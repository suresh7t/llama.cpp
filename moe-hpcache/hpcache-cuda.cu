#include "hpcache-cuda.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>

static size_t round_up(size_t n) {
    return (n + 255) & ~size_t(255);
}

bool hpcache_doorbell_init(hpcache_doorbell & db, int n_embd, int k, std::string & error) {
    // one cache line each for the three counters, then the arrays
    const size_t off_x   = 256;
    const size_t off_ids = off_x   + round_up(size_t(n_embd) * sizeof(float));
    const size_t off_w   = off_ids + round_up(size_t(k) * sizeof(int32_t));
    const size_t off_y   = off_w   + round_up(size_t(k) * sizeof(float));
    const size_t total   = off_y   + round_up(size_t(n_embd) * sizeof(float));

    void * host = nullptr;
    if (cudaHostAlloc(&host, total, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
        cudaGetLastError();
        error = "cudaHostAlloc failed";
        return false;
    }
    memset(host, 0, total);

    void * dev = nullptr;
    if (cudaHostGetDevicePointer(&dev, host, 0) != cudaSuccess) {
        cudaGetLastError();
        cudaFreeHost(host);
        error = "cudaHostGetDevicePointer failed";
        return false;
    }

    auto * h = static_cast<char *>(host);
    auto * d = static_cast<char *>(dev);

    db.n_embd = n_embd;
    db.k      = k;
    db.block  = host;

    db.seq   = reinterpret_cast<volatile uint32_t *>(h + 0);
    db.flag  = reinterpret_cast<volatile uint32_t *>(h + 64);
    db.layer = reinterpret_cast<volatile int32_t  *>(h + 128);
    db.x     = reinterpret_cast<float   *>(h + off_x);
    db.ids   = reinterpret_cast<int32_t *>(h + off_ids);
    db.w     = reinterpret_cast<float   *>(h + off_w);
    db.y     = reinterpret_cast<float   *>(h + off_y);

    db.d_seq   = reinterpret_cast<volatile uint32_t *>(d + 0);
    db.d_flag  = reinterpret_cast<volatile uint32_t *>(d + 64);
    db.d_layer = reinterpret_cast<volatile int32_t  *>(d + 128);
    db.d_x     = reinterpret_cast<float   *>(d + off_x);
    db.d_ids   = reinterpret_cast<int32_t *>(d + off_ids);
    db.d_w     = reinterpret_cast<float   *>(d + off_w);
    db.d_y     = reinterpret_cast<float   *>(d + off_y);
    return true;
}

void hpcache_doorbell_free(hpcache_doorbell & db) {
    if (db.block != nullptr) {
        cudaFreeHost(db.block);
    }
    db = hpcache_doorbell();
}

// The ring increments the counter in memory. It must not take the new value as an argument,
// because that value would be fixed when the graph is built.
static __global__ void publish_kernel(const float * x, const int32_t * ids, const float * w, int n, int k, int layer,
        float * x_out, int32_t * ids_out, float * w_out, volatile int32_t * layer_out, volatile uint32_t * seq) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        x_out[i] = x[i];
    }
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        ids_out[i] = ids[i];
        w_out[i]   = w[i];
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        *layer_out = layer;
        __threadfence_system();
        *seq = *seq + 1u;
    }
}

static __global__ void wait_merge_kernel(const volatile uint32_t * flag, const volatile uint32_t * seq,
        const volatile float * y, float * dst, int n) {
    __shared__ uint32_t want;
    if (threadIdx.x == 0) {
        want = *seq;
        while (*flag != want) {
#if __CUDA_ARCH__ >= 700
            __nanosleep(100);
#endif
        }
        __threadfence_system();
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        dst[i] = y[i];
    }
}

void hpcache_cuda_publish(void * stream, const hpcache_doorbell & db,
        const float * x, const int32_t * ids, const float * w, int layer) {
    publish_kernel<<<1, 256, 0, (cudaStream_t) stream>>>(x, ids, w, db.n_embd, db.k, layer,
            db.d_x, db.d_ids, db.d_w, db.d_layer, db.d_seq);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "moe-hpcache: publish kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}

void hpcache_cuda_wait_merge(void * stream, const hpcache_doorbell & db, float * dst) {
    wait_merge_kernel<<<1, 256, 0, (cudaStream_t) stream>>>(db.d_flag, db.d_seq, db.d_y, dst, db.n_embd);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "moe-hpcache: wait kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}

void hpcache_cuda_flush(void * stream) {
    // a query would invalidate a capture in progress
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing((cudaStream_t) stream, &status) != cudaSuccess) {
        cudaGetLastError();
        return;
    }
    if (status == cudaStreamCaptureStatusNone) {
        (void) cudaStreamQuery((cudaStream_t) stream);
    }
}

int hpcache_cuda_get_device() {
    int device = 0;
    cudaGetDevice(&device);
    return device;
}

void hpcache_cuda_set_device(int device) {
    cudaSetDevice(device);
}

void hpcache_cuda_sync_device() {
    cudaDeviceSynchronize();
}

int32_t * hpcache_cuda_table_alloc(int n_expert) {
    void * p = nullptr;
    if (cudaMalloc(&p, size_t(n_expert) * sizeof(int32_t)) != cudaSuccess) {
        cudaGetLastError();
        return nullptr;
    }
    return static_cast<int32_t *>(p);
}

void hpcache_cuda_table_free(int32_t * table) {
    if (table != nullptr) {
        cudaFree(table);
    }
}

bool hpcache_cuda_table_upload(int32_t * table, const int32_t * host, int n_expert) {
    return cudaMemcpy(table, host, size_t(n_expert) * sizeof(int32_t), cudaMemcpyHostToDevice) == cudaSuccess;
}

static cudaStream_t copy_stream() {
    static cudaStream_t s = nullptr;
    if (s == nullptr && cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) != cudaSuccess) {
        cudaGetLastError();
        s = nullptr;
    }
    return s;
}

bool hpcache_cuda_copy_to_device(void * dst, const void * src, size_t nbytes) {
    cudaStream_t s = copy_stream();
    return s != nullptr && cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyHostToDevice, s) == cudaSuccess;
}

bool hpcache_cuda_copy_wait() {
    cudaStream_t s = copy_stream();
    return s != nullptr && cudaStreamSynchronize(s) == cudaSuccess;
}

static __global__ void remap_ids_kernel(const int32_t * ids, const int32_t * table, int k, int32_t * out) {
    const int i = threadIdx.x;
    if (i < k) {
        const int32_t slot = table[ids[i]];
        out[i] = slot >= 0 ? slot : 0;
    }
}

static __global__ void remap_weights_kernel(const int32_t * ids, const float * w, const int32_t * table, int k, float * out) {
    const int i = threadIdx.x;
    if (i < k) {
        out[i] = table[ids[i]] >= 0 ? w[i] : 0.0f;
    }
}

void hpcache_cuda_remap_ids(void * stream, const int32_t * ids, const int32_t * table, int k, int32_t * out) {
    remap_ids_kernel<<<1, 32, 0, (cudaStream_t) stream>>>(ids, table, k, out);
}

void hpcache_cuda_remap_weights(void * stream, const int32_t * ids, const float * w, const int32_t * table, int k, float * out) {
    remap_weights_kernel<<<1, 32, 0, (cudaStream_t) stream>>>(ids, w, table, k, out);
}

int hpcache_cuda_device_by_pci(const char * pci_bus_id) {
    int device = -1;
    if (pci_bus_id == nullptr || cudaDeviceGetByPCIBusId(&device, pci_bus_id) != cudaSuccess) {
        cudaGetLastError();
        return -1;
    }
    return device;
}
