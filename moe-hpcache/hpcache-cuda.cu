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
    db.frac  = reinterpret_cast<volatile int32_t  *>(h + 192);
    db.wait_ns = reinterpret_cast<volatile unsigned long long *>(h + 200);
    db.x     = reinterpret_cast<float   *>(h + off_x);
    db.ids   = reinterpret_cast<int32_t *>(h + off_ids);
    db.w     = reinterpret_cast<float   *>(h + off_w);
    db.y     = reinterpret_cast<float   *>(h + off_y);

    db.d_seq   = reinterpret_cast<volatile uint32_t *>(d + 0);
    db.d_flag  = reinterpret_cast<volatile uint32_t *>(d + 64);
    db.d_layer = reinterpret_cast<volatile int32_t  *>(d + 128);
    db.d_frac  = reinterpret_cast<volatile int32_t  *>(d + 192);
    db.d_wait_ns = reinterpret_cast<volatile unsigned long long *>(d + 200);
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

static __device__ __forceinline__ unsigned long long global_ns() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

static __global__ void wait_merge_kernel(const volatile uint32_t * flag, const volatile uint32_t * seq,
        const volatile float * y, float * dst, int n, volatile unsigned long long * wait_ns) {
    __shared__ uint32_t want;
    if (threadIdx.x == 0) {
        want = *seq;
        const unsigned long long t0 = global_ns();
        while (*flag != want) {
#if __CUDA_ARCH__ >= 700
            __nanosleep(100);
#endif
        }
        __threadfence_system();
        // one wait kernel runs at a time, so a plain add is enough
        *wait_ns = *wait_ns + (global_ns() - t0);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        dst[i] = y[i];
    }
}

// On WDDM the driver can hold queued launches back, and the wait kernel that follows a publish spins until the
// CPU answers. Called by the thread that launched them, never by the follower: a query from another thread can
// wait behind a launch that is blocked on a full queue.
static void nudge_submission(cudaStream_t stream) {
    // a query would invalidate a capture in progress
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &status) != cudaSuccess) {
        cudaGetLastError();
        return;
    }
    if (status == cudaStreamCaptureStatusNone) {
        (void) cudaStreamQuery(stream);
        cudaGetLastError();   // not-ready is not an error, and the core checks the last error after its launches
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
    nudge_submission((cudaStream_t) stream);
}

void hpcache_cuda_wait_merge(void * stream, const hpcache_doorbell & db, float * dst) {
    wait_merge_kernel<<<1, 256, 0, (cudaStream_t) stream>>>(db.d_flag, db.d_seq, db.d_y, dst, db.n_embd, db.d_wait_ns);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "moe-hpcache: wait kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    nudge_submission((cudaStream_t) stream);
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

bool hpcache_cuda_tables_alloc(hpcache_table_block & b, int n_tables, int n_expert) {
    const size_t bytes = size_t(n_tables) * size_t(n_expert) * sizeof(int32_t);
    void * d = nullptr;
    void * hh = nullptr;
    if (cudaMalloc(&d, bytes) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    if (cudaHostAlloc(&hh, bytes, cudaHostAllocDefault) != cudaSuccess) {
        cudaGetLastError();
        cudaFree(d);
        return false;
    }
    memset(hh, 0xff, bytes);   // -1: not resident
    cudaMemcpy(d, hh, bytes, cudaMemcpyHostToDevice);
    b.d = static_cast<int32_t *>(d);
    b.h = static_cast<int32_t *>(hh);
    b.n_tables = n_tables;
    b.n_expert = n_expert;
    return true;
}

void hpcache_cuda_tables_free(hpcache_table_block & b) {
    if (b.d != nullptr) {
        cudaFree(b.d);
    }
    if (b.h != nullptr) {
        cudaFreeHost(b.h);
    }
    b = hpcache_table_block();
}

bool hpcache_cuda_tables_upload(const hpcache_table_block & b, int n_tables) {
    if (b.d == nullptr || n_tables <= 0 || n_tables > b.n_tables) {
        return false;
    }
    return cudaMemcpy(b.d, b.h, size_t(n_tables) * size_t(b.n_expert) * sizeof(int32_t), cudaMemcpyHostToDevice) == cudaSuccess;
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

// miss is the id written for an expert that is not in the pool: 0 computes slot 0 and the weight of that column is
// zero, a negative id makes the quantized matvec of ggml-cuda skip the column (mmvq.cu)
static __global__ void remap_ids_kernel(const int32_t * ids, const int32_t * table, int k, int32_t miss, int32_t * out) {
    const int i = threadIdx.x;
    if (i < k) {
        const int32_t slot = table[ids[i]];
        out[i] = slot >= 0 ? slot : miss;
    }
}

static __global__ void remap_weights_kernel(const int32_t * ids, const float * w, const int32_t * table, int k, float * out) {
    const int i = threadIdx.x;
    if (i < k) {
        out[i] = table[ids[i]] >= 0 ? w[i] : 0.0f;
    }
}

void hpcache_cuda_remap_ids(void * stream, const int32_t * ids, const int32_t * table, int k, int32_t miss, int32_t * out) {
    remap_ids_kernel<<<1, 32, 0, (cudaStream_t) stream>>>(ids, table, k, miss, out);
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

// page-lock a host range (or release it); not portable, so it is mapped for the current device only
bool hpcache_cuda_host_register(void * ptr, size_t nbytes, bool pin) {
    if (!pin) {
        const cudaError_t e = cudaHostUnregister(ptr);
        if (e != cudaSuccess) {
            cudaGetLastError();
        }
        return e == cudaSuccess;
    }
    const cudaError_t e = cudaHostRegister(ptr, nbytes, cudaHostRegisterMapped);
    if (e != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return true;
}

static __global__ void pcie_select_kernel(const int32_t * ids, const float * w, const int32_t * table, int k, int n_expert,
        int m, const volatile int32_t * frac_ptr, int32_t * out_ids, float * out_w) {
    if (threadIdx.x != 0) {
        return;
    }
    const int frac256 = *frac_ptr;
    int miss[32];
    int n = 0;
    for (int i = 0; i < k && i < 32; ++i) {
        const int32_t e = ids[i];
        if (e < 0 || e >= n_expert) {
            continue;
        }
        if (table == nullptr || table[e] < 0) {
            miss[n++] = i;
        }
    }
    int g = (n * frac256) >> 8;
    if (g > m) {
        g = m;
    }
    const int pad_i = g > 0 ? miss[n - g] : 0;
    const int32_t pad = k > 0 ? ids[pad_i] : 0;
    for (int j = 0; j < m; ++j) {
        if (j < g) {
            const int i = miss[n - g + j];
            if (out_ids != nullptr) {
                out_ids[j] = j;
            }
            if (out_w != nullptr) {
                out_w[j] = w[i];
            }
        } else {
            if (out_ids != nullptr) {
                out_ids[j] = j;
            }
            if (out_w != nullptr) {
                out_w[j] = 0.0f;
            }
        }
    }
}

void hpcache_cuda_pcie_select(void * stream, const int32_t * ids, const float * w, const int32_t * table,
        int k, int n_expert, int m, const volatile int32_t * frac256, int32_t * out_ids, float * out_w) {
    pcie_select_kernel<<<1, 32, 0, (cudaStream_t) stream>>>(ids, w, table, k, n_expert, m, frac256, out_ids, out_w);
}

void * hpcache_cuda_host_device_ptr(const void * ptr) {
    void * dev = nullptr;
    const cudaError_t e = cudaHostGetDevicePointer(&dev, const_cast<void *>(ptr), 0);
    if (e != cudaSuccess) {
        cudaGetLastError();
        return nullptr;
    }
    return dev;
}

// one block per (selected expert, tensor, 16 KiB chunk); every block repeats the selection, which is 32 loop steps
static __global__ void pcie_stage_kernel(const int32_t * ids, const int32_t * table, int k, int n_expert, int m,
        const volatile int32_t * frac_ptr, hpcache_stage_args a) {
    __shared__ int sel[32];
    __shared__ int g;
    if (threadIdx.x == 0) {
        const int frac256 = *frac_ptr;
        int miss[32];
        int n = 0;
        for (int i = 0; i < k && i < 32; ++i) {
            const int32_t e = ids[i];
            if (e < 0 || e >= n_expert) {
                continue;
            }
            if (table == nullptr || table[e] < 0) {
                miss[n++] = i;
            }
        }
        int gg = (n * frac256) >> 8;
        if (gg > m) {
            gg = m;
        }
        g = gg;
        for (int j = 0; j < gg; ++j) {
            sel[j] = ids[miss[n - gg + j]];
        }
    }
    __syncthreads();

    const int j = blockIdx.x;
    const int t = blockIdx.y;
    if (t >= a.n_tensors) {
        return;
    }
    const size_t n16 = a.bytes[t] / 16;
    // an unused slot is zeroed: it holds the bytes of another layer's tensor type, which can decode to NaN
    const bool fill = j < g;
    const uint4 * src = fill ? reinterpret_cast<const uint4 *>(static_cast<const char *>(a.src[t]) + size_t(sel[j]) * a.bytes[t]) : nullptr;
    uint4 * dst = reinterpret_cast<uint4 *>(static_cast<char *>(a.dst[t]) + size_t(j) * a.bytes[t]);
    const size_t chunk = 1024;   // uint4 per block-z step: 16 KiB
    for (size_t base = size_t(blockIdx.z) * chunk; base < n16; base += size_t(gridDim.z) * chunk) {
        const size_t end = base + chunk < n16 ? base + chunk : n16;
        for (size_t i = base + threadIdx.x; i < end; i += blockDim.x) {
            dst[i] = fill ? src[i] : make_uint4(0, 0, 0, 0);
        }
    }
}

void hpcache_cuda_pcie_stage(void * stream, const int32_t * ids, const int32_t * table, int k, int n_expert, int m,
        const volatile int32_t * frac256, const hpcache_stage_args & args) {
    // enough blocks along z to keep many PCIe reads in flight
    const dim3 grid(unsigned(m), unsigned(args.n_tensors), 32);
    pcie_stage_kernel<<<grid, 256, 0, (cudaStream_t) stream>>>(ids, table, k, n_expert, m, frac256, args);
}
