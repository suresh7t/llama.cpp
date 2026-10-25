#include "hpcache-cuda.h"

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

static size_t round_up(size_t n) {
    return (n + 255) & ~size_t(255);
}

bool hpcache_doorbell_init(hpcache_doorbell & db, int n_embd, int k, std::string & error) {
    // one cache line each for the counters, then the arrays
    const size_t off_x   = 384;
    const size_t off_ids = off_x   + round_up(size_t(n_embd) * sizeof(float));
    const size_t off_w   = off_ids + round_up(size_t(k) * sizeof(int32_t));
    const size_t off_y   = off_w   + round_up(size_t(k) * sizeof(float));
    const size_t off_y2  = off_y   + round_up(size_t(n_embd) * sizeof(float));
    const size_t off_gtab = off_y2 + round_up(size_t(n_embd) * sizeof(float));
    const size_t total   = off_gtab + round_up(size_t(hpcache_gtab_size));

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
    db.flag2 = reinterpret_cast<volatile uint32_t *>(h + 256);
    db.stats = reinterpret_cast<volatile unsigned long long *>(h + 320);
    db.x     = reinterpret_cast<float   *>(h + off_x);
    db.ids   = reinterpret_cast<int32_t *>(h + off_ids);
    db.w     = reinterpret_cast<float   *>(h + off_w);
    db.y     = reinterpret_cast<float   *>(h + off_y);
    db.y2    = reinterpret_cast<float   *>(h + off_y2);
    db.gtab  = reinterpret_cast<volatile int8_t *>(h + off_gtab);

    db.d_seq   = reinterpret_cast<volatile uint32_t *>(d + 0);
    db.d_flag  = reinterpret_cast<volatile uint32_t *>(d + 64);
    db.d_layer = reinterpret_cast<volatile int32_t  *>(d + 128);
    db.d_frac  = reinterpret_cast<volatile int32_t  *>(d + 192);
    db.d_wait_ns = reinterpret_cast<volatile unsigned long long *>(d + 200);
    db.d_flag2 = reinterpret_cast<volatile uint32_t *>(d + 256);
    db.d_stats = reinterpret_cast<volatile unsigned long long *>(d + 320);
    db.d_x     = reinterpret_cast<float   *>(d + off_x);
    db.d_ids   = reinterpret_cast<int32_t *>(d + off_ids);
    db.d_w     = reinterpret_cast<float   *>(d + off_w);
    db.d_y     = reinterpret_cast<float   *>(d + off_y);
    db.d_y2    = reinterpret_cast<float   *>(d + off_y2);
    db.d_gtab  = reinterpret_cast<int8_t *>(d + off_gtab);
    return true;
}

void hpcache_doorbell_free(hpcache_doorbell & db) {
    if (db.block != nullptr) {
        cudaFreeHost(db.block);
    }
    db = hpcache_doorbell();
}

static __device__ __forceinline__ unsigned long long global_ns() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

#ifdef HPCACHE_EVENT_LOG
// One slot: the ring is written by one stream only, so the counter needs no atomic. The seq (the lap) goes last.
static __device__ void device_log(hpcache_metric_dev m, unsigned code, int arg0, int arg1, int arg2, unsigned long long ts) {
    const unsigned long long h = *m.head;
    hpcache_metric_rec * p = m.rec + (h & m.mask);
    p->level = 0;
    p->ts    = ts;
    p->code  = code;
    p->arg0  = arg0;
    p->arg1  = arg1;
    p->arg2  = arg2;
    __threadfence_system();
    p->seq = (uint32_t) (h / ((unsigned long long) m.mask + 1)) + 1;
    *m.head = h + 1;
}
#endif

// The ring increments the counter in memory. It must not take the new value as an argument,
// because that value would be fixed when the graph is built.
static __global__ void publish_kernel(const float * x, const int32_t * ids, const float * w, int n, int k, int layer,
        float * x_out, int32_t * ids_out, float * w_out, volatile int32_t * layer_out, volatile uint32_t * seq
        HPCACHE_LOG_PARAM(hpcache_metric_dev m)) {
#ifdef HPCACHE_EVENT_LOG
    unsigned long long t_start = 0;
    if (m.rec != nullptr && threadIdx.x == 0) {
        t_start = global_ns();
    }
#endif
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
#ifdef HPCACHE_EVENT_LOG
        if (m.rec != nullptr) {
            device_log(m, HPE_PUBLISH_START, *seq, layer, 0, t_start);
            device_log(m, HPE_PUBLISH_STOP, *seq, layer, 0, global_ns());
        }
#endif
    }
}

#ifdef HPCACHE_EVENT_LOG
// the sequence number is read when the kernel runs, the same way as in the publish
static __global__ void device_log_kernel(hpcache_metric_dev m, unsigned code, const volatile uint32_t * seq, int layer) {
    if (threadIdx.x == 0) {
        device_log(m, code, *seq, layer, 0, global_ns());
    }
}
#endif

static __global__ void wait_merge_kernel(const volatile uint32_t * flag, const volatile uint32_t * seq,
        const volatile float * y, float * dst, int n, volatile unsigned long long * wait_ns,
        const volatile uint32_t * flag2, const volatile float * y2, volatile unsigned long long * stats
        HPCACHE_LOG_PARAM(hpcache_metric_dev m)) {
    __shared__ uint32_t want;
    if (threadIdx.x == 0) {
        want = *seq;
        const unsigned long long t0 = global_ns();
        // both answers are polled together, so each wait is counted from the same start
        unsigned long long t_cpu = 0, t_two = 0;
        bool cpu_done = false, two_done = flag2 == nullptr;
        while (!cpu_done || !two_done) {
            if (!cpu_done && *flag == want) {
                cpu_done = true;
                t_cpu = global_ns() - t0;
            }
            if (!two_done && *flag2 == want) {
                two_done = true;
                t_two = global_ns() - t0;
            }
#if __CUDA_ARCH__ >= 700
            __nanosleep(100);
#endif
        }
        __threadfence_system();
#ifdef HPCACHE_EVENT_LOG
        if (m.rec != nullptr) {
            device_log(m, HPE_WAIT_START, want, -1, 0, t0);
            device_log(m, HPE_WAIT_STOP, want, -1, 0, global_ns());
        }
#endif
        // one wait kernel runs at a time, so a plain add is enough
        *wait_ns = *wait_ns + (global_ns() - t0);
        stats[0] = stats[0] + t_cpu;
        if (flag2 != nullptr) {
            stats[1] = stats[1] + t_two;
            if (t_cpu >= t_two) {
                stats[2] = stats[2] + 1;
            } else {
                stats[3] = stats[3] + 1;
            }
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        dst[i] = y2 != nullptr ? y[i] + y2[i] : y[i];
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
            db.d_x, db.d_ids, db.d_w, db.d_layer, db.d_seq HPCACHE_LOG_ARG(db.lead_m));
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "moe-hpcache: publish kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    nudge_submission((cudaStream_t) stream);
}

static __global__ void g2_finish_kernel(const float * out, float * y2, volatile uint32_t * flag2, const volatile uint32_t * seq, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        y2[i] = out[i];
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        *flag2 = *seq;
    }
}

void hpcache_cuda_g2_finish(void * stream, const hpcache_doorbell & db, const float * out, const volatile uint32_t * seq) {
    g2_finish_kernel<<<1, 256, 0, (cudaStream_t) stream>>>(out, db.d_y2, db.d_flag2, seq, db.n_embd);
    cudaGetLastError();
}

void hpcache_cuda_wait_merge(void * stream, const hpcache_doorbell & db, float * dst) {
    wait_merge_kernel<<<1, 256, 0, (cudaStream_t) stream>>>(db.d_flag, db.d_seq, db.d_y, dst, db.n_embd, db.d_wait_ns,
            db.two ? db.d_flag2 : nullptr, db.two ? db.d_y2 : nullptr, db.d_stats HPCACHE_LOG_ARG(db.lead_m));
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "moe-hpcache: wait kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    nudge_submission((cudaStream_t) stream);
}

#ifdef HPCACHE_EVENT_LOG
bool hpcache_cuda_metric_ring_alloc(hpcache_metric_ring & r, int device, uint32_t n) {
    int prev = 0;
    cudaGetDevice(&prev);
    bool ok = cudaSetDevice(device) == cudaSuccess;
    void * host = nullptr;
    void * dev = nullptr;
    void * head = nullptr;
    const size_t bytes = size_t(n) * sizeof(hpcache_metric_rec);
    ok = ok && cudaHostAlloc(&host, bytes, cudaHostAllocMapped | cudaHostAllocPortable) == cudaSuccess;
    ok = ok && cudaHostGetDevicePointer(&dev, host, 0) == cudaSuccess;
    ok = ok && cudaMalloc(&head, sizeof(unsigned long long)) == cudaSuccess;
    ok = ok && cudaMemset(head, 0, sizeof(unsigned long long)) == cudaSuccess;
    if (ok) {
        memset(host, 0, bytes);
        r.rec    = static_cast<hpcache_metric_rec *>(host);
        r.d_rec  = static_cast<hpcache_metric_rec *>(dev);
        r.d_head = static_cast<unsigned long long *>(head);
        r.mask   = n - 1;
    } else {
        cudaGetLastError();
        if (host != nullptr) cudaFreeHost(host);
        if (head != nullptr) cudaFree(head);
    }
    cudaSetDevice(prev);
    return ok;
}

void hpcache_cuda_metric_ring_free(hpcache_metric_ring & r) {
    int prev = 0;
    cudaGetDevice(&prev);
    cudaSetDevice(r.device);
    cudaFreeHost(r.rec);
    cudaFree(r.d_head);
    cudaSetDevice(prev);
}

void hpcache_cuda_device_log(void * stream, const hpcache_metric_ring & r, uint16_t code, const volatile uint32_t * seq, int layer) {
    device_log_kernel<<<1, 32, 0, (cudaStream_t) stream>>>(r.view(), code, seq, layer);
    cudaGetLastError();
}

// The kernel answers each request word with its globaltimer, the host takes the middle of the round trip as the moment
// of the stamp. The best of many rounds has the smallest error.
static __global__ void clock_probe_kernel(volatile uint32_t * req, volatile uint32_t * resp, volatile unsigned long long * t, int rounds) {
    if (threadIdx.x != 0) {
        return;
    }
    *resp = 0xffffffffu;   // ready
    __threadfence_system();
    for (int i = 1; i <= rounds; ++i) {
        const unsigned long long w0 = global_ns();
        while (*req != (uint32_t) i) {
            if (global_ns() - w0 > 500000000ull) {
                return;
            }
        }
        t[i] = global_ns();
        __threadfence_system();
        *resp = (uint32_t) i;
    }
}

bool hpcache_cuda_clock_offset(int device, long long & off_ns, long long & rtt_ns) {
    int prev = 0;
    cudaGetDevice(&prev);
    if (cudaSetDevice(device) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    const int rounds = 200;
    void * host = nullptr;
    void * dev = nullptr;
    cudaStream_t s = nullptr;
    bool ok = false;
    const size_t bytes = 256 + size_t(rounds + 1) * sizeof(unsigned long long);
    if (cudaHostAlloc(&host, bytes, cudaHostAllocMapped | cudaHostAllocPortable) == cudaSuccess &&
            cudaHostGetDevicePointer(&dev, host, 0) == cudaSuccess &&
            cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) == cudaSuccess) {
        memset(host, 0, bytes);
        auto * req  = static_cast<volatile uint32_t *>(host);
        auto * resp = reinterpret_cast<volatile uint32_t *>(static_cast<char *>(host) + 64);
        auto * t    = reinterpret_cast<volatile unsigned long long *>(static_cast<char *>(host) + 128);
        auto * d    = static_cast<char *>(dev);
        clock_probe_kernel<<<1, 32, 0, s>>>(reinterpret_cast<volatile uint32_t *>(d), reinterpret_cast<volatile uint32_t *>(d + 64),
                reinterpret_cast<volatile unsigned long long *>(d + 128), rounds);
        cudaStreamQuery(s);
        const long long t_ready = hpcache_metric_now();
        while (*resp != 0xffffffffu && hpcache_metric_now() - t_ready < 2000000000ll) {
        }
        if (*resp == 0xffffffffu) {
            long long best_rtt = -1, best_off = 0;
            ok = true;
            for (int i = 1; i <= rounds && ok; ++i) {
                const long long h0 = hpcache_metric_now();
                *req = (uint32_t) i;
                while (*resp != (uint32_t) i) {
                    if (hpcache_metric_now() - h0 > 200000000ll) {
                        ok = false;
                        break;
                    }
                }
                const long long h1 = hpcache_metric_now();
                if (ok && (best_rtt < 0 || h1 - h0 < best_rtt)) {
                    best_rtt = h1 - h0;
                    best_off = (h0 + h1) / 2 - (long long) t[i];
                }
            }
            off_ns = best_off;
            rtt_ns = best_rtt;
        }
        cudaStreamSynchronize(s);
    }
    cudaGetLastError();
    if (s != nullptr) {
        cudaStreamDestroy(s);
    }
    if (host != nullptr) {
        cudaFreeHost(host);
    }
    cudaSetDevice(prev);
    return ok;
}
#endif

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

// One warp that spins on the GPU's own clock. It is short because WDDM resets a kernel that runs for seconds; the thread
// below always keeps the next one queued behind the running one. Other shapes were measured and are worse: kernels of 55 us
// with a sync after each flood the launch path (3.5 t/s), and a single thread that waits on memory costs as much as this one.
static __global__ void keepalive_kernel(unsigned long long ns) {
    const unsigned long long t0 = global_ns();
    while (global_ns() - t0 < ns) {
    }
}

struct hpcache_keepalive {
    std::thread       thread;
    std::atomic<bool> stop{false};
};

void * hpcache_cuda_keepalive_start(int device) {
    hpcache_keepalive * ka = new hpcache_keepalive();
    ka->thread = std::thread([ka, device] {
        constexpr unsigned long long kernel_ns = 40ull * 1000 * 1000;
        cudaSetDevice(device);
        cudaStream_t s = nullptr;
        cudaEvent_t  done[2] = {};
        int least = 0, greatest = 0;
        cudaDeviceGetStreamPriorityRange(&least, &greatest);   // the least urgent one: the work of the card goes first
        if (cudaStreamCreateWithPriority(&s, cudaStreamNonBlocking, least) != cudaSuccess ||
            cudaEventCreateWithFlags(&done[0], cudaEventDisableTiming) != cudaSuccess ||
            cudaEventCreateWithFlags(&done[1], cudaEventDisableTiming) != cudaSuccess) {
            cudaGetLastError();
            return;
        }
        // two in flight: when the older one is done the newer one is running, so the card is never without work
        for (int i = 0; i < 2; ++i) {
            keepalive_kernel<<<1, 32, 0, s>>>(kernel_ns);
            cudaEventRecord(done[i], s);
        }
        int oldest = 0;
        while (!ka->stop.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            if (cudaEventQuery(done[oldest]) == cudaSuccess) {
                keepalive_kernel<<<1, 32, 0, s>>>(kernel_ns);
                cudaEventRecord(done[oldest], s);
                oldest ^= 1;
            }
        }
        cudaStreamSynchronize(s);
        cudaEventDestroy(done[0]);
        cudaEventDestroy(done[1]);
        cudaStreamDestroy(s);
    });
    return ka;
}

void hpcache_cuda_keepalive_stop(void * handle) {
    hpcache_keepalive * ka = static_cast<hpcache_keepalive *>(handle);
    if (ka == nullptr) {
        return;
    }
    ka->stop.store(true, std::memory_order_release);
    if (ka->thread.joinable()) {
        ka->thread.join();
    }
    delete ka;
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
        int m, const volatile int32_t * frac_ptr, const volatile int8_t * gtab, int32_t * out_ids, float * out_w) {
    if (threadIdx.x != 0) {
        return;
    }
    const int frac256 = gtab != nullptr ? 0 : *frac_ptr;
    int miss[32];
    int n = 0;
    int hits = 0;
    int on_second = 0;
    for (int i = 0; i < k && i < 32; ++i) {
        const int32_t e = ids[i];
        if (e < 0 || e >= n_expert) {
            continue;
        }
        const int32_t t = table == nullptr ? -1 : table[e];
        if (t == -1) {
            miss[n++] = i;
        } else if (t == -2) {
            on_second = 1;
        } else {
            ++hits;
        }
    }
    int g = gtab != nullptr ? (int) gtab[(on_second * 33 + hits) * 33 + n] : (n * frac256) >> 8;
    if (g > m) {
        g = m;
    }
    if (g > n) {
        g = n;
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
        int k, int n_expert, int m, const volatile int32_t * frac256, const volatile int8_t * gtab, int32_t * out_ids, float * out_w) {
    pcie_select_kernel<<<1, 32, 0, (cudaStream_t) stream>>>(ids, w, table, k, n_expert, m, frac256, gtab, out_ids, out_w);
}

int8_t * hpcache_cuda_upload(int device, const void * host, size_t bytes) {
    int prev = 0;
    cudaGetDevice(&prev);
    void * dev = nullptr;
    bool ok = cudaSetDevice(device) == cudaSuccess;
    ok = ok && cudaMalloc(&dev, bytes) == cudaSuccess;
    ok = ok && cudaMemcpy(dev, host, bytes, cudaMemcpyHostToDevice) == cudaSuccess;
    if (!ok) {
        cudaGetLastError();
        if (dev != nullptr) {
            cudaFree(dev);
        }
        dev = nullptr;
    }
    cudaSetDevice(prev);
    return static_cast<int8_t *>(dev);
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
        const volatile int32_t * frac_ptr, const volatile int8_t * gtab, hpcache_stage_args a) {
    __shared__ int sel[32];
    __shared__ int g;
    if (threadIdx.x == 0) {
        const int frac256 = gtab != nullptr ? 0 : *frac_ptr;
        int miss[32];
        int n = 0;
        int hits = 0;
        int on_second = 0;
        for (int i = 0; i < k && i < 32; ++i) {
            const int32_t e = ids[i];
            if (e < 0 || e >= n_expert) {
                continue;
            }
            const int32_t t = table == nullptr ? -1 : table[e];
            if (t == -1) {
                miss[n++] = i;
            } else if (t == -2) {
                on_second = 1;
            } else {
                ++hits;
            }
        }
        int gg = gtab != nullptr ? (int) gtab[(on_second * 33 + hits) * 33 + n] : (n * frac256) >> 8;
        if (gg > m) {
            gg = m;
        }
        if (gg > n) {
            gg = n;
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

// the side stream of the fetch and its two events, made on the first call, which is before any graph is captured
static cudaStream_t stage_stream = nullptr;
static cudaEvent_t  stage_fork   = nullptr;
static cudaEvent_t  stage_join   = nullptr;

void hpcache_cuda_pcie_stage(void * stream, const int32_t * ids, const int32_t * table, int k, int n_expert, int m,
        const volatile int32_t * frac256, const volatile int8_t * gtab, const hpcache_stage_args & args, bool side) {
    // enough blocks along z to keep many PCIe reads in flight
    const dim3 grid(unsigned(m), unsigned(args.n_tensors), 32);
    if (!side) {
        pcie_stage_kernel<<<grid, 256, 0, (cudaStream_t) stream>>>(ids, table, k, n_expert, m, frac256, gtab, args);
        return;
    }
    if (stage_stream == nullptr) {
        cudaStreamCreateWithFlags(&stage_stream, cudaStreamNonBlocking);
        cudaEventCreateWithFlags(&stage_fork, cudaEventDisableTiming);
        cudaEventCreateWithFlags(&stage_join, cudaEventDisableTiming);
    }
    cudaEventRecord(stage_fork, (cudaStream_t) stream);
    cudaStreamWaitEvent(stage_stream, stage_fork, 0);
    pcie_stage_kernel<<<grid, 256, 0, stage_stream>>>(ids, table, k, n_expert, m, frac256, gtab, args);
    cudaEventRecord(stage_join, stage_stream);
}

void hpcache_cuda_pcie_stage_join(void * stream) {
    cudaStreamWaitEvent((cudaStream_t) stream, stage_join, 0);
}
