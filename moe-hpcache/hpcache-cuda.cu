#include "hpcache-cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

static size_t round_up(size_t n) {
    return (n + 255) & ~size_t(255);
}

bool hpcache_doorbell_init(hpcache_doorbell & db, int n_embd, int k, int max_tokens, std::string & error) {
    // one cache line each for the counters, then the arrays; x, ids, w and y hold max_tokens tokens
    const size_t nt      = size_t(max_tokens < 1 ? 1 : max_tokens);
    const size_t off_x   = 384;
    const size_t off_ids = off_x   + round_up(size_t(n_embd) * nt * sizeof(float));
    const size_t off_w   = off_ids + round_up(size_t(k) * nt * sizeof(int32_t));
    const size_t off_y   = off_w   + round_up(size_t(k) * nt * sizeof(float));
    const size_t off_y2  = off_y   + round_up(size_t(n_embd) * nt * sizeof(float));
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
    db.max_tokens = int(nt);
    db.block  = host;

    db.seq   = reinterpret_cast<volatile uint32_t *>(h + 0);
    db.flag  = reinterpret_cast<volatile uint32_t *>(h + 64);
    db.layer = reinterpret_cast<volatile int32_t  *>(h + 128);
    db.n_tok = reinterpret_cast<volatile int32_t  *>(h + 132);
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
    db.d_n_tok = reinterpret_cast<volatile int32_t  *>(d + 132);
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
        float * x_out, int32_t * ids_out, float * w_out, volatile int32_t * layer_out, volatile uint32_t * seq,
        int n_tokens, int x_stride, int ids_stride, int w_stride, volatile int32_t * n_tok_out
        HPCACHE_LOG_PARAM(hpcache_metric_dev m)) {
#ifdef HPCACHE_EVENT_LOG
    unsigned long long t_start = 0;
    if (m.rec != nullptr && threadIdx.x == 0) {
        t_start = global_ns();
    }
#endif
    for (int i = threadIdx.x; i < n * n_tokens; i += blockDim.x) {
        x_out[i] = x[(i / n) * x_stride + i % n];
    }
    for (int i = threadIdx.x; i < k * n_tokens; i += blockDim.x) {
        ids_out[i] = ids[(i / k) * ids_stride + i % k];
        w_out[i]   = w[(i / k) * w_stride + i % k];
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        *layer_out = layer;
        *n_tok_out = n_tokens;
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
        const float * x, const int32_t * ids, const float * w, int layer, int n_tokens, int x_stride, int ids_stride, int w_stride) {
    publish_kernel<<<1, 256, 0, (cudaStream_t) stream>>>(x, ids, w, db.n_embd, db.k, layer,
            db.d_x, db.d_ids, db.d_w, db.d_layer, db.d_seq, n_tokens, x_stride, ids_stride, w_stride, db.d_n_tok
            HPCACHE_LOG_ARG(db.lead_m));
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

void hpcache_cuda_wait_merge(void * stream, const hpcache_doorbell & db, float * dst, int n_tokens) {
    wait_merge_kernel<<<1, 256, 0, (cudaStream_t) stream>>>(db.d_flag, db.d_seq, db.d_y, dst, db.n_embd * n_tokens, db.d_wait_ns,
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

// Copy uint4 [begin, end) with U loads in flight per thread. A load from host memory (through PCIe) takes 1-2 us, and a
// thread that stores right after its load has only one in flight, so a plain loop needs many blocks to fill the link, and
// those blocks hold the load queues of every SM and slow the kernels next to them (matvecs about 3x, see nsoverlap.py).
template <int U>
static __device__ __forceinline__ void copy_u4(uint4 * dst, const uint4 * src, size_t begin, size_t end, size_t tid, size_t stride) {
    size_t i = begin + tid;
    for (; i + size_t(U - 1) * stride < end; i += size_t(U) * stride) {
        uint4 v[U];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            v[u] = src[i + size_t(u) * stride];
        }
#pragma unroll
        for (int u = 0; u < U; ++u) {
            dst[i + size_t(u) * stride] = v[u];
        }
    }
    for (; i < end; i += stride) {
        dst[i] = src[i];
    }
}

// one piece of a swap copy, at most swap_chunk_n16 uint4
struct hpcache_swap_chunk {
    const uint4 * src;
    uint4 *       dst;
    unsigned int  n16;
    unsigned int  pad;
};

static constexpr unsigned int swap_chunk_n16 = 4096;   // 64 KiB

// a grid-stride loop over the chunks; the chunk list is in mapped host memory, thread 0 reads each entry once
static __global__ void swap_copy_kernel(const hpcache_swap_chunk * chunks, int n) {
    __shared__ hpcache_swap_chunk c;
    for (int i = blockIdx.x; i < n; i += gridDim.x) {
        if (threadIdx.x == 0) {
            c = chunks[i];
        }
        __syncthreads();
        copy_u4<4>(c.dst, c.src, 0, c.n16, threadIdx.x, blockDim.x);
        __syncthreads();
    }
}

static struct {
    int                  device  = -1;
    cudaStream_t         stream  = nullptr;
    cudaEvent_t          done    = nullptr;
    hpcache_swap_chunk * h_chunks = nullptr;   // mapped, page-locked
    hpcache_swap_chunk * d_chunks = nullptr;   // its device address
    int                  cap     = 0;
    bool                 launched = false;
} swap_state;

bool hpcache_cuda_swap_copy(int device, const hpcache_swap_copy * copies, int n) {
    if (n <= 0) {
        return true;
    }
    int prev = 0;
    cudaGetDevice(&prev);
    if (device >= 0) {
        cudaSetDevice(device);
    }
    auto fail = [&]() {
        cudaGetLastError();
        cudaSetDevice(prev);
        return false;
    };
    if (swap_state.stream == nullptr) {
        int least = 0, greatest = 0;
        cudaDeviceGetStreamPriorityRange(&least, &greatest);   // the least urgent one: every other kernel goes first
        if (cudaStreamCreateWithPriority(&swap_state.stream, cudaStreamNonBlocking, least) != cudaSuccess ||
            cudaEventCreateWithFlags(&swap_state.done, cudaEventDisableTiming) != cudaSuccess) {
            return fail();
        }
        swap_state.device = device;
    }
    // the chunk list is rewritten below, so the previous kernel must be done with it
    if (swap_state.launched && cudaEventSynchronize(swap_state.done) != cudaSuccess) {
        return fail();
    }
    swap_state.launched = false;

    int n_chunks = 0;
    for (int i = 0; i < n; ++i) {
        n_chunks += int((copies[i].bytes / 16 + swap_chunk_n16 - 1) / swap_chunk_n16);
    }
    if (n_chunks > swap_state.cap) {
        if (swap_state.h_chunks != nullptr) {
            cudaFreeHost(swap_state.h_chunks);
            swap_state.h_chunks = nullptr;
            swap_state.d_chunks = nullptr;
            swap_state.cap = 0;
        }
        const int cap = std::max(n_chunks, 1024);
        void * h = nullptr;
        void * d = nullptr;
        if (cudaHostAlloc(&h, size_t(cap) * sizeof(hpcache_swap_chunk), cudaHostAllocMapped) != cudaSuccess) {
            return fail();
        }
        if (cudaHostGetDevicePointer(&d, h, 0) != cudaSuccess) {
            cudaFreeHost(h);
            return fail();
        }
        swap_state.h_chunks = static_cast<hpcache_swap_chunk *>(h);
        swap_state.d_chunks = static_cast<hpcache_swap_chunk *>(d);
        swap_state.cap = cap;
    }
    int c = 0;
    for (int i = 0; i < n; ++i) {
        const uint4 * src = static_cast<const uint4 *>(copies[i].src);
        uint4 *       dst = static_cast<uint4 *>(copies[i].dst);
        const size_t  n16 = copies[i].bytes / 16;
        for (size_t off = 0; off < n16; off += swap_chunk_n16) {
            const size_t m = std::min<size_t>(swap_chunk_n16, n16 - off);
            swap_state.h_chunks[c++] = { src + off, dst + off, unsigned(m), 0 };
        }
    }
    // a small grid: 2 blocks x 256 threads x 4 loads of 16 B in flight are enough for the link, and only 2 SMs carry the slow
    // host reads (32 blocks slowed the MTP head's matvecs about 3x); HPCACHE_SWAP_BLOCKS overrides
    static const int blocks = getenv("HPCACHE_SWAP_BLOCKS") ? std::max(1, atoi(getenv("HPCACHE_SWAP_BLOCKS"))) : 2;
    swap_copy_kernel<<<blocks, 256, 0, swap_state.stream>>>(swap_state.d_chunks, c);
    if (cudaGetLastError() != cudaSuccess || cudaEventRecord(swap_state.done, swap_state.stream) != cudaSuccess) {
        cudaStreamSynchronize(swap_state.stream);
        return fail();
    }
    swap_state.launched = true;
    cudaSetDevice(prev);
    return true;
}

bool hpcache_cuda_swap_done() {
    if (!swap_state.launched) {
        return true;
    }
    const cudaError_t e = cudaEventQuery(swap_state.done);
    if (e == cudaErrorNotReady) {
        cudaGetLastError();
        return false;
    }
    return true;
}

bool hpcache_cuda_swap_wait() {
    if (!swap_state.launched) {
        return true;
    }
    const bool ok = cudaEventSynchronize(swap_state.done) == cudaSuccess;
    swap_state.launched = false;
    return ok;
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
        int m, const volatile int32_t * frac_ptr, const volatile int8_t * gtab, int32_t * out_ids, float * out_w, bool skip_unused) {
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
                out_ids[j] = skip_unused ? -1 : j;   // -1: the quantized matvec skips the column, the slot is not read
            }
            if (out_w != nullptr) {
                out_w[j] = 0.0f;
            }
        }
    }
}

void hpcache_cuda_pcie_select(void * stream, const int32_t * ids, const float * w, const int32_t * table,
        int k, int n_expert, int m, const volatile int32_t * frac256, const volatile int8_t * gtab, int32_t * out_ids, float * out_w,
        bool skip_unused) {
    pcie_select_kernel<<<1, 32, 0, (cudaStream_t) stream>>>(ids, w, table, k, n_expert, m, frac256, gtab, out_ids, out_w, skip_unused);
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

// grid z of the stage kernels: HPCACHE_STAGE_Z (default 2), read once. With 4 loads in flight per thread, 2 blocks per
// (slot, tensor) still fill the link; 32 (the old value) filled every SM and slowed the matvecs next to it about 3x
// (story, 2026-11-21: MTP1 37.9 -> 41.2 t/s, no-MTP 33.1 -> 35.5 together with the zero-fill skip and 2-4 swap blocks)
static unsigned stage_z() {
    static const unsigned z = getenv("HPCACHE_STAGE_Z") ? unsigned(std::max(1, atoi(getenv("HPCACHE_STAGE_Z")))) : 2u;
    return z;
}

// one block per (selected expert, tensor, 16 KiB chunk); every block repeats the selection, which is 32 loop steps
static __global__ void pcie_stage_kernel(const int32_t * ids, const int32_t * table, int k, int n_expert, int m,
        const volatile int32_t * frac_ptr, const volatile int8_t * gtab, hpcache_stage_args a, bool zero_unused) {
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
    // an unused slot is zeroed: it holds the bytes of another layer's tensor type, which can decode to NaN. Not needed when
    // the select op writes id -1 for it (zero_unused false), then the matvec skips the column and never reads the slot
    const bool fill = j < g;
    if (!fill && !zero_unused) {
        return;
    }
    const uint4 * src = fill ? reinterpret_cast<const uint4 *>(static_cast<const char *>(a.src[t]) + size_t(sel[j]) * a.bytes[t]) : nullptr;
    uint4 * dst = reinterpret_cast<uint4 *>(static_cast<char *>(a.dst[t]) + size_t(j) * a.bytes[t]);
    const size_t chunk = 1024;   // uint4 per block-z step: 16 KiB
    for (size_t base = size_t(blockIdx.z) * chunk; base < n16; base += size_t(gridDim.z) * chunk) {
        const size_t end = base + chunk < n16 ? base + chunk : n16;
        if (fill) {
            copy_u4<4>(dst, src, base, end, threadIdx.x, blockDim.x);
        } else {
            for (size_t i = base + threadIdx.x; i < end; i += blockDim.x) {
                dst[i] = make_uint4(0, 0, 0, 0);
            }
        }
    }
}

// the side stream of the fetch and its two events, made on the first call, which is before any graph is captured
static cudaStream_t stage_stream = nullptr;
static cudaEvent_t  stage_fork   = nullptr;
static cudaEvent_t  stage_join   = nullptr;

void hpcache_cuda_pcie_stage(void * stream, const int32_t * ids, const int32_t * table, int k, int n_expert, int m,
        const volatile int32_t * frac256, const volatile int8_t * gtab, const hpcache_stage_args & args, bool side, bool zero_unused) {
    // enough blocks along z to keep many PCIe reads in flight
    const dim3 grid(unsigned(m), unsigned(args.n_tensors), stage_z());
    if (!side) {
        pcie_stage_kernel<<<grid, 256, 0, (cudaStream_t) stream>>>(ids, table, k, n_expert, m, frac256, gtab, args, zero_unused);
        return;
    }
    if (stage_stream == nullptr) {
        cudaStreamCreateWithFlags(&stage_stream, cudaStreamNonBlocking);
        cudaEventCreateWithFlags(&stage_fork, cudaEventDisableTiming);
        cudaEventCreateWithFlags(&stage_join, cudaEventDisableTiming);
    }
    cudaEventRecord(stage_fork, (cudaStream_t) stream);
    cudaStreamWaitEvent(stage_stream, stage_fork, 0);
    pcie_stage_kernel<<<grid, 256, 0, stage_stream>>>(ids, table, k, n_expert, m, frac256, gtab, args, zero_unused);
    cudaEventRecord(stage_join, stage_stream);
}

struct hpcache_dev_hit {
    const int32_t * table;
    __device__ bool operator()(int e) const { return table != nullptr && table[e] >= 0; }
};

// the stage kernel of a window: the selection is made by every block, block 0 also writes the table of the staged experts
static __global__ void pcie_stage_window_kernel(const int32_t * ids, int stride, int n_tok, const int32_t * table, int k,
        int n_expert, int cap, hpcache_win_model mdl, int32_t * stage_table, hpcache_stage_args a, bool zero_unused) {
    __shared__ int32_t sel[32];
    __shared__ int g;
    if (threadIdx.x == 0) {
        g = hpcache_window_select(ids, stride, n_tok, k, n_expert, hpcache_dev_hit{ table }, cap < 32 ? cap : 32, mdl, sel);
    }
    __syncthreads();
    if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        for (int e = threadIdx.x; e < n_expert; e += blockDim.x) {
            stage_table[e] = -1;
        }
        __syncthreads();
        if (threadIdx.x < g) {
            stage_table[sel[threadIdx.x]] = threadIdx.x;
        }
    }

    const int j = blockIdx.x;
    const int t = blockIdx.y;
    if (t >= a.n_tensors) {
        return;
    }
    const size_t n16 = a.bytes[t] / 16;
    const bool fill = j < g;
    if (!fill && !zero_unused) {
        return;   // an unused slot that no op reads (miss id -1)
    }
    const uint4 * src = fill ? reinterpret_cast<const uint4 *>(static_cast<const char *>(a.src[t]) + size_t(sel[j]) * a.bytes[t]) : nullptr;
    uint4 * dst = reinterpret_cast<uint4 *>(static_cast<char *>(a.dst[t]) + size_t(j) * a.bytes[t]);
    const size_t chunk = 1024;
    for (size_t base = size_t(blockIdx.z) * chunk; base < n16; base += size_t(gridDim.z) * chunk) {
        const size_t end = base + chunk < n16 ? base + chunk : n16;
        if (fill) {
            copy_u4<4>(dst, src, base, end, threadIdx.x, blockDim.x);
        } else {
            for (size_t i = base + threadIdx.x; i < end; i += blockDim.x) {
                dst[i] = make_uint4(0, 0, 0, 0);
            }
        }
    }
}

void hpcache_cuda_pcie_stage_window(void * stream, const int32_t * ids, int stride, int n_tok, const int32_t * table, int k,
        int n_expert, int cap, const hpcache_win_model & mdl, int32_t * stage_table, const hpcache_stage_args & args, bool side,
        bool zero_unused) {
    const dim3 grid(unsigned(cap), unsigned(args.n_tensors), stage_z());
    if (!side) {
        pcie_stage_window_kernel<<<grid, 256, 0, (cudaStream_t) stream>>>(ids, stride, n_tok, table, k, n_expert, cap, mdl, stage_table, args,
                zero_unused);
        return;
    }
    if (stage_stream == nullptr) {
        cudaStreamCreateWithFlags(&stage_stream, cudaStreamNonBlocking);
        cudaEventCreateWithFlags(&stage_fork, cudaEventDisableTiming);
        cudaEventCreateWithFlags(&stage_join, cudaEventDisableTiming);
    }
    cudaEventRecord(stage_fork, (cudaStream_t) stream);
    cudaStreamWaitEvent(stage_stream, stage_fork, 0);
    pcie_stage_window_kernel<<<grid, 256, 0, stage_stream>>>(ids, stride, n_tok, table, k, n_expert, cap, mdl, stage_table, args,
            zero_unused);
    cudaEventRecord(stage_join, stage_stream);
}

static __global__ void fill_neg_inf_kernel(float * out, long long n) {
    const float v = __int_as_float(0xff800000);   // -INFINITY
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long) gridDim.x * blockDim.x) {
        out[i] = v;
    }
}

static __global__ void scatter_logits_kernel(const float * sub, const int32_t * ids, int n_sub, int n_vocab, int n_tokens, float * out) {
    const long long total = (long long) n_sub * n_tokens;
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (long long) gridDim.x * blockDim.x) {
        const long long t = i / n_sub;
        const int       j = (int) (i - t * n_sub);
        out[t * n_vocab + ids[j]] = sub[i];
    }
}

void hpcache_cuda_scatter_logits(void * stream, const float * sub, const int32_t * ids, int n_sub, int n_vocab, int n_tokens, float * out) {
    const long long n_out = (long long) n_vocab * n_tokens;
    const long long n_in  = (long long) n_sub * n_tokens;
    cudaStream_t s = (cudaStream_t) stream;
    fill_neg_inf_kernel<<<(unsigned) ((n_out + 255) / 256 < 1024 ? (n_out + 255) / 256 : 1024), 256, 0, s>>>(out, n_out);
    scatter_logits_kernel<<<(unsigned) ((n_in + 255) / 256 < 1024 ? (n_in + 255) / 256 : 1024), 256, 0, s>>>(sub, ids, n_sub, n_vocab, n_tokens, out);
}

void hpcache_cuda_pcie_stage_join(void * stream) {
    cudaStreamWaitEvent((cudaStream_t) stream, stage_join, 0);
}
