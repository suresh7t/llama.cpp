#include "hpcache-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <cstdio>
#include <cstring>

static std::atomic<int64_t> g_ns_b{0}, g_ns_c{0}, g_ns_d{0}, g_runs{0};
// HPCACHE_PHASES: time summed over all threads: 0 wake delay, 1 gate/up, 2 wait, 3 quantize, 4 wait, 5 down, 6 wait, 7 reduce
static const bool g_tm = getenv("HPCACHE_PHASES") != nullptr;
static std::atomic<int64_t> g_ph[8];
static inline int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define HPCACHE_PAUSE() _mm_pause()
#else
#define HPCACHE_PAUSE() std::this_thread::yield()
#endif

struct hpcache_cpu_pool::task {
    const hpcache_expert_desc * d = nullptr;
    const int32_t * ids = nullptr;
    const float *   w   = nullptr;
    int             m   = 0;
    float *         y   = nullptr;

    const uint8_t * xq_gate = nullptr;
    const uint8_t * xq_up   = nullptr;
    float *         act     = nullptr;
    uint8_t *       actq    = nullptr;
    float *         part    = nullptr;

    int64_t t_start = 0;
    std::atomic<int> next_b{0};
    std::atomic<int> next_d{0};
};

hpcache_cpu_pool::hpcache_cpu_pool(int n) : n_threads(std::max(1, n)) {
    for (int i = 1; i < n_threads; ++i) {
        threads.emplace_back([this, i] { worker(i); });
    }
}

hpcache_cpu_pool::~hpcache_cpu_pool() {
    if (g_runs.load() > 0 && getenv("HPCACHE_PHASES")) {
        fprintf(stderr, "hpcache cpu pool: %lld runs, gate/up %.1f us, quantize %.1f us, down %.1f us per run\n",
                (long long) g_runs.load(), g_ns_b.load() / 1e3 / g_runs.load(), g_ns_c.load() / 1e3 / g_runs.load(),
                g_ns_d.load() / 1e3 / g_runs.load());
        const double k = 1e-3 / double(g_runs.load()) / n_threads;
        fprintf(stderr, "hpcache cpu pool per thread and run, us: wake %.1f | gate/up work %.1f wait %.1f | quantize %.1f wait %.1f |"
                " down work %.1f wait %.1f | reduce %.1f\n", g_ph[0] * k, g_ph[1] * k, g_ph[2] * k, g_ph[3] * k, g_ph[4] * k,
                g_ph[5] * k, g_ph[6] * k, g_ph[7] * k);
    }
    quit.store(true);
    {
        std::lock_guard<std::mutex> lock(mutex);
    }
    cv.notify_all();
    for (auto & t : threads) {
        t.join();
    }
}

#if (defined(__x86_64__) || defined(_M_X64))
#if defined(__GNUC__)
#define HPCACHE_AVX2 __attribute__((target("avx2,fma,f16c")))
#else
#define HPCACHE_AVX2
#endif

#if defined(__GNUC__)
#define HPCACHE_VNNI __attribute__((target("avx2,fma,f16c,avx512vnni,avx512vl,avx512f")))
#else
#define HPCACHE_VNNI
#endif

// AVX512 VNNI with 256-bit vectors, the state of ZMM registers must be enabled by the OS too
static bool cpu_has_vnni() {
#if defined(__GNUC__)
    return __builtin_cpu_supports("avx512vnni") && __builtin_cpu_supports("avx512vl");
#else
    int info[4];
    __cpuid(info, 0);
    if (info[0] < 7) {
        return false;
    }
    __cpuidex(info, 1, 0);
    if ((info[2] & (1 << 27)) == 0 || (_xgetbv(0) & 0xE6) != 0xE6) {   // OSXSAVE, XMM/YMM/opmask/ZMM state
        return false;
    }
    __cpuidex(info, 7, 0);
    return (info[1] & (1 << 31)) != 0 && (info[1] & (1 << 16)) != 0 && (info[2] & (1 << 11)) != 0;
#endif
}

static bool cpu_has_avx2() {
#if defined(__GNUC__)
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c");
#else
    int info[4];
    __cpuid(info, 0);
    if (info[0] < 7) {
        return false;
    }
    __cpuidex(info, 1, 0);
    const bool fma = (info[2] & (1 << 12)) != 0, f16c = (info[2] & (1 << 29)) != 0;
    __cpuidex(info, 7, 0);
    return fma && f16c && (info[1] & (1 << 5)) != 0;
#endif
}

// q2_0 (64 weights, values 0..3 meaning -1..2, four per byte) times q8_0, the CPU backend has only a scalar loop
HPCACHE_AVX2 static void dot_q2_0_q8_0_avx2(int n, float * s, size_t, const void * vx, size_t, const void * vy, size_t, int) {
    struct blk2 { uint16_t d; uint8_t qs[16]; };
    struct blk8 { uint16_t d; int8_t  qs[32]; };
    const blk2 * x = static_cast<const blk2 *>(vx);
    const blk8 * y = static_cast<const blk8 *>(vy);
    const int nb = n / 64;

    const __m256i idx  = _mm256_setr_epi8(0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3, 0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3);
    const __m256i m0   = _mm256_setr_epi8(3,0,0,0,3,0,0,0,3,0,0,0,3,0,0,0, 3,0,0,0,3,0,0,0,3,0,0,0,3,0,0,0);
    const __m256i ones8  = _mm256_set1_epi8(1);
    const __m256i ones16 = _mm256_set1_epi16(1);

    // byte n of an expanded half is at position n%4 of its group, so the four masks select
    // the fields 0..3 of the replicated byte
    const __m256i f0 = m0;
    const __m256i f1 = _mm256_setr_epi8(0,3,0,0,0,3,0,0,0,3,0,0,0,3,0,0, 0,3,0,0,0,3,0,0,0,3,0,0,0,3,0,0);
    const __m256i f2 = _mm256_setr_epi8(0,0,3,0,0,0,3,0,0,0,3,0,0,0,3,0, 0,0,3,0,0,0,3,0,0,0,3,0,0,0,3,0);
    const __m256i f3 = _mm256_setr_epi8(0,0,0,3,0,0,0,3,0,0,0,3,0,0,0,3, 0,0,0,3,0,0,0,3,0,0,0,3,0,0,0,3);

    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const __m128i bits = _mm_loadu_si128((const __m128i *) x[i].qs);
        const float d0 = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(x[i].d)));

        for (int k = 0; k < 2; ++k) {
            const __m128i src = k == 0 ? bits : _mm_srli_si128(bits, 8);
            const __m256i rep = _mm256_shuffle_epi8(_mm256_set_m128i(_mm_srli_si128(src, 4), src), idx);
            const __m256i q = _mm256_or_si256(
                    _mm256_or_si256(_mm256_and_si256(rep, f0), _mm256_and_si256(_mm256_srli_epi16(rep, 2), f1)),
                    _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(rep, 4), f2), _mm256_and_si256(_mm256_srli_epi16(rep, 6), f3)));

            const __m256i yq = _mm256_loadu_si256((const __m256i *) y[2 * i + k].qs);
            const __m256i p  = _mm256_maddubs_epi16(q, yq);
            const __m256i ps = _mm256_maddubs_epi16(ones8, yq);
            const __m256i sum = _mm256_madd_epi16(_mm256_sub_epi16(p, ps), ones16);

            const float d1 = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(y[2 * i + k].d)));
            acc = _mm256_fmadd_ps(_mm256_set1_ps(d0 * d1), _mm256_cvtepi32_ps(sum), acc);
        }
    }

    const __m128 lo = _mm256_castps256_ps128(acc);
    const __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 r = _mm_add_ps(lo, hi);
    r = _mm_add_ps(r, _mm_movehl_ps(r, r));
    r = _mm_add_ss(r, _mm_shuffle_ps(r, r, 1));
    *s = _mm_cvtss_f32(r);
}

// q2_0 (64 weights, values 0..3 meaning -1..2, four per byte) times q8_0, the CPU backend has only a scalar loop
HPCACHE_VNNI static void dot_q2_0_q8_0_vnni(int n, float * s, size_t, const void * vx, size_t, const void * vy, size_t, int) {
    struct blk2 { uint16_t d; uint8_t qs[16]; };
    struct blk8 { uint16_t d; int8_t  qs[32]; };
    const blk2 * x = static_cast<const blk2 *>(vx);
    const blk8 * y = static_cast<const blk8 *>(vy);
    const int nb = n / 64;

    const __m256i idx  = _mm256_setr_epi8(0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3, 0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3);
    const __m256i m0   = _mm256_setr_epi8(3,0,0,0,3,0,0,0,3,0,0,0,3,0,0,0, 3,0,0,0,3,0,0,0,3,0,0,0,3,0,0,0);
    const __m256i ones8  = _mm256_set1_epi8(1);

    // byte n of an expanded half is at position n%4 of its group, so the four masks select
    // the fields 0..3 of the replicated byte
    const __m256i f0 = m0;
    const __m256i f1 = _mm256_setr_epi8(0,3,0,0,0,3,0,0,0,3,0,0,0,3,0,0, 0,3,0,0,0,3,0,0,0,3,0,0,0,3,0,0);
    const __m256i f2 = _mm256_setr_epi8(0,0,3,0,0,0,3,0,0,0,3,0,0,0,3,0, 0,0,3,0,0,0,3,0,0,0,3,0,0,0,3,0);
    const __m256i f3 = _mm256_setr_epi8(0,0,0,3,0,0,0,3,0,0,0,3,0,0,0,3, 0,0,0,3,0,0,0,3,0,0,0,3,0,0,0,3);

    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const __m128i bits = _mm_loadu_si128((const __m128i *) x[i].qs);
        const float d0 = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(x[i].d)));

        for (int k = 0; k < 2; ++k) {
            const __m128i src = k == 0 ? bits : _mm_srli_si128(bits, 8);
            const __m256i rep = _mm256_shuffle_epi8(_mm256_set_m128i(_mm_srli_si128(src, 4), src), idx);
            const __m256i q = _mm256_or_si256(
                    _mm256_or_si256(_mm256_and_si256(rep, f0), _mm256_and_si256(_mm256_srli_epi16(rep, 2), f1)),
                    _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(rep, 4), f2), _mm256_and_si256(_mm256_srli_epi16(rep, 6), f3)));

            const __m256i yq = _mm256_loadu_si256((const __m256i *) y[2 * i + k].qs);
            const __m256i zero = _mm256_setzero_si256();
            const __m256i sum = _mm256_sub_epi32(_mm256_dpbusd_epi32(zero, q, yq), _mm256_dpbusd_epi32(zero, ones8, yq));

            const float d1 = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(y[2 * i + k].d)));
            acc = _mm256_fmadd_ps(_mm256_set1_ps(d0 * d1), _mm256_cvtepi32_ps(sum), acc);
        }
    }

    const __m128 lo = _mm256_castps256_ps128(acc);
    const __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 r = _mm_add_ps(lo, hi);
    r = _mm_add_ps(r, _mm_movehl_ps(r, r));
    r = _mm_add_ss(r, _mm_shuffle_ps(r, r, 1));
    *s = _mm_cvtss_f32(r);
}


#if defined(__GNUC__)
#define HPCACHE_VBMI __attribute__((target("avx2,fma,f16c,avx512f,avx512bw,avx512vl,avx512vnni,avx512vbmi")))
#else
#define HPCACHE_VBMI
#endif

static bool cpu_has_vbmi_vnni() {
#if defined(__GNUC__)
    return __builtin_cpu_supports("avx512vnni") && __builtin_cpu_supports("avx512vbmi") &&
            __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512vl") && cpu_has_avx2();
#else
    int info[4];
    __cpuid(info, 0);
    if (info[0] < 7) {
        return false;
    }
    __cpuidex(info, 1, 0);
    if ((info[2] & (1 << 27)) == 0 || (_xgetbv(0) & 0xE6) != 0xE6) {
        return false;
    }
    __cpuidex(info, 7, 0);
    const int need_b = (1 << 16) | (1 << 30) | (1 << 31);   // F, BW, VL
    return (info[1] & need_b) == need_b && (info[2] & (1 << 11)) != 0 && (info[2] & (1 << 1)) != 0 && cpu_has_avx2();   // VNNI, VBMI
#endif
}

// q8_0 activation row followed, per 64 weights, by 16 floats (the scale of each 32-block repeated 8 times) and
// 16 int32 (the sum of every group of 4 activations, one per dpbusd lane): computed once per activation, they
// spare the q2_0 dot its scale conversions and the -1 offset sum, which it would otherwise redo for every row
static ggml_from_float_t g_q8_0_from_float = nullptr;

static void quantize_q8_0_with_sums(const float * x, void * y, int64_t k) {
    g_q8_0_from_float(x, y, k);
    const int nb64 = (int) (k / 64);
    uint8_t * base = static_cast<uint8_t *>(y);
    uint8_t * tail = base + size_t(k / 32) * 34;
    for (int i = 0; i < nb64; ++i) {
        float   scale[16];
        int32_t sum[16];
        for (int h = 0; h < 2; ++h) {
            const uint8_t * blk = base + size_t(2 * i + h) * 34;
            uint16_t d16;
            memcpy(&d16, blk, 2);
            const int8_t * q = reinterpret_cast<const int8_t *>(blk + 2);
            for (int j = 0; j < 8; ++j) {
                scale[8 * h + j] = ggml_fp16_to_fp32(d16);
                sum[8 * h + j]   = q[4 * j] + q[4 * j + 1] + q[4 * j + 2] + q[4 * j + 3];   // what dpbusd(ones, q) gives
            }
        }
        memcpy(tail + size_t(i) * 128, scale, 64);
        memcpy(tail + size_t(i) * 128 + 64, sum, 64);
    }
}

// q2_0 x q8_0 with the VNNI/VBMI instructions on 256-bit vectors (512-bit ones were slower here: the core drops
// its clock, which also slows the next layer). The activation row comes from quantize_q8_0_with_sums.
HPCACHE_VBMI static void dot_q2_0_q8_0_ymm(int n, float * s, size_t, const void * vx, size_t, const void * vy, size_t, int) {
    struct blk2 { uint16_t d; uint8_t qs[16]; };
    const blk2 *    x    = static_cast<const blk2 *>(vx);
    const uint8_t * y    = static_cast<const uint8_t *>(vy);
    const uint8_t * tail = y + size_t(n / 32) * 34;
    const int nb = n / 64;

    const __m256i ctrl = _mm256_set1_epi64x((long long) 0x0E0C0A0806040200ULL);
    const __m256i c3   = _mm256_set1_epi8(3);
    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const uint8_t * ya = y + size_t(2 * i) * 34 + 2;
        const uint8_t * yb = ya + 34;
        const uint8_t * t  = tail + size_t(i) * 128;

        const __m128i bits = _mm_loadu_si128(reinterpret_cast<const __m128i *>(x[i].qs));
        const __m256i w0 = _mm256_and_si256(_mm256_multishift_epi64_epi8(ctrl, _mm256_cvtepu16_epi64(bits)), c3);
        const __m256i w1 = _mm256_and_si256(_mm256_multishift_epi64_epi8(ctrl, _mm256_cvtepu16_epi64(_mm_srli_si128(bits, 8))), c3);
        __m256i d0v = _mm256_dpbusd_epi32(_mm256_setzero_si256(), w0, _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ya)));
        __m256i d1v = _mm256_dpbusd_epi32(_mm256_setzero_si256(), w1, _mm256_loadu_si256(reinterpret_cast<const __m256i *>(yb)));
        d0v = _mm256_sub_epi32(d0v, _mm256_loadu_si256(reinterpret_cast<const __m256i *>(t + 64)));
        d1v = _mm256_sub_epi32(d1v, _mm256_loadu_si256(reinterpret_cast<const __m256i *>(t + 96)));

        uint16_t h;
        memcpy(&h, &x[i].d, 2);
        const __m256 dd = _mm256_set1_ps(_mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h))));
        acc = _mm256_fmadd_ps(_mm256_mul_ps(dd, _mm256_loadu_ps(reinterpret_cast<const float *>(t))), _mm256_cvtepi32_ps(d0v), acc);
        acc = _mm256_fmadd_ps(_mm256_mul_ps(dd, _mm256_loadu_ps(reinterpret_cast<const float *>(t + 32))), _mm256_cvtepi32_ps(d1v), acc);
    }
    const __m128 r0 = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    const __m128 r1 = _mm_add_ps(r0, _mm_movehl_ps(r0, r0));
    *s = _mm_cvtss_f32(_mm_add_ss(r1, _mm_shuffle_ps(r1, r1, 1)));
}
#endif

bool hpcache_cpu_pool::describe(hpcache_expert_desc & d, ggml_type gate, ggml_type up, ggml_type down,
        hpcache_get_traits_fn get_traits) {
    if (get_traits == nullptr) {
        return false;
    }
    struct entry {
        ggml_type type;
        ggml_vec_dot_t * dot;
        ggml_from_float_t * quant;
        size_t * xq_bytes;
        int64_t n;
    };
    const entry entries[3] = {
        { gate, &d.gate_dot, &d.gate_quant, &d.gate_xq_bytes, d.n_embd },
        { up,   &d.up_dot,   &d.up_quant,   &d.up_xq_bytes,   d.n_embd },
        { down, &d.down_dot, &d.down_quant, &d.down_xq_bytes, d.n_ff   },
    };
    for (const entry & e : entries) {
        const ggml_type_traits_cpu * t = get_traits(e.type);
        if (t == nullptr || t->vec_dot == nullptr || t->nrows != 1) {
            return false;
        }
        const ggml_type_traits_cpu * q = get_traits(t->vec_dot_type);
        if (q == nullptr || q->from_float == nullptr || e.n % ggml_blck_size(t->vec_dot_type) != 0) {
            return false;
        }
        *e.dot   = t->vec_dot;
        *e.quant = q->from_float;
        *e.xq_bytes = ggml_row_size(t->vec_dot_type, e.n);
#if defined(__x86_64__) || defined(_M_X64)
        if (e.type == GGML_TYPE_Q2_0 && cpu_has_avx2()) {
            *e.dot = (cpu_has_vnni() && !getenv("HPCACHE_NO_VNNI")) ? dot_q2_0_q8_0_vnni : dot_q2_0_q8_0_avx2;
            if (e.n % 64 == 0 && t->vec_dot_type == GGML_TYPE_Q8_0 && cpu_has_vbmi_vnni() && !getenv("HPCACHE_NO_VBMI")) {
                g_q8_0_from_float = q->from_float;
                *e.dot   = dot_q2_0_q8_0_ymm;
                *e.quant = quantize_q8_0_with_sums;
                *e.xq_bytes += size_t(e.n / 64) * 128;
            }
        }
#endif
    }
    return true;
}

void hpcache_cpu_pool::set_active(bool a) {
    {
        std::lock_guard<std::mutex> lock(mutex);
        active.store(a, std::memory_order_release);
    }
    cv.notify_all();
}

void hpcache_cpu_pool::worker(int ith) {
    uint32_t seen = 0;
    uint32_t spins = 0;

    while (!quit.load(std::memory_order_relaxed)) {
        const uint32_t gen = job_gen.load(std::memory_order_acquire);
        if (gen != seen) {
            seen = gen;
            task * t = current.load(std::memory_order_acquire);
            execute(*t, ith);
            done.fetch_add(1, std::memory_order_acq_rel);
            spins = 0;
            continue;
        }

        HPCACHE_PAUSE();
        if ((++spins & 0xfffffu) == 0 && !active.load(std::memory_order_acquire)) {
            std::unique_lock<std::mutex> lock(mutex);
            cv.wait(lock, [&] { return active.load(std::memory_order_acquire) || quit.load(); });
        }
    }
}

void hpcache_cpu_pool::barrier() {
    const uint32_t gen = bar_gen.load(std::memory_order_acquire);
    if (bar_count.fetch_add(1, std::memory_order_acq_rel) + 1 == n_threads) {
        bar_count.store(0, std::memory_order_relaxed);
        bar_gen.fetch_add(1, std::memory_order_release);
    } else {
        while (bar_gen.load(std::memory_order_acquire) == gen) {
            HPCACHE_PAUSE();
        }
    }
}

static inline float silu(float v) {
    return v / (1.0f + expf(-v));
}

#if defined(__x86_64__) || defined(_M_X64)
#define HPCACHE_PREFETCH(p) _mm_prefetch((const char *) (p), _MM_HINT_T0)
#else
#define HPCACHE_PREFETCH(p) ((void) (p))
#endif

static inline void prefetch_row(const char * p, size_t bytes) {
    for (size_t o = 0; o < bytes; o += 64) {
        HPCACHE_PREFETCH(p + o);
    }
}

// rows of one expert are streamed in blocks, so that the hardware sees long sequential runs and
// the next rows are already on their way when a dot product starts
constexpr int RB_GATE = 16;
constexpr int RB_DOWN = 64;

void hpcache_cpu_pool::execute(task & t, int ith) {
    const hpcache_expert_desc & d = *t.d;
    const int m = t.m;
    const int64_t n_ff  = d.n_ff;
    const int64_t n_out = d.n_embd;
    const int64_t t0 = ith == 0 ? now_ns() : 0;
    int64_t c0 = 0;
    if (g_tm) {
        c0 = now_ns();
        g_ph[0] += c0 - t.t_start;
    }

    // gate and up rows of every routed expert, with the activation applied
    const int blocks_b = (int) ((n_ff + RB_GATE - 1) / RB_GATE);
    const int items_b  = m * blocks_b;
    for (;;) {
        const int item = t.next_b.fetch_add(1, std::memory_order_relaxed);
        if (item >= items_b) {
            break;
        }
        const int     i    = item / blocks_b;
        const int64_t row0 = int64_t(item % blocks_b) * RB_GATE;
        const int64_t row1 = std::min<int64_t>(row0 + RB_GATE, n_ff);
        const size_t  e    = (size_t) t.ids[i];
        const char * gp = d.gate + e * d.gate_exp + size_t(row0) * d.gate_row;
        const char * up = d.up   + e * d.up_exp   + size_t(row0) * d.up_row;
        for (int64_t row = row0; row < row1; ++row, gp += d.gate_row, up += d.up_row) {
            prefetch_row(gp + 2 * d.gate_row, d.gate_row);
            prefetch_row(up + 2 * d.up_row, d.up_row);
            float g = 0.0f, u = 0.0f;
            d.gate_dot((int) d.n_embd, &g, 0, gp, 0, t.xq_gate, 0, 1);
            d.up_dot((int) d.n_embd, &u, 0, up, 0, t.xq_up, 0, 1);
            float a;
            if (d.clamp > 1e-6f) {
                a = std::min(silu(g), d.clamp) * std::max(-d.clamp, std::min(u, d.clamp));
            } else {
                a = silu(g) * u;
            }
            t.act[int64_t(i) * n_ff + row] = a;
        }
    }
    const int64_t t1 = ith == 0 ? now_ns() : 0;
    int64_t c1 = 0, c2 = 0;
    if (g_tm) {
        c1 = now_ns();
        g_ph[1] += c1 - c0;
    }
    barrier();
    const int64_t t1b = ith == 0 ? now_ns() : 0;
    if (g_tm) {
        c2 = now_ns();
        g_ph[2] += c2 - c1;
    }

    // the activations in the vec_dot type of the down tensor
    for (int i = ith; i < m; i += n_threads) {
        d.down_quant(t.act + int64_t(i) * n_ff, t.actq + size_t(i) * d.down_xq_bytes, n_ff);
    }
    int64_t c3 = 0, c4 = 0;
    if (g_tm) {
        c3 = now_ns();
        g_ph[3] += c3 - c2;
    }
    barrier();
    const int64_t t2 = ith == 0 ? now_ns() : 0;
    if (g_tm) {
        c4 = now_ns();
        g_ph[4] += c4 - c3;
    }

    // down rows per expert, weighted by the router
    const int blocks_d = (int) ((n_out + RB_DOWN - 1) / RB_DOWN);
    const int items_d  = m * blocks_d;
    for (;;) {
        const int item = t.next_d.fetch_add(1, std::memory_order_relaxed);
        if (item >= items_d) {
            break;
        }
        const int     i    = item / blocks_d;
        const int64_t row0 = int64_t(item % blocks_d) * RB_DOWN;
        const int64_t row1 = std::min<int64_t>(row0 + RB_DOWN, n_out);
        const size_t  e    = (size_t) t.ids[i];
        const char *  dp   = d.down + e * d.down_exp + size_t(row0) * d.down_row;
        const uint8_t * aq = t.actq + size_t(i) * d.down_xq_bytes;
        const float   wi   = t.w[i];
        float * part = t.part + int64_t(i) * n_out;
        for (int64_t row = row0; row < row1; ++row, dp += d.down_row) {
            prefetch_row(dp + 4 * d.down_row, d.down_row);
            float v = 0.0f;
            d.down_dot((int) n_ff, &v, 0, dp, 0, aq, 0, 1);
            part[row] = wi * v;
        }
    }
    int64_t c5 = 0, c6 = 0;
    if (g_tm) {
        c5 = now_ns();
        g_ph[5] += c5 - c4;
    }
    barrier();
    if (g_tm) {
        c6 = now_ns();
        g_ph[6] += c6 - c5;
    }

    // sum over the experts
    for (int64_t r = ith * 64; r < n_out; r += int64_t(n_threads) * 64) {
        const int64_t r1 = std::min<int64_t>(r + 64, n_out);
        for (int64_t row = r; row < r1; ++row) {
            float acc = 0.0f;
            for (int i = 0; i < m; ++i) {
                acc += t.part[int64_t(i) * n_out + row];
            }
            t.y[row] = acc;
        }
    }
    if (g_tm) {
        g_ph[7] += now_ns() - c6;
    }

    if (ith == 0) {
        const int64_t t3 = now_ns();
        g_ns_b += t1b - t0;
        g_ns_c += t2 - t1b;
        g_ns_d += t3 - t2;
        g_runs++;
    }
    (void) t1;
}

void hpcache_cpu_pool::run(const hpcache_expert_desc & d, const float * x, const int32_t * ids, const float * w,
        int m, float * y) {
    if (xq_gate.size() < d.gate_xq_bytes) {
        xq_gate.resize(d.gate_xq_bytes);
    }
    if (xq_up.size() < d.up_xq_bytes) {
        xq_up.resize(d.up_xq_bytes);
    }
    if (actq.size() < d.down_xq_bytes * 32) {
        actq.resize(d.down_xq_bytes * 32);
    }
    if (act.size() < size_t(32) * d.n_ff) {
        act.resize(size_t(32) * d.n_ff);
    }
    if (part.size() < size_t(32) * d.n_embd) {
        part.resize(size_t(32) * d.n_embd);
    }

    d.gate_quant(x, xq_gate.data(), d.n_embd);
    d.up_quant(x, xq_up.data(), d.n_embd);

    task t;
    t.d   = &d;
    t.ids = ids;
    t.w   = w;
    t.m   = m;
    t.y   = y;
    t.xq_gate = xq_gate.data();
    t.xq_up   = xq_up.data();
    t.act  = act.data();
    t.actq = actq.data();
    t.part = part.data();

    // wake workers that are parked, a job is only joined by running threads
    if (!active.load(std::memory_order_acquire)) {
        set_active(true);
    }

    done.store(0, std::memory_order_relaxed);
    bar_count.store(0, std::memory_order_relaxed);
    t.t_start = g_tm ? now_ns() : 0;
    current.store(&t, std::memory_order_release);
    job_gen.fetch_add(1, std::memory_order_acq_rel);

    execute(t, 0);

    const int others = n_threads - 1;
    while (done.load(std::memory_order_acquire) < others) {
        HPCACHE_PAUSE();
    }
}
