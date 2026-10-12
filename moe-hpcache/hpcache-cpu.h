#pragma once

#include "ggml.h"
#include "ggml-cpu.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

// One layer's routed experts: gate, up and down tensors in their file layout
struct hpcache_expert_desc {
    const char * gate = nullptr;
    const char * up   = nullptr;
    const char * down = nullptr;
    size_t gate_row = 0, up_row = 0, down_row = 0;   // bytes of one row
    size_t gate_exp = 0, up_exp = 0, down_exp = 0;   // bytes of one expert
    int64_t n_embd = 0;
    int64_t n_ff   = 0;
    float   clamp  = 0.0f;                           // swiglu clamp, 0 when unused

    ggml_vec_dot_t   gate_dot = nullptr;
    ggml_vec_dot_t   up_dot   = nullptr;
    ggml_vec_dot_t   down_dot = nullptr;
    ggml_from_float_t gate_quant = nullptr;          // activation to the vec_dot type of gate
    ggml_from_float_t up_quant   = nullptr;
    ggml_from_float_t down_quant = nullptr;
    size_t gate_xq_bytes = 0, up_xq_bytes = 0, down_xq_bytes = 0;   // bytes of the quantized activation rows
};

typedef const ggml_type_traits_cpu * (*hpcache_get_traits_fn)(ggml_type);

// A small fork-join pool of spinning threads. The caller of run() is thread 0.
// Threads park when the pool is not active.
class hpcache_cpu_pool {
public:
    explicit hpcache_cpu_pool(int n_threads);
    ~hpcache_cpu_pool();

    hpcache_cpu_pool(const hpcache_cpu_pool &) = delete;
    hpcache_cpu_pool & operator=(const hpcache_cpu_pool &) = delete;

    // fills the function pointers of desc, false when a type has no usable CPU dot product
    static bool describe(hpcache_expert_desc & desc, ggml_type gate, ggml_type up, ggml_type down,
            hpcache_get_traits_fn get_traits);

    void set_active(bool active);

    // y = sum_i w[i] * down_{ids[i]}(act(gate_{ids[i]}(x), up_{ids[i]}(x))), m <= 32 experts
    void run(const hpcache_expert_desc & d, const float * x, const int32_t * ids, const float * w, int m, float * y);

private:
    struct task;
    void worker(int ith);
    void execute(task & t, int ith);
    void barrier();

    int n_threads;
    std::vector<std::thread> threads;

    std::atomic<uint32_t> job_gen{0};
    std::atomic<int>      done{0};
    std::atomic<task *>   current{nullptr};

    std::atomic<int>      bar_count{0};
    std::atomic<uint32_t> bar_gen{0};

    std::mutex              mutex;
    std::condition_variable cv;
    std::atomic<bool>       active{false};
    std::atomic<bool>       quit{false};

    // scratch, owned by the caller of run()
    std::vector<uint8_t> xq_gate, xq_up, actq;
    std::vector<float>   act, part;
};
