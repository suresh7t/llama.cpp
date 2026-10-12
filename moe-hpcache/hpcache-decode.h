#pragma once

#include "hpcache-cpu.h"
#include "hpcache-cuda.h"
#include "llama-moe-provider.h"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

struct hpcache_pool_options {
    bool        enabled       = true;
    size_t      pool_bytes    = 0;                 // 0: free device memory minus reserve_bytes, capped by plan_bytes
    size_t      reserve_bytes = size_t(1280) << 20; // left free for compute buffers and the desktop
    size_t      plan_bytes    = 0;                 // plan budget, the pool may exceed it by 10%
    int         n_layers      = 0;                 // MoE layers, to split an unranked pool evenly
    std::string profile;                           // Strata expert profile, ranked (layer, expert) pairs
    bool        cpu_kernel    = true;              // own expert kernel, false: a ggml graph per layer
    bool        prefault      = true;              // read all expert pages once at load, so decode never waits on the disk
    int         swap_every    = 1;                 // decode steps between admission rounds
    int         max_swaps     = 12;                // experts replaced per round
    int         fill_swaps    = 64;                // experts loaded per round into empty slots
    float       decay         = 0.9f;              // score decay per round
};

// Single-token decode of the routed experts through a doorbell: the GPU publishes the router
// output to pinned memory, a CPU thread computes the experts, and the GPU merges the result.
class hpcache_decoder {
public:
    hpcache_decoder(const llama_moe_host * host, int n_threads, const hpcache_pool_options & options);
    ~hpcache_decoder();

    hpcache_decoder(const hpcache_decoder &) = delete;
    hpcache_decoder & operator=(const hpcache_decoder &) = delete;

    bool ok() const { return cpu != nullptr; }

    // returns NULL when the layer is not supported, so that the core builds the stock path
    ggml_tensor * build_layer(const llama_moe_layer_request & req);

    void begin(const llama_moe_decode_info & info);
    void end();

    // run from the graph, on the stream of the CUDA backend
    void publish(void * stream, int il, const ggml_tensor * x, const ggml_tensor * ids, const ggml_tensor * w);
    void wait_merge(void * stream, ggml_tensor * dst);

    // userdata of the graph ops
    struct op_ctx {
        hpcache_decoder * decoder;
        int il;
        const int32_t * table;   // device slot table of the layer, NULL without a pool
    };

private:
    struct layer;

    void log(enum ggml_log_level level, const std::string & message) const;
    bool ensure_doorbell(int n_embd, int k);
    void arm();
    std::unique_ptr<layer> make_layer(const llama_moe_layer_request & req);
    struct cpu_graph;
    cpu_graph * cpu_graph_for(layer & l, int n_miss);
    void init_pool(const llama_moe_layer_request & req);
    void alloc_pool(layer & l, int il);
    bool load_slot(layer & l, int slot, int e);
    void prefault_layer(layer & l, int il);
    int64_t verify_n = 0;
    double  verify_sum = 0.0, verify_max = 0.0;
    int64_t prefault_bytes = 0;
    int64_t prefault_us = 0;
    void admit();

    void follower();
    void serve(uint32_t seq);

    const llama_moe_host * host;
    hpcache_pool_options   opts;

    // GPU expert pool, sized once at the first layer
    ggml_backend_dev_t pool_dev = nullptr;
    int      pool_device_index = -1;
    bool     pool_init = false;
    size_t   pool_budget = 0;
    size_t   pool_used = 0;
    int64_t  pool_slots = 0;
    int64_t  pool_rank_limit = 0;                 // profile pairs that fit the budget
    std::vector<std::pair<int, int>> profile;     // (layer, expert), ranked
    int      steps = 0;
    std::atomic<int64_t> n_routed{0};
    std::atomic<int64_t> n_hit{0};
    int64_t  n_swapped = 0;
    int64_t  swap_us = 0;

    ggml_backend_t cpu = nullptr;
    hpcache_get_traits_fn get_traits = nullptr;
    int n_cpu_threads = 1;
    std::unique_ptr<hpcache_cpu_pool> cpu_pool;

    hpcache_doorbell db;

    std::mutex layers_mutex;
    std::map<int, std::unique_ptr<layer>> layers;
    std::deque<op_ctx> op_ctxs;

    std::atomic<void *> stream{nullptr};
    std::atomic<int>    device{-1};

    // routing histogram per layer, written at exit when HPCACHE_ROUTING_DUMP is set
    int n_expert = 0;
    std::vector<std::vector<uint32_t>> routing;

    // timing, in microseconds
    std::atomic<int64_t> n_serve{0};
    std::atomic<int64_t> serve_us{0};
    std::atomic<int64_t> n_decode{0};
    std::atomic<int64_t> decode_us{0};
    std::atomic<int64_t> decode_start_us{0};

    std::thread             thread;
    std::mutex              mutex;
    std::condition_variable cv;
    std::atomic<bool>       armed{false};
    bool                    idle = true;
    bool                    quit = false;
};
