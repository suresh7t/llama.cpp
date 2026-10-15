#pragma once

#include "hpcache-cpu.h"
#include "hpcache-arena.h"
#include "hpcache-cuda.h"
#include "llama-moe-provider.h"

#include <atomic>
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
    int         max_swaps     = 24;                // experts replaced per round
    int         fill_swaps    = 64;                // experts loaded per round into empty slots
    float       decay         = 0.9f;              // score decay per round
    bool        async_swaps   = true;              // copy replaced experts during the next step, publish them one step later
    bool        pcie          = true;              // the GPU computes a share of the missed experts from the arena, over PCIe
    int         pcie_max      = 3;                 // most experts per layer on that path
    float       pcie_frac     = 0.55f;             // start share of the missed experts of a layer
    bool        pcie_adapt    = true;              // move the share toward balance from the measured GPU wait
    bool        arena         = true;              // read the expert tensors from the GGUF into one page-locked region
    size_t      pin_bytes     = 0;                 // most that is page-locked, 0 = all of it
    size_t      arena_bytes   = 0;                 // cap of that region, 0 = the size of the expert weights
    bool        spare_slot    = false;             // one unmapped slot per layer receives the copy, the victim stays mapped until it lands
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
        int n_expert = 0;
        int pcie_m = 0;          // experts on the PCIe path, 0 = none
        const volatile int32_t * pcie_frac = nullptr;   // device address of the share, read by the kernels
        const hpcache_stage_args * stage = nullptr;
        int32_t miss_id = 0;     // id of a missed expert in the pool ops: 0, or -1 when the matvec skips it
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
    bool copy_slot(layer & l, int slot, int e);
    bool load_slot(layer & l, int slot, int e);
    void publish_pending();
    void init_arena();
    // slot tables: copy a layer's table to the pinned mirror, and upload everything that changed in one copy
    void stage_table(layer & l);
    void flush_tables();
    hpcache_table_block tables;
    int      tables_dirty_upto = 0;
    bool attach_arena(layer & l);
    bool init_pcie(layer & l);
    ggml_backend_buffer_t pcie_buf = nullptr;   // device staging slots shared by all layers, pcie_max per tensor kind
    size_t   pcie_slot_bytes = 0;               // size of one staging slot
    int      pcie_alias = -1;                   // 1 when the device address of the arena is its host address
    int64_t  n_pcie_experts = 0;
    // copier timing: how long a batch of swaps takes to land, and how long the next round waits for it
    int64_t  copy_batches = 0, copy_us_sum = 0, copy_us_max = 0;
    int64_t  pub_waits = 0, pub_blocked = 0, pub_wait_us_sum = 0, pub_wait_us_max = 0;
    // balance of the two expert paths: the share moves so that the GPU waits a little for the CPU
    void     adapt_share();
    float    share_now = 0.55f;
    int64_t  adapt_last_decode = 0, adapt_last_serve = 0;
    unsigned long long adapt_last_wait_ns = 0;
    double   last_wait_us_per_layer = 0.0;
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
    std::unique_ptr<hpcache_arena> arena;
    bool     arena_checked = false;
    int64_t  n_arena_tensors = 0;
    int      steps = 0;
    std::atomic<int64_t> n_routed{0};
    std::atomic<int64_t> n_hit{0};
    int64_t  n_swapped = 0;
    int64_t  swap_us = 0;

    // copies queued by the previous admission round, mapped once the data has landed
    struct pending_swap {
        layer * l;
        int     slot;
        int     expert;
        int     victim = -1;   // slot unmapped when the copy is published, becomes the layer's spare
    };
    std::vector<pending_swap> pending;

    // copies of the queued swaps run here: the source is pageable memory, so each copy blocks its caller
    void copier_main();
    void copier_submit(const std::vector<pending_swap> & jobs);
    void copier_wait();
    std::thread             copier;
    hpcache_sync::spinlock  copier_lock;     // copier_jobs and the copy statistics
    hpcache_sync::signal    copier_work;     // jobs queued or quit
    hpcache_sync::signal    copier_done;     // the queued jobs have landed
    std::vector<pending_swap> copier_jobs;
    std::atomic<bool>       copier_has_jobs{false};
    std::atomic<bool>       copier_busy{false};
    std::atomic<bool>       copier_quit{false};

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
    hpcache_sync::signal    follower_sig;    // armed, idle or quit changed
    std::atomic<bool>       armed{false};
    std::atomic<bool>       idle{true};
    std::atomic<bool>       quit{false};
};
