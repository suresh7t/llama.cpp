#include "hpcache-decode.h"

#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static bool trace_enabled() {
    static const bool enabled = getenv("HPCACHE_TRACE") != nullptr;
    return enabled;
}

#define HPCACHE_TRACE(...) \
    do { \
        if (trace_enabled()) { \
            fprintf(stderr, "hpcache: " __VA_ARGS__); \
            fputc('\n', stderr); \
            fflush(stderr); \
        } \
    } while (0)

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define HPCACHE_PAUSE() _mm_pause()
#else
#define HPCACHE_PAUSE() std::this_thread::yield()
#endif

// a CPU graph for a fixed number of routed experts
struct hpcache_decoder::cpu_graph {
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_cgraph *         gf  = nullptr;

    ggml_tensor * x   = nullptr;
    ggml_tensor * ids = nullptr;
    ggml_tensor * w   = nullptr;
    ggml_tensor * out = nullptr;

    ~cpu_graph() {
        if (buf != nullptr) {
            ggml_backend_buffer_free(buf);
        }
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct hpcache_decoder::layer {
    // expert tensors that the graphs read, to notice a different model
    ggml_tensor * t_gate    = nullptr;
    ggml_tensor * t_up      = nullptr;
    ggml_tensor * t_down    = nullptr;
    ggml_tensor * t_gate_up = nullptr;

    int   n_embd   = 0;
    int   k        = 0;
    int   n_expert = 0;
    float clamp    = 0.0f;

    // CPU graphs by number of experts, built when first needed
    std::vector<std::unique_ptr<cpu_graph>> graphs;

    // GPU expert pool
    ggml_context *        pool_ctx = nullptr;
    ggml_backend_buffer_t pool_buf = nullptr;
    ggml_tensor *         p_gate    = nullptr;
    ggml_tensor *         p_up      = nullptr;
    ggml_tensor *         p_down    = nullptr;
    ggml_tensor *         p_gate_up = nullptr;
    int                   n_slots = 0;
    std::vector<int32_t>  slot_expert;   // expert held by each slot, -1 when empty
    std::vector<int32_t>  slot_of;       // slot of each expert, -1 when not resident
    std::vector<float>    score;         // recent routing weight of each expert
    int32_t *             d_table = nullptr;

    // own CPU kernel, valid when fast
    hpcache_expert_desc   desc;
    bool                  fast = false;

    ~layer() {
        hpcache_cuda_table_free(d_table);
        if (pool_buf != nullptr) {
            ggml_backend_buffer_free(pool_buf);
        }
        if (pool_ctx != nullptr) {
            ggml_free(pool_ctx);
        }
    }
};

hpcache_decoder::hpcache_decoder(const llama_moe_host * host, int n_threads, const hpcache_pool_options & options) :
        host(host), opts(options) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (dev == nullptr) {
        return;
    }
    cpu = ggml_backend_dev_init(dev, nullptr);
    if (cpu == nullptr) {
        return;
    }

    typedef void (*set_n_threads_t)(ggml_backend_t, int);
    auto * reg = ggml_backend_dev_backend_reg(dev);
    auto * set_n_threads = (set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    if (set_n_threads != nullptr) {
        set_n_threads(cpu, n_threads);
    }

    n_cpu_threads = n_threads;
    get_traits = (hpcache_get_traits_fn) ggml_backend_reg_get_proc_address(reg, "ggml_get_type_traits_cpu");
    if (opts.cpu_kernel && get_traits != nullptr) {
        cpu_pool = std::make_unique<hpcache_cpu_pool>(n_threads);
    }
}

hpcache_decoder::~hpcache_decoder() {
    if (n_decode.load() > 0 && n_serve.load() > 0) {
        char buf[256];
        snprintf(buf, sizeof(buf),
                "decode: %lld steps, %.1f ms/step (wall from begin to end); CPU experts: %lld layers, %.3f ms/layer, %.1f ms/step",
                (long long) n_decode.load(), decode_us.load() / 1000.0 / n_decode.load(),
                (long long) n_serve.load(), serve_us.load() / 1000.0 / n_serve.load(),
                serve_us.load() / 1000.0 / n_decode.load());
        log(GGML_LOG_LEVEL_INFO, buf);
    }
    if (verify_n > 0) {
        char buf[160];
        snprintf(buf, sizeof(buf), "verify: %lld layers, relative L2 difference of own kernel vs ggml graph: mean %.3g, max %.3g",
                (long long) verify_n, verify_sum / verify_n, verify_max);
        log(GGML_LOG_LEVEL_INFO, buf);
    }
    if (prefault_bytes > 0) {
        char buf[160];
        snprintf(buf, sizeof(buf), "prefault: %.1f GiB of expert weights read in %.1f s",
                prefault_bytes / 1073741824.0, prefault_us / 1e6);
        log(GGML_LOG_LEVEL_INFO, buf);
    }
    if (n_routed.load() > 0) {
        char buf[256];
        snprintf(buf, sizeof(buf),
                "GPU expert pool: %lld of %lld routed experts served from the pool (%.1f%%), %lld swaps in %.1f ms",
                (long long) n_hit.load(), (long long) n_routed.load(), 100.0 * n_hit.load() / n_routed.load(),
                (long long) n_swapped, swap_us / 1000.0);
        log(GGML_LOG_LEVEL_INFO, buf);
    }
    {
        std::lock_guard<std::mutex> lock(mutex);
        quit = true;
        armed.store(false);
    }
    cv.notify_all();
    if (thread.joinable()) {
        thread.join();
    }

    if (const char * path = getenv("HPCACHE_ROUTING_DUMP")) {
        if (FILE * f = fopen(path, "w")) {
            fprintf(f, "layer,expert,count\n");
            for (size_t il = 0; il < routing.size(); ++il) {
                for (size_t e = 0; e < routing[il].size(); ++e) {
                    if (routing[il][e] > 0) {
                        fprintf(f, "%zu,%zu,%u\n", il, e, routing[il][e]);
                    }
                }
            }
            fclose(f);
        }
    }

    layers.clear();
    hpcache_doorbell_free(db);
    if (cpu != nullptr) {
        ggml_backend_free(cpu);
    }
}

void hpcache_decoder::log(enum ggml_log_level level, const std::string & message) const {
    if (host != nullptr && host->log != nullptr) {
        host->log(host->ctx, level, ("moe-hpcache: " + message + "\n").c_str());
    }
}

bool hpcache_decoder::ensure_doorbell(int n_embd, int k) {
    if (db.block != nullptr) {
        return db.n_embd == n_embd && db.k == k;
    }
    std::string error;
    if (!hpcache_doorbell_init(db, n_embd, k, error)) {
        log(GGML_LOG_LEVEL_ERROR, "doorbell: " + error);
        return false;
    }
    return true;
}

// the math of the expert part of build_moe_ffn, for m routed experts of one token
static ggml_tensor * build_experts(ggml_context * ctx, ggml_tensor * gate_up_exps, ggml_tensor * gate_exps,
        ggml_tensor * up_exps, ggml_tensor * down_exps, ggml_tensor * x, ggml_tensor * ids, ggml_tensor * w,
        int n_embd, int m, float clamp) {
    ggml_tensor * gate = nullptr;
    ggml_tensor * up   = nullptr;
    if (gate_up_exps != nullptr) {
        ggml_tensor * gate_up = ggml_mul_mat_id(ctx, gate_up_exps, x, ids);
        const int64_t n_ff = gate_up->ne[0] / 2;
        gate = ggml_view_3d(ctx, gate_up, n_ff, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], 0);
        up   = ggml_view_3d(ctx, gate_up, n_ff, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], n_ff * gate_up->nb[0]);
    } else {
        up   = ggml_mul_mat_id(ctx, up_exps,   x, ids);
        gate = ggml_mul_mat_id(ctx, gate_exps, x, ids);
    }

    ggml_tensor * act = nullptr;
    constexpr float eps = 1e-6f;
    if (gate_exps != nullptr && clamp > eps) {
        up   = ggml_clamp(ctx, up, -clamp, clamp);
        gate = ggml_clamp(ctx, ggml_silu(ctx, gate), -INFINITY, clamp);
        act  = ggml_mul(ctx, gate, up);
    } else {
        act = ggml_swiglu_split(ctx, gate, up);
    }

    ggml_tensor * experts = ggml_mul_mat_id(ctx, down_exps, act, ids);
    experts = ggml_mul(ctx, experts, w);

    // sum over the routed experts in two nodes, every node costs a thread barrier on the CPU
    ggml_tensor * by_row = ggml_cont(ctx, ggml_transpose(ctx, ggml_reshape_2d(ctx, experts, n_embd, m))); // [m, n_embd]
    return ggml_reshape_2d(ctx, ggml_sum_rows(ctx, by_row), n_embd, 1);
}

hpcache_decoder::cpu_graph * hpcache_decoder::cpu_graph_for(layer & l, int m) {
    if (m < 1 || m > l.k) {
        return nullptr;
    }
    if ((int) l.graphs.size() <= l.k) {
        l.graphs.resize(l.k + 1);
    }
    if (l.graphs[m]) {
        return l.graphs[m].get();
    }

    auto g = std::make_unique<cpu_graph>();
    const size_t graph_size = 64 + 4 * size_t(m);
    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * graph_size + ggml_graph_overhead_custom(graph_size, false),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    g->ctx = ggml_init(params);
    if (g->ctx == nullptr) {
        return nullptr;
    }
    ggml_context * ctx = g->ctx;

    g->x   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, l.n_embd, 1, 1);
    g->ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, m, 1);
    g->w   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, m, 1);
    g->out = build_experts(ctx, l.t_gate_up, l.t_gate, l.t_up, l.t_down, g->x, g->ids, g->w, l.n_embd, m, l.clamp);

    g->gf = ggml_new_graph_custom(ctx, graph_size, false);
    ggml_build_forward_expand(g->gf, g->out);

    g->buf = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    if (g->buf == nullptr) {
        return nullptr;
    }
    l.graphs[m] = std::move(g);
    return l.graphs[m].get();
}

// a Strata expert profile: "STRP", five u32 (version, layers, experts, slots, pairs), then u16 pairs
static bool read_profile(const std::string & path, std::vector<std::pair<int, int>> & ranked) {
    FILE * f = fopen(path.c_str(), "rb");
    if (f == nullptr) {
        return false;
    }
    char magic[4];
    uint32_t head[5];
    bool ok = fread(magic, 1, 4, f) == 4 && memcmp(magic, "STRP", 4) == 0 && fread(head, 4, 5, f) == 5;
    if (ok) {
        const uint32_t n = head[4];
        std::vector<uint16_t> pairs(size_t(n) * 2);
        ok = fread(pairs.data(), 2, pairs.size(), f) == pairs.size();
        if (ok) {
            ranked.reserve(n);
            for (uint32_t i = 0; i < n; ++i) {
                ranked.emplace_back(pairs[2 * i], pairs[2 * i + 1]);
            }
        }
    }
    fclose(f);
    return ok;
}

// the pool is sized once, from the free device memory at the first MoE layer
void hpcache_decoder::init_pool(const llama_moe_layer_request & req) {
    pool_init = true;
    if (!opts.enabled || host == nullptr || host->get_devices == nullptr) {
        return;
    }

    llama_moe_device_info devices[16];
    const int32_t n = host->get_devices(host->ctx, devices, 16);
    for (int32_t i = 0; i < n; ++i) {
        if (devices[i].type == GGML_BACKEND_DEVICE_TYPE_GPU && devices[i].name != nullptr &&
                strncmp(devices[i].name, "CUDA", 4) == 0) {
            pool_dev = devices[i].dev;
            break;
        }
    }
    if (pool_dev == nullptr) {
        log(GGML_LOG_LEVEL_WARN, "GPU expert pool: no CUDA device");
        return;
    }

    ggml_backend_dev_props props;
    ggml_backend_dev_get_props(pool_dev, &props);
    pool_device_index = hpcache_cuda_device_by_pci(props.device_id);
    if (pool_device_index < 0) {
        log(GGML_LOG_LEVEL_WARN, "GPU expert pool: cannot map the device to a CUDA index");
        pool_dev = nullptr;
        return;
    }

    size_t free_bytes = 0, total_bytes = 0;
    ggml_backend_dev_memory(pool_dev, &free_bytes, &total_bytes);

    size_t budget = free_bytes > opts.reserve_bytes ? free_bytes - opts.reserve_bytes : 0;
    if (opts.pool_bytes > 0) {
        budget = std::min(budget, opts.pool_bytes);
    } else if (opts.plan_bytes > 0) {
        budget = std::min(budget, opts.plan_bytes + opts.plan_bytes / 10);
    }
    pool_budget = budget;

    size_t slot_bytes = 0;
    for (const ggml_tensor * t : { req.gate_exps, req.up_exps, req.down_exps, req.gate_up_exps }) {
        if (t != nullptr) {
            slot_bytes += t->nb[2];
        }
    }
    pool_slots = slot_bytes > 0 ? int64_t(budget / slot_bytes) : 0;

    if (!opts.profile.empty() && !read_profile(opts.profile, profile)) {
        log(GGML_LOG_LEVEL_WARN, "GPU expert pool: cannot read profile " + opts.profile);
    }
    pool_rank_limit = std::min<int64_t>(pool_slots, (int64_t) profile.size());

    char buf[256];
    snprintf(buf, sizeof(buf), "GPU expert pool: %.0f MiB free of %.0f MiB, budget %.0f MiB, about %lld expert slots%s",
            free_bytes / 1048576.0, total_bytes / 1048576.0, budget / 1048576.0, (long long) pool_slots,
            profile.empty() ? "" : ", ranked by profile");
    log(GGML_LOG_LEVEL_INFO, buf);
}

void hpcache_decoder::alloc_pool(layer & l, int il) {
    if (pool_dev == nullptr) {
        return;
    }

    size_t slot_bytes = 0;
    ggml_tensor * src[4] = { l.t_gate, l.t_up, l.t_down, l.t_gate_up };
    for (ggml_tensor * t : src) {
        if (t == nullptr) {
            continue;
        }
        // the copy needs the original row layout, so a repacked CPU buffer cannot feed the GPU
        const char * name = t->buffer != nullptr ? ggml_backend_buft_name(ggml_backend_buffer_get_type(t->buffer)) : "";
        const bool host_buffer = (t->buffer != nullptr && ggml_backend_buffer_is_host(t->buffer)) || strcmp(name, "CUDA_Host") == 0;
        if (t->buffer == nullptr || !host_buffer || strstr(name, "REPACK") != nullptr ||
                strstr(name, "AMX") != nullptr || t->data == nullptr) {
            log(GGML_LOG_LEVEL_WARN, "GPU expert pool: layer " + std::to_string(il) + " expert tensor " + t->name +
                    " is in buffer type " + name + ", the layer stays on the CPU");
            return;
        }
        slot_bytes += t->nb[2];
    }

    int n_slots = 0;
    if (!profile.empty()) {
        for (int64_t i = 0; i < pool_rank_limit; ++i) {
            n_slots += profile[i].first == il;
        }
    } else if (opts.n_layers > 0) {
        n_slots = int(pool_slots / opts.n_layers);
    }
    n_slots = std::min(n_slots, l.n_expert);
    if (pool_used + size_t(n_slots) * slot_bytes > pool_budget) {
        n_slots = int((pool_budget - pool_used) / slot_bytes);
    }
    if (n_slots < 1) {
        return;
    }

    hpcache_cuda_set_device(pool_device_index);
    ggml_init_params params = { ggml_tensor_overhead() * 8, nullptr, true };
    l.pool_ctx = ggml_init(params);
    if (l.pool_ctx == nullptr) {
        return;
    }
    ggml_tensor ** dst[4] = { &l.p_gate, &l.p_up, &l.p_down, &l.p_gate_up };
    for (int i = 0; i < 4; ++i) {
        if (src[i] != nullptr) {
            *dst[i] = ggml_new_tensor_3d(l.pool_ctx, src[i]->type, src[i]->ne[0], src[i]->ne[1], n_slots);
        }
    }
    l.pool_buf = ggml_backend_alloc_ctx_tensors_from_buft(l.pool_ctx, ggml_backend_dev_buffer_type(pool_dev));
    l.d_table  = hpcache_cuda_table_alloc(l.n_expert);
    if (l.pool_buf == nullptr || l.d_table == nullptr) {
        log(GGML_LOG_LEVEL_WARN, "GPU expert pool: allocation failed at layer " + std::to_string(il));
        hpcache_cuda_table_free(l.d_table);
        l.d_table = nullptr;
        if (l.pool_buf != nullptr) {
            ggml_backend_buffer_free(l.pool_buf);
            l.pool_buf = nullptr;
        }
        ggml_free(l.pool_ctx);
        l.pool_ctx = nullptr;
        l.p_gate = l.p_up = l.p_down = l.p_gate_up = nullptr;
        pool_budget = pool_used;   // the device is full, stop here
        return;
    }

    l.n_slots = n_slots;
    l.slot_expert.assign(n_slots, -1);
    l.slot_of.assign(l.n_expert, -1);
    l.score.assign(l.n_expert, 0.0f);
    pool_used += size_t(n_slots) * slot_bytes;

    // the first slots come from the profile, in rank order
    int next = 0;
    for (int64_t i = 0; i < pool_rank_limit && next < n_slots; ++i) {
        if (profile[i].first != il || profile[i].second >= l.n_expert) {
            continue;
        }
        if (!load_slot(l, next, profile[i].second)) {
            break;
        }
        l.score[profile[i].second] = 3.0f;
        ++next;
    }
    hpcache_cuda_copy_wait();
    hpcache_cuda_table_upload(l.d_table, l.slot_of.data(), l.n_expert);
}

// copy expert e of the CPU tensors into a slot of the pool
bool hpcache_decoder::load_slot(layer & l, int slot, int e) {
    const ggml_tensor * src[4] = { l.t_gate, l.t_up, l.t_down, l.t_gate_up };
    const ggml_tensor * dst[4] = { l.p_gate, l.p_up, l.p_down, l.p_gate_up };
    for (int t = 0; t < 4; ++t) {
        if (src[t] == nullptr) {
            continue;
        }
        const size_t n = src[t]->nb[2];
        if (!hpcache_cuda_copy_to_device((char *) dst[t]->data + size_t(slot) * n, (const char *) src[t]->data + size_t(e) * n, n)) {
            return false;
        }
    }
    const int old = l.slot_expert[slot];
    if (old >= 0) {
        l.slot_of[old] = -1;
    }
    l.slot_expert[slot] = e;
    l.slot_of[e] = slot;
    return true;
}

// The expert tensors are file backed pages that are read on first use. Touch them all now, in parallel,
// so that a decode step never stalls on the disk and a measurement is not cold.
void hpcache_decoder::prefault_layer(layer & l, int il) {
    const auto t0 = std::chrono::steady_clock::now();
    ggml_tensor * ts[4] = { l.t_gate, l.t_up, l.t_down, l.t_gate_up };

    std::vector<std::thread> workers;
    std::atomic<int64_t> bytes{0};
    for (ggml_tensor * t : ts) {
        if (t == nullptr || t->data == nullptr) {
            continue;
        }
        workers.emplace_back([t, &bytes] {
            const volatile char * p = static_cast<const volatile char *>(t->data);
            const size_t n = ggml_nbytes(t);
            char sink = 0;
            for (size_t i = 0; i < n; i += 4096) {
                sink ^= p[i];
            }
            if (sink == 1 && n == 0) {
                bytes += 1;   // keeps the reads from being optimized away
            }
            bytes += (int64_t) n;
        });
    }
    for (auto & w : workers) {
        w.join();
    }
    prefault_bytes += bytes.load();
    prefault_us += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
    (void) il;
}

std::unique_ptr<hpcache_decoder::layer> hpcache_decoder::make_layer(const llama_moe_layer_request & req) {
    auto l = std::make_unique<layer>();
    l->t_gate    = req.gate_exps;
    l->t_up      = req.up_exps;
    l->t_down    = req.down_exps;
    l->t_gate_up = req.gate_up_exps;
    l->n_embd    = (int) req.cur->ne[0];
    l->k         = req.n_expert_used;
    l->n_expert  = req.n_expert;
    l->clamp     = req.swiglu_clamp;

    if (!pool_init) {
        init_pool(req);
    }
    if (opts.prefault) {
        prefault_layer(*l, req.il);
    }
    if (cpu_pool && l->t_gate_up == nullptr && l->t_gate != nullptr && l->t_up != nullptr && l->t_down != nullptr &&
            l->t_gate->data != nullptr && l->t_up->data != nullptr && l->t_down->data != nullptr) {
        hpcache_expert_desc & d = l->desc;
        d.gate = (const char *) l->t_gate->data;
        d.up   = (const char *) l->t_up->data;
        d.down = (const char *) l->t_down->data;
        d.gate_row = l->t_gate->nb[1];
        d.up_row   = l->t_up->nb[1];
        d.down_row = l->t_down->nb[1];
        d.gate_exp = l->t_gate->nb[2];
        d.up_exp   = l->t_up->nb[2];
        d.down_exp = l->t_down->nb[2];
        d.n_embd   = l->t_gate->ne[0];
        d.n_ff     = l->t_gate->ne[1];
        d.clamp    = l->clamp;
        const bool shapes = l->t_up->ne[0] == d.n_embd && l->t_up->ne[1] == d.n_ff &&
                l->t_down->ne[0] == d.n_ff && l->t_down->ne[1] == d.n_embd &&
                l->t_gate->nb[0] == (size_t) ggml_type_size(l->t_gate->type) &&
                d.gate_row == ggml_row_size(l->t_gate->type, d.n_embd) &&
                d.up_row == ggml_row_size(l->t_up->type, d.n_embd) &&
                d.down_row == ggml_row_size(l->t_down->type, d.n_ff);
        l->fast = shapes && hpcache_cpu_pool::describe(d, l->t_gate->type, l->t_up->type, l->t_down->type, get_traits);

        if (l->fast && req.il == 3 && getenv("HPCACHE_BENCH")) {
            // dot product speed with the row in L1, one thread: the compute side of the kernel
            std::vector<uint8_t> xq(std::max(d.gate_xq_bytes, d.down_xq_bytes));
            std::vector<float> xf(d.n_embd, 0.5f);
            d.gate_quant(xf.data(), xq.data(), d.n_embd);
            const int reps = 200000;
            float s = 0, v = 0;
            auto t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < reps; ++r) {
                d.gate_dot((int) d.n_embd, &v, 0, d.gate + (size_t(r) & 7) * d.gate_row, 0, xq.data(), 0, 1);
                s += v;
            }
            double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
            char buf[260];
            snprintf(buf, sizeof(buf), "bench types: gate %s up %s down %s, n_embd %lld n_ff %lld", ggml_type_name(l->t_gate->type),
                    ggml_type_name(l->t_up->type), ggml_type_name(l->t_down->type), (long long) d.n_embd, (long long) d.n_ff);
            log(GGML_LOG_LEVEL_INFO, buf);
            snprintf(buf, sizeof(buf), "bench gate dot: %.3f us/row, %.2f GB/s of weights (%zu B rows) %g", us / reps,
                    double(d.gate_row) * reps / us / 1e3, d.gate_row, s);
            log(GGML_LOG_LEVEL_INFO, buf);
            d.down_quant(xf.data(), xq.data(), d.n_ff);
            t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < reps; ++r) {
                d.down_dot((int) d.n_ff, &v, 0, d.down + (size_t(r) & 7) * d.down_row, 0, xq.data(), 0, 1);
                s += v;
            }
            us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
            {
                // compare with the stock kernel on random-ish activations
                std::vector<float> xr(d.n_ff);
                for (size_t q = 0; q < xr.size(); ++q) {
                    xr[q] = sinf(float(q) * 0.37f) * 2.0f;
                }
                d.down_quant(xr.data(), xq.data(), d.n_ff);
                const auto * stock = get_traits(l->t_down->type);
                double worst = 0.0;
                for (int row = 0; row < 64; ++row) {
                    float a = 0, b = 0;
                    d.down_dot((int) d.n_ff, &a, 0, d.down + size_t(row) * d.down_row, 0, xq.data(), 0, 1);
                    stock->vec_dot((int) d.n_ff, &b, 0, d.down + size_t(row) * d.down_row, 0, xq.data(), 0, 1);
                    worst = std::max(worst, (double) fabsf(a - b) / (fabsf(b) + 1e-3f));
                }
                char cb[120];
                snprintf(cb, sizeof(cb), "bench down dot vs stock: worst relative difference %g", worst);
                log(GGML_LOG_LEVEL_INFO, cb);
            }
            snprintf(buf, sizeof(buf), "bench down dot: %.3f us/row, %.2f GB/s of weights (%zu B rows) %g", us / reps,
                    double(d.down_row) * reps / us / 1e3, d.down_row, s);
            log(GGML_LOG_LEVEL_INFO, buf);
        }
    }
    alloc_pool(*l, req.il);
    return l;
}

static void publish_cb(ggml_tensor * dst, int, int, void * userdata);
static void remap_ids_cb(ggml_tensor * dst, int, int, void * userdata);
static void remap_w_cb(ggml_tensor * dst, int, int, void * userdata);
static void wait_merge_cb(ggml_tensor * dst, int, int, void * userdata);

ggml_tensor * hpcache_decoder::build_layer(const llama_moe_layer_request & req) {
    if (!ok() || req.ffn_op != LLAMA_MOE_FFN_SILU || req.n_tokens != 1 || req.down_exps == nullptr ||
            (req.gate_exps == nullptr && req.gate_up_exps == nullptr) ||
            (req.gate_up_exps == nullptr && req.up_exps == nullptr) || req.n_expert_used > 32) {
        return nullptr;
    }

    const int n_embd = (int) req.cur->ne[0];
    const int k      = req.n_expert_used;
    n_expert = req.n_expert;
    if (req.cur->type != GGML_TYPE_F32 || req.weights->type != GGML_TYPE_F32 || req.selected_experts->type != GGML_TYPE_I32 ||
            !ensure_doorbell(n_embd, k)) {
        return nullptr;
    }

    layer * l = nullptr;
    {
        std::lock_guard<std::mutex> lock(layers_mutex);
        auto it = layers.find(req.il);
        if (it != layers.end() && (it->second->t_up != req.up_exps || it->second->t_gate != req.gate_exps ||
                it->second->t_down != req.down_exps || it->second->t_gate_up != req.gate_up_exps)) {
            // a different model instance, as when the memory fit loads one first: size the pool again
            layers.clear();
            pool_init = false;
            pool_used = 0;
            pool_dev = nullptr;
            profile.clear();
        }
        auto & slot = layers[req.il];
        const bool same = slot != nullptr;
        if (!same) {
            slot = make_layer(req);
            if (slot == nullptr) {
                layers.erase(req.il);
                return nullptr;
            }
        }
        l = slot.get();
    }

    HPCACHE_TRACE("build layer %d, k=%d, n_embd=%d, pool slots %d", req.il, k, n_embd, l->n_slots);

    op_ctxs.push_back({ this, req.il, l->d_table });
    void * ud = &op_ctxs.back();

    ggml_tensor * srcs[3] = { req.cur, req.selected_experts, req.weights };
    ggml_tensor * pub = ggml_custom_4d(req.ctx, GGML_TYPE_F32, 1, 1, 1, 1, srcs, 3, publish_cb, 1, ud);
    ggml_custom_set_backend_stream(pub);

    ggml_tensor * wait_src[1] = { pub };
    ggml_tensor * merged = ggml_custom_4d(req.ctx, GGML_TYPE_F32, n_embd, 1, 1, 1, wait_src, 1, wait_merge_cb, 1, ud);
    ggml_custom_set_backend_stream(merged);

    if (l->n_slots == 0) {
        return merged;
    }

    // experts in the pool run on the GPU while the CPU serves the rest
    // the remap ops read pub, so that they are queued after the ring
    ggml_tensor * ids_src[2] = { req.selected_experts, pub };
    ggml_tensor * slot_ids = ggml_custom_4d(req.ctx, GGML_TYPE_I32, k, 1, 1, 1, ids_src, 2, remap_ids_cb, 1, ud);
    ggml_custom_set_backend_stream(slot_ids);

    ggml_tensor * w_src[3] = { req.selected_experts, req.weights, pub };
    ggml_tensor * w_hit = ggml_custom_4d(req.ctx, GGML_TYPE_F32, 1, k, 1, 1, w_src, 3, remap_w_cb, 1, ud);
    ggml_custom_set_backend_stream(w_hit);

    ggml_tensor * x = ggml_reshape_3d(req.ctx, req.cur, n_embd, 1, 1);
    ggml_tensor * hit = build_experts(req.ctx, l->p_gate_up, l->p_gate, l->p_up, l->p_down, x, slot_ids, w_hit,
            n_embd, k, req.swiglu_clamp);

    // the GPU part is the first operand, so it is queued before the wait
    return ggml_add(req.ctx, hit, ggml_reshape_2d(req.ctx, merged, n_embd, 1));
}

void hpcache_decoder::publish(void * s, int il, const ggml_tensor * x, const ggml_tensor * ids, const ggml_tensor * w) {
    if (s == nullptr) {
        GGML_ABORT("moe-hpcache: the publish op ran on a backend without a stream");
    }
    stream.store(s);
    if (device.load() < 0) {
        device.store(hpcache_cuda_get_device());
    }
    // the GPU waits for the follower after this ring, so it must be running whatever the caller did
    arm();
    HPCACHE_TRACE("publish layer %d, stream %p", il, s);
    hpcache_cuda_publish(s, db, (const float *) x->data, (const int32_t *) ids->data, (const float *) w->data, il);
}

void hpcache_decoder::wait_merge(void * s, ggml_tensor * dst) {
    if (s == nullptr) {
        GGML_ABORT("moe-hpcache: the wait op ran on a backend without a stream");
    }
    HPCACHE_TRACE("wait_merge enqueued");
    hpcache_cuda_wait_merge(s, db, (float *) dst->data);
}

static void publish_cb(ggml_tensor * dst, int, int, void * userdata) {
    auto * c = static_cast<hpcache_decoder::op_ctx *>(userdata);
    c->decoder->publish(ggml_custom_get_backend_stream(dst), c->il, dst->src[0], dst->src[1], dst->src[2]);
}

static void remap_ids_cb(ggml_tensor * dst, int, int, void * userdata) {
    auto * c = static_cast<hpcache_decoder::op_ctx *>(userdata);
    hpcache_cuda_remap_ids(ggml_custom_get_backend_stream(dst), (const int32_t *) dst->src[0]->data, c->table,
            (int) dst->ne[0], (int32_t *) dst->data);
}

static void remap_w_cb(ggml_tensor * dst, int, int, void * userdata) {
    auto * c = static_cast<hpcache_decoder::op_ctx *>(userdata);
    hpcache_cuda_remap_weights(ggml_custom_get_backend_stream(dst), (const int32_t *) dst->src[0]->data,
            (const float *) dst->src[1]->data, c->table, (int) dst->ne[1], (float *) dst->data);
}

static void wait_merge_cb(ggml_tensor * dst, int, int, void * userdata) {
    auto * c = static_cast<hpcache_decoder::op_ctx *>(userdata);
    c->decoder->wait_merge(ggml_custom_get_backend_stream(dst), dst);
}

void hpcache_decoder::serve(uint32_t seq) {
    const int il = *db.layer;
    HPCACHE_TRACE("serve seq %u layer %d", seq, il);

    layer * l = nullptr;
    {
        std::lock_guard<std::mutex> lock(layers_mutex);
        const auto it = layers.find(il);
        if (it != layers.end()) {
            l = it->second.get();
        }
    }

    const auto t_start = std::chrono::steady_clock::now();

    if (n_expert > 0 && il >= 0) {
        if ((size_t) il >= routing.size()) {
            routing.resize(il + 1);
        }
        auto & histogram = routing[il];
        histogram.resize(n_expert, 0);
        for (int i = 0; i < db.k; ++i) {
            const int32_t id = db.ids[i];
            if (id >= 0 && id < n_expert) {
                histogram[id]++;
            }
        }
    }

    if (const char * path = getenv("HPCACHE_ROUTING_TRACE")) {
        static FILE * trace = fopen(path, "wb");
        if (trace != nullptr) {
            int32_t rec[33];
            rec[0] = il;
            memcpy(rec + 1, db.ids, size_t(db.k) * sizeof(int32_t));
            fwrite(rec, sizeof(int32_t), size_t(db.k) + 1, trace);
            fflush(trace);
        }
    }

    if (l != nullptr) {
        // experts in the pool are served by the GPU, the rest run here
        int32_t ids[32];
        float   w[32];
        int     m = 0;
        for (int i = 0; i < db.k; ++i) {
            const int32_t e = db.ids[i];
            if (e < 0 || e >= l->n_expert) {
                continue;
            }
            if (l->n_slots > 0) {
                l->score[e] += 1.0f;
                if (l->slot_of[e] >= 0) {
                    continue;
                }
            }
            ids[m] = e;
            w[m]   = db.w[i];
            ++m;
        }
        n_routed += db.k;
        n_hit    += db.k - m;

        if (m > 0 && l->fast) {
            cpu_pool->run(l->desc, db.x, ids, w, m, db.y);
            static const bool verify = getenv("HPCACHE_VERIFY") != nullptr;
            if (verify) {
                // the same experts through the ggml graph, to compare the two kernels
                // one ggml thread, the pool threads are spinning on the other cores
                static bool single = [&] {
                    typedef void (*set_n_threads_t)(ggml_backend_t, int);
                    auto * reg = ggml_backend_dev_backend_reg(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU));
                    auto * f = (set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
                    if (f != nullptr) {
                        f(cpu, 1);
                    }
                    return true;
                }();
                (void) single;
                cpu_graph * ref = cpu_graph_for(*l, m);
                if (ref != nullptr) {
                    memcpy(ref->x->data,   db.x, size_t(db.n_embd) * sizeof(float));
                    memcpy(ref->ids->data, ids,  size_t(m) * sizeof(int32_t));
                    memcpy(ref->w->data,   w,    size_t(m) * sizeof(float));
                    ggml_backend_graph_compute(cpu, ref->gf);
                    const float * r = (const float *) ref->out->data;
                    double num = 0.0, den = 0.0;
                    for (int i = 0; i < db.n_embd; ++i) {
                        num += double(db.y[i] - r[i]) * double(db.y[i] - r[i]);
                        den += double(r[i]) * double(r[i]);
                    }
                    verify_n++;
                    verify_sum += std::sqrt(num / (den + 1e-30));
                    verify_max = std::max(verify_max, std::sqrt(num / (den + 1e-30)));
                }
            }
            m = -1;
        }
        cpu_graph * g = m > 0 ? cpu_graph_for(*l, m) : nullptr;
        if (m < 0) {
            // served by the pool
        } else if (g != nullptr) {
            memcpy(g->x->data,   db.x, size_t(db.n_embd) * sizeof(float));
            memcpy(g->ids->data, ids,  size_t(m) * sizeof(int32_t));
            memcpy(g->w->data,   w,    size_t(m) * sizeof(float));
            ggml_backend_graph_compute(cpu, g->gf);
            memcpy(db.y, g->out->data, size_t(db.n_embd) * sizeof(float));
        } else {
            if (m > 0) {
                log(GGML_LOG_LEVEL_ERROR, "no CPU graph for layer " + std::to_string(il));
            }
            memset(db.y, 0, size_t(db.n_embd) * sizeof(float));
        }
    } else {
        // release the GPU anyway, a stuck wait would end in a driver timeout
        log(GGML_LOG_LEVEL_ERROR, "no CPU graph for layer " + std::to_string(il));
        memset(db.y, 0, size_t(db.n_embd) * sizeof(float));
    }

    serve_us += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t_start).count();
    n_serve++;

    std::atomic_thread_fence(std::memory_order_seq_cst);
    *db.flag = seq;
    HPCACHE_TRACE("served seq %u", seq);
}

void hpcache_decoder::follower() {
    uint32_t seen = 0;

    for (;;) {
        {
            std::unique_lock<std::mutex> lock(mutex);
            cv.wait(lock, [&] { return armed.load() || quit; });
            if (quit) {
                return;
            }
        }

        const int dev = device.load();
        if (dev >= 0) {
            hpcache_cuda_set_device(dev);
        }

        using clock = std::chrono::steady_clock;
        auto last_flush = clock::now();
        uint32_t spins = 0;

        while (armed.load(std::memory_order_acquire)) {
            const uint32_t seq = *db.seq;
            if (seq != seen) {
                serve(seq);
                seen = seq;
                last_flush = clock::now();
                continue;
            }

            HPCACHE_PAUSE();

            // on WDDM the driver can hold queued work back, so nudge the submission now and then
            if ((++spins & 1023u) == 0) {
                const auto now = clock::now();
                if (now - last_flush > std::chrono::milliseconds(2)) {
                    void * s = stream.load();
                    if (s != nullptr) {
                        hpcache_cuda_flush(s);
                    }
                    last_flush = now;
                }
            }
        }

        {
            std::lock_guard<std::mutex> lock(mutex);
            idle = true;
        }
        cv.notify_all();
    }
}

void hpcache_decoder::arm() {
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (armed.load(std::memory_order_acquire)) {
            return;
        }
        if (!thread.joinable()) {
            thread = std::thread([this] { follower(); });
        }
        idle = false;
        armed.store(true, std::memory_order_release);
    }
    cv.notify_all();
}

void hpcache_decoder::begin(const llama_moe_decode_info & info) {
    // only single-token graphs contain the doorbell ops
    if (info.n_tokens != 1 || db.block == nullptr || !ok()) {
        return;
    }
    HPCACHE_TRACE("begin");
    if (cpu_pool) {
        cpu_pool->set_active(true);
    }
    decode_start_us.store(std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
    arm();
}

// Runs while the device and the follower are idle. Per layer, the best missed expert replaces
// the coldest resident one, and the globally best swaps of the round are applied.
void hpcache_decoder::admit() {
    struct swap {
        float gain;
        layer * l;
        int slot;
        int expert;
    };
    std::vector<swap> fills;
    std::vector<swap> swaps;

    for (auto & entry : layers) {
        layer & l = *entry.second;
        if (l.n_slots == 0) {
            continue;
        }

        int best = -1;
        for (int e = 0; e < l.n_expert; ++e) {
            if (l.slot_of[e] < 0 && (best < 0 || l.score[e] > l.score[best])) {
                best = e;
            }
        }
        if (best < 0 || l.score[best] < 2.0f) {
            continue;
        }

        int victim = -1;
        for (int s = 0; s < l.n_slots; ++s) {
            if (l.slot_expert[s] < 0) {
                victim = s;
                break;
            }
            if (victim < 0 || l.score[l.slot_expert[s]] < l.score[l.slot_expert[victim]]) {
                victim = s;
            }
        }
        if (victim < 0) {
            continue;
        }
        if (l.slot_expert[victim] < 0) {
            fills.push_back({ l.score[best], &l, victim, best });
        } else if (l.score[best] > l.score[l.slot_expert[victim]] + 1.0f) {
            swaps.push_back({ l.score[best] - l.score[l.slot_expert[victim]], &l, victim, best });
        }
    }

    auto by_gain = [](const swap & a, const swap & b) { return a.gain > b.gain; };
    std::sort(fills.begin(), fills.end(), by_gain);
    std::sort(swaps.begin(), swaps.end(), by_gain);
    if ((int) fills.size() > opts.fill_swaps) {
        fills.resize(opts.fill_swaps);
    }
    if ((int) swaps.size() > opts.max_swaps) {
        swaps.resize(opts.max_swaps);
    }

    // queue every copy, then publish the new tables once the data has landed
    std::vector<layer *> touched;
    auto apply = [&](const swap & s) {
        if (load_slot(*s.l, s.slot, s.expert)) {
            s.l->score[s.expert] = std::max(s.l->score[s.expert], 3.0f);
            touched.push_back(s.l);
            ++n_swapped;
        }
    };
    for (const auto & s : fills) {
        apply(s);
    }
    for (const auto & s : swaps) {
        apply(s);
    }
    if (!touched.empty()) {
        hpcache_cuda_copy_wait();
        std::sort(touched.begin(), touched.end());
        touched.erase(std::unique(touched.begin(), touched.end()), touched.end());
        for (layer * l : touched) {
            hpcache_cuda_table_upload(l->d_table, l->slot_of.data(), l->n_expert);
        }
    }

    for (auto & entry : layers) {
        for (float & v : entry.second->score) {
            v *= opts.decay;
        }
    }
}

void hpcache_decoder::end() {
    if (!armed.load(std::memory_order_acquire)) {
        return;
    }

    // when the device is done, every ring of this graph has been served
    const int dev = device.load();
    if (dev >= 0) {
        hpcache_cuda_set_device(dev);
    }
    HPCACHE_TRACE("end: syncing the device");
    hpcache_cuda_sync_device();
    HPCACHE_TRACE("end: device synced");

    {
        std::unique_lock<std::mutex> lock(mutex);
        armed.store(false, std::memory_order_release);
        cv.notify_all();
        cv.wait(lock, [&] { return idle; });
    }

    if (cpu_pool) {
        cpu_pool->set_active(false);
    }

    const int64_t now_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    const int64_t start_us = decode_start_us.load();
    if (start_us > 0) {
        decode_us += now_us - start_us;
        n_decode++;
    }

    if (pool_used > 0 && opts.swap_every > 0 && ++steps % opts.swap_every == 0) {
        const auto t0 = std::chrono::steady_clock::now();
        admit();
        swap_us += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
    }
}
