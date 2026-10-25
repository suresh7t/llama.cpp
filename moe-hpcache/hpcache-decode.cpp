#include "hpcache-decode.h"

#include "ggml-backend.h"
#include "ggml-impl.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <tuple>
#include <string>

#ifdef _WIN32
#    define WIN32_LEAN_AND_MEAN
#    include <windows.h>
#else
#    include <dlfcn.h>
#endif

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

    ggml_tensor * in  = nullptr;   // second card: the block that x, ids and w are windows of

    ~cpu_graph() {
        if (buf != nullptr) {
            ggml_backend_buffer_free(buf);
        }
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

// input block of the second card: x, then the ids, then the weights, each part padded to 128 bytes
static constexpr int gpu2_in_max_k = 32;   // 32 ids and 32 weights fill their 128 bytes
static size_t gpu2_in_ids_off(int n_embd) { return (size_t(n_embd) * sizeof(float) + 127) & ~size_t(127); }
static size_t gpu2_in_w_off(int n_embd)   { return gpu2_in_ids_off(n_embd) + 128; }
static size_t gpu2_in_stamp_off(int n_embd) { return gpu2_in_w_off(n_embd) + 128; }   // step trace: the sequence number, 8 bytes
static constexpr size_t gpu2_in_bytes_extra = 16;
#ifdef HPCACHE_EVENT_LOG
static void stamp_g2_cb(ggml_tensor * dst, int, int, void * userdata);
#endif
static void g2_finish_cb(ggml_tensor * dst, int, int, void * userdata);

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
    int                   spare = -1;    // empty slot that receives the next copy, -1 when none or in flight
    std::vector<int32_t>  slot_expert;   // expert held by each slot, -1 when empty
    std::vector<int32_t>  slot_of;       // slot of each expert, -1 when not resident
    std::vector<float>    score;         // recent routing weight of each expert
    int32_t *             d_table = nullptr;   // device slot table of this layer, a view into the shared array
    int32_t *             h_table = nullptr;   // its pinned host mirror
    int                   table_index = -1;

    // static pool on the second card: expert -> slot, never changes after the load
    ggml_context *        g_ctx = nullptr;
    ggml_backend_buffer_t g_buf = nullptr;
    ggml_tensor *         g_gate    = nullptr;
    ggml_tensor *         g_up      = nullptr;
    ggml_tensor *         g_down    = nullptr;
    ggml_tensor *         g_gate_up = nullptr;
    int                   n_slots2 = 0;
    std::vector<int32_t>  slot2_of;      // slot of each expert on the second card, -1 when not resident
    std::vector<std::unique_ptr<cpu_graph>> graphs2;   // graphs on the second card by number of experts

    // staged experts: the stage kernel copies the GPU's share from the page-locked arena into these views
    hpcache_stage_args stage;
    ggml_context * a_ctx      = nullptr;
    ggml_tensor *  a_gate     = nullptr;
    ggml_tensor *  a_up       = nullptr;
    ggml_tensor *  a_down     = nullptr;
    ggml_tensor *  a_gate_up  = nullptr;
    bool           pcie       = false;

    // own CPU kernel, valid when fast
    hpcache_expert_desc   desc;
    bool                  fast = false;

    ~layer() {
        if (pool_buf != nullptr) {
            ggml_backend_buffer_free(pool_buf);
        }
        if (pool_ctx != nullptr) {
            ggml_free(pool_ctx);
        }
        if (a_ctx != nullptr) {
            ggml_free(a_ctx);
        }
        if (g_buf != nullptr) {
            ggml_backend_buffer_free(g_buf);
        }
        if (g_ctx != nullptr) {
            ggml_free(g_ctx);
        }
    }
};

static int64_t g_adm_pub = 0, g_adm_sel = 0, g_adm_copy = 0, g_adm_up = 0, g_adm_decay = 0;
static inline int64_t adm_now() {
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

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
    if (opts.arena) {
        init_arena();
    }
}

// A CUDA device with its own memory. Under WSL the backend reports the card as an integrated GPU (type IGPU), with the
// system memory as its free memory, so that type counts as well.
static bool is_cuda_gpu(const llama_moe_device_info & info) {
    return info.name != nullptr && strncmp(info.name, "CUDA", 4) == 0 &&
            (info.type == GGML_BACKEND_DEVICE_TYPE_GPU || info.type == GGML_BACKEND_DEVICE_TYPE_IGPU);
}

// Read the expert tensors from the GGUF into one page-locked region. This runs as early as the provider is created,
// before the backends touch a second GPU: a CUDA context on another card limits how much can be locked.
void hpcache_decoder::init_arena() {
    const char * model = host != nullptr && host->get_param != nullptr ? host->get_param(host->ctx, "model") : nullptr;
    if (model == nullptr || host->get_devices == nullptr) {
        return;
    }
    int device_index = -1;
    int n_cuda = 0;
    std::string seen;   // what the host passed, for the warning below
    std::string pci;
    llama_moe_device_info devices[16];
    const int32_t n = host->get_devices(host->ctx, devices, 16);
    for (int32_t i = 0; i < n; ++i) {
        seen += std::string(i > 0 ? ", " : "") + (devices[i].name != nullptr ? devices[i].name : "?") + " (type " +
                std::to_string((int) devices[i].type) + ")";
        if (is_cuda_gpu(devices[i])) {
            ++n_cuda;
            if (device_index < 0) {
                ggml_backend_dev_props props;
                ggml_backend_dev_get_props(devices[i].dev, &props);
                pci = props.device_id != nullptr ? props.device_id : "(none)";
                device_index = hpcache_cuda_device_by_pci(props.device_id);
            }
        }
    }
    // measured: with a CUDA context on a second GPU, locking more than about 25 GiB fails and leaves the device
    // without memory for its own buffers, 20 GiB works; a single visible GPU takes the whole 40 GiB
    size_t pin_limit = opts.pin_bytes > 0 ? opts.pin_bytes : SIZE_MAX;
    if (opts.pin_bytes == 0 && n_cuda > 1) {
        pin_limit = size_t(16) << 30;
        log(GGML_LOG_LEVEL_INFO, "arena: " + std::to_string(n_cuda) + " CUDA devices visible, at most 16 GiB is page-locked "
                "(pin_mib=N to change; CUDA_VISIBLE_DEVICES to the compute GPU allows all of it)");
    }
    if (device_index < 0) {
        log(GGML_LOG_LEVEL_WARN, "arena: no CUDA device (" + std::to_string(n) + " devices from the host: " + seen +
                (n_cuda > 0 ? "; CUDA PCI id " + pci + " has no CUDA index" : "") +
                "), the expert weights stay in the model mapping");
        return;
    }
    hpcache_cuda_set_device(device_index);

    auto a = std::make_unique<hpcache_arena>();
    std::string error;
    const size_t cap = opts.arena_bytes > 0 ? opts.arena_bytes : SIZE_MAX;
    const bool ok = a->open(model, cap, pin_limit,
            [](void * ptr, size_t bytes, bool pin) { return hpcache_cuda_host_register(ptr, bytes, pin); },
            [this](const std::string & m) { log(GGML_LOG_LEVEL_INFO, m); }, error);
    if (!ok) {
        log(GGML_LOG_LEVEL_WARN, "arena: " + error + ", the expert weights stay in the model mapping");
        return;
    }
    arena = std::move(a);
}

// The GPU's share of the missed experts is copied from the page-locked arena into device staging slots by a kernel
// (reading through PCIe), then computed from there. The slots are shared by all layers: a layer's staged compute is
// ordered before the next layer's publish on the stream, and the follower serves a layer only after its publish.
bool hpcache_decoder::init_pcie(layer & l) {
    if (pool_dev == nullptr || !arena) {
        return false;
    }
    ggml_tensor * src[4] = { l.t_gate, l.t_up, l.t_down, l.t_gate_up };
    int n = 0;
    for (ggml_tensor * t : src) {
        if (t == nullptr) {
            continue;
        }
        if (t->data == nullptr || !arena->is_pinned(t->data, ggml_nbytes(t))) {
            return false;   // not page-locked, the GPU cannot read it
        }
        if (reinterpret_cast<uintptr_t>(t->data) % 16 != 0 || t->nb[2] % 16 != 0) {
            return false;
        }
        ++n;
    }
    if (n == 0 || n > 3) {
        return false;
    }
    const int m = opts.pcie_max;
    if (m < 1 || m > 8) {
        return false;
    }
    if (pcie_buf == nullptr) {
        // one slot size for every kind and layer: the largest tensor per expert
        pcie_slot_bytes = (arena->max_tensor_bytes() / (size_t) l.n_expert + 255) / 256 * 256;
        pcie_buf = ggml_backend_buft_alloc_buffer(ggml_backend_dev_buffer_type(pool_dev), pcie_slot_bytes * size_t(m) * 3 + (size_t(256) << 10));
        if (pcie_buf == nullptr) {
            return false;
        }
        ggml_backend_buffer_clear(pcie_buf, 0);   // zero bytes decode to zero weights, so unused slots add nothing
    }
    ggml_init_params params = { ggml_tensor_overhead() * 8, nullptr, true };
    l.a_ctx = ggml_init(params);
    if (l.a_ctx == nullptr) {
        return false;
    }
    ggml_tensor ** dst[4] = { &l.a_gate, &l.a_up, &l.a_down, &l.a_gate_up };
    char * base = static_cast<char *>(ggml_backend_buffer_get_base(pcie_buf));
    int kind = 0;
    for (int i = 0; i < 4; ++i) {
        if (src[i] == nullptr) {
            continue;
        }
        if (src[i]->nb[2] > pcie_slot_bytes) {
            return false;
        }
        void * dev = hpcache_cuda_host_device_ptr(src[i]->data);
        if (dev == nullptr) {
            log(GGML_LOG_LEVEL_WARN, "pcie: no device address for the arena, the GPU path is off");
            return false;
        }
        ggml_tensor * v = ggml_new_tensor_3d(l.a_ctx, src[i]->type, src[i]->ne[0], src[i]->ne[1], m);
        char * slot0 = base + size_t(kind) * pcie_slot_bytes * size_t(m);
        ggml_backend_tensor_alloc(pcie_buf, v, slot0);
        ggml_set_name(v, src[i]->name);
        *dst[i] = v;

        l.stage.src[l.stage.n_tensors]   = dev;
        l.stage.bytes[l.stage.n_tensors] = src[i]->nb[2];
        l.stage.dst[l.stage.n_tensors]   = slot0;
        ++l.stage.n_tensors;
        ++kind;
    }
    // the staged tensors are contiguous in the slot order of the kernel
    return true;
}

// Point the expert tensors of a layer at the arena. The first tensor is compared with the mapped one at three places.
bool hpcache_decoder::attach_arena(layer & l) {
    if (!arena) {
        return false;
    }
    ggml_tensor * ts[4] = { l.t_gate, l.t_up, l.t_down, l.t_gate_up };
    bool attached = false;
    for (ggml_tensor * t : ts) {
        if (t == nullptr || t->buffer == nullptr || t->data == nullptr || !ggml_backend_buffer_is_host(t->buffer) ||
                t->extra != nullptr) {
            continue;
        }
        const std::string buft = ggml_backend_buft_name(ggml_backend_buffer_get_type(t->buffer));
        if (buft.find("REPACK") != std::string::npos || buft.find("AMX") != std::string::npos) {
            continue;
        }
        uint8_t * p = arena->find(t->name, ggml_nbytes(t));
        if (p == nullptr) {
            continue;
        }
        if (!arena_checked) {
            arena_checked = true;
            const size_t n = ggml_nbytes(t);
            const size_t len = std::min<size_t>(4096, n);
            const size_t at[3] = { 0, (n / 2) / 64 * 64, n - len };
            for (size_t o : at) {
                if (memcmp((const uint8_t *) t->data + o, p + o, len) != 0) {
                    log(GGML_LOG_LEVEL_ERROR, std::string("arena: ") + t->name + " differs from the model mapping at offset " +
                            std::to_string(o) + ", the arena is not used");
                    arena.reset();
                    return false;
                }
            }
            log(GGML_LOG_LEVEL_INFO, std::string("arena: ") + t->name + " matches the model mapping");
        }
        t->data = p;
        attached = true;
        ++n_arena_tensors;
    }
    return attached;
}

hpcache_decoder::~hpcache_decoder() {
#ifdef HPCACHE_EVENT_LOG
    if (steps_on) {
        metrics.offset(HPC_LEAD, steps_off_lead_0, steps_off_lead, steps_host_0, steps_host_1);
        metrics.offset(HPC_G2, steps_off_g2_0, steps_off_g2, steps_host_0, steps_host_1);
        metrics.close();
    }
#endif
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
    if (g_adm_sel > 0 && n_decode.load() > 0) {
        char buf[256];
        const double k = 1.0 / double(n_decode.load());
        snprintf(buf, sizeof(buf), "admission per step, us: publish %.0f, select %.0f, copy issue %.0f, table upload %.0f, decay %.0f",
                g_adm_pub * k, g_adm_sel * k, g_adm_copy * k, g_adm_up * k, g_adm_decay * k);
        log(GGML_LOG_LEVEL_INFO, buf);
    }
    if (copy_batches > 0) {
        char buf[256];
        snprintf(buf, sizeof(buf),
                "copier: %lld batches, landed in %.0f us mean, %lld us max; the next round waited %.1f us on average "
                "(max %lld us), found the copies still running %lld of %lld times",
                (long long) copy_batches, double(copy_us_sum) / copy_batches, (long long) copy_us_max,
                pub_waits > 0 ? double(pub_wait_us_sum) / pub_waits : 0.0, (long long) pub_wait_us_max,
                (long long) pub_blocked, (long long) pub_waits);
        log(GGML_LOG_LEVEL_INFO, buf);
    }
    if (n_pcie_experts > 0 && n_decode.load() > 0) {
        char buf[160];
        snprintf(buf, sizeof(buf), "PCIe path: %lld experts computed by the GPU from the arena, %.2f per step",
                (long long) n_pcie_experts, double(n_pcie_experts) / double(n_decode.load()));
        log(GGML_LOG_LEVEL_INFO, buf);
    }
    if (n_routed.load() > 0) {
        char buf[256];
        snprintf(buf, sizeof(buf),
                "GPU expert pool: %lld of %lld routed experts served from the pool (%.1f%%), %lld swaps in %.1f ms",
                (long long) n_hit.load(), (long long) n_routed.load(), 100.0 * n_hit.load() / n_routed.load(),
                (long long) n_swapped, swap_us / 1000.0);
        log(GGML_LOG_LEVEL_INFO, buf);
        if (db.block != nullptr && n_serve.load() > 0) {
            // who waits for whom, per layer: the merge on the first card counts from the start of its wait
            const double calls = double(n_serve.load());
            const double period = double(decode_us.load()) / calls;
            const double cpu_busy = double(serve_us.load()) / calls;
            const double g2_busy = double(g2_set_us + g2_compute_us + g2_get_us + g2_graph_us) / calls;
            const double w_cpu = double(db.stats[0]) / 1000.0 / calls;
            char line[400];
            if (db.two) {
                const double w_two = double(db.stats[1]) / 1000.0 / calls;
                const double n_last = double(db.stats[2] + db.stats[3]);
                snprintf(line, sizeof(line),
                        "waits per layer (us): first GPU waited %.0f for the CPU and %.0f for the second GPU (CPU last in %.0f%% of layers, second GPU in %.0f%%); "
                        "CPU worker busy %.0f idle %.0f, second GPU worker busy %.0f idle %.0f, layer period %.0f",
                        w_cpu, w_two, n_last > 0 ? 100.0 * db.stats[2] / n_last : 0.0, n_last > 0 ? 100.0 * db.stats[3] / n_last : 0.0,
                        cpu_busy, period - cpu_busy, g2_busy, period - g2_busy, period);
            } else {
                snprintf(line, sizeof(line),
                        "waits per layer (us): first GPU waited %.0f for the CPU; CPU worker busy %.0f idle %.0f, layer period %.0f",
                        w_cpu, cpu_busy, period - cpu_busy, period);
            }
            log(GGML_LOG_LEVEL_INFO, line);
        }
        if (g2_n > 0) {
            snprintf(buf, sizeof(buf), "second GPU worker: %lld layer calls (%lld empty), %.2f experts each; per call: graph lookup %.1f us, inputs %.1f us, compute %.1f us, result %.1f us",
                    (long long) g2_n, (long long) g2_empty, double(g2_m) / g2_n, double(g2_graph_us) / g2_n,
                    double(g2_set_us) / g2_n, double(g2_compute_us) / g2_n, double(g2_get_us) / g2_n);
            log(GGML_LOG_LEVEL_INFO, buf);
        }
        if (n_gpu2.load() > 0) {
            snprintf(buf, sizeof(buf), "second GPU: %lld of %lld routed experts served by its static pool (%.1f%%), together with the first pool %.1f%%",
                    (long long) n_gpu2.load(), (long long) n_routed.load(), 100.0 * n_gpu2.load() / n_routed.load(),
                    100.0 * (n_gpu2.load() + n_hit.load()) / n_routed.load());
            log(GGML_LOG_LEVEL_INFO, buf);
        }
    }
    quit.store(true);
    armed.store(false);
    follower_sig.notify();
    follower2_sig.notify();
    if (thread.joinable()) {
        thread.join();
    }
    if (thread2.joinable()) {
        thread2.join();
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

    copier_wait();
    copier_quit.store(true);
    copier_work.notify();
    if (copier.joinable()) {
        copier.join();
    }
    pending.clear();
    layers.clear();
    if (pcie_buf != nullptr) {
        ggml_backend_buffer_free(pcie_buf);
        pcie_buf = nullptr;
    }
    hpcache_cuda_tables_free(tables);
    hpcache_doorbell_free(db);
    hpcache_cuda_keepalive_stop(gpu2_keepalive);
    if (gpu2_stage_buf != nullptr) {
        ggml_backend_buffer_free(gpu2_stage_buf);
    } else {
        free(gpu2_stage);
    }
    if (gpu2_backend != nullptr) {
        ggml_backend_free(gpu2_backend);
    }
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
    share_now = std::min(0.9f, std::max(0.0f, opts.pcie_frac));
    *db.frac = int(share_now * 256.0f + 0.5f);
    fill_gtab();
    g2_device_done = getenv("HPCACHE_G2_HOST_DONE") == nullptr || atoi(getenv("HPCACHE_G2_HOST_DONE")) == 0;
#ifdef HPCACHE_EVENT_LOG
    steps_on = getenv("HPCACHE_STEPS") != nullptr && getenv("HPCACHE_STEPS")[0] != 0;
#endif
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

// the same expert math on the second card, over its static pool; ids are slots of that pool
hpcache_decoder::cpu_graph * hpcache_decoder::gpu2_graph_for(layer & l, int m) {
    if (m < 1 || m > l.k || l.n_slots2 == 0) {
        return nullptr;
    }
    if ((int) l.graphs2.size() <= l.k) {
        l.graphs2.resize(l.k + 1);
    }
    if (l.graphs2[m]) {
        return l.graphs2[m].get();
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

    // x, ids and w are windows of the first tensor, so the one allocation below places all of them and one copy fills them
    g->in  = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, gpu2_in_ids_off(l.n_embd) + gpu2_in_max_k * 8 + gpu2_in_bytes_extra);
    g->x   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, l.n_embd, 1, 1);
    g->ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, m, 1);
    g->w   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, m, 1);
    g->x->view_src   = g->in;
    g->x->view_offs  = 0;
    g->ids->view_src  = g->in;
    g->ids->view_offs = gpu2_in_ids_off(l.n_embd);
    g->w->view_src    = g->in;
    g->w->view_offs   = gpu2_in_w_off(l.n_embd);
    g->out = build_experts(ctx, l.g_gate_up, l.g_gate, l.g_up, l.g_down, g->x, g->ids, g->w, l.n_embd, m, l.clamp);

    g->gf = ggml_new_graph_custom(ctx, graph_size, false);
    // the last op of the graph writes the result and the flag into the host memory, no host call is needed to read them
    ggml_tensor * last = g->out;
    if (g2_device_done) {
        op_ctx fc = {};
        fc.decoder = this;
        fc.il      = -1;
        fc.g2_in   = g->in;
        op_ctxs.push_back(fc);
        ggml_tensor * fs[2] = { g->out, g->in };
        last = ggml_custom_4d(ctx, GGML_TYPE_F32, 1, 1, 1, 1, fs, 2, g2_finish_cb, 1, &op_ctxs.back());
        ggml_custom_set_backend_stream(last);
    }
#ifdef HPCACHE_EVENT_LOG
    if (steps_on) {
        // a stamp before the first op, one after the experts and one after the last op, on the stream of this card
        auto stamp_op = [&](int field, ggml_tensor * after) {
            op_ctx sc = {};
            sc.decoder     = this;
            sc.stamp_event = field;
            sc.g2_in       = g->in;
            sc.il          = -1;
            steps_ctxs.push_back(sc);
            ggml_tensor * s[1] = { after };
            ggml_tensor * t = ggml_custom_4d(ctx, GGML_TYPE_F32, 1, 1, 1, 1, s, 1, stamp_g2_cb, 1, &steps_ctxs.back());
            ggml_custom_set_backend_stream(t);
            return t;
        };
        ggml_build_forward_expand(g->gf, stamp_op(HPE_G2_GRAPH_START, g->in));
        ggml_build_forward_expand(g->gf, g->out);
        ggml_build_forward_expand(g->gf, stamp_op(HPE_G2_GRAPH_STOP, g->out));
        if (last != g->out) {
            ggml_build_forward_expand(g->gf, last);
            ggml_build_forward_expand(g->gf, stamp_op(HPE_G2_FINISH, last));
        }
    } else {
        ggml_build_forward_expand(g->gf, last);
    }
#else
    ggml_build_forward_expand(g->gf, last);
#endif

    g->buf = ggml_backend_alloc_ctx_tensors(ctx, gpu2_backend);
    if (g->buf == nullptr) {
        return nullptr;
    }
    l.graphs2[m] = std::move(g);
    return l.graphs2[m].get();
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
    std::string seen;
    for (int32_t i = 0; i < n; ++i) {
        seen += std::string(i > 0 ? ", " : "") + (devices[i].name != nullptr ? devices[i].name : "?") + " (type " +
                std::to_string((int) devices[i].type) + ")";
        if (pool_dev == nullptr && is_cuda_gpu(devices[i])) {
            pool_dev = devices[i].dev;
        }
    }
    if (pool_dev == nullptr) {
        log(GGML_LOG_LEVEL_WARN, "GPU expert pool: no CUDA device (" + std::to_string(n) + " devices from the host: " + seen + ")");
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
    if (ggml_backend_dev_type(pool_dev) == GGML_BACKEND_DEVICE_TYPE_IGPU) {
        // the backend reports system memory for it: a card that only looks integrated (WSL) has NVML numbers,
        // a device on shared memory has none, and a pool there would only copy experts within the same memory
        if (!device_free_bytes(free_bytes, total_bytes)) {
            log(GGML_LOG_LEVEL_WARN, "GPU expert pool: integrated GPU without NVML memory info, no pool");
            pool_dev = nullptr;
            return;
        }
    } else {
        ggml_backend_dev_memory(pool_dev, &free_bytes, &total_bytes);
    }

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
    init_gpu2(slot_bytes, req.down_exps != nullptr && req.down_exps->data != nullptr);
    pool_rank_limit = rank_base + std::min<int64_t>(pool_slots, (int64_t) profile.size() - rank_base);

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
                    " is in buffer type " + name + (t->data == nullptr ? " (no data pointer)" : "") +
                    (host_buffer ? "" : " (not a host buffer)") + ", the layer stays on the CPU");
            return;
        }
        slot_bytes += t->nb[2];
    }

    int n_slots = 0;
    if (!profile.empty()) {
        for (int64_t i = rank_base; i < pool_rank_limit; ++i) {
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
    if (tables.d == nullptr) {
        // layer indices are small; 256 tables of n_expert entries are 0.5 MiB
        hpcache_cuda_tables_alloc(tables, 256, l.n_expert);
    }
    if (tables.d != nullptr && il >= 0 && il < tables.n_tables && l.n_expert == tables.n_expert) {
        l.table_index = il;
        l.d_table = tables.d + size_t(il) * size_t(l.n_expert);
        l.h_table = tables.h + size_t(il) * size_t(l.n_expert);
    }
    if (l.pool_buf == nullptr || l.d_table == nullptr) {
        log(GGML_LOG_LEVEL_WARN, "GPU expert pool: allocation failed at layer " + std::to_string(il));
        l.d_table = nullptr;
        l.h_table = nullptr;
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

    // the first slots come from the profile, in rank order; the last slot stays empty as the spare
    const int n_seed = (opts.spare_slot && n_slots >= 2) ? n_slots - 1 : n_slots;
    l.spare = n_seed < n_slots ? n_slots - 1 : -1;
    int next = 0;
    for (int64_t i = rank_base; i < pool_rank_limit && next < n_seed; ++i) {
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
    stage_table(l);
    flush_tables();
}

// Free memory of the pool device as total minus used, read from NVML. cudaMemGetInfo reports 0 once the share the OS
// grants the process is reached, although the card still has room. Falls back to the backend when NVML is missing.
struct nvml_memory { unsigned long long total, free, used; };

static bool nvml_used_bytes(const char * pci_bus_id, size_t & total, size_t & used) {
    static void * lib = nullptr;
    static int (*get_handle)(const char *, void **) = nullptr;
    static int (*get_memory)(void *, nvml_memory *) = nullptr;
    static bool tried = false;
    if (!tried) {
        tried = true;
#ifdef _WIN32
        HMODULE h = LoadLibraryA("nvml.dll");
        lib = (void *) h;
        auto sym = [&](const char * name) { return h ? (void *) GetProcAddress(h, name) : nullptr; };
#else
        lib = dlopen("libnvidia-ml.so.1", RTLD_NOW);
        auto sym = [&](const char * name) { return lib ? dlsym(lib, name) : nullptr; };
#endif
        auto init = (int (*)()) sym("nvmlInit_v2");
        get_handle = (int (*)(const char *, void **)) sym("nvmlDeviceGetHandleByPciBusId_v2");
        get_memory = (int (*)(void *, nvml_memory *)) sym("nvmlDeviceGetMemoryInfo");
        if (init == nullptr || init() != 0) {
            get_handle = nullptr;
        }
    }
    void * dev = nullptr;
    nvml_memory m = {};
    if (get_handle == nullptr || get_memory == nullptr || pci_bus_id == nullptr || get_handle(pci_bus_id, &dev) != 0 ||
        get_memory(dev, &m) != 0) {
        return false;
    }
    total = (size_t) m.total;
    used  = (size_t) m.used;
    return true;
}

bool hpcache_decoder::device_free_bytes(ggml_backend_dev_t dev, size_t & free_bytes, size_t & total_bytes) const {
    ggml_backend_dev_props props = {};
    ggml_backend_dev_get_props(dev, &props);
    size_t used = 0;
    if (nvml_used_bytes(props.device_id, total_bytes, used)) {
        free_bytes = total_bytes > used ? total_bytes - used : 0;
        return true;
    }
    ggml_backend_dev_memory(dev, &free_bytes, &total_bytes);
    return false;
}

bool hpcache_decoder::device_free_bytes(size_t & free_bytes, size_t & total_bytes) const {
    return device_free_bytes(pool_dev, free_bytes, total_bytes);
}

// The second CUDA card holds a static pool: the top ranks of the profile, loaded once and never swapped, because its
// link is a PCIe 3.0 x1 slot (about 0.55 GB/s, 2.3 ms for one expert). The pool of the first card then starts after
// those ranks (rank_base), so no expert is in both.
void hpcache_decoder::init_gpu2(size_t slot_bytes, bool real_model) {
    rank_base = 0;
    // the memory fit builds a first model without data: leave the second card alone then
    if (!opts.gpu2 || !real_model) {
        return;
    }
    if (profile.empty() || slot_bytes == 0) {
        log(GGML_LOG_LEVEL_WARN, "second GPU: needs an expert profile to know which experts to keep, not used");
        return;
    }
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        const char * name = ggml_backend_dev_name(d);
        if (d == pool_dev || name == nullptr || strncmp(name, "CUDA", 4) != 0) {
            continue;
        }
        if (!opts.gpu2_name.empty() && opts.gpu2_name != name) {
            continue;
        }
        gpu2_dev = d;
        break;
    }
    if (gpu2_dev == nullptr) {
        log(GGML_LOG_LEVEL_WARN, "second GPU: no other CUDA device" +
                (opts.gpu2_name.empty() ? std::string() : " named " + opts.gpu2_name) + ", not used");
        return;
    }
    ggml_backend_dev_props props;
    ggml_backend_dev_get_props(gpu2_dev, &props);
    gpu2_index = hpcache_cuda_device_by_pci(props.device_id);
    if (gpu2_index < 0) {
        log(GGML_LOG_LEVEL_WARN, "second GPU: cannot map " + std::string(ggml_backend_dev_name(gpu2_dev)) + " to a CUDA index, not used");
        gpu2_dev = nullptr;
        return;
    }
    size_t free_bytes = 0, total_bytes = 0;
    device_free_bytes(gpu2_dev, free_bytes, total_bytes);
    gpu2_budget = free_bytes > opts.gpu2_reserve_bytes ? free_bytes - opts.gpu2_reserve_bytes : 0;
    if (opts.gpu2_pool_bytes > 0) {
        gpu2_budget = std::min(gpu2_budget, opts.gpu2_pool_bytes);
    }
    gpu2_slots = int64_t(gpu2_budget / slot_bytes);
    rank_base = std::min<int64_t>(gpu2_slots, (int64_t) profile.size());
    char buf[256];
    snprintf(buf, sizeof(buf), "second GPU %s: %.0f MiB free of %.0f MiB, budget %.0f MiB, about %lld static expert slots",
            ggml_backend_dev_name(gpu2_dev), free_bytes / 1048576.0, total_bytes / 1048576.0, gpu2_budget / 1048576.0,
            (long long) gpu2_slots);
    log(GGML_LOG_LEVEL_INFO, buf);
    gpu2_backend = ggml_backend_dev_init(gpu2_dev, nullptr);
    if (gpu2_backend == nullptr) {
        log(GGML_LOG_LEVEL_WARN, "second GPU: cannot create the backend, not used");
        gpu2_dev = nullptr;
        rank_base = 0;
        gpu2_slots = 0;
    }
    hpcache_cuda_set_device(pool_device_index);   // the memory query above switched to the second card
}

// fill the static pool of one layer with its experts among the first rank_base ranks of the profile
void hpcache_decoder::alloc_gpu2(layer & l, int il) {
    if (gpu2_dev == nullptr || rank_base == 0) {
        return;
    }
    if (l.d_table == nullptr) {
        // the table of the first pool carries the marks that keep the other paths off these experts
        log(GGML_LOG_LEVEL_WARN, "second GPU: layer " + std::to_string(il) + " has no pool on the first card, nothing placed");
        return;
    }
    ggml_tensor * src[4] = { l.t_gate, l.t_up, l.t_down, l.t_gate_up };
    size_t slot_bytes = 0;
    for (ggml_tensor * t : src) {
        if (t == nullptr) {
            continue;
        }
        // the copy needs the original row layout, as for the first pool
        const char * name = t->buffer != nullptr ? ggml_backend_buft_name(ggml_backend_buffer_get_type(t->buffer)) : "";
        const bool host_buffer = (t->buffer != nullptr && ggml_backend_buffer_is_host(t->buffer)) || strcmp(name, "CUDA_Host") == 0;
        if (!host_buffer || strstr(name, "REPACK") != nullptr || strstr(name, "AMX") != nullptr || t->data == nullptr) {
            return;
        }
        slot_bytes += t->nb[2];
    }

    std::vector<int> experts;
    for (int64_t i = 0; i < rank_base; ++i) {
        if (profile[i].first == il && profile[i].second < l.n_expert) {
            experts.push_back(profile[i].second);
        }
    }
    if (gpu2_used + experts.size() * slot_bytes > gpu2_budget) {
        experts.resize((gpu2_budget - gpu2_used) / slot_bytes);
    }
    if (experts.empty()) {
        return;
    }

    const int64_t t0 = adm_now();
    hpcache_cuda_set_device(gpu2_index);
    struct device_guard {
        int back;
        ~device_guard() { if (back >= 0) hpcache_cuda_set_device(back); }
    } guard = { pool_device_index };
    ggml_init_params params = { ggml_tensor_overhead() * 8, nullptr, true };
    l.g_ctx = ggml_init(params);
    if (l.g_ctx == nullptr) {
        return;
    }
    const int n = (int) experts.size();
    ggml_tensor ** dst[4] = { &l.g_gate, &l.g_up, &l.g_down, &l.g_gate_up };
    for (int i = 0; i < 4; ++i) {
        if (src[i] != nullptr) {
            *dst[i] = ggml_new_tensor_3d(l.g_ctx, src[i]->type, src[i]->ne[0], src[i]->ne[1], n);
        }
    }
    l.g_buf = ggml_backend_alloc_ctx_tensors_from_buft(l.g_ctx, ggml_backend_dev_buffer_type(gpu2_dev));
    if (l.g_buf == nullptr) {
        log(GGML_LOG_LEVEL_WARN, "second GPU: allocation failed at layer " + std::to_string(il) + ", the card is full");
        ggml_free(l.g_ctx);
        l.g_ctx = nullptr;
        l.g_gate = l.g_up = l.g_down = l.g_gate_up = nullptr;
        gpu2_budget = gpu2_used;
        return;
    }
    l.slot2_of.assign(l.n_expert, -1);
    for (int s = 0; s < n; ++s) {
        const int e = experts[s];
        for (int t = 0; t < 4; ++t) {
            if (src[t] != nullptr) {
                const size_t nb = src[t]->nb[2];
                ggml_backend_tensor_set(*dst[t], (const char *) src[t]->data + size_t(e) * nb, size_t(s) * nb, nb);
            }
        }
        l.slot2_of[e] = s;
    }
    l.n_slots2 = n;
    gpu2_used += size_t(n) * slot_bytes;
    gpu2_placed += n;
    ++gpu2_layers;
    gpu2_load_us += adm_now() - t0;
}

// The pool is built before the compute buffer of the decode graph exists, so it leaves a reserve for it. Once that
// buffer is allocated, what is still free is idle: add slots to the layers from the next ranks of the profile.
// The pool tensors are referenced by the graph, so they keep their identity: only data, buffer and slot count change.
//
// The first failed allocation ends the growth. NVML still reports a few hundred MiB free at that point (about 330 MiB on
// the 8 GiB RTX 3060 Ti, where the process peaks near 7.7 GiB), and a 27 MiB request fails as well as an 80 MiB one, so
// smaller slots or fewer slots per layer do not help. The driver and WDDM keep a pool of their own for memory
// management (free-space bookkeeping, page tables) that the process cannot allocate from. The DXGI budget of the card
// is 7256 MiB of 8024 MiB (IDXGIAdapter3::QueryVideoMemoryInfo). Growing adds about 80 slots (+1 point of hit rate), no measurable
// speed.
void hpcache_decoder::grow_pool() {
    if (pool_dev == nullptr || pool_used == 0) {
        return;   // no pool yet (a warm-up decode before the graph with the pool), try again at the next decode
    }
    pool_grown = true;
    if (gpu2_dev != nullptr && !gpu2_logged) {
        gpu2_logged = true;
        size_t free_bytes = 0, total_bytes = 0;
        device_free_bytes(gpu2_dev, free_bytes, total_bytes);
        char msg[256];
        snprintf(msg, sizeof(msg), "second GPU pool: %lld static slots in %d layers (%.0f MiB), loaded in %.1f s, %.0f MiB free on the card",
                (long long) gpu2_placed, gpu2_layers, gpu2_used / 1048576.0, gpu2_load_us / 1e6, free_bytes / 1048576.0);
        log(GGML_LOG_LEVEL_INFO, msg);
    }
    if (!opts.grow || profile.empty()) {
        return;
    }
    copier_wait();
    if (!pending.empty()) {
        return;
    }
    hpcache_cuda_set_device(pool_device_index);

    size_t free_bytes = 0, total_bytes = 0;
    device_free_bytes(free_bytes, total_bytes);

    const size_t keep = opts.grow_margin_bytes;
    if (free_bytes <= keep) {
        char msg[200];
        snprintf(msg, sizeof(msg), "GPU expert pool not grown: %.0f MiB free, the margin needs %.0f MiB",
                free_bytes / 1048576.0, keep / 1048576.0);
        log(GGML_LOG_LEVEL_INFO, msg);
        return;
    }
    size_t avail = free_bytes - keep;
    bool full = false;   // an allocation failed: the device is at capacity

    // the next ranks of the profile that the layers can take
    std::map<layer *, std::vector<int>> added;
    int64_t n_added = 0;
    int64_t rank = pool_rank_limit;
    for (; rank < (int64_t) profile.size(); ++rank) {
        const int il = profile[rank].first;
        const int e  = profile[rank].second;
        auto it = layers.find(il);
        if (it == layers.end()) {
            continue;
        }
        layer & l = *it->second;
        if (l.n_slots == 0 || l.p_gate_up != nullptr || e >= l.n_expert || l.slot_of[e] >= 0) {
            continue;
        }
        std::vector<int> & v = added[&l];
        if (l.n_slots + (int) v.size() >= l.n_expert || std::find(v.begin(), v.end(), e) != v.end()) {
            continue;
        }
        size_t slot_bytes = 0;
        for (const ggml_tensor * t : { l.p_gate, l.p_up, l.p_down }) {
            slot_bytes += t != nullptr ? t->nb[2] : 0;
        }
        if (slot_bytes > avail) {
            break;
        }
        avail -= slot_bytes;
        v.push_back(e);
        ++n_added;
    }

    int64_t n_grown = 0;
    size_t  added_bytes = 0;
    int     n_layers_grown = 0;
    for (auto & entry : added) {
        layer & l = *entry.first;
        const std::vector<int> & extra = entry.second;
        if (extra.empty()) {
            continue;
        }
        const int n_old = l.n_slots;
        const int n_new = n_old + (int) extra.size();

        ggml_tensor * cur[4] = { l.p_gate, l.p_up, l.p_down, nullptr };
        // the old buffer goes first: its slots come again from the host copies, so the layer needs no room twice
        hpcache_cuda_sync_device();
        ggml_backend_buffer_free(l.pool_buf);
        l.pool_buf = nullptr;
        int n_got = n_old;
        ggml_backend_buffer_t buf = nullptr;
        for (int n_try : { n_new, n_old }) {
            ggml_init_params params = { ggml_tensor_overhead() * 8, nullptr, true };
            ggml_context * ctx = ggml_init(params);
            ggml_tensor * fresh[4] = { nullptr, nullptr, nullptr, nullptr };
            for (int i = 0; i < 3; ++i) {
                if (cur[i] != nullptr) {
                    fresh[i] = ggml_new_tensor_3d(ctx, cur[i]->type, cur[i]->ne[0], cur[i]->ne[1], n_try);
                }
            }
            buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_dev_buffer_type(pool_dev));
            if (buf != nullptr) {
                for (int i = 0; i < 3; ++i) {
                    if (cur[i] != nullptr) {
                        cur[i]->buffer = fresh[i]->buffer;
                        cur[i]->data   = fresh[i]->data;
                        cur[i]->ne[2]  = n_try;
                        cur[i]->nb[3]  = fresh[i]->nb[3];
                    }
                }
            }
            ggml_free(ctx);
            if (buf != nullptr) {
                n_got = n_try;
                break;
            }
            full = true;   // out of memory, the second try puts the old size back
        }
        GGML_ASSERT(buf != nullptr);
        l.pool_buf = buf;
        const int n_extra = n_got - n_old;
        l.slot_expert.resize(n_got, -1);

        // the old contents come again from the host copies, then the new experts fill the new slots
        for (int s = 0; s < n_old; ++s) {
            if (l.slot_expert[s] >= 0) {
                copy_slot(l, s, l.slot_expert[s]);
            }
        }
        for (int j = 0; j < n_extra; ++j) {
            if (!load_slot(l, n_old + j, extra[j])) {
                break;
            }
            l.score[extra[j]] = 3.0f;
        }
        hpcache_cuda_copy_wait();
        hpcache_cuda_sync_device();
        l.n_slots = n_got;
        stage_table(l);
        if (n_extra == 0) {
            break;   // the first out-of-memory is the capacity, nothing more fits after it
        }

        size_t slot_bytes = 0;
        for (const ggml_tensor * t : { l.p_gate, l.p_up, l.p_down }) {
            slot_bytes += t != nullptr ? t->nb[2] : 0;
        }
        added_bytes += slot_bytes * n_extra;
        n_grown += n_extra;
        ++n_layers_grown;
    }
    flush_tables();
    pool_used += added_bytes;
    pool_slots += n_grown;
    pool_rank_limit = std::min<int64_t>(rank, (int64_t) profile.size());

    size_t free_after = 0;
    device_free_bytes(free_after, total_bytes);
    char buf[256];
    snprintf(buf, sizeof(buf), "GPU expert pool grown: +%lld slots in %d layers (+%.0f MiB), free %.0f -> %.0f MiB, margin %.0f MiB%s",
            (long long) n_grown, n_layers_grown, added_bytes / 1048576.0, free_bytes / 1048576.0, free_after / 1048576.0,
            opts.grow_margin_bytes / 1048576.0, full ? ", device full" : "");
    log(GGML_LOG_LEVEL_INFO, buf);
}

void hpcache_decoder::stage_table(layer & l) {
    if (l.h_table == nullptr) {
        return;
    }
    memcpy(l.h_table, l.slot_of.data(), size_t(l.n_expert) * sizeof(int32_t));
    // -1: not on a GPU, -2: on the second card (neither the pool nor the PCIe path computes it)
    for (int e = 0; e < (int) l.slot2_of.size(); ++e) {
        if (l.slot2_of[e] >= 0) {
            l.h_table[e] = -2;
        }
    }
    tables_dirty_upto = std::max(tables_dirty_upto, l.table_index + 1);
}

void hpcache_decoder::flush_tables() {
    if (tables_dirty_upto > 0) {
        hpcache_cuda_tables_upload(tables, tables_dirty_upto);
        tables_dirty_upto = 0;
    }
}

// copy expert e of the CPU tensors into a slot of the pool
// queue the copy of one expert into a slot, the mapping is not touched
bool hpcache_decoder::copy_slot(layer & l, int slot, int e) {
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
    return true;
}

bool hpcache_decoder::load_slot(layer & l, int slot, int e) {
    if (!copy_slot(l, slot, e)) {
        return false;
    }
    const int old = l.slot_expert[slot];
    if (old >= 0) {
        l.slot_of[old] = -1;
    }
    l.slot_expert[slot] = e;
    l.slot_of[e] = slot;
    return true;
}

void hpcache_decoder::copier_main() {
    if (pool_device_index >= 0) {
        hpcache_cuda_set_device(pool_device_index);
    }
    for (;;) {
        copier_work.wait_until([&] { return copier_quit.load() || copier_has_jobs.load(); });
        std::vector<pending_swap> jobs;
        {
            hpcache_sync::spin_guard guard(copier_lock);
            jobs.swap(copier_jobs);
            copier_has_jobs.store(false);
        }
        if (jobs.empty()) {
            break;
        }
        const int64_t c0 = adm_now();
        for (const pending_swap & j : jobs) {
            copy_slot(*j.l, j.slot, j.expert);
        }
        hpcache_cuda_copy_wait();
        const int64_t c_us = adm_now() - c0;
        {
            hpcache_sync::spin_guard guard(copier_lock);
            ++copy_batches;
            copy_us_sum += c_us;
            copy_us_max = std::max(copy_us_max, c_us);
        }
        copier_busy.store(false);
        copier_done.notify();
    }
}

void hpcache_decoder::copier_submit(const std::vector<pending_swap> & jobs) {
    if (!copier.joinable()) {
        copier = std::thread([this] { copier_main(); });
    }
    {
        hpcache_sync::spin_guard guard(copier_lock);
        copier_jobs = jobs;
    }
    copier_busy.store(true);
    copier_has_jobs.store(true);
    copier_work.notify();
}

void hpcache_decoder::copier_wait() {
    copier_done.wait_until([&] { return !copier_busy.load(); });
}

// map the experts whose copies were queued by the previous round
void hpcache_decoder::publish_pending() {
    if (pending.empty()) {
        return;
    }
    {
        const int64_t w0 = adm_now();
        const bool blocked = copier_busy.load();
        copier_wait();
        const int64_t w_us = adm_now() - w0;
        ++pub_waits;
        pub_blocked += blocked ? 1 : 0;
        pub_wait_us_sum += w_us;
        pub_wait_us_max = std::max(pub_wait_us_max, w_us);
    }
    std::vector<layer *> touched;
    for (const pending_swap & p : pending) {
        if (p.victim >= 0) {
            const int old = p.l->slot_expert[p.victim];
            if (old >= 0) {
                p.l->slot_of[old] = -1;
            }
            p.l->slot_expert[p.victim] = -1;
            p.l->spare = p.victim;
        }
        p.l->slot_expert[p.slot] = p.expert;
        p.l->slot_of[p.expert]   = p.slot;
        touched.push_back(p.l);
    }
    pending.clear();
    std::sort(touched.begin(), touched.end());
    touched.erase(std::unique(touched.begin(), touched.end()), touched.end());
    for (layer * l : touched) {
        stage_table(*l);
    }
    flush_tables();
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
    const bool from_arena = attach_arena(*l);
    if (opts.prefault && !from_arena) {
        prefault_layer(*l, req.il);
    }
    if (from_arena && opts.pcie) {
        l->pcie = init_pcie(*l);
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

        // one benchmark per distinct (gate, up, down) type combination, layers are built one at a time
        static std::set<std::tuple<int, int, int>> bench_seen;
        const bool bench_new = l->fast && getenv("HPCACHE_BENCH") != nullptr &&
                bench_seen.insert({ (int) l->t_gate->type, (int) l->t_up->type, (int) l->t_down->type }).second;
        if (bench_new) {
            // dot product speed with the row in L1, one thread: the compute side of the kernel
            std::vector<uint8_t> xq(std::max(d.gate_xq_bytes, d.down_xq_bytes));
            std::vector<float> xf(d.n_embd);
            for (size_t q = 0; q < xf.size(); ++q) {
                xf[q] = sinf(float(q) * 0.37f) * 2.0f;
            }
            d.gate_quant(xf.data(), xq.data(), d.n_embd);
            {
                const auto * stock = get_traits(l->t_gate->type);
                double worst = 0.0;
                for (int row = 0; row < 64; ++row) {
                    float a = 0, b = 0;
                    d.gate_dot((int) d.n_embd, &a, 0, d.gate + size_t(row) * d.gate_row, 0, xq.data(), 0, 1);
                    stock->vec_dot((int) d.n_embd, &b, 0, d.gate + size_t(row) * d.gate_row, 0, xq.data(), 0, 1);
                    worst = std::max(worst, (double) fabsf(a - b) / (fabsf(b) + 1e-3f));
                }
                char cb[120];
                snprintf(cb, sizeof(cb), "bench gate dot vs stock: worst relative difference %g", worst);
                log(GGML_LOG_LEVEL_INFO, cb);
            }
            const int reps = 200000;
            float s = 0, v = 0;
            auto t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < reps; ++r) {
                d.gate_dot((int) d.n_embd, &v, 0, d.gate + (size_t(r) & 7) * d.gate_row, 0, xq.data(), 0, 1);
                s += v;
            }
            double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
            char buf[260];
            snprintf(buf, sizeof(buf), "bench layer %d types: gate %s up %s down %s, n_embd %lld n_ff %lld", req.il, ggml_type_name(l->t_gate->type),
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
                    if (row < 3) {
                        char sb[100];
                        snprintf(sb, sizeof(sb), "bench down row %d: own %g stock %g", row, a, b);
                        log(GGML_LOG_LEVEL_INFO, sb);
                    }
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
    alloc_gpu2(*l, req.il);
    if (l->n_slots2 > 0) {
        stage_table(*l);
        flush_tables();
    }
    return l;
}

static void publish_cb(ggml_tensor * dst, int, int, void * userdata);
static void remap_ids_cb(ggml_tensor * dst, int, int, void * userdata);
static void remap_w_cb(ggml_tensor * dst, int, int, void * userdata);
static void pcie_stage_cb(ggml_tensor * dst, int, int, void * userdata);
static void pcie_join_cb(ggml_tensor * dst, int, int, void * userdata);
static void pcie_ids_cb(ggml_tensor * dst, int, int, void * userdata);
static void pcie_w_cb(ggml_tensor * dst, int, int, void * userdata);
static void wait_merge_cb(ggml_tensor * dst, int, int, void * userdata);
#ifdef HPCACHE_EVENT_LOG
static void stamp_cb(ggml_tensor * dst, int, int, void * userdata);
#endif

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
        hpcache_sync::spin_guard lock(layers_lock);
        auto it = layers.find(req.il);
        if (it != layers.end() && (it->second->t_up != req.up_exps || it->second->t_gate != req.gate_exps ||
                it->second->t_down != req.down_exps || it->second->t_gate_up != req.gate_up_exps)) {
            // a different model instance, as when the memory fit loads one first: size the pool again
            copier_wait();
            pending.clear();
            layers.clear();
            pool_init = false;
            pool_grown = false;
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

    op_ctx octx = { this, req.il, l->d_table };
    octx.n_expert = req.n_expert;
    // ggml-cuda skips negative expert ids in the quantized matvec (mmvq.cu), so a missed expert costs nothing
    static const bool skip_miss = [] { const char * e = getenv("HPCACHE_SKIP_MISS"); return e == nullptr || atoi(e) != 0; }();
    if (skip_miss && l->n_slots > 0) {
        bool quantized = true;
        for (const ggml_tensor * t : { l->p_gate_up, l->p_gate, l->p_up, l->p_down }) {
            quantized = quantized && (t == nullptr || ggml_is_quantized(t->type));
        }
        if (quantized) {
            octx.miss_id = -1;
        }
    }
    if (l->pcie) {
        octx.pcie_m = std::min(opts.pcie_max, (int) req.n_expert_used);
        octx.pcie_frac = db.d_frac;
        octx.stage = &l->stage;
    }
    op_ctxs.push_back(octx);
    void * ud = &op_ctxs.back();

    ggml_tensor * srcs[3] = { req.cur, req.selected_experts, req.weights };
    ggml_tensor * pub = ggml_custom_4d(req.ctx, GGML_TYPE_F32, 1, 1, 1, 1, srcs, 3, publish_cb, 1, ud);
    ggml_custom_set_backend_stream(pub);

    ggml_tensor * wait_src[1] = { pub };
    ggml_tensor * merged = ggml_custom_4d(req.ctx, GGML_TYPE_F32, n_embd, 1, 1, 1, wait_src, 1, wait_merge_cb, 1, ud);
    ggml_custom_set_backend_stream(merged);

#ifdef HPCACHE_EVENT_LOG
    // event log: a stamp on the stream between the first card's own work and the wait, and one after the layer
    auto stamp_op = [&](int field, ggml_tensor * after) {
        op_ctx sc = octx;
        sc.stamp_event = field;
        steps_ctxs.push_back(sc);
        ggml_tensor * s[1] = { after };
        ggml_tensor * t = ggml_custom_4d(req.ctx, GGML_TYPE_F32, 1, 1, 1, 1, s, 1, stamp_cb, 1, &steps_ctxs.back());
        ggml_custom_set_backend_stream(t);
        return t;
    };
    auto finish = [&](ggml_tensor * gpu_last, ggml_tensor * result) {
        if (steps_on) {
            // the wait takes the stamp as its second source, so that it is queued after the first card's experts
            merged->src[1] = stamp_op(HPE_OWN_EXPERTS_STOP, gpu_last != nullptr ? gpu_last : pub);
            ggml_build_forward_expand(req.gf, result);
            ggml_build_forward_expand(req.gf, stamp_op(HPE_LAYER_STOP, result));
        }
        return result;
    };
#else
    auto finish = [&](ggml_tensor *, ggml_tensor * result) {
        return result;
    };
#endif

#ifdef HPCACHE_EVENT_LOG
    // The attention and the router are in the graph before the provider is called, so the two stamps of the dense steps
    // go into the node list by hand: one after the last node of the previous layer, one before the router.
    // The records are those of the previous layer, the sequence number has not moved yet.
    if (steps_on) {
        char name_last[GGML_MAX_NAME];
        char name_route[GGML_MAX_NAME];
        snprintf(name_last, sizeof(name_last), "l_last-%d", req.il - 1);
        snprintf(name_route, sizeof(name_route), "ffn_moe_logits-%d", req.il);
        int at_last = -1;
        int at_route = -1;
        for (int i = 0; i < req.gf->n_nodes; ++i) {
            const char * nm = req.gf->nodes[i]->name;
            at_last  = req.il > 0 && strcmp(nm, name_last) == 0 ? i : at_last;
            at_route = strcmp(nm, name_route) == 0 ? i : at_route;
        }
        auto insert_at = [&](int pos, ggml_tensor * t) {
            std::memmove(req.gf->nodes + pos + 1, req.gf->nodes + pos, (req.gf->n_nodes - pos) * sizeof(ggml_tensor *));
            req.gf->nodes[pos] = t;
            req.gf->n_nodes++;
            t->flags |= GGML_TENSOR_FLAG_COMPUTE;
            ggml_hash_insert(&req.gf->visited_hash_set, t);
        };
        if (req.il == 0 && req.gf->n_nodes > 1) {
            HPCACHE_LOG(ring_main, HPE_BUILD, (uint32_t) n_decode.load(), -2);
            insert_at(1, stamp_op(HPE_GRAPH_START, req.gf->nodes[0]));
            at_route++;
        }
        if (at_route > 0) {
            insert_at(at_route, stamp_op(HPE_ROUTER_START, req.cur));
        }
        if (at_last >= 0) {
            insert_at(at_last + 1, stamp_op(HPE_LAYER_IN, req.gf->nodes[at_last]));
        }
    }
#endif

    // The share of the missed experts that the GPU computes from the arena, through PCIe. The fetch is queued on a stream of
    // its own right after the ring, and the pool experts below are queued ahead of the join, so they compute while it copies.
    // The ops that read the fetched experts come after the pool experts, below.
    ggml_tensor * stg = nullptr;
    if (l->pcie) {
        ggml_tensor * ssrc[2] = { req.selected_experts, pub };
        stg = ggml_custom_4d(req.ctx, GGML_TYPE_F32, 1, 1, 1, 1, ssrc, 2, pcie_stage_cb, 1, ud);
        ggml_custom_set_backend_stream(stg);
    }
    ggml_tensor * pcie_out = nullptr;
    // the PCIe ops, queued after `after` (the last of the pool experts, or the fetch when there is no pool)
    auto build_pcie = [&](ggml_tensor * after) {
        const int m = op_ctxs.back().pcie_m;
        ggml_tensor * jsrc[2] = { stg, after };
        ggml_tensor * stj = ggml_custom_4d(req.ctx, GGML_TYPE_F32, 1, 1, 1, 1, jsrc, 2, pcie_join_cb, 1, ud);
        ggml_custom_set_backend_stream(stj);
        ggml_tensor * psrc[2] = { req.selected_experts, stj };
        ggml_tensor * p_ids = ggml_custom_4d(req.ctx, GGML_TYPE_I32, m, 1, 1, 1, psrc, 2, pcie_ids_cb, 1, ud);
        ggml_custom_set_backend_stream(p_ids);
        ggml_tensor * pw_src[3] = { req.selected_experts, req.weights, stj };
        ggml_tensor * p_w = ggml_custom_4d(req.ctx, GGML_TYPE_F32, 1, m, 1, 1, pw_src, 3, pcie_w_cb, 1, ud);
        ggml_custom_set_backend_stream(p_w);
        ggml_tensor * px = ggml_reshape_3d(req.ctx, req.cur, n_embd, 1, 1);
        pcie_out = build_experts(req.ctx, l->a_gate_up, l->a_gate, l->a_up, l->a_down, px, p_ids, p_w, n_embd, m, req.swiglu_clamp);
    };

    if (l->n_slots == 0) {
        if (stg != nullptr) {
            build_pcie(stg);
            return finish(pcie_out, ggml_add(req.ctx, pcie_out, ggml_reshape_2d(req.ctx, merged, n_embd, 1)));
        }
        return finish(nullptr, merged);
    }

    // experts in the pool run on the GPU while the CPU serves the rest
    // the remap ops read pub, so that they are queued after the ring
    // the fetch is a source of the first pool op, so that it is queued before the pool experts
    ggml_tensor * ids_src[3] = { req.selected_experts, pub, stg };
    ggml_tensor * slot_ids = ggml_custom_4d(req.ctx, GGML_TYPE_I32, k, 1, 1, 1, ids_src, stg != nullptr ? 3 : 2, remap_ids_cb, 1, ud);
    ggml_custom_set_backend_stream(slot_ids);

    ggml_tensor * w_src[3] = { req.selected_experts, req.weights, pub };
    ggml_tensor * w_hit = ggml_custom_4d(req.ctx, GGML_TYPE_F32, 1, k, 1, 1, w_src, 3, remap_w_cb, 1, ud);
    ggml_custom_set_backend_stream(w_hit);

    ggml_tensor * x = ggml_reshape_3d(req.ctx, req.cur, n_embd, 1, 1);
    ggml_tensor * hit = build_experts(req.ctx, l->p_gate_up, l->p_gate, l->p_up, l->p_down, x, slot_ids, w_hit,
            n_embd, k, req.swiglu_clamp);

    // the GPU part is the first operand, so it is queued before the wait
    if (stg != nullptr) {
        build_pcie(hit);
        hit = ggml_add(req.ctx, hit, pcie_out);
    }
    return finish(hit, ggml_add(req.ctx, hit, ggml_reshape_2d(req.ctx, merged, n_embd, 1)));
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
            (int) dst->ne[0], c->miss_id, (int32_t *) dst->data);
}

static void remap_w_cb(ggml_tensor * dst, int, int, void * userdata) {
    auto * c = static_cast<hpcache_decoder::op_ctx *>(userdata);
    hpcache_cuda_remap_weights(ggml_custom_get_backend_stream(dst), (const int32_t *) dst->src[0]->data,
            (const float *) dst->src[1]->data, c->table, (int) dst->ne[1], (float *) dst->data);
}

static void pcie_stage_cb(ggml_tensor * dst, int, int, void * userdata) {
    auto * c = static_cast<hpcache_decoder::op_ctx *>(userdata);
    static const bool side = getenv("HPCACHE_PCIE_SIDE") == nullptr || atoi(getenv("HPCACHE_PCIE_SIDE")) != 0;
    hpcache_cuda_pcie_stage(ggml_custom_get_backend_stream(dst), (const int32_t *) dst->src[0]->data, c->table,
            (int) dst->src[0]->ne[0], c->n_expert, c->pcie_m, c->pcie_frac, c->decoder->gtab_dev, *c->stage, side);
}

static void pcie_join_cb(ggml_tensor * dst, int, int, void *) {
    static const bool side = getenv("HPCACHE_PCIE_SIDE") == nullptr || atoi(getenv("HPCACHE_PCIE_SIDE")) != 0;
    if (side) {
        hpcache_cuda_pcie_stage_join(ggml_custom_get_backend_stream(dst));
    }
}

static void pcie_ids_cb(ggml_tensor * dst, int, int, void * userdata) {
    auto * c = static_cast<hpcache_decoder::op_ctx *>(userdata);
    hpcache_cuda_pcie_select(ggml_custom_get_backend_stream(dst), (const int32_t *) dst->src[0]->data, nullptr, c->table,
            (int) dst->src[0]->ne[0], c->n_expert, (int) dst->ne[0], c->pcie_frac, c->decoder->gtab_dev, (int32_t *) dst->data, nullptr);
}

static void pcie_w_cb(ggml_tensor * dst, int, int, void * userdata) {
    auto * c = static_cast<hpcache_decoder::op_ctx *>(userdata);
    hpcache_cuda_pcie_select(ggml_custom_get_backend_stream(dst), (const int32_t *) dst->src[0]->data,
            (const float *) dst->src[1]->data, c->table, (int) dst->src[0]->ne[0], c->n_expert, (int) dst->ne[1],
            c->pcie_frac, c->decoder->gtab_dev, nullptr, (float *) dst->data);
}

static void wait_merge_cb(ggml_tensor * dst, int, int, void * userdata) {
    auto * c = static_cast<hpcache_decoder::op_ctx *>(userdata);
    c->decoder->wait_merge(ggml_custom_get_backend_stream(dst), dst);
}

#ifdef HPCACHE_EVENT_LOG
static void stamp_cb(ggml_tensor * dst, int, int, void * userdata) {
    auto * c = static_cast<hpcache_decoder::op_ctx *>(userdata);
    c->decoder->stamp(ggml_custom_get_backend_stream(dst), c->stamp_event, c->il);
}

static void stamp_g2_cb(ggml_tensor * dst, int, int, void * userdata) {
    auto * c = static_cast<hpcache_decoder::op_ctx *>(userdata);
    c->decoder->stamp_g2(ggml_custom_get_backend_stream(dst), c);
}

void hpcache_decoder::stamp(void * s, int event, int il) {
    if (ring_lead != nullptr) {
        hpcache_cuda_device_log(s, *ring_lead, (uint16_t) event, db.d_seq, il);
    }
}

void hpcache_decoder::stamp_g2(void * s, const op_ctx * c) {
    // the sequence number sits in the input block of the second card's graph, in its own memory
    const auto * seq = (const volatile uint32_t *) ((const char *) c->g2_in->data + gpu2_in_stamp_off(db.n_embd));
    if (ring_g2 != nullptr) {
        hpcache_cuda_device_log(s, *ring_g2, (uint16_t) c->stamp_event, seq, -1);
    }
}

// the rings are made before the producers start: the logger thread does not look at a ring list that grows
void hpcache_decoder::steps_start() {
    ring_lead = metrics.add_ring(HPC_LEAD, pool_device_index >= 0 ? pool_device_index : hpcache_cuda_get_device());
    ring_g2   = gpu2_index >= 0 && gpu2_active() ? metrics.add_ring(HPC_G2, gpu2_index) : nullptr;
    ring_cpu  = metrics.add_ring(HPC_HOST, -1);
    ring_w2   = metrics.add_ring(HPC_HOST, -1);
    ring_main = metrics.add_ring(HPC_HOST, -1);
    db.lead_m = ring_lead != nullptr ? ring_lead->view() : hpcache_metric_dev();
    if (!metrics.open(getenv("HPCACHE_STEPS"))) {
        log(GGML_LOG_LEVEL_WARN, "event log: cannot open the file");
    }
}

// the clocks of the cards and the host, put on one timeline for the trace
void hpcache_decoder::steps_calibrate() {
    const bool first_time = !steps_calibrated;
    steps_calibrated = true;
    long long rtt = 0;
    // the first card is the pool device; the current device can be the second card, as after the load of its pool
    const int current = hpcache_cuda_get_device();
    const int dev = pool_device_index >= 0 ? pool_device_index : current;
    if (dev >= 0 && hpcache_cuda_clock_offset(dev, steps_off_lead, rtt)) {
        if (first_time) log(GGML_LOG_LEVEL_INFO, "step trace: first card clock offset " + std::to_string(steps_off_lead) + " ns, best round trip " +
                std::to_string(rtt / 1000) + " us");
    } else {
        log(GGML_LOG_LEVEL_WARN, "step trace: cannot calibrate the first card's clock");
    }
    if (gpu2_index >= 0 && gpu2_active()) {
        if (hpcache_cuda_clock_offset(gpu2_index, steps_off_g2, rtt)) {
            if (first_time) log(GGML_LOG_LEVEL_INFO, "step trace: second card clock offset " + std::to_string(steps_off_g2) + " ns, best round trip " +
                    std::to_string(rtt / 1000) + " us");
        } else {
            log(GGML_LOG_LEVEL_WARN, "step trace: cannot calibrate the second card's clock");
        }
    }
    if (current >= 0) {
        hpcache_cuda_set_device(current);
    }
    steps_host_1 = hpcache_metric_now();
    if (first_time) {
        steps_off_lead_0 = steps_off_lead;
        steps_off_g2_0   = steps_off_g2;
        steps_host_0     = steps_host_1;
    }
}
#endif

static void g2_finish_cb(ggml_tensor * dst, int, int, void * userdata) {
    auto * c = static_cast<hpcache_decoder::op_ctx *>(userdata);
    c->decoder->g2_finish(ggml_custom_get_backend_stream(dst), dst, c);
}

void hpcache_decoder::g2_finish(void * s, const ggml_tensor * dst, const op_ctx * c) {
    const auto * seq = (const volatile uint32_t *) ((const char *) c->g2_in->data + gpu2_in_stamp_off(db.n_embd));
    hpcache_cuda_g2_finish(s, db, (const float *) dst->src[0]->data, seq);
}

void hpcache_decoder::serve(uint32_t seq) {
    const int il = *db.layer;
    HPCACHE_LOG(ring_cpu, HPE_CPU_START, seq, il);
    uint32_t n_cpu_experts = 0;
    uint32_t counts = 0;   // event log: experts on the CPU | on the first card through PCIe << 8 | on the second card << 16 | pool hits << 24
    HPCACHE_TRACE("serve seq %u layer %d", seq, il);

    layer * l = nullptr;
    {
        hpcache_sync::spin_guard lock(layers_lock);
        const auto it = layers.find(il);
        if (it != layers.end()) {
            l = it->second.get();
        }
    }

    // HPCACHE_BLIND=1: answer without any expert math, to time everything that is not math (diagnostic, output is garbage)
    static const bool blind = getenv("HPCACHE_BLIND") != nullptr;
    if (blind) {
        memset(db.y, 0, size_t(db.n_embd) * sizeof(float));
        n_serve++;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        *db.flag = seq;
        return;
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
        int     n2 = 0;   // routed to the second card
        for (int i = 0; i < db.k; ++i) {
            const int32_t e = db.ids[i];
            if (e < 0 || e >= l->n_expert) {
                continue;
            }
            if (!l->slot2_of.empty() && l->slot2_of[e] >= 0) {
                ++n2;
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
        n_hit    += db.k - m - n2;
        n_gpu2   += n2;
        int g_logged = 0;
        const int hits_here = db.k - m - n2;
        if (l->pcie && m > 0) {
            // the GPU takes the last g misses in routing order (pcie_select_kernel), the CPU the first m - g
            int g = gtab_on ? int(db.gtab[hpcache_gtab_index(n2 > 0, hits_here, m)]) : (m * int(*db.frac)) >> 8;
            g = std::min(g, std::min(opts.pcie_max, db.k));
            g = std::min(g, m);
            m -= g;
            n_pcie_experts += g;
            g_logged = g;
        }

        n_cpu_experts = m > 0 ? m : 0;
        counts = n_cpu_experts | uint32_t(g_logged) << 8 | uint32_t(n2) << 16 | uint32_t(hits_here) << 24;
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

    HPCACHE_LOG(ring_cpu, HPE_CPU_STOP, seq, il, counts);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    *db.flag = seq;
    HPCACHE_TRACE("served seq %u", seq);
}

// The follower makes no CUDA call: it reads the doorbell with plain loads and writes the answer to pinned host
// memory. A CUDA call from this thread can wait behind a launch that is blocked on a full queue, and that launch
// waits for the GPU, which waits for this thread. The kernels are pushed to the device by the launching thread.
void hpcache_decoder::follower() {
    uint32_t seen = 0;

    for (;;) {
        follower_sig.wait_until([&] { return armed.load() || quit.load(); });
        if (quit.load()) {
            return;
        }

        while (armed.load(std::memory_order_acquire)) {
            const uint32_t seq = *db.seq;
            if (seq != seen) {
                serve(seq);
                seen = seq;
                continue;
            }
            HPCACHE_PAUSE();
        }

        idle.store(true);
        follower_sig.notify();
    }
}

// Everything the worker of the second card would otherwise create while a token is running: the pinned staging block, the
// graph of every layer for every number of experts, and the lazy load of the kernels that those graphs run. A CUDA
// allocation or module load takes locks of the runtime, and one that waits behind a launch of the graph thread, while the
// first card's wait kernel spins for the worker's answer, ends in a deadlock. Runs on the graph thread before the worker
// exists.
void hpcache_decoder::warm_gpu2() {
    gpu2_warm = true;
    const int64_t t0 = adm_now();
    const int dev = hpcache_cuda_get_device();

    const size_t bytes = gpu2_in_stamp_off(db.n_embd) + gpu2_in_bytes_extra;
    ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(gpu2_dev);
    if (host_buft != nullptr) {
        gpu2_stage_buf = ggml_backend_buft_alloc_buffer(host_buft, bytes);
    }
    gpu2_stage = gpu2_stage_buf != nullptr ? (char *) ggml_backend_buffer_get_base(gpu2_stage_buf) : (char *) malloc(bytes);
    memset(gpu2_stage, 0, bytes);   // x = 0, slot 0 and weight 0 for every expert: a valid call that computes zeros

    std::vector<float> y(db.n_embd);
    int n_graphs = 0;
    for (auto & kv : layers) {
        layer & l = *kv.second;
        if (l.n_slots2 == 0) {
            continue;
        }
        for (int m = 1; m <= l.k && m <= gpu2_in_max_k; ++m) {
            cpu_graph * g = gpu2_graph_for(l, m);
            if (g == nullptr) {
                continue;
            }
            // twice: ggml records the CUDA graph of the card on the second run, which must not happen later in the worker thread
            for (int run = 0; run < 2; ++run) {
                ggml_backend_tensor_set_async(gpu2_backend, g->in, gpu2_stage, 0, gpu2_in_w_off(db.n_embd) + size_t(m) * sizeof(float));
                ggml_backend_graph_compute(gpu2_backend, g->gf);
                ggml_backend_tensor_get(g->out, y.data(), 0, y.size() * sizeof(float));
            }
            ++n_graphs;
        }
    }
    if (dev >= 0) {
        hpcache_cuda_set_device(dev);
    }
    log(GGML_LOG_LEVEL_INFO, "second GPU: " + std::to_string(n_graphs) + " graphs built and run once in " +
            std::to_string((adm_now() - t0) / 1000) + " ms");
}

// The worker of the second card answers every ring: the experts of the layer that live there run on that card, the
// result goes to pinned memory, and the flag tells the merge on the first card that it can read it. A layer without
// experts there answers with zeros.
void hpcache_decoder::serve2(uint32_t seq) {
    const int il = *db.layer;
    HPCACHE_LOG(ring_w2, HPE_G2_WORKER_START, seq, il);
    uint32_t m_two = 0;
    layer * l = nullptr;
    {
        hpcache_sync::spin_guard lock(layers_lock);
        const auto it = layers.find(il);
        if (it != layers.end()) {
            l = it->second.get();
        }
    }

    if (gpu2_stage == nullptr) {
        const size_t bytes = gpu2_in_stamp_off(db.n_embd) + gpu2_in_bytes_extra;
        ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(gpu2_dev);
        if (host_buft != nullptr) {
            gpu2_stage_buf = ggml_backend_buft_alloc_buffer(host_buft, bytes);
        }
        gpu2_stage = gpu2_stage_buf != nullptr ? (char *) ggml_backend_buffer_get_base(gpu2_stage_buf) : (char *) malloc(bytes);
    }
    int32_t * ids = (int32_t *) (gpu2_stage + gpu2_in_ids_off(db.n_embd));
    float *   w   = (float *)   (gpu2_stage + gpu2_in_w_off(db.n_embd));
    int       m   = 0;
    if (l != nullptr && l->n_slots2 > 0) {
        for (int i = 0; i < db.k && m < gpu2_in_max_k; ++i) {
            const int32_t e = db.ids[i];
            if (e >= 0 && e < l->n_expert && l->slot2_of[e] >= 0) {
                ids[m] = l->slot2_of[e];
                w[m]   = db.w[i];
                ++m;
            }
        }
    }
    if (m > 0) {
        memcpy(gpu2_stage, db.x, size_t(db.n_embd) * sizeof(float));
        m_two = m;
        const int64_t t0 = adm_now();
        cpu_graph * g = gpu2_graph_for(*l, m);
        if (g == nullptr) {
            GGML_ABORT("moe-hpcache: no graph on the second GPU for layer %d with %d experts", il, m);
        }
        const int64_t t1 = adm_now();
        // the ops of the graph read the sequence number from the input block
        *(uint32_t *) (gpu2_stage + gpu2_in_stamp_off(db.n_embd)) = seq;
        const size_t upload = gpu2_in_stamp_off(db.n_embd) + sizeof(uint32_t);
        HPCACHE_LOG(ring_w2, HPE_G2_UPLOAD_START, seq, il);
        // queued on the stream of the card, the result read below waits for all of them
        ggml_backend_tensor_set_async(gpu2_backend, g->in, gpu2_stage, 0, upload);
        const int64_t t2 = adm_now();
        if (g2_device_done) {
            // the last op of the graph writes y2 and the flag, the next request cannot come before it has
            ggml_backend_graph_compute_async(gpu2_backend, g->gf);
            g2_n++; g2_m += m; g2_graph_us += t1 - t0; g2_set_us += t2 - t1; g2_compute_us += adm_now() - t2;
            HPCACHE_LOG(ring_w2, HPE_G2_WORKER_STOP, seq, il, (uint32_t) m_two);
            return;
        }
        ggml_backend_graph_compute(gpu2_backend, g->gf);
        const int64_t t3 = adm_now();
        ggml_backend_tensor_get(g->out, db.y2, 0, size_t(db.n_embd) * sizeof(float));
        const int64_t t4 = adm_now();
        HPCACHE_LOG(ring_w2, HPE_G2_DOWNLOAD_STOP, seq, il);
        g2_n++; g2_m += m; g2_graph_us += t1 - t0; g2_set_us += t2 - t1; g2_compute_us += t3 - t2; g2_get_us += t4 - t3;
    } else {
        g2_empty++;
        memset(db.y2, 0, size_t(db.n_embd) * sizeof(float));
    }
    HPCACHE_LOG(ring_w2, HPE_G2_WORKER_STOP, seq, il, (uint32_t) m_two);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    *db.flag2 = seq;
}

// The worker of the second card waits on the answer of every layer, so it gets a core of its own: the last one that the
// process may use. Its priority stays normal: a spinning thread of higher priority starves the graph thread that has to
// publish the next layer when that one is scheduled on the same core. Returns the core, or -1.
static int pin_worker_thread() {
#ifdef _WIN32
    DWORD_PTR process_mask = 0, system_mask = 0;
    if (!GetProcessAffinityMask(GetCurrentProcess(), &process_mask, &system_mask) || process_mask == 0) {
        return -1;
    }
    int core = 63;
    while (core > 0 && !(process_mask & (DWORD_PTR(1) << core))) {
        --core;
    }
    if (SetThreadAffinityMask(GetCurrentThread(), DWORD_PTR(1) << core) == 0) {
        return -1;
    }
    return core;
#else
    return -1;
#endif
}

void hpcache_decoder::follower2() {
    uint32_t seen = 0;

    // off by default: the CPU pool threads spin on cores that the process mask does not tell apart, and a worker that
    // shares its core with one of them waits for a scheduler slice at every layer
    const int core = getenv("HPCACHE_G2_PIN") != nullptr ? pin_worker_thread() : -1;
    log(GGML_LOG_LEVEL_INFO, core >= 0 ? "second GPU worker: pinned to core " + std::to_string(core)
                                       : "second GPU worker: not pinned");

    for (;;) {
        follower2_sig.wait_until([&] { return armed.load() || quit.load(); });
        if (quit.load()) {
            return;
        }

        while (armed.load(std::memory_order_acquire)) {
            const uint32_t seq = *db.seq;
            if (seq != seen) {
                serve2(seq);
                seen = seq;
                continue;
            }
            HPCACHE_PAUSE();
        }

        idle2.store(true);
        follower2_sig.notify();
    }
}

// called by the thread that runs the graph, never concurrently with end()
void hpcache_decoder::arm() {
    if (armed.load(std::memory_order_acquire)) {
        return;
    }
#ifdef HPCACHE_EVENT_LOG
    if (steps_on && ring_main == nullptr) {
        steps_start();
    }
#endif
    if (gtab_on && gtab_dev == nullptr) {
        gtab_dev = hpcache_cuda_upload(pool_device_index >= 0 ? pool_device_index : hpcache_cuda_get_device(), (const void *) db.gtab,
                hpcache_gtab_size);
        if (gtab_dev == nullptr) {
            log(GGML_LOG_LEVEL_ERROR, "cannot copy the split table to the first card, the share is used");
            gtab_on = false;
        }
    }
    if (!thread.joinable()) {
        thread = std::thread([this] { follower(); });
    }
    idle.store(false);
    db.two = gpu2_active();
    if (db.two) {
        if (!gpu2_warm) {
            warm_gpu2();
        }
    }
    // no worker is running yet and no kernel of the graph spins, so the probe kernels cannot meet a runtime lock
#ifdef HPCACHE_EVENT_LOG
    if (steps_on && !steps_calibrated) {
        steps_calibrate();
    }
#endif
    if (db.two) {
        if (!thread2.joinable()) {
            thread2 = std::thread([this] { follower2(); });
        }
        idle2.store(false);
        if (gpu2_keepalive == nullptr && opts.gpu2_keepalive) {
            gpu2_keepalive = hpcache_cuda_keepalive_start(gpu2_index);
            log(GGML_LOG_LEVEL_INFO, "second GPU: keep-alive kernel running");
        }
    }
    armed.store(true, std::memory_order_release);
    follower_sig.notify();
    if (db.two) {
        follower2_sig.notify();
    }
}

void hpcache_decoder::begin(const llama_moe_decode_info & info) {
    // only single-token graphs contain the doorbell ops
    if (info.n_tokens != 1 || db.block == nullptr || !ok()) {
        return;
    }
    HPCACHE_LOG(ring_main, HPE_BEGIN_ENTER, (uint32_t) n_decode.load(), -2);
    HPCACHE_TRACE("begin");
    if (!pool_grown) {
        grow_pool();
    }
    if (cpu_pool) {
        cpu_pool->set_active(true);
    }
    decode_start_us.store(std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
    arm();
    HPCACHE_LOG(ring_main, HPE_BEGIN_ARMED, (uint32_t) n_decode.load(), -2);
}

// Runs while the device and the follower are idle. Per layer, the best missed expert replaces
// the coldest resident one, and the globally best swaps of the round are applied.
void hpcache_decoder::admit() {
    int64_t a0 = adm_now();
    publish_pending();
    int64_t a1 = adm_now();
    g_adm_pub += a1 - a0;

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
            if (s == l.spare) {
                continue;
            }
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

    const int64_t a2 = adm_now();
    g_adm_sel += a2 - a1;
    auto by_gain = [](const swap & a, const swap & b) { return a.gain > b.gain; };
    std::sort(fills.begin(), fills.end(), by_gain);
    std::sort(swaps.begin(), swaps.end(), by_gain);
    if ((int) fills.size() > opts.fill_swaps) {
        fills.resize(opts.fill_swaps);
    }
    if ((int) swaps.size() > opts.max_swaps) {
        swaps.resize(opts.max_swaps);
    }

    // queue every copy. A replaced expert is unmapped now, so no step reads a slot that is being written;
    // the new experts are mapped by the next round once their data has landed
    std::vector<layer *> touched;
    auto apply = [&](const swap & s) {
        const int old = s.l->slot_expert[s.slot];
        if (old >= 0 && s.l->spare >= 0) {
            // the copy goes to the spare slot, the victim stays mapped until the data has landed
            pending.push_back({ s.l, s.l->spare, s.expert, s.slot });
            s.l->spare = -1;
            s.l->score[s.expert] = std::max(s.l->score[s.expert], 3.0f);
            ++n_swapped;
            return;
        }
        if (old >= 0) {
            s.l->slot_of[old] = -1;
        }
        s.l->slot_expert[s.slot] = -1;
        pending.push_back({ s.l, s.slot, s.expert });
        s.l->score[s.expert] = std::max(s.l->score[s.expert], 3.0f);
        touched.push_back(s.l);
        ++n_swapped;
    };
    for (const auto & s : fills) {
        apply(s);
    }
    for (const auto & s : swaps) {
        apply(s);
    }
    const int64_t a3 = adm_now();
    g_adm_copy += a3 - a2;
    // the victims are unmapped on the device before any copy starts
    if (!touched.empty()) {
        std::sort(touched.begin(), touched.end());
        touched.erase(std::unique(touched.begin(), touched.end()), touched.end());
        for (layer * l : touched) {
            stage_table(*l);
        }
        flush_tables();
    }
    if (!pending.empty()) {
        copier_submit(pending);
    }
    if (!opts.async_swaps) {
        publish_pending();
    }

    const int64_t a4 = adm_now();
    g_adm_up += a4 - a3;
    for (auto & entry : layers) {
        for (float & v : entry.second->score) {
            v *= opts.decay;
        }
    }
    g_adm_decay += adm_now() - a4;
}

// Every 16 steps: how long did the GPU wait for the CPU, per layer? A long wait means the CPU is the slower path and
// the GPU should take more; no wait means the GPU is the slower one. The share only changes between steps, when no
// kernel and no follower reads it.
// The time of a lane, in us, as fitted to the event log (balance.py): the first card, the CPU and the second card. For every
// case [the second card has experts][pool hits][misses] the number of misses for the PCIe path is the one that makes the
// longest lane shortest, with the CPU taking the rest. A tie goes to the smaller share.
void hpcache_decoder::fill_gtab() {
    const char * off = getenv("HPCACHE_GTAB");
    gtab_on = opts.pcie && db.gtab != nullptr && (off == nullptr || atoi(off) != 0);
    if (!gtab_on) {
        return;
    }
    double c3[3] = { 74.0, 9.0, 81.0 };    // first card: fixed, per pool hit, per PCIe expert
    double cc[2] = { 17.0, 83.0 };          // CPU: fixed, per expert
    double c2[2] = { 132.0, 22.0 };         // second card: fixed, per expert (when it has experts)
    if (const char * m = getenv("HPCACHE_LANE_MODEL")) {
        sscanf(m, "%lf,%lf,%lf,%lf,%lf,%lf,%lf", &c3[0], &c3[1], &c3[2], &cc[0], &cc[1], &c2[0], &c2[1]);
    }
    for (int two = 0; two < 2; ++two) {
        for (int hits = 0; hits <= 32; ++hits) {
            for (int miss = 0; miss <= 32; ++miss) {
                int best_g = 0;
                double best = 1e30;
                for (int g = 0; g <= std::min(miss, opts.pcie_max); ++g) {
                    const double t3 = c3[0] + c3[1] * hits + c3[2] * g;
                    const double tc = cc[0] + cc[1] * (miss - g);
                    const double t2 = two ? c2[0] + c2[1] * 2 : 0.0;   // two experts on the second card, about the mean
                    const double t = std::max(t3, std::max(tc, t2));
                    if (t < best) {
                        best = t;
                        best_g = g;
                    }
                }
                db.gtab[hpcache_gtab_index(two != 0, hits, miss)] = (int8_t) best_g;
            }
        }
    }
    log(GGML_LOG_LEVEL_INFO, "PCIe share per layer from a model of the lanes, " + std::to_string(opts.pcie_max) + " experts at most");
}

void hpcache_decoder::adapt_share() {
    const int64_t steps_done = n_decode.load();
    if (steps_done - adapt_last_decode < 16) {
        return;
    }
    const unsigned long long wait_ns = *db.wait_ns;
    const int64_t layers_served = n_serve.load() - adapt_last_serve;
    if (layers_served > 0) {
        const double us = double(wait_ns - adapt_last_wait_ns) / 1000.0 / double(layers_served);
        last_wait_us_per_layer = us;
        float next = share_now;
        if (us > 30.0) {
            next += 0.03f;
        } else if (us < 8.0) {
            next -= 0.03f;
        }
        next = std::min(0.9f, std::max(0.05f, next));
        if (next != share_now) {
            share_now = next;
            *db.frac = int(share_now * 256.0f + 0.5f);
        }
    }
    adapt_last_decode = steps_done;
    adapt_last_serve = n_serve.load();
    adapt_last_wait_ns = wait_ns;
}

void hpcache_decoder::end() {
    if (!armed.load(std::memory_order_acquire)) {
        return;
    }

    HPCACHE_LOG(ring_main, HPE_END_ENTER, (uint32_t) n_decode.load(), -2);
    // when the device is done, every ring of this graph has been served
    const int dev = device.load();
    if (dev >= 0) {
        hpcache_cuda_set_device(dev);
    }
    HPCACHE_TRACE("end: syncing the device");
    hpcache_cuda_sync_device();
    HPCACHE_TRACE("end: device synced");
    HPCACHE_LOG(ring_main, HPE_END_SYNCED, (uint32_t) n_decode.load(), -2);

    armed.store(false, std::memory_order_release);
    follower_sig.notify();
    follower_sig.wait_until([&] { return idle.load(); });
    if (db.two) {
        follower2_sig.notify();
        follower2_sig.wait_until([&] { return idle2.load(); });
    }

    HPCACHE_LOG(ring_main, HPE_END_FOLLOWERS_IDLE, (uint32_t) n_decode.load(), -2);
    if (cpu_pool) {
        cpu_pool->set_active(false);
    }
    HPCACHE_LOG(ring_main, HPE_END_POOL_OFF, (uint32_t) n_decode.load(), -2);

    const int64_t now_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    const int64_t start_us = decode_start_us.load();
    if (start_us > 0) {
        decode_us += now_us - start_us;
        n_decode++;
    }

    if (opts.pcie && opts.pcie_adapt && !gtab_on && n_pcie_experts > 0 && db.block != nullptr) {
        adapt_share();
    }
    // the second calibration, for the drift of the clocks; it takes close to 100 ms, so it runs once
#ifdef HPCACHE_EVENT_LOG
    if (steps_on && steps_calibrated && steps_host_1 == steps_host_0 && n_decode.load() >= 40) {
        steps_calibrate();
    }
#endif

    // HPCACHE_STATS_EVERY=N: one line per N decode steps with the window averages
    static const int stats_every = getenv("HPCACHE_STATS_EVERY") ? atoi(getenv("HPCACHE_STATS_EVERY")) : 0;
    if (stats_every > 0 && n_decode.load() % stats_every == 0 && n_decode.load() > 0) {
        static int64_t last_dec = 0, last_dus = 0, last_srv = 0, last_sus = 0, last_rt = 0, last_hit = 0, last_sw = 0, last_swus = 0;
        const int64_t dd = n_decode.load() - last_dec, ds = n_serve.load() - last_srv;
        const int64_t rt = n_routed.load() - last_rt;
        if (dd > 0 && rt > 0) {
            char buf[256];
            snprintf(buf, sizeof(buf),
                    "window of %lld steps (total %lld): hit %.1f%%, step %.1f ms, cpu experts %.1f ms (%.0f us/layer), other %.1f ms, swaps %.1f/step %.2f ms, gpu share %.2f (wait %.0f us/layer)",
                    (long long) dd, (long long) n_decode.load(), 100.0 * (n_hit.load() - last_hit) / rt,
                    (decode_us.load() - last_dus) / 1000.0 / dd, (serve_us.load() - last_sus) / 1000.0 / dd,
                    ds > 0 ? double(serve_us.load() - last_sus) / ds : 0.0,
                    ((decode_us.load() - last_dus) - (serve_us.load() - last_sus)) / 1000.0 / dd,
                    double(n_swapped - last_sw) / dd, double(swap_us - last_swus) / 1000.0 / dd,
                    double(share_now), last_wait_us_per_layer);
            log(GGML_LOG_LEVEL_INFO, buf);
        }
        last_dec = n_decode.load(); last_dus = decode_us.load(); last_srv = n_serve.load(); last_sus = serve_us.load();
        last_rt = n_routed.load(); last_hit = n_hit.load(); last_sw = n_swapped; last_swus = swap_us;
    }

    if (pool_used > 0 && opts.swap_every > 0 && ++steps % opts.swap_every == 0) {
        HPCACHE_LOG(ring_main, HPE_END_ADMIT_START, (uint32_t) n_decode.load(), -2);
        const auto t0 = std::chrono::steady_clock::now();
        admit();
        swap_us += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
    }
    HPCACHE_LOG(ring_main, HPE_END_EXIT, (uint32_t) n_decode.load(), -2);
}
