#include "llama-moe-provider.h"
#include "hpcache.h"
#ifdef HPCACHE_CUDA
#include "hpcache-decode.h"
#endif

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include <vector>

namespace {

// Restrict the process to one logical processor per physical core, so that the CPU experts do
// not hop between cores or share a core with an SMT sibling. cpu_list overrides the choice.
std::string pin_process(const std::string & cpu_list, int n_cores_wanted) {
#if defined(_WIN32)
    DWORD_PTR mask = 0;
    int count = 0;

    if (!cpu_list.empty()) {
        std::stringstream ss(cpu_list);
        std::string item;
        while (std::getline(ss, item, ',')) {
            const int cpu = atoi(item.c_str());
            if (cpu >= 0 && cpu < 64) {
                mask |= DWORD_PTR(1) << cpu;
                ++count;
            }
        }
    } else {
        DWORD len = 0;
        GetLogicalProcessorInformation(nullptr, &len);
        std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> info(len / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
        if (len == 0 || !GetLogicalProcessorInformation(info.data(), &len)) {
            return "cannot read the processor topology";
        }
        for (const auto & entry : info) {
            if (entry.Relationship != RelationProcessorCore || entry.ProcessorMask == 0 || count >= n_cores_wanted) {
                continue;
            }
            // the first logical processor of the core
            mask |= entry.ProcessorMask & (~entry.ProcessorMask + 1);
            ++count;
        }
    }
    if (mask == 0) {
        return "no processors selected";
    }
    if (!SetProcessAffinityMask(GetCurrentProcess(), mask)) {
        return "SetProcessAffinityMask failed";
    }
    return "pinned to " + std::to_string(count) + " cores, mask 0x" + [&] {
        char buf[32];
        snprintf(buf, sizeof(buf), "%llx", (unsigned long long) mask);
        return std::string(buf);
    }();
#else
    (void) cpu_list;
    (void) n_cores_wanted;
    return "";
#endif
}

struct hpcache_provider {
    llama_moe_provider_api api = {};
    const llama_moe_host * host = nullptr;
    common_moe_hpcache_plan plan;

    std::atomic<int64_t> n_build_layer{0};
    std::atomic<int64_t> n_layers_hooked{0};
    std::atomic<int64_t> n_decode_begin{0};
    std::atomic<int64_t> n_decode_end{0};

#ifdef HPCACHE_CUDA
    std::unique_ptr<hpcache_decoder> decoder;
#endif

    void log(enum ggml_log_level level, const std::string & message) const {
        if (host->log != nullptr) {
            host->log(host->ctx, level, ("moe-hpcache: " + message + "\n").c_str());
        }
    }

    void load_plan(const char * args) {
        const char * value = host->get_param != nullptr ? host->get_param(host->ctx, "plan") : nullptr;
        std::string path = value != nullptr ? value : (args != nullptr ? args : "");
        if (path.rfind("plan=", 0) == 0) {
            path.erase(0, 5);
        }
        if (path.empty()) {
            throw std::runtime_error("missing plan, use --moe-provider-args plan=<plan.json>");
        }

        std::ifstream input(path, std::ios::binary);
        if (!input) {
            throw std::runtime_error("cannot open plan " + path);
        }
        std::ostringstream contents;
        contents << input.rdbuf();
        plan = common_moe_hpcache_parse_plan(common_json::parse(contents.str()));

        const char * model = host->get_param != nullptr ? host->get_param(host->ctx, "model") : nullptr;
        if (model != nullptr && plan.model != model) {
            throw std::runtime_error("plan model does not match the loaded model");
        }
    }
};

// the CPU holds every expert, dense weights keep the default placement
void provider_place(void * handle, const llama_moe_device_info * devices, int32_t n_devices,
        const llama_moe_tensor_desc * tensors, int32_t n_tensors, llama_moe_tensor_home * homes) {
    auto * provider = static_cast<hpcache_provider *>(handle);
    if (n_devices < 1 || devices[0].type != GGML_BACKEND_DEVICE_TYPE_CPU) {
        provider->log(GGML_LOG_LEVEL_WARN, "device 0 is not the CPU, using the default placement");
        return;
    }

    int32_t n_expert_tensors = 0;
    for (int32_t i = 0; i < n_tensors; ++i) {
        switch (tensors[i].kind) {
            case LLAMA_MOE_TENSOR_EXPERT_GATE:
            case LLAMA_MOE_TENSOR_EXPERT_UP:
            case LLAMA_MOE_TENSOR_EXPERT_DOWN:
            case LLAMA_MOE_TENSOR_EXPERT_GATE_UP:
                homes[i].kind   = LLAMA_MOE_HOME_DEVICE;
                homes[i].device = 0;
                ++n_expert_tensors;
                break;
            default:
                break;
        }
    }
    provider->log(GGML_LOG_LEVEL_INFO, "placing " + std::to_string(n_expert_tensors) + " expert tensors on the CPU");
}

// without a decoder, the core builds the stock path
ggml_tensor * provider_build_moe_layer(void * handle, const llama_moe_layer_request * req) {
    auto * provider = static_cast<hpcache_provider *>(handle);
    provider->n_build_layer++;
#ifdef HPCACHE_CUDA
    if (provider->decoder) {
        ggml_tensor * out = provider->decoder->build_layer(*req);
        if (out != nullptr) {
            provider->n_layers_hooked++;
        }
        return out;
    }
#else
    (void) req;
#endif
    return nullptr;
}

void provider_decode_begin(void * handle, const llama_moe_decode_info * info) {
    auto * provider = static_cast<hpcache_provider *>(handle);
    provider->n_decode_begin++;
#ifdef HPCACHE_CUDA
    if (provider->decoder) {
        provider->decoder->begin(*info);
    }
#else
    (void) info;
#endif
}

void provider_decode_end(void * handle) {
    auto * provider = static_cast<hpcache_provider *>(handle);
    provider->n_decode_end++;
#ifdef HPCACHE_CUDA
    if (provider->decoder) {
        provider->decoder->end();
    }
#endif
}

void provider_free(void * handle) {
    auto * provider = static_cast<hpcache_provider *>(handle);
    provider->log(GGML_LOG_LEVEL_INFO, "hook calls: build_moe_layer " + std::to_string(provider->n_build_layer.load()) +
            " (" + std::to_string(provider->n_layers_hooked.load()) + " built by the provider)" +
            ", decode_begin " + std::to_string(provider->n_decode_begin.load()) +
            ", decode_end " + std::to_string(provider->n_decode_end.load()));
    delete provider;
}

} // namespace

#if defined(_WIN32)
#define MOE_HPCACHE_EXPORT __declspec(dllexport)
#else
#define MOE_HPCACHE_EXPORT __attribute__((visibility("default")))
#endif

extern "C" MOE_HPCACHE_EXPORT const llama_moe_provider_api * llama_moe_provider_init(
        const llama_moe_host * host, const char * args) {
    if (host == nullptr || host->abi_version != LLAMA_MOE_ABI_VERSION) {
        return nullptr;
    }

    auto provider = std::make_unique<hpcache_provider>();
    provider->host = host;

    try {
        provider->load_plan(args);
    } catch (const std::exception & e) {
        provider->log(GGML_LOG_LEVEL_ERROR, e.what());
        return nullptr;
    }
    provider->log(GGML_LOG_LEVEL_INFO, "placement plan validated");

#ifdef HPCACHE_CUDA
    const char * doorbell = host->get_param != nullptr ? host->get_param(host->ctx, "doorbell") : nullptr;
    if (doorbell != nullptr && atoi(doorbell) == 0) {
        provider->log(GGML_LOG_LEVEL_INFO, "doorbell decode disabled, placement only");
    } else {
        const char * value = host->get_param != nullptr ? host->get_param(host->ctx, "n_threads") : nullptr;
        int n_threads = value != nullptr ? atoi(value) : 0;
        if (n_threads <= 0) {
            n_threads = 4;
        }
        auto param = [&](const char * name) {
            const char * v = host->get_param != nullptr ? host->get_param(host->ctx, name) : nullptr;
            return v != nullptr ? std::string(v) : std::string();
        };
        hpcache_pool_options pool;
        if (!param("pool").empty()) {
            pool.enabled = atoi(param("pool").c_str()) != 0;
        }
        if (!param("pool_mib").empty()) {
            pool.pool_bytes = size_t(atoll(param("pool_mib").c_str())) << 20;
        }
        if (!param("reserve_mib").empty()) {
            pool.reserve_bytes = size_t(atoll(param("reserve_mib").c_str())) << 20;
        }
        if (param("kernel") == "ggml") {
            pool.cpu_kernel = false;
        }
        if (!param("prefault").empty()) {
            pool.prefault = atoi(param("prefault").c_str()) != 0;
        }
        if (!param("swap_every").empty()) {
            pool.swap_every = atoi(param("swap_every").c_str());
        }
        if (!param("max_swaps").empty()) {
            pool.max_swaps = atoi(param("max_swaps").c_str());
        }
        pool.profile = param("profile");
        for (const auto & device : provider->plan.devices) {
            if (device.capacity_owner && device.type == "gpu") {
                pool.plan_bytes = device.expert_cache_budget_bytes;
                break;
            }
        }
        if (!param("moe_layers").empty()) {
            pool.n_layers = atoi(param("moe_layers").c_str());
        }
        if (param("pin").empty() || atoi(param("pin").c_str()) != 0) {
            // the CPU threads, the host thread of the graph and the driver
            const std::string result = pin_process(param("cpus"), n_threads + 2);
            if (!result.empty()) {
                provider->log(GGML_LOG_LEVEL_INFO, "CPU affinity: " + result);
            }
        }
        provider->decoder = std::make_unique<hpcache_decoder>(host, n_threads, pool);
        if (!provider->decoder->ok()) {
            provider->log(GGML_LOG_LEVEL_WARN, "no CPU backend, the stock MoE path is used");
            provider->decoder.reset();
        } else {
            provider->log(GGML_LOG_LEVEL_INFO, "doorbell decode enabled, " + std::to_string(n_threads) + " CPU threads");
        }
    }
#else
    provider->log(GGML_LOG_LEVEL_INFO, "built without CUDA, the stock MoE path is used");
#endif

    // hooks that stay NULL use the core defaults
    provider->api.abi_version = LLAMA_MOE_ABI_VERSION;
    provider->api.struct_size = sizeof(llama_moe_provider_api);
    provider->api.handle      = provider.get();
    provider->api.place          = provider_place;
    provider->api.build_moe_layer = provider_build_moe_layer;
    provider->api.decode_begin   = provider_decode_begin;
    provider->api.decode_end     = provider_decode_end;
    provider->api.free           = provider_free;

    return &provider.release()->api;
}
