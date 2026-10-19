#include "llama-moe-provider.h"
#include "hpcache.h"
#ifdef HPCACHE_CUDA
#include "hpcache-decode.h"
#endif

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
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

// The directory of the host executable, where the provider library and the calibrate tool are installed.
std::filesystem::path executable_directory() {
#if defined(_WIN32)
    std::vector<wchar_t> buffer(512);
    for (;;) {
        const DWORD length = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
        if (length == 0) {
            return {};
        }
        if (length < buffer.size() - 1) {
            return std::filesystem::path(buffer.data(), buffer.data() + length).parent_path();
        }
        buffer.resize(buffer.size() * 2);
    }
#else
    std::vector<char> buffer(4096);
    const ssize_t length = readlink("/proc/self/exe", buffer.data(), buffer.size() - 1);
    if (length <= 0) {
        return {};
    }
    buffer[static_cast<size_t>(length)] = '\0';
    return std::filesystem::path(buffer.data()).parent_path();
#endif
}

// Run a program with the console of this process and wait for it; its exit code, or -1 when it could not start.
int run_process(const std::vector<std::string> & argv) {
#if defined(_WIN32)
    std::wstring command;
    for (const std::string & arg : argv) {
        std::wstring wide = std::filesystem::path(arg).wstring();
        command += command.empty() ? L"" : L" ";
        command += L"\"";
        for (wchar_t c : wide) {
            if (c == L'"') {
                command += L"\\";
            }
            command += c;
        }
        command += L"\"";
    }
    STARTUPINFOW si = {};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi = {};
    if (!CreateProcessW(nullptr, command.data(), nullptr, nullptr, TRUE, 0, nullptr, nullptr, &si, &pi)) {
        return -1;
    }
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 1;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    return int(code);
#else
    std::vector<char *> args;
    for (const std::string & arg : argv) {
        args.push_back(const_cast<char *>(arg.c_str()));
    }
    args.push_back(nullptr);
    const pid_t pid = fork();
    if (pid < 0) {
        return -1;
    }
    if (pid == 0) {
        execv(args[0], args.data());
        _exit(127);
    }
    int status = 0;
    if (waitpid(pid, &status, 0) < 0) {
        return -1;
    }
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
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

    std::string param(const char * name) const {
        const char * value = host->get_param != nullptr ? host->get_param(host->ctx, name) : nullptr;
        return value != nullptr ? value : "";
    }

    static bool read_plan_file(const std::string & path, common_moe_hpcache_plan & out) {
        std::ifstream input(path, std::ios::binary);
        if (!input) {
            return false;
        }
        std::ostringstream contents;
        contents << input.rdbuf();
        out = common_moe_hpcache_parse_plan(common_json::parse(contents.str()));
        return true;
    }

    // Why the plan generated for the default file no longer fits this run, empty when it does.
    std::string stale_reason(const common_moe_hpcache_plan & found) const {
        const std::string model = param("model");
        if (!model.empty() && found.model != model) {
            return "it was made for another model";
        }
        const uint32_t n_ctx = uint32_t(atoll(param("n_ctx").c_str()));
        if (n_ctx > 0 && found.context_tokens != n_ctx) {
            return "it was made for context " + std::to_string(found.context_tokens) + ", this run uses " + std::to_string(n_ctx);
        }
        const std::string ctk = param("ctk");
        const std::string ctv = param("ctv");
        if ((!ctk.empty() && found.cache_type_k != ctk) || (!ctv.empty() && found.cache_type_v != ctv)) {
            return "it was made for KV types " + found.cache_type_k + "/" + found.cache_type_v + ", this run uses " + ctk + "/" + ctv;
        }
        return "";
    }

    // Run the calibrate tool that sits next to the server and let it write the plan file.
    void run_calibration(const std::string & path) {
        const std::string model = param("model");
        if (model.empty()) {
            throw std::runtime_error("cannot calibrate, the model path is not known");
        }
        const std::string ctk = param("ctk");
        const std::string ctv = param("ctv");
        for (const std::string & type : {ctk, ctv}) {
            if (!type.empty() && type != "f16" && type != "q5_0") {
                throw std::runtime_error("automatic calibration supports KV types f16 and q5_0 only (this run uses " + type +
                        "), make a plan with llama-hpcache-calibrate and pass plan=<file>");
            }
        }

        std::vector<std::string> argv;
#if defined(_WIN32)
        argv.push_back((executable_directory() / "llama-hpcache-calibrate.exe").string());
#else
        argv.push_back((executable_directory() / "llama-hpcache-calibrate").string());
#endif
        std::error_code ec;
        if (!std::filesystem::exists(argv[0], ec)) {
            throw std::runtime_error("cannot calibrate, " + argv[0] + " not found, build the llama-hpcache-calibrate target");
        }
        argv.insert(argv.end(), {"-m", model, "-o", path});
        const std::string n_ctx = param("n_ctx");
        if (atoll(n_ctx.c_str()) > 0) {
            argv.insert(argv.end(), {"-ctx", n_ctx});
        }
        if (!ctk.empty()) {
            argv.insert(argv.end(), {"-ctk", ctk});
        }
        if (!ctv.empty()) {
            argv.insert(argv.end(), {"-ctv", ctv});
        }
        if (host->get_devices != nullptr) {
            llama_moe_device_info infos[16] = {};
            const int32_t n = host->get_devices(host->ctx, infos, 16);
            std::string names;
            for (int32_t i = 0; i < n; ++i) {
                if (infos[i].type != GGML_BACKEND_DEVICE_TYPE_CPU && infos[i].name != nullptr) {
                    names += (names.empty() ? "" : ",") + std::string(infos[i].name);
                }
            }
            if (!names.empty()) {
                argv.insert(argv.end(), {"-dev", names});
            }
        }

        log(GGML_LOG_LEVEL_INFO, "no usable plan, running the calibration (this takes a few minutes the first time)");
        const int code = run_process(argv);
        if (code != 0) {
            throw std::runtime_error("calibration failed with exit code " + std::to_string(code));
        }
        log(GGML_LOG_LEVEL_INFO, "calibration written to " + path);
    }

    void load_plan(const char * args) {
        const char * value = host->get_param != nullptr ? host->get_param(host->ctx, "plan") : nullptr;
        std::string path = value != nullptr ? value : (args != nullptr ? args : "");
        if (path.rfind("plan=", 0) == 0) {
            path.erase(0, 5);
        }
        // no plan given: use the calibration file in the working directory, and make it when it is missing or stale
        const bool explicit_plan = !path.empty();
        if (!explicit_plan) {
            path = "hpcache-calibration.json";
        }

        std::string reason;
        common_moe_hpcache_plan found;
        try {
            if (!read_plan_file(path, found)) {
                reason = path + " does not exist";
            } else if (!explicit_plan) {
                const std::string stale = stale_reason(found);
                if (!stale.empty()) {
                    reason = path + " is out of date: " + stale;
                }
            }
        } catch (const std::exception & e) {
            if (explicit_plan) {
                throw;
            }
            reason = path + " cannot be used: " + e.what();
        }
        if (!reason.empty()) {
            log(GGML_LOG_LEVEL_INFO, reason);
            run_calibration(path);
            if (!read_plan_file(path, found)) {
                throw std::runtime_error("cannot open plan " + path);
            }
        }
        plan = std::move(found);

        const std::string model = param("model");
        if (!model.empty() && plan.model != model) {
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
        } else if (!param("n_ubatch").empty() && atoll(param("n_ubatch").c_str()) > 0) {
            // the compute buffers are sized by the micro-batch: measured 1061 MiB at 512 and under 550 MiB at 128
            const int64_t ub = atoll(param("n_ubatch").c_str());
            pool.reserve_bytes = size_t(450 + ub * 8 / 5) << 20;
        }
        if (!param("grow").empty()) {
            pool.grow = atoi(param("grow").c_str()) != 0;
        }
        if (!param("grow_margin_mib").empty()) {
            pool.grow_margin_bytes = size_t(atoll(param("grow_margin_mib").c_str())) << 20;
        }
        // gpu2=1: the first other CUDA card, gpu2=CUDA1: that device; gpu2_mib caps its pool
        if (!param("gpu2").empty() && param("gpu2") != "0") {
            pool.gpu2 = true;
            if (param("gpu2") != "1") {
                pool.gpu2_name = param("gpu2");
            }
        }
        if (!param("gpu2_mib").empty()) {
            pool.gpu2_pool_bytes = size_t(atoll(param("gpu2_mib").c_str())) << 20;
        }
        if (!param("gpu2_reserve_mib").empty()) {
            pool.gpu2_reserve_bytes = size_t(atoll(param("gpu2_reserve_mib").c_str())) << 20;
        }
        if (!param("gpu2_keepalive").empty()) {
            pool.gpu2_keepalive = atoi(param("gpu2_keepalive").c_str()) != 0;
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
        if (!param("async_swaps").empty()) {
            pool.async_swaps = atoi(param("async_swaps").c_str()) != 0;
        }
        if (!param("arena").empty()) {
            pool.arena = atoi(param("arena").c_str()) != 0;
        }
        if (provider->plan.expert_split.present) {
            const auto & split = provider->plan.expert_split;
            if (split.pcie_frac > 0.0) {
                pool.pcie_frac = float(split.pcie_frac);
            }
            if (split.pcie_max > 0) {
                pool.pcie_max = split.pcie_max;
            }
            provider->log(GGML_LOG_LEVEL_INFO, "expert split from the calibration: CPU " +
                    std::to_string(split.cpu_gb_per_s).substr(0, 5) + " GB/s, PCIe " +
                    std::to_string(split.pcie_gb_per_s).substr(0, 5) + " GB/s, GPU share " +
                    std::to_string(split.pcie_frac).substr(0, 4) + ", up to " + std::to_string(split.pcie_max) + " experts");
        }
        if (!param("pcie_adapt").empty()) {
            pool.pcie_adapt = atoi(param("pcie_adapt").c_str()) != 0;
        }
        if (!param("pcie").empty()) {
            pool.pcie = atoi(param("pcie").c_str()) != 0;
        }
        if (!param("pcie_max").empty()) {
            pool.pcie_max = atoi(param("pcie_max").c_str());
        } else if (pool.gpu2) {
            // with a second card the CPU has room for what the first card streams: 2 experts per layer decode 3.5% faster than 3
            pool.pcie_max = 2;
        }
        if (!param("pcie_frac").empty()) {
            pool.pcie_frac = float(atof(param("pcie_frac").c_str()));
        }
        if (!param("pin_mib").empty()) {
            pool.pin_bytes = size_t(atoll(param("pin_mib").c_str())) << 20;
        }
        if (!param("arena_mib").empty()) {
            pool.arena_bytes = size_t(atoll(param("arena_mib").c_str())) << 20;
        }
        if (!param("spare_slot").empty()) {
            pool.spare_slot = atoi(param("spare_slot").c_str()) != 0;
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
            // the CPU threads, the host thread of the graph and the driver, and the worker of the second card
            const std::string result = pin_process(param("cpus"), n_threads + 2 + (pool.gpu2 ? 1 : 0));
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
