#include "moe-provider.h"

#include <algorithm>

#include "common.h"
#include "log.h"

#include <cctype>
#include <cstring>
#include <filesystem>
#include <sstream>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#include <limits.h>
#include <unistd.h>
#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif
#endif

static bool valid_provider_name(const std::string & name) {
    if (name.empty()) {
        return false;
    }
    for (const unsigned char c : name) {
        if (!std::isalnum(c) && c != '-' && c != '_') {
            return false;
        }
    }
    return true;
}

static std::filesystem::path executable_directory() {
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
#elif defined(__APPLE__)
    uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);
    std::vector<char> buffer(size + 1);
    if (_NSGetExecutablePath(buffer.data(), &size) != 0) {
        return {};
    }
    return std::filesystem::path(buffer.data()).parent_path();
#else
    std::vector<char> buffer(PATH_MAX);
    const ssize_t length = readlink("/proc/self/exe", buffer.data(), buffer.size() - 1);
    if (length <= 0) {
        return {};
    }
    buffer[static_cast<size_t>(length)] = '\0';
    return std::filesystem::path(buffer.data()).parent_path();
#endif
}

static std::filesystem::path provider_path(const std::string & name) {
#if defined(_WIN32)
    constexpr const char * suffix = ".dll";
#elif defined(__APPLE__)
    constexpr const char * suffix = ".dylib";
#else
    constexpr const char * suffix = ".so";
#endif
    return executable_directory() / ("moe-" + name + suffix);
}

static std::string loader_error(const std::string & prefix) {
#if defined(_WIN32)
    std::ostringstream out;
    out << prefix << " (error " << GetLastError() << ")";
    return out.str();
#else
    const char * message = dlerror();
    return prefix + (message != nullptr ? ": " + std::string(message) : "");
#endif
}

void common_moe_provider::log(void *, enum ggml_log_level level, const char * message) {
    common_log_add(common_log_main(), level, "%s", message != nullptr ? message : "");
}

const char * common_moe_provider::get_param(void * ctx, const char * name) {
    if (ctx == nullptr || name == nullptr) {
        return nullptr;
    }
    const auto & values = static_cast<const common_moe_provider *>(ctx)->values;
    const auto it = values.find(name);
    return it != values.end() ? it->second.c_str() : nullptr;
}

// The devices this run uses: every device that is not a GPU, and the GPUs of --device (all GPUs without it).
// A GPU that was not selected is not queried either, which would create a context on it.
int32_t common_moe_provider::get_devices(void * ctx, llama_moe_device_info * out, int32_t cap) {
    const auto * self = static_cast<const common_moe_provider *>(ctx);
    const size_t count = ggml_backend_dev_count();
    int32_t n = 0;
    for (size_t i = 0; i < count && n < cap; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (self != nullptr && !self->selected_devices.empty() && ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU &&
                std::find(self->selected_devices.begin(), self->selected_devices.end(), dev) == self->selected_devices.end()) {
            continue;
        }
        llama_moe_device_info & info = out[n++];
        info.index = static_cast<int32_t>(i);
        info.dev   = dev;
        info.name  = ggml_backend_dev_name(dev);
        info.type  = ggml_backend_dev_type(dev);
        ggml_backend_dev_memory(dev, &info.mem_free, &info.mem_total);
    }
    return n;
}

common_moe_provider::~common_moe_provider() {
    unload();
}

void common_moe_provider::unload() {
    // free the provider before its library goes away
    if (provider != nullptr && provider->free != nullptr) {
        provider->free(provider->handle);
    }
    provider = nullptr;

    if (module != nullptr) {
#if defined(_WIN32)
        FreeLibrary(static_cast<HMODULE>(module));
#else
        dlclose(module);
#endif
        module = nullptr;
    }
}

bool common_moe_provider::load(const common_params & params, std::string & error) {
    unload();

    const std::string & name = params.moe_provider;
    if (!valid_provider_name(name)) {
        error = "invalid provider name: " + name;
        return false;
    }

    const std::filesystem::path path = provider_path(name);
    if (path.empty()) {
        error = "could not determine the executable directory";
        return false;
    }

    selected_devices.clear();
    for (ggml_backend_dev_t dev : params.devices) {
        if (dev != nullptr) {
            selected_devices.push_back(dev);
        }
    }

    values = params.moe_provider_params;
    values["model"]         = params.model.path;
    values["n_ctx"]         = std::to_string(params.n_ctx);
    values["ctk"]           = ggml_type_name(params.cache_type_k);
    values["ctv"]           = ggml_type_name(params.cache_type_v);
    values["n_gpu_layers"]  = std::to_string(params.n_gpu_layers);
    values["n_threads"]     = std::to_string(params.cpuparams.n_threads);
    values["n_ubatch"]      = std::to_string(params.n_ubatch);

    host = {};
    host.abi_version = LLAMA_MOE_ABI_VERSION;
    host.struct_size = sizeof(host);
    host.ctx         = this;
    host.log         = log;
    host.get_param   = get_param;
    host.get_devices = get_devices;

#if defined(_WIN32)
    HMODULE loaded = LoadLibraryW(path.wstring().c_str());
    if (loaded == nullptr) {
        error = loader_error("failed to load provider '" + path.string() + "'");
        return false;
    }
    module = loaded;
    auto init = reinterpret_cast<llama_moe_provider_init_fn>(GetProcAddress(loaded, "llama_moe_provider_init"));
#else
    module = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (module == nullptr) {
        error = loader_error("failed to load provider '" + path.string() + "'");
        return false;
    }
    auto init = reinterpret_cast<llama_moe_provider_init_fn>(dlsym(module, "llama_moe_provider_init"));
#endif

    if (init == nullptr) {
        error = "provider '" + path.string() + "' does not export llama_moe_provider_init";
        unload();
        return false;
    }

    const llama_moe_provider_api * api = init(&host, params.moe_provider_args.c_str());
    if (api == nullptr) {
        error = "provider initialization failed";
        unload();
        return false;
    }

    // abi_version is the first field, so it is safe to read even when the layout differs
    if (api->abi_version != LLAMA_MOE_ABI_VERSION || api->struct_size < sizeof(llama_moe_provider_api)) {
        error = "provider ABI mismatch: provider " + std::to_string(api->abi_version) +
                ", host " + std::to_string(LLAMA_MOE_ABI_VERSION);
        unload();
        return false;
    }
    provider = api;
    return true;
}
