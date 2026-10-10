#include "moe-provider.h"

#include "common.h"
#include "log.h"

#include <cctype>
#include <cstring>
#include <exception>
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

llama_moe_provider_host_impl::llama_moe_provider_host_impl(
        const common_params & params,
        const struct llama_model_params & model_params,
        const struct llama_context_params & context_params)
    : common(params), model_info{nullptr, &model_params, &context_params} {
}

uint32_t llama_moe_provider_host_impl::abi_version() const {
    return 1;
}

void llama_moe_provider_host_impl::log(
        enum ggml_log_level level, const char * message) const {
    common_log_add(common_log_main(), level, "%s", message != nullptr ? message : "");
}

const char * llama_moe_provider_host_impl::get_param(const char * name) const {
    if (name == nullptr) {
        return nullptr;
    }

    const auto it = common.moe_provider_params.find(name);
    if (it != common.moe_provider_params.end()) {
        return it->second.c_str();
    }

    if (strcmp(name, "model") == 0) {
        return common.model.path.c_str();
    }
    if (strcmp(name, "n_ctx") == 0) {
        value = std::to_string(common.n_ctx);
    } else if (strcmp(name, "ctk") == 0) {
        value = ggml_type_name(common.cache_type_k);
    } else if (strcmp(name, "ctv") == 0) {
        value = ggml_type_name(common.cache_type_v);
    } else if (strcmp(name, "n_gpu_layers") == 0) {
        value = std::to_string(common.n_gpu_layers);
    } else {
        return nullptr;
    }
    return value.c_str();
}

const llama_moe_model_info & llama_moe_provider_host_impl::model() const {
    return model_info;
}

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
    const DWORD code = GetLastError();
    std::ostringstream out;
    out << prefix << " (error " << code << ")";
    return out.str();
#else
    const char * message = dlerror();
    return prefix + (message != nullptr ? ": " + std::string(message) : "");
#endif
}

common_moe_provider_module::~common_moe_provider_module() {
    unload();
}

void common_moe_provider_module::unload() {
    if (provider != nullptr && destroy != nullptr) {
        destroy(provider);
    }
    provider = nullptr;
    destroy = nullptr;

    if (module == nullptr) {
        return;
    }
#if defined(_WIN32)
    FreeLibrary(static_cast<HMODULE>(module));
#else
    dlclose(module);
#endif
    module = nullptr;
}

bool common_moe_provider_module::load(
        const std::string & name, const llama_moe_provider_host & host,
        llama_moe_feature_set & features, const std::string & args,
        std::string & error) {
    unload();
    if (!valid_provider_name(name)) {
        error = "invalid provider name: " + name;
        return false;
    }

    const std::filesystem::path path = provider_path(name);
    if (path.empty()) {
        error = "could not determine the executable directory";
        return false;
    }

#if defined(_WIN32)
    HMODULE loaded = LoadLibraryW(path.wstring().c_str());
    if (loaded == nullptr) {
        error = loader_error("failed to load provider '" + path.string() + "'");
        return false;
    }
    module = loaded;
    auto create = reinterpret_cast<llama_moe_provider_create>(GetProcAddress(loaded, "llama_moe_provider_create"));
    destroy = reinterpret_cast<llama_moe_provider_destroy>(GetProcAddress(loaded, "llama_moe_provider_destroy"));
#else
    module = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (module == nullptr) {
        error = loader_error("failed to load provider '" + path.string() + "'");
        return false;
    }
    auto create = reinterpret_cast<llama_moe_provider_create>(dlsym(module, "llama_moe_provider_create"));
    destroy = reinterpret_cast<llama_moe_provider_destroy>(dlsym(module, "llama_moe_provider_destroy"));
#endif

    if (create == nullptr || destroy == nullptr) {
        error = "provider '" + path.string() + "' does not export the required factory symbols";
        unload();
        return false;
    }

    try {
        provider = create(host, features, args.c_str());
    } catch (const std::exception & exception) {
        error = "provider initialization failed: " + std::string(exception.what());
        unload();
        return false;
    } catch (...) {
        error = "provider initialization failed with an unknown exception";
        unload();
        return false;
    }

    if (provider == nullptr) {
        error = "provider factory returned null";
        unload();
        return false;
    }
    return true;
}

common_moe_provider_state::common_moe_provider_state() = default;

common_moe_provider_state::~common_moe_provider_state() = default;

bool common_moe_provider_state::load(
        const std::string & name, const std::string & args,
        const llama_moe_provider_host & host,
        std::string & error) {
    return module.load(name, host, feature_set, args, error);
}

llama_moe_provider * common_moe_provider_state::get() const {
    return module.get();
}
