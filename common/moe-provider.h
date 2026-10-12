#pragma once

#include "llama-moe-provider.h"

#include <map>
#include <string>

struct common_params;

// loads a provider library and owns it for the lifetime of the model
struct common_moe_provider {
    common_moe_provider() = default;
    ~common_moe_provider();

    common_moe_provider(const common_moe_provider &) = delete;
    common_moe_provider & operator=(const common_moe_provider &) = delete;

    bool load(const common_params & params, std::string & error);
    void unload();

    const llama_moe_provider_api * api() const { return provider; }

private:
    static void log(void * ctx, enum ggml_log_level level, const char * message);
    static const char * get_param(void * ctx, const char * name);
    static int32_t get_devices(void * ctx, llama_moe_device_info * out, int32_t cap);

    llama_moe_host host{};
    std::map<std::string, std::string> values;

    void * module = nullptr;
    const llama_moe_provider_api * provider = nullptr;
};
