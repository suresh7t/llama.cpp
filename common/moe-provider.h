#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "../include/llama.h"

struct llama_moe_tensor_info {
    const char * name;
    uint16_t file_index;
    uint64_t file_offset;
    size_t byte_size;
    const int64_t * ne;
    size_t n_dims;
};

enum llama_moe_tensor_load_policy {
    LLAMA_MOE_TENSOR_LOAD_DEFAULT,
    LLAMA_MOE_TENSOR_LOAD_IMMEDIATE,
    LLAMA_MOE_TENSOR_LOAD_LAZY,
};

struct llama_moe_tensor_load_decision {
    enum llama_moe_tensor_load_policy policy;
    void * source_handle;
};

enum class llama_moe_feature : uint8_t {
    supports_placement = 0,
    supports_scheduling,
    supports_direct_io,
    supports_paging,
};

struct llama_moe_feature_set {
    uint64_t value = 0;

    static constexpr uint64_t version_mask = 0xffull;
    static constexpr uint64_t feature_mask = 0x0000ffffffffffffull;

    uint8_t version() const {
        return uint8_t(value & version_mask);
    }

    bool has(llama_moe_feature feature) const {
        const uint8_t bit = static_cast<uint8_t>(feature);
        return (value & (1ull << (8 + bit))) != 0;
    }

    void enable(llama_moe_feature feature) {
        const uint8_t bit = static_cast<uint8_t>(feature);
        value |= 1ull << (8 + bit);
    }

    void disable(llama_moe_feature feature) {
        const uint8_t bit = static_cast<uint8_t>(feature);
        value &= ~(1ull << (8 + bit));
    }
};

struct llama_moe_dispatch_request {
    void * graph;
    void * layer;
    ggml_tensor * input;
    int layer_id;
};

struct common_params;

struct llama_moe_model_info {
    const void * hparams;
    const struct llama_model_params * model_params;
    const struct llama_context_params * context_params;
};

struct llama_moe_provider_host {
    virtual ~llama_moe_provider_host() = default;

    virtual uint32_t abi_version() const = 0;
    virtual void log(enum ggml_log_level level, const char * message) const = 0;
    virtual const char * get_param(const char * name) const = 0;
    virtual const llama_moe_model_info & model() const = 0;
};

struct llama_moe_provider_host_impl final : llama_moe_provider_host {
    llama_moe_provider_host_impl(
            const common_params & params,
            const struct llama_model_params & model_params,
            const struct llama_context_params & context_params);

    uint32_t abi_version() const override;
    void log(enum ggml_log_level level, const char * message) const override;
    const char * get_param(const char * name) const override;
    const llama_moe_model_info & model() const override;

private:
    const common_params & common;
    llama_moe_model_info model_info;
    mutable std::string value;
};

struct llama_moe_provider {
public:
    virtual ~llama_moe_provider() = default;

    virtual llama_moe_tensor_load_decision tensor_load(
            const llama_moe_tensor_info &) {
        return { LLAMA_MOE_TENSOR_LOAD_DEFAULT, nullptr };
    }

    virtual void tensor_loaded(
            const llama_moe_tensor_info &, void *) {}

    virtual void * place_tensor(
            const llama_moe_tensor_info &, int) {
        return nullptr;
    }

    virtual ggml_tensor * build_dispatch(
            const llama_moe_dispatch_request &) {
        return nullptr;
    }

    virtual void before_compute() {}
    virtual void after_compute() {}
};

typedef llama_moe_provider * (*llama_moe_provider_create)(
        const llama_moe_provider_host & host, llama_moe_feature_set & features, const char * args);

typedef void (*llama_moe_provider_destroy)(llama_moe_provider * provider);

struct common_moe_provider_module {
    common_moe_provider_module() = default;
    ~common_moe_provider_module();

    common_moe_provider_module(const common_moe_provider_module &) = delete;
    common_moe_provider_module & operator=(const common_moe_provider_module &) = delete;

    bool load(const std::string & name, const llama_moe_provider_host & host,
              llama_moe_feature_set & features, const std::string & args,
              std::string & error);
    void unload();

    llama_moe_provider * get() const { return provider; }

private:
    void * module = nullptr;
    llama_moe_provider * provider = nullptr;
    llama_moe_provider_destroy destroy = nullptr;
};

struct common_moe_provider_state {
    common_moe_provider_state();
    ~common_moe_provider_state();

    common_moe_provider_state(const common_moe_provider_state &) = delete;
    common_moe_provider_state & operator=(const common_moe_provider_state &) = delete;

    bool load(const std::string & name, const std::string & args,
              const llama_moe_provider_host & host,
              std::string & error);

    llama_moe_provider * get() const;
    const llama_moe_feature_set & features() const { return feature_set; }

private:
    common_moe_provider_module module;
    llama_moe_feature_set feature_set;
};
