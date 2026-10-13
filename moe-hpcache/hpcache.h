#pragma once

#include "json.h"
#include "hpcache-io.h"
#include "hpcache-lazy-map.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

constexpr uint64_t COMMON_MOE_HPCACHE_PAGE_BYTES = 8 * 1024;
constexpr uint32_t COMMON_MOE_HPCACHE_READ_BATCH_PAGES = 16;

struct common_moe_hpcache_source_ref {
    uint16_t file_index = 0;
    uint64_t file_offset = 0;
    uint64_t byte_size = 0;

    bool operator==(const common_moe_hpcache_source_ref & other) const {
        return file_index == other.file_index && file_offset == other.file_offset && byte_size == other.byte_size;
    }
};

struct common_moe_hpcache_expert_key {
    uint32_t layer = 0;
    uint32_t expert = 0;
    common_moe_hpcache_source_ref source;

    bool operator==(const common_moe_hpcache_expert_key & other) const {
        return layer == other.layer && expert == other.expert && source == other.source;
    }
};

struct common_moe_hpcache_ple_key {
    uint32_t layer = 0;
    uint32_t head = 0;
    uint32_t chunk = 0;
    common_moe_hpcache_source_ref source;

    bool operator==(const common_moe_hpcache_ple_key & other) const {
        return layer == other.layer && head == other.head && chunk == other.chunk && source == other.source;
    }
};

struct common_moe_hpcache_key_hash {
    size_t operator()(const common_moe_hpcache_source_ref & source) const {
        size_t hash = std::hash<uint16_t>{}(source.file_index);
        hash ^= std::hash<uint64_t>{}(source.file_offset) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        return hash ^ (std::hash<uint64_t>{}(source.byte_size) + 0x9e3779b9 + (hash << 6) + (hash >> 2));
    }
    size_t operator()(const common_moe_hpcache_expert_key & key) const {
        size_t hash = std::hash<uint32_t>{}(key.layer) ^ (std::hash<uint32_t>{}(key.expert) << 1);
        return hash ^ (operator()(key.source) + 0x9e3779b9 + (hash << 6) + (hash >> 2));
    }
    size_t operator()(const common_moe_hpcache_ple_key & key) const {
        size_t hash = std::hash<uint32_t>{}(key.layer) ^ (std::hash<uint32_t>{}(key.head) << 1) ^ (std::hash<uint32_t>{}(key.chunk) << 2);
        return hash ^ (operator()(key.source) + 0x9e3779b9 + (hash << 6) + (hash >> 2));
    }
};

using common_moe_hpcache_resident_value = std::shared_ptr<std::vector<uint8_t>>;
using common_moe_hpcache_expert_map = common_moe_hpcache_lazy_value_map<common_moe_hpcache_expert_key, common_moe_hpcache_resident_value, common_moe_hpcache_key_hash>;
using common_moe_hpcache_ple_map = common_moe_hpcache_lazy_value_map<common_moe_hpcache_ple_key, common_moe_hpcache_resident_value, common_moe_hpcache_key_hash>;

struct common_moe_hpcache_device {
    std::string name;
    std::string type;
    std::string device_id;
    bool        capacity_owner = false;
    uint64_t    expert_cache_budget_bytes = 0;
    uint64_t    kv_cache_bytes = 0;
};

// measured by the calibration tool: the CPU's and the PCIe link's rate for expert bytes, and the split derived from it
struct common_moe_hpcache_expert_split {
    bool   present = false;
    double cpu_gb_per_s = 0.0;
    double pcie_gb_per_s = 0.0;
    double pcie_frac = 0.0;
    int    pcie_max = 0;
};

struct common_moe_hpcache_plan {
    int32_t     schema_version = 0;
    std::string model;
    std::string role;
    std::string architecture;
    uint32_t    context_tokens = 0;
    std::string cache_type_k;
    std::string cache_type_v;
    double      memory_pool_fraction = 0.0;
    double      gpu_kv_cache_fraction = 0.0;
    double      cpu_kv_cache_fraction = 0.0;
    bool        cpu_kv_authoritative = false;
    bool        gpu_dense_weights_required = false;
    std::string placement_policy;
    uint64_t    dense_weight_bytes = 0;
    uint64_t    ple_weight_bytes = 0;
    uint64_t    expert_weight_bytes = 0;
    double      ple_resident_fraction = 0.30;
    common_moe_hpcache_expert_split expert_split;
    std::vector<common_moe_hpcache_device> devices;
};

struct common_moe_hpcache_runtime_device {
    ggml_backend_dev_t backend = nullptr;
    std::string         name;
    std::string         type;
    uint64_t            expert_cache_budget_bytes = 0;
    bool                cpu_authoritative = false;
};

struct common_moe_hpcache_runtime {
    bool enabled = false;
    bool direct_io = false;
    bool cpu_kv_authoritative = false;
    bool gpu_dense_weights_required = false;
    uint64_t source_page_bytes = COMMON_MOE_HPCACHE_PAGE_BYTES;
    uint32_t read_batch_pages = COMMON_MOE_HPCACHE_READ_BATCH_PAGES;
    uint64_t cpu_expert_source_bytes = 0;
    uint64_t cpu_ple_cache_budget_bytes = 0;
    std::vector<common_moe_hpcache_runtime_device> devices;
    std::unique_ptr<common_moe_hpcache_expert_map> expert_values;
    std::unique_ptr<common_moe_hpcache_ple_map> ple_values;
};

common_moe_hpcache_plan common_moe_hpcache_parse_plan(const common_json & plan);
ggml_type common_moe_hpcache_cache_type_from_name(const std::string & name);
common_moe_hpcache_runtime common_moe_hpcache_build_runtime(
        const common_moe_hpcache_plan & plan,
        const std::vector<ggml_backend_dev_t> & devices,
        const std::vector<int32_t> & cuda_overrides_mib,
        const std::vector<int32_t> & vulkan_overrides_mib);
