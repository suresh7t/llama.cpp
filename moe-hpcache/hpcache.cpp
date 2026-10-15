#include "hpcache.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

static int backend_ordinal(const std::string & name, const char * prefix) {
    if (name.compare(0, std::strlen(prefix), prefix) != 0) {
        return -1;
    }
    const std::string suffix = name.substr(std::strlen(prefix));
    if (suffix.empty()) {
        return -1;
    }
    try {
        return std::stoi(suffix);
    } catch (const std::exception &) {
        return -1;
    }
}

ggml_type common_moe_hpcache_cache_type_from_name(const std::string & name) {
    for (int type = 0; type < GGML_TYPE_COUNT; ++type) {
        const auto value = static_cast<ggml_type>(type);
        if (name == ggml_type_name(value)) {
            return value;
        }
    }
    throw std::runtime_error("unsupported HP-cache type: " + name);
}

common_moe_hpcache_runtime common_moe_hpcache_build_runtime(
        const common_moe_hpcache_plan & plan,
        const std::vector<ggml_backend_dev_t> & devices,
        const std::vector<int32_t> & cuda_overrides_mib,
        const std::vector<int32_t> & vulkan_overrides_mib) {
    common_moe_hpcache_runtime result;
    result.enabled = true;
    result.direct_io = true;
    result.cpu_kv_authoritative = plan.cpu_kv_authoritative;
    result.gpu_dense_weights_required = plan.gpu_dense_weights_required;
    result.cpu_expert_source_bytes = plan.expert_weight_bytes;
    result.cpu_ple_cache_budget_bytes = uint64_t(double(plan.ple_weight_bytes) * plan.ple_resident_fraction);
    result.expert_values = std::make_unique<common_moe_hpcache_expert_map>();
    result.ple_values = std::make_unique<common_moe_hpcache_ple_map>();

    for (ggml_backend_dev_t backend : devices) {
        const std::string name = ggml_backend_dev_name(backend);
        const auto it = std::find_if(plan.devices.begin(), plan.devices.end(), [&](const auto & entry) {
            return entry.name == name && entry.capacity_owner;
        });
        if (it == plan.devices.end()) {
            continue;
        }
        common_moe_hpcache_runtime_device device;
        device.backend = backend;
        device.name = name;
        device.type = it->type;
        device.expert_cache_budget_bytes = it->expert_cache_budget_bytes;
        device.cpu_authoritative = it->type == "cpu";

        const int32_t ordinal_cuda = backend_ordinal(name, "CUDA");
        const int32_t ordinal_vulkan = backend_ordinal(name, "Vulkan");
        if (ordinal_cuda >= 0 && size_t(ordinal_cuda) < cuda_overrides_mib.size()) {
            device.expert_cache_budget_bytes = uint64_t(cuda_overrides_mib[ordinal_cuda]) * 1024 * 1024;
        } else if (ordinal_vulkan >= 0 && size_t(ordinal_vulkan) < vulkan_overrides_mib.size()) {
            device.expert_cache_budget_bytes = uint64_t(vulkan_overrides_mib[ordinal_vulkan]) * 1024 * 1024;
        }
        result.devices.push_back(std::move(device));
    }
    if (std::none_of(result.devices.begin(), result.devices.end(), [](const auto & device) {
            return device.cpu_authoritative;
        })) {
        throw std::runtime_error("HP-cache plan has no authoritative CPU device");
    }
    for (const auto & device : result.devices) {
        if (device.cpu_authoritative) {
            result.expert_values->set_budget(device.expert_cache_budget_bytes);
            result.ple_values->set_budget(result.cpu_ple_cache_budget_bytes);
            break;
        }
    }
    return result;
}

static void require_plan_field(const char * key, bool valid) {
    if (!valid) {
        throw std::runtime_error(std::string("HP-cache plan field '") + key + "' is missing or invalid");
    }
}

common_moe_hpcache_plan common_moe_hpcache_parse_plan(const common_json & plan) {
    require_plan_field("object", plan.is_object());
    require_plan_field("schema_version", plan.contains("schema_version") && plan["schema_version"].is_number_integer());
    require_plan_field("model", plan.contains("model") && plan["model"].is_string());
    require_plan_field("plan_role", plan.contains("plan_role") && plan["plan_role"].is_string() &&
                       plan["plan_role"].get<std::string>() == "moe_hpcache_capacity");
    require_plan_field("context_tokens", plan.contains("context_tokens") && plan["context_tokens"].is_number_integer());
    require_plan_field("cache_type_k", plan.contains("cache_type_k") && plan["cache_type_k"].is_string());
    require_plan_field("cache_type_v", plan.contains("cache_type_v") && plan["cache_type_v"].is_string());
    require_plan_field("memory_pool_fraction", plan.contains("memory_pool_fraction") && plan["memory_pool_fraction"].is_number());
    require_plan_field("gpu_kv_cache_fraction", plan.contains("gpu_kv_cache_fraction") && plan["gpu_kv_cache_fraction"].is_number());
    require_plan_field("cpu_kv_cache_fraction", plan.contains("cpu_kv_cache_fraction") && plan["cpu_kv_cache_fraction"].is_number());
    require_plan_field("cpu_kv_authoritative", plan.contains("cpu_kv_authoritative") && plan["cpu_kv_authoritative"].is_boolean());
    require_plan_field("gpu_dense_weights_required", plan.contains("gpu_dense_weights_required") && plan["gpu_dense_weights_required"].is_boolean());
    require_plan_field("placement_policy", plan.contains("placement_policy") && plan["placement_policy"].is_string());
    require_plan_field("dense_weight_bytes", plan.contains("dense_weight_bytes") && plan["dense_weight_bytes"].is_number_integer());
    require_plan_field("expert_weight_bytes", plan.contains("expert_weight_bytes") && plan["expert_weight_bytes"].is_number_integer());
    require_plan_field("devices", plan.contains("devices") && plan["devices"].is_array() && !plan["devices"].empty());

    common_moe_hpcache_plan result;
    result.schema_version = plan["schema_version"].get<int32_t>();
    result.model = plan["model"].get<std::string>();
    result.role = plan["plan_role"].get<std::string>();
    result.architecture = plan.value("architecture", "");
    result.context_tokens = plan["context_tokens"].get<uint32_t>();
    result.cache_type_k = plan["cache_type_k"].get<std::string>();
    result.cache_type_v = plan["cache_type_v"].get<std::string>();
    result.memory_pool_fraction = plan["memory_pool_fraction"].get<double>();
    result.gpu_kv_cache_fraction = plan["gpu_kv_cache_fraction"].get<double>();
    result.cpu_kv_cache_fraction = plan["cpu_kv_cache_fraction"].get<double>();
    result.cpu_kv_authoritative = plan["cpu_kv_authoritative"].get<bool>();
    result.gpu_dense_weights_required = plan["gpu_dense_weights_required"].get<bool>();
    result.placement_policy = plan["placement_policy"].get<std::string>();
    result.dense_weight_bytes = plan["dense_weight_bytes"].get<uint64_t>();
    result.ple_weight_bytes = plan.value("ple_table_weight_bytes", uint64_t(0));
    result.expert_weight_bytes = plan["expert_weight_bytes"].get<uint64_t>();
    result.ple_resident_fraction = plan.value("ple_resident_fraction", 0.30);
    if (plan.contains("expert_split") && plan["expert_split"].is_object()) {
        const auto & split = plan["expert_split"];
        result.expert_split.present = true;
        result.expert_split.cpu_gb_per_s = split.value("cpu_gb_per_s", 0.0);
        result.expert_split.pcie_gb_per_s = split.value("pcie_gb_per_s", 0.0);
        result.expert_split.pcie_frac = split.value("pcie_frac", 0.0);
        result.expert_split.pcie_max = split.value("pcie_max", 0);
    }

    require_plan_field("schema_version", result.schema_version == 1);
    require_plan_field("context_tokens", result.context_tokens > 0);
    require_plan_field("memory_pool_fraction", result.memory_pool_fraction >= 0.5 && result.memory_pool_fraction <= 0.95);
    require_plan_field("gpu_kv_cache_fraction", result.gpu_kv_cache_fraction > 0.0 && result.gpu_kv_cache_fraction <= 1.0);
    require_plan_field("cpu_kv_cache_fraction", result.cpu_kv_cache_fraction == 1.00);
    require_plan_field("ple_resident_fraction", result.ple_resident_fraction >= 0.0 && result.ple_resident_fraction <= 1.0);

    result.devices.reserve(plan["devices"].size());
    for (size_t i = 0; i < plan["devices"].size(); ++i) {
        const auto & value = plan["devices"][i];
        require_plan_field("devices[]", value.is_object());
        require_plan_field("devices[].name", value.contains("name") && value["name"].is_string());
        require_plan_field("devices[].type", value.contains("type") && value["type"].is_string());
        require_plan_field("devices[].capacity_owner", value.contains("capacity_owner") && value["capacity_owner"].is_boolean());
        require_plan_field("devices[].expert_cache_budget_bytes", value.contains("expert_cache_budget_bytes") &&
                           value["expert_cache_budget_bytes"].is_number_integer());
        common_moe_hpcache_device device;
        device.name = value["name"].get<std::string>();
        device.type = value["type"].get<std::string>();
        device.device_id = value.value("device_id", "");
        device.capacity_owner = value["capacity_owner"].get<bool>();
        device.expert_cache_budget_bytes = value["expert_cache_budget_bytes"].get<uint64_t>();
        device.kv_cache_bytes = value.value("kv_cache_bytes", uint64_t(0));
        result.devices.push_back(std::move(device));
    }
    return result;
}
