#if defined(_WIN32)
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

#ifdef HPCACHE_CUDA
#include <cuda_runtime.h>
#endif

#include "fit.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "gguf.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

struct expert_sample {
    std::string          architecture;
    std::string          name;
    ggml_type            type           = GGML_TYPE_F32;
    int64_t              ne0            = 0;
    int64_t              ne1            = 0;
    int64_t              n_expert       = 0;
    int32_t              n_used         = 0;
    int64_t              n_tensors      = 0;
    uint32_t             context_tokens = 0;
    size_t               group_bytes    = 0;
    size_t               expert_bytes   = 0;
    size_t               dense_bytes    = 0;
    size_t               ple_bytes      = 0;
    size_t               output_bytes   = 0;
    size_t               token_bytes    = 0;
    bool                 token_on_cpu   = true;
    double               gpu_kv_fraction = 1.0;
    int64_t              expert_slots   = 0;
    std::vector<size_t>  dense_tensor_sizes;
    std::vector<size_t>  dense_layer_sizes;
    std::vector<size_t>  expert_slot_sizes;
    std::vector<uint8_t> weight;
};

struct device_result {
    ggml_backend_dev_t      dev = nullptr;
    ggml_backend_ptr        backend;
    ggml_context_ptr        ctx;
    ggml_backend_buffer_ptr buffer;
    ggml_cgraph *           graph  = nullptr;
    ggml_tensor *           weight = nullptr;
    ggml_tensor *           input  = nullptr;
    ggml_tensor *           output = nullptr;
    std::vector<float>      input_data;
    std::vector<float>      output_data;
    size_t                  memory_free   = 0;
    size_t                  memory_total  = 0;
    size_t                  memory_pool   = 0;
    size_t                  memory_available_pool = 0;
    size_t                  weight_budget = 0;
    size_t                  kv_cache_bytes = 0;
    size_t                  expert_cache_budget = 0;
    size_t                  capacity      = 0;
    size_t                  context_bytes = 0;
    size_t                  compute_bytes = 0;
    double                  ms_transfer   = 0.0;
    double                  ms_per_expert = 0.0;
    int32_t                 assigned      = 0;
    std::string             device_id;
    bool                    capacity_owner = true;
    std::string             features;
    std::string             error;
};

static std::string json_escape(const std::string & value) {
    std::string out;
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') {
            out += '\\';
            out += char(c);
        } else if (c < 0x20) {
            char escaped[7];
            snprintf(escaped, sizeof(escaped), "\\u%04x", unsigned(c));
            out += escaped;
        } else {
            out += char(c);
        }
    }
    return out;
}

static bool is_expert_weight(const std::string & name) {
    return name.find(".ffn_") != std::string::npos && name.find("_exps.weight") != std::string::npos;
}

static size_t percentile(std::vector<size_t> values, size_t percent) {
    if (values.empty()) {
        return 0;
    }
    std::sort(values.begin(), values.end());
    return values[((values.size() - 1) * percent + 99) / 100];
}

static bool parse_context_tokens(const char * value, uint32_t & result) {
    std::string text(value);
    size_t      end = 0;
    while (end < text.size() && std::isdigit(static_cast<unsigned char>(text[end]))) {
        ++end;
    }
    if (end == 0) {
        return false;
    }
    uint64_t multiplier = 1;
    std::string suffix = text.substr(end);
    for (char & c : suffix) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    if (suffix == "k" || suffix == "ki" || suffix == "kib") {
        multiplier = 1024;
    } else if (suffix == "m" || suffix == "mi" || suffix == "mib") {
        multiplier = 1024 * 1024;
    } else if (!suffix.empty()) {
        return false;
    }
    try {
        const uint64_t base = std::stoull(text.substr(0, end));
        if (base > UINT32_MAX / multiplier) {
            return false;
        }
        const uint64_t tokens = base * multiplier;
        if (tokens == 0 || tokens > UINT32_MAX) {
            return false;
        }
        result = static_cast<uint32_t>(tokens);
        return true;
    } catch (const std::exception &) {
        return false;
    }
}

static const char * device_type_name(enum ggml_backend_dev_type type) {
    switch (type) {
        case GGML_BACKEND_DEVICE_TYPE_CPU:
            return "cpu";
        case GGML_BACKEND_DEVICE_TYPE_GPU:
            return "gpu";
        case GGML_BACKEND_DEVICE_TYPE_IGPU:
            return "igpu";
        case GGML_BACKEND_DEVICE_TYPE_ACCEL:
            return "accel";
        case GGML_BACKEND_DEVICE_TYPE_META:
            return "meta";
    }
    return "unknown";
}

static int32_t get_expert_used(const gguf_context * gguf, const std::string & arch, size_t layer) {
    const std::string key = arch + ".expert_used_count";
    const int64_t     id  = gguf_find_key(gguf, key.c_str());
    if (id < 0) {
        return 0;
    }
    if (gguf_get_kv_type(gguf, id) == GGUF_TYPE_UINT32) {
        const uint32_t value = gguf_get_val_u32(gguf, id);
        return value <= INT32_MAX ? int32_t(value) : 0;
    }
    if (gguf_get_kv_type(gguf, id) == GGUF_TYPE_ARRAY && gguf_get_arr_type(gguf, id) == GGUF_TYPE_UINT32 &&
        gguf_get_arr_n(gguf, id) > layer) {
        const uint32_t value = static_cast<const uint32_t *>(gguf_get_arr_data(gguf, id))[layer];
        return value <= INT32_MAX ? int32_t(value) : 0;
    }
    return 0;
}

static bool load_sample(const char * path, expert_sample & sample, std::string & error) {
    std::vector<gguf_context_ptr> shards;
    std::vector<std::string>      paths;
    shards.emplace_back(gguf_init_from_file(path, { true, nullptr }));
    if (!shards.front()) {
        error = "could not read GGUF metadata";
        return false;
    }
    const gguf_context * first    = shards.front().get();
    const int64_t        split_id = gguf_find_key(first, "split.count");
    const uint16_t       n_split =
        split_id >= 0 && gguf_get_kv_type(first, split_id) == GGUF_TYPE_UINT16 ? gguf_get_val_u16(first, split_id) : 1;
    if (n_split == 0) {
        error = "invalid GGUF split count";
        return false;
    }
    paths.emplace_back(path);
    if (n_split > 1) {
        const int64_t no_id = gguf_find_key(first, "split.no");
        if (no_id < 0 || gguf_get_kv_type(first, no_id) != GGUF_TYPE_UINT16 || gguf_get_val_u16(first, no_id) != 0) {
            error = "-m must name the first GGUF split";
            return false;
        }
        const std::string name(path);
        if (name.size() < 20) {
            error = "invalid GGUF split filename";
            return false;
        }
        const std::string suffix = name.substr(name.size() - 20);
        const std::string digits = suffix.substr(1, 5) + suffix.substr(10, 5);
        if (suffix[0] != '-' || suffix.substr(6, 4) != "-of-" || suffix.substr(15) != ".gguf" ||
            !std::all_of(digits.begin(), digits.end(), [](unsigned char c) { return std::isdigit(c); }) ||
            std::stoul(suffix.substr(1, 5)) != 1 || std::stoul(suffix.substr(10, 5)) != n_split) {
            error = "invalid GGUF split filename";
            return false;
        }
        const std::string prefix = name.substr(0, name.size() - 20);
        for (uint16_t i = 1; i < n_split; ++i) {
            char number[6];
            snprintf(number, sizeof(number), "%05u", unsigned(i + 1));
            paths.push_back(prefix + "-" + number + "-of-" + suffix.substr(10, 5) + ".gguf");
            shards.emplace_back(gguf_init_from_file(paths.back().c_str(), { true, nullptr }));
            if (!shards.back()) {
                error = "could not read GGUF split metadata: " + paths.back();
                return false;
            }
            const int64_t shard_no = gguf_find_key(shards.back().get(), "split.no");
            if (shard_no < 0 || gguf_get_kv_type(shards.back().get(), shard_no) != GGUF_TYPE_UINT16 ||
                gguf_get_val_u16(shards.back().get(), shard_no) != i) {
                error = "invalid GGUF split index: " + paths.back();
                return false;
            }
        }
    }
    const int64_t arch_id = gguf_find_key(first, "general.architecture");
    if (arch_id < 0 || gguf_get_kv_type(first, arch_id) != GGUF_TYPE_STRING) {
        error = "model architecture is missing";
        return false;
    }
    sample.architecture = gguf_get_val_str(first, arch_id);
    struct expert_layer_size {
        size_t  bytes = 0;
        int64_t count = 0;
    };
    std::map<unsigned, expert_layer_size> expert_layers;
    std::map<unsigned, size_t> dense_layers;
    for (const auto & shard : shards) {
        sample.n_tensors += gguf_get_n_tensors(shard.get());
        for (int64_t i = 0; i < gguf_get_n_tensors(shard.get()); ++i) {
            const std::string name = gguf_get_tensor_name(shard.get(), i);
            const size_t bytes = gguf_get_tensor_size(shard.get(), i);
            const bool expert = is_expert_weight(name);
            const bool ple    = name == "per_layer_token_embd.weight";
            const bool token  = name == "token_embd.weight";
            size_t & total = expert ? sample.expert_bytes : ple ? sample.ple_bytes : token ? sample.token_bytes : sample.dense_bytes;
            if (bytes > SIZE_MAX - total) {
                error = "model tensor size total overflows";
                return false;
            }
            total += bytes;
            if (ple || token) {
                continue;
            }
            unsigned layer = 0;
            if (expert) {
                const int64_t count = gguf_get_tensor_ne(shard.get(), i)[2];
                if (sscanf(name.c_str(), "blk.%u.", &layer) != 1 || count <= 0 || bytes % size_t(count) != 0) {
                    error = "invalid expert tensor size or layer";
                    return false;
                }
                auto & entry = expert_layers[layer];
                if ((entry.count != 0 && entry.count != count) || bytes / size_t(count) > SIZE_MAX - entry.bytes) {
                    error = "inconsistent expert tensors in layer";
                    return false;
                }
                entry.count = count;
                entry.bytes += bytes / size_t(count);
            } else {
                sample.dense_tensor_sizes.push_back(bytes);
                if (name == "output.weight") {
                    sample.output_bytes = bytes;
                }
                if (sscanf(name.c_str(), "blk.%u.", &layer) == 1) {
                    if (bytes > SIZE_MAX - dense_layers[layer]) {
                        error = "dense layer size overflows";
                        return false;
                    }
                    dense_layers[layer] += bytes;
                }
            }
        }
    }
    // input embedding lookups run on the CPU, unless the table is also the output head
    sample.token_on_cpu = sample.output_bytes != 0;
    if (!sample.token_on_cpu) {
        sample.dense_bytes += sample.token_bytes;
        sample.dense_tensor_sizes.push_back(sample.token_bytes);
    }
    for (const auto & entry : expert_layers) {
        if (entry.second.count > INT64_MAX - sample.expert_slots) {
            error = "expert count overflows";
            return false;
        }
        sample.expert_slots += entry.second.count;
        sample.expert_slot_sizes.push_back(entry.second.bytes);
    }
    for (const auto & entry : dense_layers) {
        sample.dense_layer_sizes.push_back(entry.second);
    }
    if (sample.expert_slots == 0 || sample.ple_bytes > SIZE_MAX - sample.dense_bytes ||
        sample.expert_bytes > SIZE_MAX - sample.dense_bytes - sample.ple_bytes) {
        error = "model weight total overflows or has no experts";
        return false;
    }
    if (n_split > 1) {
        const int64_t count_id = gguf_find_key(first, "split.tensors.count");
        if (count_id < 0 || gguf_get_kv_type(first, count_id) != GGUF_TYPE_INT32 ||
            gguf_get_val_i32(first, count_id) != sample.n_tensors) {
            error = "GGUF split tensor count does not match the shards";
            return false;
        }
    }
    int64_t chosen       = -1;
    size_t  chosen_shard = 0;
    for (size_t s = 0; s < shards.size() && chosen < 0; ++s) {
        const gguf_context * shard = shards[s].get();
        for (int64_t i = 0; i < gguf_get_n_tensors(shard); ++i) {
            const std::string name = gguf_get_tensor_name(shard, i);
            const int64_t *   ne   = gguf_get_tensor_ne(shard, i);
            if (is_expert_weight(name) && ne[0] > 0 && ne[0] <= 65536 && ne[1] > 0 && ne[1] <= 65536 && ne[2] > 1 &&
                ne[2] <= 4096 && ne[3] == 1) {
                chosen          = i;
                chosen_shard    = s;
                sample.name     = name;
                sample.type     = gguf_get_tensor_type(shard, i);
                sample.ne0      = ne[0];
                sample.ne1      = ne[1];
                sample.n_expert = ne[2];
                break;
            }
        }
    }
    if (chosen < 0) {
        error = "no usable 3D expert weight tensor was found";
        return false;
    }
    unsigned layer_index = 0;
    if (sscanf(sample.name.c_str(), "blk.%u.", &layer_index) != 1) {
        error = "could not identify the expert layer";
        return false;
    }
    sample.n_used = get_expert_used(first, sample.architecture, layer_index);
    if (sample.n_used <= 0 || sample.n_used > sample.n_expert) {
        error = "model has no supported expert_used_count for the selected layer";
        return false;
    }

    const std::string layer = sample.name.substr(0, sample.name.find(".ffn_"));
    for (const auto & shard : shards) {
        for (int64_t i = 0; i < gguf_get_n_tensors(shard.get()); ++i) {
            const std::string name = gguf_get_tensor_name(shard.get(), i);
            if (name.compare(0, layer.size() + 1, layer + ".") != 0 || !is_expert_weight(name)) {
                continue;
            }
            const int64_t * ne    = gguf_get_tensor_ne(shard.get(), i);
            const size_t    bytes = gguf_get_tensor_size(shard.get(), i);
            if (ne[2] == sample.n_expert && bytes % size_t(ne[2]) == 0 &&
                sample.group_bytes <= SIZE_MAX - bytes / size_t(ne[2])) {
                sample.group_bytes += bytes / size_t(ne[2]);
            }
        }
    }

    const gguf_context * selected    = shards[chosen_shard].get();
    const size_t         tensor_size = gguf_get_tensor_size(selected, chosen);
    if (tensor_size % size_t(sample.n_expert) != 0 || sample.group_bytes == 0) {
        error = "invalid expert tensor size";
        return false;
    }
    const size_t expert_size = tensor_size / size_t(sample.n_expert);
    if (sample.type < 0 || sample.type >= GGML_TYPE_COUNT || sample.ne0 % ggml_blck_size(sample.type) != 0) {
        error = "expert tensor type or shape is not supported";
        return false;
    }
    if (expert_size > 256 * 1024 * 1024) {
        error = "representative expert tensor exceeds the 256 MiB calibration limit";
        return false;
    }

    ggml_context_ptr ctx(ggml_init({ ggml_tensor_overhead() * 2, nullptr, true }));
    if (!ctx || ggml_nbytes(ggml_new_tensor_2d(ctx.get(), sample.type, sample.ne0, sample.ne1)) != expert_size) {
        error = "expert tensor layout is not supported";
        return false;
    }
    const size_t data_offset   = gguf_get_data_offset(selected);
    const size_t tensor_offset = gguf_get_tensor_offset(selected, chosen);
    if (tensor_offset > SIZE_MAX - data_offset) {
        error = "expert tensor offset overflows";
        return false;
    }
    const size_t  offset = data_offset + tensor_offset;
    std::ifstream file(paths[chosen_shard], std::ios::binary | std::ios::ate);
    if (!file || offset > size_t(file.tellg()) || expert_size > size_t(file.tellg()) - offset ||
        offset > size_t(std::numeric_limits<std::streamoff>::max())) {
        error = "expert tensor data is outside the model file";
        return false;
    }
    sample.weight.resize(expert_size);
    file.seekg(std::streamoff(offset));
    file.read(reinterpret_cast<char *>(sample.weight.data()), std::streamsize(expert_size));
    if (!file) {
        error = "could not read representative expert weights";
        return false;
    }
    return true;
}

static bool run_expert(device_result & result) {
    ggml_backend_tensor_set(result.input, result.input_data.data(), 0, result.input_data.size() * sizeof(float));
    if (ggml_backend_graph_compute(result.backend.get(), result.graph) != GGML_STATUS_SUCCESS) {
        return false;
    }
    ggml_backend_tensor_get(result.output, result.output_data.data(), 0, result.output_data.size() * sizeof(float));
    return true;
}

static std::string device_features(ggml_backend_dev_t dev) {
    ggml_backend_reg_t reg          = ggml_backend_dev_backend_reg(dev);
    auto               get_features = reinterpret_cast<ggml_backend_get_features_t>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_features"));
    if (!get_features) {
        return {};
    }
    std::string result;
    for (const ggml_backend_feature * feature = get_features(reg); feature && feature->name; ++feature) {
        if (strcmp(feature->value, "1") == 0) {
            if (!result.empty()) {
                result += ",";
            }
            result += feature->name;
        }
    }
    return result;
}

static bool estimate_working_memory(const char *                 model,
                                    expert_sample &              sample,
                                    std::vector<device_result> & devices,
                                    ggml_type                    type_k,
                                    ggml_type                    type_v,
                                    std::string &                error) {
    llama_model_params model_params     = llama_model_default_params();
    model_params.n_gpu_layers           = 0;
    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx                = sample.context_tokens;
    context_params.n_batch              = 2048;
    context_params.n_ubatch             = 512;
    context_params.n_seq_max            = 1;
    context_params.type_k               = type_k;
    context_params.type_v               = type_v;
    context_params.flash_attn_type      = ggml_is_quantized(type_k) || ggml_is_quantized(type_v) ?
        LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_AUTO;

    std::vector<ggml_backend_dev_t> backend_devices;
    uint32_t                        n_layers = 0;
    uint32_t                        n_expert = 0;
    ggml_log_callback               original_log;
    void *                          original_log_data;
    llama_log_get(&original_log, &original_log_data);
    llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
    try {
        uint32_t n_ctx_train = 0;
        const auto memory =
            common_get_device_memory_data(model, &model_params, &context_params, backend_devices, n_layers,
                                          n_ctx_train, n_expert, GGML_LOG_LEVEL_ERROR);
        if (memory.size() != backend_devices.size() + 1 || (sample.context_tokens == 0 && n_ctx_train == 0)) {
            throw std::runtime_error("could not estimate the full model context");
        }
        if (sample.context_tokens == 0) {
            sample.context_tokens = n_ctx_train;
        }
        for (auto & device : devices) {
            size_t index = memory.size();
            if (ggml_backend_dev_type(device.dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                index = memory.size() - 1;
            } else {
                // The initial probe is CPU-only. GPU memory is measured below per device.
                continue;
            }
            if (index < memory.size()) {
                device.context_bytes = memory[index].context;
                device.compute_bytes = memory[index].compute;
            } else {
                throw std::runtime_error("could not estimate working memory for a backend");
            }
        }
        for (auto & device : devices) {
            const auto type = ggml_backend_dev_type(device.dev);
            if (type != GGML_BACKEND_DEVICE_TYPE_GPU && type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
                continue;
            }
            ggml_backend_dev_t selected[] = { device.dev, nullptr };
            model_params.devices = selected;
            model_params.n_gpu_layers = -1;
            uint32_t gpu_n_ctx_train = 0;
            const auto gpu_memory =
                common_get_device_memory_data(model, &model_params, &context_params, backend_devices, n_layers,
                                              gpu_n_ctx_train, n_expert, GGML_LOG_LEVEL_ERROR);
            if (gpu_memory.size() != 2 || backend_devices.size() != 1 || backend_devices[0] != device.dev) {
                throw std::runtime_error("could not estimate full context on a GPU");
            }
            device.context_bytes = gpu_memory[0].context;
            device.compute_bytes = gpu_memory[0].compute;
        }
    } catch (const std::exception & e) {
        error = e.what();
        llama_log_set(original_log, original_log_data);
        return false;
    }
    llama_log_set(original_log, original_log_data);
    return true;
}

static void prepare_device(device_result & result, const expert_sample & sample) {
    ggml_backend_dev_props props;
    ggml_backend_dev_get_props(result.dev, &props);
    result.memory_free  = props.memory_free;
    result.memory_total = props.memory_total;
    result.device_id    = props.device_id ? props.device_id : "";
    result.features     = device_features(result.dev);
    result.memory_pool   = result.memory_total / 5 * 4 + result.memory_total % 5 * 4 / 5;
    // The calibration policy owns exactly 80% of total device memory.
    // Free-memory and compute estimates are reported for diagnostics only.
    result.memory_available_pool = result.memory_pool;
    const auto type = ggml_backend_dev_type(result.dev);
    result.kv_cache_bytes = type == GGML_BACKEND_DEVICE_TYPE_CPU ? result.context_bytes :
        size_t(double(result.context_bytes) * sample.gpu_kv_fraction);
    result.weight_budget = result.memory_available_pool;
    const size_t required = sample.dense_bytes > SIZE_MAX - result.kv_cache_bytes ? SIZE_MAX :
        sample.dense_bytes + result.kv_cache_bytes;
    if (result.weight_budget < required) {
        result.error = "dense weights and KV cache do not fit the available 80% memory pool";
        return;
    }
    result.expert_cache_budget = result.weight_budget - required;
    result.capacity = result.expert_cache_budget / sample.group_bytes;
    if (result.capacity == 0) {
        result.error = "representative expert group does not fit the expert cache budget";
        return;
    }
    result.backend.reset(ggml_backend_dev_init(result.dev, nullptr));
    if (!result.backend) {
        result.error = "backend initialization failed";
        return;
    }
    result.ctx.reset(ggml_init({ 1024 * 1024, nullptr, true }));
    if (!result.ctx) {
        result.error = "graph initialization failed";
        return;
    }
    result.weight = ggml_new_tensor_2d(result.ctx.get(), sample.type, sample.ne0, sample.ne1);
    result.input  = ggml_new_tensor_2d(result.ctx.get(), GGML_TYPE_F32, sample.ne0, 1);
    result.output = ggml_mul_mat(result.ctx.get(), result.weight, result.input);
    result.graph  = ggml_new_graph_custom(result.ctx.get(), 16, false);
    ggml_build_forward_expand(result.graph, result.output);
    if (!ggml_backend_dev_supports_op(result.dev, result.output)) {
        result.error = "expert matrix operation is unsupported";
        return;
    }
    result.buffer.reset(ggml_backend_alloc_ctx_tensors(result.ctx.get(), result.backend.get()));
    if (!result.buffer) {
        result.error = "backend buffer allocation failed";
        return;
    }
    result.input_data.assign(size_t(sample.ne0), 0.01f);
    result.output_data.resize(size_t(sample.ne1));
    ggml_backend_tensor_set(result.weight, sample.weight.data(), 0, sample.weight.size());
    constexpr int n_runs         = 7;
    const auto    transfer_start = std::chrono::steady_clock::now();
    for (int i = 0; i < n_runs; ++i) {
        ggml_backend_tensor_set(result.input, result.input_data.data(), 0, result.input_data.size() * sizeof(float));
        ggml_backend_tensor_get(result.input, result.input_data.data(), 0, result.input_data.size() * sizeof(float));
    }
    const auto transfer_end = std::chrono::steady_clock::now();
    result.ms_transfer      = std::chrono::duration<double, std::milli>(transfer_end - transfer_start).count() / n_runs;
    if (!run_expert(result)) {
        result.error = "expert benchmark failed";
        return;
    }
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < n_runs; ++i) {
        if (!run_expert(result)) {
            result.error = "expert benchmark failed";
            return;
        }
    }
    const auto end       = std::chrono::steady_clock::now();
    result.ms_per_expert = std::chrono::duration<double, std::milli>(end - start).count() / n_runs;
}

static void select_capacity_owners(std::vector<device_result> & devices) {
    for (size_t i = 0; i < devices.size(); ++i) {
        auto & device = devices[i];
        if (device.device_id.empty() || ggml_backend_dev_type(device.dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            continue;
        }
        bool seen = false;
        for (size_t j = 0; j < i; ++j) {
            if (devices[j].device_id == device.device_id) {
                seen = true;
                break;
            }
        }
        if (seen) {
            continue;
        }
        size_t owner = i;
        bool found_valid = !device.error.empty() ? false : true;
        for (size_t j = 0; j < devices.size(); ++j) {
            if (j == i || devices[j].device_id != device.device_id) {
                continue;
            }
            if (!devices[j].error.empty()) {
                continue;
            }
            if (!found_valid || devices[j].ms_per_expert < devices[owner].ms_per_expert) {
                owner = j;
                found_valid = true;
            }
        }
        for (size_t j = 0; j < devices.size(); ++j) {
            if (devices[j].device_id == device.device_id) {
                devices[j].capacity_owner = j == owner;
            }
        }
    }
}

static double measure_parallel(std::vector<device_result> & devices) {
    std::atomic<bool>        start(false);
    std::atomic<bool>        ok(true);
    std::vector<std::thread> threads;
    for (auto & device : devices) {
        if (device.assigned == 0) {
            continue;
        }
        device_result * current = &device;
        threads.emplace_back([current, &start, &ok]() {
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            for (int32_t i = 0; i < current->assigned; ++i) {
                if (!run_expert(*current)) {
                    ok.store(false, std::memory_order_relaxed);
                    return;
                }
            }
        });
    }
    const auto begin = std::chrono::steady_clock::now();
    start.store(true, std::memory_order_release);
    for (auto & thread : threads) {
        thread.join();
    }
    const auto end = std::chrono::steady_clock::now();
    return ok.load(std::memory_order_relaxed) ? std::chrono::duration<double, std::milli>(end - begin).count() : -1.0;
}

static double measure_parallel_median(std::vector<device_result> & devices) {
    double times[3];
    for (double & time : times) {
        time = measure_parallel(devices);
        if (time < 0) {
            return -1.0;
        }
    }
    std::sort(times, times + 3);
    return times[1];
}

static double improve_plan(std::vector<device_result> & devices) {
    double best_time = measure_parallel_median(devices);
    if (best_time < 0) {
        return -1.0;
    }
    for (int pass = 0; pass < 4; ++pass) {
        size_t best_from      = devices.size();
        size_t best_to        = devices.size();
        double candidate_time = best_time;
        for (size_t from = 0; from < devices.size(); ++from) {
            if (devices[from].assigned == 0) {
                continue;
            }
            for (size_t to = 0; to < devices.size(); ++to) {
                if (from == to || !devices[to].capacity_owner || !devices[to].error.empty() ||
                    devices[to].capacity <= size_t(devices[to].assigned)) {
                    continue;
                }
                devices[from].assigned--;
                devices[to].assigned++;
                const double time = measure_parallel_median(devices);
                devices[from].assigned++;
                devices[to].assigned--;
                if (time < 0) {
                    return -1.0;
                }
                if (time < candidate_time * 0.95) {
                    candidate_time = time;
                    best_from      = from;
                    best_to        = to;
                }
            }
        }
        if (best_from == devices.size()) {
            break;
        }
        devices[best_from].assigned--;
        devices[best_to].assigned++;
        best_time = candidate_time;
    }
    return best_time;
}


// What the host can page-lock for the expert arena. The probe commits one region, registers it with the backend in
// slices (the backend's register_host_buffer entry point, CUDA cudaHostRegister), times host to device copies out of
// it, and releases everything. Registering to the point of failure would leave the machine without memory, so the
// probe stops `headroom` below the available RAM and reports that as the proven limit.
struct host_arena_result {
    bool        measured = false;
    std::string device;
    std::string error;
    size_t      slice_bytes       = 0;
    size_t      available_ram     = 0;
    size_t      headroom          = 0;
    size_t      probed_bytes      = 0;   // what the probe tried to register
    size_t      registered_bytes  = 0;   // what the backend accepted
    size_t      recommended_bytes = 0;   // min(experts, registered)
    double      h2d_gbps          = 0.0;
};

static size_t available_ram_bytes() {
#if defined(_WIN32)
    MEMORYSTATUSEX ms = {};
    ms.dwLength = sizeof(ms);
    return GlobalMemoryStatusEx(&ms) ? size_t(ms.ullAvailPhys) : 0;
#else
    const long pages = sysconf(_SC_AVPHYS_PAGES), size = sysconf(_SC_PAGE_SIZE);
    return pages > 0 && size > 0 ? size_t(pages) * size_t(size) : 0;
#endif
}

static void * region_alloc(size_t bytes) {
#if defined(_WIN32)
    return VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    void * p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return p == MAP_FAILED ? nullptr : p;
#endif
}

static void region_free(void * p, size_t bytes) {
#if defined(_WIN32)
    (void) bytes;
    VirtualFree(p, 0, MEM_RELEASE);
#else
    munmap(p, bytes);
#endif
}

#ifdef HPCACHE_CUDA
// The probe runs before any backend queries device memory: a CUDA context on a second GPU (a display card on a narrow
// link) makes registrations of this size fail, and the compute process does not create one for the arena either.
static host_arena_result probe_host_arena(size_t expert_bytes, size_t headroom) {
    host_arena_result result;
    result.slice_bytes = size_t(256) << 20;
    result.headroom    = headroom;

    int n_devices = 0;
    if (cudaGetDeviceCount(&n_devices) != cudaSuccess || n_devices < 1) {
        cudaGetLastError();
        result.error = "no CUDA device";
        return result;
    }
    int best = 0;
    size_t best_mem = 0;
    for (int i = 0; i < n_devices; ++i) {   // property queries do not create a context
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, i) == cudaSuccess && prop.totalGlobalMem > best_mem) {
            best_mem = prop.totalGlobalMem;
            best     = i;
        }
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, best);
    result.device = prop.name;

    result.available_ram = available_ram_bytes();
    if (result.available_ram <= headroom + result.slice_bytes) {
        result.error = "not enough available RAM above the headroom";
        return result;
    }
    size_t want = std::min(expert_bytes, result.available_ram - headroom);
    want = (want + result.slice_bytes - 1) / result.slice_bytes * result.slice_bytes;
    if (want > result.available_ram - headroom) {
        want -= result.slice_bytes;
    }
    result.probed_bytes = want;

    uint8_t * base = static_cast<uint8_t *>(region_alloc(want));
    if (!base) {
        result.error = "could not allocate the probe region";
        return result;
    }
    {   // the pages must exist, as they will once the arena is filled from the file
        const int n_threads = int(std::max(1u, std::min(8u, std::thread::hardware_concurrency())));
        std::vector<std::thread> workers;
        for (int t = 0; t < n_threads; ++t) {
            workers.emplace_back([&, t] {
                for (size_t off = size_t(t) * result.slice_bytes; off < want; off += size_t(n_threads) * result.slice_bytes) {
                    memset(base + off, 0x5a, std::min(result.slice_bytes, want - off));
                }
            });
        }
        for (auto & w : workers) {
            w.join();
        }
    }

    cudaSetDevice(best);
    size_t registered = 0;
    unsigned flags = cudaHostRegisterMapped | cudaHostRegisterReadOnly;
    while (registered < want) {
        const size_t len = std::min(result.slice_bytes, want - registered);
        cudaError_t e = cudaHostRegister(base + registered, len, flags);
        if (e != cudaSuccess && registered == 0 && (flags & cudaHostRegisterReadOnly)) {
            cudaGetLastError();
            flags = cudaHostRegisterMapped;   // read-only registration is not supported here
            e = cudaHostRegister(base + registered, len, flags);
        }
        if (e != cudaSuccess) {
            cudaGetLastError();
            break;
        }
        registered += len;
    }

    // copy rate out of the registered region: 4 MiB pieces spread over it, every call checked
    bool usable = registered >= result.slice_bytes;
    if (usable) {
        const size_t piece = size_t(4) << 20;
        void *       dev_buf = nullptr;
        cudaStream_t stream  = nullptr;
        usable = cudaMalloc(&dev_buf, piece) == cudaSuccess && cudaStreamCreate(&stream) == cudaSuccess;
        const size_t n_slices = registered / result.slice_bytes;
        const int    n_pieces = 64;
        auto copy = [&](int i) {
            const size_t slice  = size_t(i) % n_slices;
            const size_t inside = (size_t(i) / n_slices * piece) % (result.slice_bytes - piece);
            return cudaMemcpyAsync(dev_buf, base + slice * result.slice_bytes + inside, piece, cudaMemcpyHostToDevice, stream) == cudaSuccess;
        };
        usable = usable && copy(0) && cudaStreamSynchronize(stream) == cudaSuccess;
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; usable && i < n_pieces; ++i) {
            usable = copy(i);
        }
        usable = usable && cudaStreamSynchronize(stream) == cudaSuccess;
        const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (usable) {
            result.h2d_gbps = double(piece) * n_pieces / 1e9 / sec;
        }
        if (stream) {
            cudaStreamDestroy(stream);
        }
        cudaFree(dev_buf);
    }
    for (size_t off = 0; off < registered; off += result.slice_bytes) {
        cudaHostUnregister(base + off);
    }
    cudaGetLastError();
    region_free(base, want);

    result.registered_bytes = usable ? registered : 0;
    if (!usable) {
        result.error = "registered memory could not be copied from at this size";
    }
    result.recommended_bytes = std::min(result.registered_bytes, expert_bytes);
    result.measured = true;
    return result;
}
#else
static host_arena_result probe_host_arena(size_t, size_t headroom) {
    host_arena_result result;
    result.headroom = headroom;
    result.error    = "built without CUDA";
    return result;
}
#endif

// The split of the missed experts between the CPU and the GPU's PCIe path: each gets a share proportional to its rate.
struct expert_split_result {
    bool   valid = false;
    double cpu_gb_per_s = 0.0;
    double pcie_gb_per_s = 0.0;
    double pcie_frac = 0.0;
    int    pcie_max = 0;
};

static expert_split_result derive_split(const std::vector<device_result> & devices, const expert_sample & sample,
        const host_arena_result & arena) {
    expert_split_result r;
    if (!arena.measured || arena.h2d_gbps <= 0.0) {
        return r;
    }
    for (const auto & device : devices) {
        if (ggml_backend_dev_type(device.dev) == GGML_BACKEND_DEVICE_TYPE_CPU && device.error.empty() && device.ms_per_expert > 0.0) {
            // one expert matrix of the sample tensor per timed operation
            const double bytes = double(ggml_row_size(sample.type, sample.ne0)) * double(sample.ne1);
            r.cpu_gb_per_s = bytes / (device.ms_per_expert * 1e6);
            break;
        }
    }
    if (r.cpu_gb_per_s <= 0.0) {
        return r;
    }
    r.pcie_gb_per_s = arena.h2d_gbps;
    r.pcie_frac = std::min(0.8, std::max(0.1, r.pcie_gb_per_s / (r.pcie_gb_per_s + r.cpu_gb_per_s)));
    // about half of the routed experts miss the pool: the most the GPU path has to hold
    r.pcie_max = std::min(6, std::max(2, int(std::ceil(double(sample.n_used) * 0.55 * r.pcie_frac))));
    r.valid = true;
    return r;
}

static bool write_json(const char *                       path,
                       const char *                       model,
                       const expert_sample &              sample,
                       const std::vector<device_result> & devices,
                       ggml_type                          type_k,
                       ggml_type                          type_v,
                       double                             parallel_ms,
                       size_t                             expert_shortfall,
                       const host_arena_result &          arena,
                       const expert_split_result &        split) {
    std::ofstream out(path);
    if (!out) {
        return false;
    }
    size_t weight_budget = 0;
    for (const auto & device : devices) {
        if (device.capacity_owner && device.error.empty()) {
            weight_budget += device.weight_budget;
        }
    }
    const size_t weight_bytes = sample.expert_bytes + sample.dense_bytes + sample.ple_bytes;
    const size_t mean_expert_bytes = sample.expert_bytes / size_t(sample.expert_slots);
    const size_t p99_expert_bytes  = percentile(sample.expert_slot_sizes, 99);
    out << std::fixed << std::setprecision(6);
    out << "{\n  \"schema_version\": 1,\n  \"model\": \"" << json_escape(model) << "\",\n";
    out << "  \"plan_kind\": \"active_expert_work_split\",\n";
    out << "  \"plan_role\": \"moe_hpcache_capacity\",\n";
    out << "  \"architecture\": \"" << json_escape(sample.architecture) << "\",\n";
    out << "  \"context_tokens\": " << sample.context_tokens << ",\n";
    out << "  \"cache_type_k\": \"" << ggml_type_name(type_k) << "\",\n";
    out << "  \"cache_type_v\": \"" << ggml_type_name(type_v) << "\",\n";
    out << "  \"working_memory_placement\": \"full_layers_per_backend_probe\",\n";
    out << "  \"memory_pool_fraction\": 0.80,\n";
    char kv_fraction[32];
    snprintf(kv_fraction, sizeof(kv_fraction), "%.2f", sample.gpu_kv_fraction);
    out << "  \"gpu_kv_cache_fraction\": " << kv_fraction << ",\n";
    out << "  \"cpu_kv_cache_fraction\": 1.00,\n";
    out << "  \"cpu_kv_authoritative\": true,\n";
    out << "  \"gpu_dense_weights_required\": true,\n";
    out << "  \"placement_policy\": \"ranked_capacity_with_cpu_fallback\",\n";
    out << "  \"tensor_count\": " << sample.n_tensors << ",\n";
    out << "  \"expert_weight_bytes\": " << sample.expert_bytes << ",\n";
    out << "  \"other_weight_bytes\": " << sample.dense_bytes + sample.ple_bytes + (sample.token_on_cpu ? sample.token_bytes : 0) << ",\n";
    out << "  \"dense_weight_bytes\": " << sample.dense_bytes << ",\n";
    out << "  \"ple_table_weight_bytes\": " << sample.ple_bytes << ",\n";
    out << "  \"output_head_weight_bytes\": " << sample.output_bytes << ",\n";
    out << "  \"token_embedding_weight_bytes\": " << sample.token_bytes << ",\n";
    out << "  \"dense_tensor_bytes_p95\": " << percentile(sample.dense_tensor_sizes, 95) << ",\n";
    out << "  \"dense_tensor_bytes_p99\": " << percentile(sample.dense_tensor_sizes, 99) << ",\n";
    out << "  \"dense_layer_bytes_p95\": " << percentile(sample.dense_layer_sizes, 95) << ",\n";
    out << "  \"dense_layer_bytes_p99\": " << percentile(sample.dense_layer_sizes, 99) << ",\n";
    out << "  \"expert_slots_total\": " << sample.expert_slots << ",\n";
    out << "  \"expert_slot_bytes_mean\": " << mean_expert_bytes << ",\n";
    out << "  \"expert_slot_bytes_p95\": " << percentile(sample.expert_slot_sizes, 95) << ",\n";
    out << "  \"expert_slot_bytes_p99\": " << p99_expert_bytes << ",\n";
    out << "  \"expert_capacity_shortfall_bytes\": " << expert_shortfall << ",\n";
    out << "  \"resident_weight_budget_bytes\": " << weight_budget << ",\n";
    out << "  \"all_weights_resident_feasible\": " << (weight_bytes <= weight_budget ? "true" : "false") << ",\n";
    out << "  \"resident_weight_shortfall_bytes\": "
        << (weight_bytes > weight_budget ? weight_bytes - weight_budget : 0) << ",\n";
    out << "  \"sample_tensor\": \"" << json_escape(sample.name) << "\",\n";
    out << "  \"sample_type\": \"" << ggml_type_name(sample.type) << "\",\n";
    out << "  \"sample_shape\": [" << sample.ne0 << ", " << sample.ne1 << ", " << sample.n_expert << "],\n";
    out << "  \"active_experts_per_layer\": " << sample.n_used << ",\n";
    out << "  \"bytes_per_expert_group\": " << sample.group_bytes << ",\n";
    out << "  \"expert_cache_capacity_assumption\": \"all dense weights on each device, PLE offloaded\",\n";
    out << "  \"host_arena\": {\"measured\": " << (arena.measured ? "true" : "false") << ", "
        << "\"device\": \"" << json_escape(arena.device) << "\", "
        << "\"slice_bytes\": " << arena.slice_bytes << ", "
        << "\"available_ram_bytes\": " << arena.available_ram << ", "
        << "\"headroom_bytes\": " << arena.headroom << ", "
        << "\"probed_bytes\": " << arena.probed_bytes << ", "
        << "\"registered_bytes\": " << arena.registered_bytes << ", "
        << "\"recommended_bytes\": " << arena.recommended_bytes << ", "
        << "\"h2d_gb_per_s\": " << arena.h2d_gbps << ", "
        << "\"error\": \"" << json_escape(arena.error) << "\"},\n";
    if (split.valid) {
        out << "  \"expert_split\": {\"cpu_gb_per_s\": " << split.cpu_gb_per_s << ", "
            << "\"pcie_gb_per_s\": " << split.pcie_gb_per_s << ", "
            << "\"pcie_frac\": " << split.pcie_frac << ", "
            << "\"pcie_max\": " << split.pcie_max << "},\n";
    }
    out << "  \"parallel_sample_ms\": " << parallel_ms << ",\n  \"devices\": [\n";
    for (size_t i = 0; i < devices.size(); ++i) {
        const auto &           device = devices[i];
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(device.dev, &props);
        const size_t expert_cache_bytes = device.expert_cache_budget;
        const size_t slots_mean = mean_expert_bytes ? std::min(size_t(sample.expert_slots), expert_cache_bytes / mean_expert_bytes) : 0;
        const size_t slots_p99  = p99_expert_bytes ? std::min(size_t(sample.expert_slots), expert_cache_bytes / p99_expert_bytes) : 0;
        out << "    {\"name\": \"" << json_escape(ggml_backend_dev_name(device.dev)) << "\", "
            << "\"description\": \"" << json_escape(props.description ? props.description : "") << "\", "
            << "\"type\": \"" << device_type_name(props.type) << "\", "
            << "\"device_id\": \"" << json_escape(props.device_id ? props.device_id : "") << "\", "
            << "\"capacity_owner\": " << (device.capacity_owner ? "true" : "false") << ", "
            << "\"async\": " << (props.caps.async ? "true" : "false") << ", "
            << "\"host_buffer\": " << (props.caps.host_buffer ? "true" : "false") << ", "
            << "\"events\": " << (props.caps.events ? "true" : "false") << ", "
            << "\"memory_free_bytes\": " << device.memory_free << ", "
            << "\"memory_total_bytes\": " << device.memory_total << ", "
            << "\"memory_pool_bytes\": " << device.memory_pool << ", "
            << "\"memory_available_pool_bytes\": " << device.memory_available_pool << ", "
            << "\"context_bytes\": " << device.context_bytes << ", "
            << "\"compute_bytes\": " << device.compute_bytes << ", "
            << "\"dense_weight_bytes\": " << sample.dense_bytes << ", "
            << "\"kv_cache_bytes\": " << device.kv_cache_bytes << ", "
            << "\"weight_budget_bytes\": " << device.weight_budget << ", "
            << "\"expert_cache_budget_bytes\": " << device.expert_cache_budget << ", "
            << "\"max_resident_expert_groups\": " << device.capacity << ", "
            << "\"expert_cache_budget_after_dense_bytes\": " << expert_cache_bytes << ", "
            << "\"expert_cache_slots_mean\": " << slots_mean << ", "
            << "\"expert_cache_slots_p99\": " << slots_p99 << ", "
            << "\"features\": \"" << json_escape(device.features) << "\", "
            << "\"ms_transfer_roundtrip\": " << device.ms_transfer << ", "
            << "\"ms_per_sample_expert\": " << device.ms_per_expert << ", "
            << "\"benchmark_active_experts\": " << device.assigned << ", "
            << "\"error\": \"" << json_escape(device.error) << "\"}" << (i + 1 < devices.size() ? "," : "") << "\n";
    }
    out << "  ]\n}\n";
    return bool(out);
}

int main(int argc, char ** argv) {
    const char * model  = nullptr;
    const char * output = "hpcache-calibration.json";
    uint32_t     context_tokens = 0;
    bool         context_explicit = false;
    ggml_type    type_k = GGML_TYPE_F16;
    ggml_type    type_v = GGML_TYPE_F16;
    double       kv_fraction = 1.0;
    double       arena_headroom_gib = 12.0;
    bool         probe_arena = true;
    static const char * usage = "usage: %s -m model.gguf [-o hpcache-calibration.json] [-ctx tokens|128kib] [-ctk f16|q5_0] [-ctv f16|q5_0] [-kvf 0.0-1.0] [-arena-headroom-gib N] [-no-arena]\n";
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model = argv[++i];
        } else if (strcmp(argv[i], "-kvf") == 0 && i + 1 < argc) {
            char * end = nullptr;
            kv_fraction = strtod(argv[++i], &end);
            if (end == argv[i] || *end != '\0' || !(kv_fraction > 0.0 && kv_fraction <= 1.0)) {
                fprintf(stderr, "calibration: -kvf must be in (0, 1]\n");
                return 1;
            }
        } else if (strcmp(argv[i], "-arena-headroom-gib") == 0 && i + 1 < argc) {
            arena_headroom_gib = atof(argv[++i]);
        } else if (strcmp(argv[i], "-no-arena") == 0) {
            probe_arena = false;
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            output = argv[++i];
        } else if (strcmp(argv[i], "-ctx") == 0 && i + 1 < argc) {
            if (!parse_context_tokens(argv[++i], context_tokens)) {
                fprintf(stderr, "calibration: invalid context size (use tokens or a k/m suffix, for example 128kib)\n");
                return 1;
            }
            context_explicit = true;
        } else if (strcmp(argv[i], "-ctk") == 0 && i + 1 < argc) {
            const char * value = argv[++i];
            if (strcmp(value, "f16") == 0) {
                type_k = GGML_TYPE_F16;
            } else if (strcmp(value, "q5_0") == 0) {
                type_k = GGML_TYPE_Q5_0;
            } else {
                fprintf(stderr, "calibration: unsupported K cache type: %s\n", value);
                return 1;
            }
        } else if (strcmp(argv[i], "-ctv") == 0 && i + 1 < argc) {
            const char * value = argv[++i];
            if (strcmp(value, "f16") == 0) {
                type_v = GGML_TYPE_F16;
            } else if (strcmp(value, "q5_0") == 0) {
                type_v = GGML_TYPE_Q5_0;
            } else {
                fprintf(stderr, "calibration: unsupported V cache type: %s\n", value);
                return 1;
            }
        } else {
            fprintf(stderr, usage, argv[0]);
            return 1;
        }
    }
    if (!model) {
        fprintf(stderr, usage, argv[0]);
        return 1;
    }
    expert_sample sample;
    std::string   error;
    if (!load_sample(model, sample, error)) {
        fprintf(stderr, "calibration: %s\n", error.c_str());
        return 1;
    }
    sample.gpu_kv_fraction = kv_fraction;
    if (context_explicit) {
        sample.context_tokens = context_tokens;
    }
    host_arena_result arena;
    if (probe_arena) {
        arena = probe_host_arena(sample.expert_bytes, size_t(arena_headroom_gib * double(1ull << 30)));
    }
#ifdef GGML_BACKEND_DL
    ggml_backend_load_all();
#endif
    std::vector<device_result> devices;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        devices.emplace_back();
        devices.back().dev = dev;
    }
    if (!estimate_working_memory(model, sample, devices, type_k, type_v, error)) {
        fprintf(stderr, "calibration: %s\n", error.c_str());
        return 1;
    }
    for (auto & device : devices) {
        prepare_device(device, sample);
    }
    select_capacity_owners(devices);
    for (int32_t i = 0; i < sample.n_used; ++i) {
        device_result * best = nullptr;
        for (auto & device : devices) {
            if (!device.capacity_owner || !device.error.empty() || device.capacity <= size_t(device.assigned)) {
                continue;
            }
            if (!best || (device.assigned + 1) * device.ms_per_expert < (best->assigned + 1) * best->ms_per_expert) {
                best = &device;
            }
        }
        if (!best) {
            fprintf(stderr, "calibration: not enough resident expert capacity for %d active experts\n", sample.n_used);
            return 1;
        }
        best->assigned++;
    }
    const double parallel_ms = improve_plan(devices);
    size_t expert_capacity = 0;
    for (const auto & device : devices) {
        if (device.error.empty()) {
            expert_capacity += device.expert_cache_budget;
        }
    }
    const size_t expert_shortfall = sample.expert_bytes > expert_capacity ? sample.expert_bytes - expert_capacity : 0;
    const expert_split_result split = derive_split(devices, sample, arena);
    if (parallel_ms < 0 || !write_json(output, model, sample, devices, type_k, type_v, parallel_ms, expert_shortfall, arena, split)) {
        fprintf(stderr, "calibration: could not complete or write calibration\n");
        return 1;
    }
    printf("calibration: %s (%d active experts), %u context tokens, parallel sample %.3f ms\n", sample.name.c_str(),
           sample.n_used, sample.context_tokens, parallel_ms);
    printf("calibration: dense %.2f GiB, PLE %.2f GiB, experts %.2f GiB; expert slot mean %.2f MiB, P99 %.2f MiB\n",
           double(sample.dense_bytes) / (1ull << 30), double(sample.ple_bytes) / (1ull << 30),
           double(sample.expert_bytes) / (1ull << 30),
           double(sample.expert_bytes) / double(sample.expert_slots) / (1ull << 20),
           double(percentile(sample.expert_slot_sizes, 99)) / (1ull << 20));
    for (const auto & device : devices) {
        printf("  %s: %d experts, %.3f ms/expert%s%s\n", ggml_backend_dev_name(device.dev), device.assigned,
               device.ms_per_expert, device.error.empty() ? "" : " - ", device.error.c_str());
        if (ggml_backend_dev_type(device.dev) == GGML_BACKEND_DEVICE_TYPE_GPU && device.capacity_owner) {
            const size_t cache_bytes = device.expert_cache_budget;
            const size_t mean_bytes = sample.expert_bytes / size_t(sample.expert_slots);
            const size_t p99_bytes  = percentile(sample.expert_slot_sizes, 99);
            printf("    80%% pool: %.2f GiB, dense %.2f GiB, GPU KV %.2f GiB, expert cache %.2f GiB; %zu mean-size or %zu P99-size slots\n",
                   double(device.memory_pool) / (1ull << 30), double(sample.dense_bytes) / (1ull << 30),
                   double(device.kv_cache_bytes) / (1ull << 30),
                   double(cache_bytes) / (1ull << 30), mean_bytes ? cache_bytes / mean_bytes : 0,
                   p99_bytes ? cache_bytes / p99_bytes : 0);
        }
    }
    if (split.valid) {
        printf("calibration: expert split: CPU %.1f GB/s, PCIe %.1f GB/s -> GPU share %.2f, up to %d experts per layer\n",
               split.cpu_gb_per_s, split.pcie_gb_per_s, split.pcie_frac, split.pcie_max);
    }
    if (arena.measured) {
        printf("calibration: host arena: %.2f GiB registered of %.2f GiB probed (RAM available %.2f GiB, headroom %.2f GiB), "
               "recommended %.2f GiB, copy %.1f GB/s\n",
               double(arena.registered_bytes) / (1ull << 30), double(arena.probed_bytes) / (1ull << 30),
               double(arena.available_ram) / (1ull << 30), double(arena.headroom) / (1ull << 30),
               double(arena.recommended_bytes) / (1ull << 30), arena.h2d_gbps);
    } else if (probe_arena) {
        printf("calibration: host arena not measured: %s\n", arena.error.c_str());
    }
    printf("calibration: wrote %s\n", output);
    if (expert_shortfall > 0) {
        fprintf(stderr, "calibration: expert weights exceed the backend memory pools by %.2f GiB\n",
                double(expert_shortfall) / (1ull << 30));
        return 1;
    }
    return 0;
}
