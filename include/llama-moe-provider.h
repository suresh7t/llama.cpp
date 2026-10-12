#pragma once

// Experimental MoE provider interface. A provider is a shared library that can replace
// expert placement and per-layer MoE scheduling. Every hook left NULL uses the core default.

#include "ggml.h"
#include "ggml-backend.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LLAMA_MOE_ABI_VERSION 1

struct llama_model;

//
// placement
//

enum llama_moe_tensor_kind {
    LLAMA_MOE_TENSOR_OTHER,
    LLAMA_MOE_TENSOR_DENSE,
    LLAMA_MOE_TENSOR_EXPERT_GATE,
    LLAMA_MOE_TENSOR_EXPERT_UP,
    LLAMA_MOE_TENSOR_EXPERT_DOWN,
    LLAMA_MOE_TENSOR_EXPERT_GATE_UP,
    LLAMA_MOE_TENSOR_PLE,
};

struct llama_moe_tensor_desc {
    const char * name;
    enum llama_moe_tensor_kind kind;
    int32_t layer;          // -1 if not tied to a layer
    enum ggml_type type;
    int64_t ne[GGML_MAX_DIMS];
    size_t  nbytes;
    size_t  expert_nbytes;  // bytes of one expert slice, 0 if not an expert tensor
    int32_t n_expert;       // 0 if not an expert tensor
    uint16_t file_index;    // GGUF split file
    uint64_t file_offset;   // byte offset of the tensor data in that file
};

struct llama_moe_device_info {
    int32_t index;          // position in the array of the call that returned it
    ggml_backend_dev_t dev;
    const char * name;
    enum ggml_backend_dev_type type;
    size_t mem_total;
    size_t mem_free;
};

enum llama_moe_home_kind {
    LLAMA_MOE_HOME_DEFAULT,   // core default placement
    LLAMA_MOE_HOME_DEVICE,    // core loads the tensor into a buffer of devices[device]
};

struct llama_moe_tensor_home {
    enum llama_moe_home_kind kind;
    int32_t device;           // index into the devices array that place() received
};

//
// scheduling
//

enum llama_moe_ffn_op {
    LLAMA_MOE_FFN_SILU,   // gate and up combined as silu(gate) * up
    LLAMA_MOE_FFN_OTHER,  // any other activation, the provider should return NULL
};

struct llama_moe_layer_request {
    struct ggml_context * ctx;      // graph build context, new nodes go here
    struct ggml_cgraph  * gf;
    struct ggml_tensor  * cur;               // layer input, [n_embd, n_tokens]
    struct ggml_tensor  * selected_experts;  // I32, [n_expert_used, n_tokens]
    struct ggml_tensor  * weights;           // F32, [1, n_expert_used, n_tokens], already normalized and scaled
    struct ggml_tensor  * gate_exps;         // NULL when absent or provider owned
    struct ggml_tensor  * up_exps;
    struct ggml_tensor  * down_exps;
    struct ggml_tensor  * gate_up_exps;
    int32_t il;
    int32_t n_expert;
    int32_t n_expert_used;
    int32_t n_tokens;
    enum llama_moe_ffn_op ffn_op;
    float   swiglu_clamp;           // clamp limit of the gate and up activations, 0 when unused
};

//
// decode lifecycle
//

struct llama_moe_decode_info {
    int32_t n_tokens;
    int32_t n_layer;
    int32_t seq_id;                 // -1 when the ubatch has more than one sequence
};

//
// services that the core gives to the provider
// a service that the core does not implement yet is NULL, the provider must check
//

struct llama_moe_host {
    uint32_t abi_version;
    size_t   struct_size;
    void   * ctx;                   // pass as the first argument of every call

    void (*log)(void * ctx, enum ggml_log_level level, const char * message);

    // --moe-provider-args entries, plus "model", "n_ctx", "ctk", "ctv", "n_gpu_layers"
    const char * (*get_param)(void * ctx, const char * name);

    // fill out[] with up to cap devices and return the device count
    int32_t (*get_devices)(void * ctx, struct llama_moe_device_info * out, int32_t cap);

    // valid after the model is loaded, NULL for provider owned tensors
    struct ggml_tensor * (*get_tensor)(void * ctx, const char * name);

    // K or V tensor of an attention layer, NULL when the layer has no KV cache
    struct ggml_tensor * (*get_kv_tensor)(void * ctx, int32_t il, int32_t is_v);

    // core default algorithms, so that a provider can compose them
    void (*default_placement)(void * ctx, const struct llama_moe_tensor_desc * tensors, int32_t n_tensors,
            struct llama_moe_tensor_home * homes);
    struct ggml_tensor * (*default_moe_layer)(void * ctx, const struct llama_moe_layer_request * req);
};

//
// what the provider gives to the core
//

struct llama_moe_provider_api {
    uint32_t abi_version;
    size_t   struct_size;
    void   * handle;                // pass as the first argument of every call

    // called once, after GGUF metadata is read and before any tensor is created
    // devices[0] is the CPU, the rest are the devices that the model uses
    // homes[i] is the answer for tensors[i], it is DEFAULT on entry
    // a --override-tensor pattern wins over the provider
    void (*place)(void * handle, const struct llama_moe_device_info * devices, int32_t n_devices,
            const struct llama_moe_tensor_desc * tensors, int32_t n_tensors, struct llama_moe_tensor_home * homes);

    // called once, after tensors are loaded and before the first decode
    int32_t (*model_loaded)(void * handle, const struct llama_model * model);

    // called while the graph is built, after top-k selection and weight normalization
    // returns the weighted sum of routed expert outputs, [n_embd, n_tokens]
    // the core adds shared experts and the residual
    // returns NULL to use the stock path
    // the core calls this only when n_tokens == 1, without expert biases, scales or LoRA adapters
    struct ggml_tensor * (*build_moe_layer)(void * handle, const struct llama_moe_layer_request * req);

    // right before the graph is submitted, must not block
    void (*decode_begin)(void * handle, const struct llama_moe_decode_info * info);
    // right after the graph is submitted, the device may still be running it
    // the provider can wait here for its own work on this graph
    // always paired with decode_begin, also when the submission failed
    void (*decode_end)(void * handle);

    // after compute, cells [cell_begin, cell_begin + n_cells) of the KV cache were written
    void (*kv_commit)(void * handle, int32_t seq_id, int32_t cell_begin, int32_t n_cells);

    void (*free)(void * handle);
};

// the only symbol that a provider library must export, named llama_moe_provider_init
// returns NULL on failure and must not throw
// the returned table and its handle stay valid until api->free is called
typedef const struct llama_moe_provider_api * (*llama_moe_provider_init_fn)(
        const struct llama_moe_host * host, const char * args);

#ifdef __cplusplus
}
#endif
