#pragma once

// A draft head over a subset of the vocabulary, for the MTP head of a model (llama_moe_provider_api::build_output).
//
// The draft layer only has to propose tokens that the target then checks, so its output projection can run over the rows of
// output.weight that belong to a list of token ids (--moe-provider-args draft_vocab=<file of int32 ids>, the format of the
// draft_vocab*.bin files of Strata). The rows are gathered once, on the first graph that asks, into a buffer of their own;
// the graph then computes mul_mat over those rows and a small op scatters the result into logits of the full width with -inf
// elsewhere, so the sampler and the speculative code do not change. A token outside the list is never drafted.

#include "llama-moe-provider.h"

#include <memory>
#include <string>

struct ggml_tensor;

class hpcache_draft_head {
public:
    // ids_path: file of int32 token ids, log: the provider's log
    hpcache_draft_head(std::string ids_path, void (*log)(void * ctx, enum ggml_log_level level, const std::string & message), void * log_ctx);
    ~hpcache_draft_head();

    // the logits for req, or NULL for the stock path (no list, output not in a CUDA buffer, list not usable)
    ggml_tensor * build(const llama_moe_output_request & req);

private:
    struct impl;
    std::unique_ptr<impl> p;
};
