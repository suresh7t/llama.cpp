#include "hpcache-draft-head.h"

#include "hpcache-cuda.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <vector>

// the CUDA backend runs a custom op on its own stream through two calls that it exports in its registry
typedef void   (*custom_set_stream_t)(ggml_tensor *);
typedef void * (*custom_get_stream_t)(const ggml_tensor *);

static void * cuda_proc(const char * name) {
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("CUDA");
    return reg != nullptr ? ggml_backend_reg_get_proc_address(reg, name) : nullptr;
}

namespace {

struct op_data {
    int32_t         n_sub   = 0;
    int32_t         n_vocab = 0;
    const int32_t * ids     = nullptr;   // device, n_sub entries
};

// the head of one output tensor
struct head_state {
    const ggml_tensor *   key  = nullptr;     // the output.weight this head was cut from
    ggml_context *        ctx  = nullptr;
    ggml_backend_buffer_t buf  = nullptr;
    ggml_tensor *         head = nullptr;     // [n_embd, n_sub], the type of output.weight
    ggml_tensor *         ids  = nullptr;     // I32 [n_sub]
    op_data               op;
    bool                  usable = false;

    ~head_state() {
        if (buf != nullptr) {
            ggml_backend_buffer_free(buf);
        }
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

void scatter_cb(ggml_tensor * dst, int, int, void * userdata) {
    static const custom_get_stream_t get_stream = (custom_get_stream_t) cuda_proc("ggml_backend_cuda_custom_get_stream");
    const auto * d = static_cast<const op_data *>(userdata);
    hpcache_cuda_scatter_logits(get_stream(dst), (const float *) dst->src[0]->data, d->ids, d->n_sub, d->n_vocab, (int) dst->ne[1],
            (float *) dst->data);
}

} // namespace

struct hpcache_draft_head::impl {
    std::string ids_path;
    void (*log)(void *, enum ggml_log_level, const std::string &) = nullptr;
    void * log_ctx = nullptr;

    std::mutex mutex;
    std::vector<std::unique_ptr<head_state>> heads;

    void say(enum ggml_log_level level, const std::string & msg) const {
        if (log != nullptr) {
            log(log_ctx, level, msg);
        }
    }

    bool read_ids(std::vector<int32_t> & ids) const {
        std::ifstream f(ids_path, std::ios::binary | std::ios::ate);
        if (!f) {
            say(GGML_LOG_LEVEL_WARN, "draft vocabulary: cannot open " + ids_path);
            return false;
        }
        const std::streamsize bytes = f.tellg();
        if (bytes < 4 || bytes % 4 != 0) {
            say(GGML_LOG_LEVEL_WARN, "draft vocabulary: " + ids_path + " is not a file of int32 ids");
            return false;
        }
        ids.resize((size_t) bytes / 4);
        f.seekg(0);
        f.read((char *) ids.data(), bytes);
        return (bool) f;
    }

    head_state * find_or_create(const llama_moe_output_request & req) {
        for (auto & h : heads) {
            if (h->key == req.output) {
                return h.get();
            }
        }
        auto st = std::make_unique<head_state>();
        st->key = req.output;
        head_state * out = st.get();
        heads.push_back(std::move(st));
        create(*out, req);
        return out;
    }

    void create(head_state & st, const llama_moe_output_request & req) {
        const ggml_tensor * w = req.output;
        if (w == nullptr || w->buffer == nullptr || w->ne[2] != 1 || w->ne[3] != 1 || w->ne[0] != req.n_embd || w->ne[1] != req.n_vocab ||
                w->nb[1] != ggml_row_size(w->type, w->ne[0])) {
            say(GGML_LOG_LEVEL_WARN, "draft vocabulary: output.weight is not a plain 2D tensor, the full head is used");
            return;
        }
        // the head must sit on the card of the graph: a CUDA buffer that is not pinned host memory
        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(w->buffer);
        if (ggml_backend_buffer_is_host(w->buffer) || strncmp(ggml_backend_buft_name(buft), "CUDA", 4) != 0) {
            say(GGML_LOG_LEVEL_WARN, std::string("draft vocabulary: output.weight is in a '") + ggml_backend_buft_name(buft) +
                    "' buffer, not on a card, the full head is used");
            return;
        }
        if (cuda_proc("ggml_backend_cuda_custom_set_stream") == nullptr || cuda_proc("ggml_backend_cuda_custom_get_stream") == nullptr) {
            say(GGML_LOG_LEVEL_WARN, "draft vocabulary: the CUDA backend does not export the custom op stream calls, the full head is used");
            return;
        }

        std::vector<int32_t> ids;
        if (!read_ids(ids)) {
            return;
        }
        // distinct ids in range, in file order
        {
            std::vector<uint8_t> seen((size_t) req.n_vocab, 0);
            size_t n = 0;
            for (int32_t id : ids) {
                if (id >= 0 && id < req.n_vocab && !seen[(size_t) id]) {
                    seen[(size_t) id] = 1;
                    ids[n++] = id;
                }
            }
            if (n != ids.size()) {
                say(GGML_LOG_LEVEL_WARN, "draft vocabulary: " + std::to_string(ids.size() - n) + " ids are repeated or outside the vocabulary of " +
                        std::to_string(req.n_vocab) + " tokens, dropped");
            }
            ids.resize(n);
        }
        if (ids.empty() || ids.size() >= (size_t) req.n_vocab) {
            say(GGML_LOG_LEVEL_WARN, "draft vocabulary: no usable subset (" + std::to_string(ids.size()) + " ids), the full head is used");
            return;
        }

        const auto t0 = std::chrono::steady_clock::now();
        const size_t row_bytes = ggml_row_size(w->type, w->ne[0]);
        const size_t n_sub     = ids.size();

        // the rows of the subset, gathered from a copy of the tensor
        std::vector<uint8_t> all(ggml_nbytes(w));
        ggml_backend_tensor_get(w, all.data(), 0, all.size());
        std::vector<uint8_t> rows(n_sub * row_bytes);
        for (size_t j = 0; j < n_sub; ++j) {
            memcpy(rows.data() + j * row_bytes, all.data() + (size_t) ids[j] * row_bytes, row_bytes);
        }
        all = std::vector<uint8_t>();

        ggml_init_params ip = { ggml_tensor_overhead() * 4 + 4096, nullptr, true };
        st.ctx = ggml_init(ip);
        if (st.ctx == nullptr) {
            return;
        }
        st.head = ggml_new_tensor_2d(st.ctx, w->type, w->ne[0], (int64_t) n_sub);
        ggml_set_name(st.head, "draft_head");
        st.ids = ggml_new_tensor_1d(st.ctx, GGML_TYPE_I32, (int64_t) n_sub);
        ggml_set_name(st.ids, "draft_head_ids");
        st.buf = ggml_backend_alloc_ctx_tensors_from_buft(st.ctx, buft);
        if (st.buf == nullptr) {
            say(GGML_LOG_LEVEL_WARN, "draft vocabulary: the head of " + std::to_string(n_sub * row_bytes / (1024 * 1024)) +
                    " MiB does not fit on the card, the full head is used");
            ggml_free(st.ctx);
            st.ctx = nullptr;
            return;
        }
        ggml_backend_buffer_set_usage(st.buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_tensor_set(st.head, rows.data(), 0, rows.size());
        ggml_backend_tensor_set(st.ids, ids.data(), 0, ids.size() * sizeof(int32_t));

        st.op.n_sub   = (int32_t) n_sub;
        st.op.n_vocab = req.n_vocab;
        st.op.ids     = (const int32_t *) st.ids->data;
        st.usable     = true;

        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        char msg[256];
        snprintf(msg, sizeof(msg), "draft head over %zu of %d tokens (%.1f MiB instead of %.1f MiB), gathered in %.0f ms", n_sub, req.n_vocab,
                (double) (n_sub * row_bytes) / 1048576.0, (double) ggml_nbytes(w) / 1048576.0, ms);
        say(GGML_LOG_LEVEL_INFO, msg);
    }
};

hpcache_draft_head::hpcache_draft_head(std::string ids_path,
        void (*log)(void *, enum ggml_log_level, const std::string &), void * log_ctx) : p(new impl()) {
    p->ids_path = std::move(ids_path);
    p->log      = log;
    p->log_ctx  = log_ctx;
}

hpcache_draft_head::~hpcache_draft_head() = default;

ggml_tensor * hpcache_draft_head::build(const llama_moe_output_request & req) {
    if (p->ids_path.empty()) {
        return nullptr;
    }
    head_state * st;
    {
        std::lock_guard<std::mutex> lock(p->mutex);
        st = p->find_or_create(req);
    }
    if (!st->usable) {
        return nullptr;
    }

    static const custom_set_stream_t set_stream = (custom_set_stream_t) cuda_proc("ggml_backend_cuda_custom_set_stream");

    ggml_tensor * sub = ggml_mul_mat(req.ctx, st->head, req.cur);   // [n_sub, n_tokens]
    ggml_set_name(sub, "draft_head_logits");
    ggml_tensor * srcs[1] = { sub };
    ggml_tensor * full = ggml_custom_4d(req.ctx, GGML_TYPE_F32, req.n_vocab, req.n_tokens, 1, 1, srcs, 1, scatter_cb, 1, &st->op);
    set_stream(full);
    ggml_set_name(full, "draft_head_full");
    return full;
}
