#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#ifdef HPCACHE_EVENT_LOG
#include "hpcache-metrics.h"
// the log's argument of a kernel, nothing when the log is not built
#define HPCACHE_LOG_PARAM(x) , x
#define HPCACHE_LOG_ARG(x) , x
#else
#define HPCACHE_LOG_PARAM(x)
#define HPCACHE_LOG_ARG(x)
#endif

// pinned host memory that the GPU and the CPU follower share
struct hpcache_doorbell {
    int n_embd = 0;
    int k      = 0;
    int max_tokens = 1;                   // tokens of one window that the arrays below hold

    // host side
    float *             x     = nullptr;   // activation, [n_embd] per token
    int32_t *           ids   = nullptr;   // routed experts, [k] per token
    float *             w     = nullptr;   // router weights, [k] per token
    float *             y     = nullptr;   // CPU result, [n_embd] per token
    volatile int32_t *  layer = nullptr;   // layer of the last publish
    volatile int32_t *  n_tok = nullptr;   // tokens of the last publish
    volatile uint32_t * seq   = nullptr;   // bumped by the GPU after each publish
    volatile uint32_t * flag  = nullptr;   // set by the host to the seq it served
    volatile int32_t *  frac  = nullptr;   // GPU share of the missed experts, 0..256, written by the host between steps
    volatile unsigned long long * wait_ns = nullptr;   // total time the GPU waited for the follower, written by the GPU
    // second card: its worker answers each ring with its own flag and result, written by the worker only
    volatile uint32_t * flag2 = nullptr;   // set by the second card's worker to the seq it served
    float *             y2    = nullptr;   // result of the experts on the second card, [n_embd]
    bool                two   = false;     // the merge waits for flag2 as well
    // written by the merge kernel on the GPU: [0] ns until the CPU answer was seen, [1] ns until the second card's, summed over
    // layers (both counted from the start of the wait), [2] layers where the CPU answered last, [3] where the second card did
    volatile unsigned long long * stats = nullptr;

    // the same memory, as the device sees it
    float *             d_x     = nullptr;
    int32_t *           d_ids   = nullptr;
    float *             d_w     = nullptr;
    float *             d_y     = nullptr;
    volatile int32_t *  d_layer = nullptr;
    volatile int32_t *  d_n_tok = nullptr;
    volatile uint32_t * d_seq   = nullptr;
    volatile uint32_t * d_flag  = nullptr;
    volatile int32_t *  d_frac  = nullptr;
    volatile unsigned long long * d_wait_ns = nullptr;
    volatile uint32_t * d_flag2 = nullptr;
    float *             d_y2    = nullptr;
    volatile unsigned long long * d_stats = nullptr;

    // The split of the missed experts of a layer between the CPU and the PCIe path: the number of experts for the PCIe path
    // for [the second card has experts of the layer][pool hits][misses], see hpcache_gtab_index. Both sides read it, so
    // they pick the same experts. The kernels get d_gtab only when the table is in use, else the split follows the share frac.
    volatile int8_t * gtab   = nullptr;
    int8_t *          d_gtab = nullptr;

#ifdef HPCACHE_EVENT_LOG
    // event log of the first card, see hpcache-metrics.h; rec is NULL until the log is started
    hpcache_metric_dev lead_m;
#endif

    void * block = nullptr;
};

// The last op of the graph of the second card: copies the result into the pinned memory that the first card reads and
// sets the flag, so the worker thread does not wait for the card or copy the result back. seq is the sequence number
// of the request, in the input block of the graph.
void hpcache_cuda_g2_finish(void * stream, const hpcache_doorbell & db, const float * out, const volatile uint32_t * seq);

// a copy of host memory in the memory of a card, NULL on failure (the split table: a kernel that read it from the pinned
// memory waited behind the PCIe copy of the experts for every block)
int8_t * hpcache_cuda_upload(int device, const void * host, size_t bytes);

static constexpr int hpcache_gtab_size = 2 * 33 * 33;
static inline int hpcache_gtab_index(bool two, int hits, int misses) { return ((two ? 1 : 0) * 33 + hits) * 33 + misses; }

bool hpcache_doorbell_init(hpcache_doorbell & db, int n_embd, int k, int max_tokens, std::string & error);
void hpcache_doorbell_free(hpcache_doorbell & db);

// stream is the cudaStream_t of the ggml CUDA backend
// copy the layer inputs of n_tokens tokens to host memory, then ring; the strides are the distances between the rows
// of the tokens, in elements
void hpcache_cuda_publish(void * stream, const hpcache_doorbell & db,
        const float * x, const int32_t * ids, const float * w, int layer, int n_tokens, int x_stride, int ids_stride, int w_stride);
// wait until the host served the last ring, then copy the CPU result of n_tokens tokens to dst
void hpcache_cuda_wait_merge(void * stream, const hpcache_doorbell & db, float * dst, int n_tokens);

int  hpcache_cuda_get_device();
void hpcache_cuda_set_device(int device);

void hpcache_cuda_sync_device();

// Keeps a light kernel running on a card so that the driver does not take it for idle and lower its clocks.
// A card that is busy for a quarter of every layer drops to a third of its clock, and the drop makes every kernel on it
// three times as long. Returns a handle for hpcache_cuda_keepalive_stop, or nullptr.
void * hpcache_cuda_keepalive_start(int device);
void   hpcache_cuda_keepalive_stop(void * handle);


// slot table of one layer: expert id -> pool slot or -1, in device memory
int32_t * hpcache_cuda_table_alloc(int n_expert);
void      hpcache_cuda_table_free(int32_t * table);
bool      hpcache_cuda_table_upload(int32_t * table, const int32_t * host, int n_expert);

// The slot tables of all layers live in one device array (table i at d + i * n_expert) with a pinned host mirror, so a
// round of admission changes them with one copy instead of one per layer.
struct hpcache_table_block {
    int32_t * d = nullptr;
    int32_t * h = nullptr;
    int       n_tables = 0;
    int       n_expert = 0;
};
bool hpcache_cuda_tables_alloc(hpcache_table_block & b, int n_tables, int n_expert);
void hpcache_cuda_tables_free(hpcache_table_block & b);
// copy the first n_tables tables of the mirror to the device
bool hpcache_cuda_tables_upload(const hpcache_table_block & b, int n_tables);

// queued host -> device copy on a private stream, hpcache_cuda_copy_wait returns when all have landed
bool hpcache_cuda_copy_to_device(void * dst, const void * src, size_t nbytes);
bool hpcache_cuda_copy_wait();

// Swap copies done by a kernel instead of the copy engine. The kernel reads the page-locked, mapped arena through PCIe,
// so the host -> device copy engine stays free and the small input copies of other work (the MTP head right after a
// window) do not queue behind tens of MB of swaps. It runs on a least-priority stream with a small grid, so it takes
// few SMs from other kernels. src is the device address of mapped host memory; src, dst and bytes are 16-byte aligned.
struct hpcache_swap_copy {
    const void * src;
    void *       dst;
    size_t       bytes;
};
// launches one kernel for all copies on the device's swap stream; false when it could not, then nothing was launched
bool hpcache_cuda_swap_copy(int device, const hpcache_swap_copy * copies, int n);
// true when the last launched copies have landed
bool hpcache_cuda_swap_done();
// returns when the last launched copies have landed
bool hpcache_cuda_swap_wait();

// device ids: pool slot of each routed expert, 0 for a miss
void hpcache_cuda_remap_ids(void * stream, const int32_t * ids, const int32_t * table, int k, int32_t miss, int32_t * out);
// device weights: the router weight of a hit, 0 for a miss
void hpcache_cuda_remap_weights(void * stream, const int32_t * ids, const float * w, const int32_t * table, int k, float * out);

// CUDA device index of a PCI bus id such as "0000:01:00.0", -1 when unknown
int hpcache_cuda_device_by_pci(const char * pci_bus_id);

// The GPU's share of the missed experts of one token: of the routed experts that are not in the pool, in routing
// order, the last g, g = min(m, (misses * frac256) >> 8). out_ids / out_w hold m entries, the unused ones repeat an
// expert with weight 0. Either output may be NULL. table may be NULL (no pool: every expert is a miss).
void hpcache_cuda_pcie_select(void * stream, const int32_t * ids, const float * w, const int32_t * table,
        int k, int n_expert, int m, const volatile int32_t * frac256, const volatile int8_t * gtab, int32_t * out_ids, float * out_w,
        bool skip_unused);   // skip_unused: id -1 for an unused slot (the quantized matvec skips it), else the slot index with weight 0

// Experts that the GPU fetches itself: the stage kernel copies the experts selected by the rule above from the
// page-locked arena (read through PCIe) into device staging slots, one slot per selected expert.
struct hpcache_stage_args {
    int          n_tensors = 0;      // 3 (gate, up, down) or fewer
    const void * src[4]    = {};     // arena device address of expert 0 of each tensor
    size_t       bytes[4]  = {};     // bytes of one expert, also the distance between experts
    void *       dst[4]    = {};     // staging slot 0 of each tensor
};
// With side the copy runs on a stream of its own that forks from `stream` at this point, so what is queued on `stream` after
// it computes while the copy runs; hpcache_cuda_pcie_stage_join makes `stream` wait for the copy.
void hpcache_cuda_pcie_stage(void * stream, const int32_t * ids, const int32_t * table, int k, int n_expert, int m,
        const volatile int32_t * frac256, const volatile int8_t * gtab, const hpcache_stage_args & args, bool side,
        bool zero_unused);   // false only together with skip_unused in hpcache_cuda_pcie_select
void hpcache_cuda_pcie_stage_join(void * stream);

// The GPU's share of a window of several tokens. One set of experts is fetched for the whole window, each expert once,
// and every token that routed to one of them computes it on the GPU. The CPU follower and the stage kernel run the same
// function on the same ids and the same pool table, so they agree on the set. The lane model is the one of fill_gtab.
#ifdef __CUDACC__
#define HPCACHE_HD __host__ __device__
#else
#define HPCACHE_HD
#endif
static constexpr int hpcache_window_distinct = 256;   // distinct missed experts that the selection looks at
// lane model of a window, microseconds: first card fixed / per pool hit / per staged expert, CPU fixed / per distinct expert
struct hpcache_win_model {
    float g0 = 74.0f, gh = 9.0f, gs = 150.0f, c0 = 17.0f, cc = 83.0f;
};

// hit(e) tells whether expert e is in the pool. Returns the number of selected experts, they are written to sel[].
template <class Hit>
HPCACHE_HD static inline int hpcache_window_select(const int32_t * ids, int stride, int n_tok, int k, int n_expert,
        Hit hit, int cap, const hpcache_win_model & mdl, int32_t * sel) {
    int32_t ex[hpcache_window_distinct];
    int     cnt[hpcache_window_distinct];
    int nd = 0;
    int hits = 0;
    for (int t = 0; t < n_tok; ++t) {
        for (int i = 0; i < k && i < 32; ++i) {
            const int32_t e = ids[t * stride + i];
            if (e < 0 || e >= n_expert) {
                continue;
            }
            if (hit(e)) {
                ++hits;
                continue;
            }
            int j = 0;
            while (j < nd && ex[j] != e) {
                ++j;
            }
            if (j < nd) {
                ++cnt[j];
            } else if (nd < hpcache_window_distinct) {
                ex[nd] = e;
                cnt[nd] = 1;
                ++nd;
            }
        }
    }
    int g_max = nd < cap ? nd : cap;
    int best_g = 0;
    float best = 1e30f;
    for (int g = 0; g <= g_max; ++g) {
        const float t3 = mdl.g0 + mdl.gh * (float) hits + mdl.gs * (float) g;   // first card
        const float tc = mdl.c0 + mdl.cc * (float) (nd - g);                    // CPU
        const float t  = t3 > tc ? t3 : tc;
        if (t < best) {
            best = t;
            best_g = g;
        }
    }
    // the experts that most tokens share first, a tie goes to the first one seen
    for (int j = 0; j < best_g; ++j) {
        int at = 0;
        for (int i = 1; i < nd; ++i) {
            if (cnt[i] > cnt[at]) {
                at = i;
            }
        }
        sel[j] = ex[at];
        cnt[at] = -1;
    }
    return best_g;
}

// Stage the experts of hpcache_window_select (at most cap of them) and write the table expert -> staging slot (-1 for an
// expert that is not staged) into stage_table, n_expert entries. ids has a row of k per token, stride apart.
// zero_unused: zero the slots past the staged ones. Needed only when the expert ops compute an unstaged expert as a slot with
// weight 0 (miss id 0); with miss id -1 the matvec skips it and the slots are never read, so the zeroing (about 4.6 slots of
// 3 tensors per layer, most of the grid) is skipped.
void hpcache_cuda_pcie_stage_window(void * stream, const int32_t * ids, int stride, int n_tok, const int32_t * table, int k,
        int n_expert, int cap, const hpcache_win_model & mdl, int32_t * stage_table, const hpcache_stage_args & args, bool side,
        bool zero_unused);

// Logits over a subset of the vocabulary to logits over all of it: out[t * n_vocab + ids[j]] = sub[t * n_sub + j], every other entry
// -INFINITY. sub is [n_sub, n_tokens] and out [n_vocab, n_tokens], both F32 and contiguous, ids are n_sub distinct token ids.
void hpcache_cuda_scatter_logits(void * stream, const float * sub, const int32_t * ids, int n_sub, int n_vocab, int n_tokens, float * out);

// the device address of a registered host pointer, NULL when it is not registered for mapping
void * hpcache_cuda_host_device_ptr(const void * ptr);

// page-lock (pin = true) or release a host range for the current device, false on refusal
bool hpcache_cuda_host_register(void * ptr, size_t nbytes, bool pin);
