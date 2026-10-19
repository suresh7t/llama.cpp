#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

// pinned host memory that the GPU and the CPU follower share
struct hpcache_doorbell {
    int n_embd = 0;
    int k      = 0;

    // host side
    float *             x     = nullptr;   // activation, [n_embd]
    int32_t *           ids   = nullptr;   // routed experts, [k]
    float *             w     = nullptr;   // router weights, [k]
    float *             y     = nullptr;   // CPU result, [n_embd]
    volatile int32_t *  layer = nullptr;   // layer of the last publish
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
    volatile uint32_t * d_seq   = nullptr;
    volatile uint32_t * d_flag  = nullptr;
    volatile int32_t *  d_frac  = nullptr;
    volatile unsigned long long * d_wait_ns = nullptr;
    volatile uint32_t * d_flag2 = nullptr;
    float *             d_y2    = nullptr;
    volatile unsigned long long * d_stats = nullptr;

    void * block = nullptr;
};

bool hpcache_doorbell_init(hpcache_doorbell & db, int n_embd, int k, std::string & error);
void hpcache_doorbell_free(hpcache_doorbell & db);

// stream is the cudaStream_t of the ggml CUDA backend
// copy the layer inputs to host memory, then ring
void hpcache_cuda_publish(void * stream, const hpcache_doorbell & db,
        const float * x, const int32_t * ids, const float * w, int layer);
// wait until the host served the last ring, then copy the CPU result to dst
void hpcache_cuda_wait_merge(void * stream, const hpcache_doorbell & db, float * dst);

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
        int k, int n_expert, int m, const volatile int32_t * frac256, int32_t * out_ids, float * out_w);

// Experts that the GPU fetches itself: the stage kernel copies the experts selected by the rule above from the
// page-locked arena (read through PCIe) into device staging slots, one slot per selected expert.
struct hpcache_stage_args {
    int          n_tensors = 0;      // 3 (gate, up, down) or fewer
    const void * src[4]    = {};     // arena device address of expert 0 of each tensor
    size_t       bytes[4]  = {};     // bytes of one expert, also the distance between experts
    void *       dst[4]    = {};     // staging slot 0 of each tensor
};
void hpcache_cuda_pcie_stage(void * stream, const int32_t * ids, const int32_t * table, int k, int n_expert, int m,
        const volatile int32_t * frac256, const hpcache_stage_args & args);

// the device address of a registered host pointer, NULL when it is not registered for mapping
void * hpcache_cuda_host_device_ptr(const void * ptr);

// page-lock (pin = true) or release a host range for the current device, false on refusal
bool hpcache_cuda_host_register(void * ptr, size_t nbytes, bool pin);
