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

    // the same memory, as the device sees it
    float *             d_x     = nullptr;
    int32_t *           d_ids   = nullptr;
    float *             d_w     = nullptr;
    float *             d_y     = nullptr;
    volatile int32_t *  d_layer = nullptr;
    volatile uint32_t * d_seq   = nullptr;
    volatile uint32_t * d_flag  = nullptr;

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

// nonblocking, makes the driver submit queued work
void hpcache_cuda_flush(void * stream);
int  hpcache_cuda_get_device();
void hpcache_cuda_set_device(int device);
void hpcache_cuda_sync_device();

// slot table of one layer: expert id -> pool slot or -1, in device memory
int32_t * hpcache_cuda_table_alloc(int n_expert);
void      hpcache_cuda_table_free(int32_t * table);
bool      hpcache_cuda_table_upload(int32_t * table, const int32_t * host, int n_expert);

// queued host -> device copy on a private stream, hpcache_cuda_copy_wait returns when all have landed
bool hpcache_cuda_copy_to_device(void * dst, const void * src, size_t nbytes);
bool hpcache_cuda_copy_wait();

// device ids: pool slot of each routed expert, 0 for a miss
void hpcache_cuda_remap_ids(void * stream, const int32_t * ids, const int32_t * table, int k, int32_t * out);
// device weights: the router weight of a hit, 0 for a miss
void hpcache_cuda_remap_weights(void * stream, const int32_t * ids, const float * w, const int32_t * table, int k, float * out);

// CUDA device index of a PCI bus id such as "0000:01:00.0", -1 when unknown
int hpcache_cuda_device_by_pci(const char * pci_bus_id);
