#pragma once

#ifdef HPCACHE_EVENT_LOG

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

// Event log for benchmarks, on when HPCACHE_STEPS=<csv path> is set.
//
// Every producer, a CUDA stream or a host thread, owns a ring of fixed-size records. A producer fills the next record and
// writes its ticket last; it never waits and never takes a lock. The rings of the cards are page-locked and mapped, so a
// kernel writes into them over PCIe and the logger thread reads them from host memory. The logger thread drains the rings
// into a csv, steps.py analyses it. One writer per ring and one reader, no atomics are needed besides the ticket.

#define HPCACHE_EVENTS(X) \
    X(PUBLISH_START,       "publish:start")       /* the router is done, the publish kernel starts */ \
    X(PUBLISH_STOP,        "publish:stop")        /* the doorbell is rung */ \
    X(OWN_EXPERTS_STOP,    "own_experts:stop")    /* the first card's experts are done */ \
    X(WAIT_START,          "wait:start")          /* the wait kernel starts */ \
    X(WAIT_STOP,           "wait:stop")           /* both answers are in */ \
    X(LAYER_STOP,          "layer:stop")          /* the MoE output of the layer is complete */ \
    X(ROUTER_START,        "router:start")        /* the layer's dense work is done, the router starts */ \
    X(LAYER_IN,            "layer_in:ready")      /* the layer's output is ready, the next layer starts */ \
    X(GRAPH_START,         "graph:start")         /* the first node of the next token's graph ran */ \
    X(CPU_START,           "cpu_experts:start")   /* the CPU follower sees the ring */ \
    X(CPU_STOP,            "cpu_experts:stop")    /* it writes its answer, arg = experts it computed */ \
    X(G2_WORKER_START,     "g2_worker:start")     /* the worker of the second card sees the ring */ \
    X(G2_UPLOAD_START,     "g2_upload:start")     /* it starts the upload */ \
    X(G2_GRAPH_START,      "g2_graph:start")      /* the input has landed on the second card */ \
    X(G2_GRAPH_STOP,       "g2_graph:stop")       /* its graph is done */ \
    X(G2_DOWNLOAD_STOP,    "g2_download:stop")    /* the worker has the result back */ \
    X(G2_FINISH,           "g2_finish:stop")      /* the card has written the result and the flag to host memory */ \
    X(G2_WORKER_STOP,      "g2_worker:stop")      /* it writes its answer, arg = experts it computed */ \
    X(BUILD,               "build:il0")           /* the graph of a token is built */ \
    X(BEGIN_ENTER,         "begin:enter")         /* decode_begin */ \
    X(BEGIN_ARMED,         "begin:armed") \
    X(END_ENTER,           "end:enter")           /* decode_end */ \
    X(END_SYNCED,          "end:synced")          /* the device is idle */ \
    X(END_FOLLOWERS_IDLE,  "end:followers_idle") \
    X(END_POOL_OFF,        "end:pool_off")        /* the CPU pool is parked */ \
    X(END_ADMIT_START,     "end:admit_start")     /* admit() is about to run */ \
    X(END_EXIT,            "end:exit")

enum hpcache_event : uint16_t {
#define X(id, name) HPE_##id,
    HPCACHE_EVENTS(X)
#undef X
    HPE_COUNT
};

// the clock of a ring: host events use the steady clock, a card its globaltimer
enum hpcache_clock : int { HPC_HOST = 0, HPC_LEAD = 1, HPC_G2 = 2, HPC_COUNT };

// One slot, 32 bytes, generic: a code (the event) and three arguments. The log of the doorbell uses arg0 = sequence number
// of the ring, arg1 = layer, arg2 = a count or 0. level is free.
struct hpcache_metric_rec {
    volatile uint32_t seq;           // lap of the ring + 1, written last
    uint32_t level;
    uint64_t ts;                     // ns on the clock of the ring
    uint32_t code;
    int32_t  arg0, arg1, arg2;
};

// what a kernel needs to write into a ring
struct hpcache_metric_dev {
    hpcache_metric_rec *        rec  = nullptr;   // NULL: no log
    volatile unsigned long long * head = nullptr; // number of records written, in device memory
    uint32_t                    mask = 0;
};

struct hpcache_metric_ring {
    hpcache_metric_rec * rec = nullptr;           // host address
    hpcache_metric_rec * d_rec = nullptr;         // device address, device rings only
    unsigned long long * d_head = nullptr;
    uint32_t mask = 0;
    int      clock = HPC_HOST;
    int      device = -1;
    uint64_t head = 0;                            // host rings: slots written
    uint64_t tail = 0;                            // logger: slots read
    uint64_t lost = 0;                            // overwritten before the logger read them

    hpcache_metric_dev view() const { return { d_rec, d_head, mask }; }
};

static inline long long hpcache_metric_now() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// implemented in hpcache-cuda.cu
// a ring of n records (a power of two) in page-locked memory, with its counter in the memory of the device
bool hpcache_cuda_metric_ring_alloc(hpcache_metric_ring & r, int device, uint32_t n);
void hpcache_cuda_metric_ring_free(hpcache_metric_ring & r);
// queues a device_log on the stream: code, arg0 = the sequence number read from seq when the kernel runs, arg1 = layer
void hpcache_cuda_device_log(void * stream, const hpcache_metric_ring & r, uint16_t code, const volatile uint32_t * seq, int layer);
// host clock minus the globaltimer of a card, by a ping-pong through pinned memory; rtt is the best round trip
bool hpcache_cuda_clock_offset(int device, long long & off_ns, long long & rtt_ns);

class hpcache_metrics {
public:
    ~hpcache_metrics() { close(); }

    bool on() const { return out != nullptr; }
    bool open(const char * path);
    // the rings are created before the producers start; device >= 0 makes a ring of that card
    hpcache_metric_ring * add_ring(int clock, int device);
    // host clock minus the card clock at two moments, to correct the drift
    void offset(int clock, long long off0, long long off1, long long host0, long long host1);
    void close();

    // a host producer, one thread per ring. The same slot as device_log in hpcache-cuda.cu.
    static void host_log(hpcache_metric_ring & r, uint16_t code, int32_t arg0 = 0, int32_t arg1 = 0, int32_t arg2 = 0);

private:
    void run();
    size_t drain(hpcache_metric_ring & r);

    FILE * out = nullptr;
    std::vector<hpcache_metric_ring *> rings;
    std::thread logger;
    std::atomic<bool> stopping { false };
    struct offsets { long long off0 = 0, off1 = 0, host0 = 0, host1 = 0; bool set = false; } clocks[HPC_COUNT];
};

#endif
