#include "hpcache-metrics.h"

#ifdef HPCACHE_EVENT_LOG

#include <chrono>
#include <cstdlib>
#include <cstring>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

static const char * const event_names[HPE_COUNT] = {
#define X(id, name) name,
    HPCACHE_EVENTS(X)
#undef X
};

static const char * const clock_names[HPC_COUNT] = { "host", "lead", "g2" };

// 4 MiB per ring; the busiest produces about 25000 records per second, so the logger can be away for seconds
static const uint32_t ring_records = 1u << 17;
// the logger sleeps this long between sweeps, it must not take a core from the spinning workers
static const int sweep_ms = 20;

bool hpcache_metrics::open(const char * path) {
    out = path != nullptr && path[0] != 0 ? fopen(path, "w") : nullptr;
    if (out == nullptr) {
        return false;
    }
    setvbuf(out, nullptr, _IOFBF, 1 << 20);
    fprintf(out, "seq,layer,event,clock,ts_ns,arg\n");
    logger = std::thread([this] { run(); });
    return true;
}

hpcache_metric_ring * hpcache_metrics::add_ring(int clock, int device) {
    auto * r = new hpcache_metric_ring();
    r->clock  = clock;
    r->device = device;
    if (device >= 0) {
        if (!hpcache_cuda_metric_ring_alloc(*r, device, ring_records)) {
            delete r;
            return nullptr;
        }
    } else {
        r->rec  = static_cast<hpcache_metric_rec *>(calloc(ring_records, sizeof(hpcache_metric_rec)));
        r->mask = ring_records - 1;
    }
    rings.push_back(r);
    return r;
}

void hpcache_metrics::offset(int clock, long long off0, long long off1, long long host0, long long host1) {
    clocks[clock] = { off0, off1, host0, host1, true };
}

void hpcache_metrics::host_log(hpcache_metric_ring & r, uint16_t code, int32_t arg0, int32_t arg1, int32_t arg2) {
    hpcache_metric_rec & p = r.rec[r.head & r.mask];
    p.level = 0;
    p.ts    = (uint64_t) hpcache_metric_now();
    p.code  = code;
    p.arg0  = arg0;
    p.arg1  = arg1;
    p.arg2  = arg2;
    const uint32_t lap = (uint32_t) (r.head / (r.mask + 1)) + 1;
    reinterpret_cast<std::atomic<uint32_t> *>(const_cast<uint32_t *>(&p.seq))->store(lap, std::memory_order_release);
    r.head++;
}

size_t hpcache_metrics::drain(hpcache_metric_ring & r) {
    size_t n = 0;
    const uint64_t slots = (uint64_t) r.mask + 1;
    for (;;) {
        hpcache_metric_rec & p = r.rec[r.tail & r.mask];
        const uint32_t want = (uint32_t) (r.tail / slots) + 1;
        const uint32_t seq = reinterpret_cast<std::atomic<uint32_t> *>(const_cast<uint32_t *>(&p.seq))->load(std::memory_order_acquire);
        if (seq != want) {
            const int32_t ahead = (int32_t) (seq - want);
            if (ahead <= 0) {            // not written yet
                break;
            }
            r.lost += (uint64_t) ahead * slots;    // the producer lapped us
            r.tail += (uint64_t) ahead * slots;
            continue;
        }
        fprintf(out, "%d,%d,%s,%s,%llu,%d\n", p.arg0, p.arg1, event_names[p.code], clock_names[r.clock],
                (unsigned long long) p.ts, p.arg2);
        r.tail++;
        n++;
    }
    return n;
}

void hpcache_metrics::run() {
#ifdef _WIN32
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
#endif
    for (;;) {
        const bool last = stopping.load(std::memory_order_acquire);
        size_t n = 0;
        for (hpcache_metric_ring * r : rings) {
            n += drain(*r);
        }
        if (last && n == 0) {
            break;
        }
        if (n == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(sweep_ms));
        }
    }
}

void hpcache_metrics::close() {
    if (out == nullptr) {
        return;
    }
    stopping.store(true, std::memory_order_release);
    logger.join();
    // host clock minus the card clock at the first and at the second calibration, then the host times of the two
    for (int c = HPC_LEAD; c < HPC_COUNT; ++c) {
        if (clocks[c].set) {
            fprintf(out, "0,-1,offset,%s,%lld,%lld,%lld,%lld\n", clock_names[c], clocks[c].off0, clocks[c].off1,
                    clocks[c].host0, clocks[c].host1);
        }
    }
    uint64_t lost = 0;
    for (hpcache_metric_ring * r : rings) {
        lost += r->lost;
        if (r->device >= 0) {
            hpcache_cuda_metric_ring_free(*r);
        } else {
            free(r->rec);
        }
        delete r;
    }
    rings.clear();
    if (lost > 0) {
        fprintf(stderr, "moe-hpcache: metrics: %llu records were overwritten before the logger read them\n", (unsigned long long) lost);
    }
    fclose(out);
    out = nullptr;
}

#endif
