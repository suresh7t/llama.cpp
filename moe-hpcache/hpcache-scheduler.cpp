#include "hpcache-scheduler.h"

common_moe_hpcache_dispatch common_moe_hpcache_scheduler::route(
        const std::vector<common_moe_hpcache_routed_expert> & selected) const {
    common_moe_hpcache_dispatch result;
    result.gpu.reserve(selected.size());
    result.cpu.reserve(selected.size());
    for (const auto & expert : selected) {
        if (expert.gpu_resident) {
            result.gpu.push_back(expert);
        } else {
            // GPU misses execute on CPU. The CPU miss refill is handled by the LRU and I/O layer.
            result.cpu.push_back(expert);
        }
    }
    return result;
}
