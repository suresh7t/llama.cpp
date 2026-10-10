#pragma once

#include "hpcache-placement.h"
#include "hpcache-lazy-map.h"

#include <cstddef>
#include <cstdint>
#include <vector>

struct common_moe_hpcache_routed_expert {
    common_moe_hpcache_expert_key key;
    float router_weight = 0.0f;
    bool gpu_resident = false;
    bool cpu_resident = false;
};

struct common_moe_hpcache_dispatch {
    std::vector<common_moe_hpcache_routed_expert> gpu;
    std::vector<common_moe_hpcache_routed_expert> cpu;
};

class common_moe_hpcache_scheduler {
public:
    common_moe_hpcache_dispatch route(const std::vector<common_moe_hpcache_routed_expert> & selected) const;
};
