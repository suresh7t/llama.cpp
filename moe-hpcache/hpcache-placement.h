#pragma once

#include "hpcache.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <unordered_map>

class common_moe_hpcache_lru {
public:
    using evict_fn = std::function<bool(const common_moe_hpcache_expert_key &)>;

    explicit common_moe_hpcache_lru(size_t budget_bytes = 0);

    void set_budget(size_t budget_bytes);
    bool contains(common_moe_hpcache_expert_key key) const;
    bool ensure(common_moe_hpcache_expert_key key, size_t bytes, const evict_fn & evict);
    void touch(common_moe_hpcache_expert_key key);
    void pin(common_moe_hpcache_expert_key key);
    void unpin(common_moe_hpcache_expert_key key);

    size_t used_bytes() const;
    size_t budget_bytes() const;

private:
    struct entry {
        size_t bytes = 0;
        uint64_t tick = 0;
        uint32_t pins = 0;
    };

    size_t budget = 0;
    size_t used = 0;
    uint64_t clock = 0;
    std::unordered_map<common_moe_hpcache_expert_key, entry, common_moe_hpcache_key_hash> entries;
};
