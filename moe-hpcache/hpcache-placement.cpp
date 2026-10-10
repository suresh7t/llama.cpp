#include "hpcache-placement.h"

common_moe_hpcache_lru::common_moe_hpcache_lru(size_t budget_bytes) : budget(budget_bytes) {}

void common_moe_hpcache_lru::set_budget(size_t budget_bytes) {
    budget = budget_bytes;
}

bool common_moe_hpcache_lru::contains(common_moe_hpcache_expert_key key) const {
    return entries.find(key) != entries.end();
}

bool common_moe_hpcache_lru::ensure(common_moe_hpcache_expert_key key, size_t bytes, const evict_fn & evict) {
    if (bytes > budget) {
        return false;
    }
    auto found = entries.find(key);
    if (found != entries.end()) {
        found->second.tick = ++clock;
        return true;
    }
    while (used + bytes > budget) {
        auto victim = entries.end();
        for (auto it = entries.begin(); it != entries.end(); ++it) {
            if (it->second.pins == 0 && (victim == entries.end() || it->second.tick < victim->second.tick)) {
                victim = it;
            }
        }
        if (victim == entries.end() || !evict(victim->first)) {
            return false;
        }
        used -= victim->second.bytes;
        entries.erase(victim);
    }
    entries.emplace(key, entry{bytes, ++clock, 0});
    used += bytes;
    return true;
}

void common_moe_hpcache_lru::touch(common_moe_hpcache_expert_key key) {
    auto it = entries.find(key);
    if (it != entries.end()) {
        it->second.tick = ++clock;
    }
}

void common_moe_hpcache_lru::pin(common_moe_hpcache_expert_key key) {
    auto it = entries.find(key);
    if (it != entries.end()) {
        ++it->second.pins;
    }
}

void common_moe_hpcache_lru::unpin(common_moe_hpcache_expert_key key) {
    auto it = entries.find(key);
    if (it != entries.end() && it->second.pins > 0) {
        --it->second.pins;
    }
}

size_t common_moe_hpcache_lru::used_bytes() const {
    return used;
}

size_t common_moe_hpcache_lru::budget_bytes() const {
    return budget;
}
