#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <functional>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <utility>

// A sharded map with demand-loaded values. Keys remain indexed after eviction.
// Miss loading and eviction are serialized so the byte budget remains exact.
template<typename K, typename V, typename Hash = std::hash<K>>
class common_moe_hpcache_lazy_value_map {
public:
    using load_callback = std::function<bool(const K &, V &, size_t &)>;
    using evict_callback = std::function<bool(const K &, V &)>;
    using create_callback = std::function<V()>;

    explicit common_moe_hpcache_lazy_value_map(size_t budget_bytes = 0) : budget(budget_bytes) {}

    bool register_key(const K & key) {
        shard & bucket = buckets[shard_for(key)];
        std::lock_guard<std::mutex> lock(bucket.mutex);
        return bucket.entries.emplace(key, entry{}).second;
    }

    bool get_registered(const K & key, V & value) const {
        const shard & bucket = buckets[shard_for(key)];
        std::lock_guard<std::mutex> lock(bucket.mutex);
        auto it = bucket.entries.find(key);
        if (it == bucket.entries.end()) {
            return false;
        }
        value = it->second.value;
        return true;
    }

    V get_or_create(const K & key, const create_callback & create) {
        shard & bucket = buckets[shard_for(key)];
        std::lock_guard<std::mutex> lock(bucket.mutex);
        auto it = bucket.entries.find(key);
        if (it == bucket.entries.end()) {
            entry value;
            value.value = create();
            it = bucket.entries.emplace(key, std::move(value)).first;
        }
        return it->second.value;
    }

    bool get(const K & key, V & value) {
        shard & bucket = buckets[shard_for(key)];
        std::lock_guard<std::mutex> lock(bucket.mutex);
        auto it = bucket.entries.find(key);
        if (it == bucket.entries.end()) {
            return false;
        }
        it->second.tick = clock.fetch_add(1, std::memory_order_relaxed) + 1;
        if (!it->second.resident) {
            return false;
        }
        value = *it->second.resident;
        return true;
    }

    bool get_or_load(const K & key, V & value, const load_callback & load, const evict_callback & evict) {
        if (get(key, value)) {
            return true;
        }
        V loaded;
        size_t bytes = 0;
        if (!load(key, loaded, bytes)) {
            return false;
        }
        std::lock_guard<std::mutex> mutation_lock(mutation_mutex);
        if (get(key, value)) {
            return true;
        }
        const size_t limit = budget.load(std::memory_order_acquire);
        if (bytes > limit) {
            return false;
        }
        while (used.load(std::memory_order_acquire) + bytes > limit) {
            shard * victim_bucket = nullptr;
            K victim_key{};
            uint64_t victim_tick = 0;
            for (shard & candidate : buckets) {
                std::lock_guard<std::mutex> lock(candidate.mutex);
                for (const auto & item : candidate.entries) {
                    if (item.second.resident && item.second.pins == 0 && (victim_bucket == nullptr || item.second.tick < victim_tick)) {
                        victim_bucket = &candidate;
                        victim_key = item.first;
                        victim_tick = item.second.tick;
                    }
                }
            }
            if (victim_bucket == nullptr) {
                return false;
            }
            std::lock_guard<std::mutex> lock(victim_bucket->mutex);
            auto current = victim_bucket->entries.find(victim_key);
            if (current == victim_bucket->entries.end() || current->second.pins != 0) {
                continue;
            }
            if (!current->second.resident || !evict(current->first, *current->second.resident)) {
                return false;
            }
            used.fetch_sub(current->second.bytes, std::memory_order_release);
            current->second.resident.reset();
            current->second.bytes = 0;
        }
        shard & bucket = buckets[shard_for(key)];
        std::lock_guard<std::mutex> lock(bucket.mutex);
        auto inserted = bucket.entries.emplace(key, entry{});
        inserted.first->second.value = loaded;
        if (inserted.first->second.resident) {
            value = *inserted.first->second.resident;
            return true;
        }
        inserted.first->second.resident.emplace(std::move(loaded));
        inserted.first->second.bytes = bytes;
        inserted.first->second.tick = clock.fetch_add(1, std::memory_order_relaxed) + 1;
        used.fetch_add(bytes, std::memory_order_release);
        value = *inserted.first->second.resident;
        return true;
    }

    void set_budget(size_t budget_bytes) { budget.store(budget_bytes, std::memory_order_release); }
    void pin(const K & key) { shard & bucket = buckets[shard_for(key)]; std::lock_guard<std::mutex> lock(bucket.mutex); auto it = bucket.entries.find(key); if (it != bucket.entries.end()) ++it->second.pins; }
    void unpin(const K & key) { shard & bucket = buckets[shard_for(key)]; std::lock_guard<std::mutex> lock(bucket.mutex); auto it = bucket.entries.find(key); if (it != bucket.entries.end() && it->second.pins > 0) --it->second.pins; }
    size_t used_bytes() const { return used.load(std::memory_order_acquire); }
    size_t budget_bytes() const { return budget.load(std::memory_order_acquire); }
    size_t key_count() const {
        size_t count = 0;
        for (const shard & bucket : buckets) {
            std::lock_guard<std::mutex> lock(bucket.mutex);
            count += bucket.entries.size();
        }
        return count;
    }

    void for_each_registered(const std::function<void(const K &, V &)> & callback) {
        for (shard & bucket : buckets) {
            std::lock_guard<std::mutex> lock(bucket.mutex);
            for (auto & item : bucket.entries) {
                callback(item.first, item.second.value);
            }
        }
    }

private:
    struct entry {
        V value{};
        std::optional<V> resident;
        size_t bytes;
        uint64_t tick;
        uint32_t pins;
    };

    static constexpr size_t shard_count = 16;

    struct shard {
        mutable std::mutex mutex;
        std::unordered_map<K, entry, Hash> entries;
    };

    size_t shard_for(const K & key) const { return Hash{}(key) % shard_count; }

    std::atomic<size_t> budget{0};
    std::atomic<size_t> used{0};
    std::atomic<uint64_t> clock{0};
    std::array<shard, shard_count> buckets;
    mutable std::mutex mutation_mutex;
};
