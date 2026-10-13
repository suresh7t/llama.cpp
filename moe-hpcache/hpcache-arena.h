#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

// One 2 MiB aligned host region that holds the routed expert tensors of a GGUF model, read from the file with
// unbuffered 2 MiB requests, so no page cache copy of the weights exists. The region is page-locked in one range
// per layer, which makes copies to the GPU asynchronous and lets the GPU read experts directly.
//
// Layout: layers in order, 2 MiB aligned; inside a layer the expert tensors back to back at 4 KiB granularity, each
// at the same offset modulo 4 KiB as in the file. A tensor never spans two layer ranges.
class hpcache_arena {
public:
    // pin(ptr, bytes, true) page-locks a range, pin(ptr, bytes, false) releases it; returns false on failure
    using pin_fn = std::function<bool(void *, size_t, bool)>;
    using log_fn = std::function<void(const std::string &)>;

    hpcache_arena() = default;
    ~hpcache_arena();
    hpcache_arena(const hpcache_arena &) = delete;
    hpcache_arena & operator=(const hpcache_arena &) = delete;

    // first_shard: the "-00001-of-0000N.gguf" path (or a single file). max_bytes caps the region, pin may be empty.
    bool open(const std::string & first_shard, size_t max_bytes, size_t max_pinned, const pin_fn & pin, const log_fn & log, std::string & error);

    // the bytes of a tensor in the arena, nullptr when unknown or when nbytes does not match the file
    uint8_t * find(const std::string & name, size_t nbytes) const;

    // true when [p, p + nbytes) lies in a page-locked layer range
    bool is_pinned(const void * p, size_t nbytes) const;

    size_t max_tensor_bytes() const;

    size_t total_bytes() const  { return region_bytes; }
    size_t pinned_bytes() const { return n_pinned_bytes; }
    size_t n_tensors() const    { return index.size(); }
    double load_seconds() const { return seconds; }

private:
    struct entry {
        uint8_t * data   = nullptr;
        size_t    nbytes = 0;
    };

    void release();

    uint8_t *                              raw = nullptr;       // as allocated
    uint8_t *                              base = nullptr;      // 2 MiB aligned
    size_t                                 raw_bytes = 0;
    size_t                                 region_bytes = 0;
    size_t                                 n_pinned_bytes = 0;
    double                                 seconds = 0.0;
    pin_fn                                 pin;
    std::vector<std::pair<uint8_t *, size_t>> pinned;           // ranges to release
    std::unordered_map<std::string, entry> index;
};
