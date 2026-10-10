#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

class common_moe_hpcache_io {
public:
    common_moe_hpcache_io() = default;
    ~common_moe_hpcache_io();

    common_moe_hpcache_io(const common_moe_hpcache_io &) = delete;
    common_moe_hpcache_io & operator=(const common_moe_hpcache_io &) = delete;

    void open(const std::string & path);
    void read(uint64_t offset, void * dst, size_t size);

    uint64_t size() const;
    size_t alignment() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
