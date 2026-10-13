#include "hpcache-arena.h"

#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>

#if defined(_WIN32)
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace {

constexpr size_t kGranule = size_t(2) << 20;   // request size and layer alignment
constexpr size_t kSector  = 4096;              // unbuffered I/O alignment

size_t round_up(size_t v, size_t a) { return (v + a - 1) / a * a; }

struct tensor_info {
    std::string name;
    int         layer  = -1;
    int         shard  = 0;
    uint64_t    offset = 0;   // in the file
    size_t      nbytes = 0;
    size_t      cursor = 0;   // start of the (sector aligned) read in the arena
    size_t      head   = 0;   // offset % 4096
};

struct read_job {
    uint64_t  file_offset;
    uint8_t * dst;
    uint32_t  len;
};

// blk.N.ffn_*_exps.weight -> N, otherwise -1
int expert_layer(const std::string & name) {
    if (name.compare(0, 4, "blk.") != 0 || name.find("_exps.weight") == std::string::npos ||
            name.find(".ffn_") == std::string::npos) {
        return -1;
    }
    return atoi(name.c_str() + 4);
}

// "...-00001-of-00003.gguf" -> the paths of all shards, a single file -> itself
std::vector<std::string> shard_paths(const std::string & first) {
    std::vector<std::string> out;
    const size_t of = first.rfind("-of-");
    if (of != std::string::npos && of >= 6 && first.size() >= of + 4 + 5 + 5 && first.compare(first.size() - 5, 5, ".gguf") == 0) {
        const int count = atoi(first.c_str() + of + 4);
        const std::string prefix = first.substr(0, of - 5);
        const std::string tail = first.substr(of, first.size() - of);
        char num[16];
        for (int i = 1; i <= count; ++i) {
            snprintf(num, sizeof(num), "%05d", i);
            out.push_back(prefix + num + tail);
        }
    }
    if (out.empty()) {
        out.push_back(first);
    }
    return out;
}

#if defined(_WIN32)
bool read_jobs(const std::string & path, std::vector<read_job> & jobs, uint64_t file_size, std::string & error) {
    HANDLE file = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
            FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        error = "cannot open " + path + " for unbuffered reads";
        return false;
    }
    HANDLE port = CreateIoCompletionPort(file, nullptr, 0, 0);
    if (port == nullptr) {
        CloseHandle(file);
        error = "cannot create the completion port";
        return false;
    }
    const size_t depth = 4;
    std::vector<OVERLAPPED> ovs(depth);
    std::vector<size_t>     job_of(depth, 0);
    size_t next = 0;
    size_t in_flight = 0;
    bool   ok = true;

    auto issue = [&](size_t slot) -> bool {
        if (next >= jobs.size()) {
            return false;
        }
        const read_job & j = jobs[next];
        memset(&ovs[slot], 0, sizeof(OVERLAPPED));
        ovs[slot].Offset     = DWORD(j.file_offset & 0xffffffffu);
        ovs[slot].OffsetHigh = DWORD(j.file_offset >> 32);
        job_of[slot] = next++;
        DWORD got = 0;
        if (!ReadFile(file, j.dst, j.len, &got, &ovs[slot]) && GetLastError() != ERROR_IO_PENDING) {
            error = "read failed at offset " + std::to_string(j.file_offset) + " of " + path;
            ok = false;
            return false;
        }
        ++in_flight;
        return true;
    };
    for (size_t i = 0; i < depth && issue(i); ++i) {
    }
    while (in_flight > 0 && ok) {
        DWORD got = 0;
        ULONG_PTR key = 0;
        OVERLAPPED * done = nullptr;
        const BOOL r = GetQueuedCompletionStatus(port, &got, &key, &done, INFINITE);
        if (done == nullptr) {
            error = "completion port failed";
            ok = false;
            break;
        }
        --in_flight;
        const size_t slot = size_t(done - ovs.data());
        const read_job & j = jobs[job_of[slot]];
        const uint64_t want = std::min<uint64_t>(j.len, file_size > j.file_offset ? file_size - j.file_offset : 0);
        if (!r || got < want) {
            error = "short read at offset " + std::to_string(j.file_offset) + " of " + path;
            ok = false;
            break;
        }
        issue(slot);
    }
    // drain what is still in flight before the buffers are touched again
    while (in_flight > 0) {
        DWORD got = 0;
        ULONG_PTR key = 0;
        OVERLAPPED * done = nullptr;
        GetQueuedCompletionStatus(port, &got, &key, &done, INFINITE);
        --in_flight;
    }
    CloseHandle(port);
    CloseHandle(file);
    return ok;
}

uint8_t * region_alloc(size_t bytes) {
    return static_cast<uint8_t *>(VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
}

void region_free(uint8_t * p, size_t) {
    VirtualFree(p, 0, MEM_RELEASE);
}

uint64_t file_size_of(const std::string & path) {
    WIN32_FILE_ATTRIBUTE_DATA data;
    if (!GetFileAttributesExA(path.c_str(), GetFileExInfoStandard, &data)) {
        return 0;
    }
    return (uint64_t(data.nFileSizeHigh) << 32) | data.nFileSizeLow;
}
#else
bool read_jobs(const std::string & path, std::vector<read_job> & jobs, uint64_t file_size, std::string & error) {
#ifdef O_DIRECT
    int fd = open(path.c_str(), O_RDONLY | O_DIRECT);
    if (fd < 0) {
        fd = open(path.c_str(), O_RDONLY);
    }
#else
    int fd = open(path.c_str(), O_RDONLY);
#endif
    if (fd < 0) {
        error = "cannot open " + path;
        return false;
    }
    bool ok = true;
    for (const read_job & j : jobs) {
        const uint64_t want = std::min<uint64_t>(j.len, file_size > j.file_offset ? file_size - j.file_offset : 0);
        size_t done = 0;
        while (done < want) {
            const ssize_t n = pread(fd, j.dst + done, j.len - done, off_t(j.file_offset + done));
            if (n <= 0) {
                error = "short read at offset " + std::to_string(j.file_offset) + " of " + path;
                ok = false;
                break;
            }
            done += size_t(n);
        }
        if (!ok) {
            break;
        }
    }
    close(fd);
    return ok;
}

uint8_t * region_alloc(size_t bytes) {
    void * p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return p == MAP_FAILED ? nullptr : static_cast<uint8_t *>(p);
}

void region_free(uint8_t * p, size_t bytes) {
    munmap(p, bytes);
}

uint64_t file_size_of(const std::string & path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0 ? uint64_t(st.st_size) : 0;
}
#endif

}  // namespace

hpcache_arena::~hpcache_arena() {
    release();
}

void hpcache_arena::release() {
    if (pin) {
        for (auto & r : pinned) {
            pin(r.first, r.second, false);
        }
    }
    pinned.clear();
    n_pinned_bytes = 0;
    if (raw != nullptr) {
        region_free(raw, raw_bytes);
    }
    raw = base = nullptr;
    raw_bytes = region_bytes = 0;
    index.clear();
}

size_t hpcache_arena::max_tensor_bytes() const {
    size_t m = 0;
    for (const auto & kv : index) {
        m = std::max(m, kv.second.nbytes);
    }
    return m;
}

bool hpcache_arena::is_pinned(const void * p, size_t nbytes) const {
    const uint8_t * b = static_cast<const uint8_t *>(p);
    for (const auto & r : pinned) {
        if (b >= r.first && b + nbytes <= r.first + r.second) {
            return true;
        }
    }
    return false;
}

uint8_t * hpcache_arena::find(const std::string & name, size_t nbytes) const {
    auto it = index.find(name);
    if (it == index.end() || it->second.nbytes != nbytes) {
        return nullptr;
    }
    return it->second.data;
}

bool hpcache_arena::open(const std::string & first_shard, size_t max_bytes, size_t max_pinned, const pin_fn & pin_function,
        const log_fn & log, std::string & error) {
    release();
    pin = pin_function;
    const auto t0 = std::chrono::steady_clock::now();

    // 1. the expert tensors of every shard, from the GGUF headers
    const std::vector<std::string> shards = shard_paths(first_shard);
    std::vector<tensor_info> tensors;
    std::vector<uint64_t> sizes;
    for (size_t s = 0; s < shards.size(); ++s) {
        gguf_init_params params = { /*no_alloc =*/ true, /*ctx =*/ nullptr };
        gguf_context * g = gguf_init_from_file(shards[s].c_str(), params);
        if (g == nullptr) {
            error = "cannot read the GGUF header of " + shards[s];
            return false;
        }
        const size_t data_offset = gguf_get_data_offset(g);
        for (int64_t i = 0; i < gguf_get_n_tensors(g); ++i) {
            tensor_info t;
            t.name = gguf_get_tensor_name(g, i);
            t.layer = expert_layer(t.name);
            if (t.layer < 0) {
                continue;
            }
            t.shard  = int(s);
            t.offset = data_offset + gguf_get_tensor_offset(g, i);
            t.nbytes = gguf_get_tensor_size(g, i);
            tensors.push_back(std::move(t));
        }
        gguf_free(g);
        sizes.push_back(file_size_of(shards[s]));
    }
    if (tensors.empty()) {
        error = "no expert tensors in " + first_shard;
        return false;
    }

    // 2. layout: layers in order, 2 MiB aligned, tensors packed at sector granularity
    std::stable_sort(tensors.begin(), tensors.end(), [](const tensor_info & a, const tensor_info & b) {
        return a.layer != b.layer ? a.layer < b.layer : a.name < b.name;
    });
    struct range { size_t begin, end; };
    std::vector<range> layer_ranges;
    size_t cursor = 0;
    for (size_t i = 0; i < tensors.size();) {
        const int layer = tensors[i].layer;
        cursor = round_up(cursor, kGranule);
        const size_t begin = cursor;
        for (; i < tensors.size() && tensors[i].layer == layer; ++i) {
            tensor_info & t = tensors[i];
            t.head   = size_t(t.offset % kSector);
            t.cursor = cursor;
            cursor  += round_up(t.head + t.nbytes, kSector);
        }
        layer_ranges.push_back({ begin, cursor });
    }
    const size_t total = round_up(cursor, kGranule);
    if (total > max_bytes) {
        error = "the expert weights need " + std::to_string(total >> 20) + " MiB, the arena is limited to " +
                std::to_string(max_bytes >> 20) + " MiB";
        return false;
    }

    // 3. the region
    raw_bytes = total + kGranule;
    raw = region_alloc(raw_bytes);
    if (raw == nullptr) {
        raw_bytes = 0;
        error = "cannot allocate " + std::to_string(total >> 20) + " MiB";
        return false;
    }
    base = reinterpret_cast<uint8_t *>(round_up(reinterpret_cast<uintptr_t>(raw), kGranule));
    region_bytes = total;

    // 4. read, per shard, in file order, 2 MiB requests
    for (size_t s = 0; s < shards.size(); ++s) {
        std::vector<const tensor_info *> mine;
        for (const tensor_info & t : tensors) {
            if (size_t(t.shard) == s) {
                mine.push_back(&t);
            }
        }
        std::sort(mine.begin(), mine.end(), [](const tensor_info * a, const tensor_info * b) { return a->offset < b->offset; });
        std::vector<read_job> jobs;
        for (const tensor_info * t : mine) {
            const uint64_t start = t->offset - t->head;
            const size_t   span  = round_up(t->head + t->nbytes, kSector);
            for (size_t done = 0; done < span; done += kGranule) {
                jobs.push_back({ start + done, base + t->cursor + done, uint32_t(std::min(kGranule, span - done)) });
            }
        }
        if (!read_jobs(shards[s], jobs, sizes[s], error)) {
            release();
            return false;
        }
    }

    for (const tensor_info & t : tensors) {
        index[t.name] = { base + t.cursor + t.head, t.nbytes };
    }
    seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    // 5. page-lock one range per layer, in order, and stop at the first refusal
    if (pin) {
        for (const range & r : layer_ranges) {
            if (n_pinned_bytes + (r.end - r.begin) > max_pinned || !pin(base + r.begin, r.end - r.begin, true)) {
                break;
            }
            pinned.push_back({ base + r.begin, r.end - r.begin });
            n_pinned_bytes += r.end - r.begin;
        }
    }
    if (log) {
        char buf[256];
        snprintf(buf, sizeof(buf), "arena: %zu expert tensors, %.2f GiB read in %.1f s (%.2f GB/s), %.2f GiB page-locked in %zu of %zu layers",
                tensors.size(), double(total) / (1ull << 30), seconds, double(total) / 1e9 / std::max(seconds, 1e-9),
                double(n_pinned_bytes) / (1ull << 30), pinned.size(), layer_ranges.size());
        log(buf);
    }
    return true;
}
