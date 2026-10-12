#include "hpcache-io.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <malloc.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#endif

static constexpr size_t HPCACHE_DIO_PAGE = 8 * 1024;
static constexpr size_t HPCACHE_DIO_BATCH = 16 * HPCACHE_DIO_PAGE;

struct common_moe_hpcache_io::impl {
    std::string path;
    uint64_t file_size = 0;

#ifdef _WIN32
    HANDLE handle = INVALID_HANDLE_VALUE;
#else
    int fd = -1;
#endif

    ~impl() {
#ifdef _WIN32
        if (handle != INVALID_HANDLE_VALUE) {
            CloseHandle(handle);
        }
#else
        if (fd >= 0) {
            close(fd);
        }
#endif
    }
};

common_moe_hpcache_io::common_moe_hpcache_io() = default;
common_moe_hpcache_io::~common_moe_hpcache_io() = default;

void common_moe_hpcache_io::open(const std::string & path) {
    auto next = std::make_unique<impl>();
    next->path = path;
#ifdef _WIN32
    next->handle = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                               FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
    if (next->handle == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("HP-cache direct I/O open failed: " + path);
    }
    LARGE_INTEGER size;
    if (!GetFileSizeEx(next->handle, &size)) {
        throw std::runtime_error("HP-cache direct I/O size query failed: " + path);
    }
    next->file_size = static_cast<uint64_t>(size.QuadPart);
#else
    next->fd = ::open(path.c_str(), O_RDONLY | O_DIRECT);
    if (next->fd < 0) {
        throw std::runtime_error("HP-cache direct I/O open failed: " + path + ": " + std::strerror(errno));
    }
    struct stat st {};
    if (fstat(next->fd, &st) != 0) {
        throw std::runtime_error("HP-cache direct I/O size query failed: " + path);
    }
    next->file_size = static_cast<uint64_t>(st.st_size);
#endif
    pimpl = std::move(next);
}

void common_moe_hpcache_io::read(uint64_t offset, void * dst, size_t size) {
    if (!pimpl || offset > pimpl->file_size || size > pimpl->file_size - offset) {
        throw std::runtime_error("HP-cache direct I/O read is outside the file");
    }
    auto * out = static_cast<uint8_t *>(dst);
    while (size > 0) {
        const size_t part = (std::min)(size, HPCACHE_DIO_BATCH);
        const uint64_t aligned_offset = offset & ~(uint64_t(HPCACHE_DIO_PAGE) - 1);
        const size_t skip = static_cast<size_t>(offset - aligned_offset);
        const size_t physical_size = (skip + part + HPCACHE_DIO_PAGE - 1) & ~(HPCACHE_DIO_PAGE - 1);
        void * raw = nullptr;
#ifdef _WIN32
        raw = _aligned_malloc(physical_size, HPCACHE_DIO_PAGE);
        if (!raw) {
            throw std::runtime_error("HP-cache direct I/O buffer allocation failed");
        }
        LARGE_INTEGER pos;
        pos.QuadPart = aligned_offset;
        if (!SetFilePointerEx(pimpl->handle, pos, nullptr, FILE_BEGIN)) {
            _aligned_free(raw);
            throw std::runtime_error("HP-cache direct I/O seek failed");
        }
        const size_t available = (std::min<uint64_t>)(physical_size, pimpl->file_size - aligned_offset);
        DWORD got = 0;
        const bool ok = available == 0 || (ReadFile(pimpl->handle, raw, static_cast<DWORD>(available), &got, nullptr) && got == available);
        if (!ok) {
            _aligned_free(raw);
            throw std::runtime_error("HP-cache direct I/O read failed");
        }
        if (available < physical_size) {
            std::memset(static_cast<uint8_t *>(raw) + available, 0, physical_size - available);
        }
#else
        if (posix_memalign(&raw, HPCACHE_DIO_PAGE, physical_size) != 0) {
            throw std::runtime_error("HP-cache direct I/O buffer allocation failed");
        }
        const ssize_t got = pread(pimpl->fd, raw, physical_size, static_cast<off_t>(aligned_offset));
        if (got < 0 || static_cast<size_t>(got) != physical_size) {
            free(raw);
            throw std::runtime_error("HP-cache direct I/O read failed");
        }
#endif
        std::memcpy(out, static_cast<uint8_t *>(raw) + skip, part);
#ifdef _WIN32
        _aligned_free(raw);
#else
        free(raw);
#endif
        offset += part;
        out += part;
        size -= part;
    }
}

uint64_t common_moe_hpcache_io::size() const {
    return pimpl ? pimpl->file_size : 0;
}

size_t common_moe_hpcache_io::alignment() const {
    return HPCACHE_DIO_PAGE;
}
