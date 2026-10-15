#pragma once

// Small synchronization primitives on the address-wait facility of the OS: WaitOnAddress on Windows,
// futex on Linux. A waiter sleeps in the kernel until a 32-bit word changes, a waker does not need a mutex.

#include <atomic>
#include <cstdint>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#elif defined(__linux__)
#include <climits>
#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>
#else
#include <chrono>
#include <thread>
#endif

#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#define HPCACHE_SYNC_PAUSE() _mm_pause()
#else
#define HPCACHE_SYNC_PAUSE() std::atomic_signal_fence(std::memory_order_seq_cst)
#endif

namespace hpcache_sync {

static_assert(sizeof(std::atomic<uint32_t>) == sizeof(uint32_t), "atomic<uint32_t> must be a plain word");

// Sleep while *word == expected. Returns on a wake, a changed value, or spuriously: callers re-check.
inline void wait_word(std::atomic<uint32_t> & word, uint32_t expected) {
#if defined(_WIN32)
    WaitOnAddress(reinterpret_cast<volatile VOID *>(&word), &expected, sizeof(expected), INFINITE);
#elif defined(__linux__)
    syscall(SYS_futex, reinterpret_cast<uint32_t *>(&word), FUTEX_WAIT_PRIVATE, expected, nullptr, nullptr, 0);
#else
    if (word.load(std::memory_order_acquire) == expected) {
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
#endif
}

inline void wake_one(std::atomic<uint32_t> & word) {
#if defined(_WIN32)
    WakeByAddressSingle(&word);
#elif defined(__linux__)
    syscall(SYS_futex, reinterpret_cast<uint32_t *>(&word), FUTEX_WAKE_PRIVATE, 1, nullptr, nullptr, 0);
#else
    (void) word;
#endif
}

inline void wake_all(std::atomic<uint32_t> & word) {
#if defined(_WIN32)
    WakeByAddressAll(&word);
#elif defined(__linux__)
    syscall(SYS_futex, reinterpret_cast<uint32_t *>(&word), FUTEX_WAKE_PRIVATE, INT_MAX, nullptr, nullptr, 0);
#else
    (void) word;
#endif
}

// Wait until a condition holds. The waker changes the condition first, then calls notify().
// wait_until() reads the epoch before it checks the condition, so a notify in between is not lost.
class signal {
public:
    template <typename Pred>
    void wait_until(Pred && pred, int spins = 0) {
        for (int i = 0; i < spins; ++i) {
            if (pred()) {
                return;
            }
            HPCACHE_SYNC_PAUSE();
        }
        for (;;) {
            const uint32_t seen = epoch.load(std::memory_order_acquire);
            if (pred()) {
                return;
            }
            wait_word(epoch, seen);
        }
    }

    void notify() {
        epoch.fetch_add(1, std::memory_order_seq_cst);
        wake_all(epoch);
    }

private:
    std::atomic<uint32_t> epoch{0};
};

// Mutual exclusion for very short sections: spins a little, then sleeps on the word.
// 0 = free, 1 = held, 2 = held and somebody sleeps.
class spinlock {
public:
    void lock() {
        uint32_t expected = 0;
        for (int i = 0; i < 128; ++i) {
            if (state.load(std::memory_order_relaxed) == 0 &&
                    state.compare_exchange_weak(expected, 1, std::memory_order_acquire, std::memory_order_relaxed)) {
                return;
            }
            expected = 0;
            HPCACHE_SYNC_PAUSE();
        }
        while (state.exchange(2, std::memory_order_acquire) != 0) {
            wait_word(state, 2);
        }
    }

    void unlock() {
        if (state.exchange(0, std::memory_order_release) == 2) {
            wake_one(state);
        }
    }

private:
    std::atomic<uint32_t> state{0};
};

class spin_guard {
public:
    explicit spin_guard(spinlock & l) : lk(l) { lk.lock(); }
    ~spin_guard() { lk.unlock(); }
    spin_guard(const spin_guard &) = delete;
    spin_guard & operator=(const spin_guard &) = delete;

private:
    spinlock & lk;
};

} // namespace hpcache_sync
