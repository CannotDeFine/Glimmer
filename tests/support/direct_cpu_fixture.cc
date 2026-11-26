#include <pthread.h>
#include <time.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {
std::atomic<bool> finished{false};
std::uint64_t foreign_chunks{};

std::uint64_t clock_ns(clockid_t clock = CLOCK_MONOTONIC) {
    timespec value{};
    if (clock_gettime(clock, &value) != 0) {
        std::_Exit(81);
    }
    return static_cast<std::uint64_t>(value.tv_sec) * 1000000000 +
           static_cast<std::uint64_t>(value.tv_nsec);
}
}  // namespace

// Keep distinct symbols and frames: the profile must distinguish every phase.
extern "C" __attribute__((noinline)) std::uint64_t target_leaf(std::uint64_t seed) {
    asm volatile("" : "+r"(seed));
    for (int i = 0; i < 16384; ++i) {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
    }
    return seed;
}

extern "C" __attribute__((noinline)) std::uint64_t target_chain(std::uint64_t end) {
    std::uint64_t count = 0;
    while (clock_ns() < end) {
        if (target_leaf(1) != 3024703106425976043ULL) {
            std::_Exit(82);
        }
        ++count;
    }
    return count;
}

extern "C" __attribute__((noinline)) void before_chain(std::uint64_t end) {
    while (clock_ns() < end) {
        if (target_leaf(2) == 0) {
            std::_Exit(83);
        }
    }
}

extern "C" __attribute__((noinline)) void after_chain(std::uint64_t end) {
    while (clock_ns() < end) {
        if (target_leaf(3) == 0) {
            std::_Exit(83);
        }
    }
}

extern "C" __attribute__((noinline)) void* foreign_chain(void*) {
    while (!finished.load(std::memory_order_relaxed)) {
        if (target_leaf(4) == 0) {
            std::_Exit(83);
        }
        ++foreign_chunks;
    }
    return nullptr;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        return 80;
    }
    const auto begin = std::strtoull(argv[1], nullptr, 10);
    const auto end = std::strtoull(argv[2], nullptr, 10);
    pthread_t foreign{};
    if (end <= begin || pthread_create(&foreign, nullptr, foreign_chain, nullptr) != 0) {
        return 80;
    }
    before_chain(begin - 100000000);
    const auto cpu_begin = clock_ns(CLOCK_THREAD_CPUTIME_ID);
    const auto wall_begin = clock_ns();
    const auto count = target_chain(end + 100000000);
    const auto wall_end = clock_ns();
    const auto cpu_end = clock_ns(CLOCK_THREAD_CPUTIME_ID);
    after_chain(end + 300000000);
    finished.store(true, std::memory_order_relaxed);
    if (pthread_join(foreign, nullptr) != 0 || foreign_chunks == 0 || count == 0) {
        return 84;
    }
    std::printf("{\"chunks\":%llu,\"foreign_chunks\":%llu,\"cpu_ns\":%llu,\"wall_ns\":%llu}\n",
                static_cast<unsigned long long>(count),
                static_cast<unsigned long long>(foreign_chunks),
                static_cast<unsigned long long>(cpu_end - cpu_begin),
                static_cast<unsigned long long>(wall_end - wall_begin));
}
