#pragma once

#include <cstdint>

namespace glimmer::test::gemm_completion {
inline constexpr unsigned int kWorkers = 4, kThreads = 32, kGuard = 32;
inline constexpr std::uint32_t kSentinel = 0x91ae73c5;
struct alignas(64) Shared {
    std::uint32_t release{}, completed{};
    std::uint32_t output[kWorkers * kThreads + 2 * kGuard]{};
    std::uint32_t publishers[kWorkers]{}, done[kWorkers]{};
};
}  // namespace glimmer::test::gemm_completion
