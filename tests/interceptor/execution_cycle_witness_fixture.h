#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

// Fixed SDK-only diagnostic layout, not an execution-service protocol.
namespace cycle_witness {
inline constexpr std::uint32_t kBlocks = 320, kThreads = 128, kWarp = 32, kSamples = 1024;
inline constexpr std::uint32_t kWarps = kBlocks * (kThreads / kWarp);
inline constexpr std::uint32_t kRecords = kWarps * kSamples;
inline constexpr std::uint32_t kValues = kBlocks * kThreads;
inline constexpr std::uint32_t kGuard = 0xA5A5A5A5U, kGuardRecords = 4;
struct Sample {
    std::uint64_t before_ns, cycle, after_ns;
    std::uint32_t sm_before, sm_after, identity, value, group, reserved;
};
static_assert(sizeof(Sample) == 48 && std::is_trivially_copyable_v<Sample>);
inline constexpr std::uint32_t kGuardBytes = kGuardRecords * sizeof(Sample);
inline constexpr std::uint32_t kGuardValues = kGuardBytes / sizeof(std::uint32_t);
inline constexpr std::size_t kTraceBytes =
    (std::size_t{kRecords} + 2 * kGuardRecords) * sizeof(Sample);
inline constexpr std::size_t kValueBytes =
    (std::size_t{kValues} + 2 * kGuardValues) * sizeof(std::uint32_t);
static_assert(kTraceBytes + kValueBytes < 64 * 1024 * 1024);
inline constexpr std::uint32_t kMultiplier = 1664525U, kIncrement = 1013904223U;
inline constexpr std::uint32_t kSeed = 0x12345678U, kGroupSeed = 0x10203U;
}  // namespace cycle_witness
