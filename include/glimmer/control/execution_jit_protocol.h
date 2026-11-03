#pragma once

#include "glimmer/control/execution_protocol.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace glimmer::control {

inline constexpr std::size_t kExecutionMaxJitOptions = 34;
inline constexpr std::size_t kExecutionMaxJitLogBytes = 65536;
inline constexpr std::size_t kExecutionMaxJitNames = 32;
// Includes the final NUL. Empty C strings occupy one byte.
inline constexpr std::size_t kExecutionMaxJitNameBytes = 1025;
inline constexpr std::size_t kExecutionMaxJitNamesBytes =
    4 + kExecutionMaxJitNames * (4 + kExecutionMaxJitNameBytes);
inline constexpr std::size_t kExecutionMaxJitWireBytes = 8 + kExecutionMaxJitOptions * 16 +
                                                         2 * kExecutionMaxJitLogBytes + 4 +
                                                         2 * kExecutionMaxJitNamesBytes;
struct ExecutionJitOption {
    std::uint32_t option{};
    // Integer slot bits only. Float low-slot bytes and log contents use
    // canonical zero plus owned data. Names use pointer-presence0/1 and a
    // u32 count plus length-prefixed NUL-terminated blobs, never host pointers.
    std::uint64_t value{};
    std::vector<std::byte> data;
    bool operator==(const ExecutionJitOption&) const = default;
};
enum class ExecutionJitStorage : std::uint8_t { kUnsupported, kScalar, kFloat, kLog, kNames };
[[nodiscard]] ExecutionJitStorage execution_jit_storage(std::uint32_t option) noexcept;
// JIT1: u32 magic/count followed by ordered u32 option/u64 scalar/blob records.
// Structural codec supports both input and SDK output. Pair capacities and exact
// request/output correspondence are checked separately, never inferred here.
[[nodiscard]] std::optional<std::vector<std::byte>> encode_execution_jit_options(
    std::span<const ExecutionJitOption> options) noexcept;
[[nodiscard]] std::optional<std::vector<ExecutionJitOption>> decode_execution_jit_options(
    std::span<const std::byte> bytes) noexcept;
[[nodiscard]] bool valid_execution_jit_input(std::span<const ExecutionJitOption> options) noexcept;
[[nodiscard]] bool execution_jit_output_matches(
    std::span<const ExecutionJitOption> input, std::span<const ExecutionJitOption> output) noexcept;

}  // namespace glimmer::control
