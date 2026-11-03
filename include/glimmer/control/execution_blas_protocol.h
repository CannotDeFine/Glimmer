#pragma once

#include "glimmer/control/execution_protocol.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace glimmer::control {

inline constexpr std::size_t kExecutionGeamBytes = 160;
// The fixed underlying type and values are part of the wire protocol.
// NOLINTNEXTLINE(performance-enum-size)
enum class ExecutionGeamType : std::uint32_t { kFloat = 1, kDouble = 2 };
struct ExecutionGeamBuffer {
    std::uint64_t allocation{}, offset{};
};
struct ExecutionGeam {
    ExecutionGeamType type{};
    std::uint32_t transpose_a{}, transpose_b{};
    std::int32_t rows{}, columns{}, leading_a{}, leading_b{}, leading_c{};
    std::uint64_t handle{};
    std::uint32_t pointer_mode{}, presence{};
    // A/B/C/alpha/beta. Presence bits are alpha=1, beta=2, A=4, B=8, C=16.
    // A present zero identity is unresolved, never a native address or lease.
    std::array<ExecutionGeamBuffer, 5> buffers{};
    // HOST mode preserves exactly 4/8 bytes; unused tails and DEVICE slots are zero.
    std::array<std::byte, 16> alpha{}, beta{};
};

// Exact original cuBLAS entries. Both scalar ABIs are FP32 regardless of the
// SgemmEx matrix types; neither packet conveys compute-replacement authority.
inline constexpr std::size_t kExecutionSgemvBytes = 152;
inline constexpr std::size_t kExecutionSgemmExBytes = 172;
struct ExecutionSgemv {
    std::uint32_t transpose{};
    std::int32_t rows{}, columns{}, leading_a{}, increment_x{}, increment_y{};
    std::uint64_t handle{};
    std::uint32_t pointer_mode{}, presence{};
    // A/x/y/alpha/beta; presence bits are 4/8/16/1/2 respectively.
    std::array<ExecutionGeamBuffer, 5> buffers{};
    std::array<std::byte, 16> alpha{}, beta{};
};
struct ExecutionSgemmEx {
    std::uint32_t transpose_a{}, transpose_b{}, type_a{}, type_b{}, type_c{};
    std::int32_t rows{}, columns{}, depth{}, leading_a{}, leading_b{}, leading_c{};
    std::uint64_t handle{};
    std::uint32_t pointer_mode{}, presence{};
    // A/B/C/alpha/beta; presence bits are 4/8/16/1/2 respectively.
    std::array<ExecutionGeamBuffer, 5> buffers{};
    std::array<std::byte, 16> alpha{}, beta{};
};

// Failure preserves the writer prefix or the reader cursor and destination.
[[nodiscard]] bool write_execution_sgemv(ExecutionPayloadWriter& output,
                                         const ExecutionSgemv& value) noexcept;
[[nodiscard]] bool read_execution_sgemv(ExecutionPayloadReader& input,
                                        ExecutionSgemv& value) noexcept;
[[nodiscard]] bool write_execution_sgemm_ex(ExecutionPayloadWriter& output,
                                            const ExecutionSgemmEx& value) noexcept;
[[nodiscard]] bool read_execution_sgemm_ex(ExecutionPayloadReader& input,
                                           ExecutionSgemmEx& value) noexcept;

[[nodiscard]] bool write_execution_geam(ExecutionPayloadWriter& output,
                                        const ExecutionGeam& value) noexcept;
// Failure leaves both the input cursor and destination unchanged. Enclosing
// messages independently reject trailing data; this codec permits composition.
[[nodiscard]] bool read_execution_geam(ExecutionPayloadReader& input,
                                       ExecutionGeam& value) noexcept;

}  // namespace glimmer::control
