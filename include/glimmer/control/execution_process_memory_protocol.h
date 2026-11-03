#pragma once

#include "glimmer/control/execution_protocol.h"

#include <cstddef>
#include <cstdint>

namespace glimmer::control {

inline constexpr std::size_t kExecutionProcessMemoryRequestBytes = 48;
inline constexpr std::size_t kExecutionProcessMemoryObservationBytes = 8;
inline constexpr std::uint64_t kExecutionProcessAllocationTag = std::uint64_t{1} << 63;
[[nodiscard]] constexpr bool execution_is_process_allocation(std::uint64_t id) noexcept {
    return (id & kExecutionProcessAllocationTag) && (id & ~kExecutionProcessAllocationTag);
}

// The fixed underlying type and values are part of the wire protocol.
// NOLINTNEXTLINE(performance-enum-size)
enum class ExecutionProcessMemoryOperation : std::uint32_t {
    kGetDefaultPool = 1,
    kGetCurrentPool = 2,
    kSetCurrentPool = 3,
    kAllocateAsync = 4,
    kAllocateFromPoolAsync = 5,
    kTrimPool = 6,
};
struct ExecutionProcessMemoryRequest {
    ExecutionProcessMemoryOperation operation{ExecutionProcessMemoryOperation::kGetDefaultPool};
    std::int32_t device{};
    bool has_output{};
    std::uint64_t pool{}, bytes{}, stream{}, output_seed{};
    bool operator==(const ExecutionProcessMemoryRequest&) const = default;
};
struct ExecutionProcessMemoryObservation {
    // An observed write is independent of SDK success, physical completion and
    // usable resource identity. Exact output bits and status live in WireReply.
    bool output_written{};
    bool operator==(const ExecutionProcessMemoryObservation&) const = default;
};

// Canonical, versioned, little-endian records. Failure preserves the writer
// prefix or the complete reader cursor/destination. These codecs grant no
// resource authority and do not validate native device/pool/stream arguments.
// Composed readers leave trailing-byte rejection to the enclosing request.
[[nodiscard]] bool write_execution_process_memory_request(
    ExecutionPayloadWriter& output, const ExecutionProcessMemoryRequest& value) noexcept;
[[nodiscard]] bool read_execution_process_memory_request(
    ExecutionPayloadReader& input, ExecutionProcessMemoryRequest& value) noexcept;
[[nodiscard]] bool write_execution_process_memory_observation(
    ExecutionPayloadWriter& output, ExecutionProcessMemoryObservation value) noexcept;
[[nodiscard]] bool read_execution_process_memory_observation(
    ExecutionPayloadReader& input, ExecutionProcessMemoryObservation& value) noexcept;

}  // namespace glimmer::control
