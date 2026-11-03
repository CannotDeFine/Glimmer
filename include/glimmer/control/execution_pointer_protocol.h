#pragma once

#include "glimmer/control/execution_protocol.h"

#include <cstddef>
#include <cstdint>

namespace glimmer::control {

inline constexpr std::size_t kExecutionPointerQueryBytes = 48;
inline constexpr std::size_t kExecutionPointerObservationBytes = 20;

// Fixed-width enum types and values are part of the wire protocol.
// NOLINTBEGIN(performance-enum-size)
enum class ExecutionPointerDomain : std::uint32_t { kDriver = 1, kRuntime = 2 };
enum class ExecutionPointerKind : std::uint32_t {
    kNull = 0,
    kUnregistered = 1,
    kDevice = 2,
    kHost = 3,
    // source_session names a process carrier, not the allocating CUDA context.
    kProcessDevice = 4,
};
enum class ExecutionPointerAlias : std::uint32_t { kNull = 0, kInput = 1 };
// NOLINTEND(performance-enum-size)

struct ExecutionPointerQuery {
    ExecutionPointerDomain domain{ExecutionPointerDomain::kDriver};
    std::uint32_t selector{3};
    ExecutionPointerKind kind{ExecutionPointerKind::kNull};
    std::uint64_t query_session{};
    std::uint64_t source_session{};
    std::uint64_t allocation{};
    std::uint64_t offset{};
    bool operator==(const ExecutionPointerQuery&) const = default;
};

struct ExecutionPointerObservation {
    // Independent type, device, device-pointer and host-pointer effects: 1/2/4/8.
    // Active scalars are raw signed bits, including native-error partial writes.
    // Success tuple restrictions belong to the domain-aware receipt validator.
    std::uint32_t effects{};
    std::int32_t type{};
    std::int32_t device{};
    ExecutionPointerAlias device_alias{ExecutionPointerAlias::kNull};
    ExecutionPointerAlias host_alias{ExecutionPointerAlias::kNull};
    bool operator==(const ExecutionPointerObservation&) const = default;
};

// Canonical, explicit little-endian values only. No caller/native pointer bits.
// Failure preserves the complete writer prefix or reader cursor/destination.
// Readers permit enclosing composition; the caller validates trailing bytes,
// domain-specific effects, ownership, native-size conversion and success tuples.
[[nodiscard]] bool write_execution_pointer_query(ExecutionPayloadWriter& output,
                                                 const ExecutionPointerQuery& value) noexcept;
[[nodiscard]] bool read_execution_pointer_query(ExecutionPayloadReader& input,
                                                ExecutionPointerQuery& value) noexcept;
[[nodiscard]] bool write_execution_pointer_observation(
    ExecutionPayloadWriter& output, const ExecutionPointerObservation& value) noexcept;
[[nodiscard]] bool read_execution_pointer_observation(ExecutionPayloadReader& input,
                                                      ExecutionPointerObservation& value) noexcept;

}  // namespace glimmer::control
