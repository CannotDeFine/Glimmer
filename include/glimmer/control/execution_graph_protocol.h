#pragma once

#include "glimmer/control/execution_protocol.h"

namespace glimmer::control {

// The fixed underlying type and values are part of the wire protocol.
// NOLINTNEXTLINE(performance-enum-size)
enum class ExecutionGraphAction : std::uint32_t {
    kBegin = 1,
    kEnd = 2,
    kState = 3,
    kInstantiate = 4,
    kDestroyGraph = 5,
    kDestroyExecutable = 6,
    kCaptureKernel = 7,
    kCaptureMemory = 8,
    kInvalidate = 9,
};
struct ExecutionGraphRequest {
    ExecutionGraphAction action{ExecutionGraphAction::kState};
    bool has_output{};
    std::uint64_t object{}, stream{}, caller{}, flags{}, output_seed{};
    // Borrowed from the enclosing authenticated message; never a native pointer.
    std::span<const std::byte> payload{};
};
inline constexpr std::size_t kExecutionGraphRequestBytes = 60;
struct ExecutionGraphObservation {
    bool output_written{}, is_quarantined{};
    std::uint32_t capture_status{};
    std::uint64_t output{};
    bool operator==(const ExecutionGraphObservation&) const = default;
};
inline constexpr std::size_t kExecutionGraphObservationBytes = 24;
struct ExecutionGraphLaunch {
    std::uint64_t executable{}, stream{};
    bool operator==(const ExecutionGraphLaunch&) const = default;
};
inline constexpr std::size_t kExecutionGraphLaunchBytes = 16;

// Transactional allocation-bounded codecs. They validate carrier shape, not SDK
// handles, capture authority, native flags or physical completion. Readers leave
// the destination/cursor unchanged on failure; callers check enclosing extent.
[[nodiscard]] bool write_execution_graph_request(ExecutionPayloadWriter& output,
                                                 const ExecutionGraphRequest& value) noexcept;
[[nodiscard]] bool read_execution_graph_request(ExecutionPayloadReader& input,
                                                ExecutionGraphRequest& value) noexcept;
[[nodiscard]] bool write_execution_graph_observation(
    ExecutionPayloadWriter& output, const ExecutionGraphObservation& value) noexcept;
[[nodiscard]] bool read_execution_graph_observation(ExecutionPayloadReader& input,
                                                    ExecutionGraphObservation& value) noexcept;
[[nodiscard]] bool write_execution_graph_launch(ExecutionPayloadWriter& output,
                                                ExecutionGraphLaunch value) noexcept;
[[nodiscard]] bool read_execution_graph_launch(ExecutionPayloadReader& input,
                                               ExecutionGraphLaunch& value) noexcept;

}  // namespace glimmer::control
