#pragma once

#include "glimmer/control/task_protocol.h"

namespace glimmer::control {

// A separate C++ ABI and wire envelope for generation-bound execution.
// Decoding does not authenticate, authorize, retry, or prove physical completion.
inline constexpr std::string_view kTaskExecutionProtocolVersion = "GLIMMER_TASK_EXEC_V1";

enum class TaskExecutionProtocolOperation : std::uint8_t {
    kSubmit = 0,
    // Submit a task and claim it immediately when a scheduler slot is free.
    // A queued task is returned as ACCEPTED and can be claimed by task id.
    kAcquire = 1,
    kCancel = 2,
    kQuery = 3,
    kClaim = 4,
    // Wait for a task-specific claim without client-side polling. The
    // request carries a bounded wait timeout in milliseconds.
    kWait = 5,
    kHeartbeat = 6,
    kComplete = 7,
    kFail = 8,
    kStats = 9,
    kMetrics = 10,
    kExecutionClaim = 11,
    kRetreat = 12,
    kSuspend = 13,
    kExecutionHeartbeat = 14,
    kExecutionComplete = 15,
    kExecutionFail = 16,
    kExecutionCancel = 17,
    // Submit cooperative work and attempt its first generation-bound claim.
    kExecutionAcquire = 18,
    // Complete one proven-stopped execution, then admit distinct new work.
    kExecutionCompleteAcquire = 19,
};

struct TaskExecutionProtocolRequest {
    TaskExecutionProtocolOperation operation = TaskExecutionProtocolOperation::kSubmit;
    TaskAdmissionRequest admission;
    core::TaskId task_id = 0;
    std::uint32_t wait_timeout_ms = 0;
    std::uint64_t generation = 0;
};

struct TaskExecutionProtocolRequestParseResult {
    TaskProtocolParseError error = TaskProtocolParseError::kInternalError;
    std::optional<TaskExecutionProtocolRequest> request;

    [[nodiscard]] bool parsed() const noexcept {
        return error == TaskProtocolParseError::kNone && request.has_value();
    }
};

enum class TaskExecutionProtocolResponseKind : std::uint8_t {
    kAccepted = 0,
    kState = 1,
    kError = 2,
    kLease = 3,
    kEmpty = 4,
    kStats = 5,
    kMetrics = 6,
    kExecutionLease = 7,
    kRetreat = 8,
    kExecutionHandoff = 9,
};

struct TaskExecutionProtocolResponse {
    TaskExecutionProtocolResponseKind kind = TaskExecutionProtocolResponseKind::kError;
    core::TaskId task_id = 0;
    TaskProtocolState state = TaskProtocolState::kQueued;
    TaskProtocolErrorCode error = TaskProtocolErrorCode::kInternalError;
    std::string tenant_id;
    core::MemoryBytes memory_bytes = 0;
    std::uint32_t weight = 0;
    std::uint32_t work_units = 0;
    TaskProtocolStats stats;
    std::uint64_t generation = 0;
    bool should_retreat = false;
    // Only EXEC_HANDOFF carries the independently acknowledged old identity.
    TaskExecutionIdentity completed_execution{};
};

struct TaskExecutionProtocolResponseParseResult {
    TaskProtocolParseError error = TaskProtocolParseError::kInternalError;
    std::optional<TaskExecutionProtocolResponse> response;

    [[nodiscard]] bool parsed() const noexcept {
        return error == TaskProtocolParseError::kNone && response.has_value();
    }
};

[[nodiscard]] TaskExecutionProtocolRequestParseResult parse_task_execution_protocol_request(
    std::string_view line) noexcept;
[[nodiscard]] std::optional<std::string> format_task_execution_protocol_request(
    const TaskExecutionProtocolRequest& request);
[[nodiscard]] TaskExecutionProtocolResponseParseResult parse_task_execution_protocol_response(
    std::string_view line) noexcept;
[[nodiscard]] std::optional<std::string> format_task_execution_protocol_response(
    const TaskExecutionProtocolResponse& response);

}  // namespace glimmer::control
