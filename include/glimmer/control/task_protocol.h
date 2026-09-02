#pragma once

#include "glimmer/control/task_admission.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace glimmer::control {

inline constexpr std::string_view kTaskProtocolVersion = "GLIMMER_TASK_V1";
inline constexpr std::size_t kTaskProtocolMaxLineBytes = 256;
inline constexpr std::size_t kTaskProtocolMaxTenantIdBytes = 96;

enum class TaskProtocolOperation : std::uint8_t {
    kSubmit,
    // Submit a task and claim it immediately when a scheduler slot is free.
    // A queued task is returned as ACCEPTED and can be claimed by task id.
    kAcquire,
    kCancel,
    kQuery,
    kClaim,
    // Wait for a task-specific claim without client-side polling. The
    // request carries a bounded wait timeout in milliseconds.
    kWait,
    kHeartbeat,
    kComplete,
    kFail,
    kStats,
    kMetrics,
};

enum class TaskProtocolParseError : std::uint8_t {
    kNone,
    kEmpty,
    kLineTooLong,
    kUnsupportedVersion,
    kMalformed,
    kInvalidTenant,
    kInvalidValue,
    kInternalError,
};

struct TaskProtocolRequest {
    TaskProtocolOperation operation = TaskProtocolOperation::kSubmit;
    TaskAdmissionRequest admission;
    core::TaskId task_id = 0;
    std::uint32_t wait_timeout_ms = 0;
};

struct TaskProtocolRequestParseResult {
    TaskProtocolParseError error = TaskProtocolParseError::kInternalError;
    std::optional<TaskProtocolRequest> request;

    [[nodiscard]] bool parsed() const noexcept {
        return error == TaskProtocolParseError::kNone && request.has_value();
    }
};

enum class TaskProtocolState : std::uint8_t {
    kQueued,
    kRunning,
    kCompleted,
    kCancelled,
    kFailed,
};

enum class TaskProtocolErrorCode : std::uint8_t {
    kInvalidRequest,
    kQuotaExceeded,
    kQueueFull,
    kUnknownTask,
    kInternalError,
    kUnsupportedVersion,
};

enum class TaskProtocolResponseKind : std::uint8_t {
    kAccepted,
    kState,
    kError,
    kLease,
    kEmpty,
    kStats,
    kMetrics,
};

struct TaskProtocolStats {
    std::uint64_t total_task_count = 0;
    std::uint64_t queued_task_count = 0;
    std::uint64_t running_task_count = 0;
    std::uint64_t completed_task_count = 0;
    std::uint64_t cancelled_task_count = 0;
    std::uint64_t failed_task_count = 0;
    core::MemoryBytes quota_limit_bytes = 0;
    core::MemoryBytes quota_reserved_bytes = 0;
    core::MemoryBytes quota_allocated_bytes = 0;
    std::uint64_t max_running_tasks = 0;
    std::uint64_t max_queued_tasks = 0;
    std::uint64_t total_queue_wait_microseconds = 0;
    std::uint64_t max_queue_wait_microseconds = 0;
    std::uint64_t total_service_time_microseconds = 0;
    std::uint64_t max_service_time_microseconds = 0;
};

struct TaskProtocolResponse {
    TaskProtocolResponseKind kind = TaskProtocolResponseKind::kError;
    core::TaskId task_id = 0;
    TaskProtocolState state = TaskProtocolState::kQueued;
    TaskProtocolErrorCode error = TaskProtocolErrorCode::kInternalError;
    std::string tenant_id;
    core::MemoryBytes memory_bytes = 0;
    std::uint32_t weight = 0;
    std::uint32_t work_units = 0;
    TaskProtocolStats stats;
};

struct TaskProtocolResponseParseResult {
    TaskProtocolParseError error = TaskProtocolParseError::kInternalError;
    std::optional<TaskProtocolResponse> response;

    [[nodiscard]] bool parsed() const noexcept {
        return error == TaskProtocolParseError::kNone && response.has_value();
    }
};

[[nodiscard]] TaskProtocolRequestParseResult parse_task_protocol_request(
    std::string_view line) noexcept;

[[nodiscard]] std::optional<std::string> format_task_protocol_request(
    const TaskProtocolRequest& request);

[[nodiscard]] TaskProtocolResponseParseResult parse_task_protocol_response(
    std::string_view line) noexcept;

[[nodiscard]] std::optional<std::string> format_task_protocol_response(
    const TaskProtocolResponse& response);

}  // namespace glimmer::control
