#include "glimmer/control/task_endpoint.h"
#include "glimmer/control/task_protocol.h"

#include "glimmer/core/scheduler.h"

namespace glimmer::control {
namespace {

[[nodiscard]] TaskProtocolResponse error_response(TaskProtocolErrorCode error) noexcept {
    return {.kind = TaskProtocolResponseKind::kError,
            .task_id = 0,
            .state = TaskProtocolState::kQueued,
            .error = error,
            .tenant_id = {},
            .memory_bytes = 0,
            .weight = 0,
            .work_units = 0,
            .stats = {}};
}

[[nodiscard]] TaskProtocolState protocol_state(core::TaskState state) noexcept {
    switch (state) {
        case core::TaskState::kQueued:
            return TaskProtocolState::kQueued;
        case core::TaskState::kRunning:
            return TaskProtocolState::kRunning;
        case core::TaskState::kCompleted:
            return TaskProtocolState::kCompleted;
        case core::TaskState::kCancelled:
            return TaskProtocolState::kCancelled;
        case core::TaskState::kFailed:
            return TaskProtocolState::kFailed;
    }
    return TaskProtocolState::kFailed;
}

[[nodiscard]] TaskProtocolResponse state_response(core::TaskId task_id,
                                                  core::TaskState state) noexcept {
    return {.kind = TaskProtocolResponseKind::kState,
            .task_id = task_id,
            .state = protocol_state(state),
            .error = TaskProtocolErrorCode::kInternalError,
            .tenant_id = {},
            .memory_bytes = 0,
            .weight = 0,
            .work_units = 0,
            .stats = {}};
}

[[nodiscard]] TaskProtocolResponse lease_response(const core::TaskSnapshot& snapshot) {
    return {.kind = TaskProtocolResponseKind::kLease,
            .task_id = snapshot.task_id,
            .state = TaskProtocolState::kRunning,
            .error = TaskProtocolErrorCode::kInternalError,
            .tenant_id = snapshot.tenant_id,
            .memory_bytes = snapshot.memory_bytes,
            .weight = snapshot.weight,
            .work_units = snapshot.work_units,
            .stats = {}};
}

[[nodiscard]] TaskProtocolResponse accepted_response(core::TaskId task_id) noexcept {
    return {.kind = TaskProtocolResponseKind::kAccepted,
            .task_id = task_id,
            .state = TaskProtocolState::kQueued,
            .error = TaskProtocolErrorCode::kInternalError,
            .tenant_id = {},
            .memory_bytes = 0,
            .weight = 0,
            .work_units = 0,
            .stats = {}};
}

[[nodiscard]] TaskProtocolResponse stats_response(const core::SchedulerStats& stats,
                                                  bool include_metrics) {
    return {
        .kind =
            include_metrics ? TaskProtocolResponseKind::kMetrics : TaskProtocolResponseKind::kStats,
        .task_id = 0,
        .state = TaskProtocolState::kQueued,
        .error = TaskProtocolErrorCode::kInternalError,
        .tenant_id = {},
        .memory_bytes = 0,
        .weight = 0,
        .work_units = 0,
        .stats = {.total_task_count = static_cast<std::uint64_t>(stats.total_task_count),
                  .queued_task_count = static_cast<std::uint64_t>(stats.queued_task_count),
                  .running_task_count = static_cast<std::uint64_t>(stats.running_task_count),
                  .completed_task_count = static_cast<std::uint64_t>(stats.completed_task_count),
                  .cancelled_task_count = static_cast<std::uint64_t>(stats.cancelled_task_count),
                  .failed_task_count = static_cast<std::uint64_t>(stats.failed_task_count),
                  .quota_limit_bytes = stats.quota.limit_bytes,
                  .quota_reserved_bytes = stats.quota.reserved_bytes,
                  .quota_allocated_bytes = stats.quota.allocated_bytes,
                  .max_running_tasks = static_cast<std::uint64_t>(stats.max_running_tasks),
                  .max_queued_tasks = static_cast<std::uint64_t>(stats.max_queued_tasks),
                  .total_queue_wait_microseconds = stats.total_queue_wait_microseconds,
                  .max_queue_wait_microseconds = stats.max_queue_wait_microseconds,
                  .total_service_time_microseconds = stats.total_service_time_microseconds,
                  .max_service_time_microseconds = stats.max_service_time_microseconds}};
}

[[nodiscard]] TaskProtocolErrorCode admission_error(core::SubmitStatus status) noexcept {
    switch (status) {
        case core::SubmitStatus::kInvalidTask:
        case core::SubmitStatus::kTenantWeightMismatch:
            return TaskProtocolErrorCode::kInvalidRequest;
        case core::SubmitStatus::kQuotaExceeded:
            return TaskProtocolErrorCode::kQuotaExceeded;
        case core::SubmitStatus::kQueueFull:
            return TaskProtocolErrorCode::kQueueFull;
        case core::SubmitStatus::kAccepted:
        case core::SubmitStatus::kInternalError:
            return TaskProtocolErrorCode::kInternalError;
    }
    return TaskProtocolErrorCode::kInternalError;
}

[[nodiscard]] TaskProtocolErrorCode parse_error_code(TaskProtocolParseError error) noexcept {
    switch (error) {
        case TaskProtocolParseError::kUnsupportedVersion:
            return TaskProtocolErrorCode::kUnsupportedVersion;
        case TaskProtocolParseError::kInternalError:
            return TaskProtocolErrorCode::kInternalError;
        case TaskProtocolParseError::kNone:
        case TaskProtocolParseError::kEmpty:
        case TaskProtocolParseError::kLineTooLong:
        case TaskProtocolParseError::kMalformed:
        case TaskProtocolParseError::kInvalidTenant:
        case TaskProtocolParseError::kInvalidValue:
            return TaskProtocolErrorCode::kInvalidRequest;
    }
    return TaskProtocolErrorCode::kInternalError;
}

[[nodiscard]] std::optional<std::string> format_response(
    const TaskProtocolResponse& response) noexcept {
    try {
        return format_task_protocol_response(response);
    } catch (...) {
        return std::nullopt;
    }
}

}  // namespace

TaskControlEndpoint::TaskControlEndpoint(TaskAdmissionService& admission_service,
                                         TaskResourceRegistrar registrar,
                                         void* registrar_context) noexcept
    : admission_service_(admission_service),
      registrar_(registrar),
      registrar_context_(registrar_context) {}

std::optional<std::string> TaskControlEndpoint::handle(std::string_view line) noexcept {
    return handle(line, std::nullopt);
}

std::optional<std::string> TaskControlEndpoint::handle(
    std::string_view line, std::optional<TaskPeerIdentity> peer) noexcept {
    try {
        const TaskProtocolRequestParseResult parsed = parse_task_protocol_request(line);
        if (!parsed.parsed()) {
            return format_response(error_response(parse_error_code(parsed.error)));
        }
        const TaskProtocolRequest request = parsed.request.value_or(TaskProtocolRequest{});
        switch (request.operation) {
            case TaskProtocolOperation::kSubmit: {
                const core::SubmitResult result = admission_service_.submit(
                    request.admission, registrar_, registrar_context_, peer);
                if (!result.accepted()) {
                    return format_response(error_response(admission_error(result.status)));
                }
                return format_response(accepted_response(result.task_id));
            }
            case TaskProtocolOperation::kAcquire: {
                const core::SubmitResult result = admission_service_.submit(
                    request.admission, registrar_, registrar_context_, peer);
                if (!result.accepted()) {
                    return format_response(error_response(admission_error(result.status)));
                }
                const auto snapshot = admission_service_.claim(result.task_id, peer);
                if (snapshot.has_value()) {
                    return format_response(lease_response(snapshot.value()));
                }
                const auto current = admission_service_.find(result.task_id);
                if (current.has_value() && current->state == core::TaskState::kQueued) {
                    return format_response(accepted_response(result.task_id));
                }
                static_cast<void>(admission_service_.cancel(result.task_id));
                return format_response(error_response(TaskProtocolErrorCode::kInternalError));
            }
            case TaskProtocolOperation::kCancel: {
                if (admission_service_.cancel_queued(request.task_id)) {
                    return format_response(
                        state_response(request.task_id, core::TaskState::kCancelled));
                }
                if (!admission_service_.find(request.task_id).has_value()) {
                    return format_response(error_response(TaskProtocolErrorCode::kUnknownTask));
                }
                return format_response(error_response(TaskProtocolErrorCode::kInvalidRequest));
            }
            case TaskProtocolOperation::kQuery: {
                const auto snapshot = admission_service_.find(request.task_id);
                if (!snapshot.has_value()) {
                    return format_response(error_response(TaskProtocolErrorCode::kUnknownTask));
                }
                return format_response(state_response(snapshot->task_id, snapshot->state));
            }
            case TaskProtocolOperation::kHeartbeat: {
                if (admission_service_.heartbeat(request.task_id, peer)) {
                    return format_response(
                        state_response(request.task_id, core::TaskState::kRunning));
                }
                if (!admission_service_.find(request.task_id).has_value()) {
                    return format_response(error_response(TaskProtocolErrorCode::kUnknownTask));
                }
                return format_response(error_response(TaskProtocolErrorCode::kInvalidRequest));
            }
            case TaskProtocolOperation::kClaim: {
                const auto snapshot = request.task_id == 0
                                          ? admission_service_.claim_next(peer)
                                          : admission_service_.claim(request.task_id, peer);
                if (!snapshot.has_value()) {
                    if (request.task_id != 0 &&
                        !admission_service_.find(request.task_id).has_value()) {
                        return format_response(error_response(TaskProtocolErrorCode::kUnknownTask));
                    }
                    return format_response({.kind = TaskProtocolResponseKind::kEmpty,
                                            .task_id = 0,
                                            .state = TaskProtocolState::kQueued,
                                            .error = TaskProtocolErrorCode::kInternalError,
                                            .tenant_id = {},
                                            .memory_bytes = 0,
                                            .weight = 0,
                                            .work_units = 0,
                                            .stats = {}});
                }
                return format_response(lease_response(snapshot.value()));
            }
            case TaskProtocolOperation::kWait: {
                const auto snapshot = admission_service_.wait_claim(
                    request.task_id, std::chrono::milliseconds{request.wait_timeout_ms}, peer);
                if (snapshot.has_value()) {
                    return format_response(lease_response(snapshot.value()));
                }
                const auto current = admission_service_.find(request.task_id);
                if (!current.has_value()) {
                    return format_response(error_response(TaskProtocolErrorCode::kUnknownTask));
                }
                if (current->state != core::TaskState::kQueued) {
                    return format_response(state_response(current->task_id, current->state));
                }
                return format_response({.kind = TaskProtocolResponseKind::kEmpty,
                                        .task_id = 0,
                                        .state = TaskProtocolState::kQueued,
                                        .error = TaskProtocolErrorCode::kInternalError,
                                        .tenant_id = {},
                                        .memory_bytes = 0,
                                        .weight = 0,
                                        .work_units = 0,
                                        .stats = {}});
            }
            case TaskProtocolOperation::kStats:
                return format_response(stats_response(admission_service_.stats(), false));
            case TaskProtocolOperation::kMetrics:
                return format_response(stats_response(admission_service_.stats(), true));
            case TaskProtocolOperation::kComplete:
            case TaskProtocolOperation::kFail: {
                const bool updated = request.operation == TaskProtocolOperation::kComplete
                                         ? admission_service_.complete(request.task_id, peer)
                                         : admission_service_.fail(request.task_id, peer);
                if (updated) {
                    const core::TaskState state =
                        request.operation == TaskProtocolOperation::kComplete
                            ? core::TaskState::kCompleted
                            : core::TaskState::kFailed;
                    return format_response(state_response(request.task_id, state));
                }
                if (!admission_service_.find(request.task_id).has_value()) {
                    return format_response(error_response(TaskProtocolErrorCode::kUnknownTask));
                }
                return format_response(error_response(TaskProtocolErrorCode::kInvalidRequest));
            }
        }
        return format_response(error_response(TaskProtocolErrorCode::kInternalError));
    } catch (...) {
        return format_response(error_response(TaskProtocolErrorCode::kInternalError));
    }
}

}  // namespace glimmer::control
