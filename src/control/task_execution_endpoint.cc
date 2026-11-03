#include "glimmer/control/task_execution_endpoint.h"
#include "glimmer/control/native_epoch_protocol.h"
#include "glimmer/control/task_execution_protocol.h"

#include "glimmer/core/scheduler.h"

#include <new>
#include <stdexcept>
#include <utility>

namespace glimmer::control {
#ifdef GLIMMER_ENDPOINT_TESTING
namespace testing {
thread_local unsigned fail_next_response = 0;
void fail_next_endpoint_response_format() noexcept {
    fail_next_response = 1;
}
void throw_next_endpoint_response_format() noexcept {
    fail_next_response = 2;
}
}  // namespace testing
#endif
namespace {

[[nodiscard]] bool fail_response_for_test() {
#ifdef GLIMMER_ENDPOINT_TESTING
    const auto mode = std::exchange(testing::fail_next_response, 0U);
    if (mode == 2) {
        throw std::bad_alloc();
    }
    return mode == 1;
#else
    return false;
#endif
}

[[nodiscard]] TaskExecutionProtocolResponse error_response(TaskProtocolErrorCode error) noexcept {
    return {.kind = TaskExecutionProtocolResponseKind::kError,
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

[[nodiscard]] TaskExecutionProtocolResponse state_response(core::TaskId task_id,
                                                           core::TaskState state) noexcept {
    return {.kind = TaskExecutionProtocolResponseKind::kState,
            .task_id = task_id,
            .state = protocol_state(state),
            .error = TaskProtocolErrorCode::kInternalError,
            .tenant_id = {},
            .memory_bytes = 0,
            .weight = 0,
            .work_units = 0,
            .stats = {}};
}

[[nodiscard]] TaskExecutionProtocolResponse lease_response(const core::TaskSnapshot& snapshot) {
    return {.kind = TaskExecutionProtocolResponseKind::kLease,
            .task_id = snapshot.task_id,
            .state = TaskProtocolState::kRunning,
            .error = TaskProtocolErrorCode::kInternalError,
            .tenant_id = snapshot.tenant_id,
            .memory_bytes = snapshot.memory_bytes,
            .weight = snapshot.weight,
            .work_units = snapshot.work_units,
            .stats = {}};
}

[[nodiscard]] TaskExecutionProtocolResponse accepted_response(core::TaskId task_id) noexcept {
    return {.kind = TaskExecutionProtocolResponseKind::kAccepted,
            .task_id = task_id,
            .state = TaskProtocolState::kQueued,
            .error = TaskProtocolErrorCode::kInternalError,
            .tenant_id = {},
            .memory_bytes = 0,
            .weight = 0,
            .work_units = 0,
            .stats = {}};
}

[[nodiscard]] TaskExecutionProtocolResponse stats_response(const core::SchedulerStats& stats,
                                                           bool include_metrics) {
    return {
        .kind = include_metrics ? TaskExecutionProtocolResponseKind::kMetrics
                                : TaskExecutionProtocolResponseKind::kStats,
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

[[nodiscard]] std::optional<TaskExecutionProtocolOperation> execution_operation(
    TaskProtocolOperation operation) noexcept {
    switch (operation) {
        case TaskProtocolOperation::kSubmit:
            return TaskExecutionProtocolOperation::kSubmit;
        case TaskProtocolOperation::kAcquire:
            return TaskExecutionProtocolOperation::kAcquire;
        case TaskProtocolOperation::kCancel:
            return TaskExecutionProtocolOperation::kCancel;
        case TaskProtocolOperation::kQuery:
            return TaskExecutionProtocolOperation::kQuery;
        case TaskProtocolOperation::kClaim:
            return TaskExecutionProtocolOperation::kClaim;
        case TaskProtocolOperation::kWait:
            return TaskExecutionProtocolOperation::kWait;
        case TaskProtocolOperation::kHeartbeat:
            return TaskExecutionProtocolOperation::kHeartbeat;
        case TaskProtocolOperation::kComplete:
            return TaskExecutionProtocolOperation::kComplete;
        case TaskProtocolOperation::kFail:
            return TaskExecutionProtocolOperation::kFail;
        case TaskProtocolOperation::kStats:
            return TaskExecutionProtocolOperation::kStats;
        case TaskProtocolOperation::kMetrics:
            return TaskExecutionProtocolOperation::kMetrics;
        default:
            return std::nullopt;
    }
    return std::nullopt;
}

[[nodiscard]] std::optional<TaskProtocolResponseKind> legacy_kind(
    TaskExecutionProtocolResponseKind kind) noexcept {
    switch (kind) {
        case TaskExecutionProtocolResponseKind::kAccepted:
            return TaskProtocolResponseKind::kAccepted;
        case TaskExecutionProtocolResponseKind::kState:
            return TaskProtocolResponseKind::kState;
        case TaskExecutionProtocolResponseKind::kError:
            return TaskProtocolResponseKind::kError;
        case TaskExecutionProtocolResponseKind::kLease:
            return TaskProtocolResponseKind::kLease;
        case TaskExecutionProtocolResponseKind::kEmpty:
            return TaskProtocolResponseKind::kEmpty;
        case TaskExecutionProtocolResponseKind::kStats:
            return TaskProtocolResponseKind::kStats;
        case TaskExecutionProtocolResponseKind::kMetrics:
            return TaskProtocolResponseKind::kMetrics;
        case TaskExecutionProtocolResponseKind::kExecutionLease:
        case TaskExecutionProtocolResponseKind::kRetreat:
        case TaskExecutionProtocolResponseKind::kExecutionHandoff:
            return std::nullopt;
    }
    return std::nullopt;
}

[[nodiscard]] std::optional<std::string> encode_response(
    const TaskExecutionProtocolResponse& response, bool legacy) noexcept {
    try {
        if (fail_response_for_test()) {
            return std::nullopt;
        }
        if (!legacy) {
            return format_task_execution_protocol_response(response);
        }
        const auto kind = legacy_kind(response.kind);
        if (!kind || response.generation != 0 || response.completed_execution.task_id != 0 ||
            response.completed_execution.generation != 0) {
            return std::nullopt;
        }
        const TaskProtocolResponse result{.kind = *kind,
                                          .task_id = response.task_id,
                                          .state = response.state,
                                          .error = response.error,
                                          .tenant_id = response.tenant_id,
                                          .memory_bytes = response.memory_bytes,
                                          .weight = response.weight,
                                          .work_units = response.work_units,
                                          .stats = response.stats};
        return format_task_protocol_response(result);
    } catch (...) {
        return std::nullopt;
    }
}

[[nodiscard]] std::optional<std::string> encode_epoch_response(
    const NativeEpochResponse& response) noexcept {
    try {
        if (fail_response_for_test()) {
            return std::nullopt;
        }
        return format_native_epoch_response(response);
    } catch (...) {
        return std::nullopt;
    }
}

[[nodiscard]] std::string_view envelope(std::string_view line) noexcept {
    const auto start = line.find_first_not_of(" \t");
    if (start == std::string_view::npos) {
        return {};
    }
    line.remove_prefix(start);
    return line.substr(0, line.find_first_of(" \t\r\n"));
}

// Only this call may publish a newly issued cooperative/native grant. If its
// response cannot be built, physical issue has not been authorized yet.
class UnpublishedTask final {
   public:
    enum class Kind : std::uint8_t {
        kNone,
        kLegacyPending,
        kCooperativePending,
        kCooperativeActive,
        kNativePending,
        kNativeActive
    };

    UnpublishedTask(TaskExecutionAdmissionService& service, TaskResourceRollback rollback,
                    void* context, std::optional<TaskPeerIdentity> peer) noexcept
        : service_(service), rollback_(rollback), context_(context), peer_(peer) {}
    UnpublishedTask(const UnpublishedTask&) = delete;
    UnpublishedTask& operator=(const UnpublishedTask&) = delete;

    ~UnpublishedTask() noexcept {
        try {
            bool retired = false;
            switch (kind_) {
                case Kind::kNone:
                    break;
                case Kind::kLegacyPending:
                    retired = service_.cancel_queued(execution_.task_id, peer_);
                    break;
                case Kind::kCooperativePending:
                    retired = service_.cancel_pending_cooperative(execution_.task_id, peer_);
                    break;
                case Kind::kCooperativeActive:
                    retired = service_.fail_execution(execution_, peer_);
                    break;
                case Kind::kNativePending:
                    retired = service_.cancel_pending_native_epoch(execution_.task_id, peer_);
                    break;
                case Kind::kNativeActive:
                    retired = service_.finish_native_epoch(execution_, 0, true, peer_);
                    break;
            }
            if (retired) {
                rollback_(context_, execution_.task_id);
            }
        } catch (...) {
            // Uncertain ownership remains retained; never release it by guess.
            return;
        }
    }

    void arm(Kind kind, TaskExecutionIdentity execution) noexcept {
        kind_ = kind;
        execution_ = execution;
    }
    void publish() noexcept {
        kind_ = Kind::kNone;
    }

   private:
    TaskExecutionAdmissionService& service_;
    TaskResourceRollback rollback_;
    void* context_;
    std::optional<TaskPeerIdentity> peer_;
    Kind kind_ = Kind::kNone;
    TaskExecutionIdentity execution_{};
};

}  // namespace

struct TaskExecutionEndpoint::Impl {
    Impl(std::unique_ptr<TaskExecutionAdmissionService> service, TaskResourceRegistrar registrar,
         TaskResourceRollback rollback, void* context)
        : admission_service_(std::move(service)),
          registrar_(registrar),
          rollback_(rollback),
          registrar_context_(context) {
        if (!admission_service_ || registrar_ == nullptr || rollback_ == nullptr) {
            throw std::invalid_argument("execution service, registrar and rollback are required");
        }
    }

    std::optional<std::string> handle(std::string_view line,
                                      std::optional<TaskPeerIdentity> peer) noexcept;
    std::optional<std::string> handle_native_epoch(std::string_view line,
                                                   std::optional<TaskPeerIdentity> peer);

    std::unique_ptr<TaskExecutionAdmissionService> admission_service_;
    TaskResourceRegistrar registrar_;
    TaskResourceRollback rollback_;
    void* registrar_context_;
};

TaskExecutionEndpoint::TaskExecutionEndpoint(std::unique_ptr<TaskExecutionAdmissionService> service,
                                             TaskResourceRegistrar registrar,
                                             TaskResourceRollback rollback, void* context)
    : impl_(std::make_unique<Impl>(std::move(service), registrar, rollback, context)) {}

TaskExecutionEndpoint::~TaskExecutionEndpoint() = default;

std::optional<std::string> TaskExecutionEndpoint::handle(
    std::string_view line, std::optional<TaskPeerIdentity> peer) noexcept {
    return impl_->handle(line, peer);
}

std::optional<std::string> TaskExecutionEndpoint::Impl::handle(
    std::string_view line, std::optional<TaskPeerIdentity> peer) noexcept {
    const auto version = envelope(line);
    const bool legacy = version != kTaskExecutionProtocolVersion;
    const auto format_response = [legacy](const TaskExecutionProtocolResponse& response) {
        return encode_response(response, legacy);
    };
    try {
        if (version == kNativeEpochProtocolVersion) {
            try {
                return handle_native_epoch(line, peer);
            } catch (...) {
                return std::nullopt;
            }
        }
        TaskExecutionProtocolRequest request;
        if (legacy) {
            const auto parsed = parse_task_protocol_request(line);
            if (!parsed.parsed() || !parsed.request.has_value()) {
                return format_response(error_response(parse_error_code(parsed.error)));
            }
            const auto operation = execution_operation(parsed.request->operation);
            if (!operation.has_value()) {
                return format_response(error_response(TaskProtocolErrorCode::kInvalidRequest));
            }
            request.operation = *operation;
            request.admission = parsed.request->admission;
            request.task_id = parsed.request->task_id;
            request.wait_timeout_ms = parsed.request->wait_timeout_ms;
        } else {
            const auto parsed = parse_task_execution_protocol_request(line);
            if (!parsed.parsed() || !parsed.request.has_value()) {
                return format_response(error_response(parse_error_code(parsed.error)));
            }
            request = *parsed.request;
        }
        switch (request.operation) {
            case TaskExecutionProtocolOperation::kSubmit: {
                UnpublishedTask unpublished(*admission_service_, rollback_, registrar_context_,
                                            peer);
                const core::SubmitResult result =
                    legacy ? admission_service_->submit(request.admission, registrar_,
                                                        registrar_context_, peer)
                           : admission_service_->submit_cooperative(request.admission, registrar_,
                                                                    registrar_context_, peer);
                if (!result.accepted()) {
                    return format_response(error_response(admission_error(result.status)));
                }
                unpublished.arm(legacy ? UnpublishedTask::Kind::kLegacyPending
                                       : UnpublishedTask::Kind::kCooperativePending,
                                {result.task_id, 0});
                auto encoded = format_response(accepted_response(result.task_id));
                if (encoded) unpublished.publish();
                return encoded;
            }
            case TaskExecutionProtocolOperation::kExecutionAcquire: {
                if (!admission_service_->supports_cooperative_execution()) {
                    return format_response(error_response(TaskProtocolErrorCode::kInvalidRequest));
                }
                UnpublishedTask unpublished(*admission_service_, rollback_, registrar_context_,
                                            peer);
                const auto result = admission_service_->submit_cooperative(
                    request.admission, registrar_, registrar_context_, peer);
                if (!result.accepted()) {
                    return format_response(error_response(admission_error(result.status)));
                }
                unpublished.arm(UnpublishedTask::Kind::kCooperativePending, {result.task_id, 0});
                const auto snapshot = admission_service_->claim_execution(result.task_id, peer);
                if (!snapshot) {
                    // Preserve accepted ownership; denial is not cancellation authority.
                    auto encoded = format_response(accepted_response(result.task_id));
                    if (encoded) unpublished.publish();
                    return encoded;
                }
                unpublished.arm(UnpublishedTask::Kind::kCooperativeActive,
                                {result.task_id, snapshot->dispatch_sequence});
                auto response = lease_response(*snapshot);
                response.kind = TaskExecutionProtocolResponseKind::kExecutionLease;
                response.generation = snapshot->dispatch_sequence;
                auto encoded = format_response(response);
                if (encoded) unpublished.publish();
                return encoded;
            }
            case TaskExecutionProtocolOperation::kExecutionCompleteAcquire: {
                const TaskExecutionIdentity execution{request.task_id, request.generation};
                if (!admission_service_->supports_cooperative_execution() ||
                    !admission_service_->complete_execution(execution, peer)) {
                    return format_response(error_response(TaskProtocolErrorCode::kInvalidRequest));
                }
                TaskExecutionProtocolResponse response;
                response.kind = TaskExecutionProtocolResponseKind::kExecutionHandoff;
                response.completed_execution = execution;
                UnpublishedTask unpublished(*admission_service_, rollback_, registrar_context_,
                                            peer);
                // Completion is independently acknowledged even if new admission
                // rejects. Separate service locks preserve competing policy work.
                const auto result = admission_service_->submit_cooperative(
                    request.admission, registrar_, registrar_context_, peer);
                if (result.accepted()) {
                    response.task_id = result.task_id;
                    unpublished.arm(UnpublishedTask::Kind::kCooperativePending,
                                    {result.task_id, 0});
                    const auto snapshot = admission_service_->claim_execution(result.task_id, peer);
                    if (snapshot) {
                        response.generation = snapshot->dispatch_sequence;
                        unpublished.arm(UnpublishedTask::Kind::kCooperativeActive,
                                        {result.task_id, snapshot->dispatch_sequence});
                    }
                }
                auto encoded = format_response(response);
                if (encoded) unpublished.publish();
                return encoded;
            }
            case TaskExecutionProtocolOperation::kAcquire: {
                UnpublishedTask unpublished(*admission_service_, rollback_, registrar_context_,
                                            peer);
                const core::SubmitResult result = admission_service_->submit(
                    request.admission, registrar_, registrar_context_, peer);
                if (!result.accepted()) {
                    return format_response(error_response(admission_error(result.status)));
                }
                unpublished.arm(UnpublishedTask::Kind::kLegacyPending, {result.task_id, 0});
                const auto snapshot = admission_service_->claim(result.task_id, peer);
                if (snapshot.has_value()) {
                    // Legacy active claims can be repeated; another caller may
                    // already have observed this lease, so do not roll it back.
                    unpublished.publish();
                    return format_response(lease_response(snapshot.value()));
                }
                const auto current = admission_service_->find(result.task_id);
                if (current.has_value() && current->state == core::TaskState::kQueued) {
                    auto encoded = format_response(accepted_response(result.task_id));
                    if (encoded) unpublished.publish();
                    return encoded;
                }
                return format_response(error_response(TaskProtocolErrorCode::kInternalError));
            }
            case TaskExecutionProtocolOperation::kCancel: {
                const auto before = admission_service_->find(request.task_id);
                if (before.has_value() && before->dispatch_sequence != 0) {
                    // A stopped continuation must use its authenticated generation.
                    return format_response(error_response(TaskProtocolErrorCode::kInvalidRequest));
                }
                if ((!legacy &&
                     admission_service_->cancel_pending_cooperative(request.task_id, peer)) ||
                    admission_service_->cancel_queued(request.task_id, peer)) {
                    return format_response(
                        state_response(request.task_id, core::TaskState::kCancelled));
                }
                if (!admission_service_->find(request.task_id).has_value()) {
                    return format_response(error_response(TaskProtocolErrorCode::kUnknownTask));
                }
                return format_response(error_response(TaskProtocolErrorCode::kInvalidRequest));
            }
            case TaskExecutionProtocolOperation::kQuery: {
                const auto snapshot = admission_service_->find(request.task_id);
                if (!snapshot.has_value()) {
                    return format_response(error_response(TaskProtocolErrorCode::kUnknownTask));
                }
                return format_response(state_response(snapshot->task_id, snapshot->state));
            }
            case TaskExecutionProtocolOperation::kExecutionClaim: {
                if (!admission_service_->supports_cooperative_execution()) {
                    return format_response(error_response(TaskProtocolErrorCode::kInvalidRequest));
                }
                if (!admission_service_->find(request.task_id).has_value()) {
                    return format_response(error_response(TaskProtocolErrorCode::kUnknownTask));
                }
                const auto snapshot = admission_service_->claim_execution(request.task_id, peer);
                if (!snapshot.has_value()) {
                    TaskExecutionProtocolResponse response;
                    response.kind = TaskExecutionProtocolResponseKind::kEmpty;
                    return format_response(response);
                }
                UnpublishedTask unpublished(*admission_service_, rollback_, registrar_context_,
                                            peer);
                unpublished.arm(UnpublishedTask::Kind::kCooperativeActive,
                                {snapshot->task_id, snapshot->dispatch_sequence});
                auto response = lease_response(*snapshot);
                response.kind = TaskExecutionProtocolResponseKind::kExecutionLease;
                response.generation = snapshot->dispatch_sequence;
                auto encoded = format_response(response);
                if (encoded) unpublished.publish();
                return encoded;
            }
            case TaskExecutionProtocolOperation::kRetreat: {
                const auto retreat = admission_service_->retreat_requested(
                    {request.task_id, request.generation}, peer);
                if (!retreat.has_value()) {
                    return format_response(error_response(TaskProtocolErrorCode::kInvalidRequest));
                }
                TaskExecutionProtocolResponse response;
                response.kind = TaskExecutionProtocolResponseKind::kRetreat;
                response.task_id = request.task_id;
                response.generation = request.generation;
                response.should_retreat = *retreat;
                return format_response(response);
            }
            case TaskExecutionProtocolOperation::kSuspend:
            case TaskExecutionProtocolOperation::kExecutionHeartbeat:
            case TaskExecutionProtocolOperation::kExecutionComplete:
            case TaskExecutionProtocolOperation::kExecutionFail:
            case TaskExecutionProtocolOperation::kExecutionCancel: {
                const TaskExecutionIdentity execution{request.task_id, request.generation};
                bool updated = false;
                if (request.operation == TaskExecutionProtocolOperation::kSuspend) {
                    updated = admission_service_->suspend_execution(execution, peer);
                } else if (request.operation ==
                           TaskExecutionProtocolOperation::kExecutionHeartbeat) {
                    updated = admission_service_->heartbeat_execution(execution, peer);
                } else if (request.operation ==
                           TaskExecutionProtocolOperation::kExecutionComplete) {
                    updated = admission_service_->complete_execution(execution, peer);
                } else if (request.operation == TaskExecutionProtocolOperation::kExecutionFail) {
                    updated = admission_service_->fail_execution(execution, peer);
                } else {
                    updated = admission_service_->cancel_execution(execution, peer);
                }
                const auto current = admission_service_->find(request.task_id);
                if (!updated || !current.has_value()) {
                    return format_response(error_response(TaskProtocolErrorCode::kInvalidRequest));
                }
                return format_response(state_response(current->task_id, current->state));
            }
            case TaskExecutionProtocolOperation::kHeartbeat: {
                if (admission_service_->heartbeat(request.task_id, peer)) {
                    return format_response(
                        state_response(request.task_id, core::TaskState::kRunning));
                }
                if (!admission_service_->find(request.task_id).has_value()) {
                    return format_response(error_response(TaskProtocolErrorCode::kUnknownTask));
                }
                return format_response(error_response(TaskProtocolErrorCode::kInvalidRequest));
            }
            case TaskExecutionProtocolOperation::kClaim: {
                const auto snapshot = request.task_id == 0
                                          ? admission_service_->claim_next(peer)
                                          : admission_service_->claim(request.task_id, peer);
                if (!snapshot.has_value()) {
                    if (request.task_id != 0 &&
                        !admission_service_->find(request.task_id).has_value()) {
                        return format_response(error_response(TaskProtocolErrorCode::kUnknownTask));
                    }
                    return format_response({.kind = TaskExecutionProtocolResponseKind::kEmpty,
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
            case TaskExecutionProtocolOperation::kWait: {
                const auto snapshot = admission_service_->wait_claim(
                    request.task_id, std::chrono::milliseconds{request.wait_timeout_ms}, peer);
                if (snapshot.has_value()) {
                    return format_response(lease_response(snapshot.value()));
                }
                const auto current = admission_service_->find(request.task_id);
                if (!current.has_value()) {
                    return format_response(error_response(TaskProtocolErrorCode::kUnknownTask));
                }
                if (current->state != core::TaskState::kQueued) {
                    return format_response(state_response(current->task_id, current->state));
                }
                return format_response({.kind = TaskExecutionProtocolResponseKind::kEmpty,
                                        .task_id = 0,
                                        .state = TaskProtocolState::kQueued,
                                        .error = TaskProtocolErrorCode::kInternalError,
                                        .tenant_id = {},
                                        .memory_bytes = 0,
                                        .weight = 0,
                                        .work_units = 0,
                                        .stats = {}});
            }
            case TaskExecutionProtocolOperation::kStats:
                return format_response(stats_response(admission_service_->stats(), false));
            case TaskExecutionProtocolOperation::kMetrics:
                return format_response(stats_response(admission_service_->stats(), true));
            case TaskExecutionProtocolOperation::kComplete:
            case TaskExecutionProtocolOperation::kFail: {
                const bool updated = request.operation == TaskExecutionProtocolOperation::kComplete
                                         ? admission_service_->complete(request.task_id, peer)
                                         : admission_service_->fail(request.task_id, peer);
                if (updated) {
                    const core::TaskState state =
                        request.operation == TaskExecutionProtocolOperation::kComplete
                            ? core::TaskState::kCompleted
                            : core::TaskState::kFailed;
                    return format_response(state_response(request.task_id, state));
                }
                if (!admission_service_->find(request.task_id).has_value()) {
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

namespace {

TaskProtocolErrorCode epoch_admission_error(core::SubmitStatus status) noexcept {
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

}  // namespace

std::optional<std::string> TaskExecutionEndpoint::Impl::handle_native_epoch(
    std::string_view line, std::optional<TaskPeerIdentity> peer) {
    NativeEpochResponse response;
    const auto request = parse_native_epoch_request(line);
    if (!request) {
        return encode_epoch_response(response);
    }
    response.operation = request->operation;
    response.request_sequence = request->request_sequence;
    response.grant.execution = request->execution;
    UnpublishedTask unpublished(*admission_service_, rollback_, registrar_context_, peer);
    switch (request->operation) {
        case NativeEpochOperation::kAcquire:
        case NativeEpochOperation::kClaim: {
            auto task_id = request->execution.task_id;
            if (request->operation == NativeEpochOperation::kAcquire) {
                const auto result = admission_service_->submit_native_epoch(
                    request->admission, request->extent, registrar_, registrar_context_, peer);
                if (!result.accepted()) {
                    response.error = epoch_admission_error(result.status);
                    break;
                }
                task_id = result.task_id;
                unpublished.arm(UnpublishedTask::Kind::kNativePending, {task_id, 0});
            }
            const auto granted = admission_service_->claim_native_epoch(task_id, peer);
            if (granted) {
                response.kind = NativeEpochResponseKind::kGranted;
                response.grant = *granted;
                unpublished.arm(UnpublishedTask::Kind::kNativeActive, granted->execution);
                break;
            }
            const auto pending = admission_service_->pending_native_epoch_extent(task_id, peer);
            if (pending) {
                response.kind = NativeEpochResponseKind::kPending;
                response.grant = {.execution = {task_id, 0}, .extent = *pending};
            }
            break;
        }
        case NativeEpochOperation::kRetreat: {
            const auto retreat =
                admission_service_->native_epoch_retreat_requested(request->execution, peer);
            if (retreat) {
                response.kind = NativeEpochResponseKind::kRetreat;
                response.should_retreat = *retreat;
            }
            break;
        }
        case NativeEpochOperation::kHeartbeat:
            if (admission_service_->heartbeat_native_epoch(request->execution, peer)) {
                response.kind = NativeEpochResponseKind::kAcknowledged;
            }
            break;
        case NativeEpochOperation::kComplete:
        case NativeEpochOperation::kFail:
            if (admission_service_->finish_native_epoch(
                    request->execution, request->issued_calls,
                    request->operation == NativeEpochOperation::kFail, peer)) {
                response.kind = NativeEpochResponseKind::kAcknowledged;
                response.issued_calls = request->issued_calls;
            }
            break;
        case NativeEpochOperation::kCancelPending:
            if (admission_service_->cancel_pending_native_epoch(request->execution.task_id, peer)) {
                response.kind = NativeEpochResponseKind::kAcknowledged;
            }
            break;
    }
    auto encoded = encode_epoch_response(response);
    if (encoded) {
        unpublished.publish();
    }
    return encoded;
}

}  // namespace glimmer::control
