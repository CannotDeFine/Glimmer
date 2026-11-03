#include "remote_control_operation.h"

#include "control_request_sequence.h"

#include <algorithm>
#include <limits>
#include <type_traits>
#include <unistd.h>
#include <utility>

namespace glimmer::control::internal {
namespace {

static_assert(std::is_nothrow_copy_assignable_v<RemoteControlResult>);

TaskAdmissionRequest admission_request(const LaunchGateOptions& options,
                                       CooperativeLaunchRequest settings) {
    return {.tenant_id = options.tenant_id.empty() ? "default" : options.tenant_id,
            .memory_bytes = 1,
            .weight = options.tenant_weight,
            .work_units = settings.work_units,
            .priority = options.task_priority + std::min(settings.relative_priority,
                                                         std::numeric_limits<std::uint32_t>::max() -
                                                             options.task_priority)};
}

bool matching_execution(TaskExecutionIdentity first, TaskExecutionIdentity second) noexcept {
    return first.task_id == second.task_id && first.generation == second.generation;
}

bool allows_epoch(std::optional<NativeEpochExtent> policy, NativeEpochExtent extent) noexcept {
    return policy && valid_native_epoch_extent(*policy) && valid_native_epoch_extent(extent) &&
           extent.max_calls <= policy->max_calls &&
           extent.calls_per_prefix <= policy->calls_per_prefix &&
           extent.max_pending_prefixes <= policy->max_pending_prefixes &&
           extent.acceptance_microseconds <= policy->acceptance_microseconds;
}

bool valid_epoch_grant(std::optional<NativeEpochExtent> policy, NativeEpochGrant grant,
                       bool is_pending) noexcept {
    return allows_epoch(policy, grant.extent) && grant.execution.task_id != 0 &&
           (is_pending
                ? grant.execution.generation == 0 && grant.acceptance_deadline_nanoseconds == 0
                : grant.execution.generation != 0 && grant.acceptance_deadline_nanoseconds != 0);
}

bool acknowledged(const TaskExecutionProtocolResponse& response, TaskExecutionIdentity execution,
                  TaskProtocolState expected) noexcept {
    // STATE has no generation field. The generation is in the exact request,
    // validated by the service and correlated by the qualified transport.
    return response.kind == TaskExecutionProtocolResponseKind::kState &&
           response.task_id == execution.task_id && response.state == expected;
}

std::optional<CooperativeSubmission> submission(
    const TaskExecutionProtocolResponse& response) noexcept {
    if (response.task_id == 0) return std::nullopt;
    if (response.kind == TaskExecutionProtocolResponseKind::kAccepted)
        return CooperativeSubmission{response.task_id,
                                     {.status = CooperativeClaimStatus::kPending}};
    if (response.kind == TaskExecutionProtocolResponseKind::kExecutionLease &&
        response.generation != 0)
        return CooperativeSubmission{
            response.task_id,
            {CooperativeClaimStatus::kGranted, {response.task_id, response.generation}}};
    return std::nullopt;
}

}  // namespace

RemoteControlOperation::RemoteControlOperation() : process_id(::getpid()) {
    response_line_.reserve(kTaskProtocolMaxLineBytes);
}

RemoteControlOperation::~RemoteControlOperation() {
    if (state_ == State::kPending && endpoint_ != nullptr && process_id == ::getpid()) quarantine();
}

std::unique_ptr<RemoteControlOperation> RemoteControlOperation::prepare(
    RemoteCooperativeCommand command, const LaunchGateOptions& options) noexcept {
    try {
        auto value = std::unique_ptr<RemoteControlOperation>(new RemoteControlOperation);
        auto& request = value->task_request_;
        request.operation = command.operation;
        request.task_id = command.execution.task_id;
        request.generation = command.operation == TaskExecutionProtocolOperation::kExecutionClaim ||
                                     command.operation == TaskExecutionProtocolOperation::kCancel
                                 ? 0
                                 : command.execution.generation;
        switch (command.operation) {
            case TaskExecutionProtocolOperation::kSubmit:
                value->result_ = std::optional<core::TaskId>{};
                break;
            case TaskExecutionProtocolOperation::kExecutionAcquire:
                value->result_ = CooperativeSubmission{};
                break;
            case TaskExecutionProtocolOperation::kExecutionCompleteAcquire:
                value->result_ = CooperativeHandoff{};
                if (request.task_id == 0 || request.generation == 0) return value;
                break;
            case TaskExecutionProtocolOperation::kExecutionClaim:
                value->result_ = CooperativeClaim{};
                break;
            case TaskExecutionProtocolOperation::kRetreat:
                value->result_ = std::optional<bool>{};
                break;
            case TaskExecutionProtocolOperation::kExecutionHeartbeat:
            case TaskExecutionProtocolOperation::kSuspend:
            case TaskExecutionProtocolOperation::kExecutionComplete:
            case TaskExecutionProtocolOperation::kExecutionFail:
            case TaskExecutionProtocolOperation::kExecutionCancel:
            case TaskExecutionProtocolOperation::kCancel:
                value->result_ = false;
                break;
            default:
                return nullptr;
        }
        if (command.operation == TaskExecutionProtocolOperation::kSubmit ||
            command.operation == TaskExecutionProtocolOperation::kExecutionAcquire ||
            command.operation == TaskExecutionProtocolOperation::kExecutionCompleteAcquire)
            request.admission = admission_request(options, command.settings);
        auto encoded = format_task_execution_protocol_request(request);
        if (!encoded &&
            command.operation == TaskExecutionProtocolOperation::kExecutionCompleteAcquire) {
            value->has_fallback_ = true;
            TaskExecutionProtocolRequest complete{};
            complete.operation = TaskExecutionProtocolOperation::kExecutionComplete;
            complete.task_id = request.task_id;
            complete.generation = request.generation;
            encoded = format_task_execution_protocol_request(complete);
            TaskExecutionProtocolRequest acquire{};
            acquire.operation = TaskExecutionProtocolOperation::kExecutionAcquire;
            acquire.admission = request.admission;
            const auto secondary = format_task_execution_protocol_request(acquire);
            if (secondary) value->secondary_line_ = *secondary;
        }
        if (encoded) value->request_line_ = std::move(*encoded);
        value->failure_result_ = value->result_;
        return value;
    } catch (...) {
        // Preparation is before acceptance/publication; allocation cannot
        // create a grant or ambiguously execute a report.
        return nullptr;
    }
}

std::unique_ptr<RemoteControlOperation> RemoteControlOperation::prepare(
    RemoteNativeCommand command, const LaunchGateOptions& options,
    std::optional<NativeEpochExtent> policy) noexcept {
    try {
        auto value = std::unique_ptr<RemoteControlOperation>(new RemoteControlOperation);
        value->is_native_ = true;
        value->epoch_policy_ = policy;
        auto& request = value->epoch_request_;
        request.operation = command.operation;
        request.execution = command.grant.execution;
        request.issued_calls = command.issued_calls;
        value->expected_epoch_ = command.grant;
        switch (command.operation) {
            case NativeEpochOperation::kAcquire:
                value->result_ = NativeEpochClaim{};
                if (!allows_epoch(policy, command.settings.extent)) return value;
                value->expected_epoch_ = {.execution = {}, .extent = command.settings.extent};
                value->result_ =
                    NativeEpochClaim{NativeEpochClaimStatus::kUncertain, value->expected_epoch_};
                request.admission = admission_request(
                    options, {.relative_priority = command.settings.relative_priority,
                              .work_units = command.settings.extent.max_calls});
                request.extent = command.settings.extent;
                request.execution = {};
                break;
            case NativeEpochOperation::kClaim:
                value->result_ =
                    NativeEpochClaim{NativeEpochClaimStatus::kUncertain, command.grant};
                if (!valid_epoch_grant(policy, command.grant, true)) return value;
                break;
            case NativeEpochOperation::kRetreat:
                value->result_ = std::optional<bool>{};
                if (!valid_epoch_grant(policy, command.grant, false)) return value;
                break;
            case NativeEpochOperation::kHeartbeat:
            case NativeEpochOperation::kComplete:
            case NativeEpochOperation::kFail:
            case NativeEpochOperation::kCancelPending:
                value->result_ = false;
                if (!valid_epoch_grant(policy, command.grant,
                                       command.operation == NativeEpochOperation::kCancelPending) ||
                    command.issued_calls > command.grant.extent.max_calls)
                    return value;
                break;
            default:
                return nullptr;
        }
        request.request_sequence = next_epoch_request_sequence();
        const auto encoded = format_native_epoch_request(request);
        if (encoded) value->request_line_ = *encoded;
        value->failure_result_ = value->result_;
        return value;
    } catch (...) {
        return nullptr;
    }
}

std::unique_ptr<RemoteControlOperation> RemoteControlOperation::prepare_handoff(
    TaskExecutionIdentity completed, const RemoteControlOperation& successor) noexcept {
    if (!successor.is_prepared() || successor.is_native_ || completed.task_id == 0 ||
        completed.generation == 0 ||
        successor.task_request_.operation != TaskExecutionProtocolOperation::kExecutionAcquire)
        return nullptr;
    try {
        auto value = std::unique_ptr<RemoteControlOperation>(new RemoteControlOperation);
        value->task_request_ = successor.task_request_;
        value->task_request_.operation = TaskExecutionProtocolOperation::kExecutionCompleteAcquire;
        value->task_request_.task_id = completed.task_id;
        value->task_request_.generation = completed.generation;
        const auto encoded = format_task_execution_protocol_request(value->task_request_);
        if (!encoded) return nullptr;
        value->request_line_ = *encoded;
        value->result_ = CooperativeHandoff{};
        value->failure_result_ = value->result_;
        return value;
    } catch (...) {
        return nullptr;
    }
}

RemoteControlStep RemoteControlOperation::publish(std::string_view line,
                                                  Clock::time_point now) noexcept {
    const auto publication = endpoint_->begin(line, now);
    ticket_ = publication.ticket;
    if (publication.status == SharedControlPublication::kRejected) {
        state_ = State::kFinished;
        return RemoteControlStep::kFinished;
    }
    has_publication_ = true;
    state_ = State::kPending;
    if (publication.status == SharedControlPublication::kAmbiguous) {
        quarantine();
        return RemoteControlStep::kUncertain;
    }
    return RemoteControlStep::kPending;
}

RemoteControlStep RemoteControlOperation::begin(OwnedControlExchange& endpoint,
                                                Clock::time_point now) noexcept {
    if (process_id != ::getpid() || state_ != State::kPrepared) return RemoteControlStep::kRejected;
    endpoint_ = &endpoint;
    if (request_line_.empty()) {
        state_ = State::kFinished;
        return RemoteControlStep::kFinished;
    }
    return publish(request_line_, now);
}

void RemoteControlOperation::quarantine() noexcept {
    state_ = State::kUncertain;
    // Use preconstructed, nonthrowing copies: never publish a partly decoded
    // positive result after an exception, and retain any independent old ACK.
    result_ = has_completed_ ? acknowledged_failure_ : failure_result_;
    if (endpoint_ != nullptr && process_id == ::getpid()) {
        if (endpoint_->wait_state(ticket_)) {
            static_cast<void>(endpoint_->abandon(ticket_));
        } else {
            // Actual transport completion may still have an invalid domain
            // reply. Retire its idle CPU endpoint, not the ownership ledger.
            static_cast<void>(endpoint_->close_idle());
        }
    }
}

RemoteControlStep RemoteControlOperation::poll(Clock::time_point now) noexcept {
    if (process_id != ::getpid()) return RemoteControlStep::kRejected;
    if (state_ == State::kUncertain) return RemoteControlStep::kUncertain;
    if (state_ != State::kPending) return RemoteControlStep::kRejected;
    try {
        const auto progress = endpoint_->poll(ticket_, response_line_, now);
        if (progress == OwnedExchangeProgress::kPending) return RemoteControlStep::kPending;
        if (progress == OwnedExchangeProgress::kResponse) {
            bool has_result = false;
            if (is_native_) {
                const auto parsed = parse_native_epoch_response(response_line_);
                has_result = parsed && decode_native(*parsed);
            } else {
                const auto parsed = parse_task_execution_protocol_response(response_line_);
                has_result = parsed.parsed() && decode_cooperative(*parsed.response);
            }
            if (has_result) {
                if (has_fallback_ && has_completed_ && !secondary_line_.empty()) {
                    // This is a distinct fresh request after the actual primary
                    // ACK, never replay of a published combined request.
                    auto secondary = std::move(secondary_line_);
                    secondary_line_.clear();
                    return publish(secondary, now);
                }
                state_ = State::kFinished;
                return RemoteControlStep::kFinished;
            }
        }
    } catch (...) {
        // Decode/variant failure cannot escape an owner progress boundary or
        // turn a partly assigned result into authorization/acknowledgement.
        quarantine();
        return RemoteControlStep::kUncertain;
    }
    quarantine();
    return RemoteControlStep::kUncertain;
}

bool RemoteControlOperation::decode_cooperative(const TaskExecutionProtocolResponse& response) {
    const auto operation = task_request_.operation;
    const TaskExecutionIdentity expected{task_request_.task_id, task_request_.generation};
    if (response.kind == TaskExecutionProtocolResponseKind::kError)
        return response.error != TaskProtocolErrorCode::kInternalError;
    switch (operation) {
        case TaskExecutionProtocolOperation::kSubmit:
            if (response.kind != TaskExecutionProtocolResponseKind::kAccepted ||
                response.task_id == 0)
                return false;
            result_ = std::optional{response.task_id};
            return true;
        case TaskExecutionProtocolOperation::kExecutionAcquire: {
            const auto decoded = submission(response);
            if (!decoded) return false;
            result_ = *decoded;
            return true;
        }
        case TaskExecutionProtocolOperation::kExecutionCompleteAcquire: {
            if (has_fallback_) {
                if (!has_completed_) {
                    if (!acknowledged(response, expected, TaskProtocolState::kCompleted))
                        return false;
                    has_completed_ = true;
                    result_ = CooperativeHandoff{.is_completed = true};
                    return true;
                }
                const auto decoded = submission(response);
                if (!decoded) return false;
                result_ = CooperativeHandoff{true, *decoded};
                return true;
            }
            if (response.kind != TaskExecutionProtocolResponseKind::kExecutionHandoff ||
                !matching_execution(response.completed_execution, expected))
                return false;
            has_completed_ = true;
            CooperativeHandoff decoded{.is_completed = true};
            decoded.submission.task_id = response.task_id;
            if (response.task_id != 0) {
                decoded.submission.claim.status = response.generation != 0
                                                      ? CooperativeClaimStatus::kGranted
                                                      : CooperativeClaimStatus::kPending;
                if (response.generation != 0)
                    decoded.submission.claim.execution = {response.task_id, response.generation};
            }
            result_ = decoded;
            return true;
        }
        case TaskExecutionProtocolOperation::kExecutionClaim:
            if (response.kind == TaskExecutionProtocolResponseKind::kEmpty) {
                result_ = CooperativeClaim{.status = CooperativeClaimStatus::kPending};
                return true;
            }
            if (response.kind != TaskExecutionProtocolResponseKind::kExecutionLease ||
                response.task_id != expected.task_id || response.generation == 0)
                return false;
            result_ = CooperativeClaim{CooperativeClaimStatus::kGranted,
                                       {response.task_id, response.generation}};
            return true;
        case TaskExecutionProtocolOperation::kRetreat:
            if (response.kind != TaskExecutionProtocolResponseKind::kRetreat ||
                response.task_id != expected.task_id || response.generation != expected.generation)
                return false;
            result_ = std::optional{response.should_retreat};
            return true;
        case TaskExecutionProtocolOperation::kExecutionHeartbeat:
            if (!acknowledged(response, expected, TaskProtocolState::kRunning) &&
                !acknowledged(response, expected, TaskProtocolState::kQueued))
                return false;
            result_ = true;
            return true;
        case TaskExecutionProtocolOperation::kSuspend:
        case TaskExecutionProtocolOperation::kExecutionComplete:
        case TaskExecutionProtocolOperation::kExecutionFail:
        case TaskExecutionProtocolOperation::kExecutionCancel:
        case TaskExecutionProtocolOperation::kCancel: {
            const auto state = operation == TaskExecutionProtocolOperation::kSuspend
                                   ? TaskProtocolState::kQueued
                               : operation == TaskExecutionProtocolOperation::kExecutionComplete
                                   ? TaskProtocolState::kCompleted
                               : operation == TaskExecutionProtocolOperation::kExecutionFail
                                   ? TaskProtocolState::kFailed
                                   : TaskProtocolState::kCancelled;
            if (!acknowledged(response, expected, state)) return false;
            result_ = true;
            return true;
        }
        default:
            return false;
    }
}

bool RemoteControlOperation::decode_native(const NativeEpochResponse& response) {
    if (response.request_sequence != epoch_request_.request_sequence ||
        ((response.kind == NativeEpochResponseKind::kAcknowledged ||
          response.kind == NativeEpochResponseKind::kRetreat) &&
         response.operation != epoch_request_.operation))
        return false;
    const auto operation = epoch_request_.operation;
    // Native errors do not encode whether admission was published.
    if (response.kind == NativeEpochResponseKind::kError) return false;
    if (operation == NativeEpochOperation::kAcquire || operation == NativeEpochOperation::kClaim) {
        const auto grant = response.grant;
        if ((response.kind != NativeEpochResponseKind::kPending &&
             response.kind != NativeEpochResponseKind::kGranted) ||
            grant.extent != expected_epoch_.extent ||
            (expected_epoch_.execution.task_id != 0 &&
             grant.execution.task_id != expected_epoch_.execution.task_id))
            return false;
        const bool is_pending = response.kind == NativeEpochResponseKind::kPending;
        if (!valid_epoch_grant(epoch_policy_, grant, is_pending)) return false;
        if (!is_pending) {
            const auto count = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                   Clock::now().time_since_epoch())
                                   .count();
            const auto now = count > 0 ? static_cast<std::uint64_t>(count) : 0;
            const auto duration =
                static_cast<std::uint64_t>(grant.extent.acceptance_microseconds) * 1000;
            const auto latest = duration > std::numeric_limits<std::uint64_t>::max() - now
                                    ? std::numeric_limits<std::uint64_t>::max()
                                    : now + duration;
            if (grant.acceptance_deadline_nanoseconds > latest) return false;
        }
        result_ = NativeEpochClaim{
            is_pending ? NativeEpochClaimStatus::kPending : NativeEpochClaimStatus::kGranted,
            grant};
        return true;
    }
    if (!matching_execution(response.grant.execution, expected_epoch_.execution)) return false;
    if (operation == NativeEpochOperation::kRetreat) {
        if (response.kind != NativeEpochResponseKind::kRetreat) return false;
        result_ = std::optional{response.should_retreat};
        return true;
    }
    if (response.kind != NativeEpochResponseKind::kAcknowledged ||
        response.operation != operation || response.issued_calls != epoch_request_.issued_calls)
        return false;
    result_ = true;
    return true;
}

std::optional<OwnedExchangeWait> RemoteControlOperation::wait_state() const noexcept {
    if (state_ != State::kPending || process_id != ::getpid()) return std::nullopt;
    return endpoint_->wait_state(ticket_);
}

std::optional<RemoteControlResult> RemoteControlOperation::take_result() noexcept {
    if (process_id != ::getpid() || is_consumed_ ||
        (state_ != State::kFinished && state_ != State::kUncertain))
        return std::nullopt;
    is_consumed_ = true;
    return result_;
}

bool RemoteControlOperation::was_published() const noexcept {
    return has_publication_;
}

bool RemoteControlOperation::reject_unpublished() noexcept {
    if (process_id != ::getpid() || state_ != State::kPrepared) return false;
    state_ = State::kFinished;
    return true;
}

bool RemoteControlOperation::is_prepared() const noexcept {
    return process_id == ::getpid() && state_ == State::kPrepared;
}

bool RemoteControlOperation::completion_acknowledged() const noexcept {
    return has_completed_ && process_id == ::getpid();
}

}  // namespace glimmer::control::internal
