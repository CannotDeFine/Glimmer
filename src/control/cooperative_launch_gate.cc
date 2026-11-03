#include "glimmer/control/cooperative_launch_gate.h"

#include "glimmer/control/native_epoch_protocol.h"
#include "glimmer/control/shared_task_control_client.h"
#include "glimmer/control/task_execution_admission.h"
#include "glimmer/control/task_execution_protocol.h"
#include "internal/process_identity.h"
#include "internal/control_request_sequence.h"

#include <algorithm>
#include <atomic>
#include <limits>
#include <mutex>
#include <utility>

namespace glimmer::control {
namespace {
bool registered(void*, core::TaskId) noexcept {
    return true;
}

std::uint64_t monotonic_nanoseconds() noexcept {
    const auto count = std::chrono::duration_cast<std::chrono::nanoseconds>(
                           std::chrono::steady_clock::now().time_since_epoch())
                           .count();
    return count > 0 ? static_cast<std::uint64_t>(count) : 0;
}

bool matching_execution(TaskExecutionIdentity first, TaskExecutionIdentity second) noexcept {
    return first.task_id == second.task_id && first.generation == second.generation;
}

bool matches_reply(const TaskExecutionProtocolRequest& request,
                   const TaskExecutionProtocolResponse& response) noexcept {
    using Op = TaskExecutionProtocolOperation;
    using Kind = TaskExecutionProtocolResponseKind;
    if (response.kind == Kind::kError) {
        return response.error != TaskProtocolErrorCode::kInternalError;
    }
    switch (request.operation) {
        case Op::kSubmit:
            return response.kind == Kind::kAccepted && response.task_id != 0;
        case Op::kExecutionAcquire:
            return (response.kind == Kind::kAccepted && response.task_id != 0) ||
                   (response.kind == Kind::kExecutionLease && response.task_id != 0 &&
                    response.generation != 0);
        case Op::kExecutionClaim:
            return response.kind == Kind::kEmpty ||
                   (response.kind == Kind::kState && response.task_id == request.task_id) ||
                   (response.kind == Kind::kExecutionLease && response.task_id == request.task_id &&
                    response.generation != 0);
        case Op::kRetreat:
            return response.kind == Kind::kRetreat && response.task_id == request.task_id &&
                   response.generation == request.generation;
        case Op::kExecutionCompleteAcquire:
            return response.kind == Kind::kExecutionHandoff &&
                   response.completed_execution.task_id == request.task_id &&
                   response.completed_execution.generation == request.generation;
        case Op::kCancel:
        case Op::kSuspend:
        case Op::kExecutionHeartbeat:
        case Op::kExecutionComplete:
        case Op::kExecutionFail:
        case Op::kExecutionCancel:
            return response.kind == Kind::kState && response.task_id == request.task_id;
        default:
            return false;
    }
}

}  // namespace

struct CooperativeLaunchGate::Implementation {
    TaskExecutionAdmissionService admission;
    TaskAdmissionRequest request;
    const std::optional<TaskPeerIdentity> peer{current_process_identity()};
    const bool is_remote;
    std::unique_ptr<SharedTaskControlClient> client;
    mutable std::mutex remote_mutex;
    mutable std::atomic_bool remote_uncertain{false};
    const std::optional<NativeEpochExtent> epoch_policy;

    Implementation(LaunchGateOptions options, std::optional<NativeEpochExtent> policy)
        : admission(std::numeric_limits<core::MemoryBytes>::max(),
                    {.max_running_tasks = options.max_concurrent_launches,
                     .max_queued_tasks = 65536,
                     .priority_reserved_slots = options.priority_reserved_slots,
                     .priority_reservation_threshold = options.priority_reservation_threshold,
                     .scheduling_policy = options.scheduling_policy},
                    std::chrono::milliseconds::zero(), true, nullptr, nullptr, nullptr, nullptr,
                    policy),
          request{.tenant_id = options.tenant_id.empty() ? "default" : std::move(options.tenant_id),
                  .memory_bytes = 1,
                  .weight = options.tenant_weight,
                  .work_units = 1,
                  .priority = options.task_priority},
          is_remote(!options.control_socket.empty()),
          epoch_policy(policy) {
        if (is_remote) {
            client = std::make_unique<SharedTaskControlClient>(std::move(options.control_socket));
        }
    }
    bool allows_epoch(NativeEpochExtent extent) const noexcept {
        return epoch_policy && valid_native_epoch_extent(*epoch_policy) &&
               valid_native_epoch_extent(extent) && extent.max_calls <= epoch_policy->max_calls &&
               extent.calls_per_prefix <= epoch_policy->calls_per_prefix &&
               extent.max_pending_prefixes <= epoch_policy->max_pending_prefixes &&
               extent.acceptance_microseconds <= epoch_policy->acceptance_microseconds;
    }
    bool valid_epoch_grant(NativeEpochGrant grant, bool is_pending = false) const noexcept {
        return allows_epoch(grant.extent) && grant.execution.task_id != 0 &&
               (is_pending
                    ? grant.execution.generation == 0 && grant.acceptance_deadline_nanoseconds == 0
                    : grant.execution.generation != 0 &&
                          grant.acceptance_deadline_nanoseconds != 0);
    }
    bool valid_epoch_reply(const NativeEpochRequest& request, const NativeEpochGrant& expected,
                           const NativeEpochResponse& response) const noexcept {
        // A native endpoint can return its default error after admission was
        // published. Until errors carry a transactional outcome, none proves
        // that ownership was rejected or retired.
        if (response.kind == NativeEpochResponseKind::kError) return false;
        if (request.operation == NativeEpochOperation::kAcquire ||
            request.operation == NativeEpochOperation::kClaim) {
            const auto grant = response.grant;
            const bool pending = response.kind == NativeEpochResponseKind::kPending;
            if ((!pending && response.kind != NativeEpochResponseKind::kGranted) ||
                grant.extent != expected.extent ||
                (expected.execution.task_id != 0 &&
                 grant.execution.task_id != expected.execution.task_id) ||
                !valid_epoch_grant(grant, pending))
                return false;
            if (!pending) {
                const auto now = monotonic_nanoseconds();
                const auto duration =
                    static_cast<std::uint64_t>(grant.extent.acceptance_microseconds) * 1000;
                const auto latest = duration > std::numeric_limits<std::uint64_t>::max() - now
                                        ? std::numeric_limits<std::uint64_t>::max()
                                        : now + duration;
                if (grant.acceptance_deadline_nanoseconds > latest) return false;
            }
            return true;
        }
        if (!matching_execution(response.grant.execution, expected.execution)) return false;
        if (request.operation == NativeEpochOperation::kRetreat) {
            return response.kind == NativeEpochResponseKind::kRetreat;
        }
        return response.kind == NativeEpochResponseKind::kAcknowledged &&
               response.issued_calls == request.issued_calls;
    }
    std::optional<NativeEpochResponse> epoch_exchange(const NativeEpochRequest& message,
                                                      NativeEpochGrant expected) const {
        auto correlated = message;
        correlated.request_sequence = internal::next_epoch_request_sequence();
        const auto encoded = format_native_epoch_request(correlated);
        if (!client || !encoded) {
            return std::nullopt;
        }
        std::lock_guard lock(remote_mutex);
        if (remote_uncertain.load()) {
            return std::nullopt;
        }
        const auto response = client->request(*encoded);
        const auto parsed = response ? parse_native_epoch_response(*response) : std::nullopt;
        if (!parsed || parsed->request_sequence != correlated.request_sequence ||
            parsed->operation != correlated.operation ||
            !valid_epoch_reply(correlated, expected, *parsed)) {
            remote_uncertain.store(true);
            return std::nullopt;
        }
        return parsed;
    }
    NativeEpochClaim epoch_claim_response(const std::optional<NativeEpochResponse>& response,
                                          NativeEpochGrant expected) const {
        const NativeEpochClaim uncertain{NativeEpochClaimStatus::kUncertain, expected};
        if (!response) {
            return uncertain;
        }
        if (response->kind == NativeEpochResponseKind::kError && expected.execution.task_id == 0 &&
            (response->error == TaskProtocolErrorCode::kInvalidRequest ||
             response->error == TaskProtocolErrorCode::kQuotaExceeded ||
             response->error == TaskProtocolErrorCode::kQueueFull)) {
            return {};
        }
        const auto grant = response->grant;
        if ((response->kind != NativeEpochResponseKind::kPending &&
             response->kind != NativeEpochResponseKind::kGranted) ||
            grant.extent != expected.extent ||
            (expected.execution.task_id != 0 &&
             grant.execution.task_id != expected.execution.task_id)) {
            return uncertain;
        }
        const bool pending = response->kind == NativeEpochResponseKind::kPending;
        if (!valid_epoch_grant(grant, pending)) {
            return uncertain;
        }
        if (!pending) {
            const auto now = monotonic_nanoseconds();
            const auto duration =
                static_cast<std::uint64_t>(grant.extent.acceptance_microseconds) * 1000;
            const auto latest = duration > std::numeric_limits<std::uint64_t>::max() - now
                                    ? std::numeric_limits<std::uint64_t>::max()
                                    : now + duration;
            if (grant.acceptance_deadline_nanoseconds > latest) {
                return uncertain;
            }
        }
        return {pending ? NativeEpochClaimStatus::kPending : NativeEpochClaimStatus::kGranted,
                grant};
    }
    bool epoch_report(NativeEpochGrant grant, NativeEpochOperation operation,
                      std::uint32_t issued_calls = 0) {
        const bool pending = operation == NativeEpochOperation::kCancelPending;
        if (!valid_epoch_grant(grant, pending) || issued_calls > grant.extent.max_calls) {
            return false;
        }
        if (is_remote) {
            const auto response = epoch_exchange({.operation = operation,
                                                  .admission = {},
                                                  .extent = {},
                                                  .execution = grant.execution,
                                                  .issued_calls = issued_calls},
                                                 grant);
            return response && response->kind == NativeEpochResponseKind::kAcknowledged &&
                   response->operation == operation &&
                   matching_execution(response->grant.execution, grant.execution) &&
                   response->issued_calls == issued_calls;
        }
        if (operation == NativeEpochOperation::kHeartbeat) {
            return admission.heartbeat_native_epoch(grant.execution, peer);
        }
        const bool updated =
            pending ? admission.cancel_pending_native_epoch(grant.execution.task_id, peer)
                    : admission.finish_native_epoch(grant.execution, issued_calls,
                                                    operation == NativeEpochOperation::kFail, peer);
        return updated && admission.forget_terminal(grant.execution.task_id);
    }
    std::optional<TaskExecutionProtocolResponse> exchange(
        const TaskExecutionProtocolRequest& message) const {
        const auto encoded = format_task_execution_protocol_request(message);
        if (!client || !encoded) {
            return std::nullopt;
        }
        return exchange_encoded(message, *encoded);
    }
    std::optional<TaskExecutionProtocolResponse> exchange_encoded(
        const TaskExecutionProtocolRequest& request, std::string_view encoded) const {
        std::lock_guard lock(remote_mutex);
        if (!client || remote_uncertain.load()) {
            return std::nullopt;
        }
        const auto text = client->request(encoded);
        if (!text) {
            remote_uncertain.store(true);
            return std::nullopt;
        }
        auto parsed = parse_task_execution_protocol_response(*text);
        if (!parsed.parsed() || !parsed.response || !matches_reply(request, *parsed.response)) {
            remote_uncertain.store(true);
            return std::nullopt;
        }
        return std::move(parsed.response);
    }
    bool transition_unchecked(TaskExecutionIdentity execution,
                              TaskExecutionProtocolOperation operation,
                              TaskProtocolState expected_state) {
        if (execution.task_id == 0 || execution.generation == 0) {
            return false;
        }
        if (is_remote) {
            const auto response = exchange({.operation = operation,
                                            .admission = {},
                                            .task_id = execution.task_id,
                                            .generation = execution.generation});
            return response && response->kind == TaskExecutionProtocolResponseKind::kState &&
                   response->task_id == execution.task_id &&
                   (response->state == expected_state ||
                    (operation == TaskExecutionProtocolOperation::kExecutionHeartbeat &&
                     response->state == TaskProtocolState::kQueued));
        }
        bool updated = false;
        switch (operation) {
            case TaskExecutionProtocolOperation::kSuspend:
                return admission.suspend_execution(execution, peer);
            case TaskExecutionProtocolOperation::kExecutionHeartbeat:
                return admission.heartbeat_execution(execution, peer);
            case TaskExecutionProtocolOperation::kExecutionComplete:
                updated = admission.complete_execution(execution, peer);
                break;
            case TaskExecutionProtocolOperation::kExecutionFail:
                updated = admission.fail_execution(execution, peer);
                break;
            case TaskExecutionProtocolOperation::kExecutionCancel:
                updated = admission.cancel_execution(execution, peer);
                break;
            default:
                return false;
        }
        return updated && admission.forget_terminal(execution.task_id);
    }
    bool transition(TaskExecutionIdentity execution, TaskExecutionProtocolOperation operation,
                    TaskProtocolState expected_state) noexcept {
        try {
            return transition_unchecked(execution, operation, expected_state);
        } catch (...) {
            return false;
        }
    }
};

CooperativeLaunchGate::CooperativeLaunchGate(LaunchGateOptions options,
                                             std::optional<NativeEpochExtent> native_epoch_policy)
    : implementation_(std::make_unique<Implementation>(std::move(options), native_epoch_policy)) {}
CooperativeLaunchGate::~CooperativeLaunchGate() = default;

NativeEpochClaim CooperativeLaunchGate::acquire_native_epoch(
    NativeEpochLaunchRequest settings) noexcept {
    NativeEpochClaim result;
    try {
        auto& state = *implementation_;
        if (!state.allows_epoch(settings.extent)) {
            return result;
        }
        auto request = state.request;
        request.work_units = settings.extent.max_calls;
        request.priority += std::min(settings.relative_priority,
                                     std::numeric_limits<std::uint32_t>::max() - request.priority);
        if (state.is_remote) {
            return state.epoch_claim_response(
                state.epoch_exchange({.operation = NativeEpochOperation::kAcquire,
                                      .admission = std::move(request),
                                      .extent = settings.extent,
                                      .execution = {}},
                                     {.execution = {}, .extent = settings.extent}),
                {.execution = {}, .extent = settings.extent});
        }
        const auto accepted = state.admission.submit_native_epoch(request, settings.extent,
                                                                  registered, nullptr, state.peer);
        if (!accepted.accepted()) {
            if (accepted.task_id != 0) {
                static_cast<void>(state.admission.forget_terminal(accepted.task_id));
            }
            return result;
        }
        result = {NativeEpochClaimStatus::kUncertain,
                  {.execution = {accepted.task_id, 0}, .extent = settings.extent}};
        return claim_native_epoch(result.grant);
    } catch (...) {
        result.status = NativeEpochClaimStatus::kUncertain;
        return result;
    }
}

NativeEpochClaim CooperativeLaunchGate::claim_native_epoch(NativeEpochGrant pending) noexcept {
    NativeEpochClaim result{NativeEpochClaimStatus::kUncertain, pending};
    try {
        auto& state = *implementation_;
        if (!state.valid_epoch_grant(pending, true)) {
            return result;
        }
        if (state.is_remote) {
            return state.epoch_claim_response(
                state.epoch_exchange({.operation = NativeEpochOperation::kClaim,
                                      .admission = {},
                                      .extent = {},
                                      .execution = pending.execution},
                                     pending),
                pending);
        }
        const auto grant =
            state.admission.claim_native_epoch(pending.execution.task_id, state.peer);
        if (grant && grant->extent == pending.extent) {
            return {NativeEpochClaimStatus::kGranted, *grant};
        }
        const auto extent =
            state.admission.pending_native_epoch_extent(pending.execution.task_id, state.peer);
        return extent && *extent == pending.extent
                   ? NativeEpochClaim{NativeEpochClaimStatus::kPending, pending}
                   : result;
    } catch (...) {
        return result;
    }
}

std::optional<bool> CooperativeLaunchGate::retreat_native_epoch(NativeEpochGrant grant) noexcept {
    try {
        auto& state = *implementation_;
        if (!state.valid_epoch_grant(grant)) {
            return std::nullopt;
        }
        if (!state.is_remote) {
            return state.admission.native_epoch_retreat_requested(grant.execution, state.peer);
        }
        const auto response = state.epoch_exchange({.operation = NativeEpochOperation::kRetreat,
                                                    .admission = {},
                                                    .extent = {},
                                                    .execution = grant.execution},
                                                   grant);
        return response && response->kind == NativeEpochResponseKind::kRetreat &&
                       matching_execution(response->grant.execution, grant.execution)
                   ? std::optional{response->should_retreat}
                   : std::nullopt;
    } catch (...) {
        return std::nullopt;
    }
}

bool CooperativeLaunchGate::heartbeat_native_epoch(NativeEpochGrant grant) noexcept {
    try {
        return implementation_->epoch_report(grant, NativeEpochOperation::kHeartbeat);
    } catch (...) {
        return false;
    }
}

bool CooperativeLaunchGate::finish_native_epoch(NativeEpochGrant grant, std::uint32_t issued_calls,
                                                bool has_failed) noexcept {
    try {
        return implementation_->epoch_report(
            grant, has_failed ? NativeEpochOperation::kFail : NativeEpochOperation::kComplete,
            issued_calls);
    } catch (...) {
        return false;
    }
}

bool CooperativeLaunchGate::cancel_native_epoch(NativeEpochGrant pending) noexcept {
    try {
        return implementation_->epoch_report(pending, NativeEpochOperation::kCancelPending);
    } catch (...) {
        return false;
    }
}

std::optional<core::TaskId> CooperativeLaunchGate::submit(
    CooperativeLaunchRequest settings) noexcept {
    try {
        auto& state = *implementation_;
        auto request = state.request;
        request.priority += std::min(settings.relative_priority,
                                     std::numeric_limits<std::uint32_t>::max() - request.priority);
        request.work_units = settings.work_units;
        if (state.is_remote) {
            const auto response =
                state.exchange({.operation = TaskExecutionProtocolOperation::kSubmit,
                                .admission = std::move(request)});
            return response && response->kind == TaskExecutionProtocolResponseKind::kAccepted &&
                           response->task_id != 0
                       ? std::optional{response->task_id}
                       : std::nullopt;
        }
        const auto admitted =
            state.admission.submit_cooperative(request, registered, nullptr, state.peer);
        if (!admitted.accepted() && admitted.task_id != 0) {
            static_cast<void>(state.admission.forget_terminal(admitted.task_id));
        }
        return admitted.accepted() ? std::optional{admitted.task_id} : std::nullopt;
    } catch (...) {
        return std::nullopt;
    }
}

CooperativeSubmission CooperativeLaunchGate::submit_and_claim(
    CooperativeLaunchRequest settings) noexcept {
    if (!implementation_->is_remote) {
        const auto task = submit(settings);
        return task ? CooperativeSubmission{*task, claim(*task)} : CooperativeSubmission{};
    }
    try {
        auto& state = *implementation_;
        auto request = state.request;
        request.priority += std::min(settings.relative_priority,
                                     std::numeric_limits<std::uint32_t>::max() - request.priority);
        request.work_units = settings.work_units;
        const auto response =
            state.exchange({.operation = TaskExecutionProtocolOperation::kExecutionAcquire,
                            .admission = std::move(request)});
        if (!response || response->task_id == 0) {
            return {};
        }
        if (response->kind == TaskExecutionProtocolResponseKind::kAccepted) {
            return {response->task_id, {.status = CooperativeClaimStatus::kPending}};
        }
        if (response->kind == TaskExecutionProtocolResponseKind::kExecutionLease &&
            response->generation != 0) {
            return {response->task_id,
                    {CooperativeClaimStatus::kGranted, {response->task_id, response->generation}}};
        }
        return {};
    } catch (...) {
        return {};
    }
}

CooperativeHandoff CooperativeLaunchGate::complete_and_submit_and_claim(
    TaskExecutionIdentity execution, CooperativeLaunchRequest settings) noexcept {
    if (execution.task_id == 0 || execution.generation == 0) {
        return {};
    }
    try {
        auto& state = *implementation_;
        if (!state.is_remote) {
            return complete(execution) ? CooperativeHandoff{true, submit_and_claim(settings)}
                                       : CooperativeHandoff{};
        }
        auto request = state.request;
        request.priority += std::min(settings.relative_priority,
                                     std::numeric_limits<std::uint32_t>::max() - request.priority);
        request.work_units = settings.work_units;
        const TaskExecutionProtocolRequest message{
            .operation = TaskExecutionProtocolOperation::kExecutionCompleteAcquire,
            .admission = std::move(request),
            .task_id = execution.task_id,
            .generation = execution.generation};
        // Only a local encoding rejection proves no combined request was sent.
        // Preserve the independent completion for an invalid new submission.
        const auto encoded = format_task_execution_protocol_request(message);
        if (!encoded) {
            return complete(execution) ? CooperativeHandoff{true, submit_and_claim(settings)}
                                       : CooperativeHandoff{};
        }
        const auto parsed = state.exchange_encoded(message, *encoded);
        if (!parsed) {
            return {};
        }
        const auto& response = *parsed;
        if (response.kind != TaskExecutionProtocolResponseKind::kExecutionHandoff ||
            response.completed_execution.task_id != execution.task_id ||
            response.completed_execution.generation != execution.generation) {
            return {};
        }
        CooperativeHandoff result{.is_completed = true};
        result.submission.task_id = response.task_id;
        if (response.task_id != 0) {
            result.submission.claim.status = response.generation != 0
                                                 ? CooperativeClaimStatus::kGranted
                                                 : CooperativeClaimStatus::kPending;
            if (response.generation != 0) {
                result.submission.claim.execution = {response.task_id, response.generation};
            }
        }
        return result;
    } catch (...) {
        return {};
    }
}

CooperativeClaim CooperativeLaunchGate::claim(core::TaskId task) noexcept {
    if (task == 0) {
        return {};
    }
    try {
        auto& state = *implementation_;
        if (state.is_remote) {
            const auto response =
                state.exchange({.operation = TaskExecutionProtocolOperation::kExecutionClaim,
                                .admission = {},
                                .task_id = task});
            if (!response) {
                return {};
            }
            if (response->kind == TaskExecutionProtocolResponseKind::kEmpty) {
                return {.status = CooperativeClaimStatus::kPending};
            }
            // EXEC_LEASE implies running; its wire schema has no STATE field.
            return response->kind == TaskExecutionProtocolResponseKind::kExecutionLease &&
                           response->task_id == task && response->generation != 0
                       ? CooperativeClaim{CooperativeClaimStatus::kGranted,
                                          {task, response->generation}}
                       : CooperativeClaim{};
        }
        const auto snapshot = state.admission.claim_execution(task, state.peer);
        if (snapshot && snapshot->dispatch_sequence != 0) {
            return {CooperativeClaimStatus::kGranted, {task, snapshot->dispatch_sequence}};
        }
        const auto pending = state.admission.find(task);
        return pending && pending->state == core::TaskState::kQueued
                   ? CooperativeClaim{.status = CooperativeClaimStatus::kPending}
                   : CooperativeClaim{};
    } catch (...) {
        return {};
    }
}

std::optional<bool> CooperativeLaunchGate::retreat_requested(
    TaskExecutionIdentity execution) noexcept {
    try {
        auto& state = *implementation_;
        if (!state.is_remote) {
            return state.admission.retreat_requested(execution, state.peer);
        }
        const auto response = state.exchange({.operation = TaskExecutionProtocolOperation::kRetreat,
                                              .admission = {},
                                              .task_id = execution.task_id,
                                              .generation = execution.generation});
        return response && response->kind == TaskExecutionProtocolResponseKind::kRetreat &&
                       response->task_id == execution.task_id &&
                       response->generation == execution.generation
                   ? std::optional{response->should_retreat}
                   : std::nullopt;
    } catch (...) {
        return std::nullopt;
    }
}

bool CooperativeLaunchGate::heartbeat(TaskExecutionIdentity execution) noexcept {
    return implementation_->transition(execution,
                                       TaskExecutionProtocolOperation::kExecutionHeartbeat,
                                       TaskProtocolState::kRunning);
}
bool CooperativeLaunchGate::suspend(TaskExecutionIdentity execution) noexcept {
    return implementation_->transition(execution, TaskExecutionProtocolOperation::kSuspend,
                                       TaskProtocolState::kQueued);
}
bool CooperativeLaunchGate::complete(TaskExecutionIdentity execution) noexcept {
    return implementation_->transition(execution,
                                       TaskExecutionProtocolOperation::kExecutionComplete,
                                       TaskProtocolState::kCompleted);
}
bool CooperativeLaunchGate::fail(TaskExecutionIdentity execution) noexcept {
    return implementation_->transition(execution, TaskExecutionProtocolOperation::kExecutionFail,
                                       TaskProtocolState::kFailed);
}
bool CooperativeLaunchGate::cancel_stopped(TaskExecutionIdentity execution) noexcept {
    return implementation_->transition(execution, TaskExecutionProtocolOperation::kExecutionCancel,
                                       TaskProtocolState::kCancelled);
}

bool CooperativeLaunchGate::cancel_pending(core::TaskId task) noexcept {
    try {
        auto& state = *implementation_;
        if (!state.is_remote) {
            return state.admission.cancel_pending_cooperative(task, state.peer) &&
                   state.admission.forget_terminal(task);
        }
        const auto response = state.exchange({.operation = TaskExecutionProtocolOperation::kCancel,
                                              .admission = {},
                                              .task_id = task});
        return response && response->kind == TaskExecutionProtocolResponseKind::kState &&
               response->task_id == task && response->state == TaskProtocolState::kCancelled;
    } catch (...) {
        return false;
    }
}
bool CooperativeLaunchGate::remote_mode() const noexcept {
    return implementation_->is_remote;
}

bool CooperativeLaunchGate::remote_outcome_uncertain() const noexcept {
    return implementation_->remote_uncertain.load();
}

std::optional<core::SchedulerStats> CooperativeLaunchGate::local_stats() const noexcept {
    try {
        return implementation_->is_remote ? std::nullopt
                                          : std::optional{implementation_->admission.stats()};
    } catch (...) {
        return std::nullopt;
    }
}

}  // namespace glimmer::control
