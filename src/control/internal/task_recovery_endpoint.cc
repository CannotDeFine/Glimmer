#include "task_recovery_endpoint.h"

#include <algorithm>
#include <span>
#include <string>

namespace glimmer::control::internal {
namespace {

[[nodiscard]] bool register_task(void*, core::TaskId) noexcept {
    return true;
}

[[nodiscard]] TaskRecoveryResponse rejection(const TaskRecoveryRequest& request,
                                             TaskRecoveryToken incarnation,
                                             TaskRecoveryError error) noexcept {
    return {.kind = TaskRecoveryResultKind::kRejected,
            .session_id = request.session_id,
            .operation_id = request.operation_id,
            .nonce = request.nonce,
            .incarnation = incarnation,
            .error = error};
}

[[nodiscard]] std::span<const std::byte> bytes(std::string_view value) noexcept {
    return std::as_bytes(std::span(value.data(), value.size()));
}

}  // namespace

TaskRecoveryEndpoint::TaskRecoveryEndpoint(TaskRecoveryToken incarnation,
                                           core::MemoryBytes memory_limit,
                                           core::SchedulerOptions options)
    : incarnation(incarnation),
      journal_(incarnation, {}),
      admission_(memory_limit, options, std::chrono::milliseconds::zero(), true) {}

TaskRecoveryEndpoint::SessionSlot* TaskRecoveryEndpoint::find_session(
    const TaskRecoveryRequest& request, TaskPeerIdentity peer) noexcept {
    for (auto& slot : sessions_) {
        if (slot.used && slot.session.id == request.session_id &&
            slot.session.nonce == request.nonce && slot.session.peer == peer &&
            slot.session.incarnation == request.incarnation) {
            return &slot;
        }
    }
    return nullptr;
}

TaskRecoveryEndpoint::Binding* TaskRecoveryEndpoint::find_binding(std::uint64_t session_id,
                                                                  core::TaskId task_id) noexcept {
    for (auto& binding : bindings_) {
        if (binding.state != BindingState::kFree && binding.session_id == session_id &&
            binding.execution.task_id == task_id) {
            return &binding;
        }
    }
    return nullptr;
}

void TaskRecoveryEndpoint::reclaim_forgotten(std::uint64_t session_id,
                                             std::uint64_t through) noexcept {
    for (auto& binding : bindings_) {
        if (binding.state == BindingState::kForgotten && binding.session_id == session_id &&
            binding.forget_operation <= through) {
            binding = {};
        }
    }
}

TaskRecoveryResponse TaskRecoveryEndpoint::execute(
    const TaskRecoveryRequest& request, TaskPeerIdentity peer,
    const TaskAdmissionRequest* prepared_submission) {
    auto result = rejection(request, incarnation, TaskRecoveryError::kInvalid);
    if (request.operation == TaskRecoveryOperation::kStatus) {
        result.kind = TaskRecoveryResultKind::kStatus;
        result.error = TaskRecoveryError::kNone;
        for (const auto& binding : bindings_) {
            if (binding.session_id != request.session_id) continue;
            result.pending_tasks += binding.state == BindingState::kPending ? 1U : 0U;
            result.active_tasks += binding.state == BindingState::kActive ? 1U : 0U;
        }
        return result;
    }
    if (request.operation == TaskRecoveryOperation::kSubmit) {
        auto free = std::find_if(bindings_.begin(), bindings_.end(),
                                 [](const Binding& b) { return b.state == BindingState::kFree; });
        if (free == bindings_.end()) {
            result.error = TaskRecoveryError::kCapacity;
            return result;
        }
        *free = {.state = BindingState::kReserved, .session_id = request.session_id};
        if (prepared_submission == nullptr) {
            uncertain_ = true;
            return result;
        }
        free->state = BindingState::kSubmitting;
        const auto submitted =
            admission_.submit_cooperative(*prepared_submission, register_task, nullptr, peer);
        if (!submitted.accepted()) {
            if (submitted.task_id != 0 && !admission_.forget_terminal(submitted.task_id)) {
                uncertain_ = true;
                return result;
            }
            *free = {};
            switch (submitted.status) {
                case core::SubmitStatus::kQuotaExceeded:
                    result.error = TaskRecoveryError::kQuota;
                    break;
                case core::SubmitStatus::kQueueFull:
                    result.error = TaskRecoveryError::kCapacity;
                    break;
                case core::SubmitStatus::kInvalidTask:
                case core::SubmitStatus::kTenantWeightMismatch:
                    result.error = TaskRecoveryError::kInvalid;
                    break;
                case core::SubmitStatus::kAccepted:
                case core::SubmitStatus::kInternalError:
                    result.error = TaskRecoveryError::kInternal;
                    break;
            }
            return result;
        }
        free->state = BindingState::kPending;
        free->execution.task_id = submitted.task_id;
        result.kind = TaskRecoveryResultKind::kAccepted;
        result.error = TaskRecoveryError::kNone;
        result.execution.task_id = submitted.task_id;
        return result;
    }
    auto* binding = find_binding(request.session_id, request.execution.task_id);
    if (binding == nullptr || binding->state == BindingState::kReserved ||
        binding->state == BindingState::kSubmitting || binding->state == BindingState::kForgotten) {
        result.error = TaskRecoveryError::kUnknownTask;
        return result;
    }
    if (request.operation == TaskRecoveryOperation::kClaim) {
        if (binding->state != BindingState::kPending) return result;
        const auto claimed = admission_.claim_execution_detailed(binding->execution.task_id, peer);
        if (claimed.status == core::TaskDispatchStatus::kQueued) {
            result.kind = TaskRecoveryResultKind::kPending;
            result.error = TaskRecoveryError::kNone;
            result.execution.task_id = binding->execution.task_id;
            return result;
        }
        if (claimed.status == core::TaskDispatchStatus::kTerminalClean) {
            binding->state = BindingState::kTerminal;
            result.error = TaskRecoveryError::kInternal;
            return result;
        }
        if (claimed.status != core::TaskDispatchStatus::kGranted || !claimed.task) {
            uncertain_ = true;
            return result;
        }
        binding->state = BindingState::kActive;
        binding->execution.generation = claimed.task->dispatch_sequence;
        result.kind = TaskRecoveryResultKind::kGranted;
        result.error = TaskRecoveryError::kNone;
        result.execution = binding->execution;
        return result;
    }
    if (request.operation == TaskRecoveryOperation::kForget) {
        if (binding->state != BindingState::kTerminal ||
            !admission_.forget_terminal(binding->execution.task_id))
            return result;
        binding->state = BindingState::kForgotten;
        binding->forget_operation = request.operation_id;
    } else if (request.operation == TaskRecoveryOperation::kCancelPending) {
        if (binding->state != BindingState::kPending) return result;
        const bool cancelled =
            binding->execution.generation == 0
                ? admission_.cancel_pending_cooperative(binding->execution.task_id, peer)
                : admission_.cancel_execution(binding->execution, peer);
        if (!cancelled) return result;
        binding->state = BindingState::kTerminal;
    } else {
        const bool suspended_heartbeat = request.operation == TaskRecoveryOperation::kHeartbeat &&
                                         binding->state == BindingState::kPending &&
                                         binding->execution.generation != 0;
        if ((binding->state != BindingState::kActive && !suspended_heartbeat) ||
            binding->execution.generation != request.execution.generation)
            return result;
        if (request.operation == TaskRecoveryOperation::kRetreat) {
            const auto retreat = admission_.retreat_requested(binding->execution, peer);
            if (!retreat) return result;
            result.kind = TaskRecoveryResultKind::kRetreat;
            result.error = TaskRecoveryError::kNone;
            result.execution = binding->execution;
            result.should_retreat = *retreat;
            return result;
        }
        bool success = false;
        switch (request.operation) {
            case TaskRecoveryOperation::kHeartbeat:
                success = admission_.heartbeat_execution(binding->execution, peer);
                break;
            case TaskRecoveryOperation::kSuspend:
                success = admission_.suspend_execution(binding->execution, peer);
                if (success) binding->state = BindingState::kPending;
                break;
            case TaskRecoveryOperation::kComplete:
                success = admission_.complete_execution(binding->execution, peer);
                if (success) binding->state = BindingState::kTerminal;
                break;
            case TaskRecoveryOperation::kFail:
                success = admission_.fail_execution(binding->execution, peer);
                if (success) binding->state = BindingState::kTerminal;
                break;
            default:
                return result;
        }
        if (!success) return result;
    }
    result.kind = TaskRecoveryResultKind::kAcknowledged;
    result.error = TaskRecoveryError::kNone;
    return result;
}

std::optional<TaskRecoveryLine> TaskRecoveryEndpoint::handle(std::string_view line,
                                                             TaskPeerIdentity peer) noexcept {
    try {
        const auto parsed = parse_task_recovery_request(line);
        if (!parsed || peer.pid == 0 || peer.start_time_ticks == 0) return std::nullopt;
        const auto& request = *parsed;
        std::scoped_lock lock(mutex_);
        try {
            if (stopped_ || uncertain_ || incarnation == TaskRecoveryToken{}) return std::nullopt;
            if (request.command == TaskRecoveryCommand::kOpen) {
                auto free = std::find_if(sessions_.begin(), sessions_.end(),
                                         [](const SessionSlot& slot) { return !slot.used; });
                const auto existing =
                    std::find_if(sessions_.begin(), sessions_.end(), [&](const SessionSlot& slot) {
                        return slot.used && slot.session.nonce == request.nonce &&
                               slot.session.peer == peer;
                    });
                if (free == sessions_.end() && existing == sessions_.end())
                    return format_task_recovery_response(
                        rejection(request, incarnation, TaskRecoveryError::kCapacity));
                const auto session = journal_.open(request.nonce, peer);
                if (!session)
                    return format_task_recovery_response(
                        rejection(request, incarnation, TaskRecoveryError::kUnauthorized));
                auto& slot = existing == sessions_.end() ? *free : *existing;
                slot = {.used = true, .session = *session};
                last_session_id_ = std::max(last_session_id_, session->id);
                return format_task_recovery_response({.kind = TaskRecoveryResultKind::kOpened,
                                                      .session_id = session->id,
                                                      .nonce = session->nonce,
                                                      .incarnation = incarnation});
            }
            // Reject stale incarnation before journal, ACK, mapping or admission.
            if (request.incarnation != incarnation) return std::nullopt;
            auto* session = find_session(request, peer);
            if (session == nullptr) {
                if (request.command != TaskRecoveryCommand::kClose ||
                    request.session_id > last_session_id_)
                    return std::nullopt;
                // A close reply may be lost after the session is removed.
                // CLOSED means absence, not historical ownership or physical stop.
                // Never report closed for a live ID owned by another session.
                for (const auto& live : sessions_) {
                    if (live.used && live.session.id == request.session_id) return std::nullopt;
                }
                return format_task_recovery_response({.kind = TaskRecoveryResultKind::kClosed,
                                                      .session_id = request.session_id,
                                                      .nonce = request.nonce,
                                                      .incarnation = incarnation});
            }
            if (request.command == TaskRecoveryCommand::kClose) {
                for (const auto& binding : bindings_) {
                    if (binding.state != BindingState::kFree &&
                        binding.session_id == request.session_id)
                        return format_task_recovery_response(
                            rejection(request, incarnation, TaskRecoveryError::kNotReady));
                }
                const auto response =
                    format_task_recovery_response({.kind = TaskRecoveryResultKind::kClosed,
                                                   .session_id = request.session_id,
                                                   .nonce = request.nonce,
                                                   .incarnation = incarnation});
                if (!response || !journal_.close(session->session)) return std::nullopt;
                *session = {};
                return response;
            }
            std::optional<TaskAdmissionRequest> prepared_submission;
            if (request.command == TaskRecoveryCommand::kExecute &&
                request.operation == TaskRecoveryOperation::kSubmit) {
                try {
                    prepared_submission.emplace(
                        TaskAdmissionRequest{.tenant_id = std::string(request.tenant),
                                             .memory_bytes = request.memory_bytes,
                                             .weight = request.weight,
                                             .work_units = request.work_units,
                                             .priority = request.priority});
                } catch (...) {
                    return std::nullopt;
                }
            }
            if (!journal_.acknowledge(session->session, request.acknowledged_through))
                return std::nullopt;
            reclaim_forgotten(request.session_id, request.acknowledged_through);
            if (request.command == TaskRecoveryCommand::kAcknowledge) {
                return format_task_recovery_response(
                    {.kind = TaskRecoveryResultKind::kAcknowledged,
                     .session_id = request.session_id,
                     .nonce = request.nonce,
                     .incarnation = incarnation,
                     .acknowledged_through = request.acknowledged_through});
            }
            const auto key = format_task_recovery_journal_key(request);
            if (!key) return std::nullopt;
            const auto begun = journal_.begin(session->session, request.operation_id,
                                              request.acknowledged_through, bytes(key->view()));
            if (begun.decision == TaskRecoveryJournal::Decision::kReplay) {
                TaskRecoveryLine replay;
                replay.size = begun.result_size;
                std::copy_n(reinterpret_cast<const char*>(begun.result.data()), replay.size,
                            replay.bytes.begin());
                return replay;
            }
            // No journal entry means no definitive response can be committed.
            // The caller must retain and retry this same operation identity.
            if (begun.decision == TaskRecoveryJournal::Decision::kFull) return std::nullopt;
            if (begun.decision != TaskRecoveryJournal::Decision::kExecute) return std::nullopt;
            const auto response = format_task_recovery_response(
                execute(request, peer, prepared_submission ? &*prepared_submission : nullptr));
            if (uncertain_ || !response ||
                !journal_.commit(session->session, request.operation_id, bytes(response->view()))) {
                uncertain_ = true;
                return std::nullopt;
            }
            return response;
        } catch (...) {
            // Fence later callers before releasing the endpoint lock.
            uncertain_ = true;
            return std::nullopt;
        }
    } catch (...) {
        return std::nullopt;
    }
}

bool TaskRecoveryEndpoint::retire_after_backend_stopped() noexcept {
    try {
        std::scoped_lock lock(mutex_);
        stopped_ = true;
        for (auto& binding : bindings_) {
            if (binding.state == BindingState::kFree || binding.state == BindingState::kForgotten)
                continue;
            if (binding.state == BindingState::kReserved) {
                binding = {};
                continue;
            }
            if (binding.state == BindingState::kSubmitting) return false;
            const auto session = std::find_if(
                sessions_.begin(), sessions_.end(),
                [&](const SessionSlot& s) { return s.used && s.session.id == binding.session_id; });
            if (session == sessions_.end()) return false;
            const auto peer = session->session.peer;
            if (binding.state == BindingState::kPending) {
                const bool cancelled =
                    binding.execution.generation == 0
                        ? admission_.cancel_pending_cooperative(binding.execution.task_id, peer)
                        : admission_.cancel_execution(binding.execution, peer);
                if (!cancelled) return false;
                binding.state = BindingState::kTerminal;
            } else if (binding.state == BindingState::kActive) {
                if (!admission_.fail_execution(binding.execution, peer)) return false;
                binding.state = BindingState::kTerminal;
            }
            if (!admission_.forget_terminal(binding.execution.task_id)) return false;
            binding = {};
        }
        return true;
    } catch (...) {
        return false;
    }
}

}  // namespace glimmer::control::internal
