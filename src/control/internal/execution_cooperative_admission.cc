#include "execution_admission_state.h"

#if defined(GLIMMER_ENABLE_TEST_HOOKS)
#include "dispatch_test_hooks.h"
#include <new>
#endif
#include <limits>

namespace glimmer::control {

std::optional<core::TaskSnapshot> ExecutionAdmissionState::claim_execution(
    core::TaskId task_id, std::optional<TaskPeerIdentity> peer) {
    return claim_execution_detailed(task_id, peer).task;
}

core::TaskDispatchResult ExecutionAdmissionState::claim_execution_detailed(
    core::TaskId task_id, std::optional<TaskPeerIdentity> peer) {
    if (!bind_leases_to_process_) return {};
    std::unique_lock lock(admission_mutex_);
    const auto pending = pending_leases_.find(task_id);
    if (pending == pending_leases_.end() || !pending->second.is_cooperative ||
        pending->second.native_epoch.has_value() || pending->second.reaping ||
        !owner_matches(pending->second, peer)) {
        return {};
    }
    const auto current = scheduler_.find(task_id);
    if (!current || current->state != core::TaskState::kQueued) {
        return {.status = core::TaskDispatchStatus::kResourceUncertain, .task = std::nullopt};
    }
    // Allocate the active record before the scheduler can issue a grant.
    decltype(active_leases_)::iterator active;
    try {
#if defined(GLIMMER_ENABLE_TEST_HOOKS)
        if ((core::internal::dispatch_test_faults & core::internal::kDispatchFailActiveRecord) != 0)
            throw std::bad_alloc();
#endif
        const auto inserted = active_leases_.emplace(task_id, pending->second);
        if (!inserted.second)
            return {.status = core::TaskDispatchStatus::kResourceUncertain, .task = std::nullopt};
        active = inserted.first;
    } catch (...) {
        return {};
    }
    core::TaskDispatchResult result;
    try {
        result = scheduler_.dispatch_task_detailed(task_id);
    } catch (...) {
        pending->second.reaping = true;
        active->second.reaping = true;
        return {.status = core::TaskDispatchStatus::kResourceUncertain, .task = std::nullopt};
    }
    if (result.status == core::TaskDispatchStatus::kQueued) {
        active_leases_.erase(active);
        return result;
    }
    if (result.status == core::TaskDispatchStatus::kResourceUncertain) {
        pending->second.reaping = true;
        active_leases_.erase(active);
        return result;
    }
    if (result.status == core::TaskDispatchStatus::kTerminalClean) {
        const auto record = pending->second;
        active_leases_.erase(active);
        pending_leases_.erase(pending);
        lock.unlock();
        observe_completion(*current, core::TaskState::kFailed, record);
        notify_state_change();
        return result;
    }
    if (!result.task || result.task->dispatch_sequence == 0) {
        pending->second.reaping = true;
        active->second.reaping = true;
        return {.status = core::TaskDispatchStatus::kResourceUncertain, .task = std::nullopt};
    }
    auto& lease = active->second;
    const auto now = Clock::now();
    const auto queued_at =
        lease.queued_at == Clock::time_point{} ? lease.submitted_at : lease.queued_at;
    if (queued_at != Clock::time_point{} && now > queued_at) {
        const auto elapsed =
            std::chrono::duration_cast<std::chrono::microseconds>(now - queued_at).count();
        const auto duration = elapsed <= 0 ? std::uint64_t{0} : static_cast<std::uint64_t>(elapsed);
        lease.queue_wait_microseconds =
            duration > std::numeric_limits<std::uint64_t>::max() - lease.queue_wait_microseconds
                ? std::numeric_limits<std::uint64_t>::max()
                : lease.queue_wait_microseconds + duration;
    }
    lease.started_at = now;
    lease.deadline = lease_timeout_ == std::chrono::milliseconds::zero() ? Clock::time_point::max()
                                                                         : now + lease_timeout_;
    pending_leases_.erase(pending);
    lock.unlock();
    observe_dispatch(*result.task);
    notify_state_change();
    return result;
}

bool ExecutionAdmissionState::cancel_pending_cooperative(core::TaskId task_id,
                                                         std::optional<TaskPeerIdentity> peer) {
    std::unique_lock lock(admission_mutex_);
    const auto lease = pending_leases_.find(task_id);
    const auto task = scheduler_.find(task_id);
    if (lease == pending_leases_.end() || !lease->second.is_cooperative ||
        lease->second.native_epoch.has_value() || lease->second.reaping ||
        !owner_matches(lease->second, peer) || !task.has_value() ||
        task->state != core::TaskState::kQueued || task->dispatch_sequence != 0 ||
        !scheduler_.cancel_queued(task_id)) {
        return false;
    }
    const auto record = lease->second;
    pending_leases_.erase(lease);
    lock.unlock();
    observe_completion(*task, core::TaskState::kCancelled, record);
    notify_state_change();
    return true;
}

std::optional<bool> ExecutionAdmissionState::retreat_requested(
    TaskExecutionIdentity execution, std::optional<TaskPeerIdentity> peer) {
    const auto [task_id, generation] = execution;
    std::scoped_lock lock(admission_mutex_);
    const auto snapshot = scheduler_.find(task_id);
    const auto lease = active_leases_.find(task_id);
    if (!bind_leases_to_process_ || generation == 0 || !snapshot.has_value() ||
        snapshot->state != core::TaskState::kRunning || snapshot->dispatch_sequence != generation ||
        lease == active_leases_.end() || !lease->second.is_cooperative || lease->second.reaping ||
        lease->second.native_epoch.has_value() || !owner_matches(lease->second, peer)) {
        return std::nullopt;
    }
    return scheduler_.retreat_requested(task_id);
}

bool ExecutionAdmissionState::suspend_execution(TaskExecutionIdentity execution,
                                                std::optional<TaskPeerIdentity> peer) {
    const auto [task_id, generation] = execution;
    std::unique_lock lock(admission_mutex_);
    const auto snapshot = scheduler_.find(task_id);
    const auto lease = active_leases_.find(task_id);
    if (!bind_leases_to_process_ || generation == 0 || !snapshot.has_value() ||
        snapshot->state != core::TaskState::kRunning || snapshot->dispatch_sequence != generation ||
        lease == active_leases_.end() || !lease->second.is_cooperative || lease->second.reaping ||
        lease->second.native_epoch.has_value() || !owner_matches(lease->second, peer)) {
        return false;
    }
    auto pending = lease->second;
    stop_lease_timing(pending);
    pending.is_cooperative = true;
    pending.deadline = lease_timeout_ == std::chrono::milliseconds::zero()
                           ? Clock::time_point::max()
                           : Clock::now() + lease_timeout_;
    try {
        if (!pending_leases_.emplace(task_id, pending).second) {
            return false;
        }
    } catch (...) {
        return false;
    }
    bool suspended = false;
    try {
        suspended = scheduler_.suspend(task_id);
    } catch (...) {
        suspended = false;
    }
    if (!suspended) {
        pending_leases_.erase(task_id);
        return false;
    }
    active_leases_.erase(lease);
    lock.unlock();
    notify_state_change();
    return true;
}

bool ExecutionAdmissionState::heartbeat_execution(TaskExecutionIdentity execution,
                                                  std::optional<TaskPeerIdentity> peer) {
    const auto [task_id, generation] = execution;
    std::scoped_lock lock(admission_mutex_);
    const auto snapshot = scheduler_.find(task_id);
    if (!bind_leases_to_process_ || generation == 0 || !snapshot.has_value() ||
        snapshot->dispatch_sequence != generation ||
        (snapshot->state != core::TaskState::kRunning &&
         snapshot->state != core::TaskState::kQueued)) {
        return false;
    }
    auto& leases = snapshot->state == core::TaskState::kRunning ? active_leases_ : pending_leases_;
    const auto lease = leases.find(task_id);
    if (lease == leases.end() || !lease->second.is_cooperative || lease->second.reaping ||
        lease->second.native_epoch.has_value() || !owner_matches(lease->second, peer)) {
        return false;
    }
    if (lease_timeout_ != std::chrono::milliseconds::zero()) {
        lease->second.deadline = Clock::now() + lease_timeout_;
    }
    return true;
}

bool ExecutionAdmissionState::finish_execution(TaskExecutionIdentity execution,
                                               std::optional<TaskPeerIdentity> peer,
                                               core::TaskState state) {
    const auto [task_id, generation] = execution;
    std::unique_lock lock(admission_mutex_);
    const auto snapshot = scheduler_.find(task_id);
    const bool is_cancel = state == core::TaskState::kCancelled;
    if (!bind_leases_to_process_ || generation == 0 || !snapshot.has_value() ||
        snapshot->dispatch_sequence != generation ||
        snapshot->state != (is_cancel ? core::TaskState::kQueued : core::TaskState::kRunning)) {
        return false;
    }
    auto& leases = is_cancel ? pending_leases_ : active_leases_;
    const auto lease = leases.find(task_id);
    if (lease == leases.end() || !lease->second.is_cooperative || lease->second.reaping ||
        lease->second.native_epoch.has_value() || !owner_matches(lease->second, peer)) {
        return false;
    }
    const auto record = lease->second;
    const bool finished = is_cancel                              ? scheduler_.cancel_queued(task_id)
                          : state == core::TaskState::kCompleted ? scheduler_.complete(task_id)
                                                                 : scheduler_.fail(task_id);
    if (!finished) {
        return false;
    }
    leases.erase(lease);
    lock.unlock();
    observe_completion(*snapshot, state, record);
    notify_state_change();
    return true;
}

bool ExecutionAdmissionState::complete_execution(TaskExecutionIdentity execution,
                                                 std::optional<TaskPeerIdentity> peer) {
    return finish_execution(execution, peer, core::TaskState::kCompleted);
}
bool ExecutionAdmissionState::fail_execution(TaskExecutionIdentity execution,
                                             std::optional<TaskPeerIdentity> peer) {
    return finish_execution(execution, peer, core::TaskState::kFailed);
}
bool ExecutionAdmissionState::cancel_execution(TaskExecutionIdentity execution,
                                               std::optional<TaskPeerIdentity> peer) {
    return finish_execution(execution, peer, core::TaskState::kCancelled);
}

}  // namespace glimmer::control
