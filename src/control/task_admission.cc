#include "glimmer/control/task_admission.h"

#include <utility>

namespace glimmer::control {
namespace {

[[nodiscard]] bool valid_peer_identity(const std::optional<TaskPeerIdentity>& peer) noexcept {
    return peer.has_value() && peer->pid != 0 && peer->start_time_ticks != 0;
}

}  // namespace

TaskAdmissionService::TaskAdmissionService(core::Scheduler& scheduler,
                                           std::chrono::milliseconds lease_timeout,
                                           bool bind_leases_to_process) noexcept
    : scheduler_(scheduler),
      lease_timeout_(lease_timeout > std::chrono::milliseconds::zero()
                         ? lease_timeout
                         : std::chrono::milliseconds::zero()),
      bind_leases_to_process_(bind_leases_to_process) {}

core::SubmitResult TaskAdmissionService::submit(const TaskAdmissionRequest& request,
                                                TaskResourceRegistrar registrar,
                                                void* registrar_context,
                                                std::optional<TaskPeerIdentity> peer) {
    if (registrar == nullptr) {
        return {.status = core::SubmitStatus::kInternalError};
    }
    if (bind_leases_to_process_ && !valid_peer_identity(peer)) {
        return {.status = core::SubmitStatus::kInvalidTask};
    }

    const core::SubmitResult admission =
        scheduler_.submit(core::TaskSpec{.tenant_id = request.tenant_id,
                                         .memory_bytes = request.memory_bytes,
                                         .weight = request.weight,
                                         .work_units = request.work_units,
                                         .priority = request.priority});
    if (!admission.accepted()) {
        return admission;
    }

    if (registrar(registrar_context, admission.task_id)) {
        if (bind_leases_to_process_ || lease_timeout_ != std::chrono::milliseconds::zero()) {
            bool recorded = false;
            try {
                std::scoped_lock lock(lease_mutex_);
                const auto deadline = lease_timeout_ == std::chrono::milliseconds::zero()
                                          ? Clock::time_point::max()
                                          : Clock::now() + lease_timeout_;
                recorded =
                    pending_leases_
                        .emplace(admission.task_id,
                                 LeaseRecord{.deadline = deadline, .reaping = false, .owner = peer})
                        .second;
            } catch (...) {
                recorded = false;
            }
            if (!recorded) {
                static_cast<void>(scheduler_.cancel(admission.task_id));
                return {.status = core::SubmitStatus::kInternalError, .task_id = admission.task_id};
            }
        }
        return admission;
    }

    static_cast<void>(scheduler_.cancel(admission.task_id));
    return {.status = core::SubmitStatus::kInternalError, .task_id = admission.task_id};
}

bool TaskAdmissionService::cancel(core::TaskId task_id) {
    const bool cancelled = scheduler_.cancel(task_id);
    if (cancelled) {
        std::scoped_lock lock(lease_mutex_);
        active_leases_.erase(task_id);
        pending_leases_.erase(task_id);
    }
    return cancelled;
}

bool TaskAdmissionService::cancel_queued(core::TaskId task_id) {
    const bool cancelled = scheduler_.cancel_queued(task_id);
    if (cancelled) {
        std::scoped_lock lock(lease_mutex_);
        pending_leases_.erase(task_id);
    }
    return cancelled;
}

std::optional<core::TaskSnapshot> TaskAdmissionService::claim_next(
    std::optional<TaskPeerIdentity> peer) {
    if (bind_leases_to_process_ && !valid_peer_identity(peer)) {
        return std::nullopt;
    }
    return record_lease(scheduler_.dispatch_next(), peer);
}

std::optional<core::TaskSnapshot> TaskAdmissionService::claim(
    core::TaskId task_id, std::optional<TaskPeerIdentity> peer) {
    if (task_id == 0 || (bind_leases_to_process_ && !valid_peer_identity(peer))) {
        return std::nullopt;
    }
    auto existing = scheduler_.find(task_id);
    if (existing.has_value() && existing->state == core::TaskState::kRunning) {
        std::scoped_lock lock(lease_mutex_);
        const auto iterator = active_leases_.find(task_id);
        if (iterator != active_leases_.end() && owner_matches(iterator->second, peer)) {
            return existing;
        }
        if (!bind_leases_to_process_ && lease_timeout_ == std::chrono::milliseconds::zero()) {
            return existing;
        }
        return std::nullopt;
    }
    return record_lease(scheduler_.dispatch_task(task_id), peer);
}

std::optional<core::TaskSnapshot> TaskAdmissionService::record_lease(
    std::optional<core::TaskSnapshot> snapshot, std::optional<TaskPeerIdentity> peer) {
    if (!snapshot.has_value() ||
        (!bind_leases_to_process_ && lease_timeout_ == std::chrono::milliseconds::zero())) {
        return snapshot;
    }
    {
        std::scoped_lock lock(lease_mutex_);
        pending_leases_.erase(snapshot->task_id);
    }
    bool lease_recorded = false;
    try {
        std::scoped_lock lock(lease_mutex_);
        const auto deadline = lease_timeout_ == std::chrono::milliseconds::zero()
                                  ? Clock::time_point::max()
                                  : Clock::now() + lease_timeout_;
        lease_recorded =
            active_leases_
                .emplace(snapshot->task_id,
                         LeaseRecord{.deadline = deadline, .reaping = false, .owner = peer})
                .second;
    } catch (...) {
        lease_recorded = false;
    }
    if (!lease_recorded) {
        static_cast<void>(scheduler_.fail(snapshot->task_id));
        return std::nullopt;
    }
    return snapshot;
}

bool TaskAdmissionService::heartbeat(core::TaskId task_id, std::optional<TaskPeerIdentity> peer) {
    const auto snapshot = scheduler_.find(task_id);
    if (!snapshot.has_value() || snapshot->state != core::TaskState::kRunning) {
        return false;
    }
    if (!bind_leases_to_process_ && lease_timeout_ == std::chrono::milliseconds::zero()) {
        return true;
    }
    std::scoped_lock lock(lease_mutex_);
    const auto lease_iterator = active_leases_.find(task_id);
    if (lease_iterator == active_leases_.end() || lease_iterator->second.reaping ||
        !owner_matches(lease_iterator->second, peer)) {
        return false;
    }
    if (lease_timeout_ != std::chrono::milliseconds::zero()) {
        lease_iterator->second.deadline = Clock::now() + lease_timeout_;
    }
    return true;
}

bool TaskAdmissionService::complete(core::TaskId task_id, std::optional<TaskPeerIdentity> peer) {
    {
        std::scoped_lock lock(lease_mutex_);
        const auto lease_iterator = active_leases_.find(task_id);
        if (bind_leases_to_process_ && (lease_iterator == active_leases_.end() ||
                                        !owner_matches(lease_iterator->second, peer))) {
            return false;
        }
    }
    const bool completed = scheduler_.complete(task_id);
    if (completed) {
        std::scoped_lock lock(lease_mutex_);
        active_leases_.erase(task_id);
    }
    return completed;
}

bool TaskAdmissionService::fail(core::TaskId task_id, std::optional<TaskPeerIdentity> peer) {
    {
        std::scoped_lock lock(lease_mutex_);
        const auto lease_iterator = active_leases_.find(task_id);
        if (bind_leases_to_process_ && (lease_iterator == active_leases_.end() ||
                                        !owner_matches(lease_iterator->second, peer))) {
            return false;
        }
    }
    const bool failed = scheduler_.fail(task_id);
    if (failed) {
        std::scoped_lock lock(lease_mutex_);
        active_leases_.erase(task_id);
    }
    return failed;
}

bool TaskAdmissionService::owner_matches(
    const LeaseRecord& lease, const std::optional<TaskPeerIdentity>& peer) const noexcept {
    if (!bind_leases_to_process_) {
        return true;
    }
    return valid_peer_identity(peer) && valid_peer_identity(lease.owner) &&
           peer.value_or(TaskPeerIdentity{}) == lease.owner.value_or(TaskPeerIdentity{});
}

bool TaskAdmissionService::reap_expired() {
    if (lease_timeout_ == std::chrono::milliseconds::zero()) {
        return false;
    }
    std::optional<core::TaskId> expired_pending_task;
    std::optional<core::TaskId> expired_running_task;
    {
        std::scoped_lock lock(lease_mutex_);
        const auto now = Clock::now();
        for (auto& [task_id, lease] : pending_leases_) {
            if (!lease.reaping && now >= lease.deadline) {
                expired_pending_task = task_id;
                lease.reaping = true;
                break;
            }
        }
        if (expired_pending_task.has_value()) {
            // Reclaim queued work before considering running leases. This
            // prevents a crashed submitter from blocking the policy head.
            expired_running_task = std::nullopt;
        } else {
            for (auto& [task_id, lease] : active_leases_) {
                if (!lease.reaping && now >= lease.deadline) {
                    expired_running_task = task_id;
                    lease.reaping = true;
                    break;
                }
            }
        }
    }
    if (expired_pending_task.has_value()) {
        const bool cancelled = scheduler_.cancel_queued(expired_pending_task.value());
        std::scoped_lock lock(lease_mutex_);
        const auto iterator = pending_leases_.find(expired_pending_task.value());
        if (iterator == pending_leases_.end()) {
            return cancelled;
        }
        if (cancelled) {
            pending_leases_.erase(iterator);
            return cancelled;
        }
        iterator->second.reaping = false;
        iterator->second.deadline = Clock::now() + lease_timeout_;
        return false;
    }
    if (!expired_running_task.has_value()) {
        return false;
    }
    const bool failed = scheduler_.fail(expired_running_task.value());
    const auto snapshot = scheduler_.find(expired_running_task.value());
    std::scoped_lock lock(lease_mutex_);
    const auto lease_iterator = active_leases_.find(expired_running_task.value());
    if (lease_iterator == active_leases_.end()) {
        return failed;
    }
    if (failed || !snapshot.has_value() || snapshot->state != core::TaskState::kRunning) {
        active_leases_.erase(lease_iterator);
        return failed;
    }

    // Keep the lease recoverable when the scheduler could not release its
    // accounting. The next expiry retries the terminal transition instead of
    // leaving an untracked running task behind.
    lease_iterator->second.reaping = false;
    lease_iterator->second.deadline = Clock::now() + lease_timeout_;
    return failed;
}

std::optional<core::TaskSnapshot> TaskAdmissionService::find(core::TaskId task_id) const {
    return scheduler_.find(task_id);
}

core::SchedulerStats TaskAdmissionService::stats() const {
    return scheduler_.stats();
}

}  // namespace glimmer::control
