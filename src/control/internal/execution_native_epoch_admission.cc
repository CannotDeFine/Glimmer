#include "execution_admission_state.h"

#include <limits>

namespace glimmer::control {
namespace {

std::uint64_t monotonic_nanoseconds() noexcept {
    const auto value = std::chrono::duration_cast<std::chrono::nanoseconds>(
                           std::chrono::steady_clock::now().time_since_epoch())
                           .count();
    return value <= 0 ? 0 : static_cast<std::uint64_t>(value);
}

NativeEpochGrant grant_from(const core::TaskSnapshot& task, NativeEpochExtent extent,
                            std::uint64_t deadline_nanoseconds) {
    return {.execution = {task.task_id, task.dispatch_sequence},
            .extent = extent,
            .acceptance_deadline_nanoseconds = deadline_nanoseconds};
}

}  // namespace

core::SubmitResult ExecutionAdmissionState::submit_native_epoch(
    const TaskAdmissionRequest& request, NativeEpochExtent extent, TaskResourceRegistrar registrar,
    void* registrar_context, std::optional<TaskPeerIdentity> peer) {
    if (!bind_leases_to_process_ || !native_epoch_policy ||
        !valid_native_epoch_extent(*native_epoch_policy) || !valid_native_epoch_extent(extent) ||
        extent.max_calls > native_epoch_policy->max_calls ||
        extent.calls_per_prefix > native_epoch_policy->calls_per_prefix ||
        extent.max_pending_prefixes > native_epoch_policy->max_pending_prefixes ||
        extent.acceptance_microseconds > native_epoch_policy->acceptance_microseconds ||
        request.work_units != extent.max_calls) {
        return {.status = core::SubmitStatus::kInvalidTask};
    }
    return submit_impl(request, registrar, registrar_context, peer, extent, AdmissionKind::kNative);
}

std::optional<NativeEpochGrant> ExecutionAdmissionState::claim_native_epoch(
    core::TaskId task_id, std::optional<TaskPeerIdentity> peer) {
    std::unique_lock lock(admission_mutex_);
    const auto active = active_leases_.find(task_id);
    if (active != active_leases_.end()) {
        const auto task = scheduler_.find(task_id);
        if (!task || !active->second.native_epoch.has_value() ||
            !native_epoch_owner_matches({task_id, task->dispatch_sequence}, peer)) {
            return std::nullopt;
        }
        return grant_from(*task, active->second.native_epoch.value_or(NativeEpochExtent{}),
                          active->second.acceptance_deadline_nanoseconds);
    }
    const auto pending = pending_leases_.find(task_id);
    if (pending == pending_leases_.end() || !pending->second.native_epoch ||
        !pending_owner_matches(task_id, peer)) {
        return std::nullopt;
    }
    auto task = record_lease(scheduler_.dispatch_task(task_id), peer);
    if (!task) {
        return std::nullopt;
    }
    const auto active_lease = active_leases_.find(task_id);
    if (active_lease == active_leases_.end() || !active_lease->second.native_epoch.has_value()) {
        return std::nullopt;
    }
    auto& lease = active_lease->second;
    const auto extent = lease.native_epoch.value_or(NativeEpochExtent{});
    const auto now = monotonic_nanoseconds();
    const std::uint64_t duration =
        static_cast<std::uint64_t>(extent.acceptance_microseconds) * 1'000;
    lease.acceptance_deadline_nanoseconds =
        duration > std::numeric_limits<std::uint64_t>::max() - now
            ? std::numeric_limits<std::uint64_t>::max()
            : now + duration;
    const auto grant = grant_from(*task, extent, lease.acceptance_deadline_nanoseconds);
    lock.unlock();
    observe_dispatch(*task);
    notify_state_change();
    return grant;
}

bool ExecutionAdmissionState::native_epoch_owner_matches(
    TaskExecutionIdentity execution, const std::optional<TaskPeerIdentity>& peer) const {
    const auto task = scheduler_.find(execution.task_id);
    const auto lease = active_leases_.find(execution.task_id);
    if (lease == active_leases_.end() || !lease->second.native_epoch.has_value()) {
        return false;
    }
    return bind_leases_to_process_ && execution.generation != 0 && task &&
           task->state == core::TaskState::kRunning &&
           task->dispatch_sequence == execution.generation && !lease->second.reaping &&
           owner_matches(lease->second, peer);
}

std::optional<NativeEpochExtent> ExecutionAdmissionState::pending_native_epoch_extent(
    core::TaskId task_id, std::optional<TaskPeerIdentity> peer) {
    std::scoped_lock lock(admission_mutex_);
    const auto lease = pending_leases_.find(task_id);
    const auto task = scheduler_.find(task_id);
    if (lease == pending_leases_.end() || !lease->second.native_epoch || !task ||
        task->state != core::TaskState::kQueued || !pending_owner_matches(task_id, peer)) {
        return std::nullopt;
    }
    return lease->second.native_epoch;
}

std::optional<bool> ExecutionAdmissionState::native_epoch_retreat_requested(
    TaskExecutionIdentity execution, std::optional<TaskPeerIdentity> peer) {
    std::scoped_lock lock(admission_mutex_);
    if (!native_epoch_owner_matches(execution, peer)) {
        return std::nullopt;
    }
    if (monotonic_nanoseconds() >=
        active_leases_.at(execution.task_id).acceptance_deadline_nanoseconds) {
        return true;
    }
    return scheduler_.retreat_requested(execution.task_id);
}

bool ExecutionAdmissionState::heartbeat_native_epoch(TaskExecutionIdentity execution,
                                                     std::optional<TaskPeerIdentity> peer) {
    std::scoped_lock lock(admission_mutex_);
    // A heartbeat acknowledges identity, not continued acceptance or renewal.
    return native_epoch_owner_matches(execution, peer);
}

bool ExecutionAdmissionState::finish_native_epoch(TaskExecutionIdentity execution,
                                                  std::uint32_t issued_calls, bool has_failed,
                                                  std::optional<TaskPeerIdentity> peer) {
    std::unique_lock lock(admission_mutex_);
    if (!native_epoch_owner_matches(execution, peer)) {
        return false;
    }
    const auto lease = active_leases_.find(execution.task_id);
    if (lease == active_leases_.end() || !lease->second.native_epoch.has_value()) {
        return false;
    }
    if (issued_calls > lease->second.native_epoch.value_or(NativeEpochExtent{}).max_calls) {
        return false;
    }
    const auto task = scheduler_.find(execution.task_id);
    if (!task.has_value()) {
        return false;
    }
    const auto record = lease->second;
    const bool finished =
        has_failed ? scheduler_.fail(execution.task_id) : scheduler_.complete(execution.task_id);
    if (!finished) {
        return false;
    }
    active_leases_.erase(lease);
    lock.unlock();
    observe_completion(*task, has_failed ? core::TaskState::kFailed : core::TaskState::kCompleted,
                       record);
    notify_state_change();
    return true;
}

bool ExecutionAdmissionState::cancel_pending_native_epoch(core::TaskId task_id,
                                                          std::optional<TaskPeerIdentity> peer) {
    std::unique_lock lock(admission_mutex_);
    const auto lease = pending_leases_.find(task_id);
    if (lease == pending_leases_.end() || !lease->second.native_epoch ||
        !pending_owner_matches(task_id, peer)) {
        return false;
    }
    const auto task = scheduler_.find(task_id);
    if (!task || task->dispatch_sequence != 0 || !scheduler_.cancel_queued(task_id)) {
        return false;
    }
    const auto record = lease->second;
    pending_leases_.erase(lease);
    lock.unlock();
    observe_completion(*task, core::TaskState::kCancelled, record);
    notify_state_change();
    return true;
}

}  // namespace glimmer::control
