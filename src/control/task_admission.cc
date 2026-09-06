#include "glimmer/control/task_admission.h"

#include <limits>

namespace glimmer::control {
namespace {

[[nodiscard]] bool valid_peer_identity(const std::optional<TaskPeerIdentity>& peer) noexcept {
    return peer.has_value() && peer->pid != 0 && peer->start_time_ticks != 0;
}

[[nodiscard]] std::uint64_t elapsed_microseconds(
    std::chrono::steady_clock::time_point start,
    std::chrono::steady_clock::time_point end) noexcept {
    if (start == std::chrono::steady_clock::time_point{} || end <= start) {
        return 0;
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return elapsed.count() <= 0 ? 0 : static_cast<std::uint64_t>(elapsed.count());
}

}  // namespace

TaskAdmissionService::TaskAdmissionService(core::Scheduler& scheduler,
                                           std::chrono::milliseconds lease_timeout,
                                           bool bind_leases_to_process,
                                           TaskDispatchObserver dispatch_observer,
                                           void* dispatch_observer_context,
                                           TaskCompletionObserver completion_observer,
                                           void* completion_observer_context) noexcept
    : scheduler_(scheduler),
      lease_timeout_(lease_timeout > std::chrono::milliseconds::zero()
                         ? lease_timeout
                         : std::chrono::milliseconds::zero()),
      bind_leases_to_process_(bind_leases_to_process),
      dispatch_observer_(dispatch_observer),
      dispatch_observer_context_(dispatch_observer_context),
      completion_observer_(completion_observer),
      completion_observer_context_(completion_observer_context) {}

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

    // A task must not become running between Scheduler::submit and completion
    // of resource registration. Serialize this transaction with dispatches.
    std::scoped_lock admission_lock(admission_mutex_);
    const bool tracks_pending_lease = bind_leases_to_process_ ||
                                      lease_timeout_ != std::chrono::milliseconds::zero() ||
                                      completion_observer_ != nullptr;

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
        if (tracks_pending_lease) {
            bool recorded = false;
            try {
                const auto submitted_at = Clock::now();
                const auto deadline = lease_timeout_ == std::chrono::milliseconds::zero()
                                          ? Clock::time_point::max()
                                          : submitted_at + lease_timeout_;
                recorded = pending_leases_
                               .emplace(admission.task_id, LeaseRecord{.deadline = deadline,
                                                                       .submitted_at = submitted_at,
                                                                       .reaping = false,
                                                                       .owner = peer})
                               .second;
            } catch (...) {
                recorded = false;
            }
            if (!recorded) {
                static_cast<void>(scheduler_.cancel(admission.task_id));
                notify_state_change();
                return {.status = core::SubmitStatus::kInternalError, .task_id = admission.task_id};
            }
        }
        return admission;
    }

    static_cast<void>(scheduler_.cancel(admission.task_id));
    notify_state_change();
    return {.status = core::SubmitStatus::kInternalError, .task_id = admission.task_id};
}

bool TaskAdmissionService::cancel(core::TaskId task_id) {
    std::scoped_lock admission_lock(admission_mutex_);
    const bool cancelled = scheduler_.cancel(task_id);
    if (cancelled) {
        active_leases_.erase(task_id);
        pending_leases_.erase(task_id);
        notify_state_change();
    }
    return cancelled;
}

bool TaskAdmissionService::cancel_queued(core::TaskId task_id) {
    std::scoped_lock admission_lock(admission_mutex_);
    const bool cancelled = scheduler_.cancel_queued(task_id);
    if (cancelled) {
        pending_leases_.erase(task_id);
        notify_state_change();
    }
    return cancelled;
}

std::optional<core::TaskSnapshot> TaskAdmissionService::claim_next(
    std::optional<TaskPeerIdentity> peer) {
    if (bind_leases_to_process_) {
        // A policy-agnostic claim cannot prove that the selected task belongs
        // to this process. Process-bound clients must claim their submitted
        // task by id so the pending owner can be checked before dispatch.
        return std::nullopt;
    }
    std::optional<core::TaskSnapshot> dispatched;
    {
        std::scoped_lock lock(admission_mutex_);
        dispatched = record_lease(scheduler_.dispatch_next(), peer);
    }
    if (dispatched.has_value()) {
        observe_dispatch(dispatched.value());
        notify_state_change();
    }
    return dispatched;
}

std::optional<core::TaskSnapshot> TaskAdmissionService::claim(
    core::TaskId task_id, std::optional<TaskPeerIdentity> peer) {
    if (task_id == 0 || (bind_leases_to_process_ && !valid_peer_identity(peer))) {
        return std::nullopt;
    }
    std::unique_lock admission_lock(admission_mutex_);
    auto existing = scheduler_.find(task_id);
    if (existing.has_value() && existing->state == core::TaskState::kRunning) {
        const auto iterator = active_leases_.find(task_id);
        if (iterator != active_leases_.end() && owner_matches(iterator->second, peer)) {
            return existing;
        }
        if (!bind_leases_to_process_ && lease_timeout_ == std::chrono::milliseconds::zero()) {
            return existing;
        }
        return std::nullopt;
    }
    if (!pending_owner_matches(task_id, peer)) {
        return std::nullopt;
    }
    const auto result = record_lease(scheduler_.dispatch_task(task_id), peer);
    admission_lock.unlock();
    if (result.has_value()) {
        observe_dispatch(result.value());
        notify_state_change();
    }
    return result;
}

std::optional<core::TaskSnapshot> TaskAdmissionService::wait_claim(
    core::TaskId task_id, std::chrono::milliseconds timeout, std::optional<TaskPeerIdentity> peer) {
    if (task_id == 0 || timeout < std::chrono::milliseconds::zero() ||
        (bind_leases_to_process_ && !valid_peer_identity(peer))) {
        return std::nullopt;
    }

    const auto deadline = timeout == std::chrono::milliseconds::zero() ? Clock::time_point::max()
                                                                       : Clock::now() + timeout;
    std::unique_lock state_lock(state_mutex_);
    while (true) {
        const std::uint64_t observed_generation = state_generation_;
        state_lock.unlock();
        std::unique_lock admission_lock(admission_mutex_);
        auto existing = scheduler_.find(task_id);
        if (!existing.has_value()) {
            return std::nullopt;
        }
        if (existing->state == core::TaskState::kRunning) {
            const auto lease_iterator = active_leases_.find(task_id);
            if ((!bind_leases_to_process_ && lease_timeout_ == std::chrono::milliseconds::zero()) ||
                (lease_iterator != active_leases_.end() &&
                 owner_matches(lease_iterator->second, peer))) {
                return existing;
            }
            return std::nullopt;
        }
        if (existing->state != core::TaskState::kQueued) {
            return std::nullopt;
        }
        if (!pending_owner_matches(task_id, peer)) {
            return std::nullopt;
        }

        auto snapshot = record_lease(scheduler_.dispatch_task(task_id), peer);
        admission_lock.unlock();
        if (snapshot.has_value()) {
            observe_dispatch(snapshot.value());
            notify_state_change();
            return snapshot;
        }

        if (deadline != Clock::time_point::max() && Clock::now() >= deadline) {
            return std::nullopt;
        }
        state_lock.lock();
        if (state_generation_ != observed_generation) {
            continue;
        }
        if (deadline == Clock::time_point::max()) {
            state_condition_.wait(state_lock, [this, observed_generation] {
                return state_generation_ != observed_generation;
            });
        } else if (!state_condition_.wait_until(state_lock, deadline, [this, observed_generation] {
                       return state_generation_ != observed_generation;
                   })) {
            return std::nullopt;
        }
    }
}

std::optional<core::TaskSnapshot> TaskAdmissionService::record_lease(
    std::optional<core::TaskSnapshot> snapshot, std::optional<TaskPeerIdentity> peer) {
    const bool tracks_leases = bind_leases_to_process_ ||
                               lease_timeout_ != std::chrono::milliseconds::zero() ||
                               completion_observer_ != nullptr;
    if (!snapshot.has_value() || !tracks_leases) {
        return snapshot;
    }
    bool pending_owner_valid = true;
    Clock::time_point submitted_at{};
    {
        const auto pending_iterator = pending_leases_.find(snapshot->task_id);
        if (bind_leases_to_process_ &&
            (pending_iterator == pending_leases_.end() || pending_iterator->second.reaping ||
             !owner_matches(pending_iterator->second, peer))) {
            pending_owner_valid = false;
            if (pending_iterator != pending_leases_.end()) {
                pending_leases_.erase(pending_iterator);
            }
        } else if (pending_iterator != pending_leases_.end()) {
            submitted_at = pending_iterator->second.submitted_at;
            pending_leases_.erase(pending_iterator);
        }
    }
    if (!pending_owner_valid) {
        static_cast<void>(scheduler_.fail(snapshot->task_id));
        notify_state_change();
        return std::nullopt;
    }
    bool lease_recorded = false;
    try {
        if (!bind_leases_to_process_ &&
            active_leases_.find(snapshot->task_id) != active_leases_.end()) {
            // A repeated external-dispatch notification is idempotent. Do
            // not fail an already-running task because its observation was
            // delivered twice.
            return snapshot;
        }
        const auto started_at = Clock::now();
        const auto deadline = lease_timeout_ == std::chrono::milliseconds::zero()
                                  ? Clock::time_point::max()
                                  : started_at + lease_timeout_;
        const auto [iterator, inserted] =
            active_leases_.emplace(snapshot->task_id, LeaseRecord{.deadline = deadline,
                                                                  .submitted_at = submitted_at,
                                                                  .started_at = started_at,
                                                                  .reaping = false,
                                                                  .owner = peer});
        static_cast<void>(iterator);
        if (!inserted && !bind_leases_to_process_) {
            // Preserve idempotence if an external dispatch was already recorded.
            return snapshot;
        }
        lease_recorded = inserted;
    } catch (...) {
        lease_recorded = false;
    }
    if (!lease_recorded) {
        static_cast<void>(scheduler_.fail(snapshot->task_id));
        notify_state_change();
        return std::nullopt;
    }
    return snapshot;
}

bool TaskAdmissionService::pending_owner_matches(
    core::TaskId task_id, const std::optional<TaskPeerIdentity>& peer) noexcept {
    if (!bind_leases_to_process_) {
        return true;
    }
    const auto iterator = pending_leases_.find(task_id);
    return iterator != pending_leases_.end() && !iterator->second.reaping &&
           owner_matches(iterator->second, peer);
}

void TaskAdmissionService::observe_dispatch(const core::TaskSnapshot& snapshot) noexcept {
    if (dispatch_observer_ == nullptr) {
        return;
    }
    try {
        const auto elapsed =
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch());
        const std::uint64_t dispatched_at_nanoseconds =
            elapsed.count() <= 0 ? 0 : static_cast<std::uint64_t>(elapsed.count());
        const TaskDispatchObservation observation{
            .task = snapshot,
            .sequence = snapshot.dispatch_sequence,
            .dispatched_at_nanoseconds = dispatched_at_nanoseconds};
        dispatch_observer_(dispatch_observer_context_, observation);
    } catch (...) {
        // Tracing is strictly best effort. An allocation or callback failure
        // must never turn a successful scheduler dispatch into a failed one.
        return;
    }
}

void TaskAdmissionService::observe_completion(const core::TaskSnapshot& snapshot,
                                              core::TaskState terminal_state,
                                              const std::optional<LeaseRecord>& lease) noexcept {
    if (completion_observer_ == nullptr) {
        return;
    }
    try {
        const auto now = Clock::now();
        const TaskCompletionObservation observation{
            .task = snapshot,
            .terminal_state = terminal_state,
            .queue_wait_microseconds =
                lease.has_value() ? elapsed_microseconds(lease->submitted_at, lease->started_at)
                                  : 0,
            .service_time_microseconds =
                lease.has_value() ? elapsed_microseconds(lease->started_at, now) : 0};
        completion_observer_(completion_observer_context_, observation);
    } catch (...) {
        // Observability is strictly best effort. A callback or allocation
        // failure must never change a terminal scheduler transition.
        return;
    }
}

bool TaskAdmissionService::heartbeat(core::TaskId task_id, std::optional<TaskPeerIdentity> peer) {
    std::scoped_lock admission_lock(admission_mutex_);
    const auto snapshot = scheduler_.find(task_id);
    if (!snapshot.has_value() || snapshot->state != core::TaskState::kRunning) {
        return false;
    }
    if (!bind_leases_to_process_ && lease_timeout_ == std::chrono::milliseconds::zero()) {
        return true;
    }
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
    std::unique_lock admission_lock(admission_mutex_);
    {
        const auto lease_iterator = active_leases_.find(task_id);
        if (bind_leases_to_process_ && (lease_iterator == active_leases_.end() ||
                                        !owner_matches(lease_iterator->second, peer))) {
            return false;
        }
    }
    const auto before = completion_observer_ == nullptr ? std::nullopt : scheduler_.find(task_id);
    const bool completed = scheduler_.complete(task_id);
    if (completed) {
        std::optional<LeaseRecord> lease;
        {
            const auto iterator = active_leases_.find(task_id);
            if (iterator != active_leases_.end()) {
                lease = iterator->second;
                active_leases_.erase(iterator);
            }
        }
        admission_lock.unlock();
        if (before.has_value()) {
            observe_completion(before.value(), core::TaskState::kCompleted, lease);
        }
        notify_state_change();
    }
    return completed;
}

bool TaskAdmissionService::fail(core::TaskId task_id, std::optional<TaskPeerIdentity> peer) {
    std::unique_lock admission_lock(admission_mutex_);
    {
        const auto lease_iterator = active_leases_.find(task_id);
        if (bind_leases_to_process_ && (lease_iterator == active_leases_.end() ||
                                        !owner_matches(lease_iterator->second, peer))) {
            return false;
        }
    }
    const auto before = completion_observer_ == nullptr ? std::nullopt : scheduler_.find(task_id);
    const bool failed = scheduler_.fail(task_id);
    if (failed) {
        std::optional<LeaseRecord> lease;
        {
            const auto iterator = active_leases_.find(task_id);
            if (iterator != active_leases_.end()) {
                lease = iterator->second;
                active_leases_.erase(iterator);
            }
        }
        admission_lock.unlock();
        if (before.has_value()) {
            observe_completion(before.value(), core::TaskState::kFailed, lease);
        }
        notify_state_change();
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
    std::unique_lock admission_lock(admission_mutex_);
    std::optional<core::TaskId> expired_pending_task;
    std::optional<core::TaskId> expired_running_task;
    {
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
        bool removed = false;
        {
            const auto iterator = pending_leases_.find(expired_pending_task.value());
            if (iterator == pending_leases_.end()) {
                return cancelled;
            }
            if (cancelled) {
                pending_leases_.erase(iterator);
                removed = true;
            } else {
                iterator->second.reaping = false;
                iterator->second.deadline = Clock::now() + lease_timeout_;
            }
        }
        if (removed) {
            admission_lock.unlock();
            notify_state_change();
            return cancelled;
        }
        return false;
    }
    if (!expired_running_task.has_value()) {
        return false;
    }
    const auto before = completion_observer_ == nullptr
                            ? std::nullopt
                            : scheduler_.find(expired_running_task.value());
    const bool failed = scheduler_.fail(expired_running_task.value());
    const auto snapshot = scheduler_.find(expired_running_task.value());
    bool lease_removed = false;
    std::optional<LeaseRecord> lease;
    {
        const auto lease_iterator = active_leases_.find(expired_running_task.value());
        if (lease_iterator == active_leases_.end()) {
            return failed;
        }
        if (failed || !snapshot.has_value() || snapshot->state != core::TaskState::kRunning) {
            lease = lease_iterator->second;
            active_leases_.erase(lease_iterator);
            lease_removed = true;
        } else {
            // Keep the lease recoverable when the scheduler could not release
            // its accounting. The next expiry retries the terminal transition
            // instead of leaving an untracked running task behind.
            lease_iterator->second.reaping = false;
            lease_iterator->second.deadline = Clock::now() + lease_timeout_;
        }
    }
    if (lease_removed) {
        admission_lock.unlock();
        if (failed && before.has_value()) {
            observe_completion(before.value(), core::TaskState::kFailed, lease);
        }
        notify_state_change();
    }
    return failed;
}

void TaskAdmissionService::notify_state_change() noexcept {
    {
        std::scoped_lock lock(state_mutex_);
        if (state_generation_ != std::numeric_limits<std::uint64_t>::max()) {
            ++state_generation_;
        }
    }
    state_condition_.notify_all();
}

void TaskAdmissionService::notify_scheduler_change() noexcept {
    notify_state_change();
}

std::optional<core::TaskSnapshot> TaskAdmissionService::find(core::TaskId task_id) const {
    return scheduler_.find(task_id);
}

core::SchedulerStats TaskAdmissionService::stats() const {
    return scheduler_.stats();
}

void TaskAdmissionService::observe_external_dispatch(core::TaskId task_id) noexcept {
    if (task_id == 0 || (dispatch_observer_ == nullptr && completion_observer_ == nullptr)) {
        return;
    }
    std::unique_lock admission_lock(admission_mutex_);
    if (active_leases_.contains(task_id)) {
        return;
    }
    const auto snapshot = scheduler_.find(task_id);
    if (!snapshot.has_value() || snapshot->state != core::TaskState::kRunning) {
        return;
    }

    std::optional<TaskPeerIdentity> peer;
    if (bind_leases_to_process_) {
        const auto iterator = pending_leases_.find(task_id);
        if (iterator == pending_leases_.end() || iterator->second.reaping) {
            return;
        }
        peer = iterator->second.owner;
    }
    const auto recorded = record_lease(snapshot, peer);
    admission_lock.unlock();
    if (recorded.has_value()) {
        observe_dispatch(recorded.value());
        notify_state_change();
    }
}

void TaskAdmissionService::observe_external_completion(core::TaskId task_id,
                                                       core::TaskState terminal_state) noexcept {
    if (task_id == 0 || completion_observer_ == nullptr ||
        (terminal_state != core::TaskState::kCompleted &&
         terminal_state != core::TaskState::kFailed &&
         terminal_state != core::TaskState::kCancelled)) {
        return;
    }
    std::unique_lock admission_lock(admission_mutex_);
    const auto snapshot = scheduler_.find(task_id);
    if (!snapshot.has_value() || snapshot->state != terminal_state) {
        return;
    }
    std::optional<LeaseRecord> lease;
    {
        const auto iterator = active_leases_.find(task_id);
        if (iterator == active_leases_.end()) {
            return;
        }
        lease = iterator->second;
        active_leases_.erase(iterator);
    }
    admission_lock.unlock();
    observe_completion(snapshot.value(), terminal_state, lease);
    notify_state_change();
}

}  // namespace glimmer::control
