#pragma once

#include "glimmer/core/scheduler.h"

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

namespace glimmer::control {

struct TaskAdmissionRequest {
    std::string tenant_id;
    core::MemoryBytes memory_bytes = 0;
    std::uint32_t weight = 1;
    std::uint32_t work_units = 1;
    std::uint32_t priority = 0;
};

// Identity obtained from a trusted local transport such as SO_PEERCRED. The
// process start-time value prevents a recycled PID from inheriting a lease.
struct TaskPeerIdentity {
    std::uint32_t pid = 0;
    std::uint32_t uid = 0;
    std::uint64_t start_time_ticks = 0;

    friend bool operator==(const TaskPeerIdentity&, const TaskPeerIdentity&) = default;
};

using TaskResourceRegistrar = bool (*)(void* context, core::TaskId task_id) noexcept;

// Best-effort observation emitted after a task is dispatched through this
// admission service. The callback must be noexcept, must not call back into
// the service, and should keep its work bounded because it runs on the
// dispatching caller's thread. Observations are disabled when the callback is
// null and never change admission behavior.
struct TaskDispatchObservation {
    core::TaskSnapshot task;
    std::uint64_t sequence = 0;
    std::uint64_t dispatched_at_nanoseconds = 0;
};

using TaskDispatchObserver = void (*)(void* context,
                                      const TaskDispatchObservation& observation) noexcept;

// Thread-safe for scheduler operations. The registrar is invoked synchronously
// after admission and must not throw or call back into this service. It owns
// backend-specific resource binding and must keep those resources alive until
// the task reaches a terminal state.
class TaskAdmissionService final {
   public:
    explicit TaskAdmissionService(
        core::Scheduler& scheduler,
        std::chrono::milliseconds lease_timeout = std::chrono::milliseconds::zero(),
        bool bind_leases_to_process = false, TaskDispatchObserver dispatch_observer = nullptr,
        void* dispatch_observer_context = nullptr) noexcept;

    [[nodiscard]] core::SubmitResult submit(const TaskAdmissionRequest& request,
                                            TaskResourceRegistrar registrar,
                                            void* registrar_context,
                                            std::optional<TaskPeerIdentity> peer = std::nullopt);
    [[nodiscard]] bool cancel(core::TaskId task_id);
    [[nodiscard]] bool cancel_queued(core::TaskId task_id);
    // Claims the next queued task and transitions it to running. The returned
    // snapshot is the worker lease metadata; no backend resource is owned by
    // this service. When process binding is enabled, unscoped claims are
    // rejected; callers must use claim(task_id, peer) for their own task.
    [[nodiscard]] std::optional<core::TaskSnapshot> claim_next(
        std::optional<TaskPeerIdentity> peer = std::nullopt);
    // Claims a specific queued task when it is next under the scheduler
    // policy. This is used by process-bound transparent launch clients so a
    // process cannot accidentally execute another tenant's lease. In
    // process-bound mode, the peer must match the identity that submitted the
    // pending lease.
    [[nodiscard]] std::optional<core::TaskSnapshot> claim(
        core::TaskId task_id, std::optional<TaskPeerIdentity> peer = std::nullopt);
    // Waits until a specific queued task is next under the scheduler policy,
    // a running lease can be returned to its owner, or timeout expires. A
    // zero timeout waits indefinitely for in-process callers.
    [[nodiscard]] std::optional<core::TaskSnapshot> wait_claim(
        core::TaskId task_id, std::chrono::milliseconds timeout,
        std::optional<TaskPeerIdentity> peer = std::nullopt);
    [[nodiscard]] bool heartbeat(core::TaskId task_id,
                                 std::optional<TaskPeerIdentity> peer = std::nullopt);
    [[nodiscard]] bool complete(core::TaskId task_id,
                                std::optional<TaskPeerIdentity> peer = std::nullopt);
    [[nodiscard]] bool fail(core::TaskId task_id,
                            std::optional<TaskPeerIdentity> peer = std::nullopt);
    // Expires the active lease when a timeout is configured. Returns true when
    // a running task was transitioned to FAILED and its quota was released.
    [[nodiscard]] bool reap_expired();
    [[nodiscard]] std::optional<core::TaskSnapshot> find(core::TaskId task_id) const;
    [[nodiscard]] core::SchedulerStats stats() const;
    // Wakes WAIT callers after a component outside this service advances the
    // referenced scheduler. The service itself notifies automatically for
    // transitions performed through its admission methods.
    void notify_scheduler_change() noexcept;

   private:
    core::Scheduler& scheduler_;
    using Clock = std::chrono::steady_clock;
    struct LeaseRecord {
        Clock::time_point deadline{};
        bool reaping = false;
        std::optional<TaskPeerIdentity> owner;
    };

    [[nodiscard]] bool owner_matches(const LeaseRecord& lease,
                                     const std::optional<TaskPeerIdentity>& peer) const noexcept;
    [[nodiscard]] std::optional<core::TaskSnapshot> record_lease(
        std::optional<core::TaskSnapshot> snapshot, std::optional<TaskPeerIdentity> peer);
    [[nodiscard]] bool pending_owner_matches(core::TaskId task_id,
                                             const std::optional<TaskPeerIdentity>& peer) noexcept;
    void observe_dispatch(const core::TaskSnapshot& snapshot) noexcept;
    void notify_state_change() noexcept;

    // Serializes submission/resource registration with dispatch so a queued
    // task cannot become running before its registrar has completed.
    std::mutex admission_mutex_;
    std::chrono::milliseconds lease_timeout_;
    bool bind_leases_to_process_ = false;
    TaskDispatchObserver dispatch_observer_ = nullptr;
    void* dispatch_observer_context_ = nullptr;
    std::mutex lease_mutex_;
    std::mutex state_mutex_;
    std::condition_variable state_condition_;
    std::uint64_t state_generation_ = 0;
    std::unordered_map<core::TaskId, LeaseRecord> active_leases_;
    std::unordered_map<core::TaskId, LeaseRecord> pending_leases_;
};

}  // namespace glimmer::control
