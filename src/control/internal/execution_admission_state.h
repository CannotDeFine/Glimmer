#pragma once

#include "glimmer/control/task_execution_admission.h"

namespace glimmer::control {

// Private state used by the versioned admission service.
class ExecutionAdmissionState final {
   public:
    // The optional final hook exercises pending-record allocation failure.
    // Production callers leave it null; it runs under the admission lock.
    explicit ExecutionAdmissionState(
        core::Scheduler& scheduler,
        std::chrono::milliseconds lease_timeout = std::chrono::milliseconds::zero(),
        bool bind_leases_to_process = false, TaskDispatchObserver dispatch_observer = nullptr,
        void* dispatch_observer_context = nullptr,
        TaskCompletionObserver completion_observer = nullptr,
        void* completion_observer_context = nullptr,
        std::optional<NativeEpochExtent> native_epoch_policy = std::nullopt,
        void (*before_pending_insert)() = nullptr) noexcept;

    [[nodiscard]] core::SubmitResult submit(const TaskAdmissionRequest& request,
                                            TaskResourceRegistrar registrar,
                                            void* registrar_context,
                                            std::optional<TaskPeerIdentity> peer = std::nullopt);
    [[nodiscard]] core::SubmitResult submit_cooperative(const TaskAdmissionRequest& request,
                                                        TaskResourceRegistrar registrar,
                                                        void* registrar_context,
                                                        std::optional<TaskPeerIdentity> peer);
    [[nodiscard]] bool cancel_pending_cooperative(core::TaskId task_id,
                                                  std::optional<TaskPeerIdentity> peer);
    [[nodiscard]] bool cancel(core::TaskId task_id);
    [[nodiscard]] bool cancel_queued(core::TaskId task_id,
                                     std::optional<TaskPeerIdentity> peer = std::nullopt);
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
    [[nodiscard]] bool forget_terminal(core::TaskId task_id);
    // Cooperative execution requires process-bound leases and a nonzero
    // dispatch generation. Backend suspension is reported only after the
    // physical kernel has stopped; CUDA handles never enter this service.
    [[nodiscard]] std::optional<core::TaskSnapshot> claim_execution(
        core::TaskId task_id, std::optional<TaskPeerIdentity> peer);
    [[nodiscard]] bool supports_cooperative_execution() const noexcept {
        return bind_leases_to_process_;
    }
    [[nodiscard]] std::optional<bool> retreat_requested(TaskExecutionIdentity execution,
                                                        std::optional<TaskPeerIdentity> peer);
    [[nodiscard]] bool suspend_execution(TaskExecutionIdentity execution,
                                         std::optional<TaskPeerIdentity> peer);
    [[nodiscard]] bool heartbeat_execution(TaskExecutionIdentity execution,
                                           std::optional<TaskPeerIdentity> peer);
    [[nodiscard]] bool complete_execution(TaskExecutionIdentity execution,
                                          std::optional<TaskPeerIdentity> peer);
    [[nodiscard]] bool fail_execution(TaskExecutionIdentity execution,
                                      std::optional<TaskPeerIdentity> peer);
    [[nodiscard]] bool cancel_execution(TaskExecutionIdentity execution,
                                        std::optional<TaskPeerIdentity> peer);
    // Default-off, process-bound, distinct authorization. Registration follows
    // submit's contract; work_units must equal extent.max_calls. No CUDA proof
    // is supplied by these methods. Only a physically stopped backend may
    // report completion/failure; uncertain reports retain ownership.
    [[nodiscard]] core::SubmitResult submit_native_epoch(const TaskAdmissionRequest& request,
                                                         NativeEpochExtent extent,
                                                         TaskResourceRegistrar registrar,
                                                         void* registrar_context,
                                                         std::optional<TaskPeerIdentity> peer);
    [[nodiscard]] std::optional<NativeEpochGrant> claim_native_epoch(
        core::TaskId task_id, std::optional<TaskPeerIdentity> peer);
    [[nodiscard]] std::optional<NativeEpochExtent> pending_native_epoch_extent(
        core::TaskId task_id, std::optional<TaskPeerIdentity> peer);
    [[nodiscard]] std::optional<bool> native_epoch_retreat_requested(
        TaskExecutionIdentity execution, std::optional<TaskPeerIdentity> peer);
    [[nodiscard]] bool heartbeat_native_epoch(TaskExecutionIdentity execution,
                                              std::optional<TaskPeerIdentity> peer);
    [[nodiscard]] bool finish_native_epoch(TaskExecutionIdentity execution,
                                           std::uint32_t issued_calls, bool has_failed,
                                           std::optional<TaskPeerIdentity> peer);
    [[nodiscard]] bool cancel_pending_native_epoch(core::TaskId task_id,
                                                   std::optional<TaskPeerIdentity> peer);
    // Reclaims expired legacy running leases or physically stopped pending
    // leases. Cooperative running work retains capacity until an authenticated
    // physical stop or terminal report: a deadline alone is not a CUDA fence.
    // Returns true when a task reached a terminal state and quota was released.
    [[nodiscard]] bool reap_expired();
    [[nodiscard]] std::optional<core::TaskSnapshot> find(core::TaskId task_id) const;
    [[nodiscard]] core::SchedulerStats stats() const;
    [[nodiscard]] TaskExecutionLeaseStatus lease_status() const;

   private:
    core::Scheduler& scheduler_;
    using Clock = std::chrono::steady_clock;
    struct LeaseRecord {
        Clock::time_point deadline{};
        Clock::time_point submitted_at{};
        Clock::time_point started_at{};
        bool reaping = false;
        std::optional<TaskPeerIdentity> owner;
        bool is_cooperative = false;
        Clock::time_point queued_at{};
        std::uint64_t queue_wait_microseconds = 0;
        std::uint64_t service_time_microseconds = 0;
        std::optional<NativeEpochExtent> native_epoch = std::nullopt;
        std::uint64_t acceptance_deadline_nanoseconds = 0;
    };

    enum class AdmissionKind : std::uint8_t { kLegacy, kCooperative, kNative };
    [[nodiscard]] core::SubmitResult submit_impl(const TaskAdmissionRequest& request,
                                                 TaskResourceRegistrar registrar,
                                                 void* registrar_context,
                                                 std::optional<TaskPeerIdentity> peer,
                                                 std::optional<NativeEpochExtent> extent,
                                                 AdmissionKind kind);
    // Requires admission_mutex_; identity/type checking never renews an extent.
    [[nodiscard]] bool native_epoch_owner_matches(
        TaskExecutionIdentity execution, const std::optional<TaskPeerIdentity>& peer) const;

    [[nodiscard]] bool owner_matches(const LeaseRecord& lease,
                                     const std::optional<TaskPeerIdentity>& peer) const noexcept;
    // These helpers require admission_mutex_. No observer runs inside them.
    [[nodiscard]] std::optional<core::TaskSnapshot> record_lease(
        std::optional<core::TaskSnapshot> snapshot, std::optional<TaskPeerIdentity> peer);
    [[nodiscard]] bool pending_owner_matches(core::TaskId task_id,
                                             const std::optional<TaskPeerIdentity>& peer) noexcept;
    [[nodiscard]] bool finish_execution(TaskExecutionIdentity execution,
                                        std::optional<TaskPeerIdentity> peer,
                                        core::TaskState state);
    static void stop_lease_timing(LeaseRecord& lease) noexcept;
    void observe_dispatch(const core::TaskSnapshot& snapshot) noexcept;
    void observe_completion(const core::TaskSnapshot& snapshot, core::TaskState terminal_state,
                            const std::optional<LeaseRecord>& lease) noexcept;
    void notify_state_change() noexcept;

    // Serializes lifecycle transitions and protects both lease maps. Scheduler
    // dispatch and lease publication form one transaction. The lock order is
    // admission -> scheduler; state_mutex_ is never held while acquiring it.
    // Observers run after unlocking, before notifying state waiters. The
    // registrar remains part of the serialized submission transaction.
    mutable std::mutex admission_mutex_;
    std::chrono::milliseconds lease_timeout_;
    bool bind_leases_to_process_ = false;
    const std::optional<NativeEpochExtent> native_epoch_policy;
    void (*const before_pending_insert)();
    TaskDispatchObserver dispatch_observer_ = nullptr;
    void* dispatch_observer_context_ = nullptr;
    TaskCompletionObserver completion_observer_ = nullptr;
    void* completion_observer_context_ = nullptr;
    std::mutex state_mutex_;
    std::condition_variable state_condition_;
    std::uint64_t state_generation_ = 0;
    std::unordered_map<core::TaskId, LeaseRecord> active_leases_;
    std::unordered_map<core::TaskId, LeaseRecord> pending_leases_;
};

}  // namespace glimmer::control
