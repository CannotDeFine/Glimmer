#pragma once

#include "glimmer/control/task_admission.h"

#include <memory>

namespace glimmer::control {

// Owns its scheduler and serializes its admission state. No other admission
// service or external caller can mutate that scheduler. Construction may throw
// on allocation failure; no partially constructed service is published. Before
// destruction, the owner must stop and join backend work, terminate or cancel
// every accepted pending/active lease, release registrar-owned external
// resources, and ensure no concurrent calls remain. Destruction is not proof of
// physical stop and does not itself fence or tear down a CUDA context.
class TaskExecutionAdmissionService final {
   public:
    explicit TaskExecutionAdmissionService(
        core::MemoryBytes memory_limit_bytes, core::SchedulerOptions scheduler_options = {},
        std::chrono::milliseconds lease_timeout = std::chrono::milliseconds::zero(),
        bool bind_leases_to_process = false, TaskDispatchObserver dispatch_observer = nullptr,
        void* dispatch_observer_context = nullptr,
        TaskCompletionObserver completion_observer = nullptr,
        void* completion_observer_context = nullptr,
        std::optional<NativeEpochExtent> native_epoch_policy = std::nullopt);
    ~TaskExecutionAdmissionService();

    TaskExecutionAdmissionService(const TaskExecutionAdmissionService&) = delete;
    TaskExecutionAdmissionService& operator=(const TaskExecutionAdmissionService&) = delete;
    TaskExecutionAdmissionService(TaskExecutionAdmissionService&&) = delete;
    TaskExecutionAdmissionService& operator=(TaskExecutionAdmissionService&&) = delete;

    [[nodiscard]] core::SubmitResult submit(const TaskAdmissionRequest& request,
                                            TaskResourceRegistrar registrar,
                                            void* registrar_context,
                                            std::optional<TaskPeerIdentity> peer = std::nullopt);
    // Atomically publishes an owner-bound cooperative admission.
    [[nodiscard]] core::SubmitResult submit_cooperative(const TaskAdmissionRequest& request,
                                                        TaskResourceRegistrar registrar,
                                                        void* registrar_context,
                                                        std::optional<TaskPeerIdentity> peer);
    // Cancels only owner-bound cooperative work that has never been dispatched.
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
    // Cooperative execution requires process-bound leases and a nonzero
    // dispatch generation. Backend suspension is reported only after the
    // physical kernel has stopped; CUDA handles never enter this service.
    [[nodiscard]] std::optional<core::TaskSnapshot> claim_execution(
        core::TaskId task_id, std::optional<TaskPeerIdentity> peer);
    [[nodiscard]] bool supports_cooperative_execution() const noexcept;
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
    // A native grant is issued once. A lost reply is not safe to retry: the
    // owner must retain the lease until it proves non-issue or physical stop.
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

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace glimmer::control
