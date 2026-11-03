#include "glimmer/control/task_execution_admission.h"

#include "internal/execution_admission_state.h"

#include <memory>

namespace glimmer::control {

struct TaskExecutionAdmissionService::Impl {
    Impl(core::MemoryBytes memory_limit_bytes, core::SchedulerOptions scheduler_options,
         std::chrono::milliseconds lease_timeout, bool bind_leases_to_process,
         TaskDispatchObserver dispatch_observer, void* dispatch_observer_context,
         TaskCompletionObserver completion_observer, void* completion_observer_context,
         std::optional<NativeEpochExtent> native_epoch_policy)
        : scheduler(memory_limit_bytes, scheduler_options),
          state(scheduler, lease_timeout, bind_leases_to_process, dispatch_observer,
                dispatch_observer_context, completion_observer, completion_observer_context,
                native_epoch_policy) {}

    core::Scheduler scheduler;
    ExecutionAdmissionState state;
};

TaskExecutionAdmissionService::TaskExecutionAdmissionService(
    core::MemoryBytes memory_limit_bytes, core::SchedulerOptions scheduler_options,
    std::chrono::milliseconds lease_timeout, bool bind_leases_to_process,
    TaskDispatchObserver dispatch_observer, void* dispatch_observer_context,
    TaskCompletionObserver completion_observer, void* completion_observer_context,
    std::optional<NativeEpochExtent> native_epoch_policy)
    : impl_(std::make_unique<Impl>(memory_limit_bytes, scheduler_options, lease_timeout,
                                   bind_leases_to_process, dispatch_observer,
                                   dispatch_observer_context, completion_observer,
                                   completion_observer_context, native_epoch_policy)) {}

TaskExecutionAdmissionService::~TaskExecutionAdmissionService() = default;

core::SubmitResult TaskExecutionAdmissionService::submit(const TaskAdmissionRequest& request,
                                                         TaskResourceRegistrar registrar,
                                                         void* context,
                                                         std::optional<TaskPeerIdentity> peer) {
    return impl_->state.submit(request, registrar, context, peer);
}

core::SubmitResult TaskExecutionAdmissionService::submit_cooperative(
    const TaskAdmissionRequest& request, TaskResourceRegistrar registrar, void* context,
    std::optional<TaskPeerIdentity> peer) {
    return impl_->state.submit_cooperative(request, registrar, context, peer);
}

bool TaskExecutionAdmissionService::cancel_pending_cooperative(
    core::TaskId id, std::optional<TaskPeerIdentity> peer) {
    return impl_->state.cancel_pending_cooperative(id, peer);
}

bool TaskExecutionAdmissionService::cancel(core::TaskId id) {
    return impl_->state.cancel(id);
}

bool TaskExecutionAdmissionService::cancel_queued(core::TaskId id,
                                                  std::optional<TaskPeerIdentity> peer) {
    return impl_->state.cancel_queued(id, peer);
}

std::optional<core::TaskSnapshot> TaskExecutionAdmissionService::claim_next(
    std::optional<TaskPeerIdentity> peer) {
    return impl_->state.claim_next(peer);
}

std::optional<core::TaskSnapshot> TaskExecutionAdmissionService::claim(
    core::TaskId id, std::optional<TaskPeerIdentity> peer) {
    return impl_->state.claim(id, peer);
}

std::optional<core::TaskSnapshot> TaskExecutionAdmissionService::wait_claim(
    core::TaskId id, std::chrono::milliseconds timeout, std::optional<TaskPeerIdentity> peer) {
    return impl_->state.wait_claim(id, timeout, peer);
}

bool TaskExecutionAdmissionService::heartbeat(core::TaskId id,
                                              std::optional<TaskPeerIdentity> peer) {
    return impl_->state.heartbeat(id, peer);
}

bool TaskExecutionAdmissionService::complete(core::TaskId id,
                                             std::optional<TaskPeerIdentity> peer) {
    return impl_->state.complete(id, peer);
}

bool TaskExecutionAdmissionService::fail(core::TaskId id, std::optional<TaskPeerIdentity> peer) {
    return impl_->state.fail(id, peer);
}

bool TaskExecutionAdmissionService::forget_terminal(core::TaskId id) {
    return impl_->state.forget_terminal(id);
}

std::optional<core::TaskSnapshot> TaskExecutionAdmissionService::claim_execution(
    core::TaskId id, std::optional<TaskPeerIdentity> peer) {
    return impl_->state.claim_execution(id, peer);
}

bool TaskExecutionAdmissionService::supports_cooperative_execution() const noexcept {
    return impl_->state.supports_cooperative_execution();
}

std::optional<bool> TaskExecutionAdmissionService::retreat_requested(
    TaskExecutionIdentity execution, std::optional<TaskPeerIdentity> peer) {
    return impl_->state.retreat_requested(execution, peer);
}

bool TaskExecutionAdmissionService::suspend_execution(TaskExecutionIdentity execution,
                                                      std::optional<TaskPeerIdentity> peer) {
    return impl_->state.suspend_execution(execution, peer);
}

bool TaskExecutionAdmissionService::heartbeat_execution(TaskExecutionIdentity execution,
                                                        std::optional<TaskPeerIdentity> peer) {
    return impl_->state.heartbeat_execution(execution, peer);
}

bool TaskExecutionAdmissionService::complete_execution(TaskExecutionIdentity execution,
                                                       std::optional<TaskPeerIdentity> peer) {
    return impl_->state.complete_execution(execution, peer);
}

bool TaskExecutionAdmissionService::fail_execution(TaskExecutionIdentity execution,
                                                   std::optional<TaskPeerIdentity> peer) {
    return impl_->state.fail_execution(execution, peer);
}

bool TaskExecutionAdmissionService::cancel_execution(TaskExecutionIdentity execution,
                                                     std::optional<TaskPeerIdentity> peer) {
    return impl_->state.cancel_execution(execution, peer);
}

core::SubmitResult TaskExecutionAdmissionService::submit_native_epoch(
    const TaskAdmissionRequest& request, NativeEpochExtent extent, TaskResourceRegistrar registrar,
    void* context, std::optional<TaskPeerIdentity> peer) {
    return impl_->state.submit_native_epoch(request, extent, registrar, context, peer);
}

std::optional<NativeEpochGrant> TaskExecutionAdmissionService::claim_native_epoch(
    core::TaskId id, std::optional<TaskPeerIdentity> peer) {
    return impl_->state.claim_native_epoch(id, peer);
}

std::optional<NativeEpochExtent> TaskExecutionAdmissionService::pending_native_epoch_extent(
    core::TaskId id, std::optional<TaskPeerIdentity> peer) {
    return impl_->state.pending_native_epoch_extent(id, peer);
}

std::optional<bool> TaskExecutionAdmissionService::native_epoch_retreat_requested(
    TaskExecutionIdentity execution, std::optional<TaskPeerIdentity> peer) {
    return impl_->state.native_epoch_retreat_requested(execution, peer);
}

bool TaskExecutionAdmissionService::heartbeat_native_epoch(TaskExecutionIdentity execution,
                                                           std::optional<TaskPeerIdentity> peer) {
    return impl_->state.heartbeat_native_epoch(execution, peer);
}

bool TaskExecutionAdmissionService::finish_native_epoch(TaskExecutionIdentity execution,
                                                        std::uint32_t issued_calls, bool has_failed,
                                                        std::optional<TaskPeerIdentity> peer) {
    return impl_->state.finish_native_epoch(execution, issued_calls, has_failed, peer);
}

bool TaskExecutionAdmissionService::cancel_pending_native_epoch(
    core::TaskId id, std::optional<TaskPeerIdentity> peer) {
    return impl_->state.cancel_pending_native_epoch(id, peer);
}

bool TaskExecutionAdmissionService::reap_expired() {
    return impl_->state.reap_expired();
}

std::optional<core::TaskSnapshot> TaskExecutionAdmissionService::find(core::TaskId id) const {
    return impl_->state.find(id);
}

core::SchedulerStats TaskExecutionAdmissionService::stats() const {
    return impl_->state.stats();
}

TaskExecutionLeaseStatus TaskExecutionAdmissionService::lease_status() const {
    return impl_->state.lease_status();
}

}  // namespace glimmer::control
