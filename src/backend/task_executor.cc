#include "glimmer/backend/task_backend.h"

namespace glimmer::backend {

TaskExecutor::TaskExecutor(core::Scheduler& scheduler, TaskBackend& backend) noexcept
    : scheduler_(scheduler), backend_(backend) {}

ExecutorStepResult TaskExecutor::step() {
    if (!active_task_id_.has_value()) {
        const std::optional<core::TaskSnapshot> dispatched = scheduler_.dispatch_next();
        if (!dispatched.has_value()) {
            return {.status = ExecutorStepStatus::kIdle, .task_id = std::nullopt};
        }

        const core::TaskId task_id = dispatched->task_id;
        bool submitted = false;
        try {
            submitted = backend_.submit({.task_id = task_id, .work_units = dispatched->work_units});
        } catch (...) {
            submitted = false;
        }
        if (!submitted) {
            if (!scheduler_.fail(task_id)) {
                // Keep ownership visible so a later step can retry the
                // scheduler transition if accounting recovery temporarily
                // fails. Do not silently orphan a running task.
                active_task_id_ = task_id;
            }
            return {.status = ExecutorStepStatus::kBackendError, .task_id = task_id};
        }
        active_task_id_ = task_id;
        return {.status = ExecutorStepStatus::kSubmitted, .task_id = task_id};
    }

    const core::TaskId task_id = *active_task_id_;
    TaskPollResult poll_result{.status = TaskStatus::kUnknown};
    try {
        poll_result = backend_.poll(task_id);
    } catch (...) {
        // A backend exception is an execution failure. Keep it inside the
        // scheduler boundary and use the existing unknown-state recovery path.
        poll_result.status = TaskStatus::kUnknown;
    }
    switch (poll_result.status) {
        case TaskStatus::kPending:
            return {.status = ExecutorStepStatus::kPending, .task_id = task_id};
        case TaskStatus::kCompleted:
            if (!scheduler_.complete(task_id)) {
                return {.status = ExecutorStepStatus::kBackendError, .task_id = task_id};
            }
            active_task_id_.reset();
            return {.status = ExecutorStepStatus::kCompleted, .task_id = task_id};
        case TaskStatus::kFailed:
            if (!scheduler_.fail(task_id)) {
                return {.status = ExecutorStepStatus::kBackendError, .task_id = task_id};
            }
            active_task_id_.reset();
            return {.status = ExecutorStepStatus::kFailed, .task_id = task_id};
        case TaskStatus::kCancelled:
            if (!scheduler_.cancel(task_id)) {
                return {.status = ExecutorStepStatus::kBackendError, .task_id = task_id};
            }
            active_task_id_.reset();
            return {.status = ExecutorStepStatus::kCancelled, .task_id = task_id};
        case TaskStatus::kUnknown:
            if (scheduler_.fail(task_id)) {
                active_task_id_.reset();
            }
            return {.status = ExecutorStepStatus::kBackendError, .task_id = task_id};
    }
    return {.status = ExecutorStepStatus::kBackendError, .task_id = task_id};
}

std::optional<core::TaskId> TaskExecutor::active_task() const noexcept {
    return active_task_id_;
}

}  // namespace glimmer::backend
