#include "glimmer/backend/task_backend.h"

#include <cstdlib>
#include <iostream>
#include <string_view>

namespace {

using glimmer::backend::ExecutorStepStatus;
using glimmer::backend::SimulatedBackend;
using glimmer::backend::TaskBackend;
using glimmer::backend::TaskExecutor;
using glimmer::backend::TaskPollResult;
using glimmer::backend::TaskStatus;
using glimmer::backend::TaskSubmission;
using glimmer::core::Scheduler;
using glimmer::core::TaskSpec;

void expect(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "Test failed: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

class RejectingBackend final : public TaskBackend {
   public:
    [[nodiscard]] bool submit(const TaskSubmission&) override {
        return false;
    }
    [[nodiscard]] TaskPollResult poll(glimmer::core::TaskId) const override {
        return {.status = TaskStatus::kUnknown};
    }
    [[nodiscard]] bool cancel(glimmer::core::TaskId) override {
        return false;
    }
};

class UnknownBackend final : public TaskBackend {
   public:
    [[nodiscard]] bool submit(const TaskSubmission&) override {
        return true;
    }
    [[nodiscard]] TaskPollResult poll(glimmer::core::TaskId) const override {
        return {.status = TaskStatus::kUnknown};
    }
    [[nodiscard]] bool cancel(glimmer::core::TaskId) override {
        return false;
    }
};

class ThrowingSubmitBackend final : public TaskBackend {
   public:
    [[nodiscard]] bool submit(const TaskSubmission&) override {
        throw 1;
    }
    [[nodiscard]] TaskPollResult poll(glimmer::core::TaskId) const override {
        return {.status = TaskStatus::kUnknown};
    }
    [[nodiscard]] bool cancel(glimmer::core::TaskId) override {
        return false;
    }
};

class ThrowingPollBackend final : public TaskBackend {
   public:
    [[nodiscard]] bool submit(const TaskSubmission&) override {
        return true;
    }
    [[nodiscard]] TaskPollResult poll(glimmer::core::TaskId) const override {
        throw 1;
    }
    [[nodiscard]] bool cancel(glimmer::core::TaskId) override {
        return false;
    }
};

void test_executor_drives_completion() {
    Scheduler scheduler(64);
    SimulatedBackend backend(2);
    TaskExecutor executor(scheduler, backend);
    const auto admission = scheduler.submit(
        TaskSpec{.tenant_id = "tenant-a", .memory_bytes = 16, .weight = 1, .work_units = 3});
    expect(admission.accepted(), "scheduler should admit the task");

    const auto submitted = executor.step();
    expect(submitted.status == ExecutorStepStatus::kSubmitted,
           "executor should submit the dispatched task");
    expect(submitted.task_id == admission.task_id, "executor should report the submitted task");
    expect(executor.active_task() == admission.task_id, "submitted task should be active");
    expect(scheduler.running_task_count() == 1, "scheduler should have one running task");

    const auto pending = executor.step();
    expect(pending.status == ExecutorStepStatus::kPending,
           "executor should report backend progress");
    backend.advance();
    expect(executor.step().status == ExecutorStepStatus::kPending,
           "task should remain pending before backend completion");
    backend.advance();
    const auto completed = executor.step();
    expect(completed.status == ExecutorStepStatus::kCompleted,
           "executor should complete the scheduler task");
    expect(!executor.active_task().has_value(), "completed task should no longer be active");
    expect(scheduler.usage().used_bytes() == 0, "completion should release scheduler quota");
}

void test_executor_reports_idle_and_failure() {
    Scheduler scheduler(64);
    SimulatedBackend backend;
    TaskExecutor executor(scheduler, backend);
    expect(executor.step().status == ExecutorStepStatus::kIdle,
           "empty scheduler should report idle");

    const auto admission = scheduler.submit(TaskSpec{.tenant_id = "tenant-a", .memory_bytes = 16});
    expect(admission.accepted(), "scheduler should admit the task");
    const auto submitted = executor.step();
    expect(submitted.status == ExecutorStepStatus::kSubmitted,
           "executor should submit the task before failure");
    expect(backend.cancel(admission.task_id), "backend task should be cancellable");
    const auto cancelled = executor.step();
    expect(cancelled.status == ExecutorStepStatus::kCancelled,
           "executor should propagate backend cancellation");
    expect(scheduler.usage().used_bytes() == 0, "cancellation should release scheduler quota");
}

void test_executor_releases_after_backend_errors() {
    Scheduler scheduler(64);
    RejectingBackend rejecting_backend;
    TaskExecutor rejecting_executor(scheduler, rejecting_backend);
    const auto rejected_admission =
        scheduler.submit(TaskSpec{.tenant_id = "tenant-a", .memory_bytes = 16});
    expect(rejected_admission.accepted(), "scheduler should admit the rejected backend task");
    const auto rejected = rejecting_executor.step();
    expect(rejected.status == ExecutorStepStatus::kBackendError,
           "backend submission failure should be reported");
    expect(!rejecting_executor.active_task().has_value(),
           "rejected backend task should not remain active");
    expect(scheduler.usage().used_bytes() == 0,
           "backend submission failure should release scheduler quota");

    Scheduler unknown_scheduler(64);
    UnknownBackend unknown_backend;
    TaskExecutor unknown_executor(unknown_scheduler, unknown_backend);
    const auto unknown_admission =
        unknown_scheduler.submit(TaskSpec{.tenant_id = "tenant-a", .memory_bytes = 16});
    expect(unknown_admission.accepted(), "scheduler should admit the unknown backend task");
    expect(unknown_executor.step().status == ExecutorStepStatus::kSubmitted,
           "unknown backend task should be submitted");
    expect(unknown_executor.step().status == ExecutorStepStatus::kBackendError,
           "unknown backend status should be reported as an error");
    expect(!unknown_executor.active_task().has_value(),
           "unknown backend task should not remain active after failure");
    expect(unknown_scheduler.usage().used_bytes() == 0,
           "unknown backend failure should release scheduler quota");
}

void test_executor_contains_backend_exceptions() {
    Scheduler submit_scheduler(64);
    ThrowingSubmitBackend submit_backend;
    TaskExecutor submit_executor(submit_scheduler, submit_backend);
    const auto submit_admission =
        submit_scheduler.submit(TaskSpec{.tenant_id = "tenant-a", .memory_bytes = 16});
    expect(submit_admission.accepted(), "scheduler should admit the throwing submit task");
    expect(submit_executor.step().status == ExecutorStepStatus::kBackendError,
           "throwing submission should become a backend error");
    expect(!submit_executor.active_task().has_value(),
           "throwing submission should not remain active");
    expect(submit_scheduler.usage().used_bytes() == 0,
           "throwing submission should release scheduler quota");

    Scheduler poll_scheduler(64);
    ThrowingPollBackend poll_backend;
    TaskExecutor poll_executor(poll_scheduler, poll_backend);
    const auto poll_admission =
        poll_scheduler.submit(TaskSpec{.tenant_id = "tenant-a", .memory_bytes = 16});
    expect(poll_admission.accepted(), "scheduler should admit the throwing poll task");
    expect(poll_executor.step().status == ExecutorStepStatus::kSubmitted,
           "throwing poll task should be submitted before polling");
    expect(poll_executor.step().status == ExecutorStepStatus::kBackendError,
           "throwing poll should become a backend error");
    expect(!poll_executor.active_task().has_value(), "throwing poll should clear active state");
    expect(poll_scheduler.usage().used_bytes() == 0,
           "throwing poll should release scheduler quota");
}

}  // namespace

int main() {
    test_executor_drives_completion();
    test_executor_reports_idle_and_failure();
    test_executor_releases_after_backend_errors();
    test_executor_contains_backend_exceptions();
    return EXIT_SUCCESS;
}
