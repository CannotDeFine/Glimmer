#include "glimmer/backend/task_backend.h"

#include <cstdlib>
#include <iostream>
#include <string_view>

namespace {

using glimmer::backend::SimulatedBackend;
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

void test_submission_progress_and_completion() {
    SimulatedBackend backend(2);
    expect(backend.submit(TaskSubmission{.task_id = 1, .work_units = 4}),
           "valid task should be submitted");
    expect(backend.active_task_count() == 1, "submitted task should be active");
    expect(backend.poll(1).status == TaskStatus::kPending, "task should initially be pending");

    backend.advance();
    expect(backend.poll(1).status == TaskStatus::kPending,
           "task should remain pending before completion steps elapse");
    backend.advance();
    expect(backend.poll(1).status == TaskStatus::kCompleted,
           "task should complete after explicit progress");
    expect(backend.active_task_count() == 0, "completed task should not remain active");
}

void test_cancellation_and_invalid_submission() {
    SimulatedBackend backend;
    expect(!backend.submit(TaskSubmission{}), "empty task should be rejected");
    expect(backend.submit(TaskSubmission{.task_id = 2, .work_units = 1}),
           "valid task should be submitted");
    expect(!backend.submit(TaskSubmission{.task_id = 2, .work_units = 1}),
           "duplicate task should be rejected");
    expect(backend.cancel(2), "pending task should be cancellable");
    expect(backend.poll(2).status == TaskStatus::kCancelled,
           "cancelled task state should be observable");
    expect(!backend.cancel(2), "terminal task should not be cancelled twice");
    expect(backend.poll(99).status == TaskStatus::kUnknown,
           "unknown task should be reported as unknown");
}

void test_scheduler_and_backend_execution_boundary() {
    Scheduler scheduler(64);
    SimulatedBackend backend(2);
    const auto admission = scheduler.submit(TaskSpec{.tenant_id = "tenant-a", .memory_bytes = 16});
    expect(admission.accepted(), "scheduler should admit the task");

    const auto dispatched = scheduler.dispatch_next();
    expect(dispatched.has_value(), "scheduler should dispatch the admitted task");
    if (!dispatched.has_value()) {
        return;
    }
    expect(backend.submit(TaskSubmission{.task_id = dispatched->task_id, .work_units = 1}),
           "backend should accept the dispatched task");
    expect(backend.poll(dispatched->task_id).status == TaskStatus::kPending,
           "backend task should start pending");
    backend.advance();
    expect(backend.poll(dispatched->task_id).status == TaskStatus::kPending,
           "backend task should remain pending before completion");
    backend.advance();
    expect(backend.poll(dispatched->task_id).status == TaskStatus::kCompleted,
           "backend task should complete");
    expect(scheduler.complete(dispatched->task_id),
           "scheduler should complete after backend completion");
    expect(scheduler.usage().used_bytes() == 0,
           "scheduler quota should be released after backend completion");
}

}  // namespace

int main() {
    test_submission_progress_and_completion();
    test_cancellation_and_invalid_submission();
    test_scheduler_and_backend_execution_boundary();
    return EXIT_SUCCESS;
}
