#include "glimmer/core/scheduler.h"
#include "glimmer/core/scheduler_mode.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace {

using glimmer::core::MemoryBytes;
using glimmer::core::Scheduler;
using glimmer::core::SchedulerMode;
using glimmer::core::SubmitStatus;
using glimmer::core::TaskId;
using glimmer::core::TaskSpec;
using glimmer::core::TaskState;

void expect(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "Test failed: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

void expect_state(const Scheduler& scheduler, TaskId task_id, TaskState expected_state,
                  std::string_view message) {
    const auto snapshot = scheduler.find(task_id);
    expect(snapshot.has_value(), "task snapshot should exist");
    if (snapshot.has_value()) {
        expect(snapshot.value().state == expected_state, message);
    }
}

TaskId submit(Scheduler& scheduler, std::string_view tenant_id, MemoryBytes memory_bytes,
              std::uint32_t weight = 1) {
    auto result = scheduler.submit(TaskSpec{
        .tenant_id = std::string(tenant_id), .memory_bytes = memory_bytes, .weight = weight});
    expect(result.accepted(), "task should be accepted");
    return result.task_id;
}

void test_admission_dispatch_and_completion() {
    Scheduler scheduler(100);
    const TaskId task_id = submit(scheduler, "tenant-a", 60);
    const auto rejected = scheduler.submit(TaskSpec{.tenant_id = "tenant-b", .memory_bytes = 41});
    expect(rejected.status == SubmitStatus::kQuotaExceeded, "task beyond quota should be rejected");
    expect(scheduler.queued_task_count() == 1, "accepted task should be queued");

    const auto dispatched = scheduler.dispatch_next();
    expect(dispatched.has_value(), "queued task should be dispatched");
    if (!dispatched.has_value()) {
        return;
    }
    expect(dispatched->task_id == task_id, "unexpected task was dispatched");
    expect(dispatched->state == TaskState::kRunning, "dispatched task should be running");
    expect(scheduler.running_task_count() == 1, "one task should be running");
    expect(scheduler.usage().allocated_bytes == 60, "running task should consume quota");
    expect(!scheduler.dispatch_next().has_value(), "a second task cannot run concurrently");
    expect(scheduler.complete(task_id), "running task should complete");
    expect(scheduler.usage().used_bytes() == 0, "completed task should release quota");
    expect_state(scheduler, task_id, TaskState::kCompleted,
                 "completed task state should be observable");
    expect(scheduler.forget(task_id), "terminal task should be removable");
    expect(!scheduler.find(task_id).has_value(), "forgotten task should no longer be observable");
}

void test_weighted_round_robin_order() {
    Scheduler scheduler(100);
    const TaskId a1 = submit(scheduler, "tenant-a", 1, 2);
    const TaskId a2 = submit(scheduler, "tenant-a", 1, 2);
    const TaskId a3 = submit(scheduler, "tenant-a", 1, 2);
    const TaskId b1 = submit(scheduler, "tenant-b", 1, 1);
    const TaskId b2 = submit(scheduler, "tenant-b", 1, 1);

    const std::vector<TaskId> expected{a1, a2, b1, a3, b2};
    for (const TaskId expected_id : expected) {
        const auto dispatched = scheduler.dispatch_next();
        expect(dispatched.has_value(), "weighted queue should dispatch a task");
        if (!dispatched.has_value()) {
            return;
        }
        expect(dispatched->task_id == expected_id, "weighted round-robin order was not preserved");
        expect(scheduler.complete(expected_id), "dispatched task should complete");
    }
    expect(!scheduler.dispatch_next().has_value(), "all queued tasks should be consumed");
}

void test_cancellation_and_failure_release_quota() {
    Scheduler scheduler(100);
    const TaskId queued_id = submit(scheduler, "tenant-a", 40);
    expect(scheduler.cancel(queued_id), "queued task should be cancellable");
    expect(scheduler.usage().used_bytes() == 0, "queued cancellation should release reservation");
    expect_state(scheduler, queued_id, TaskState::kCancelled,
                 "cancelled task state should be observable");
    expect(scheduler.forget(queued_id), "cancelled task should be removable");

    const TaskId running_id = submit(scheduler, "tenant-a", 50);
    const auto dispatched = scheduler.dispatch_next();
    expect(dispatched.has_value() && dispatched->task_id == running_id,
           "task should dispatch before failure");
    expect(!scheduler.forget(running_id), "running task should not be removable");
    expect(scheduler.fail(running_id), "running task should fail");
    expect(scheduler.usage().used_bytes() == 0, "failed task should release quota");
    expect_state(scheduler, running_id, TaskState::kFailed,
                 "failed task state should be observable");
    expect(!scheduler.complete(running_id), "terminal task should not complete twice");
    expect(scheduler.forget(running_id), "failed task should be removable");
}

void test_input_validation_and_tenant_weight_consistency() {
    Scheduler scheduler(100);
    expect(scheduler.submit(TaskSpec{}).status == SubmitStatus::kInvalidTask,
           "empty task should be rejected");
    expect(submit(scheduler, "tenant-a", 10, 3) != 0, "valid task should have an id");
    const auto mismatched =
        scheduler.submit(TaskSpec{.tenant_id = "tenant-a", .memory_bytes = 10, .weight = 2});
    expect(mismatched.status == SubmitStatus::kTenantWeightMismatch,
           "tenant weight should remain stable");
}

void test_scheduler_mode_configuration() {
    const auto off_mode = glimmer::core::parse_scheduler_mode("off");
    const auto observe_mode = glimmer::core::parse_scheduler_mode("observe");
    const auto enforce_mode = glimmer::core::parse_scheduler_mode("enforce");
    expect(off_mode.has_value() && off_mode.value() == SchedulerMode::kOff,
           "off mode should parse");
    expect(observe_mode.has_value() && observe_mode.value() == SchedulerMode::kObserve,
           "observe mode should parse");
    expect(enforce_mode.has_value() && enforce_mode.value() == SchedulerMode::kEnforce,
           "enforce mode should parse");
    expect(!glimmer::core::parse_scheduler_mode("invalid").has_value(),
           "invalid mode should be rejected");
    expect(glimmer::core::scheduler_mode_name(SchedulerMode::kObserve) == "observe",
           "mode name should be stable");
}

}  // namespace

int main() {
    test_admission_dispatch_and_completion();
    test_weighted_round_robin_order();
    test_cancellation_and_failure_release_quota();
    test_input_validation_and_tenant_weight_consistency();
    test_scheduler_mode_configuration();
    return EXIT_SUCCESS;
}
