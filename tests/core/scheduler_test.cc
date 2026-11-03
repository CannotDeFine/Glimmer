#include "glimmer/core/scheduler.h"
#include "glimmer/core/scheduler_mode.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <string>
#include <string_view>
#include <atomic>
#include <vector>

namespace {

using glimmer::core::MemoryBytes;
using glimmer::core::Scheduler;
using glimmer::core::SchedulerMode;
using glimmer::core::SchedulerOptions;
using glimmer::core::SchedulingPolicy;
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

void test_configurable_concurrent_dispatch() {
    Scheduler scheduler(100, SchedulerOptions{.max_running_tasks = 2});
    const TaskId first_id = submit(scheduler, "tenant-a", 20);
    const TaskId second_id = submit(scheduler, "tenant-a", 20);
    const TaskId third_id = submit(scheduler, "tenant-a", 20);

    const auto first_lease = scheduler.dispatch_next();
    const auto second_lease = scheduler.dispatch_next();
    expect(first_lease.has_value() && second_lease.has_value(),
           "scheduler should fill both configured running slots");
    if (!first_lease.has_value() || !second_lease.has_value()) {
        return;
    }
    expect(first_lease->task_id == first_id && second_lease->task_id == second_id,
           "concurrent dispatch should preserve queue order");
    expect(scheduler.running_task_count() == 2, "both running slots should be observable");
    expect(scheduler.usage().allocated_bytes == 40,
           "concurrent dispatch should charge both running tasks");
    expect(!scheduler.dispatch_next().has_value(),
           "scheduler should not exceed the configured running capacity");

    expect(scheduler.complete(first_id), "first concurrent task should complete");
    const auto third_lease = scheduler.dispatch_next();
    expect(third_lease.has_value() && third_lease->task_id == third_id,
           "a released slot should dispatch the next queued task");
    expect(scheduler.fail(second_id), "second concurrent task should fail");
    expect(scheduler.complete(third_id), "third concurrent task should complete");
    expect(scheduler.running_task_count() == 0, "all concurrent tasks should be terminal");
    expect(scheduler.usage().used_bytes() == 0,
           "all concurrent terminal transitions should release quota");
}

void test_zero_concurrency_falls_back_to_one() {
    Scheduler scheduler(100, SchedulerOptions{.max_running_tasks = 0});
    const TaskId task_id = submit(scheduler, "tenant-a", 20);
    const auto lease = scheduler.dispatch_next();
    expect(lease.has_value() && lease->task_id == task_id,
           "zero concurrency should retain one safe running slot");
    expect(scheduler.fail(task_id), "fallback running task should fail cleanly");
}

void test_queue_backpressure_releases_with_dispatch() {
    Scheduler scheduler(100, SchedulerOptions{.max_queued_tasks = 2});
    const TaskId first_id = submit(scheduler, "tenant-a", 10);
    const TaskId second_id = submit(scheduler, "tenant-a", 10);
    const auto rejected = scheduler.submit(
        TaskSpec{.tenant_id = "tenant-a", .memory_bytes = 10, .weight = 1, .work_units = 1});
    expect(rejected.status == SubmitStatus::kQueueFull,
           "a full waiting queue should reject new submissions explicitly");
    expect(scheduler.usage().reserved_bytes == 20,
           "queue backpressure should not reserve rejected work");
    const auto first_lease = scheduler.dispatch_next();
    expect(first_lease.has_value() && first_lease->task_id == first_id,
           "dispatch should free one waiting-queue slot");
    const auto accepted_after_dispatch = scheduler.submit(
        TaskSpec{.tenant_id = "tenant-a", .memory_bytes = 10, .weight = 1, .work_units = 1});
    expect(accepted_after_dispatch.accepted() && accepted_after_dispatch.task_id != second_id,
           "a new task should be accepted after queue space is freed");
    expect(scheduler.fail(first_id), "the first running task should fail cleanly");
    expect(scheduler.cancel_queued(second_id), "the second task should be cancellable");
    expect(scheduler.cancel_queued(accepted_after_dispatch.task_id),
           "the replacement task should be cancellable");
    expect(scheduler.usage().used_bytes() == 0,
           "queue backpressure test cleanup should release all quota");
}

void test_scheduler_stats_snapshot() {
    Scheduler scheduler(100, SchedulerOptions{.max_running_tasks = 2, .max_queued_tasks = 3});
    const auto empty = scheduler.stats();
    expect(empty.quota.limit_bytes == 100 && empty.total_task_count == 0 &&
               empty.queued_task_count == 0 && empty.running_task_count == 0,
           "empty scheduler stats should report zero tasks and the quota limit");

    const TaskId completed_id = submit(scheduler, "tenant-a", 20);
    const TaskId failed_id = submit(scheduler, "tenant-b", 20);
    const TaskId cancelled_id = submit(scheduler, "tenant-c", 20);
    const auto queued = scheduler.stats();
    expect(queued.total_task_count == 3 && queued.queued_task_count == 3 &&
               queued.running_task_count == 0 && queued.quota.reserved_bytes == 60,
           "queued scheduler stats should expose reservations and queue depth");

    expect(scheduler.dispatch_next().has_value() && scheduler.dispatch_next().has_value(),
           "stats test should dispatch both running slots");
    const auto running = scheduler.stats();
    expect(running.queued_task_count == 1 && running.running_task_count == 2 &&
               running.quota.reserved_bytes == 20 && running.quota.allocated_bytes == 40,
           "running scheduler stats should separate reserved and allocated bytes");

    expect(scheduler.complete(completed_id), "stats test should complete one task");
    expect(scheduler.fail(failed_id), "stats test should fail one task");
    expect(scheduler.cancel_queued(cancelled_id), "stats test should cancel one task");
    const auto terminal = scheduler.stats();
    expect(terminal.total_task_count == 3 && terminal.queued_task_count == 0 &&
               terminal.running_task_count == 0 && terminal.completed_task_count == 1 &&
               terminal.cancelled_task_count == 1 && terminal.failed_task_count == 1 &&
               terminal.quota.used_bytes() == 0 && terminal.max_running_tasks == 2 &&
               terminal.max_queued_tasks == 3 &&
               terminal.scheduling_policy == SchedulingPolicy::kWeightedRoundRobin,
           "terminal scheduler stats should expose states, limits, and released quota");
}

void test_fifo_policy_order() {
    Scheduler scheduler(100, SchedulerOptions{.scheduling_policy = SchedulingPolicy::kFifo});
    const TaskId first = submit(scheduler, "tenant-a", 1, 4);
    const TaskId second = submit(scheduler, "tenant-b", 1, 1);
    const TaskId third = submit(scheduler, "tenant-a", 1, 4);

    for (const TaskId expected_id : {first, second, third}) {
        const auto dispatched = scheduler.dispatch_next();
        expect(dispatched.has_value() && dispatched->task_id == expected_id,
               "FIFO policy should dispatch tasks by submission order");
        expect(scheduler.complete(expected_id), "FIFO task should complete cleanly");
    }
    expect(scheduler.stats().scheduling_policy == SchedulingPolicy::kFifo,
           "FIFO policy should be visible in scheduler stats");
}

void test_scheduling_policy_configuration() {
    const auto fifo = glimmer::core::parse_scheduling_policy("fifo");
    const auto weighted = glimmer::core::parse_scheduling_policy("weighted_rr");
    const auto deficit = glimmer::core::parse_scheduling_policy("drr");
    const auto priority = glimmer::core::parse_scheduling_policy("priority");
    expect(fifo.has_value() && fifo.value() == SchedulingPolicy::kFifo, "FIFO policy should parse");
    expect(weighted.has_value() && weighted.value() == SchedulingPolicy::kWeightedRoundRobin,
           "weighted policy should parse");
    expect(deficit.has_value() && deficit.value() == SchedulingPolicy::kDeficitRoundRobin,
           "deficit policy should parse");
    expect(priority.has_value() && priority.value() == SchedulingPolicy::kPriority,
           "priority policy should parse");
    expect(!glimmer::core::parse_scheduling_policy("unknown").has_value(),
           "unknown policy should be rejected");
    expect(glimmer::core::scheduling_policy_name(SchedulingPolicy::kFifo) == "fifo",
           "FIFO policy name should be stable");
    expect(glimmer::core::scheduling_policy_name(SchedulingPolicy::kPriority) == "priority",
           "priority policy name should be stable");
    const auto invalid_policy = static_cast<SchedulingPolicy>(255);
    const Scheduler scheduler(100, SchedulerOptions{.scheduling_policy = invalid_policy});
    expect(scheduler.stats().scheduling_policy == SchedulingPolicy::kWeightedRoundRobin,
           "invalid direct API policy should fall back to weighted round-robin");
}

void test_priority_policy_order_and_tie_breaking() {
    Scheduler scheduler(100, SchedulerOptions{.scheduling_policy = SchedulingPolicy::kPriority});
    const auto low =
        scheduler.submit(TaskSpec{.tenant_id = "tenant-a", .memory_bytes = 1, .priority = 1});
    const auto high_first =
        scheduler.submit(TaskSpec{.tenant_id = "tenant-b", .memory_bytes = 1, .priority = 9});
    const auto high_second =
        scheduler.submit(TaskSpec{.tenant_id = "tenant-c", .memory_bytes = 1, .priority = 9});
    expect(low.accepted() && high_first.accepted() && high_second.accepted(),
           "priority tasks should be accepted");

    for (const TaskId expected_id : {high_first.task_id, high_second.task_id, low.task_id}) {
        const auto lease = scheduler.dispatch_next();
        expect(lease.has_value() && lease->task_id == expected_id,
               "priority policy should prefer larger priorities and preserve ties");
        expect(scheduler.complete(expected_id), "priority task should complete cleanly");
    }

    Scheduler specific_scheduler(
        100, SchedulerOptions{.scheduling_policy = SchedulingPolicy::kPriority});
    const auto specific_low = specific_scheduler.submit(
        TaskSpec{.tenant_id = "tenant-a", .memory_bytes = 1, .priority = 1});
    const auto specific_high = specific_scheduler.submit(
        TaskSpec{.tenant_id = "tenant-b", .memory_bytes = 1, .priority = 9});
    expect(specific_low.accepted() && specific_high.accepted(),
           "specific priority tasks should be accepted");
    expect(!specific_scheduler.dispatch_task(specific_low.task_id).has_value(),
           "specific dispatch must not bypass a higher-priority task");
    const auto specific_high_lease = specific_scheduler.dispatch_task(specific_high.task_id);
    expect(specific_high_lease.has_value() && specific_high_lease->task_id == specific_high.task_id,
           "specific dispatch should claim the priority head");
    expect(specific_scheduler.complete(specific_high.task_id),
           "specific priority task should complete cleanly");
    const auto specific_low_lease = specific_scheduler.dispatch_task(specific_low.task_id);
    expect(specific_low_lease.has_value() && specific_low_lease->task_id == specific_low.task_id,
           "specific dispatch should claim the remaining priority task");
    expect(specific_scheduler.complete(specific_low.task_id),
           "remaining specific priority task should complete cleanly");
}

void test_priority_latency_task_not_starved_by_training_backlog() {
    Scheduler scheduler(256, SchedulerOptions{.max_running_tasks = 1,
                                              .scheduling_policy = SchedulingPolicy::kPriority});
    std::vector<TaskId> training_tasks;
    training_tasks.reserve(32);
    for (int index = 0; index < 32; ++index) {
        const auto result = scheduler.submit(TaskSpec{.tenant_id = "training",
                                                      .memory_bytes = 1,
                                                      .weight = 1,
                                                      .work_units = 1,
                                                      .priority = 1});
        expect(result.accepted(), "training backlog task should be accepted");
        training_tasks.push_back(result.task_id);
    }
    const auto inference_result = scheduler.submit(TaskSpec{.tenant_id = "inference",
                                                            .memory_bytes = 1,
                                                            .weight = 1,
                                                            .work_units = 1,
                                                            .priority = 100});
    expect(inference_result.accepted(), "latency-sensitive inference task should be accepted");
    const TaskId inference_task = inference_result.task_id;

    const auto inference_lease = scheduler.dispatch_next();
    expect(inference_lease.has_value() && inference_lease->task_id == inference_task,
           "priority scheduling must dispatch inference before a training backlog");
    expect(scheduler.complete(inference_task), "inference task should complete cleanly");

    for (const TaskId training_task : training_tasks) {
        const auto training_lease = scheduler.dispatch_next();
        expect(training_lease.has_value() && training_lease->task_id == training_task,
               "training backlog should remain executable after inference admission");
        expect(scheduler.complete(training_task), "training task should complete cleanly");
    }
    const auto stats = scheduler.stats();
    expect(stats.completed_task_count == training_tasks.size() + 1,
           "priority starvation test should complete every task");
}

void test_priority_reserved_slot_protects_latency_class() {
    Scheduler scheduler(100, SchedulerOptions{.max_running_tasks = 2,
                                              .priority_reserved_slots = 1,
                                              .priority_reservation_threshold = 100,
                                              .scheduling_policy = SchedulingPolicy::kPriority});
    const auto first_training =
        scheduler.submit(TaskSpec{.tenant_id = "training", .memory_bytes = 1, .priority = 1});
    const auto second_training =
        scheduler.submit(TaskSpec{.tenant_id = "training", .memory_bytes = 1, .priority = 1});
    expect(first_training.accepted() && second_training.accepted(),
           "reserved-slot training tasks should be accepted");

    const auto first_lease = scheduler.dispatch_next();
    expect(first_lease.has_value() && first_lease->task_id == first_training.task_id,
           "training should use the unreserved slot before inference arrives");
    expect(!scheduler.dispatch_next(), "lower-priority work must not borrow the reserved slot");
    expect(!scheduler.dispatch_task(second_training.task_id),
           "specific low-priority dispatch must honor the reserved slot");

    const auto inference =
        scheduler.submit(TaskSpec{.tenant_id = "inference", .memory_bytes = 1, .priority = 100});
    expect(inference.accepted(), "reserved-slot inference task should be accepted");
    const auto inference_lease = scheduler.dispatch_next();
    expect(inference_lease.has_value() && inference_lease->task_id == inference.task_id,
           "the reserved slot should admit high-priority inference");
    const auto reservation_stats = scheduler.stats();
    expect(reservation_stats.priority_reservation_active &&
               reservation_stats.priority_reserved_slots == 1 &&
               reservation_stats.priority_reservation_threshold == 100,
           "scheduler stats should expose the configured priority reservation");
    expect(scheduler.set_priority_reserved_slots(99),
           "priority scheduler should accept a runtime reservation update");
    expect(scheduler.stats().priority_reserved_slots == 2,
           "runtime reservation updates should clamp to running capacity");
    expect(scheduler.set_priority_reserved_slots(1),
           "priority scheduler should restore a smaller runtime reservation");

    expect(scheduler.complete(first_training.task_id), "first training task should complete");
    const auto replacement_training = scheduler.dispatch_task(second_training.task_id);
    expect(replacement_training.has_value(),
           "inference in the reserved slot must not block the free unreserved slot");
    if (replacement_training.has_value()) {
        expect(scheduler.complete(second_training.task_id), "replacement training should complete");
    }
    expect(scheduler.complete(inference.task_id), "inference task should complete");
    expect(!scheduler.dispatch_next(), "all reserved-slot test tasks should have drained");
}

void test_dynamic_reservation_rejects_non_priority_policy() {
    Scheduler scheduler(100, SchedulerOptions{.max_running_tasks = 2});
    expect(!scheduler.set_priority_reserved_slots(1),
           "non-priority schedulers must reject adaptive reservation updates");
}

void test_specific_dispatch_preserves_policy_order() {
    Scheduler scheduler(100);
    const TaskId first = submit(scheduler, "tenant-a", 1);
    const TaskId second = submit(scheduler, "tenant-b", 1);
    expect(!scheduler.dispatch_task(second).has_value(),
           "a task behind the policy head must not bypass fairness");
    const auto first_lease = scheduler.dispatch_task(first);
    expect(first_lease.has_value() && first_lease->task_id == first,
           "the policy head should be claimable by its submitting process");
    expect(scheduler.complete(first), "the first specific lease should complete");
    const auto second_lease = scheduler.dispatch_task(second);
    expect(second_lease.has_value() && second_lease->task_id == second,
           "the next specific lease should become claimable after completion");
    expect(scheduler.complete(second), "the second specific lease should complete");
}

void test_deficit_round_robin_and_latency_stats() {
    Scheduler scheduler(
        100, SchedulerOptions{.scheduling_policy = SchedulingPolicy::kDeficitRoundRobin});
    const auto result = scheduler.submit(
        TaskSpec{.tenant_id = "tenant-a", .memory_bytes = 1, .weight = 2, .work_units = 2});
    expect(result.accepted(), "DRR task should be accepted");
    expect(!scheduler.dispatch_next().has_value(),
           "DRR should accumulate a quantum before dispatching an expensive task");
    const auto lease = scheduler.dispatch_next();
    expect(lease.has_value() && lease->task_id == result.task_id,
           "DRR should dispatch once the deficit covers task cost");
    expect(scheduler.complete(result.task_id), "DRR task should complete");
    const auto stats = scheduler.stats();
    expect(stats.total_queue_wait_microseconds >= stats.max_queue_wait_microseconds &&
               stats.total_service_time_microseconds >= stats.max_service_time_microseconds,
           "scheduler stats should expose consistent queue and service latency aggregates");
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
    expect(!scheduler.cancel_queued(queued_id),
           "a terminal task should not be cancellable through the queued-only path");
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
    expect(scheduler
                   .submit(TaskSpec{
                       .tenant_id = "tenant-a", .memory_bytes = 10, .weight = 1, .work_units = 0})
                   .status == SubmitStatus::kInvalidTask,
           "zero-work task should be rejected");
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

void test_concurrent_claim_is_single_owner() {
    Scheduler scheduler(100);
    const TaskId task_id = submit(scheduler, "tenant-a", 40);
    std::atomic<int> successful_claims = 0;
    std::vector<std::thread> claimers;
    claimers.reserve(8);
    for (int index = 0; index < 8; ++index) {
        claimers.emplace_back([&scheduler, &successful_claims] {
            const auto lease = scheduler.dispatch_next();
            if (lease.has_value()) {
                ++successful_claims;
            }
        });
    }
    for (std::thread& claimer : claimers) {
        claimer.join();
    }
    expect(successful_claims.load() == 1, "concurrent claimers must receive one lease");
    expect(scheduler.running_task_count() == 1, "one task should remain running after claim race");
    expect(scheduler.fail(task_id), "the sole running lease should fail cleanly");
    expect(scheduler.usage().used_bytes() == 0,
           "failed concurrent lease should release its reservation");
}

void test_cooperative_suspend_and_priority_resume() {
    Scheduler scheduler(100, SchedulerOptions{.max_queued_tasks = 1,
                                              .scheduling_policy = SchedulingPolicy::kPriority});
    const auto low =
        scheduler.submit(TaskSpec{.tenant_id = "low", .memory_bytes = 60, .priority = 1});
    expect(low.accepted(), "low-priority work should submit");
    const auto first = scheduler.dispatch_task(low.task_id);
    if (!first) {
        expect(false, "low-priority work should dispatch");
        return;
    }
    expect(!scheduler.retreat_requested(low.task_id),
           "running work should not retreat without higher priority queued work");
    const auto high =
        scheduler.submit(TaskSpec{.tenant_id = "high", .memory_bytes = 30, .priority = 9});
    expect(high.accepted() && scheduler.retreat_requested(low.task_id),
           "higher-priority arrival must request cooperative retreat");
    expect(!scheduler.dispatch_task(high.task_id).has_value(),
           "a retreat request must not release capacity before the backend stops");
    expect(scheduler.suspend(low.task_id) && !scheduler.suspend(low.task_id),
           "physical completion may suspend exactly once, even with a full submission queue");
    expect(scheduler.running_task_count() == 0 && scheduler.queued_task_count() == 2 &&
               scheduler.usage().allocated_bytes == 60 && scheduler.usage().reserved_bytes == 30,
           "suspension must retain committed memory while returning only the execution slot");
    expect(!scheduler.dispatch_task(low.task_id).has_value() &&
               scheduler.dispatch_task(high.task_id).has_value(),
           "priority must run high work before resuming the continuation");
    expect(scheduler.complete(high.task_id), "high-priority task must complete");
    for (unsigned int iteration = 0; iteration < 4; ++iteration) {
        const auto resumed = scheduler.dispatch_task(low.task_id);
        expect(resumed.has_value() && resumed->dispatch_sequence > first->dispatch_sequence &&
                   scheduler.usage().allocated_bytes == 60 && scheduler.usage().reserved_bytes == 0,
               "resumption must obtain a fresh dispatch generation without recommitting memory");
        expect(scheduler.suspend(low.task_id), "repeated continuation must suspend cleanly");
    }
    expect(scheduler.cancel_queued(low.task_id) && !scheduler.cancel_queued(low.task_id) &&
               scheduler.usage().used_bytes() == 0,
           "cancelling a stopped continuation must release committed memory exactly once");
    expect(!scheduler.retreat_requested(low.task_id) && !scheduler.suspend(999),
           "terminal and unknown tasks must not participate in retreat");
    Scheduler fifo(100, SchedulerOptions{.scheduling_policy = SchedulingPolicy::kFifo});
    const auto id = submit(fifo, "fifo", 10);
    expect(fifo.dispatch_task(id).has_value() && fifo.suspend(id) &&
               fifo.dispatch_task(id).has_value() && fifo.complete(id) &&
               fifo.usage().used_bytes() == 0,
           "non-priority continuation must preserve completion accounting");
}

void test_cooperative_continuations_across_policies() {
    for (const auto policy :
         {SchedulingPolicy::kFifo, SchedulingPolicy::kPriority,
          SchedulingPolicy::kWeightedRoundRobin, SchedulingPolicy::kDeficitRoundRobin}) {
        Scheduler scheduler(100, SchedulerOptions{.scheduling_policy = policy});
        const auto task = scheduler.submit(TaskSpec{
            .tenant_id = "continuation", .memory_bytes = 60, .weight = 2, .work_units = 3});
        expect(task.accepted(), "continuation task should submit");
        auto claim = [&] {
            std::optional<glimmer::core::TaskSnapshot> result;
            for (unsigned int attempt = 0; attempt < 5 && !result; ++attempt)
                result = scheduler.dispatch_task(task.task_id);
            return result;
        };
        const auto first = claim();
        if (!first) {
            expect(false, "task-specific claims must progress under every policy");
            return;
        }
        expect(scheduler.suspend(task.task_id), "running task should suspend after physical stop");
        const auto resumed = claim();
        expect(resumed.has_value() && resumed->dispatch_sequence > first->dispatch_sequence &&
                   scheduler.usage().allocated_bytes == 60,
               "every policy must resume with a fresh generation and retain quota once");
        expect(scheduler.complete(task.task_id) && scheduler.usage().used_bytes() == 0,
               "resumed completion must release quota exactly once");
    }
}

void test_cooperative_terminal_races() {
    for (unsigned int iteration = 0; iteration < 64; ++iteration) {
        Scheduler scheduler(100);
        const auto task = submit(scheduler, "race", 20);
        expect(scheduler.dispatch_task(task).has_value(), "race fixture should start");
        std::atomic<bool> start{false};
        bool suspended = false;
        bool completed = false;
        std::thread first([&] {
            while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
            suspended = scheduler.suspend(task);
        });
        std::thread second([&] {
            while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
            completed = scheduler.complete(task);
        });
        start.store(true, std::memory_order_release);
        first.join();
        second.join();
        expect(suspended != completed, "suspend and completion cannot both win the same run");
        if (suspended)
            expect(scheduler.cancel_queued(task), "stopped continuation can be cancelled");
        expect(scheduler.usage().used_bytes() == 0 && scheduler.running_task_count() == 0 &&
                   scheduler.queued_task_count() == 0,
               "terminal race leaves no duplicate quota or execution slot");

        Scheduler double_suspend(100);
        const auto duplicate = submit(double_suspend, "duplicate", 20);
        expect(double_suspend.dispatch_task(duplicate).has_value(),
               "double suspend fixture starts");
        std::atomic<bool> duplicate_start{false};
        bool first_suspended = false;
        bool second_suspended = false;
        std::thread third([&] {
            while (!duplicate_start.load(std::memory_order_acquire)) std::this_thread::yield();
            first_suspended = double_suspend.suspend(duplicate);
        });
        std::thread fourth([&] {
            while (!duplicate_start.load(std::memory_order_acquire)) std::this_thread::yield();
            second_suspended = double_suspend.suspend(duplicate);
        });
        duplicate_start.store(true, std::memory_order_release);
        third.join();
        fourth.join();
        expect(first_suspended != second_suspended && double_suspend.cancel_queued(duplicate) &&
                   double_suspend.usage().used_bytes() == 0,
               "concurrent suspension queues once and releases retained quota once");

        Scheduler cancel_race(100);
        const auto cancellable = submit(cancel_race, "cancel", 20);
        expect(cancel_race.dispatch_task(cancellable).has_value(), "cancel race fixture starts");
        std::atomic<bool> cancel_start{false};
        bool cancelled = false;
        std::thread fifth([&] {
            while (!cancel_start.load(std::memory_order_acquire)) std::this_thread::yield();
            static_cast<void>(cancel_race.suspend(cancellable));
        });
        std::thread sixth([&] {
            while (!cancel_start.load(std::memory_order_acquire)) std::this_thread::yield();
            cancelled = cancel_race.cancel(cancellable);
        });
        cancel_start.store(true, std::memory_order_release);
        fifth.join();
        sixth.join();
        expect(cancelled && cancel_race.usage().used_bytes() == 0 &&
                   cancel_race.running_task_count() == 0 && cancel_race.queued_task_count() == 0,
               "cancel and suspension race must release quota and leave no queued work");
    }
}

void test_pending_selection_preserves_history_and_continuations() {
    for (const auto policy : {SchedulingPolicy::kFifo, SchedulingPolicy::kPriority}) {
        Scheduler scheduler(
            100, SchedulerOptions{
                     .max_running_tasks = 2, .max_queued_tasks = 2, .scheduling_policy = policy});
        TaskId first_history = 0;
        for (unsigned int index = 0; index < 128; ++index) {
            const auto task = scheduler.submit(
                TaskSpec{.tenant_id = "history", .memory_bytes = 1, .priority = 100});
            expect(task.accepted(), "historical task should submit");
            if (index == 0) {
                first_history = task.task_id;
            }
            if (index % 3 == 0) {
                expect(scheduler.cancel_queued(task.task_id),
                       "history cancellation should release");
            } else {
                expect(scheduler.dispatch_task(task.task_id).has_value(),
                       "history task should dispatch");
                expect(index % 3 == 1 ? scheduler.complete(task.task_id)
                                      : scheduler.fail(task.task_id),
                       "history terminal transition should release");
            }
        }
        const auto cancelled =
            scheduler.submit(TaskSpec{.tenant_id = "empty", .memory_bytes = 1, .priority = 200});
        expect(cancelled.accepted() && scheduler.cancel_queued(cancelled.task_id) &&
                   scheduler.forget(cancelled.task_id),
               "cancelled forgotten candidate must leave no queue entry");

        const auto first =
            scheduler.submit(TaskSpec{.tenant_id = "active", .memory_bytes = 10, .priority = 7});
        expect(first.accepted(), "first continuation should submit");
        const auto first_lease = scheduler.dispatch_task(first.task_id);
        if (!first_lease) {
            expect(false, "first continuation should run");
            return;
        }
        const auto first_sequence = first_lease->dispatch_sequence;
        const auto second =
            scheduler.submit(TaskSpec{.tenant_id = "active", .memory_bytes = 10, .priority = 7});
        expect(second.accepted(), "second continuation should submit");
        const auto second_lease = scheduler.dispatch_task(second.task_id);
        if (!second_lease) {
            expect(false, "second continuation should run");
            return;
        }
        const auto second_sequence = second_lease->dispatch_sequence;
        const auto third =
            scheduler.submit(TaskSpec{.tenant_id = "active", .memory_bytes = 10, .priority = 7});
        const auto high =
            scheduler.submit(TaskSpec{.tenant_id = "active", .memory_bytes = 10, .priority = 9});
        expect(third.accepted() && high.accepted(), "pending candidates should fill queue");
        expect(
            scheduler.retreat_requested(first.task_id) == (policy == SchedulingPolicy::kPriority),
            "only priority policy should request retreat for actual pending high work");
        expect(!scheduler.dispatch_task(high.task_id).has_value(),
               "retreat request must not return occupied physical slots");
        expect(scheduler.suspend(first.task_id) && scheduler.suspend(second.task_id),
               "continuations may exceed submission limit and reorder the tenant deque");
        expect(scheduler.queued_task_count() == 4 && scheduler.usage().allocated_bytes == 20 &&
                   scheduler.usage().reserved_bytes == 20,
               "stopped work retains quota while all four candidates remain queued");
        expect(scheduler.submit(TaskSpec{.tenant_id = "extra", .memory_bytes = 1}).status ==
                   SubmitStatus::kQueueFull,
               "continuations must not weaken admission backpressure");
        const std::vector<TaskId> expected =
            policy == SchedulingPolicy::kPriority
                ? std::vector<TaskId>{high.task_id, first.task_id, second.task_id, third.task_id}
                : std::vector<TaskId>{first.task_id, second.task_id, third.task_id, high.task_id};
        for (std::size_t index = 0; index < expected.size(); ++index) {
            if (index + 1 < expected.size()) {
                expect(!scheduler.dispatch_task(expected[index + 1]).has_value(),
                       "task-specific peek must reject a candidate behind the policy head");
            }
            const auto lease = index % 2 == 0 ? scheduler.dispatch_task(expected[index])
                                              : scheduler.dispatch_next();
            if (!lease) {
                expect(false, "selected candidate should dispatch");
                return;
            }
            expect(lease->task_id == expected[index],
                   "selection must preserve global ID ties despite tenant deque order");
            if (expected[index] == first.task_id || expected[index] == second.task_id) {
                const auto previous =
                    expected[index] == first.task_id ? first_sequence : second_sequence;
                expect(lease->dispatch_sequence > previous,
                       "resumed continuation must obtain a fresh generation");
            }
            expect(scheduler.complete(expected[index]), "selected candidate should complete");
        }
        expect_state(scheduler, first_history, TaskState::kCancelled,
                     "selection must retain original historical queries");
        expect(!scheduler.forget(999999) && !scheduler.dispatch_next().has_value(),
               "unknown forget and empty dispatch must not manufacture candidates");
        const auto stats = scheduler.stats();
        expect(stats.total_task_count == 132 && stats.queued_task_count == 0 &&
                   stats.running_task_count == 0 && stats.quota.used_bytes() == 0,
               "historical records and terminal quota must reconcile");
    }
}

}  // namespace

int main() {
    test_admission_dispatch_and_completion();
    test_configurable_concurrent_dispatch();
    test_zero_concurrency_falls_back_to_one();
    test_queue_backpressure_releases_with_dispatch();
    test_scheduler_stats_snapshot();
    test_fifo_policy_order();
    test_scheduling_policy_configuration();
    test_specific_dispatch_preserves_policy_order();
    test_deficit_round_robin_and_latency_stats();
    test_priority_policy_order_and_tie_breaking();
    test_priority_latency_task_not_starved_by_training_backlog();
    test_priority_reserved_slot_protects_latency_class();
    test_dynamic_reservation_rejects_non_priority_policy();
    test_weighted_round_robin_order();
    test_cancellation_and_failure_release_quota();
    test_input_validation_and_tenant_weight_consistency();
    test_scheduler_mode_configuration();
    test_concurrent_claim_is_single_owner();
    test_cooperative_suspend_and_priority_resume();
    test_cooperative_continuations_across_policies();
    test_cooperative_terminal_races();
    test_pending_selection_preserves_history_and_continuations();
    return EXIT_SUCCESS;
}
