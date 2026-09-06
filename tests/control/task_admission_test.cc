#include "glimmer/control/task_admission.h"

#include <array>
#include <atomic>
#include <cstdlib>
#include <chrono>
#include <iostream>
#include <latch>
#include <string_view>
#include <thread>

namespace {

using glimmer::control::TaskAdmissionRequest;
using glimmer::control::TaskAdmissionService;
using glimmer::control::TaskResourceRegistrar;
using glimmer::core::Scheduler;
using glimmer::core::SubmitStatus;
using glimmer::core::TaskId;

struct RegistrarState {
    bool should_register = true;
    int calls = 0;
    TaskId last_task_id = 0;
};

struct DispatchTraceState {
    std::size_t calls = 0;
    std::array<TaskId, 4> task_ids{};
    std::array<std::uint32_t, 4> priorities{};
    std::array<std::uint64_t, 4> sequences{};
};

struct CompletionTraceState {
    std::size_t calls = 0;
    std::array<TaskId, 4> task_ids{};
    std::array<glimmer::core::TaskState, 4> states{};
    std::array<std::uint64_t, 4> queue_wait_microseconds{};
    std::array<std::uint64_t, 4> service_time_microseconds{};
};

struct BlockingRegistrarState {
    std::latch entered{1};
    std::latch release{1};
};

bool register_resource(void* context, TaskId task_id) noexcept {
    if (context == nullptr) {
        return false;
    }
    auto* state = static_cast<RegistrarState*>(context);
    ++state->calls;
    state->last_task_id = task_id;
    return state->should_register;
}

bool register_blocking_resource(void* context, TaskId) noexcept {
    if (context == nullptr) {
        return false;
    }
    auto* state = static_cast<BlockingRegistrarState*>(context);
    state->entered.count_down();
    state->release.wait();
    return true;
}

void observe_dispatch(void* context,
                      const glimmer::control::TaskDispatchObservation& observation) noexcept {
    if (context == nullptr) {
        return;
    }
    auto* state = static_cast<DispatchTraceState*>(context);
    if (state->calls < state->task_ids.size()) {
        state->task_ids[state->calls] = observation.task.task_id;
        state->priorities[state->calls] = observation.task.priority;
        state->sequences[state->calls] = observation.sequence;
    }
    ++state->calls;
}

void observe_completion(void* context,
                        const glimmer::control::TaskCompletionObservation& observation) noexcept {
    if (context == nullptr) {
        return;
    }
    auto* state = static_cast<CompletionTraceState*>(context);
    if (state->calls < state->task_ids.size()) {
        state->task_ids[state->calls] = observation.task.task_id;
        state->states[state->calls] = observation.terminal_state;
        state->queue_wait_microseconds[state->calls] = observation.queue_wait_microseconds;
        state->service_time_microseconds[state->calls] = observation.service_time_microseconds;
    }
    ++state->calls;
}

void expect(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "Test failed: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

void test_admission_registers_and_cancels() {
    Scheduler scheduler(100);
    TaskAdmissionService service(scheduler);
    RegistrarState state;
    const auto admission = service.submit(
        TaskAdmissionRequest{.tenant_id = "tenant-a", .memory_bytes = 20, .weight = 2},
        register_resource, &state);
    expect(admission.accepted(), "valid request should be admitted");
    expect(state.calls == 1 && state.last_task_id == admission.task_id,
           "registrar should receive the accepted task id");
    expect(service.find(admission.task_id).has_value(), "admitted task should be observable");
    expect(service.cancel(admission.task_id), "queued task should be cancellable");
    expect(!service.cancel_queued(admission.task_id),
           "a terminal task should not be cancellable through the queued-only path");
    expect(scheduler.usage().used_bytes() == 0, "cancellation should release reservation");
}

void test_admission_propagates_priority() {
    Scheduler scheduler(100);
    TaskAdmissionService service(scheduler);
    RegistrarState state;
    const auto admission = service.submit(
        TaskAdmissionRequest{.tenant_id = "tenant-a", .memory_bytes = 20, .priority = 7},
        register_resource, &state);
    expect(admission.accepted(), "prioritized request should be admitted");
    const auto lease = service.claim_next();
    expect(lease.has_value() && lease->priority == 7,
           "admission should propagate priority to the scheduler snapshot");
    expect(service.complete(admission.task_id), "prioritized task should complete");
}

void test_dispatch_observer_records_policy_order() {
    Scheduler scheduler(100, glimmer::core::SchedulerOptions{
                                 .max_running_tasks = 1,
                                 .scheduling_policy = glimmer::core::SchedulingPolicy::kPriority});
    DispatchTraceState trace;
    TaskAdmissionService service(scheduler, std::chrono::milliseconds::zero(), false,
                                 observe_dispatch, &trace);
    RegistrarState registrar;
    const auto low = service.submit(
        TaskAdmissionRequest{.tenant_id = "tenant-a", .memory_bytes = 10, .priority = 10},
        register_resource, &registrar);
    const auto high = service.submit(
        TaskAdmissionRequest{.tenant_id = "tenant-b", .memory_bytes = 10, .priority = 100},
        register_resource, &registrar);
    expect(low.accepted() && high.accepted(), "observer tasks should be admitted");

    const auto first = service.claim_next();
    expect(first.has_value() && first->task_id == high.task_id,
           "priority policy should dispatch the high-priority task first");
    expect(service.complete(high.task_id), "high-priority task should complete");
    const auto second = service.claim_next();
    expect(second.has_value() && second->task_id == low.task_id,
           "priority policy should dispatch the remaining task");
    expect(service.complete(low.task_id), "low-priority task should complete");

    expect(trace.calls == 2, "observer should see each successful dispatch exactly once");
    expect(trace.task_ids[0] == high.task_id && trace.task_ids[1] == low.task_id,
           "observer order should match scheduler dispatch order");
    expect(trace.priorities[0] == 100 && trace.priorities[1] == 10,
           "observer should expose task priorities");
    expect(trace.sequences[0] == 1 && trace.sequences[1] == 2,
           "observer sequence should be monotonic");
    expect(scheduler.usage().used_bytes() == 0, "observed tasks should release quota");
}

void test_completion_observer_records_terminal_timings() {
    Scheduler scheduler(100, glimmer::core::SchedulerOptions{.max_running_tasks = 1});
    CompletionTraceState trace;
    TaskAdmissionService service(scheduler, std::chrono::milliseconds::zero(), false, nullptr,
                                 nullptr, observe_completion, &trace);
    RegistrarState registrar;
    const auto first = service.submit(
        TaskAdmissionRequest{.tenant_id = "tenant-a", .memory_bytes = 10, .priority = 1},
        register_resource, &registrar);
    const auto second = service.submit(
        TaskAdmissionRequest{.tenant_id = "tenant-b", .memory_bytes = 10, .priority = 10},
        register_resource, &registrar);
    expect(first.accepted() && second.accepted(), "completion observer tasks should be admitted");
    expect(service.claim(first.task_id).has_value(), "first completion observer task should run");
    expect(service.complete(first.task_id), "first completion observer task should complete");
    expect(service.claim(second.task_id).has_value(), "second completion observer task should run");
    expect(service.complete(second.task_id), "second completion observer task should complete");

    expect(trace.calls == 2, "completion observer should see each terminal transition once");
    expect(trace.task_ids[0] == first.task_id && trace.task_ids[1] == second.task_id,
           "completion observer should preserve terminal transition order");
    expect(trace.states[0] == glimmer::core::TaskState::kCompleted &&
               trace.states[1] == glimmer::core::TaskState::kCompleted,
           "completion observer should expose the terminal state");
    expect(scheduler.usage().used_bytes() == 0,
           "completion observation must not retain scheduler quota");
}

void test_external_progress_observations_pair_dispatch_and_completion() {
    Scheduler scheduler(100);
    CompletionTraceState trace;
    TaskAdmissionService service(scheduler, std::chrono::milliseconds::zero(), false, nullptr,
                                 nullptr, observe_completion, &trace);
    RegistrarState registrar;
    const auto admission = service.submit(
        TaskAdmissionRequest{.tenant_id = "tenant-a", .memory_bytes = 10, .priority = 9},
        register_resource, &registrar);
    expect(admission.accepted(), "external observation task should be admitted");
    const auto lease = scheduler.dispatch_next();
    expect(lease.has_value(), "external scheduler should dispatch the observation task");
    service.observe_external_dispatch(admission.task_id);
    service.observe_external_dispatch(admission.task_id);
    expect(scheduler.complete(admission.task_id),
           "external scheduler should complete the observation task");
    service.observe_external_completion(admission.task_id, glimmer::core::TaskState::kCompleted);
    expect(trace.calls == 1 && trace.task_ids[0] == admission.task_id,
           "external progress should emit one completion observation");
    expect(scheduler.usage().used_bytes() == 0,
           "external completion observation should preserve quota release");
}

void test_registration_failure_rolls_back() {
    Scheduler scheduler(100);
    TaskAdmissionService service(scheduler);
    RegistrarState state{.should_register = false};
    const auto admission =
        service.submit(TaskAdmissionRequest{.tenant_id = "tenant-a", .memory_bytes = 20},
                       register_resource, &state);
    expect(admission.status == SubmitStatus::kInternalError,
           "registration failure should be reported");
    expect(admission.task_id != 0, "rollback result should preserve task identity");
    expect(state.calls == 1, "failed registration should be attempted once");
    const auto snapshot = service.find(admission.task_id);
    expect(snapshot.has_value(), "rolled-back task should remain inspectable");
    if (snapshot.has_value()) {
        expect(snapshot->state == glimmer::core::TaskState::kCancelled,
               "rolled-back task should be cancelled");
    }
    expect(scheduler.usage().used_bytes() == 0, "registration failure should release reservation");
}

void test_registration_serializes_dispatch() {
    Scheduler scheduler(100);
    TaskAdmissionService service(scheduler, std::chrono::milliseconds{100});
    BlockingRegistrarState registrar_state;
    std::optional<glimmer::core::SubmitResult> admission;
    std::thread submitter([&] {
        admission =
            service.submit(TaskAdmissionRequest{.tenant_id = "tenant-a", .memory_bytes = 20},
                           register_blocking_resource, &registrar_state);
    });
    registrar_state.entered.wait();

    std::latch claim_started{1};
    std::atomic_bool claim_finished = false;
    std::optional<glimmer::core::TaskSnapshot> lease;
    std::thread claimer([&] {
        claim_started.count_down();
        lease = service.claim_next();
        claim_finished.store(true, std::memory_order_release);
    });
    claim_started.wait();
    std::this_thread::sleep_for(std::chrono::milliseconds{5});
    expect(!claim_finished.load(std::memory_order_acquire),
           "dispatch must wait until resource registration finishes");

    registrar_state.release.count_down();
    submitter.join();
    claimer.join();
    if (!admission.has_value() || !admission->accepted()) {
        expect(false, "blocking registration should eventually admit the task");
        return;
    }
    const TaskId task_id = admission->task_id;
    if (!lease.has_value()) {
        expect(false, "dispatch should follow completed resource registration");
        return;
    }
    expect(lease->task_id == task_id, "dispatch should follow completed resource registration");
    expect(service.complete(task_id), "serialized lease should complete");
    expect(scheduler.usage().used_bytes() == 0,
           "serialized registration test should release its reservation");
}

void test_wait_claim_avoids_client_polling() {
    Scheduler scheduler(100, glimmer::core::SchedulerOptions{.max_running_tasks = 1});
    TaskAdmissionService service(scheduler);
    RegistrarState state;
    const auto first =
        service.submit(TaskAdmissionRequest{.tenant_id = "tenant-a", .memory_bytes = 20},
                       register_resource, &state);
    const auto second =
        service.submit(TaskAdmissionRequest{.tenant_id = "tenant-b", .memory_bytes = 20},
                       register_resource, &state);
    expect(first.accepted() && second.accepted(), "wait test tasks should be admitted");
    expect(service.claim(first.task_id).has_value(), "first task should hold the only slot");
    expect(!service.wait_claim(second.task_id, std::chrono::milliseconds{1}).has_value(),
           "wait claim should time out while the slot is occupied");

    std::latch waiter_started{1};
    std::atomic_bool waiter_finished = false;
    std::optional<glimmer::core::TaskSnapshot> second_lease;
    std::thread waiter([&] {
        waiter_started.count_down();
        second_lease = service.wait_claim(second.task_id, std::chrono::milliseconds{1000});
        waiter_finished.store(true, std::memory_order_release);
    });
    waiter_started.wait();
    std::this_thread::sleep_for(std::chrono::milliseconds{5});
    expect(!waiter_finished.load(std::memory_order_acquire),
           "wait claim should remain blocked while the slot is occupied");
    expect(service.complete(first.task_id), "first task should complete");
    waiter.join();
    expect(second_lease.has_value() && second_lease->task_id == second.task_id,
           "wait claim should dispatch the task after the slot is released");
    expect(service.complete(second.task_id), "waited task should complete");
    expect(scheduler.usage().used_bytes() == 0, "waited task should release its reservation");
}

void test_wait_claim_observes_external_scheduler_progress() {
    Scheduler scheduler(100, glimmer::core::SchedulerOptions{.max_running_tasks = 1});
    TaskAdmissionService service(scheduler);
    RegistrarState state;
    const auto first =
        service.submit(TaskAdmissionRequest{.tenant_id = "tenant-a", .memory_bytes = 20},
                       register_resource, &state);
    const auto second =
        service.submit(TaskAdmissionRequest{.tenant_id = "tenant-b", .memory_bytes = 20},
                       register_resource, &state);
    expect(first.accepted() && second.accepted(), "external progress tasks should be admitted");
    expect(service.claim(first.task_id).has_value(), "external progress task should hold the slot");

    std::latch waiter_started{1};
    std::optional<glimmer::core::TaskSnapshot> second_lease;
    std::thread waiter([&] {
        waiter_started.count_down();
        second_lease = service.wait_claim(second.task_id, std::chrono::milliseconds{1000});
    });
    waiter_started.wait();
    std::this_thread::sleep_for(std::chrono::milliseconds{5});
    expect(scheduler.complete(first.task_id), "external scheduler should complete the first task");
    service.notify_scheduler_change();
    waiter.join();
    expect(second_lease.has_value() && second_lease->task_id == second.task_id,
           "wait claim should observe externally completed scheduler work");
    expect(service.complete(second.task_id), "externally awakened task should complete");
    expect(scheduler.usage().used_bytes() == 0,
           "external scheduler progress should preserve quota release");
}

void test_admission_rejects_invalid_or_unavailable_requests() {
    Scheduler scheduler(10);
    TaskAdmissionService service(scheduler);
    RegistrarState state;
    const TaskAdmissionRequest invalid_request{.tenant_id = "tenant-a", .memory_bytes = 0};
    expect(service.submit(invalid_request, register_resource, &state).status ==
               SubmitStatus::kInvalidTask,
           "invalid request should be rejected by scheduler validation");
    expect(service.submit(TaskAdmissionRequest{.tenant_id = "tenant-a", .memory_bytes = 1},
                          static_cast<TaskResourceRegistrar>(nullptr), nullptr)
                   .status == SubmitStatus::kInternalError,
           "missing registrar should be rejected");
    expect(service.submit(TaskAdmissionRequest{.tenant_id = "tenant-a", .memory_bytes = 11},
                          register_resource, &state)
                   .status == SubmitStatus::kQuotaExceeded,
           "request beyond quota should be rejected");
    expect(state.calls == 0, "rejected requests must not invoke registrar");
}

void test_pending_lease_expiry_releases_queue() {
    Scheduler scheduler(100);
    TaskAdmissionService service(scheduler, std::chrono::milliseconds{1}, true);
    RegistrarState state;
    const auto admission = service.submit(
        TaskAdmissionRequest{.tenant_id = "tenant-a", .memory_bytes = 20}, register_resource,
        &state, glimmer::control::TaskPeerIdentity{.pid = 1, .uid = 1, .start_time_ticks = 1});
    expect(admission.accepted(), "process-bound pending lease should be admitted");
    bool reaped = false;
    for (int attempt = 0; attempt < 20 && !reaped; ++attempt) {
        reaped = service.reap_expired();
        if (!reaped) {
            std::this_thread::sleep_for(std::chrono::milliseconds{1});
        }
    }
    expect(reaped, "expired queued lease should be reclaimed");
    const auto snapshot = service.find(admission.task_id);
    expect(snapshot.has_value() && snapshot->state == glimmer::core::TaskState::kCancelled,
           "expired queued lease should become cancelled");
    expect(scheduler.usage().used_bytes() == 0,
           "expired queued lease should release its reservation");
}

}  // namespace

int main() {
    test_admission_registers_and_cancels();
    test_admission_propagates_priority();
    test_dispatch_observer_records_policy_order();
    test_completion_observer_records_terminal_timings();
    test_external_progress_observations_pair_dispatch_and_completion();
    test_registration_failure_rolls_back();
    test_registration_serializes_dispatch();
    test_admission_rejects_invalid_or_unavailable_requests();
    test_wait_claim_avoids_client_polling();
    test_wait_claim_observes_external_scheduler_progress();
    test_pending_lease_expiry_releases_queue();
    return EXIT_SUCCESS;
}
