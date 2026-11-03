#include "glimmer/control/task_execution_admission.h"

#include <array>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <string_view>
#include <thread>

namespace {

using namespace glimmer::control;
using namespace std::chrono_literals;
constexpr TaskPeerIdentity kOwner{10, 1000, 20};
constexpr TaskPeerIdentity kForeign{10, 1000, 21};

void expect(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "Test failed: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

bool accept_resource(void*, glimmer::core::TaskId) noexcept {
    return true;
}

bool reject_resource(void*, glimmer::core::TaskId) noexcept {
    return false;
}

void test_legacy_equivalence() {
    glimmer::core::Scheduler scheduler(100);
    TaskAdmissionService old_service(scheduler);
    TaskExecutionAdmissionService new_service(100);
    const TaskAdmissionRequest request{.tenant_id = "tenant", .memory_bytes = 20};
    const auto old_reject = old_service.submit(request, reject_resource, nullptr);
    const auto new_reject = new_service.submit(request, reject_resource, nullptr);
    expect(old_reject.status == new_reject.status &&
               old_service.stats().quota.used_bytes() == new_service.stats().quota.used_bytes(),
           "registrar rollback matches legacy admission");
    const auto old_submit = old_service.submit(request, accept_resource, nullptr);
    const auto new_submit = new_service.submit(request, accept_resource, nullptr);
    expect(
        old_submit.accepted() && new_submit.accepted() && old_submit.task_id == new_submit.task_id,
        "legacy submissions match");
    const auto old_claim = old_service.claim(old_submit.task_id);
    const auto new_claim = new_service.claim(new_submit.task_id);
    expect(old_claim.has_value() && new_claim.has_value() && old_claim->state == new_claim->state &&
               old_claim->dispatch_sequence == new_claim->dispatch_sequence &&
               old_service.heartbeat(old_submit.task_id) &&
               new_service.heartbeat(new_submit.task_id),
           "legacy claims and heartbeats match");
    expect(old_service.complete(old_submit.task_id) && new_service.complete(new_submit.task_id) &&
               !old_service.complete(old_submit.task_id) &&
               !new_service.complete(new_submit.task_id),
           "legacy completion is exactly once");
    const auto old_stats = old_service.stats();
    const auto new_stats = new_service.stats();
    expect(old_stats.completed_task_count == new_stats.completed_task_count &&
               old_stats.quota.used_bytes() == new_stats.quota.used_bytes() &&
               new_stats.quota.used_bytes() == 0,
           "legacy completion releases quota equally");
}

void test_cooperative_ownership() {
    TaskExecutionAdmissionService service(100, {}, 5ms, true);
    const TaskAdmissionRequest request{.tenant_id = "tenant", .memory_bytes = 20};
    const auto submitted = service.submit(request, accept_resource, nullptr, kOwner);
    expect(submitted.accepted() && !service.claim_execution(submitted.task_id, kForeign),
           "only the submitter may claim");
    const auto first = service.claim_execution(submitted.task_id, kOwner);
    expect(first.has_value() && first->dispatch_sequence != 0, "first execution claimed");
    const TaskExecutionIdentity original{
        submitted.task_id, first.value_or(glimmer::core::TaskSnapshot{}).dispatch_sequence};
    expect(!service.cancel(original.task_id) && !service.complete(original.task_id, kOwner) &&
               !service.fail(original.task_id, kOwner) &&
               !service.suspend_execution(original, kForeign),
           "legacy or foreign reports cannot terminate cooperative work");
    std::this_thread::sleep_for(20ms);
    expect(!service.reap_expired() && service.stats().quota.used_bytes() == 20,
           "timeout cannot prove physical stop or release quota");
    expect(service.suspend_execution(original, kOwner), "owner reports physical suspension");
    expect(!service.cancel(original.task_id) && !service.cancel_queued(original.task_id, kOwner),
           "legacy cancellation cannot bypass a generation");
    const auto second = service.claim_execution(original.task_id, kOwner);
    expect(second.has_value() && second->dispatch_sequence != original.generation,
           "resume receives a fresh generation");
    const TaskExecutionIdentity resumed{
        original.task_id, second.value_or(glimmer::core::TaskSnapshot{}).dispatch_sequence};
    expect(!service.complete_execution(original, kOwner) &&
               !service.complete_execution(resumed, kForeign) &&
               service.complete_execution(resumed, kOwner) &&
               !service.complete_execution(resumed, kOwner) &&
               service.stats().quota.used_bytes() == 0,
           "stale or replayed terminal reports cannot release quota");
}

void test_native_epoch() {
    constexpr NativeEpochExtent extent{4, 2, 2, 1000000};
    TaskExecutionAdmissionService service(100, {}, 5ms, true, nullptr, nullptr, nullptr, nullptr,
                                          extent);
    const TaskAdmissionRequest request{.tenant_id = "tenant", .memory_bytes = 20, .work_units = 4};
    const auto submitted =
        service.submit_native_epoch(request, extent, accept_resource, nullptr, kOwner);
    expect(submitted.accepted() && !service.claim_native_epoch(submitted.task_id, kForeign),
           "native grant is owner-bound");
    const auto grant = service.claim_native_epoch(submitted.task_id, kOwner);
    expect(grant.has_value() && grant->extent == extent, "native extent granted");
    const auto execution = grant.value_or(NativeEpochGrant{}).execution;
    expect(!service.cancel(execution.task_id) &&
               !service.finish_native_epoch(execution, 5, false, kOwner) &&
               !service.finish_native_epoch(execution, 4, false, kForeign) &&
               service.stats().quota.used_bytes() == 20,
           "foreign and oversized reports retain native ownership");
    std::this_thread::sleep_for(20ms);
    expect(!service.reap_expired() && service.stats().quota.used_bytes() == 20,
           "native timeout does not imply completion");
    expect(service.finish_native_epoch(execution, 4, false, kOwner) &&
               !service.finish_native_epoch(execution, 4, false, kOwner) &&
               service.stats().quota.used_bytes() == 0,
           "native completion is exactly once");
}

void test_priority_and_continuation() {
    glimmer::core::SchedulerOptions options;
    options.max_queued_tasks = 1;
    options.scheduling_policy = glimmer::core::SchedulingPolicy::kPriority;
    TaskExecutionAdmissionService service(100, options, 0ms, true);
    const TaskAdmissionRequest low{.tenant_id = "low", .memory_bytes = 20, .priority = 1};
    const TaskAdmissionRequest high{.tenant_id = "high", .memory_bytes = 20, .priority = 9};
    const auto first = service.submit(low, accept_resource, nullptr, kOwner);
    const auto initial = service.claim_execution(first.task_id, kOwner);
    expect(initial.has_value(), "low priority execution starts");
    auto execution = TaskExecutionIdentity{
        first.task_id, initial.value_or(glimmer::core::TaskSnapshot{}).dispatch_sequence};
    const auto second = service.submit(high, accept_resource, nullptr, kOwner);
    expect(second.accepted() && service.retreat_requested(execution, kOwner).value_or(false) &&
               service.suspend_execution(execution, kOwner),
           "high priority requests retreat even when the waiting queue is full");
    expect(!service.claim_execution(first.task_id, kOwner),
           "low continuation cannot bypass higher priority work");
    const auto high_run = service.claim_execution(second.task_id, kOwner);
    expect(high_run.has_value(), "higher priority task claims the released slot");
    const TaskExecutionIdentity high_execution{
        second.task_id, high_run.value_or(glimmer::core::TaskSnapshot{}).dispatch_sequence};
    expect(service.complete_execution(high_execution, kOwner), "higher priority task completes");
    for (int i = 0; i < 16; ++i) {
        const auto resumed = service.claim_execution(first.task_id, kOwner);
        expect(resumed.has_value(), "continuation resumes");
        const TaskExecutionIdentity next{
            first.task_id, resumed.value_or(glimmer::core::TaskSnapshot{}).dispatch_sequence};
        expect(next.generation > execution.generation &&
                   !service.complete_execution(execution, kOwner) &&
                   service.suspend_execution(next, kOwner),
               "every continuation rejects its predecessor generation");
        execution = next;
    }
    expect(service.cancel_execution(execution, kOwner) && service.stats().quota.used_bytes() == 0,
           "physically stopped continuation can be cancelled exactly once");
}

void test_pending_expiry_and_native_policy() {
    constexpr NativeEpochExtent allowed{4, 2, 2, 1000000};
    TaskExecutionAdmissionService service(100, {}, 5ms, true, nullptr, nullptr, nullptr, nullptr,
                                          allowed);
    const TaskAdmissionRequest request{.tenant_id = "tenant", .memory_bytes = 20, .work_units = 4};
    const NativeEpochExtent excessive{8, 2, 2, 1000000};
    expect(!service.submit_native_epoch(request, excessive, accept_resource, nullptr, kOwner)
                .accepted(),
           "native extent cannot exceed server policy");
    const auto pending =
        service.submit_native_epoch(request, allowed, accept_resource, nullptr, kOwner);
    expect(pending.accepted() && !service.cancel_pending_native_epoch(pending.task_id, kForeign) &&
               service.pending_native_epoch_extent(pending.task_id, kOwner) == allowed,
           "pending native lease is owner-bound");
    std::this_thread::sleep_for(20ms);
    expect(service.reap_expired() && service.stats().quota.used_bytes() == 0 &&
               !service.claim_native_epoch(pending.task_id, kOwner),
           "never-issued expired native work is reclaimed");
    const auto next =
        service.submit_native_epoch(request, allowed, accept_resource, nullptr, kOwner);
    expect(next.accepted() && service.cancel_pending_native_epoch(next.task_id, kOwner) &&
               !service.cancel_pending_native_epoch(next.task_id, kOwner) &&
               service.stats().quota.used_bytes() == 0,
           "pending native cancellation is owner-bound and exactly once");
}

void test_native_acceptance_deadline() {
    constexpr NativeEpochExtent allowed{4, 2, 2, 10000};
    TaskExecutionAdmissionService service(100, {}, 0ms, true, nullptr, nullptr, nullptr, nullptr,
                                          allowed);
    const TaskAdmissionRequest request{.tenant_id = "tenant", .memory_bytes = 20, .work_units = 4};
    const auto submitted =
        service.submit_native_epoch(request, allowed, accept_resource, nullptr, kOwner);
    const auto grant = service.claim_native_epoch(submitted.task_id, kOwner);
    expect(grant.has_value(), "bounded native epoch granted");
    const auto execution = grant.value_or(NativeEpochGrant{}).execution;
    expect(service.heartbeat_native_epoch(execution, kOwner) &&
               !service.heartbeat_native_epoch(execution, kForeign),
           "native heartbeats validate the owner");
    std::this_thread::sleep_for(20ms);
    expect(service.heartbeat_native_epoch(execution, kOwner) &&
               service.native_epoch_retreat_requested(execution, kOwner).value_or(false) &&
               service.stats().quota.used_bytes() == 20,
           "heartbeat cannot renew the native acceptance deadline");
    expect(service.finish_native_epoch(execution, 0, true, kOwner) &&
               service.stats().quota.used_bytes() == 0,
           "physical failure releases native quota once");
}

struct ObservationCounts {
    std::atomic<unsigned> dispatches{0};
    std::atomic<unsigned> completions{0};
};

void on_dispatch(void* context, const TaskDispatchObservation&) noexcept {
    static_cast<ObservationCounts*>(context)->dispatches.fetch_add(1, std::memory_order_relaxed);
}

void on_completion(void* context, const TaskCompletionObservation&) noexcept {
    static_cast<ObservationCounts*>(context)->completions.fetch_add(1, std::memory_order_relaxed);
}

void test_competing_terminal_reports() {
    ObservationCounts observed;
    TaskExecutionAdmissionService service(100, {}, 0ms, true, on_dispatch, &observed, on_completion,
                                          &observed);
    const TaskAdmissionRequest request{.tenant_id = "tenant", .memory_bytes = 20};
    const auto submitted = service.submit(request, accept_resource, nullptr, kOwner);
    const auto running = service.claim_execution(submitted.task_id, kOwner);
    expect(running.has_value(), "concurrent fixture has an execution");
    const TaskExecutionIdentity execution{
        submitted.task_id, running.value_or(glimmer::core::TaskSnapshot{}).dispatch_sequence};
    std::atomic<unsigned> successes{0};
    std::array<std::thread, 8> threads;
    std::barrier start(static_cast<std::ptrdiff_t>(threads.size() + 1));
    for (auto& thread : threads) {
        thread = std::thread([&] {
            start.arrive_and_wait();
            if (service.complete_execution(execution, kOwner)) {
                successes.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    start.arrive_and_wait();
    for (auto& thread : threads) {
        thread.join();
    }
    expect(successes.load(std::memory_order_relaxed) == 1 &&
               service.stats().quota.used_bytes() == 0 &&
               observed.dispatches.load(std::memory_order_relaxed) == 1 &&
               observed.completions.load(std::memory_order_relaxed) == 1,
           "competing terminal reports release quota exactly once");
}

void test_suspend_completion_race() {
    TaskExecutionAdmissionService service(100, {}, 0ms, true);
    const TaskAdmissionRequest request{.tenant_id = "tenant", .memory_bytes = 20};
    for (int i = 0; i < 16; ++i) {
        const auto submitted = service.submit(request, accept_resource, nullptr, kOwner);
        const auto running = service.claim_execution(submitted.task_id, kOwner);
        expect(running.has_value(), "race fixture is running");
        const TaskExecutionIdentity execution{
            submitted.task_id, running.value_or(glimmer::core::TaskSnapshot{}).dispatch_sequence};
        bool suspended = false;
        bool completed = false;
        std::barrier start(3);
        std::thread first([&] {
            start.arrive_and_wait();
            suspended = service.suspend_execution(execution, kOwner);
        });
        std::thread second([&] {
            start.arrive_and_wait();
            completed = service.complete_execution(execution, kOwner);
        });
        start.arrive_and_wait();
        first.join();
        second.join();
        expect(suspended != completed, "only one racing transition succeeds");
        if (suspended) {
            expect(service.cancel_execution(execution, kOwner), "suspended race is retired");
        }
        expect(service.stats().quota.used_bytes() == 0, "race releases quota exactly once");
    }
}

}  // namespace

int main() {
    test_legacy_equivalence();
    test_cooperative_ownership();
    test_native_epoch();
    test_priority_and_continuation();
    test_pending_expiry_and_native_policy();
    test_native_acceptance_deadline();
    test_competing_terminal_reports();
    test_suspend_completion_race();
    return EXIT_SUCCESS;
}
