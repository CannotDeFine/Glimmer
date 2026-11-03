#include "glimmer/control/task_execution_admission.h"
#include "internal/execution_admission_state.h"

#include <array>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <new>
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

bool count_registration(void* context, glimmer::core::TaskId) noexcept {
    ++*static_cast<unsigned*>(context);
    return true;
}

void fail_pending_insert() {
    throw std::bad_alloc();
}

void test_pending_allocation_failure() {
    glimmer::core::Scheduler scheduler(100);
    ExecutionAdmissionState state(scheduler, 0ms, true, nullptr, nullptr, nullptr, nullptr,
                                  std::nullopt, fail_pending_insert);
    const TaskAdmissionRequest request{.tenant_id = "tenant", .memory_bytes = 20};
    unsigned registrations = 0;
    const auto result =
        state.submit_cooperative(request, count_registration, &registrations, kOwner);
    expect(!result.accepted() && registrations == 0 && scheduler.stats().quota.used_bytes() == 0,
           "failed pending allocation rolls back before registrar publishes resources");
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
    const auto submitted = service.submit_cooperative(request, accept_resource, nullptr, kOwner);
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
    const auto first = service.submit_cooperative(low, accept_resource, nullptr, kOwner);
    const auto initial = service.claim_execution(first.task_id, kOwner);
    expect(initial.has_value(), "low priority execution starts");
    auto execution = TaskExecutionIdentity{
        first.task_id, initial.value_or(glimmer::core::TaskSnapshot{}).dispatch_sequence};
    const auto second = service.submit_cooperative(high, accept_resource, nullptr, kOwner);
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

void test_cooperative_admission_kind() {
    ObservationCounts observed;
    TaskExecutionAdmissionService service(100, {}, 0ms, true, on_dispatch, &observed, on_completion,
                                          &observed);
    TaskExecutionAdmissionService unbound(100);
    const TaskAdmissionRequest request{.tenant_id = "tenant", .memory_bytes = 20};
    expect(
        !unbound.submit_cooperative(request, accept_resource, nullptr, kOwner).accepted() &&
            !service.submit_cooperative(request, accept_resource, nullptr, std::nullopt).accepted(),
        "cooperative admission requires process binding and a peer");
    expect(!service.submit_cooperative(request, reject_resource, nullptr, kOwner).accepted() &&
               service.stats().quota.used_bytes() == 0,
           "registrar rejection rolls back the pending lease and quota");
    const auto legacy = service.submit(request, accept_resource, nullptr, kOwner);
    expect(legacy.accepted() && !service.claim_execution(legacy.task_id, kOwner) &&
               service.cancel_queued(legacy.task_id, kOwner),
           "legacy work cannot be silently upgraded to cooperative execution");
    const auto pending = service.submit_cooperative(request, accept_resource, nullptr, kOwner);
    expect(pending.accepted() && !service.claim(pending.task_id, kOwner) &&
               !service.wait_claim(pending.task_id, 1ms, kOwner) &&
               !service.cancel(pending.task_id) &&
               !service.cancel_queued(pending.task_id, kOwner) &&
               !service.cancel_pending_cooperative(pending.task_id, kForeign) &&
               !service.complete(pending.task_id, kOwner),
           "legacy and foreign paths cannot claim or terminate cooperative work");
    const auto claimed = service.claim_execution(pending.task_id, kOwner);
    expect(claimed.has_value() && !service.claim_execution(pending.task_id, kOwner),
           "a running generation is not returned as a second launch grant");
    const TaskExecutionIdentity execution{
        pending.task_id, claimed.value_or(glimmer::core::TaskSnapshot{}).dispatch_sequence};
    expect(service.complete_execution(execution, kOwner) && service.stats().quota.used_bytes() == 0,
           "cooperative completion retires the grant");
    expect(observed.dispatches.load(std::memory_order_relaxed) == 1 &&
               observed.completions.load(std::memory_order_relaxed) == 1,
           "cooperative transitions notify observers once");
}

void test_pending_claim_cancel_race() {
    TaskExecutionAdmissionService service(100, {}, 0ms, true);
    const TaskAdmissionRequest request{.tenant_id = "tenant", .memory_bytes = 20};
    for (int i = 0; i < 16; ++i) {
        const auto pending = service.submit_cooperative(request, accept_resource, nullptr, kOwner);
        expect(pending.accepted(), "race fixture admitted");
        std::optional<glimmer::core::TaskSnapshot> claimed;
        bool cancelled = false;
        std::barrier start(3);
        std::thread first([&] {
            start.arrive_and_wait();
            claimed = service.claim_execution(pending.task_id, kOwner);
        });
        std::thread second([&] {
            start.arrive_and_wait();
            cancelled = service.cancel_pending_cooperative(pending.task_id, kOwner);
        });
        start.arrive_and_wait();
        first.join();
        second.join();
        expect(claimed.has_value() != cancelled, "claim and pending cancellation are exclusive");
        if (claimed.has_value()) {
            const TaskExecutionIdentity execution{pending.task_id, claimed->dispatch_sequence};
            expect(service.complete_execution(execution, kOwner), "race winner is retired");
        }
        expect(service.stats().quota.used_bytes() == 0, "pending race releases quota once");
    }
}

void test_competing_cooperative_claims() {
    TaskExecutionAdmissionService service(100, {}, 0ms, true);
    const TaskAdmissionRequest request{.tenant_id = "tenant", .memory_bytes = 20};
    const auto pending = service.submit_cooperative(request, accept_resource, nullptr, kOwner);
    std::array<std::optional<glimmer::core::TaskSnapshot>, 2> claims;
    std::barrier start(3);
    std::thread first([&] {
        start.arrive_and_wait();
        claims[0] = service.claim_execution(pending.task_id, kOwner);
    });
    std::thread second([&] {
        start.arrive_and_wait();
        claims[1] = service.claim_execution(pending.task_id, kOwner);
    });
    start.arrive_and_wait();
    first.join();
    second.join();
    expect(claims[0].has_value() != claims[1].has_value(), "only one physical launch grant wins");
    const auto winner = claims[0].has_value() ? claims[0] : claims[1];
    const TaskExecutionIdentity execution{
        pending.task_id, winner.value_or(glimmer::core::TaskSnapshot{}).dispatch_sequence};
    expect(service.complete_execution(execution, kOwner) && service.stats().quota.used_bytes() == 0,
           "winning grant is retired");
}

void test_pending_claim_expiry_race() {
    ObservationCounts observed;
    TaskExecutionAdmissionService service(100, {}, 1ms, true, nullptr, nullptr, on_completion,
                                          &observed);
    const TaskAdmissionRequest request{.tenant_id = "tenant", .memory_bytes = 20};
    for (int i = 0; i < 8; ++i) {
        const auto pending = service.submit_cooperative(request, accept_resource, nullptr, kOwner);
        expect(pending.accepted(), "expiry race fixture admitted");
        std::this_thread::sleep_for(5ms);
        std::optional<glimmer::core::TaskSnapshot> claimed;
        bool expired = false;
        std::barrier start(3);
        std::thread first([&] {
            start.arrive_and_wait();
            claimed = service.claim_execution(pending.task_id, kOwner);
        });
        std::thread second([&] {
            start.arrive_and_wait();
            expired = service.reap_expired();
        });
        start.arrive_and_wait();
        first.join();
        second.join();
        expect(claimed.has_value() != expired, "claim and expiry are exclusive");
        if (claimed.has_value()) {
            const TaskExecutionIdentity execution{pending.task_id, claimed->dispatch_sequence};
            expect(service.complete_execution(execution, kOwner), "claimed race winner retires");
        }
        expect(service.stats().quota.used_bytes() == 0 &&
                   observed.completions.load(std::memory_order_relaxed) ==
                       static_cast<unsigned>(i + 1),
               "expiry or completion releases quota and notifies once");
    }
}

void test_competing_terminal_reports() {
    ObservationCounts observed;
    TaskExecutionAdmissionService service(100, {}, 0ms, true, on_dispatch, &observed, on_completion,
                                          &observed);
    const TaskAdmissionRequest request{.tenant_id = "tenant", .memory_bytes = 20};
    const auto submitted = service.submit_cooperative(request, accept_resource, nullptr, kOwner);
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
        const auto submitted =
            service.submit_cooperative(request, accept_resource, nullptr, kOwner);
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
    test_pending_allocation_failure();
    test_legacy_equivalence();
    test_cooperative_ownership();
    test_native_epoch();
    test_priority_and_continuation();
    test_pending_expiry_and_native_policy();
    test_native_acceptance_deadline();
    test_cooperative_admission_kind();
    test_pending_claim_cancel_race();
    test_competing_cooperative_claims();
    test_pending_claim_expiry_race();
    test_competing_terminal_reports();
    test_suspend_completion_race();
    return EXIT_SUCCESS;
}
