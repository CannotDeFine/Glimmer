#include "glimmer/control/task_admission.h"

#include <array>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <future>
#include <iostream>
#include <latch>
#include <thread>

namespace {

bool register_resource(void*, glimmer::core::TaskId) noexcept {
    return true;
}

void observe_completion(void* context,
                        const glimmer::control::TaskCompletionObservation&) noexcept {
    auto* count = static_cast<std::atomic_size_t*>(context);
    count->fetch_add(1, std::memory_order_relaxed);
}

void observe_dispatch(void* context, const glimmer::control::TaskDispatchObservation&) noexcept {
    auto* count = static_cast<std::atomic_size_t*>(context);
    count->fetch_add(1, std::memory_order_relaxed);
}

bool test_competing_claims_and_terminal_reports() {
    glimmer::core::Scheduler scheduler(128);
    std::atomic_size_t dispatches{0};
    std::atomic_size_t completions{0};
    glimmer::control::TaskAdmissionService service(scheduler, std::chrono::seconds{10}, true,
                                                   observe_dispatch, &dispatches,
                                                   observe_completion, &completions);
    constexpr glimmer::control::TaskPeerIdentity owner{.pid = 1, .uid = 1, .start_time_ticks = 1};
    constexpr glimmer::control::TaskPeerIdentity other{.pid = 2, .uid = 1, .start_time_ticks = 1};
    const auto task = service.submit({.tenant_id = "tenant", .memory_bytes = 1}, register_resource,
                                     nullptr, owner);
    if (!task.accepted() || service.claim(task.task_id, other).has_value()) {
        return false;
    }
    constexpr std::size_t claimant_count = 4;
    std::barrier claim_start(static_cast<std::ptrdiff_t>(claimant_count + 1));
    std::array<std::jthread, claimant_count> claimants;
    std::atomic_size_t claimed{0};
    for (auto& claimant : claimants) {
        claimant = std::jthread([&] {
            claim_start.arrive_and_wait();
            const auto lease = service.claim(task.task_id, owner);
            if (lease.has_value() && lease->task_id == task.task_id &&
                lease->state == glimmer::core::TaskState::kRunning) {
                claimed.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    claim_start.arrive_and_wait();
    for (auto& claimant : claimants) {
        claimant.join();
    }
    if (claimed.load() != claimant_count || dispatches.load() != 1 ||
        service.complete(task.task_id, other) || service.heartbeat(task.task_id, other)) {
        return false;
    }
    std::barrier terminal_start(3);
    std::atomic_size_t terminal_reports{0};
    std::jthread completion([&] {
        terminal_start.arrive_and_wait();
        if (service.complete(task.task_id, owner)) {
            terminal_reports.fetch_add(1, std::memory_order_relaxed);
        }
    });
    std::jthread failure([&] {
        terminal_start.arrive_and_wait();
        if (service.fail(task.task_id, owner)) {
            terminal_reports.fetch_add(1, std::memory_order_relaxed);
        }
    });
    terminal_start.arrive_and_wait();
    completion.join();
    failure.join();
    const auto stats = service.stats();
    return terminal_reports.load() == 1 && completions.load() == 1 &&
           stats.completed_task_count + stats.failed_task_count == 1 &&
           stats.running_task_count == 0 && stats.quota.used_bytes() == 0;
}

bool test_concurrent_submit_claim_complete_and_reap() {
    constexpr std::size_t worker_count = 4;
    constexpr std::size_t iterations = 1000;
    glimmer::core::Scheduler scheduler(
        128, glimmer::core::SchedulerOptions{.max_running_tasks = worker_count});
    std::atomic_size_t observations{0};
    glimmer::control::TaskAdmissionService service(scheduler, std::chrono::hours{1}, true, nullptr,
                                                   nullptr, observe_completion, &observations);
    std::barrier start(static_cast<std::ptrdiff_t>(worker_count + 1));
    std::atomic_size_t active_workers{worker_count};
    std::atomic_bool failed{false};
    std::array<std::jthread, worker_count> workers;
    for (std::size_t index = 0; index < worker_count; ++index) {
        workers[index] = std::jthread([&, index] {
            const glimmer::control::TaskPeerIdentity peer{
                .pid = static_cast<std::uint32_t>(index + 1), .uid = 1, .start_time_ticks = 1};
            start.arrive_and_wait();
            for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
                const auto task = service.submit({.tenant_id = "tenant", .memory_bytes = 1},
                                                 register_resource, nullptr, peer);
                if (!task.accepted()) {
                    failed.store(true, std::memory_order_relaxed);
                    break;
                }
                const auto lease = service.wait_claim(task.task_id, std::chrono::seconds{5}, peer);
                if (!lease.has_value() || !service.heartbeat(task.task_id, peer) ||
                    !service.complete(task.task_id, peer)) {
                    failed.store(true, std::memory_order_relaxed);
                    break;
                }
            }
            active_workers.fetch_sub(1, std::memory_order_release);
        });
    }
    start.arrive_and_wait();
    while (active_workers.load(std::memory_order_acquire) != 0) {
        if (service.reap_expired()) {
            failed.store(true, std::memory_order_relaxed);
        }
        std::this_thread::yield();
    }
    for (auto& worker : workers) {
        worker.join();
    }
    const auto stats = service.stats();
    return !failed.load(std::memory_order_relaxed) &&
           observations.load(std::memory_order_relaxed) == worker_count * iterations &&
           stats.completed_task_count == worker_count * iterations &&
           stats.running_task_count == 0 && stats.queued_task_count == 0 &&
           stats.quota.used_bytes() == 0;
}

struct CapacityUpdate {
    glimmer::core::Scheduler& scheduler;
    glimmer::core::TaskId observed_task = 0;
    std::latch entered{1};
    std::latch release{1};
    bool applied = false;
};

void release_reserved_capacity(
    void* context, const glimmer::control::TaskCompletionObservation& observation) noexcept {
    auto* update = static_cast<CapacityUpdate*>(context);
    if (observation.task.task_id != update->observed_task) {
        return;
    }
    update->entered.count_down();
    update->release.wait();
    update->applied = update->scheduler.set_priority_reserved_slots(0);
}

bool test_completion_feedback_wakes_waiter_without_holding_admission_lock() {
    glimmer::core::Scheduler scheduler(
        128, glimmer::core::SchedulerOptions{
                 .max_running_tasks = 2,
                 .priority_reserved_slots = 1,
                 .priority_reservation_threshold = 100,
                 .scheduling_policy = glimmer::core::SchedulingPolicy::kPriority});
    CapacityUpdate update{.scheduler = scheduler};
    glimmer::control::TaskAdmissionService service(scheduler, std::chrono::seconds{10}, false,
                                                   nullptr, nullptr, release_reserved_capacity,
                                                   &update);
    const auto running =
        service.submit({.tenant_id = "training", .memory_bytes = 1}, register_resource, nullptr);
    if (!running.accepted() || !service.claim(running.task_id).has_value()) {
        return false;
    }
    const auto high = service.submit({.tenant_id = "inference", .memory_bytes = 1, .priority = 100},
                                     register_resource, nullptr);
    if (!high.accepted() || !service.claim(high.task_id).has_value()) {
        return false;
    }
    const auto waiting =
        service.submit({.tenant_id = "training", .memory_bytes = 1}, register_resource, nullptr);
    if (!waiting.accepted()) {
        return false;
    }
    update.observed_task = high.task_id;
    auto completion =
        std::async(std::launch::async, [&] { return service.complete(high.task_id); });
    update.entered.wait();
    // An observer must not hold admission's lock: another lease can renew
    // while the feedback callback is paused.
    auto heartbeat =
        std::async(std::launch::async, [&] { return service.heartbeat(running.task_id); });
    const bool independent_progress =
        heartbeat.wait_for(std::chrono::seconds{1}) == std::future_status::ready;
    auto waiter = std::async(std::launch::async, [&] {
        return service.wait_claim(waiting.task_id, std::chrono::seconds{2});
    });
    // The remaining slot is still reserved. A waiter started inside the
    // callback window must be notified after the reservation is removed.
    const bool initially_blocked =
        waiter.wait_for(std::chrono::milliseconds{50}) == std::future_status::timeout;
    update.release.count_down();
    const bool completed = completion.get();
    const bool renewed = heartbeat.get();
    const auto claimed = waiter.get();
    const bool released_waiting = service.complete(waiting.task_id);
    const bool released_running = service.complete(running.task_id);
    return independent_progress && initially_blocked && completed && renewed && update.applied &&
           claimed.has_value() && released_waiting && released_running &&
           scheduler.usage().used_bytes() == 0;
}

}  // namespace

int main() {
    if (!test_competing_claims_and_terminal_reports()) {
        std::cerr << "Competing claims or terminal reports violated lease ownership\n";
        return EXIT_FAILURE;
    }
    if (!test_concurrent_submit_claim_complete_and_reap()) {
        std::cerr << "Concurrent admission lost progress, observations, or quota\n";
        return EXIT_FAILURE;
    }
    if (!test_completion_feedback_wakes_waiter_without_holding_admission_lock()) {
        std::cerr << "Completion feedback blocked admission or lost a capacity-change wakeup\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
