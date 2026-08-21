#include "glimmer/control/launch_gate.h"

#include <chrono>
#include <cstdlib>
#include <atomic>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <optional>
#include <string_view>
#include <thread>

namespace {

using glimmer::control::LaunchGate;
using glimmer::control::LaunchGateOptions;

void expect(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "Test failed: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

void test_capacity_blocks_and_releases() {
    LaunchGate gate(LaunchGateOptions{.max_concurrent_launches = 1, .control_socket = ""});
    const auto first = gate.acquire();
    const auto first_id = first.value_or(0);
    expect(first_id != 0, "the first launch should acquire the gate");

    std::optional<glimmer::core::TaskId> second;
    std::condition_variable waiter_condition;
    std::mutex waiter_mutex;
    bool waiter_started = false;
    std::atomic_bool waiter_finished = false;
    std::thread waiter(
        [&second, &waiter_condition, &waiter_finished, &waiter_mutex, &gate, &waiter_started] {
            {
                std::scoped_lock lock(waiter_mutex);
                waiter_started = true;
            }
            waiter_condition.notify_one();
            second = gate.acquire();
            waiter_finished.store(true, std::memory_order_release);
        });
    {
        std::unique_lock lock(waiter_mutex);
        expect(waiter_condition.wait_for(lock, std::chrono::milliseconds(100),
                                         [&waiter_started] { return waiter_started; }),
               "the waiting thread should start");
    }
    expect(!waiter_finished.load(std::memory_order_acquire),
           "a full gate should keep the second launch waiting");

    expect(gate.complete(first_id), "completion should release the first launch");
    waiter.join();
    const auto second_id = second.value_or(0);
    expect(second_id != 0, "the waiting launch should acquire the released slot");
    expect(gate.fail(second_id), "failure should release the second launch");
    expect(gate.stats().running_task_count == 0 && gate.stats().queued_task_count == 0,
           "all launch slots should be released after terminal transitions");
}

void test_timeout_cancels_queued_launch() {
    LaunchGate gate(LaunchGateOptions{.max_concurrent_launches = 1, .control_socket = ""});
    const auto first = gate.acquire();
    const auto first_id = first.value_or(0);
    expect(first_id != 0, "the first launch should acquire the gate");

    const auto timed_out = gate.acquire(std::chrono::milliseconds(1));
    expect(!timed_out.has_value(), "a timed-out launch should not be admitted");
    const auto stats = gate.stats();
    expect(stats.queued_task_count == 0 && stats.running_task_count == 1,
           "a timed-out launch should be removed without affecting the running slot");
    expect(gate.fail(first_id), "the running launch should fail cleanly");
}

}  // namespace

int main() {
    test_capacity_blocks_and_releases();
    test_timeout_cancels_queued_launch();
    return EXIT_SUCCESS;
}
