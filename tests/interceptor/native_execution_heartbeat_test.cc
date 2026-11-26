#include "internal/native_execution_heartbeat.h"

#include <atomic>
#include <barrier>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <future>
#include <iostream>
#include <latch>
#include <mutex>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <vector>

namespace {
using glimmer::control::TaskExecutionIdentity;
using glimmer::interceptor::NativeExecutionHeartbeat;
using Session = NativeExecutionHeartbeat::Session;

void require(bool passed, std::string_view message) {
    if (!passed) {
        std::cerr << "Test failed: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

struct Observations {
    std::mutex mutex;
    std::condition_variable condition;
    std::vector<TaskExecutionIdentity> executions;
    std::vector<std::thread::id> threads;

    bool record(TaskExecutionIdentity execution) {
        {
            const std::lock_guard lock(mutex);
            executions.push_back(execution);
            threads.push_back(std::this_thread::get_id());
        }
        condition.notify_all();
        return true;
    }

    void await(std::size_t count) {
        std::unique_lock lock(mutex);
        require(condition.wait_for(lock, std::chrono::seconds{5},
                                   [&] { return executions.size() >= count; }),
                "heartbeat callback should run within the harness deadline");
    }

    std::size_t size() {
        const std::lock_guard lock(mutex);
        return executions.size();
    }
};

void test_validation_and_short_sessions() {
    for (const auto interval : {std::chrono::milliseconds{0}, std::chrono::milliseconds{-1},
                                std::chrono::milliseconds::max()}) {
        bool rejected = false;
        try {
            NativeExecutionHeartbeat owner([](TaskExecutionIdentity) { return true; }, interval);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected, "nonpositive heartbeat interval must be rejected before thread creation");
    }
    bool rejected = false;
    try {
        NativeExecutionHeartbeat owner({});
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "empty heartbeat callback must be rejected");
    std::atomic<unsigned int> calls{};
    NativeExecutionHeartbeat owner(
        [&](TaskExecutionIdentity) {
            ++calls;
            return true;
        },
        std::chrono::hours{1});
    Session zero_task(owner, {0, 1});
    Session zero_generation(owner, {1, 0});
    require(!zero_task.valid() && !zero_generation.valid() && !zero_task.stop(),
            "both identity fields must be valid and rejected sessions cannot stop another owner");
    for (unsigned int index = 1; index <= 1000; ++index) {
        Session session(owner, {index, index});
        require(session.valid() && session.stop() && session.stop(),
                "short sessions should reuse the owner with idempotent stop");
    }
    require(calls.load() == 0, "short and idle sessions must not fabricate heartbeat requests");
}

void test_generation_and_worker_reuse() {
    Observations observations;
    NativeExecutionHeartbeat owner(
        [&](TaskExecutionIdentity value) { return observations.record(value); },
        std::chrono::milliseconds{1});
    Session first(owner, {17, 3});
    require(first.valid(), "first session should start");
    Session competing(owner, {18, 4});
    require(!competing.valid() && !competing.stop(), "live session must exclude a competitor");
    observations.await(1);
    require(first.stop(), "first session should settle");
    const auto split = observations.size();
    Session second(owner, {17, 4});
    require(second.valid() && first.stop(),
            "cached old stop must not stop the next generation of the same task");
    observations.await(split + 1);
    require(second.stop(), "second generation should settle");
    const std::lock_guard lock(observations.mutex);
    for (std::size_t index = 0; index < observations.executions.size(); ++index) {
        const auto value = observations.executions[index];
        require(value.task_id == 17 && value.generation == (index < split ? 3 : 4),
                "no old-generation callback may arrive after stop returns");
        require(observations.threads[index] == observations.threads.front() &&
                    observations.threads[index] != std::this_thread::get_id(),
                "sessions must use the same background worker, never the caller");
    }
}

void test_in_flight_stop(bool should_fail, bool should_throw) {
    std::latch entered(1);
    std::latch release(1);
    std::once_flag first_callback;
    std::atomic<bool> has_exited{};
    NativeExecutionHeartbeat owner(
        [&](TaskExecutionIdentity identity) {
            if (identity.task_id == 24 && identity.generation == 6) {
                return true;
            }
            require(identity.task_id == 23 && identity.generation == 5,
                    "blocked callback must retain its exact execution identity");
            std::call_once(first_callback, [&] { entered.count_down(); });
            release.wait();
            has_exited = true;
            if (should_throw) {
                throw std::runtime_error("injected heartbeat failure");
            }
            return !should_fail;
        },
        std::chrono::milliseconds{1});
    Session session(owner, {23, 5});
    require(session.valid(), "blocked session should start");
    entered.wait();
    auto stopped = std::async(std::launch::async, [&] { return session.stop(); });
    require(stopped.wait_for(std::chrono::milliseconds{10}) == std::future_status::timeout,
            "stop must not return while an accepted callback is withheld");
    Session competing(owner, {24, 6});
    require(!competing.valid(), "in-flight stop must exclude another generation");
    release.count_down();
    require(stopped.wait_for(std::chrono::seconds{5}) == std::future_status::ready,
            "released callback should allow stop to finish within the harness deadline");
    const bool expected = !should_fail && !should_throw;
    require(stopped.get() == expected && has_exited.load() && session.stop() == expected,
            "stop must consume the actual result and cache failure without throwing");
    Session next(owner, {24, 6});
    require(next.valid() == expected,
            "failure or exception must quarantine the owner rather than accept another session");
    if (next.valid()) {
        require(next.stop(), "healthy owner should remain reusable");
    }
}

void test_scope_exception_and_competing_starts() {
    NativeExecutionHeartbeat owner([](TaskExecutionIdentity) { return true; },
                                   std::chrono::hours{1});
    try {
        Session session(owner, {31, 1});
        require(session.valid(), "exception scope should start");
        throw std::runtime_error("original operation exception");
    } catch (const std::runtime_error&) {
    }
    Session restored(owner, {31, 2});
    require(restored.valid() && restored.stop(), "scope unwinding must retire the CPU session");
    std::barrier boundary(9);
    std::atomic<unsigned int> winners{};
    std::vector<std::thread> callers;
    for (unsigned int index = 0; index < 8; ++index) {
        callers.emplace_back([&, index] {
            boundary.arrive_and_wait();
            Session session(owner, {32 + index, 3});
            if (session.valid()) {
                ++winners;
            }
            boundary.arrive_and_wait();
        });
    }
    boundary.arrive_and_wait();
    boundary.arrive_and_wait();
    for (auto& caller : callers) {
        caller.join();
    }
    require(winners.load() == 1, "competing starts must publish only one heartbeat session");
    Session final(owner, {42, 4});
    require(final.valid() && final.stop(), "the winning session must retire before later reuse");
}
}  // namespace

int main() {
    test_validation_and_short_sessions();
    test_generation_and_worker_reuse();
    test_in_flight_stop(false, false);
    test_in_flight_stop(true, false);
    test_in_flight_stop(false, true);
    test_scope_exception_and_competing_starts();
    return EXIT_SUCCESS;
}
