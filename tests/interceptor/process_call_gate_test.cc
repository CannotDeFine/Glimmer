#include "internal/process_call_gate.h"

#include <array>
#include <atomic>
#include <barrier>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <latch>
#include <stdexcept>
#include <thread>

namespace {
using glimmer::interceptor::ProcessCallGate;

void require(bool value, const char* message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}

template <typename Operation>
void require_logic_error(Operation&& operation) {
    bool was_rejected = false;
    try {
        operation();
    } catch (const std::logic_error&) {
        was_rejected = true;
    }
    require(was_rejected, "invalid process gate nesting must reject");
}

void nesting_phase_and_misuse() {
    ProcessCallGate gate;
    ProcessCallGate foreign;
    require(gate.snapshot().active_calls == 0 && gate.snapshot().admitted_callers == 0 &&
                !gate.snapshot().owns_transition && !gate.snapshot().is_native_only,
            "fresh CPU-only observation does not admit a caller or change phase");
    require_logic_error([&] { const ProcessCallGate::Transition transition(gate); });
    {
        const ProcessCallGate::Call outer(gate);
        const ProcessCallGate::Call nested(gate);
        require(gate.snapshot().active_calls == 1 && gate.snapshot().admitted_callers == 1,
                "nested lexical calls contribute only one observed caller");
        require(!outer.is_native_only() && !nested.is_native_only(),
                "ordinary nested calls share the noncapture snapshot");
        require_logic_error([&] { const ProcessCallGate::Call wrong(foreign); });
        require_logic_error([&] { const ProcessCallGate::Transition wrong(foreign); });
        {
            ProcessCallGate::Transition transition(gate);
            require(outer.is_native_only(), "transition callbacks must be native-only");
            const ProcessCallGate::Call callback(gate);
            const auto observed = gate.snapshot();
            require(observed.active_calls == 0 && observed.admitted_callers == 1 &&
                        observed.waiting_transitions == 0 && observed.owns_transition,
                    "exclusive observation is CPU-only and does not count nested callbacks");
            require(callback.is_native_only(), "nested native callback must not deadlock");
            require_logic_error([&] { const ProcessCallGate::Transition duplicate(gate); });
            transition.set_native_only(true);
        }
        require(outer.is_native_only() && nested.is_native_only(),
                "enclosing calls cannot retain old routing after their own transition");
    }
    {
        const ProcessCallGate::Call captured(gate);
        require(captured.is_native_only(), "native-only phase persists after transition exits");
        ProcessCallGate::Transition transition(gate);
        transition.set_native_only(false);
    }
    const ProcessCallGate::Call normal(gate);
    require(!normal.is_native_only(), "explicit proof owner can restore the ordinary phase");
}

void exception_restores_admitted_count_and_phase() {
    ProcessCallGate gate;
    {
        const ProcessCallGate::Call outer(gate);
        try {
            const ProcessCallGate::Transition transition(gate);
            throw std::runtime_error("native callback failure");
        } catch (const std::runtime_error&) {
            require(!outer.is_native_only(), "exception must preserve unchanged phase");
        }
        try {
            ProcessCallGate::Transition transition(gate);
            transition.set_native_only(true);
            throw std::runtime_error("uncertain native capture ownership");
        } catch (const std::runtime_error&) {
            require(outer.is_native_only(), "explicit fail-closed phase survives exception");
        }
    }
    // This second upgrade would hang if either exceptional scope lost its count.
    const ProcessCallGate::Call next(gate);
    ProcessCallGate::Transition transition(gate);
    transition.set_native_only(false);
}

void transition_drains_existing_calls() {
    ProcessCallGate gate;
    std::latch admitted(1);
    std::latch can_leave(1);
    std::latch upgrading(1);
    std::atomic<bool> has_left = false;
    std::atomic<bool> was_drained = false;
    std::thread existing([&] {
        const ProcessCallGate::Call call(gate);
        admitted.count_down();
        can_leave.wait();
        has_left.store(true);
    });
    admitted.wait();
    std::thread changer([&] {
        const ProcessCallGate::Call call(gate);
        upgrading.count_down();
        const ProcessCallGate::Transition transition(gate);
        was_drained.store(has_left.load());
        const ProcessCallGate::Call callback(gate);
        if (!callback.is_native_only()) {
            was_drained.store(false);
        }
    });
    upgrading.wait();
    can_leave.count_down();
    existing.join();
    changer.join();
    require(was_drained.load(), "transition must wait for previously admitted outer calls");
}

void transition_excludes_new_calls_and_publishes_phase() {
    ProcessCallGate gate;
    std::latch owns_transition(1);
    std::latch attempted_entry(1);
    std::atomic<bool> is_transition_active = false;
    std::atomic<bool> entered_during_transition = false;
    std::atomic<bool> inherited_phase = false;
    std::thread changer([&] {
        const ProcessCallGate::Call call(gate);
        ProcessCallGate::Transition transition(gate);
        is_transition_active.store(true);
        owns_transition.count_down();
        attempted_entry.wait();
        transition.set_native_only(true);
        is_transition_active.store(false);
    });
    owns_transition.wait();
    std::thread newcomer([&] {
        attempted_entry.count_down();
        const ProcessCallGate::Call call(gate);
        entered_during_transition.store(is_transition_active.load());
        inherited_phase.store(call.is_native_only());
    });
    changer.join();
    newcomer.join();
    require(!entered_during_transition.load() && inherited_phase.load(),
            "new outer calls must enter only after stable phase publication");
}

void competing_upgrades_release_their_counts_before_waiting() {
    ProcessCallGate gate;
    std::barrier ready(3);
    std::atomic<int> concurrent_transitions = 0;
    std::atomic<int> completed = 0;
    std::atomic<bool> was_exclusive = true;
    auto upgrade = [&] {
        const ProcessCallGate::Call outer(gate);
        const ProcessCallGate::Call nested(gate);
        ready.arrive_and_wait();
        ProcessCallGate::Transition transition(gate);
        if (concurrent_transitions.fetch_add(1) != 0 || !nested.is_native_only()) {
            was_exclusive.store(false);
        }
        transition.set_native_only(true);
        const ProcessCallGate::Call callback(gate);
        if (!callback.is_native_only()) {
            was_exclusive.store(false);
        }
        concurrent_transitions.fetch_sub(1);
        completed.fetch_add(1);
    };
    std::thread first(upgrade);
    std::thread second(upgrade);
    ready.arrive_and_wait();
    first.join();
    second.join();
    require(completed.load() == 2 && was_exclusive.load(),
            "competing upgrades must finish exactly once without overlap or deadlock");
    const ProcessCallGate::Call after(gate);
    require(after.is_native_only(), "last competing transition publishes the stable phase");
}

void concurrent_calls_and_repeated_transitions(bool should_wait_for_admission = false) {
    ProcessCallGate gate;
    constexpr int k_iterations = 1000;
    constexpr std::size_t k_callers = 4;
    std::barrier start(static_cast<std::ptrdiff_t>(k_callers + 1));
    std::atomic<int> callbacks = 0;
    std::atomic<bool> has_wrong_snapshot = false;
    std::array<std::thread, k_callers> callers;
    for (auto& caller : callers) {
        caller = std::thread([&] {
            start.arrive_and_wait();
            for (int iteration = 0; iteration < k_iterations; ++iteration) {
                const ProcessCallGate::Call outer(gate);
                if (should_wait_for_admission) {
                    const ProcessCallGate::AdmissionWait waiting(gate);
                    std::this_thread::yield();
                }
                const bool native_only = outer.is_native_only();
                std::this_thread::yield();
                const ProcessCallGate::Call nested(gate);
                if (nested.is_native_only() != native_only ||
                    outer.is_native_only() != native_only) {
                    has_wrong_snapshot.store(true);
                }
                callbacks.fetch_add(1);
            }
        });
    }
    start.arrive_and_wait();
    for (int iteration = 0; iteration < k_iterations; ++iteration) {
        const ProcessCallGate::Call outer(gate);
        ProcessCallGate::Transition transition(gate);
        transition.set_native_only(iteration % 2 == 0);
        const ProcessCallGate::Call callback(gate);
        if (!callback.is_native_only()) {
            has_wrong_snapshot.store(true);
        }
    }
    for (auto& caller : callers) {
        caller.join();
    }
    require(callbacks.load() == static_cast<int>(k_callers) * k_iterations &&
                !has_wrong_snapshot.load(),
            "repeated transitions preserve every admitted nested call and routing snapshot");
    const ProcessCallGate::Call after(gate);
    require(!after.is_native_only(), "last transition restores ordinary routing");
}

void admission_wait_releases_count_and_refreshes_phase() {
    ProcessCallGate gate;
    ProcessCallGate foreign;
    require_logic_error([&] { const ProcessCallGate::AdmissionWait invalid(gate); });
    const ProcessCallGate::Call outer(gate);
    const ProcessCallGate::Call nested(gate);
    require_logic_error([&] { const ProcessCallGate::AdmissionWait invalid(foreign); });
    {
        const ProcessCallGate::Transition transition(gate);
        require_logic_error([&] { const ProcessCallGate::AdmissionWait invalid(gate); });
    }
    ProcessCallGate::AdmissionWait waiting(gate);
    const auto detached = gate.snapshot();
    require(
        detached.active_calls == 0 && detached.admitted_callers == 1 && !detached.owns_transition,
        "diagnostic snapshot distinguishes detached admission from no caller");
    require(!gate.is_entered_by_current_thread(),
            "detached admission cannot authorize source issue");
    require_logic_error([&] { const ProcessCallGate::Call callback(gate); });
    require_logic_error([&] { const ProcessCallGate::Transition transition(gate); });
    require_logic_error([&] { const ProcessCallGate::AdmissionWait duplicate(gate); });
    std::latch owns_transition{1};
    std::latch resuming{1};
    std::atomic<bool> is_transition_active{};
    std::thread changer([&] {
        const ProcessCallGate::Call call(gate);
        ProcessCallGate::Transition transition(gate);
        is_transition_active.store(true);
        owns_transition.count_down();
        resuming.wait();
        transition.set_native_only(true);
        is_transition_active.store(false);
    });
    owns_transition.wait();
    resuming.count_down();
    waiting.resume();
    waiting.resume();
    require(!is_transition_active.load() && gate.is_entered_by_current_thread() &&
                outer.is_native_only() && nested.is_native_only() &&
                gate.is_native_only_on_current_thread(),
            "resumption waits for transition and refreshes one shared caller snapshot");
    changer.join();
}

void admission_wait_unwind_and_competing_callers() {
    ProcessCallGate gate;
    {
        const ProcessCallGate::Call call(gate);
        try {
            const ProcessCallGate::AdmissionWait waiting(gate);
            throw std::runtime_error("control wait failed");
        } catch (const std::runtime_error&) {
            require(gate.is_entered_by_current_thread(), "wait unwind must restore admission");
        }
        const ProcessCallGate::Transition transition(gate);
    }
    constexpr std::size_t k_callers = 4;
    std::latch detached{static_cast<std::ptrdiff_t>(k_callers)};
    std::latch changed{1};
    std::atomic<std::size_t> resumed{};
    std::array<std::thread, k_callers> callers;
    for (auto& caller : callers) {
        caller = std::thread([&] {
            const ProcessCallGate::Call call(gate);
            ProcessCallGate::AdmissionWait waiting(gate);
            detached.count_down();
            changed.wait();
            waiting.resume();
            require(call.is_native_only(), "competing detached caller refreshes published phase");
            resumed.fetch_add(1);
        });
    }
    detached.wait();
    {
        const ProcessCallGate::Call call(gate);
        ProcessCallGate::Transition transition(gate);
        transition.set_native_only(true);
    }
    changed.count_down();
    for (auto& caller : callers) {
        caller.join();
    }
    require(resumed.load() == k_callers, "all detached admission owners resume exactly once");
    // A leaked contribution would block this final transition.
    const ProcessCallGate::Call call(gate);
    ProcessCallGate::Transition transition(gate);
    transition.set_native_only(false);
}
}  // namespace

int main() {
    try {
        nesting_phase_and_misuse();
        exception_restores_admitted_count_and_phase();
        transition_drains_existing_calls();
        transition_excludes_new_calls_and_publishes_phase();
        competing_upgrades_release_their_counts_before_waiting();
        concurrent_calls_and_repeated_transitions();
        admission_wait_releases_count_and_refreshes_phase();
        admission_wait_unwind_and_competing_callers();
        concurrent_calls_and_repeated_transitions(true);
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
