#include "process_call_gate.h"

#include <limits>
#include <exception>
#include <stdexcept>

namespace glimmer::interceptor {

struct ProcessCallGate::Caller {
    ProcessCallGate* gate{};
    std::size_t depth{};
    bool owns_transition{};
    bool is_native_only{};
    bool is_detached{};
};
thread_local ProcessCallGate::Caller ProcessCallGate::current_caller;

ProcessCallGate::Snapshot ProcessCallGate::snapshot() {
    const std::lock_guard lock(mutex_);
    return {active_calls_, admitted_callers_, waiting_transitions_, owns_transition_,
            is_native_only_};
}

bool ProcessCallGate::is_entered_by_current_thread() const noexcept {
    return current_caller.gate == this && current_caller.depth != 0 && !current_caller.is_detached;
}

bool ProcessCallGate::is_native_only_on_current_thread() const noexcept {
    return is_entered_by_current_thread() &&
           (current_caller.owns_transition || current_caller.is_native_only);
}

ProcessCallGate::Call::Call(ProcessCallGate& gate) : gate_(gate) {
    auto& caller = ProcessCallGate::current_caller;
    if (caller.gate) {
        if (caller.gate != &gate_ || caller.is_detached ||
            caller.depth == std::numeric_limits<std::size_t>::max()) {
            throw std::logic_error("invalid process call gate nesting");
        }
        ++caller.depth;
        return;
    }
    std::unique_lock lock(gate_.mutex_);
    gate_.condition_.wait(
        lock, [&] { return !gate_.owns_transition_ && gate_.waiting_transitions_ == 0; });
    if (gate_.admitted_callers_ == std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error("process call count exhausted");
    }
    ++gate_.active_calls_;
    ++gate_.admitted_callers_;
    caller = {&gate_, 1, false, gate_.is_native_only_};
}

ProcessCallGate::Call::~Call() {
    auto& caller = ProcessCallGate::current_caller;
    if (--caller.depth != 0) {
        return;
    }
    caller = {};
    {
        const std::lock_guard lock(gate_.mutex_);
        --gate_.active_calls_;
        --gate_.admitted_callers_;
    }
    gate_.condition_.notify_all();
}

bool ProcessCallGate::Call::is_native_only() const noexcept {
    return ProcessCallGate::current_caller.owns_transition ||
           ProcessCallGate::current_caller.is_native_only;
}

ProcessCallGate::Transition::Transition(ProcessCallGate& gate) : gate_(gate) {
    auto& caller = ProcessCallGate::current_caller;
    if (caller.gate != &gate_ || caller.depth == 0 || caller.owns_transition ||
        caller.is_detached) {
        throw std::logic_error("process transition requires one admitted caller");
    }
    std::unique_lock lock(gate_.mutex_);
    if (gate_.waiting_transitions_ == std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error("process transition count exhausted");
    }
    ++gate_.waiting_transitions_;
    --gate_.active_calls_;
    gate_.condition_.notify_all();
    bool has_acquired_transition = false;
    try {
        gate_.condition_.wait(lock, [&] { return !gate_.owns_transition_; });
        gate_.owns_transition_ = true;
        has_acquired_transition = true;
        gate_.condition_.wait(lock, [&] { return gate_.active_calls_ == 0; });
    } catch (...) {
        // Only the thread that acquired the transition may clear ownership.
        // condition_variable failures are not ordinary timeout/capture proof.
        if (has_acquired_transition) {
            gate_.owns_transition_ = false;
        }
        --gate_.waiting_transitions_;
        ++gate_.active_calls_;
        gate_.condition_.notify_all();
        throw;
    }
    --gate_.waiting_transitions_;
    caller.owns_transition = true;
}

ProcessCallGate::Transition::~Transition() {
    {
        const std::lock_guard lock(gate_.mutex_);
        ++gate_.active_calls_;
        gate_.owns_transition_ = false;
        ProcessCallGate::current_caller.owns_transition = false;
        // The enclosing call may finish native begin/end validation. Its old
        // routing snapshot must not authorize private work after a phase change.
        ProcessCallGate::current_caller.is_native_only = gate_.is_native_only_;
    }
    gate_.condition_.notify_all();
}

void ProcessCallGate::Transition::set_native_only(bool is_native_only) noexcept {
    const std::lock_guard lock(gate_.mutex_);
    gate_.is_native_only_ = is_native_only;
}

ProcessCallGate::AdmissionWait::AdmissionWait(ProcessCallGate& gate) : gate_(gate) {
    auto& caller = ProcessCallGate::current_caller;
    if (caller.gate != &gate_ || caller.depth == 0 || caller.owns_transition ||
        caller.is_detached) {
        throw std::logic_error("admission wait requires an admitted source caller");
    }
    {
        const std::lock_guard lock(gate_.mutex_);
        caller.is_detached = true;
        --gate_.active_calls_;
    }
    gate_.condition_.notify_all();
}

ProcessCallGate::AdmissionWait::~AdmissionWait() {
    try {
        resume();
    } catch (...) {
        // A failed mutex/condition restoration cannot leave an enclosing Call
        // pretending to own a contribution. No destructor exception may escape.
        std::terminate();
    }
}

void ProcessCallGate::AdmissionWait::resume() {
    if (has_resumed_) {
        return;
    }
    auto& caller = ProcessCallGate::current_caller;
    if (caller.gate != &gate_ || !caller.is_detached || caller.depth == 0) {
        throw std::logic_error("invalid admission wait resumption");
    }
    std::unique_lock lock(gate_.mutex_);
    gate_.condition_.wait(
        lock, [&] { return !gate_.owns_transition_ && gate_.waiting_transitions_ == 0; });
    if (gate_.active_calls_ == std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error("process call count exhausted on admission resume");
    }
    ++gate_.active_calls_;
    caller.is_native_only = gate_.is_native_only_;
    caller.is_detached = false;
    has_resumed_ = true;
}

}  // namespace glimmer::interceptor
