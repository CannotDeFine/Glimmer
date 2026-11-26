#include "native_execution_heartbeat.h"

#include <stdexcept>
#include <utility>

namespace glimmer::interceptor {
namespace {
bool same_execution(control::TaskExecutionIdentity left,
                    control::TaskExecutionIdentity right) noexcept {
    return left.task_id == right.task_id && left.generation == right.generation;
}
}  // namespace

NativeExecutionHeartbeat::NativeExecutionHeartbeat(Callback callback_value,
                                                   std::chrono::milliseconds interval_value)
    : callback(std::move(callback_value)), interval(interval_value) {
    const auto max_interval = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::duration::max());
    if (!callback || interval <= std::chrono::milliseconds::zero() || interval > max_interval) {
        throw std::invalid_argument("native heartbeat requires a callback and positive interval");
    }
    thread_ = std::thread([this] { run(); });
}

NativeExecutionHeartbeat::~NativeExecutionHeartbeat() {
    {
        const std::lock_guard lock(mutex_);
        is_closing_ = true;
        execution_.reset();
    }
    condition_.notify_all();
    if (thread_.joinable()) {
        thread_.join();
    }
}

NativeExecutionHeartbeat::Session::Session(NativeExecutionHeartbeat& owner,
                                           control::TaskExecutionIdentity execution_value)
    : owner_(owner), execution(execution_value), has_started_(owner_.start(execution)) {}

NativeExecutionHeartbeat::Session::~Session() {
    (void)stop();
}

bool NativeExecutionHeartbeat::Session::stop() noexcept {
    if (!has_stopped_) {
        has_passed_ = has_started_ && owner_.stop(execution);
        has_stopped_ = true;
    }
    return has_passed_;
}

bool NativeExecutionHeartbeat::start(control::TaskExecutionIdentity execution) {
    if (execution.task_id == 0 || execution.generation == 0) {
        return false;
    }
    {
        const std::lock_guard lock(mutex_);
        if (execution_ || is_in_flight_ || has_failed_ || is_closing_) {
            return false;
        }
        execution_ = execution;
        deadline_ = next_deadline();
    }
    condition_.notify_all();
    return true;
}

bool NativeExecutionHeartbeat::stop(control::TaskExecutionIdentity execution) noexcept {
    std::unique_lock lock(mutex_);
    if (execution_ && !same_execution(*execution_, execution)) {
        return false;
    }
    execution_.reset();
    condition_.notify_all();
    condition_.wait(lock, [&] { return !is_in_flight_; });
    return !has_failed_ && !is_closing_;
}

void NativeExecutionHeartbeat::run() noexcept {
    std::unique_lock lock(mutex_);
    for (;;) {
        condition_.wait(lock, [&] { return is_closing_ || (execution_ && !has_failed_); });
        if (is_closing_) {
            return;
        }
        if (!execution_) continue;
        const auto execution = *execution_;
        if (condition_.wait_until(lock, deadline_, [&] {
                return is_closing_ || !execution_ || !same_execution(*execution_, execution);
            })) {
            continue;
        }
        is_in_flight_ = true;
        lock.unlock();
        bool passed = false;
        try {
            passed = callback(execution);
        } catch (...) {
            passed = false;
        }
        lock.lock();
        if (!passed) {
            has_failed_ = true;
        }
        is_in_flight_ = false;
        if (passed && execution_ && same_execution(*execution_, execution)) {
            deadline_ = next_deadline();
        }
        condition_.notify_all();
    }
}

std::chrono::steady_clock::time_point NativeExecutionHeartbeat::next_deadline() const noexcept {
    const auto now = std::chrono::steady_clock::now();
    const auto delay = std::chrono::duration_cast<std::chrono::steady_clock::duration>(interval);
    const auto maximum = std::chrono::steady_clock::time_point::max();
    return now > maximum - delay ? maximum : now + delay;
}

}  // namespace glimmer::interceptor
