#include "internal/launch_completion_tracker.h"

#include "internal/diagnostics.h"

#include <algorithm>
#include <chrono>

namespace glimmer::interceptor {
namespace {

constexpr std::chrono::milliseconds kHeartbeatInterval{100};

}  // namespace

LaunchCompletionTracker::LaunchCompletionTracker(DriverDispatch& driver, control::LaunchGate& gate,
                                                 bool trace_launch_timings) noexcept
    : driver_(driver), gate_(gate), trace_launch_timings(trace_launch_timings) {}

LaunchCompletionTracker::~LaunchCompletionTracker() noexcept {
    stop();
}

bool LaunchCompletionTracker::start() noexcept {
    try {
        std::scoped_lock lock(mutex_);
        if (monitor_thread_.joinable() || stopping_) {
            return false;
        }
        monitor_thread_ = std::thread([this] { monitor(); });
        return true;
    } catch (...) {
        return false;
    }
}

bool LaunchCompletionTracker::track(core::TaskId task_id, CUstream stream) noexcept {
    if (task_id == 0) {
        return false;
    }

    CUevent event = nullptr;
    if (driver_.event_create(&event, CU_EVENT_DISABLE_TIMING) != CUDA_SUCCESS || event == nullptr) {
        static_cast<void>(fail(task_id));
        return false;
    }
    if (driver_.event_record(event, stream) != CUDA_SUCCESS) {
        static_cast<void>(driver_.event_destroy(event));
        static_cast<void>(fail(task_id));
        return false;
    }

    bool rejected_stopped = false;
    try {
        std::scoped_lock lock(mutex_);
        if (stopping_) {
            rejected_stopped = true;
        } else {
            auto [batch_iterator, inserted] = batches_.try_emplace(task_id, BatchState{});
            if (!inserted && batch_iterator->second.failed) {
                rejected_stopped = true;
            } else {
                const std::uint64_t sequence = next_sequence_++;
                pending_.push_back(
                    PendingLaunch{.task_id = task_id,
                                  .sequence = sequence,
                                  .event = event,
                                  .last_heartbeat = std::chrono::steady_clock::now()});
            }
        }
    } catch (...) {
        static_cast<void>(driver_.event_destroy(event));
        static_cast<void>(fail(task_id));
        return false;
    }
    if (rejected_stopped) {
        static_cast<void>(driver_.event_destroy(event));
        static_cast<void>(fail(task_id));
        return false;
    }
    condition_.notify_one();
    return true;
}

bool LaunchCompletionTracker::close_batch(core::TaskId task_id) noexcept {
    if (task_id == 0) {
        return false;
    }
    bool should_complete = false;
    bool should_fail = false;
    {
        std::scoped_lock lock(mutex_);
        const auto batch_iterator = batches_.find(task_id);
        if (batch_iterator == batches_.end()) {
            return false;
        }
        batch_iterator->second.closed = true;
        const bool has_pending = std::any_of(
            pending_.begin(), pending_.end(),
            [task_id](const PendingLaunch& pending) { return pending.task_id == task_id; });
        if (!has_pending) {
            should_fail = batch_iterator->second.failed;
            should_complete = !should_fail;
            batches_.erase(batch_iterator);
        }
    }
    if (should_fail) {
        return gate_.fail(task_id);
    }
    if (should_complete) {
        return gate_.complete(task_id);
    }
    return true;
}

bool LaunchCompletionTracker::fail(core::TaskId task_id) noexcept {
    if (task_id == 0) {
        return false;
    }
    bool should_fail = false;
    try {
        std::scoped_lock lock(mutex_);
        auto [batch_iterator, inserted] = batches_.try_emplace(task_id, BatchState{});
        static_cast<void>(inserted);
        batch_iterator->second.closed = true;
        batch_iterator->second.failed = true;
        for (PendingLaunch& pending : pending_) {
            if (pending.task_id == task_id) {
                pending.lease_lost = true;
            }
        }
        const bool has_pending = std::any_of(
            pending_.begin(), pending_.end(),
            [task_id](const PendingLaunch& pending) { return pending.task_id == task_id; });
        if (!has_pending) {
            batches_.erase(batch_iterator);
            should_fail = true;
        }
    } catch (...) {
        return gate_.fail(task_id);
    }
    return should_fail ? gate_.fail(task_id) : true;
}

void LaunchCompletionTracker::monitor() noexcept {
    while (true) {
        PendingLaunch candidate;
        {
            std::unique_lock lock(mutex_);
            condition_.wait_for(lock, std::chrono::milliseconds(1),
                                [this] { return stopping_ || !pending_.empty(); });
            if (stopping_) {
                return;
            }
            if (pending_.empty()) {
                continue;
            }
            if (next_index_ >= pending_.size()) {
                next_index_ = 0;
            }
            candidate = pending_[next_index_];
            next_index_ = (next_index_ + 1) % pending_.size();
        }

        bool heartbeat_due = false;
        {
            std::scoped_lock lock(mutex_);
            for (const PendingLaunch& pending : pending_) {
                if (pending.task_id == candidate.task_id &&
                    pending.sequence == candidate.sequence) {
                    const auto now = std::chrono::steady_clock::now();
                    heartbeat_due = now - pending.last_heartbeat >= kHeartbeatInterval;
                    if (heartbeat_due) {
                        for (PendingLaunch& mutable_pending : pending_) {
                            if (mutable_pending.task_id == candidate.task_id &&
                                mutable_pending.sequence == candidate.sequence) {
                                mutable_pending.last_heartbeat = now;
                                break;
                            }
                        }
                    }
                    break;
                }
            }
        }
        if (heartbeat_due && !gate_.heartbeat(candidate.task_id)) {
            static_cast<void>(fail(candidate.task_id));
        }

        const CUresult query_result = driver_.event_query(candidate.event);
        if (query_result == CUDA_ERROR_NOT_READY) {
            continue;
        }

        // Keep the event pending until cleanup finishes. Otherwise close_batch()
        // can publish success while this thread still owns the terminal result.
        const bool event_destroyed = driver_.event_destroy(candidate.event) == CUDA_SUCCESS;
        bool should_complete = false;
        bool should_fail = false;
        {
            std::scoped_lock lock(mutex_);
            for (auto iterator = pending_.begin(); iterator != pending_.end(); ++iterator) {
                if (iterator->task_id != candidate.task_id ||
                    iterator->sequence != candidate.sequence) {
                    continue;
                }
                const std::size_t index = static_cast<std::size_t>(iterator - pending_.begin());
                pending_.erase(iterator);
                next_index_ = pending_.empty() ? 0 : index % pending_.size();
                break;
            }
            const auto batch_iterator = batches_.find(candidate.task_id);
            if (batch_iterator != batches_.end()) {
                if (!event_destroyed || query_result != CUDA_SUCCESS || candidate.lease_lost) {
                    batch_iterator->second.failed = true;
                    for (PendingLaunch& pending : pending_) {
                        if (pending.task_id == candidate.task_id) {
                            pending.lease_lost = true;
                        }
                    }
                }
                const bool has_pending =
                    std::any_of(pending_.begin(), pending_.end(),
                                [task_id = candidate.task_id](const PendingLaunch& pending) {
                                    return pending.task_id == task_id;
                                });
                if (!has_pending && batch_iterator->second.closed) {
                    should_fail = batch_iterator->second.failed;
                    should_complete = !should_fail;
                    batches_.erase(batch_iterator);
                }
            } else {
                should_fail = true;
            }
        }
        if (!should_complete && !should_fail) {
            continue;
        }

        control::LaunchGateTiming gate_timing;
        const auto completion_started = trace_launch_timings
                                            ? std::chrono::steady_clock::now()
                                            : std::chrono::steady_clock::time_point{};
        const bool terminal_transition =
            should_complete
                ? gate_.complete(candidate.task_id, trace_launch_timings ? &gate_timing : nullptr)
                : gate_.fail(candidate.task_id, trace_launch_timings ? &gate_timing : nullptr);
        if (trace_launch_timings) {
            const auto completion_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - completion_started);
            report_launch_completion_diagnostic({
                .task_id = candidate.task_id,
                .remote = gate_timing.remote,
                .completion_nanoseconds =
                    completion_elapsed.count() < 0
                        ? std::uint64_t{0}
                        : static_cast<std::uint64_t>(completion_elapsed.count()),
                .completion_transport_nanoseconds = gate_timing.transport_nanoseconds,
                .completion_request_count = gate_timing.request_count,
                .completed = should_complete && terminal_transition,
            });
        }
    }
}

void LaunchCompletionTracker::stop() noexcept {
    {
        std::scoped_lock lock(mutex_);
        stopping_ = true;
    }
    condition_.notify_all();
    if (monitor_thread_.joinable()) {
        monitor_thread_.join();
    }

    std::vector<PendingLaunch> pending;
    std::unordered_map<core::TaskId, BatchState> batches;
    {
        std::scoped_lock lock(mutex_);
        pending.swap(pending_);
        batches.swap(batches_);
        next_index_ = 0;
    }
    for (const PendingLaunch& launch : pending) {
        static_cast<void>(driver_.event_destroy(launch.event));
    }
    for (const auto& [task_id, state] : batches) {
        static_cast<void>(state);
        static_cast<void>(gate_.fail(task_id));
    }
}

}  // namespace glimmer::interceptor
