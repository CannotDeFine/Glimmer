#include "internal/launch_completion_tracker.h"

#include <chrono>

namespace glimmer::interceptor {
namespace {

constexpr std::chrono::milliseconds kHeartbeatInterval{100};

}  // namespace

LaunchCompletionTracker::LaunchCompletionTracker(DriverDispatch& driver,
                                                 control::LaunchGate& gate) noexcept
    : driver_(driver), gate_(gate) {}

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
        static_cast<void>(gate_.fail(task_id));
        return false;
    }
    if (driver_.event_record(event, stream) != CUDA_SUCCESS) {
        static_cast<void>(driver_.event_destroy(event));
        static_cast<void>(gate_.fail(task_id));
        return false;
    }

    try {
        std::scoped_lock lock(mutex_);
        if (stopping_) {
            static_cast<void>(driver_.event_destroy(event));
            static_cast<void>(gate_.fail(task_id));
            return false;
        }
        pending_.push_back(PendingLaunch{.task_id = task_id,
                                         .event = event,
                                         .last_heartbeat = std::chrono::steady_clock::now()});
    } catch (...) {
        static_cast<void>(driver_.event_destroy(event));
        static_cast<void>(gate_.fail(task_id));
        return false;
    }
    condition_.notify_one();
    return true;
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
                if (pending.task_id == candidate.task_id && pending.event == candidate.event) {
                    const auto now = std::chrono::steady_clock::now();
                    heartbeat_due = now - pending.last_heartbeat >= kHeartbeatInterval;
                    if (heartbeat_due) {
                        for (PendingLaunch& mutable_pending : pending_) {
                            if (mutable_pending.task_id == candidate.task_id &&
                                mutable_pending.event == candidate.event) {
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
            bool should_fail = false;
            {
                std::scoped_lock lock(mutex_);
                for (PendingLaunch& pending : pending_) {
                    if (pending.task_id == candidate.task_id && pending.event == candidate.event &&
                        !pending.lease_lost) {
                        pending.lease_lost = true;
                        should_fail = true;
                        break;
                    }
                }
            }
            if (should_fail) {
                candidate.lease_lost = true;
                static_cast<void>(gate_.fail(candidate.task_id));
            }
        }

        const CUresult query_result = driver_.event_query(candidate.event);
        if (query_result == CUDA_ERROR_NOT_READY) {
            continue;
        }

        bool removed = false;
        {
            std::scoped_lock lock(mutex_);
            for (auto iterator = pending_.begin(); iterator != pending_.end(); ++iterator) {
                if (iterator->task_id != candidate.task_id || iterator->event != candidate.event) {
                    continue;
                }
                const std::size_t index = static_cast<std::size_t>(iterator - pending_.begin());
                pending_.erase(iterator);
                next_index_ = pending_.empty() ? 0 : index % pending_.size();
                removed = true;
                break;
            }
        }
        if (!removed) {
            continue;
        }

        const bool event_destroyed = driver_.event_destroy(candidate.event) == CUDA_SUCCESS;
        if (event_destroyed && query_result == CUDA_SUCCESS && !candidate.lease_lost) {
            static_cast<void>(gate_.complete(candidate.task_id));
        } else {
            static_cast<void>(gate_.fail(candidate.task_id));
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
    {
        std::scoped_lock lock(mutex_);
        pending.swap(pending_);
        next_index_ = 0;
    }
    for (const PendingLaunch& launch : pending) {
        static_cast<void>(driver_.event_destroy(launch.event));
        static_cast<void>(gate_.fail(launch.task_id));
    }
}

}  // namespace glimmer::interceptor
