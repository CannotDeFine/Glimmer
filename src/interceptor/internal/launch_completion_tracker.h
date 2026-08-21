#pragma once

#include "glimmer/control/launch_gate.h"

#include "internal/driver_dispatch.h"

#include <cuda.h>

#include <chrono>
#include <cstddef>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

namespace glimmer::interceptor {

class LaunchCompletionTracker final {
   public:
    LaunchCompletionTracker(DriverDispatch& driver, control::LaunchGate& gate) noexcept;
    ~LaunchCompletionTracker() noexcept;

    LaunchCompletionTracker(const LaunchCompletionTracker&) = delete;
    LaunchCompletionTracker& operator=(const LaunchCompletionTracker&) = delete;
    LaunchCompletionTracker(LaunchCompletionTracker&&) = delete;
    LaunchCompletionTracker& operator=(LaunchCompletionTracker&&) = delete;

    [[nodiscard]] bool start() noexcept;
    [[nodiscard]] bool track(core::TaskId task_id, CUstream stream) noexcept;

   private:
    struct PendingLaunch {
        core::TaskId task_id = 0;
        CUevent event = nullptr;
        std::chrono::steady_clock::time_point last_heartbeat{};
        bool lease_lost = false;
    };

    void monitor() noexcept;
    void stop() noexcept;

    DriverDispatch& driver_;
    control::LaunchGate& gate_;
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::vector<PendingLaunch> pending_;
    std::size_t next_index_ = 0;
    std::thread monitor_thread_;
    bool stopping_ = false;
};

}  // namespace glimmer::interceptor
