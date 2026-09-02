#pragma once

#include "glimmer/core/scheduler.h"
#include "glimmer/control/unix_socket_client.h"

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

namespace glimmer::control {

struct LaunchGateOptions {
    std::size_t max_concurrent_launches = 1;
    core::SchedulingPolicy scheduling_policy = core::SchedulingPolicy::kWeightedRoundRobin;
    std::string tenant_id = "default";
    std::uint32_t tenant_weight = 1;
    std::uint32_t task_priority = 0;
    // When set, launch leases are coordinated by the Linux control service
    // instead of this process-local scheduler.
    std::string control_socket;
    std::chrono::milliseconds remote_acquire_timeout = std::chrono::milliseconds{0};
    std::chrono::milliseconds remote_poll_interval = std::chrono::milliseconds{2};
};

struct LaunchGateTiming {
    bool remote = false;
    bool lease_reused = false;
    std::uint64_t elapsed_nanoseconds = 0;
    std::uint64_t transport_nanoseconds = 0;
    std::uint32_t request_count = 0;
    std::uint32_t claim_poll_count = 0;
    std::uint32_t wait_request_count = 0;
};

// A launch gate admits host-side launch calls at explicit scheduler boundaries.
// It owns no CUDA state and releases a scheduler slot only after the caller
// reports that the corresponding CUDA work reached a completion boundary.
class LaunchGate final {
   public:
    explicit LaunchGate(LaunchGateOptions options = {});

    LaunchGate(const LaunchGate&) = delete;
    LaunchGate& operator=(const LaunchGate&) = delete;
    LaunchGate(LaunchGate&&) = delete;
    LaunchGate& operator=(LaunchGate&&) = delete;

    // A zero timeout waits indefinitely. A positive timeout bounds the wait
    // for admission and returns no lease if it expires while the task remains
    // queued.
    [[nodiscard]] std::optional<core::TaskId> acquire(
        std::chrono::milliseconds timeout = std::chrono::milliseconds::zero(),
        LaunchGateTiming* timing = nullptr);
    [[nodiscard]] bool heartbeat(core::TaskId task_id);
    [[nodiscard]] bool complete(core::TaskId task_id, LaunchGateTiming* timing = nullptr);
    [[nodiscard]] bool fail(core::TaskId task_id, LaunchGateTiming* timing = nullptr);
    [[nodiscard]] core::SchedulerStats stats() const;
    [[nodiscard]] bool remote_mode() const noexcept {
        return remote_mode_requested_;
    }

   private:
    [[nodiscard]] bool pump_locked();
    [[nodiscard]] bool finish_locked(core::TaskId task_id, core::TaskState state);

    mutable std::mutex mutex_;
    std::condition_variable condition_;
    core::Scheduler scheduler_;
    std::string tenant_id_;
    std::uint32_t tenant_weight_ = 1;
    std::uint32_t task_priority_ = 0;
    bool remote_mode_requested_ = false;
    std::unique_ptr<UnixSocketControlClient> remote_client_;
    std::chrono::milliseconds remote_acquire_timeout_{};
    std::chrono::milliseconds remote_poll_interval_{2};
};

}  // namespace glimmer::control
