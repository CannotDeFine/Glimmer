#pragma once

#include "glimmer/control/task_admission.h"

#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <optional>
#include <thread>

namespace glimmer::interceptor {

// CPU-only heartbeat owner. Sessions may compete for activation, but one
// session object is externally serialized. Destruction requires all sessions
// retired. Callback execution is outside locks and must not reenter this owner.
// A stopped session settles control work, NOT CUDA work or execution ownership.
class NativeExecutionHeartbeat final {
   public:
    using Callback = std::function<bool(control::TaskExecutionIdentity)>;
    explicit NativeExecutionHeartbeat(
        Callback callback, std::chrono::milliseconds interval = std::chrono::milliseconds{25});
    ~NativeExecutionHeartbeat();
    NativeExecutionHeartbeat(const NativeExecutionHeartbeat&) = delete;
    NativeExecutionHeartbeat& operator=(const NativeExecutionHeartbeat&) = delete;

    class Session final {
       public:
        Session(NativeExecutionHeartbeat& owner, control::TaskExecutionIdentity execution);
        ~Session();
        Session(const Session&) = delete;
        Session& operator=(const Session&) = delete;
        [[nodiscard]] bool valid() const noexcept {
            return has_started_;
        }
        [[nodiscard]] bool stop() noexcept;

       private:
        NativeExecutionHeartbeat& owner_;
        const control::TaskExecutionIdentity execution;
        bool has_started_{};
        bool has_stopped_{};
        bool has_passed_{};
    };

   private:
    [[nodiscard]] bool start(control::TaskExecutionIdentity execution);
    [[nodiscard]] bool stop(control::TaskExecutionIdentity execution) noexcept;
    void run() noexcept;
    [[nodiscard]] std::chrono::steady_clock::time_point next_deadline() const noexcept;

    const Callback callback;
    const std::chrono::milliseconds interval;
    std::mutex mutex_;
    std::condition_variable condition_;
    std::optional<control::TaskExecutionIdentity> execution_;
    std::chrono::steady_clock::time_point deadline_{};
    bool is_in_flight_{};
    bool has_failed_{};
    bool is_closing_{};
    std::thread thread_;
};

}  // namespace glimmer::interceptor
