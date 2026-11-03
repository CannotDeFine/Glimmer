#pragma once

#include "process_control_mailbox.h"

#include <atomic>
#include <mutex>
#include <vector>

namespace glimmer::control::internal {

struct OwnerPumpedCpuRegistration;

// Shared CPU session behind the public experimental facade. No CUDA or thread.
class OwnerPumpedControlState final : public std::enable_shared_from_this<OwnerPumpedControlState> {
   public:
    using Gate = AsyncCooperativeLaunchGate;
    [[nodiscard]] static std::shared_ptr<OwnerPumpedControlState> create(
        LaunchGateOptions options, Gate::OwnerPumpOptions bounds,
        std::optional<NativeEpochExtent> policy) noexcept;
    ~OwnerPumpedControlState();
    [[nodiscard]] bool same_process() const noexcept;
    [[nodiscard]] std::shared_ptr<OwnerPumpedCpuRegistration> register_cpu() noexcept;
    [[nodiscard]] std::shared_ptr<Gate::ProgressSignal> subscribe() noexcept;
    void broadcast() noexcept;
    void close() noexcept;
    void contain() noexcept;
    void notify_activity() noexcept;
    [[nodiscard]] Gate::OwnerPumpSnapshot snapshot() const noexcept;

    template <typename Result>
    [[nodiscard]] std::future<Result> enqueue(RemoteCooperativeCommand command,
                                              bool should_measure_claim = false) noexcept {
        if (!same_process() || is_closed_.load(std::memory_order_acquire)) return {};
        auto prepared =
            PreparedControlFuture::prepare<Result>(command, options, {}, should_measure_claim);
        return accept<Result>(std::move(prepared));
    }
    template <typename Result>
    [[nodiscard]] std::future<Result> enqueue(RemoteNativeCommand command) noexcept {
        if (!same_process() || is_closed_.load(std::memory_order_acquire)) return {};
        auto prepared = PreparedControlFuture::prepare<Result>(command, options, epoch_policy);
        return accept<Result>(std::move(prepared));
    }
    ProcessControlMailbox mailbox;

   private:
    OwnerPumpedControlState(LaunchGateOptions options, Gate::OwnerPumpOptions bounds,
                            std::optional<NativeEpochExtent> policy);
    template <typename Result>
    std::future<Result> accept(std::unique_ptr<PreparedControlFuture> prepared) noexcept {
        if (!prepared) return {};
        auto future = prepared->template take_future<Result>();
        if (!future.valid() || !mailbox.submit(producer_, std::move(prepared))) return {};
        return future;
    }
    const pid_t process_id = ::getpid();
    const LaunchGateOptions options;
    const std::optional<NativeEpochExtent> epoch_policy;
    const std::size_t subscriber_limit;
    ControlOwnerToken producer_;
    std::mutex subscribers_mutex_;
    std::vector<std::weak_ptr<Gate::ProgressSignal>> subscribers_;
    std::atomic<bool> is_closed_{false};
};

// Owner methods are externally serialized; detach status is observable safely.
// A lease retains this registration, whose CPU state cannot vanish mid-step.
struct OwnerPumpedCpuRegistration {
    std::shared_ptr<OwnerPumpedControlState> state;
    ControlOwnerToken token;
    std::atomic<bool> is_detached{false};
    std::optional<ControlExecutorLease> lease;
    ~OwnerPumpedCpuRegistration();
    [[nodiscard]] bool detach() noexcept;
};

}  // namespace glimmer::control::internal
