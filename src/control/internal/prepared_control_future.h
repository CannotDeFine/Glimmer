#pragma once

#include "remote_control_operation.h"

#include "glimmer/control/async_cooperative_launch_gate.h"

#include <future>
#include <memory>
#include <type_traits>
#include <unistd.h>
#include <utility>
#include <variant>

namespace glimmer::control::internal {

struct ProcessControlState;

// An original typed promise and its operation, prepared before FIFO acceptance.
// Externally serialize preparation/future retrieval, then transfer ownership to
// the mailbox. No callbacks or executor threads are created. A retrieved future
// is NOT acceptance: callers must check submit's ticket, returning an invalid
// public future on rejection. Raw std::future is not fork authorization; domain
// owners must independently guard PID before using any cached positive result.
class PreparedControlFuture final {
   public:
    template <typename Result>
    [[nodiscard]] static std::unique_ptr<PreparedControlFuture> prepare(
        RemoteCooperativeCommand command, const LaunchGateOptions& options,
        std::shared_ptr<AsyncCooperativeLaunchGate::ProgressSignal> signal = {},
        bool should_measure_claim_timing = false) noexcept {
        const bool should_measure =
            command.settings.should_measure_timing || should_measure_claim_timing;
        const auto enqueued = should_measure ? timestamp_now() : 0;
        return create<Result>(RemoteControlOperation::prepare(command, options), std::move(signal),
                              should_measure, enqueued);
    }
    template <typename Result>
    [[nodiscard]] static std::unique_ptr<PreparedControlFuture> prepare(
        RemoteNativeCommand command, const LaunchGateOptions& options,
        std::optional<NativeEpochExtent> policy,
        std::shared_ptr<AsyncCooperativeLaunchGate::ProgressSignal> signal = {}) noexcept {
        return create<Result>(RemoteControlOperation::prepare(command, options, policy),
                              std::move(signal), false, 0);
    }
    template <typename Result>
    [[nodiscard]] std::future<Result> take_future() noexcept {
        if (process_id != ::getpid() || has_future_ || is_published_) return {};
        auto* promise = std::get_if<std::promise<Result>>(&promise_);
        if (promise == nullptr) return {};
        try {
            auto future = promise->get_future();
            has_future_ = true;
            return future;
        } catch (...) {
            return {};
        }
    }
    ~PreparedControlFuture() = default;
    PreparedControlFuture(const PreparedControlFuture&) = delete;
    PreparedControlFuture& operator=(const PreparedControlFuture&) = delete;
    PreparedControlFuture(PreparedControlFuture&&) = delete;
    PreparedControlFuture& operator=(PreparedControlFuture&&) = delete;

   private:
    friend struct ProcessControlState;
    using Promise =
        std::variant<std::monostate, std::promise<std::optional<core::TaskId>>,
                     std::promise<CooperativeClaim>, std::promise<CooperativeSubmission>,
                     std::promise<CooperativeHandoff>, std::promise<std::optional<bool>>,
                     std::promise<bool>, std::promise<NativeEpochClaim>>;

    PreparedControlFuture() = default;
    template <typename Result>
    static std::unique_ptr<PreparedControlFuture> create(
        std::unique_ptr<RemoteControlOperation> operation,
        std::shared_ptr<AsyncCooperativeLaunchGate::ProgressSignal> signal, bool should_measure,
        std::uint64_t enqueued) noexcept {
        if (!operation || !operation->is_prepared() ||
            std::get_if<Result>(&operation->result_) == nullptr)
            return nullptr;
        if constexpr (!std::is_same_v<Result, CooperativeClaim> &&
                      !std::is_same_v<Result, CooperativeSubmission> &&
                      !std::is_same_v<Result, CooperativeHandoff>) {
            if (should_measure) return nullptr;
        }
        try {
            auto prepared = std::unique_ptr<PreparedControlFuture>(new PreparedControlFuture);
            prepared->promise_.template emplace<std::promise<Result>>();
            prepared->operation_ = std::move(operation);
            prepared->needs_terminal_ack_ = requires_terminal_ack(*prepared->operation_);
            prepared->signal_ = std::move(signal);
            prepared->should_measure_timing_ = should_measure;
            prepared->enqueued_nanoseconds_ = enqueued;
            return prepared;
        } catch (...) {
            return nullptr;
        }
    }
    [[nodiscard]] bool can_accept() const noexcept;
    void started() noexcept;
    [[nodiscard]] bool publish(const RemoteControlResult& result) noexcept;
    void notify() noexcept;
    [[nodiscard]] static bool requires_terminal_ack(
        const RemoteControlOperation& operation) noexcept;
    [[nodiscard]] static bool terminal_acknowledged(const RemoteControlResult& result) noexcept;
    [[nodiscard]] static std::uint64_t timestamp_now() noexcept;

    const pid_t process_id = ::getpid();
    std::unique_ptr<RemoteControlOperation> operation_;
    Promise promise_;
    std::shared_ptr<AsyncCooperativeLaunchGate::ProgressSignal> signal_;
    std::uint64_t enqueued_nanoseconds_ = 0;
    std::uint64_t started_nanoseconds_ = 0;
    bool should_measure_timing_ = false;
    bool has_future_ = false;
    bool is_published_ = false;
    bool needs_terminal_ack_ = false;
};

}  // namespace glimmer::control::internal
