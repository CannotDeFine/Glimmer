#pragma once

#include "glimmer/control/cooperative_launch_gate.h"

#include <chrono>
#include <condition_variable>
#include <future>
#include <mutex>
#include <sys/types.h>

namespace glimmer::control::internal {
class OwnerPumpedControlState;
struct OwnerPumpedCpuRegistration;
}  // namespace glimmer::control::internal

namespace glimmer::control {

// Thread-safe submission during lifetime; teardown is externally serialized.
// No CUDA or warm domain transport executes on a submitter (local wake hints
// are CPU-only socket I/O). The experimental factory performs cold startup.
// Invalid futures mean
// rejection BEFORE execution; valid futures must be consumed to inspect the
// actual control result. A CUDA pump may get only an already-ready future.
class AsyncCooperativeLaunchGate final {
   public:
    // Independent, coalescing CPU-only hints. A wakeup is neither a control
    // result nor physical completion proof; consumers must inspect their futures.
    class ProgressSignal final {
       public:
        void notify() noexcept;
        // Consumes an earlier or concurrent notification; false means timeout.
        [[nodiscard]] bool wait_for(std::chrono::microseconds timeout);

       private:
        friend class internal::OwnerPumpedControlState;
        void notify_result() noexcept;
        std::mutex mutex_;
        std::condition_variable condition_;
        bool is_pending_{};
        pid_t owner_process_ = 0;
        std::weak_ptr<internal::OwnerPumpedControlState> activity_;
    };

    struct OwnerPumpOptions {
        std::size_t max_queued_requests = 1024;
        std::size_t max_cpu_owners = 64;
        std::chrono::milliseconds transport_timeout{1000};
    };
    // Preserve the existing public enum representations.
    // NOLINTNEXTLINE(performance-enum-size)
    enum class CpuProgress { kRejected, kIdle, kPending, kCompleted, kQuarantined };
    // NOLINTNEXTLINE(performance-enum-size)
    enum class CpuWaitStatus { kReady, kDue, kInterrupted, kRejected, kError };
    struct CpuWaitOutcome {
        CpuWaitStatus status = CpuWaitStatus::kRejected;
        bool has_activity = false;
        bool has_reply = false;
        bool has_peer_closure = false;
    };
    struct OwnerPumpSnapshot {
        bool is_enabled = false;
        std::size_t outstanding_requests = 0;
        std::size_t eligible_cpu_owners = 0;
        bool has_executor = false;
        bool is_closed = false;
        bool is_quarantined = false;
        std::size_t accepting_cpu_owners = 0;
    };
    class CpuOwner final {
       public:
        CpuOwner() = default;
        ~CpuOwner();
        CpuOwner(CpuOwner&&) noexcept;
        CpuOwner(const CpuOwner&) = delete;
        CpuOwner& operator=(const CpuOwner&) = delete;
        CpuOwner& operator=(CpuOwner&&) = delete;
        [[nodiscard]] explicit operator bool() const noexcept;
        // Refused while leased or last with accepted work. Dropping a busy last
        // registration quarantines, NOT successful detach/drain or GPU proof.
        [[nodiscard]] bool detach() noexcept;
        // Atomically withdraw new acceptance eligibility, but remain eligible
        // to advance accepted work before detach. Idempotent, never GPU proof.
        [[nodiscard]] bool begin_retirement() noexcept;

       private:
        friend class AsyncCooperativeLaunchGate;
        std::shared_ptr<internal::OwnerPumpedCpuRegistration> registration_;
    };
    class CpuExecutor final {
       public:
        ~CpuExecutor();
        CpuExecutor(CpuExecutor&&) noexcept;
        CpuExecutor(const CpuExecutor&) = delete;
        CpuExecutor& operator=(const CpuExecutor&) = delete;
        CpuExecutor& operator=(CpuExecutor&&) = delete;
        [[nodiscard]] CpuProgress progress() noexcept;
        // A hint only; inspect original futures and all due timers afterwards.
        [[nodiscard]] CpuWaitOutcome wait_until(
            std::chrono::steady_clock::time_point deadline) noexcept;

       private:
        friend class AsyncCooperativeLaunchGate;
        explicit CpuExecutor(
            std::shared_ptr<internal::OwnerPumpedCpuRegistration> registration) noexcept;
        std::shared_ptr<internal::OwnerPumpedCpuRegistration> registration_;
    };

    // Explicit experiment only. Finite V3 cold startup; no control thread or
    // legacy fallback. Ordinary construction below remains unchanged. Register
    // real CPU progress sources before accepting metadata; no CUDA is moved.
    [[nodiscard]] static std::unique_ptr<AsyncCooperativeLaunchGate> create_owner_pumped(
        LaunchGateOptions options, OwnerPumpOptions pump_options,
        std::optional<NativeEpochExtent> native_epoch_policy = std::nullopt) noexcept;
    [[nodiscard]] CpuOwner register_cpu_owner() noexcept;
    // Externally serialize each owner's acquire/detach/handle lifecycle and each
    // executor's methods/move/destruction. Different owners may contend safely.
    [[nodiscard]] std::optional<CpuExecutor> try_acquire_cpu(const CpuOwner& owner) noexcept;
    [[nodiscard]] OwnerPumpSnapshot owner_pump_snapshot() const noexcept;
    // Explicit CPU containment only. No terminal ACK, physical proof or replay;
    // consumers must observe quarantine instead of getting an unready future.
    void quarantine_owner_pump() noexcept;

    // An ambiguous remote result is not a clean rejection. Containment is
    // visible before its future becomes ready and prevents later sends.
    [[nodiscard]] bool remote_outcome_uncertain() const noexcept;

    explicit AsyncCooperativeLaunchGate(
        LaunchGateOptions options = {}, std::size_t max_queued_requests = 1024,
        std::optional<NativeEpochExtent> native_epoch_policy = std::nullopt);
    ~AsyncCooperativeLaunchGate();
    AsyncCooperativeLaunchGate(const AsyncCooperativeLaunchGate&) = delete;
    AsyncCooperativeLaunchGate& operator=(const AsyncCooperativeLaunchGate&) = delete;
    AsyncCooperativeLaunchGate(AsyncCooperativeLaunchGate&&) = delete;
    AsyncCooperativeLaunchGate& operator=(AsyncCooperativeLaunchGate&&) = delete;

    // Weak subscriptions are bounded by max_queued_requests. Null means closed,
    // full or allocation failure. Each consumer must use its own signal.
    [[nodiscard]] std::shared_ptr<ProgressSignal> subscribe_progress() noexcept;

    [[nodiscard]] std::future<std::optional<core::TaskId>> submit(
        CooperativeLaunchRequest request = {}) noexcept;
    [[nodiscard]] std::future<CooperativeClaim> claim(core::TaskId task,
                                                      bool should_measure_timing = false) noexcept;
    [[nodiscard]] std::future<CooperativeSubmission> submit_and_claim(
        CooperativeLaunchRequest request = {}) noexcept;
    [[nodiscard]] std::future<std::optional<bool>> retreat_requested(
        TaskExecutionIdentity execution) noexcept;
    [[nodiscard]] std::future<bool> heartbeat(TaskExecutionIdentity execution) noexcept;
    [[nodiscard]] std::future<bool> suspend(TaskExecutionIdentity execution) noexcept;
    [[nodiscard]] std::future<bool> complete(TaskExecutionIdentity execution) noexcept;
    [[nodiscard]] std::future<bool> fail(TaskExecutionIdentity execution) noexcept;
    [[nodiscard]] std::future<bool> cancel_stopped(TaskExecutionIdentity execution) noexcept;
    [[nodiscard]] std::future<bool> cancel_pending(core::TaskId task) noexcept;
    [[nodiscard]] std::future<NativeEpochClaim> acquire_native_epoch(
        NativeEpochLaunchRequest request) noexcept;
    [[nodiscard]] std::future<NativeEpochClaim> claim_native_epoch(
        NativeEpochGrant pending) noexcept;
    [[nodiscard]] std::future<std::optional<bool>> retreat_native_epoch(
        NativeEpochGrant grant) noexcept;
    [[nodiscard]] std::future<bool> heartbeat_native_epoch(NativeEpochGrant grant) noexcept;
    [[nodiscard]] std::future<bool> finish_native_epoch(NativeEpochGrant grant,
                                                        std::uint32_t issued_calls,
                                                        bool has_failed) noexcept;
    [[nodiscard]] std::future<bool> cancel_native_epoch(NativeEpochGrant pending) noexcept;
    // Ordinary gate drains/joins metadata, NOT GPU work. Experimental factory
    // initiates close only: retain CPU owners and advance accepted work until
    // snapshot shows settled or quarantine, before final detach/join. Never
    // call an unready get on the sole executor or infer physical stop here.
    void shutdown() noexcept;

   private:
    struct OwnerPumpTag {};
    AsyncCooperativeLaunchGate(OwnerPumpTag,
                               std::shared_ptr<internal::OwnerPumpedControlState> state);
    struct Implementation;
    std::unique_ptr<Implementation> implementation_;
};

}  // namespace glimmer::control
