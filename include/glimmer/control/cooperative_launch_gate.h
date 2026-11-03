#pragma once

#include "glimmer/control/launch_gate.h"
#include "glimmer/control/task_admission.h"

#include <memory>

namespace glimmer::control {

// Preserve the existing enum representation used by the execution components.
// NOLINTNEXTLINE(performance-enum-size)
enum class CooperativeClaimStatus { kPending, kGranted, kFailed };
// CPU diagnostic metadata only; finish precedes future publication.
struct CooperativeRequestTiming {
    std::uint64_t enqueued_nanoseconds{};
    std::uint64_t started_nanoseconds{};
    std::uint64_t finished_nanoseconds{};
};
struct CooperativeClaim {
    CooperativeClaimStatus status{CooperativeClaimStatus::kFailed};
    TaskExecutionIdentity execution{};
    CooperativeRequestTiming timing{};
};
struct CooperativeSubmission {
    core::TaskId task_id{};
    CooperativeClaim claim{};
};
struct CooperativeHandoff {
    bool is_completed{};
    CooperativeSubmission submission{};
};
struct CooperativeLaunchRequest {
    std::uint32_t relative_priority{};
    std::uint32_t work_units{1};
    // Used only by the asynchronous adapter, never serialized or policy input.
    bool should_measure_timing{};
};

// NOLINTNEXTLINE(performance-enum-size)
enum class NativeEpochClaimStatus { kPending, kGranted, kRejected, kUncertain };
struct NativeEpochClaim {
    NativeEpochClaimStatus status{NativeEpochClaimStatus::kRejected};
    // Uncertain pending claims keep their known identity/extent. An uncertain
    // acquisition may have no safely known identity and cannot be resubmitted.
    NativeEpochGrant grant{};
};
struct NativeEpochLaunchRequest {
    NativeEpochExtent extent{};
    std::uint32_t relative_priority{};
};

// Thread-safe, CUDA-free cooperative admission. No capacity wait is performed.
// Remote transport has a finite I/O deadline and must not run inside a CUDA
// nonblocking pump: backend owners need an asynchronous transport adapter.
// Suspend/terminal reports require backend physical-stop proof, never timeout.
class CooperativeLaunchGate final {
   public:
    explicit CooperativeLaunchGate(
        LaunchGateOptions options = {},
        std::optional<NativeEpochExtent> native_epoch_policy = std::nullopt);
    ~CooperativeLaunchGate();
    CooperativeLaunchGate(const CooperativeLaunchGate&) = delete;
    CooperativeLaunchGate& operator=(const CooperativeLaunchGate&) = delete;
    CooperativeLaunchGate(CooperativeLaunchGate&&) = delete;
    CooperativeLaunchGate& operator=(CooperativeLaunchGate&&) = delete;

    [[nodiscard]] std::optional<core::TaskId> submit(
        CooperativeLaunchRequest request = {}) noexcept;
    [[nodiscard]] CooperativeClaim claim(core::TaskId task) noexcept;
    // No automatic retry or legacy downgrade after any remote failure.
    [[nodiscard]] CooperativeSubmission submit_and_claim(
        CooperativeLaunchRequest request = {}) noexcept;
    // Physical completion must already be proven. No replay after remote ambiguity.
    [[nodiscard]] CooperativeHandoff complete_and_submit_and_claim(
        TaskExecutionIdentity execution, CooperativeLaunchRequest request = {}) noexcept;
    [[nodiscard]] std::optional<bool> retreat_requested(TaskExecutionIdentity execution) noexcept;
    [[nodiscard]] bool heartbeat(TaskExecutionIdentity execution) noexcept;
    [[nodiscard]] bool suspend(TaskExecutionIdentity execution) noexcept;
    [[nodiscard]] bool complete(TaskExecutionIdentity execution) noexcept;
    [[nodiscard]] bool fail(TaskExecutionIdentity execution) noexcept;
    [[nodiscard]] bool cancel_stopped(TaskExecutionIdentity execution) noexcept;
    [[nodiscard]] bool cancel_pending(core::TaskId task) noexcept;
    // Distinct explicit bounded authorization; never widens a unit grant.
    // No retry/downgrade after uncertain acquisition or claim. Expired grants
    // remain owned but authorize no new original call. Backend proof is required
    // before finish; a false/unknown result cannot release ownership.
    [[nodiscard]] NativeEpochClaim acquire_native_epoch(NativeEpochLaunchRequest request) noexcept;
    [[nodiscard]] NativeEpochClaim claim_native_epoch(NativeEpochGrant pending) noexcept;
    [[nodiscard]] std::optional<bool> retreat_native_epoch(NativeEpochGrant grant) noexcept;
    [[nodiscard]] bool heartbeat_native_epoch(NativeEpochGrant grant) noexcept;
    [[nodiscard]] bool finish_native_epoch(NativeEpochGrant grant, std::uint32_t issued_calls,
                                           bool has_failed) noexcept;
    [[nodiscard]] bool cancel_native_epoch(NativeEpochGrant pending) noexcept;
    [[nodiscard]] bool remote_mode() const noexcept;
    // Once a remote reply is lost or cannot be authenticated, the outcome may
    // have committed even if an operation returned failure. No subsequent
    // requests are sent. Ownership is not released by this quarantine.
    [[nodiscard]] bool remote_outcome_uncertain() const noexcept;
    [[nodiscard]] std::optional<core::SchedulerStats> local_stats() const noexcept;

   private:
    struct Implementation;
    std::unique_ptr<Implementation> implementation_;
};

}  // namespace glimmer::control
