#pragma once

#include "glimmer/control/launch_gate.h"
#include "glimmer/control/versioned_task_control_client.h"

#include <memory>
#include <optional>
#include <vector>

namespace glimmer::control::internal {

// Private opt-in path. It never replaces the frozen V1 LaunchGate layout.
// All returned tasks remain owned until an exact terminal reply is observed.
// Before normal destruction all operations must be joined and every task must
// have an exact terminal acknowledgement. After quarantine, stop and join
// backend work, independently retire registrar resources and retire the entire
// non-reusable service/accounting domain. Destruction or disconnect alone is
// never a cancellation or physical-stop proof.
// A fork child must promptly exec or _exit; inherited state is intentionally
// left for process teardown to avoid touching locks inherited from threads.
class VersionedRemoteLaunchGate final {
   public:
    enum class RetainedPhase : std::uint8_t { kUnknownIdentity, kPending, kRunning, kUncertain };
    struct RetainedTask {
        core::TaskId task{};
        RetainedPhase phase{RetainedPhase::kUnknownIdentity};
    };
    VersionedRemoteLaunchGate(LaunchGateOptions options, VersionedControlTransport transport);
    ~VersionedRemoteLaunchGate();
    VersionedRemoteLaunchGate(const VersionedRemoteLaunchGate&) = delete;
    VersionedRemoteLaunchGate& operator=(const VersionedRemoteLaunchGate&) = delete;
    VersionedRemoteLaunchGate(VersionedRemoteLaunchGate&&) = delete;
    VersionedRemoteLaunchGate& operator=(VersionedRemoteLaunchGate&&) = delete;

    // This private route supports the interceptor's indefinite acquire only.
    [[nodiscard]] std::optional<core::TaskId> acquire(LaunchGateTiming* timing = nullptr) noexcept;
    [[nodiscard]] bool heartbeat(core::TaskId task) noexcept;
    [[nodiscard]] bool complete(core::TaskId task, LaunchGateTiming* timing = nullptr) noexcept;
    [[nodiscard]] bool fail(core::TaskId task, LaunchGateTiming* timing = nullptr) noexcept;
    // Called when physical work may still be running without a reliable
    // completion observation. This never publishes a terminal transition.
    void quarantine() noexcept;
    [[nodiscard]] bool quarantined() const noexcept;
    // Only a quarantined session exposes these diagnostic records. A task ID
    // is not a recovery handle unless an external owner independently proves
    // the same service incarnation and authenticated peer. An unknown identity
    // requires service-wide containment; no snapshot proves cancellation or
    // physical stop.
    [[nodiscard]] std::optional<std::vector<RetainedTask>> retained() const noexcept;
    // Defined only in testing builds.
    struct TestLanes {
        std::uint64_t normal_next{}, normal_turn{}, urgent_next{}, urgent_turn{};
    };
    [[nodiscard]] std::optional<TestLanes> test_lanes() const noexcept;

   private:
    struct State;
    std::unique_ptr<State> state_;
};

}  // namespace glimmer::control::internal
