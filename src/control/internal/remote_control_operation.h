#pragma once

#include "owned_control_exchange.h"

#include "glimmer/control/cooperative_launch_gate.h"
#include "glimmer/control/native_epoch_protocol.h"
#include "glimmer/control/task_execution_protocol.h"

#include <memory>
#include <optional>
#include <string>
#include <variant>

namespace glimmer::control::internal {

// Only cooperative gate operations are supported. Identity/settings fields
// have their canonical meaning for the operation; unrelated fields are ignored.
struct RemoteCooperativeCommand {
    TaskExecutionProtocolOperation operation;
    TaskExecutionIdentity execution{};
    CooperativeLaunchRequest settings{};
};
struct RemoteNativeCommand {
    NativeEpochOperation operation;
    NativeEpochGrant grant{};
    NativeEpochLaunchRequest settings{};
    std::uint32_t issued_calls = 0;
};

using RemoteControlResult =
    std::variant<std::optional<core::TaskId>, CooperativeClaim, CooperativeSubmission,
                 CooperativeHandoff, std::optional<bool>, bool, NativeEpochClaim>;
enum class RemoteControlStep : std::uint8_t { kRejected, kPending, kFinished, kUncertain };

// Prepared before mailbox acceptance; thereafter one externally serialized
// executor owns progress, result consumption and destruction. The endpoint
// borrowed by begin MUST outlive this operation. No CUDA, wait, synchronous
// client/gate, callback, future or policy mutation runs here. Domain reports
// require physical proof from the backend before preparation/acceptance.
// Finished may be a definite failure: inspect the typed result. Uncertain
// retains ownership outside this object and MUST NOT be replayed or repaired
// by a replacement endpoint. This layer is not the process FIFO/gate ledger.
class RemoteControlOperation final {
   public:
    using Clock = OwnedControlExchange::Clock;
    [[nodiscard]] static std::unique_ptr<RemoteControlOperation> prepare(
        RemoteCooperativeCommand command, const LaunchGateOptions& options) noexcept;
    [[nodiscard]] static std::unique_ptr<RemoteControlOperation> prepare(
        RemoteNativeCommand command, const LaunchGateOptions& options,
        std::optional<NativeEpochExtent> policy) noexcept;
    // Optimization preparation only: copy an already-owned acquire admission.
    // Return null if one combined wire cannot encode; callers then keep their
    // independent operations. No hidden split, endpoint or domain publication.
    [[nodiscard]] static std::unique_ptr<RemoteControlOperation> prepare_handoff(
        TaskExecutionIdentity completed, const RemoteControlOperation& successor) noexcept;
    ~RemoteControlOperation();
    RemoteControlOperation(const RemoteControlOperation&) = delete;
    RemoteControlOperation& operator=(const RemoteControlOperation&) = delete;
    RemoteControlOperation(RemoteControlOperation&&) = delete;
    RemoteControlOperation& operator=(RemoteControlOperation&&) = delete;

    [[nodiscard]] RemoteControlStep begin(OwnedControlExchange& endpoint,
                                          Clock::time_point now = Clock::now()) noexcept;
    [[nodiscard]] RemoteControlStep poll(Clock::time_point now = Clock::now()) noexcept;
    [[nodiscard]] std::optional<OwnedExchangeWait> wait_state() const noexcept;
    [[nodiscard]] std::optional<RemoteControlResult> take_result() noexcept;
    // Process FIFO containment only: settle a never-published prepared record
    // without touching transport. Published/pending/consumed work cannot reset.
    [[nodiscard]] bool reject_unpublished() noexcept;
    [[nodiscard]] bool is_prepared() const noexcept;
    [[nodiscard]] bool was_published() const noexcept;
    // A real primary ACK is retained through secondary rejection/uncertainty.
    // This is metadata acknowledgement, never physical-stop proof.
    [[nodiscard]] bool completion_acknowledged() const noexcept;

   private:
    friend class PreparedControlFuture;
    friend struct ProcessControlState;
    RemoteControlOperation();
    enum class State : std::uint8_t { kPrepared, kPending, kFinished, kUncertain };
    [[nodiscard]] bool decode_cooperative(const TaskExecutionProtocolResponse& response);
    [[nodiscard]] bool decode_native(const NativeEpochResponse& response);
    [[nodiscard]] RemoteControlStep publish(std::string_view line, Clock::time_point now) noexcept;
    void quarantine() noexcept;

    const pid_t process_id;
    TaskExecutionProtocolRequest task_request_{};
    NativeEpochRequest epoch_request_{};
    std::optional<NativeEpochExtent> epoch_policy_;
    NativeEpochGrant expected_epoch_{};
    RemoteControlResult result_;
    RemoteControlResult failure_result_;
    RemoteControlResult acknowledged_failure_{CooperativeHandoff{.is_completed = true}};
    std::string request_line_;
    std::string secondary_line_;
    std::string response_line_;
    OwnedControlExchange* endpoint_ = nullptr;
    OwnedExchangeTicket ticket_;
    State state_ = State::kPrepared;
    bool is_native_ = false;
    bool has_fallback_ = false;
    bool has_completed_ = false;
    bool has_publication_ = false;
    bool is_consumed_ = false;
};

}  // namespace glimmer::control::internal
