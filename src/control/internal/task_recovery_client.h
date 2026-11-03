#pragma once

#include "task_recovery_protocol.h"

#include "glimmer/control/shared_task_control_client.h"

#include <cstdint>
#include <optional>
#include <string>

namespace glimmer::control::internal {

// Serialized, process-local session. Each instance obtains a fresh random
// nonce; it cannot be copied or resumed by a second instance. The shared transport pins its service
// identity; this client additionally pins the recovery incarnation and session. A response is
// delivered at most once per operation by this object. The caller must itself issue a returned
// grant at most once and may acknowledge only after recording acceptance. State is not durable
// across process death. Stop and join callers before destruction. Inherited objects are not usable
// after fork. All methods require external serialization.
class TaskRecoveryClient final {
   public:
    explicit TaskRecoveryClient(std::string socket_path, std::uint32_t timeout_ms = 1000);
    TaskRecoveryClient(const TaskRecoveryClient&) = delete;
    TaskRecoveryClient& operator=(const TaskRecoveryClient&) = delete;

    // Retrying after an ambiguous transport result uses the same OPEN.
    [[nodiscard]] bool open() noexcept;
    // On an ambiguous transport result, only the identical request may be
    // retried. A delivered result is never returned again by this object.
    [[nodiscard]] std::optional<TaskRecoveryResponse> execute(TaskRecoveryRequest request) noexcept;
    // Advances only after the caller records acceptance of the delivered result.
    [[nodiscard]] bool acknowledge() noexcept;
    // Requires the last result acknowledged and all task resources retired.
    [[nodiscard]] bool close() noexcept;

   private:
    enum class State : std::uint8_t {
        kNew,
        kReady,
        kWaiting,
        kDelivered,
        kClosing,
        kClosed,
        kFailed
    };
    [[nodiscard]] std::optional<TaskRecoveryResponse> exchange(
        const TaskRecoveryLine& line) noexcept;

    SharedTaskControlClient transport_;
    const TaskRecoveryToken nonce;
    TaskRecoveryToken incarnation_{};
    std::uint64_t session_id_{};
    std::uint64_t next_operation_ = 1;
    std::uint64_t acknowledged_through_{};
    TaskRecoveryLine pending_{};
    State state_ = State::kNew;
};

}  // namespace glimmer::control::internal
