#pragma once

#include "owned_control_exchange.h"

#include <array>
#include <atomic>
#include <optional>

namespace glimmer::control::internal {

enum class ControlWaitStatus : std::uint8_t { kReady, kDue, kInterrupted, kRejected, kError };
struct ControlWaitOutcome {
    ControlWaitStatus status = ControlWaitStatus::kRejected;
    bool has_activity = false;
    bool has_reply = false;
    bool has_peer_closure = false;
};
struct ControlWaitPlan {
    OwnedControlExchange::Clock::time_point deadline;
    std::optional<OwnedExchangeWait> reply;
};

// Thread-safe bounded notification; one externally serialized consumer waits.
// Owned local descriptors are never returned or transferred. No hint is result
// authority. Wait/close/destruction are externally serialized with each other;
// close/destruction also require notifier quiescence. Fork may close CPU copies
// but cannot notify/consume the parent's shared open-file descriptions.
class LocalControlWake final {
   public:
    LocalControlWake();
    ~LocalControlWake();
    LocalControlWake(const LocalControlWake&) = delete;
    LocalControlWake& operator=(const LocalControlWake&) = delete;
    // True may mean coalesced with an in-progress sender, not confirmed I/O.
    // A failed sender makes subsequent operations fail closed.
    [[nodiscard]] bool notify() noexcept;
    [[nodiscard]] ControlWaitOutcome wait_until(ControlWaitPlan plan) noexcept;
    [[nodiscard]] bool close() noexcept;

   private:
    [[nodiscard]] ControlWaitOutcome consume() noexcept;
    const pid_t process_id;
    std::array<int, 2> descriptors_{-1, -1};
    std::atomic<bool> has_error_{false};
    std::atomic<bool> has_pending_wake_{false};
};

}  // namespace glimmer::control::internal
