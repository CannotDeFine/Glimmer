#pragma once

#include "shared_control_channel.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace glimmer::control::internal {

// A logical session outlives its endpoints. Binding is thread-safe, write-once,
// fork-retained and deliberately has no reset. Concurrent conflicting first
// offers can poison the pair conservatively, but cannot admit both services.
class ServiceIncarnationBinding final {
   public:
    [[nodiscard]] bool bind(ServiceIncarnation incarnation) noexcept;
    [[nodiscard]] bool matches(ServiceIncarnation incarnation) const noexcept;
    [[nodiscard]] bool is_bound() const noexcept;

   private:
    std::array<std::atomic<std::uint64_t>, 2> words_{};
};

class OwnedControlExchange;

// Retained CPU identity prevents stale tickets from aliasing a replacement
// endpoint, including allocator/fd reuse. A ticket is not domain authority.
class OwnedExchangeTicket final {
   public:
    OwnedExchangeTicket() noexcept = default;
    [[nodiscard]] explicit operator bool() const noexcept;

   private:
    friend class OwnedControlExchange;
    OwnedExchangeTicket(std::shared_ptr<const void> identity, std::uint64_t generation) noexcept;
    std::shared_ptr<const void> identity_;
    std::uint64_t generation_ = 0;
};

struct OwnedExchangeConfig {
    std::string_view socket_path;
    std::chrono::milliseconds io_timeout{1000};
};

enum class OwnedExchangeColdStatus : std::uint8_t { kReady, kUnsupported, kRejected };
enum class OwnedExchangeProgress : std::uint8_t { kRejected, kPending, kResponse, kUncertain };

struct OwnedExchangePublication {
    SharedControlPublication status = SharedControlPublication::kRejected;
    OwnedExchangeTicket ticket;
};

struct OwnedExchangeWait {
    // Borrowed only until the next exchange mutation. The executor must retain
    // and validate the ticket, not treat fd readiness as a result or authority.
    int descriptor;
    std::chrono::steady_clock::time_point deadline;
};

// Owns CPU transport only. Non-copyable/non-movable; externally serialize all
// operations and destruction with one executor lease. The binding must outlive
// this endpoint. Cold initialization may wait/allocate outside owner progress.
// Warm begin/poll are bounded, CUDA-free and never reconnect, replay or wait.
// Uncertainty is sticky for this endpoint; destruction cannot release a grant.
class OwnedControlExchange final {
   public:
    using Clock = std::chrono::steady_clock;

    explicit OwnedControlExchange(ServiceIncarnationBinding& binding);
    ~OwnedControlExchange();
    OwnedControlExchange(const OwnedControlExchange&) = delete;
    OwnedControlExchange& operator=(const OwnedControlExchange&) = delete;
    OwnedControlExchange(OwnedControlExchange&&) = delete;
    OwnedControlExchange& operator=(OwnedControlExchange&&) = delete;

    [[nodiscard]] OwnedExchangeColdStatus initialize(OwnedExchangeConfig config) noexcept;
    // Time arguments are trusted monotonic observations (default: actual clock),
    // not a public application override. Explicit values enable CPU fixtures.
    [[nodiscard]] OwnedExchangePublication begin(std::string_view line,
                                                 Clock::time_point now = Clock::now()) noexcept;
    [[nodiscard]] OwnedExchangeProgress poll(const OwnedExchangeTicket& ticket, std::string& line,
                                             Clock::time_point now = Clock::now()) noexcept;
    [[nodiscard]] std::optional<OwnedExchangeWait> wait_state(
        const OwnedExchangeTicket& ticket) const noexcept;
    [[nodiscard]] bool abandon(const OwnedExchangeTicket& ticket) noexcept;
    [[nodiscard]] bool close_idle() noexcept;
    [[nodiscard]] bool has_cleanup_error() const noexcept;

   private:
    enum class State : std::uint8_t { kDisconnected, kReady, kPending, kUncertain };
    [[nodiscard]] bool owns(const OwnedExchangeTicket& ticket) const noexcept;
    [[nodiscard]] bool close_transport() noexcept;
    void quarantine() noexcept;

    ServiceIncarnationBinding& binding_;
    const std::shared_ptr<const void> identity;
    SharedControlChannel channel_;
    int socket_fd_ = -1;
    const pid_t process_id;
    State state_ = State::kDisconnected;
    std::chrono::milliseconds io_timeout_{};
    Clock::time_point deadline_{};
    Clock::time_point last_observation_{};
    std::uint64_t generation_ = 0;
    bool has_cleanup_error_ = false;
};

}  // namespace glimmer::control::internal
