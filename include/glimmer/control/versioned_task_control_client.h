#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace glimmer::control {

enum class VersionedControlTransport : std::uint8_t { kSocket = 0, kSharedMemory = 1, kAuto = 2 };

// A separate process-local session for explicit transport selection. In auto
// mode socket fallback is allowed only after an exact unsupported response,
// before any domain request, on the same connection. Once a shared service is
// pinned it cannot downgrade. Once a socket connection is lost the session is
// unusable: legacy socket replies do not identify a service incarnation.
// Failed domain requests are never replayed. A shared session may reconnect to
// the same incarnation on a later explicit request; owners with uncertain
// resource state must quarantine themselves. Fork children must not use or
// destroy the inherited object and must promptly exec or _exit. All request
// calls must finish before destruction. The socket directory must be trusted.
class VersionedTaskControlClient final {
   public:
    explicit VersionedTaskControlClient(std::string socket_path,
                                        VersionedControlTransport transport,
                                        std::uint32_t io_timeout_ms = 1000);
    ~VersionedTaskControlClient();
    VersionedTaskControlClient(const VersionedTaskControlClient&) = delete;
    VersionedTaskControlClient& operator=(const VersionedTaskControlClient&) = delete;
    VersionedTaskControlClient(VersionedTaskControlClient&&) = delete;
    VersionedTaskControlClient& operator=(VersionedTaskControlClient&&) = delete;

    [[nodiscard]] std::optional<std::string> request(std::string_view line) noexcept;

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace glimmer::control
