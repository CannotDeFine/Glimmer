#pragma once

#include <cstdint>
#include <array>
#include <atomic>
#include <optional>
#include <string>
#include <string_view>

namespace glimmer::control {

struct ControlRequestTiming {
    std::uint64_t elapsed_nanoseconds = 0;
    bool used_shared_memory = false;
};

enum class ControlTransport : std::uint8_t {
    kEnvironment,
    kSocket,
    kSharedMemory,
    kAuto,
    kInvalid,
};

// Linux client for the control protocol, with optional authenticated shared
// memory negotiation over the socket. Requests
// issued by the same thread reuse one authenticated connection; different
// threads keep independent connections. The client owns no scheduler or CUDA
// state.
class UnixSocketControlClient final {
   public:
    explicit UnixSocketControlClient(
        std::string socket_path, std::uint32_t io_timeout_ms = 1000,
        ControlTransport transport = ControlTransport::kEnvironment) noexcept;

    UnixSocketControlClient(const UnixSocketControlClient&) = delete;
    UnixSocketControlClient& operator=(const UnixSocketControlClient&) = delete;
    UnixSocketControlClient(UnixSocketControlClient&&) = delete;
    UnixSocketControlClient& operator=(UnixSocketControlClient&&) = delete;

    // Sends one complete protocol line and returns the server's response line.
    // A null result indicates an invalid configuration, timeout, transport
    // failure, or response exceeding the protocol bound. Failed connections
    // are discarded without retrying the request because protocol operations
    // may have side effects.
    // The size bound includes the newline. Non-empty EOF-framed responses
    // remain supported; read-ahead never survives a connection reset.
    // When timing is non-null, it receives the complete request duration,
    // including a connection setup when the thread has no reusable connection.
    // kEnvironment captures GLIMMER_CONTROL_TRANSPORT at construction: unset or
    // "socket" retains the socket path; "shared_memory" requires negotiation;
    // "auto" allows fallback only after an explicit unsupported negotiation.
    // Invalid selections fail closed. Domain requests are never retried or
    // switched to another transport after publication.
    // Shared-memory negotiation permanently pins this logical client's server
    // incarnation, across threads, reconnects and fork. Changed services and
    // pinned auto-mode downgrade reject before domain publication. Create a new
    // client only for a new logical session, never to migrate retained lease IDs.
    [[nodiscard]] std::optional<std::string> request(
        std::string_view line, ControlRequestTiming* timing = nullptr) const noexcept;

   private:
    std::string socket_path_;
    std::uint32_t io_timeout_ms_;
    ControlTransport transport_;
    // Each word changes at most once, from zero to nonzero. Both must match
    // before publication; even concurrent first binds cannot accept a hybrid.
    mutable std::array<std::atomic<std::uint64_t>, 2> service_incarnation_{};
    [[nodiscard]] bool bind_service(const std::array<std::uint64_t, 2>& incarnation) const noexcept;
};

}  // namespace glimmer::control
