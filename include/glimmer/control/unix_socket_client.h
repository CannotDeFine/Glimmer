#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace glimmer::control {

struct ControlRequestTiming {
    std::uint64_t elapsed_nanoseconds = 0;
};

// Linux client for the line-oriented Unix socket control protocol. Requests
// issued by the same thread reuse one authenticated connection; different
// threads keep independent connections. The client owns no scheduler or CUDA
// state.
class UnixSocketControlClient final {
   public:
    explicit UnixSocketControlClient(std::string socket_path,
                                     std::uint32_t io_timeout_ms = 1000) noexcept;

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
    [[nodiscard]] std::optional<std::string> request(
        std::string_view line, ControlRequestTiming* timing = nullptr) const noexcept;

   private:
    std::string socket_path_;
    std::uint32_t io_timeout_ms_;
};

}  // namespace glimmer::control
