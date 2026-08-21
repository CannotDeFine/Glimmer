#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace glimmer::control {

struct ControlRequestTiming {
    std::uint64_t elapsed_nanoseconds = 0;
};

// Linux client for the one-request/one-response Unix socket control protocol.
// The client owns no scheduler or CUDA state and is safe to use sequentially.
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
    // failure, or response exceeding the protocol bound.
    // When timing is non-null, it receives the complete request duration,
    // including connect, I/O, response parsing, and close handling.
    [[nodiscard]] std::optional<std::string> request(
        std::string_view line, ControlRequestTiming* timing = nullptr) const noexcept;

   private:
    std::string socket_path_;
    std::uint32_t io_timeout_ms_;
};

}  // namespace glimmer::control
