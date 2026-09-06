#pragma once

#include "glimmer/control/task_protocol.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>

namespace glimmer::control::internal {

enum class SocketLineStatus : std::uint8_t { kLine, kTooLong, kIdle, kClosed, kError };

// Connection-local framing, not descriptor ownership. Use from one thread only.
// Read-ahead is bounded and retained for the next line; reset on every new fd,
// even when the descriptor number is reused. Discard the connection on error.
class SocketLineReader final {
   public:
    explicit SocketLineReader(int file_descriptor = -1) noexcept;

    void reset(int file_descriptor = -1) noexcept;

    // The limit includes the newline. A non-empty EOF-terminated line remains
    // supported by the codec. Idle timeouts may be retried; partial-line
    // timeouts and overlong lines require closing the connection.
    [[nodiscard]] SocketLineStatus read_line(std::string& line, std::uint32_t timeout_ms,
                                             const std::atomic<bool>* stopping = nullptr) noexcept;

   private:
    int file_descriptor_;
    std::array<char, kTaskProtocolMaxLineBytes + 1> buffer_{};
    std::size_t begin_ = 0;
    std::size_t end_ = 0;
};

}  // namespace glimmer::control::internal
