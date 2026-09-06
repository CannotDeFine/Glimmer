#include "socket_line_reader.h"

#include <algorithm>
#include <cerrno>
#include <poll.h>
#include <string_view>
#include <sys/socket.h>

namespace glimmer::control::internal {

SocketLineReader::SocketLineReader(int file_descriptor) noexcept
    : file_descriptor_(file_descriptor) {}

void SocketLineReader::reset(int file_descriptor) noexcept {
    file_descriptor_ = file_descriptor;
    begin_ = 0;
    end_ = 0;
}

SocketLineStatus SocketLineReader::read_line(std::string& line, std::uint32_t timeout_ms,
                                             const std::atomic<bool>* stopping) noexcept {
    line.clear();
    if (file_descriptor_ < 0 || timeout_ms == 0 || timeout_ms > 60'000) {
        return SocketLineStatus::kError;
    }
    try {
        line.reserve(kTaskProtocolMaxLineBytes + 1);
        while (true) {
            if (stopping != nullptr && stopping->load(std::memory_order_acquire)) {
                return SocketLineStatus::kError;
            }
            if (begin_ < end_) {
                const std::string_view pending(buffer_.data() + begin_, end_ - begin_);
                const std::size_t newline = pending.find('\n');
                const std::size_t count =
                    std::min(newline == std::string_view::npos ? pending.size() : newline + 1,
                             kTaskProtocolMaxLineBytes + 1 - line.size());
                line.append(pending.data(), count);
                begin_ += count;
                if (line.size() > kTaskProtocolMaxLineBytes) {
                    return SocketLineStatus::kTooLong;
                }
                if (newline != std::string_view::npos) {
                    return SocketLineStatus::kLine;
                }
            }
            pollfd descriptor{.fd = file_descriptor_, .events = POLLIN, .revents = 0};
            const int ready = ::poll(&descriptor, 1, static_cast<int>(timeout_ms));
            if (ready == 0) {
                return line.empty() ? SocketLineStatus::kIdle : SocketLineStatus::kError;
            }
            if (ready < 0) {
                if (errno == EINTR) {
                    continue;
                }
                return SocketLineStatus::kError;
            }
            if (stopping != nullptr && stopping->load(std::memory_order_acquire)) {
                return SocketLineStatus::kError;
            }
            // Drain readable bytes on hangup, including a final EOF-framed
            // line. POLLHUP alone is also readable EOF, not an idle timeout.
            if ((descriptor.revents & (POLLERR | POLLNVAL)) != 0 ||
                (descriptor.revents & (POLLIN | POLLHUP)) == 0) {
                return SocketLineStatus::kError;
            }
            const ssize_t received = ::recv(file_descriptor_, buffer_.data(), buffer_.size(), 0);
            if (received < 0) {
                if (errno == EINTR) {
                    continue;
                }
                return SocketLineStatus::kError;
            }
            if (received == 0) {
                return line.empty() ? SocketLineStatus::kClosed : SocketLineStatus::kLine;
            }
            begin_ = 0;
            end_ = static_cast<std::size_t>(received);
        }
    } catch (...) {
        return SocketLineStatus::kError;
    }
}

}  // namespace glimmer::control::internal
