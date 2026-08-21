#include "glimmer/control/unix_socket_client.h"

#include "glimmer/control/task_protocol.h"

#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <poll.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <utility>

namespace glimmer::control {
namespace {

class RequestTimingScope final {
   public:
    explicit RequestTimingScope(ControlRequestTiming* timing) noexcept
        : timing_(timing), started_(timing == nullptr ? Clock::time_point{} : Clock::now()) {}

    RequestTimingScope(const RequestTimingScope&) = delete;
    RequestTimingScope& operator=(const RequestTimingScope&) = delete;

    ~RequestTimingScope() noexcept {
        if (timing_ != nullptr) {
            const auto elapsed =
                std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - started_);
            timing_->elapsed_nanoseconds = elapsed.count() < 0
                                               ? std::uint64_t{0}
                                               : static_cast<std::uint64_t>(elapsed.count());
        }
    }

   private:
    using Clock = std::chrono::steady_clock;

    ControlRequestTiming* timing_ = nullptr;
    Clock::time_point started_{};
};

constexpr std::uint32_t kMaxIoTimeoutMs = 60'000;

enum class PollEvent : std::uint8_t {
    kRead = POLLIN,
    kWrite = POLLOUT,
};

[[nodiscard]] bool wait_for_io(int file_descriptor, PollEvent event,
                               std::uint32_t timeout_ms) noexcept {
    const short events = static_cast<short>(event);
    pollfd descriptor{.fd = file_descriptor, .events = events, .revents = 0};
    while (true) {
        const int result = ::poll(&descriptor, 1, static_cast<int>(timeout_ms));
        if (result > 0) {
            return (descriptor.revents & events) != 0 &&
                   (descriptor.revents & (POLLERR | POLLNVAL)) == 0;
        }
        if (result == 0) {
            return false;
        }
        if (errno != EINTR) {
            return false;
        }
    }
}

[[nodiscard]] int connect_to_server(const std::string& path) noexcept {
    if (path.empty() || path.size() >= sizeof(sockaddr_un::sun_path)) {
        return -1;
    }
    const int file_descriptor = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (file_descriptor < 0) {
        return -1;
    }
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    std::memcpy(address.sun_path, path.data(), path.size());
    address.sun_path[path.size()] = '\0';
    const auto address_length =
        static_cast<socklen_t>(offsetof(sockaddr_un, sun_path) + path.size() + 1);
    if (::connect(file_descriptor, reinterpret_cast<const sockaddr*>(&address), address_length) !=
        0) {
        static_cast<void>(::close(file_descriptor));
        return -1;
    }
    return file_descriptor;
}

[[nodiscard]] bool send_all(int file_descriptor, std::string_view line,
                            std::uint32_t timeout_ms) noexcept {
    std::size_t offset = 0;
    while (offset < line.size()) {
        if (!wait_for_io(file_descriptor, PollEvent::kWrite, timeout_ms)) {
            return false;
        }
        const ssize_t sent =
            ::send(file_descriptor, line.data() + offset, line.size() - offset, MSG_NOSIGNAL);
        if (sent < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        if (sent == 0) {
            return false;
        }
        offset += static_cast<std::size_t>(sent);
    }
    return true;
}

[[nodiscard]] bool receive_line(int file_descriptor, std::string* response,
                                std::uint32_t timeout_ms) noexcept {
    if (response == nullptr) {
        return false;
    }
    try {
        response->clear();
        response->reserve(kTaskProtocolMaxLineBytes + 1);
        while (response->size() <= kTaskProtocolMaxLineBytes) {
            if (!wait_for_io(file_descriptor, PollEvent::kRead, timeout_ms)) {
                return false;
            }
            char value = '\0';
            const ssize_t received = ::recv(file_descriptor, &value, 1, 0);
            if (received == 0) {
                return !response->empty();
            }
            if (received < 0) {
                if (errno == EINTR) {
                    continue;
                }
                return false;
            }
            response->push_back(value);
            if (value == '\n') {
                return true;
            }
        }
    } catch (...) {
        return false;
    }
    return false;
}

}  // namespace

UnixSocketControlClient::UnixSocketControlClient(std::string socket_path,
                                                 std::uint32_t io_timeout_ms) noexcept
    : socket_path_(std::move(socket_path)), io_timeout_ms_(io_timeout_ms) {}

std::optional<std::string> UnixSocketControlClient::request(
    std::string_view line, ControlRequestTiming* timing) const noexcept {
    const RequestTimingScope timing_scope(timing);
    if (io_timeout_ms_ == 0 || io_timeout_ms_ > kMaxIoTimeoutMs || line.empty() ||
        line.back() != '\n' || line.size() > kTaskProtocolMaxLineBytes) {
        return std::nullopt;
    }
    const int file_descriptor = connect_to_server(socket_path_);
    if (file_descriptor < 0) {
        return std::nullopt;
    }
    std::optional<std::string> response;
    try {
        std::string response_line;
        if (send_all(file_descriptor, line, io_timeout_ms_) &&
            receive_line(file_descriptor, &response_line, io_timeout_ms_)) {
            response = std::move(response_line);
        }
    } catch (...) {
        response = std::nullopt;
    }
    static_cast<void>(::close(file_descriptor));
    return response;
}

}  // namespace glimmer::control
