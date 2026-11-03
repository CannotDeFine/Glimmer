#include "glimmer/control/shared_task_control_client.h"

#include "glimmer/control/task_protocol.h"
#include "internal/shared_control_channel.h"
#include "internal/socket_line_reader.h"

#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <poll.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <utility>

namespace glimmer::control {
namespace {

constexpr std::uint32_t kMaxTimeoutMs = 60'000;

[[nodiscard]] bool wait_writable(int fd, std::chrono::milliseconds timeout) noexcept {
    pollfd descriptor{.fd = fd, .events = POLLOUT, .revents = 0};
    for (;;) {
        const int result = ::poll(&descriptor, 1, static_cast<int>(timeout.count()));
        if (result > 0) {
            return (descriptor.revents & POLLOUT) != 0 &&
                   (descriptor.revents & (POLLERR | POLLHUP | POLLNVAL)) == 0;
        }
        if (result == 0 || errno != EINTR) {
            return false;
        }
    }
}

[[nodiscard]] int connect_server(const std::string& path, std::uint32_t timeout_ms) noexcept {
    if (path.empty() || path.size() >= sizeof(sockaddr_un::sun_path) ||
        path.find('\0') != std::string::npos) {
        return -1;
    }
    const int fd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0);
    if (fd < 0) {
        return -1;
    }
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    std::memcpy(address.sun_path, path.c_str(), path.size() + 1);
    const auto length = static_cast<socklen_t>(offsetof(sockaddr_un, sun_path) + path.size() + 1);
    const int connected = ::connect(fd, reinterpret_cast<const sockaddr*>(&address), length);
    if (connected != 0 &&
        (errno != EINPROGRESS || !wait_writable(fd, std::chrono::milliseconds(timeout_ms)))) {
        static_cast<void>(::close(fd));
        return -1;
    }
    int error = 0;
    socklen_t error_length = sizeof(error);
    struct ucred peer {};
    socklen_t peer_length = sizeof(peer);
    const int flags = ::fcntl(fd, F_GETFL);
    if (::getsockopt(fd, SOL_SOCKET, SO_ERROR, &error, &error_length) != 0 || error != 0 ||
        ::getsockopt(fd, SOL_SOCKET, SO_PEERCRED, &peer, &peer_length) != 0 ||
        peer_length != sizeof(peer) || peer.pid <= 0 || peer.uid != ::geteuid() || flags < 0 ||
        ::fcntl(fd, F_SETFL, flags & ~O_NONBLOCK) != 0) {
        static_cast<void>(::close(fd));
        return -1;
    }
    return fd;
}

[[nodiscard]] bool send_open(int fd, std::uint32_t timeout_ms) noexcept {
    std::string_view remaining = internal::kSharedControlOpen;
    while (!remaining.empty()) {
        if (!wait_writable(fd, std::chrono::milliseconds(timeout_ms))) {
            return false;
        }
        const ssize_t sent = ::send(fd, remaining.data(), remaining.size(), MSG_NOSIGNAL);
        if (sent < 0 && errno == EINTR) {
            continue;
        }
        if (sent <= 0) {
            return false;
        }
        remaining.remove_prefix(static_cast<std::size_t>(sent));
    }
    return true;
}

}  // namespace

struct SharedTaskControlClient::Impl {
    explicit Impl(std::string path, std::uint32_t timeout)
        : socket_path(std::move(path)), io_timeout_ms(timeout), process_id(::getpid()) {}

    ~Impl() {
        reset();
    }

    void reset() noexcept {
        channel.reset();
        reader.reset(-1);
        if (socket_fd >= 0) {
            static_cast<void>(::close(socket_fd));
            socket_fd = -1;
        }
    }

    [[nodiscard]] bool connect() noexcept {
        if (socket_fd >= 0) {
            return true;
        }
        socket_fd = connect_server(socket_path, io_timeout_ms);
        if (socket_fd < 0) {
            return false;
        }
        reader.reset(socket_fd);
        std::string response;
        if (!send_open(socket_fd, io_timeout_ms) ||
            reader.read_line(response, io_timeout_ms) != internal::SocketLineStatus::kLine ||
            response != internal::kSharedControlReady ||
            !channel.receive_offer(socket_fd, std::chrono::milliseconds(io_timeout_ms))) {
            reset();
            return false;
        }
        const auto observed = channel.service_incarnation();
        if (observed[0] == 0 || observed[1] == 0 || (pinned && observed != incarnation)) {
            reset();
            return false;
        }
        incarnation = observed;
        pinned = true;
        return true;
    }

    const std::string socket_path;
    const std::uint32_t io_timeout_ms;
    const pid_t process_id;
    std::mutex mutex;
    int socket_fd = -1;
    internal::SocketLineReader reader;
    internal::SharedControlChannel channel;
    internal::ServiceIncarnation incarnation{};
    bool pinned = false;
};

SharedTaskControlClient::SharedTaskControlClient(std::string socket_path,
                                                 std::uint32_t io_timeout_ms)
    : impl_(std::make_unique<Impl>(std::move(socket_path), io_timeout_ms)) {}

SharedTaskControlClient::~SharedTaskControlClient() = default;

std::optional<std::string> SharedTaskControlClient::request(std::string_view line) noexcept {
    if (::getpid() != impl_->process_id || impl_->io_timeout_ms == 0 ||
        impl_->io_timeout_ms > kMaxTimeoutMs || line.empty() || line.back() != '\n' ||
        line.size() > kTaskProtocolMaxLineBytes || line.find('\n') != line.size() - 1) {
        return std::nullopt;
    }
    try {
        std::scoped_lock lock(impl_->mutex);
        try {
            if (!impl_->connect()) {
                return std::nullopt;
            }
            std::string response;
            if (impl_->channel.publish_request(line) &&
                impl_->channel.receive_response(impl_->socket_fd, impl_->io_timeout_ms, response)) {
                return response;
            }
        } catch (...) {
            // Reset under the same lock as request publication.
            impl_->reset();
            return std::nullopt;
        }
        // Publication may have succeeded. Do not replay or clear the pinned identity.
        impl_->reset();
    } catch (...) {
        // A lock failure happens before the channel is accessed.
        return std::nullopt;
    }
    return std::nullopt;
}

}  // namespace glimmer::control
