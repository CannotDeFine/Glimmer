#include "glimmer/control/unix_socket_server.h"

#include "glimmer/control/task_protocol.h"

#include <charconv>
#include <cerrno>
#include <cstddef>
#include <cstring>
#include <fcntl.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>
#include <utility>

namespace glimmer::control {
namespace {

constexpr std::uint32_t kMaxIoTimeoutMs = 60'000;

enum class PollEvent : std::uint8_t { kRead = POLLIN, kWrite = POLLOUT };

enum class PollResult : std::uint8_t { kReady, kTimedOut, kError };

[[nodiscard]] PollResult wait_for_io(int file_descriptor, PollEvent event,
                                     std::uint32_t timeout_ms) noexcept {
    const short events = static_cast<short>(event);
    pollfd descriptor{.fd = file_descriptor, .events = events, .revents = 0};
    while (true) {
        const int result = ::poll(&descriptor, 1, static_cast<int>(timeout_ms));
        if (result > 0) {
            if ((descriptor.revents & (POLLERR | POLLNVAL | POLLHUP)) != 0) {
                return PollResult::kError;
            }
            return (descriptor.revents & events) != 0 ? PollResult::kReady : PollResult::kError;
        }
        if (result == 0) {
            return PollResult::kTimedOut;
        }
        if (errno != EINTR) {
            return PollResult::kError;
        }
    }
}

[[nodiscard]] bool stat_socket_path(const std::string& path, std::uint64_t* device,
                                    std::uint64_t* inode) noexcept {
    if (device == nullptr || inode == nullptr) {
        return false;
    }
    struct stat path_stat {};
    if (::lstat(path.c_str(), &path_stat) != 0) {
        return false;
    }
    *device = static_cast<std::uint64_t>(path_stat.st_dev);
    *inode = static_cast<std::uint64_t>(path_stat.st_ino);
    return true;
}

[[nodiscard]] bool read_process_start_time(std::uint32_t pid,
                                           std::uint64_t* start_time_ticks) noexcept {
    if (pid == 0 || start_time_ticks == nullptr) {
        return false;
    }
    try {
        const std::string path = "/proc/" + std::to_string(pid) + "/stat";
        const int file_descriptor = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (file_descriptor < 0) {
            return false;
        }
        char buffer[4096]{};
        const ssize_t bytes_read = ::read(file_descriptor, buffer, sizeof(buffer) - 1);
        static_cast<void>(::close(file_descriptor));
        if (bytes_read <= 0) {
            return false;
        }

        const std::string_view stat_line(buffer, static_cast<std::size_t>(bytes_read));
        const std::size_t command_end = stat_line.rfind(')');
        if (command_end == std::string_view::npos || command_end + 2 > stat_line.size()) {
            return false;
        }
        std::size_t offset = command_end + 2;
        for (unsigned int field = 3; field <= 22; ++field) {
            while (offset < stat_line.size() && stat_line[offset] == ' ') {
                ++offset;
            }
            if (offset == stat_line.size()) {
                return false;
            }
            const std::size_t token_start = offset;
            while (offset < stat_line.size() && stat_line[offset] != ' ') {
                ++offset;
            }
            if (field == 22) {
                std::uint64_t parsed = 0;
                const auto result = std::from_chars(stat_line.data() + token_start,
                                                    stat_line.data() + offset, parsed);
                if (result.ec != std::errc{} || result.ptr != stat_line.data() + offset) {
                    return false;
                }
                *start_time_ticks = parsed;
                return true;
            }
        }
    } catch (...) {
        return false;
    }
    return false;
}

void unlink_socket_path_if_owned(int socket_fd, const std::string& path) noexcept {
    struct stat socket_stat {};
    struct stat path_stat {};
    if (::fstat(socket_fd, &socket_stat) != 0 || ::lstat(path.c_str(), &path_stat) != 0 ||
        socket_stat.st_dev != path_stat.st_dev || socket_stat.st_ino != path_stat.st_ino) {
        return;
    }
    static_cast<void>(::unlink(path.c_str()));
}

}  // namespace

UnixSocketControlServer::UnixSocketControlServer(TaskControlEndpoint& endpoint,
                                                 UnixSocketControlConfig config) noexcept
    : endpoint_(endpoint), config_(std::move(config)) {}

UnixSocketControlServer::~UnixSocketControlServer() {
    stop();
}

bool UnixSocketControlServer::start() noexcept {
    if (listen_fd_ >= 0 || config_.socket_path.empty() || config_.io_timeout_ms == 0 ||
        config_.io_timeout_ms > kMaxIoTimeoutMs ||
        config_.socket_path.size() >= sizeof(sockaddr_un::sun_path)) {
        return false;
    }

    struct stat existing_path {};
    const int path_status = ::lstat(config_.socket_path.c_str(), &existing_path);
    if (path_status == 0 || errno != ENOENT) {
        return false;
    }

    const int socket_fd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (socket_fd < 0) {
        return false;
    }

    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    std::memcpy(address.sun_path, config_.socket_path.c_str(), config_.socket_path.size() + 1);
    const auto address_length =
        static_cast<socklen_t>(offsetof(sockaddr_un, sun_path) + config_.socket_path.size() + 1);
    if (::bind(socket_fd, reinterpret_cast<const sockaddr*>(&address), address_length) != 0 ||
        ::listen(socket_fd, 16) != 0) {
        const int failure_errno = errno;
        unlink_socket_path_if_owned(socket_fd, config_.socket_path);
        static_cast<void>(::close(socket_fd));
        errno = failure_errno;
        return false;
    }

    if (!stat_socket_path(config_.socket_path, &socket_device_, &socket_inode_)) {
        const int failure_errno = errno;
        unlink_socket_path_if_owned(socket_fd, config_.socket_path);
        static_cast<void>(::close(socket_fd));
        socket_device_ = 0;
        socket_inode_ = 0;
        errno = failure_errno;
        return false;
    }
    listen_fd_ = socket_fd;
    stopping_.store(false, std::memory_order_release);
    served_request_count_.store(0, std::memory_order_release);
    return true;
}

bool UnixSocketControlServer::serve_one() noexcept {
    return serve_one_for(0) == UnixSocketServeStatus::kServed;
}

UnixSocketServeStatus UnixSocketControlServer::serve_one_for(std::uint32_t timeout_ms) noexcept {
    if (listen_fd_ < 0) {
        return UnixSocketServeStatus::kError;
    }
    reap_finished_clients();
    if (timeout_ms != 0) {
        pollfd descriptor{.fd = listen_fd_, .events = POLLIN, .revents = 0};
        while (true) {
            const int result = ::poll(&descriptor, 1, static_cast<int>(timeout_ms));
            if (result == 0) {
                return UnixSocketServeStatus::kTimedOut;
            }
            if (result > 0) {
                if ((descriptor.revents & POLLIN) == 0 ||
                    (descriptor.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0) {
                    return UnixSocketServeStatus::kError;
                }
                break;
            }
            if (errno != EINTR) {
                return UnixSocketServeStatus::kError;
            }
        }
    }
    const int client_fd = ::accept4(listen_fd_, nullptr, nullptr, SOCK_CLOEXEC);
    if (client_fd < 0) {
        return errno == EINTR ? UnixSocketServeStatus::kTimedOut : UnixSocketServeStatus::kError;
    }
    register_client(client_fd);
    std::thread client_thread;
    try {
        const auto finished = std::make_shared<std::atomic<bool>>(false);
        client_thread = std::thread([this, client_fd, finished] {
            static_cast<void>(handle_client(client_fd));
            close_client(client_fd);
            finished->store(true, std::memory_order_release);
        });
        client_threads_.push_back(
            ClientThread{.thread = std::move(client_thread), .finished = finished});
    } catch (...) {
        if (client_thread.joinable()) {
            static_cast<void>(::shutdown(client_fd, SHUT_RDWR));
            client_thread.join();
        } else {
            close_client(client_fd);
        }
        return UnixSocketServeStatus::kError;
    }
    return UnixSocketServeStatus::kServed;
}

void UnixSocketControlServer::stop() noexcept {
    stopping_.store(true, std::memory_order_release);
    if (listen_fd_ >= 0) {
        static_cast<void>(::close(listen_fd_));
        listen_fd_ = -1;
    }
    {
        std::scoped_lock lock(clients_mutex_);
        for (const int client_fd : client_fds_) {
            static_cast<void>(::shutdown(client_fd, SHUT_RDWR));
        }
    }
    for (ClientThread& client_thread : client_threads_) {
        if (client_thread.thread.joinable()) {
            client_thread.thread.join();
        }
    }
    client_threads_.clear();
    if (socket_device_ != 0 || socket_inode_ != 0) {
        std::uint64_t current_device = 0;
        std::uint64_t current_inode = 0;
        if (stat_socket_path(config_.socket_path, &current_device, &current_inode) &&
            current_device == socket_device_ && current_inode == socket_inode_) {
            static_cast<void>(::unlink(config_.socket_path.c_str()));
        }
        socket_device_ = 0;
        socket_inode_ = 0;
    }
}

bool UnixSocketControlServer::running() const noexcept {
    return listen_fd_ >= 0;
}

std::uint64_t UnixSocketControlServer::served_request_count() const noexcept {
    return served_request_count_.load(std::memory_order_acquire);
}

bool UnixSocketControlServer::handle_client(int client_fd) noexcept {
    TaskPeerIdentity peer;
    if (!authenticate_client(client_fd, &peer)) {
        return true;
    }
    bool handled_request = false;
    while (!stopping_.load(std::memory_order_acquire)) {
        std::string request;
        if (!read_line(client_fd, &request)) {
            break;
        }
        const auto response = endpoint_.handle(request, peer);
        if (!response.has_value() || !write_response(client_fd, response.value())) {
            break;
        }
        handled_request = true;
        served_request_count_.fetch_add(1, std::memory_order_release);
    }
    return handled_request;
}

bool UnixSocketControlServer::authenticate_client(int client_fd,
                                                  TaskPeerIdentity* peer) const noexcept {
    if (peer == nullptr) {
        return false;
    }
    struct ucred credentials {};
    socklen_t credentials_length = sizeof(credentials);
    if (::getsockopt(client_fd, SOL_SOCKET, SO_PEERCRED, &credentials, &credentials_length) != 0 ||
        credentials_length != sizeof(credentials)) {
        return false;
    }
    const std::uint32_t allowed_uid =
        config_.allowed_uid.value_or(static_cast<std::uint32_t>(::geteuid()));
    if (static_cast<std::uint32_t>(credentials.uid) != allowed_uid || credentials.pid <= 0) {
        return false;
    }
    peer->pid = static_cast<std::uint32_t>(credentials.pid);
    peer->uid = static_cast<std::uint32_t>(credentials.uid);
    // Unbound services only need the kernel-authenticated UID. A missing or
    // restricted /proc does not reduce that default compatibility; a bound
    // service will reject the zero start-time identity when it handles CLAIM.
    static_cast<void>(read_process_start_time(peer->pid, &peer->start_time_ticks));
    return true;
}

bool UnixSocketControlServer::read_line(int client_fd, std::string* line) const noexcept {
    if (line == nullptr) {
        return false;
    }
    try {
        line->clear();
        line->reserve(kTaskProtocolMaxLineBytes + 1);
        while (line->size() <= kTaskProtocolMaxLineBytes) {
            const PollResult poll_result =
                wait_for_io(client_fd, PollEvent::kRead, config_.io_timeout_ms);
            if (poll_result == PollResult::kTimedOut) {
                if (!line->empty()) {
                    // Do not allow a partially written line to hold a worker
                    // indefinitely. An idle, already-authenticated client is
                    // different: it may be between long-running CUDA calls.
                    return false;
                }
                // A persistent client may be idle while its CUDA work is
                // running. The timeout bounds each poll, not the lifetime of
                // an authenticated connection.
                continue;
            }
            if (poll_result == PollResult::kError || stopping_.load(std::memory_order_acquire)) {
                return false;
            }
            char value = '\0';
            const ssize_t received = ::recv(client_fd, &value, 1, 0);
            if (received == 0) {
                return !line->empty();
            }
            if (received < 0) {
                if (errno == EINTR) {
                    continue;
                }
                return false;
            }
            line->push_back(value);
            if (value == '\n') {
                return true;
            }
        }
        return true;
    } catch (...) {
        return false;
    }
}

bool UnixSocketControlServer::write_response(int client_fd,
                                             const std::string& response) const noexcept {
    std::size_t offset = 0;
    while (offset < response.size()) {
        if (wait_for_io(client_fd, PollEvent::kWrite, config_.io_timeout_ms) !=
            PollResult::kReady) {
            return false;
        }
        const ssize_t sent =
            ::send(client_fd, response.data() + offset, response.size() - offset, MSG_NOSIGNAL);
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

void UnixSocketControlServer::register_client(int client_fd) noexcept {
    if (client_fd < 0) {
        return;
    }
    try {
        std::scoped_lock lock(clients_mutex_);
        client_fds_.insert(client_fd);
    } catch (...) {
        // The worker still owns the descriptor. Failure to track it only
        // means stop() cannot proactively wake this client, so fail closed
        // rather than leaving an unbounded worker behind.
        static_cast<void>(::shutdown(client_fd, SHUT_RDWR));
        return;
    }
}

void UnixSocketControlServer::unregister_client(int client_fd) noexcept {
    if (client_fd < 0) {
        return;
    }
    std::scoped_lock lock(clients_mutex_);
    client_fds_.erase(client_fd);
}

void UnixSocketControlServer::close_client(int client_fd) noexcept {
    if (client_fd >= 0) {
        unregister_client(client_fd);
        static_cast<void>(::close(client_fd));
    }
}

void UnixSocketControlServer::reap_finished_clients() noexcept {
    for (auto iterator = client_threads_.begin(); iterator != client_threads_.end();) {
        if (!iterator->finished->load(std::memory_order_acquire)) {
            ++iterator;
            continue;
        }
        if (iterator->thread.joinable()) {
            iterator->thread.join();
        }
        iterator = client_threads_.erase(iterator);
    }
}

}  // namespace glimmer::control
