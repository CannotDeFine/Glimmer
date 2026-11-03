#include "glimmer/control/unix_socket_task_execution_server.h"

#include "internal/process_identity.h"
#include "internal/shared_control_channel.h"
#include "internal/socket_line_reader.h"
#include "internal/socket_path.h"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <mutex>
#include <poll.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <thread>
#include <unordered_set>
#include <unistd.h>
#include <utility>
#include <vector>

namespace glimmer::control {
namespace {

constexpr std::uint32_t kMaxTimeoutMs = 60'000;
constexpr std::size_t kMaxClients = 256;

[[nodiscard]] bool send_ready(int fd, std::chrono::milliseconds timeout) noexcept {
    std::string_view remaining = internal::kSharedControlReady;
    while (!remaining.empty()) {
        pollfd descriptor{.fd = fd, .events = POLLOUT, .revents = 0};
        const int result = ::poll(&descriptor, 1, static_cast<int>(timeout.count()));
        if (result < 0 && errno == EINTR) {
            continue;
        }
        if (result <= 0 || (descriptor.revents & POLLOUT) == 0 ||
            (descriptor.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0) {
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

struct UnixSocketTaskExecutionServer::Impl {
    struct Worker {
        std::thread thread;
        std::shared_ptr<std::atomic<bool>> finished;
    };

    Impl(TaskExecutionEndpoint& task_endpoint, UnixSocketTaskExecutionConfig server_config)
        : endpoint(task_endpoint), config(std::move(server_config)) {}
    ~Impl() {
        stop();
    }

    void unlink_owned() noexcept {
        std::uint64_t current_device = 0;
        std::uint64_t current_inode = 0;
        if ((device != 0 || inode != 0) &&
            internal::stat_socket_path(config.socket_path, &current_device, &current_inode) &&
            current_device == device && current_inode == inode) {
            static_cast<void>(::unlink(config.socket_path.c_str()));
        }
        device = 0;
        inode = 0;
    }

    [[nodiscard]] bool start() noexcept {
        if (listener >= 0 || config.socket_path.empty() ||
            config.socket_path.size() >= sizeof(sockaddr_un::sun_path) ||
            config.socket_path.find('\0') != std::string::npos || config.io_timeout_ms == 0 ||
            config.io_timeout_ms > kMaxTimeoutMs || config.max_clients == 0 ||
            config.max_clients > kMaxClients) {
            return false;
        }
        struct stat existing {};
        if (::lstat(config.socket_path.c_str(), &existing) == 0 || errno != ENOENT) {
            return false;
        }
        incarnation = internal::new_service_incarnation();
        if (incarnation[0] == 0 || incarnation[1] == 0) {
            return false;
        }
        const int fd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
        if (fd < 0) {
            return false;
        }
        sockaddr_un address{};
        address.sun_family = AF_UNIX;
        std::memcpy(address.sun_path, config.socket_path.c_str(), config.socket_path.size() + 1);
        const auto length =
            static_cast<socklen_t>(offsetof(sockaddr_un, sun_path) + config.socket_path.size() + 1);
        if (::bind(fd, reinterpret_cast<const sockaddr*>(&address), length) != 0) {
            static_cast<void>(::close(fd));
            return false;
        }
        if (!internal::stat_socket_path(config.socket_path, &device, &inode) ||
            ::listen(fd, static_cast<int>(config.max_clients)) != 0) {
            unlink_owned();
            static_cast<void>(::close(fd));
            return false;
        }
        listener = fd;
        stopping.store(false, std::memory_order_release);
        return true;
    }

    [[nodiscard]] std::optional<TaskPeerIdentity> authenticate(int fd) const noexcept {
        struct ucred credentials {};
        socklen_t length = sizeof(credentials);
        if (::getsockopt(fd, SOL_SOCKET, SO_PEERCRED, &credentials, &length) != 0 ||
            length != sizeof(credentials) || credentials.pid <= 0 ||
            static_cast<std::uint32_t>(credentials.uid) !=
                config.allowed_uid.value_or(static_cast<std::uint32_t>(::geteuid()))) {
            return std::nullopt;
        }
        TaskPeerIdentity peer{static_cast<std::uint32_t>(credentials.pid),
                              static_cast<std::uint32_t>(credentials.uid), 0};
        return read_process_start_time(peer.pid, &peer.start_time_ticks) ? std::optional{peer}
                                                                         : std::nullopt;
    }

    void handle(int fd) noexcept {
        const auto peer = authenticate(fd);
        if (!peer) {
            return;
        }
        internal::SocketLineReader reader(fd);
        std::string request;
        if (reader.read_line(request, config.io_timeout_ms, &stopping) !=
                internal::SocketLineStatus::kLine ||
            request != internal::kSharedControlOpen) {
            return;
        }
        internal::SharedControlChannel channel;
        if (!channel.create_server(incarnation) ||
            !send_ready(fd, std::chrono::milliseconds(config.io_timeout_ms)) ||
            !channel.send_offer(fd, std::chrono::milliseconds(config.io_timeout_ms))) {
            return;
        }
        while (!stopping.load(std::memory_order_acquire)) {
            const auto status = channel.receive_request(fd, config.io_timeout_ms, request);
            if (status == internal::SharedControlReadStatus::kIdle) {
                continue;
            }
            if (status != internal::SharedControlReadStatus::kLine) {
                break;
            }
            const auto response = endpoint.handle(request, peer);
            if (!response || !channel.publish_response(*response)) {
                // The result may already be published; never cancel or replay.
                break;
            }
        }
    }

    void reap() noexcept {
        for (auto it = workers.begin(); it != workers.end();) {
            if (!it->finished->load(std::memory_order_acquire)) {
                ++it;
            } else {
                it->thread.join();
                it = workers.erase(it);
            }
        }
    }

    [[nodiscard]] UnixSocketServeStatus serve(std::uint32_t timeout_ms) noexcept {
        if (listener < 0 || timeout_ms > kMaxTimeoutMs) {
            return UnixSocketServeStatus::kError;
        }
        reap();
        if (timeout_ms != 0) {
            pollfd descriptor{.fd = listener, .events = POLLIN, .revents = 0};
            int result = 0;
            do {
                result = ::poll(&descriptor, 1, static_cast<int>(timeout_ms));
            } while (result < 0 && errno == EINTR);
            if (result == 0) {
                return UnixSocketServeStatus::kTimedOut;
            }
            if (result < 0 || (descriptor.revents & POLLIN) == 0 ||
                (descriptor.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0) {
                return UnixSocketServeStatus::kError;
            }
        }
        const int fd = ::accept4(listener, nullptr, nullptr, SOCK_CLOEXEC);
        if (fd < 0) {
            return errno == EINTR ? UnixSocketServeStatus::kTimedOut
                                  : UnixSocketServeStatus::kError;
        }
        std::thread worker;
        try {
            // Reserve before starting a joinable thread so allocation failure
            // cannot destroy an unjoined thread during vector insertion.
            workers.reserve(workers.size() + 1);
            {
                std::scoped_lock lock(clients_mutex);
                if (clients.size() >= config.max_clients) {
                    static_cast<void>(::close(fd));
                    return UnixSocketServeStatus::kError;
                }
                clients.insert(fd);
            }
            const auto finished = std::make_shared<std::atomic<bool>>(false);
            worker = std::thread([this, fd, finished] {
                handle(fd);
                {
                    std::scoped_lock lock(clients_mutex);
                    clients.erase(fd);
                }
                static_cast<void>(::close(fd));
                finished->store(true, std::memory_order_release);
            });
            workers.push_back(Worker{.thread = std::move(worker), .finished = finished});
            return UnixSocketServeStatus::kServed;
        } catch (...) {
            if (worker.joinable()) {
                static_cast<void>(::shutdown(fd, SHUT_RDWR));
                worker.join();
            } else {
                std::scoped_lock lock(clients_mutex);
                clients.erase(fd);
                static_cast<void>(::close(fd));
            }
            return UnixSocketServeStatus::kError;
        }
    }

    void stop() noexcept {
        stopping.store(true, std::memory_order_release);
        if (listener >= 0) {
            static_cast<void>(::close(listener));
            listener = -1;
        }
        {
            std::scoped_lock lock(clients_mutex);
            for (const int fd : clients) {
                static_cast<void>(::shutdown(fd, SHUT_RDWR));
            }
        }
        for (auto& worker : workers) {
            worker.thread.join();
        }
        workers.clear();
        unlink_owned();
    }

    TaskExecutionEndpoint& endpoint;
    UnixSocketTaskExecutionConfig config;
    internal::ServiceIncarnation incarnation{};
    int listener = -1;
    std::uint64_t device = 0;
    std::uint64_t inode = 0;
    std::atomic<bool> stopping{false};
    std::mutex clients_mutex;
    std::unordered_set<int> clients;
    std::vector<Worker> workers;
};

UnixSocketTaskExecutionServer::UnixSocketTaskExecutionServer(TaskExecutionEndpoint& endpoint,
                                                             UnixSocketTaskExecutionConfig config)
    : impl_(std::make_unique<Impl>(endpoint, std::move(config))) {}

UnixSocketTaskExecutionServer::~UnixSocketTaskExecutionServer() = default;

bool UnixSocketTaskExecutionServer::start() noexcept {
    return impl_->start();
}

UnixSocketServeStatus UnixSocketTaskExecutionServer::serve_one_for(
    std::uint32_t timeout_ms) noexcept {
    return impl_->serve(timeout_ms);
}

void UnixSocketTaskExecutionServer::stop() noexcept {
    impl_->stop();
}

}  // namespace glimmer::control
