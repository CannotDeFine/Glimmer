#pragma once

#include "glimmer/control/task_endpoint.h"

#include <cstdint>
#include <atomic>
#include <mutex>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

namespace glimmer::control {

struct UnixSocketControlConfig {
    std::string socket_path;
    // If unset, only the effective uid of the server process is accepted.
    std::optional<std::uint32_t> allowed_uid;
    std::uint32_t io_timeout_ms = 1000;
};

enum class UnixSocketServeStatus : std::uint8_t {
    kServed,
    kTimedOut,
    kError,
};

class UnixSocketControlServer final {
   public:
    UnixSocketControlServer(TaskControlEndpoint& endpoint, UnixSocketControlConfig config) noexcept;
    ~UnixSocketControlServer();

    UnixSocketControlServer(const UnixSocketControlServer&) = delete;
    UnixSocketControlServer& operator=(const UnixSocketControlServer&) = delete;
    UnixSocketControlServer(UnixSocketControlServer&&) = delete;
    UnixSocketControlServer& operator=(UnixSocketControlServer&&) = delete;

    // Binds and listens on the configured path. Existing filesystem entries
    // are never removed by start(); stale paths must be handled explicitly.
    [[nodiscard]] bool start() noexcept;

    // Accepts one client connection and starts its request loop. The caller
    // may invoke this in a loop while existing clients exchange additional
    // request/response pairs on their persistent connections.
    [[nodiscard]] bool serve_one() noexcept;

    // Like serve_one(), but returns to the caller when no client arrives
    // within timeout_ms. A zero timeout keeps the call blocking.
    [[nodiscard]] UnixSocketServeStatus serve_one_for(std::uint32_t timeout_ms) noexcept;

    // Counts requests whose complete response was written successfully. The
    // count is safe to sample from a supervising service loop.
    [[nodiscard]] std::uint64_t served_request_count() const noexcept;

    // Closes the listening descriptor and removes only the socket inode that
    // this instance created.
    void stop() noexcept;

    [[nodiscard]] bool running() const noexcept;

   private:
    [[nodiscard]] bool handle_client(int client_fd) noexcept;
    [[nodiscard]] bool authenticate_client(int client_fd, TaskPeerIdentity* peer) const noexcept;
    [[nodiscard]] bool read_line(int client_fd, std::string* line) const noexcept;
    [[nodiscard]] bool write_response(int client_fd, const std::string& response) const noexcept;
    void register_client(int client_fd) noexcept;
    void unregister_client(int client_fd) noexcept;
    void close_client(int client_fd) noexcept;
    void reap_finished_clients() noexcept;

    struct ClientThread {
        std::thread thread;
        std::shared_ptr<std::atomic<bool>> finished;
    };

    TaskControlEndpoint& endpoint_;
    UnixSocketControlConfig config_;
    int listen_fd_ = -1;
    std::uint64_t socket_device_ = 0;
    std::uint64_t socket_inode_ = 0;
    std::atomic<bool> stopping_{false};
    std::atomic<std::uint64_t> served_request_count_{0};
    std::mutex clients_mutex_;
    std::unordered_set<int> client_fds_;
    std::vector<ClientThread> client_threads_;
};

}  // namespace glimmer::control
