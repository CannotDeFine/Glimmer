#pragma once

#include "glimmer/control/task_execution_endpoint.h"
#include "glimmer/control/unix_socket_server.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace glimmer::control {

struct UnixSocketTaskExecutionConfig {
    std::string socket_path;
    // The bundled shared client also requires the server to have its own UID.
    std::optional<std::uint32_t> allowed_uid = std::nullopt;
    std::uint32_t io_timeout_ms = 1000;
    std::size_t max_clients = 32;
};

// The endpoint and its dependencies must outlive the server. start(),
// serve_one_for(), stop(), and destruction require external serialization.
// stop() shuts down connections and joins all workers before returning; it
// does not cancel tasks, release external resources, or prove GPU completion.
// Only negotiated shared-channel requests reach the endpoint. Keep the socket
// path in a directory trusted against replacement by other same-UID actors.
// If start binds but cannot identify the pathname, it leaves a possible stale
// entry rather than risk deleting an unrelated file. No domain work is accepted.
// Peer identity is authenticated per connection, not per inherited descriptor;
// fork children must not use inherited connections or task identities.
class UnixSocketTaskExecutionServer final {
   public:
    explicit UnixSocketTaskExecutionServer(TaskExecutionEndpoint& endpoint,
                                           UnixSocketTaskExecutionConfig config);
    ~UnixSocketTaskExecutionServer();
    UnixSocketTaskExecutionServer(const UnixSocketTaskExecutionServer&) = delete;
    UnixSocketTaskExecutionServer& operator=(const UnixSocketTaskExecutionServer&) = delete;
    UnixSocketTaskExecutionServer(UnixSocketTaskExecutionServer&&) = delete;
    UnixSocketTaskExecutionServer& operator=(UnixSocketTaskExecutionServer&&) = delete;

    [[nodiscard]] bool start() noexcept;
    // Use a nonzero timeout when the owner must stop the server cooperatively.
    [[nodiscard]] UnixSocketServeStatus serve_one_for(std::uint32_t timeout_ms) noexcept;
    void stop() noexcept;

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace glimmer::control
