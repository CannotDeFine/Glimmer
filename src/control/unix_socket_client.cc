#include "glimmer/control/unix_socket_client.h"

#include "glimmer/control/task_protocol.h"
#include "internal/socket_line_reader.h"
#include "internal/shared_control_channel.h"

#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <cstdlib>
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

[[nodiscard]] ControlTransport selected_transport(ControlTransport transport) noexcept {
    if (transport == ControlTransport::kSocket || transport == ControlTransport::kSharedMemory ||
        transport == ControlTransport::kAuto) {
        return transport;
    }
    if (transport != ControlTransport::kEnvironment) {
        return ControlTransport::kInvalid;
    }
    const char* value = std::getenv("GLIMMER_CONTROL_TRANSPORT");
    if (value == nullptr || std::strcmp(value, "socket") == 0) {
        return ControlTransport::kSocket;
    }
    if (std::strcmp(value, "shared_memory") == 0) {
        return ControlTransport::kSharedMemory;
    }
    if (std::strcmp(value, "auto") == 0) {
        return ControlTransport::kAuto;
    }
    return ControlTransport::kInvalid;
}

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

class ThreadConnection final {
   public:
    ThreadConnection() noexcept = default;
    ~ThreadConnection() noexcept {
        reset();
    }

    ThreadConnection(const ThreadConnection&) = delete;
    ThreadConnection& operator=(const ThreadConnection&) = delete;

    void reset() noexcept {
        channel_.reset();
        reader_.reset();
        if (file_descriptor_ >= 0) {
            static_cast<void>(::close(file_descriptor_));
            file_descriptor_ = -1;
        }
        socket_path_.clear();
        io_timeout_ms_ = 0;
        process_id_ = 0;
    }

    [[nodiscard]] bool ensure(const std::string& socket_path, std::uint32_t io_timeout_ms,
                              ControlTransport transport) noexcept {
        if (file_descriptor_ >= 0 &&
            (socket_path_ != socket_path || io_timeout_ms_ != io_timeout_ms ||
             process_id_ != ::getpid() || transport_ != transport)) {
            reset();
        }
        if (file_descriptor_ >= 0) {
            return true;
        }
        const int file_descriptor = connect_to_server(socket_path);
        if (file_descriptor < 0) {
            return false;
        }
        try {
            socket_path_ = socket_path;
            io_timeout_ms_ = io_timeout_ms;
            file_descriptor_ = file_descriptor;
            process_id_ = ::getpid();
            transport_ = transport;
            reader_.reset(file_descriptor);
            if (transport != ControlTransport::kSocket && !negotiate(transport)) {
                reset();
                return false;
            }
            return true;
        } catch (...) {
            if (file_descriptor_ >= 0) {
                reset();
            } else {
                static_cast<void>(::close(file_descriptor));
            }
            return false;
        }
    }

    [[nodiscard]] int file_descriptor() const noexcept {
        return file_descriptor_;
    }

    [[nodiscard]] bool receive_line(std::string& response) noexcept {
        return reader_.read_line(response, io_timeout_ms_) == internal::SocketLineStatus::kLine;
    }

    [[nodiscard]] bool shared_memory() const noexcept {
        return channel_.active();
    }

    [[nodiscard]] internal::ServiceIncarnation service_incarnation() const noexcept {
        return channel_.service_incarnation();
    }

    [[nodiscard]] bool shared_exchange(std::string_view line, std::string& response) noexcept {
        return channel_.publish_request(line) &&
               channel_.receive_response(file_descriptor_, io_timeout_ms_, response);
    }

   private:
    [[nodiscard]] bool negotiate(ControlTransport transport) noexcept {
        if (!send_all(file_descriptor_, internal::kSharedControlOpen, io_timeout_ms_)) {
            return false;
        }
        std::string response;
        if (!receive_line(response)) {
            return false;
        }
        if (response == internal::kSharedControlReady) {
            return channel_.receive_offer(file_descriptor_,
                                          std::chrono::milliseconds(io_timeout_ms_));
        }
        // Only a complete negative negotiation reply allows fallback. No domain
        // request has been submitted, and ambiguous outcomes never switch paths.
        return transport == ControlTransport::kAuto &&
               (response == internal::kSharedControlUnsupported ||
                response == "GLIMMER_TASK_V1 ERROR UNSUPPORTED_VERSION\n");
    }

    std::string socket_path_;
    std::uint32_t io_timeout_ms_ = 0;
    pid_t process_id_ = 0;
    int file_descriptor_ = -1;
    internal::SocketLineReader reader_;
    internal::SharedControlChannel channel_;
    ControlTransport transport_ = ControlTransport::kSocket;
};

thread_local ThreadConnection g_thread_connection;

}  // namespace

UnixSocketControlClient::UnixSocketControlClient(std::string socket_path,
                                                 std::uint32_t io_timeout_ms,
                                                 ControlTransport transport) noexcept
    : socket_path_(std::move(socket_path)),
      io_timeout_ms_(io_timeout_ms),
      transport_(selected_transport(transport)) {}

bool UnixSocketControlClient::bind_service(
    const std::array<std::uint64_t, 2>& incarnation) const noexcept {
    static_assert(std::atomic<std::uint64_t>::is_always_lock_free);
    for (std::size_t i = 0; i < incarnation.size(); ++i) {
        if (incarnation[i] == 0) {
            return false;
        }
        auto observed = service_incarnation_[i].load(std::memory_order_relaxed);
        if (observed == 0 && service_incarnation_[i].compare_exchange_strong(
                                 observed, incarnation[i], std::memory_order_relaxed)) {
            observed = incarnation[i];
        }
        // Failed CAS returns the winner. Both words are write-once, including
        // after ambiguous failures or in a forked child.
        if (observed != incarnation[i]) {
            return false;
        }
    }
    return true;
}

std::optional<std::string> UnixSocketControlClient::request(
    std::string_view line, ControlRequestTiming* timing) const noexcept {
    const RequestTimingScope timing_scope(timing);
    if (timing != nullptr) {
        timing->used_shared_memory = false;
    }
    if (io_timeout_ms_ == 0 || io_timeout_ms_ > kMaxIoTimeoutMs || line.empty() ||
        line.back() != '\n' || line.size() > kTaskProtocolMaxLineBytes ||
        transport_ == ControlTransport::kInvalid || transport_ == ControlTransport::kEnvironment) {
        return std::nullopt;
    }
    if (!g_thread_connection.ensure(socket_path_, io_timeout_ms_, transport_)) {
        return std::nullopt;
    }
    const bool is_shared = g_thread_connection.shared_memory();
    if (is_shared ? !bind_service(g_thread_connection.service_incarnation())
                  : transport_ == ControlTransport::kAuto &&
                        (service_incarnation_[0].load(std::memory_order_relaxed) != 0 ||
                         service_incarnation_[1].load(std::memory_order_relaxed) != 0)) {
        g_thread_connection.reset();
        return std::nullopt;
    }
    std::optional<std::string> response;
    try {
        std::string response_line;
        if (timing != nullptr) {
            timing->used_shared_memory = is_shared;
        }
        if (is_shared ? g_thread_connection.shared_exchange(line, response_line)
                      : send_all(g_thread_connection.file_descriptor(), line, io_timeout_ms_) &&
                            g_thread_connection.receive_line(response_line)) {
            response = std::move(response_line);
        }
    } catch (...) {
        response = std::nullopt;
    }
    if (!response.has_value()) {
        // Do not retry a failed request: a side-effecting operation may have
        // reached the service even when its response was lost.
        g_thread_connection.reset();
    }
    return response;
}

}  // namespace glimmer::control
