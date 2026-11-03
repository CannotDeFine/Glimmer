#include "glimmer/control/execution_socket.h"
#include "internal/process_identity.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <future>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <fcntl.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {
std::atomic<bool> should_fragment_rights{};
std::atomic<unsigned int> rights_send_attempts{}, rights_positive_transfers{};
thread_local std::size_t* blocked_send_attempts{};
thread_local int send_error{};
thread_local int receive_error{};
thread_local std::size_t receive_attempts{};
}  // namespace
// NOLINTNEXTLINE(bugprone-reserved-identifier,readability-identifier-naming): GNU ld wrapping ABI.
extern "C" ssize_t __real_send(int descriptor, const void* data, std::size_t bytes, int flags);
// NOLINTNEXTLINE(bugprone-reserved-identifier,readability-identifier-naming): GNU ld wrapping ABI.
extern "C" ssize_t __wrap_send(int descriptor, const void* data, std::size_t bytes, int flags) {
    if (send_error) {
        errno = send_error;
        return -1;
    }
    if (!blocked_send_attempts) return __real_send(descriptor, data, bytes, flags);
    ++*blocked_send_attempts;
    errno = EAGAIN;
    return -1;
}
// NOLINTNEXTLINE(bugprone-reserved-identifier,readability-identifier-naming): GNU ld wrapping ABI.
extern "C" ssize_t __real_sendmsg(int descriptor, const msghdr* message, int flags);
// NOLINTNEXTLINE(bugprone-reserved-identifier,readability-identifier-naming): GNU ld wrapping ABI.
extern "C" ssize_t __wrap_sendmsg(int descriptor, const msghdr* message, int flags) {
    if (blocked_send_attempts) {
        ++*blocked_send_attempts;
        errno = EAGAIN;
        return -1;
    }
    if (!should_fragment_rights.load()) return __real_sendmsg(descriptor, message, flags);
    const auto attempt = ++rights_send_attempts;
    if (attempt == 1) {
        errno = EINTR;
        return -1;
    }
    if (attempt == 2) {
        errno = EAGAIN;
        return -1;
    }
    auto limited = *message;
    auto data = *message->msg_iov;
    data.iov_len = 1;
    limited.msg_iov = &data;
    limited.msg_iovlen = 1;
    const auto count = __real_sendmsg(descriptor, &limited, flags);
    if (count > 0) ++rights_positive_transfers;
    return count;
}
// NOLINTNEXTLINE(bugprone-reserved-identifier,readability-identifier-naming): GNU ld wrapping ABI.
extern "C" ssize_t __real_recvmsg(int descriptor, msghdr* message, int flags);
// NOLINTNEXTLINE(bugprone-reserved-identifier,readability-identifier-naming): GNU ld wrapping ABI.
extern "C" ssize_t __wrap_recvmsg(int descriptor, msghdr* message, int flags) {
    if (!receive_error) return __real_recvmsg(descriptor, message, flags);
    ++receive_attempts;
    errno = receive_error;
    return -1;
}

namespace {
using namespace glimmer::control;
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
template <typename T>
T& require_value(std::optional<T>& value) {
    if (!value) throw std::runtime_error("expected an engaged optional");
    return *value;
}
template <typename T>
const T& require_value(const std::optional<T>& value) {
    if (!value) throw std::runtime_error("expected an engaged optional");
    return *value;
}
template <typename T>
T require_value(std::optional<T>&& value) {
    if (!value) throw std::runtime_error("expected an engaged optional");
    return std::move(*value);
}
class SendBackpressureScope final {
   public:
    explicit SendBackpressureScope(std::size_t& attempts) {
        expect(!blocked_send_attempts, "send backpressure scopes cannot nest");
        blocked_send_attempts = &attempts;
    }
    ~SendBackpressureScope() {
        blocked_send_attempts = nullptr;
    }
    SendBackpressureScope(const SendBackpressureScope&) = delete;
    SendBackpressureScope& operator=(const SendBackpressureScope&) = delete;
};
struct Directory {
    std::string path;
    Directory() {
        char pattern[] = "/tmp/glimmer-exec-socket-XXXXXX";
        const auto created = ::mkdtemp(pattern);
        expect(created, "create private temporary socket directory");
        path = created;
    }
    ~Directory() {
        if (::rmdir(path.c_str()) != 0) std::cerr << "Socket fixture directory was not empty\n";
    }
};
void authenticated_round_trip(const std::string& path) {
    ExecutionSocketConfig config;
    config.socket_path = path;
    ExecutionSocketServer server(std::move(config));
    expect(server.start(), "start private server");
    struct stat info {};
    expect(::lstat(path.c_str(), &info) == 0 && (info.st_mode & 0777) == 0600, "socket is private");
    const auto identity = current_process_identity();
    expect(identity.has_value(), "trusted server identity");
    auto client = std::async(std::launch::async, [&] {
        ExecutionSocketClient connection({path, *identity, 3000});
        const auto hello = connection.open();
        expect(hello && hello->status == 0 && connection.session() == 7, "authenticated bootstrap");
        const auto child = ::fork();
        expect(child >= 0, "fork inherited client guard");
        if (child == 0) {
            const bool guarded =
                !connection.exchange(ExecutionOpcode::kSnapshot) && connection.session() == 0;
            connection.~ExecutionSocketClient();
            _exit(guarded ? 0 : 1);
        }
        int status{};
        expect(
            ::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
            "child cannot use or shutdown parent channel");
        const auto reply = connection.exchange(ExecutionOpcode::kSnapshot);
        expect(reply && reply->id == 19 && reply->original_status == 0,
               "parent channel survives child destruction");
        const auto closed = connection.exchange(ExecutionOpcode::kClose);
        expect(closed && closed->status == 0 && !connection.exchange(ExecutionOpcode::kSnapshot),
               "closed instance never reconnects");
    });
    unsigned int served{};
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready) {
        expect(server.poll_for(1), "nonblocking server progress");
        if (auto request = server.take_request()) {
            expect(request->peer == *identity,
                   "kernel credentials bind the actual sending process");
            ExecutionWireReply reply;
            reply.disposition = ExecutionWireDisposition::kComplete;
            reply.original_status = 0;
            reply.id = request->message.header.operation == ExecutionOpcode::kSnapshot ? 19 : 0;
            auto encoded = encode_execution_reply(reply);
            expect(encoded.has_value(), "encode reply");
            auto header = request->message.header;
            header.kind = ExecutionMessageKind::kReply;
            header.session = 7;
            header.incarnation = {11, 13};
            expect(server.reply(request->connection, {header, std::move(*encoded)}),
                   "queue reply without blocking");
            ++served;
        }
        expect(std::chrono::steady_clock::now() < deadline, "bounded authenticated exchange");
    }
    client.get();
    expect(served == 3, "inherited client generated no extra request");
    server.stop();
    expect(::lstat(path.c_str(), &info) != 0, "remove only own socket inode");
}
void inherited_sender_is_rejected(const std::string& path) {
    ExecutionSocketConfig config;
    config.socket_path = path;
    config.should_record_receive_time = true;
    ExecutionSocketServer server(std::move(config));
    expect(server.start(), "start credential guard server");
    int descriptor = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    expect(descriptor >= 0, "open raw peer");
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    expect(path.size() < sizeof(address.sun_path), "fixture socket path fits");
    std::copy(path.begin(), path.end(), address.sun_path);
    expect(
        ::connect(descriptor, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) == 0 &&
            server.poll_for(0),
        "authenticate parent connection");
    const auto child = ::fork();
    expect(child >= 0, "fork actual wrong sender");
    if (child == 0) {
        const auto bytes = encode_execution_message(
            {{ExecutionMessageKind::kRequest, ExecutionOpcode::kHello, 0, 0, 1, {}}, {}});
        const auto sent = ::send(descriptor, require_value(bytes).data(),
                                 require_value(bytes).size(), MSG_NOSIGNAL);
        _exit(sent == static_cast<ssize_t>(require_value(bytes).size()) ? 0 : 1);
    }
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "child sends through inherited socket");
    for (unsigned int iteration = 0; iteration < 10; ++iteration) {
        expect(server.poll_for(1), "reject wrong sender progress");
        expect(!server.take_request(),
               "SCM sender cannot borrow the original SO_PEERCRED identity");
    }
    expect(::close(descriptor) == 0, "close raw peer");
    server.stop();
}
int connect_raw(const std::string& path) {
    const auto descriptor = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    expect(descriptor >= 0, "create raw protocol peer");
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    expect(path.size() < sizeof(address.sun_path), "bounded socket fixture path");
    std::copy(path.begin(), path.end(), address.sun_path);
    if (::connect(descriptor, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) != 0) {
        const auto closed = ::close(descriptor);
        expect(closed == 0, "close failed raw fixture connection");
        throw std::runtime_error("connect raw protocol peer");
    }
    return descriptor;
}
void process_identity_boundaries() {
    std::uint64_t start = 77;
    expect(!read_process_start_time(0, &start) && start == 77 &&
               !read_process_start_time(static_cast<std::uint32_t>(::getpid()), nullptr),
           "invalid identity inputs preserve the output");
    const auto identity = current_process_identity();
    expect(identity && identity->pid == static_cast<std::uint32_t>(::getpid()) &&
               identity->uid == static_cast<std::uint32_t>(::geteuid()) &&
               read_process_start_time(identity->pid, &start) &&
               start == identity->start_time_ticks,
           "current identity matches the kernel process start time");
    const auto child = ::fork();
    expect(child >= 0, "fork identity boundary child");
    if (child == 0) _exit(0);
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status), "reap identity child");
    start = 77;
    expect(!read_process_start_time(static_cast<std::uint32_t>(child), &start) && start == 77,
           "a reaped process cannot retain an identity");
}
void wrong_uid_is_rejected(const std::string& path) {
    ExecutionSocketConfig config;
    config.socket_path = path;
    config.allowed_uid = static_cast<std::uint32_t>(::geteuid()) ^ 1U;
    ExecutionSocketServer server(std::move(config));
    expect(server.start(), "start UID boundary server");
    ExecutionOwnedDescriptor peer(connect_raw(path));
    expect(server.poll_for(0) && !server.has_peer(1) && !server.take_request(),
           "mismatched actual UID cannot become an authenticated peer");
    pollfd ready{peer.get(), POLLIN | POLLHUP, 0};
    expect(::poll(&ready, 1, 1000) > 0, "rejected peer observes closure");
    char byte{};
    expect(::recv(peer.get(), &byte, 1, 0) <= 0, "wrong UID connection closes");
    server.stop();
}
void forked_server_destruction_preserves_parent(const std::string& path) {
    ExecutionSocketConfig config;
    config.socket_path = path;
    auto server = std::make_unique<ExecutionSocketServer>(std::move(config));
    expect(server->start(), "start fork boundary server");
    ExecutionOwnedDescriptor peer(connect_raw(path));
    expect(server->poll_for(0) && server->has_peer(1), "accept parent-owned peer");
    struct stat original {};
    expect(::lstat(path.c_str(), &original) == 0, "capture original endpoint inode");
    const auto child = ::fork();
    expect(child >= 0, "fork inherited server");
    if (child == 0) {
        server.reset();
        _exit(0);
    }
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "child disposes its inherited server wrapper");
    struct stat current {};
    expect(::lstat(path.c_str(), &current) == 0 && current.st_dev == original.st_dev &&
               current.st_ino == original.st_ino && server->has_peer(1),
           "child destruction preserves the parent endpoint and connection");
    const auto frame = encode_execution_message(
        {{ExecutionMessageKind::kRequest, ExecutionOpcode::kHello, 0, 0, 1, {}}, {}});
    const auto& bytes = require_value(frame);
    expect(::send(peer.get(), bytes.data(), bytes.size(), MSG_NOSIGNAL) ==
                   static_cast<ssize_t>(bytes.size()) &&
               server->poll_for(0) && server->take_request().has_value(),
           "parent can still receive from its original peer");
    server->stop();
    expect(::lstat(path.c_str(), &current) != 0 && errno == ENOENT,
           "parent retains responsibility for endpoint cleanup");
}
std::optional<ExecutionReceivedRequest> inspect_retained_frame(ExecutionSocketServer& server,
                                                               bool takes_exactly) {
    const auto view = server.peek_request(1);
    if (!view) return std::nullopt;
    const std::vector<std::byte> bytes(view->payload.begin(), view->payload.end());
    const auto again = server.peek_request(1);
    expect(again && again->payload.data() == view->payload.data() &&
               std::ranges::equal(again->payload, bytes) &&
               again->header.sequence == view->header.sequence &&
               again->has_descriptor == view->has_descriptor,
           "repeated peek leaves the complete frame and FD in the original channel");
    for (unsigned int mutation = 0; mutation < 9; ++mutation) {
        auto wrong = *view;
        switch (mutation) {
            case 0:
                ++wrong.connection;
                break;
            case 1:
                ++wrong.peer.start_time_ticks;
                break;
            case 2:
                wrong.header.kind = ExecutionMessageKind::kReply;
                break;
            case 3:
                wrong.header.operation = ExecutionOpcode::kDrain;
                break;
            case 4:
                ++wrong.header.payload_bytes;
                break;
            case 5:
                ++wrong.header.session;
                break;
            case 6:
                ++wrong.header.sequence;
                break;
            case 7:
                ++wrong.header.incarnation[0];
                break;
            case 8:
                wrong.has_descriptor = !wrong.has_descriptor;
                break;
            default:
                break;
        }
        expect(!server.take_request(wrong), "mismatched frame token never consumes or closes");
    }
    const auto fresh = server.peek_request(1);
    expect(fresh && std::ranges::equal(fresh->payload, bytes),
           "failed takes retain all payload bytes");
    auto request =
        takes_exactly ? server.take_request(require_value(fresh)) : server.take_request();
    expect(request && request->message.payload == bytes &&
               request->message.header.sequence == view->header.sequence &&
               (request->descriptor.get() >= 0) == view->has_descriptor &&
               !server.peek_request(1) && !server.take_request(*view),
           "exact or ordinary take transfers a retained frame only once");
    return request;
}
void retained_frame_boundaries(const std::string& path) {
    ExecutionSocketConfig config;
    config.socket_path = path;
    config.max_pending_requests = 2;
    ExecutionSocketServer server(std::move(config));
    expect(server.start(), "start retained-frame endpoint");
    ExecutionOwnedDescriptor peer(connect_raw(path));
    expect(server.poll_for(0) && !server.peek_request(1), "accept without inventing a frame");
    const auto bytes = encode_execution_message(
        {{ExecutionMessageKind::kRequest, ExecutionOpcode::kSnapshot, 0, 1, 1, {2, 3}}, {}});
    expect(bytes && ::send(peer.get(), bytes->data(), 1, MSG_NOSIGNAL) == 1 && server.poll_for(0) &&
               !server.peek_request(1),
           "partial header cannot be borrowed");
    expect(::send(peer.get(), require_value(bytes).data() + 1, require_value(bytes).size() - 1,
                  MSG_NOSIGNAL) == static_cast<ssize_t>(require_value(bytes).size() - 1),
           "send the remaining authenticated frame bytes");
    for (const int error : {EAGAIN, EINTR}) {
        receive_error = error;
        receive_attempts = 0;
        const bool progressed = server.poll_for(0);
        receive_error = 0;
        expect(
            progressed && !server.peek_request(1) && receive_attempts == (error == EINTR ? 8 : 1),
            "partial input retains bounded would-block/interruption behavior");
    }
    expect(server.poll_for(0), "complete retained frame");
    const auto view = server.peek_request(1);
    expect(view.has_value(), "borrow complete frame only");
    const auto child = ::fork();
    expect(child >= 0, "fork borrowed frame ownership control");
    if (!child) _exit(!server.peek_request(1) && !server.take_request(require_value(view)) ? 0 : 1);
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) && !WEXITSTATUS(status),
           "fork child cannot inspect or consume parent frame");
    expect(inspect_retained_frame(server, true).has_value(), "parent frame survives fork");
    expect(::send(peer.get(), require_value(bytes).data(), require_value(bytes).size(),
                  MSG_NOSIGNAL) == static_cast<ssize_t>(require_value(bytes).size()) &&
               server.poll_for(0),
           "queue duplicate pending sequence");
    const auto duplicate = server.peek_request(1);
    expect(duplicate && !server.take_request(*duplicate) && !server.has_peer(1),
           "exact take retains ordinary duplicate-sequence poisoning");
    server.stop();
}
void first_positive_read_boundaries(const std::string& path) {
    const auto now = [] {
        return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                              std::chrono::steady_clock::now().time_since_epoch())
                                              .count());
    };
    for (const bool enabled : {false, true}) {
        ExecutionSocketConfig config;
        config.socket_path = path;
        config.should_record_receive_time = enabled;
        config.max_pending_requests = 2;
        ExecutionSocketServer server(std::move(config));
        expect(server.start(), "start first-positive-read fixture");
        ExecutionOwnedDescriptor peer(connect_raw(path));
        expect(server.poll_for(0), "accept first-positive-read peer");
        const auto bytes = encode_execution_message(
            {{ExecutionMessageKind::kRequest, ExecutionOpcode::kHello, 0, 0, 1, {}},
             std::vector<std::byte>(32, std::byte{17})});
        const auto next_bytes = encode_execution_message(
            {{ExecutionMessageKind::kRequest, ExecutionOpcode::kSnapshot, 0, 1, 2, {11, 13}},
             std::vector<std::byte>(32, std::byte{17})});
        expect(bytes && ::send(peer.get(), bytes->data(), 1, MSG_NOSIGNAL) == 1,
               "queue first byte without completing a header");
        for (const auto error : {EINTR, EAGAIN}) {
            receive_error = error;
            receive_attempts = 0;
            const bool polled = server.poll_for(0);
            receive_error = 0;
            expect(
                polled && !server.take_request() && receive_attempts == (error == EINTR ? 8U : 1U),
                "interrupted/would-block reads consume no frame and remain bounded");
        }
        const auto begin = now();
        expect(server.poll_for(0) && !server.take_request(), "retain first positive byte only");
        const auto first_end = now();
        expect(::send(peer.get(), bytes->data() + 1, kExecutionHeaderBytes - 1, MSG_NOSIGNAL) ==
                       static_cast<ssize_t>(kExecutionHeaderBytes - 1) &&
                   server.poll_for(0) && !server.take_request(),
               "later header fragment does not complete the payload");
        expect(::send(peer.get(), bytes->data() + kExecutionHeaderBytes,
                      bytes->size() - kExecutionHeaderBytes, MSG_NOSIGNAL) == 32 &&
                   server.poll_for(0),
               "complete the same frame after multiple positive reads");
        const auto complete_end = now();
        expect(next_bytes &&
                   ::send(peer.get(), next_bytes->data(), next_bytes->size(), MSG_NOSIGNAL) ==
                       static_cast<ssize_t>(next_bytes->size()) &&
                   server.poll_for(0),
               "next readable frame stays behind the retained frame");
        auto first = server.take_request();
        expect(first && first->message.payload.size() == 32 && first->message.header.sequence == 1,
               "take exact first frame");
        const auto second_begin = now();
        expect(server.poll_for(0), "only the next owner poll may read the next frame");
        auto second = server.take_request();
        const auto second_end = now();
        expect(second && second->message.payload == first->message.payload &&
                   second->message.header.sequence == 2,
               "next frame preserves all payload bytes");
        if (enabled) {
            expect(
                first->first_read_ns >= begin && first->first_read_ns <= first_end &&
                    first->frame_received_ns >= first->first_read_ns &&
                    first->frame_received_ns <= complete_end &&
                    second->first_read_ns >= second_begin &&
                    second->frame_received_ns >= second->first_read_ns &&
                    second->frame_received_ns <= second_end,
                "first-read stamp survives fragments, excludes failed reads and resets per frame");
        } else {
            expect(!first->first_read_ns && !first->frame_received_ns && !second->first_read_ns &&
                       !second->frame_received_ns,
                   "default-off transport publishes no diagnostic timestamps");
        }
        server.stop();
    }
}
std::size_t descriptor_count() {
    return static_cast<std::size_t>(
        std::distance(std::filesystem::directory_iterator("/proc/self/fd"),
                      std::filesystem::directory_iterator{}));
}
void rights_and_partial_frames_are_rejected(const std::string& path) {
    const auto baseline = descriptor_count();
    {
        ExecutionSocketConfig config;
        config.socket_path = path;
        config.io_timeout_ms = 10;
        ExecutionSocketServer server(std::move(config));
        expect(server.start(), "start ancillary rejection server");
        const auto descriptor = connect_raw(path);
        const auto passed = ::open("/dev/null", O_RDONLY | O_CLOEXEC);
        expect(passed >= 0 && server.poll_for(0), "accept raw rights fixture");
        const auto bytes = encode_execution_message(
            {{ExecutionMessageKind::kRequest, ExecutionOpcode::kHello, 0, 0, 1, {}}, {}});
        expect(bytes.has_value(), "encode rights fixture header");
        iovec data{const_cast<std::byte*>(require_value(bytes).data()),
                   require_value(bytes).size()};
        alignas(cmsghdr) std::array<std::byte, CMSG_SPACE(sizeof(int))> ancillary{};
        msghdr message{};
        message.msg_iov = &data;
        message.msg_iovlen = 1;
        message.msg_control = ancillary.data();
        message.msg_controllen = ancillary.size();
        auto* item = CMSG_FIRSTHDR(&message);
        item->cmsg_level = SOL_SOCKET;
        item->cmsg_type = SCM_RIGHTS;
        item->cmsg_len = CMSG_LEN(sizeof(int));
        std::memcpy(CMSG_DATA(item), &passed, sizeof(passed));
        expect(::sendmsg(descriptor, &message, MSG_NOSIGNAL) ==
                   static_cast<ssize_t>(require_value(bytes).size()),
               "send prohibited descriptor");
        expect(server.poll_for(1) && !server.take_request() && !server.has_peer(1),
               "descriptor-bearing request is rejected before application dispatch");
        expect(::close(passed) == 0 && ::close(descriptor) == 0, "close original fixture handles");
        const auto partial = connect_raw(path);
        expect(server.poll_for(0) &&
                   ::send(partial, require_value(bytes).data(), 1, MSG_NOSIGNAL) == 1 &&
                   server.poll_for(1),
               "receive incomplete header");
        expect(server.poll_for(30) && !server.take_request() && !server.has_peer(2),
               "partial frame expires without execution");
        expect(::close(partial) == 0, "close partial-frame fixture");
        server.stop();
    }
    expect(descriptor_count() == baseline, "rejected SCM_RIGHTS descriptors cannot leak");
}

void wrong_reply_incarnation_poisoned(const std::string& path) {
    ExecutionSocketConfig config;
    config.socket_path = path;
    ExecutionSocketServer server(std::move(config));
    expect(server.start(), "start reply incarnation fixture");
    const auto identity = current_process_identity();
    expect(identity.has_value(), "trusted server identity");
    auto client = std::async(std::launch::async, [&] {
        ExecutionSocketClient connection({path, *identity, 3000});
        expect(connection.open().has_value(), "bootstrap fixture");
        expect(!connection.exchange(ExecutionOpcode::kSnapshot) &&
                   !connection.exchange(ExecutionOpcode::kSnapshot) && !connection.open(),
               "wrong incarnation poisons channel permanently without reconnect/replay");
    });
    unsigned int requests{};
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready) {
        expect(server.poll_for(1), "progress reply incarnation fixture");
        if (auto request = server.take_request()) {
            ExecutionWireReply reply;
            reply.disposition = ExecutionWireDisposition::kComplete;
            auto encoded = encode_execution_reply(reply);
            expect(encoded.has_value(), "encode incarnation reply");
            auto header = request->message.header;
            header.kind = ExecutionMessageKind::kReply;
            header.session = 1;
            header.incarnation = requests == 0 ? std::array<std::uint64_t, 2>{3, 5}
                                               : std::array<std::uint64_t, 2>{3, 7};
            expect(server.reply(request->connection, {header, std::move(*encoded)}),
                   "queue deliberately stale incarnation reply");
            ++requests;
        }
        expect(std::chrono::steady_clock::now() < deadline, "bounded poisoned exchange");
    }
    client.get();
    expect(requests == 2, "no request replay after a mismatched reply");
    server.stop();
}
void bootstrap_ownership_validation(const std::string& path, bool wraps_continuation = false) {
    ExecutionSocketConfig config;
    config.socket_path = path;
    ExecutionSocketServer server(std::move(config));
    expect(server.start(), "start bootstrap metadata endpoint");
    const auto identity = current_process_identity();
    expect(identity.has_value(), "trusted bootstrap endpoint identity");
    struct Case {
        std::vector<std::byte> data;
        std::int32_t status{};
        bool has_reply{};
        std::uint32_t metadata_role{};
    };
    const auto record = [](ExecutionBootstrapOwnership value) {
        const auto encoded = encode_execution_bootstrap_ownership(value);
        expect(encoded.has_value(), "encode known bootstrap ownership");
        return std::vector<std::byte>(encoded->begin(), encoded->end());
    };
    std::vector<Case> cases{
        {{}, 0, true, 0},
        {record(ExecutionBootstrapOwnership::kRetained), 0, true, 0},
        {record(ExecutionBootstrapOwnership::kNone), 0, false, 0},
        {record(ExecutionBootstrapOwnership::kUnproven), 0, false, 0},
        {{std::byte{7}}, 0, false, 0},
        {record(ExecutionBootstrapOwnership::kNone), 2, true, 0},
        {record(ExecutionBootstrapOwnership::kRetained), 2, true, 0},
        {record(ExecutionBootstrapOwnership::kUnproven), 2, true, 0},
        {{}, 2, true, 0},
        {record(ExecutionBootstrapOwnership::kMetadataRetained), 0, false, 0},
        {record(ExecutionBootstrapOwnership::kMetadataRetained), 0, true, kExecutionLtMetadataRole},
        {record(ExecutionBootstrapOwnership::kRetained), 0, false, kExecutionLtMetadataRole},
        {record(ExecutionBootstrapOwnership::kNone), 0, false, kExecutionLtMetadataRole},
        {{}, 0, false, kExecutionLtMetadataRole},
        {record(ExecutionBootstrapOwnership::kNone), 2, true, kExecutionLtMetadataRole},
        {record(ExecutionBootstrapOwnership::kMetadataRetained), 0, true,
         kExecutionDnnMetadataRole},
        {record(ExecutionBootstrapOwnership::kRetained), 0, false, kExecutionDnnMetadataRole},
        {record(ExecutionBootstrapOwnership::kNone), 0, false, kExecutionDnnMetadataRole},
        {{}, 0, false, kExecutionDnnMetadataRole},
        {record(ExecutionBootstrapOwnership::kNone), 2, true, kExecutionDnnMetadataRole},
        {record(ExecutionBootstrapOwnership::kMetadataRetained), 0, true,
         kExecutionPointerMetadataRole},
        {record(ExecutionBootstrapOwnership::kRetained), 0, false, kExecutionPointerMetadataRole},
        {record(ExecutionBootstrapOwnership::kNone), 0, false, kExecutionPointerMetadataRole},
        {{}, 0, false, kExecutionPointerMetadataRole},
        {record(ExecutionBootstrapOwnership::kNone), 2, true, kExecutionPointerMetadataRole},
        {record(ExecutionBootstrapOwnership::kMetadataRetained), 0, true,
         kExecutionProcessMemoryRole},
        {record(ExecutionBootstrapOwnership::kRetained), 0, false, kExecutionProcessMemoryRole},
        {record(ExecutionBootstrapOwnership::kNone), 0, false, kExecutionProcessMemoryRole},
        {{}, 0, false, kExecutionProcessMemoryRole},
        {record(ExecutionBootstrapOwnership::kNone), 2, true, kExecutionProcessMemoryRole}};
    auto client = std::async(std::launch::async, [&] {
        for (const auto& test : cases) {
            ExecutionSocketClient connection({path, *identity, 3000});
            ExecutionPayloadWriter bootstrap;
            if (test.metadata_role)
                expect(bootstrap.write_u32(test.metadata_role) && bootstrap.write_u32(0),
                       "CPU metadata role");
            std::optional<ExecutionWireReply> result;
            if (wraps_continuation) {
                const auto submitted = connection.submit_open(
                    bootstrap.take(), ExecutionSourceContinuation{0, {17, 19}, 23, 1, 0, 0});
                expect(submitted.status == ExecutionSocketSubmitStatus::kAccepted &&
                           submitted.sequence == 1,
                       "wrapped nonblocking bootstrap exposes one exact accepted receipt");
                for (;;) {
                    result = connection.take_reply(submitted.sequence);
                    if (result) break;
                    if (!connection.poll_for(1)) {
                        result = connection.take_reply(submitted.sequence);
                        break;
                    }
                }
            } else {
                result = connection.open(bootstrap.take());
            }
            expect(
                result.has_value() == test.has_reply && (!result || result->status == test.status),
                "legacy/typed success and failure remain distinct from malformed ownership");
            expect(!connection.open(), "one instance can never replay bootstrap");
            if (test.status || !test.has_reply)
                expect(!connection.exchange(ExecutionOpcode::kSnapshot),
                       "failed or malformed bootstrap remains permanently poisoned");
        }
    });
    std::size_t served{};
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready) {
        expect(server.poll_for(1), "progress actual bootstrap metadata transport");
        if (const auto request = server.take_request()) {
            const auto effective = normalize_execution_request(request->message.header.operation,
                                                               request->message.payload);
            expect(served < cases.size() && effective &&
                       effective->operation == ExecutionOpcode::kHello &&
                       effective->continuation.has_value() == wraps_continuation,
                   "a new case uses a fresh authenticated connection, never a replay");
            ExecutionWireReply reply;
            reply.disposition = ExecutionWireDisposition::kComplete;
            reply.status = cases[served].status;
            reply.original_status = reply.status;
            reply.data = cases[served++].data;
            auto header = request->message.header;
            header.kind = ExecutionMessageKind::kReply;
            header.session = request->connection;
            header.incarnation = {31, 37};
            const auto payload = encode_execution_reply(reply);
            expect(payload && server.reply(request->connection, {header, *payload}),
                   "return owned optional bootstrap metadata");
        }
        expect(std::chrono::steady_clock::now() < deadline, "bounded metadata validation");
    }
    client.get();
    expect(served == cases.size(), "all ownership states and malformed/legacy cases exercised");
    server.stop();
}
struct ResponseContent {
    std::uint64_t id{};
    std::int32_t status{};
    std::size_t bytes{};
};
ExecutionWireMessage response_to(const ExecutionReceivedRequest& request, ResponseContent content) {
    ExecutionWireReply reply;
    reply.disposition = ExecutionWireDisposition::kComplete;
    reply.id = content.id;
    reply.status = content.status;
    reply.original_status = content.status;
    reply.data.assign(content.bytes, std::byte{53});
    const auto payload = encode_execution_reply(reply);
    expect(payload.has_value(), "encode multiplexed reply");
    auto header = request.message.header;
    header.kind = ExecutionMessageKind::kReply;
    header.session = 41;
    header.incarnation = {43, 47};
    return {header, require_value(payload)};
}
void source_continuation_round_trip(const std::string& path) {
    constexpr std::size_t limit = 128;
    ExecutionSocketConfig config;
    config.socket_path = path;
    config.max_payload_bytes = limit;
    config.max_pending_requests = 2;
    ExecutionSocketServer server(std::move(config));
    expect(server.start(), "start bounded SCN1 transport endpoint");
    const auto identity = current_process_identity();
    expect(identity.has_value(), "actual SCN1 endpoint identity");
    auto client = std::async(std::launch::async, [&] {
        ExecutionSocketClientConfig options{path, *identity, 3000, limit, 2};
        ExecutionSocketClient connection(std::move(options));
        ExecutionSourceContinuation scope{0, {17, 19}, 23, 1, 0, 0};
        ExecutionPayloadWriter bootstrap;
        expect(bootstrap.write_u32(kExecutionStagingOwnerRole) && bootstrap.write_u32(0),
               "unchanged STG1 bootstrap bytes");
        const auto hello = connection.submit_open(bootstrap.take(), scope);
        expect(hello.status == ExecutionSocketSubmitStatus::kAccepted && hello.sequence == 1 &&
                   connection.submit_open().status == ExecutionSocketSubmitStatus::kUnavailable,
               "cold wrapped bootstrap enqueues without replay or blocking");
        const auto take = [&](std::uint64_t sequence) {
            std::optional<ExecutionWireReply> reply;
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{4};
            while (!(reply = connection.take_reply(sequence))) {
                if (!connection.poll_for(1)) {
                    reply = connection.take_reply(sequence);
                    expect(reply.has_value(),
                           "a verified wrapped Close survives transport poisoning");
                    break;
                }
                expect(std::chrono::steady_clock::now() < deadline,
                       "bounded exact wrapped receipt, preserving other pending replies");
            }
            return *reply;
        };
        expect(!take(hello.sequence).status && connection.session(),
               "wrapped Hello preserves source session and checked bootstrap ownership");
        const auto frame = [&](ExecutionOpcode operation, std::span<const std::byte> bytes = {}) {
            const auto wrapped =
                encode_execution_source_continuation(scope, operation, bytes, limit);
            expect(wrapped.has_value(), "canonical client continuation");
            return *wrapped;
        };
        const std::vector<std::byte> full_payload(limit, std::byte{0x5a});
        scope.origin_source = 71;
        scope.ordinary_sequence = 2;
        const auto upload = connection.submit(ExecutionOpcode::kSourceContinuation,
                                              frame(ExecutionOpcode::kUpload, full_payload));
        scope.ordinary_sequence = 4;
        const auto close = frame(ExecutionOpcode::kClose);
        expect(upload.status == ExecutionSocketSubmitStatus::kAccepted && upload.sequence == 2 &&
                   connection.submit(ExecutionOpcode::kSourceContinuation, close).status ==
                       ExecutionSocketSubmitStatus::kFull,
               "effective Close cannot pass a retained receipt or seal the channel on Full");
        scope.ordinary_sequence = 3;
        const auto metadata = connection.submit(ExecutionOpcode::kSourceContinuation,
                                                frame(ExecutionOpcode::kSnapshot));
        expect(metadata.status == ExecutionSocketSubmitStatus::kAccepted &&
                   metadata.sequence == 3 && take(metadata.sequence).id == 3 &&
                   take(upload.sequence).id == 2,
               "outer-op replies correlate exact sequence despite reversed completion");
        ExecutionOwnedDescriptor descriptor(::open("/dev/null", O_RDONLY | O_CLOEXEC));
        expect(descriptor.get() >= 0, "create exact wrapped import descriptor");
        scope.ordinary_sequence = 4;
        const auto import = frame(ExecutionOpcode::kImportHostMemory);
        expect(connection.submit(ExecutionOpcode::kSourceContinuation, import).status ==
                       ExecutionSocketSubmitStatus::kInvalid &&
                   connection
                           .submit(ExecutionOpcode::kSourceContinuation,
                                   frame(ExecutionOpcode::kSnapshot), descriptor.get())
                           .status == ExecutionSocketSubmitStatus::kInvalid,
               "effective import descriptor presence rejects locally without sequence use");
        const auto imported =
            connection.submit(ExecutionOpcode::kSourceContinuation, import, descriptor.get());
        expect(imported.status == ExecutionSocketSubmitStatus::kAccepted &&
                   imported.sequence == 4 && take(imported.sequence).id == 4,
               "one descriptor follows normalized import, not opaque wrapper classification");
        scope.ordinary_sequence = 5;
        const auto closed =
            connection.submit(ExecutionOpcode::kSourceContinuation, frame(ExecutionOpcode::kClose));
        expect(closed.status == ExecutionSocketSubmitStatus::kAccepted && closed.sequence == 5 &&
                   connection.submit(ExecutionOpcode::kSnapshot).status ==
                       ExecutionSocketSubmitStatus::kUnavailable &&
                   !take(closed.sequence).status && !connection.poll_for(0),
               "effective Close seals before reply and permanently poisons afterward");
    });
    std::optional<ExecutionReceivedRequest> held;
    unsigned int requests{};
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{6};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready) {
        expect(server.poll_for(1), "nonblocking wrapped endpoint progress");
        if (auto request = inspect_retained_frame(server, false)) {
            ++requests;
            const auto view = normalize_execution_request(request->message.header.operation,
                                                          request->message.payload, limit);
            expect(
                view && view->continuation && request->peer == *identity &&
                    request->message.header.operation == ExecutionOpcode::kSourceContinuation &&
                    view->continuation->ordinary_sequence == request->message.header.sequence &&
                    execution_requires_descriptor(view->operation) ==
                        (request->descriptor.get() >= 0),
                "server preserves outer identity and authenticates exact effective FD semantics");
            if (view->operation == ExecutionOpcode::kUpload) {
                expect(!held && view->payload.size() == limit &&
                           std::ranges::all_of(view->payload,
                                               [](auto byte) { return byte == std::byte{0x5a}; }),
                       "full old payload capacity reaches normalized application bytes");
                held = std::move(*request);
            } else {
                auto response = response_to(*request, {request->message.header.sequence});
                if (view->operation == ExecutionOpcode::kHello) {
                    ExecutionWireReply reply;
                    reply.disposition = ExecutionWireDisposition::kComplete;
                    const auto ownership = encode_execution_bootstrap_ownership(
                        ExecutionBootstrapOwnership::kRetained);
                    expect(ownership.has_value(), "encode exact wrapped bootstrap ownership");
                    reply.data.assign(ownership->begin(), ownership->end());
                    response.payload = *encode_execution_reply(reply, limit);
                }
                expect(server.reply(request->connection, response),
                       "reply using original outer opcode");
                if (view->operation == ExecutionOpcode::kSnapshot) {
                    expect(held && server.reply(held->connection, response_to(*held, {2})),
                           "independent exact metadata reply precedes held ordinary reply");
                    held.reset();
                }
            }
        }
        expect(std::chrono::steady_clock::now() < deadline, "bounded wrapped transport fixture");
    }
    client.get();
    expect(requests == 5 && !held,
           "invalid/full wrappers never become requests or consume wire sequence");
    server.stop();
}
void closing_host_function_controls(const std::string& path, bool wraps_close) {
    constexpr std::size_t limit = 128;
    constexpr std::uint64_t token = 53;
    ExecutionSocketConfig config;
    config.socket_path = path;
    config.max_payload_bytes = limit;
    config.max_pending_requests = 2;
    ExecutionSocketServer server(std::move(config));
    expect(server.start(), "start callback-closing transport endpoint");
    const auto identity = current_process_identity();
    expect(identity.has_value(), "actual callback-closing endpoint identity");
    auto client = std::async(std::launch::async, [&] {
        ExecutionSocketClient connection({path, *identity, 3000, limit, 2});
        ExecutionPayloadWriter stream, query, finish;
        expect(stream.write_u64(0) && query.write_u64(token) && finish.write_u64(token) &&
                   finish.write_u32(1),
               "encode exact callback stream, token and return controls");
        const auto stream_bytes = stream.take();
        const auto query_bytes = query.take();
        const auto finish_bytes = finish.take();
        const auto wrap = [&](ExecutionOpcode operation, std::span<const std::byte> bytes = {}) {
            const auto encoded = encode_execution_source_continuation({0, {17, 19}, 23, 3, 71, 0},
                                                                      operation, bytes, limit);
            expect(encoded.has_value(), "encode valid closing continuation");
            return *encoded;
        };
        const auto close_operation =
            wraps_close ? ExecutionOpcode::kSourceContinuation : ExecutionOpcode::kClose;
        const auto close_bytes =
            wraps_close ? wrap(ExecutionOpcode::kClose) : std::vector<std::byte>{};
        expect(connection.submit(ExecutionOpcode::kQueryHostFunction, query_bytes).status ==
                       ExecutionSocketSubmitStatus::kUnavailable &&
                   connection.open().has_value(),
               "callback helpers require an authenticated opened session");
        const auto enqueued =
            connection.submit(ExecutionOpcode::kEnqueueHostFunction, stream_bytes);
        expect(enqueued.status == ExecutionSocketSubmitStatus::kAccepted &&
                   enqueued.sequence == 2 &&
                   connection.submit(close_operation, close_bytes).status ==
                       ExecutionSocketSubmitStatus::kFull,
               "Close still rejects any retained receipt even with spare window capacity");
        const auto take = [&](std::uint64_t sequence) {
            std::optional<ExecutionWireReply> reply;
            while (!(reply = connection.take_reply(sequence))) {
                if (!connection.poll_for(1)) {
                    reply = connection.take_reply(sequence);
                    expect(reply.has_value(), "only a verified closing result survives poison");
                    break;
                }
            }
            return *reply;
        };
        expect(take(enqueued.sequence).id == token, "accepted callback receipt retires normally");
        const auto closed = connection.submit(close_operation, close_bytes);
        expect(closed.status == ExecutionSocketSubmitStatus::kAccepted && closed.sequence == 3 &&
                   connection.submit(ExecutionOpcode::kSnapshot).status ==
                       ExecutionSocketSubmitStatus::kUnavailable &&
                   connection.submit(ExecutionOpcode::kEnqueueHostFunction, stream_bytes).status ==
                       ExecutionSocketSubmitStatus::kUnavailable &&
                   connection
                           .submit(ExecutionOpcode::kSourceContinuation,
                                   wrap(ExecutionOpcode::kSnapshot))
                           .status == ExecutionSocketSubmitStatus::kUnavailable &&
                   connection
                           .submit(ExecutionOpcode::kSourceContinuation,
                                   wrap(ExecutionOpcode::kEnqueueHostFunction, stream_bytes))
                           .status == ExecutionSocketSubmitStatus::kUnavailable &&
                   connection.submit(close_operation, close_bytes).status ==
                       ExecutionSocketSubmitStatus::kUnavailable,
               "pending Close seals ordinary, enqueue, continuation and duplicate Close work");
        ExecutionOwnedDescriptor descriptor(::open("/dev/null", O_RDONLY | O_CLOEXEC));
        expect(descriptor.get() >= 0, "create invalid callback descriptor input");
        const std::vector<std::byte> oversized(limit + 1);
        for (const auto operation :
             {ExecutionOpcode::kQueryHostFunction, ExecutionOpcode::kFinishHostFunction}) {
            const auto& bytes =
                operation == ExecutionOpcode::kQueryHostFunction ? query_bytes : finish_bytes;
            auto wrapped = wrap(ExecutionOpcode::kEnqueueHostFunction, bytes);
            wrapped[56] = std::byte{static_cast<unsigned char>(operation)};
            expect(connection.submit(operation, bytes, descriptor.get()).status ==
                           ExecutionSocketSubmitStatus::kInvalid &&
                       connection.submit(operation, bytes, -2).status ==
                           ExecutionSocketSubmitStatus::kInvalid &&
                       connection.submit(operation, oversized).status ==
                           ExecutionSocketSubmitStatus::kInvalid &&
                       connection.submit(ExecutionOpcode::kSourceContinuation, wrapped).status ==
                           ExecutionSocketSubmitStatus::kInvalid,
                   "closing helpers retain descriptor, payload and direct-only validation");
        }
        const auto queried = connection.submit(ExecutionOpcode::kQueryHostFunction, query_bytes);
        expect(queried.status == ExecutionSocketSubmitStatus::kAccepted && queried.sequence == 4 &&
                   connection.submit(ExecutionOpcode::kFinishHostFunction, finish_bytes).status ==
                       ExecutionSocketSubmitStatus::kFull &&
                   !connection.exchange(ExecutionOpcode::kFinishHostFunction, finish_bytes),
               "direct query drains behind Close without bypassing capacity or exchange exclusion");
        expect(take(queried.sequence).id == token && !connection.take_reply(closed.sequence),
               "query permission arrives while the original Close remains pending");
        const auto finished = connection.submit(ExecutionOpcode::kFinishHostFunction, finish_bytes);
        expect(finished.status == ExecutionSocketSubmitStatus::kAccepted &&
                   finished.sequence == 5 &&
                   connection.submit(ExecutionOpcode::kSnapshot).status ==
                       ExecutionSocketSubmitStatus::kUnavailable,
               "direct finish consumes the freed slot without reopening ordinary admission");
        while (connection.poll_for(1)) {
        }
        const auto finish_reply = connection.take_reply(finished.sequence);
        const auto close_reply = connection.take_reply(closed.sequence);
        expect(finish_reply && finish_reply->id == token && !finish_reply->status && close_reply &&
                   !close_reply->status && !connection.poll_for(0) && !connection.open() &&
                   connection.submit(ExecutionOpcode::kQueryHostFunction, query_bytes).status ==
                       ExecutionSocketSubmitStatus::kUnavailable &&
                   connection.submit(ExecutionOpcode::kFinishHostFunction, finish_bytes).status ==
                       ExecutionSocketSubmitStatus::kUnavailable,
               "Close still poisons permanently while preserving already verified helper replies");
    });
    std::optional<ExecutionReceivedRequest> held_close;
    unsigned int requests{};
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready) {
        expect(server.poll_for(1), "progress closing callback endpoint");
        if (auto request = server.take_request()) {
            ++requests;
            const auto view = normalize_execution_request(request->message.header.operation,
                                                          request->message.payload, limit);
            expect(view && request->peer == *identity && request->descriptor.get() < 0 &&
                       request->message.header.sequence == requests &&
                       (requests == 1 || (request->message.header.session == 41 &&
                                          request->message.header.incarnation ==
                                              std::array<std::uint64_t, 2>{43, 47})),
                   "every accepted control retains exact authentication, session and sequence");
            const auto operation = view->operation;
            if (operation == ExecutionOpcode::kClose) {
                expect(
                    requests == 3 && !held_close && view->continuation.has_value() == wraps_close,
                    "hold the exact original direct or wrapped Close");
                held_close = std::move(*request);
            } else {
                expect((requests == 1 && operation == ExecutionOpcode::kHello) ||
                           (requests == 2 && operation == ExecutionOpcode::kEnqueueHostFunction) ||
                           (requests == 4 && operation == ExecutionOpcode::kQueryHostFunction) ||
                           (requests == 5 && operation == ExecutionOpcode::kFinishHostFunction),
                       "only existing callback controls arrive after pending Close");
                if (requests >= 4) {
                    ExecutionPayloadReader payload(view->payload);
                    std::uint64_t actual_token{};
                    std::uint32_t returned{};
                    expect(held_close && !view->continuation && payload.read_u64(actual_token) &&
                               actual_token == token &&
                               (requests == 4 || (payload.read_u32(returned) && returned == 1)) &&
                               payload.empty(),
                           "callback controls retain all original payload bytes");
                }
                expect(server.reply(request->connection,
                                    response_to(*request, {requests == 1 ? 0 : token})),
                       "return actual helper receipt before close completion");
                if (operation == ExecutionOpcode::kFinishHostFunction) {
                    expect(held_close &&
                               server.reply(held_close->connection, response_to(*held_close, {})),
                           "complete Close only after callback finish receipt");
                    held_close.reset();
                }
            }
        }
        expect(std::chrono::steady_clock::now() < deadline, "bounded closing callback controls");
    }
    client.get();
    expect(requests == 5 && !held_close, "rejections consume no sequence or remote work");
    server.stop();
}
void multiplexed_owned_round_trip(const std::string& path) {
    ExecutionSocketConfig config;
    config.socket_path = path;
    config.max_pending_requests = 3;
    ExecutionSocketServer server(std::move(config));
    expect(server.start(), "start bounded multiplexed server");
    const auto identity = current_process_identity();
    expect(identity.has_value(), "multiplexed trusted identity");
    constexpr auto input_bytes = std::size_t{2} * 1024 * 1024;
    auto client = std::async(std::launch::async, [&] {
        ExecutionSocketClientConfig options{path, *identity, 3000};
        options.max_pending_requests = 3;
        options.max_payload_bytes = input_bytes + 1024;
        ExecutionSocketClient connection(std::move(options));
        expect(connection.open().has_value(), "multiplexed bootstrap");
        std::vector<std::byte> input(input_bytes + 2048, std::byte{37});
        expect(connection.submit(ExecutionOpcode::kUpload, input).status ==
                   ExecutionSocketSubmitStatus::kInvalid,
               "oversized input is rejected before ownership or sequence consumption");
        input.resize(input_bytes);
        const auto first = connection.submit(ExecutionOpcode::kUpload, input);
        std::ranges::fill(input, std::byte{99});
        const auto second = connection.submit(ExecutionOpcode::kSnapshot);
        const auto third = connection.submit(ExecutionOpcode::kQueryFrontier);
        expect(first.status == ExecutionSocketSubmitStatus::kAccepted && first.sequence == 2 &&
                   second.status == ExecutionSocketSubmitStatus::kAccepted &&
                   second.sequence == 3 && third.status == ExecutionSocketSubmitStatus::kAccepted &&
                   third.sequence == 4,
               "owned submissions receive monotonic identities before any I/O or reply");
        expect(!connection.take_reply(first.sequence) &&
                   !connection.exchange(ExecutionOpcode::kSnapshot) &&
                   connection.submit(ExecutionOpcode::kSnapshot).status ==
                       ExecutionSocketSubmitStatus::kFull &&
                   connection.submit(ExecutionOpcode::kClose).status ==
                       ExecutionSocketSubmitStatus::kFull,
               "no fabricated success, overlapping blocking exchange or full-window submission");
        const auto child = ::fork();
        expect(child >= 0, "fork with queued multiplexed requests");
        if (!child) {
            const bool rejected = connection.submit(ExecutionOpcode::kSnapshot).status ==
                                      ExecutionSocketSubmitStatus::kUnavailable &&
                                  !connection.poll_for(0) && !connection.take_reply(first.sequence);
            connection.~ExecutionSocketClient();
            _exit(rejected ? 0 : 1);
        }
        int child_status{};
        expect(::waitpid(child, &child_status, 0) == child && WIFEXITED(child_status) &&
                   WEXITSTATUS(child_status) == 0,
               "child cannot pump, consume or close parent requests");
        std::vector<std::uint64_t> returned;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{4};
        while (returned.size() != 3) {
            expect(connection.poll_for(1), "pump partial owned request and reply frames");
            for (const auto sequence : {first.sequence, second.sequence, third.sequence}) {
                if (auto reply = connection.take_reply(sequence)) {
                    expect(reply->id == sequence + 100 &&
                               reply->original_status == (sequence == 3 ? 701 : 0) &&
                               reply->status == (sequence == 3 ? 701 : 0),
                           "real remote statuses route to exact original request");
                    expect(reply->data.size() == (sequence == 2 ? input_bytes : 0) &&
                               std::ranges::all_of(reply->data,
                                                   [](auto byte) { return byte == std::byte{53}; }),
                           "partial large response frames preserve all owned bytes");
                    returned.push_back(sequence);
                }
            }
            expect(std::chrono::steady_clock::now() < deadline,
                   "first deferred response cannot exclude independent later requests");
        }
        expect(returned == std::vector<std::uint64_t>{4, 3, 2},
               "out-of-order replies are not mistaken for queue order");
        const auto closed = connection.submit(ExecutionOpcode::kClose);
        expect(closed.status == ExecutionSocketSubmitStatus::kAccepted && closed.sequence == 5 &&
                   connection.submit(ExecutionOpcode::kSnapshot).status ==
                       ExecutionSocketSubmitStatus::kUnavailable,
               "full-window rejection consumed no sequence and close forbids new work");
        std::optional<ExecutionWireReply> reply;
        while (!(reply = connection.take_reply(closed.sequence)))
            expect(connection.poll_for(1), "receive actual close result");
        expect(reply->status == 0 && !connection.poll_for(0) && !connection.open(),
               "closed channel never reconnects");
    });
    std::vector<ExecutionReceivedRequest> held;
    unsigned int requests{};
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready) {
        expect(server.poll_for(1), "progress multiplexed server");
        if (auto request = server.take_request()) {
            expect(request->peer == *identity, "every multiplexed request retains credentials");
            ++requests;
            if (request->message.header.operation == ExecutionOpcode::kHello ||
                request->message.header.operation == ExecutionOpcode::kClose) {
                expect(server.reply(request->connection, response_to(*request, {})),
                       "queue bootstrap/close result");
            } else {
                held.push_back(std::move(*request));
                if (held.size() == 3) {
                    expect(held.front().message.payload.size() == input_bytes &&
                               std::ranges::all_of(held.front().message.payload,
                                                   [](auto byte) { return byte == std::byte{37}; }),
                           "submission owns bytes before caller overwrites its large buffer");
                    for (auto position = held.rbegin(); position != held.rend(); ++position) {
                        const auto sequence = position->message.header.sequence;
                        const auto reply =
                            response_to(*position, {sequence + 100, sequence == 3 ? 701 : 0,
                                                    sequence == 2 ? input_bytes : 0});
                        expect(server.reply(position->connection, reply) &&
                                   !server.reply(position->connection, reply),
                               "queue multiple reversed replies; reject duplicate server reply");
                    }
                }
            }
        }
        expect(std::chrono::steady_clock::now() < deadline, "bounded multiplexed fixture");
    }
    client.get();
    expect(requests == 5, "no replay, full-window request or forked submission reached server");
    server.stop();
}

void server_window_is_bounded(const std::string& path) {
    ExecutionSocketConfig config;
    config.socket_path = path;
    ExecutionSocketServer server(std::move(config));
    expect(server.start(), "start default single-request server");
    const auto identity = current_process_identity();
    expect(identity.has_value(), "window endpoint identity");
    std::atomic<bool> has_pumped{};
    auto client = std::async(std::launch::async, [&] {
        ExecutionSocketClient connection({path, *identity, 3000});
        expect(connection.open().has_value(), "window bootstrap");
        const auto first = connection.submit(ExecutionOpcode::kSnapshot);
        const auto second = connection.submit(ExecutionOpcode::kDrain);
        expect(first.status == ExecutionSocketSubmitStatus::kAccepted &&
                   second.status == ExecutionSocketSubmitStatus::kAccepted &&
                   connection.poll_for(0) && connection.poll_for(0),
               "queue both requests without expanding server admission");
        has_pumped = true;
        bool has_first{}, has_second{};
        while (!has_first || !has_second) {
            expect(connection.poll_for(1), "bounded-window replies make progress");
            if (const auto reply = connection.take_reply(first.sequence))
                has_first = reply->id == 102;
            if (const auto reply = connection.take_reply(second.sequence))
                has_second = reply->id == 103;
        }
    });
    unsigned int requests{};
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{4};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready) {
        expect(server.poll_for(1), "progress bounded server admission");
        if (auto request = server.take_request()) {
            ++requests;
            const auto sequence = request->message.header.sequence;
            if (sequence == 2) {
                while (!has_pumped) {
                    expect(server.poll_for(1) && !server.take_request(),
                           "pending request prevents second admission");
                    expect(std::chrono::steady_clock::now() < deadline,
                           "bounded queued-frame barrier");
                }
                for (unsigned int step = 0; step != 4; ++step)
                    expect(server.poll_for(1) && !server.take_request(),
                           "server does not read ahead past its configured request window");
            }
            const auto reply = response_to(*request, {sequence + 100});
            auto unknown = reply;
            unknown.header.sequence = 999;
            expect(!server.reply(request->connection, unknown),
                   "uncorrelated replies cannot release request-window capacity");
            std::size_t blocked_attempts{};
            bool is_queued{};
            {
                // Hold only this owner-thread reply's actual eager send. The
                // concurrent client and subsequent polling remain unmodified.
                const SendBackpressureScope scope(blocked_attempts);
                is_queued = server.reply(request->connection, reply);
            }
            expect(is_queued && blocked_attempts == 1 &&
                       server.response_pending(request->connection) && !server.take_request(),
                   "only a correlated reply is queued and retains capacity until delivery");
        }
        expect(std::chrono::steady_clock::now() < deadline, "bounded window fixture");
    }
    client.get();
    expect(requests == 3, "both requests complete once under default serial server policy");
    server.stop();
}
void invalid_windows_are_rejected() {
    const auto identity = current_process_identity();
    expect(identity.has_value(), "bounds fixture identity");
    for (const auto count : {std::size_t{0}, std::size_t{65}}) {
        bool server_rejected{}, client_rejected{};
        try {
            ExecutionSocketConfig config;
            config.max_pending_requests = count;
            ExecutionSocketServer server(std::move(config));
        } catch (const std::invalid_argument&) {
            server_rejected = true;
        }
        try {
            ExecutionSocketClientConfig config{"/unused.sock", require_value(identity), 1000};
            config.max_pending_requests = count;
            ExecutionSocketClient client(std::move(config));
        } catch (const std::invalid_argument&) {
            client_rejected = true;
        }
        expect(server_rejected && client_rejected,
               "zero and unbounded receipt windows reject before creating any endpoint");
    }
}

enum class MultiplexFailure : std::uint8_t { kTimeout, kDisconnect, kWrongIncarnation };
void multiplexed_failure(const std::string& path, MultiplexFailure failure) {
    ExecutionSocketConfig config;
    config.socket_path = path;
    config.max_pending_requests = 2;
    ExecutionSocketServer server(std::move(config));
    expect(server.start(), "start multiplexed failure endpoint");
    const auto identity = current_process_identity();
    expect(identity.has_value(), "failure endpoint identity");
    std::atomic<bool> has_reaped{};
    auto client = std::async(std::launch::async, [&] {
        ExecutionSocketClientConfig options{path, *identity, 500};
        options.max_pending_requests = 2;
        ExecutionSocketClient connection(std::move(options));
        expect(connection.open().has_value(), "failure bootstrap");
        const auto first = connection.submit(ExecutionOpcode::kSnapshot);
        const auto second = connection.submit(ExecutionOpcode::kDrain);
        expect(first.status == ExecutionSocketSubmitStatus::kAccepted &&
                   second.status == ExecutionSocketSubmitStatus::kAccepted,
               "own both pending requests");
        bool has_failed{};
        std::optional<ExecutionWireReply> known;
        while (!has_failed) {
            has_failed = !connection.poll_for(1);
            if (failure == MultiplexFailure::kDisconnect && !known) {
                known = connection.take_reply(second.sequence);
                if (known) has_reaped = true;
            }
        }
        expect(!connection.take_reply(first.sequence),
               "unresolved request cannot become success after transport failure");
        if (!known) known = connection.take_reply(second.sequence);
        expect(known && known->id == 103 && known->original_status == 701,
               "verified original result survives later unrelated uncertainty");
        expect(!connection.poll_for(0) && !connection.open() &&
                   connection.submit(ExecutionOpcode::kSnapshot).status ==
                       ExecutionSocketSubmitStatus::kUnavailable &&
                   !connection.exchange(ExecutionOpcode::kSnapshot),
               "all uncertain paths permanently stop new issue without reconnect/replay");
        const auto expired = connection.expired_request();
        if (failure == MultiplexFailure::kTimeout)
            expect(expired && expired->sequence == first.sequence &&
                       expired->operation == ExecutionOpcode::kSnapshot &&
                       expired->effective_operation == ExecutionOpcode::kSnapshot &&
                       expired->session == 41 && expired->timeout_ms == 500,
                   "retain the unresolved timeout origin, never the verified later reply");
        else
            expect(!expired, "disconnect and protocol failures cannot invent timeout provenance");
    });
    std::vector<ExecutionReceivedRequest> held;
    bool has_failed{};
    unsigned int requests{};
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{4};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready) {
        expect(server.poll_for(1), "progress failure fixture");
        if (auto request = server.take_request()) {
            ++requests;
            if (request->message.header.operation == ExecutionOpcode::kHello) {
                expect(server.reply(request->connection, response_to(*request, {})),
                       "bootstrap failure fixture");
            } else {
                held.push_back(std::move(*request));
                if (held.size() == 2)
                    expect(
                        server.reply(held.back().connection, response_to(held.back(), {103, 701})),
                        "return one verified original failure before uncertainty");
            }
        }
        if (held.size() == 2 && !has_failed && !server.response_pending(held.back().connection) &&
            (failure != MultiplexFailure::kDisconnect || has_reaped.load())) {
            has_failed = true;
            if (failure == MultiplexFailure::kDisconnect) {
                server.close_peer(held.front().connection);
            } else if (failure != MultiplexFailure::kTimeout) {
                auto invalid = response_to(held.front(), {102});
                invalid.header.incarnation[0] = 999;
                expect(server.reply(held.front().connection, invalid),
                       "inject stale response after an independently verified result");
            }
        }
        expect(std::chrono::steady_clock::now() < deadline, "bounded failure quarantine");
    }
    client.get();
    expect(requests == 3 && has_failed, "uncertain pending requests are never replayed");
    server.stop();
}
struct RawEndpoint {
    std::string path;
    int listener{-1}, peer{-1};
    bool has_bound{};
    explicit RawEndpoint(std::string value) : path(std::move(value)) {}
    ~RawEndpoint() {
        for (const auto descriptor : {peer, listener})
            if (descriptor >= 0 && ::close(descriptor) != 0)
                std::cerr << "Raw endpoint descriptor cleanup failed\n";
        if (has_bound && ::unlink(path.c_str()) != 0)
            std::cerr << "Raw endpoint socket cleanup failed\n";
    }
};
void raw_read(int descriptor, std::span<std::byte> bytes) {
    while (!bytes.empty()) {
        pollfd waiting{descriptor, POLLIN, 0};
        expect(::poll(&waiting, 1, 1000) > 0, "bounded raw request receive");
        const auto count = ::recv(descriptor, bytes.data(), bytes.size(), MSG_DONTWAIT);
        expect(count > 0, "read authenticated client's original bytes");
        bytes = bytes.subspan(static_cast<std::size_t>(count));
    }
}
ExecutionReceivedRequest raw_request(int descriptor) {
    std::array<std::byte, kExecutionHeaderBytes> bytes{};
    raw_read(descriptor, bytes);
    const auto header = decode_execution_header(bytes);
    expect(header.has_value(), "decode actual client header");
    ExecutionReceivedRequest result;
    result.message.header = require_value(header);
    result.message.payload.resize(require_value(header).payload_bytes);
    raw_read(descriptor, result.message.payload);
    return result;
}
void raw_send(int descriptor, const ExecutionWireMessage& response) {
    const auto encoded = encode_execution_message(response);
    expect(encoded.has_value(), "encode intentionally miscorrelated server frame");
    std::span<const std::byte> bytes(require_value(encoded));
    while (!bytes.empty()) {
        pollfd waiting{descriptor, POLLOUT, 0};
        expect(::poll(&waiting, 1, 1000) > 0, "bounded raw response send");
        const auto count =
            ::send(descriptor, bytes.data(), bytes.size(), MSG_DONTWAIT | MSG_NOSIGNAL);
        expect(count > 0, "send raw response from actual authenticated server process");
        bytes = bytes.subspan(static_cast<std::size_t>(count));
    }
}
void owned_readiness_snapshots(const std::string& path) {
    const auto baseline = descriptor_count();
    {
        RawEndpoint endpoint(path);
        endpoint.listener = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
        expect(endpoint.listener >= 0, "create readiness endpoint");
        sockaddr_un address{};
        address.sun_family = AF_UNIX;
        expect(path.size() < sizeof(address.sun_path), "bounded readiness endpoint path");
        std::copy(path.begin(), path.end(), address.sun_path);
        expect(::bind(endpoint.listener, reinterpret_cast<const sockaddr*>(&address),
                      sizeof(address)) == 0,
               "bind exact readiness endpoint");
        endpoint.has_bound = true;
        expect(::chmod(path.c_str(), 0600) == 0 && ::listen(endpoint.listener, 1) == 0,
               "private readiness endpoint");
        const auto identity = current_process_identity();
        expect(identity.has_value(), "readiness endpoint identity");
        auto connection = std::make_unique<ExecutionSocketClient>(
            ExecutionSocketClientConfig{path, require_value(identity), 1000});
        bool fully_flushed = true;
        expect(
            connection->submit_observed(ExecutionOpcode::kSnapshot, {}, -1, nullptr, &fully_flushed)
                        .status == ExecutionSocketSubmitStatus::kUnavailable &&
                !fully_flushed,
            "cold submission cannot claim delivery or connect");
        expect(!connection->readiness(), "cold readiness never connects");
        const auto opened = connection->submit_open();
        expect(opened.status == ExecutionSocketSubmitStatus::kAccepted, "submit exact bootstrap");
        endpoint.peer = ::accept4(endpoint.listener, nullptr, nullptr, SOCK_CLOEXEC);
        expect(endpoint.peer >= 0, "accept readiness client");
        raw_send(endpoint.peer, response_to(raw_request(endpoint.peer), {}));
        expect(connection->poll_for(100) && connection->take_reply(opened.sequence).has_value(),
               "authenticate before taking readiness ownership");
        const auto before_hint = descriptor_count();
        auto hint = connection->readiness();
        expect(hint && descriptor_count() == before_hint + 1 && !hint->wait_for_ns(0) &&
                   !hint->wait_for_ns(50001) && hint->wait_for_ns(50000),
               "one owned duplicate; bounded timeout is not a reply");
        std::size_t blocked{};
        ExecutionSocketSubmission submitted;
        {
            SendBackpressureScope backpressure(blocked);
            fully_flushed = true;
            submitted = connection->submit_observed(ExecutionOpcode::kSnapshot, {}, -1, nullptr,
                                                    &fully_flushed);
            const auto writable = connection->readiness();
            expect(!fully_flushed && blocked && writable && writable->wait_for_ns(50000) &&
                       !connection->take_reply(submitted.sequence),
                   "writable output remains owned until real transport progress");
        }
        expect(
            submitted.status == ExecutionSocketSubmitStatus::kAccepted && connection->poll_for(0),
            "original transport flushes the unsent request");
        raw_send(endpoint.peer, response_to(raw_request(endpoint.peer), {37}));
        expect(
            require_value(hint).wait_for_ns(50000) && !connection->take_reply(submitted.sequence),
            "readiness consumes no response bytes or receipt");
        expect(connection->poll_for(100), "normal authentication still receives the response");
        const auto reply = connection->take_reply(submitted.sequence);
        expect(reply && reply->id == 37, "only exact validated reply completes the request");
        const auto child = ::fork();
        expect(child >= 0, "fork with owned readiness");
        if (child == 0) {
            fully_flushed = true;
            const auto rejected = connection->submit_observed(ExecutionOpcode::kSnapshot, {}, -1,
                                                              nullptr, &fully_flushed);
            _exit(!require_value(hint).wait_for_ns(50000) && !connection->readiness() &&
                          !fully_flushed &&
                          rejected.status == ExecutionSocketSubmitStatus::kUnavailable
                      ? 0
                      : 1);
        }
        int status{};
        expect(
            ::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
            "fork child cannot wait on or snapshot parent transport");
        auto moved = std::move(require_value(hint));
        expect(!require_value(hint).wait_for_ns(50000) && moved.wait_for_ns(50000),
               "move transfers one owned descriptor without reusing the old handle");
        const auto pending =
            connection->submit_observed(ExecutionOpcode::kDrain, {}, -1, nullptr, &fully_flushed);
        expect(pending.status == ExecutionSocketSubmitStatus::kAccepted && fully_flushed,
               "small request is fully flushed at submit return, not execution complete");
        auto malformed = response_to(raw_request(endpoint.peer), {});
        ++malformed.header.sequence;
        raw_send(endpoint.peer, malformed);
        expect(!connection->poll_for(100) && !connection->readiness() &&
                   !connection->take_reply(pending.sequence),
               "wrong exact sequence poisons; hint cannot mint or revive a receipt");
        fully_flushed = true;
        expect(
            connection->submit_observed(ExecutionOpcode::kSnapshot, {}, -1, nullptr, &fully_flushed)
                        .status == ExecutionSocketSubmitStatus::kUnavailable &&
                !fully_flushed,
            "poisoned empty queue cannot claim successful delivery");
        connection.reset();
        ExecutionOwnedDescriptor reused{::open("/dev/null", O_RDONLY | O_CLOEXEC)};
        expect(reused.get() >= 0 && moved.wait_for_ns(50000),
               "snapshot remains independently owned after client destruction and new fd creation");
        expect(::close(endpoint.peer) == 0, "retire old raw peer before hard-send control");
        endpoint.peer = -1;
        ExecutionSocketClient failed({path, require_value(identity), 1000});
        const auto retry_open = failed.submit_open();
        expect(retry_open.status == ExecutionSocketSubmitStatus::kAccepted,
               "independent hard-send fixture owns its bootstrap");
        endpoint.peer = ::accept4(endpoint.listener, nullptr, nullptr, SOCK_CLOEXEC);
        expect(endpoint.peer >= 0, "accept independent hard-send control");
        raw_send(endpoint.peer, response_to(raw_request(endpoint.peer), {}));
        expect(failed.poll_for(100) && failed.take_reply(retry_open.sequence).has_value(),
               "authenticate before forcing one hard send failure");
        fully_flushed = true;
        send_error = EPIPE;
        const auto uncertain =
            failed.submit_observed(ExecutionOpcode::kSnapshot, {}, -1, nullptr, &fully_flushed);
        send_error = 0;
        expect(uncertain.status == ExecutionSocketSubmitStatus::kAccepted &&
                   uncertain.sequence == 2 && !fully_flushed && !failed.poll_for(0),
               "hard send preserves committed acceptance but poisoned empty queue is not delivery");
    }
    expect(descriptor_count() == baseline, "readiness duplicates retire without leaking handles");
}
void raw_send_rights(int descriptor, std::span<const std::byte> bytes,
                     std::span<const int> passed) {
    expect(!passed.empty() && passed.size() <= 12, "bounded raw ancillary fixture");
    alignas(cmsghdr) std::array<std::byte, CMSG_SPACE(sizeof(int) * 12)> ancillary{};
    iovec data{const_cast<std::byte*>(bytes.data()), bytes.size()};
    msghdr message{};
    message.msg_iov = &data;
    message.msg_iovlen = 1;
    message.msg_control = ancillary.data();
    message.msg_controllen = CMSG_SPACE(passed.size_bytes());
    auto* item = CMSG_FIRSTHDR(&message);
    item->cmsg_level = SOL_SOCKET;
    item->cmsg_type = SCM_RIGHTS;
    item->cmsg_len = CMSG_LEN(passed.size_bytes());
    std::memcpy(CMSG_DATA(item), passed.data(), passed.size_bytes());
    expect(::sendmsg(descriptor, &message, MSG_NOSIGNAL) == static_cast<ssize_t>(bytes.size()),
           "send complete bounded raw descriptor fixture");
}
void retained_descriptor_disconnect(const std::string& path) {
    const auto baseline = descriptor_count();
    {
        ExecutionSocketConfig config;
        config.socket_path = path;
        ExecutionSocketServer server(std::move(config));
        expect(server.start(), "start retained descriptor disconnect control");
        ExecutionOwnedDescriptor peer(connect_raw(path));
        ExecutionOwnedDescriptor source(::open("/dev/null", O_RDONLY | O_CLOEXEC));
        expect(source.get() >= 0 && server.poll_for(0), "authenticate descriptor source");
        const auto frame = encode_execution_message(
            {{ExecutionMessageKind::kRequest, ExecutionOpcode::kLoadModuleFile, 0, 1, 1, {2, 3}},
             {std::byte{37}}});
        expect(frame.has_value(), "encode descriptor-bearing complete frame");
        const std::array descriptors{source.get()};
        raw_send_rights(peer.get(), require_value(frame), descriptors);
        expect(server.poll_for(0), "receive authenticated descriptor");
        const auto view = server.peek_request(1);
        expect(view && view->has_descriptor && view->payload.size() == 1,
               "untaken complete frame retains its descriptor in Channel");
        expect(::shutdown(peer.get(), SHUT_RDWR) == 0 && server.poll_for(0) &&
                   !server.peek_request(1) && !server.has_peer(1) &&
                   ::fcntl(source.get(), F_GETFD) >= 0,
               "disconnect releases only Channel's untaken descriptor, not the caller's source");
        server.stop();
    }
    expect(descriptor_count() == baseline, "untaken ready descriptor has no cleanup leak");
}
void owned_descriptor_round_trip(const std::string& path, ExecutionOpcode operation) {
    const auto baseline = descriptor_count();
    {
        ExecutionOwnedDescriptor backing(
            ::memfd_create("glimmer-execution-host-fixture", MFD_CLOEXEC | MFD_ALLOW_SEALING));
        expect(
            backing.get() >= 0 && ::ftruncate(backing.get(), 4096) == 0 &&
                ::fcntl(backing.get(), F_ADD_SEALS, F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL) == 0,
            "create fixed-extent shared backing without a filesystem pathname");
        struct stat original {};
        expect(::fstat(backing.get(), &original) == 0, "retain backing object identity");
        auto* mapping = static_cast<unsigned char*>(
            ::mmap(nullptr, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, backing.get(), 0));
        expect(mapping != MAP_FAILED, "map actual shared host bytes");
        mapping[0] = 37;
        ExecutionSocketConfig config;
        config.socket_path = path;
        config.max_pending_requests = 2;
        ExecutionSocketServer server(std::move(config));
        expect(server.start(), "start owned descriptor endpoint");
        const auto identity = current_process_identity();
        expect(identity.has_value(), "owned descriptor endpoint identity");
        rights_send_attempts = rights_positive_transfers = 0;
        auto client = std::async(std::launch::async, [&, source = std::move(backing)]() mutable {
            ExecutionSocketClientConfig options{path, *identity, 4000};
            options.max_pending_requests = 2;
            ExecutionSocketClient connection(std::move(options));
            expect(connection.open().has_value(), "descriptor bootstrap has no attachment");
            bool invalid_flushed = true;
            expect(
                connection.submit_observed(ExecutionOpcode::kSnapshot, {}, source.get(), nullptr,
                                           &invalid_flushed)
                            .status == ExecutionSocketSubmitStatus::kInvalid &&
                    !invalid_flushed &&
                    connection.submit(operation).status == ExecutionSocketSubmitStatus::kInvalid &&
                    connection.submit(operation, {}, 1000000).status ==
                        ExecutionSocketSubmitStatus::kInvalid,
                "wrong opcode, missing and invalid descriptors consume no sequence");
            std::vector<std::byte> payload(std::size_t{2} * 1024 * 1024, std::byte{37});
            should_fragment_rights = true;
            // One case enters EINTR/EAGAIN; the other performs a positive one-byte
            // SCM transfer during submit. Neither has flushed its full request.
            if (operation == ExecutionOpcode::kLoadModuleFile) rights_send_attempts = 2;
            bool first_flushed = true, second_flushed = true, rejected_flushed = true;
            const auto first = connection.submit_observed(operation, payload, source.get(), nullptr,
                                                          &first_flushed);
            const auto second = connection.submit_observed(ExecutionOpcode::kSnapshot, {}, -1,
                                                           nullptr, &second_flushed);
            expect(first.status == ExecutionSocketSubmitStatus::kAccepted && first.sequence == 2 &&
                       second.status == ExecutionSocketSubmitStatus::kAccepted &&
                       second.sequence == 3 && !first_flushed && !second_flushed &&
                       connection
                               .submit_observed(operation, {}, source.get(), nullptr,
                                                &rejected_flushed)
                               .status == ExecutionSocketSubmitStatus::kFull &&
                       !rejected_flushed,
                   "descriptor ownership remains within the original bounded receipt window");
            source = ExecutionOwnedDescriptor{};
            std::ranges::fill(payload, std::byte{99});
            const auto child = ::fork();
            expect(child >= 0, "fork with a retained unsent descriptor");
            if (!child) {
                const bool guarded = connection.submit(operation).status ==
                                         ExecutionSocketSubmitStatus::kUnavailable &&
                                     !connection.poll_for(0);
                connection.~ExecutionSocketClient();
                _exit(guarded ? 0 : 1);
            }
            int child_status{};
            expect(::waitpid(child, &child_status, 0) == child && WIFEXITED(child_status) &&
                       WEXITSTATUS(child_status) == 0,
                   "fork child cannot send or dispose the parent's pending descriptor");
            std::vector<std::uint64_t> replies;
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
            while (replies.size() != 2) {
                expect(connection.poll_for(1), "progress first-EAGAIN and one-byte SCM transfer");
                for (const auto sequence : {first.sequence, second.sequence})
                    if (auto reply = connection.take_reply(sequence)) {
                        expect(reply->id == sequence && reply->original_status == 0,
                               "reversed descriptor replies retain original request identity");
                        replies.push_back(sequence);
                    }
                expect(std::chrono::steady_clock::now() < deadline, "bounded descriptor exchange");
            }
            should_fragment_rights = false;
            expect(replies == std::vector<std::uint64_t>{3, 2} && mapping[0] == 88 &&
                       rights_send_attempts == 3 && rights_positive_transfers == 1,
                   "one actual descriptor transfer survives source close and short writes");
            expect(connection.exchange(ExecutionOpcode::kClose).has_value(),
                   "close transport independently of retained backing ownership");
        });
        std::vector<ExecutionReceivedRequest> held;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{6};
        while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready) {
            expect(server.poll_for(1), "progress owned descriptor endpoint");
            if (auto request = inspect_retained_frame(server, true)) {
                if (request->message.header.operation == operation) {
                    struct stat received {};
                    expect(
                        request->descriptor.get() >= 0 &&
                            (::fcntl(request->descriptor.get(), F_GETFD) & FD_CLOEXEC) &&
                            ::fstat(request->descriptor.get(), &received) == 0 &&
                            received.st_dev == original.st_dev &&
                            received.st_ino == original.st_ino &&
                            request->message.payload.size() == std::size_t{2} * 1024 * 1024 &&
                            std::ranges::all_of(request->message.payload,
                                                [](auto value) { return value == std::byte{37}; }),
                        "actual CLOEXEC descriptor and owned partial payload retain identity");
                    const unsigned char value = 88;
                    expect(::pwrite(request->descriptor.get(), &value, 1, 0) == 1,
                           "server writes the same backing observed by the application");
                    held.push_back(std::move(*request));
                } else if (request->message.header.operation == ExecutionOpcode::kSnapshot) {
                    expect(server.reply(request->connection, response_to(*request, {3})) &&
                               held.size() == 1 &&
                               server.reply(held[0].connection, response_to(held[0], {2})),
                           "independent reply precedes the descriptor request reply");
                } else {
                    expect(request->descriptor.get() < 0 &&
                               server.reply(request->connection, response_to(*request, {})),
                           "ordinary bootstrap/close frames carry no descriptor");
                }
            }
            expect(std::chrono::steady_clock::now() < deadline,
                   "bounded retained request lifetime");
        }
        client.get();
        server.stop();
        expect(held.size() == 1 && ::fcntl(held[0].descriptor.get(), F_GETFD) >= 0,
               "transport teardown cannot close a capability already handed to its native owner");
        const auto descriptor = held[0].descriptor.get();
        auto moved = std::move(held[0].descriptor);
        expect(held[0].descriptor.get() == -1 && moved.get() == descriptor,
               "move-only descriptor transfer has one owner");
        moved = ExecutionOwnedDescriptor{};
        expect(::fcntl(descriptor, F_GETFD) < 0 && errno == EBADF && ::munmap(mapping, 4096) == 0,
               "last capability closes once; the caller mapping remains separately owned");
    }
    expect(descriptor_count() == baseline,
           "all descriptor frame and retained request references close");
}
void invalid_descriptor_frames(const std::string& path, ExecutionOpcode attached_operation,
                               bool wraps_continuation = false) {
    const auto baseline = descriptor_count();
    for (unsigned int scenario = 0; scenario < (wraps_continuation ? 11U : 8U); ++scenario) {
        ExecutionSocketConfig config;
        config.socket_path = path;
        config.io_timeout_ms = 10;
        ExecutionSocketServer server(std::move(config));
        expect(server.start(), "start strict descriptor framing endpoint");
        ExecutionOwnedDescriptor peer(connect_raw(path));
        ExecutionOwnedDescriptor source(::open("/dev/null", O_RDONLY | O_CLOEXEC));
        expect(source.get() >= 0 && server.poll_for(0), "accept raw descriptor peer");
        const auto operation = scenario == 1 ? ExecutionOpcode::kMemory : attached_operation;
        std::vector<std::byte> body;
        if (wraps_continuation) {
            auto wrapped =
                encode_execution_source_continuation({0, {17, 19}, 23, 1, 29, 0}, operation, {});
            expect(wrapped.has_value(), "encode descriptor continuation");
            body = std::move(require_value(wrapped));
        }
        auto bytes = require_value(encode_execution_message(
            {{ExecutionMessageKind::kRequest,
              wraps_continuation ? ExecutionOpcode::kSourceContinuation : operation,
              0,
              1,
              2,
              {3, 5}},
             std::move(body)}));
        std::array<int, 12> descriptors;
        descriptors.fill(source.get());
        const auto count = scenario == 2 ? 2U : scenario == 3 ? 12U : 1U;
        if (scenario == 0) {
            expect(::send(peer.get(), bytes.data(), bytes.size(), MSG_NOSIGNAL) ==
                       static_cast<ssize_t>(bytes.size()),
                   "send missing required descriptor");
        } else if (scenario == 4) {
            expect(::send(peer.get(), bytes.data(), 1, MSG_NOSIGNAL) == 1 && server.poll_for(1),
                   "begin an incomplete header without ancillary data");
            raw_send_rights(peer.get(), std::span{bytes}.subspan(1),
                            std::span{descriptors}.first(1));
        } else if (scenario == 5) {
            const auto child = ::fork();
            expect(child >= 0, "fork wrong actual descriptor sender");
            if (!child) {
                try {
                    raw_send_rights(peer.get(), bytes, std::span{descriptors}.first(1));
                    _exit(0);
                } catch (...) {
                    _exit(1);
                }
            }
            int status{};
            expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) &&
                       WEXITSTATUS(status) == 0,
                   "inherited socket does not authenticate the child's transferred descriptor");
        } else if (scenario == 6) {
            raw_send_rights(peer.get(), std::span{bytes}.first(1), std::span{descriptors}.first(1));
            expect(server.poll_for(1) && server.poll_for(30),
                   "expire retained partial descriptor frame");
        } else {
            if (scenario == 7) bytes[8] = std::byte{255};
            if (scenario == 8) bytes[kExecutionHeaderBytes] = std::byte{255};
            if (scenario == 9) bytes[kExecutionHeaderBytes + 56] = std::byte{132};
            if (scenario == 10) bytes[kExecutionHeaderBytes + 60] = std::byte{1};
            raw_send_rights(peer.get(), bytes, std::span{descriptors}.first(count));
        }
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{1};
        while (server.has_peer(1)) {
            expect(server.poll_for(1) && !server.take_request() &&
                       std::chrono::steady_clock::now() < deadline,
                   "bounded full-frame descriptor classification before dispatch");
        }
        expect(!server.take_request() && !server.has_peer(1),
               "invalid attachment never reaches dispatch or survives failure");
        server.stop();
    }
    expect(descriptor_count() == baseline,
           "every malformed/truncated/wrong-sender descriptor closes");
}
void unsent_descriptor_cleanup(const std::string& path, bool should_poison) {
    const auto baseline = descriptor_count();
    {
        ExecutionSocketConfig config;
        config.socket_path = path;
        ExecutionSocketServer server(std::move(config));
        expect(server.start(), "start unsent descriptor cleanup endpoint");
        const auto identity = current_process_identity();
        expect(identity.has_value(), "unsent cleanup endpoint identity");
        std::atomic<bool> is_queued{}, is_closed{};
        auto client = std::async(std::launch::async, [&] {
            ExecutionOwnedDescriptor source(::open("/dev/null", O_RDONLY | O_CLOEXEC));
            expect(source.get() >= 0, "create unsent capability source");
            ExecutionSocketClient connection({path, *identity, 1000});
            expect(connection.open().has_value(), "bootstrap before deliberate send backpressure");
            std::size_t blocked_attempts{};
            const SendBackpressureScope scope(blocked_attempts);
            expect(connection.submit(ExecutionOpcode::kImportHostMemory, {}, source.get()).status ==
                       ExecutionSocketSubmitStatus::kAccepted,
                   "retain the original descriptor through actual would-block backpressure");
            expect(blocked_attempts == 1, "eager submission attempts but cannot transfer rights");
            source = ExecutionOwnedDescriptor{};
            is_queued = true;
            if (should_poison) {
                const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{2};
                while (!is_closed.load()) {
                    expect(std::chrono::steady_clock::now() < deadline,
                           "bounded unsent poisoning setup");
                    std::this_thread::yield();
                }
                expect(!connection.poll_for(10) && !connection.poll_for(0),
                       "peer close poisons without sending or retaining the unsent descriptor");
            }
        });
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{3};
        while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready) {
            expect(server.poll_for(1), "progress unsent descriptor cleanup");
            if (auto request = server.take_request())
                expect(request->message.header.operation == ExecutionOpcode::kHello &&
                           server.reply(request->connection, response_to(*request, {})),
                       "only bootstrap may arrive before deliberate client abandonment");
            if (should_poison && is_queued.load() && !is_closed.load()) {
                server.close_peer(1);
                is_closed = true;
            }
            expect(std::chrono::steady_clock::now() < deadline,
                   "bounded unsent descriptor owner retirement");
        }
        client.get();
        server.stop();
    }
    expect(descriptor_count() == baseline,
           "destruction and poisoning both retire every unsent descriptor reference");
}
enum class ReplyCorruption : std::uint8_t {
    kUnknownSequence,
    kDuplicate,
    kReapedDuplicate,
    kWrongOpcode,
    kWrongKind,
    kWrongSession,
    kMalformedPayload,
    kDescriptor,
};
void corrupted_multiplexed_reply(const std::string& path, ReplyCorruption corruption) {
    RawEndpoint endpoint(path);
    endpoint.listener = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    expect(endpoint.listener >= 0, "create raw authenticated reply endpoint");
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    expect(path.size() < sizeof(address.sun_path), "raw endpoint path fits");
    std::copy(path.begin(), path.end(), address.sun_path);
    expect(::bind(endpoint.listener, reinterpret_cast<const sockaddr*>(&address),
                  sizeof(address)) == 0,
           "bind owned raw reply endpoint");
    endpoint.has_bound = true;
    expect(::chmod(path.c_str(), 0600) == 0 && ::listen(endpoint.listener, 1) == 0,
           "private raw authenticated listener");
    const auto identity = current_process_identity();
    expect(identity.has_value(), "raw endpoint trusted identity");
    std::atomic<bool> has_reaped{};
    auto client = std::async(std::launch::async, [&] {
        ExecutionSocketClient connection({path, *identity, 1500});
        expect(connection.open().has_value(), "raw authenticated bootstrap");
        const auto first = connection.submit(ExecutionOpcode::kSnapshot);
        const auto second = connection.submit(ExecutionOpcode::kDrain);
        expect(first.status == ExecutionSocketSubmitStatus::kAccepted &&
                   second.status == ExecutionSocketSubmitStatus::kAccepted,
               "own exact original requests before corruption");
        std::optional<ExecutionWireReply> known;
        while (connection.poll_for(1)) {
            if (corruption == ReplyCorruption::kReapedDuplicate && !known) {
                known = connection.take_reply(second.sequence);
                if (known) has_reaped = true;
            }
        }
        if (!known) known = connection.take_reply(second.sequence);
        expect(known && known->id == 103 && known->original_status == 701 &&
                   !connection.take_reply(first.sequence) &&
                   !connection.take_reply(second.sequence),
               "corruption cannot steal a receipt or undo an independently verified result");
        expect(connection.submit(ExecutionOpcode::kSnapshot).status ==
                       ExecutionSocketSubmitStatus::kUnavailable &&
                   !connection.open() && !connection.poll_for(0),
               "corruption permanently poisons without native fallback or replay");
    });
    pollfd waiting{endpoint.listener, POLLIN, 0};
    expect(::poll(&waiting, 1, 1000) > 0, "bounded raw accept");
    endpoint.peer = ::accept4(endpoint.listener, nullptr, nullptr, SOCK_CLOEXEC);
    expect(endpoint.peer >= 0, "accept actual client");
    const auto hello = raw_request(endpoint.peer);
    raw_send(endpoint.peer, response_to(hello, {}));
    const auto first = raw_request(endpoint.peer);
    const auto second = raw_request(endpoint.peer);
    expect(first.message.header.sequence == 2 && second.message.header.sequence == 3,
           "both original requests arrived before any result");
    raw_send(endpoint.peer, response_to(second, {103, 701}));
    if (corruption == ReplyCorruption::kReapedDuplicate) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{1};
        while (!has_reaped && std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds{1});
        expect(has_reaped, "duplicate is sent only after exact receipt has been consumed");
    }
    auto invalid = response_to(first, {102});
    switch (corruption) {
        case ReplyCorruption::kUnknownSequence:
            invalid.header.sequence = 999;
            break;
        case ReplyCorruption::kDuplicate:
        case ReplyCorruption::kReapedDuplicate:
            invalid = response_to(second, {104});
            break;
        case ReplyCorruption::kWrongOpcode:
            invalid.header.operation = ExecutionOpcode::kLaunch;
            break;
        case ReplyCorruption::kWrongKind:
            invalid.header.kind = ExecutionMessageKind::kRequest;
            break;
        case ReplyCorruption::kWrongSession:
            invalid.header.session = 999;
            break;
        case ReplyCorruption::kMalformedPayload:
            invalid.payload = {std::byte{7}};
            break;
        case ReplyCorruption::kDescriptor:
            break;
    }
    if (corruption == ReplyCorruption::kDescriptor) {
        ExecutionOwnedDescriptor source(::open("/dev/null", O_RDONLY | O_CLOEXEC));
        expect(source.get() >= 0, "prepare prohibited response descriptor");
        const auto bytes = require_value(encode_execution_message(invalid));
        const std::array descriptors{source.get()};
        raw_send_rights(endpoint.peer, bytes, descriptors);
    } else {
        raw_send(endpoint.peer, invalid);
    }
    client.get();
}

void preparation_policy_bounds() {
    const auto identity = current_process_identity();
    expect(identity.has_value(), "deadline policy identity");
    expect(!ExecutionSocketClientConfig{}.preparation_timeout_ms &&
               kExecutionMaxPreparationTimeoutMs == 600000,
           "cold override is absent by default and has an independent finite ceiling");
    for (unsigned int raw = 0; raw <= static_cast<unsigned int>(ExecutionOpcode::kProcessMemory);
         ++raw) {
        const auto operation = static_cast<ExecutionOpcode>(raw);
        const bool cold = operation == ExecutionOpcode::kLoadPtx ||
                          operation == ExecutionOpcode::kLoadBinary ||
                          operation == ExecutionOpcode::kLoadModuleFile ||
                          operation == ExecutionOpcode::kLoadModuleJit ||
                          operation == ExecutionOpcode::kGetFunction;
        expect(execution_uses_preparation_deadline(operation) == cold,
               "only the five explicit cold opcodes select the trusted timeout");
    }
    const auto descriptors = descriptor_count();
    for (const auto value : {std::uint32_t{0}, std::uint32_t{600001}, UINT32_MAX}) {
        bool rejected{};
        try {
            ExecutionSocketClientConfig config{"/unused-cold-policy.sock", require_value(identity),
                                               1000};
            config.preparation_timeout_ms = value;
            ExecutionSocketClient client(std::move(config));
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        expect(rejected, "invalid present preparation timeout rejects before connection");
    }
    for (const auto value : {std::uint32_t{1}, std::uint32_t{600000}}) {
        ExecutionSocketClientConfig config{"/unused-cold-policy.sock", require_value(identity),
                                           1000};
        config.preparation_timeout_ms = value;
        ExecutionSocketClient client(std::move(config));
        expect(client.session() == 0, "inclusive finite bounds do not open a connection");
    }
    expect(descriptor_count() == descriptors, "policy validation creates no transport resource");
}

void cold_reply_outlives_ordinary_deadline(const std::string& path, ExecutionOpcode operation,
                                           bool wraps_continuation = false) {
    ExecutionSocketConfig config;
    config.socket_path = path;
    ExecutionSocketServer server(std::move(config));
    expect(server.start(), "start cold reply endpoint");
    const auto identity = current_process_identity();
    expect(identity.has_value(), "cold endpoint identity");
    auto client = std::async(std::launch::async, [&] {
        ExecutionSocketClientConfig options{path, *identity, 200};
        options.preparation_timeout_ms = 1500;
        ExecutionSocketClient connection(std::move(options));
        expect(connection.open().has_value(), "ordinary bootstrap is immediate");
        ExecutionOwnedDescriptor attachment;
        if (operation == ExecutionOpcode::kLoadModuleFile) {
            attachment = ExecutionOwnedDescriptor(::open("/dev/null", O_RDONLY | O_CLOEXEC));
            expect(attachment.get() >= 0, "valid owned transport attachment for cold file opcode");
        }
        const auto wrapped = wraps_continuation ? encode_execution_source_continuation(
                                                      {0, {17, 19}, 23, 1, 29, 0}, operation, {})
                                                : std::nullopt;
        expect(!wraps_continuation || wrapped, "encode cold continuation");
        const auto submitted = connection.submit(
            wraps_continuation ? ExecutionOpcode::kSourceContinuation : operation,
            wrapped ? std::span<const std::byte>{*wrapped} : std::span<const std::byte>{},
            attachment.get());
        expect(submitted.status == ExecutionSocketSubmitStatus::kAccepted,
               "cold request owns exactly one immutable receipt");
        const auto child = ::fork();
        expect(child >= 0, "fork cold pending receipt control");
        if (child == 0) {
            const bool guarded =
                !connection.poll_for(0) && !connection.take_reply(submitted.sequence) &&
                connection.submit(operation).status == ExecutionSocketSubmitStatus::kUnavailable;
            connection.~ExecutionSocketClient();
            ::_exit(guarded ? 0 : 1);
        }
        int status{};
        expect(
            ::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
            "fork child cannot renew, replay or destroy parent preparation receipt");
        std::optional<ExecutionWireReply> reply;
        while (!reply) {
            expect(connection.poll_for(1), "cold request survives the ordinary deadline");
            reply = connection.take_reply(submitted.sequence);
        }
        expect(reply->id == 73 && reply->original_status == 0 &&
                   !connection.take_reply(submitted.sequence),
               "late cold reply is verified and consumed exactly once");
    });
    std::optional<ExecutionReceivedRequest> held;
    std::chrono::steady_clock::time_point received{};
    unsigned int requests{};
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{3};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready) {
        expect(server.poll_for(1), "cold reply server progress");
        if (auto request = server.take_request()) {
            ++requests;
            if (request->message.header.operation == ExecutionOpcode::kHello)
                expect(server.reply(request->connection, response_to(*request, {})),
                       "cold bootstrap reply");
            else {
                const auto effective = normalize_execution_request(
                    request->message.header.operation, request->message.payload);
                expect(!held && effective && effective->operation == operation &&
                           effective->continuation.has_value() == wraps_continuation,
                       "one original cold request reaches server");
                held = std::move(*request);
                received = std::chrono::steady_clock::now();
            }
        }
        if (held && std::chrono::steady_clock::now() - received >= std::chrono::milliseconds{300}) {
            expect(server.reply(held->connection, response_to(*held, {73})),
                   "reply deliberately follows ordinary timeout but precedes cold deadline");
            held.reset();
        }
        expect(std::chrono::steady_clock::now() < deadline, "bounded cold deadline fixture");
    }
    client.get();
    expect(requests == 2, "cold waiting and child cannot reconnect or replay");
    server.stop();
}

void preparation_deadline_expiry(const std::string& path, unsigned int mode,
                                 bool wraps_continuation = false) {
    ExecutionSocketConfig config;
    config.socket_path = path;
    config.max_pending_requests = 3;
    ExecutionSocketServer server(std::move(config));
    expect(server.start(), "start immutable expiry endpoint");
    const auto identity = current_process_identity();
    expect(identity.has_value(), "expiry endpoint identity");
    std::optional<ExecutionSocketExpiry> expired;
    auto client = std::async(std::launch::async, [&] {
        ExecutionSocketClientConfig options{path, *identity, mode == 1 ? 1500U : 200U};
        options.max_pending_requests = 3;
        if (mode != 0) options.preparation_timeout_ms = mode == 1 ? 200 : 1500;
        ExecutionSocketClient connection(std::move(options));
        expect(!connection.expired_request(), "cold client has no invented expiry");
        expect(connection.open().has_value(), "expiry bootstrap");
        std::uint64_t ordinal{};
        const auto submit = [&](ExecutionOpcode operation) {
            if (!wraps_continuation) return connection.submit(operation);
            const auto wrapped = encode_execution_source_continuation(
                {0, {17, 19}, 23, ++ordinal, 29, 0}, operation, {});
            expect(wrapped.has_value(), "encode immutable mixed-deadline continuation");
            return connection.submit(ExecutionOpcode::kSourceContinuation, *wrapped);
        };
        const auto first = submit(ExecutionOpcode::kLoadPtx);
        const auto second =
            submit(mode == 2 ? ExecutionOpcode::kSnapshot : ExecutionOpcode::kGetFunction);
        const auto third = submit(ExecutionOpcode::kLoadBinary);
        expect(first.status == ExecutionSocketSubmitStatus::kAccepted &&
                   second.status == ExecutionSocketSubmitStatus::kAccepted &&
                   third.status == ExecutionSocketSubmitStatus::kAccepted,
               "mixed requests retain independent original submission deadlines");
        const auto started = std::chrono::steady_clock::now();
        while (connection.poll_for(1))
            expect(std::chrono::steady_clock::now() - started < std::chrono::seconds{3},
                   "repeated polling cannot keep an expired request alive");
        expect(std::chrono::steady_clock::now() - started < std::chrono::milliseconds{1000},
               "ordinary/default/shorter override expires before unrelated long policy");
        const auto known = connection.take_reply(third.sequence);
        expect(
            known && known->id == 99 && known->original_status == 701 &&
                !connection.take_reply(first.sequence) && !connection.take_reply(second.sequence) &&
                !connection.take_reply(third.sequence),
            "verified original failure survives other expired requests without fabricated replies");
        expect(!connection.open() && !connection.poll_for(0) &&
                   connection.submit(ExecutionOpcode::kLoadPtx).status ==
                       ExecutionSocketSubmitStatus::kUnavailable,
               "expiry poisons once and never reconnects or replays cold work");
        expired = connection.expired_request();
        expect(expired == connection.expired_request(),
               "expiry observation is stable and nonconsuming");
        const auto child = fork();
        expect(child >= 0, "fork with retained expiry provenance");
        if (!child) _exit(connection.expired_request() ? 1 : 0);
        int child_status{};
        expect(waitpid(child, &child_status, 0) == child && WIFEXITED(child_status) &&
                   !WEXITSTATUS(child_status),
               "child cannot inspect the parent expiry");
    });
    unsigned int requests{};
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{4};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready) {
        expect(server.poll_for(1), "expiry server progress");
        if (auto request = server.take_request()) {
            ++requests;
            if (request->message.header.operation == ExecutionOpcode::kHello)
                expect(server.reply(request->connection, response_to(*request, {})),
                       "ordinary bootstrap retains original success");
            else if (request->message.header.sequence == 4)
                expect(server.reply(request->connection, response_to(*request, {99, 701})),
                       "one verified reply cannot renew other pending deadlines");
        }
        expect(std::chrono::steady_clock::now() < deadline, "bounded expiry handshake");
    }
    client.get();
    expect(requests == 4, "expired pending work is observed once, never replayed");
    server.stop();
    const auto effective = mode == 2 ? ExecutionOpcode::kSnapshot : ExecutionOpcode::kLoadPtx;
    expect(
        expired == ExecutionSocketExpiry{wraps_continuation ? ExecutionOpcode::kSourceContinuation
                                                            : effective,
                                         effective, mode == 2 ? 3U : 2U, 41, 200},
        "after exact cleanup, retain the earliest expired wire/effective request and budget");
}

void partial_cold_reply_deadline(const std::string& path, bool wraps_continuation = false) {
    RawEndpoint endpoint(path);
    endpoint.listener = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    expect(endpoint.listener >= 0, "partial cold endpoint");
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    expect(path.size() < sizeof(address.sun_path), "partial cold endpoint path fits");
    std::copy(path.begin(), path.end(), address.sun_path);
    expect(::bind(endpoint.listener, reinterpret_cast<const sockaddr*>(&address),
                  sizeof(address)) == 0,
           "bind partial cold endpoint");
    endpoint.has_bound = true;
    expect(::chmod(path.c_str(), 0600) == 0 && ::listen(endpoint.listener, 1) == 0,
           "private partial cold endpoint");
    const auto identity = current_process_identity();
    expect(identity.has_value(), "partial endpoint identity");
    std::atomic<unsigned int> fragments{};
    auto client = std::async(std::launch::async, [&] {
        ExecutionSocketClientConfig options{path, *identity, 1500};
        options.preparation_timeout_ms = 160;
        ExecutionSocketClient connection(std::move(options));
        expect(connection.open().has_value(), "partial cold bootstrap");
        const auto wrapped =
            wraps_continuation ? encode_execution_source_continuation(
                                     {0, {17, 19}, 23, 1, 29, 0}, ExecutionOpcode::kGetFunction, {})
                               : std::nullopt;
        expect(!wraps_continuation || wrapped, "encode partial cold continuation");
        const auto submitted = connection.submit(
            wraps_continuation ? ExecutionOpcode::kSourceContinuation
                               : ExecutionOpcode::kGetFunction,
            wrapped ? std::span<const std::byte>{*wrapped} : std::span<const std::byte>{});
        expect(submitted.status == ExecutionSocketSubmitStatus::kAccepted, "partial cold request");
        const auto started = std::chrono::steady_clock::now();
        while (connection.poll_for(1))
            expect(std::chrono::steady_clock::now() - started < std::chrono::seconds{2},
                   "partial progress cannot renew absolute deadline indefinitely");
        expect(fragments >= 2 &&
                   std::chrono::steady_clock::now() - started < std::chrono::milliseconds{350} &&
                   !connection.take_reply(submitted.sequence) && !connection.open(),
               "incomplete verified-prefix progress does not extend submission deadline or publish "
               "success");
        expect(
            connection.expired_request() ==
                ExecutionSocketExpiry{wraps_continuation ? ExecutionOpcode::kSourceContinuation
                                                         : ExecutionOpcode::kGetFunction,
                                      ExecutionOpcode::kGetFunction, submitted.sequence, 41, 160},
            "partial progress preserves the exact cold request expiry cause");
    });
    pollfd ready{endpoint.listener, POLLIN, 0};
    expect(::poll(&ready, 1, 1000) > 0, "bounded partial cold accept");
    endpoint.peer = ::accept4(endpoint.listener, nullptr, nullptr, SOCK_CLOEXEC);
    expect(endpoint.peer >= 0, "accept partial cold client");
    raw_send(endpoint.peer, response_to(raw_request(endpoint.peer), {}));
    const auto request = raw_request(endpoint.peer);
    const auto bytes = encode_execution_message(response_to(request, {71}));
    expect(bytes && bytes->size() > 10, "complete reply exists but is deliberately withheld");
    for (unsigned int index = 0; index < 10; ++index) {
        if (client.wait_for(std::chrono::milliseconds{35}) == std::future_status::ready) break;
        const auto count = ::send(endpoint.peer, bytes->data() + index, 1, MSG_NOSIGNAL);
        if (count < 0 && (errno == EPIPE || errno == ECONNRESET)) break;
        expect(count == 1, "one new reply byte makes bounded transport progress");
        ++fragments;
    }
    client.get();
}

void preparation_deadlines(const std::string& path) {
    preparation_policy_bounds();
    for (const auto operation : {ExecutionOpcode::kLoadPtx, ExecutionOpcode::kLoadBinary,
                                 ExecutionOpcode::kLoadModuleFile, ExecutionOpcode::kGetFunction})
        cold_reply_outlives_ordinary_deadline(path, operation);
    for (unsigned int mode = 0; mode < 3; ++mode) preparation_deadline_expiry(path, mode);
    partial_cold_reply_deadline(path);
    for (const auto operation : {ExecutionOpcode::kLoadPtx, ExecutionOpcode::kLoadBinary,
                                 ExecutionOpcode::kLoadModuleFile, ExecutionOpcode::kGetFunction})
        cold_reply_outlives_ordinary_deadline(path, operation, true);
    for (unsigned int mode = 0; mode < 3; ++mode) preparation_deadline_expiry(path, mode, true);
    partial_cold_reply_deadline(path, true);
}
}  // namespace
int main(int argc, char** argv) {
    try {
        const auto descriptors = descriptor_count();
        invalid_windows_are_rejected();
        Directory directory;
        if (argc == 2 && std::string_view(argv[1]) == "--preparation-deadlines") {
            preparation_deadlines(directory.path + "/preparation.sock");
            expect(descriptor_count() == descriptors, "cold deadline controls release descriptors");
            std::cout << "Bounded cold preparation deadline checks passed\n";
            return EXIT_SUCCESS;
        }
        expect(argc == 1, "unknown execution socket test mode");
        process_identity_boundaries();
        wrong_uid_is_rejected(directory.path + "/wrong-uid.sock");
        forked_server_destruction_preserves_parent(directory.path + "/fork-server.sock");
        preparation_deadlines(directory.path + "/preparation.sock");
        authenticated_round_trip(directory.path + "/roundtrip.sock");
        inherited_sender_is_rejected(directory.path + "/fork.sock");
        retained_frame_boundaries(directory.path + "/retained.sock");
        retained_descriptor_disconnect(directory.path + "/retained-descriptor.sock");
        first_positive_read_boundaries(directory.path + "/first-read.sock");
        rights_and_partial_frames_are_rejected(directory.path + "/rights.sock");
        wrong_reply_incarnation_poisoned(directory.path + "/incarnation.sock");
        bootstrap_ownership_validation(directory.path + "/ownership.sock");
        bootstrap_ownership_validation(directory.path + "/wrapped-ownership.sock", true);
        source_continuation_round_trip(directory.path + "/continuation.sock");
        closing_host_function_controls(directory.path + "/callback-close.sock", false);
        closing_host_function_controls(directory.path + "/callback-wrapped-close.sock", true);
        multiplexed_owned_round_trip(directory.path + "/multiplex.sock");
        owned_readiness_snapshots(directory.path + "/readiness.sock");
        owned_descriptor_round_trip(directory.path + "/descriptor.sock",
                                    ExecutionOpcode::kImportHostMemory);
        owned_descriptor_round_trip(directory.path + "/module-descriptor.sock",
                                    ExecutionOpcode::kLoadModuleFile);
        invalid_descriptor_frames(directory.path + "/descriptor-invalid.sock",
                                  ExecutionOpcode::kImportHostMemory);
        invalid_descriptor_frames(directory.path + "/module-descriptor-invalid.sock",
                                  ExecutionOpcode::kLoadModuleFile);
        invalid_descriptor_frames(directory.path + "/wrapped-descriptor-invalid.sock",
                                  ExecutionOpcode::kImportHostMemory, true);
        invalid_descriptor_frames(directory.path + "/wrapped-module-descriptor-invalid.sock",
                                  ExecutionOpcode::kLoadModuleFile, true);
        unsent_descriptor_cleanup(directory.path + "/descriptor-abandon.sock", false);
        unsent_descriptor_cleanup(directory.path + "/descriptor-poison.sock", true);
        server_window_is_bounded(directory.path + "/window.sock");
        multiplexed_failure(directory.path + "/timeout.sock", MultiplexFailure::kTimeout);
        multiplexed_failure(directory.path + "/disconnect.sock", MultiplexFailure::kDisconnect);
        multiplexed_failure(directory.path + "/stale.sock", MultiplexFailure::kWrongIncarnation);
        for (const auto corruption :
             {ReplyCorruption::kUnknownSequence, ReplyCorruption::kDuplicate,
              ReplyCorruption::kReapedDuplicate, ReplyCorruption::kWrongOpcode,
              ReplyCorruption::kWrongKind, ReplyCorruption::kWrongSession,
              ReplyCorruption::kMalformedPayload, ReplyCorruption::kDescriptor})
            corrupted_multiplexed_reply(directory.path + "/corrupt.sock", corruption);
        expect(descriptor_count() == descriptors, "all pending/failed endpoint descriptors retire");
        std::cout << "Authenticated execution socket checks passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
