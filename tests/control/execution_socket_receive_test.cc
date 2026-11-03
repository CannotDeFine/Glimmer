#include "glimmer/control/execution_socket.h"
#include "internal/process_identity.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <thread>
#include <vector>
#include <fcntl.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/eventfd.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>
#include <time.h>

namespace {
struct ReceiveStats {
    int descriptor{-1};
    std::size_t attempts{}, bytes{};
    std::size_t maximum_read_bytes{};
};
struct ReceiveProbe {
    std::array<ReceiveStats, 4> channels{};
    std::size_t attempts{}, interrupted{}, would_block_at{}, chunk_limit{};
    bool limits_first_only{}, invalid_flags{};
};
thread_local ReceiveProbe* receive_probe{};
struct WakeProbe {
    int descriptor{-1}, read_error{}, write_error{};
    std::size_t read_interruptions{}, write_interruptions{}, reads{}, writes{};
    bool fails_creation{};
};
thread_local WakeProbe* wake_probe{};
struct PublicationProbe {
    std::size_t bytes{}, maximum{}, interrupted{}, fail_after{};
};
thread_local PublicationProbe* publication_probe{};
struct PollProbe {
    std::vector<std::uint64_t> timeouts;
    bool interrupts{}, fails{};
    std::atomic<bool>* entered{};
};
thread_local PollProbe* poll_probe{};
}  // namespace
// NOLINTNEXTLINE(bugprone-reserved-identifier,readability-identifier-naming): GNU ld wrapping ABI.
extern "C" int __real_ppoll(pollfd*, nfds_t, const timespec*, const sigset_t*);
// NOLINTNEXTLINE(bugprone-reserved-identifier,readability-identifier-naming): GNU ld wrapping ABI.
extern "C" int __wrap_ppoll(pollfd* descriptors, nfds_t count, const timespec* timeout,
                            const sigset_t* mask) {
    if (poll_probe) {
        poll_probe->timeouts.push_back(static_cast<std::uint64_t>(timeout->tv_sec) * 1000000000 +
                                       static_cast<std::uint64_t>(timeout->tv_nsec));
        if (std::exchange(poll_probe->interrupts, false)) {
            errno = EINTR;
            return -1;
        }
        if (poll_probe->fails) {
            errno = EIO;
            return -1;
        }
        if (poll_probe->entered) poll_probe->entered->store(true, std::memory_order_release);
    }
    return __real_ppoll(descriptors, count, timeout, mask);
}
// NOLINTNEXTLINE(bugprone-reserved-identifier,readability-identifier-naming): GNU ld wrapping ABI.
extern "C" int __real_eventfd(unsigned int initial, int flags);
// NOLINTNEXTLINE(bugprone-reserved-identifier,readability-identifier-naming): GNU ld wrapping ABI.
extern "C" int __wrap_eventfd(unsigned int initial, int flags) {
    if (wake_probe && wake_probe->fails_creation) {
        errno = EMFILE;
        return -1;
    }
    return __real_eventfd(initial, flags);
}
// NOLINTNEXTLINE(bugprone-reserved-identifier,readability-identifier-naming): GNU ld wrapping ABI.
extern "C" ssize_t __real_read(int descriptor, void* buffer, std::size_t bytes);
// NOLINTNEXTLINE(bugprone-reserved-identifier,readability-identifier-naming): GNU ld wrapping ABI.
extern "C" ssize_t __wrap_read(int descriptor, void* buffer, std::size_t bytes) {
    auto* probe = wake_probe;
    if (!probe || probe->descriptor != descriptor) return __real_read(descriptor, buffer, bytes);
    ++probe->reads;
    if (probe->read_interruptions) {
        --probe->read_interruptions;
        errno = EINTR;
        return -1;
    }
    if (probe->read_error) {
        errno = probe->read_error;
        return -1;
    }
    return __real_read(descriptor, buffer, bytes);
}
// NOLINTNEXTLINE(bugprone-reserved-identifier,readability-identifier-naming): GNU ld wrapping ABI.
extern "C" ssize_t __real_write(int descriptor, const void* buffer, std::size_t bytes);
// NOLINTNEXTLINE(bugprone-reserved-identifier,readability-identifier-naming): GNU ld wrapping ABI.
extern "C" ssize_t __wrap_write(int descriptor, const void* buffer, std::size_t bytes) {
    if (auto* probe = publication_probe) {
        if (probe->interrupted) {
            --probe->interrupted;
            errno = EINTR;
            return -1;
        }
        if (probe->fail_after && probe->bytes >= probe->fail_after) {
            errno = ENOSPC;
            return -1;
        }
        auto limit = probe->maximum ? std::min(bytes, probe->maximum) : bytes;
        if (probe->fail_after) limit = std::min(limit, probe->fail_after - probe->bytes);
        const auto count = __real_write(descriptor, buffer, limit);
        if (count > 0) probe->bytes += static_cast<std::size_t>(count);
        return count;
    }
    auto* probe = wake_probe;
    if (!probe || probe->descriptor != descriptor) return __real_write(descriptor, buffer, bytes);
    ++probe->writes;
    if (probe->write_interruptions) {
        --probe->write_interruptions;
        errno = EINTR;
        return -1;
    }
    if (probe->write_error) {
        errno = probe->write_error;
        return -1;
    }
    return __real_write(descriptor, buffer, bytes);
}
// NOLINTNEXTLINE(bugprone-reserved-identifier,readability-identifier-naming): GNU ld wrapping ABI.
extern "C" ssize_t __real_recvmsg(int descriptor, msghdr* message, int flags);
// NOLINTNEXTLINE(bugprone-reserved-identifier,readability-identifier-naming): GNU ld wrapping ABI.
extern "C" ssize_t __wrap_recvmsg(int descriptor, msghdr* message, int flags) {
    auto* probe = receive_probe;
    if (!probe) return __real_recvmsg(descriptor, message, flags);
    ++probe->attempts;
    probe->invalid_flags = probe->invalid_flags || (flags & (MSG_DONTWAIT | MSG_CMSG_CLOEXEC)) !=
                                                       (MSG_DONTWAIT | MSG_CMSG_CLOEXEC);
    auto found = std::ranges::find(probe->channels, descriptor, &ReceiveStats::descriptor);
    if (found == probe->channels.end()) {
        found = std::ranges::find(probe->channels, -1, &ReceiveStats::descriptor);
        if (found == probe->channels.end()) {
            errno = EIO;
            return -1;
        }
        found->descriptor = descriptor;
    }
    ++found->attempts;
    found->maximum_read_bytes = std::max(found->maximum_read_bytes, message->msg_iov[0].iov_len);
    if (probe->interrupted) {
        --probe->interrupted;
        errno = EINTR;
        return -1;
    }
    if (probe->attempts == probe->would_block_at) {
        errno = EAGAIN;
        return -1;
    }
    const auto original_size = message->msg_iov[0].iov_len;
    if (probe->chunk_limit && (!probe->limits_first_only || found == probe->channels.begin()))
        message->msg_iov[0].iov_len = std::min(original_size, probe->chunk_limit);
    const auto count = __real_recvmsg(descriptor, message, flags);
    message->msg_iov[0].iov_len = original_size;
    if (count > 0) found->bytes += static_cast<std::size_t>(count);
    return count;
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
class ProbeScope final {
   public:
    explicit ProbeScope(ReceiveProbe& probe) {
        expect(!receive_probe, "receive probes cannot nest");
        receive_probe = &probe;
    }
    ~ProbeScope() {
        receive_probe = nullptr;
    }
    ProbeScope(const ProbeScope&) = delete;
    ProbeScope& operator=(const ProbeScope&) = delete;
};
class WakeProbeScope final {
   public:
    explicit WakeProbeScope(WakeProbe& probe) {
        expect(!wake_probe, "wake probes cannot nest");
        wake_probe = &probe;
    }
    ~WakeProbeScope() {
        wake_probe = nullptr;
    }
    WakeProbeScope(const WakeProbeScope&) = delete;
    WakeProbeScope& operator=(const WakeProbeScope&) = delete;
};
struct Directory {
    std::string path;
    Directory() {
        char pattern[] = "/tmp/glimmer-receive-XXXXXX";
        const auto* created = ::mkdtemp(pattern);
        expect(created, "create private receive-test directory");
        path = created;
    }
    ~Directory() {
        if (::rmdir(path.c_str()) != 0) std::cerr << "Receive-test directory cleanup failed\n";
    }
};
sockaddr_un address_for(const std::string& path) {
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    expect(path.size() < sizeof(address.sun_path), "bounded private socket path");
    std::copy(path.begin(), path.end(), address.sun_path);
    return address;
}
ExecutionSocketConfig options(const std::string& path, bool should_record_receive_time = false,
                              std::shared_ptr<ExecutionSocketWake> wake = {}) {
    ExecutionSocketConfig config;
    config.socket_path = path;
    config.max_pending_requests = 2;
    config.should_record_receive_time = should_record_receive_time;
    config.wake = std::move(wake);
    return config;
}
struct Endpoint {
    Directory directory;
    ExecutionSocketServer server;
    explicit Endpoint(bool should_record_receive_time = false,
                      std::shared_ptr<ExecutionSocketWake> wake = {})
        : server(options(directory.path + "/socket", should_record_receive_time, std::move(wake))) {
        expect(server.start(), "start actual receive-test endpoint");
    }
    explicit Endpoint(std::shared_ptr<ExecutionSocketPollJournal> journal,
                      std::shared_ptr<ExecutionSocketWake> wake = {})
        : server(options(directory.path + "/socket", true, std::move(wake)), std::move(journal)) {
        expect(server.start(), "start diagnostic receive-test endpoint");
    }
    ExecutionOwnedDescriptor connect() {
        ExecutionOwnedDescriptor peer(::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0));
        const auto address = address_for(directory.path + "/socket");
        expect(peer.get() >= 0 &&
                   ::connect(peer.get(), reinterpret_cast<const sockaddr*>(&address),
                             sizeof(address)) == 0 &&
                   server.poll_for(0),
               "connect and authenticate actual raw peer");
        return peer;
    }
};
std::vector<std::byte> frame(std::size_t payload_size = 512, std::uint64_t sequence = 1,
                             ExecutionOpcode operation = ExecutionOpcode::kHello) {
    const auto result =
        encode_execution_message({{ExecutionMessageKind::kRequest, operation, 0, 0, sequence, {}},
                                  std::vector<std::byte>(payload_size, std::byte{0x35})});
    expect(result.has_value(), "encode complete literal transport frame");
    return require_value(result);
}
void send_bytes(int descriptor, std::span<const std::byte> bytes, int attachment = -1) {
    pollfd ready{descriptor, POLLOUT, 0};
    expect(::poll(&ready, 1, 1000) > 0, "bounded raw fixture write");
    ssize_t count{};
    if (attachment < 0) {
        count = ::send(descriptor, bytes.data(), bytes.size(), MSG_DONTWAIT | MSG_NOSIGNAL);
    } else {
        alignas(cmsghdr) std::array<std::byte, CMSG_SPACE(sizeof(int))> ancillary{};
        iovec data{const_cast<std::byte*>(bytes.data()), bytes.size()};
        msghdr message{};
        message.msg_iov = &data;
        message.msg_iovlen = 1;
        message.msg_control = ancillary.data();
        message.msg_controllen = ancillary.size();
        auto* item = CMSG_FIRSTHDR(&message);
        item->cmsg_level = SOL_SOCKET;
        item->cmsg_type = SCM_RIGHTS;
        item->cmsg_len = CMSG_LEN(sizeof(int));
        std::memcpy(CMSG_DATA(item), &attachment, sizeof(attachment));
        count = ::sendmsg(descriptor, &message, MSG_DONTWAIT | MSG_NOSIGNAL);
    }
    expect(count == static_cast<ssize_t>(bytes.size()), "send complete bounded fixture extent");
}
bool progress(ExecutionSocketServer& server, ReceiveProbe& probe) {
    const ProbeScope scope(probe);
    const auto result = server.poll_for(0);
    expect(!probe.invalid_flags, "every chunk remains nonblocking and descriptor-CLOEXEC");
    return result;
}
void small_frame_and_no_read_ahead() {
    Endpoint endpoint;
    auto peer = endpoint.connect();
    const auto first = frame();
    const auto second = frame(23, 2);
    send_bytes(peer.get(), first);
    send_bytes(peer.get(), second);
    ReceiveProbe probe;
    expect(progress(endpoint.server, probe), "one owner turn receives queued bytes");
    const auto received = endpoint.server.take_request();
    expect(received && received->message.payload == std::vector<std::byte>(512, std::byte{0x35}) &&
               probe.attempts == 2 && probe.channels[0].bytes == first.size(),
           "complete small frame must publish in one receive turn");
    expect(!endpoint.server.take_request(), "receive never reads ahead into a successor frame");
    ReceiveProbe next;
    expect(progress(endpoint.server, next), "next turn receives successor");
    const auto successor = endpoint.server.take_request();
    expect(successor && successor->message.header.sequence == 2 && next.attempts == 2 &&
               next.channels[0].bytes == second.size(),
           "successor remains independent and once-only");
    send_bytes(peer.get(), frame(0, 3));
    ReceiveProbe blocked;
    expect(progress(endpoint.server, blocked) && blocked.attempts == 0 &&
               !endpoint.server.take_request(),
           "completed receive batches cannot exceed the pending receipt window");
}
std::uint64_t monotonic_ns() {
    timespec value{};
    expect(::clock_gettime(CLOCK_MONOTONIC, &value) == 0 && value.tv_sec >= 0,
           "read independent Linux monotonic clock");
    return static_cast<std::uint64_t>(value.tv_sec) * 1000000000ULL +
           static_cast<std::uint64_t>(value.tv_nsec);
}
void complete_frame_timestamps(bool enabled) {
    Endpoint endpoint(enabled);
    auto peer = endpoint.connect();
    const auto bytes = frame();
    const auto prefix = kExecutionHeaderBytes + 3;
    send_bytes(peer.get(), std::span{bytes}.first(prefix));
    expect(endpoint.server.poll_for(0) && !endpoint.server.take_request(),
           "partial frame publishes neither a request nor a timestamp");
    const auto before = monotonic_ns();
    send_bytes(peer.get(), std::span{bytes}.subspan(prefix));
    expect(endpoint.server.poll_for(0), "complete original frame");
    const auto after = monotonic_ns();
    const auto first = endpoint.server.take_request();
    expect(
        first && first->message.payload == std::vector<std::byte>(512, std::byte{0x35}) &&
            (enabled ? first->frame_received_ns >= before && first->frame_received_ns <= after
                     : first->frame_received_ns == 0),
        "opt-in complete-frame stamp uses the actual server monotonic clock; default stays zero");
    const auto next_before = monotonic_ns();
    send_bytes(peer.get(), frame(17, 2));
    expect(endpoint.server.poll_for(0), "complete successive original frame");
    const auto next = endpoint.server.take_request();
    const auto next_after = monotonic_ns();
    expect(next && next->message.header.sequence == 2 &&
               (enabled ? next->frame_received_ns >= next_before &&
                              next->frame_received_ns <= next_after &&
                              next->frame_received_ns >= require_value(first).frame_received_ns
                        : next->frame_received_ns == 0) &&
               !endpoint.server.take_request(),
           "successor cannot inherit a prior frame's timestamp or publish twice");
}
void fragmented_yield_and_other_peer() {
    Endpoint endpoint;
    auto first = endpoint.connect();
    auto second = endpoint.connect();
    const auto bytes = frame();
    send_bytes(first.get(), bytes);
    send_bytes(second.get(), frame(19));
    ReceiveProbe probe;
    probe.chunk_limit = 1;
    probe.limits_first_only = true;
    expect(progress(endpoint.server, probe), "fragmented and ordinary peers share one poll");
    const auto ordinary = endpoint.server.take_request();
    expect(probe.channels[0].attempts == 8 && probe.channels[0].bytes == 8 && ordinary &&
               ordinary->connection == 2 && ordinary->message.payload.size() == 19 &&
               !endpoint.server.take_request(),
           "eight short reads yield without preventing another peer's complete receive");
    ReceiveProbe remainder;
    expect(progress(endpoint.server, remainder), "resume fragmented frame without replay");
    const auto completed = endpoint.server.take_request();
    expect(completed && completed->connection == 1 && completed->message.payload.size() == 512 &&
               remainder.channels[0].bytes == bytes.size() - 8,
           "partial header survives budget exhaustion exactly once");
}
void bounded_interrupts_and_would_block() {
    Endpoint endpoint;
    auto peer = endpoint.connect();
    send_bytes(peer.get(), frame());
    ReceiveProbe interrupted;
    interrupted.interrupted = 9;
    expect(progress(endpoint.server, interrupted) && interrupted.attempts == 8 &&
               interrupted.interrupted == 1 && !endpoint.server.take_request(),
           "EINTR retries consume the finite per-turn receive budget");
    ReceiveProbe blocked;
    blocked.would_block_at = 2;
    expect(progress(endpoint.server, blocked) && blocked.attempts == 2 &&
               blocked.channels[0].bytes == kExecutionHeaderBytes &&
               !endpoint.server.take_request(),
           "EAGAIN immediately yields with the verified header retained");
    ReceiveProbe resumed;
    expect(progress(endpoint.server, resumed), "resume after would-block without reconnect");
    const auto completed = endpoint.server.take_request();
    expect(completed && completed->message.payload.size() == 512 && resumed.attempts == 1 &&
               resumed.channels[0].bytes == 512,
           "retained payload completes without replaying header");
}
void large_frame_yields() {
    Endpoint endpoint;
    auto large = endpoint.connect();
    auto small = endpoint.connect();
    const int capacity = 512 * 1024;
    expect(::setsockopt(large.get(), SOL_SOCKET, SO_SNDBUF, &capacity, sizeof(capacity)) == 0,
           "make the complete large fixture readable without producer timing");
    const auto bytes = frame(128 * 1024 + 97);
    send_bytes(large.get(), bytes);
    send_bytes(small.get(), frame(31));
    ReceiveProbe probe;
    expect(progress(endpoint.server, probe), "large input has a bounded receive turn");
    const auto ordinary = endpoint.server.take_request();
    expect(probe.channels[0].attempts == 8 && probe.channels[0].maximum_read_bytes == 16384 &&
               probe.channels[0].bytes == kExecutionHeaderBytes + std::size_t{7} * 16384 &&
               ordinary && ordinary->connection == 2 && ordinary->message.payload.size() == 31 &&
               !endpoint.server.take_request(),
           "large queued input exhausts the byte/attempt budget while another peer completes");
    ReceiveProbe remainder;
    expect(progress(endpoint.server, remainder), "large frame resumes in the next turn");
    const auto completed = endpoint.server.take_request();
    expect(completed && completed->connection == 1 &&
               completed->message.payload ==
                   std::vector<std::byte>(128 * 1024 + 97, std::byte{0x35}) &&
               probe.channels[0].bytes + remainder.channels[0].bytes == bytes.size() &&
               !endpoint.server.take_request(),
           "large frame publishes only its complete unchanged payload exactly once");
}
void descriptor_handoff_and_late_rejection(bool late) {
    Endpoint endpoint;
    auto peer = endpoint.connect();
    ExecutionOwnedDescriptor attachment(::open("/dev/null", O_RDONLY | O_CLOEXEC));
    expect(attachment.get() >= 0, "own original descriptor capability");
    const auto bytes = frame(97, 1, ExecutionOpcode::kImportHostMemory);
    send_bytes(peer.get(), std::span{bytes}.first(1), attachment.get());
    send_bytes(peer.get(), std::span{bytes}.subspan(1), late ? attachment.get() : -1);
    ReceiveProbe probe;
    expect(progress(endpoint.server, probe), "receive authenticated ancillary chunks");
    const auto received = endpoint.server.take_request();
    if (late) {
        expect(!received && !endpoint.server.has_peer(1),
               "a late second descriptor rejects the entire frame before handoff");
    } else {
        struct stat original {
        }, transferred{};
        expect(received && received->descriptor.get() >= 0 && probe.attempts == 3 &&
                   ::fstat(attachment.get(), &original) == 0 &&
                   ::fstat(received->descriptor.get(), &transferred) == 0 &&
                   original.st_dev == transferred.st_dev && original.st_ino == transferred.st_ino &&
                   (::fcntl(received->descriptor.get(), F_GETFD) & FD_CLOEXEC) != 0 &&
                   !endpoint.server.take_request(),
               "first-byte capability transfers once across a batched fragmented header");
    }
}
void foreign_payload_rejects() {
    Endpoint endpoint;
    auto peer = endpoint.connect();
    const auto bytes = frame();
    send_bytes(peer.get(), std::span{bytes}.first(kExecutionHeaderBytes));
    const auto child = ::fork();
    expect(child >= 0, "fork actual foreign payload sender");
    if (child == 0) {
        const auto remaining = std::span{bytes}.subspan(kExecutionHeaderBytes);
        const auto count = ::send(peer.get(), remaining.data(), remaining.size(), MSG_NOSIGNAL);
        ::_exit(count == static_cast<ssize_t>(remaining.size()) ? 0 : 1);
    }
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "foreign child has queued a payload behind the parent's header");
    ReceiveProbe probe;
    expect(progress(endpoint.server, probe) && probe.attempts == 2 &&
               !endpoint.server.take_request() && !endpoint.server.has_peer(1),
           "batched payload cannot borrow its earlier header sender credentials");
}
void one_turn_client_reply() {
    Directory directory;
    const auto path = directory.path + "/reply";
    ExecutionOwnedDescriptor listener(::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0));
    const auto address = address_for(path);
    expect(
        listener.get() >= 0 && ::bind(listener.get(), reinterpret_cast<const sockaddr*>(&address),
                                      sizeof(address)) == 0,
        "bind private raw reply endpoint");
    struct PathGuard {
        const std::string& path;
        ~PathGuard() {
            if (::unlink(path.c_str()) != 0) std::cerr << "Raw reply pathname cleanup failed\n";
        }
    } path_guard{path};
    expect(::chmod(path.c_str(), 0600) == 0 && ::listen(listener.get(), 1) == 0,
           "private raw endpoint accepts one authenticated client");
    const auto identity = current_process_identity();
    expect(identity.has_value(), "raw endpoint authenticates the actual server process");
    ExecutionSocketClient client({path, require_value(identity), 3000});
    const auto submitted = client.submit_open();
    expect(submitted.status == ExecutionSocketSubmitStatus::kAccepted && client.poll_for(0),
           "ordinary client connects and sends the original bootstrap");
    ExecutionOwnedDescriptor peer(::accept4(listener.get(), nullptr, nullptr, SOCK_CLOEXEC));
    expect(peer.get() >= 0, "accept authenticated bootstrap socket");
    ExecutionWireReply reply;
    reply.disposition = ExecutionWireDisposition::kComplete;
    const auto payload = encode_execution_reply(reply);
    expect(payload.has_value(), "encode original reply payload");
    const auto encoded = encode_execution_message({{ExecutionMessageKind::kReply,
                                                    ExecutionOpcode::kHello,
                                                    0,
                                                    7,
                                                    submitted.sequence,
                                                    {11, 13}},
                                                   require_value(payload)});
    expect(encoded.has_value(), "encode original complete bootstrap reply");
    send_bytes(peer.get(), require_value(encoded));
    ReceiveProbe probe;
    {
        const ProbeScope scope(probe);
        expect(client.poll_for(0), "one client turn receives a complete queued reply");
    }
    const auto completed = client.take_reply(submitted.sequence);
    expect(completed && completed->status == 0 && client.session() == 7 && probe.attempts == 2 &&
               probe.channels[0].bytes == require_value(encoded).size() && !probe.invalid_flags &&
               !client.take_reply(submitted.sequence),
           "client verifies and publishes complete reply in one turn exactly once");
}
void coalesced_process_local_wakes() {
    auto wake = std::make_shared<ExecutionSocketWake>();
    Endpoint endpoint(false, wake);
    const auto descriptor = wake->descriptor();
    const auto flags = ::fcntl(descriptor, F_GETFL),
               descriptor_flags = ::fcntl(descriptor, F_GETFD);
    expect(descriptor >= 0 && flags >= 0 && descriptor_flags >= 0 && (flags & O_NONBLOCK) &&
               (descriptor_flags & FD_CLOEXEC),
           "nonblocking close-on-exec CPU hint");
    errno = EDOM;
    expect(wake->consume() && errno == EDOM, "empty hint is harmless and preserves errno");
    for (unsigned int count = 0; count < 16; ++count)
        expect(wake->notify() && errno == EDOM, "coalescing notifications preserve errno");
    pollfd observed{descriptor, POLLIN, 0};
    expect(
        ::poll(&observed, 1, 0) == 1 && observed.revents == POLLIN && endpoint.server.poll_for(0),
        "transport drains actual coalesced readiness");
    observed.revents = 0;
    expect(::poll(&observed, 1, 0) == 0 && !endpoint.server.take_request(),
           "one wake drain leaves neither a ready descriptor nor an invented request");
    const auto child = ::fork();
    expect(child >= 0, "fork an idle wake reference");
    if (!child)
        ::_exit(wake->descriptor() == -1 && !wake->notify() && !wake->consume() ? EXIT_SUCCESS
                                                                                : EXIT_FAILURE);
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0 &&
               ::poll(&observed, 1, 0) == 0,
           "inherited hints cannot notify or consume parent state");
    std::atomic<bool> is_notified{};
    std::thread producer([wake, &is_notified] { is_notified = wake->notify(); });
    const auto progressed = endpoint.server.poll_for(1000);
    producer.join();
    expect(progressed && is_notified && ::poll(&observed, 1, 0) == 0,
           "background publication wakes actual socket poll without a socket frame");
    auto peer = endpoint.connect();
    const auto original = frame(13);
    send_bytes(peer.get(), original);
    expect(wake->notify() && endpoint.server.poll_for(0),
           "wake does not hide actual peer readiness");
    const auto request = endpoint.server.take_request();
    expect(request && request->message.payload == std::vector<std::byte>(13, std::byte{0x35}),
           "ordinary authenticated receive remains unchanged alongside a hint");
}
void bounded_wake_failures() {
    WakeProbe probe;
    const WakeProbeScope scope(probe);
    probe.fails_creation = true;
    bool is_rejected{};
    try {
        const ExecutionSocketWake rejected;
    } catch (const std::runtime_error&) {
        is_rejected = true;
    }
    expect(is_rejected, "failed eventfd construction rejects without a live descriptor");
    probe.fails_creation = false;
    ExecutionSocketWake wake;
    probe.descriptor = wake.descriptor();
    probe.write_interruptions = 9;
    errno = EDOM;
    expect(!wake.notify() && probe.writes == 8 && probe.write_interruptions == 1 && errno == EDOM,
           "interrupted notification has an exact finite budget and preserves errno");
    probe.write_interruptions = 7;
    expect(wake.notify() && probe.writes == 16 && errno == EDOM,
           "last bounded notification attempt can publish the actual hint");
    probe.write_error = EAGAIN;
    expect(wake.notify() && errno == EDOM,
           "already pending saturated notification stays coalesced");
    probe.write_error = EIO;
    expect(!wake.notify() && errno == EDOM,
           "notification failure changes no external errno or result");
    probe.write_error = 0;
    probe.read_interruptions = 9;
    expect(!wake.consume() && probe.reads == 8 && probe.read_interruptions == 1 && errno == EDOM,
           "interrupted consumption is bounded without losing actual pending readiness");
    probe.read_interruptions = 0;
    probe.read_error = EIO;
    expect(!wake.consume() && errno == EDOM, "failed drain cannot report successful consumption");
    probe.read_error = 0;
    probe.read_interruptions = 7;
    expect(wake.consume() && probe.reads == 17 && errno == EDOM && wake.consume(),
           "real pending hint survives failures then drains on the last permitted attempt");
}
void precise_transport_waits() {
    Endpoint endpoint;
    PollProbe probe;
    poll_probe = &probe;
    const bool precise = endpoint.server.poll_for_ns(100000);
    const bool legacy = endpoint.server.poll_for(1);
    const bool zero = endpoint.server.poll_for_ns(0);
    const auto calls = probe.timeouts;
    probe.timeouts.clear();
    probe.interrupts = true;
    const bool interrupted = endpoint.server.poll_for_ns(10000000);
    const auto retry = probe.timeouts;
    probe.fails = true;
    const bool failed = endpoint.server.poll_for_ns(0);
    const auto before = probe.timeouts.size();
    const bool oversized = endpoint.server.poll_for_ns(
        static_cast<std::uint64_t>(std::numeric_limits<int>::max()) * 1000000 + 1);
    const bool rejected_before_poll = before == probe.timeouts.size();
    poll_probe = nullptr;
    expect(precise && legacy && zero && calls == std::vector<std::uint64_t>{100000, 1000000, 0},
           "actual ppoll receives exact submillisecond, legacy and nonblocking durations");
    expect(interrupted && retry.size() >= 1 && retry.front() == 10000000 &&
               std::ranges::all_of(retry, [](auto value) { return value <= 10000000; }),
           "EINTR retries spend the original deadline instead of restarting the interval");
    expect(!failed && !oversized && rejected_before_poll && !endpoint.server.take_request(),
           "poll failure and overflow reject without manufacturing input");
}

using Journal = ExecutionSocketPollJournal;
struct JournalRows {
    std::vector<Journal::Cycle> cycles;
    std::vector<Journal::Poll> polls;
    std::vector<Journal::Peer> peers;
    std::vector<Journal::Read> reads;
    std::vector<Journal::Frame> frames;
    explicit JournalRows(const Journal& journal)
        : cycles(journal.cycles().begin(), journal.cycles().end()),
          polls(journal.polls().begin(), journal.polls().end()),
          peers(journal.peers().begin(), journal.peers().end()),
          reads(journal.reads().begin(), journal.reads().end()),
          frames(journal.frames().begin(), journal.frames().end()) {}
};
void validate_journal(const JournalRows& rows) {
    std::uint64_t last{};
    std::size_t peer_cursor{}, poll_cursor{};
    for (std::size_t index = 0; index < rows.cycles.size(); ++index) {
        const auto& cycle = rows.cycles[index];
        expect(cycle.id == index + 1 && last <= cycle.entry && cycle.entry <= cycle.exit &&
                   cycle.peer_begin == peer_cursor && cycle.poll_begin == poll_cursor &&
                   cycle.peer_count <= rows.peers.size() - peer_cursor &&
                   cycle.poll_count <= rows.polls.size() - poll_cursor,
               "exact ordered cycle ranges");
        auto poll_end = cycle.entry;
        for (std::size_t offset = 0; offset < cycle.poll_count; ++offset) {
            const auto& poll = rows.polls[poll_cursor++];
            expect(poll.cycle == cycle.id && poll_end <= poll.entry &&
                       poll.entry <= poll.returned && poll.returned <= cycle.exit &&
                       poll.result >= -1 && ((poll.result < 0) == (poll.error != 0)),
                   "exact ppoll attempt clock/result");
            poll_end = poll.returned;
        }
        auto visit_end = poll_end;
        for (std::size_t offset = 0; offset < cycle.peer_count; ++offset) {
            const auto& peer = rows.peers[peer_cursor++];
            expect(peer.cycle == cycle.id && peer.connection && peer.pid && peer.start &&
                       peer.flags <= 3 && (!(peer.flags & 3) || !(peer.events & POLLIN)),
                   "exact authenticated poll interest and exclusion flags");
            if (cycle.success) {
                expect(visit_end <= peer.visit && peer.visit <= cycle.exit && peer.alive <= 1,
                       "ordered original peer traversal");
                visit_end = peer.visit;
            }
        }
        last = cycle.exit;
    }
    expect(peer_cursor == rows.peers.size() && poll_cursor == rows.polls.size(),
           "all peer and syscall records belong to exact cycles");
    std::vector<bool> associated(rows.reads.size());
    std::size_t first_count{};
    for (std::size_t index = 0; index < rows.reads.size(); ++index) {
        const auto& read = rows.reads[index];
        expect(read.cycle && read.cycle <= rows.cycles.size(), "read has exact cycle");
        const auto& cycle = rows.cycles[read.cycle - 1];
        const auto peers = std::span{rows.peers}.subspan(cycle.peer_begin, cycle.peer_count);
        const auto peer = std::ranges::find(peers, read.connection, &Journal::Peer::connection);
        expect(peer != peers.end() && peer->alive && (peer->revents & POLLIN) &&
                   peer->visit <= read.entry && read.entry <= read.returned &&
                   read.returned <= cycle.exit && read.result >= -1 &&
                   ((read.result < 0) == (read.error != 0)),
               "read clock and exact readable peer");
        expect(read.result <= 0 || read.token, "positive read requires stable frame token");
        if (read.token) {
            expect(read.token <= index + 1 && rows.reads[read.token - 1].first_read &&
                       rows.reads[read.token - 1].connection == read.connection,
                   "read token belongs to this peer's first positive syscall");
        }
        if (read.first_read) {
            expect(read.token == index + 1 && read.result > 0 && read.returned <= read.first_read &&
                       read.first_read <= cycle.exit,
                   "first-positive read is never backfilled");
            ++first_count;
        }
    }
    std::uint64_t completed_cycle{}, completed_at{};
    for (const auto& frame : rows.frames) {
        expect(frame.token && frame.token <= rows.reads.size() && !associated[frame.token - 1] &&
                   frame.cycle && frame.cycle <= rows.cycles.size() &&
                   frame.cycle >= completed_cycle && frame.received >= completed_at,
               "unique complete frame association");
        const auto& first = rows.reads[frame.token - 1];
        const auto& cycle = rows.cycles[frame.cycle - 1];
        const auto peers = std::span{rows.peers}.subspan(cycle.peer_begin, cycle.peer_count);
        const auto peer = std::ranges::find(peers, frame.connection, &Journal::Peer::connection);
        expect(peer != peers.end() && peer->pid == frame.pid && peer->start == frame.start &&
                   frame.connection == first.connection && frame.first_read == first.first_read &&
                   frame.first_read && first.cycle <= frame.cycle &&
                   cycle.entry <= frame.received && frame.first_read <= frame.received &&
                   frame.received <= cycle.exit && frame.wire,
               "decoded frame has exact first-read token and authenticated completion identity");
        associated[frame.token - 1] = true;
        completed_cycle = frame.cycle;
        completed_at = frame.received;
    }
    expect(first_count == rows.frames.size(), "no unmatched first-positive read");
    for (const auto& read : rows.reads)
        expect(!read.token || associated[read.token - 1],
               "all partial reads close through their frame");
}
std::shared_ptr<Journal> journal_with(Journal::Limits limits = {64, 512, 128, 4096, 512},
                                      Journal::Clock clock = Journal::system_clock) {
    return std::make_shared<Journal>(limits, clock);
}
void journal_arrival_and_association() {
    auto journal = journal_with();
    Endpoint endpoint(journal);
    auto peer = endpoint.connect();
    const auto queued = monotonic_ns();
    send_bytes(peer.get(), frame(17));
    expect(endpoint.server.poll_for(0), "observe request already queued before poll");
    const auto request = endpoint.server.take_request();
    expect(request && journal->valid() && journal->frames().size() == 1,
           "complete journal frame and unchanged transport receipt");
    const auto initial = journal->frames().front();
    expect(initial.first_read == require_value(request).first_read_ns &&
               initial.received == require_value(request).frame_received_ns &&
               queued <= journal->cycles().back().entry,
           "exact unchanged intake stamps");
    std::atomic<bool> polling{false};
    std::uint64_t sent{};
    bool sent_all{};
    const auto bytes = frame(19, 2);
    PollProbe probe;
    probe.entered = &polling;
    std::thread sender([&] {
        while (!polling.load(std::memory_order_acquire)) std::this_thread::yield();
        timespec now{};
        sent_all = ::clock_gettime(CLOCK_MONOTONIC, &now) == 0;
        sent = static_cast<std::uint64_t>(now.tv_sec) * 1000000000 +
               static_cast<std::uint64_t>(now.tv_nsec);
        sent_all = sent_all &&
                   ::send(peer.get(), bytes.data(), bytes.size(), MSG_DONTWAIT | MSG_NOSIGNAL) ==
                       static_cast<ssize_t>(bytes.size());
    });
    poll_probe = &probe;
    const bool progressed = endpoint.server.poll_for_ns(50000000);
    poll_probe = nullptr;
    sender.join();
    expect(progressed && sent_all && endpoint.server.take_request() && journal->valid(),
           "arrival during actual ppoll call remains an ordinary complete receipt");
    const auto& poll = journal->polls().back();
    expect(poll.entry <= sent && sent <= poll.returned && poll.result > 0,
           "sender timestamp is bracketed by the actual ppoll attempt");
    const JournalRows original(*journal);
    validate_journal(original);
    for (int mutation = 0; mutation < 7; ++mutation) {
        auto changed = original;
        if (mutation == 0) changed.frames[0].token = 0;
        if (mutation == 1) ++changed.frames[0].pid;
        if (mutation == 2) changed.frames.push_back(changed.frames.front());
        if (mutation == 3) changed.frames.pop_back();
        if (mutation == 4) changed.reads[0].cycle = 0;
        if (mutation == 5) changed.reads[0].returned = changed.reads[0].entry - 1;
        if (mutation == 6) std::swap(changed.frames.front(), changed.frames.back());
        bool rejected{};
        try {
            validate_journal(changed);
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        expect(rejected, "independent reconstruction rejects corrupt/foreign/missing associations");
    }
}
void journal_partial_peer_and_credit() {
    auto journal = journal_with();
    auto wake = std::make_shared<ExecutionSocketWake>();
    Endpoint endpoint(journal, wake);
    auto first = endpoint.connect();
    auto second = endpoint.connect();
    expect(wake->notify() && endpoint.server.poll_for(0) && journal->valid(),
           "unrelated owner wake records a cycle without inventing peer readability");
    for (const auto& peer : std::span{journal->peers()}.last(2))
        expect((peer.events & POLLIN) && !peer.revents, "unrelated wake is not target readability");
    send_bytes(first.get(), frame(23));
    send_bytes(second.get(), frame(19));
    ReceiveProbe partial;
    partial.chunk_limit = 1;
    partial.limits_first_only = true;
    expect(progress(endpoint.server, partial),
           "bounded partial peer does not monopolize traversal");
    const auto ordinary = endpoint.server.take_request();
    expect(ordinary && ordinary->connection == 2 && !journal->valid(),
           "dangling first-positive token cannot be complete evidence yet");
    const auto first_read = journal->reads().front();
    const auto delayed = journal->peers().back();
    expect(delayed.connection == 2 && delayed.visit >= journal->reads()[7].returned,
           "later readable peer retains preceding traversal wall interval");
    expect(endpoint.server.poll_for(0), "finish original partial frame in later cycle");
    const auto completed = endpoint.server.take_request();
    expect(completed && completed->connection == 1 && journal->valid(),
           "same partial frame completes once");
    const auto& joined = journal->frames().back();
    expect(joined.token == first_read.token && joined.first_read == first_read.first_read &&
               joined.cycle > first_read.cycle &&
               journal->reads().front().cycle == first_read.cycle,
           "frame identity appended separately without rewriting completed first syscall");
    send_bytes(first.get(), frame(17, 2));
    expect(endpoint.server.poll_for(0), "hold a ready frame without taking it");
    expect(endpoint.server.poll_for(0) && (journal->peers()[journal->peers().size() - 2].flags & 1),
           "already ready frame explicitly excludes further input");
    expect(endpoint.server.take_request().has_value(), "take held frame normally");
    send_bytes(first.get(), frame(17, 3));
    expect(endpoint.server.poll_for(0) && !endpoint.server.take_request(),
           "credit exclusion unchanged");
    const auto& excluded = journal->peers()[journal->peers().size() - 2];
    expect((excluded.flags & 2) && !(excluded.events & POLLIN) && excluded.pending == 2,
           "credit-excluded bytes are not attributed to kernel readiness delay");
    validate_journal(JournalRows(*journal));
    first = ExecutionOwnedDescriptor{};
    expect(endpoint.server.poll_for(0) && !endpoint.server.has_peer(1) && journal->valid(),
           "disconnect removal preserves stable snapshot indexing for following peers");
    validate_journal(JournalRows(*journal));
}
thread_local std::size_t journal_clock_calls{}, journal_clock_fail_after{};
thread_local bool journal_clock_reverses{};
bool counting_journal_clock(std::uint64_t& now) noexcept {
    ++journal_clock_calls;
    const bool result = Journal::system_clock(now);
    if (journal_clock_reverses) now = 1;
    errno = EBUSY;
    return result && (!journal_clock_fail_after || journal_clock_calls < journal_clock_fail_after);
}
void journal_failures_and_legacy_control() {
    journal_clock_calls = journal_clock_fail_after = 0;
    const auto unused = journal_with({64, 512, 128, 4096, 512}, counting_journal_clock);
    {
        Endpoint legacy(true);
        auto peer = legacy.connect();
        send_bytes(peer.get(), frame(17));
        expect(legacy.server.poll_for(0) && legacy.server.take_request() && !journal_clock_calls &&
                   unused->cycles().empty(),
               "legacy constructor never enters a recorder or its clock");
    }
    for (const auto limits :
         {Journal::Limits{}, Journal::Limits{524288, 4194304, 1048576, 4194304, 1048576}}) {
        bool rejected{};
        try {
            const auto invalid = journal_with(limits);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        expect(rejected, "zero or combined overflow capacity rejects before allocation");
    }
    for (std::size_t dimension = 0; dimension < 5; ++dimension) {
        auto limits = Journal::Limits{64, 512, 128, 4096, 512};
        if (dimension == 0) limits.cycles = 1;
        if (dimension == 1) limits.peers = 1;
        if (dimension == 2) limits.polls = 1;
        if (dimension == 3) limits.reads = 1;
        if (dimension == 4) limits.frames = 1;
        auto journal = journal_with(limits);
        Endpoint endpoint(journal);
        auto peer = endpoint.connect();
        for (std::uint64_t sequence = 1; sequence <= 2; ++sequence) {
            send_bytes(peer.get(), frame(17, sequence));
            expect(endpoint.server.poll_for(0), "journal saturation cannot fail transport");
            const auto received = endpoint.server.take_request();
            expect(received && received->message.header.sequence == sequence &&
                       received->message.payload.size() == 17,
                   "saturated journal preserves every payload and sequence");
        }
        expect(!journal->valid() && journal->failure() == Journal::Failure::kCapacity,
               "each independent storage bound invalidates rather than truncates evidence");
    }
    auto journal = journal_with({64, 512, 128, 4096, 512}, counting_journal_clock);
    Endpoint endpoint(journal);
    auto peer = endpoint.connect();
    PollProbe poll;
    poll.interrupts = true;
    poll_probe = &poll;
    const bool retried = endpoint.server.poll_for_ns(100000);
    poll_probe = nullptr;
    expect(retried && journal->valid() && journal->polls()[1].error == EINTR &&
               journal->polls()[1].result == -1,
           "journal clock preserves ppoll EINTR and exact retry rows");
    send_bytes(peer.get(), frame(17));
    ReceiveProbe interrupted;
    interrupted.interrupted = 2;
    expect(progress(endpoint.server, interrupted) && endpoint.server.take_request() &&
               journal->valid() && journal->reads()[0].error == EINTR &&
               journal->reads()[1].error == EINTR,
           "journal clock preserves nonblocking receive retry and errno");
    validate_journal(JournalRows(*journal));
    journal_clock_fail_after = journal_clock_calls + 1;
    send_bytes(peer.get(), frame(17, 2));
    expect(endpoint.server.poll_for(0) && endpoint.server.take_request() && !journal->valid() &&
               journal->failure() == Journal::Failure::kClock,
           "clock failure invalidates only diagnostics");
    journal_clock_fail_after = 0;
    {
        auto reversed = journal_with({64, 512, 128, 4096, 512}, counting_journal_clock);
        Endpoint reversed_endpoint(reversed);
        auto reversed_peer = reversed_endpoint.connect();
        send_bytes(reversed_peer.get(), frame(17));
        journal_clock_reverses = true;
        const bool progressed = reversed_endpoint.server.poll_for(0);
        journal_clock_reverses = false;
        expect(progressed && reversed_endpoint.server.take_request() && !reversed->valid() &&
                   reversed->failure() == Journal::Failure::kClock,
               "reversed clock cannot change transport or qualify diagnostic evidence");
    }
    auto foreign = journal_with();
    Endpoint foreign_endpoint(foreign);
    bool progressed{}, empty_view{};
    std::thread other([&] {
        empty_view = foreign->cycles().empty() && !foreign->valid();
        progressed = foreign_endpoint.server.poll_for(0);
    });
    other.join();
    expect(progressed && empty_view && !foreign->valid() &&
               foreign->failure() == Journal::Failure::kOwner,
           "foreign-thread observation invalidates evidence without changing serialized transport");
}
void journal_descriptors_and_errors() {
    for (const bool late : {false, true}) {
        auto journal = journal_with();
        Endpoint endpoint(journal);
        auto peer = endpoint.connect();
        ExecutionOwnedDescriptor attachment(::open("/dev/null", O_RDONLY | O_CLOEXEC));
        expect(attachment.get() >= 0, "open diagnostic descriptor control");
        const auto bytes = frame(97, 1, ExecutionOpcode::kImportHostMemory);
        send_bytes(peer.get(), std::span{bytes}.first(1), attachment.get());
        send_bytes(peer.get(), std::span{bytes}.subspan(1), late ? attachment.get() : -1);
        expect(endpoint.server.poll_for(0), "diagnostic descriptor intake completes normally");
        const auto received = endpoint.server.take_request();
        if (late) {
            expect(!received && !endpoint.server.has_peer(1) && !journal->valid(),
                   "rejected ancillary frame remains rejected and unassociated");
        } else {
            struct stat original {
            }, transferred{};
            expect(received && received->descriptor.get() >= 0 && journal->valid() &&
                       ::fstat(attachment.get(), &original) == 0 &&
                       ::fstat(received->descriptor.get(), &transferred) == 0 &&
                       original.st_dev == transferred.st_dev &&
                       original.st_ino == transferred.st_ino &&
                       (::fcntl(received->descriptor.get(), F_GETFD) & FD_CLOEXEC) != 0,
                   "diagnostics preserve authenticated descriptor ownership and bytes");
            validate_journal(JournalRows(*journal));
        }
    }
    auto journal = journal_with();
    Endpoint endpoint(journal);
    PollProbe probe;
    probe.fails = true;
    poll_probe = &probe;
    const bool result = endpoint.server.poll_for(0);
    poll_probe = nullptr;
    expect(!result && journal->valid() && !journal->cycles().front().success &&
               journal->polls().front().error == EIO,
           "failed poll is an observed error, never a manufactured transport success");
    validate_journal(JournalRows(*journal));
}

std::uint64_t thread_cpu_ns() {
    timespec value{};
    expect(::clock_gettime(CLOCK_THREAD_CPUTIME_ID, &value) == 0 && value.tv_sec >= 0,
           "read owner thread CPU clock");
    return static_cast<std::uint64_t>(value.tv_sec) * 1000000000ULL +
           static_cast<std::uint64_t>(value.tv_nsec);
}
struct JournalBenchResult {
    std::uint64_t owner_wall_ns{}, owner_cpu_ns{}, whole_wall_ns{}, whole_cpu_ns{};
    std::size_t cycles{}, peers{}, polls{}, reads{}, frames{}, storage_bytes{};
};
JournalBenchResult journal_bench(bool enabled, std::uint64_t iterations,
                                 const std::string& output = {}) {
    const auto whole_wall = monotonic_ns();
    const auto whole_cpu = thread_cpu_ns();
    JournalBenchResult result;
    {
        auto journal = enabled ? journal_with({12000, 84000, 24000, 672000, 84000}) : nullptr;
        auto endpoint =
            enabled ? std::make_unique<Endpoint>(journal) : std::make_unique<Endpoint>(true);
        std::vector<ExecutionOwnedDescriptor> peers;
        for (std::size_t index = 0; index < 7; ++index) peers.push_back(endpoint->connect());
        ExecutionWireReply reply;
        reply.disposition = ExecutionWireDisposition::kComplete;
        const auto payload = encode_execution_reply(reply);
        expect(payload.has_value(), "encode fixed uncontended response payload");
        const std::vector<std::byte> expected_payload(17, std::byte{0x35});
        for (std::uint64_t sequence = 1; sequence <= iterations; ++sequence) {
            const auto bytes = frame(17, sequence, ExecutionOpcode::kLaunch);
            const ExecutionWireMessage response{
                {ExecutionMessageKind::kReply, ExecutionOpcode::kLaunch, 0, 7, sequence, {11, 13}},
                *payload};
            const auto encoded = encode_execution_message(response);
            expect(encoded.has_value(), "encode exact expected reply bytes outside owner interval");
            std::vector<std::byte> observed(encoded->size());
            send_bytes(peers.back().get(), bytes);
            const auto before_wall = monotonic_ns();
            const auto before_cpu = thread_cpu_ns();
            const bool progressed = endpoint->server.poll_for(0);
            const auto request = endpoint->server.take_request();
            const bool replied = request && endpoint->server.reply(request->connection, response);
            result.owner_cpu_ns += thread_cpu_ns() - before_cpu;
            result.owner_wall_ns += monotonic_ns() - before_wall;
            expect(progressed && replied && request->connection == 7 &&
                       request->message.header.sequence == sequence &&
                       request->message.payload == expected_payload &&
                       request->descriptor.get() < 0 && request->first_read_ns &&
                       request->frame_received_ns >= request->first_read_ns &&
                       !endpoint->server.take_request() && !endpoint->server.response_pending(7),
                   "every owner cycle receives and replies exactly once with unchanged bytes");
            std::size_t received{};
            while (received < observed.size()) {
                pollfd ready{peers.back().get(), POLLIN, 0};
                expect(::poll(&ready, 1, 1000) > 0,
                       "bounded wait for complete reply outside owner interval");
                const auto count = ::recv(peers.back().get(), observed.data() + received,
                                          observed.size() - received, MSG_DONTWAIT);
                expect(count > 0, "receive exact ordinary reply extent");
                received += static_cast<std::size_t>(count);
            }
            expect(observed == *encoded,
                   "enabled and legacy paths deliver the same complete reply bytes");
        }
        if (journal) {
            expect(journal->valid() && journal->storage_bytes() == 58944000 &&
                       journal->frames().size() == iterations,
                   "bounded complete benchmark journal");
            validate_journal(JournalRows(*journal));
            result.cycles = journal->cycles().size();
            result.peers = journal->peers().size();
            result.polls = journal->polls().size();
            result.reads = journal->reads().size();
            result.frames = journal->frames().size();
            result.storage_bytes = journal->storage_bytes();
            if (!output.empty())
                expect(journal->publish(output), "publish complete fixture journal");
        }
    }
    result.whole_cpu_ns = thread_cpu_ns() - whole_cpu;
    result.whole_wall_ns = monotonic_ns() - whole_wall;
    return result;
}

void journal_publication() {
    auto journal = journal_with();
    Endpoint endpoint(journal);
    const auto path = endpoint.directory.path + "/journal.bin";
    expect(!journal->publish(path) && !std::filesystem::exists(path),
           "empty evidence cannot be published as a complete journal");
    auto peer = endpoint.connect();
    send_bytes(peer.get(), frame(17));
    expect(endpoint.server.poll_for(0) && endpoint.server.take_request(),
           "complete publication fixture");
    bool foreign{};
    std::thread other([&] { foreign = journal->publish(path); });
    other.join();
    expect(!foreign && !std::filesystem::exists(path),
           "foreign journal publisher cannot create output");
    expect(!journal->publish("relative.bin") &&
               !journal->publish(endpoint.directory.path + "/missing/../journal.bin"),
           "only normalized absolute owned publication paths are accepted");
    PublicationProbe short_writes;
    short_writes.maximum = 17;
    short_writes.interrupted = 2;
    publication_probe = &short_writes;
    const bool published = journal->publish(path);
    publication_probe = nullptr;
    const auto expected = 128 + journal->cycles().size() * sizeof(Journal::Cycle) +
                          journal->polls().size() * sizeof(Journal::Poll) +
                          journal->peers().size() * sizeof(Journal::Peer) +
                          journal->reads().size() * sizeof(Journal::Read) +
                          journal->frames().size() * sizeof(Journal::Frame);
    struct stat status {};
    expect(published && short_writes.bytes == expected && ::lstat(path.c_str(), &status) == 0 &&
               S_ISREG(status.st_mode) && (status.st_mode & 0777) == 0600 && status.st_nlink == 1 &&
               static_cast<std::size_t>(status.st_size) == expected && !journal->publish(path),
           "short/EINTR publication completes exact bounded bytes without overwrite");
    std::ifstream stream(path, std::ios::binary);
    std::array<unsigned char, 128> header{};
    expect(static_cast<bool>(stream.read(reinterpret_cast<char*>(header.data()), header.size())) &&
               std::equal(header.begin(), header.begin() + 8, "GLMPOLL1"),
           "explicit binary magic is independent of struct layout");
    const auto word = [&](std::size_t index) {
        std::uint64_t value{};
        for (std::size_t byte = 0; byte < 8; ++byte)
            value |= static_cast<std::uint64_t>(header[index * 8 + byte]) << (byte * 8);
        return value;
    };
    expect(word(1) == 1 && word(2) == 128 && word(3) == static_cast<std::uint64_t>(::getpid()) &&
               word(4) && word(5) == journal->cycles().size() &&
               word(6) == journal->polls().size() && word(7) == journal->peers().size() &&
               word(8) == journal->reads().size() && word(9) == journal->frames().size() &&
               word(10) == expected - 128 && word(11) == 0 && word(12) == 0 && word(13) == 0 &&
               word(14) == 0 && word(15) == 0,
           "exact little-endian owner, lengths and reserved header fields");
    stream.close();
    const auto alias = endpoint.directory.path + "/alias.bin";
    expect(::symlink(path.c_str(), alias.c_str()) == 0 && !journal->publish(alias) &&
               std::filesystem::file_size(path) == expected && ::unlink(alias.c_str()) == 0,
           "publisher never follows or replaces a destination symlink");
    const auto failed = endpoint.directory.path + "/failed.bin";
    PublicationProbe full_disk;
    full_disk.fail_after = 137;
    publication_probe = &full_disk;
    const bool partial = journal->publish(failed);
    publication_probe = nullptr;
    expect(!partial && std::filesystem::file_size(failed) == 137 && journal->valid() &&
               !journal->publish(failed),
           "failed partial file is retained, never overwritten or qualified");
    const auto restricted = endpoint.directory.path + "/restricted.bin";
    const auto old_mask = ::umask(0777);
    const bool wrong_mode = journal->publish(restricted);
    ::umask(old_mask);
    expect(!wrong_mode && ::lstat(restricted.c_str(), &status) == 0 &&
               (status.st_mode & 0777) == 0 && !journal->publish(restricted),
           "restrictive creation mask cannot produce a successful publication receipt");
    expect(::unlink(path.c_str()) == 0 && ::unlink(failed.c_str()) == 0 &&
               ::unlink(restricted.c_str()) == 0,
           "remove only this fixture's exact temporary publication files");
}

std::size_t descriptors() {
    return static_cast<std::size_t>(
        std::distance(std::filesystem::directory_iterator("/proc/self/fd"),
                      std::filesystem::directory_iterator{}));
}
}  // namespace
int main(int argc, char** argv) {
    try {
        const auto baseline = descriptors();
        if (argc == 3 && std::string_view(argv[1]) == "--poll-journal-export") {
            const auto result = journal_bench(true, 128, argv[2]);
            expect(result.frames == 128 && descriptors() == baseline,
                   "exported multi-buffer fixture closes every descriptor");
            std::cout << "Complete multi-buffer poll journal exported\n";
            return EXIT_SUCCESS;
        }
        if (argc == 3 && std::string_view(argv[1]) == "--poll-journal-bench" &&
            (std::string_view(argv[2]) == "off" || std::string_view(argv[2]) == "on")) {
            const bool enabled = std::string_view(argv[2]) == "on";
            const auto result = journal_bench(enabled, 10000);
            expect(descriptors() == baseline, "benchmark descriptors retire completely");
            std::cout << "{\"enabled\":" << (enabled ? "true" : "false")
                      << ",\"iterations\":10000,\"authenticated_peers\":7,\"active_peers\":1"
                      << ",\"owner_wall_ns\":" << result.owner_wall_ns
                      << ",\"owner_cpu_ns\":" << result.owner_cpu_ns
                      << ",\"whole_wall_ns\":" << result.whole_wall_ns
                      << ",\"whole_cpu_ns\":" << result.whole_cpu_ns
                      << ",\"cycles\":" << result.cycles << ",\"peers\":" << result.peers
                      << ",\"polls\":" << result.polls << ",\"reads\":" << result.reads
                      << ",\"frames\":" << result.frames
                      << ",\"storage_bytes\":" << result.storage_bytes
                      << ",\"correctness_and_cleanup\":true}\n";
            return EXIT_SUCCESS;
        }
        expect(argc == 1, "unknown receive-test arguments");
        small_frame_and_no_read_ahead();
        complete_frame_timestamps(false);
        complete_frame_timestamps(true);
        fragmented_yield_and_other_peer();
        bounded_interrupts_and_would_block();
        large_frame_yields();
        descriptor_handoff_and_late_rejection(false);
        descriptor_handoff_and_late_rejection(true);
        foreign_payload_rejects();
        one_turn_client_reply();
        coalesced_process_local_wakes();
        bounded_wake_failures();
        precise_transport_waits();
        journal_arrival_and_association();
        journal_partial_peer_and_credit();
        journal_failures_and_legacy_control();
        journal_descriptors_and_errors();
        journal_publication();
        (void)journal_bench(false, 96);
        (void)journal_bench(true, 96);
        expect(descriptors() == baseline, "all normal and rejected receive resources retire");
        std::cout << "Bounded execution receive checks passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
