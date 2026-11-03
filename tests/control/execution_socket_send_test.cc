#include "glimmer/control/execution_socket.h"
#include "internal/process_identity.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <fcntl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>

namespace {
// Only the explicitly scoped owner call is intercepted. No fake positive write
// is returned: short writes and SCM_RIGHTS still pass through the actual socket.
struct SendProbe {
    int descriptor{-1}, error{};
    std::size_t attempts{}, positive_writes{}, bytes{}, rights_attempts{}, rights_transfers{};
    std::size_t interruptions{}, chunk_limit{};
    std::uint64_t first_send_ns{}, last_send_ns{};
    bool returns_zero{}, has_invalid_flags{}, has_foreign_descriptor{}, has_invalid_clock{};
};
thread_local SendProbe* send_probe{};

bool injected_failure(int descriptor, SendProbe& probe, int flags) noexcept {
    ++probe.attempts;
    timespec observed{};
    const auto saved_errno = errno;
    if (::clock_gettime(CLOCK_MONOTONIC, &observed) != 0 || observed.tv_sec < 0) {
        probe.has_invalid_clock = true;
    } else {
        probe.last_send_ns = static_cast<std::uint64_t>(observed.tv_sec) * 1000000000ULL +
                             static_cast<std::uint64_t>(observed.tv_nsec);
        if (probe.attempts == 1) probe.first_send_ns = probe.last_send_ns;
    }
    errno = saved_errno;
    if (probe.descriptor < 0) probe.descriptor = descriptor;
    probe.has_foreign_descriptor = probe.has_foreign_descriptor || probe.descriptor != descriptor;
    constexpr auto required = MSG_DONTWAIT | MSG_NOSIGNAL;
    probe.has_invalid_flags = probe.has_invalid_flags || (flags & required) != required;
    if (probe.interruptions) {
        --probe.interruptions;
        errno = EINTR;
        return true;
    }
    if (probe.error) {
        errno = probe.error;
        return true;
    }
    return false;
}
void record_write(SendProbe& probe, ssize_t count, bool has_rights) noexcept {
    if (count <= 0) return;
    ++probe.positive_writes;
    probe.bytes += static_cast<std::size_t>(count);
    if (has_rights) ++probe.rights_transfers;
}
}  // namespace

// NOLINTNEXTLINE(bugprone-reserved-identifier,readability-identifier-naming): GNU ld wrapping ABI.
extern "C" ssize_t __real_send(int descriptor, const void* data, std::size_t bytes, int flags);
// NOLINTNEXTLINE(bugprone-reserved-identifier,readability-identifier-naming): GNU ld wrapping ABI.
extern "C" ssize_t __wrap_send(int descriptor, const void* data, std::size_t bytes, int flags) {
    auto* probe = send_probe;
    if (!probe) return __real_send(descriptor, data, bytes, flags);
    if (injected_failure(descriptor, *probe, flags)) return -1;
    if (probe->returns_zero) return 0;
    if (probe->chunk_limit) bytes = std::min(bytes, probe->chunk_limit);
    const auto count = __real_send(descriptor, data, bytes, flags);
    record_write(*probe, count, false);
    return count;
}
// NOLINTNEXTLINE(bugprone-reserved-identifier,readability-identifier-naming): GNU ld wrapping ABI.
extern "C" ssize_t __real_sendmsg(int descriptor, const msghdr* message, int flags);
// NOLINTNEXTLINE(bugprone-reserved-identifier,readability-identifier-naming): GNU ld wrapping ABI.
extern "C" ssize_t __wrap_sendmsg(int descriptor, const msghdr* message, int flags) {
    auto* probe = send_probe;
    if (!probe) return __real_sendmsg(descriptor, message, flags);
    bool has_rights{};
    auto copy = *message;
    for (auto* item = CMSG_FIRSTHDR(&copy); item; item = CMSG_NXTHDR(&copy, item))
        has_rights =
            has_rights || (item->cmsg_level == SOL_SOCKET && item->cmsg_type == SCM_RIGHTS);
    if (has_rights) ++probe->rights_attempts;
    if (injected_failure(descriptor, *probe, flags)) return -1;
    if (probe->returns_zero) return 0;
    if (copy.msg_iovlen != 1) {
        errno = EINVAL;
        return -1;
    }
    auto data = *copy.msg_iov;
    if (probe->chunk_limit) data.iov_len = std::min(data.iov_len, probe->chunk_limit);
    copy.msg_iov = &data;
    const auto count = __real_sendmsg(descriptor, &copy, flags);
    record_write(*probe, count, has_rights);
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
    explicit ProbeScope(SendProbe& probe) {
        expect(!send_probe, "send probes cannot nest");
        send_probe = &probe;
    }
    ~ProbeScope() {
        send_probe = nullptr;
    }
    ProbeScope(const ProbeScope&) = delete;
    ProbeScope& operator=(const ProbeScope&) = delete;
};
struct Directory {
    std::string path;
    Directory() {
        char pattern[] = "/tmp/glimmer-send-XXXXXX";
        const auto* created = ::mkdtemp(pattern);
        expect(created, "create private send-test directory");
        path = created;
    }
    ~Directory() {
        if (::rmdir(path.c_str()) != 0) std::cerr << "Send-test directory cleanup failed\n";
    }
};
ExecutionSocketConfig server_options(const std::string& path) {
    ExecutionSocketConfig config;
    config.socket_path = path;
    config.max_pending_requests = 2;
    return config;
}
ExecutionSocketClientConfig client_options(const std::string& path) {
    const auto identity = current_process_identity();
    expect(identity.has_value(), "authenticate actual current process");
    return {path, require_value(identity), 3000};
}
std::vector<std::byte> pattern(std::size_t size) {
    std::vector<std::byte> result(size);
    for (std::size_t index = 0; index < size; ++index)
        result[index] = static_cast<std::byte>((index * 17 + 31) % 251);
    return result;
}
ExecutionWireMessage reply_for(const ExecutionReceivedRequest& request,
                               std::span<const std::byte> data = {}) {
    ExecutionWireReply reply;
    reply.disposition = ExecutionWireDisposition::kComplete;
    reply.original_status = 0;
    reply.id = request.message.header.sequence;
    reply.data.assign(data.begin(), data.end());
    const auto payload = encode_execution_reply(reply);
    expect(payload.has_value(), "encode exact successful transport reply");
    auto header = request.message.header;
    header.kind = ExecutionMessageKind::kReply;
    header.session = 7;
    header.incarnation = {11, 13};
    return {header, require_value(payload)};
}
void verify_reply(const ExecutionWireReply& reply, std::uint64_t sequence,
                  std::span<const std::byte> data = {}) {
    expect(reply.disposition == ExecutionWireDisposition::kComplete && reply.status == 0 &&
               reply.original_status == 0 && reply.id == sequence &&
               std::ranges::equal(reply.data, data),
           "actual reply preserves exact sequence, status and every payload byte");
}
struct Endpoint {
    Directory directory;
    ExecutionSocketServer server;
    ExecutionSocketClient client;
    std::uint64_t connection{};

    Endpoint()
        : server(server_options(directory.path + "/socket")),
          client(client_options(directory.path + "/socket")) {
        expect(server.start(), "start actual authenticated send-test endpoint");
    }
    ExecutionReceivedRequest receive(ExecutionSocketSubmission submission) {
        expect(submission.status == ExecutionSocketSubmitStatus::kAccepted,
               "actual client accepts the bounded request");
        for (unsigned int turn = 0; turn < 32; ++turn) {
            expect(client.poll_for(0) && server.poll_for(0), "advance actual request bytes");
            if (auto request = server.take_request()) {
                expect(request->message.header.sequence == submission.sequence &&
                           require_value(request).peer == require_value(current_process_identity()),
                       "dispatch the exact authenticated source request");
                connection = request->connection;
                return std::move(*request);
            }
        }
        throw std::runtime_error("bounded request did not arrive");
    }
    ExecutionWireReply drain_reply(std::uint64_t sequence) {
        for (unsigned int turn = 0; turn < 32; ++turn) {
            if (auto reply = client.take_reply(sequence)) return std::move(*reply);
            expect(server.poll_for(0), "advance retained output through ordinary server polling");
            const bool is_live = client.poll_for(0);
            if (auto reply = client.take_reply(sequence)) return std::move(*reply);
            expect(is_live, "ordinary reply remains live until exact receipt");
        }
        throw std::runtime_error("bounded reply did not arrive");
    }
    void respond(const ExecutionReceivedRequest& request, std::span<const std::byte> data = {}) {
        expect(server.reply(connection, reply_for(request, data)), "accept exact original reply");
        verify_reply(drain_reply(request.message.header.sequence), request.message.header.sequence,
                     data);
    }
    void open() {
        const auto request = receive(client.submit_open());
        respond(request);
        expect(client.session() == 7, "bootstrap retains authenticated session identity");
    }
    void finish_transport() {
        expect(client.submit(ExecutionOpcode::kSnapshot).status ==
                   ExecutionSocketSubmitStatus::kUnavailable,
               "completed Close seals the original client without reconnecting");
        expect(server.poll_for(0) && !server.has_peer(connection),
               "normal client Close retires the authenticated peer");
        server.stop();
        struct stat info {};
        expect(::lstat((directory.path + "/socket").c_str(), &info) != 0 && errno == ENOENT,
               "normal stop removes the owned socket pathname");
    }
    void close() {
        const auto request = receive(client.submit(ExecutionOpcode::kClose));
        respond(request);
        finish_transport();
    }
};
bool server_poll(Endpoint& endpoint, SendProbe& probe) {
    const ProbeScope scope(probe);
    return endpoint.server.poll_for(0);
}
bool client_poll(Endpoint& endpoint, SendProbe& probe) {
    const ProbeScope scope(probe);
    return endpoint.client.poll_for(0);
}
void verify_probe(const SendProbe& probe) {
    expect(!probe.has_invalid_flags && !probe.has_foreign_descriptor && !probe.has_invalid_clock,
           "bounded send uses only its nonblocking SIGPIPE-safe channel");
}
ExecutionSocketSubmission submit_request(Endpoint& endpoint, SendProbe& probe,
                                         ExecutionOpcode operation,
                                         std::span<const std::byte> payload = {},
                                         int descriptor = -1, std::uint64_t* queued_ns = nullptr) {
    const ProbeScope scope(probe);
    return endpoint.client.submit(operation, payload, descriptor, queued_ns);
}
bool client_eager_after_normal_cleanup() {
    Endpoint endpoint;
    std::uint64_t rejected_ns = 99;
    expect(endpoint.client.submit(ExecutionOpcode::kSnapshot, {}, -1, &rejected_ns).status ==
                   ExecutionSocketSubmitStatus::kUnavailable &&
               !rejected_ns,
           "unopened client clears the trusted timestamp without admission");
    endpoint.open();
    SendProbe probe;
    const auto data = pattern(71);
    std::uint64_t queued_ns{};
    const auto submitted =
        submit_request(endpoint, probe, ExecutionOpcode::kSnapshot, data, -1, &queued_ns);
    expect(submitted.status == ExecutionSocketSubmitStatus::kAccepted,
           "submit exact original frame before observing eager delivery");
    std::optional<ExecutionReceivedRequest> request;
    for (unsigned int turn = 0; turn < 4 && !request; ++turn) {
        expect(endpoint.server.poll_for(0), "observe actual request without client polling");
        request = endpoint.server.take_request();
    }
    const bool is_eager =
        request.has_value() && probe.attempts == 1 && probe.positive_writes == 1 && queued_ns &&
        queued_ns <= probe.first_send_ns && probe.bytes == kExecutionHeaderBytes + data.size();
    // The old queued-only policy must also finish normally before comparison.
    if (!request) request = endpoint.receive(submitted);
    expect(request->message.header.sequence == submitted.sequence &&
               require_value(request).message.payload == data &&
               require_value(request).peer == require_value(current_process_identity()),
           "eager or deferred delivery retains every byte and authenticated identity");
    endpoint.connection = require_value(request).connection;
    expect(!endpoint.client.take_reply(submitted.sequence),
           "socket delivery cannot invent a host result");
    rejected_ns = 99;
    expect(endpoint.client.submit(ExecutionOpcode::kClose, {}, -1, &rejected_ns).status ==
                   ExecutionSocketSubmitStatus::kFull &&
               !rejected_ns,
           "Close with a retained receipt clears its no-admission clock without consuming work");
    endpoint.respond(require_value(request));
    request.reset();
    endpoint.close();
    return is_eager && !probe.has_invalid_flags && !probe.has_foreign_descriptor &&
           !probe.has_invalid_clock;
}
void client_backpressure_fifo() {
    Endpoint endpoint;
    endpoint.open();
    SendProbe probe;
    probe.error = EAGAIN;
    const auto first_data = pattern(71), second_data = pattern(29);
    std::uint64_t first_queued_ns{}, second_queued_ns{};
    const auto first = submit_request(endpoint, probe, ExecutionOpcode::kSnapshot, first_data, -1,
                                      &first_queued_ns);
    const auto second = submit_request(endpoint, probe, ExecutionOpcode::kSnapshot, second_data, -1,
                                       &second_queued_ns);
    expect(first.status == ExecutionSocketSubmitStatus::kAccepted &&
               second.status == ExecutionSocketSubmitStatus::kAccepted && first_queued_ns &&
               first_queued_ns <= probe.first_send_ns && second_queued_ns >= first_queued_ns &&
               second.sequence == first.sequence + 1 && probe.attempts == 1 && !probe.bytes &&
               endpoint.server.poll_for(0) && !endpoint.server.take_request(),
           "only an empty queue attempts eager sending; backlog cannot bypass would-block");
    probe.error = 0;
    probe.chunk_limit = 7;
    expect(client_poll(endpoint, probe) && probe.attempts == 2 && probe.bytes == 7 &&
               endpoint.server.poll_for(0) && !endpoint.server.take_request(),
           "positive short request write stays partial and cannot deliver its successor");
    probe.chunk_limit = 0;
    expect(client_poll(endpoint, probe) && probe.attempts == 3 &&
               probe.bytes == kExecutionHeaderBytes + first_data.size() &&
               endpoint.server.poll_for(0),
           "one ordinary poll finishes only the exact retained head");
    auto request = endpoint.server.take_request();
    expect(request && request->message.header.sequence == first.sequence &&
               request->message.payload == first_data,
           "partial head retains original sequence and all bytes");
    endpoint.connection = require_value(request).connection;
    endpoint.respond(require_value(request));
    const auto successor = endpoint.receive(second);
    expect(successor.message.header.sequence == second.sequence &&
               successor.message.payload == second_data,
           "backlogged successor follows the complete original head");
    endpoint.respond(successor);
    verify_probe(probe);
    endpoint.close();
}
void client_interrupted_final_attempt() {
    Endpoint endpoint;
    endpoint.open();
    SendProbe probe;
    probe.interruptions = 7;
    const auto data = pattern(83);
    const auto submitted = submit_request(endpoint, probe, ExecutionOpcode::kSnapshot, data);
    expect(submitted.status == ExecutionSocketSubmitStatus::kAccepted && probe.attempts == 8 &&
               !probe.interruptions && probe.positive_writes == 1 &&
               probe.bytes == kExecutionHeaderBytes + data.size(),
           "the eighth bounded attempt can eagerly deliver a complete original request");
    const auto request = endpoint.receive(submitted);
    expect(request.message.payload == data, "EINTR retries preserve every original request byte");
    endpoint.respond(request);
    verify_probe(probe);
    endpoint.close();
}
void client_hard_eager_failure(bool returns_zero) {
    Endpoint endpoint;
    endpoint.open();
    SendProbe probe;
    probe.error = returns_zero ? 0 : EPIPE;
    probe.returns_zero = returns_zero;
    std::uint64_t queued_ns{};
    const auto submitted =
        submit_request(endpoint, probe, ExecutionOpcode::kSnapshot, {}, -1, &queued_ns);
    expect(submitted.status == ExecutionSocketSubmitStatus::kAccepted && submitted.sequence &&
               probe.attempts == 1 && queued_ns && queued_ns <= probe.first_send_ns &&
               !probe.positive_writes && !endpoint.client.poll_for(0) &&
               !endpoint.client.take_reply(submitted.sequence) &&
               endpoint.client.submit(ExecutionOpcode::kSnapshot).status ==
                   ExecutionSocketSubmitStatus::kUnavailable &&
               endpoint.server.poll_for(0) && !endpoint.server.has_peer(endpoint.connection) &&
               !endpoint.server.take_request(),
           "hard eager failure consumes admission, poisons transport and cannot authorize replay");
    verify_probe(probe);
    endpoint.server.stop();
}
bool send_reply(Endpoint& endpoint, const ExecutionReceivedRequest& request, SendProbe& probe,
                std::span<const std::byte> data = {}, std::uint64_t* queued_ns = nullptr) {
    const ProbeScope scope(probe);
    return endpoint.server.reply(request.connection, reply_for(request, data), queued_ns);
}
std::uint64_t monotonic_ns() {
    timespec observed{};
    expect(::clock_gettime(CLOCK_MONOTONIC, &observed) == 0 && observed.tv_sec >= 0,
           "read independent send-boundary monotonic clock");
    return static_cast<std::uint64_t>(observed.tv_sec) * 1000000000ULL +
           static_cast<std::uint64_t>(observed.tv_nsec);
}

bool eager_receipts_after_normal_cleanup() {
    Endpoint endpoint;
    bool is_eager = true;
    for (const auto operation :
         {ExecutionOpcode::kHello, ExecutionOpcode::kSnapshot, ExecutionOpcode::kClose}) {
        const auto request = endpoint.receive(operation == ExecutionOpcode::kHello
                                                  ? endpoint.client.submit_open()
                                                  : endpoint.client.submit(operation));
        const auto data =
            operation == ExecutionOpcode::kSnapshot ? pattern(512) : std::vector<std::byte>{};
        SendProbe probe;
        expect(send_reply(endpoint, request, probe, data), "queue each actual original reply");
        // No server poll is permitted between reply() and this client observation.
        const bool is_live = endpoint.client.poll_for(0);
        auto reply = endpoint.client.take_reply(request.message.header.sequence);
        is_eager = is_eager && is_live && reply.has_value() && probe.attempts == 1 &&
                   probe.positive_writes == 1 &&
                   !endpoint.server.response_pending(request.connection);
        verify_probe(probe);
        // The old implementation may need its ordinary next poll. Complete that
        // same request and normal Close before evaluating the regression oracle.
        if (!reply) reply = endpoint.drain_reply(request.message.header.sequence);
        verify_reply(*reply, request.message.header.sequence, data);
        expect(!endpoint.client.take_reply(request.message.header.sequence),
               "a verified eager or delayed reply is consumed exactly once");
    }
    endpoint.finish_transport();
    return is_eager;
}

void partial_frame_fifo() {
    Endpoint endpoint;
    endpoint.open();
    const auto first = endpoint.receive(endpoint.client.submit(ExecutionOpcode::kSnapshot));
    const auto second = endpoint.receive(endpoint.client.submit(ExecutionOpcode::kSnapshot));
    const auto first_data = pattern(512), second_data = pattern(37);
    SendProbe probe;
    probe.chunk_limit = 7;
    expect(send_reply(endpoint, first, probe, first_data) && probe.attempts == 1 &&
               probe.bytes == 7 && endpoint.server.response_pending(first.connection),
           "eager positive partial write yields after exactly one syscall");
    expect(
        endpoint.client.poll_for(0) && !endpoint.client.take_reply(first.message.header.sequence),
        "partial header cannot publish a completed reply");
    expect(send_reply(endpoint, second, probe, second_data) && probe.attempts == 1,
           "appending a successor cannot bypass or retry the pending front frame");
    expect(server_poll(endpoint, probe) && probe.attempts == 2 && probe.bytes == 14,
           "next turn resumes only the existing frame offset");
    probe.chunk_limit = 0;
    expect(server_poll(endpoint, probe) && probe.attempts == 3 &&
               endpoint.server.response_pending(first.connection) && endpoint.client.poll_for(0),
           "one positive continuation finishes only the first frame");
    const auto first_reply = endpoint.client.take_reply(first.message.header.sequence);
    expect(first_reply && !endpoint.client.take_reply(second.message.header.sequence),
           "successor remains queued after the first complete wire frame");
    verify_reply(require_value(first_reply), first.message.header.sequence, first_data);
    expect(server_poll(endpoint, probe) && probe.attempts == 4 &&
               !endpoint.server.response_pending(first.connection) && endpoint.client.poll_for(0),
           "successor sends exactly once in its own ordinary poll turn");
    const auto second_reply = endpoint.client.take_reply(second.message.header.sequence);
    expect(second_reply.has_value(), "FIFO successor reaches its exact recipient");
    verify_reply(require_value(second_reply), second.message.header.sequence, second_data);
    verify_probe(probe);
    endpoint.close();
}

void would_block_fifo() {
    Endpoint endpoint;
    endpoint.open();
    const auto first = endpoint.receive(endpoint.client.submit(ExecutionOpcode::kSnapshot));
    const auto second = endpoint.receive(endpoint.client.submit(ExecutionOpcode::kSnapshot));
    const auto data = pattern(41);
    SendProbe probe;
    probe.error = EAGAIN;
    expect(send_reply(endpoint, first, probe, data) && probe.attempts == 1 && !probe.bytes &&
               endpoint.server.response_pending(first.connection),
           "eager would-block retains the complete frame without spinning");
    expect(send_reply(endpoint, second, probe, data) && probe.attempts == 1 &&
               endpoint.client.poll_for(0) &&
               !endpoint.client.take_reply(first.message.header.sequence) &&
               !endpoint.client.take_reply(second.message.header.sequence),
           "backlogged replies remain FIFO and cannot fabricate readiness");
    expect(server_poll(endpoint, probe) && probe.attempts == 2 && !probe.bytes,
           "a continued would-block consumes one attempt in the next turn");
    probe.error = 0;
    expect(server_poll(endpoint, probe) && probe.attempts == 3 &&
               endpoint.server.response_pending(first.connection) && endpoint.client.poll_for(0),
           "readiness resumes the retained head without crossing a frame");
    const auto first_reply = endpoint.client.take_reply(first.message.header.sequence);
    expect(first_reply && !endpoint.client.take_reply(second.message.header.sequence),
           "head receipt arrives before its successor");
    verify_reply(require_value(first_reply), first.message.header.sequence, data);
    expect(server_poll(endpoint, probe) && probe.attempts == 4 && endpoint.client.poll_for(0),
           "next owner turn drains the successor");
    const auto second_reply = endpoint.client.take_reply(second.message.header.sequence);
    expect(second_reply.has_value(), "backpressured successor is not lost");
    verify_reply(require_value(second_reply), second.message.header.sequence, data);
    verify_probe(probe);
    endpoint.close();
}

void queued_timestamps_precede_delivery() {
    Endpoint endpoint;
    endpoint.open();
    const auto first = endpoint.receive(endpoint.client.submit(ExecutionOpcode::kSnapshot));
    const auto second = endpoint.receive(endpoint.client.submit(ExecutionOpcode::kSnapshot));
    SendProbe probe;
    auto invalid = reply_for(first);
    invalid.header.sequence += 100;
    std::uint64_t rejected_ns = 99;
    {
        const ProbeScope scope(probe);
        expect(!endpoint.server.reply(first.connection, invalid, &rejected_ns) && !rejected_ns &&
                   !probe.attempts,
               "unmatched reply clears diagnostic output without queueing or sending");
        rejected_ns = 99;
        invalid.header.kind = ExecutionMessageKind::kRequest;
        expect(!endpoint.server.reply(first.connection, invalid, &rejected_ns) && !rejected_ns &&
                   !probe.attempts,
               "early invalid-envelope rejection also clears its diagnostic output");
    }
    const auto data = pattern(57);
    probe.chunk_limit = 7;
    std::uint64_t first_queued_ns{}, second_queued_ns{};
    const auto first_before = monotonic_ns();
    expect(send_reply(endpoint, first, probe, data, &first_queued_ns),
           "queue timestamped partial reply");
    const auto first_after = monotonic_ns();
    expect(probe.attempts == 1 && probe.bytes == 7 && first_queued_ns >= first_before &&
               first_queued_ns > 0 && first_queued_ns <= probe.first_send_ns &&
               probe.first_send_ns <= first_after,
           "actual insertion timestamp precedes even the first partial eager delivery");
    const auto second_before = monotonic_ns();
    expect(send_reply(endpoint, second, probe, data, &second_queued_ns),
           "queue timestamped successor behind partial output");
    const auto second_after = monotonic_ns();
    expect(probe.attempts == 1 && second_queued_ns >= second_before &&
               second_queued_ns >= first_queued_ns && second_queued_ns <= second_after,
           "backlogged reply records its own insertion without another eager send");
    probe.chunk_limit = 0;
    expect(server_poll(endpoint, probe) && probe.attempts == 2 && endpoint.client.poll_for(0),
           "finish the timestamped partial head");
    const auto first_reply = endpoint.client.take_reply(first.message.header.sequence);
    expect(first_reply && !endpoint.client.take_reply(second.message.header.sequence),
           "timestamp publication cannot bypass FIFO or publish a receipt");
    verify_reply(require_value(first_reply), first.message.header.sequence, data);
    expect(server_poll(endpoint, probe) && probe.attempts == 3 &&
               probe.last_send_ns >= second_queued_ns && endpoint.client.poll_for(0),
           "successor's actual socket delivery follows its original insertion timestamp");
    const auto second_reply = endpoint.client.take_reply(second.message.header.sequence);
    expect(second_reply.has_value(), "timestamped successor completes normally");
    verify_reply(require_value(second_reply), second.message.header.sequence, data);
    verify_probe(probe);
    endpoint.close();
}

bool interrupted_send_after_normal_cleanup(std::size_t interruptions) {
    Endpoint endpoint;
    endpoint.open();
    const auto request = endpoint.receive(endpoint.client.submit(ExecutionOpcode::kSnapshot));
    SendProbe probe;
    probe.interruptions = interruptions;
    expect(send_reply(endpoint, request, probe), "interrupted eager reply remains accepted");
    const bool first_turn =
        probe.attempts == 8 &&
        (interruptions == 9 ? probe.interruptions == 1 && !probe.positive_writes &&
                                  endpoint.server.response_pending(request.connection)
                            : !probe.interruptions && probe.positive_writes == 1 &&
                                  !endpoint.server.response_pending(request.connection));
    bool next_turn = true;
    if (interruptions == 9) {
        expect(endpoint.client.poll_for(0) &&
                   !endpoint.client.take_reply(request.message.header.sequence),
               "only EINTR cannot deliver an invented reply");
        const auto before = probe.attempts;
        expect(server_poll(endpoint, probe), "next poll resumes the same interrupted frame");
        next_turn = probe.attempts - before == 2 && probe.positive_writes == 1 &&
                    !probe.interruptions && !endpoint.server.response_pending(request.connection);
    }
    verify_reply(endpoint.drain_reply(request.message.header.sequence),
                 request.message.header.sequence);
    verify_probe(probe);
    endpoint.close();
    return first_turn && next_turn;
}

void hard_eager_failure(bool returns_zero) {
    Endpoint endpoint;
    endpoint.open();
    const auto request = endpoint.receive(endpoint.client.submit(ExecutionOpcode::kSnapshot));
    SendProbe probe;
    probe.error = returns_zero ? 0 : EPIPE;
    probe.returns_zero = returns_zero;
    std::uint64_t queued_ns{};
    expect(!send_reply(endpoint, request, probe, {}, &queued_ns) && probe.attempts == 1 &&
               queued_ns > 0 && queued_ns <= probe.first_send_ns && !probe.positive_writes &&
               !endpoint.server.has_peer(request.connection) &&
               !endpoint.server.response_pending(request.connection),
           "hard eager failure returns false and closes its exact peer");
    expect(!endpoint.client.poll_for(0) &&
               !endpoint.client.take_reply(request.message.header.sequence) &&
               endpoint.client.submit(ExecutionOpcode::kSnapshot).status ==
                   ExecutionSocketSubmitStatus::kUnavailable,
           "transport loss never manufactures a rejection receipt or permits replay");
    verify_probe(probe);
    endpoint.server.stop();
}

void descriptor_retry_and_partial_transfer() {
    Endpoint endpoint;
    endpoint.open();
    ExecutionOwnedDescriptor original(::open("/dev/null", O_RDONLY | O_CLOEXEC));
    struct stat original_info {};
    expect(original.get() >= 0 && ::fstat(original.get(), &original_info) == 0,
           "create identifiable borrowed descriptor");
    const auto data = pattern(257);
    SendProbe probe;
    probe.interruptions = 9;
    const auto submitted =
        submit_request(endpoint, probe, ExecutionOpcode::kImportHostMemory, data, original.get());
    expect(submitted.status == ExecutionSocketSubmitStatus::kAccepted,
           "descriptor-bearing request copies its payload and duplicates the borrowed handle");
    original = ExecutionOwnedDescriptor{};
    expect(
        probe.attempts == 8 && probe.interruptions == 1 && !probe.bytes && !probe.rights_transfers,
        "eight interrupted sendmsg attempts retain an unsent descriptor");
    probe.error = EAGAIN;
    expect(client_poll(endpoint, probe) && probe.attempts == 10 && !probe.interruptions &&
               !probe.bytes && endpoint.server.poll_for(0) && !endpoint.server.take_request(),
           "remaining EINTR and would-block preserve the same descriptor and original frame");
    probe.error = 0;
    probe.chunk_limit = 1;
    expect(client_poll(endpoint, probe) && probe.attempts == 11 && probe.bytes == 1 &&
               probe.rights_transfers == 1 && endpoint.server.poll_for(0) &&
               !endpoint.server.take_request(),
           "one actual byte transfers rights without prematurely dispatching a partial header");
    probe.chunk_limit = 0;
    expect(client_poll(endpoint, probe) && probe.attempts == 12 &&
               probe.bytes == kExecutionHeaderBytes + data.size() && probe.rights_attempts == 11 &&
               probe.rights_transfers == 1 && endpoint.server.poll_for(0),
           "remaining bytes use send without retransmitting transferred rights");
    auto request = endpoint.server.take_request();
    struct stat received_info {};
    const auto descriptor_flags = request ? ::fcntl(request->descriptor.get(), F_GETFD) : -1;
    expect(request && request->message.header.sequence == submitted.sequence &&
               request->message.payload == data && request->descriptor.get() >= 0 &&
               ::fstat(request->descriptor.get(), &received_info) == 0 &&
               received_info.st_dev == original_info.st_dev &&
               received_info.st_ino == original_info.st_ino && descriptor_flags >= 0 &&
               (descriptor_flags & FD_CLOEXEC) != 0,
           "exact original bytes and one live CLOEXEC descriptor reach the authenticated owner");
    verify_probe(probe);
    endpoint.respond(require_value(request));
    request.reset();
    endpoint.close();
}

void descriptor_hard_failure() {
    Endpoint endpoint;
    endpoint.open();
    ExecutionOwnedDescriptor original(::open("/dev/null", O_RDONLY | O_CLOEXEC));
    expect(original.get() >= 0, "create hard-failure borrowed descriptor");
    SendProbe probe;
    probe.error = EPIPE;
    const auto submitted = submit_request(endpoint, probe, ExecutionOpcode::kImportHostMemory,
                                          pattern(13), original.get());
    expect(submitted.status == ExecutionSocketSubmitStatus::kAccepted,
           "retain descriptor before an actual send failure");
    expect(!client_poll(endpoint, probe) && probe.attempts == 1 && probe.rights_attempts == 1 &&
               !probe.rights_transfers && ::fcntl(original.get(), F_GETFD) >= 0 &&
               !endpoint.client.take_reply(submitted.sequence),
           "hard sendmsg failure discards only the transport-owned duplicate, not its borrower");
    expect(endpoint.server.poll_for(0) && !endpoint.server.has_peer(endpoint.connection) &&
               !endpoint.server.take_request(),
           "failed descriptor frame never reaches application dispatch");
    verify_probe(probe);
    endpoint.server.stop();
}

std::size_t descriptors() {
    return static_cast<std::size_t>(
        std::distance(std::filesystem::directory_iterator("/proc/self/fd"),
                      std::filesystem::directory_iterator{}));
}
}  // namespace

int main(int argc, char** argv) {
    try {
        const bool eager_only = argc == 2 && std::string_view(argv[1]) == "--eager-only";
        const bool client_only = argc == 2 && std::string_view(argv[1]) == "--client-eager-only";
        expect(argc == 1 || eager_only || client_only, "only isolated eager modes are supported");
        const auto baseline = descriptors();
        if (!eager_only) {
            const bool is_client_eager = client_eager_after_normal_cleanup();
            expect(descriptors() == baseline, "client eager regression completes normal cleanup");
            expect(is_client_eager, "accepted request must send without a later client poll");
            if (client_only) {
                std::cout << "Client eager request and normal-cleanup checks passed\n";
                return EXIT_SUCCESS;
            }
            client_backpressure_fifo();
            client_interrupted_final_attempt();
            client_hard_eager_failure(false);
            client_hard_eager_failure(true);
        }
        const bool is_eager = eager_receipts_after_normal_cleanup();
        expect(descriptors() == baseline, "eager regression completes normal descriptor cleanup");
        expect(is_eager, "ready replies require no later server poll after normal cleanup");
        if (!eager_only) {
            partial_frame_fifo();
            would_block_fifo();
            queued_timestamps_precede_delivery();
            const bool exhausted = interrupted_send_after_normal_cleanup(9);
            const bool last_attempt = interrupted_send_after_normal_cleanup(7);
            expect(descriptors() == baseline, "interrupted sends complete normal cleanup");
            expect(exhausted && last_attempt,
                   "exactly eight attempts bound EINTR while preserving next-turn recovery");
            hard_eager_failure(false);
            hard_eager_failure(true);
            descriptor_retry_and_partial_transfer();
            descriptor_hard_failure();
        }
        expect(descriptors() == baseline, "all successful and failed send resources retire");
        std::cout << "Bounded execution send checks passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
