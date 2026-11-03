#include "glimmer/control/native_epoch_protocol.h"
#include "glimmer/control/shared_task_control_client.h"
#include "glimmer/control/task_execution_admission.h"
#include "glimmer/control/task_execution_endpoint.h"
#include "glimmer/control/task_execution_protocol.h"
#include "glimmer/control/unix_socket_task_execution_server.h"
#include "internal/task_execution_endpoint_test_hooks.h"
#include "internal/owned_control_exchange.h"
#include "internal/task_recovery_endpoint.h"
#include "internal/task_recovery_client.h"
#include "internal/task_recovery_transport_test_hooks.h"
#include "../../src/core/dispatch_test_hooks.h"

#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <future>
#include <iostream>
#include <memory>
#include <poll.h>
#include <string>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

namespace {
std::atomic<int> fail_path_stat_after{0};
std::atomic<bool> replace_path_at_stat{false};
std::atomic<bool> fail_listen{false};
std::atomic<bool> fail_next_entropy{false};
const char* replacement_path = nullptr;
const char* socket_path = nullptr;
}  // namespace

// GNU ld --wrap requires these exact names.
// NOLINTBEGIN(bugprone-reserved-identifier,readability-identifier-naming)
extern "C" int __real_lstat(const char*, struct stat*);
extern "C" int __real_listen(int, int);
extern "C" ssize_t __real_getrandom(void*, size_t, unsigned int);
extern "C" ssize_t __wrap_getrandom(void* buffer, size_t length, unsigned int flags) {
    if (fail_next_entropy.exchange(false)) {
        errno = EAGAIN;
        return -1;
    }
    return __real_getrandom(buffer, length, flags);
}
extern "C" int __wrap_lstat(const char* path, struct stat* status) {
    if (fail_path_stat_after.load() > 0 && fail_path_stat_after.fetch_sub(1) == 1) {
        if (replace_path_at_stat.exchange(false)) {
            if (replacement_path == nullptr || socket_path == nullptr ||
                ::rename(replacement_path, socket_path) != 0) {
                std::abort();
            }
            return __real_lstat(path, status);
        }
        errno = EIO;
        return -1;
    }
    return __real_lstat(path, status);
}
extern "C" int __wrap_listen(int fd, int backlog) {
    if (fail_listen.exchange(false)) {
        if (replacement_path != nullptr && socket_path != nullptr &&
            ::rename(replacement_path, socket_path) != 0) {
            std::abort();
        }
        errno = EIO;
        return -1;
    }
    return __real_listen(fd, backlog);
}
// NOLINTEND(bugprone-reserved-identifier,readability-identifier-naming)

namespace {
using namespace glimmer::control;

struct Resources {
    std::atomic<unsigned> registered{0};
    std::atomic<unsigned> rolled_back{0};
    std::atomic<bool> fail_next_format{false};
    std::atomic<bool> fail_next_dispatch{false};
};

bool register_resource(void* context, glimmer::core::TaskId) noexcept {
    auto* resources = static_cast<Resources*>(context);
    resources->registered.fetch_add(1);
    if (resources->fail_next_format.exchange(false)) {
        testing::fail_next_endpoint_response_format();
    }
    if (resources->fail_next_dispatch.exchange(false)) {
        glimmer::core::internal::dispatch_test_faults =
            glimmer::core::internal::kDispatchFailSnapshot |
            glimmer::core::internal::kDispatchFailCancel;
    }
    return true;
}

void rollback_resource(void* context, glimmer::core::TaskId) noexcept {
    static_cast<Resources*>(context)->rolled_back.fetch_add(1);
}

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "Test failed: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

int connect_client(const std::string& path) {
    const int fd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    expect(fd >= 0, "create raw client");
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    std::memcpy(address.sun_path, path.c_str(), path.size() + 1);
    expect(::connect(fd, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) == 0,
           "connect raw client");
    return fd;
}

void reject_raw(const std::string& path, std::string_view line) {
    const int fd = connect_client(path);
    expect(::send(fd, line.data(), line.size(), MSG_NOSIGNAL) == static_cast<ssize_t>(line.size()),
           "send raw request");
    char byte{};
    expect(::recv(fd, &byte, 1, 0) <= 0, "raw domain request receives no response");
    static_cast<void>(::close(fd));
}

void test_uncertain_dispatch_is_not_replayed() {
    char directory[] = "/tmp/glimmer-exec-uncertain-XXXXXX";
    expect(::mkdtemp(directory) != nullptr, "create private uncertain-dispatch directory");
    const std::string path = std::string(directory) + "/server.sock";
    Resources resources;
    auto admission = std::make_unique<TaskExecutionAdmissionService>(
        100, glimmer::core::SchedulerOptions{}, std::chrono::milliseconds::zero(), true);
    TaskExecutionEndpoint endpoint(std::move(admission), register_resource, rollback_resource,
                                   &resources);
    UnixSocketTaskExecutionServer server(endpoint, {.socket_path = path, .io_timeout_ms = 1000});
    expect(server.start(), "start uncertain-dispatch server");
    std::atomic<bool> running{true};
    std::thread acceptor([&] {
        while (running.load()) {
            expect(server.serve_one_for(20) != UnixSocketServeStatus::kError,
                   "serve uncertain-dispatch client");
        }
    });
    {
        SharedTaskControlClient client(path);
        resources.fail_next_dispatch.store(true);
        expect(!client.request("GLIMMER_TASK_EXEC_V1 EXEC_ACQUIRE tenant 20 1 1\n"),
               "uncertain dispatch returns a lost response");
        const auto status = endpoint.lease_status();
        expect(status && status->pending_cooperative == 1 && status->reaping == 1 &&
                   resources.registered.load() == 1 && resources.rolled_back.load() == 0,
               "client does not replay or release uncertain ownership");
    }
    expect(endpoint.begin_shutdown(), "stop admission before teardown");
    running.store(false);
    acceptor.join();
    server.stop();
    // The test registrar owns no external resources and no backend work exists.
    // The non-reusable, poisoned accounting is destroyed only after joining.
    expect(::rmdir(directory) == 0, "remove private uncertain-dispatch directory");
}

void test_owner_reply_hint_channel() {
    using namespace glimmer::control::internal;
    char directory[] = "/tmp/glimmer-exec-owner-XXXXXX";
    expect(::mkdtemp(directory) != nullptr, "create private owner-channel directory");
    const std::string path = std::string(directory) + "/server.sock";
    Resources resources;
    auto admission = std::make_unique<TaskExecutionAdmissionService>(
        100, glimmer::core::SchedulerOptions{}, std::chrono::milliseconds::zero(), true);
    TaskExecutionEndpoint endpoint(std::move(admission), register_resource, rollback_resource,
                                   &resources);
    UnixSocketTaskExecutionServer server(endpoint, {.socket_path = path, .io_timeout_ms = 1000});
    expect(server.start(), "start owner-channel server");
    std::atomic<bool> running{true};
    std::thread acceptor([&] {
        while (running.load()) {
            expect(server.serve_one_for(20) != UnixSocketServeStatus::kError,
                   "serve owner-channel client");
        }
    });
    {
        ServiceIncarnationBinding binding;
        OwnedControlExchange client(binding);
        expect(client.initialize({path, std::chrono::milliseconds(1000)}) ==
                   OwnedExchangeColdStatus::kReady,
               "owner channel negotiates authenticated V3");
        const auto exchange = [&](std::string_view request) {
            const auto publication = client.begin(request);
            expect(publication.status == SharedControlPublication::kPublished,
                   "owner channel publishes exactly once");
            std::string response;
            for (;;) {
                const auto result = client.poll(publication.ticket, response);
                if (result == OwnedExchangeProgress::kResponse) return response;
                expect(result == OwnedExchangeProgress::kPending, "owner response remains valid");
                const auto wait = client.wait_state(publication.ticket);
                expect(wait.has_value(), "pending owner response exposes a wait descriptor");
                pollfd descriptor{.fd = wait->descriptor, .events = POLLIN, .revents = 0};
                expect(::poll(&descriptor, 1, 1000) > 0, "owner response hint arrives");
            }
        };
        const auto lease = parse_task_execution_protocol_response(
            exchange("GLIMMER_TASK_EXEC_V1 EXEC_ACQUIRE tenant 20 1 1\n"));
        expect(lease.parsed() && lease.response &&
                   lease.response->kind == TaskExecutionProtocolResponseKind::kExecutionLease,
               "owner channel receives a cooperative grant");
        const auto granted = lease.response.value_or(TaskExecutionProtocolResponse{});
        const auto done = "GLIMMER_TASK_EXEC_V1 EXEC_COMPLETE " + std::to_string(granted.task_id) +
                          " " + std::to_string(granted.generation) + "\n";
        const auto completed = parse_task_execution_protocol_response(exchange(done));
        expect(completed.parsed() && completed.response &&
                   completed.response->state == TaskProtocolState::kCompleted,
               "owner channel receives a terminal acknowledgment");
        expect(client.close_idle(), "close idle owner channel");
    }
    expect(endpoint.begin_shutdown(), "stop owner-channel admission");
    running.store(false);
    acceptor.join();
    server.stop();
    const auto status = endpoint.lease_status();
    expect(status && status->queued_tasks == 0 && status->running_tasks == 0,
           "owner-channel tasks retire before teardown");
    expect(::rmdir(directory) == 0, "remove private owner-channel directory");
}

void test_recovery_after_lost_replies() {
    using namespace glimmer::control::internal;
    char directory[] = "/tmp/glimmer-recovery-server-XXXXXX";
    expect(::mkdtemp(directory) != nullptr, "create recovery socket directory");
    const std::string path = std::string(directory) + "/server.sock";
    TaskRecoveryEndpoint endpoint({91, 92}, 100);
    UnixSocketTaskExecutionServer server(endpoint, {.socket_path = path, .io_timeout_ms = 1000});
    expect(server.start(), "start recovery server");
    std::atomic<bool> serving{true};
    std::atomic<unsigned> accepted_connections{0};
    std::thread acceptor([&] {
        while (serving.load()) {
            const auto result = server.serve_one_for(20);
            expect(result != UnixSocketServeStatus::kError, "serve recovery clients");
            if (result == UnixSocketServeStatus::kServed) accepted_connections.fetch_add(1);
        }
    });
    fail_next_entropy.store(true);
    TaskRecoveryClient no_entropy(path);
    expect(!no_entropy.open() && accepted_connections.load() == 0,
           "entropy failure never sends OPEN");
    SharedTaskControlClient client(path);
    auto exchange = [&](const TaskRecoveryRequest& request) -> std::optional<std::string> {
        const auto wire = format_task_recovery_request(request);
        expect(wire.has_value(), "format recovery request");
        const auto line = wire.value_or(TaskRecoveryLine{});
        return client.request(line.view());
    };
    auto decode = [&](const TaskRecoveryRequest& request) -> TaskRecoveryResponse {
        const auto reply = exchange(request);
        expect(reply.has_value(), "receive recovery response");
        const auto parsed = parse_task_recovery_response(reply.value_or(""));
        expect(parsed.has_value(), "decode recovery response");
        return parsed.value_or(TaskRecoveryResponse{});
    };
    TaskRecoveryRequest request;
    request.nonce = {10, 11};
    testing::drop_next_recovery_response();
    expect(!exchange(request), "opened session reply is lost");
    const auto opened = decode(request);
    expect(opened.kind == TaskRecoveryResultKind::kOpened,
           "same recovery session opens after reconnect");
    request.command = TaskRecoveryCommand::kExecute;
    request.incarnation = opened.incarnation;
    request.session_id = opened.session_id;
    request.operation_id = 1;
    request.operation = TaskRecoveryOperation::kSubmit;
    request.tenant = "worker";
    request.memory_bytes = 10;
    request.weight = 1;
    request.work_units = 1;
    testing::drop_next_recovery_response();
    expect(!exchange(request), "committed submit reply is lost");
    const auto accepted = decode(request);
    expect(accepted.kind == TaskRecoveryResultKind::kAccepted, "submit recovered after reconnect");
    request.operation_id = 2;
    request.acknowledged_through = 1;
    request.operation = TaskRecoveryOperation::kClaim;
    request.tenant = {};
    request.memory_bytes = 0;
    request.weight = 0;
    request.work_units = 0;
    request.execution.task_id = accepted.execution.task_id;
    const auto foreign_request = format_task_recovery_request(request);
    expect(foreign_request.has_value(), "format foreign process probe");
    const std::string foreign_line(foreign_request.value_or(TaskRecoveryLine{}).view());
    const pid_t child = ::fork();
    expect(child >= 0, "start separate process probe");
    if (child == 0) {
        ::execl("/proc/self/exe", "glimmer-recovery-probe", "--recovery-peer-probe", path.c_str(),
                foreign_line.c_str(), static_cast<char*>(nullptr));
        ::_exit(127);
    }
    int child_status = 0;
    expect(::waitpid(child, &child_status, 0) == child && WIFEXITED(child_status) &&
               WEXITSTATUS(child_status) == 0,
           "different process cannot take over an existing session");
    testing::drop_next_recovery_response();
    expect(!exchange(request), "committed grant reply is lost");
    const auto grant = decode(request);
    expect(grant.kind == TaskRecoveryResultKind::kGranted, "same grant recovered after reconnect");
    request.operation_id = 3;
    request.acknowledged_through = 2;
    expect(decode(request).kind == TaskRecoveryResultKind::kRejected,
           "new operation cannot obtain a second grant");
    request.operation_id = 4;
    request.acknowledged_through = 3;
    request.operation = TaskRecoveryOperation::kComplete;
    request.execution = grant.execution;
    expect(decode(request).kind == TaskRecoveryResultKind::kAcknowledged,
           "complete recovered work");
    request.operation_id = 5;
    request.acknowledged_through = 4;
    request.operation = TaskRecoveryOperation::kForget;
    request.execution.generation = 0;
    expect(decode(request).kind == TaskRecoveryResultKind::kAcknowledged, "forget terminal task");
    request.command = TaskRecoveryCommand::kAcknowledge;
    request.operation = TaskRecoveryOperation::kStatus;
    request.operation_id = 0;
    request.acknowledged_through = 5;
    request.execution = {};
    expect(decode(request).kind == TaskRecoveryResultKind::kAcknowledged, "acknowledge results");
    request.command = TaskRecoveryCommand::kClose;
    request.acknowledged_through = 0;
    testing::drop_next_recovery_response();
    expect(!exchange(request), "close reply is lost");
    expect(decode(request).kind == TaskRecoveryResultKind::kClosed,
           "close recovered after reconnect");

    TaskRecoveryClient guarded(path);
    expect(guarded.open(), "guarded client opens a separate session");
    TaskRecoveryRequest operation;
    operation.operation = TaskRecoveryOperation::kSubmit;
    operation.tenant = "guarded";
    operation.memory_bytes = 10;
    operation.weight = 1;
    operation.work_units = 1;
    testing::drop_next_recovery_response();
    expect(!guarded.execute(operation), "guarded submit sees lost response");
    auto changed = operation;
    changed.memory_bytes = 11;
    expect(!guarded.execute(changed), "ambiguous operation cannot change payload");
    const auto guarded_accepted = guarded.execute(operation);
    expect(guarded_accepted && guarded_accepted->kind == TaskRecoveryResultKind::kAccepted,
           "guarded submit recovers original result");
    expect(!guarded.execute(operation), "delivered result is not returned twice");
    testing::drop_next_recovery_response();
    expect(!guarded.acknowledge() && guarded.acknowledge(), "lost ACK is idempotently retried");
    operation = {};
    operation.operation = TaskRecoveryOperation::kClaim;
    operation.execution.task_id =
        guarded_accepted.value_or(TaskRecoveryResponse{}).execution.task_id;
    testing::drop_next_recovery_response();
    expect(!guarded.execute(operation), "guarded grant sees lost response");
    const auto guarded_grant = guarded.execute(operation);
    expect(guarded_grant && guarded_grant->kind == TaskRecoveryResultKind::kGranted &&
               !guarded.execute(operation),
           "grant is delivered once after reconnect");
    expect(guarded.acknowledge(), "acknowledge accepted grant");
    operation.operation = TaskRecoveryOperation::kComplete;
    operation.execution = guarded_grant.value_or(TaskRecoveryResponse{}).execution;
    expect(guarded.execute(operation).has_value() && guarded.acknowledge(),
           "complete guarded work");
    operation.operation = TaskRecoveryOperation::kForget;
    operation.execution.generation = 0;
    expect(guarded.execute(operation).has_value() && guarded.acknowledge(), "forget guarded work");
    testing::drop_next_recovery_response();
    expect(!guarded.close() && guarded.close(), "guarded close recovers its lost reply");

    TaskRecoveryClient first_instance(path);
    TaskRecoveryClient second_instance(path);
    expect(first_instance.open() && second_instance.open(),
           "client instances open distinct sessions");
    operation = {};
    operation.operation = TaskRecoveryOperation::kSubmit;
    operation.tenant = "separate";
    operation.memory_bytes = 10;
    operation.weight = 1;
    operation.work_units = 1;
    const auto first_task = first_instance.execute(operation);
    const auto second_task = second_instance.execute(operation);
    expect(first_task && second_task &&
               first_task.value_or(TaskRecoveryResponse{}).execution.task_id !=
                   second_task.value_or(TaskRecoveryResponse{}).execution.task_id,
           "client instances never share the same task grant");
    expect(first_instance.acknowledge() && second_instance.acknowledge(),
           "acknowledge both submissions");
    operation = {};
    operation.operation = TaskRecoveryOperation::kCancelPending;
    operation.execution.task_id = first_task.value_or(TaskRecoveryResponse{}).execution.task_id;
    expect(first_instance.execute(operation).has_value() && first_instance.acknowledge(),
           "cancel first pending task");
    operation.operation = TaskRecoveryOperation::kForget;
    expect(first_instance.execute(operation).has_value() && first_instance.acknowledge() &&
               first_instance.close(),
           "retire first independent session");
    operation.operation = TaskRecoveryOperation::kCancelPending;
    operation.execution.task_id = second_task.value_or(TaskRecoveryResponse{}).execution.task_id;
    expect(second_instance.execute(operation).has_value() && second_instance.acknowledge(),
           "cancel second pending task");
    operation.operation = TaskRecoveryOperation::kForget;
    expect(second_instance.execute(operation).has_value() && second_instance.acknowledge() &&
               second_instance.close(),
           "retire second independent session");
    serving.store(false);
    acceptor.join();
    server.stop();
    expect(endpoint.retire_after_backend_stopped(), "retire stopped recovery endpoint");

    TaskRecoveryEndpoint replacement({93, 94}, 100);
    UnixSocketTaskExecutionServer restarted(replacement,
                                            {.socket_path = path, .io_timeout_ms = 1000});
    expect(restarted.start(), "restart recovery server with fresh incarnation");
    serving.store(true);
    std::thread restarted_acceptor([&] {
        while (serving.load()) {
            expect(restarted.serve_one_for(20) != UnixSocketServeStatus::kError,
                   "serve restarted recovery clients");
        }
    });
    expect(!exchange(request) && !exchange(request),
           "old client cannot reconnect to restarted service");
    TaskRecoveryClient fresh(path);
    expect(fresh.open(), "fresh client opens after restart");
    operation = {};
    operation.operation = TaskRecoveryOperation::kStatus;
    const auto fresh_status = fresh.execute(operation);
    expect(fresh_status && fresh_status->kind == TaskRecoveryResultKind::kStatus &&
               fresh_status->pending_tasks == 0 && fresh_status->active_tasks == 0 &&
               fresh.acknowledge() && fresh.close(),
           "old requests made no task changes in new service");
    serving.store(false);
    restarted_acceptor.join();
    restarted.stop();
    expect(replacement.retire_after_backend_stopped(), "retire restarted endpoint");
    expect(::rmdir(directory) == 0, "remove recovery socket directory");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 4 && std::string_view(argv[1]) == "--recovery-peer-probe") {
        SharedTaskControlClient foreign(argv[2]);
        return foreign.request(argv[3]).has_value() ? EXIT_FAILURE : EXIT_SUCCESS;
    }
    test_uncertain_dispatch_is_not_replayed();
    test_owner_reply_hint_channel();
    char directory[] = "/tmp/glimmer-exec-server-XXXXXX";
    expect(::mkdtemp(directory) != nullptr, "create private socket directory");
    const std::string path = std::string(directory) + "/server.sock";
    constexpr NativeEpochExtent extent{4, 2, 2, 1000000};
    Resources resources;
    auto admission = std::make_unique<TaskExecutionAdmissionService>(
        100, glimmer::core::SchedulerOptions{}, std::chrono::milliseconds::zero(), true, nullptr,
        nullptr, nullptr, nullptr, extent);
    TaskExecutionEndpoint endpoint(std::move(admission), register_resource, rollback_resource,
                                   &resources);
    {
        glimmer::core::Scheduler legacy_scheduler(100);
        TaskAdmissionService legacy_admission(legacy_scheduler);
        TaskControlEndpoint legacy_endpoint(legacy_admission, register_resource, &resources);
        UnixSocketControlServer legacy_server(legacy_endpoint,
                                              {.socket_path = path, .allowed_uid = std::nullopt});
        fail_listen.store(true);
        expect(!legacy_server.start(), "legacy server rejects listen failure");
        struct stat status {};
        expect(::lstat(path.c_str(), &status) != 0 && errno == ENOENT,
               "legacy server removes its identified socket after listen failure");
    }
    {
        UnixSocketTaskExecutionServer failed(endpoint, {.socket_path = path});
        fail_path_stat_after.store(2);
        expect(!failed.start(), "post-bind identity failure is rejected");
        struct stat orphan {};
        expect(::lstat(path.c_str(), &orphan) == 0 && S_ISSOCK(orphan.st_mode),
               "unknown identity is retained instead of blindly unlinked");
        expect(::unlink(path.c_str()) == 0, "remove test-owned orphan");
        fail_listen.store(true);
        expect(!failed.start(), "listen failure is rejected");
        expect(::lstat(path.c_str(), &orphan) != 0 && errno == ENOENT,
               "identified path is removed after listen failure");
        const std::string replacement = std::string(directory) + "/replacement";
        int replacement_fd = ::open(replacement.c_str(), O_CREAT | O_EXCL | O_WRONLY, 0600);
        expect(replacement_fd >= 0 && ::close(replacement_fd) == 0, "create test replacement");
        replacement_path = replacement.c_str();
        socket_path = path.c_str();
        replace_path_at_stat.store(true);
        fail_path_stat_after.store(2);
        expect(!failed.start(), "replacement before identity capture is rejected");
        expect(::lstat(path.c_str(), &orphan) == 0 && S_ISREG(orphan.st_mode),
               "replacement before identity capture is preserved");
        expect(::unlink(path.c_str()) == 0, "remove test-owned pre-stat replacement");
        replacement_fd = ::open(replacement.c_str(), O_CREAT | O_EXCL | O_WRONLY, 0600);
        expect(replacement_fd >= 0 && ::close(replacement_fd) == 0, "recreate test replacement");
        fail_listen.store(true);
        expect(!failed.start(), "listen failure after replacement is rejected");
        replacement_path = nullptr;
        socket_path = nullptr;
        expect(::lstat(path.c_str(), &orphan) == 0 && S_ISREG(orphan.st_mode),
               "replacement path is not removed");
        expect(::unlink(path.c_str()) == 0, "remove test-owned replacement");
    }
    UnixSocketTaskExecutionServer server(endpoint, {.socket_path = path, .io_timeout_ms = 1000});
    expect(server.start(), "start execution server");
    std::atomic<bool> serving{true};
    std::thread acceptor([&] {
        while (serving.load()) {
            expect(server.serve_one_for(20) != UnixSocketServeStatus::kError, "serve clients");
        }
    });

    reject_raw(path, "GLIMMER_TASK_EXEC_V1 EXEC_ACQUIRE tenant 20 1 1\n");
    reject_raw(path, std::string(300, 'x') + "\nGLIMMER_TASK_V1 SUBMIT tenant 1 1 1\n");
    expect(resources.registered.load() == 0, "unnegotiated requests cannot register resources");

    SharedTaskControlClient client(path);
    resources.fail_next_format.store(true);
    expect(!client.request("GLIMMER_TASK_EXEC_V1 EXEC_ACQUIRE tenant 20 1 1\n") &&
               resources.rolled_back.load() == 1,
           "endpoint formatting failure rolls back exactly once");
    const auto acquired = client.request("GLIMMER_TASK_EXEC_V1 EXEC_ACQUIRE tenant 20 1 1\n");
    expect(acquired.has_value(), "execute cooperative acquire");
    const auto lease = parse_task_execution_protocol_response(acquired.value_or(""));
    expect(lease.parsed() && lease.response.value_or(TaskExecutionProtocolResponse{}).kind ==
                                 TaskExecutionProtocolResponseKind::kExecutionLease,
           "cooperative lease decodes");
    const auto lease_value = lease.response.value_or(TaskExecutionProtocolResponse{});
    const std::string done = "GLIMMER_TASK_EXEC_V1 EXEC_COMPLETE " +
                             std::to_string(lease_value.task_id) + " " +
                             std::to_string(lease_value.generation) + "\n";
    const auto completed = client.request(done);
    expect(completed && parse_task_execution_protocol_response(*completed).parsed(),
           "cooperative lease completes");

    const auto legacy = client.request("GLIMMER_TASK_V1 STATS\n");
    expect(legacy && parse_task_protocol_response(*legacy).parsed(), "legacy wire routes");

    NativeEpochRequest epoch;
    epoch.request_sequence = 1;
    epoch.admission = {.tenant_id = "tenant", .memory_bytes = 20, .work_units = 4};
    epoch.extent = extent;
    const auto encoded = format_native_epoch_request(epoch);
    expect(encoded.has_value(), "encode native request");
    const auto epoch_reply = client.request(encoded.value_or(""));
    expect(epoch_reply.has_value(), "native acquire response");
    const auto grant = parse_native_epoch_response(epoch_reply.value_or(""));
    expect(grant && grant->kind == NativeEpochResponseKind::kGranted, "native grant decodes");
    epoch.operation = NativeEpochOperation::kComplete;
    epoch.execution = grant.value_or(NativeEpochResponse{}).grant.execution;
    epoch.issued_calls = 4;
    epoch.request_sequence = 2;
    const auto completion = format_native_epoch_request(epoch);
    expect(completion.has_value(), "encode native completion");
    const auto native_done = client.request(completion.value_or(""));
    const auto native_status =
        native_done ? parse_native_epoch_response(*native_done) : std::nullopt;
    expect(native_status && native_status->kind == NativeEpochResponseKind::kAcknowledged,
           "native lease completes");
    const auto queued = client.request("GLIMMER_TASK_EXEC_V1 SUBMIT tenant 10 1 1\n");
    const auto queued_result = parse_task_execution_protocol_response(queued.value_or(""));
    expect(queued_result.parsed() &&
               queued_result.response.value_or(TaskExecutionProtocolResponse{}).kind ==
                   TaskExecutionProtocolResponseKind::kAccepted,
           "cooperative task is queued");
    const std::string claim =
        "GLIMMER_TASK_EXEC_V1 EXEC_CLAIM " +
        std::to_string(queued_result.response.value_or(TaskExecutionProtocolResponse{}).task_id) +
        "\n";
    SharedTaskControlClient second_client(path);
    auto first_claim = std::async(std::launch::async, [&] { return client.request(claim); });
    auto second_claim =
        std::async(std::launch::async, [&] { return second_client.request(claim); });
    const auto first_result =
        parse_task_execution_protocol_response(first_claim.get().value_or(""));
    const auto second_result =
        parse_task_execution_protocol_response(second_claim.get().value_or(""));
    expect(first_result.parsed() && second_result.parsed(), "both claim responses decode");
    const auto a = first_result.response.value_or(TaskExecutionProtocolResponse{});
    const auto b = second_result.response.value_or(TaskExecutionProtocolResponse{});
    expect((a.kind == TaskExecutionProtocolResponseKind::kExecutionLease &&
            b.kind == TaskExecutionProtocolResponseKind::kEmpty) ||
               (b.kind == TaskExecutionProtocolResponseKind::kExecutionLease &&
                a.kind == TaskExecutionProtocolResponseKind::kEmpty),
           "two connections cannot duplicate a grant");
    const auto winner = a.kind == TaskExecutionProtocolResponseKind::kExecutionLease ? a : b;
    const std::string finish_claim = "GLIMMER_TASK_EXEC_V1 EXEC_COMPLETE " +
                                     std::to_string(winner.task_id) + " " +
                                     std::to_string(winner.generation) + "\n";
    const auto claim_done = client.request(finish_claim);
    expect(claim_done && parse_task_execution_protocol_response(*claim_done).parsed(),
           "winning claim completes");
    expect(resources.registered.load() == 4 && resources.rolled_back.load() == 1,
           "published tasks are not rolled back");

    expect(endpoint.begin_shutdown(), "stop new authorization before transport teardown");
    serving.store(false);
    acceptor.join();
    server.stop();
    const auto status_after_join = endpoint.lease_status();
    expect(status_after_join && status_after_join->queued_tasks == 0 &&
               status_after_join->running_tasks == 0 && status_after_join->reaping == 0,
           "transport is joined before final empty lease snapshot");
    expect(::access(path.c_str(), F_OK) != 0, "stop unlinks owned socket");
    UnixSocketTaskExecutionServer wrong_uid(
        endpoint,
        {.socket_path = path, .allowed_uid = static_cast<std::uint32_t>(::geteuid()) + 1});
    expect(wrong_uid.start(), "start UID rejection server");
    std::thread denied([&] {
        expect(wrong_uid.serve_one_for(2000) == UnixSocketServeStatus::kServed,
               "accept unauthorized connection");
    });
    reject_raw(path, "GLIMMER_SHM_V2 OPEN\n");
    denied.join();
    wrong_uid.stop();
    expect(resources.registered.load() == 4 && resources.rolled_back.load() == 1,
           "wrong UID cannot reach endpoint or change resources");
    expect(::rmdir(directory) == 0, "remove private directory");
    test_recovery_after_lost_replies();
    return EXIT_SUCCESS;
}
