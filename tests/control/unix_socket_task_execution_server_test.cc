#include "glimmer/control/native_epoch_protocol.h"
#include "glimmer/control/shared_task_control_client.h"
#include "glimmer/control/task_execution_admission.h"
#include "glimmer/control/task_execution_endpoint.h"
#include "glimmer/control/task_execution_protocol.h"
#include "glimmer/control/unix_socket_task_execution_server.h"
#include "internal/task_execution_endpoint_test_hooks.h"

#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <future>
#include <iostream>
#include <memory>
#include <string>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <thread>
#include <unistd.h>

namespace {
std::atomic<int> fail_path_stat_after{0};
std::atomic<bool> fail_listen{false};
const char* replacement_path = nullptr;
const char* socket_path = nullptr;
}  // namespace

// GNU ld --wrap requires these exact names.
// NOLINTBEGIN(bugprone-reserved-identifier,readability-identifier-naming)
extern "C" int __real_lstat(const char*, struct stat*);
extern "C" int __real_listen(int, int);
extern "C" int __wrap_lstat(const char* path, struct stat* status) {
    if (fail_path_stat_after.load() > 0 && fail_path_stat_after.fetch_sub(1) == 1) {
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
};

bool register_resource(void* context, glimmer::core::TaskId) noexcept {
    auto* resources = static_cast<Resources*>(context);
    resources->registered.fetch_add(1);
    if (resources->fail_next_format.exchange(false)) {
        testing::fail_next_endpoint_response_format();
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

}  // namespace

int main() {
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
        const int replacement_fd = ::open(replacement.c_str(), O_CREAT | O_EXCL | O_WRONLY, 0600);
        expect(replacement_fd >= 0 && ::close(replacement_fd) == 0, "create test replacement");
        replacement_path = replacement.c_str();
        socket_path = path.c_str();
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
    return EXIT_SUCCESS;
}
