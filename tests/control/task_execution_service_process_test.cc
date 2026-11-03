#include "glimmer/control/native_epoch_protocol.h"
#include "glimmer/control/shared_task_control_client.h"
#include "glimmer/control/task_execution_protocol.h"
#include "internal/owned_control_exchange.h"

#include <chrono>
#include <cerrno>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <poll.h>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

namespace {
using namespace glimmer::control;
using namespace std::chrono_literals;
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
pid_t wait_child(pid_t pid, int* status, int options) noexcept {
    pid_t result = 0;
    do {
        result = ::waitpid(pid, status, options);
    } while (result < 0 && errno == EINTR);
    return result;
}
struct Service {
    std::string directory;
    std::string path;
    pid_t pid = -1;
    explicit Service(const char* executable, bool poison = false, bool with_native = true) {
        char pattern[] = "/tmp/glimmer-task-service-XXXXXX";
        const auto* made = ::mkdtemp(pattern);
        require(made != nullptr, "create socket directory");
        directory = made;
        path = directory + "/service.sock";
        pid = ::fork();
        require(pid >= 0, "fork service");
        if (pid == 0) {
            if (poison) ::setenv("GLIMMER_TEST_DISPATCH_UNCERTAIN", "1", 1);
            if (with_native) {
                ::execl(executable, executable, "--socket", path.c_str(), "--quota-bytes", "100",
                        "--max-concurrent-tasks", "1", "--max-queued-tasks", "8",
                        "--native-epoch-max-calls", "4", "--native-epoch-prefix-calls", "2",
                        "--native-epoch-pending-prefixes", "2", "--native-epoch-acceptance-us",
                        "1000000", static_cast<char*>(nullptr));
            } else {
                ::execl(executable, executable, "--socket", path.c_str(), "--quota-bytes", "100",
                        "--max-concurrent-tasks", "1", "--max-queued-tasks", "8",
                        static_cast<char*>(nullptr));
            }
            ::_exit(127);
        }
        const auto deadline = std::chrono::steady_clock::now() + 3s;
        struct stat status {};
        while (::lstat(path.c_str(), &status) != 0 && std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(10ms);
        if (!S_ISSOCK(status.st_mode)) {
            if (wait_child(pid, nullptr, WNOHANG) == 0) {
                ::kill(pid, SIGKILL);
                wait_child(pid, nullptr, 0);
            }
            pid = -1;
            ::unlink(path.c_str());
            ::rmdir(directory.c_str());
            throw std::runtime_error("service creates socket");
        }
    }
    ~Service() {
        if (pid > 0) {
            if (wait_child(pid, nullptr, WNOHANG) == 0) {
                ::kill(pid, SIGKILL);
                wait_child(pid, nullptr, 0);
            }
        }
        ::unlink(path.c_str());
        ::rmdir(directory.c_str());
    }
    bool still_running() {
        const auto result = wait_child(pid, nullptr, WNOHANG);
        if (result == pid || (result < 0 && errno == ECHILD)) pid = -1;
        require(result >= 0, "failed to observe owned service process");
        return result == 0;
    }
    void join_success() {
        const auto deadline = std::chrono::steady_clock::now() + 3s;
        int status = 0;
        pid_t result = 0;
        do {
            result = wait_child(pid, &status, WNOHANG);
            if (result == 0) std::this_thread::sleep_for(10ms);
        } while (result == 0 && std::chrono::steady_clock::now() < deadline);
        if (result == pid || (result < 0 && errno == ECHILD)) pid = -1;
        require(result >= 0, "failed to join owned service process");
        require(pid == -1 && WIFEXITED(status) && WEXITSTATUS(status) == 0,
                "drained service exits successfully");
        struct stat socket_status {};
        require(::lstat(path.c_str(), &socket_status) != 0, "service removes its socket");
    }
};
TaskExecutionProtocolResponse exec(SharedTaskControlClient& client, const std::string& request) {
    const auto wire = client.request(request);
    const auto parsed = wire ? parse_task_execution_protocol_response(*wire)
                             : TaskExecutionProtocolResponseParseResult{};
    require(parsed.parsed(), "execution response decodes");
    return *parsed.response;
}
NativeEpochResponse epoch(SharedTaskControlClient& client, const NativeEpochRequest& request) {
    const auto wire = format_native_epoch_request(request);
    const auto reply = wire ? client.request(*wire) : std::nullopt;
    const auto parsed = reply ? parse_native_epoch_response(*reply) : std::nullopt;
    require(parsed.has_value(), "native response decodes");
    return *parsed;
}
void test_drain(const char* executable) {
    Service service(executable);
    SharedTaskControlClient client(service.path);
    const auto grant = exec(client, "GLIMMER_TASK_EXEC_V1 EXEC_ACQUIRE tenant 10 1 1\n");
    require(grant.kind == TaskExecutionProtocolResponseKind::kExecutionLease,
            "cooperative grant is issued");
    const auto done = "GLIMMER_TASK_EXEC_V1 EXEC_COMPLETE " + std::to_string(grant.task_id) + " " +
                      std::to_string(grant.generation) + "\n";
    const auto child = ::fork();
    require(child >= 0, "fork foreign owner");
    if (child == 0) {
        SharedTaskControlClient foreign(service.path);
        const auto wire = foreign.request(done);
        const auto parsed = wire ? parse_task_execution_protocol_response(*wire)
                                 : TaskExecutionProtocolResponseParseResult{};
        ::_exit(parsed.parsed() &&
                        parsed.response->kind == TaskExecutionProtocolResponseKind::kError
                    ? 0
                    : 1);
    }
    int child_status = 0;
    require(wait_child(child, &child_status, 0) == child && WIFEXITED(child_status) &&
                WEXITSTATUS(child_status) == 0,
            "foreign process cannot complete another owner's task");
    {
        internal::ServiceIncarnationBinding binding;
        internal::OwnedControlExchange owner(binding);
        require(
            owner.initialize({service.path, 1000ms}) == internal::OwnedExchangeColdStatus::kReady,
            "V3 owner channel connects to process service");
        const auto sent = owner.begin("GLIMMER_TASK_EXEC_V1 STATS\n");
        require(sent.status == internal::SharedControlPublication::kPublished,
                "V3 request publishes");
        std::string response;
        auto progress = owner.poll(sent.ticket, response);
        while (progress == internal::OwnedExchangeProgress::kPending) {
            const auto wait = owner.wait_state(sent.ticket);
            require(wait.has_value(), "V3 response exposes wait descriptor");
            pollfd descriptor{.fd = wait->descriptor, .events = POLLIN, .revents = 0};
            require(::poll(&descriptor, 1, 1000) > 0, "V3 response wakes owner");
            progress = owner.poll(sent.ticket, response);
        }
        require(progress == internal::OwnedExchangeProgress::kResponse &&
                    parse_task_execution_protocol_response(response).parsed() && owner.close_idle(),
                "V3 response is consumed and owner channel closes");
    }
    NativeEpochRequest request;
    request.request_sequence = 1;
    request.admission = {.tenant_id = "tenant", .memory_bytes = 10, .work_units = 4};
    request.extent = {4, 2, 2, 1000000};
    const auto pending = epoch(client, request);
    require(pending.kind == NativeEpochResponseKind::kPending, "native work is pending");
    require(::kill(service.pid, SIGTERM) == 0, "request service shutdown");
    std::this_thread::sleep_for(200ms);
    require(service.still_running(), "live leases prevent exit");
    require(exec(client, "GLIMMER_TASK_EXEC_V1 EXEC_ACQUIRE tenant 10 1 1\n").kind ==
                TaskExecutionProtocolResponseKind::kError,
            "draining rejects new admission");
    request.operation = NativeEpochOperation::kCancelPending;
    request.execution = pending.grant.execution;
    request.request_sequence = 2;
    require(epoch(client, request).kind == NativeEpochResponseKind::kAcknowledged,
            "owner can cancel pending native work during drain");
    require(service.still_running(), "active lease still prevents exit");
    require(exec(client, done).state == TaskProtocolState::kCompleted,
            "owner completes active work during drain");
    service.join_success();
}

void test_uncertain_shutdown(const char* executable) {
    Service service(executable, true);
    SharedTaskControlClient client(service.path);
    require(!client.request("GLIMMER_TASK_EXEC_V1 EXEC_ACQUIRE tenant 10 1 1\n"),
            "uncertain dispatch does not produce a normal reply");
    const auto stats = exec(client, "GLIMMER_TASK_EXEC_V1 STATS\n");
    require(stats.kind == TaskExecutionProtocolResponseKind::kStats &&
                stats.stats.total_task_count == 1,
            "uncertain request was not replayed or silently retired");
    require(::kill(service.pid, SIGTERM) == 0, "request shutdown with poisoned accounting");
    std::this_thread::sleep_for(200ms);
    require(service.still_running(), "uncertain accounting cannot produce clean exit");
    // The test host owns no CUDA or external resources; its destructor stops
    // this exact child after the fail-closed assertion.
}

void test_owner_channel(const char* executable) {
    Service service(executable);
    internal::ServiceIncarnationBinding binding;
    internal::OwnedControlExchange owner(binding);
    require(owner.initialize({service.path, 1000ms}) == internal::OwnedExchangeColdStatus::kReady,
            "V3 owner channel connects");
    const auto exchange = [&](std::string_view request) {
        const auto sent = owner.begin(request);
        require(sent.status == internal::SharedControlPublication::kPublished,
                "V3 domain request publishes once");
        std::string response;
        auto progress = owner.poll(sent.ticket, response);
        while (progress == internal::OwnedExchangeProgress::kPending) {
            const auto wait = owner.wait_state(sent.ticket);
            require(wait.has_value(), "V3 pending result has a wait descriptor");
            pollfd descriptor{.fd = wait->descriptor, .events = POLLIN, .revents = 0};
            require(::poll(&descriptor, 1, 1000) > 0, "V3 domain reply wakes");
            progress = owner.poll(sent.ticket, response);
        }
        require(progress == internal::OwnedExchangeProgress::kResponse,
                "V3 domain reply is unambiguous");
        const auto parsed = parse_task_execution_protocol_response(response);
        require(parsed.parsed(), "V3 domain reply parses");
        return *parsed.response;
    };
    const auto grant = exchange("GLIMMER_TASK_EXEC_V1 EXEC_ACQUIRE tenant 10 1 1\n");
    require(grant.kind == TaskExecutionProtocolResponseKind::kExecutionLease,
            "V3 service grants actual task");
    const auto completed =
        exchange("GLIMMER_TASK_EXEC_V1 EXEC_COMPLETE " + std::to_string(grant.task_id) + " " +
                 std::to_string(grant.generation) + "\n");
    require(completed.state == TaskProtocolState::kCompleted && owner.close_idle(),
            "V3 owner completes and closes");
    require(::kill(service.pid, SIGTERM) == 0, "stop drained V3 service");
    service.join_success();
}

void test_native_disabled(const char* executable) {
    Service service(executable, false, false);
    SharedTaskControlClient client(service.path);
    NativeEpochRequest request;
    request.request_sequence = 1;
    request.admission = {.tenant_id = "tenant", .memory_bytes = 10, .work_units = 4};
    request.extent = {4, 2, 2, 1000000};
    require(epoch(client, request).kind == NativeEpochResponseKind::kError,
            "native work is rejected when not configured");
    require(::kill(service.pid, SIGTERM) == 0, "stop service without native work");
    service.join_success();
}

void test_invalid_options(const char* executable) {
    char pattern[] = "/tmp/glimmer-task-options-XXXXXX";
    const auto* made = ::mkdtemp(pattern);
    require(made != nullptr, "create private option-test directory");
    const std::string path = std::string(made) + "/service.sock";
    for (const bool partial_native : {false, true}) {
        const auto child = ::fork();
        require(child >= 0, "fork invalid-option service");
        if (child == 0) {
            if (partial_native) {
                ::execl(executable, executable, "--socket", path.c_str(), "--quota-bytes", "100",
                        "--max-concurrent-tasks", "1", "--max-queued-tasks", "8",
                        "--native-epoch-max-calls", "4", static_cast<char*>(nullptr));
            } else {
                ::execl(executable, executable, "--socket", path.c_str(), "--quota-bytes", "100",
                        "--max-concurrent-tasks", "1", "--max-queued-tasks", "0",
                        static_cast<char*>(nullptr));
            }
            ::_exit(127);
        }
        int status = 0;
        pid_t observed = 0;
        const auto deadline = std::chrono::steady_clock::now() + 2s;
        do {
            observed = wait_child(child, &status, WNOHANG);
            if (observed == 0) std::this_thread::sleep_for(10ms);
        } while (observed == 0 && std::chrono::steady_clock::now() < deadline);
        if (observed == 0) {
            ::kill(child, SIGKILL);
            wait_child(child, &status, 0);
        }
        const bool rejected = observed == child && WIFEXITED(status) && WEXITSTATUS(status) != 0;
        if (!rejected) {
            ::unlink(path.c_str());
            ::rmdir(made);
            throw std::runtime_error("invalid options unexpectedly started service");
        }
    }
    require(::rmdir(made) == 0, "remove private option-test directory");
}
}  // namespace
int main(int argc, char** argv) {
    try {
        require(argc == 3, "service and test-host executables required");
        test_drain(argv[1]);
        test_owner_channel(argv[1]);
        test_native_disabled(argv[1]);
        test_invalid_options(argv[1]);
        test_uncertain_shutdown(argv[2]);
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
