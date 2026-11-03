#include "glimmer/control/native_epoch_protocol.h"
#include "glimmer/control/shared_task_control_client.h"
#include "glimmer/control/task_execution_protocol.h"
#include "internal/owned_control_exchange.h"

#include <chrono>
#include <cerrno>
#include <csignal>
#include <cstdlib>
#include <fcntl.h>
#include <iostream>
#include <poll.h>
#include <stdexcept>
#include <string>
#include <string_view>
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
    int trace_fd = -1;
    std::string trace;
    explicit Service(const char* executable, bool poison = false, bool with_native = true,
                     bool adaptive = false) {
        char pattern[] = "/tmp/glimmer-task-service-XXXXXX";
        const auto* made = ::mkdtemp(pattern);
        require(made != nullptr, "create socket directory");
        directory = made;
        path = directory + "/service.sock";
        int trace_pipe[2] = {-1, -1};
        if (adaptive) {
            require(::pipe2(trace_pipe, O_CLOEXEC | O_NONBLOCK) == 0, "create trace pipe");
        }
        pid = ::fork();
        require(pid >= 0, "fork service");
        if (pid == 0) {
            if (adaptive) {
                ::close(trace_pipe[0]);
                if (::dup2(trace_pipe[1], STDERR_FILENO) < 0) ::_exit(126);
                ::close(trace_pipe[1]);
            }
            if (poison) ::setenv("GLIMMER_TEST_DISPATCH_UNCERTAIN", "1", 1);
            if (adaptive) {
                ::execl(executable, executable, "--socket", path.c_str(), "--quota-bytes", "100",
                        "--max-concurrent-tasks", "2", "--max-queued-tasks", "8",
                        "--scheduler-policy", "priority", "--adaptive-slo-target-queue-us", "1",
                        "--adaptive-slo-window", "1", "--trace-scheduler",
                        static_cast<char*>(nullptr));
            } else if (with_native) {
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
        if (adaptive) {
            ::close(trace_pipe[1]);
            trace_fd = trace_pipe[0];
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
            if (trace_fd >= 0) ::close(trace_fd);
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
        if (trace_fd >= 0) ::close(trace_fd);
    }
    void wait_for_trace(std::string_view expected) {
        const auto deadline = std::chrono::steady_clock::now() + 3s;
        while (trace.find(expected) == std::string::npos &&
               std::chrono::steady_clock::now() < deadline) {
            pollfd descriptor{.fd = trace_fd, .events = POLLIN, .revents = 0};
            const auto result = ::poll(&descriptor, 1, 100);
            if (result < 0 && errno == EINTR) continue;
            require(result >= 0, "poll trace pipe");
            if (result == 0) continue;
            char buffer[1024];
            const auto size = ::read(trace_fd, buffer, sizeof(buffer));
            if (size > 0)
                trace.append(buffer, static_cast<std::size_t>(size));
            else
                require(size < 0 && (errno == EAGAIN || errno == EINTR), "read trace pipe");
        }
        require(trace.find(expected) != std::string::npos, "expected scheduler trace arrives");
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
    return parsed.response.value_or(TaskExecutionProtocolResponse{});
}
NativeEpochResponse epoch(SharedTaskControlClient& client, const NativeEpochRequest& request) {
    const auto wire = format_native_epoch_request(request);
    const auto reply = wire ? client.request(*wire) : std::nullopt;
    const auto parsed = reply ? parse_native_epoch_response(*reply) : std::nullopt;
    require(parsed.has_value(), "native response decodes");
    return parsed.value_or(NativeEpochResponse{});
}

void test_execution_client(Service& service, const char* client_executable) {
    for (const auto* operation : {"stats", "metrics", "claim"}) {
        int descriptors[2];
        require(::pipe2(descriptors, O_CLOEXEC) == 0, "create client output pipe");
        const auto child = ::fork();
        require(child >= 0, "fork execution client");
        if (child == 0) {
            ::close(descriptors[0]);
            if (::dup2(descriptors[1], STDOUT_FILENO) < 0) ::_exit(126);
            ::close(descriptors[1]);
            ::execl(client_executable, client_executable, "--execution", "--socket",
                    service.path.c_str(), operation, static_cast<char*>(nullptr));
            ::_exit(127);
        }
        ::close(descriptors[1]);
        std::string output;
        char buffer[512];
        ssize_t size = 0;
        do {
            size = ::read(descriptors[0], buffer, sizeof(buffer));
            if (size > 0) output.append(buffer, static_cast<std::size_t>(size));
        } while (size > 0 || (size < 0 && errno == EINTR));
        ::close(descriptors[0]);
        int status = 0;
        require(wait_child(child, &status, 0) == child && WIFEXITED(status) && size == 0,
                "execution client exits and output is read completely");
        if (std::string_view(operation) == "claim") {
            require(WEXITSTATUS(status) != 0 && output.empty(),
                    "execution client rejects unsupported state-changing operations");
            continue;
        }
        require(WEXITSTATUS(status) == 0, "execution query completes successfully");
        const auto parsed = parse_task_execution_protocol_response(output);
        const auto expected = std::string_view(operation) == "stats"
                                  ? TaskExecutionProtocolResponseKind::kStats
                                  : TaskExecutionProtocolResponseKind::kMetrics;
        require(parsed.parsed() && parsed.response && parsed.response->kind == expected,
                "execution client prints the versioned stats or metrics reply");
    }
    require(::kill(service.pid, SIGTERM) == 0, "stop execution client fixture");
    service.join_success();
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
        ::_exit(parsed.parsed() && parsed.response &&
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
            pollfd descriptor{.fd = wait ? wait->descriptor : -1, .events = POLLIN, .revents = 0};
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

void test_adaptive_reservation(const char* executable) {
    Service service(executable, false, false, true);
    SharedTaskControlClient client(service.path);
    const auto first = exec(client, "GLIMMER_TASK_EXEC_V1 EXEC_ACQUIRE low 10 1 1\n");
    const auto second = exec(client, "GLIMMER_TASK_EXEC_V1 EXEC_ACQUIRE low 10 1 1\n");
    const auto high = exec(client, "GLIMMER_TASK_EXEC_V1 EXEC_ACQUIRE high 10 1 1 1\n");
    require(first.kind == TaskExecutionProtocolResponseKind::kExecutionLease &&
                second.kind == TaskExecutionProtocolResponseKind::kExecutionLease &&
                high.kind == TaskExecutionProtocolResponseKind::kAccepted,
            "adaptive fixture queues high priority behind two running tasks");
    std::this_thread::sleep_for(20ms);
    const auto complete = [&](const TaskExecutionProtocolResponse& lease) {
        return exec(client, "GLIMMER_TASK_EXEC_V1 EXEC_COMPLETE " + std::to_string(lease.task_id) +
                                " " + std::to_string(lease.generation) + "\n");
    };
    require(complete(first).state == TaskProtocolState::kCompleted,
            "first low-priority task completes");
    const auto granted =
        exec(client, "GLIMMER_TASK_EXEC_V1 EXEC_CLAIM " + std::to_string(high.task_id) + "\n");
    require(granted.kind == TaskExecutionProtocolResponseKind::kExecutionLease &&
                complete(granted).state == TaskProtocolState::kCompleted,
            "high-priority queued task completes");
    service.wait_for_trace("scheduler dispatch");
    service.wait_for_trace("adaptive reservation slots=1");
    const auto blocked = exec(client, "GLIMMER_TASK_EXEC_V1 EXEC_ACQUIRE low 10 1 1\n");
    require(blocked.kind == TaskExecutionProtocolResponseKind::kAccepted,
            "adaptive reservation blocks a second low-priority task");
    require(complete(second).state == TaskProtocolState::kCompleted,
            "remaining low-priority task completes");
    const auto released =
        exec(client, "GLIMMER_TASK_EXEC_V1 EXEC_CLAIM " + std::to_string(blocked.task_id) + "\n");
    require(released.kind == TaskExecutionProtocolResponseKind::kExecutionLease &&
                complete(released).state == TaskProtocolState::kCompleted,
            "blocked task completes after capacity is available");
    require(::kill(service.pid, SIGTERM) == 0, "stop adaptive service");
    service.join_success();
}

void test_invalid_options(const char* executable) {
    char pattern[] = "/tmp/glimmer-task-options-XXXXXX";
    const auto* made = ::mkdtemp(pattern);
    require(made != nullptr, "create private option-test directory");
    const std::string path = std::string(made) + "/service.sock";
    for (int invalid = 0; invalid < 6; ++invalid) {
        const auto child = ::fork();
        require(child >= 0, "fork invalid-option service");
        if (child == 0) {
            if (invalid == 1) {
                ::execl(executable, executable, "--socket", path.c_str(), "--quota-bytes", "100",
                        "--max-concurrent-tasks", "1", "--max-queued-tasks", "8",
                        "--native-epoch-max-calls", "4", static_cast<char*>(nullptr));
            } else if (invalid == 2) {
                ::execl(executable, executable, "--socket", path.c_str(), "--quota-bytes", "100",
                        "--max-concurrent-tasks", "2", "--max-queued-tasks", "8",
                        "--adaptive-slo-target-queue-us", "1", static_cast<char*>(nullptr));
            } else if (invalid == 3) {
                ::execl(executable, executable, "--socket", path.c_str(), "--quota-bytes", "100",
                        "--max-concurrent-tasks", "2", "--max-queued-tasks", "8",
                        "--scheduler-policy", "priority", "--adaptive-slo-target-queue-us", "1",
                        "--adaptive-slo-max-reserved-slots", "3", static_cast<char*>(nullptr));
            } else if (invalid == 4) {
                ::execl(executable, executable, "--socket", path.c_str(), "--quota-bytes", "100",
                        "--max-concurrent-tasks", "2", "--max-queued-tasks", "8",
                        "--scheduler-policy", "priority", "--adaptive-slo-target-queue-us", "1",
                        "--adaptive-slo-window", "0", static_cast<char*>(nullptr));
            } else if (invalid == 5) {
                ::execl(executable, executable, "--socket", path.c_str(), "--quota-bytes", "100",
                        "--max-concurrent-tasks", "1", "--max-queued-tasks", "8",
                        "--scheduler-policy", "priority", "--adaptive-slo-target-queue-us", "1",
                        static_cast<char*>(nullptr));
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
        require(argc == 4, "service, test-host and client executables required");
        test_drain(argv[1]);
        test_owner_channel(argv[1]);
        test_native_disabled(argv[1]);
        test_adaptive_reservation(argv[1]);
        test_invalid_options(argv[1]);
        test_uncertain_shutdown(argv[2]);
        Service client_service(argv[1]);
        test_execution_client(client_service, argv[3]);
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
