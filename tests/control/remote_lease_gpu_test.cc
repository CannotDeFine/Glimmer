#include "glimmer/control/task_endpoint.h"
#include "glimmer/control/task_protocol.h"
#include "glimmer/control/unix_socket_client.h"
#include "glimmer/control/unix_socket_server.h"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

namespace {

constexpr int kSkipExitCode = 77;

bool register_resource(void* context, glimmer::core::TaskId) noexcept {
    return context != nullptr;
}

void expect(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "Test failed: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

std::optional<glimmer::control::TaskProtocolResponse> request(
    const glimmer::control::UnixSocketControlClient& client,
    const glimmer::control::TaskProtocolRequest& task_request) {
    const auto encoded = glimmer::control::format_task_protocol_request(task_request);
    if (!encoded.has_value()) {
        expect(false, "test request should format");
        return std::nullopt;
    }
    const auto response_line = client.request(encoded.value());
    if (!response_line.has_value()) {
        expect(false, "test client should receive a response");
        return std::nullopt;
    }
    const auto parsed = glimmer::control::parse_task_protocol_response(response_line.value());
    if (!parsed.parsed() || !parsed.response.has_value()) {
        expect(false, "test response should parse");
        return std::nullopt;
    }
    return parsed.response;
}

bool socket_available(const std::string& path) {
    glimmer::core::Scheduler scheduler(1);
    glimmer::control::TaskAdmissionService admission_service(scheduler);
    glimmer::control::TaskControlEndpoint endpoint(admission_service, register_resource,
                                                   &scheduler);
    glimmer::control::UnixSocketControlServer server(
        endpoint,
        glimmer::control::UnixSocketControlConfig{.socket_path = path, .allowed_uid = ::geteuid()});
    if (server.start()) {
        server.stop();
        return true;
    }
    return errno != EPERM && errno != EACCES && errno != ENOSYS;
}

pid_t start_worker(const std::string& worker_path, const std::string& socket_path) {
    const pid_t worker_pid = ::fork();
    expect(worker_pid >= 0, "GPU worker should fork");
    if (worker_pid == 0) {
        ::execl(worker_path.c_str(), worker_path.c_str(), "--socket", socket_path.c_str(),
                "--tasks", "1", "--idle-polls", "20", "--heartbeat-interval-ms", "20",
                "--execution-delay-ms", "500", static_cast<char*>(nullptr));
        _exit(EXIT_FAILURE);
    }
    return worker_pid;
}

void wait_for_worker(pid_t worker_pid) {
    int worker_status = 0;
    while (::waitpid(worker_pid, &worker_status, 0) < 0 && errno == EINTR) {
    }
    expect(WIFEXITED(worker_status) && WEXITSTATUS(worker_status) == EXIT_SUCCESS,
           "CUDA worker should execute and report the lease");
}

}  // namespace

int main(int argc, char** argv) {
    expect(argc == 1, "GPU lease test does not accept arguments");
    const std::string socket_path =
        "/tmp/glimmer-remote-lease-" + std::to_string(::getpid()) + ".sock";
    if (!socket_available(socket_path)) {
        std::cerr << "Skipping remote lease GPU test: Unix socket bind is unavailable\n";
        return kSkipExitCode;
    }

    glimmer::core::Scheduler scheduler(glimmer::core::MemoryBytes{8} * 1024 * 1024,
                                       glimmer::core::SchedulerOptions{.max_running_tasks = 2});
    // This fixture submits work from the parent and lets independent workers
    // claim it. Process-bound ownership is covered by the dedicated remote
    // launch-gate process test, where each worker submits its own lease.
    // CUDA context/module startup can take longer than the worker's first
    // polling window. Keep the lease timeout long enough for admission while
    // still exercising heartbeat-based renewal during execution.
    glimmer::control::TaskAdmissionService admission_service(
        scheduler, std::chrono::milliseconds{2'000}, false);
    glimmer::control::TaskControlEndpoint endpoint(admission_service, register_resource,
                                                   &scheduler);
    glimmer::control::UnixSocketControlServer server(
        endpoint, glimmer::control::UnixSocketControlConfig{.socket_path = socket_path,
                                                            .allowed_uid = ::geteuid()});
    expect(server.start(), "remote lease test server should start");

    std::atomic_bool stop_requested = false;
    std::thread server_thread([&admission_service, &server, &stop_requested] {
        while (!stop_requested.load()) {
            static_cast<void>(admission_service.reap_expired());
            const auto status = server.serve_one_for(10);
            if (status == glimmer::control::UnixSocketServeStatus::kError &&
                !stop_requested.load()) {
                break;
            }
        }
    });

    const glimmer::control::UnixSocketControlClient client(socket_path);
    const glimmer::control::TaskProtocolRequest submit{
        .operation = glimmer::control::TaskProtocolOperation::kSubmit,
        .admission = {.tenant_id = "gpu-worker",
                      .memory_bytes = glimmer::core::MemoryBytes{1} * 1024 * 1024,
                      .weight = 1,
                      .work_units = 1},
        .task_id = 0};
    const auto accepted = request(client, submit);
    expect(accepted.has_value() &&
               accepted->kind == glimmer::control::TaskProtocolResponseKind::kAccepted &&
               accepted->task_id == 1,
           "remote GPU task should be admitted");
    const auto second_accepted = request(client, submit);
    expect(second_accepted.has_value() &&
               second_accepted->kind == glimmer::control::TaskProtocolResponseKind::kAccepted &&
               second_accepted->task_id == 2,
           "second remote GPU task should be admitted");

    const std::string executable_path = argv[0];
    const std::size_t separator = executable_path.find_last_of('/');
    expect(separator != std::string::npos, "GPU test executable should have a directory");
    const std::string test_directory = executable_path.substr(0, separator);
    const std::string build_directory = test_directory.substr(0, test_directory.find_last_of('/'));
    const std::string worker_path =
        build_directory + "/examples/cuda_task_backend/glimmer_cuda_lease_worker";

    const pid_t first_worker_pid = start_worker(worker_path, socket_path);
    const pid_t second_worker_pid = start_worker(worker_path, socket_path);
    wait_for_worker(first_worker_pid);
    wait_for_worker(second_worker_pid);

    const glimmer::control::TaskProtocolRequest query{
        .operation = glimmer::control::TaskProtocolOperation::kQuery,
        .admission = {},
        .task_id = 1};
    const auto completed = request(client, query);
    expect(completed.has_value() &&
               completed->kind == glimmer::control::TaskProtocolResponseKind::kState &&
               completed->state == glimmer::control::TaskProtocolState::kCompleted,
           "remote worker should leave a completed task state");
    const auto second_completed =
        request(client, glimmer::control::TaskProtocolRequest{
                            .operation = glimmer::control::TaskProtocolOperation::kQuery,
                            .admission = {},
                            .task_id = 2});
    expect(second_completed.has_value() &&
               second_completed->kind == glimmer::control::TaskProtocolResponseKind::kState &&
               second_completed->state == glimmer::control::TaskProtocolState::kCompleted,
           "second remote worker should leave a completed task state");

    stop_requested.store(true);
    server_thread.join();
    server.stop();
    static_cast<void>(::unlink(socket_path.c_str()));
    return EXIT_SUCCESS;
}
