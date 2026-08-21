#include "glimmer/control/launch_gate.h"

#include <chrono>
#include <cerrno>
#include <cstdlib>
#include <iostream>
#include <poll.h>
#include <signal.h>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

namespace {

constexpr int kSkipExitCode = 77;
constexpr int kStartupTimeoutMs = 2'000;

void expect(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "Test failed: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

bool wait_for_socket(const std::string& socket_path) {
    for (int elapsed_ms = 0; elapsed_ms < kStartupTimeoutMs; elapsed_ms += 10) {
        struct stat socket_stat {};
        if (::lstat(socket_path.c_str(), &socket_stat) == 0) {
            return true;
        }
        static_cast<void>(::poll(nullptr, 0, 10));
    }
    return false;
}

int run_worker(const char* socket_path, const char* tenant_id) {
    glimmer::control::LaunchGate gate(glimmer::control::LaunchGateOptions{
        .max_concurrent_launches = 1,
        .tenant_id = tenant_id == nullptr ? "default" : tenant_id,
        .control_socket = socket_path == nullptr ? "" : socket_path,
        .remote_acquire_timeout = std::chrono::seconds{2},
        .remote_poll_interval = std::chrono::milliseconds{2}});
    glimmer::control::LaunchGateTiming acquire_timing;
    const auto lease = gate.acquire(std::chrono::seconds{2}, &acquire_timing);
    if (!lease.has_value()) {
        return EXIT_FAILURE;
    }
    if (!acquire_timing.remote || acquire_timing.request_count == 0 ||
        acquire_timing.transport_nanoseconds == 0) {
        return EXIT_FAILURE;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds{100});
    glimmer::control::LaunchGateTiming completion_timing;
    if (!gate.complete(lease.value(), &completion_timing)) {
        return EXIT_FAILURE;
    }
    return completion_timing.remote && completion_timing.request_count == 1 &&
                   completion_timing.transport_nanoseconds != 0
               ? EXIT_SUCCESS
               : EXIT_FAILURE;
}

pid_t start_service(const std::string& executable, const std::string& socket_path) {
    const pid_t service_pid = ::fork();
    expect(service_pid >= 0, "control service should fork");
    if (service_pid == 0) {
        ::execl(executable.c_str(), executable.c_str(), "--socket", socket_path.c_str(),
                "--quota-bytes", "1048576", "--execution-mode", "remote", "--max-concurrent-tasks",
                "1", "--lease-timeout-ms", "2000", "--bind-leases-to-process",
                static_cast<char*>(nullptr));
        _exit(EXIT_FAILURE);
    }
    return service_pid;
}

void stop_service(pid_t service_pid) {
    static_cast<void>(::kill(service_pid, SIGTERM));
    int status = 0;
    while (::waitpid(service_pid, &status, 0) < 0 && errno == EINTR) {
    }
}

pid_t start_worker(const std::string& executable, const std::string& socket_path,
                   std::string_view tenant_id) {
    const pid_t worker_pid = ::fork();
    expect(worker_pid >= 0, "launch worker should fork");
    if (worker_pid == 0) {
        ::execl(executable.c_str(), executable.c_str(), "--worker", socket_path.c_str(),
                std::string(tenant_id).c_str(), static_cast<char*>(nullptr));
        _exit(EXIT_FAILURE);
    }
    return worker_pid;
}

int run_parent(const char* executable_path) {
    const std::string prefix = "/tmp/glimmer-launch-gate-" + std::to_string(::getpid());
    const std::string socket_path = prefix + ".sock";
    const std::size_t separator = std::string(executable_path).find_last_of('/');
    expect(separator != std::string::npos, "test executable should have a directory");
    const std::string build_directory = std::string(executable_path).substr(0, separator);
    const std::string service_path = build_directory + "/../bin/glimmer_control_service";
    static_cast<void>(::unlink(socket_path.c_str()));
    const pid_t service_pid = start_service(service_path, socket_path);
    if (!wait_for_socket(socket_path)) {
        stop_service(service_pid);
        std::cerr << "Skipping remote launch-gate test: service did not start\n";
        return kSkipExitCode;
    }

    const auto start = std::chrono::steady_clock::now();
    const pid_t first_worker = start_worker(executable_path, socket_path, "tenant-a");
    const pid_t second_worker = start_worker(executable_path, socket_path, "tenant-b");
    int first_status = 0;
    int second_status = 0;
    while (::waitpid(first_worker, &first_status, 0) < 0 && errno == EINTR) {
    }
    while (::waitpid(second_worker, &second_status, 0) < 0 && errno == EINTR) {
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start);
    stop_service(service_pid);
    static_cast<void>(::unlink(socket_path.c_str()));
    expect(WIFEXITED(first_status) && WEXITSTATUS(first_status) == EXIT_SUCCESS,
           "first process should complete a remote launch lease");
    expect(WIFEXITED(second_status) && WEXITSTATUS(second_status) == EXIT_SUCCESS,
           "second process should complete a remote launch lease");
    expect(elapsed >= std::chrono::milliseconds{160},
           "a global single-slot scheduler should serialize the two processes");
    return EXIT_SUCCESS;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 4 && std::string_view(argv[1]) == "--worker") {
        return run_worker(argv[2], argv[3]);
    }
    expect(argc == 1, "remote launch-gate test does not accept arguments");
    return run_parent(argv[0]);
}
