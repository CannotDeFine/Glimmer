#include <cerrno>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <poll.h>
#include <signal.h>
#include <string>
#include <string_view>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

constexpr int kSkipExitCode = 77;
constexpr int kStartupTimeoutMs = 2'000;

struct CommandResult {
    int exit_code = -1;
    std::string output;
};

void expect(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "Test failed: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

bool socket_bind_is_supported(const std::string& socket_path) {
    static_cast<void>(::unlink(socket_path.c_str()));
    const int socket_fd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (socket_fd < 0) {
        return errno != EPERM && errno != EACCES && errno != ENOSYS;
    }
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    std::memcpy(address.sun_path, socket_path.c_str(), socket_path.size() + 1);
    const auto address_length =
        static_cast<socklen_t>(offsetof(sockaddr_un, sun_path) + socket_path.size() + 1);
    const int bind_result =
        ::bind(socket_fd, reinterpret_cast<const sockaddr*>(&address), address_length);
    const int bind_errno = errno;
    static_cast<void>(::close(socket_fd));
    static_cast<void>(::unlink(socket_path.c_str()));
    if (bind_result == 0) {
        return true;
    }
    if (bind_errno == EPERM || bind_errno == EACCES || bind_errno == ENOSYS) {
        return false;
    }
    expect(false, "Unix socket bind should be supported or explicitly unavailable");
    return false;
}

bool wait_for_socket(const std::string& socket_path) {
    int elapsed_ms = 0;
    while (elapsed_ms < kStartupTimeoutMs) {
        struct stat socket_stat {};
        if (::lstat(socket_path.c_str(), &socket_stat) == 0) {
            return true;
        }
        static_cast<void>(::poll(nullptr, 0, 10));
        elapsed_ms += 10;
    }
    return false;
}

pid_t start_service(const std::string& executable, const std::string& socket_path,
                    bool bounded_queue = false) {
    const pid_t child_pid = ::fork();
    expect(child_pid >= 0, "service process should fork");
    if (child_pid == 0) {
        const int null_fd = ::open("/dev/null", O_WRONLY | O_CLOEXEC);
        if (null_fd >= 0) {
            static_cast<void>(::dup2(null_fd, STDERR_FILENO));
            static_cast<void>(::close(null_fd));
        }
        if (bounded_queue) {
            ::execl(executable.c_str(), executable.c_str(), "--socket", socket_path.c_str(),
                    "--quota-bytes", "1024", "--execution-mode", "remote", "--max-concurrent-tasks",
                    "1", "--max-queued-tasks", "1", "--max-requests", "16",
                    static_cast<char*>(nullptr));
        } else {
            ::execl(executable.c_str(), executable.c_str(), "--socket", socket_path.c_str(),
                    "--quota-bytes", "1024", "--execution-mode", "remote", "--max-concurrent-tasks",
                    "2", "--lease-timeout-ms", "0", "--max-queued-tasks", "0", "--max-requests",
                    "16", "--scheduler-policy", "fifo", static_cast<char*>(nullptr));
        }
        _exit(EXIT_FAILURE);
    }
    return child_pid;
}

void stop_service(pid_t service_pid) {
    if (service_pid <= 0) {
        return;
    }
    static_cast<void>(::kill(service_pid, SIGTERM));
    int status = 0;
    while (::waitpid(service_pid, &status, 0) < 0 && errno == EINTR) {
    }
}

CommandResult run_client(const std::string& executable, const std::string& socket_path,
                         const std::vector<std::string>& arguments) {
    int output_pipe[2]{};
    expect(::pipe2(output_pipe, O_CLOEXEC) == 0, "client output pipe should be created");
    const pid_t child_pid = ::fork();
    expect(child_pid >= 0, "client process should fork");
    if (child_pid == 0) {
        static_cast<void>(::dup2(output_pipe[1], STDOUT_FILENO));
        static_cast<void>(::close(output_pipe[0]));
        static_cast<void>(::close(output_pipe[1]));
        std::vector<std::string> command;
        command.reserve(arguments.size() + 4);
        command.push_back(executable);
        command.emplace_back("--socket");
        command.push_back(socket_path);
        command.insert(command.end(), arguments.begin(), arguments.end());
        std::vector<char*> command_argv;
        command_argv.reserve(command.size() + 1);
        for (std::string& argument : command) {
            command_argv.push_back(argument.data());
        }
        command_argv.push_back(nullptr);
        ::execv(executable.c_str(), command_argv.data());
        _exit(EXIT_FAILURE);
    }
    static_cast<void>(::close(output_pipe[1]));
    int status = 0;
    while (::waitpid(child_pid, &status, 0) < 0 && errno == EINTR) {
    }
    std::string output;
    char buffer[256]{};
    while (true) {
        const ssize_t received = ::read(output_pipe[0], buffer, sizeof(buffer));
        if (received == 0) {
            break;
        }
        if (received < 0) {
            if (errno == EINTR) {
                continue;
            }
            break;
        }
        output.append(buffer, static_cast<std::size_t>(received));
    }
    static_cast<void>(::close(output_pipe[0]));
    return {.exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1, .output = std::move(output)};
}

}  // namespace

int main(int argc, char** argv) {
    expect(argc == 1, "process test does not accept arguments");
    const std::string test_prefix = "/tmp/glimmer-control-process-" + std::to_string(::getpid());
    const std::string socket_path = test_prefix + ".sock";
    if (!socket_bind_is_supported(socket_path)) {
        std::cerr << "Skipping process control test: Unix socket bind is unavailable\n";
        return kSkipExitCode;
    }

    const std::string executable_path = argv[0];
    const std::size_t separator = executable_path.find_last_of('/');
    expect(separator != std::string::npos, "test executable should have a directory");
    const std::string test_directory = executable_path.substr(0, separator);
    const std::string build_directory = test_directory.substr(0, test_directory.find_last_of('/'));
    const std::string service_path = build_directory + "/bin/glimmer_control_service";
    const std::string client_path = build_directory + "/bin/glimmer_control_client";

    const pid_t service_pid = start_service(service_path, socket_path);
    if (!wait_for_socket(socket_path)) {
        stop_service(service_pid);
        expect(false, "control service should create its socket");
    }

    const CommandResult empty_claim = run_client(client_path, socket_path, {"claim"});
    expect(empty_claim.exit_code == 0 && empty_claim.output == "GLIMMER_TASK_V1 EMPTY\n",
           "claim should report an empty queue before submission");
    const CommandResult empty_stats = run_client(client_path, socket_path, {"stats"});
    expect(empty_stats.exit_code == 0 &&
               empty_stats.output == "GLIMMER_TASK_V1 STATS 0 0 0 0 0 0 1024 0 0 2 0\n",
           "stats should report an empty service and configured limits");

    const CommandResult quota_rejection =
        run_client(client_path, socket_path, {"submit", "tenant-a", "2048", "1", "1"});
    expect(quota_rejection.exit_code != 0 &&
               quota_rejection.output.find("ERROR QUOTA_EXCEEDED") != std::string::npos,
           "client should expose quota rejection");

    const CommandResult accepted =
        run_client(client_path, socket_path, {"submit", "tenant-a", "512", "1", "1"});
    expect(accepted.exit_code == 0 && accepted.output == "GLIMMER_TASK_V1 OK 1\n",
           "client should submit a task with the first task id");

    const CommandResult second_accepted =
        run_client(client_path, socket_path, {"submit", "tenant-b", "512", "1", "1"});
    expect(second_accepted.exit_code == 0 && second_accepted.output == "GLIMMER_TASK_V1 OK 2\n",
           "client should submit a second task for the second running slot");

    const CommandResult lease = run_client(client_path, socket_path, {"claim"});
    expect(lease.exit_code == 0 && lease.output == "GLIMMER_TASK_V1 LEASE 1 tenant-a 512 1 1\n",
           "claim should return the queued task metadata");

    const CommandResult second_lease = run_client(client_path, socket_path, {"claim"});
    expect(second_lease.exit_code == 0 &&
               second_lease.output == "GLIMMER_TASK_V1 LEASE 2 tenant-b 512 1 1\n",
           "a second worker should receive the second running slot");

    const CommandResult initial_query = run_client(client_path, socket_path, {"query", "1"});
    expect(
        initial_query.exit_code == 0 && initial_query.output == "GLIMMER_TASK_V1 STATE 1 RUNNING\n",
        "first query should observe a running task");
    const CommandResult second_query = run_client(client_path, socket_path, {"query", "2"});
    expect(
        second_query.exit_code == 0 && second_query.output == "GLIMMER_TASK_V1 STATE 2 RUNNING\n",
        "the second query should observe the second running task");
    const CommandResult blocked_claim = run_client(client_path, socket_path, {"claim"});
    expect(blocked_claim.exit_code == 0 && blocked_claim.output == "GLIMMER_TASK_V1 EMPTY\n",
           "a third worker must wait when both running slots are occupied");
    const CommandResult heartbeat = run_client(client_path, socket_path, {"heartbeat", "1"});
    expect(heartbeat.exit_code == 0 && heartbeat.output == "GLIMMER_TASK_V1 STATE 1 RUNNING\n",
           "worker heartbeat should observe the active lease");
    const CommandResult second_heartbeat = run_client(client_path, socket_path, {"heartbeat", "2"});
    expect(second_heartbeat.exit_code == 0 &&
               second_heartbeat.output == "GLIMMER_TASK_V1 STATE 2 RUNNING\n",
           "the second worker heartbeat should observe its active lease");

    const CommandResult completed_response =
        run_client(client_path, socket_path, {"complete", "1"});
    expect(completed_response.exit_code == 0 &&
               completed_response.output == "GLIMMER_TASK_V1 STATE 1 COMPLETED\n",
           "worker completion should transition the leased task");
    const CommandResult completed_query = run_client(client_path, socket_path, {"query", "1"});
    expect(completed_query.exit_code == 0 &&
               completed_query.output == "GLIMMER_TASK_V1 STATE 1 COMPLETED\n",
           "query should observe the completed leased task");

    const CommandResult failed_response = run_client(client_path, socket_path, {"fail", "2"});
    expect(failed_response.exit_code == 0 &&
               failed_response.output == "GLIMMER_TASK_V1 STATE 2 FAILED\n",
           "worker failure should transition the second task");
    const CommandResult failed_query = run_client(client_path, socket_path, {"query", "2"});
    expect(failed_query.exit_code == 0 && failed_query.output == "GLIMMER_TASK_V1 STATE 2 FAILED\n",
           "query should observe the failed leased task");
    stop_service(service_pid);
    static_cast<void>(::unlink(socket_path.c_str()));

    const std::string bounded_socket_path = test_prefix + ".bounded.sock";
    const pid_t bounded_service_pid = start_service(service_path, bounded_socket_path, true);
    if (!wait_for_socket(bounded_socket_path)) {
        stop_service(bounded_service_pid);
        expect(false, "bounded control service should create its socket");
    }

    const CommandResult bounded_first =
        run_client(client_path, bounded_socket_path, {"submit", "tenant-a", "512", "1", "1"});
    expect(bounded_first.exit_code == 0 && bounded_first.output == "GLIMMER_TASK_V1 OK 1\n",
           "bounded service should accept work within its queue capacity");
    const CommandResult queue_full =
        run_client(client_path, bounded_socket_path, {"submit", "tenant-b", "512", "1", "1"});
    expect(queue_full.exit_code != 0 &&
               queue_full.output.find("ERROR QUEUE_FULL") != std::string::npos,
           "bounded service should expose queue backpressure to the client");

    const CommandResult bounded_lease = run_client(client_path, bounded_socket_path, {"claim"});
    expect(bounded_lease.exit_code == 0 &&
               bounded_lease.output == "GLIMMER_TASK_V1 LEASE 1 tenant-a 512 1 1\n",
           "claim should release one bounded waiting-queue slot");
    const CommandResult bounded_second =
        run_client(client_path, bounded_socket_path, {"submit", "tenant-b", "512", "1", "1"});
    expect(bounded_second.exit_code == 0 && bounded_second.output == "GLIMMER_TASK_V1 OK 2\n",
           "bounded service should accept work after a task leaves the queue");
    const CommandResult bounded_complete =
        run_client(client_path, bounded_socket_path, {"complete", "1"});
    expect(bounded_complete.exit_code == 0 &&
               bounded_complete.output == "GLIMMER_TASK_V1 STATE 1 COMPLETED\n",
           "bounded service should complete the first leased task");
    const CommandResult bounded_second_lease =
        run_client(client_path, bounded_socket_path, {"claim"});
    expect(bounded_second_lease.exit_code == 0 &&
               bounded_second_lease.output == "GLIMMER_TASK_V1 LEASE 2 tenant-b 512 1 1\n",
           "the task admitted after backpressure should become claimable");
    const CommandResult bounded_second_complete =
        run_client(client_path, bounded_socket_path, {"complete", "2"});
    expect(bounded_second_complete.exit_code == 0 &&
               bounded_second_complete.output == "GLIMMER_TASK_V1 STATE 2 COMPLETED\n",
           "the task admitted after backpressure should complete normally");
    const CommandResult bounded_stats = run_client(client_path, bounded_socket_path, {"stats"});
    expect(bounded_stats.exit_code == 0 &&
               bounded_stats.output == "GLIMMER_TASK_V1 STATS 2 0 0 2 0 0 1024 0 0 1 1\n",
           "stats should expose bounded service terminal counts and limits");
    stop_service(bounded_service_pid);
    static_cast<void>(::unlink(bounded_socket_path.c_str()));
    return EXIT_SUCCESS;
}
