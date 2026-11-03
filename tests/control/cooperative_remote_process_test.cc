#include "glimmer/control/task_execution_protocol.h"
#include "glimmer/control/shared_task_control_client.h"

#ifdef GLIMMER_TEST_COOPERATIVE_GPU
#include "cooperative_gpu_execution_fixture.h"
#endif

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <fcntl.h>
#include <iostream>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;

namespace {

using glimmer::control::SharedTaskControlClient;
using glimmer::control::TaskExecutionProtocolResponse;
using glimmer::control::TaskExecutionProtocolResponseKind;
using glimmer::control::TaskProtocolErrorCode;
using glimmer::control::TaskProtocolState;

void expect(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

class ChildProcess final {
   public:
    ChildProcess() = default;
    ~ChildProcess() {
        if (pid > 0) {
            static_cast<void>(::kill(pid, SIGTERM));
            int status = 0;
            while (::waitpid(pid, &status, 0) < 0 && errno == EINTR) {
            }
        }
    }
    ChildProcess(const ChildProcess&) = delete;
    ChildProcess& operator=(const ChildProcess&) = delete;
    ChildProcess(ChildProcess&&) = delete;
    ChildProcess& operator=(ChildProcess&&) = delete;

    void join() {
        int status = 0;
        pid_t result = -1;
        do {
            result = ::waitpid(pid, &status, 0);
        } while (result < 0 && errno == EINTR);
        expect(result == pid, "child wait should succeed");
        pid = -1;
        expect(WIFEXITED(status) && WEXITSTATUS(status) == EXIT_SUCCESS,
               "remote worker should exit successfully");
    }

    pid_t pid = -1;
};

class Pipe final {
   public:
    Pipe() {
        expect(::pipe2(descriptors.data(), O_CLOEXEC) == 0, "coordination pipe should be created");
    }
    ~Pipe() {
        for (const auto descriptor : descriptors) {
            static_cast<void>(::close(descriptor));
        }
    }
    Pipe(const Pipe&) = delete;
    Pipe& operator=(const Pipe&) = delete;
    Pipe(Pipe&&) = delete;
    Pipe& operator=(Pipe&&) = delete;
    std::array<int, 2> descriptors{};
};

class TemporarySocket final {
   public:
    TemporarySocket() {
        std::array<char, 48> pattern{};
        constexpr std::string_view name = "/tmp/glimmer-cooperative-XXXXXX";
        std::copy(name.begin(), name.end(), pattern.begin());
        const auto* created = ::mkdtemp(pattern.data());
        expect(created != nullptr, "isolated socket directory should be created");
        directory = created;
        path = directory + "/control.sock";
    }
    ~TemporarySocket() {
        static_cast<void>(::unlink(path.c_str()));
        static_cast<void>(::rmdir(directory.c_str()));
    }
    TemporarySocket(const TemporarySocket&) = delete;
    TemporarySocket& operator=(const TemporarySocket&) = delete;
    TemporarySocket(TemporarySocket&&) = delete;
    TemporarySocket& operator=(TemporarySocket&&) = delete;
    std::string directory;
    std::string path;
};

void signal_peer(int descriptor, char signal) {
    ssize_t written = -1;
    do {
        written = ::write(descriptor, &signal, 1);
    } while (written < 0 && errno == EINTR);
    expect(written == 1, "coordination signal should be delivered");
}

// The file descriptor and phase byte are separate protocol roles.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
void await_peer(int descriptor, char expected) {
    pollfd descriptor_state{.fd = descriptor, .events = POLLIN, .revents = 0};
    int ready = -1;
    do {
        ready = ::poll(&descriptor_state, 1,
#ifdef GLIMMER_TEST_COOPERATIVE_GPU
                       30'000
#else
                       5000
#endif
        );
    } while (ready < 0 && errno == EINTR);
    expect(ready == 1, "coordination should finish within the deadline");
    char signal = 0;
    ssize_t received = -1;
    do {
        received = ::read(descriptor, &signal, 1);
    } while (received < 0 && errno == EINTR);
    expect(received == 1 && signal == expected, "coordination signal should match the phase");
}
// NOLINTEND(bugprone-easily-swappable-parameters)

TaskExecutionProtocolResponse request(SharedTaskControlClient& client, std::string_view operation) {
    const auto encoded = client.request("GLIMMER_TASK_EXEC_V1 " + std::string(operation) + "\n");
    expect(encoded.has_value(), "authenticated remote request should receive a response");
    const auto parsed =
        glimmer::control::parse_task_execution_protocol_response(encoded.value_or(""));
    expect(parsed.parsed(), "remote execution response should parse");
    return parsed.response.value_or(TaskExecutionProtocolResponse{});
}

std::string identity(const TaskExecutionProtocolResponse& lease) {
    return std::to_string(lease.task_id) + " " + std::to_string(lease.generation);
}

bool invalid(const TaskExecutionProtocolResponse& response) {
    return response.kind == TaskExecutionProtocolResponseKind::kError &&
           response.error == TaskProtocolErrorCode::kInvalidRequest;
}

#ifdef GLIMMER_TEST_COOPERATIVE_GPU
void await_gpu_completion(int descriptor, SharedTaskControlClient& client,
                          const TaskExecutionProtocolResponse& paused) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{30};
    while (std::chrono::steady_clock::now() < deadline) {
        pollfd state{.fd = descriptor, .events = POLLIN, .revents = 0};
        const int ready = ::poll(&state, 1, 500);
        if (ready < 0 && errno == EINTR) {
            continue;
        }
        expect(ready >= 0, "GPU completion notification poll must succeed");
        if (ready > 0) {
            await_peer(descriptor, 'D');
            return;
        }
        expect(request(client, "EXEC_HEARTBEAT " + identity(paused)).state ==
                   TaskProtocolState::kQueued,
               "paused owner must renew its generation while another GPU process runs");
    }
    throw std::runtime_error("high-priority GPU completion deadline expired");
}
#endif

void run_worker(const char* socket, glimmer::control::TaskExecutionIdentity low) {
    SharedTaskControlClient client(socket);
    const std::string low_task = std::to_string(low.task_id);
    const std::string low_identity = low_task + " " + std::to_string(low.generation);
    for (const std::string_view operation :
         {"RETREAT", "SUSPEND", "EXEC_HEARTBEAT", "EXEC_COMPLETE", "EXEC_FAIL", "EXEC_CANCEL"}) {
        expect(invalid(request(client, std::string(operation) + " " + low_identity)),
               "SO_PEERCRED must prevent another process from reporting for the low task");
    }
    expect(request(client, "EXEC_CLAIM " + std::string(low_task)).kind ==
               TaskExecutionProtocolResponseKind::kEmpty,
           "another process cannot claim the low task");
    signal_peer(STDOUT_FILENO, 'R');
    await_peer(STDIN_FILENO, 'L');
    const auto high = request(client, "SUBMIT high 30 1 1 9");
    expect(high.kind == TaskExecutionProtocolResponseKind::kAccepted,
           "higher-priority process should submit successfully");
    expect(request(client, "EXEC_CLAIM " + std::to_string(high.task_id)).kind ==
               TaskExecutionProtocolResponseKind::kEmpty,
           "high task must wait until low physical execution acknowledges stop");
    signal_peer(STDOUT_FILENO, 'Q');
    await_peer(STDIN_FILENO, 'S');
    const auto lease = request(client, "EXEC_CLAIM " + std::to_string(high.task_id));
    expect(lease.kind == TaskExecutionProtocolResponseKind::kExecutionLease,
           "high process should own the next execution generation");
#ifdef GLIMMER_TEST_COOPERATIVE_GPU
    glimmer::test::CooperativeGpuExecution gpu;
    gpu.priority_probe();
#endif
    expect(
        request(client, "EXEC_COMPLETE " + identity(lease)).state == TaskProtocolState::kCompleted,
        "high process should complete its own generation");
    signal_peer(STDOUT_FILENO, 'D');
}

void run_parent(const char* executable, const char* service_executable) {
    TemporarySocket socket;
    ChildProcess service;
    const std::array<const char*, 12> service_arguments{service_executable,
                                                        "--socket",
                                                        socket.path.c_str(),
                                                        "--quota-bytes",
                                                        "100",
                                                        "--max-concurrent-tasks",
                                                        "1",
                                                        "--max-queued-tasks",
                                                        "8",
                                                        "--scheduler-policy",
                                                        "priority",
                                                        nullptr};
    expect(::posix_spawn(&service.pid, service_executable, nullptr, nullptr,
                         const_cast<char* const*>(service_arguments.data()), environ) == 0,
           "authenticated priority service should start");
    const auto startup_deadline = std::chrono::steady_clock::now() + std::chrono::seconds{3};
    struct stat state {};
    while (::lstat(socket.path.c_str(), &state) != 0 &&
           std::chrono::steady_clock::now() < startup_deadline) {
        static_cast<void>(::poll(nullptr, 0, 2));
    }
    expect(S_ISSOCK(state.st_mode), "control service must create its socket");
    SharedTaskControlClient client(socket.path);
    const auto low = request(client, "SUBMIT low 60 1 1 1");
    const auto original = request(client, "EXEC_CLAIM " + std::to_string(low.task_id));
    expect(original.kind == TaskExecutionProtocolResponseKind::kExecutionLease,
           "low process should acquire initial execution");
#ifdef GLIMMER_TEST_COOPERATIVE_GPU
    glimmer::test::CooperativeGpuExecution gpu;
#endif

    Pipe to_worker;
    Pipe from_worker;
    ChildProcess worker;
    posix_spawn_file_actions_t actions{};
    expect(::posix_spawn_file_actions_init(&actions) == 0, "spawn actions should initialize");
    const int input_action =
        ::posix_spawn_file_actions_adddup2(&actions, to_worker.descriptors[0], STDIN_FILENO);
    const int output_action =
        ::posix_spawn_file_actions_adddup2(&actions, from_worker.descriptors[1], STDOUT_FILENO);
    const std::string task_id = std::to_string(low.task_id);
    const std::string generation = std::to_string(original.generation);
    const std::array<const char*, 6> worker_arguments{
        executable, "--worker", socket.path.c_str(), task_id.c_str(), generation.c_str(), nullptr};
    const int spawn_result =
        input_action == 0 && output_action == 0
            ? ::posix_spawn(&worker.pid, executable, &actions, nullptr,
                            const_cast<char* const*>(worker_arguments.data()), environ)
            : -1;
    static_cast<void>(::posix_spawn_file_actions_destroy(&actions));
    expect(spawn_result == 0, "second execution process should start");
    await_peer(from_worker.descriptors[0], 'R');
#ifdef GLIMMER_TEST_COOPERATIVE_GPU
    gpu.launch(3);
    gpu.await_progress();
#endif
    signal_peer(to_worker.descriptors[1], 'L');
    await_peer(from_worker.descriptors[0], 'Q');
    const auto retreat = request(client, "RETREAT " + identity(original));
    expect(retreat.kind == TaskExecutionProtocolResponseKind::kRetreat && retreat.should_retreat,
           "cross-process higher-priority arrival must request retreat");
#ifdef GLIMMER_TEST_COOPERATIVE_GPU
    const auto progress = gpu.stop();
    std::cout << "remote physical retreat retained " << progress << " logical blocks\n";
#endif
    expect(request(client, "SUSPEND " + identity(original)).state == TaskProtocolState::kQueued,
           "owner physical-stop report should release running capacity");
    signal_peer(to_worker.descriptors[1], 'S');
#ifdef GLIMMER_TEST_COOPERATIVE_GPU
    await_gpu_completion(from_worker.descriptors[0], client, original);
#else
    await_peer(from_worker.descriptors[0], 'D');
#endif
    worker.join();
    const auto resumed = request(client, "EXEC_CLAIM " + std::to_string(low.task_id));
    expect(resumed.kind == TaskExecutionProtocolResponseKind::kExecutionLease &&
               resumed.generation > original.generation,
           "low process resumes the same task under a fresh generation");
    expect(invalid(request(client, "EXEC_COMPLETE " + identity(original))),
           "delayed pre-retreat completion cannot retire resumed execution over the socket");
#ifdef GLIMMER_TEST_COOPERATIVE_GPU
    gpu.launch(7);
    gpu.finish();
#endif
    expect(request(client, "EXEC_COMPLETE " + identity(resumed)).state ==
               TaskProtocolState::kCompleted,
           "resumed owner can complete execution");
    const auto stats = request(client, "STATS");
    expect(stats.stats.quota_reserved_bytes == 0 && stats.stats.quota_allocated_bytes == 0 &&
               stats.stats.completed_task_count == 2 && stats.stats.running_task_count == 0,
           "two-process priority resume should release quota exactly once for each logical task");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 5 && std::string_view(argv[1]) == "--worker") {
            const auto parsed = glimmer::control::parse_task_execution_protocol_request(
                "GLIMMER_TASK_EXEC_V1 RETREAT " + std::string(argv[3]) + " " + argv[4]);
            expect(parsed.parsed(), "worker identity must parse");
            if (!parsed.request) throw std::runtime_error("missing worker identity");
            run_worker(argv[2], {parsed.request->task_id, parsed.request->generation});
        } else {
            expect(argc == 2, "remote test expects the service executable path");
            run_parent(argv[0], argv[1]);
        }
    } catch (const std::exception& error) {
        std::cerr << "Test failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
