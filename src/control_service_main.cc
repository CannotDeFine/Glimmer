#include "glimmer/backend/task_backend.h"
#include "glimmer/control/task_endpoint.h"
#include "glimmer/control/unix_socket_server.h"

#include <charconv>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>

namespace {

volatile std::sig_atomic_t g_stop_requested = 0;

void request_stop(int) noexcept {
    g_stop_requested = 1;
}

struct ServiceOptions {
    enum class ExecutionMode : std::uint8_t { kSimulated, kRemote };

    std::string socket_path;
    glimmer::core::MemoryBytes quota_bytes = 0;
    std::uint32_t completion_steps = 1;
    std::uint32_t lease_timeout_ms = 0;
    std::uint32_t max_concurrent_tasks = 1;
    std::uint32_t max_queued_tasks = 0;
    std::uint64_t max_requests = 0;
    glimmer::core::SchedulingPolicy scheduling_policy =
        glimmer::core::SchedulingPolicy::kWeightedRoundRobin;
    bool bind_leases_to_process = false;
    ExecutionMode execution_mode = ExecutionMode::kSimulated;
};

void print_usage(std::ostream& output, std::string_view program) {
    output << "Usage: " << program
           << " --socket PATH --quota-bytes BYTES [--execution-mode simulated|remote]"
              " [--completion-steps N] [--lease-timeout-ms N]"
              " [--max-concurrent-tasks N] [--max-queued-tasks N]"
              " [--scheduler-policy fifo|weighted_rr|drr|priority]"
              " [--max-requests N] [--bind-leases-to-process]\n";
}

template <typename Integer>
bool parse_positive(std::string_view text, Integer* value) {
    if (value == nullptr || text.empty()) {
        return false;
    }
    Integer parsed = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), parsed);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size() || parsed == 0) {
        return false;
    }
    *value = parsed;
    return true;
}

template <typename Integer>
bool parse_nonnegative(std::string_view text, Integer* value) {
    if (value == nullptr || text.empty()) {
        return false;
    }
    Integer parsed = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), parsed);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) {
        return false;
    }
    *value = parsed;
    return true;
}

bool parse_options(int argc, char** argv, ServiceOptions* options) {
    if (options == nullptr) {
        return false;
    }
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument = argv[index];
        if (argument == "--help") {
            print_usage(std::cout, argv[0]);
            return false;
        }
        if (argument == "--bind-leases-to-process") {
            options->bind_leases_to_process = true;
            continue;
        }
        if (index + 1 >= argc) {
            return false;
        }
        const std::string_view value = argv[++index];
        if (argument == "--socket") {
            options->socket_path = std::string(value);
        } else if (argument == "--quota-bytes") {
            if (!parse_positive(value, &options->quota_bytes)) {
                return false;
            }
        } else if (argument == "--execution-mode") {
            if (value == "simulated") {
                options->execution_mode = ServiceOptions::ExecutionMode::kSimulated;
            } else if (value == "remote") {
                options->execution_mode = ServiceOptions::ExecutionMode::kRemote;
            } else {
                return false;
            }
        } else if (argument == "--completion-steps") {
            if (!parse_positive(value, &options->completion_steps)) {
                return false;
            }
        } else if (argument == "--lease-timeout-ms") {
            if (!parse_nonnegative(value, &options->lease_timeout_ms)) {
                return false;
            }
        } else if (argument == "--max-concurrent-tasks") {
            if (!parse_positive(value, &options->max_concurrent_tasks)) {
                return false;
            }
        } else if (argument == "--max-queued-tasks") {
            if (!parse_nonnegative(value, &options->max_queued_tasks)) {
                return false;
            }
        } else if (argument == "--scheduler-policy") {
            const auto policy = glimmer::core::parse_scheduling_policy(value);
            if (!policy.has_value()) {
                return false;
            }
            options->scheduling_policy = policy.value();
        } else if (argument == "--max-requests") {
            if (!parse_positive(value, &options->max_requests)) {
                return false;
            }
        } else {
            return false;
        }
    }
    return !options->socket_path.empty() && options->quota_bytes != 0;
}

bool register_simulated_task(void* context, glimmer::core::TaskId) noexcept {
    return context != nullptr;
}

}  // namespace

int run_service(int argc, char** argv) {
    ServiceOptions options;
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        print_usage(std::cout, argv[0]);
        return EXIT_SUCCESS;
    }
    if (!parse_options(argc, argv, &options)) {
        print_usage(std::cerr, argv[0]);
        return EXIT_FAILURE;
    }

    struct sigaction action {};
    action.sa_handler = request_stop;
    sigemptyset(&action.sa_mask);
    if (::sigaction(SIGINT, &action, nullptr) != 0 || ::sigaction(SIGTERM, &action, nullptr) != 0) {
        std::cerr << "failed to install shutdown handlers\n";
        return EXIT_FAILURE;
    }

    glimmer::core::Scheduler scheduler(
        options.quota_bytes,
        glimmer::core::SchedulerOptions{.max_running_tasks = options.max_concurrent_tasks,
                                        .max_queued_tasks = options.max_queued_tasks,
                                        .scheduling_policy = options.scheduling_policy});
    glimmer::backend::SimulatedBackend backend(options.completion_steps);
    glimmer::backend::TaskExecutor executor(scheduler, backend);
    glimmer::control::TaskAdmissionService admission_service(
        scheduler, std::chrono::milliseconds(options.lease_timeout_ms),
        options.bind_leases_to_process);
    glimmer::control::TaskControlEndpoint endpoint(admission_service, register_simulated_task,
                                                   &backend);
    glimmer::control::UnixSocketControlServer server(
        endpoint, glimmer::control::UnixSocketControlConfig{.socket_path = options.socket_path,
                                                            .allowed_uid = std::nullopt,
                                                            .io_timeout_ms = 1000});
    if (!server.start()) {
        std::cerr << "failed to start control service on " << options.socket_path << '\n';
        return EXIT_FAILURE;
    }

    std::uint64_t request_count = 0;
    while (g_stop_requested == 0 &&
           (options.max_requests == 0 || request_count < options.max_requests)) {
        static_cast<void>(admission_service.reap_expired());
        if (options.execution_mode == ServiceOptions::ExecutionMode::kSimulated) {
            backend.advance();
            static_cast<void>(executor.step());
        }
        const auto serve_status = server.serve_one_for(100);
        if (serve_status == glimmer::control::UnixSocketServeStatus::kTimedOut) {
            continue;
        }
        if (serve_status == glimmer::control::UnixSocketServeStatus::kError) {
            if (g_stop_requested != 0) {
                break;
            }
            std::cerr << "control service failed while serving a client\n";
            server.stop();
            return EXIT_FAILURE;
        }
        ++request_count;
    }
    server.stop();
    return EXIT_SUCCESS;
}

int main(int argc, char** argv) noexcept {
    try {
        return run_service(argc, argv);
    } catch (...) {
        std::cerr << "control service terminated after an unexpected exception\n";
        return EXIT_FAILURE;
    }
}
