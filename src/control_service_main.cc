#include "glimmer/core/adaptive_slo.h"
#include "glimmer/backend/task_backend.h"
#include "glimmer/control/task_endpoint.h"
#include "glimmer/control/unix_socket_server.h"

#include <charconv>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <mutex>
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
    std::uint32_t priority_reserved_slots = 0;
    std::uint32_t priority_reservation_threshold = 1;
    std::uint64_t adaptive_slo_target_queue_wait_us = 0;
    std::uint32_t adaptive_slo_window = 32;
    std::uint32_t adaptive_slo_max_reserved_slots = 0;
    std::uint64_t max_requests = 0;
    glimmer::core::SchedulingPolicy scheduling_policy =
        glimmer::core::SchedulingPolicy::kWeightedRoundRobin;
    bool bind_leases_to_process = false;
    bool trace_scheduler = false;
    ExecutionMode execution_mode = ExecutionMode::kSimulated;
};

void print_usage(std::ostream& output, std::string_view program) {
    output << "Usage: " << program
           << " --socket PATH --quota-bytes BYTES [--execution-mode simulated|remote]"
              " [--completion-steps N] [--lease-timeout-ms N]"
              " [--max-concurrent-tasks N] [--max-queued-tasks N]"
              " [--priority-reserved-slots N] [--priority-threshold N]"
              " [--adaptive-slo-target-queue-us N] [--adaptive-slo-window N]"
              " [--adaptive-slo-max-reserved-slots N]"
              " [--scheduler-policy fifo|weighted_rr|drr|priority]"
              " [--max-requests N] [--bind-leases-to-process] [--trace-scheduler]\n";
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
        if (argument == "--trace-scheduler") {
            options->trace_scheduler = true;
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
        } else if (argument == "--priority-reserved-slots") {
            if (!parse_nonnegative(value, &options->priority_reserved_slots)) {
                return false;
            }
        } else if (argument == "--priority-threshold") {
            if (!parse_nonnegative(value, &options->priority_reservation_threshold)) {
                return false;
            }
        } else if (argument == "--adaptive-slo-target-queue-us") {
            if (!parse_nonnegative(value, &options->adaptive_slo_target_queue_wait_us)) {
                return false;
            }
        } else if (argument == "--adaptive-slo-window") {
            if (!parse_positive(value, &options->adaptive_slo_window)) {
                return false;
            }
        } else if (argument == "--adaptive-slo-max-reserved-slots") {
            if (!parse_nonnegative(value, &options->adaptive_slo_max_reserved_slots)) {
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
    if (options->socket_path.empty() || options->quota_bytes == 0 ||
        options->priority_reserved_slots > options->max_concurrent_tasks) {
        return false;
    }
    if (options->adaptive_slo_target_queue_wait_us == 0) {
        return true;
    }
    if (options->scheduling_policy != glimmer::core::SchedulingPolicy::kPriority) {
        return false;
    }
    const std::uint32_t maximum_reserved_slots =
        options->adaptive_slo_max_reserved_slots == 0
            ? (options->max_concurrent_tasks > 1 ? options->max_concurrent_tasks - 1 : 0)
            : options->adaptive_slo_max_reserved_slots;
    return maximum_reserved_slots != 0 && maximum_reserved_slots <= options->max_concurrent_tasks &&
           options->priority_reserved_slots <= maximum_reserved_slots;
}

bool register_simulated_task(void* context, glimmer::core::TaskId) noexcept {
    return context != nullptr;
}

void trace_scheduler_dispatch(
    void*, const glimmer::control::TaskDispatchObservation& observation) noexcept {
    static_cast<void>(std::fprintf(
        stderr,
        "[glimmer] scheduler dispatch sequence=%llu task_id=%llu tenant=%s priority=%u "
        "memory_bytes=%llu work_units=%u timestamp_ns=%llu\n",
        static_cast<unsigned long long>(observation.sequence),
        static_cast<unsigned long long>(observation.task.task_id),
        observation.task.tenant_id.c_str(), observation.task.priority,
        static_cast<unsigned long long>(observation.task.memory_bytes), observation.task.work_units,
        static_cast<unsigned long long>(observation.dispatched_at_nanoseconds)));
}

struct AdaptiveSloObserverContext {
    glimmer::core::Scheduler* scheduler = nullptr;
    glimmer::core::AdaptiveSloController* controller = nullptr;
    // Keep recommendation and application ordered across completion threads.
    std::mutex update_mutex;
    bool trace = false;
};

void observe_adaptive_completion(
    void* context, const glimmer::control::TaskCompletionObservation& observation) noexcept {
    if (context == nullptr) {
        return;
    }
    auto* state = static_cast<AdaptiveSloObserverContext*>(context);
    if (state->scheduler == nullptr || state->controller == nullptr) {
        return;
    }
    std::scoped_lock update_lock(state->update_mutex);
    const auto recommendation = state->controller->observe(
        {.priority = observation.task.priority,
         .queue_wait_microseconds = observation.queue_wait_microseconds});
    if (!recommendation.has_value() ||
        !state->scheduler->set_priority_reserved_slots(recommendation.value())) {
        return;
    }
    if (state->trace) {
        static_cast<void>(
            std::fprintf(stderr,
                         "[glimmer] adaptive reservation slots=%zu priority=%u queue_wait_us=%llu "
                         "window_samples=%zu\n",
                         recommendation.value(), observation.task.priority,
                         static_cast<unsigned long long>(observation.queue_wait_microseconds),
                         state->controller->observed_window_samples()));
    }
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
        glimmer::core::SchedulerOptions{
            .max_running_tasks = options.max_concurrent_tasks,
            .max_queued_tasks = options.max_queued_tasks,
            .priority_reserved_slots = options.priority_reserved_slots,
            .priority_reservation_threshold = options.priority_reservation_threshold,
            .scheduling_policy = options.scheduling_policy});
    glimmer::backend::SimulatedBackend backend(options.completion_steps);
    glimmer::backend::TaskExecutor executor(scheduler, backend);
    std::optional<glimmer::core::AdaptiveSloController> adaptive_controller;
    AdaptiveSloObserverContext adaptive_context;
    if (options.adaptive_slo_target_queue_wait_us != 0) {
        const std::size_t maximum_reserved_slots =
            options.adaptive_slo_max_reserved_slots == 0
                ? (options.max_concurrent_tasks > 1 ? options.max_concurrent_tasks - 1 : 0)
                : options.adaptive_slo_max_reserved_slots;
        adaptive_controller.emplace(glimmer::core::AdaptiveSloOptions{
            .initial_reserved_slots = options.priority_reserved_slots,
            .min_reserved_slots = 0,
            .max_reserved_slots = maximum_reserved_slots,
            .priority_threshold = options.priority_reservation_threshold,
            .target_queue_wait_microseconds = options.adaptive_slo_target_queue_wait_us,
            .observation_window = options.adaptive_slo_window,
            .violation_ratio_percent = 25,
            .increase_step = 1,
            .decrease_step = 1});
        adaptive_context.scheduler = &scheduler;
        adaptive_context.controller = &adaptive_controller.value();
        adaptive_context.trace = options.trace_scheduler;
    }
    glimmer::control::TaskAdmissionService admission_service(
        scheduler, std::chrono::milliseconds(options.lease_timeout_ms),
        options.bind_leases_to_process,
        options.trace_scheduler ? trace_scheduler_dispatch : nullptr, nullptr,
        adaptive_controller.has_value() ? observe_adaptive_completion : nullptr, &adaptive_context);
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

    while (g_stop_requested == 0 &&
           (options.max_requests == 0 || server.served_request_count() < options.max_requests)) {
        static_cast<void>(admission_service.reap_expired());
        if (options.execution_mode == ServiceOptions::ExecutionMode::kSimulated) {
            backend.advance();
            const auto execution = executor.step();
            if (execution.status == glimmer::backend::ExecutorStepStatus::kSubmitted &&
                execution.task_id.has_value()) {
                admission_service.observe_external_dispatch(execution.task_id.value());
            } else if (execution.status == glimmer::backend::ExecutorStepStatus::kCompleted &&
                       execution.task_id.has_value()) {
                admission_service.observe_external_completion(execution.task_id.value(),
                                                              glimmer::core::TaskState::kCompleted);
            } else if (execution.status == glimmer::backend::ExecutorStepStatus::kFailed &&
                       execution.task_id.has_value()) {
                admission_service.observe_external_completion(execution.task_id.value(),
                                                              glimmer::core::TaskState::kFailed);
            } else if (execution.status == glimmer::backend::ExecutorStepStatus::kCancelled &&
                       execution.task_id.has_value()) {
                admission_service.observe_external_completion(execution.task_id.value(),
                                                              glimmer::core::TaskState::kCancelled);
            }
            if (execution.status != glimmer::backend::ExecutorStepStatus::kIdle &&
                execution.status != glimmer::backend::ExecutorStepStatus::kPending) {
                admission_service.notify_scheduler_change();
            }
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
