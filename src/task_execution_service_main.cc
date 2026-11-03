#include "glimmer/control/task_execution_endpoint.h"
#include "glimmer/control/unix_socket_task_execution_server.h"
#if defined(GLIMMER_TASK_SERVICE_TESTING)
#include "dispatch_test_hooks.h"
#endif

#include <charconv>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

namespace {
using namespace glimmer;
volatile std::sig_atomic_t stop_requested = 0;
void request_stop(int) noexcept {
    stop_requested = 1;
}

struct Options {
    std::string socket;
    core::MemoryBytes quota = 0;
    std::size_t max_running = 0;
    std::size_t max_queued = 0;
    bool has_queued = false;
    core::SchedulingPolicy policy = core::SchedulingPolicy::kWeightedRoundRobin;
    std::size_t reserved = 0;
    std::uint32_t threshold = 1;
    control::NativeEpochExtent epoch{};
    unsigned epoch_fields = 0;
};

template <typename T>
bool number(std::string_view text, T& result) {
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), result);
    return !text.empty() && parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size();
}

bool parse_options(int argc, char** argv, Options& options) {
    for (int index = 1; index < argc; ++index) {
        const std::string_view option = argv[index];
        if (index + 1 >= argc) return false;
        const std::string_view value = argv[++index];
        if (option == "--socket")
            options.socket = value;
        else if (option == "--quota-bytes") {
            if (!number(value, options.quota)) return false;
        } else if (option == "--max-concurrent-tasks") {
            if (!number(value, options.max_running)) return false;
        } else if (option == "--max-queued-tasks") {
            if (!number(value, options.max_queued)) return false;
            options.has_queued = true;
        } else if (option == "--scheduler-policy") {
            const auto policy = core::parse_scheduling_policy(value);
            if (!policy) return false;
            options.policy = *policy;
        } else if (option == "--priority-reserved-slots") {
            if (!number(value, options.reserved)) return false;
        } else if (option == "--priority-threshold") {
            if (!number(value, options.threshold)) return false;
        } else if (option == "--native-epoch-max-calls") {
            if ((options.epoch_fields & 1U) || !number(value, options.epoch.max_calls))
                return false;
            options.epoch_fields |= 1U;
        } else if (option == "--native-epoch-prefix-calls") {
            if ((options.epoch_fields & 2U) || !number(value, options.epoch.calls_per_prefix))
                return false;
            options.epoch_fields |= 2U;
        } else if (option == "--native-epoch-pending-prefixes") {
            if ((options.epoch_fields & 4U) || !number(value, options.epoch.max_pending_prefixes))
                return false;
            options.epoch_fields |= 4U;
        } else if (option == "--native-epoch-acceptance-us") {
            if ((options.epoch_fields & 8U) ||
                !number(value, options.epoch.acceptance_microseconds))
                return false;
            options.epoch_fields |= 8U;
        } else
            return false;
    }
    return !options.socket.empty() && options.quota > 0 && options.max_running > 0 &&
           options.has_queued && options.max_queued > 0 &&
           options.reserved <= options.max_running &&
           (options.epoch_fields == 0 ||
            (options.epoch_fields == 15 && control::valid_native_epoch_extent(options.epoch)));
}

bool trusted_parent(const std::string& socket) {
    const std::filesystem::path path(socket);
    const auto parent = path.has_parent_path() ? path.parent_path() : std::filesystem::path(".");
    struct stat status {};
    return ::stat(parent.c_str(), &status) == 0 && S_ISDIR(status.st_mode) &&
           status.st_uid == ::geteuid() && (status.st_mode & (S_IWGRP | S_IWOTH)) == 0;
}
bool register_task(void*, core::TaskId) noexcept {
#if defined(GLIMMER_TASK_SERVICE_TESTING)
    if (std::getenv("GLIMMER_TEST_DISPATCH_UNCERTAIN") != nullptr) {
        core::internal::dispatch_test_faults =
            core::internal::kDispatchFailSnapshot | core::internal::kDispatchFailCancel;
    }
#endif
    return true;
}
void rollback_task(void*, core::TaskId) noexcept {}
bool empty(const control::TaskExecutionLeaseStatus& state) {
    return state.queued_tasks == 0 && state.running_tasks == 0 && state.pending_legacy == 0 &&
           state.pending_cooperative == 0 && state.pending_native == 0 &&
           state.active_legacy == 0 && state.active_cooperative == 0 && state.active_native == 0 &&
           state.reaping == 0;
}

int run(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        std::cout << "Usage: " << argv[0]
                  << " --socket PATH --quota-bytes N --max-concurrent-tasks N "
                     "--max-queued-tasks N [--scheduler-policy fifo|weighted_rr|drr|priority] "
                     "[--priority-reserved-slots N] [--priority-threshold N] "
                     "[--native-epoch-max-calls N --native-epoch-prefix-calls N "
                     "--native-epoch-pending-prefixes N --native-epoch-acceptance-us N]\n";
        return 0;
    }
    Options options;
    if (!parse_options(argc, argv, options) || !trusted_parent(options.socket)) {
        std::cerr << "invalid options or untrusted socket directory\n";
        return 1;
    }
    struct sigaction action {};
    action.sa_handler = request_stop;
    ::sigemptyset(&action.sa_mask);
    if (::sigaction(SIGINT, &action, nullptr) != 0 || ::sigaction(SIGTERM, &action, nullptr) != 0)
        return 1;
    auto admission = std::make_unique<control::TaskExecutionAdmissionService>(
        options.quota,
        core::SchedulerOptions{.max_running_tasks = options.max_running,
                               .max_queued_tasks = options.max_queued,
                               .priority_reserved_slots = options.reserved,
                               .priority_reservation_threshold = options.threshold,
                               .scheduling_policy = options.policy},
        std::chrono::milliseconds::zero(), true, nullptr, nullptr, nullptr, nullptr,
        options.epoch_fields == 15 ? std::optional{options.epoch} : std::nullopt);
    control::TaskExecutionEndpoint endpoint(std::move(admission), register_task, rollback_task,
                                            nullptr);
    control::UnixSocketTaskExecutionServer server(endpoint, {.socket_path = options.socket});
    if (!server.start()) {
        std::cerr << "unable to start task execution service\n";
        return 1;
    }
    bool draining = false;
    bool failed = false;
    for (;;) {
        if (stop_requested && !draining) {
            draining = endpoint.begin_shutdown();
            if (!draining) failed = true;
        }
        if (draining) {
            const auto status = endpoint.lease_status();
            if (status && empty(*status)) break;
        }
        if (server.serve_one_for(100) == control::UnixSocketServeStatus::kError) {
            failed = true;
            stop_requested = 1;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
    server.stop();
    return failed ? 1 : 0;
}
}  // namespace

int main(int argc, char** argv) {
    return run(argc, argv);
}
