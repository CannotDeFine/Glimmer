#include "glimmer/control/task_protocol.h"
#include "glimmer/control/unix_socket_client.h"

#include <cuda.h>

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>

namespace {

constexpr std::uint32_t kDefaultTasks = 1;
constexpr std::uint32_t kDefaultIdlePolls = 100;
constexpr std::uint32_t kDefaultHeartbeatIntervalMilliseconds = 1'000;
constexpr std::uint32_t kIdleSleepMilliseconds = 10;

constexpr char kNoopPtx[] = R"ptx(
.version 7.0
.target sm_50
.address_size 64

.visible .entry glimmer_remote_noop()
{
    ret;
}
)ptx";

struct Options {
    std::string socket_path;
    std::uint32_t task_count = kDefaultTasks;
    std::uint32_t idle_polls = kDefaultIdlePolls;
    std::uint32_t heartbeat_interval_ms = kDefaultHeartbeatIntervalMilliseconds;
    std::uint32_t execution_delay_ms = 0;
    int device_index = 0;
};

struct ContextGuard {
    CUcontext handle = nullptr;

    ~ContextGuard() {
        if (handle != nullptr) {
            static_cast<void>(cuCtxDestroy(handle));
        }
    }
};

struct ModuleGuard {
    CUmodule handle = nullptr;

    ~ModuleGuard() {
        if (handle != nullptr) {
            static_cast<void>(cuModuleUnload(handle));
        }
    }
};

void print_usage(std::ostream& output, std::string_view program) {
    output << "Usage: " << program
           << " --socket PATH [--tasks N] [--idle-polls N] [--device INDEX]"
              " [--heartbeat-interval-ms N] [--execution-delay-ms N]\n";
}

template <typename Integer>
bool parse_unsigned(std::string_view text, Integer* value) {
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

bool parse_options(int argc, char** argv, Options* options) {
    if (options == nullptr) {
        return false;
    }
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument = argv[index];
        if (argument == "--help") {
            print_usage(std::cout, argv[0]);
            return false;
        }
        if (index + 1 >= argc) {
            return false;
        }
        const std::string_view value = argv[++index];
        if (argument == "--socket") {
            options->socket_path = std::string(value);
        } else if (argument == "--tasks") {
            if (!parse_unsigned(value, &options->task_count) || options->task_count == 0) {
                return false;
            }
        } else if (argument == "--idle-polls") {
            if (!parse_unsigned(value, &options->idle_polls)) {
                return false;
            }
        } else if (argument == "--heartbeat-interval-ms") {
            if (!parse_unsigned(value, &options->heartbeat_interval_ms)) {
                return false;
            }
        } else if (argument == "--execution-delay-ms") {
            if (!parse_unsigned(value, &options->execution_delay_ms)) {
                return false;
            }
        } else if (argument == "--device") {
            std::uint32_t parsed_device = 0;
            if (!parse_unsigned(value, &parsed_device) ||
                parsed_device > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
                return false;
            }
            options->device_index = static_cast<int>(parsed_device);
        } else {
            return false;
        }
    }
    return !options->socket_path.empty();
}

bool check_cuda(CUresult result, std::string_view operation) {
    if (result == CUDA_SUCCESS) {
        return true;
    }
    std::cerr << operation << " failed with CUresult " << static_cast<int>(result) << '\n';
    return false;
}

std::optional<glimmer::control::TaskProtocolResponse> send_request(
    const glimmer::control::UnixSocketControlClient& client,
    const glimmer::control::TaskProtocolRequest& request) {
    const auto encoded = glimmer::control::format_task_protocol_request(request);
    if (!encoded.has_value()) {
        return std::nullopt;
    }
    const auto response_line = client.request(encoded.value());
    if (!response_line.has_value()) {
        return std::nullopt;
    }
    const auto parsed = glimmer::control::parse_task_protocol_response(response_line.value());
    if (!parsed.parsed() || !parsed.response.has_value()) {
        return std::nullopt;
    }
    return parsed.response;
}

bool report_terminal(const glimmer::control::UnixSocketControlClient& client,
                     glimmer::core::TaskId task_id, bool success) {
    const glimmer::control::TaskProtocolRequest request{
        .operation = success ? glimmer::control::TaskProtocolOperation::kComplete
                             : glimmer::control::TaskProtocolOperation::kFail,
        .admission = {},
        .task_id = task_id};
    const auto response = send_request(client, request);
    return response.has_value() &&
           response->kind == glimmer::control::TaskProtocolResponseKind::kState &&
           response->task_id == task_id &&
           response->state == (success ? glimmer::control::TaskProtocolState::kCompleted
                                       : glimmer::control::TaskProtocolState::kFailed);
}

bool heartbeat_lease(const glimmer::control::UnixSocketControlClient& client,
                     glimmer::core::TaskId task_id) {
    const glimmer::control::TaskProtocolRequest request{
        .operation = glimmer::control::TaskProtocolOperation::kHeartbeat,
        .admission = {},
        .task_id = task_id};
    const auto response = send_request(client, request);
    return response.has_value() &&
           response->kind == glimmer::control::TaskProtocolResponseKind::kState &&
           response->task_id == task_id &&
           response->state == glimmer::control::TaskProtocolState::kRunning;
}

bool execute_lease(const glimmer::control::UnixSocketControlClient& client,
                   const glimmer::control::TaskProtocolResponse& lease, CUfunction function,
                   const Options& options) {
    std::atomic_bool heartbeat_stop = false;
    std::atomic_bool heartbeat_failed = false;
    std::condition_variable heartbeat_condition;
    std::mutex heartbeat_mutex;
    std::thread heartbeat_thread;
    if (options.heartbeat_interval_ms != 0) {
        try {
            const auto heartbeat_interval =
                std::chrono::milliseconds{options.heartbeat_interval_ms};
            heartbeat_thread = std::thread([&client, task_id = lease.task_id, heartbeat_interval,
                                            &heartbeat_stop, &heartbeat_failed,
                                            &heartbeat_condition, &heartbeat_mutex] {
                std::unique_lock lock(heartbeat_mutex);
                while (!heartbeat_stop.load()) {
                    if (heartbeat_condition.wait_for(lock, heartbeat_interval, [&heartbeat_stop] {
                            return heartbeat_stop.load();
                        })) {
                        break;
                    }
                    lock.unlock();
                    bool heartbeat_succeeded = false;
                    try {
                        heartbeat_succeeded = heartbeat_lease(client, task_id);
                    } catch (...) {
                        heartbeat_succeeded = false;
                    }
                    if (!heartbeat_succeeded) {
                        heartbeat_failed.store(true);
                        lock.lock();
                        break;
                    }
                    lock.lock();
                }
            });
        } catch (...) {
            std::cerr << "failed to start lease heartbeat\n";
            return false;
        }
    }

    CUdeviceptr allocation = 0;
    if (!check_cuda(cuMemAlloc(&allocation, lease.memory_bytes), "cuMemAlloc")) {
        heartbeat_stop.store(true);
        heartbeat_condition.notify_one();
        if (heartbeat_thread.joinable()) {
            heartbeat_thread.join();
        }
        return false;
    }

    const unsigned int grid_x = std::min(lease.work_units, 65'535U);
    const CUresult launch_result =
        cuLaunchKernel(function, grid_x, 1, 1, 1, 1, 1, 0, nullptr, nullptr, nullptr);
    const bool launched = check_cuda(launch_result, "cuLaunchKernel");
    if (launched && options.execution_delay_ms != 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds{options.execution_delay_ms});
    }
    const bool synchronized = launched && check_cuda(cuCtxSynchronize(), "cuCtxSynchronize");
    const bool released = check_cuda(cuMemFree(allocation), "cuMemFree");
    heartbeat_stop.store(true);
    heartbeat_condition.notify_one();
    if (heartbeat_thread.joinable()) {
        heartbeat_thread.join();
    }
    return launched && synchronized && released && !heartbeat_failed.load();
}

int run_worker(const Options& options) {
    if (!check_cuda(cuInit(0), "cuInit")) {
        return EXIT_FAILURE;
    }
    CUdevice device{};
    if (!check_cuda(cuDeviceGet(&device, options.device_index), "cuDeviceGet")) {
        return EXIT_FAILURE;
    }
    ContextGuard context;
    if (!check_cuda(cuCtxCreate(&context.handle, nullptr, 0, device), "cuCtxCreate")) {
        return EXIT_FAILURE;
    }
    ModuleGuard module;
    if (!check_cuda(cuModuleLoadData(&module.handle, kNoopPtx), "cuModuleLoadData")) {
        return EXIT_FAILURE;
    }
    CUfunction function = nullptr;
    if (!check_cuda(cuModuleGetFunction(&function, module.handle, "glimmer_remote_noop"),
                    "cuModuleGetFunction")) {
        return EXIT_FAILURE;
    }

    const glimmer::control::UnixSocketControlClient client(options.socket_path);
    std::uint32_t completed_tasks = 0;
    std::uint32_t idle_polls = 0;
    while (completed_tasks < options.task_count) {
        const glimmer::control::TaskProtocolRequest claim{
            .operation = glimmer::control::TaskProtocolOperation::kClaim,
            .admission = {},
            .task_id = 0};
        const auto response = send_request(client, claim);
        if (!response.has_value()) {
            std::cerr << "claim request failed\n";
            return EXIT_FAILURE;
        }
        if (response->kind == glimmer::control::TaskProtocolResponseKind::kEmpty) {
            if (idle_polls++ >= options.idle_polls) {
                std::cout << "glimmer_cuda_lease_worker status=idle completed=" << completed_tasks
                          << '\n';
                return completed_tasks == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(kIdleSleepMilliseconds));
            continue;
        }
        if (response->kind != glimmer::control::TaskProtocolResponseKind::kLease) {
            std::cerr << "claim returned an unexpected protocol response\n";
            return EXIT_FAILURE;
        }
        idle_polls = 0;
        std::cout << "glimmer_cuda_lease_worker lease task_id=" << response->task_id
                  << " tenant=" << response->tenant_id << " memory_bytes=" << response->memory_bytes
                  << " work_units=" << response->work_units << '\n';
        const bool execution_succeeded = execute_lease(client, response.value(), function, options);
        if (!report_terminal(client, response->task_id, execution_succeeded)) {
            std::cerr << "failed to report terminal state for task " << response->task_id << '\n';
            return EXIT_FAILURE;
        }
        if (!execution_succeeded) {
            return EXIT_FAILURE;
        }
        ++completed_tasks;
    }
    std::cout << "glimmer_cuda_lease_worker status=ok completed=" << completed_tasks << '\n';
    return EXIT_SUCCESS;
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        print_usage(std::cout, argv[0]);
        return EXIT_SUCCESS;
    }
    if (!parse_options(argc, argv, &options)) {
        print_usage(std::cerr, argv[0]);
        return EXIT_FAILURE;
    }
    return run_worker(options);
}
