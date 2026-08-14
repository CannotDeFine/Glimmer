#include "internal/cuda_task_backend.h"

#include <cuda.h>

#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <system_error>
#include <string_view>
#include <thread>

namespace {

constexpr std::uint64_t kDefaultQuotaBytes = std::uint64_t{8} * 1024 * 1024;
constexpr std::uint64_t kDefaultTaskMemoryBytes = std::uint64_t{1} * 1024 * 1024;
constexpr std::uint32_t kDefaultTaskCount = 2;
constexpr std::size_t kMaximumPolls = 1'000'000;

constexpr char kNoopPtx[] = R"ptx(
.version 7.0
.target sm_50
.address_size 64

.visible .entry glimmer_noop()
{
    ret;
}
)ptx";

struct Options {
    std::uint64_t quota_bytes = kDefaultQuotaBytes;
    std::uint64_t task_memory_bytes = kDefaultTaskMemoryBytes;
    std::uint32_t task_count = kDefaultTaskCount;
};

struct ContextGuard {
    CUcontext handle = nullptr;

    ~ContextGuard() {
        if (handle != nullptr) {
            const CUresult result = cuCtxDestroy(handle);
            if (result != CUDA_SUCCESS) {
                std::cerr << "cuCtxDestroy failed with CUresult " << static_cast<int>(result)
                          << '\n';
            }
        }
    }
};

struct ModuleGuard {
    CUmodule handle = nullptr;

    ~ModuleGuard() {
        if (handle != nullptr) {
            const CUresult result = cuModuleUnload(handle);
            if (result != CUDA_SUCCESS) {
                std::cerr << "cuModuleUnload failed with CUresult " << static_cast<int>(result)
                          << '\n';
            }
        }
    }
};

void print_usage() {
    std::cout << "Usage: glimmer_cuda_task_backend_demo [options]\n"
              << "  --quota-bytes BYTES        scheduler quota (default: 8388608)\n"
              << "  --task-memory-bytes BYTES  reservation per task (default: 1048576)\n"
              << "  --tasks COUNT              queued tasks (default: 2)\n"
              << "  --help                     show this message\n";
}

bool parse_unsigned(std::string_view value, std::uint64_t* result) {
    if (result == nullptr || value.empty()) {
        return false;
    }
    std::uint64_t parsed = 0;
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (error != std::errc{} || end != value.data() + value.size()) {
        return false;
    }
    *result = parsed;
    return true;
}

bool parse_options(int argc, char** argv, Options* options, bool* help_requested) {
    if (options == nullptr || help_requested == nullptr) {
        return false;
    }
    *help_requested = false;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument = argv[index];
        if (argument == "--help") {
            print_usage();
            *help_requested = true;
            return false;
        }
        if (index + 1 >= argc) {
            std::cerr << "Missing value for " << argument << '\n';
            return false;
        }

        std::uint64_t value = 0;
        if (!parse_unsigned(argv[++index], &value)) {
            std::cerr << "Invalid numeric value for " << argument << '\n';
            return false;
        }
        if (argument == "--quota-bytes") {
            options->quota_bytes = value;
        } else if (argument == "--task-memory-bytes") {
            options->task_memory_bytes = value;
        } else if (argument == "--tasks" && value <= std::numeric_limits<std::uint32_t>::max()) {
            options->task_count = static_cast<std::uint32_t>(value);
        } else {
            std::cerr << "Unknown or invalid option: " << argument << '\n';
            return false;
        }
    }
    return options->quota_bytes != 0 && options->task_memory_bytes != 0 && options->task_count != 0;
}

bool check(CUresult result, std::string_view operation) {
    if (result == CUDA_SUCCESS) {
        return true;
    }
    std::cerr << operation << " failed with CUresult " << static_cast<int>(result) << '\n';
    return false;
}

bool run_demo(const Options& options) {
    if (!check(cuInit(0), "cuInit")) {
        return false;
    }

    CUdevice device{};
    if (!check(cuDeviceGet(&device, 0), "cuDeviceGet")) {
        return false;
    }

    ContextGuard context;
    if (!check(cuCtxCreate(&context.handle, nullptr, 0, device), "cuCtxCreate")) {
        return false;
    }

    ModuleGuard module;
    if (!check(cuModuleLoadData(&module.handle, kNoopPtx), "cuModuleLoadData")) {
        return false;
    }

    CUfunction function = nullptr;
    if (!check(cuModuleGetFunction(&function, module.handle, "glimmer_noop"),
               "cuModuleGetFunction")) {
        return false;
    }

    const glimmer::backend::cuda::CudaFunctionTable functions{.launch_kernel = &cuLaunchKernel,
                                                              .event_create = &cuEventCreate,
                                                              .event_record = &cuEventRecord,
                                                              .event_query = &cuEventQuery,
                                                              .event_destroy = &cuEventDestroy_v2};
    glimmer::backend::cuda::CudaTaskBackend backend(functions);
    glimmer::core::Scheduler scheduler(options.quota_bytes);
    glimmer::backend::cuda::CudaTaskController controller(scheduler, backend);

    const glimmer::backend::cuda::CudaKernelLaunch launch{.function = function,
                                                          .grid_dim_x = 1,
                                                          .grid_dim_y = 1,
                                                          .grid_dim_z = 1,
                                                          .block_dim_x = 1,
                                                          .block_dim_y = 1,
                                                          .block_dim_z = 1};
    for (std::uint32_t index = 0; index < options.task_count; ++index) {
        const auto admission =
            controller.submit(glimmer::core::TaskSpec{.tenant_id = "explicit-demo",
                                                      .memory_bytes = options.task_memory_bytes,
                                                      .work_units = 1},
                              launch);
        if (!admission.accepted()) {
            std::cerr << "task admission failed with status " << static_cast<int>(admission.status)
                      << '\n';
            return false;
        }
    }

    std::size_t completed_tasks = 0;
    for (std::size_t polls = 0; completed_tasks < options.task_count && polls < kMaximumPolls;
         ++polls) {
        const auto step = controller.step();
        switch (step.status) {
            case glimmer::backend::ExecutorStepStatus::kSubmitted:
            case glimmer::backend::ExecutorStepStatus::kPending:
                if (step.status == glimmer::backend::ExecutorStepStatus::kPending) {
                    std::this_thread::yield();
                }
                break;
            case glimmer::backend::ExecutorStepStatus::kCompleted:
                ++completed_tasks;
                break;
            case glimmer::backend::ExecutorStepStatus::kIdle:
            case glimmer::backend::ExecutorStepStatus::kFailed:
            case glimmer::backend::ExecutorStepStatus::kCancelled:
            case glimmer::backend::ExecutorStepStatus::kBackendError:
                std::cerr << "explicit task execution stopped with status "
                          << static_cast<int>(step.status) << '\n';
                return false;
        }
    }

    const bool completed = completed_tasks == options.task_count;
    const bool quota_released = scheduler.usage().used_bytes() == 0;
    if (!completed || !quota_released) {
        std::cerr << "explicit task demo did not reach a clean terminal state\n";
        return false;
    }

    std::cout << "explicit_cuda_task_demo status=ok tasks=" << options.task_count
              << " completed=" << completed_tasks << " quota_bytes=" << options.quota_bytes << '\n';
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    bool help_requested = false;
    if (!parse_options(argc, argv, &options, &help_requested)) {
        if (help_requested) {
            return EXIT_SUCCESS;
        }
        return EXIT_FAILURE;
    }
    return run_demo(options) ? EXIT_SUCCESS : EXIT_FAILURE;
}
