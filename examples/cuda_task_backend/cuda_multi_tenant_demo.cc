#include "internal/cuda_task_backend.h"

#include "glimmer/backend/task_backend.h"
#include "glimmer/control/task_admission.h"
#include "glimmer/control/task_endpoint.h"
#include "glimmer/control/task_protocol.h"

#include <cuda.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

constexpr std::size_t kQuotaBytes = std::size_t{8} * 1024 * 1024;
constexpr std::size_t kTaskMemoryBytes = std::size_t{1} * 1024 * 1024;
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

struct TaskInfo {
    glimmer::core::TaskId task_id = 0;
    std::string label;
};

struct RegistrarContext {
    glimmer::backend::cuda::CudaTaskBackend* backend = nullptr;
    glimmer::backend::cuda::CudaKernelLaunch launch;
};

bool register_cuda_launch(void* context, glimmer::core::TaskId task_id) noexcept {
    if (context == nullptr) {
        return false;
    }
    auto* registrar = static_cast<RegistrarContext*>(context);
    return registrar->backend != nullptr &&
           registrar->backend->register_launch(task_id, registrar->launch);
}

bool check(CUresult result, std::string_view operation) {
    if (result == CUDA_SUCCESS) {
        return true;
    }
    std::cerr << operation << " failed with CUresult " << static_cast<int>(result) << '\n';
    return false;
}

const TaskInfo* find_task(const std::vector<TaskInfo>& tasks, glimmer::core::TaskId task_id) {
    for (const TaskInfo& task : tasks) {
        if (task.task_id == task_id) {
            return &task;
        }
    }
    return nullptr;
}

bool run_demo() {
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
    glimmer::core::Scheduler scheduler(kQuotaBytes);
    glimmer::control::TaskAdmissionService admission_service(scheduler);
    glimmer::backend::TaskExecutor executor(scheduler, backend);

    const glimmer::backend::cuda::CudaKernelLaunch launch{.function = function,
                                                          .grid_dim_x = 1,
                                                          .grid_dim_y = 1,
                                                          .grid_dim_z = 1,
                                                          .block_dim_x = 1,
                                                          .block_dim_y = 1,
                                                          .block_dim_z = 1};
    RegistrarContext registrar_context{.backend = &backend, .launch = launch};
    glimmer::control::TaskControlEndpoint control_endpoint(admission_service, register_cuda_launch,
                                                           &registrar_context);
    struct TaskRequest {
        std::string_view tenant_id;
        std::string_view label;
        std::uint32_t weight;
    };
    constexpr TaskRequest requests[] = {
        {.tenant_id = "tenant-a", .label = "tenant-a1", .weight = 2},
        {.tenant_id = "tenant-a", .label = "tenant-a2", .weight = 2},
        {.tenant_id = "tenant-a", .label = "tenant-a3", .weight = 2},
        {.tenant_id = "tenant-b", .label = "tenant-b1", .weight = 1},
        {.tenant_id = "tenant-b", .label = "tenant-b2", .weight = 1},
    };

    std::vector<TaskInfo> tasks;
    tasks.reserve(std::size(requests));
    for (const TaskRequest& request : requests) {
        const std::string command = "GLIMMER_TASK_V1 SUBMIT " + std::string(request.tenant_id) +
                                    " " + std::to_string(kTaskMemoryBytes) + " " +
                                    std::to_string(request.weight) + " 1\n";
        const auto response_text = control_endpoint.handle(command);
        if (!response_text.has_value()) {
            std::cerr << "task admission produced no control response for " << request.label
                      << '\n';
            return false;
        }
        const auto response = glimmer::control::parse_task_protocol_response(response_text.value());
        if (!response.parsed() || !response.response.has_value() ||
            response.response->kind != glimmer::control::TaskProtocolResponseKind::kAccepted) {
            std::cerr << "task admission failed for " << request.label << '\n';
            return false;
        }
        tasks.push_back(
            TaskInfo{.task_id = response.response->task_id, .label = std::string(request.label)});
    }

    std::vector<std::string> dispatch_order;
    dispatch_order.reserve(tasks.size());
    for (std::size_t polls = 0; dispatch_order.size() < tasks.size() && polls < kMaximumPolls;
         ++polls) {
        const auto step = executor.step();
        switch (step.status) {
            case glimmer::backend::ExecutorStepStatus::kSubmitted:
            case glimmer::backend::ExecutorStepStatus::kPending:
                if (step.status == glimmer::backend::ExecutorStepStatus::kPending) {
                    std::this_thread::yield();
                }
                break;
            case glimmer::backend::ExecutorStepStatus::kCompleted: {
                if (!step.task_id.has_value()) {
                    std::cerr << "completed task did not report an id\n";
                    return false;
                }
                const TaskInfo* task = find_task(tasks, step.task_id.value());
                if (task == nullptr) {
                    std::cerr << "completed unknown task " << step.task_id.value() << '\n';
                    return false;
                }
                dispatch_order.push_back(task->label);
                break;
            }
            case glimmer::backend::ExecutorStepStatus::kIdle:
            case glimmer::backend::ExecutorStepStatus::kFailed:
            case glimmer::backend::ExecutorStepStatus::kCancelled:
            case glimmer::backend::ExecutorStepStatus::kBackendError:
                std::cerr << "multi-tenant execution stopped with status "
                          << static_cast<int>(step.status) << '\n';
                return false;
        }
    }

    constexpr std::string_view expected_order = "tenant-a1,tenant-a2,tenant-b1,tenant-a3,tenant-b2";
    std::string actual_order;
    for (std::size_t index = 0; index < dispatch_order.size(); ++index) {
        if (index != 0) {
            actual_order += ',';
        }
        actual_order += dispatch_order[index];
    }
    const bool order_matches = actual_order == expected_order;
    const bool quota_released = scheduler.usage().used_bytes() == 0;
    if (!order_matches || !quota_released) {
        std::cerr << "unexpected scheduling result order=" << actual_order
                  << " quota_used_bytes=" << scheduler.usage().used_bytes() << '\n';
        return false;
    }

    std::cout << "explicit_cuda_multi_tenant_demo status=ok dispatch_order=" << actual_order
              << " quota_bytes=" << kQuotaBytes << '\n';
    return true;
}

}  // namespace

int main() {
    return run_demo() ? EXIT_SUCCESS : EXIT_FAILURE;
}
