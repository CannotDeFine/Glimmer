#pragma once

#include "glimmer/backend/task_backend.h"

#include <cuda.h>

#include <cstdint>
#include <optional>
#include <unordered_map>

namespace glimmer::backend::cuda {

using LaunchKernelFunction = decltype(&::cuLaunchKernel);
using EventCreateFunction = decltype(&::cuEventCreate);
using EventRecordFunction = decltype(&::cuEventRecord);
using EventQueryFunction = decltype(&::cuEventQuery);
using EventDestroyFunction = decltype(&::cuEventDestroy_v2);

struct CudaFunctionTable {
    LaunchKernelFunction launch_kernel = nullptr;
    EventCreateFunction event_create = nullptr;
    EventRecordFunction event_record = nullptr;
    EventQueryFunction event_query = nullptr;
    EventDestroyFunction event_destroy = nullptr;
};

struct CudaKernelLaunch {
    CUfunction function = nullptr;
    std::uint32_t grid_dim_x = 0;
    std::uint32_t grid_dim_y = 0;
    std::uint32_t grid_dim_z = 0;
    std::uint32_t block_dim_x = 0;
    std::uint32_t block_dim_y = 0;
    std::uint32_t block_dim_z = 0;
    std::uint32_t shared_memory_bytes = 0;
    CUstream stream = nullptr;
    void** kernel_parameters = nullptr;
    void** extra = nullptr;
};

class CudaTaskBackend final : public backend::TaskBackend {
   public:
    // Single-threaded. poll() reaps completed events and may update health.
    explicit CudaTaskBackend(CudaFunctionTable functions) noexcept;
    ~CudaTaskBackend() override;

    CudaTaskBackend(const CudaTaskBackend&) = delete;
    CudaTaskBackend& operator=(const CudaTaskBackend&) = delete;
    CudaTaskBackend(CudaTaskBackend&&) = delete;
    CudaTaskBackend& operator=(CudaTaskBackend&&) = delete;

    // Register the trusted launch descriptor before the scheduler dispatches
    // the task. Descriptor fields are copied, but the caller must keep CUDA
    // argument storage and referenced resources alive until terminal status.
    [[nodiscard]] bool register_launch(core::TaskId task_id, CudaKernelLaunch launch);

    [[nodiscard]] bool submit(const backend::TaskSubmission& submission) override;
    [[nodiscard]] backend::TaskPollResult poll(core::TaskId task_id) const override;
    [[nodiscard]] bool cancel(core::TaskId task_id) override;

    [[nodiscard]] bool is_healthy() const noexcept;

   private:
    struct TaskRecord {
        CudaKernelLaunch launch;
        CUevent completion_event = nullptr;
    };

    [[nodiscard]] static bool valid_launch(const CudaKernelLaunch& launch) noexcept;
    [[nodiscard]] bool destroy_event(CUevent event) const noexcept;

    CudaFunctionTable functions_;
    mutable std::unordered_map<core::TaskId, TaskRecord> tasks_;
    mutable bool is_healthy_ = true;
};

class CudaTaskController final {
   public:
    CudaTaskController(core::Scheduler& scheduler, CudaTaskBackend& backend) noexcept;

    [[nodiscard]] core::SubmitResult submit(core::TaskSpec spec, CudaKernelLaunch launch);
    [[nodiscard]] backend::ExecutorStepResult step();
    [[nodiscard]] std::optional<core::TaskId> active_task() const noexcept;

   private:
    core::Scheduler& scheduler_;
    CudaTaskBackend& backend_;
    backend::TaskExecutor executor_;
};

}  // namespace glimmer::backend::cuda
