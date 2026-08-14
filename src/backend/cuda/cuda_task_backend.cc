#include "internal/cuda_task_backend.h"

#include <utility>

namespace glimmer::backend::cuda {

CudaTaskBackend::CudaTaskBackend(CudaFunctionTable functions) noexcept
    : functions_(functions),
      is_healthy_(functions_.launch_kernel != nullptr && functions_.event_create != nullptr &&
                  functions_.event_record != nullptr && functions_.event_query != nullptr &&
                  functions_.event_destroy != nullptr) {}

CudaTaskBackend::~CudaTaskBackend() {
    if (functions_.event_destroy == nullptr) {
        tasks_.clear();
        return;
    }

    for (const auto& [task_id, task] : tasks_) {
        static_cast<void>(task_id);
        if (task.completion_event != nullptr && !destroy_event(task.completion_event)) {
            is_healthy_ = false;
        }
    }
    tasks_.clear();
}

bool CudaTaskBackend::register_launch(core::TaskId task_id, CudaKernelLaunch launch) {
    if (!is_healthy() || task_id == 0 || !valid_launch(launch) || tasks_.contains(task_id)) {
        return false;
    }

    try {
        tasks_.emplace(task_id, TaskRecord{.launch = launch});
        return true;
    } catch (...) {
        return false;
    }
}

bool CudaTaskBackend::submit(const TaskSubmission& submission) {
    if (!is_healthy() || submission.task_id == 0 || submission.work_units == 0 ||
        functions_.launch_kernel == nullptr || functions_.event_create == nullptr ||
        functions_.event_record == nullptr) {
        return false;
    }

    const auto task_iterator = tasks_.find(submission.task_id);
    if (task_iterator == tasks_.end() || task_iterator->second.completion_event != nullptr) {
        return false;
    }

    TaskRecord& task = task_iterator->second;
    CUevent completion_event = nullptr;
    const auto cleanup_failed_submission = [this, task_iterator](CUevent event) {
        if (destroy_event(event)) {
            tasks_.erase(task_iterator);
            return;
        }
        task_iterator->second.completion_event = event;
        is_healthy_ = false;
    };

    const CUresult event_create_result =
        functions_.event_create(&completion_event, CU_EVENT_DISABLE_TIMING);
    if (event_create_result != CUDA_SUCCESS || completion_event == nullptr) {
        if (completion_event != nullptr) {
            cleanup_failed_submission(completion_event);
        } else {
            tasks_.erase(task_iterator);
        }
        return false;
    }
    task.completion_event = completion_event;

    const CudaKernelLaunch& launch = task.launch;
    if (functions_.launch_kernel(launch.function, launch.grid_dim_x, launch.grid_dim_y,
                                 launch.grid_dim_z, launch.block_dim_x, launch.block_dim_y,
                                 launch.block_dim_z, launch.shared_memory_bytes, launch.stream,
                                 launch.kernel_parameters, launch.extra) != CUDA_SUCCESS) {
        cleanup_failed_submission(completion_event);
        return false;
    }

    if (functions_.event_record(completion_event, launch.stream) != CUDA_SUCCESS) {
        cleanup_failed_submission(completion_event);
        return false;
    }
    return true;
}

TaskPollResult CudaTaskBackend::poll(core::TaskId task_id) const {
    const auto task_iterator = tasks_.find(task_id);
    if (task_iterator == tasks_.end() || task_iterator->second.completion_event == nullptr ||
        functions_.event_query == nullptr) {
        return {.status = TaskStatus::kUnknown};
    }

    const CUresult query_result = functions_.event_query(task_iterator->second.completion_event);
    if (query_result == CUDA_ERROR_NOT_READY) {
        return {.status = TaskStatus::kPending};
    }

    const TaskStatus terminal_status =
        query_result == CUDA_SUCCESS ? TaskStatus::kCompleted : TaskStatus::kFailed;
    const CUevent completion_event = task_iterator->second.completion_event;
    if (!destroy_event(completion_event)) {
        is_healthy_ = false;
        return {.status = TaskStatus::kFailed};
    }
    tasks_.erase(task_iterator);
    return {.status = terminal_status};
}

bool CudaTaskBackend::cancel(core::TaskId task_id) {
    static_cast<void>(task_id);
    // CUDA Driver events provide completion observation, not kernel preemption.
    return false;
}

bool CudaTaskBackend::is_healthy() const noexcept {
    return is_healthy_;
}

bool CudaTaskBackend::valid_launch(const CudaKernelLaunch& launch) noexcept {
    return launch.function != nullptr && launch.grid_dim_x != 0 && launch.grid_dim_y != 0 &&
           launch.grid_dim_z != 0 && launch.block_dim_x != 0 && launch.block_dim_y != 0 &&
           launch.block_dim_z != 0;
}

bool CudaTaskBackend::destroy_event(CUevent event) const noexcept {
    return functions_.event_destroy != nullptr && functions_.event_destroy(event) == CUDA_SUCCESS;
}

CudaTaskController::CudaTaskController(core::Scheduler& scheduler,
                                       CudaTaskBackend& backend) noexcept
    : scheduler_(scheduler), backend_(backend), executor_(scheduler, backend) {}

core::SubmitResult CudaTaskController::submit(core::TaskSpec spec, CudaKernelLaunch launch) {
    const core::SubmitResult admission = scheduler_.submit(std::move(spec));
    if (!admission.accepted()) {
        return admission;
    }
    if (backend_.register_launch(admission.task_id, launch)) {
        return admission;
    }

    static_cast<void>(scheduler_.cancel(admission.task_id));
    return {.status = core::SubmitStatus::kInternalError, .task_id = admission.task_id};
}

ExecutorStepResult CudaTaskController::step() {
    return executor_.step();
}

std::optional<core::TaskId> CudaTaskController::active_task() const noexcept {
    return executor_.active_task();
}

}  // namespace glimmer::backend::cuda
