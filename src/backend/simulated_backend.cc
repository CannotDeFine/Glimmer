#include "glimmer/backend/task_backend.h"

#include <utility>

namespace glimmer::backend {

SimulatedBackend::SimulatedBackend(std::uint32_t completion_steps) noexcept
    : completion_steps_(completion_steps == 0 ? 1 : completion_steps) {}

bool SimulatedBackend::submit(const TaskSubmission& submission) {
    if (submission.task_id == 0 || submission.work_units == 0 ||
        tasks_.contains(submission.task_id)) {
        return false;
    }

    try {
        tasks_.emplace(submission.task_id, TaskRecord{.status = TaskStatus::kPending,
                                                      .remaining_steps = completion_steps_});
        return true;
    } catch (...) {
        return false;
    }
}

TaskPollResult SimulatedBackend::poll(core::TaskId task_id) const {
    const auto task_iterator = tasks_.find(task_id);
    if (task_iterator == tasks_.end()) {
        return {.status = TaskStatus::kUnknown};
    }
    return {.status = task_iterator->second.status};
}

bool SimulatedBackend::cancel(core::TaskId task_id) {
    const auto task_iterator = tasks_.find(task_id);
    if (task_iterator == tasks_.end() || task_iterator->second.status != TaskStatus::kPending) {
        return false;
    }
    task_iterator->second.status = TaskStatus::kCancelled;
    return true;
}

void SimulatedBackend::advance() noexcept {
    for (auto& [task_id, task] : tasks_) {
        static_cast<void>(task_id);
        if (task.status != TaskStatus::kPending) {
            continue;
        }
        --task.remaining_steps;
        if (task.remaining_steps == 0) {
            task.status = TaskStatus::kCompleted;
        }
    }
}

std::size_t SimulatedBackend::active_task_count() const noexcept {
    std::size_t count = 0;
    for (const auto& [task_id, task] : tasks_) {
        static_cast<void>(task_id);
        if (task.status == TaskStatus::kPending) {
            ++count;
        }
    }
    return count;
}

}  // namespace glimmer::backend
