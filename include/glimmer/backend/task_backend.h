#pragma once

#include "glimmer/core/scheduler.h"

#include <cstddef>
#include <cstdint>
#include <unordered_map>

namespace glimmer::backend {

enum class TaskStatus : std::uint8_t {
    kUnknown,
    kPending,
    kCompleted,
    kFailed,
    kCancelled,
};

struct TaskSubmission {
    core::TaskId task_id = 0;
    std::uint32_t work_units = 0;
};

struct TaskPollResult {
    TaskStatus status = TaskStatus::kUnknown;
};

class TaskBackend {
   public:
    virtual ~TaskBackend() = default;

    // Implementations are called from the scheduler's execution context. A
    // backend that supports calls from multiple threads must provide its own
    // synchronization.
    [[nodiscard]] virtual bool submit(const TaskSubmission& submission) = 0;
    [[nodiscard]] virtual TaskPollResult poll(core::TaskId task_id) const = 0;
    [[nodiscard]] virtual bool cancel(core::TaskId task_id) = 0;
};

class SimulatedBackend final : public TaskBackend {
   public:
    // Each pending task completes after this many explicit advance() calls.
    explicit SimulatedBackend(std::uint32_t completion_steps = 1) noexcept;

    [[nodiscard]] bool submit(const TaskSubmission& submission) override;
    [[nodiscard]] TaskPollResult poll(core::TaskId task_id) const override;
    [[nodiscard]] bool cancel(core::TaskId task_id) override;

    void advance() noexcept;

    [[nodiscard]] std::size_t active_task_count() const noexcept;

   private:
    struct TaskRecord {
        TaskStatus status = TaskStatus::kPending;
        std::uint32_t remaining_steps = 0;
    };

    std::uint32_t completion_steps_;
    std::unordered_map<core::TaskId, TaskRecord> tasks_;
};

}  // namespace glimmer::backend
