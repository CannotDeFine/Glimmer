#pragma once

#include "glimmer/core/quota_ledger.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace glimmer::core {

using TaskId = std::uint64_t;
using TenantId = std::string;

struct TaskSpec {
    TenantId tenant_id;
    MemoryBytes memory_bytes = 0;
    std::uint32_t weight = 1;
    std::uint32_t work_units = 1;
    std::uint32_t priority = 0;
};

enum class TaskState : std::uint8_t {
    kQueued,
    kRunning,
    kCompleted,
    kCancelled,
    kFailed,
};

enum class SchedulingPolicy : std::uint8_t {
    kWeightedRoundRobin,
    kDeficitRoundRobin,
    kFifo,
    kPriority,
};

[[nodiscard]] std::optional<SchedulingPolicy> parse_scheduling_policy(
    std::string_view value) noexcept;
[[nodiscard]] std::string_view scheduling_policy_name(SchedulingPolicy policy) noexcept;

enum class SubmitStatus : std::uint8_t {
    kAccepted,
    kInvalidTask,
    kQuotaExceeded,
    kQueueFull,
    kTenantWeightMismatch,
    kInternalError,
};

struct SubmitResult {
    SubmitStatus status = SubmitStatus::kInternalError;
    TaskId task_id = 0;

    [[nodiscard]] bool accepted() const noexcept;
};

struct SchedulerOptions {
    std::size_t max_running_tasks = 1;
    // Zero means that queued-task count is not bounded.
    std::size_t max_queued_tasks = 0;
    SchedulingPolicy scheduling_policy = SchedulingPolicy::kWeightedRoundRobin;
};

struct TaskSnapshot {
    TaskId task_id = 0;
    TenantId tenant_id;
    MemoryBytes memory_bytes = 0;
    std::uint32_t weight = 0;
    std::uint32_t work_units = 0;
    std::uint32_t priority = 0;
    TaskState state = TaskState::kFailed;
};

struct SchedulerStats {
    QuotaUsage quota;
    std::size_t total_task_count = 0;
    std::size_t queued_task_count = 0;
    std::size_t running_task_count = 0;
    std::size_t completed_task_count = 0;
    std::size_t cancelled_task_count = 0;
    std::size_t failed_task_count = 0;
    std::size_t max_running_tasks = 0;
    std::size_t max_queued_tasks = 0;
    SchedulingPolicy scheduling_policy = SchedulingPolicy::kWeightedRoundRobin;
    std::uint64_t total_queue_wait_microseconds = 0;
    std::uint64_t max_queue_wait_microseconds = 0;
    std::uint64_t total_service_time_microseconds = 0;
    std::uint64_t max_service_time_microseconds = 0;
};

class Scheduler {
   public:
    // Thread-safe. The scheduler admits up to max_running_tasks at a time and
    // applies the configured task-selection policy. A zero capacity is treated
    // as one to keep construction fail-safe.
    explicit Scheduler(MemoryBytes memory_limit_bytes, SchedulerOptions options = {});

    Scheduler(const Scheduler&) = delete;
    Scheduler& operator=(const Scheduler&) = delete;
    Scheduler(Scheduler&&) = delete;
    Scheduler& operator=(Scheduler&&) = delete;

    [[nodiscard]] SubmitResult submit(TaskSpec spec);
    [[nodiscard]] std::optional<TaskSnapshot> dispatch_next();
    // Dispatches task_id only when it is next under the configured policy.
    [[nodiscard]] std::optional<TaskSnapshot> dispatch_task(TaskId task_id);
    [[nodiscard]] bool complete(TaskId task_id);
    [[nodiscard]] bool fail(TaskId task_id);
    [[nodiscard]] bool cancel(TaskId task_id);
    [[nodiscard]] bool cancel_queued(TaskId task_id);
    [[nodiscard]] bool forget(TaskId task_id);

    [[nodiscard]] std::optional<TaskSnapshot> find(TaskId task_id) const;
    [[nodiscard]] std::optional<TaskState> task_state(TaskId task_id) const;
    [[nodiscard]] QuotaUsage usage() const;
    [[nodiscard]] SchedulerStats stats() const;
    [[nodiscard]] std::size_t queued_task_count() const;
    [[nodiscard]] std::size_t running_task_count() const;

   private:
    struct TaskRecord {
        TaskSpec spec;
        TaskState state = TaskState::kQueued;
        std::optional<QuotaReservation> reservation;
        std::chrono::steady_clock::time_point queued_at{};
        std::chrono::steady_clock::time_point running_at{};
    };

    struct TenantQueue {
        std::uint32_t weight = 1;
        std::deque<TaskId> task_ids;
        std::uint64_t deficit = 0;
    };

    [[nodiscard]] std::optional<TaskId> select_next_task_locked();
    [[nodiscard]] std::optional<TaskId> select_fifo_task_locked();
    [[nodiscard]] std::optional<TaskId> select_weighted_round_robin_task_locked();
    [[nodiscard]] std::optional<TaskId> select_deficit_round_robin_task_locked();
    [[nodiscard]] std::optional<TaskId> select_priority_task_locked();
    [[nodiscard]] std::optional<TaskId> peek_next_task_locked() const;
    [[nodiscard]] std::optional<TaskId> peek_weighted_round_robin_task_locked() const;
    [[nodiscard]] std::optional<TaskId> peek_deficit_round_robin_task_locked() const;
    [[nodiscard]] std::optional<TaskId> peek_priority_task_locked() const;
    [[nodiscard]] std::optional<TaskSnapshot> dispatch_selected_task_locked(TaskId task_id);
    [[nodiscard]] TaskSnapshot snapshot_locked(TaskId task_id, const TaskRecord& task) const;
    [[nodiscard]] bool finish_running_task_locked(TaskId task_id, TaskState terminal_state);
    [[nodiscard]] bool cancel_queued_task_locked(TaskId task_id, TaskRecord& task) noexcept;
    void remove_task_from_tenant_queue_locked(const TenantId& tenant_id, TaskId task_id) noexcept;
    void advance_tenant_locked();

    mutable std::mutex mutex_;
    QuotaLedger quota_ledger_;
    TaskId next_task_id_ = 1;
    std::unordered_map<TaskId, TaskRecord> tasks_;
    std::unordered_map<TenantId, TenantQueue> tenant_queues_;
    std::vector<TenantId> tenant_order_;
    std::size_t tenant_cursor_ = 0;
    std::uint32_t tenant_budget_ = 0;
    const SchedulingPolicy scheduling_policy;
    const std::size_t max_running_tasks;
    const std::size_t max_queued_tasks;
    std::vector<TaskId> running_task_ids_;
    std::size_t queued_task_count_ = 0;
    std::uint64_t total_queue_wait_microseconds_ = 0;
    std::uint64_t max_queue_wait_microseconds_ = 0;
    std::uint64_t total_service_time_microseconds_ = 0;
    std::uint64_t max_service_time_microseconds_ = 0;
};

}  // namespace glimmer::core
