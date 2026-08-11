#pragma once

#include "glimmer/core/quota_ledger.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace glimmer::core {

using TaskId = std::uint64_t;
using TenantId = std::string;

struct TaskSpec {
    TenantId tenant_id;
    MemoryBytes memory_bytes = 0;
    std::uint32_t weight = 1;
};

enum class TaskState : std::uint8_t {
    kQueued,
    kRunning,
    kCompleted,
    kCancelled,
    kFailed,
};

enum class SubmitStatus : std::uint8_t {
    kAccepted,
    kInvalidTask,
    kQuotaExceeded,
    kTenantWeightMismatch,
    kInternalError,
};

struct SubmitResult {
    SubmitStatus status = SubmitStatus::kInternalError;
    TaskId task_id = 0;

    [[nodiscard]] bool accepted() const noexcept;
};

struct TaskSnapshot {
    TaskId task_id = 0;
    TenantId tenant_id;
    MemoryBytes memory_bytes = 0;
    std::uint32_t weight = 0;
    TaskState state = TaskState::kFailed;
};

class Scheduler {
   public:
    // Thread-safe. The first scheduler version runs at most one task at a
    // time and applies weighted round-robin order between tenants.
    explicit Scheduler(MemoryBytes memory_limit_bytes);

    Scheduler(const Scheduler&) = delete;
    Scheduler& operator=(const Scheduler&) = delete;
    Scheduler(Scheduler&&) = delete;
    Scheduler& operator=(Scheduler&&) = delete;

    [[nodiscard]] SubmitResult submit(TaskSpec spec);
    [[nodiscard]] std::optional<TaskSnapshot> dispatch_next();
    [[nodiscard]] bool complete(TaskId task_id);
    [[nodiscard]] bool fail(TaskId task_id);
    [[nodiscard]] bool cancel(TaskId task_id);
    [[nodiscard]] bool forget(TaskId task_id);

    [[nodiscard]] std::optional<TaskSnapshot> find(TaskId task_id) const;
    [[nodiscard]] QuotaUsage usage() const;
    [[nodiscard]] std::size_t queued_task_count() const;
    [[nodiscard]] std::size_t running_task_count() const;

   private:
    struct TaskRecord {
        TaskSpec spec;
        TaskState state = TaskState::kQueued;
        std::optional<QuotaReservation> reservation;
    };

    struct TenantQueue {
        std::uint32_t weight = 1;
        std::deque<TaskId> task_ids;
    };

    [[nodiscard]] std::optional<TaskId> select_next_task_locked();
    [[nodiscard]] TaskSnapshot snapshot_locked(TaskId task_id, const TaskRecord& task) const;
    [[nodiscard]] bool finish_running_task_locked(TaskId task_id, TaskState terminal_state);
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
    std::optional<TaskId> running_task_id_;
    std::size_t queued_task_count_ = 0;
};

}  // namespace glimmer::core
