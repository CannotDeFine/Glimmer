#include "glimmer/core/scheduler.h"

#include <algorithm>
#include <limits>
#include <utility>

namespace glimmer::core {

namespace {

[[nodiscard]] bool is_supported_scheduling_policy(SchedulingPolicy policy) noexcept {
    switch (policy) {
        case SchedulingPolicy::kWeightedRoundRobin:
        case SchedulingPolicy::kFifo:
            return true;
    }
    return false;
}

}  // namespace

bool SubmitResult::accepted() const noexcept {
    return status == SubmitStatus::kAccepted;
}

std::optional<SchedulingPolicy> parse_scheduling_policy(std::string_view value) noexcept {
    if (value == "weighted_rr") {
        return SchedulingPolicy::kWeightedRoundRobin;
    }
    if (value == "fifo") {
        return SchedulingPolicy::kFifo;
    }
    return std::nullopt;
}

std::string_view scheduling_policy_name(SchedulingPolicy policy) noexcept {
    switch (policy) {
        case SchedulingPolicy::kWeightedRoundRobin:
            return "weighted_rr";
        case SchedulingPolicy::kFifo:
            return "fifo";
    }
    return "unknown";
}

Scheduler::Scheduler(MemoryBytes memory_limit_bytes, SchedulerOptions options)
    : quota_ledger_(memory_limit_bytes),
      scheduling_policy(is_supported_scheduling_policy(options.scheduling_policy)
                            ? options.scheduling_policy
                            : SchedulingPolicy::kWeightedRoundRobin),
      max_running_tasks(std::max<std::size_t>(options.max_running_tasks, 1)),
      max_queued_tasks(options.max_queued_tasks) {
    running_task_ids_.reserve(max_running_tasks);
}

SubmitResult Scheduler::submit(TaskSpec spec) {
    if (spec.tenant_id.empty() || spec.memory_bytes == 0 || spec.weight == 0 ||
        spec.work_units == 0) {
        return {.status = SubmitStatus::kInvalidTask};
    }

    std::scoped_lock lock(mutex_);
    if (next_task_id_ == std::numeric_limits<TaskId>::max()) {
        return {.status = SubmitStatus::kInternalError};
    }
    if (max_queued_tasks != 0 && queued_task_count_ >= max_queued_tasks) {
        return {.status = SubmitStatus::kQueueFull};
    }

    auto tenant_iterator = tenant_queues_.find(spec.tenant_id);
    if (tenant_iterator != tenant_queues_.end() && tenant_iterator->second.weight != spec.weight) {
        return {.status = SubmitStatus::kTenantWeightMismatch};
    }

    std::optional<QuotaReservation> reservation;
    try {
        reservation = quota_ledger_.try_reserve(spec.memory_bytes);
    } catch (...) {
        return {.status = SubmitStatus::kInternalError};
    }
    if (!reservation.has_value()) {
        return {.status = SubmitStatus::kQuotaExceeded};
    }

    bool inserted_tenant = false;
    if (tenant_iterator == tenant_queues_.end()) {
        try {
            auto [inserted_iterator, inserted] = tenant_queues_.emplace(
                spec.tenant_id, TenantQueue{.weight = spec.weight, .task_ids = {}});
            if (!inserted) {
                return {.status = SubmitStatus::kInternalError};
            }
            tenant_order_.push_back(spec.tenant_id);
            tenant_iterator = inserted_iterator;
            inserted_tenant = true;
        } catch (...) {
            tenant_queues_.erase(spec.tenant_id);
            return {.status = SubmitStatus::kInternalError};
        }
    }

    const TaskId task_id = next_task_id_;
    try {
        const auto [task_iterator, inserted] =
            tasks_.emplace(task_id, TaskRecord{.spec = std::move(spec),
                                               .state = TaskState::kQueued,
                                               .reservation = std::move(reservation)});
        if (!inserted) {
            if (inserted_tenant) {
                tenant_queues_.erase(tenant_iterator);
                tenant_order_.pop_back();
            }
            return {.status = SubmitStatus::kInternalError};
        }
        tenant_iterator->second.task_ids.push_back(task_id);
        ++next_task_id_;
        ++queued_task_count_;
        static_cast<void>(task_iterator);
    } catch (...) {
        tasks_.erase(task_id);
        if (inserted_tenant) {
            tenant_queues_.erase(tenant_iterator);
            tenant_order_.pop_back();
        }
        return {.status = SubmitStatus::kInternalError};
    }

    return {.status = SubmitStatus::kAccepted, .task_id = task_id};
}

std::optional<TaskSnapshot> Scheduler::dispatch_next() {
    std::scoped_lock lock(mutex_);
    if (running_task_ids_.size() >= max_running_tasks) {
        return std::nullopt;
    }

    const auto task_id = select_next_task_locked();
    if (!task_id.has_value()) {
        return std::nullopt;
    }

    const TaskId selected_task_id = task_id.value();
    const auto task_iterator = tasks_.find(selected_task_id);
    if (task_iterator == tasks_.end() || task_iterator->second.state != TaskState::kQueued) {
        return std::nullopt;
    }
    std::optional<QuotaReservation>& reservation = task_iterator->second.reservation;
    if (!reservation.has_value()) {
        task_iterator->second.state = TaskState::kFailed;
        return std::nullopt;
    }

    TaskSnapshot dispatched_snapshot;
    try {
        // Build the return value before committing the reservation. A task
        // snapshot owns tenant metadata and can therefore allocate; keeping
        // that work before the state transition prevents an exception from
        // leaving a running task without a lease owner.
        dispatched_snapshot = snapshot_locked(selected_task_id, task_iterator->second);
        dispatched_snapshot.state = TaskState::kRunning;
    } catch (...) {
        reservation->cancel();
        task_iterator->second.state = TaskState::kFailed;
        return std::nullopt;
    }

    try {
        running_task_ids_.push_back(selected_task_id);
    } catch (...) {
        reservation->cancel();
        task_iterator->second.state = TaskState::kFailed;
        return std::nullopt;
    }

    bool committed = false;
    try {
        committed = reservation.value().commit();
    } catch (...) {
        committed = false;
    }
    if (!committed) {
        reservation->cancel();
        running_task_ids_.pop_back();
        task_iterator->second.state = TaskState::kFailed;
        return std::nullopt;
    }
    task_iterator->second.state = TaskState::kRunning;
    return dispatched_snapshot;
}

bool Scheduler::complete(TaskId task_id) {
    std::scoped_lock lock(mutex_);
    return finish_running_task_locked(task_id, TaskState::kCompleted);
}

bool Scheduler::fail(TaskId task_id) {
    std::scoped_lock lock(mutex_);
    return finish_running_task_locked(task_id, TaskState::kFailed);
}

bool Scheduler::cancel(TaskId task_id) {
    std::scoped_lock lock(mutex_);
    const auto task_iterator = tasks_.find(task_id);
    if (task_iterator == tasks_.end()) {
        return false;
    }

    TaskRecord& task = task_iterator->second;
    if (task.state == TaskState::kQueued) {
        return cancel_queued_task_locked(task_id, task);
    }
    if (task.state != TaskState::kRunning) {
        return false;
    }

    return finish_running_task_locked(task_id, TaskState::kCancelled);
}

bool Scheduler::cancel_queued(TaskId task_id) {
    std::scoped_lock lock(mutex_);
    const auto task_iterator = tasks_.find(task_id);
    if (task_iterator == tasks_.end() || task_iterator->second.state != TaskState::kQueued) {
        return false;
    }
    return cancel_queued_task_locked(task_id, task_iterator->second);
}

bool Scheduler::forget(TaskId task_id) {
    std::scoped_lock lock(mutex_);
    const auto task_iterator = tasks_.find(task_id);
    if (task_iterator == tasks_.end()) {
        return false;
    }
    const TaskState state = task_iterator->second.state;
    if (state == TaskState::kQueued || state == TaskState::kRunning) {
        return false;
    }
    tasks_.erase(task_iterator);
    return true;
}

std::optional<TaskSnapshot> Scheduler::find(TaskId task_id) const {
    std::scoped_lock lock(mutex_);
    const auto task_iterator = tasks_.find(task_id);
    if (task_iterator == tasks_.end()) {
        return std::nullopt;
    }
    return snapshot_locked(task_id, task_iterator->second);
}

QuotaUsage Scheduler::usage() const {
    return quota_ledger_.usage();
}

SchedulerStats Scheduler::stats() const {
    std::scoped_lock lock(mutex_);
    SchedulerStats result{.quota = quota_ledger_.usage(),
                          .total_task_count = tasks_.size(),
                          .queued_task_count = queued_task_count_,
                          .running_task_count = running_task_ids_.size(),
                          .max_running_tasks = max_running_tasks,
                          .max_queued_tasks = max_queued_tasks,
                          .scheduling_policy = scheduling_policy};
    for (const auto& [task_id, task] : tasks_) {
        static_cast<void>(task_id);
        switch (task.state) {
            case TaskState::kQueued:
            case TaskState::kRunning:
                break;
            case TaskState::kCompleted:
                ++result.completed_task_count;
                break;
            case TaskState::kCancelled:
                ++result.cancelled_task_count;
                break;
            case TaskState::kFailed:
                ++result.failed_task_count;
                break;
        }
    }
    return result;
}

std::size_t Scheduler::queued_task_count() const {
    std::scoped_lock lock(mutex_);
    return queued_task_count_;
}

std::size_t Scheduler::running_task_count() const {
    std::scoped_lock lock(mutex_);
    return running_task_ids_.size();
}

std::optional<TaskId> Scheduler::select_next_task_locked() {
    switch (scheduling_policy) {
        case SchedulingPolicy::kFifo:
            return select_fifo_task_locked();
        case SchedulingPolicy::kWeightedRoundRobin:
            return select_weighted_round_robin_task_locked();
    }
    return std::nullopt;
}

std::optional<TaskId> Scheduler::select_fifo_task_locked() {
    std::optional<TaskId> selected_task_id;
    for (const auto& [task_id, task] : tasks_) {
        if (task.state != TaskState::kQueued ||
            (selected_task_id.has_value() && task_id >= selected_task_id.value())) {
            continue;
        }
        selected_task_id = task_id;
    }
    if (!selected_task_id.has_value()) {
        return std::nullopt;
    }

    const auto task_iterator = tasks_.find(selected_task_id.value());
    if (task_iterator == tasks_.end() || task_iterator->second.state != TaskState::kQueued) {
        return std::nullopt;
    }
    remove_task_from_tenant_queue_locked(task_iterator->second.spec.tenant_id,
                                         selected_task_id.value());
    --queued_task_count_;
    return selected_task_id;
}

std::optional<TaskId> Scheduler::select_weighted_round_robin_task_locked() {
    if (tenant_order_.empty()) {
        return std::nullopt;
    }

    const std::size_t tenant_count = tenant_order_.size();
    for (std::size_t attempt = 0; attempt < tenant_count; ++attempt) {
        if (tenant_cursor_ >= tenant_count) {
            tenant_cursor_ = 0;
            tenant_budget_ = 0;
        }

        const auto tenant_iterator = tenant_queues_.find(tenant_order_[tenant_cursor_]);
        if (tenant_iterator == tenant_queues_.end()) {
            advance_tenant_locked();
            continue;
        }

        TenantQueue& queue = tenant_iterator->second;
        while (!queue.task_ids.empty()) {
            const TaskId task_id = queue.task_ids.front();
            queue.task_ids.pop_front();
            const auto task_iterator = tasks_.find(task_id);
            if (task_iterator == tasks_.end() ||
                task_iterator->second.state != TaskState::kQueued) {
                continue;
            }

            if (tenant_budget_ == 0) {
                tenant_budget_ = queue.weight;
            }
            --tenant_budget_;
            if (tenant_budget_ == 0 || queue.task_ids.empty()) {
                advance_tenant_locked();
            }
            --queued_task_count_;
            return task_id;
        }
        advance_tenant_locked();
    }
    return std::nullopt;
}

TaskSnapshot Scheduler::snapshot_locked(TaskId task_id, const TaskRecord& task) const {
    return TaskSnapshot{.task_id = task_id,
                        .tenant_id = task.spec.tenant_id,
                        .memory_bytes = task.spec.memory_bytes,
                        .weight = task.spec.weight,
                        .work_units = task.spec.work_units,
                        .state = task.state};
}

bool Scheduler::finish_running_task_locked(TaskId task_id, TaskState terminal_state) {
    const auto task_iterator = tasks_.find(task_id);
    const auto running_iterator =
        std::find(running_task_ids_.begin(), running_task_ids_.end(), task_id);
    if (task_iterator == tasks_.end() || task_iterator->second.state != TaskState::kRunning ||
        running_iterator == running_task_ids_.end()) {
        return false;
    }
    if (!quota_ledger_.release(task_iterator->second.spec.memory_bytes)) {
        return false;
    }

    task_iterator->second.state = terminal_state;
    running_task_ids_.erase(running_iterator);
    return true;
}

bool Scheduler::cancel_queued_task_locked(TaskId task_id, TaskRecord& task) noexcept {
    if (!task.reservation.has_value() || !task.reservation->is_active()) {
        return false;
    }
    task.reservation->cancel();
    if (task.reservation->is_active()) {
        return false;
    }
    task.state = TaskState::kCancelled;
    --queued_task_count_;
    remove_task_from_tenant_queue_locked(task.spec.tenant_id, task_id);
    return true;
}

void Scheduler::remove_task_from_tenant_queue_locked(const TenantId& tenant_id,
                                                     TaskId task_id) noexcept {
    const auto tenant_iterator = tenant_queues_.find(tenant_id);
    if (tenant_iterator == tenant_queues_.end()) {
        return;
    }

    auto& task_ids = tenant_iterator->second.task_ids;
    const auto task_iterator = std::find(task_ids.begin(), task_ids.end(), task_id);
    if (task_iterator != task_ids.end()) {
        task_ids.erase(task_iterator);
    }
}

void Scheduler::advance_tenant_locked() {
    tenant_budget_ = 0;
    if (tenant_order_.empty()) {
        tenant_cursor_ = 0;
        return;
    }
    tenant_cursor_ = (tenant_cursor_ + 1) % tenant_order_.size();
}

}  // namespace glimmer::core
