#include "glimmer/core/scheduler.h"

#include <algorithm>
#include <limits>
#include <utility>

namespace glimmer::core {

bool SubmitResult::accepted() const noexcept {
    return status == SubmitStatus::kAccepted;
}

Scheduler::Scheduler(MemoryBytes memory_limit_bytes) : quota_ledger_(memory_limit_bytes) {}

SubmitResult Scheduler::submit(TaskSpec spec) {
    if (spec.tenant_id.empty() || spec.memory_bytes == 0 || spec.weight == 0) {
        return {.status = SubmitStatus::kInvalidTask};
    }

    std::scoped_lock lock(mutex_);
    if (next_task_id_ == std::numeric_limits<TaskId>::max()) {
        return {.status = SubmitStatus::kInternalError};
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
    if (running_task_id_.has_value()) {
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
        return std::nullopt;
    }

    try {
        if (!reservation.value().commit()) {
            task_iterator->second.state = TaskState::kFailed;
            return std::nullopt;
        }
    } catch (...) {
        task_iterator->second.state = TaskState::kFailed;
        return std::nullopt;
    }

    task_iterator->second.state = TaskState::kRunning;
    running_task_id_ = selected_task_id;
    return snapshot_locked(selected_task_id, task_iterator->second);
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
    if (task.state != TaskState::kRunning) {
        return false;
    }

    return finish_running_task_locked(task_id, TaskState::kCancelled);
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

std::size_t Scheduler::queued_task_count() const {
    std::scoped_lock lock(mutex_);
    return queued_task_count_;
}

std::size_t Scheduler::running_task_count() const {
    std::scoped_lock lock(mutex_);
    return running_task_id_.has_value() ? 1 : 0;
}

std::optional<TaskId> Scheduler::select_next_task_locked() {
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
                        .state = task.state};
}

bool Scheduler::finish_running_task_locked(TaskId task_id, TaskState terminal_state) {
    const auto task_iterator = tasks_.find(task_id);
    if (task_iterator == tasks_.end() || task_iterator->second.state != TaskState::kRunning ||
        !running_task_id_.has_value() || *running_task_id_ != task_id) {
        return false;
    }
    if (!quota_ledger_.release(task_iterator->second.spec.memory_bytes)) {
        return false;
    }

    task_iterator->second.state = terminal_state;
    running_task_id_.reset();
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
