#include "glimmer/control/launch_gate.h"
#include "glimmer/control/task_protocol.h"

#include <limits>
#include <thread>
#include <utility>

namespace glimmer::control {

LaunchGate::LaunchGate(LaunchGateOptions options)
    : scheduler_(std::numeric_limits<core::MemoryBytes>::max(),
                 core::SchedulerOptions{.max_running_tasks = options.max_concurrent_launches,
                                        .scheduling_policy = options.scheduling_policy}),
      tenant_id_(std::move(options.tenant_id)),
      tenant_weight_(options.tenant_weight == 0 ? 1 : options.tenant_weight),
      task_priority_(options.task_priority),
      remote_mode_requested_(!options.control_socket.empty()),
      remote_acquire_timeout_(options.remote_acquire_timeout),
      remote_poll_interval_(options.remote_poll_interval > std::chrono::milliseconds::zero()
                                ? options.remote_poll_interval
                                : std::chrono::milliseconds{1}) {
    if (tenant_id_.empty()) {
        tenant_id_ = "default";
    }
    if (remote_mode_requested_) {
        try {
            remote_client_ =
                std::make_unique<UnixSocketControlClient>(std::move(options.control_socket));
        } catch (...) {
            remote_client_.reset();
        }
    }
}

std::optional<core::TaskId> LaunchGate::acquire(std::chrono::milliseconds timeout) {
    if (remote_mode_requested_) {
        if (remote_client_ == nullptr) {
            return std::nullopt;
        }
        const auto effective_timeout =
            timeout == std::chrono::milliseconds::zero() ? remote_acquire_timeout_ : timeout;
        const auto deadline = effective_timeout == std::chrono::milliseconds::zero()
                                  ? std::chrono::steady_clock::time_point::max()
                                  : std::chrono::steady_clock::now() + effective_timeout;
        const TaskProtocolRequest submit{.operation = TaskProtocolOperation::kSubmit,
                                         .admission = {.tenant_id = tenant_id_,
                                                       .memory_bytes = 1,
                                                       .weight = tenant_weight_,
                                                       .work_units = 1,
                                                       .priority = task_priority_},
                                         .task_id = 0};
        const auto encoded_submit = format_task_protocol_request(submit);
        if (!encoded_submit.has_value()) {
            return std::nullopt;
        }
        const auto submit_response = remote_client_->request(encoded_submit.value());
        if (!submit_response.has_value()) {
            return std::nullopt;
        }
        const auto parsed_submit = parse_task_protocol_response(submit_response.value());
        if (!parsed_submit.parsed() || !parsed_submit.response.has_value() ||
            parsed_submit.response->kind != TaskProtocolResponseKind::kAccepted) {
            return std::nullopt;
        }
        const core::TaskId task_id = parsed_submit.response->task_id;
        const auto cancel_task = [this, task_id] {
            const TaskProtocolRequest cancel{
                .operation = TaskProtocolOperation::kCancel, .admission = {}, .task_id = task_id};
            const auto encoded_cancel = format_task_protocol_request(cancel);
            if (encoded_cancel.has_value()) {
                static_cast<void>(remote_client_->request(encoded_cancel.value()));
            }
        };
        const TaskProtocolRequest claim{
            .operation = TaskProtocolOperation::kClaim, .admission = {}, .task_id = task_id};
        const auto encoded_claim = format_task_protocol_request(claim);
        if (!encoded_claim.has_value()) {
            cancel_task();
            return std::nullopt;
        }
        while (true) {
            const auto claim_response = remote_client_->request(encoded_claim.value());
            if (!claim_response.has_value()) {
                cancel_task();
                return std::nullopt;
            }
            const auto parsed_claim = parse_task_protocol_response(claim_response.value());
            if (!parsed_claim.parsed() || !parsed_claim.response.has_value()) {
                cancel_task();
                return std::nullopt;
            }
            if (parsed_claim.response->kind == TaskProtocolResponseKind::kLease &&
                parsed_claim.response->task_id == task_id) {
                return task_id;
            }
            if (parsed_claim.response->kind == TaskProtocolResponseKind::kError) {
                cancel_task();
                return std::nullopt;
            }
            if (effective_timeout != std::chrono::milliseconds::zero() &&
                std::chrono::steady_clock::now() >= deadline) {
                cancel_task();
                return std::nullopt;
            }
            std::this_thread::sleep_for(remote_poll_interval_);
        }
    }
    std::unique_lock lock(mutex_);
    const core::SubmitResult submission =
        scheduler_.submit(core::TaskSpec{.tenant_id = tenant_id_,
                                         .memory_bytes = 1,
                                         .weight = tenant_weight_,
                                         .priority = task_priority_});
    if (!submission.accepted()) {
        return std::nullopt;
    }

    const core::TaskId task_id = submission.task_id;
    static_cast<void>(pump_locked());
    const auto admitted = [this, task_id] {
        const auto state = scheduler_.task_state(task_id);
        return !state.has_value() || state.value() != core::TaskState::kQueued;
    };
    if (timeout == std::chrono::milliseconds::zero()) {
        condition_.wait(lock, admitted);
    } else {
        static_cast<void>(condition_.wait_for(lock, timeout, admitted));
    }

    const auto state = scheduler_.task_state(task_id);
    if (state.has_value() && state.value() == core::TaskState::kRunning) {
        return task_id;
    }
    if (state.has_value() && state.value() == core::TaskState::kQueued) {
        static_cast<void>(scheduler_.cancel_queued(task_id));
    }
    static_cast<void>(scheduler_.forget(task_id));
    static_cast<void>(pump_locked());
    condition_.notify_all();
    return std::nullopt;
}

bool LaunchGate::heartbeat(core::TaskId task_id) {
    if (task_id == 0) {
        return false;
    }
    if (remote_mode_requested_) {
        if (remote_client_ == nullptr) {
            return false;
        }
        const TaskProtocolRequest request{
            .operation = TaskProtocolOperation::kHeartbeat, .admission = {}, .task_id = task_id};
        const auto encoded = format_task_protocol_request(request);
        if (!encoded.has_value()) {
            return false;
        }
        const auto response = remote_client_->request(encoded.value());
        if (!response.has_value()) {
            return false;
        }
        const auto parsed = parse_task_protocol_response(response.value());
        return parsed.parsed() && parsed.response.has_value() &&
               parsed.response->kind == TaskProtocolResponseKind::kState &&
               parsed.response->task_id == task_id &&
               parsed.response->state == TaskProtocolState::kRunning;
    }
    std::scoped_lock lock(mutex_);
    const auto state = scheduler_.task_state(task_id);
    return state.has_value() && state.value() == core::TaskState::kRunning;
}

bool LaunchGate::complete(core::TaskId task_id) {
    if (remote_mode_requested_) {
        if (remote_client_ == nullptr) {
            return false;
        }
        const TaskProtocolRequest request{
            .operation = TaskProtocolOperation::kComplete, .admission = {}, .task_id = task_id};
        const auto encoded = format_task_protocol_request(request);
        if (!encoded.has_value()) {
            return false;
        }
        const auto response = remote_client_->request(encoded.value());
        if (!response.has_value()) {
            return false;
        }
        const auto parsed = parse_task_protocol_response(response.value());
        return parsed.parsed() && parsed.response.has_value() &&
               parsed.response->kind == TaskProtocolResponseKind::kState &&
               parsed.response->task_id == task_id &&
               parsed.response->state == TaskProtocolState::kCompleted;
    }
    std::scoped_lock lock(mutex_);
    return finish_locked(task_id, core::TaskState::kCompleted);
}

bool LaunchGate::fail(core::TaskId task_id) {
    if (remote_mode_requested_) {
        if (remote_client_ == nullptr) {
            return false;
        }
        const TaskProtocolRequest request{
            .operation = TaskProtocolOperation::kFail, .admission = {}, .task_id = task_id};
        const auto encoded = format_task_protocol_request(request);
        if (!encoded.has_value()) {
            return false;
        }
        const auto response = remote_client_->request(encoded.value());
        if (!response.has_value()) {
            return false;
        }
        const auto parsed = parse_task_protocol_response(response.value());
        return parsed.parsed() && parsed.response.has_value() &&
               parsed.response->kind == TaskProtocolResponseKind::kState &&
               parsed.response->task_id == task_id &&
               parsed.response->state == TaskProtocolState::kFailed;
    }
    std::scoped_lock lock(mutex_);
    return finish_locked(task_id, core::TaskState::kFailed);
}

core::SchedulerStats LaunchGate::stats() const {
    if (remote_mode_requested_ && remote_client_ != nullptr) {
        const TaskProtocolRequest request{
            .operation = TaskProtocolOperation::kMetrics, .admission = {}, .task_id = 0};
        const auto encoded = format_task_protocol_request(request);
        if (encoded.has_value()) {
            const auto response = remote_client_->request(encoded.value());
            if (response.has_value()) {
                const auto parsed = parse_task_protocol_response(response.value());
                if (parsed.parsed() && parsed.response.has_value() &&
                    (parsed.response->kind == TaskProtocolResponseKind::kStats ||
                     parsed.response->kind == TaskProtocolResponseKind::kMetrics)) {
                    const auto& stats = parsed.response->stats;
                    return core::SchedulerStats{
                        .quota = {.limit_bytes = stats.quota_limit_bytes,
                                  .reserved_bytes = stats.quota_reserved_bytes,
                                  .allocated_bytes = stats.quota_allocated_bytes},
                        .total_task_count = static_cast<std::size_t>(stats.total_task_count),
                        .queued_task_count = static_cast<std::size_t>(stats.queued_task_count),
                        .running_task_count = static_cast<std::size_t>(stats.running_task_count),
                        .completed_task_count =
                            static_cast<std::size_t>(stats.completed_task_count),
                        .cancelled_task_count =
                            static_cast<std::size_t>(stats.cancelled_task_count),
                        .failed_task_count = static_cast<std::size_t>(stats.failed_task_count),
                        .max_running_tasks = static_cast<std::size_t>(stats.max_running_tasks),
                        .max_queued_tasks = static_cast<std::size_t>(stats.max_queued_tasks),
                        .total_queue_wait_microseconds = stats.total_queue_wait_microseconds,
                        .max_queue_wait_microseconds = stats.max_queue_wait_microseconds,
                        .total_service_time_microseconds = stats.total_service_time_microseconds,
                        .max_service_time_microseconds = stats.max_service_time_microseconds};
                }
            }
        }
    }
    std::scoped_lock lock(mutex_);
    return scheduler_.stats();
}

bool LaunchGate::pump_locked() {
    bool dispatched = false;
    while (true) {
        const auto task = scheduler_.dispatch_next();
        if (!task.has_value()) {
            return dispatched;
        }
        dispatched = true;
        condition_.notify_all();
    }
}

bool LaunchGate::finish_locked(core::TaskId task_id, core::TaskState state) {
    bool finished = false;
    if (state == core::TaskState::kCompleted) {
        finished = scheduler_.complete(task_id);
    } else if (state == core::TaskState::kFailed) {
        finished = scheduler_.fail(task_id);
    }
    if (!finished) {
        return false;
    }
    static_cast<void>(scheduler_.forget(task_id));
    static_cast<void>(pump_locked());
    condition_.notify_all();
    return true;
}

}  // namespace glimmer::control
