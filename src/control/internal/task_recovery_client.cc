#include "task_recovery_client.h"
#include "shared_control_channel.h"

#include <limits>
#include <utility>

namespace glimmer::control::internal {
namespace {

[[nodiscard]] bool response_matches_operation(const TaskRecoveryRequest& request,
                                              const TaskRecoveryResponse& response) noexcept {
    if (response.kind == TaskRecoveryResultKind::kRejected) return true;
    switch (request.operation) {
        case TaskRecoveryOperation::kSubmit:
            return response.kind == TaskRecoveryResultKind::kAccepted;
        case TaskRecoveryOperation::kClaim:
            return (response.kind == TaskRecoveryResultKind::kPending ||
                    response.kind == TaskRecoveryResultKind::kGranted) &&
                   response.execution.task_id == request.execution.task_id;
        case TaskRecoveryOperation::kRetreat:
            return response.kind == TaskRecoveryResultKind::kRetreat &&
                   response.execution.task_id == request.execution.task_id &&
                   response.execution.generation == request.execution.generation;
        case TaskRecoveryOperation::kStatus:
            return response.kind == TaskRecoveryResultKind::kStatus;
        case TaskRecoveryOperation::kHeartbeat:
        case TaskRecoveryOperation::kSuspend:
        case TaskRecoveryOperation::kComplete:
        case TaskRecoveryOperation::kFail:
        case TaskRecoveryOperation::kCancelPending:
        case TaskRecoveryOperation::kForget:
            return response.kind == TaskRecoveryResultKind::kAcknowledged;
    }
    return false;
}

}  // namespace

TaskRecoveryClient::TaskRecoveryClient(std::string socket_path, std::uint32_t timeout_ms)
    : transport_(std::move(socket_path), timeout_ms), nonce(new_service_incarnation()) {}

std::optional<TaskRecoveryResponse> TaskRecoveryClient::exchange(
    const TaskRecoveryLine& line) noexcept {
    const auto reply = transport_.request(line.view());
    if (!reply) return std::nullopt;
    const auto parsed = parse_task_recovery_response(*reply);
    if (!parsed) state_ = State::kFailed;
    return parsed;
}

bool TaskRecoveryClient::open() noexcept {
    if (state_ == State::kReady) return true;
    if (state_ != State::kNew) return false;
    TaskRecoveryRequest request;
    request.nonce = nonce;
    const auto line = format_task_recovery_request(request);
    if (!line) {
        state_ = State::kFailed;
        return false;
    }
    const auto response = exchange(*line);
    if (!response) return false;
    if (response->kind != TaskRecoveryResultKind::kOpened || response->nonce != nonce) {
        state_ = State::kFailed;
        return false;
    }
    incarnation_ = response->incarnation;
    session_id_ = response->session_id;
    state_ = State::kReady;
    return true;
}

std::optional<TaskRecoveryResponse> TaskRecoveryClient::execute(
    TaskRecoveryRequest request) noexcept {
    if ((state_ != State::kReady && state_ != State::kWaiting) || next_operation_ == 0)
        return std::nullopt;
    request.command = TaskRecoveryCommand::kExecute;
    request.nonce = nonce;
    request.incarnation = incarnation_;
    request.session_id = session_id_;
    request.operation_id = next_operation_;
    request.acknowledged_through = acknowledged_through_;
    const auto line = format_task_recovery_request(request);
    if (!line) return std::nullopt;
    if (state_ == State::kWaiting && pending_.view() != line->view()) return std::nullopt;
    pending_ = *line;
    state_ = State::kWaiting;
    const auto response = exchange(pending_);
    if (!response) return std::nullopt;
    if (response->nonce != nonce || response->incarnation != incarnation_ ||
        response->session_id != session_id_ || response->operation_id != next_operation_ ||
        !response_matches_operation(request, *response)) {
        state_ = State::kFailed;
        return std::nullopt;
    }
    state_ = State::kDelivered;
    return response;
}

bool TaskRecoveryClient::acknowledge() noexcept {
    if (state_ != State::kDelivered) return false;
    TaskRecoveryRequest request;
    request.command = TaskRecoveryCommand::kAcknowledge;
    request.nonce = nonce;
    request.incarnation = incarnation_;
    request.session_id = session_id_;
    request.acknowledged_through = next_operation_;
    const auto line = format_task_recovery_request(request);
    if (!line) {
        state_ = State::kFailed;
        return false;
    }
    const auto response = exchange(*line);
    if (!response) return false;
    if (response->kind != TaskRecoveryResultKind::kAcknowledged || response->nonce != nonce ||
        response->incarnation != incarnation_ || response->session_id != session_id_ ||
        response->operation_id != 0 || response->acknowledged_through != next_operation_) {
        state_ = State::kFailed;
        return false;
    }
    acknowledged_through_ = next_operation_;
    next_operation_ =
        next_operation_ == std::numeric_limits<std::uint64_t>::max() ? 0 : next_operation_ + 1;
    pending_ = {};
    state_ = State::kReady;
    return true;
}

bool TaskRecoveryClient::close() noexcept {
    if (state_ == State::kClosed) return true;
    if (state_ != State::kReady && state_ != State::kClosing) return false;
    TaskRecoveryRequest request;
    request.command = TaskRecoveryCommand::kClose;
    request.nonce = nonce;
    request.incarnation = incarnation_;
    request.session_id = session_id_;
    const auto line = format_task_recovery_request(request);
    if (!line) {
        state_ = State::kFailed;
        return false;
    }
    state_ = State::kClosing;
    const auto response = exchange(*line);
    if (!response) return false;
    if (response->nonce != nonce || response->incarnation != incarnation_ ||
        response->session_id != session_id_ || response->operation_id != 0) {
        state_ = State::kFailed;
        return false;
    }
    if (response->kind == TaskRecoveryResultKind::kRejected) {
        state_ = State::kReady;
        return false;
    }
    if (response->kind != TaskRecoveryResultKind::kClosed) {
        state_ = State::kFailed;
        return false;
    }
    state_ = State::kClosed;
    return true;
}

}  // namespace glimmer::control::internal
