#include "glimmer/control/native_epoch_protocol.h"
#include "glimmer/control/task_endpoint.h"

namespace glimmer::control {
namespace {

TaskProtocolErrorCode epoch_admission_error(core::SubmitStatus status) noexcept {
    switch (status) {
        case core::SubmitStatus::kInvalidTask:
        case core::SubmitStatus::kTenantWeightMismatch:
            return TaskProtocolErrorCode::kInvalidRequest;
        case core::SubmitStatus::kQuotaExceeded:
            return TaskProtocolErrorCode::kQuotaExceeded;
        case core::SubmitStatus::kQueueFull:
            return TaskProtocolErrorCode::kQueueFull;
        case core::SubmitStatus::kAccepted:
        case core::SubmitStatus::kInternalError:
            return TaskProtocolErrorCode::kInternalError;
    }
    return TaskProtocolErrorCode::kInternalError;
}

}  // namespace

std::optional<std::string> TaskControlEndpoint::handle_native_epoch(
    std::string_view line, std::optional<TaskPeerIdentity> peer) {
    NativeEpochResponse response;
    const auto request = parse_native_epoch_request(line);
    if (!request) {
        return format_native_epoch_response(response);
    }
    response.operation = request->operation;
    response.request_sequence = request->request_sequence;
    response.grant.execution = request->execution;
    switch (request->operation) {
        case NativeEpochOperation::kAcquire:
        case NativeEpochOperation::kClaim: {
            auto task_id = request->execution.task_id;
            if (request->operation == NativeEpochOperation::kAcquire) {
                const auto result = admission_service_.submit_native_epoch(
                    request->admission, request->extent, registrar_, registrar_context_, peer);
                if (!result.accepted()) {
                    response.error = epoch_admission_error(result.status);
                    break;
                }
                task_id = result.task_id;
            }
            const auto granted = admission_service_.claim_native_epoch(task_id, peer);
            if (granted) {
                response.kind = NativeEpochResponseKind::kGranted;
                response.grant = *granted;
                break;
            }
            const auto pending = admission_service_.pending_native_epoch_extent(task_id, peer);
            if (pending) {
                response.kind = NativeEpochResponseKind::kPending;
                response.grant = {.execution = {task_id, 0}, .extent = *pending};
            }
            break;
        }
        case NativeEpochOperation::kRetreat: {
            const auto retreat =
                admission_service_.native_epoch_retreat_requested(request->execution, peer);
            if (retreat) {
                response.kind = NativeEpochResponseKind::kRetreat;
                response.should_retreat = *retreat;
            }
            break;
        }
        case NativeEpochOperation::kHeartbeat:
            if (admission_service_.heartbeat_native_epoch(request->execution, peer)) {
                response.kind = NativeEpochResponseKind::kAcknowledged;
            }
            break;
        case NativeEpochOperation::kComplete:
        case NativeEpochOperation::kFail:
            if (admission_service_.finish_native_epoch(
                    request->execution, request->issued_calls,
                    request->operation == NativeEpochOperation::kFail, peer)) {
                response.kind = NativeEpochResponseKind::kAcknowledged;
                response.issued_calls = request->issued_calls;
            }
            break;
        case NativeEpochOperation::kCancelPending:
            if (admission_service_.cancel_pending_native_epoch(request->execution.task_id, peer)) {
                response.kind = NativeEpochResponseKind::kAcknowledged;
            }
            break;
    }
    return format_native_epoch_response(response);
}

}  // namespace glimmer::control
