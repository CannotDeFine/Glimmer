#pragma once

#include "glimmer/control/task_endpoint.h"
#include "glimmer/control/task_protocol.h"
#include "glimmer/control/unix_socket_client.h"
#include "glimmer/control/unix_socket_server.h"

#include <array>
#include <cstddef>

namespace {

// Internal linkage ensures each translation unit uses its own header version.
[[nodiscard]] auto control_layout() noexcept {
    using namespace glimmer::control;
    return std::array{
        sizeof(glimmer::core::Scheduler),
        alignof(glimmer::core::Scheduler),
        sizeof(glimmer::core::TaskSnapshot),
        alignof(glimmer::core::TaskSnapshot),
        sizeof(glimmer::core::SchedulerStats),
        alignof(glimmer::core::SchedulerStats),
        sizeof(glimmer::core::SubmitResult),
        alignof(glimmer::core::SubmitResult),
        sizeof(TaskAdmissionService),
        alignof(TaskAdmissionService),
        sizeof(TaskControlEndpoint),
        alignof(TaskControlEndpoint),
        sizeof(TaskProtocolRequest),
        alignof(TaskProtocolRequest),
        sizeof(TaskProtocolRequestParseResult),
        alignof(TaskProtocolRequestParseResult),
        sizeof(TaskProtocolResponse),
        alignof(TaskProtocolResponse),
        sizeof(TaskProtocolResponseParseResult),
        alignof(TaskProtocolResponseParseResult),
        sizeof(ControlRequestTiming),
        alignof(ControlRequestTiming),
        sizeof(UnixSocketControlClient),
        alignof(UnixSocketControlClient),
        sizeof(UnixSocketControlConfig),
        alignof(UnixSocketControlConfig),
        sizeof(UnixSocketControlServer),
        alignof(UnixSocketControlServer),
        offsetof(TaskProtocolRequest, operation),
        offsetof(TaskProtocolRequest, admission),
        offsetof(TaskProtocolRequest, task_id),
        offsetof(TaskProtocolRequest, wait_timeout_ms),
        offsetof(TaskProtocolRequestParseResult, error),
        offsetof(TaskProtocolRequestParseResult, request),
        offsetof(TaskProtocolResponse, kind),
        offsetof(TaskProtocolResponse, task_id),
        offsetof(TaskProtocolResponse, state),
        offsetof(TaskProtocolResponse, error),
        offsetof(TaskProtocolResponse, tenant_id),
        offsetof(TaskProtocolResponse, memory_bytes),
        offsetof(TaskProtocolResponse, weight),
        offsetof(TaskProtocolResponse, work_units),
        offsetof(TaskProtocolResponse, stats),
        offsetof(TaskProtocolResponseParseResult, error),
        offsetof(TaskProtocolResponseParseResult, response),
        offsetof(ControlRequestTiming, elapsed_nanoseconds),
        offsetof(UnixSocketControlConfig, socket_path),
        offsetof(UnixSocketControlConfig, allowed_uid),
        offsetof(UnixSocketControlConfig, io_timeout_ms),
        sizeof(TaskProtocolOperation),
        static_cast<std::size_t>(TaskProtocolOperation::kSubmit),
        static_cast<std::size_t>(TaskProtocolOperation::kAcquire),
        static_cast<std::size_t>(TaskProtocolOperation::kCancel),
        static_cast<std::size_t>(TaskProtocolOperation::kQuery),
        static_cast<std::size_t>(TaskProtocolOperation::kClaim),
        static_cast<std::size_t>(TaskProtocolOperation::kWait),
        static_cast<std::size_t>(TaskProtocolOperation::kHeartbeat),
        static_cast<std::size_t>(TaskProtocolOperation::kComplete),
        static_cast<std::size_t>(TaskProtocolOperation::kFail),
        static_cast<std::size_t>(TaskProtocolOperation::kStats),
        static_cast<std::size_t>(TaskProtocolOperation::kMetrics),
        sizeof(TaskProtocolResponseKind),
        static_cast<std::size_t>(TaskProtocolResponseKind::kAccepted),
        static_cast<std::size_t>(TaskProtocolResponseKind::kState),
        static_cast<std::size_t>(TaskProtocolResponseKind::kError),
        static_cast<std::size_t>(TaskProtocolResponseKind::kLease),
        static_cast<std::size_t>(TaskProtocolResponseKind::kEmpty),
        static_cast<std::size_t>(TaskProtocolResponseKind::kStats),
        static_cast<std::size_t>(TaskProtocolResponseKind::kMetrics),
    };
}

}  // namespace
