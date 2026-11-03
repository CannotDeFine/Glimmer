#pragma once

#include "glimmer/control/task_admission.h"
#include "glimmer/control/task_protocol.h"

#include <optional>
#include <string>
#include <string_view>

namespace glimmer::control {

inline constexpr std::string_view kNativeEpochProtocolVersion = "GLIMMER_EPOCH_V2";

enum class NativeEpochOperation : std::uint8_t {
    kAcquire = 0,
    kClaim = 1,
    kRetreat = 2,
    kHeartbeat = 3,
    kComplete = 4,
    kFail = 5,
    kCancelPending = 6,
};

struct NativeEpochRequest {
    NativeEpochOperation operation = NativeEpochOperation::kAcquire;
    TaskAdmissionRequest admission;
    NativeEpochExtent extent;
    TaskExecutionIdentity execution;
    std::uint32_t issued_calls = 0;
    // Fresh process-lifetime correlation, not permission or CUDA proof. Every
    // normal reply echoes it; zero is reserved for an unparseable request error.
    std::uint64_t request_sequence = 0;
};

enum class NativeEpochResponseKind : std::uint8_t {
    kPending = 0,
    kGranted = 1,
    kRetreat = 2,
    kAcknowledged = 3,
    kError = 4,
};

struct NativeEpochResponse {
    NativeEpochResponseKind kind = NativeEpochResponseKind::kError;
    NativeEpochOperation operation = NativeEpochOperation::kAcquire;
    NativeEpochGrant grant;
    std::uint32_t issued_calls = 0;
    bool should_retreat = false;
    TaskProtocolErrorCode error = TaskProtocolErrorCode::kInvalidRequest;
    std::uint64_t request_sequence = 0;
};

// Strict bounded codecs. Malformed/oversized input fails closed; no transport,
// retry, fallback or authentication is performed here. The endpoint supplies
// trusted peer identity. Callers must authenticate the peer and correlate the
// sequence before treating a reply as authority.
[[nodiscard]] std::optional<NativeEpochRequest> parse_native_epoch_request(
    std::string_view line) noexcept;
[[nodiscard]] std::optional<std::string> format_native_epoch_request(
    const NativeEpochRequest& request);
[[nodiscard]] std::optional<NativeEpochResponse> parse_native_epoch_response(
    std::string_view line) noexcept;
[[nodiscard]] std::optional<std::string> format_native_epoch_response(
    const NativeEpochResponse& response);

}  // namespace glimmer::control
