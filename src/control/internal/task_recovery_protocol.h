#pragma once

#include "glimmer/control/task_admission.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>

namespace glimmer::control::internal {

// This separate envelope never falls back to either older task protocol.
inline constexpr std::string_view kTaskRecoveryVersion = "GLIMMER_TASK_EXEC_V2";
inline constexpr std::size_t kTaskRecoveryMaxLineBytes = 256;
// Worst-case decimal identities and submission values leave 29 tenant bytes.
inline constexpr std::size_t kTaskRecoveryMaxTenantIdBytes = 29;
using TaskRecoveryToken = std::array<std::uint64_t, 2>;

enum class TaskRecoveryCommand : std::uint8_t { kOpen, kExecute, kAcknowledge, kClose };
enum class TaskRecoveryOperation : std::uint8_t {
    kSubmit,
    kClaim,
    kHeartbeat,
    kRetreat,
    kSuspend,
    kComplete,
    kFail,
    kCancelPending,
    kForget,
    kStatus,
};

// Tenant is a borrowed view into the request line. The view expires with it.
struct TaskRecoveryRequest {
    TaskRecoveryCommand command{TaskRecoveryCommand::kOpen};
    TaskRecoveryOperation operation{TaskRecoveryOperation::kStatus};
    TaskRecoveryToken nonce{};
    // OPEN has no expected incarnation; all later commands bind to one.
    TaskRecoveryToken incarnation{};
    std::uint64_t session_id{};
    std::uint64_t operation_id{};
    std::uint64_t acknowledged_through{};
    std::string_view tenant;
    core::MemoryBytes memory_bytes{};
    std::uint32_t weight{};
    std::uint32_t work_units{};
    std::uint32_t priority{};
    TaskExecutionIdentity execution{};
};

enum class TaskRecoveryResultKind : std::uint8_t {
    kOpened,
    kAccepted,
    kPending,
    kGranted,
    kAcknowledged,
    kRetreat,
    kRejected,
    kClosed,
    kStatus,
};
enum class TaskRecoveryError : std::uint8_t {
    kNone = 0,
    kInvalid = 1,
    kUnauthorized = 2,
    kUnknownTask = 3,
    kOwnership = 4,
    kNotReady = 5,
    kQuota = 6,
    kCapacity = 7,
    kPoisoned = 8,
    kUncertain = 9,
    kInternal = 10,
};
struct TaskRecoveryResponse {
    TaskRecoveryResultKind kind{TaskRecoveryResultKind::kRejected};
    std::uint64_t session_id{};
    std::uint64_t operation_id{};
    TaskRecoveryToken nonce{};
    // Every response is bound to the same authenticated service incarnation.
    TaskRecoveryToken incarnation{};
    TaskExecutionIdentity execution{};
    std::uint64_t acknowledged_through{};
    // Session-scoped accounting only; neither value proves GPU quiescence.
    std::uint64_t pending_tasks{};
    std::uint64_t active_tasks{};
    bool should_retreat{};
    TaskRecoveryError error{TaskRecoveryError::kNone};
};

// Fixed-size, allocation-free formatting storage; valid bytes exclude padding.
struct TaskRecoveryLine {
    std::array<char, kTaskRecoveryMaxLineBytes> bytes{};
    std::size_t size{};
    [[nodiscard]] std::string_view view() const noexcept {
        return {bytes.data(), size};
    }
};

// Parsing accepts only canonical encoding. Format rejects unused nonzero fields,
// invalid combinations, and lines that cannot fit the negotiated transport.
// Before any state access, a server must compare non-OPEN request incarnations
// with its own. Clients must match responses to their session and operation.
[[nodiscard]] std::optional<TaskRecoveryRequest> parse_task_recovery_request(
    std::string_view line) noexcept;
[[nodiscard]] std::optional<TaskRecoveryLine> format_task_recovery_request(
    const TaskRecoveryRequest& request) noexcept;
// The journal compares this canonical form, which excludes only the cumulative
// ACK. It never mutates scheduler state or authenticates peer identity.
[[nodiscard]] std::optional<TaskRecoveryLine> format_task_recovery_journal_key(
    const TaskRecoveryRequest& request) noexcept;
[[nodiscard]] std::optional<TaskRecoveryResponse> parse_task_recovery_response(
    std::string_view line) noexcept;
[[nodiscard]] std::optional<TaskRecoveryLine> format_task_recovery_response(
    const TaskRecoveryResponse& response) noexcept;

}  // namespace glimmer::control::internal
