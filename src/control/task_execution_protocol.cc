#include "glimmer/control/task_execution_protocol.h"

#include <array>
#include <charconv>
#include <utility>

namespace glimmer::control {
namespace {

constexpr std::size_t kMaxTokens = 24;
constexpr std::uint32_t kMaxWaitTimeoutMs = 60'000;

constexpr std::array<std::pair<std::string_view, TaskExecutionProtocolOperation>, 7>
    kExecutionOperations{{
        {"EXEC_CLAIM", TaskExecutionProtocolOperation::kExecutionClaim},
        {"RETREAT", TaskExecutionProtocolOperation::kRetreat},
        {"SUSPEND", TaskExecutionProtocolOperation::kSuspend},
        {"EXEC_HEARTBEAT", TaskExecutionProtocolOperation::kExecutionHeartbeat},
        {"EXEC_COMPLETE", TaskExecutionProtocolOperation::kExecutionComplete},
        {"EXEC_FAIL", TaskExecutionProtocolOperation::kExecutionFail},
        {"EXEC_CANCEL", TaskExecutionProtocolOperation::kExecutionCancel},
    }};

std::optional<TaskExecutionProtocolOperation> execution_operation(std::string_view name) {
    for (const auto& [token, operation] : kExecutionOperations) {
        if (token == name) {
            return operation;
        }
    }
    return std::nullopt;
}

std::optional<std::string_view> execution_token(TaskExecutionProtocolOperation operation) {
    for (const auto& [token, candidate] : kExecutionOperations) {
        if (candidate == operation) {
            return token;
        }
    }
    return std::nullopt;
}

struct TokenList {
    std::array<std::string_view, kMaxTokens> values{};
    std::size_t count = 0;
    bool overflow = false;
};

[[nodiscard]] bool is_separator(char value) noexcept {
    return value == ' ' || value == '\t';
}

[[nodiscard]] TokenList split_tokens(std::string_view line) noexcept {
    TokenList result;
    std::size_t offset = 0;
    while (offset < line.size()) {
        while (offset < line.size() && is_separator(line[offset])) {
            ++offset;
        }
        if (offset == line.size()) {
            break;
        }
        const std::size_t start = offset;
        while (offset < line.size() && !is_separator(line[offset])) {
            ++offset;
        }
        if (result.count == result.values.size()) {
            result.overflow = true;
            continue;
        }
        result.values[result.count] = line.substr(start, offset - start);
        ++result.count;
    }
    return result;
}

[[nodiscard]] TaskExecutionProtocolRequestParseResult request_error(
    TaskProtocolParseError error) noexcept {
    return TaskExecutionProtocolRequestParseResult{.error = error, .request = std::nullopt};
}

[[nodiscard]] TaskExecutionProtocolResponseParseResult response_error(
    TaskProtocolParseError error) noexcept {
    return TaskExecutionProtocolResponseParseResult{.error = error, .response = std::nullopt};
}

[[nodiscard]] bool is_ascii_alphanumeric(char value) noexcept {
    return (value >= 'a' && value <= 'z') || (value >= 'A' && value <= 'Z') ||
           (value >= '0' && value <= '9');
}

[[nodiscard]] bool is_valid_tenant_id(std::string_view tenant_id) noexcept {
    if (tenant_id.empty() || tenant_id.size() > kTaskProtocolMaxTenantIdBytes) {
        return false;
    }
    for (const char value : tenant_id) {
        if (!is_ascii_alphanumeric(value) && value != '.' && value != '_' && value != '-') {
            return false;
        }
    }
    return true;
}

template <typename Integer>
[[nodiscard]] bool parse_unsigned_integer(std::string_view token, Integer* value) noexcept {
    if (value == nullptr || token.empty()) {
        return false;
    }
    Integer parsed = 0;
    const auto result = std::from_chars(token.data(), token.data() + token.size(), parsed);
    if (result.ec != std::errc{} || result.ptr != token.data() + token.size()) {
        return false;
    }
    *value = parsed;
    return true;
}

template <typename Integer>
[[nodiscard]] bool parse_positive_integer(std::string_view token, Integer* value) noexcept {
    return parse_unsigned_integer(token, value) && *value != 0;
}

[[nodiscard]] std::optional<TaskProtocolState> parse_state(std::string_view token) noexcept {
    if (token == "QUEUED") {
        return TaskProtocolState::kQueued;
    }
    if (token == "RUNNING") {
        return TaskProtocolState::kRunning;
    }
    if (token == "COMPLETED") {
        return TaskProtocolState::kCompleted;
    }
    if (token == "CANCELLED") {
        return TaskProtocolState::kCancelled;
    }
    if (token == "FAILED") {
        return TaskProtocolState::kFailed;
    }
    return std::nullopt;
}

[[nodiscard]] std::optional<TaskProtocolErrorCode> parse_error_code(
    std::string_view token) noexcept {
    if (token == "INVALID_REQUEST") {
        return TaskProtocolErrorCode::kInvalidRequest;
    }
    if (token == "QUOTA_EXCEEDED") {
        return TaskProtocolErrorCode::kQuotaExceeded;
    }
    if (token == "QUEUE_FULL") {
        return TaskProtocolErrorCode::kQueueFull;
    }
    if (token == "UNKNOWN_TASK") {
        return TaskProtocolErrorCode::kUnknownTask;
    }
    if (token == "INTERNAL_ERROR") {
        return TaskProtocolErrorCode::kInternalError;
    }
    if (token == "UNSUPPORTED_VERSION") {
        return TaskProtocolErrorCode::kUnsupportedVersion;
    }
    return std::nullopt;
}

[[nodiscard]] std::optional<std::string_view> state_token(TaskProtocolState state) noexcept {
    switch (state) {
        case TaskProtocolState::kQueued:
            return "QUEUED";
        case TaskProtocolState::kRunning:
            return "RUNNING";
        case TaskProtocolState::kCompleted:
            return "COMPLETED";
        case TaskProtocolState::kCancelled:
            return "CANCELLED";
        case TaskProtocolState::kFailed:
            return "FAILED";
    }
    return std::nullopt;
}

[[nodiscard]] std::optional<std::string_view> error_token(TaskProtocolErrorCode error) noexcept {
    switch (error) {
        case TaskProtocolErrorCode::kInvalidRequest:
            return "INVALID_REQUEST";
        case TaskProtocolErrorCode::kQuotaExceeded:
            return "QUOTA_EXCEEDED";
        case TaskProtocolErrorCode::kQueueFull:
            return "QUEUE_FULL";
        case TaskProtocolErrorCode::kUnknownTask:
            return "UNKNOWN_TASK";
        case TaskProtocolErrorCode::kInternalError:
            return "INTERNAL_ERROR";
        case TaskProtocolErrorCode::kUnsupportedVersion:
            return "UNSUPPORTED_VERSION";
    }
    return std::nullopt;
}

[[nodiscard]] std::optional<std::string> format_task_id(core::TaskId task_id) {
    if (task_id == 0) {
        return std::nullopt;
    }
    return std::to_string(task_id);
}

}  // namespace

TaskExecutionProtocolRequestParseResult parse_task_execution_protocol_request(
    std::string_view line) noexcept {
    try {
        if (line.size() > kTaskProtocolMaxLineBytes) {
            return request_error(TaskProtocolParseError::kLineTooLong);
        }
        if (!line.empty() && line.back() == '\n') {
            line.remove_suffix(1);
        }
        if (!line.empty() && line.back() == '\r') {
            line.remove_suffix(1);
        }
        const TokenList tokens = split_tokens(line);
        if (tokens.count == 0) {
            return request_error(TaskProtocolParseError::kEmpty);
        }
        if (tokens.overflow) {
            return request_error(TaskProtocolParseError::kMalformed);
        }
        if (tokens.values[0] != kTaskExecutionProtocolVersion) {
            return request_error(TaskProtocolParseError::kUnsupportedVersion);
        }
        if (tokens.count < 2) {
            return request_error(TaskProtocolParseError::kMalformed);
        }

        TaskExecutionProtocolRequest request;
        const bool is_handoff = tokens.values[1] == "EXEC_COMPLETE_ACQUIRE";
        if (tokens.values[1] == "SUBMIT" || tokens.values[1] == "ACQUIRE" ||
            tokens.values[1] == "EXEC_ACQUIRE" || is_handoff) {
            const std::size_t offset = is_handoff ? 2 : 0;
            if ((tokens.count != 6 + offset && tokens.count != 7 + offset) ||
                !is_valid_tenant_id(tokens.values[2 + offset])) {
                return request_error(tokens.count == 6 + offset || tokens.count == 7 + offset
                                         ? TaskProtocolParseError::kInvalidTenant
                                         : TaskProtocolParseError::kMalformed);
            }
            if (is_handoff && (!parse_positive_integer(tokens.values[2], &request.task_id) ||
                               !parse_positive_integer(tokens.values[3], &request.generation))) {
                return request_error(TaskProtocolParseError::kInvalidValue);
            }
            request.operation =
                is_handoff ? TaskExecutionProtocolOperation::kExecutionCompleteAcquire
                : tokens.values[1] == "SUBMIT"  ? TaskExecutionProtocolOperation::kSubmit
                : tokens.values[1] == "ACQUIRE" ? TaskExecutionProtocolOperation::kAcquire
                                                : TaskExecutionProtocolOperation::kExecutionAcquire;
            request.admission.tenant_id = std::string(tokens.values[2 + offset]);
            if (!parse_positive_integer(tokens.values[3 + offset],
                                        &request.admission.memory_bytes) ||
                !parse_positive_integer(tokens.values[4 + offset], &request.admission.weight) ||
                !parse_positive_integer(tokens.values[5 + offset], &request.admission.work_units)) {
                return request_error(TaskProtocolParseError::kInvalidValue);
            }
            if (tokens.count == 7 + offset &&
                !parse_unsigned_integer(tokens.values[6 + offset], &request.admission.priority)) {
                return request_error(TaskProtocolParseError::kInvalidValue);
            }
            return TaskExecutionProtocolRequestParseResult{.error = TaskProtocolParseError::kNone,
                                                           .request = std::move(request)};
        }

        if (tokens.values[1] == "CLAIM") {
            if (tokens.count != 2 && tokens.count != 3) {
                return request_error(TaskProtocolParseError::kMalformed);
            }
            request.operation = TaskExecutionProtocolOperation::kClaim;
            if (tokens.count == 3) {
                std::uint64_t parsed_task_id = 0;
                if (!parse_unsigned_integer(tokens.values[2], &parsed_task_id)) {
                    return request_error(TaskProtocolParseError::kMalformed);
                }
                if (parsed_task_id == 0) {
                    return request_error(TaskProtocolParseError::kInvalidValue);
                }
                request.task_id = parsed_task_id;
            }
            return TaskExecutionProtocolRequestParseResult{.error = TaskProtocolParseError::kNone,
                                                           .request = std::move(request)};
        }

        if (const auto operation = execution_operation(tokens.values[1]); operation.has_value()) {
            const bool is_claim = *operation == TaskExecutionProtocolOperation::kExecutionClaim;
            if (tokens.count != (is_claim ? 3 : 4)) {
                return request_error(TaskProtocolParseError::kMalformed);
            }
            request.operation = *operation;
            if (!parse_positive_integer(tokens.values[2], &request.task_id) ||
                (!is_claim && !parse_positive_integer(tokens.values[3], &request.generation))) {
                return request_error(TaskProtocolParseError::kInvalidValue);
            }
            return {.error = TaskProtocolParseError::kNone, .request = std::move(request)};
        }

        if (tokens.values[1] == "WAIT") {
            if (tokens.count != 4) {
                return request_error(TaskProtocolParseError::kMalformed);
            }
            request.operation = TaskExecutionProtocolOperation::kWait;
            if (!parse_positive_integer(tokens.values[2], &request.task_id) ||
                !parse_positive_integer(tokens.values[3], &request.wait_timeout_ms) ||
                request.wait_timeout_ms > kMaxWaitTimeoutMs) {
                return request_error(TaskProtocolParseError::kInvalidValue);
            }
            return TaskExecutionProtocolRequestParseResult{.error = TaskProtocolParseError::kNone,
                                                           .request = std::move(request)};
        }

        if (tokens.values[1] == "STATS" || tokens.values[1] == "METRICS") {
            if (tokens.count != 2) {
                return request_error(TaskProtocolParseError::kMalformed);
            }
            request.operation = tokens.values[1] == "STATS"
                                    ? TaskExecutionProtocolOperation::kStats
                                    : TaskExecutionProtocolOperation::kMetrics;
            return TaskExecutionProtocolRequestParseResult{.error = TaskProtocolParseError::kNone,
                                                           .request = std::move(request)};
        }

        if (tokens.count != 3) {
            return request_error(TaskProtocolParseError::kMalformed);
        }
        if (!parse_positive_integer(tokens.values[2], &request.task_id)) {
            return request_error(TaskProtocolParseError::kInvalidValue);
        }
        if (tokens.values[1] == "CANCEL") {
            request.operation = TaskExecutionProtocolOperation::kCancel;
        } else if (tokens.values[1] == "QUERY") {
            request.operation = TaskExecutionProtocolOperation::kQuery;
        } else if (tokens.values[1] == "HEARTBEAT") {
            request.operation = TaskExecutionProtocolOperation::kHeartbeat;
        } else if (tokens.values[1] == "COMPLETE") {
            request.operation = TaskExecutionProtocolOperation::kComplete;
        } else if (tokens.values[1] == "FAIL") {
            request.operation = TaskExecutionProtocolOperation::kFail;
        } else {
            return request_error(TaskProtocolParseError::kMalformed);
        }
        return TaskExecutionProtocolRequestParseResult{.error = TaskProtocolParseError::kNone,
                                                       .request = std::move(request)};
    } catch (...) {
        return request_error(TaskProtocolParseError::kInternalError);
    }
}

std::optional<std::string> format_task_execution_protocol_request(
    const TaskExecutionProtocolRequest& request) {
    if (request.operation != TaskExecutionProtocolOperation::kExecutionCompleteAcquire &&
        request.operation != TaskExecutionProtocolOperation::kRetreat &&
        request.operation != TaskExecutionProtocolOperation::kSuspend &&
        request.operation != TaskExecutionProtocolOperation::kExecutionHeartbeat &&
        request.operation != TaskExecutionProtocolOperation::kExecutionComplete &&
        request.operation != TaskExecutionProtocolOperation::kExecutionFail &&
        request.operation != TaskExecutionProtocolOperation::kExecutionCancel &&
        request.generation != 0) {
        return std::nullopt;
    }
    if ((request.operation == TaskExecutionProtocolOperation::kSubmit ||
         request.operation == TaskExecutionProtocolOperation::kAcquire ||
         request.operation == TaskExecutionProtocolOperation::kExecutionAcquire ||
         request.operation == TaskExecutionProtocolOperation::kStats ||
         request.operation == TaskExecutionProtocolOperation::kMetrics) &&
        request.task_id != 0) {
        return std::nullopt;
    }
    if (const auto operation = execution_token(request.operation); operation.has_value()) {
        const bool is_claim = request.operation == TaskExecutionProtocolOperation::kExecutionClaim;
        if (request.task_id == 0 || (is_claim && request.generation != 0) ||
            (!is_claim && request.generation == 0)) {
            return std::nullopt;
        }
        auto formatted = std::string(kTaskExecutionProtocolVersion) + " " +
                         std::string(*operation) + " " + std::to_string(request.task_id);
        if (!is_claim) {
            formatted += " " + std::to_string(request.generation);
        }
        return formatted + "\n";
    }
    std::optional<std::string> formatted;
    if (request.operation == TaskExecutionProtocolOperation::kSubmit ||
        request.operation == TaskExecutionProtocolOperation::kAcquire ||
        request.operation == TaskExecutionProtocolOperation::kExecutionAcquire ||
        request.operation == TaskExecutionProtocolOperation::kExecutionCompleteAcquire) {
        const bool is_handoff =
            request.operation == TaskExecutionProtocolOperation::kExecutionCompleteAcquire;
        if (!is_valid_tenant_id(request.admission.tenant_id) ||
            request.admission.memory_bytes == 0 || request.admission.weight == 0 ||
            request.admission.work_units == 0 ||
            (is_handoff && (request.task_id == 0 || request.generation == 0))) {
            return std::nullopt;
        }
        const std::string_view operation =
            is_handoff                                                     ? "EXEC_COMPLETE_ACQUIRE"
            : request.operation == TaskExecutionProtocolOperation::kSubmit ? "SUBMIT"
            : request.operation == TaskExecutionProtocolOperation::kAcquire ? "ACQUIRE"
                                                                            : "EXEC_ACQUIRE";
        const auto prefix =
            std::string(kTaskExecutionProtocolVersion) + " " + std::string(operation) + " " +
            (is_handoff
                 ? std::to_string(request.task_id) + " " + std::to_string(request.generation) + " "
                 : "");
        formatted = prefix + request.admission.tenant_id + " " +
                    std::to_string(request.admission.memory_bytes) + " " +
                    std::to_string(request.admission.weight) + " " +
                    std::to_string(request.admission.work_units) + "\n";
        if (request.admission.priority != 0) {
            formatted = prefix + request.admission.tenant_id + " " +
                        std::to_string(request.admission.memory_bytes) + " " +
                        std::to_string(request.admission.weight) + " " +
                        std::to_string(request.admission.work_units) + " " +
                        std::to_string(request.admission.priority) + "\n";
        }
    } else if (request.operation == TaskExecutionProtocolOperation::kCancel ||
               request.operation == TaskExecutionProtocolOperation::kQuery ||
               request.operation == TaskExecutionProtocolOperation::kHeartbeat ||
               request.operation == TaskExecutionProtocolOperation::kComplete ||
               request.operation == TaskExecutionProtocolOperation::kFail) {
        const auto task_id = format_task_id(request.task_id);
        if (!task_id.has_value()) {
            return std::nullopt;
        }
        const std::string_view operation =
            request.operation == TaskExecutionProtocolOperation::kCancel      ? "CANCEL"
            : request.operation == TaskExecutionProtocolOperation::kQuery     ? "QUERY"
            : request.operation == TaskExecutionProtocolOperation::kHeartbeat ? "HEARTBEAT"
            : request.operation == TaskExecutionProtocolOperation::kComplete  ? "COMPLETE"
            : request.operation == TaskExecutionProtocolOperation::kFail      ? "FAIL"
                                                                              : "";
        if (operation.empty()) {
            return std::nullopt;
        }
        formatted = std::string(kTaskExecutionProtocolVersion) + " " + std::string(operation) +
                    " " + task_id.value() + "\n";
    } else if (request.operation == TaskExecutionProtocolOperation::kClaim) {
        if (request.task_id == 0) {
            formatted = std::string(kTaskExecutionProtocolVersion) + " CLAIM\n";
        } else {
            formatted = std::string(kTaskExecutionProtocolVersion) + " CLAIM " +
                        std::to_string(request.task_id) + "\n";
        }
    } else if (request.operation == TaskExecutionProtocolOperation::kWait) {
        if (request.task_id == 0 || request.wait_timeout_ms == 0 ||
            request.wait_timeout_ms > kMaxWaitTimeoutMs) {
            return std::nullopt;
        }
        formatted = std::string(kTaskExecutionProtocolVersion) + " WAIT " +
                    std::to_string(request.task_id) + " " +
                    std::to_string(request.wait_timeout_ms) + "\n";
    } else if (request.operation == TaskExecutionProtocolOperation::kStats ||
               request.operation == TaskExecutionProtocolOperation::kMetrics) {
        formatted = std::string(kTaskExecutionProtocolVersion) +
                    (request.operation == TaskExecutionProtocolOperation::kStats ? " STATS\n"
                                                                                 : " METRICS\n");
    }
    if (!formatted.has_value() || formatted->size() > kTaskProtocolMaxLineBytes) {
        return std::nullopt;
    }
    return formatted;
}

TaskExecutionProtocolResponseParseResult parse_task_execution_protocol_response(
    std::string_view line) noexcept {
    try {
        if (line.size() > kTaskProtocolMaxLineBytes) {
            return response_error(TaskProtocolParseError::kLineTooLong);
        }
        if (!line.empty() && line.back() == '\n') {
            line.remove_suffix(1);
        }
        if (!line.empty() && line.back() == '\r') {
            line.remove_suffix(1);
        }
        const TokenList tokens = split_tokens(line);
        if (tokens.count == 0) {
            return response_error(TaskProtocolParseError::kEmpty);
        }
        if (tokens.overflow || tokens.values[0] != kTaskExecutionProtocolVersion) {
            return response_error(tokens.overflow ? TaskProtocolParseError::kMalformed
                                                  : TaskProtocolParseError::kUnsupportedVersion);
        }

        TaskExecutionProtocolResponse response;
        if (tokens.count == 6 && tokens.values[1] == "EXEC_HANDOFF" &&
            parse_positive_integer(tokens.values[2], &response.completed_execution.task_id) &&
            parse_positive_integer(tokens.values[3], &response.completed_execution.generation) &&
            parse_unsigned_integer(tokens.values[4], &response.task_id) &&
            parse_unsigned_integer(tokens.values[5], &response.generation) &&
            response.task_id != response.completed_execution.task_id &&
            (response.task_id != 0 || response.generation == 0)) {
            response.kind = TaskExecutionProtocolResponseKind::kExecutionHandoff;
            return {.error = TaskProtocolParseError::kNone, .response = response};
        }
        if (tokens.count == 5 && tokens.values[1] == "RETREAT" &&
            parse_positive_integer(tokens.values[2], &response.task_id) &&
            parse_positive_integer(tokens.values[3], &response.generation) &&
            (tokens.values[4] == "0" || tokens.values[4] == "1")) {
            response.kind = TaskExecutionProtocolResponseKind::kRetreat;
            response.should_retreat = tokens.values[4] == "1";
            return {.error = TaskProtocolParseError::kNone, .response = response};
        }
        if (tokens.count == 2 && tokens.values[1] == "EMPTY") {
            response.kind = TaskExecutionProtocolResponseKind::kEmpty;
            return TaskExecutionProtocolResponseParseResult{.error = TaskProtocolParseError::kNone,
                                                            .response = response};
        }
        if (((tokens.count == 13 && tokens.values[1] == "STATS") ||
             (tokens.count == 17 && tokens.values[1] == "METRICS")) &&
            parse_unsigned_integer(tokens.values[2], &response.stats.total_task_count) &&
            parse_unsigned_integer(tokens.values[3], &response.stats.queued_task_count) &&
            parse_unsigned_integer(tokens.values[4], &response.stats.running_task_count) &&
            parse_unsigned_integer(tokens.values[5], &response.stats.completed_task_count) &&
            parse_unsigned_integer(tokens.values[6], &response.stats.cancelled_task_count) &&
            parse_unsigned_integer(tokens.values[7], &response.stats.failed_task_count) &&
            parse_unsigned_integer(tokens.values[8], &response.stats.quota_limit_bytes) &&
            parse_unsigned_integer(tokens.values[9], &response.stats.quota_reserved_bytes) &&
            parse_unsigned_integer(tokens.values[10], &response.stats.quota_allocated_bytes) &&
            parse_unsigned_integer(tokens.values[11], &response.stats.max_running_tasks) &&
            parse_unsigned_integer(tokens.values[12], &response.stats.max_queued_tasks) &&
            (tokens.count == 13 ||
             (parse_unsigned_integer(tokens.values[13],
                                     &response.stats.total_queue_wait_microseconds) &&
              parse_unsigned_integer(tokens.values[14],
                                     &response.stats.max_queue_wait_microseconds) &&
              parse_unsigned_integer(tokens.values[15],
                                     &response.stats.total_service_time_microseconds) &&
              parse_unsigned_integer(tokens.values[16],
                                     &response.stats.max_service_time_microseconds)))) {
            response.kind = tokens.values[1] == "STATS"
                                ? TaskExecutionProtocolResponseKind::kStats
                                : TaskExecutionProtocolResponseKind::kMetrics;
            return TaskExecutionProtocolResponseParseResult{.error = TaskProtocolParseError::kNone,
                                                            .response = response};
        }
        if (tokens.count == 3 && tokens.values[1] == "OK" &&
            parse_positive_integer(tokens.values[2], &response.task_id)) {
            response.kind = TaskExecutionProtocolResponseKind::kAccepted;
            return TaskExecutionProtocolResponseParseResult{.error = TaskProtocolParseError::kNone,
                                                            .response = response};
        }
        if (tokens.count == 4 && tokens.values[1] == "STATE" &&
            parse_positive_integer(tokens.values[2], &response.task_id)) {
            const auto state = parse_state(tokens.values[3]);
            if (state.has_value()) {
                response.kind = TaskExecutionProtocolResponseKind::kState;
                response.state = state.value();
                return TaskExecutionProtocolResponseParseResult{
                    .error = TaskProtocolParseError::kNone, .response = response};
            }
        }
        if (tokens.count == 3 && tokens.values[1] == "ERROR") {
            const auto error = parse_error_code(tokens.values[2]);
            if (error.has_value()) {
                response.kind = TaskExecutionProtocolResponseKind::kError;
                response.error = error.value();
                return TaskExecutionProtocolResponseParseResult{
                    .error = TaskProtocolParseError::kNone, .response = response};
            }
        }
        if (((tokens.count == 7 && tokens.values[1] == "LEASE") ||
             (tokens.count == 8 && tokens.values[1] == "EXEC_LEASE" &&
              parse_positive_integer(tokens.values[7], &response.generation))) &&
            parse_positive_integer(tokens.values[2], &response.task_id) &&
            is_valid_tenant_id(tokens.values[3]) &&
            parse_positive_integer(tokens.values[4], &response.memory_bytes) &&
            parse_positive_integer(tokens.values[5], &response.weight) &&
            parse_positive_integer(tokens.values[6], &response.work_units)) {
            response.kind = tokens.count == 8 ? TaskExecutionProtocolResponseKind::kExecutionLease
                                              : TaskExecutionProtocolResponseKind::kLease;
            response.tenant_id = std::string(tokens.values[3]);
            return TaskExecutionProtocolResponseParseResult{.error = TaskProtocolParseError::kNone,
                                                            .response = response};
        }
        return response_error(TaskProtocolParseError::kMalformed);
    } catch (...) {
        return response_error(TaskProtocolParseError::kInternalError);
    }
}

std::optional<std::string> format_task_execution_protocol_response(
    const TaskExecutionProtocolResponse& response) {
    if (response.kind != TaskExecutionProtocolResponseKind::kExecutionHandoff &&
        (response.completed_execution.task_id != 0 ||
         response.completed_execution.generation != 0)) {
        return std::nullopt;
    }
    if (response.kind != TaskExecutionProtocolResponseKind::kExecutionHandoff &&
        response.kind != TaskExecutionProtocolResponseKind::kExecutionLease &&
        response.kind != TaskExecutionProtocolResponseKind::kRetreat && response.generation != 0) {
        return std::nullopt;
    }
    if ((response.kind == TaskExecutionProtocolResponseKind::kEmpty ||
         response.kind == TaskExecutionProtocolResponseKind::kStats ||
         response.kind == TaskExecutionProtocolResponseKind::kMetrics ||
         response.kind == TaskExecutionProtocolResponseKind::kError) &&
        response.task_id != 0) {
        return std::nullopt;
    }
    if (response.kind == TaskExecutionProtocolResponseKind::kExecutionHandoff) {
        if (response.completed_execution.task_id == 0 ||
            response.completed_execution.generation == 0 ||
            response.task_id == response.completed_execution.task_id ||
            (response.task_id == 0 && response.generation != 0)) {
            return std::nullopt;
        }
        return std::string(kTaskExecutionProtocolVersion) + " EXEC_HANDOFF " +
               std::to_string(response.completed_execution.task_id) + " " +
               std::to_string(response.completed_execution.generation) + " " +
               std::to_string(response.task_id) + " " + std::to_string(response.generation) + "\n";
    }
    if (response.kind == TaskExecutionProtocolResponseKind::kAccepted) {
        const auto task_id = format_task_id(response.task_id);
        if (!task_id.has_value()) {
            return std::nullopt;
        }
        return std::string(kTaskExecutionProtocolVersion) + " OK " + task_id.value() + "\n";
    }
    if (response.kind == TaskExecutionProtocolResponseKind::kEmpty) {
        return std::string(kTaskExecutionProtocolVersion) + " EMPTY\n";
    }
    if (response.kind == TaskExecutionProtocolResponseKind::kLease ||
        response.kind == TaskExecutionProtocolResponseKind::kExecutionLease) {
        const auto task_id = format_task_id(response.task_id);
        if (!task_id.has_value() || !is_valid_tenant_id(response.tenant_id) ||
            response.memory_bytes == 0 || response.weight == 0 || response.work_units == 0 ||
            (response.kind == TaskExecutionProtocolResponseKind::kExecutionLease &&
             response.generation == 0)) {
            return std::nullopt;
        }
        std::string formatted =
            std::string(kTaskExecutionProtocolVersion) +
            (response.kind == TaskExecutionProtocolResponseKind::kExecutionLease ? " EXEC_LEASE "
                                                                                 : " LEASE ") +
            task_id.value() + " " + response.tenant_id + " " +
            std::to_string(response.memory_bytes) + " " + std::to_string(response.weight) + " " +
            std::to_string(response.work_units);
        if (response.kind == TaskExecutionProtocolResponseKind::kExecutionLease) {
            formatted += " " + std::to_string(response.generation);
        }
        formatted += "\n";
        if (formatted.size() > kTaskProtocolMaxLineBytes) {
            return std::nullopt;
        }
        return formatted;
    }
    if (response.kind == TaskExecutionProtocolResponseKind::kRetreat) {
        if (response.task_id == 0 || response.generation == 0) {
            return std::nullopt;
        }
        return std::string(kTaskExecutionProtocolVersion) + " RETREAT " +
               std::to_string(response.task_id) + " " + std::to_string(response.generation) +
               (response.should_retreat ? " 1\n" : " 0\n");
    }
    if (response.kind == TaskExecutionProtocolResponseKind::kState) {
        const auto task_id = format_task_id(response.task_id);
        const auto state = state_token(response.state);
        if (!task_id.has_value() || !state.has_value()) {
            return std::nullopt;
        }
        return std::string(kTaskExecutionProtocolVersion) + " STATE " + task_id.value() + " " +
               std::string(state.value()) + "\n";
    }
    if (response.kind == TaskExecutionProtocolResponseKind::kStats ||
        response.kind == TaskExecutionProtocolResponseKind::kMetrics) {
        const auto& stats = response.stats;
        std::string formatted =
            std::string(kTaskExecutionProtocolVersion) +
            (response.kind == TaskExecutionProtocolResponseKind::kStats ? " STATS " : " METRICS ") +
            std::to_string(stats.total_task_count) + " " + std::to_string(stats.queued_task_count) +
            " " + std::to_string(stats.running_task_count) + " " +
            std::to_string(stats.completed_task_count) + " " +
            std::to_string(stats.cancelled_task_count) + " " +
            std::to_string(stats.failed_task_count) + " " +
            std::to_string(stats.quota_limit_bytes) + " " +
            std::to_string(stats.quota_reserved_bytes) + " " +
            std::to_string(stats.quota_allocated_bytes) + " " +
            std::to_string(stats.max_running_tasks) + " " + std::to_string(stats.max_queued_tasks);
        if (response.kind == TaskExecutionProtocolResponseKind::kMetrics) {
            formatted += " " + std::to_string(stats.total_queue_wait_microseconds) + " " +
                         std::to_string(stats.max_queue_wait_microseconds) + " " +
                         std::to_string(stats.total_service_time_microseconds) + " " +
                         std::to_string(stats.max_service_time_microseconds);
        }
        formatted += "\n";
        if (formatted.size() > kTaskProtocolMaxLineBytes) {
            return std::nullopt;
        }
        return formatted;
    }
    if (response.kind != TaskExecutionProtocolResponseKind::kError) {
        return std::nullopt;
    }
    const auto error = error_token(response.error);
    if (!error.has_value()) {
        return std::nullopt;
    }
    return std::string(kTaskExecutionProtocolVersion) + " ERROR " + std::string(error.value()) +
           "\n";
}

}  // namespace glimmer::control
