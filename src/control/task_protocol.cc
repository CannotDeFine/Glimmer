#include "glimmer/control/task_protocol.h"

#include <array>
#include <charconv>
#include <utility>

namespace glimmer::control {
namespace {

constexpr std::size_t kMaxTokens = 24;
constexpr std::uint32_t kMaxWaitTimeoutMs = 60'000;

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

[[nodiscard]] TaskProtocolRequestParseResult request_error(TaskProtocolParseError error) noexcept {
    return TaskProtocolRequestParseResult{.error = error, .request = std::nullopt};
}

[[nodiscard]] TaskProtocolResponseParseResult response_error(
    TaskProtocolParseError error) noexcept {
    return TaskProtocolResponseParseResult{.error = error, .response = std::nullopt};
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

TaskProtocolRequestParseResult parse_task_protocol_request(std::string_view line) noexcept {
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
        if (tokens.values[0] != kTaskProtocolVersion) {
            return request_error(TaskProtocolParseError::kUnsupportedVersion);
        }
        if (tokens.count < 2) {
            return request_error(TaskProtocolParseError::kMalformed);
        }

        TaskProtocolRequest request;
        if (tokens.values[1] == "SUBMIT" || tokens.values[1] == "ACQUIRE") {
            if ((tokens.count != 6 && tokens.count != 7) || !is_valid_tenant_id(tokens.values[2])) {
                return request_error(tokens.count == 6 || tokens.count == 7
                                         ? TaskProtocolParseError::kInvalidTenant
                                         : TaskProtocolParseError::kMalformed);
            }
            request.operation = tokens.values[1] == "SUBMIT" ? TaskProtocolOperation::kSubmit
                                                             : TaskProtocolOperation::kAcquire;
            request.admission.tenant_id = std::string(tokens.values[2]);
            if (!parse_positive_integer(tokens.values[3], &request.admission.memory_bytes) ||
                !parse_positive_integer(tokens.values[4], &request.admission.weight) ||
                !parse_positive_integer(tokens.values[5], &request.admission.work_units)) {
                return request_error(TaskProtocolParseError::kInvalidValue);
            }
            if (tokens.count == 7 &&
                !parse_unsigned_integer(tokens.values[6], &request.admission.priority)) {
                return request_error(TaskProtocolParseError::kInvalidValue);
            }
            return TaskProtocolRequestParseResult{.error = TaskProtocolParseError::kNone,
                                                  .request = std::move(request)};
        }

        if (tokens.values[1] == "CLAIM") {
            if (tokens.count != 2 && tokens.count != 3) {
                return request_error(TaskProtocolParseError::kMalformed);
            }
            request.operation = TaskProtocolOperation::kClaim;
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
            return TaskProtocolRequestParseResult{.error = TaskProtocolParseError::kNone,
                                                  .request = std::move(request)};
        }

        if (tokens.values[1] == "WAIT") {
            if (tokens.count != 4) {
                return request_error(TaskProtocolParseError::kMalformed);
            }
            request.operation = TaskProtocolOperation::kWait;
            if (!parse_positive_integer(tokens.values[2], &request.task_id) ||
                !parse_positive_integer(tokens.values[3], &request.wait_timeout_ms) ||
                request.wait_timeout_ms > kMaxWaitTimeoutMs) {
                return request_error(TaskProtocolParseError::kInvalidValue);
            }
            return TaskProtocolRequestParseResult{.error = TaskProtocolParseError::kNone,
                                                  .request = std::move(request)};
        }

        if (tokens.values[1] == "STATS" || tokens.values[1] == "METRICS") {
            if (tokens.count != 2) {
                return request_error(TaskProtocolParseError::kMalformed);
            }
            request.operation = tokens.values[1] == "STATS" ? TaskProtocolOperation::kStats
                                                            : TaskProtocolOperation::kMetrics;
            return TaskProtocolRequestParseResult{.error = TaskProtocolParseError::kNone,
                                                  .request = std::move(request)};
        }

        if (tokens.count != 3) {
            return request_error(TaskProtocolParseError::kMalformed);
        }
        if (!parse_positive_integer(tokens.values[2], &request.task_id)) {
            return request_error(TaskProtocolParseError::kInvalidValue);
        }
        if (tokens.values[1] == "CANCEL") {
            request.operation = TaskProtocolOperation::kCancel;
        } else if (tokens.values[1] == "QUERY") {
            request.operation = TaskProtocolOperation::kQuery;
        } else if (tokens.values[1] == "HEARTBEAT") {
            request.operation = TaskProtocolOperation::kHeartbeat;
        } else if (tokens.values[1] == "COMPLETE") {
            request.operation = TaskProtocolOperation::kComplete;
        } else if (tokens.values[1] == "FAIL") {
            request.operation = TaskProtocolOperation::kFail;
        } else {
            return request_error(TaskProtocolParseError::kMalformed);
        }
        return TaskProtocolRequestParseResult{.error = TaskProtocolParseError::kNone,
                                              .request = std::move(request)};
    } catch (...) {
        return request_error(TaskProtocolParseError::kInternalError);
    }
}

std::optional<std::string> format_task_protocol_request(const TaskProtocolRequest& request) {
    std::optional<std::string> formatted;
    if (request.operation == TaskProtocolOperation::kSubmit ||
        request.operation == TaskProtocolOperation::kAcquire) {
        if (!is_valid_tenant_id(request.admission.tenant_id) ||
            request.admission.memory_bytes == 0 || request.admission.weight == 0 ||
            request.admission.work_units == 0) {
            return std::nullopt;
        }
        const std::string_view operation =
            request.operation == TaskProtocolOperation::kSubmit ? "SUBMIT" : "ACQUIRE";
        formatted = std::string(kTaskProtocolVersion) + " " + std::string(operation) + " " +
                    request.admission.tenant_id + " " +
                    std::to_string(request.admission.memory_bytes) + " " +
                    std::to_string(request.admission.weight) + " " +
                    std::to_string(request.admission.work_units) + "\n";
        if (request.admission.priority != 0) {
            formatted = std::string(kTaskProtocolVersion) + " " + std::string(operation) + " " +
                        request.admission.tenant_id + " " +
                        std::to_string(request.admission.memory_bytes) + " " +
                        std::to_string(request.admission.weight) + " " +
                        std::to_string(request.admission.work_units) + " " +
                        std::to_string(request.admission.priority) + "\n";
        }
    } else if (request.operation == TaskProtocolOperation::kCancel ||
               request.operation == TaskProtocolOperation::kQuery ||
               request.operation == TaskProtocolOperation::kHeartbeat ||
               request.operation == TaskProtocolOperation::kComplete ||
               request.operation == TaskProtocolOperation::kFail) {
        const auto task_id = format_task_id(request.task_id);
        if (!task_id.has_value()) {
            return std::nullopt;
        }
        const std::string_view operation =
            request.operation == TaskProtocolOperation::kCancel      ? "CANCEL"
            : request.operation == TaskProtocolOperation::kQuery     ? "QUERY"
            : request.operation == TaskProtocolOperation::kHeartbeat ? "HEARTBEAT"
            : request.operation == TaskProtocolOperation::kComplete  ? "COMPLETE"
            : request.operation == TaskProtocolOperation::kFail      ? "FAIL"
                                                                     : "";
        if (operation.empty()) {
            return std::nullopt;
        }
        formatted = std::string(kTaskProtocolVersion) + " " + std::string(operation) + " " +
                    task_id.value() + "\n";
    } else if (request.operation == TaskProtocolOperation::kClaim) {
        if (request.task_id == 0) {
            formatted = std::string(kTaskProtocolVersion) + " CLAIM\n";
        } else {
            formatted = std::string(kTaskProtocolVersion) + " CLAIM " +
                        std::to_string(request.task_id) + "\n";
        }
    } else if (request.operation == TaskProtocolOperation::kWait) {
        if (request.task_id == 0 || request.wait_timeout_ms == 0 ||
            request.wait_timeout_ms > kMaxWaitTimeoutMs) {
            return std::nullopt;
        }
        formatted = std::string(kTaskProtocolVersion) + " WAIT " + std::to_string(request.task_id) +
                    " " + std::to_string(request.wait_timeout_ms) + "\n";
    } else if (request.operation == TaskProtocolOperation::kStats ||
               request.operation == TaskProtocolOperation::kMetrics) {
        formatted =
            std::string(kTaskProtocolVersion) +
            (request.operation == TaskProtocolOperation::kStats ? " STATS\n" : " METRICS\n");
    }
    if (!formatted.has_value() || formatted->size() > kTaskProtocolMaxLineBytes) {
        return std::nullopt;
    }
    return formatted;
}

TaskProtocolResponseParseResult parse_task_protocol_response(std::string_view line) noexcept {
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
        if (tokens.overflow || tokens.values[0] != kTaskProtocolVersion) {
            return response_error(tokens.overflow ? TaskProtocolParseError::kMalformed
                                                  : TaskProtocolParseError::kUnsupportedVersion);
        }

        TaskProtocolResponse response;
        if (tokens.count == 2 && tokens.values[1] == "EMPTY") {
            response.kind = TaskProtocolResponseKind::kEmpty;
            return TaskProtocolResponseParseResult{.error = TaskProtocolParseError::kNone,
                                                   .response = response};
        }
        if ((tokens.count == 13 || tokens.count == 17) &&
            (tokens.values[1] == "STATS" || tokens.values[1] == "METRICS") &&
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
            response.kind = tokens.values[1] == "STATS" ? TaskProtocolResponseKind::kStats
                                                        : TaskProtocolResponseKind::kMetrics;
            return TaskProtocolResponseParseResult{.error = TaskProtocolParseError::kNone,
                                                   .response = response};
        }
        if (tokens.count == 3 && tokens.values[1] == "OK" &&
            parse_positive_integer(tokens.values[2], &response.task_id)) {
            response.kind = TaskProtocolResponseKind::kAccepted;
            return TaskProtocolResponseParseResult{.error = TaskProtocolParseError::kNone,
                                                   .response = response};
        }
        if (tokens.count == 4 && tokens.values[1] == "STATE" &&
            parse_positive_integer(tokens.values[2], &response.task_id)) {
            const auto state = parse_state(tokens.values[3]);
            if (state.has_value()) {
                response.kind = TaskProtocolResponseKind::kState;
                response.state = state.value();
                return TaskProtocolResponseParseResult{.error = TaskProtocolParseError::kNone,
                                                       .response = response};
            }
        }
        if (tokens.count == 3 && tokens.values[1] == "ERROR") {
            const auto error = parse_error_code(tokens.values[2]);
            if (error.has_value()) {
                response.kind = TaskProtocolResponseKind::kError;
                response.error = error.value();
                return TaskProtocolResponseParseResult{.error = TaskProtocolParseError::kNone,
                                                       .response = response};
            }
        }
        if (tokens.count == 7 && tokens.values[1] == "LEASE" &&
            parse_positive_integer(tokens.values[2], &response.task_id) &&
            is_valid_tenant_id(tokens.values[3]) &&
            parse_positive_integer(tokens.values[4], &response.memory_bytes) &&
            parse_positive_integer(tokens.values[5], &response.weight) &&
            parse_positive_integer(tokens.values[6], &response.work_units)) {
            response.kind = TaskProtocolResponseKind::kLease;
            response.tenant_id = std::string(tokens.values[3]);
            return TaskProtocolResponseParseResult{.error = TaskProtocolParseError::kNone,
                                                   .response = response};
        }
        return response_error(TaskProtocolParseError::kMalformed);
    } catch (...) {
        return response_error(TaskProtocolParseError::kInternalError);
    }
}

std::optional<std::string> format_task_protocol_response(const TaskProtocolResponse& response) {
    if (response.kind == TaskProtocolResponseKind::kAccepted) {
        const auto task_id = format_task_id(response.task_id);
        if (!task_id.has_value()) {
            return std::nullopt;
        }
        return std::string(kTaskProtocolVersion) + " OK " + task_id.value() + "\n";
    }
    if (response.kind == TaskProtocolResponseKind::kEmpty) {
        return std::string(kTaskProtocolVersion) + " EMPTY\n";
    }
    if (response.kind == TaskProtocolResponseKind::kLease) {
        const auto task_id = format_task_id(response.task_id);
        if (!task_id.has_value() || !is_valid_tenant_id(response.tenant_id) ||
            response.memory_bytes == 0 || response.weight == 0 || response.work_units == 0) {
            return std::nullopt;
        }
        std::string formatted =
            std::string(kTaskProtocolVersion) + " LEASE " + task_id.value() + " " +
            response.tenant_id + " " + std::to_string(response.memory_bytes) + " " +
            std::to_string(response.weight) + " " + std::to_string(response.work_units) + "\n";
        if (formatted.size() > kTaskProtocolMaxLineBytes) {
            return std::nullopt;
        }
        return formatted;
    }
    if (response.kind == TaskProtocolResponseKind::kState) {
        const auto task_id = format_task_id(response.task_id);
        const auto state = state_token(response.state);
        if (!task_id.has_value() || !state.has_value()) {
            return std::nullopt;
        }
        return std::string(kTaskProtocolVersion) + " STATE " + task_id.value() + " " +
               std::string(state.value()) + "\n";
    }
    if (response.kind == TaskProtocolResponseKind::kStats ||
        response.kind == TaskProtocolResponseKind::kMetrics) {
        const auto& stats = response.stats;
        std::string formatted =
            std::string(kTaskProtocolVersion) +
            (response.kind == TaskProtocolResponseKind::kStats ? " STATS " : " METRICS ") +
            std::to_string(stats.total_task_count) + " " + std::to_string(stats.queued_task_count) +
            " " + std::to_string(stats.running_task_count) + " " +
            std::to_string(stats.completed_task_count) + " " +
            std::to_string(stats.cancelled_task_count) + " " +
            std::to_string(stats.failed_task_count) + " " +
            std::to_string(stats.quota_limit_bytes) + " " +
            std::to_string(stats.quota_reserved_bytes) + " " +
            std::to_string(stats.quota_allocated_bytes) + " " +
            std::to_string(stats.max_running_tasks) + " " + std::to_string(stats.max_queued_tasks);
        if (response.kind == TaskProtocolResponseKind::kMetrics) {
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
    const auto error = error_token(response.error);
    if (!error.has_value()) {
        return std::nullopt;
    }
    return std::string(kTaskProtocolVersion) + " ERROR " + std::string(error.value()) + "\n";
}

}  // namespace glimmer::control
