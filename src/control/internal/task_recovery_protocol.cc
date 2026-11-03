#include "task_recovery_protocol.h"

#include <array>
#include <charconv>
#include <cstring>
#include <type_traits>

namespace glimmer::control::internal {
namespace {

static_assert(kTaskRecoveryVersion.size() + 2 + std::size_t{7} * 21 + 2 + 1 +
                  kTaskRecoveryMaxTenantIdBytes + 21 + std::size_t{3} * 11 + 1 ==
              kTaskRecoveryMaxLineBytes);

struct Tokens {
    std::array<std::string_view, 16> values{};
    std::size_t size{};
};

[[nodiscard]] std::optional<Tokens> split(std::string_view line) noexcept {
    if (line.empty() || line.size() > kTaskRecoveryMaxLineBytes || line.back() != '\n' ||
        line.find('\n') != line.size() - 1 || line.find('\0') != std::string_view::npos) {
        return std::nullopt;
    }
    line.remove_suffix(1);
    Tokens tokens;
    while (!line.empty()) {
        const auto separator = line.find(' ');
        const auto token = line.substr(0, separator);
        if (token.empty() || tokens.size == tokens.values.size()) return std::nullopt;
        tokens.values[tokens.size++] = token;
        if (separator == std::string_view::npos) break;
        line.remove_prefix(separator + 1);
        if (line.empty()) return std::nullopt;
    }
    return tokens;
}

template <typename Integer>
[[nodiscard]] bool number(std::string_view token, Integer& result) noexcept {
    if (token.empty()) return false;
    const auto parsed = std::from_chars(token.data(), token.data() + token.size(), result);
    return parsed.ec == std::errc{} && parsed.ptr == token.data() + token.size();
}

struct Builder {
    TaskRecoveryLine line;
    [[nodiscard]] bool append(std::string_view value) noexcept {
        if (value.size() > line.bytes.size() - line.size) return false;
        std::memcpy(line.bytes.data() + line.size, value.data(), value.size());
        line.size += value.size();
        return true;
    }
    [[nodiscard]] bool token(std::string_view value) noexcept {
        return append(" ") && append(value);
    }
    template <typename Integer>
        requires std::is_integral_v<Integer>
    [[nodiscard]] bool token(Integer value) noexcept {
        std::array<char, 24> buffer{};
        const auto formatted = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);
        return formatted.ec == std::errc{} &&
               token(std::string_view(buffer.data(),
                                      static_cast<std::size_t>(formatted.ptr - buffer.data())));
    }
    [[nodiscard]] std::optional<TaskRecoveryLine> finish() noexcept {
        return append("\n") ? std::optional{line} : std::nullopt;
    }
};

[[nodiscard]] bool nonzero(TaskRecoveryToken token) noexcept {
    return token[0] != 0 || token[1] != 0;
}

[[nodiscard]] bool tenant_valid(std::string_view tenant) noexcept {
    if (tenant.empty() || tenant.size() > kTaskRecoveryMaxTenantIdBytes) return false;
    for (const char c : tenant) {
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
              c == '_' || c == '-' || c == '.')) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] bool empty_payload(const TaskRecoveryRequest& request) noexcept {
    return request.tenant.empty() && request.memory_bytes == 0 && request.weight == 0 &&
           request.work_units == 0 && request.priority == 0 && request.execution.task_id == 0 &&
           request.execution.generation == 0;
}

[[nodiscard]] std::string_view operation_token(TaskRecoveryOperation operation) noexcept {
    switch (operation) {
        case TaskRecoveryOperation::kSubmit:
            return "S";
        case TaskRecoveryOperation::kClaim:
            return "C";
        case TaskRecoveryOperation::kHeartbeat:
            return "H";
        case TaskRecoveryOperation::kRetreat:
            return "R";
        case TaskRecoveryOperation::kSuspend:
            return "P";
        case TaskRecoveryOperation::kComplete:
            return "D";
        case TaskRecoveryOperation::kFail:
            return "F";
        case TaskRecoveryOperation::kCancelPending:
            return "X";
        case TaskRecoveryOperation::kForget:
            return "G";
        case TaskRecoveryOperation::kStatus:
            return "T";
    }
    return {};
}

[[nodiscard]] std::optional<TaskRecoveryOperation> parse_operation(
    std::string_view token) noexcept {
    constexpr std::array operations{
        TaskRecoveryOperation::kSubmit,    TaskRecoveryOperation::kClaim,
        TaskRecoveryOperation::kHeartbeat, TaskRecoveryOperation::kRetreat,
        TaskRecoveryOperation::kSuspend,   TaskRecoveryOperation::kComplete,
        TaskRecoveryOperation::kFail,      TaskRecoveryOperation::kCancelPending,
        TaskRecoveryOperation::kForget,    TaskRecoveryOperation::kStatus};
    for (const auto operation : operations) {
        if (operation_token(operation) == token) return operation;
    }
    return std::nullopt;
}

[[nodiscard]] bool task_only(TaskRecoveryOperation operation) noexcept {
    return operation == TaskRecoveryOperation::kClaim ||
           operation == TaskRecoveryOperation::kCancelPending ||
           operation == TaskRecoveryOperation::kForget;
}

[[nodiscard]] std::string_view result_token(TaskRecoveryResultKind kind) noexcept {
    switch (kind) {
        case TaskRecoveryResultKind::kOpened:
            return "O";
        case TaskRecoveryResultKind::kAccepted:
            return "A";
        case TaskRecoveryResultKind::kPending:
            return "P";
        case TaskRecoveryResultKind::kGranted:
            return "G";
        case TaskRecoveryResultKind::kAcknowledged:
            return "K";
        case TaskRecoveryResultKind::kRetreat:
            return "R";
        case TaskRecoveryResultKind::kRejected:
            return "E";
        case TaskRecoveryResultKind::kClosed:
            return "Z";
        case TaskRecoveryResultKind::kStatus:
            return "T";
    }
    return {};
}

[[nodiscard]] std::optional<TaskRecoveryResultKind> parse_result(std::string_view token) noexcept {
    constexpr std::array kinds{
        TaskRecoveryResultKind::kOpened,       TaskRecoveryResultKind::kAccepted,
        TaskRecoveryResultKind::kPending,      TaskRecoveryResultKind::kGranted,
        TaskRecoveryResultKind::kAcknowledged, TaskRecoveryResultKind::kRetreat,
        TaskRecoveryResultKind::kRejected,     TaskRecoveryResultKind::kClosed,
        TaskRecoveryResultKind::kStatus};
    for (const auto kind : kinds) {
        if (result_token(kind) == token) return kind;
    }
    return std::nullopt;
}

}  // namespace

std::optional<TaskRecoveryLine> format_task_recovery_request(
    const TaskRecoveryRequest& request) noexcept {
    if (!nonzero(request.nonce)) return std::nullopt;
    Builder builder;
    if (!builder.append(kTaskRecoveryVersion)) return std::nullopt;
    if (request.command == TaskRecoveryCommand::kOpen) {
        if (nonzero(request.incarnation) || request.session_id != 0 || request.operation_id != 0 ||
            request.acknowledged_through != 0 ||
            request.operation != TaskRecoveryOperation::kStatus || !empty_payload(request) ||
            !builder.token("O") || !builder.token(request.nonce[0]) ||
            !builder.token(request.nonce[1])) {
            return std::nullopt;
        }
        return builder.finish();
    }
    if (request.session_id == 0 || !nonzero(request.incarnation)) return std::nullopt;
    if (request.command == TaskRecoveryCommand::kAcknowledge ||
        request.command == TaskRecoveryCommand::kClose) {
        if (request.operation_id != 0 || request.operation != TaskRecoveryOperation::kStatus ||
            !empty_payload(request) ||
            (request.command == TaskRecoveryCommand::kClose && request.acknowledged_through != 0) ||
            !builder.token(request.command == TaskRecoveryCommand::kClose ? "Z" : "A") ||
            !builder.token(request.session_id) || !builder.token(request.nonce[0]) ||
            !builder.token(request.nonce[1]) || !builder.token(request.incarnation[0]) ||
            !builder.token(request.incarnation[1]) ||
            !builder.token(request.acknowledged_through)) {
            return std::nullopt;
        }
        return builder.finish();
    }
    const auto operation = operation_token(request.operation);
    if (request.command != TaskRecoveryCommand::kExecute || request.operation_id == 0 ||
        request.acknowledged_through >= request.operation_id || operation.empty() ||
        !builder.token("X") || !builder.token(request.session_id) ||
        !builder.token(request.nonce[0]) || !builder.token(request.nonce[1]) ||
        !builder.token(request.incarnation[0]) || !builder.token(request.incarnation[1]) ||
        !builder.token(request.operation_id) || !builder.token(request.acknowledged_through) ||
        !builder.token(operation)) {
        return std::nullopt;
    }
    if (request.operation == TaskRecoveryOperation::kSubmit) {
        if (!tenant_valid(request.tenant) || request.memory_bytes == 0 || request.weight == 0 ||
            request.work_units == 0 || request.execution.task_id != 0 ||
            request.execution.generation != 0 || !builder.token(request.tenant) ||
            !builder.token(request.memory_bytes) || !builder.token(request.weight) ||
            !builder.token(request.work_units) || !builder.token(request.priority)) {
            return std::nullopt;
        }
    } else if (request.operation == TaskRecoveryOperation::kStatus) {
        if (!empty_payload(request)) return std::nullopt;
    } else {
        if (!request.tenant.empty() || request.memory_bytes != 0 || request.weight != 0 ||
            request.work_units != 0 || request.priority != 0 || request.execution.task_id == 0 ||
            (task_only(request.operation) ? request.execution.generation != 0
                                          : request.execution.generation == 0) ||
            !builder.token(request.execution.task_id)) {
            return std::nullopt;
        }
        if (!task_only(request.operation) && !builder.token(request.execution.generation)) {
            return std::nullopt;
        }
    }
    return builder.finish();
}

std::optional<TaskRecoveryLine> format_task_recovery_journal_key(
    const TaskRecoveryRequest& request) noexcept {
    if (request.command != TaskRecoveryCommand::kExecute || !format_task_recovery_request(request))
        return std::nullopt;
    auto canonical = request;
    canonical.acknowledged_through = 0;
    return format_task_recovery_request(canonical);
}

std::optional<TaskRecoveryRequest> parse_task_recovery_request(std::string_view line) noexcept {
    const auto split_line = split(line);
    if (!split_line || split_line->size < 4 || split_line->values[0] != kTaskRecoveryVersion) {
        return std::nullopt;
    }
    const auto& tokens = *split_line;
    TaskRecoveryRequest request;
    const auto command = tokens.values[1];
    if (command == "O") {
        if (tokens.size != 4 || !number(tokens.values[2], request.nonce[0]) ||
            !number(tokens.values[3], request.nonce[1]))
            return std::nullopt;
    } else if (command == "A" || command == "Z") {
        request.command =
            command == "A" ? TaskRecoveryCommand::kAcknowledge : TaskRecoveryCommand::kClose;
        if (tokens.size != 8 || !number(tokens.values[2], request.session_id) ||
            !number(tokens.values[3], request.nonce[0]) ||
            !number(tokens.values[4], request.nonce[1]) ||
            !number(tokens.values[5], request.incarnation[0]) ||
            !number(tokens.values[6], request.incarnation[1]) ||
            !number(tokens.values[7], request.acknowledged_through))
            return std::nullopt;
    } else if (command == "X") {
        request.command = TaskRecoveryCommand::kExecute;
        if (tokens.size < 10 || !number(tokens.values[2], request.session_id) ||
            !number(tokens.values[3], request.nonce[0]) ||
            !number(tokens.values[4], request.nonce[1]) ||
            !number(tokens.values[5], request.incarnation[0]) ||
            !number(tokens.values[6], request.incarnation[1]) ||
            !number(tokens.values[7], request.operation_id) ||
            !number(tokens.values[8], request.acknowledged_through))
            return std::nullopt;
        const auto operation = parse_operation(tokens.values[9]);
        if (!operation) return std::nullopt;
        request.operation = *operation;
        if (*operation == TaskRecoveryOperation::kSubmit) {
            if (tokens.size != 15 || !number(tokens.values[11], request.memory_bytes) ||
                !number(tokens.values[12], request.weight) ||
                !number(tokens.values[13], request.work_units) ||
                !number(tokens.values[14], request.priority))
                return std::nullopt;
            request.tenant = tokens.values[10];
        } else if (*operation == TaskRecoveryOperation::kStatus) {
            if (tokens.size != 10) return std::nullopt;
        } else {
            if (tokens.size != (task_only(*operation) ? 11U : 12U) ||
                !number(tokens.values[10], request.execution.task_id) ||
                (!task_only(*operation) &&
                 !number(tokens.values[11], request.execution.generation))) {
                return std::nullopt;
            }
        }
    } else {
        return std::nullopt;
    }
    const auto canonical = format_task_recovery_request(request);
    return canonical && canonical->view() == line ? std::optional{request} : std::nullopt;
}

std::optional<TaskRecoveryLine> format_task_recovery_response(
    const TaskRecoveryResponse& response) noexcept {
    const auto kind = result_token(response.kind);
    if (kind.empty() || !nonzero(response.nonce) || !nonzero(response.incarnation)) {
        return std::nullopt;
    }
    const bool has_incarnation = nonzero(response.incarnation);
    const bool has_execution =
        response.execution.task_id != 0 || response.execution.generation != 0;
    const bool has_counts = response.pending_tasks != 0 || response.active_tasks != 0;
    const bool has_error = response.error != TaskRecoveryError::kNone;
    const auto error = static_cast<std::uint32_t>(response.error);
    if (error > static_cast<std::uint32_t>(TaskRecoveryError::kInternal)) return std::nullopt;
    switch (response.kind) {
        case TaskRecoveryResultKind::kOpened:
            if (response.session_id == 0 || response.operation_id != 0 || !has_incarnation ||
                has_execution || response.acknowledged_through != 0 || has_counts ||
                response.should_retreat || has_error)
                return std::nullopt;
            break;
        case TaskRecoveryResultKind::kAccepted:
        case TaskRecoveryResultKind::kPending:
            if (response.session_id == 0 || response.operation_id == 0 ||
                response.execution.task_id == 0 || response.execution.generation != 0 ||
                response.acknowledged_through != 0 || has_counts || response.should_retreat ||
                has_error)
                return std::nullopt;
            break;
        case TaskRecoveryResultKind::kGranted:
            if (response.session_id == 0 || response.operation_id == 0 ||
                response.execution.task_id == 0 || response.execution.generation == 0 ||
                response.acknowledged_through != 0 || has_counts || response.should_retreat ||
                has_error)
                return std::nullopt;
            break;
        case TaskRecoveryResultKind::kAcknowledged:
            if (response.session_id == 0 || has_execution || has_counts ||
                response.should_retreat || has_error ||
                (response.operation_id != 0 && response.acknowledged_through != 0)) {
                return std::nullopt;
            }
            break;
        case TaskRecoveryResultKind::kRetreat:
            if (response.session_id == 0 || response.operation_id == 0 ||
                response.execution.task_id == 0 || response.execution.generation == 0 ||
                response.acknowledged_through != 0 || has_counts || has_error)
                return std::nullopt;
            break;
        case TaskRecoveryResultKind::kRejected:
            if ((response.session_id == 0 && response.operation_id != 0) || has_execution ||
                response.acknowledged_through != 0 || has_counts || response.should_retreat ||
                !has_error)
                return std::nullopt;
            break;
        case TaskRecoveryResultKind::kClosed:
            if (response.session_id == 0 || response.operation_id != 0 || has_execution ||
                response.acknowledged_through != 0 || has_counts || response.should_retreat ||
                has_error)
                return std::nullopt;
            break;
        case TaskRecoveryResultKind::kStatus:
            if (response.session_id == 0 || response.operation_id == 0 || has_execution ||
                response.acknowledged_through != 0 || response.should_retreat || has_error)
                return std::nullopt;
            break;
    }
    Builder builder;
    if (!builder.append(kTaskRecoveryVersion) || !builder.token("R") || !builder.token(kind) ||
        !builder.token(response.session_id) || !builder.token(response.operation_id) ||
        !builder.token(response.nonce[0]) || !builder.token(response.nonce[1]) ||
        !builder.token(response.incarnation[0]) || !builder.token(response.incarnation[1]) ||
        !builder.token(response.execution.task_id) ||
        !builder.token(response.execution.generation) ||
        !builder.token(response.acknowledged_through) || !builder.token(response.pending_tasks) ||
        !builder.token(response.active_tasks) ||
        !builder.token(response.should_retreat ? 1U : 0U) || !builder.token(error))
        return std::nullopt;
    return builder.finish();
}

std::optional<TaskRecoveryResponse> parse_task_recovery_response(std::string_view line) noexcept {
    const auto parsed = split(line);
    if (!parsed || parsed->size != 16 || parsed->values[0] != kTaskRecoveryVersion ||
        parsed->values[1] != "R")
        return std::nullopt;
    const auto kind = parse_result(parsed->values[2]);
    if (!kind) return std::nullopt;
    TaskRecoveryResponse response;
    response.kind = *kind;
    std::uint32_t retreat = 0;
    std::uint32_t error = 0;
    if (!number(parsed->values[3], response.session_id) ||
        !number(parsed->values[4], response.operation_id) ||
        !number(parsed->values[5], response.nonce[0]) ||
        !number(parsed->values[6], response.nonce[1]) ||
        !number(parsed->values[7], response.incarnation[0]) ||
        !number(parsed->values[8], response.incarnation[1]) ||
        !number(parsed->values[9], response.execution.task_id) ||
        !number(parsed->values[10], response.execution.generation) ||
        !number(parsed->values[11], response.acknowledged_through) ||
        !number(parsed->values[12], response.pending_tasks) ||
        !number(parsed->values[13], response.active_tasks) ||
        !number(parsed->values[14], retreat) || retreat > 1 || !number(parsed->values[15], error) ||
        error > static_cast<std::uint32_t>(TaskRecoveryError::kInternal))
        return std::nullopt;
    response.should_retreat = retreat != 0;
    response.error = static_cast<TaskRecoveryError>(error);
    const auto canonical = format_task_recovery_response(response);
    return canonical && canonical->view() == line ? std::optional{response} : std::nullopt;
}

}  // namespace glimmer::control::internal
