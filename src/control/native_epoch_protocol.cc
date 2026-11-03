#include "glimmer/control/native_epoch_protocol.h"

#include <array>
#include <charconv>

namespace glimmer::control {
namespace {

constexpr std::array<std::string_view, 7> kOperations{
    "ACQUIRE", "CLAIM", "RETREAT", "HEARTBEAT", "COMPLETE", "FAIL", "CANCEL_PENDING"};

struct Tokens {
    std::array<std::string_view, 16> values{};
    std::size_t count = 0;
};

std::optional<Tokens> tokenize(std::string_view line) noexcept {
    if (line.empty() || line.size() > kTaskProtocolMaxLineBytes) {
        return std::nullopt;
    }
    if (line.back() == '\n') {
        line.remove_suffix(1);
    }
    if (!line.empty() && line.back() == '\r') {
        line.remove_suffix(1);
    }
    Tokens result;
    while (!line.empty()) {
        const auto start = line.find_first_not_of(" \t");
        if (start == std::string_view::npos) {
            break;
        }
        line.remove_prefix(start);
        const auto end = line.find_first_of(" \t");
        if (result.count == result.values.size()) {
            return std::nullopt;
        }
        result.values[result.count++] = line.substr(0, end);
        if (end == std::string_view::npos) {
            break;
        }
        line.remove_prefix(end);
    }
    if (result.count < 2 || result.values[0] != kNativeEpochProtocolVersion) {
        return std::nullopt;
    }
    return result;
}

template <typename Integer>
bool integer(std::string_view text, Integer& value) noexcept {
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    return !text.empty() && parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size();
}

std::optional<NativeEpochOperation> operation(std::string_view token) noexcept {
    for (std::size_t index = 0; index < kOperations.size(); ++index) {
        if (kOperations[index] == token) {
            return static_cast<NativeEpochOperation>(index);
        }
    }
    return std::nullopt;
}

std::optional<std::string_view> operation_token(NativeEpochOperation value) noexcept {
    const auto index = static_cast<std::size_t>(value);
    return index < kOperations.size() ? std::optional{kOperations[index]} : std::nullopt;
}

bool parse_extent(const Tokens& tokens, std::size_t offset, NativeEpochExtent& extent) noexcept {
    return integer(tokens.values[offset], extent.max_calls) &&
           integer(tokens.values[offset + 1], extent.calls_per_prefix) &&
           integer(tokens.values[offset + 2], extent.max_pending_prefixes) &&
           integer(tokens.values[offset + 3], extent.acceptance_microseconds) &&
           valid_native_epoch_extent(extent);
}

std::string format_extent(const NativeEpochExtent& extent) {
    return std::to_string(extent.max_calls) + " " + std::to_string(extent.calls_per_prefix) + " " +
           std::to_string(extent.max_pending_prefixes) + " " +
           std::to_string(extent.acceptance_microseconds);
}

bool is_terminal(NativeEpochOperation value) noexcept {
    return value == NativeEpochOperation::kComplete || value == NativeEpochOperation::kFail;
}

bool is_pending_operation(NativeEpochOperation value) noexcept {
    return value == NativeEpochOperation::kClaim || value == NativeEpochOperation::kCancelPending;
}

}  // namespace

bool valid_native_epoch_extent(const NativeEpochExtent& extent) noexcept {
    return extent.max_calls != 0 && extent.max_calls <= 65'536 && extent.calls_per_prefix != 0 &&
           extent.calls_per_prefix <= extent.max_calls && extent.max_pending_prefixes != 0 &&
           extent.max_pending_prefixes <= 16 && extent.acceptance_microseconds != 0 &&
           extent.acceptance_microseconds <= 1'000'000;
}

std::optional<NativeEpochRequest> parse_native_epoch_request(std::string_view line) noexcept {
    try {
        auto tokens = tokenize(line);
        if (!tokens) {
            return std::nullopt;
        }
        const auto op = operation(tokens->values[1]);
        if (!op) {
            return std::nullopt;
        }
        NativeEpochRequest request;
        if (tokens->count < 3 ||
            !integer(tokens->values[tokens->count - 1], request.request_sequence) ||
            request.request_sequence == 0) {
            return std::nullopt;
        }
        --tokens->count;
        request.operation = *op;
        if (*op == NativeEpochOperation::kAcquire) {
            if (tokens->count != 11 || !parse_extent(*tokens, 7, request.extent)) {
                return std::nullopt;
            }
            // Reuse the ordinary admission validation, never its authorization.
            const auto admission = parse_task_protocol_request(
                std::string(kTaskProtocolVersion) + " SUBMIT " + std::string(tokens->values[2]) +
                " " + std::string(tokens->values[3]) + " " + std::string(tokens->values[4]) + " " +
                std::string(tokens->values[5]) + " " + std::string(tokens->values[6]));
            if (!admission.parsed() || !admission.request.has_value() ||
                admission.request->admission.work_units != request.extent.max_calls) {
                return std::nullopt;
            }
            request.admission = admission.request->admission;
            return request;
        }
        const bool pending = is_pending_operation(*op);
        const auto count = pending ? 3U : is_terminal(*op) ? 5U : 4U;
        if (tokens->count != count || !integer(tokens->values[2], request.execution.task_id) ||
            request.execution.task_id == 0 ||
            (!pending && (!integer(tokens->values[3], request.execution.generation) ||
                          request.execution.generation == 0)) ||
            (is_terminal(*op) && (!integer(tokens->values[4], request.issued_calls) ||
                                  request.issued_calls > 65'536))) {
            return std::nullopt;
        }
        return request;
    } catch (...) {
        return std::nullopt;
    }
}

std::optional<std::string> format_native_epoch_request(const NativeEpochRequest& request) {
    const auto token = operation_token(request.operation);
    if (!token) {
        return std::nullopt;
    }
    if ((is_pending_operation(request.operation) && request.execution.generation != 0) ||
        (!is_terminal(request.operation) && request.issued_calls != 0) ||
        (request.operation == NativeEpochOperation::kAcquire &&
         request.admission.tenant_id.size() > kTaskProtocolMaxTenantIdBytes)) {
        return std::nullopt;
    }
    std::string line = std::string(kNativeEpochProtocolVersion) + " " + std::string(*token) + " ";
    if (request.operation == NativeEpochOperation::kAcquire) {
        line += request.admission.tenant_id + " " + std::to_string(request.admission.memory_bytes) +
                " " + std::to_string(request.admission.weight) + " " +
                std::to_string(request.admission.work_units) + " " +
                std::to_string(request.admission.priority) + " " + format_extent(request.extent);
    } else {
        line += std::to_string(request.execution.task_id);
        if (!is_pending_operation(request.operation)) {
            line += " " + std::to_string(request.execution.generation);
        }
        if (is_terminal(request.operation)) {
            line += " " + std::to_string(request.issued_calls);
        }
    }
    line += " " + std::to_string(request.request_sequence) + "\n";
    return parse_native_epoch_request(line) ? std::optional{line} : std::nullopt;
}

std::optional<NativeEpochResponse> parse_native_epoch_response(std::string_view line) noexcept {
    try {
        auto tokens = tokenize(line);
        if (!tokens) {
            return std::nullopt;
        }
        NativeEpochResponse response;
        const auto kind = tokens->values[1];
        if (tokens->count < 3 ||
            !integer(tokens->values[tokens->count - 1], response.request_sequence) ||
            (kind != "ERROR" && response.request_sequence == 0)) {
            return std::nullopt;
        }
        --tokens->count;
        if (kind == "ERROR" && tokens->count == 3) {
            const auto error = parse_task_protocol_response(
                std::string(kTaskProtocolVersion) + " ERROR " + std::string(tokens->values[2]));
            if (!error.parsed() || !error.response.has_value()) {
                return std::nullopt;
            }
            response.error = error.response->error;
            return response;
        }
        if ((kind == "PENDING" && tokens->count == 7) ||
            (kind == "GRANTED" && tokens->count == 9)) {
            const bool granted = kind == "GRANTED";
            if (!integer(tokens->values[2], response.grant.execution.task_id) ||
                response.grant.execution.task_id == 0 ||
                !parse_extent(*tokens, granted ? 4 : 3, response.grant.extent) ||
                (granted &&
                 (!integer(tokens->values[3], response.grant.execution.generation) ||
                  response.grant.execution.generation == 0 ||
                  !integer(tokens->values[8], response.grant.acceptance_deadline_nanoseconds) ||
                  response.grant.acceptance_deadline_nanoseconds == 0))) {
                return std::nullopt;
            }
            response.kind =
                granted ? NativeEpochResponseKind::kGranted : NativeEpochResponseKind::kPending;
            return response;
        }
        if (kind == "RETREAT" && tokens->count == 5 &&
            integer(tokens->values[2], response.grant.execution.task_id) &&
            response.grant.execution.task_id != 0 &&
            integer(tokens->values[3], response.grant.execution.generation) &&
            response.grant.execution.generation != 0 &&
            (tokens->values[4] == "0" || tokens->values[4] == "1")) {
            response.kind = NativeEpochResponseKind::kRetreat;
            response.operation = NativeEpochOperation::kRetreat;
            response.should_retreat = tokens->values[4] == "1";
            return response;
        }
        if (kind == "ACK" && tokens->count == 6) {
            const auto op = operation(tokens->values[2]);
            if (!op ||
                (*op != NativeEpochOperation::kHeartbeat && !is_terminal(*op) &&
                 *op != NativeEpochOperation::kCancelPending) ||
                !integer(tokens->values[3], response.grant.execution.task_id) ||
                response.grant.execution.task_id == 0 ||
                !integer(tokens->values[4], response.grant.execution.generation) ||
                ((*op == NativeEpochOperation::kCancelPending) !=
                 (response.grant.execution.generation == 0)) ||
                !integer(tokens->values[5], response.issued_calls) ||
                response.issued_calls > 65'536 ||
                (!is_terminal(*op) && response.issued_calls != 0)) {
                return std::nullopt;
            }
            response.kind = NativeEpochResponseKind::kAcknowledged;
            response.operation = *op;
            return response;
        }
        return std::nullopt;
    } catch (...) {
        return std::nullopt;
    }
}

std::optional<std::string> format_native_epoch_response(const NativeEpochResponse& response) {
    std::string line = std::string(kNativeEpochProtocolVersion) + " ";
    const auto& grant = response.grant;
    switch (response.kind) {
        case NativeEpochResponseKind::kPending:
            if (grant.execution.generation != 0 || grant.acceptance_deadline_nanoseconds != 0) {
                return std::nullopt;
            }
            line += "PENDING " + std::to_string(grant.execution.task_id) + " " +
                    format_extent(grant.extent);
            break;
        case NativeEpochResponseKind::kGranted:
            line += "GRANTED " + std::to_string(grant.execution.task_id) + " " +
                    std::to_string(grant.execution.generation) + " " + format_extent(grant.extent) +
                    " " + std::to_string(grant.acceptance_deadline_nanoseconds);
            break;
        case NativeEpochResponseKind::kRetreat:
            line += "RETREAT " + std::to_string(grant.execution.task_id) + " " +
                    std::to_string(grant.execution.generation) +
                    (response.should_retreat ? " 1" : " 0");
            break;
        case NativeEpochResponseKind::kAcknowledged: {
            const auto token = operation_token(response.operation);
            if (!token) {
                return std::nullopt;
            }
            line += "ACK " + std::string(*token) + " " + std::to_string(grant.execution.task_id) +
                    " " + std::to_string(grant.execution.generation) + " " +
                    std::to_string(response.issued_calls);
            break;
        }
        case NativeEpochResponseKind::kError: {
            TaskProtocolResponse error;
            error.error = response.error;
            const auto formatted = format_task_protocol_response(error);
            if (!formatted) {
                return std::nullopt;
            }
            line += formatted->substr(kTaskProtocolVersion.size() + 1);
            line.pop_back();
            break;
        }
    }
    line += " " + std::to_string(response.request_sequence) + "\n";
    return parse_native_epoch_response(line) ? std::optional{line} : std::nullopt;
}

}  // namespace glimmer::control
