#include "glimmer/control/execution_graph_protocol.h"

namespace glimmer::control {
namespace {
bool valid(const ExecutionGraphRequest& value) noexcept {
    if ((!value.has_output && value.output_seed) ||
        value.payload.size() > kExecutionMaxPayloadBytes - kExecutionGraphRequestBytes)
        return false;
    const bool has_payload = !value.payload.empty();
    switch (value.action) {
        case ExecutionGraphAction::kBegin:
            return !value.object && value.caller && !value.has_output && !has_payload;
        case ExecutionGraphAction::kEnd:
            return !value.object && value.caller && !value.flags && !has_payload;
        case ExecutionGraphAction::kState:
            return !value.object && value.caller && !value.flags && !value.has_output &&
                   !has_payload;
        case ExecutionGraphAction::kInstantiate:
            return !value.stream && !value.caller && !has_payload;
        case ExecutionGraphAction::kDestroyGraph:
        case ExecutionGraphAction::kDestroyExecutable:
            return !value.stream && !value.caller && !value.flags && !value.has_output &&
                   !has_payload;
        case ExecutionGraphAction::kCaptureKernel:
        case ExecutionGraphAction::kCaptureMemory:
            return !value.object && value.caller && !value.flags && !value.has_output &&
                   has_payload;
        case ExecutionGraphAction::kInvalidate:
            return !value.object && value.caller && !value.flags && !value.has_output &&
                   !has_payload;
    }
    return false;
}
bool valid(const ExecutionGraphObservation& value) noexcept {
    return value.capture_status <= 2 && (value.output_written || !value.output);
}
}  // namespace

bool write_execution_graph_request(ExecutionPayloadWriter& output,
                                   const ExecutionGraphRequest& value) noexcept {
    if (!valid(value) ||
        !output.reserve_additional_capacity(kExecutionGraphRequestBytes + value.payload.size()))
        return false;
    return output.write_u32(1) && output.write_u32(static_cast<std::uint32_t>(value.action)) &&
           output.write_u32(value.has_output ? 1 : 0) && output.write_u32(0) &&
           output.write_u64(value.object) && output.write_u64(value.stream) &&
           output.write_u64(value.caller) && output.write_u64(value.flags) &&
           output.write_u64(value.output_seed) && output.write_blob(value.payload);
}
bool read_execution_graph_request(ExecutionPayloadReader& input,
                                  ExecutionGraphRequest& value) noexcept {
    auto cursor = input;
    ExecutionGraphRequest decoded;
    std::uint32_t version{}, action{}, presence{}, reserved{};
    if (!cursor.read_u32(version) || version != 1 || !cursor.read_u32(action) ||
        !cursor.read_u32(presence) || presence > 1 || !cursor.read_u32(reserved) || reserved ||
        !cursor.read_u64(decoded.object) || !cursor.read_u64(decoded.stream) ||
        !cursor.read_u64(decoded.caller) || !cursor.read_u64(decoded.flags) ||
        !cursor.read_u64(decoded.output_seed) || !cursor.read_blob(decoded.payload))
        return false;
    decoded.action = static_cast<ExecutionGraphAction>(action);
    decoded.has_output = presence != 0;
    if (!valid(decoded)) return false;
    value = decoded;
    input = cursor;
    return true;
}
bool write_execution_graph_observation(ExecutionPayloadWriter& output,
                                       const ExecutionGraphObservation& value) noexcept {
    if (!valid(value) || !output.reserve_additional_capacity(kExecutionGraphObservationBytes))
        return false;
    return output.write_u32(1) &&
           output.write_u32((value.output_written ? 1U : 0U) | (value.is_quarantined ? 2U : 0U)) &&
           output.write_u32(value.capture_status) && output.write_u32(0) &&
           output.write_u64(value.output);
}
bool read_execution_graph_observation(ExecutionPayloadReader& input,
                                      ExecutionGraphObservation& value) noexcept {
    auto cursor = input;
    ExecutionGraphObservation decoded;
    std::uint32_t version{}, flags{}, reserved{};
    if (!cursor.read_u32(version) || version != 1 || !cursor.read_u32(flags) || flags > 3 ||
        !cursor.read_u32(decoded.capture_status) || !cursor.read_u32(reserved) || reserved ||
        !cursor.read_u64(decoded.output))
        return false;
    decoded.output_written = (flags & 1U) != 0;
    decoded.is_quarantined = (flags & 2U) != 0;
    if (!valid(decoded)) return false;
    value = decoded;
    input = cursor;
    return true;
}
bool write_execution_graph_launch(ExecutionPayloadWriter& output,
                                  ExecutionGraphLaunch value) noexcept {
    return output.reserve_additional_capacity(kExecutionGraphLaunchBytes) &&
           output.write_u64(value.executable) && output.write_u64(value.stream);
}
bool read_execution_graph_launch(ExecutionPayloadReader& input,
                                 ExecutionGraphLaunch& value) noexcept {
    auto cursor = input;
    ExecutionGraphLaunch decoded;
    if (!cursor.read_u64(decoded.executable) || !cursor.read_u64(decoded.stream)) return false;
    input = cursor;
    value = decoded;
    return true;
}

}  // namespace glimmer::control
