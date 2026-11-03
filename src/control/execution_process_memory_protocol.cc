#include "glimmer/control/execution_process_memory_protocol.h"

#include <bit>

namespace glimmer::control {
namespace {
bool valid(const ExecutionProcessMemoryRequest& value) noexcept {
    if (!value.has_output && value.output_seed) return false;
    switch (value.operation) {
        case ExecutionProcessMemoryOperation::kGetDefaultPool:
        case ExecutionProcessMemoryOperation::kGetCurrentPool:
            return !value.pool && !value.bytes && !value.stream;
        case ExecutionProcessMemoryOperation::kSetCurrentPool:
            return !value.bytes && !value.stream && !value.has_output;
        case ExecutionProcessMemoryOperation::kAllocateAsync:
            return !value.device && !value.pool;
        case ExecutionProcessMemoryOperation::kAllocateFromPoolAsync:
            return !value.device;
        case ExecutionProcessMemoryOperation::kTrimPool:
            return !value.device && !value.stream && !value.has_output;
    }
    return false;
}
}  // namespace

bool write_execution_process_memory_request(ExecutionPayloadWriter& output,
                                            const ExecutionProcessMemoryRequest& value) noexcept {
    if (!valid(value) || !output.reserve_additional_capacity(kExecutionProcessMemoryRequestBytes))
        return false;
    return output.write_u32(1) && output.write_u32(static_cast<std::uint32_t>(value.operation)) &&
           output.write_u32(std::bit_cast<std::uint32_t>(value.device)) &&
           output.write_u32(value.has_output ? 1 : 0) && output.write_u64(value.pool) &&
           output.write_u64(value.bytes) && output.write_u64(value.stream) &&
           output.write_u64(value.output_seed);
}

bool read_execution_process_memory_request(ExecutionPayloadReader& input,
                                           ExecutionProcessMemoryRequest& value) noexcept {
    auto cursor = input;
    ExecutionProcessMemoryRequest decoded;
    std::uint32_t version{}, operation{}, device{}, has_output{};
    if (!cursor.read_u32(version) || version != 1 || !cursor.read_u32(operation) ||
        !cursor.read_u32(device) || !cursor.read_u32(has_output) || has_output > 1 ||
        !cursor.read_u64(decoded.pool) || !cursor.read_u64(decoded.bytes) ||
        !cursor.read_u64(decoded.stream) || !cursor.read_u64(decoded.output_seed))
        return false;
    decoded.operation = static_cast<ExecutionProcessMemoryOperation>(operation);
    decoded.device = std::bit_cast<std::int32_t>(device);
    decoded.has_output = has_output != 0;
    if (!valid(decoded)) return false;
    input = cursor;
    value = decoded;
    return true;
}

bool write_execution_process_memory_observation(ExecutionPayloadWriter& output,
                                                ExecutionProcessMemoryObservation value) noexcept {
    return output.reserve_additional_capacity(kExecutionProcessMemoryObservationBytes) &&
           output.write_u32(1) && output.write_u32(value.output_written ? 1 : 0);
}

bool read_execution_process_memory_observation(ExecutionPayloadReader& input,
                                               ExecutionProcessMemoryObservation& value) noexcept {
    auto cursor = input;
    std::uint32_t version{}, written{};
    if (!cursor.read_u32(version) || version != 1 || !cursor.read_u32(written) || written > 1)
        return false;
    input = cursor;
    value.output_written = written != 0;
    return true;
}

}  // namespace glimmer::control
