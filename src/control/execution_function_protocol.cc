#include "glimmer/control/execution_function_protocol.h"

#include <bit>

namespace glimmer::control {

bool write_execution_function_attributes(ExecutionPayloadWriter& output,
                                         const ExecutionFunctionAttributes& value) noexcept {
    if (!output.reserve_additional_capacity(kExecutionFunctionAttributesBytes)) return false;
    for (const auto size : value.sizes)
        if (!output.write_u64(size)) return false;
    for (const auto field : value.values)
        if (!output.write_u32(std::bit_cast<std::uint32_t>(field))) return false;
    return true;
}

bool read_execution_function_attributes(ExecutionPayloadReader& input,
                                        ExecutionFunctionAttributes& value) noexcept {
    auto cursor = input;
    ExecutionFunctionAttributes decoded;
    for (auto& size : decoded.sizes)
        if (!cursor.read_u64(size)) return false;
    for (auto& field : decoded.values) {
        std::uint32_t bits{};
        if (!cursor.read_u32(bits)) return false;
        field = std::bit_cast<std::int32_t>(bits);
    }
    input = cursor;
    value = decoded;
    return true;
}

}  // namespace glimmer::control
