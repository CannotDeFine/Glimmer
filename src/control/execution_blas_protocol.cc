#include "glimmer/control/execution_blas_protocol.h"

#include <algorithm>
#include <bit>

namespace glimmer::control {
namespace {
bool is_zero(std::span<const std::byte> bytes) noexcept {
    return std::ranges::all_of(bytes, [](auto value) { return value == std::byte{}; });
}
template <typename Operation>
bool valid_original_blas(const Operation& value, std::size_t width) noexcept {
    if (value.pointer_mode > 1 || value.presence > 31) return false;
    constexpr std::array<std::uint32_t, 5> bits{4, 8, 16, 1, 2};
    for (std::size_t index = 0; index != value.buffers.size(); ++index) {
        const auto& buffer = value.buffers[index];
        if ((!buffer.allocation && buffer.offset) ||
            (!(value.presence & bits[index]) && buffer.allocation))
            return false;
    }
    if (value.pointer_mode == 1) return is_zero(value.alpha) && is_zero(value.beta);
    if (value.buffers[3].allocation || value.buffers[4].allocation) return false;
    return is_zero(std::span{value.alpha}.subspan((value.presence & 1U) ? width : 0)) &&
           is_zero(std::span{value.beta}.subspan((value.presence & 2U) ? width : 0));
}
bool valid_geam(const ExecutionGeam& value) noexcept {
    return (value.type == ExecutionGeamType::kFloat || value.type == ExecutionGeamType::kDouble) &&
           valid_original_blas(value, value.type == ExecutionGeamType::kFloat ? 4U : 8U);
}
bool write_scalar(ExecutionPayloadWriter& output, const std::array<std::byte, 16>& bytes) noexcept {
    for (std::size_t index = 0; index != bytes.size(); index += 4) {
        std::uint32_t word{};
        for (std::size_t part = 0; part != 4; ++part)
            word |= std::to_integer<std::uint32_t>(bytes[index + part]) << (part * 8);
        if (!output.write_u32(word)) return false;
    }
    return true;
}
bool read_scalar(ExecutionPayloadReader& input, std::array<std::byte, 16>& bytes) noexcept {
    for (std::size_t index = 0; index != bytes.size(); index += 4) {
        std::uint32_t word{};
        if (!input.read_u32(word)) return false;
        for (std::size_t part = 0; part != 4; ++part)
            bytes[index + part] = static_cast<std::byte>(word >> (part * 8));
    }
    return true;
}
template <typename Operation>
bool write_original_blas_tail(ExecutionPayloadWriter& output, const Operation& value) noexcept {
    if (!output.write_u64(value.handle) || !output.write_u32(value.pointer_mode) ||
        !output.write_u32(value.presence))
        return false;
    for (const auto& buffer : value.buffers)
        if (!output.write_u64(buffer.allocation) || !output.write_u64(buffer.offset)) return false;
    return write_scalar(output, value.alpha) && write_scalar(output, value.beta);
}
template <typename Operation>
bool read_original_blas_tail(ExecutionPayloadReader& input, Operation& value) noexcept {
    if (!input.read_u64(value.handle) || !input.read_u32(value.pointer_mode) ||
        !input.read_u32(value.presence))
        return false;
    for (auto& buffer : value.buffers)
        if (!input.read_u64(buffer.allocation) || !input.read_u64(buffer.offset)) return false;
    return read_scalar(input, value.alpha) && read_scalar(input, value.beta) &&
           valid_original_blas(value, 4);
}
}  // namespace

bool write_execution_sgemv(ExecutionPayloadWriter& output, const ExecutionSgemv& value) noexcept {
    if (!valid_original_blas(value, 4) ||
        !output.reserve_additional_capacity(kExecutionSgemvBytes) ||
        !output.write_u32(value.transpose))
        return false;
    for (auto field :
         {value.rows, value.columns, value.leading_a, value.increment_x, value.increment_y})
        if (!output.write_u32(std::bit_cast<std::uint32_t>(field))) return false;
    return write_original_blas_tail(output, value);
}
bool read_execution_sgemv(ExecutionPayloadReader& input, ExecutionSgemv& value) noexcept {
    auto cursor = input;
    ExecutionSgemv decoded;
    if (!cursor.read_u32(decoded.transpose)) return false;
    for (auto* field : {&decoded.rows, &decoded.columns, &decoded.leading_a, &decoded.increment_x,
                        &decoded.increment_y}) {
        std::uint32_t bits{};
        if (!cursor.read_u32(bits)) return false;
        *field = std::bit_cast<std::int32_t>(bits);
    }
    if (!read_original_blas_tail(cursor, decoded)) return false;
    input = cursor;
    value = decoded;
    return true;
}
bool write_execution_sgemm_ex(ExecutionPayloadWriter& output,
                              const ExecutionSgemmEx& value) noexcept {
    if (!valid_original_blas(value, 4) ||
        !output.reserve_additional_capacity(kExecutionSgemmExBytes))
        return false;
    for (auto field :
         {value.transpose_a, value.transpose_b, value.type_a, value.type_b, value.type_c})
        if (!output.write_u32(field)) return false;
    for (auto field : {value.rows, value.columns, value.depth, value.leading_a, value.leading_b,
                       value.leading_c})
        if (!output.write_u32(std::bit_cast<std::uint32_t>(field))) return false;
    return write_original_blas_tail(output, value);
}
bool read_execution_sgemm_ex(ExecutionPayloadReader& input, ExecutionSgemmEx& value) noexcept {
    auto cursor = input;
    ExecutionSgemmEx decoded;
    for (auto* field : {&decoded.transpose_a, &decoded.transpose_b, &decoded.type_a,
                        &decoded.type_b, &decoded.type_c})
        if (!cursor.read_u32(*field)) return false;
    for (auto* field : {&decoded.rows, &decoded.columns, &decoded.depth, &decoded.leading_a,
                        &decoded.leading_b, &decoded.leading_c}) {
        std::uint32_t bits{};
        if (!cursor.read_u32(bits)) return false;
        *field = std::bit_cast<std::int32_t>(bits);
    }
    if (!read_original_blas_tail(cursor, decoded)) return false;
    input = cursor;
    value = decoded;
    return true;
}

bool write_execution_geam(ExecutionPayloadWriter& output, const ExecutionGeam& value) noexcept {
    if (!valid_geam(value) || !output.reserve_additional_capacity(kExecutionGeamBytes))
        return false;
    if (!output.write_u32(static_cast<std::uint32_t>(value.type)) ||
        !output.write_u32(value.transpose_a) || !output.write_u32(value.transpose_b))
        return false;
    for (auto field :
         {value.rows, value.columns, value.leading_a, value.leading_b, value.leading_c})
        if (!output.write_u32(std::bit_cast<std::uint32_t>(field))) return false;
    if (!output.write_u64(value.handle) || !output.write_u32(value.pointer_mode) ||
        !output.write_u32(value.presence))
        return false;
    for (const auto& buffer : value.buffers)
        if (!output.write_u64(buffer.allocation) || !output.write_u64(buffer.offset)) return false;
    return write_scalar(output, value.alpha) && write_scalar(output, value.beta);
}

bool read_execution_geam(ExecutionPayloadReader& input, ExecutionGeam& value) noexcept {
    auto cursor = input;
    ExecutionGeam decoded;
    std::uint32_t type{};
    if (!cursor.read_u32(type) || !cursor.read_u32(decoded.transpose_a) ||
        !cursor.read_u32(decoded.transpose_b))
        return false;
    decoded.type = static_cast<ExecutionGeamType>(type);
    for (auto* field : {&decoded.rows, &decoded.columns, &decoded.leading_a, &decoded.leading_b,
                        &decoded.leading_c}) {
        std::uint32_t bits{};
        if (!cursor.read_u32(bits)) return false;
        *field = std::bit_cast<std::int32_t>(bits);
    }
    if (!cursor.read_u64(decoded.handle) || !cursor.read_u32(decoded.pointer_mode) ||
        !cursor.read_u32(decoded.presence))
        return false;
    for (auto& buffer : decoded.buffers)
        if (!cursor.read_u64(buffer.allocation) || !cursor.read_u64(buffer.offset)) return false;
    if (!read_scalar(cursor, decoded.alpha) || !read_scalar(cursor, decoded.beta) ||
        !valid_geam(decoded))
        return false;
    input = cursor;
    value = decoded;
    return true;
}
}  // namespace glimmer::control
