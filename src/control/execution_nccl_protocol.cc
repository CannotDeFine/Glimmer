#include "glimmer/control/execution_nccl_protocol.h"

#include <algorithm>
#include <bit>
#include <limits>
#include <type_traits>
#include <utility>

namespace glimmer::control {
namespace {
constexpr std::size_t kRequestPrefixBytes = 8 + kExecutionNcclScopeBytes;
constexpr std::array<std::size_t, 14> kConfigOffsets{16, 20, 24, 28, 40, 44, 56,
                                                     60, 64, 68, 72, 76, 80, 84};

bool valid(const ExecutionNcclScope& value) noexcept {
    return (value.thread[0] || value.thread[1]) && value.sequence;
}
bool valid(const ExecutionNcclCommunicator& value) noexcept {
    return static_cast<bool>(value.domain) == static_cast<bool>(value.object);
}
bool valid(const ExecutionNcclStream& value) noexcept {
    switch (value.kind) {
        case ExecutionNcclStreamKind::kDefault:
        case ExecutionNcclStreamKind::kLegacy:
        case ExecutionNcclStreamKind::kPerThread:
            return !value.owner && !value.object;
        case ExecutionNcclStreamKind::kOwned:
            return value.owner && value.object;
    }
    return false;
}
bool valid(const ExecutionNcclBuffer& value) noexcept {
    switch (value.kind) {
        case ExecutionNcclBufferKind::kNull:
        case ExecutionNcclBufferKind::kUnresolved:
            return !value.source && !value.allocation && !value.offset;
        case ExecutionNcclBufferKind::kOwned:
            return value.source && value.allocation;
    }
    return false;
}
struct ScalarPrefix {
    std::uint32_t bits;
    std::size_t offset;
};
bool canonical_prefix(const ExecutionNcclConfig& config, ScalarPrefix field) noexcept {
    for (std::size_t index = 0; index != 4; ++index)
        if (config.declared_bytes <= field.offset + index && ((field.bits >> (index * 8)) & 255U))
            return false;
    return true;
}
bool valid(const std::optional<std::string>& value) noexcept {
    return !value || (value->size() <= kExecutionNcclConfigStringLimit &&
                      value->find('\0') == std::string::npos);
}
bool valid(const ExecutionNcclConfig& value) noexcept {
    const auto size = value.declared_bytes;
    if ((size > 32 && size < 40) || (size > 48 && size < 56) || (size < 40 && value.net_name) ||
        (size < 56 && value.comm_name) || !canonical_prefix(value, {value.magic, 8}) ||
        !canonical_prefix(value, {value.version, 12}) || !valid(value.net_name) ||
        !valid(value.comm_name))
        return false;
    for (std::size_t index = 0; index != value.fields.size(); ++index)
        if (!canonical_prefix(
                value, {std::bit_cast<std::uint32_t>(value.fields[index]), kConfigOffsets[index]}))
            return false;
    return true;
}
bool valid(const ExecutionNcclAllGather& value) noexcept {
    return valid(value.communicator) && valid(value.stream) && valid(value.send) &&
           valid(value.receive);
}

std::optional<std::size_t> body_bytes(const ExecutionNcclCall& value) noexcept {
    if (std::get_if<ExecutionNcclGetUniqueId>(&value)) return 4;
    if (std::get_if<ExecutionNcclInitRank>(&value)) return 140;
    if (const auto* call = std::get_if<ExecutionNcclInitRankConfig>(&value)) {
        if (!call->config) return 144;
        const auto& config = *call->config;
        if (!valid(config)) return std::nullopt;
        // Both string lengths have already been capped, so this sum cannot
        // overflow even on a 32-bit control-only build.
        return 224 + (config.net_name ? 4 + config.net_name->size() : 0) +
               (config.comm_name ? 4 + config.comm_name->size() : 0);
    }
    if (const auto* call = std::get_if<ExecutionNcclGetAsyncError>(&value))
        return valid(call->communicator) ? std::optional<std::size_t>{20} : std::nullopt;
    if (const auto* call = std::get_if<ExecutionNcclDestroy>(&value))
        return valid(call->communicator) ? std::optional<std::size_t>{16} : std::nullopt;
    if (const auto* call = std::get_if<ExecutionNcclAbort>(&value))
        return valid(call->communicator) ? std::optional<std::size_t>{16} : std::nullopt;
    if (std::get_if<ExecutionNcclGroupStart>(&value) || std::get_if<ExecutionNcclGroupEnd>(&value))
        return 0;
    if (const auto* call = std::get_if<ExecutionNcclAllGather>(&value))
        return valid(*call) ? std::optional<std::size_t>{116} : std::nullopt;
    if (const auto* call = std::get_if<ExecutionNcclAllReduce>(&value))
        return valid(call->collective) ? std::optional<std::size_t>{120} : std::nullopt;
    if (const auto* call = std::get_if<ExecutionNcclBcast>(&value))
        return valid(call->communicator) && valid(call->stream) && valid(call->buffer)
                   ? std::optional<std::size_t>{88}
                   : std::nullopt;
    return std::nullopt;
}

bool write_signed(ExecutionPayloadWriter& output, std::int32_t value) noexcept {
    return output.write_u32(std::bit_cast<std::uint32_t>(value));
}
bool read_signed(ExecutionPayloadReader& input, std::int32_t& value) noexcept {
    std::uint32_t bits{};
    if (!input.read_u32(bits)) return false;
    value = std::bit_cast<std::int32_t>(bits);
    return true;
}
bool read_flag(ExecutionPayloadReader& input, bool& value) noexcept {
    std::uint32_t bits{};
    if (!input.read_u32(bits) || bits > 1) return false;
    value = bits != 0;
    return true;
}
template <std::size_t Size>
bool write_fixed(ExecutionPayloadWriter& output,
                 const std::array<std::byte, Size>& value) noexcept {
    static_assert(Size % 4 == 0);
    for (std::size_t index = 0; index != Size; index += 4) {
        std::uint32_t word{};
        for (std::size_t byte = 0; byte != 4; ++byte)
            word |= std::to_integer<std::uint32_t>(value[index + byte]) << (byte * 8);
        if (!output.write_u32(word)) return false;
    }
    return true;
}
template <std::size_t Size>
bool read_fixed(ExecutionPayloadReader& input, std::array<std::byte, Size>& value) noexcept {
    static_assert(Size % 4 == 0);
    for (std::size_t index = 0; index != Size; index += 4) {
        std::uint32_t word{};
        if (!input.read_u32(word)) return false;
        for (std::size_t byte = 0; byte != 4; ++byte)
            value[index + byte] = static_cast<std::byte>((word >> (byte * 8)) & 255U);
    }
    return true;
}
bool write_value(ExecutionPayloadWriter& output, const ExecutionNcclCommunicator& value) noexcept {
    return output.write_u64(value.domain) && output.write_u64(value.object);
}
bool read_value(ExecutionPayloadReader& input, ExecutionNcclCommunicator& value) noexcept {
    return input.read_u64(value.domain) && input.read_u64(value.object) && valid(value);
}
bool write_value(ExecutionPayloadWriter& output, const ExecutionNcclStream& value) noexcept {
    return output.write_u32(static_cast<std::uint32_t>(value.kind)) && output.write_u32(0) &&
           output.write_u64(value.owner) && output.write_u64(value.object);
}
bool read_value(ExecutionPayloadReader& input, ExecutionNcclStream& value) noexcept {
    std::uint32_t kind{}, reserved{};
    if (!input.read_u32(kind) || !input.read_u32(reserved) || reserved ||
        !input.read_u64(value.owner) || !input.read_u64(value.object))
        return false;
    value.kind = static_cast<ExecutionNcclStreamKind>(kind);
    return valid(value);
}
bool write_value(ExecutionPayloadWriter& output, const ExecutionNcclBuffer& value) noexcept {
    return output.write_u32(static_cast<std::uint32_t>(value.kind)) && output.write_u32(0) &&
           output.write_u64(value.source) && output.write_u64(value.allocation) &&
           output.write_u64(value.offset);
}
bool read_value(ExecutionPayloadReader& input, ExecutionNcclBuffer& value) noexcept {
    std::uint32_t kind{}, reserved{};
    if (!input.read_u32(kind) || !input.read_u32(reserved) || reserved ||
        !input.read_u64(value.source) || !input.read_u64(value.allocation) ||
        !input.read_u64(value.offset))
        return false;
    value.kind = static_cast<ExecutionNcclBufferKind>(kind);
    return valid(value);
}
bool write_text(ExecutionPayloadWriter& output, const std::optional<std::string>& value) noexcept {
    return output.write_u32(value ? 1U : 0U) && (!value || output.write_string(*value));
}
bool read_text(ExecutionPayloadReader& input, std::optional<std::string>& value) {
    bool present{};
    if (!read_flag(input, present)) return false;
    if (!present) return true;
    std::string_view text;
    if (!input.read_string(text) || text.size() > kExecutionNcclConfigStringLimit ||
        text.find('\0') != std::string_view::npos)
        return false;
    value.emplace(text);
    return true;
}
bool write_config(ExecutionPayloadWriter& output, const ExecutionNcclConfig& value) noexcept {
    if (!output.write_u64(value.declared_bytes) || !output.write_u32(value.magic) ||
        !output.write_u32(value.version))
        return false;
    for (const auto field : value.fields)
        if (!write_signed(output, field)) return false;
    return write_text(output, value.net_name) && write_text(output, value.comm_name);
}
bool read_config(ExecutionPayloadReader& input, ExecutionNcclConfig& value) {
    if (!input.read_u64(value.declared_bytes) || !input.read_u32(value.magic) ||
        !input.read_u32(value.version))
        return false;
    for (auto& field : value.fields)
        if (!read_signed(input, field)) return false;
    return valid(value) && read_text(input, value.net_name) && read_text(input, value.comm_name) &&
           valid(value);
}
bool write_body(ExecutionPayloadWriter& output, const ExecutionNcclGetUniqueId& value) noexcept {
    return output.write_u32(value.has_output ? 1U : 0U);
}
bool read_body(ExecutionPayloadReader& input, ExecutionNcclGetUniqueId& value) noexcept {
    return read_flag(input, value.has_output);
}
bool write_body(ExecutionPayloadWriter& output, const ExecutionNcclInitRank& value) noexcept {
    return output.write_u32(value.has_output ? 1U : 0U) && write_signed(output, value.ranks) &&
           write_signed(output, value.rank) && write_fixed(output, value.unique_id);
}
bool read_body(ExecutionPayloadReader& input, ExecutionNcclInitRank& value) noexcept {
    return read_flag(input, value.has_output) && read_signed(input, value.ranks) &&
           read_signed(input, value.rank) && read_fixed(input, value.unique_id);
}
bool write_body(ExecutionPayloadWriter& output, const ExecutionNcclInitRankConfig& value) noexcept {
    return write_body(output, value.initialization) && output.write_u32(value.config ? 1U : 0U) &&
           (!value.config || write_config(output, *value.config));
}
bool read_body(ExecutionPayloadReader& input, ExecutionNcclInitRankConfig& value) {
    bool present{};
    if (!read_body(input, value.initialization) || !read_flag(input, present)) return false;
    if (!present) return true;
    value.config.emplace();
    return read_config(input, *value.config);
}
bool write_body(ExecutionPayloadWriter& output, const ExecutionNcclGetAsyncError& value) noexcept {
    return write_value(output, value.communicator) && output.write_u32(value.has_output ? 1U : 0U);
}
bool read_body(ExecutionPayloadReader& input, ExecutionNcclGetAsyncError& value) noexcept {
    return read_value(input, value.communicator) && read_flag(input, value.has_output);
}
bool write_body(ExecutionPayloadWriter& output, const ExecutionNcclDestroy& value) noexcept {
    return write_value(output, value.communicator);
}
bool read_body(ExecutionPayloadReader& input, ExecutionNcclDestroy& value) noexcept {
    return read_value(input, value.communicator);
}
bool write_body(ExecutionPayloadWriter& output, const ExecutionNcclAbort& value) noexcept {
    return write_value(output, value.communicator);
}
bool read_body(ExecutionPayloadReader& input, ExecutionNcclAbort& value) noexcept {
    return read_value(input, value.communicator);
}
bool write_body(ExecutionPayloadWriter&, const ExecutionNcclGroupStart&) noexcept {
    return true;
}
bool read_body(ExecutionPayloadReader&, ExecutionNcclGroupStart&) noexcept {
    return true;
}
bool write_body(ExecutionPayloadWriter&, const ExecutionNcclGroupEnd&) noexcept {
    return true;
}
bool read_body(ExecutionPayloadReader&, ExecutionNcclGroupEnd&) noexcept {
    return true;
}
bool write_body(ExecutionPayloadWriter& output, const ExecutionNcclAllGather& value) noexcept {
    return write_value(output, value.communicator) && write_value(output, value.stream) &&
           write_value(output, value.send) && write_value(output, value.receive) &&
           output.write_u64(value.count) && write_signed(output, value.datatype);
}
bool read_body(ExecutionPayloadReader& input, ExecutionNcclAllGather& value) noexcept {
    return read_value(input, value.communicator) && read_value(input, value.stream) &&
           read_value(input, value.send) && read_value(input, value.receive) &&
           input.read_u64(value.count) && read_signed(input, value.datatype);
}
bool write_body(ExecutionPayloadWriter& output, const ExecutionNcclAllReduce& value) noexcept {
    return write_body(output, value.collective) && write_signed(output, value.reduction);
}
bool read_body(ExecutionPayloadReader& input, ExecutionNcclAllReduce& value) noexcept {
    return read_body(input, value.collective) && read_signed(input, value.reduction);
}
bool write_body(ExecutionPayloadWriter& output, const ExecutionNcclBcast& value) noexcept {
    return write_value(output, value.communicator) && write_value(output, value.stream) &&
           write_value(output, value.buffer) && output.write_u64(value.count) &&
           write_signed(output, value.datatype) && write_signed(output, value.root);
}
bool read_body(ExecutionPayloadReader& input, ExecutionNcclBcast& value) noexcept {
    return read_value(input, value.communicator) && read_value(input, value.stream) &&
           read_value(input, value.buffer) && input.read_u64(value.count) &&
           read_signed(input, value.datatype) && read_signed(input, value.root);
}
template <class Call>
bool read_call(ExecutionPayloadReader& input, ExecutionNcclCall& value) {
    Call decoded;
    if (!read_body(input, decoded)) return false;
    value = std::move(decoded);
    return true;
}
bool read_call(ExecutionPayloadReader& input, ExecutionNcclCall& value,
               ExecutionNcclOperation operation) {
    switch (operation) {
        case ExecutionNcclOperation::kGetUniqueId:
            return read_call<ExecutionNcclGetUniqueId>(input, value);
        case ExecutionNcclOperation::kInitRank:
            return read_call<ExecutionNcclInitRank>(input, value);
        case ExecutionNcclOperation::kInitRankConfig:
            return read_call<ExecutionNcclInitRankConfig>(input, value);
        case ExecutionNcclOperation::kGetAsyncError:
            return read_call<ExecutionNcclGetAsyncError>(input, value);
        case ExecutionNcclOperation::kDestroy:
            return read_call<ExecutionNcclDestroy>(input, value);
        case ExecutionNcclOperation::kAbort:
            return read_call<ExecutionNcclAbort>(input, value);
        case ExecutionNcclOperation::kGroupStart:
            return read_call<ExecutionNcclGroupStart>(input, value);
        case ExecutionNcclOperation::kGroupEnd:
            return read_call<ExecutionNcclGroupEnd>(input, value);
        case ExecutionNcclOperation::kAllGather:
            return read_call<ExecutionNcclAllGather>(input, value);
        case ExecutionNcclOperation::kAllReduce:
            return read_call<ExecutionNcclAllReduce>(input, value);
        case ExecutionNcclOperation::kBcast:
            return read_call<ExecutionNcclBcast>(input, value);
    }
    return false;
}
bool valid(const ExecutionNcclObservation& value) noexcept {
    if (value.effects > 7 || value.async_error_mask > 15 || !valid(value.communicator))
        return false;
    const bool has_uid = std::ranges::any_of(value.unique_id_mask,
                                             [](std::byte byte) { return byte != std::byte{}; });
    if (static_cast<bool>(value.effects & 1U) != has_uid ||
        (!(value.effects & 2U) && (value.communicator.domain || value.communicator.object)) ||
        static_cast<bool>(value.effects & 4U) != static_cast<bool>(value.async_error_mask))
        return false;
    for (std::size_t index = 0; index != value.unique_id.size(); ++index)
        if (!(std::to_integer<unsigned int>(value.unique_id_mask[index / 8]) &
              (1U << (index % 8))) &&
            value.unique_id[index] != std::byte{})
            return false;
    const auto scalar = std::bit_cast<std::uint32_t>(value.async_error);
    for (unsigned int index = 0; index != 4; ++index)
        if (!(value.async_error_mask & (1U << index)) && ((scalar >> (index * 8)) & 255U))
            return false;
    return true;
}
bool valid(const ExecutionNcclContextEffect& value) noexcept {
    switch (value.kind) {
        case ExecutionNcclContextEffectKind::kNone:
        case ExecutionNcclContextEffectKind::kClearCurrent:
            return !value.device && !value.domain;
        case ExecutionNcclContextEffectKind::kSelectPrimary:
            return value.device >= 0 && value.domain;
    }
    return false;
}
bool valid_reply_prefix(const ExecutionNcclReply& value) noexcept {
    if (!valid(value.reserved_communicator) || !valid(value.observation) ||
        !valid(value.context_effect))
        return false;
    if (value.is_accepted) return value.operation != 0;
    return !value.operation && !value.reserved_communicator.domain && !value.original_status &&
           !value.observation.effects &&
           value.context_effect.kind == ExecutionNcclContextEffectKind::kNone;
}
std::optional<std::size_t> reply_bytes(std::size_t publications) noexcept {
    if (publications > kExecutionNcclPublicationLimit ||
        publications > (std::numeric_limits<std::size_t>::max() - kExecutionNcclReplyPrefixBytes) /
                           kExecutionNcclPublicationBytes)
        return std::nullopt;
    return kExecutionNcclReplyPrefixBytes + publications * kExecutionNcclPublicationBytes;
}
bool write_scope(ExecutionPayloadWriter& output, const ExecutionNcclScope& value) noexcept {
    return output.write_u64(value.thread[0]) && output.write_u64(value.thread[1]) &&
           output.write_u64(value.sequence) && output.write_u64(value.group) &&
           output.write_u64(value.context);
}
bool read_scope(ExecutionPayloadReader& input, ExecutionNcclScope& value) noexcept {
    return input.read_u64(value.thread[0]) && input.read_u64(value.thread[1]) &&
           input.read_u64(value.sequence) && input.read_u64(value.group) &&
           input.read_u64(value.context) && valid(value);
}
bool read_publication(ExecutionPayloadReader& input, ExecutionNcclPublication& value) noexcept {
    return read_scope(input, value.scope) &&
           read_execution_nccl_observation(input, value.observation);
}
}  // namespace

bool write_execution_nccl_request(ExecutionPayloadWriter& output,
                                  const ExecutionNcclRequest& value) noexcept {
    const auto body = body_bytes(value.call);
    if (!valid(value.scope) || !body ||
        !output.reserve_additional_capacity(kRequestPrefixBytes + *body))
        return false;
    if (!output.write_u32(static_cast<std::uint32_t>(value.call.index() + 1)) ||
        !output.write_u32(0) || !output.write_u64(value.scope.thread[0]) ||
        !output.write_u64(value.scope.thread[1]) || !output.write_u64(value.scope.sequence) ||
        !output.write_u64(value.scope.group) || !output.write_u64(value.scope.context))
        return false;
    if (const auto* call = std::get_if<ExecutionNcclGetUniqueId>(&value.call))
        return write_body(output, *call);
    if (const auto* call = std::get_if<ExecutionNcclInitRank>(&value.call))
        return write_body(output, *call);
    if (const auto* call = std::get_if<ExecutionNcclInitRankConfig>(&value.call))
        return write_body(output, *call);
    if (const auto* call = std::get_if<ExecutionNcclGetAsyncError>(&value.call))
        return write_body(output, *call);
    if (const auto* call = std::get_if<ExecutionNcclDestroy>(&value.call))
        return write_body(output, *call);
    if (const auto* call = std::get_if<ExecutionNcclAbort>(&value.call))
        return write_body(output, *call);
    if (const auto* call = std::get_if<ExecutionNcclGroupStart>(&value.call))
        return write_body(output, *call);
    if (const auto* call = std::get_if<ExecutionNcclGroupEnd>(&value.call))
        return write_body(output, *call);
    if (const auto* call = std::get_if<ExecutionNcclAllGather>(&value.call))
        return write_body(output, *call);
    if (const auto* call = std::get_if<ExecutionNcclAllReduce>(&value.call))
        return write_body(output, *call);
    if (const auto* call = std::get_if<ExecutionNcclBcast>(&value.call))
        return write_body(output, *call);
    return false;
}

bool read_execution_nccl_request(ExecutionPayloadReader& input,
                                 ExecutionNcclRequest& value) noexcept {
    try {
        auto cursor = input;
        ExecutionNcclRequest decoded;
        std::uint32_t operation{}, reserved{};
        if (!cursor.read_u32(operation) || !cursor.read_u32(reserved) || reserved ||
            !cursor.read_u64(decoded.scope.thread[0]) ||
            !cursor.read_u64(decoded.scope.thread[1]) || !cursor.read_u64(decoded.scope.sequence) ||
            !cursor.read_u64(decoded.scope.group) || !cursor.read_u64(decoded.scope.context) ||
            !valid(decoded.scope) ||
            !read_call(cursor, decoded.call, static_cast<ExecutionNcclOperation>(operation)))
            return false;
        static_assert(std::is_nothrow_move_assignable_v<ExecutionNcclRequest>);
        value = std::move(decoded);
        input = cursor;
        return true;
    } catch (...) {
        return false;
    }
}

bool write_execution_nccl_observation(ExecutionPayloadWriter& output,
                                      const ExecutionNcclObservation& value) noexcept {
    if (!valid(value) || !output.reserve_additional_capacity(kExecutionNcclObservationBytes))
        return false;
    return output.write_u32(value.effects) && write_fixed(output, value.unique_id_mask) &&
           write_fixed(output, value.unique_id) && write_value(output, value.communicator) &&
           output.write_u32(value.async_error_mask) && write_signed(output, value.async_error);
}

bool read_execution_nccl_observation(ExecutionPayloadReader& input,
                                     ExecutionNcclObservation& value) noexcept {
    auto cursor = input;
    ExecutionNcclObservation decoded;
    if (!cursor.read_u32(decoded.effects) || !read_fixed(cursor, decoded.unique_id_mask) ||
        !read_fixed(cursor, decoded.unique_id) || !read_value(cursor, decoded.communicator) ||
        !cursor.read_u32(decoded.async_error_mask) || !read_signed(cursor, decoded.async_error) ||
        !valid(decoded))
        return false;
    value = decoded;
    input = cursor;
    return true;
}

bool write_execution_nccl_reply(ExecutionPayloadWriter& output,
                                const ExecutionNcclReply& value) noexcept {
    const auto bytes = reply_bytes(value.publications.size());
    if (!bytes || !valid_reply_prefix(value) || (!value.is_accepted && !value.publications.empty()))
        return false;
    for (const auto& publication : value.publications)
        if (!valid(publication.scope) || !valid(publication.observation)) return false;
    if (!output.reserve_additional_capacity(*bytes)) return false;
    if (!output.write_u32(kExecutionNcclReplyMagic) ||
        !output.write_u32(kExecutionNcclReplyVersion) || !output.write_u32(0) ||
        !output.write_u32(value.is_accepted ? 1U : 0U) || !output.write_u64(value.operation) ||
        !write_value(output, value.reserved_communicator) ||
        !output.write_u32(value.original_status ? 1U : 0U) ||
        !write_signed(output, value.original_status.value_or(0)) ||
        !write_execution_nccl_observation(output, value.observation) ||
        !output.write_u32(static_cast<std::uint32_t>(value.context_effect.kind)) ||
        !write_signed(output, value.context_effect.device) ||
        !output.write_u64(value.context_effect.domain) ||
        !output.write_u32(static_cast<std::uint32_t>(value.publications.size())))
        return false;
    for (const auto& publication : value.publications)
        if (!write_scope(output, publication.scope) ||
            !write_execution_nccl_observation(output, publication.observation))
            return false;
    return true;
}

bool read_execution_nccl_reply(ExecutionPayloadReader& input, ExecutionNcclReply& value) noexcept {
    try {
        auto cursor = input;
        ExecutionNcclReply decoded;
        std::uint32_t magic{}, version{}, reserved{}, context_kind{}, count{};
        bool has_original{};
        std::int32_t original{};
        if (!cursor.read_u32(magic) || magic != kExecutionNcclReplyMagic ||
            !cursor.read_u32(version) || version != kExecutionNcclReplyVersion ||
            !cursor.read_u32(reserved) || reserved || !read_flag(cursor, decoded.is_accepted) ||
            !cursor.read_u64(decoded.operation) ||
            !read_value(cursor, decoded.reserved_communicator) ||
            !read_flag(cursor, has_original) || !read_signed(cursor, original) ||
            (!has_original && original) ||
            !read_execution_nccl_observation(cursor, decoded.observation) ||
            !cursor.read_u32(context_kind) || !read_signed(cursor, decoded.context_effect.device) ||
            !cursor.read_u64(decoded.context_effect.domain) || !cursor.read_u32(count) ||
            !reply_bytes(count))
            return false;
        if (has_original) decoded.original_status = original;
        decoded.context_effect.kind = static_cast<ExecutionNcclContextEffectKind>(context_kind);
        if (!valid_reply_prefix(decoded) || (!decoded.is_accepted && count)) return false;

        // The reader has no remaining-byte accessor. A bounded, allocation-free
        // pass proves every fixed-size publication is present and canonical
        // before allocating storage from the untrusted count.
        auto publications = cursor;
        for (std::uint32_t index = 0; index != count; ++index) {
            ExecutionNcclPublication publication;
            if (!read_publication(cursor, publication)) return false;
        }
        decoded.publications.reserve(count);
        for (std::uint32_t index = 0; index != count; ++index) {
            ExecutionNcclPublication publication;
            if (!read_publication(publications, publication)) return false;
            decoded.publications.push_back(publication);
        }
        static_assert(std::is_nothrow_move_assignable_v<ExecutionNcclReply>);
        value = std::move(decoded);
        input = cursor;
        return true;
    } catch (...) {
        return false;
    }
}

}  // namespace glimmer::control
