#include "glimmer/control/execution_protocol.h"

#include <algorithm>
#include <bit>
#include <limits>
#include <utility>

namespace glimmer::control {
namespace {
constexpr std::uint32_t kMagic = 0x31584547;  // GEX1
std::uint64_t read_integer(std::span<const std::byte> bytes) noexcept {
    std::uint64_t value{};
    for (std::size_t index = 0; index < bytes.size(); ++index)
        value |= std::uint64_t{std::to_integer<unsigned char>(bytes[index])} << (index * 8);
    return value;
}
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): explicit value/byte-width codec primitive.
void append_integer(std::vector<std::byte>& bytes, std::uint64_t value, std::size_t count) {
    for (std::size_t index = 0; index < count; ++index)
        bytes.push_back(std::byte{static_cast<unsigned char>(value >> (index * 8))});
}
bool valid_async_frontier_key(ExecutionAsyncFrontierKey key,
                              std::span<const ExecutionAsyncFrontierKey> previous) noexcept {
    return key.id && key.sequence && std::ranges::none_of(previous, [key](const auto& prior) {
               return prior.id == key.id || prior.sequence == key.sequence;
           });
}
bool valid_async_frontier_status(std::int32_t status, bool is_retired) noexcept {
    // CUDA_ERROR_NOT_READY is 600 in the stable wire status namespace. This
    // protocol layer intentionally has no CUDA headers or runtime dependency.
    return status >= 0 && status <= 999 && !(status == 600 && is_retired);
}
bool valid_reply_argument(const ExecutionWireArgument& argument, std::uint64_t parameter_bytes,
                          std::uint64_t previous_end) noexcept {
    return argument.bytes && argument.offset >= previous_end &&
           argument.offset <= parameter_bytes &&
           argument.bytes <= parameter_bytes - argument.offset;
}
bool valid_copy_endpoint(const ExecutionCopyEndpoint& endpoint, std::uint64_t bytes) noexcept {
    if (bytes > UINT64_MAX - endpoint.offset) return false;
    switch (endpoint.kind) {
        case ExecutionCopyEndpointKind::kNull:
            return !bytes && !endpoint.connection && !endpoint.allocation && !endpoint.offset;
        case ExecutionCopyEndpointKind::kDevice:
            return !endpoint.connection && endpoint.allocation;
        case ExecutionCopyEndpointKind::kHost:
        case ExecutionCopyEndpointKind::kPageable:
            return endpoint.connection && endpoint.allocation;
    }
    return false;
}
}  // namespace

bool write_execution_unified_copy(ExecutionPayloadWriter& output,
                                  const ExecutionUnifiedCopy& value) noexcept {
    if ((!value.is_asynchronous && value.stream) ||
        !valid_copy_endpoint(value.destination, value.bytes) ||
        !valid_copy_endpoint(value.source, value.bytes) ||
        !output.reserve_additional_capacity(kExecutionUnifiedCopyBytes))
        return false;
    for (const auto& endpoint : {value.destination, value.source})
        if (!output.write_u32(static_cast<std::uint32_t>(endpoint.kind)) ||
            !output.write_u64(endpoint.connection) || !output.write_u64(endpoint.allocation) ||
            !output.write_u64(endpoint.offset))
            return false;
    return output.write_u64(value.bytes) && output.write_u64(value.stream) &&
           output.write_u32(value.is_asynchronous ? 1 : 0);
}
bool read_execution_unified_copy(ExecutionPayloadReader& input,
                                 ExecutionUnifiedCopy& value) noexcept {
    ExecutionUnifiedCopy decoded;
    for (auto* endpoint : {&decoded.destination, &decoded.source}) {
        std::uint32_t kind{};
        if (!input.read_u32(kind) ||
            kind > static_cast<std::uint32_t>(ExecutionCopyEndpointKind::kPageable) ||
            !input.read_u64(endpoint->connection) || !input.read_u64(endpoint->allocation) ||
            !input.read_u64(endpoint->offset))
            return false;
        endpoint->kind = static_cast<ExecutionCopyEndpointKind>(kind);
    }
    std::uint32_t asynchronous{};
    if (!input.read_u64(decoded.bytes) || !input.read_u64(decoded.stream) ||
        !input.read_u32(asynchronous) || asynchronous > 1 || !input.empty() ||
        (!asynchronous && decoded.stream) ||
        !valid_copy_endpoint(decoded.destination, decoded.bytes) ||
        !valid_copy_endpoint(decoded.source, decoded.bytes))
        return false;
    decoded.is_asynchronous = asynchronous != 0;
    value = decoded;
    return true;
}

std::optional<std::vector<std::byte>> encode_execution_async_frontier_batch(
    std::span<const ExecutionAsyncFrontierKey> keys) noexcept {
    if (keys.empty() || keys.size() > kExecutionMaxAsyncFrontiers) return std::nullopt;
    ExecutionPayloadWriter output(8 + 16 * kExecutionMaxAsyncFrontiers);
    if (!output.reserve_capacity(8 + 16 * keys.size()) || !output.write_u32(1) ||
        !output.write_u32(static_cast<std::uint32_t>(keys.size())))
        return std::nullopt;
    for (std::size_t index = 0; index < keys.size(); ++index) {
        const auto key = keys[index];
        if (!valid_async_frontier_key(key, keys.first(index)) || !output.write_u64(key.id) ||
            !output.write_u64(key.sequence))
            return std::nullopt;
    }
    return output.take();
}
std::optional<std::vector<ExecutionAsyncFrontierKey>> decode_execution_async_frontier_batch(
    std::span<const std::byte> bytes) noexcept {
    ExecutionPayloadReader input(bytes);
    std::uint32_t version{}, count{};
    if (!input.read_u32(version) || version != 1 || !input.read_u32(count) || !count ||
        count > kExecutionMaxAsyncFrontiers || bytes.size() != 8 + 16 * count)
        return std::nullopt;
    try {
        std::vector<ExecutionAsyncFrontierKey> result;
        result.reserve(count);
        for (std::uint32_t index = 0; index < count; ++index) {
            ExecutionAsyncFrontierKey key;
            if (!input.read_u64(key.id) || !input.read_u64(key.sequence) ||
                !valid_async_frontier_key(key, result))
                return std::nullopt;
            result.push_back(key);
        }
        return result;
    } catch (...) {
        return std::nullopt;
    }
}
std::optional<std::vector<std::byte>> encode_execution_async_frontier_observations(
    std::span<const ExecutionAsyncFrontierObservation> observations) noexcept {
    if (observations.empty() || observations.size() > kExecutionMaxAsyncFrontiers)
        return std::nullopt;
    std::array<ExecutionAsyncFrontierKey, kExecutionMaxAsyncFrontiers> keys{};
    ExecutionPayloadWriter output(8 + 24 * kExecutionMaxAsyncFrontiers);
    if (!output.reserve_capacity(8 + 24 * observations.size()) || !output.write_u32(1) ||
        !output.write_u32(static_cast<std::uint32_t>(observations.size())))
        return std::nullopt;
    for (std::size_t index = 0; index < observations.size(); ++index) {
        const auto& observation = observations[index];
        if (!valid_async_frontier_key(observation.key, std::span{keys}.first(index)) ||
            !valid_async_frontier_status(observation.status, observation.is_retired) ||
            !output.write_u64(observation.key.id) || !output.write_u64(observation.key.sequence) ||
            !output.write_u32(static_cast<std::uint32_t>(observation.status)) ||
            !output.write_u32(observation.is_retired ? 1 : 0))
            return std::nullopt;
        keys[index] = observation.key;
    }
    return output.take();
}
std::optional<std::vector<ExecutionAsyncFrontierObservation>>
decode_execution_async_frontier_observations(std::span<const std::byte> bytes) noexcept {
    ExecutionPayloadReader input(bytes);
    std::uint32_t version{}, count{};
    if (!input.read_u32(version) || version != 1 || !input.read_u32(count) || !count ||
        count > kExecutionMaxAsyncFrontiers || bytes.size() != 8 + 24 * count)
        return std::nullopt;
    try {
        std::vector<ExecutionAsyncFrontierObservation> result;
        std::array<ExecutionAsyncFrontierKey, kExecutionMaxAsyncFrontiers> keys{};
        result.reserve(count);
        for (std::uint32_t index = 0; index < count; ++index) {
            ExecutionAsyncFrontierObservation observation;
            std::uint32_t status{}, retired{};
            if (!input.read_u64(observation.key.id) || !input.read_u64(observation.key.sequence) ||
                !input.read_u32(status) || !input.read_u32(retired) || retired > 1 ||
                status > 999 ||
                !valid_async_frontier_key(observation.key, std::span{keys}.first(index)))
                return std::nullopt;
            observation.status = static_cast<std::int32_t>(status);
            observation.is_retired = retired != 0;
            if (!valid_async_frontier_status(observation.status, observation.is_retired))
                return std::nullopt;
            keys[index] = observation.key;
            result.push_back(observation);
        }
        return result;
    } catch (...) {
        return std::nullopt;
    }
}

std::optional<ExecutionRequestView> normalize_execution_request(
    ExecutionOpcode outer, std::span<const std::byte> payload,
    std::size_t ordinary_limit) noexcept {
    const auto opcode = static_cast<std::uint32_t>(outer);
    if (opcode < static_cast<std::uint32_t>(ExecutionOpcode::kHello) ||
        opcode > static_cast<std::uint32_t>(ExecutionOpcode::kDnnLegacyCompute))
        return std::nullopt;
    if (outer != ExecutionOpcode::kSourceContinuation) {
        if (payload.size() > ordinary_limit) return std::nullopt;
        return ExecutionRequestView{outer, payload, std::nullopt};
    }
    if (payload.size() < kExecutionSourceContinuationBytes ||
        payload.size() - kExecutionSourceContinuationBytes > ordinary_limit ||
        read_integer(payload.first(4)) != kExecutionSourceContinuationMagic)
        return std::nullopt;
    ExecutionSourceContinuation scope;
    scope.flags = static_cast<std::uint32_t>(read_integer(payload.subspan(4, 4)));
    scope.caller = {read_integer(payload.subspan(8, 8)), read_integer(payload.subspan(16, 8))};
    scope.group = read_integer(payload.subspan(24, 8));
    scope.ordinary_sequence = read_integer(payload.subspan(32, 8));
    scope.origin_source = read_integer(payload.subspan(40, 8));
    scope.cleanup_parent = read_integer(payload.subspan(48, 8));
    const auto inner = read_integer(payload.subspan(56, 4));
    const auto bytes = read_integer(payload.subspan(60, 4));
    if (scope.flags > kExecutionSourceContinuationCleanup || !scope.caller[0] || !scope.caller[1] ||
        !scope.group || !scope.ordinary_sequence || (scope.flags == 0 && scope.cleanup_parent) ||
        (scope.flags == kExecutionSourceContinuationCleanup && !scope.cleanup_parent) ||
        inner < static_cast<std::uint32_t>(ExecutionOpcode::kHello) ||
        (inner >= static_cast<std::uint32_t>(ExecutionOpcode::kNcclCall) &&
         inner != static_cast<std::uint32_t>(ExecutionOpcode::kEnqueueHostFunction) &&
         inner != static_cast<std::uint32_t>(ExecutionOpcode::kSgemv) &&
         inner != static_cast<std::uint32_t>(ExecutionOpcode::kSgemmEx) &&
         inner != static_cast<std::uint32_t>(ExecutionOpcode::kUnifiedCopy) &&
         inner != static_cast<std::uint32_t>(ExecutionOpcode::kLoadModuleJit) &&
         inner != static_cast<std::uint32_t>(ExecutionOpcode::kLtAlgorithmCapability) &&
         inner != static_cast<std::uint32_t>(ExecutionOpcode::kProcessMemory) &&
         inner != static_cast<std::uint32_t>(ExecutionOpcode::kDnnLegacyMetadata) &&
         inner != static_cast<std::uint32_t>(ExecutionOpcode::kDnnLegacyControl) &&
         inner != static_cast<std::uint32_t>(ExecutionOpcode::kDnnLegacyCompute)) ||
        bytes != payload.size() - kExecutionSourceContinuationBytes)
        return std::nullopt;
    const auto effective = static_cast<ExecutionOpcode>(inner);
    const auto body = payload.subspan(kExecutionSourceContinuationBytes);
    if (scope.flags == kExecutionSourceContinuationCleanup &&
        (effective != ExecutionOpcode::kReleaseHostMemory || body.size() != 8 ||
         !read_integer(body)))
        return std::nullopt;
    if (effective == ExecutionOpcode::kHello && body.size() == 8 &&
        read_integer(body.first(4)) == kExecutionNcclRole)
        return std::nullopt;
    return ExecutionRequestView{effective, body, scope};
}

std::optional<std::vector<std::byte>> encode_execution_source_continuation(
    const ExecutionSourceContinuation& continuation, ExecutionOpcode inner,
    std::span<const std::byte> payload, std::size_t ordinary_limit) noexcept {
    if (payload.size() > ordinary_limit ||
        ordinary_limit >
            std::numeric_limits<std::uint32_t>::max() - kExecutionSourceContinuationBytes ||
        ordinary_limit >
            std::numeric_limits<std::size_t>::max() - kExecutionSourceContinuationBytes)
        return std::nullopt;
    ExecutionPayloadWriter writer(ordinary_limit + kExecutionSourceContinuationBytes);
    if (!writer.write_u32(kExecutionSourceContinuationMagic) ||
        !writer.write_u32(continuation.flags) || !writer.write_u64(continuation.caller[0]) ||
        !writer.write_u64(continuation.caller[1]) || !writer.write_u64(continuation.group) ||
        !writer.write_u64(continuation.ordinary_sequence) ||
        !writer.write_u64(continuation.origin_source) ||
        !writer.write_u64(continuation.cleanup_parent) ||
        !writer.write_u32(static_cast<std::uint32_t>(inner)) || !writer.write_blob(payload))
        return std::nullopt;
    auto encoded = writer.take();
    if (!normalize_execution_request(ExecutionOpcode::kSourceContinuation, encoded, ordinary_limit))
        return std::nullopt;
    return encoded;
}

std::optional<ExecutionWireHeader> decode_execution_header(std::span<const std::byte> bytes,
                                                           std::size_t max_payload) noexcept {
    if (bytes.size() != kExecutionHeaderBytes || read_integer(bytes.first(4)) != kMagic ||
        read_integer(bytes.subspan(4, 2)) != 1)
        return std::nullopt;
    const auto kind = read_integer(bytes.subspan(6, 2));
    const auto operation = read_integer(bytes.subspan(8, 4));
    const auto payload = read_integer(bytes.subspan(12, 4));
    const bool is_continuation =
        kind == static_cast<std::uint16_t>(ExecutionMessageKind::kRequest) &&
        operation == static_cast<std::uint32_t>(ExecutionOpcode::kSourceContinuation);
    const auto overhead = is_continuation ? kExecutionSourceContinuationBytes : 0;
    if (kind < 1 || kind > 2 || operation < 1 ||
        operation > static_cast<std::uint32_t>(ExecutionOpcode::kDnnLegacyCompute) ||
        (payload > max_payload && payload - max_payload > overhead))
        return std::nullopt;
    return ExecutionWireHeader{
        static_cast<ExecutionMessageKind>(kind),
        static_cast<ExecutionOpcode>(operation),
        static_cast<std::uint32_t>(payload),
        read_integer(bytes.subspan(16, 8)),
        read_integer(bytes.subspan(24, 8)),
        {read_integer(bytes.subspan(32, 8)), read_integer(bytes.subspan(40, 8))}};
}
std::optional<std::vector<std::byte>> encode_execution_message(const ExecutionWireMessage& message,
                                                               std::size_t max_payload) noexcept {
    const auto overhead = message.header.kind == ExecutionMessageKind::kRequest &&
                                  message.header.operation == ExecutionOpcode::kSourceContinuation
                              ? kExecutionSourceContinuationBytes
                              : 0;
    if ((message.payload.size() > max_payload && message.payload.size() - max_payload > overhead) ||
        message.payload.size() > std::numeric_limits<std::uint32_t>::max() ||
        message.payload.size() > std::numeric_limits<std::size_t>::max() - kExecutionHeaderBytes)
        return std::nullopt;
    try {
        std::vector<std::byte> result;
        result.reserve(kExecutionHeaderBytes + message.payload.size());
        append_integer(result, kMagic, 4);
        append_integer(result, 1, 2);
        append_integer(result, static_cast<std::uint16_t>(message.header.kind), 2);
        append_integer(result, static_cast<std::uint32_t>(message.header.operation), 4);
        append_integer(result, message.payload.size(), 4);
        append_integer(result, message.header.session, 8);
        append_integer(result, message.header.sequence, 8);
        append_integer(result, message.header.incarnation[0], 8);
        append_integer(result, message.header.incarnation[1], 8);
        if (!decode_execution_header(result, max_payload)) return std::nullopt;
        result.insert(result.end(), message.payload.begin(), message.payload.end());
        return result;
    } catch (...) {
        return std::nullopt;
    }
}

ExecutionPayloadWriter::ExecutionPayloadWriter(std::size_t limit) noexcept : limit_(limit) {}
bool ExecutionPayloadWriter::reserve_capacity(std::size_t bytes) noexcept {
    if (bytes > limit_) return false;
    try {
        bytes_.reserve(bytes);
        return true;
    } catch (...) {
        return false;
    }
}
bool ExecutionPayloadWriter::reserve_additional_capacity(std::size_t bytes) noexcept {
    return reserve_extra(bytes);
}
bool ExecutionPayloadWriter::reserve_extra(std::size_t bytes) noexcept {
    if (bytes_.size() > limit_ || bytes > limit_ - bytes_.size()) return false;
    const auto needed = bytes_.size() + bytes;
    if (needed <= bytes_.capacity()) return true;
    const auto grown = bytes_.capacity() > limit_ / 2 ? limit_ : bytes_.capacity() * 2;
    try {
        bytes_.reserve(std::max(needed, std::max(std::min<std::size_t>(64, limit_), grown)));
        return true;
    } catch (...) {
        return false;
    }
}
bool ExecutionPayloadWriter::write_u32(std::uint32_t value) noexcept {
    if (bytes_.size() > limit_ || limit_ - bytes_.size() < 4) return false;
    try {
        if (!reserve_extra(4)) return false;
        append_integer(bytes_, value, 4);
        return true;
    } catch (...) {
        return false;
    }
}
bool ExecutionPayloadWriter::write_u64(std::uint64_t value) noexcept {
    if (bytes_.size() > limit_ || limit_ - bytes_.size() < 8) return false;
    try {
        if (!reserve_extra(8)) return false;
        append_integer(bytes_, value, 8);
        return true;
    } catch (...) {
        return false;
    }
}
bool ExecutionPayloadWriter::write_blob(std::span<const std::byte> bytes) noexcept {
    if (bytes.size() > std::numeric_limits<std::uint32_t>::max() || bytes_.size() > limit_ ||
        limit_ - bytes_.size() < 4 || bytes.size() > limit_ - bytes_.size() - 4)
        return false;
    try {
        if (!reserve_extra(4 + bytes.size())) return false;
        append_integer(bytes_, bytes.size(), 4);
        if (!bytes.empty()) bytes_.insert(bytes_.end(), bytes.begin(), bytes.end());
        return true;
    } catch (...) {
        return false;
    }
}
bool ExecutionPayloadWriter::write_string(std::string_view value) noexcept {
    return write_blob(std::as_bytes(std::span{value.data(), value.size()}));
}
std::vector<std::byte> ExecutionPayloadWriter::take() noexcept {
    return std::move(bytes_);
}
ExecutionPayloadReader::ExecutionPayloadReader(std::span<const std::byte> bytes) noexcept
    : bytes_(bytes) {}
bool ExecutionPayloadReader::read_u32(std::uint32_t& value) noexcept {
    if (bytes_.size() - position_ < 4) return false;
    value = static_cast<std::uint32_t>(read_integer(bytes_.subspan(position_, 4)));
    position_ += 4;
    return true;
}
bool ExecutionPayloadReader::read_u64(std::uint64_t& value) noexcept {
    if (bytes_.size() - position_ < 8) return false;
    value = read_integer(bytes_.subspan(position_, 8));
    position_ += 8;
    return true;
}
bool ExecutionPayloadReader::read_blob(std::span<const std::byte>& value) noexcept {
    std::uint32_t length{};
    if (!read_u32(length) || length > bytes_.size() - position_) return false;
    value = bytes_.subspan(position_, length);
    position_ += length;
    return true;
}
bool ExecutionPayloadReader::read_string(std::string_view& value) noexcept {
    std::span<const std::byte> bytes;
    if (!read_blob(bytes)) return false;
    value = std::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    return true;
}
bool ExecutionPayloadReader::empty() const noexcept {
    return position_ == bytes_.size();
}

std::optional<std::array<std::byte, kExecutionContextBootstrapBytes>>
encode_execution_context_bootstrap(ExecutionContextBootstrap descriptor) noexcept {
    const auto purpose = static_cast<std::uint32_t>(descriptor.purpose);
    if (purpose < static_cast<std::uint32_t>(ExecutionContextPurpose::kExplicitUser) ||
        purpose > static_cast<std::uint32_t>(ExecutionContextPurpose::kRuntimePrimary) ||
        !descriptor.source_generation)
        return std::nullopt;
    std::array<std::byte, kExecutionContextBootstrapBytes> result{};
    const std::array<std::uint32_t, 4> prefix{kExecutionContextRole, purpose, descriptor.flags, 0};
    for (std::size_t word = 0; word < prefix.size(); ++word)
        for (std::size_t byte = 0; byte < 4; ++byte)
            result[word * 4 + byte] =
                std::byte{static_cast<unsigned char>(prefix[word] >> (byte * 8))};
    for (std::size_t byte = 0; byte < 8; ++byte)
        result[16 + byte] =
            std::byte{static_cast<unsigned char>(descriptor.source_generation >> (byte * 8))};
    return result;
}

std::optional<ExecutionContextBootstrap> decode_execution_context_bootstrap(
    std::span<const std::byte> bytes) noexcept {
    if (bytes.empty()) return ExecutionContextBootstrap{};
    if (bytes.size() == 4)
        return ExecutionContextBootstrap{ExecutionContextPurpose::kLegacyIsolated,
                                         static_cast<std::uint32_t>(read_integer(bytes)), 0};
    if (bytes.size() != kExecutionContextBootstrapBytes ||
        read_integer(bytes.first(4)) != kExecutionContextRole || read_integer(bytes.subspan(12, 4)))
        return std::nullopt;
    const auto purpose = read_integer(bytes.subspan(4, 4));
    const auto generation = read_integer(bytes.subspan(16, 8));
    if (purpose < static_cast<std::uint32_t>(ExecutionContextPurpose::kExplicitUser) ||
        purpose > static_cast<std::uint32_t>(ExecutionContextPurpose::kRuntimePrimary) ||
        !generation)
        return std::nullopt;
    return ExecutionContextBootstrap{static_cast<ExecutionContextPurpose>(purpose),
                                     static_cast<std::uint32_t>(read_integer(bytes.subspan(8, 4))),
                                     generation};
}

std::optional<std::array<std::byte, 8>> encode_execution_bootstrap_ownership(
    ExecutionBootstrapOwnership ownership) noexcept {
    const auto value = static_cast<std::uint32_t>(ownership);
    if (value > static_cast<std::uint32_t>(ExecutionBootstrapOwnership::kMetadataRetained))
        return std::nullopt;
    std::array<std::byte, 8> result{std::byte{'G'}, std::byte{'B'}, std::byte{'O'}, std::byte{'1'}};
    for (std::size_t index = 0; index < 4; ++index)
        result[4 + index] = std::byte{static_cast<unsigned char>(value >> (8 * index))};
    return result;
}
std::optional<ExecutionBootstrapOwnership> decode_execution_bootstrap_ownership(
    std::span<const std::byte> bytes) noexcept {
    if (bytes.size() != 8 || read_integer(bytes.first(4)) != 0x314f4247)
        return std::nullopt;  // GBO1, explicit little-endian versioned record.
    const auto value = read_integer(bytes.subspan(4, 4));
    return value <= static_cast<std::uint32_t>(ExecutionBootstrapOwnership::kMetadataRetained)
               ? std::optional{static_cast<ExecutionBootstrapOwnership>(value)}
               : std::nullopt;
}

std::optional<std::vector<std::byte>> encode_execution_reply(const ExecutionWireReply& reply,
                                                             std::size_t limit) noexcept {
    if (reply.arguments.size() > 4096 ||
        (reply.disposition != ExecutionWireDisposition::kComplete &&
         reply.disposition != ExecutionWireDisposition::kRejected &&
         reply.disposition != ExecutionWireDisposition::kComputeReplacement &&
         reply.disposition != ExecutionWireDisposition::kAdmissionAccepted &&
         reply.disposition != ExecutionWireDisposition::kAdmissionBusy) ||
        (reply.disposition == ExecutionWireDisposition::kAdmissionBusy &&
         (reply.status || reply.original_status || reply.id || reply.address ||
          reply.parameter_bytes || !reply.arguments.empty() || !reply.data.empty())) ||
        (reply.disposition == ExecutionWireDisposition::kComputeReplacement &&
         (reply.original_status || !reply.id)) ||
        (reply.disposition == ExecutionWireDisposition::kAdmissionAccepted &&
         (reply.status || reply.original_status || !reply.id || !reply.address ||
          reply.parameter_bytes || !reply.arguments.empty() || !reply.data.empty())))
        return std::nullopt;
    std::uint64_t previous_end{};
    for (const auto& argument : reply.arguments) {
        if (!valid_reply_argument(argument, reply.parameter_bytes, previous_end))
            return std::nullopt;
        previous_end = argument.offset + argument.bytes;
    }
    ExecutionPayloadWriter output(limit);
    if (!output.write_u32(static_cast<std::uint32_t>(reply.disposition)) ||
        !output.write_u32(std::bit_cast<std::uint32_t>(reply.status)) ||
        !output.write_u32(reply.original_status ? 1 : 0) ||
        !output.write_u32(
            reply.original_status ? std::bit_cast<std::uint32_t>(*reply.original_status) : 0) ||
        !output.write_u64(reply.id) || !output.write_u64(reply.address) ||
        !output.write_u64(reply.parameter_bytes) ||
        !output.write_u32(static_cast<std::uint32_t>(reply.arguments.size())))
        return std::nullopt;
    for (const auto& argument : reply.arguments)
        if (!output.write_u64(argument.offset) || !output.write_u64(argument.bytes))
            return std::nullopt;
    if (!output.write_blob(reply.data)) return std::nullopt;
    return output.take();
}
std::optional<ExecutionWireReply> decode_execution_reply(
    std::span<const std::byte> bytes) noexcept {
    try {
        ExecutionPayloadReader input(bytes);
        ExecutionWireReply result;
        std::uint32_t disposition{}, status{}, has_original{}, original{}, count{};
        if (!input.read_u32(disposition) || disposition < 1 || disposition > 5 ||
            !input.read_u32(status) || !input.read_u32(has_original) || has_original > 1 ||
            !input.read_u32(original) || (!has_original && original) ||
            !input.read_u64(result.id) || !input.read_u64(result.address) ||
            !input.read_u64(result.parameter_bytes) || !input.read_u32(count) || count > 4096)
            return std::nullopt;
        result.disposition = static_cast<ExecutionWireDisposition>(disposition);
        if (result.disposition == ExecutionWireDisposition::kAdmissionBusy &&
            (status || has_original || result.id || result.address || result.parameter_bytes ||
             count))
            return std::nullopt;
        if (result.disposition == ExecutionWireDisposition::kComputeReplacement &&
            (has_original || !result.id))
            return std::nullopt;
        result.status = std::bit_cast<std::int32_t>(status);
        if (has_original) result.original_status = std::bit_cast<std::int32_t>(original);
        result.arguments.reserve(count);
        std::uint64_t previous_end{};
        for (std::uint32_t index = 0; index < count; ++index) {
            ExecutionWireArgument argument;
            if (!input.read_u64(argument.offset) || !input.read_u64(argument.bytes) ||
                !valid_reply_argument(argument, result.parameter_bytes, previous_end))
                return std::nullopt;
            previous_end = argument.offset + argument.bytes;
            result.arguments.push_back(argument);
        }
        std::span<const std::byte> data;
        if (!input.read_blob(data) || !input.empty()) return std::nullopt;
        if (result.disposition == ExecutionWireDisposition::kAdmissionBusy && !data.empty())
            return std::nullopt;
        if (result.disposition == ExecutionWireDisposition::kAdmissionAccepted &&
            (result.status || result.original_status || !result.id || !result.address ||
             result.parameter_bytes || !result.arguments.empty() || !data.empty()))
            return std::nullopt;
        if (!data.empty()) result.data.assign(data.begin(), data.end());
        return result;
    } catch (...) {
        return std::nullopt;
    }
}
}  // namespace glimmer::control
