#include "glimmer/control/execution_nccl_protocol.h"

#include <array>
#include <bit>
#include <cstdlib>
#include <iostream>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace {
using namespace glimmer::control;
static_assert(static_cast<std::uint32_t>(ExecutionNcclOperation::kGetUniqueId) == 1);
static_assert(static_cast<std::uint32_t>(ExecutionNcclOperation::kInitRank) == 2);
static_assert(static_cast<std::uint32_t>(ExecutionNcclOperation::kInitRankConfig) == 3);
static_assert(static_cast<std::uint32_t>(ExecutionNcclOperation::kGetAsyncError) == 4);
static_assert(static_cast<std::uint32_t>(ExecutionNcclOperation::kDestroy) == 5);
static_assert(static_cast<std::uint32_t>(ExecutionNcclOperation::kAbort) == 6);
static_assert(static_cast<std::uint32_t>(ExecutionNcclOperation::kGroupStart) == 7);
static_assert(static_cast<std::uint32_t>(ExecutionNcclOperation::kGroupEnd) == 8);
static_assert(static_cast<std::uint32_t>(ExecutionNcclOperation::kAllGather) == 9);
static_assert(static_cast<std::uint32_t>(ExecutionNcclOperation::kAllReduce) == 10);
static_assert(static_cast<std::uint32_t>(ExecutionNcclOperation::kBcast) == 11);
static_assert(static_cast<std::uint32_t>(ExecutionNcclStreamKind::kDefault) == 0);
static_assert(static_cast<std::uint32_t>(ExecutionNcclStreamKind::kOwned) == 1);
static_assert(static_cast<std::uint32_t>(ExecutionNcclStreamKind::kLegacy) == 2);
static_assert(static_cast<std::uint32_t>(ExecutionNcclStreamKind::kPerThread) == 3);
static_assert(static_cast<std::uint32_t>(ExecutionNcclBufferKind::kNull) == 0);
static_assert(static_cast<std::uint32_t>(ExecutionNcclBufferKind::kOwned) == 1);
static_assert(static_cast<std::uint32_t>(ExecutionNcclBufferKind::kUnresolved) == 2);
static_assert(static_cast<std::uint32_t>(ExecutionNcclContextEffectKind::kNone) == 0);
static_assert(static_cast<std::uint32_t>(ExecutionNcclContextEffectKind::kSelectPrimary) == 1);
static_assert(static_cast<std::uint32_t>(ExecutionNcclContextEffectKind::kClearCurrent) == 2);
using Bytes = std::vector<std::byte>;
void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template <std::size_t Width>
void append(Bytes& bytes, std::uint64_t value) {
    static_assert(Width == 4 || Width == 8);
    bytes.reserve(bytes.size() + Width);
    for (std::size_t index = 0; index != Width; ++index)
        bytes.push_back(static_cast<std::byte>((value >> (index * 8)) & 255U));
}
template <std::size_t Width>
void replace(Bytes& bytes, std::size_t offset, const std::array<std::byte, Width>& replacement) {
    for (std::size_t index = 0; index != Width; ++index)
        bytes.at(offset + index) = replacement[index];
}
template <std::size_t Width>
std::array<std::byte, Width> little(std::uint64_t value) {
    std::array<std::byte, Width> result{};
    for (std::size_t index = 0; index != Width; ++index)
        result[index] = static_cast<std::byte>((value >> (index * 8)) & 255U);
    return result;
}
bool same_cursor(ExecutionPayloadReader left, ExecutionPayloadReader right) {
    for (;;) {
        std::uint32_t first{}, second{};
        const bool has_first = left.read_u32(first);
        const bool has_second = right.read_u32(second);
        if (has_first != has_second || first != second) return false;
        if (!has_first) return left.empty() == right.empty();
    }
}
ExecutionNcclScope scope() {
    return {{0x0102030405060708ULL, 0x1112131415161718ULL},
            UINT64_MAX,
            0x2122232425262728ULL,
            0x3132333435363738ULL};
}
Bytes prefix(std::uint32_t operation) {
    Bytes result;
    result.reserve(48);
    append<4>(result, operation);
    append<4>(result, 0);
    // Literal scope bytes deliberately do not use the production scope writer.
    constexpr std::array<unsigned char, 40> literal{
        8,    7,    6,    5,    4,    3,    2,    1,    0x18, 0x17, 0x16, 0x15, 0x14, 0x13,
        0x12, 0x11, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x28, 0x27, 0x26, 0x25,
        0x24, 0x23, 0x22, 0x21, 0x38, 0x37, 0x36, 0x35, 0x34, 0x33, 0x32, 0x31};
    for (const auto byte : literal) result.push_back(static_cast<std::byte>(byte));
    return result;
}
ExecutionNcclInitRank initialization() {
    ExecutionNcclInitRank value{true, INT32_MIN, INT32_MAX, {}};
    for (std::size_t index = 0; index != value.unique_id.size(); ++index)
        value.unique_id[index] = static_cast<std::byte>(index);
    return value;
}
Bytes initialization_wire(std::uint32_t operation) {
    auto result = prefix(operation);
    result.reserve(188);
    append<4>(result, 1);
    append<4>(result, 0x80000000U);
    append<4>(result, 0x7fffffffU);
    for (unsigned int index = 0; index != 128; ++index)
        result.push_back(static_cast<std::byte>(index));
    return result;
}
ExecutionNcclAllGather gather() {
    return {{41, 43},
            {ExecutionNcclStreamKind::kOwned, 47, 53},
            {ExecutionNcclBufferKind::kOwned, 59, 61, UINT64_MAX},
            {ExecutionNcclBufferKind::kOwned, 67, 71, 73},
            UINT64_MAX,
            INT32_MIN};
}
Bytes gather_wire(std::uint32_t operation) {
    auto result = prefix(operation);
    result.reserve(164);
    append<8>(result, 41);
    append<8>(result, 43);
    append<4>(result, 1);
    append<4>(result, 0);
    append<8>(result, 47);
    append<8>(result, 53);
    append<4>(result, 1);
    append<4>(result, 0);
    append<8>(result, 59);
    append<8>(result, 61);
    append<8>(result, UINT64_MAX);
    append<4>(result, 1);
    append<4>(result, 0);
    append<8>(result, 67);
    append<8>(result, 71);
    append<8>(result, 73);
    append<8>(result, UINT64_MAX);
    append<4>(result, 0x80000000U);
    return result;
}
ExecutionNcclRequest request_sentinel() {
    ExecutionNcclConfig config;
    config.declared_bytes = 88;
    config.net_name = "destination must remain unchanged";
    return {scope(), ExecutionNcclInitRankConfig{initialization(), config}};
}
ExecutionNcclObservation observation_sentinel() {
    ExecutionNcclObservation value;
    value.effects = 2;
    value.communicator = {93, 97};
    return value;
}
template <class Value>
struct Record {
    Value value;
    Bytes wire;
};
template <class Value, class Write, class Read>
void boundaries(const Record<Value>& record, const Value& sentinel, Write write, Read read,
                bool exhaustive = true) {
    ExecutionPayloadWriter exact(record.wire.size());
    expect(write(exact, record.value) && exact.take() == record.wire,
           "independent literal encoding has exact bytes, widths and field order");
    auto decoded = sentinel;
    ExecutionPayloadReader complete(record.wire);
    expect(read(complete, decoded) && complete.empty() && decoded == record.value,
           "literal decodes exactly, without a round-trip-derived oracle");
    const auto failure = [&](std::size_t length) {
        ExecutionPayloadReader short_input{std::span{record.wire}.first(length)};
        const auto before = short_input;
        decoded = sentinel;
        expect(
            !read(short_input, decoded) && decoded == sentinel && same_cursor(short_input, before),
            "truncation preserves both cursor and preexisting output");
        ExecutionPayloadWriter short_output(length);
        expect(!write(short_output, record.value) && short_output.take().empty(),
               "capacity failure appends no partial record");
        ExecutionPayloadWriter prefixed(length + 4);
        expect(prefixed.write_u32(0x76543210) && !write(prefixed, record.value) &&
                   prefixed.take() ==
                       Bytes{std::byte{0x10}, std::byte{0x32}, std::byte{0x54}, std::byte{0x76}},
               "whole-record preflight accounts for an existing writer prefix");
    };
    if (exhaustive) {
        for (std::size_t length = 0; length != record.wire.size(); ++length) failure(length);
    } else {
        for (const auto length :
             std::array<std::size_t, 4>{0, 47, record.wire.size() / 2, record.wire.size() - 1})
            failure(length);
    }
    ExecutionPayloadWriter composed(record.wire.size() + 12);
    expect(
        composed.write_u32(17) && write(composed, record.value) && composed.write_u64(UINT64_MAX),
        "records compose in an enclosing prefix and suffix");
    const auto encoded = composed.take();
    ExecutionPayloadReader input(encoded);
    std::uint32_t first{};
    std::uint64_t last{};
    expect(input.read_u32(first) && first == 17 && read(input, decoded) &&
               decoded == record.value && !input.empty() && input.read_u64(last) &&
               last == UINT64_MAX && input.empty(),
           "successful decode consumes only one exact record, not its enclosing suffix");
}
template <class Value, class Read>
void reject_wire(const Bytes& wire, const Value& sentinel, Read read) {
    auto actual = sentinel;
    ExecutionPayloadReader input(wire);
    const auto before = input;
    expect(!read(input, actual) && actual == sentinel && same_cursor(input, before),
           "hostile bytes preserve destination and reader position");
}
template <class Value, class Write>
void reject_value(const Value& value, Write write) {
    ExecutionPayloadWriter output;
    expect(output.write_u32(19) && !write(output, value) &&
               output.take() == Bytes{std::byte{19}, std::byte{}, std::byte{}, std::byte{}},
           "invalid value cannot mutate even the envelope prefix");
}
void reject_request(const Record<ExecutionNcclRequest>& record) {
    reject_value(record.value, write_execution_nccl_request);
    reject_wire(record.wire, request_sentinel(), read_execution_nccl_request);
}

std::vector<Record<ExecutionNcclRequest>> literals() {
    std::vector<Record<ExecutionNcclRequest>> result;
    result.reserve(11);
    auto wire = prefix(1);
    append<4>(wire, 1);
    result.push_back({{scope(), ExecutionNcclGetUniqueId{true}}, std::move(wire)});
    result.push_back({{scope(), initialization()}, initialization_wire(2)});
    ExecutionNcclConfig config;
    config.declared_bytes = 88;
    config.magic = 0x12345678U;
    config.version = UINT32_MAX;
    config.fields = {INT32_MIN, -1, 0, INT32_MAX, -7, 2, 3, 4, 5, 6, 7, 8, 9, -17};
    config.net_name = std::string{"ib"} + static_cast<char>(0xff);
    config.comm_name = "";
    wire = initialization_wire(3);
    append<4>(wire, 1);
    append<8>(wire, 88);
    append<4>(wire, 0x12345678U);
    append<4>(wire, UINT32_MAX);
    for (const auto bits :
         std::array<std::uint32_t, 14>{0x80000000U, 0xffffffffU, 0, 0x7fffffffU, 0xfffffff9U, 2, 3,
                                       4, 5, 6, 7, 8, 9, 0xffffffefU})
        append<4>(wire, bits);
    append<4>(wire, 1);
    append<4>(wire, 3);
    wire.insert(wire.end(), {std::byte{'i'}, std::byte{'b'}, std::byte{0xff}});
    append<4>(wire, 1);
    append<4>(wire, 0);
    result.push_back(
        {{scope(), ExecutionNcclInitRankConfig{initialization(), config}}, std::move(wire)});
    wire = prefix(4);
    append<8>(wire, 41);
    append<8>(wire, 43);
    append<4>(wire, 0);
    result.push_back({{scope(), ExecutionNcclGetAsyncError{{41, 43}, false}}, std::move(wire)});
    wire = prefix(5);
    append<8>(wire, 0);
    append<8>(wire, 0);
    result.push_back({{scope(), ExecutionNcclDestroy{}}, std::move(wire)});
    wire = prefix(6);
    append<8>(wire, 41);
    append<8>(wire, 43);
    result.push_back({{scope(), ExecutionNcclAbort{{41, 43}}}, std::move(wire)});
    result.push_back({{scope(), ExecutionNcclGroupStart{}}, prefix(7)});
    result.push_back({{scope(), ExecutionNcclGroupEnd{}}, prefix(8)});
    result.push_back({{scope(), gather()}, gather_wire(9)});
    wire = gather_wire(10);
    append<4>(wire, 0x7fffffffU);
    result.push_back({{scope(), ExecutionNcclAllReduce{gather(), INT32_MAX}}, std::move(wire)});
    wire = prefix(11);
    append<8>(wire, 0);
    append<8>(wire, 0);
    append<4>(wire, 3);
    append<4>(wire, 0);
    append<8>(wire, 0);
    append<8>(wire, 0);
    append<4>(wire, 2);
    append<4>(wire, 0);
    append<8>(wire, 0);
    append<8>(wire, 0);
    append<8>(wire, 0);
    append<8>(wire, UINT64_MAX);
    append<4>(wire, UINT32_MAX);
    append<4>(wire, 0x80000000U);
    result.push_back({{scope(), ExecutionNcclBcast{{},
                                                   {ExecutionNcclStreamKind::kPerThread, 0, 0},
                                                   {ExecutionNcclBufferKind::kUnresolved, 0, 0, 0},
                                                   UINT64_MAX,
                                                   -1,
                                                   INT32_MIN}},
                      std::move(wire)});
    return result;
}
void operation_literals() {
    static_assert(kExecutionNcclUniqueIdBytes == 128 && kExecutionNcclScopeBytes == 40);
    const auto records = literals();
    constexpr std::array<std::size_t, 11> lengths{52, 188, 283, 68, 64, 64, 48, 48, 164, 168, 136};
    expect(records.size() == 11,
           "all eleven implemented-reference API request variants have literals");
    for (std::size_t index = 0; index != records.size(); ++index) {
        expect(records[index].wire.size() == lengths[index],
               "literal request has independent fixed or variable length");
        boundaries(records[index], request_sentinel(), write_execution_nccl_request,
                   read_execution_nccl_request);
    }
    ExecutionPayloadWriter joined;
    for (const auto& record : records)
        expect(write_execution_nccl_request(joined, record.value),
               "append heterogeneous request sequence");
    const auto all = joined.take();
    ExecutionPayloadReader input(all);
    for (const auto& record : records) {
        auto value = request_sentinel();
        expect(read_execution_nccl_request(input, value) && value == record.value,
               "operation discriminator determines exact body, not caller destination variant");
    }
    expect(input.empty(), "all eleven composed requests consume exactly their own bytes");
}
void request_structure() {
    const auto records = literals();
    for (const auto bad : std::array<std::uint32_t, 3>{0, 12, UINT32_MAX}) {
        auto wire = prefix(bad);
        reject_wire(wire, request_sentinel(), read_execution_nccl_request);
    }
    for (const auto bad : std::array<std::uint32_t, 2>{1, UINT32_MAX}) {
        auto wire = records[0].wire;
        replace(wire, 4, little<4>(bad));
        reject_wire(wire, request_sentinel(), read_execution_nccl_request);
    }
    for (const bool missing_thread : {false, true}) {
        auto record = records[0];
        if (missing_thread) {
            record.value.scope.thread = {};
            replace(record.wire, 8, little<8>(0));
            replace(record.wire, 16, little<8>(0));
        } else {
            record.value.scope.sequence = 0;
            replace(record.wire, 24, little<8>(0));
        }
        reject_request(record);
    }
    for (const bool high_only : {false, true}) {
        auto record = records[7];
        record.value.scope = {{high_only ? 0U : 1U, high_only ? 1U : 0U}, 1, 0, 0};
        replace(record.wire, 8, little<8>(high_only ? 0 : 1));
        replace(record.wire, 16, little<8>(high_only ? 1 : 0));
        replace(record.wire, 24, little<8>(1));
        replace(record.wire, 32, little<8>(0));
        replace(record.wire, 40, little<8>(0));
        boundaries(record, request_sentinel(), write_execution_nccl_request,
                   read_execution_nccl_request);
    }
    for (const auto index : std::array<std::size_t, 4>{0, 1, 2, 3}) {
        auto wire = records[index].wire;
        replace(wire, index == 3 ? 64 : 48, little<4>(2));
        reject_wire(wire, request_sentinel(), read_execution_nccl_request);
        auto changed = records[index];
        std::visit(
            [](auto& call) {
                using Call = std::decay_t<decltype(call)>;
                if constexpr (std::is_same_v<Call, ExecutionNcclInitRankConfig>)
                    call.initialization.has_output = false;
                else if constexpr (requires { call.has_output; })
                    call.has_output = !call.has_output;
            },
            changed.value.call);
        replace(changed.wire, index == 3 ? 64 : 48, little<4>(index == 3 ? 1 : 0));
        boundaries(changed, request_sentinel(), write_execution_nccl_request,
                   read_execution_nccl_request);
    }
    for (const auto index : std::array<std::size_t, 6>{3, 4, 5, 8, 9, 10}) {
        for (const bool missing_domain : {false, true}) {
            auto record = records[index];
            const ExecutionNcclCommunicator invalid{missing_domain ? 0U : 1U,
                                                    missing_domain ? 1U : 0U};
            std::visit(
                [&](auto& call) {
                    using Call = std::decay_t<decltype(call)>;
                    if constexpr (std::is_same_v<Call, ExecutionNcclAllReduce>)
                        call.collective.communicator = invalid;
                    else if constexpr (requires { call.communicator; })
                        call.communicator = invalid;
                },
                record.value.call);
            replace(record.wire, 48, little<8>(invalid.domain));
            replace(record.wire, 56, little<8>(invalid.object));
            reject_request(record);
        }
    }
    for (std::uint32_t kind = 0; kind != 4; ++kind) {
        auto record = records[8];
        auto& stream = std::get<ExecutionNcclAllGather>(record.value.call).stream;
        stream = {static_cast<ExecutionNcclStreamKind>(kind), kind == 1 ? 47U : 0U,
                  kind == 1 ? 53U : 0U};
        replace(record.wire, 64, little<4>(kind));
        replace(record.wire, 72, little<8>(stream.owner));
        replace(record.wire, 80, little<8>(stream.object));
        boundaries(record, request_sentinel(), write_execution_nccl_request,
                   read_execution_nccl_request);
        for (const bool first : {false, true}) {
            auto invalid = record;
            auto& changed = std::get<ExecutionNcclAllGather>(invalid.value.call).stream;
            auto& field = first ? changed.owner : changed.object;
            field = kind == 1 ? 0 : 1;
            replace(invalid.wire, first ? 72 : 80, little<8>(field));
            reject_request(invalid);
        }
    }
    for (std::uint32_t kind = 0; kind != 3; ++kind) {
        auto record = records[8];
        auto& buffer = std::get<ExecutionNcclAllGather>(record.value.call).send;
        buffer = {static_cast<ExecutionNcclBufferKind>(kind), kind == 1 ? 59U : 0U,
                  kind == 1 ? 61U : 0U, kind == 1 ? UINT64_MAX : 0};
        replace(record.wire, 88, little<4>(kind));
        replace(record.wire, 96, little<8>(buffer.source));
        replace(record.wire, 104, little<8>(buffer.allocation));
        replace(record.wire, 112, little<8>(buffer.offset));
        boundaries(record, request_sentinel(), write_execution_nccl_request,
                   read_execution_nccl_request);
        for (unsigned int field = 0; field != (kind == 1 ? 2U : 3U); ++field) {
            auto invalid = record;
            auto& changed = std::get<ExecutionNcclAllGather>(invalid.value.call).send;
            auto& target = field == 0   ? changed.source
                           : field == 1 ? changed.allocation
                                        : changed.offset;
            target = kind == 1 ? 0 : 1;
            replace(invalid.wire, 96 + field * 8, little<8>(target));
            reject_request(invalid);
        }
    }
    for (const auto offset : std::array<std::size_t, 3>{68, 92, 124}) {
        auto wire = records[8].wire;
        replace(wire, offset, little<4>(1));
        reject_wire(wire, request_sentinel(), read_execution_nccl_request);
    }
    for (const auto invalid_kind : std::array<std::uint32_t, 2>{4, UINT32_MAX}) {
        auto record = records[8];
        std::get<ExecutionNcclAllGather>(record.value.call).stream.kind =
            static_cast<ExecutionNcclStreamKind>(invalid_kind);
        replace(record.wire, 64, little<4>(invalid_kind));
        reject_request(record);
    }
    for (const auto offset : std::array<std::size_t, 2>{88, 120}) {
        auto record = records[8];
        auto& call = std::get<ExecutionNcclAllGather>(record.value.call);
        (offset == 88 ? call.send : call.receive).kind = static_cast<ExecutionNcclBufferKind>(3);
        replace(record.wire, offset, little<4>(3));
        reject_request(record);
    }
}

void raw_scalar_bits() {
    const auto records = literals();
    for (const auto bits : std::array<std::uint32_t, 5>{0, 7, 12, 0x80000000U, UINT32_MAX}) {
        const auto signed_value = std::bit_cast<std::int32_t>(bits);
        auto init = records[1];
        auto& initialization_value = std::get<ExecutionNcclInitRank>(init.value.call);
        initialization_value.ranks = initialization_value.rank = signed_value;
        replace(init.wire, 52, little<4>(bits));
        replace(init.wire, 56, little<4>(bits));
        boundaries(init, request_sentinel(), write_execution_nccl_request,
                   read_execution_nccl_request);
        auto reduce = records[9];
        auto& reduction = std::get<ExecutionNcclAllReduce>(reduce.value.call);
        reduction.collective.count = 0;
        reduction.collective.datatype = reduction.reduction = signed_value;
        replace(reduce.wire, 152, little<8>(0));
        replace(reduce.wire, 160, little<4>(bits));
        replace(reduce.wire, 164, little<4>(bits));
        boundaries(reduce, request_sentinel(), write_execution_nccl_request,
                   read_execution_nccl_request);
        auto broadcast = records[10];
        auto& bcast = std::get<ExecutionNcclBcast>(broadcast.value.call);
        bcast.count = std::uint64_t{1} << 63;
        bcast.datatype = bcast.root = signed_value;
        replace(broadcast.wire, 120, little<8>(std::uint64_t{1} << 63));
        replace(broadcast.wire, 128, little<4>(bits));
        replace(broadcast.wire, 132, little<4>(bits));
        boundaries(broadcast, request_sentinel(), write_execution_nccl_request,
                   read_execution_nccl_request);
    }
}

Record<ExecutionNcclRequest> config_record(std::uint64_t declared) {
    // Construct an independent native-prefix byte oracle, then project only its
    // scalar positions. No production config validator or serializer is used.
    std::array<std::byte, 88> native{};
    for (std::size_t index = 8; index != native.size(); ++index)
        if (index < declared && !(index >= 32 && index < 40) && !(index >= 48 && index < 56))
            native[index] = static_cast<std::byte>(0x80U + static_cast<unsigned int>(index));
    const auto scalar = [&](std::size_t offset) {
        std::uint32_t bits{};
        for (std::size_t index = 0; index != 4; ++index)
            bits |= std::to_integer<std::uint32_t>(native[offset + index]) << (index * 8);
        return bits;
    };
    ExecutionNcclConfig config;
    config.declared_bytes = declared;
    config.magic = scalar(8);
    config.version = scalar(12);
    auto wire = initialization_wire(3);
    wire.reserve(272);
    append<4>(wire, 1);
    append<8>(wire, declared);
    wire.insert(wire.end(), native.begin() + 8, native.begin() + 16);
    constexpr std::array<std::size_t, 14> offsets{16, 20, 24, 28, 40, 44, 56,
                                                  60, 64, 68, 72, 76, 80, 84};
    for (std::size_t index = 0; index != offsets.size(); ++index) {
        config.fields[index] = std::bit_cast<std::int32_t>(scalar(offsets[index]));
        wire.insert(wire.end(), native.begin() + static_cast<std::ptrdiff_t>(offsets[index]),
                    native.begin() + static_cast<std::ptrdiff_t>(offsets[index] + 4));
    }
    append<4>(wire, 0);
    append<4>(wire, 0);
    return {{scope(), ExecutionNcclInitRankConfig{initialization(), config}}, std::move(wire)};
}
void config_prefixes() {
    auto absent = initialization_wire(3);
    append<4>(absent, 0);
    boundaries(
        Record<ExecutionNcclRequest>{
            {scope(), ExecutionNcclInitRankConfig{initialization(), std::nullopt}}, absent},
        request_sentinel(), write_execution_nccl_request, read_execution_nccl_request);
    for (std::uint64_t declared = 0; declared <= 96; ++declared) {
        auto record = config_record(declared);
        if ((declared >= 33 && declared <= 39) || (declared >= 49 && declared <= 55))
            reject_request(record);
        else
            boundaries(record, request_sentinel(), write_execution_nccl_request,
                       read_execution_nccl_request);
    }
    boundaries(config_record(UINT64_MAX), request_sentinel(), write_execution_nccl_request,
               read_execution_nccl_request);
    // Every missing scalar byte must be zero, including short magic/version.
    constexpr std::array<std::size_t, 16> native_offsets{8,  12, 16, 20, 24, 28, 40, 44,
                                                         56, 60, 64, 68, 72, 76, 80, 84};
    for (std::size_t field = 0; field != native_offsets.size(); ++field) {
        for (std::size_t prefix_bytes = 0; prefix_bytes != 4; ++prefix_bytes) {
            auto record = config_record(native_offsets[field] + prefix_bytes);
            auto& maybe_config = std::get<ExecutionNcclInitRankConfig>(record.value.call).config;
            if (!maybe_config) throw std::runtime_error("config fixture must be present");
            auto& config = *maybe_config;
            const auto forbidden = std::uint32_t{1} << (prefix_bytes * 8);
            if (field == 0)
                config.magic |= forbidden;
            else if (field == 1)
                config.version |= forbidden;
            else
                config.fields[field - 2] = std::bit_cast<std::int32_t>(
                    std::bit_cast<std::uint32_t>(config.fields[field - 2]) | forbidden);
            record.wire[200 + field * 4 + prefix_bytes] |= std::byte{1};
            reject_request(record);
        }
    }
    auto bad_presence = absent;
    replace(bad_presence, 188, little<4>(2));
    reject_wire(bad_presence, request_sentinel(), read_execution_nccl_request);
}
void config_strings() {
    for (unsigned int net = 0; net != 3; ++net) {
        for (unsigned int comm = 0; comm != 3; ++comm) {
            auto record = config_record(88);
            auto& config = *std::get<ExecutionNcclInitRankConfig>(record.value.call).config;
            record.wire.resize(264);
            record.wire.reserve(292);
            for (const bool is_net : {true, false}) {
                const auto mode = is_net ? net : comm;
                auto& value = is_net ? config.net_name : config.comm_name;
                append<4>(record.wire, mode ? 1U : 0U);
                if (mode) {
                    value = mode == 1 ? "" : "opaque";
                    append<4>(record.wire, mode == 1 ? 0 : 6);
                    if (mode == 2)
                        for (const auto byte : std::array{'o', 'p', 'a', 'q', 'u', 'e'})
                            record.wire.push_back(static_cast<std::byte>(byte));
                }
            }
            boundaries(record, request_sentinel(), write_execution_nccl_request,
                       read_execution_nccl_request);
        }
    }
    for (const bool is_net : {true, false}) {
        auto record = config_record(is_net ? 32 : 48);
        auto& config = *std::get<ExecutionNcclInitRankConfig>(record.value.call).config;
        (is_net ? config.net_name : config.comm_name) = "";
        const auto offset = is_net ? 264U : 268U;
        replace(record.wire, offset, little<4>(1));
        record.wire.insert(record.wire.begin() + offset + 4, 4, std::byte{});
        reject_request(record);
    }
    for (const auto offset : std::array<std::size_t, 2>{264, 268}) {
        auto record = config_record(88);
        replace(record.wire, offset, little<4>(2));
        reject_wire(record.wire, request_sentinel(), read_execution_nccl_request);
    }
    for (const auto lengths : std::array<std::array<std::size_t, 2>, 3>{
             {{65535, 65535}, {65536, 65535}, {65535, 65536}}}) {
        auto record = config_record(UINT64_MAX);
        auto& config = *std::get<ExecutionNcclInitRankConfig>(record.value.call).config;
        config.net_name = std::string(lengths[0], 'n');
        config.comm_name = std::string(lengths[1], 'c');
        record.wire.resize(264);
        record.wire.reserve(280 + lengths[0] + lengths[1]);
        append<4>(record.wire, 1);
        append<4>(record.wire, lengths[0]);
        record.wire.insert(record.wire.end(), lengths[0], std::byte{'n'});
        append<4>(record.wire, 1);
        append<4>(record.wire, lengths[1]);
        record.wire.insert(record.wire.end(), lengths[1], std::byte{'c'});
        if (lengths[0] == 65535 && lengths[1] == 65535)
            boundaries(record, request_sentinel(), write_execution_nccl_request,
                       read_execution_nccl_request, false);
        else
            reject_request(record);
    }
    for (const bool is_net : {true, false}) {
        auto record = config_record(88);
        auto& config = *std::get<ExecutionNcclInitRankConfig>(record.value.call).config;
        (is_net ? config.net_name : config.comm_name) = std::string("a\0b", 3);
        record.wire.resize(264);
        if (!is_net) append<4>(record.wire, 0);
        append<4>(record.wire, 1);
        append<4>(record.wire, 3);
        record.wire.insert(record.wire.end(), {std::byte{'a'}, std::byte{}, std::byte{'b'}});
        if (is_net) append<4>(record.wire, 0);
        reject_request(record);
    }
    auto giant_length = config_record(88).wire;
    giant_length.resize(264);
    append<4>(giant_length, 1);
    append<4>(giant_length, UINT32_MAX);
    reject_wire(giant_length, request_sentinel(), read_execution_nccl_request);
}

Record<ExecutionNcclObservation> observation_literal() {
    ExecutionNcclObservation value;
    value.effects = 7;
    value.unique_id_mask[0] = std::byte{0x81};
    value.unique_id_mask[15] = std::byte{0x80};
    value.unique_id[0] = std::byte{0x11};
    value.unique_id[127] = std::byte{0xff};
    value.communicator = {81, 83};
    value.async_error_mask = 10;
    value.async_error = std::bit_cast<std::int32_t>(0x8000ff00U);
    Bytes wire(172);
    wire[0] = std::byte{7};
    wire[4] = std::byte{0x81};
    wire[19] = std::byte{0x80};
    wire[20] = std::byte{0x11};
    wire[147] = std::byte{0xff};
    wire[148] = std::byte{81};
    wire[156] = std::byte{83};
    wire[164] = std::byte{10};
    wire[169] = std::byte{0xff};
    wire[171] = std::byte{0x80};
    return {value, std::move(wire)};
}
void reject_observation(const Record<ExecutionNcclObservation>& record) {
    reject_value(record.value, write_execution_nccl_observation);
    reject_wire(record.wire, observation_sentinel(), read_execution_nccl_observation);
}
void observation_masks() {
    static_assert(kExecutionNcclObservationBytes == 172);
    boundaries(observation_literal(), observation_sentinel(), write_execution_nccl_observation,
               read_execution_nccl_observation);
    boundaries(Record<ExecutionNcclObservation>{{}, Bytes(172)}, observation_sentinel(),
               write_execution_nccl_observation, read_execution_nccl_observation);
    for (const auto bits : std::array<std::uint32_t, 2>{0x80000000U, 0x7fffffffU}) {
        ExecutionNcclObservation value;
        Bytes wire(172);
        value.effects = 4;
        wire[0] = std::byte{4};
        value.async_error_mask = 15;
        wire[164] = std::byte{15};
        value.async_error = std::bit_cast<std::int32_t>(bits);
        replace(wire, 168, little<4>(bits));
        boundaries(Record<ExecutionNcclObservation>{value, wire}, observation_sentinel(),
                   write_execution_nccl_observation, read_execution_nccl_observation);
    }
    {
        ExecutionNcclObservation value;
        Bytes wire(172);
        value.effects = 1;
        wire[0] = std::byte{1};
        value.unique_id_mask.fill(std::byte{0xff});
        for (std::size_t index = 0; index != 16; ++index) wire[4 + index] = std::byte{0xff};
        for (std::size_t index = 0; index != 128; ++index) {
            value.unique_id[index] = static_cast<std::byte>(index);
            wire[20 + index] = static_cast<std::byte>(index);
        }
        boundaries(Record<ExecutionNcclObservation>{value, wire}, observation_sentinel(),
                   write_execution_nccl_observation, read_execution_nccl_observation);
    }
    for (unsigned int uid = 0; uid != 2; ++uid) {
        for (unsigned int comm = 0; comm != 2; ++comm) {
            for (std::uint32_t mask = 0; mask != 16; ++mask) {
                ExecutionNcclObservation value;
                Bytes wire(172);
                value.effects = uid + 2 * comm + (mask ? 4U : 0U);
                wire[0] = static_cast<std::byte>(value.effects);
                if (uid) {
                    value.unique_id_mask[0] = std::byte{1};
                    wire[4] = std::byte{1};
                }
                // Active communicator effect with 0/0 is an actual NULL write.
                value.async_error_mask = mask;
                wire[164] = static_cast<std::byte>(mask);
                std::uint32_t bits{};
                for (unsigned int byte = 0; byte != 4; ++byte)
                    if (mask & (1U << byte)) {
                        bits |= 0xffU << (byte * 8);
                        wire[168 + byte] = std::byte{0xff};
                    }
                value.async_error = std::bit_cast<std::int32_t>(bits);
                boundaries(Record<ExecutionNcclObservation>{value, wire}, observation_sentinel(),
                           write_execution_nccl_observation, read_execution_nccl_observation);
                if (!uid && !comm) {
                    for (unsigned int byte = 0; byte != 4; ++byte) {
                        if (mask & (1U << byte)) continue;
                        auto invalid = Record<ExecutionNcclObservation>{value, wire};
                        invalid.value.async_error =
                            std::bit_cast<std::int32_t>(bits | (1U << (byte * 8)));
                        invalid.wire[168 + byte] = std::byte{1};
                        reject_observation(invalid);
                    }
                }
            }
        }
    }
    for (std::size_t index = 0; index != 128; ++index) {
        for (const auto payload : std::array<std::byte, 2>{std::byte{}, std::byte{0xa5}}) {
            ExecutionNcclObservation value;
            Bytes wire(172);
            value.effects = 1;
            wire[0] = std::byte{1};
            value.unique_id_mask[index / 8] = static_cast<std::byte>(1U << (index % 8));
            wire[4 + index / 8] = value.unique_id_mask[index / 8];
            value.unique_id[index] = payload;
            wire[20 + index] = payload;
            boundaries(Record<ExecutionNcclObservation>{value, wire}, observation_sentinel(),
                       write_execution_nccl_observation, read_execution_nccl_observation, false);
            const auto inactive = (index + 1) % 128;
            value.unique_id[inactive] = std::byte{1};
            wire[20 + inactive] = std::byte{1};
            reject_observation({value, wire});
        }
    }
    for (unsigned int test = 0; test != 11; ++test) {
        auto record = observation_literal();
        switch (test) {
            case 0:
                record.value.effects = 8;
                record.wire[0] = std::byte{8};
                break;
            case 1:
                record.value.effects = 6;
                record.wire[0] = std::byte{6};
                break;
            case 2:
                record.value.effects = 5;
                record.wire[0] = std::byte{5};
                break;
            case 3:
                record.value.effects = 3;
                record.wire[0] = std::byte{3};
                break;
            case 4:
                record.value.unique_id_mask = {};
                for (std::size_t index = 4; index != 20; ++index) record.wire[index] = std::byte{};
                break;
            case 5:
                record.value.communicator.domain = 0;
                record.wire[148] = std::byte{};
                break;
            case 6:
                record.value.communicator.object = 0;
                record.wire[156] = std::byte{};
                break;
            case 7:
                record.value.async_error_mask = 0;
                record.wire[164] = std::byte{};
                break;
            case 8:
                record.value.async_error_mask = 16;
                record.wire[164] = std::byte{16};
                break;
            case 9:
                record.value.async_error = std::bit_cast<std::int32_t>(0x8000ff01U);
                record.wire[168] = std::byte{1};
                break;
            case 10:
                record.value.unique_id[1] = std::byte{1};
                record.wire[21] = std::byte{1};
                break;
            default:
                throw std::runtime_error("unexpected observation corruption case");
        }
        reject_observation(record);
    }
    for (const bool uid : {false, true}) {
        ExecutionNcclObservation value;
        Bytes wire(172);
        value.effects = uid ? 1U : 4U;
        wire[0] = static_cast<std::byte>(value.effects);
        reject_observation({value, wire});
    }
}

Bytes reply_prefix(bool is_accepted) {
    // Literal NCR1/version1/reserved0/accepted followed by zero-valued fields.
    // Do not derive offsets, sizes or magic from production codec constants.
    Bytes wire(240);
    wire[0] = std::byte{'N'};
    wire[1] = std::byte{'C'};
    wire[2] = std::byte{'R'};
    wire[3] = std::byte{'1'};
    wire[4] = std::byte{1};
    wire[12] = is_accepted ? std::byte{1} : std::byte{};
    return wire;
}
Record<ExecutionNcclReply> reply_literal() {
    const auto observed = observation_literal();
    ExecutionNcclReply value;
    value.is_accepted = true;
    value.operation = 0x0102030405060708ULL;
    value.reserved_communicator = {71, 73};
    value.original_status = INT32_MIN;
    value.observation = observed.value;
    value.context_effect = {ExecutionNcclContextEffectKind::kSelectPrimary, 7,
                            0x1112131415161718ULL};
    value.publications = {{scope(), observed.value}, {{{1, 0}, 3, 0, 0}, {}}};
    auto wire = reply_prefix(true);
    replace(wire, 16, little<8>(0x0102030405060708ULL));
    replace(wire, 24, little<8>(71));
    replace(wire, 32, little<8>(73));
    replace(wire, 40, little<4>(1));
    replace(wire, 44, little<4>(0x80000000U));
    for (std::size_t index = 0; index != 172; ++index) wire[48 + index] = observed.wire[index];
    replace(wire, 220, little<4>(1));
    replace(wire, 224, little<4>(7));
    replace(wire, 228, little<8>(0x1112131415161718ULL));
    replace(wire, 236, little<4>(2));
    const auto scope_bytes = prefix(0);
    wire.insert(wire.end(), scope_bytes.begin() + 8, scope_bytes.end());
    wire.insert(wire.end(), observed.wire.begin(), observed.wire.end());
    for (const auto field : std::array<std::uint64_t, 5>{1, 0, 3, 0, 0}) append<8>(wire, field);
    wire.resize(wire.size() + 172);
    return {std::move(value), std::move(wire)};
}
ExecutionNcclReply reply_sentinel() {
    auto value = reply_literal().value;
    value.operation = 83;
    value.publications.front().scope.sequence = 97;
    return value;
}
void reject_reply(const Record<ExecutionNcclReply>& record) {
    reject_value(record.value, write_execution_nccl_reply);
    reject_wire(record.wire, reply_sentinel(), read_execution_nccl_reply);
}
void reply_literals() {
    static_assert(kExecutionNcclReplyMagic == 0x3152434eU && kExecutionNcclReplyVersion == 1);
    static_assert(kExecutionNcclReplyPrefixBytes == 240 && kExecutionNcclPublicationBytes == 212);
    static_assert(kExecutionNcclPublicationLimit == 256);
    boundaries(Record<ExecutionNcclReply>{{}, reply_prefix(false)}, reply_sentinel(),
               write_execution_nccl_reply, read_execution_nccl_reply);
    ExecutionNcclReply accepted;
    accepted.is_accepted = true;
    accepted.operation = 1;
    auto wire = reply_prefix(true);
    replace(wire, 16, little<8>(1));
    boundaries(Record<ExecutionNcclReply>{accepted, wire}, reply_sentinel(),
               write_execution_nccl_reply, read_execution_nccl_reply);
    boundaries(reply_literal(), reply_sentinel(), write_execution_nccl_reply,
               read_execution_nccl_reply);
    for (const auto bits :
         std::array<std::uint32_t, 5>{0, 7, 0xffffffffU, 0x80000000U, 0x7fffffffU}) {
        auto record = reply_literal();
        record.value.original_status = std::bit_cast<std::int32_t>(bits);
        replace(record.wire, 44, little<4>(bits));
        boundaries(record, reply_sentinel(), write_execution_nccl_reply, read_execution_nccl_reply);
    }
    {
        // Missing requested status and retained representation are independent;
        // the enclosing owner, not this codec, proves a carrier failure/effect.
        auto record = reply_literal();
        record.value.original_status.reset();
        replace(record.wire, 40, little<4>(0));
        replace(record.wire, 44, little<4>(0));
        boundaries(record, reply_sentinel(), write_execution_nccl_reply, read_execution_nccl_reply);
    }
    for (const auto device : std::array<std::int32_t, 2>{0, INT32_MAX}) {
        auto record = reply_literal();
        record.value.context_effect.device = device;
        record.value.context_effect.domain = UINT64_MAX;
        replace(record.wire, 224, little<4>(static_cast<std::uint32_t>(device)));
        replace(record.wire, 228, little<8>(UINT64_MAX));
        boundaries(record, reply_sentinel(), write_execution_nccl_reply, read_execution_nccl_reply);
    }
    for (const auto kind :
         {ExecutionNcclContextEffectKind::kNone, ExecutionNcclContextEffectKind::kClearCurrent}) {
        auto record = reply_literal();
        record.value.reserved_communicator = {};
        record.value.context_effect = {kind, 0, 0};
        replace(record.wire, 24, little<8>(0));
        replace(record.wire, 32, little<8>(0));
        replace(record.wire, 220, little<4>(static_cast<std::uint32_t>(kind)));
        replace(record.wire, 224, little<4>(0));
        replace(record.wire, 228, little<8>(0));
        boundaries(record, reply_sentinel(), write_execution_nccl_reply, read_execution_nccl_reply);
    }
    {
        // Preserve ordered scope identities, even duplicates; the public actor
        // must reject duplicate publication authority after decoding.
        auto record = reply_literal();
        record.value.publications[1].scope = record.value.publications[0].scope;
        for (std::size_t index = 0; index != 40; ++index)
            record.wire[452 + index] = record.wire[240 + index];
        boundaries(record, reply_sentinel(), write_execution_nccl_reply, read_execution_nccl_reply);
    }
}
void reply_structure() {
    for (const auto offset : std::array<std::size_t, 3>{0, 4, 8}) {
        auto wire = reply_literal().wire;
        replace(wire, offset, little<4>(0xffffffffU));
        reject_wire(wire, reply_sentinel(), read_execution_nccl_reply);
    }
    for (const auto offset : std::array<std::size_t, 2>{12, 40}) {
        auto wire = reply_literal().wire;
        replace(wire, offset, little<4>(2));
        reject_wire(wire, reply_sentinel(), read_execution_nccl_reply);
    }
    {
        auto wire = reply_literal().wire;
        replace(wire, 40, little<4>(0));
        reject_wire(wire, reply_sentinel(), read_execution_nccl_reply);
    }
    for (unsigned int index = 0; index != 11; ++index) {
        auto record = reply_literal();
        switch (index) {
            case 0:
                record.value.operation = 0;
                replace(record.wire, 16, little<8>(0));
                break;
            case 1:
                record.value.reserved_communicator.domain = 0;
                replace(record.wire, 24, little<8>(0));
                break;
            case 2:
                record.value.reserved_communicator.object = 0;
                replace(record.wire, 32, little<8>(0));
                break;
            case 3:
                record.value.context_effect.kind = static_cast<ExecutionNcclContextEffectKind>(3);
                replace(record.wire, 220, little<4>(3));
                break;
            case 4:
                record.value.context_effect.device = -1;
                replace(record.wire, 224, little<4>(0xffffffffU));
                break;
            case 5:
                record.value.context_effect.domain = 0;
                replace(record.wire, 228, little<8>(0));
                break;
            case 6:
                record.value.context_effect = {ExecutionNcclContextEffectKind::kNone, 0, 9};
                replace(record.wire, 220, little<4>(0));
                replace(record.wire, 224, little<4>(0));
                replace(record.wire, 228, little<8>(9));
                break;
            case 7:
                record.value.context_effect = {ExecutionNcclContextEffectKind::kNone, 1, 0};
                replace(record.wire, 220, little<4>(0));
                replace(record.wire, 224, little<4>(1));
                replace(record.wire, 228, little<8>(0));
                break;
            case 8:
                record.value.observation.effects = 8;
                replace(record.wire, 48, little<4>(8));
                break;
            case 9:
                record.value.context_effect = {ExecutionNcclContextEffectKind::kClearCurrent, 0, 1};
                replace(record.wire, 220, little<4>(2));
                replace(record.wire, 224, little<4>(0));
                replace(record.wire, 228, little<8>(1));
                break;
            case 10:
                record.value.context_effect = {ExecutionNcclContextEffectKind::kClearCurrent, -1,
                                               0};
                replace(record.wire, 220, little<4>(2));
                replace(record.wire, 224, little<4>(0xffffffffU));
                replace(record.wire, 228, little<8>(0));
                break;
            default:
                throw std::runtime_error("unexpected accepted reply corruption case");
        }
        reject_reply(record);
    }
    for (unsigned int index = 0; index != 7; ++index) {
        Record<ExecutionNcclReply> record{{}, reply_prefix(false)};
        switch (index) {
            case 0:
                record.value.operation = 1;
                replace(record.wire, 16, little<8>(1));
                break;
            case 1:
                record.value.reserved_communicator = {1, 2};
                replace(record.wire, 24, little<8>(1));
                replace(record.wire, 32, little<8>(2));
                break;
            case 2:
                record.value.original_status = 0;
                replace(record.wire, 40, little<4>(1));
                break;
            case 3:
                record.value.observation.effects = 2;  // Even a NULL write is an effect.
                replace(record.wire, 48, little<4>(2));
                break;
            case 4:
                record.value.context_effect = {ExecutionNcclContextEffectKind::kSelectPrimary, 0,
                                               1};
                replace(record.wire, 220, little<4>(1));
                replace(record.wire, 228, little<8>(1));
                break;
            case 5: {
                record.value.publications.push_back({scope(), {}});
                replace(record.wire, 236, little<4>(1));
                const auto scope_bytes = prefix(0);
                record.wire.insert(record.wire.end(), scope_bytes.begin() + 8, scope_bytes.end());
                record.wire.resize(record.wire.size() + 172);
                break;
            }
            case 6:
                record.value.context_effect = {ExecutionNcclContextEffectKind::kClearCurrent, 0, 0};
                replace(record.wire, 220, little<4>(2));
                break;
            default:
                throw std::runtime_error("unexpected rejected reply corruption case");
        }
        reject_reply(record);
    }
    // Corrupt every scope identity field with a structural obligation, at both
    // the first and the final publication. The entire writer must stay atomic.
    for (std::size_t member = 0; member != 2; ++member) {
        for (unsigned int invalid = 0; invalid != 4; ++invalid) {
            auto record = reply_literal();
            auto& publication = record.value.publications[member];
            const auto offset = 240 + 212 * member;
            if (invalid == 0) {
                publication.scope.thread = {};
                replace(record.wire, offset, little<8>(0));
                replace(record.wire, offset + 8, little<8>(0));
            } else if (invalid == 1) {
                publication.scope.sequence = 0;
                replace(record.wire, offset + 16, little<8>(0));
            } else if (invalid == 2) {
                publication.observation.effects = 8;
                replace(record.wire, offset + 40, little<4>(8));
            } else {
                publication.observation.unique_id[1] = std::byte{1};
                record.wire[offset + 40 + 21] = std::byte{1};
            }
            reject_reply(record);
        }
    }
    // A claimed array cannot borrow a short enclosing suffix as a full entry.
    auto missing = reply_literal().wire;
    missing.resize(240);
    reject_wire(missing, reply_sentinel(), read_execution_nccl_reply);
}
void reply_quota() {
    Record<ExecutionNcclReply> record{{}, reply_prefix(true)};
    record.value.is_accepted = true;
    record.value.operation = UINT64_MAX;
    replace(record.wire, 16, little<8>(UINT64_MAX));
    replace(record.wire, 236, little<4>(256));
    record.value.publications.reserve(256);
    for (std::uint64_t index = 1; index <= 256; ++index) {
        const ExecutionNcclScope original{{0, 1}, index, UINT64_MAX, 0};
        record.value.publications.push_back({original, {}});
        for (const auto field : std::array<std::uint64_t, 5>{0, 1, index, UINT64_MAX, 0})
            append<8>(record.wire, field);
        record.wire.resize(record.wire.size() + 172);
    }
    expect(record.wire.size() == 54512,
           "maximum reply has an independently checked aggregate size");
    boundaries(record, reply_sentinel(), write_execution_nccl_reply, read_execution_nccl_reply,
               false);
    {
        auto last_invalid = record;
        last_invalid.value.publications.back().scope.sequence = 0;
        replace(last_invalid.wire, 240 + 255 * 212 + 16, little<8>(0));
        reject_reply(last_invalid);
    }
    record.value.publications.push_back({{{1, 0}, 257, 0, 0}, {}});
    replace(record.wire, 236, little<4>(257));
    for (const auto field : std::array<std::uint64_t, 5>{1, 0, 257, 0, 0})
        append<8>(record.wire, field);
    record.wire.resize(record.wire.size() + 172);
    reject_reply(record);
    for (const auto count : std::array<std::uint32_t, 2>{257, UINT32_MAX}) {
        auto claimed = reply_literal().wire;
        claimed.resize(240);
        replace(claimed, 236, little<4>(count));
        reject_wire(claimed, reply_sentinel(), read_execution_nccl_reply);
    }
}
}  // namespace

int main() {
    try {
        operation_literals();
        request_structure();
        raw_scalar_bits();
        config_prefixes();
        config_strings();
        observation_masks();
        reply_literals();
        reply_structure();
        reply_quota();
        std::cout << "NCCL wire-only request/config/effect/reply contracts passed; native support "
                     "unqualified\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
