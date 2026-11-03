#include "glimmer/control/execution_pointer_protocol.h"
#include "glimmer/control/execution_process_memory_protocol.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace {
using namespace glimmer::control;
static_assert(static_cast<std::uint32_t>(ExecutionPointerDomain::kDriver) == 1);
static_assert(static_cast<std::uint32_t>(ExecutionPointerDomain::kRuntime) == 2);
static_assert(static_cast<std::uint32_t>(ExecutionPointerKind::kNull) == 0);
static_assert(static_cast<std::uint32_t>(ExecutionPointerKind::kUnregistered) == 1);
static_assert(static_cast<std::uint32_t>(ExecutionPointerKind::kDevice) == 2);
static_assert(static_cast<std::uint32_t>(ExecutionPointerKind::kHost) == 3);
static_assert(static_cast<std::uint32_t>(ExecutionPointerKind::kProcessDevice) == 4);
static_assert(static_cast<std::uint32_t>(ExecutionPointerAlias::kNull) == 0);
static_assert(static_cast<std::uint32_t>(ExecutionPointerAlias::kInput) == 1);
void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
bool same_cursor(ExecutionPayloadReader first, ExecutionPayloadReader second) {
    for (;;) {
        std::uint32_t left{}, right{};
        const bool has_left = first.read_u32(left);
        const bool has_right = second.read_u32(right);
        if (has_left != has_right || left != right) return false;
        if (!has_left) return first.empty() == second.empty();
    }
}
template <std::size_t Size>
std::vector<std::byte> bytes(const std::array<unsigned char, Size>& values) {
    std::vector<std::byte> result;
    result.reserve(Size);
    for (const auto value : values) result.push_back(static_cast<std::byte>(value));
    return result;
}
ExecutionPointerQuery query() {
    return {ExecutionPointerDomain::kDriver,
            9,
            ExecutionPointerKind::kDevice,
            0x0102030405060708ULL,
            0x1112131415161718ULL,
            UINT64_MAX,
            0x8182838485868788ULL};
}
std::vector<std::byte> query_bytes() {
    return bytes(std::array<unsigned char, 48>{
        1,    0,    0,    0,    9,    0,    0,    0,    2,    0,    0,    0,
        0,    0,    0,    0,    8,    7,    6,    5,    4,    3,    2,    1,
        0x18, 0x17, 0x16, 0x15, 0x14, 0x13, 0x12, 0x11, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xff, 0x88, 0x87, 0x86, 0x85, 0x84, 0x83, 0x82, 0x81});
}
ExecutionPointerObservation observation() {
    return {15, 0, -2, ExecutionPointerAlias::kNull, ExecutionPointerAlias::kInput};
}
std::vector<std::byte> observation_bytes() {
    return bytes(std::array<unsigned char, 20>{15,   0,    0, 0, 0, 0, 0, 0, 0xfe, 0xff,
                                               0xff, 0xff, 0, 0, 0, 0, 1, 0, 0,    0});
}
template <class Value, class Write, class Read>
void boundaries(const Value& expected, const std::vector<std::byte>& literal, Write write,
                Read read) {
    ExecutionPayloadWriter exact(literal.size());
    expect(write(exact, expected) && exact.take() == literal,
           "encoder agrees with independent literal little-endian bytes");
    Value sentinel{};
    ExecutionPayloadReader full(literal);
    auto decoded = sentinel;
    expect(read(full, decoded) && full.empty() && decoded == expected,
           "independent literal decodes exactly, without host padding");
    for (std::size_t length = 0; length < literal.size(); ++length) {
        ExecutionPayloadReader truncated{std::span{literal}.first(length)};
        const auto before = truncated;
        decoded = expected;
        expect(!read(truncated, decoded) && decoded == expected && same_cursor(truncated, before),
               "every truncated record preserves destination and reader position");
        ExecutionPayloadWriter short_writer(length);
        expect(!write(short_writer, expected) && short_writer.take().empty(),
               "every insufficient capacity preserves an empty writer");
        ExecutionPayloadWriter prefixed(length + 4);
        expect(prefixed.write_u32(0x76543210) && !write(prefixed, expected) &&
                   prefixed.take() == bytes(std::array<unsigned char, 4>{0x10, 0x32, 0x54, 0x76}),
               "preflight includes the existing writer prefix and never partially appends");
    }
    ExecutionPayloadWriter composed(literal.size() + 12);
    expect(composed.write_u32(17) && write(composed, expected) && composed.write_u64(UINT64_MAX),
           "record composes with independently framed prefix and suffix");
    const auto encoded = composed.take();
    ExecutionPayloadReader input(encoded);
    std::uint32_t prefix{};
    std::uint64_t suffix{};
    expect(input.read_u32(prefix) && prefix == 17 && read(input, decoded) && decoded == expected &&
               !input.empty() && input.read_u64(suffix) && suffix == UINT64_MAX && input.empty(),
           "decoder consumes only its exact record, leaving trailing validation to enclosing API");
}
template <class Value, class Write, class Read>
void invalid(const Value& value, const std::vector<std::byte>& wire, Write write, Read read) {
    ExecutionPayloadWriter output;
    expect(output.write_u32(17) && !write(output, value) &&
               output.take() == bytes(std::array<unsigned char, 4>{17, 0, 0, 0}),
           "invalid canonical value cannot alter a writer prefix");
    ExecutionPayloadReader input(wire);
    const auto before = input;
    Value sentinel{};
    auto observed = sentinel;
    expect(!read(input, observed) && observed == sentinel && same_cursor(input, before),
           "malformed record cannot change reader position or output");
}
void process_query_controls() {
    for (const auto pair :
         std::array<std::array<std::uint32_t, 2>, 4>{{{1, 1}, {1, 3}, {1, 9}, {2, 0}}}) {
        auto value = query();
        value.domain = static_cast<ExecutionPointerDomain>(pair[0]);
        value.selector = pair[1];
        value.kind = ExecutionPointerKind::kProcessDevice;
        auto wire = query_bytes();
        wire[0] = static_cast<std::byte>(pair[0]);
        wire[4] = static_cast<std::byte>(pair[1]);
        wire[8] = std::byte{4};
        boundaries(value, wire, write_execution_pointer_query, read_execution_pointer_query);
        for (const auto bad_id :
             {std::uint64_t{0}, std::uint64_t{1}, kExecutionProcessAllocationTag,
              kExecutionProcessAllocationTag - 1}) {
            auto bad_value = value;
            bad_value.allocation = bad_id;
            auto malformed = wire;
            for (unsigned int index = 0; index < 8; ++index)
                malformed[32 + index] =
                    std::byte{static_cast<unsigned char>(bad_id >> (index * 8))};
            invalid(bad_value, malformed, write_execution_pointer_query,
                    read_execution_pointer_query);
        }
        auto no_source = value;
        no_source.source_session = 0;
        auto no_source_wire = wire;
        for (std::size_t index = 24; index < 32; ++index) no_source_wire[index] = std::byte{};
        invalid(no_source, no_source_wire, write_execution_pointer_query,
                read_execution_pointer_query);
    }
    auto local_context = query();
    local_context.selector = 1;
    auto local_wire = query_bytes();
    local_wire[4] = std::byte{1};
    invalid(local_context, local_wire, write_execution_pointer_query, read_execution_pointer_query);
    const ExecutionPointerObservation null_context{4, 0, 0, ExecutionPointerAlias::kNull,
                                                   ExecutionPointerAlias::kNull};
    boundaries(null_context, bytes(std::array<unsigned char, 20>{4}),
               write_execution_pointer_observation, read_execution_pointer_observation);
}
void query_controls() {
    static_assert(kExecutionPointerQueryBytes == 48);
    boundaries(query(), query_bytes(), write_execution_pointer_query, read_execution_pointer_query);
    // Exhaust all admitted domain/selector/kind forms independently of the codec.
    for (const auto pair : std::array<std::array<std::uint32_t, 2>, 3>{{{1, 3}, {1, 9}, {2, 0}}}) {
        for (std::uint32_t kind = 0; kind != 4; ++kind) {
            auto value = query();
            value.domain = static_cast<ExecutionPointerDomain>(pair[0]);
            value.selector = pair[1];
            value.kind = static_cast<ExecutionPointerKind>(kind);
            value.query_session = 0;
            if (kind < 2)
                value.source_session = value.allocation = value.offset = 0;
            else
                value.offset = UINT64_MAX;
            ExecutionPayloadWriter output;
            expect(write_execution_pointer_query(output, value),
                   "all twelve structural forms encode");
            const auto encoded = output.take();
            ExecutionPayloadReader input(encoded);
            ExecutionPointerQuery decoded;
            expect(
                read_execution_pointer_query(input, decoded) && input.empty() && decoded == value,
                "codec preserves no-current and full-width offset; authority is separate");
        }
    }
    for (unsigned int test = 0; test != 10; ++test) {
        auto value = query();
        auto wire = query_bytes();
        switch (test) {
            case 0:
                value.domain = static_cast<ExecutionPointerDomain>(0);
                wire[0] = std::byte{};
                break;
            case 1:
                value.domain = static_cast<ExecutionPointerDomain>(3);
                wire[0] = std::byte{3};
                break;
            case 2:
                value.selector = 8;
                wire[4] = std::byte{8};
                break;
            case 3:
                value.domain = ExecutionPointerDomain::kRuntime;
                wire[0] = std::byte{2};
                break;
            case 4:
                value.kind = static_cast<ExecutionPointerKind>(5);
                wire[8] = std::byte{5};
                break;
            case 5:
                value.kind = ExecutionPointerKind::kNull;
                wire[8] = std::byte{};
                break;
            case 6:
                value.kind = ExecutionPointerKind::kUnregistered;
                wire[8] = std::byte{1};
                break;
            case 7:
                value.source_session = 0;
                for (std::size_t index = 24; index != 32; ++index) wire[index] = std::byte{};
                break;
            case 8:
                value.allocation = 0;
                for (std::size_t index = 32; index != 40; ++index) wire[index] = std::byte{};
                break;
            case 9:
                value.selector = UINT32_MAX;
                for (std::size_t index = 4; index != 8; ++index) wire[index] = std::byte{0xff};
                break;
            default:
                throw std::runtime_error("unexpected query corruption case");
        }
        invalid(value, wire, write_execution_pointer_query, read_execution_pointer_query);
    }
    auto wire = query_bytes();
    wire[12] = std::byte{1};
    ExecutionPayloadReader input(wire);
    const auto before = input;
    auto value = query();
    expect(!read_execution_pointer_query(input, value) && same_cursor(input, before) &&
               value == query(),
           "reserved field is exactly zero, not an ignored future discriminator");
    for (unsigned int field = 0; field != 3; ++field) {
        ExecutionPointerQuery value_with_hidden_id;
        value_with_hidden_id.kind = ExecutionPointerKind::kUnregistered;
        if (field == 0) value_with_hidden_id.source_session = 1;
        if (field == 1) value_with_hidden_id.allocation = 1;
        if (field == 2) value_with_hidden_id.offset = 1;
        ExecutionPayloadWriter output;
        expect(
            !write_execution_pointer_query(output, value_with_hidden_id) && output.take().empty(),
            "unregistered requests cannot smuggle any single authority or caller pointer field");
    }
}
void observation_controls() {
    static_assert(kExecutionPointerObservationBytes == 20);
    boundaries(observation(), observation_bytes(), write_execution_pointer_observation,
               read_execution_pointer_observation);
    boundaries(ExecutionPointerObservation{3, INT32_MIN, INT32_MAX},
               bytes(std::array<unsigned char, 20>{3,    0,    0, 0, 0, 0, 0, 0x80, 0xff, 0xff,
                                                   0xff, 0x7f, 0, 0, 0, 0, 0, 0,    0,    0}),
               write_execution_pointer_observation, read_execution_pointer_observation);
    boundaries(
        ExecutionPointerObservation{3, -1, -17},
        bytes(std::array<unsigned char, 20>{3,    0,    0, 0, 0xff, 0xff, 0xff, 0xff, 0xef, 0xff,
                                            0xff, 0xff, 0, 0, 0,    0,    0,    0,    0,    0}),
        write_execution_pointer_observation, read_execution_pointer_observation);
    for (std::uint32_t mask = 0; mask != 16; ++mask) {
        for (std::int32_t type = 0; type != 4; ++type) {
            for (const std::int32_t device : std::array{-2, 0, INT32_MAX}) {
                ExecutionPointerObservation value{
                    mask, (mask & 1U) ? type : 0, (mask & 2U) ? device : 0,
                    (mask & 4U) ? ExecutionPointerAlias::kInput : ExecutionPointerAlias::kNull,
                    (mask & 8U) ? ExecutionPointerAlias::kInput : ExecutionPointerAlias::kNull};
                ExecutionPayloadWriter output;
                expect(write_execution_pointer_observation(output, value),
                       "every independent effect mask and bounded scalar value encodes");
                const auto wire = output.take();
                ExecutionPayloadReader input(wire);
                ExecutionPointerObservation decoded;
                expect(read_execution_pointer_observation(input, decoded) && decoded == value &&
                           input.empty(),
                       "all192 structural effect records round trip");
            }
        }
    }
    for (unsigned int test = 0; test != 12; ++test) {
        auto value = observation();
        auto wire = observation_bytes();
        switch (test) {
            case 0:
                value.effects = 16;
                wire[0] = std::byte{16};
                break;
            case 1:
                value.effects = 14;
                wire[0] = std::byte{14};
                value.type = -1;
                for (std::size_t i = 4; i != 8; ++i) wire[i] = std::byte{0xff};
                break;
            case 2:
                value.effects = 14;
                wire[0] = std::byte{14};
                value.type = 4;
                wire[4] = std::byte{4};
                break;
            case 3:
                value.effects = 13;
                wire[0] = std::byte{13};
                value.device = -1;
                wire[8] = std::byte{0xff};
                break;
            case 4:
                value.effects = 13;
                wire[0] = std::byte{13};
                value.device = -3;
                wire[8] = std::byte{0xfd};
                break;
            case 5:
                value.device_alias = static_cast<ExecutionPointerAlias>(2);
                wire[12] = std::byte{2};
                break;
            case 6:
                value.host_alias = static_cast<ExecutionPointerAlias>(2);
                wire[16] = std::byte{2};
                break;
            case 7:
                value.effects = 13;
                wire[0] = std::byte{13};
                break;
            case 8:
                value.effects = 7;
                wire[0] = std::byte{7};
                break;
            case 9:
                value.effects = 14;
                value.type = 1;
                wire[0] = std::byte{14};
                wire[4] = std::byte{1};
                break;
            case 10:
                value.effects = 11;
                value.device_alias = ExecutionPointerAlias::kInput;
                wire[0] = std::byte{11};
                wire[12] = std::byte{1};
                break;
            case 11:
                value.effects = 13;
                wire[0] = std::byte{13};
                value.device = INT32_MIN;
                wire[8] = wire[9] = wire[10] = std::byte{};
                wire[11] = std::byte{0x80};
                break;
            default:
                throw std::runtime_error("unexpected observation corruption case");
        }
        invalid(value, wire, write_execution_pointer_observation,
                read_execution_pointer_observation);
    }
}
}  // namespace
int main() {
    try {
        query_controls();
        process_query_controls();
        observation_controls();
        std::cout
            << "Pointer protocol literal, composition, canonical and transactional checks passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
