#include "glimmer/control/execution_dnn_backend_protocol.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <span>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace {
using namespace glimmer::control;
static_assert(static_cast<std::uint32_t>(ExecutionDnnBackendValueClass::kInvalid) == 0);
static_assert(static_cast<std::uint32_t>(ExecutionDnnBackendValueClass::kScalar) == 1);
static_assert(static_cast<std::uint32_t>(ExecutionDnnBackendValueClass::kHandle) == 2);
static_assert(static_cast<std::uint32_t>(ExecutionDnnBackendValueClass::kDescriptor) == 3);
static_assert(static_cast<std::uint32_t>(ExecutionDnnBackendValueClass::kPointerSlot) == 4);
static_assert(static_cast<std::uint32_t>(ExecutionDnnBackendReferenceKind::kNull) == 0);
static_assert(static_cast<std::uint32_t>(ExecutionDnnBackendReferenceKind::kHandle) == 1);
static_assert(static_cast<std::uint32_t>(ExecutionDnnBackendReferenceKind::kDescriptor) == 2);
static_assert(static_cast<std::uint32_t>(ExecutionDnnBackendReferenceKind::kPointerSlot) == 3);
static_assert(static_cast<std::uint32_t>(ExecutionDnnBackendAttributeOperation::kSet) == 1);
static_assert(static_cast<std::uint32_t>(ExecutionDnnBackendAttributeOperation::kGet) == 2);
static_assert(static_cast<std::uint32_t>(ExecutionDnnBackendAttributeDirection::kNone) == 0);
static_assert(static_cast<std::uint32_t>(ExecutionDnnBackendAttributeDirection::kInput) == 1);
static_assert(static_cast<std::uint32_t>(ExecutionDnnBackendAttributeDirection::kScalarOutput) ==
              2);
static_assert(
    static_cast<std::uint32_t>(ExecutionDnnBackendAttributeDirection::kDescriptorOutput) == 3);
static_assert(static_cast<std::uint32_t>(ExecutionDnnBackendAttributeDirection::kHandleOutput) ==
              4);
static_assert(static_cast<std::uint32_t>(ExecutionDnnBackendAttributeDirection::kPointerOutput) ==
              5);
static_assert(static_cast<std::uint32_t>(ExecutionDnnBackendBindingRole::kTensor) == 1);
static_assert(static_cast<std::uint32_t>(ExecutionDnnBackendBindingRole::kWorkspace) == 2);
static_assert(static_cast<std::uint32_t>(ExecutionDnnBackendBindingClass::kIgnored) == 0);
static_assert(static_cast<std::uint32_t>(ExecutionDnnBackendBindingClass::kDevice) == 1);
static_assert(static_cast<std::uint32_t>(ExecutionDnnBackendBindingClass::kByValue) == 2);
static_assert(static_cast<std::uint32_t>(ExecutionDnnBackendBindingAccess::kNone) == 0);
static_assert(static_cast<std::uint32_t>(ExecutionDnnBackendBindingAccess::kRead) == 1);
static_assert(static_cast<std::uint32_t>(ExecutionDnnBackendBindingAccess::kWrite) == 2);
static_assert(static_cast<std::uint32_t>(ExecutionDnnBackendBindingAccess::kReadWrite) == 3);
using Values = ExecutionDnnBackendAttributeValues;
using Reference = ExecutionDnnBackendReference;
using ReferenceKind = ExecutionDnnBackendReferenceKind;

// Independent installed-ABI table, not populated by the codec under test.
constexpr std::array<unsigned, 30> kClasses{2, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1,
                                            3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
constexpr std::array<std::size_t, 30> kWidths{0, 4, 1, 8, 4, 8, 0, 4, 4, 4, 4, 4,  4, 4, 4,
                                              0, 4, 4, 4, 4, 4, 4, 4, 4, 1, 4, 16, 4, 4, 4};
constexpr std::size_t kElementLimit = 65536;
constexpr std::size_t kByteLimit = 2097152;

static_assert(noexcept(execution_dnn_backend_value_class(0)));
static_assert(noexcept(execution_dnn_backend_scalar_width(0)));
static_assert(noexcept(write_execution_dnn_backend_values(std::declval<ExecutionPayloadWriter&>(),
                                                          std::declval<const Values&>())));
static_assert(noexcept(read_execution_dnn_backend_values(std::declval<ExecutionPayloadReader&>(),
                                                         std::declval<Values&>())));

void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

bool same_cursor(ExecutionPayloadReader left, ExecutionPayloadReader right) {
    for (;;) {
        std::uint32_t a{}, b{};
        const bool has_a = left.read_u32(a);
        const bool has_b = right.read_u32(b);
        if (has_a != has_b || a != b || left.empty() != right.empty()) return false;
        if (!has_a) return true;
    }
}

bool same(const Values& left, const Values& right) {
    if (left.type != right.type || left.scalars != right.scalars ||
        left.references.size() != right.references.size())
        return false;
    for (std::size_t index = 0; index < left.references.size(); ++index) {
        const auto& a = left.references[index];
        const auto& b = right.references[index];
        if (a.kind != b.kind || a.owner != b.owner || a.id != b.id) return false;
    }
    return true;
}

template <std::size_t Width>
void append(std::vector<std::byte>& bytes, std::uint64_t value) {
    static_assert(Width == 4 || Width == 8);
    for (std::size_t index = 0; index < Width; ++index)
        bytes.push_back(static_cast<std::byte>((value >> (8 * index)) & 255));
}

std::vector<std::byte> packet(std::uint32_t type, std::span<const std::byte> body) {
    expect(body.size() <= UINT32_MAX, "fixture blob is representable");
    std::vector<std::byte> bytes;
    append<4>(bytes, type);
    append<4>(bytes, body.size());
    bytes.insert(bytes.end(), body.begin(), body.end());
    return bytes;
}

std::vector<std::byte> reference_body(std::span<const Reference> references) {
    std::vector<std::byte> bytes;
    for (const auto& value : references) {
        append<4>(bytes, static_cast<std::uint32_t>(value.kind));
        append<8>(bytes, value.owner);
        append<8>(bytes, value.id);
    }
    return bytes;
}

std::vector<std::byte> expected_wire(const Values& value) {
    return packet(value.type,
                  kClasses.at(value.type) == 1 ? value.scalars : reference_body(value.references));
}

ReferenceKind reference_kind(std::uint32_t type) {
    if (type == 0) return ReferenceKind::kHandle;
    if (type == 6) return ReferenceKind::kPointerSlot;
    expect(type == 15, "reference fixture uses only documented reference types");
    return ReferenceKind::kDescriptor;
}

Values sample(std::uint32_t type, std::size_t count) {
    Values value;
    value.type = type;
    if (kClasses.at(type) == 1) {
        value.scalars.resize(kWidths.at(type) * count);
        for (std::size_t index = 0; index < value.scalars.size(); ++index)
            value.scalars[index] =
                static_cast<std::byte>(type == 2 ? index % 2 : (index * 37 + 129) % 256);
    } else {
        value.references.resize(count);
        for (std::size_t index = 0; index < count; ++index) {
            if (index % 3 == 0) continue;
            value.references[index] = {reference_kind(type), UINT64_C(0x0102030405060708),
                                       UINT64_C(0xffeeddccbbaa9988)};
        }
    }
    return value;
}

void reject_wire(std::span<const std::byte> bytes) {
    // Nonempty fields of both classes make an accidental partial assignment visible.
    Values output{0x89abcdef,
                  {std::byte{0x12}, std::byte{0xff}},
                  {{ReferenceKind::kPointerSlot, 53, UINT64_MAX}}};
    const auto before = output;
    ExecutionPayloadReader reader(bytes);
    const auto cursor = reader;
    expect(!read_execution_dnn_backend_values(reader, output), "invalid value decoding rejects");
    expect(same(output, before) && same_cursor(reader, cursor),
           "failed value decoding preserves destination and cursor");
}

void reject_value(const Values& value) {
    ExecutionPayloadWriter writer;
    expect(writer.write_u32(0x9abcdef0), "seed a preceding field");
    expect(!write_execution_dnn_backend_values(writer, value), "invalid value encoding rejects");
    const std::vector<std::byte> prefix{std::byte{0xf0}, std::byte{0xde}, std::byte{0xbc},
                                        std::byte{0x9a}};
    expect(writer.take() == prefix, "invalid values reject before writing any field");
}

void check_round_trip(const Values& original, bool check_short_buffers) {
    const auto expected = expected_wire(original);
    ExecutionPayloadWriter writer;
    expect(write_execution_dnn_backend_values(writer, original), "encode valid typed values");
    expect(writer.take() == expected, "wire matches independent type/length/element bytes");
    auto decoded = sample(0, 2);
    ExecutionPayloadReader reader(expected);
    expect(read_execution_dnn_backend_values(reader, decoded) && reader.empty() &&
               same(decoded, original),
           "independent wire decodes exact typed data without normalization");
    ExecutionPayloadWriter exact(expected.size());
    expect(write_execution_dnn_backend_values(exact, original) && exact.take() == expected,
           "exact writer bound includes type and blob-length fields");
    if (!check_short_buffers) return;
    for (std::size_t length = 0; length < expected.size(); ++length) {
        reject_wire(std::span{expected}.first(length));
        ExecutionPayloadWriter short_writer(length + 4);
        expect(short_writer.write_u32(0x9abcdef0) &&
                   !write_execution_dnn_backend_values(short_writer, original) &&
                   short_writer.take() == std::vector<std::byte>{std::byte{0xf0}, std::byte{0xde},
                                                                 std::byte{0xbc}, std::byte{0x9a}},
               "every too-small writer preserves its prefix");
    }
}

void type_table_and_all_values() {
    expect(kExecutionDnnBackendMaxValueElements == kElementLimit &&
               kExecutionDnnBackendMaxValueBytes == kByteLimit,
           "transport quotas match the frozen wire contract");
    expect(static_cast<unsigned>(ReferenceKind::kNull) == 0 &&
               static_cast<unsigned>(ReferenceKind::kHandle) == 1 &&
               static_cast<unsigned>(ReferenceKind::kDescriptor) == 2 &&
               static_cast<unsigned>(ReferenceKind::kPointerSlot) == 3,
           "symbolic reference kind numbers are fixed");
    for (std::uint32_t type = 0; type < 30; ++type) {
        expect(static_cast<unsigned>(execution_dnn_backend_value_class(type)) == kClasses[type] &&
                   execution_dnn_backend_scalar_width(type) == kWidths[type],
               "all installed types have independent expected class and scalar width");
        check_round_trip(sample(type, 0), true);
        check_round_trip(sample(type, 3), true);
    }
    for (const auto type : {30U, 31U, 255U, UINT32_C(0x80000000), UINT32_MAX}) {
        expect(execution_dnn_backend_value_class(type) == ExecutionDnnBackendValueClass::kInvalid &&
                   execution_dnn_backend_scalar_width(type) == 0,
               "unknown types are not an untyped scalar escape hatch");
        reject_value({type, {}, {}});
        reject_wire(packet(type, {}));
        reject_wire(packet(type, std::array{std::byte{1}, std::byte{2}}));
    }
}

void explicit_wire_and_bits() {
    const Values single{
        4, {std::byte{0x45}, std::byte{0x23}, std::byte{0xc1}, std::byte{0x7f}}, {}};
    const std::vector<std::byte> literal{std::byte{4},    std::byte{0},    std::byte{0},
                                         std::byte{0},    std::byte{4},    std::byte{0},
                                         std::byte{0},    std::byte{0},    std::byte{0x45},
                                         std::byte{0x23}, std::byte{0xc1}, std::byte{0x7f}};
    ExecutionPayloadWriter writer;
    expect(write_execution_dnn_backend_values(writer, single) && writer.take() == literal,
           "literal twelve-byte packet preserves a float NaN payload");

    const Values pointer{6,
                         {},
                         {{ReferenceKind::kPointerSlot, UINT64_C(0x0102030405060708),
                           UINT64_C(0xffeeddccbbaa9988)}}};
    const std::vector<std::byte> pointer_literal{
        std::byte{6},    std::byte{0},    std::byte{0},    std::byte{0},    std::byte{20},
        std::byte{0},    std::byte{0},    std::byte{0},    std::byte{3},    std::byte{0},
        std::byte{0},    std::byte{0},    std::byte{8},    std::byte{7},    std::byte{6},
        std::byte{5},    std::byte{4},    std::byte{3},    std::byte{2},    std::byte{1},
        std::byte{0x88}, std::byte{0x99}, std::byte{0xaa}, std::byte{0xbb}, std::byte{0xcc},
        std::byte{0xdd}, std::byte{0xee}, std::byte{0xff}};
    ExecutionPayloadWriter pointer_writer;
    expect(write_execution_dnn_backend_values(pointer_writer, pointer) &&
               pointer_writer.take() == pointer_literal,
           "pointer slots contain symbolic full-width owner/id, not native pointers");

    for (const auto type : {1U,  4U,  7U,  8U,  9U,  10U, 11U, 12U, 13U, 14U, 16U,
                            17U, 18U, 19U, 20U, 21U, 22U, 23U, 25U, 27U, 28U, 29U}) {
        Values value{type, {}, {}};
        for (const auto bits : {0U, UINT32_C(0x80000000), UINT32_C(0x7f800000),
                                UINT32_C(0xff800000), UINT32_C(0x7fc12345), UINT32_MAX})
            append<4>(value.scalars, bits);
        check_round_trip(value, true);
    }
    for (const auto type : {3U, 5U, 26U}) {
        Values value{type, {}, {}};
        for (const auto bits :
             {UINT64_C(0), UINT64_C(0x8000000000000000), UINT64_C(0x7ff0000000000000),
              UINT64_C(0xfff0000000000000), UINT64_C(0x7ff8123456789abc), UINT64_MAX})
            append<8>(value.scalars, bits);
        check_round_trip(value, true);
    }
    Values characters{24, {}, {}};
    for (unsigned byte = 0; byte < 256; ++byte)
        characters.scalars.push_back(static_cast<std::byte>(byte));
    check_round_trip(characters, true);
}

void malformed_values() {
    for (std::uint32_t type = 0; type < 30; ++type) {
        auto wrong_class = sample(type, 3);
        if (kClasses[type] == 1)
            wrong_class.references = {{ReferenceKind::kNull, 0, 0}};
        else
            wrong_class.scalars = {std::byte{0}};
        reject_value(wrong_class);
        const auto width = kClasses[type] == 1 ? kWidths[type] : 20;
        for (std::size_t remainder = 1; remainder < width; ++remainder) {
            const std::vector<std::byte> malformed(width + remainder, std::byte{0});
            reject_wire(packet(type, malformed));
            if (kClasses[type] == 1) reject_value({type, malformed, {}});
        }
    }
    for (unsigned byte = 2; byte < 256; ++byte) {
        const Values boolean{2, {std::byte{0}, std::byte{1}, static_cast<std::byte>(byte)}, {}};
        reject_value(boolean);
        reject_wire(packet(2, boolean.scalars));
    }
    for (const auto type : {0U, 6U, 15U}) {
        const auto reject_reference = [type](Reference reference) {
            auto value = sample(type, 3);
            value.references.back() = reference;
            reject_value(value);
            reject_wire(packet(type, reference_body(value.references)));
        };
        for (const auto kind : {0U, 1U, 2U, 3U, 4U, UINT32_MAX}) {
            if (kind == static_cast<unsigned>(reference_kind(type))) continue;
            reject_reference({static_cast<ReferenceKind>(kind), 1, 2});
        }
        reject_reference({ReferenceKind::kNull, 1, 0});
        reject_reference({ReferenceKind::kNull, 0, 1});
        reject_reference({reference_kind(type), 0, 1});
        reject_reference({reference_kind(type), 1, 0});
        reject_reference({reference_kind(type), 0, 0});
        auto full_bits = sample(type, 3);
        full_bits.references = {{reference_kind(type), UINT64_MAX, UINT64_MAX},
                                {ReferenceKind::kNull, 0, 0},
                                {reference_kind(type), UINT64_MAX, UINT64_MAX}};
        check_round_trip(full_bits, true);
    }
}

void quota_boundaries() {
    for (std::uint32_t type = 0; type < 30; ++type) {
        check_round_trip(sample(type, kElementLimit), false);
        const auto too_many = sample(type, kElementLimit + 1);
        reject_value(too_many);
        reject_wire(expected_wire(too_many));
    }
    // With current widths, the element cap is tighter than the byte cap for
    // every valid type. A two-MiB blob is not an independently valid boundary.
    for (const auto size : {kByteLimit, kByteLimit + 1}) {
        const std::vector<std::byte> oversized(size, std::byte{0});
        reject_wire(packet(24, oversized));
        reject_value({24, oversized, {}});
    }
    for (const auto type : {0U, 2U, 3U, 6U, 15U, 26U}) {
        for (const auto claimed : {1U, UINT32_C(0x200001), UINT32_C(0x80000000), UINT32_MAX}) {
            std::vector<std::byte> header;
            append<4>(header, type);
            append<4>(header, claimed);
            reject_wire(header);
        }
    }
}

void composition() {
    const auto first = sample(26, 2);
    const auto second = sample(15, 3);
    ExecutionPayloadWriter writer;
    expect(writer.write_u64(UINT64_C(0x123456789abcdef0)) &&
               write_execution_dnn_backend_values(writer, first) &&
               write_execution_dnn_backend_values(writer, second) && writer.write_u32(0x11223344),
           "typed values compose with distinct enclosing fields");
    const auto bytes = writer.take();
    ExecutionPayloadReader reader(bytes);
    std::uint64_t prefix{};
    std::uint32_t suffix{};
    Values a, b;
    expect(reader.read_u64(prefix) && prefix == UINT64_C(0x123456789abcdef0) &&
               read_execution_dnn_backend_values(reader, a) && same(a, first) && !reader.empty() &&
               read_execution_dnn_backend_values(reader, b) && same(b, second) && !reader.empty() &&
               reader.read_u32(suffix) && suffix == 0x11223344 && reader.empty(),
           "value decoder consumes exactly its own fields, not enclosing suffixes");

    auto trailing = expected_wire(first);
    trailing.push_back(std::byte{0x99});
    ExecutionPayloadReader inner(trailing);
    expect(read_execution_dnn_backend_values(inner, a) && same(a, first) && !inner.empty(),
           "trailing data belongs to the enclosing decoder's validation");
}

bool same(const ExecutionDnnBackendCreate& a, const ExecutionDnnBackendCreate& b) {
    return a.kind == b.kind && a.has_output == b.has_output;
}
bool same(const ExecutionDnnBackendAttributeHeader& a,
          const ExecutionDnnBackendAttributeHeader& b) {
    return std::tie(a.descriptor, a.name, a.type, a.count, a.presence) ==
           std::tie(b.descriptor, b.name, b.type, b.count, b.presence);
}
bool same(const ExecutionDnnBackendPrepare& a, const ExecutionDnnBackendPrepare& b) {
    return a.operation == b.operation && same(a.header, b.header);
}
bool same(const ExecutionDnnBackendAttributePreparation& a,
          const ExecutionDnnBackendAttributePreparation& b) {
    return std::tie(a.generation, a.direction, a.input_elements, a.output_elements, a.count_bytes,
                    a.output_bytes) == std::tie(b.generation, b.direction, b.input_elements,
                                                b.output_elements, b.count_bytes, b.output_bytes);
}
bool same(const ExecutionDnnBackendAttribute& a, const ExecutionDnnBackendAttribute& b) {
    return same(a.header, b.header) && a.generation == b.generation && same(a.values, b.values);
}
bool same(const ExecutionDnnBackendGetResult& a, const ExecutionDnnBackendGetResult& b) {
    return a.count_bytes == b.count_bytes && a.count_bits == b.count_bits &&
           same(a.values, b.values);
}

template <typename T, typename Read>
void reject_message(std::span<const std::byte> bytes, const T& sentinel, Read read) {
    auto value = sentinel;
    ExecutionPayloadReader reader(bytes);
    const auto cursor = reader;
    expect(!read(reader, value) && same(value, sentinel) && same_cursor(reader, cursor),
           "malformed outer message preserves destination and cursor");
}
template <typename T, typename Write>
void reject_message_value(const T& value, Write write) {
    ExecutionPayloadWriter writer;
    expect(writer.write_u32(0x44332211), "prefix before invalid outer message");
    expect(!write(writer, value), "invalid outer message cannot encode");
    const std::vector<std::byte> prefix{std::byte{0x11}, std::byte{0x22}, std::byte{0x33},
                                        std::byte{0x44}};
    expect(writer.take() == prefix, "invalid message validates before modifying preceding packet");
}
template <typename T, typename Write, typename Read>
void message_round_trip(const T& value, const std::vector<std::byte>& bytes, const T& sentinel,
                        Write write, Read read) {
    ExecutionPayloadWriter writer;
    expect(write(writer, value) && writer.take() == bytes,
           "outer message matches independent literal field order/width/endian oracle");
    auto decoded = sentinel;
    ExecutionPayloadReader reader(bytes);
    expect(read(reader, decoded) && reader.empty() && same(decoded, value),
           "outer message decodes independent bytes exactly");
    for (std::size_t size = 0; size < bytes.size(); ++size) {
        reject_message(std::span{bytes}.first(size), sentinel, read);
        ExecutionPayloadWriter bounded(size + 4);
        expect(bounded.write_u32(0x44332211) && !write(bounded, value) &&
                   bounded.take() == std::vector<std::byte>{std::byte{0x11}, std::byte{0x22},
                                                            std::byte{0x33}, std::byte{0x44}},
               "every short outer writer preserves its prefix");
    }
    auto trailing = bytes;
    trailing.push_back(std::byte{0x7a});
    ExecutionPayloadReader composed(trailing);
    expect(read(composed, decoded) && same(decoded, value) && !composed.empty(),
           "outer helper composes and leaves enclosing trailing-byte check to caller");
}
template <std::size_t Width, std::size_t Position>
void replace(std::vector<std::byte>& bytes, std::uint64_t value) {
    expect(Position <= bytes.size() && Width <= bytes.size() - Position, "bounded wire mutation");
    for (std::size_t index = 0; index != Width; ++index)
        bytes[Position + index] = static_cast<std::byte>((value >> (index * 8)) & 255);
}
std::vector<std::byte> header_wire(const ExecutionDnnBackendAttributeHeader& header) {
    std::vector<std::byte> bytes;
    append<8>(bytes, header.descriptor);
    append<4>(bytes, header.name);
    append<4>(bytes, header.type);
    append<8>(bytes, static_cast<std::uint64_t>(header.count));
    append<4>(bytes, header.presence);
    return bytes;
}

void metadata_requests() {
    using Header = ExecutionDnnBackendAttributeHeader;
    using Prepare = ExecutionDnnBackendPrepare;
    using Operation = ExecutionDnnBackendAttributeOperation;
    const ExecutionDnnBackendCreate create_sentinel{17, 1};
    for (const auto kind : {0U, 17U, 40U, 63U, UINT32_MAX}) {
        for (const auto has_output : {0U, 1U}) {
            std::vector<std::byte> bytes;
            append<4>(bytes, kind);
            append<4>(bytes, has_output);
            message_round_trip(ExecutionDnnBackendCreate{kind, has_output}, bytes, create_sentinel,
                               write_execution_dnn_backend_create,
                               read_execution_dnn_backend_create);
        }
    }
    reject_message_value(ExecutionDnnBackendCreate{17, 2}, write_execution_dnn_backend_create);
    reject_message(std::array{std::byte{17}, std::byte{0}, std::byte{0}, std::byte{0}, std::byte{2},
                              std::byte{0}, std::byte{0}, std::byte{0}},
                   create_sentinel, read_execution_dnn_backend_create);
    const Header sentinel{71, 902, 3, 6, 3};
    for (const auto count : {INT64_MIN, std::int64_t{-1}, std::int64_t{0}, INT64_MAX}) {
        for (const auto presence : {0U, 1U, 2U, 3U}) {
            const Header header{UINT64_MAX, UINT32_MAX, UINT32_MAX, count, presence};
            const auto bytes = header_wire(header);
            expect(bytes.size() == 28, "attribute header fixed width is28B");
            message_round_trip(header, bytes, sentinel,
                               write_execution_dnn_backend_attribute_header,
                               read_execution_dnn_backend_attribute_header);
            const Prepare prepare{Operation::kGet, header};
            std::vector<std::byte> request;
            append<4>(request, 2);
            request.insert(request.end(), bytes.begin(), bytes.end());
            message_round_trip(prepare, request, Prepare{Operation::kSet, {9, 906, 3, 1, 1}},
                               write_execution_dnn_backend_prepare,
                               read_execution_dnn_backend_prepare);
        }
    }
    for (const auto presence : {4U, UINT32_MAX}) {
        auto invalid = sentinel;
        invalid.presence = presence;
        reject_message_value(invalid, write_execution_dnn_backend_attribute_header);
        reject_message(header_wire(invalid), sentinel, read_execution_dnn_backend_attribute_header);
    }
    for (const auto operation : {0U, 3U, UINT32_MAX}) {
        const Prepare invalid{static_cast<Operation>(operation), sentinel};
        reject_message_value(invalid, write_execution_dnn_backend_prepare);
        std::vector<std::byte> bytes;
        append<4>(bytes, operation);
        const auto header = header_wire(sentinel);
        bytes.insert(bytes.end(), header.begin(), header.end());
        reject_message(bytes, Prepare{Operation::kGet, sentinel},
                       read_execution_dnn_backend_prepare);
    }
    for (const auto presence : {2U, 3U}) {
        auto header = sentinel;
        header.presence = presence;
        reject_message_value(Prepare{Operation::kSet, header}, write_execution_dnn_backend_prepare);
        std::vector<std::byte> bytes;
        append<4>(bytes, 1);
        const auto fields = header_wire(header);
        bytes.insert(bytes.end(), fields.begin(), fields.end());
        reject_message(bytes, Prepare{Operation::kGet, sentinel},
                       read_execution_dnn_backend_prepare);
    }
}

void metadata_preparation_results() {
    using Preparation = ExecutionDnnBackendAttributePreparation;
    using Direction = ExecutionDnnBackendAttributeDirection;
    const Preparation sentinel{19, Direction::kScalarOutput, 0, 8, 1, 64};
    const auto wire = [](const Preparation& value) {
        std::vector<std::byte> bytes;
        append<8>(bytes, value.generation);
        append<4>(bytes, static_cast<std::uint32_t>(value.direction));
        append<4>(bytes, value.input_elements);
        append<4>(bytes, value.output_elements);
        append<4>(bytes, value.count_bytes);
        append<4>(bytes, value.output_bytes);
        expect(bytes.size() == 28, "preparation carries the exact byte prefix in 28 bytes");
        return bytes;
    };
    for (const auto direction : {Direction::kNone, Direction::kInput, Direction::kScalarOutput,
                                 Direction::kDescriptorOutput}) {
        for (const auto count_bytes : {0U, 1U, 8U}) {
            Preparation value{UINT64_MAX, direction, 0, 0, count_bytes};
            if (direction == Direction::kInput || direction == Direction::kDescriptorOutput)
                value.input_elements = 65536;
            if (direction == Direction::kScalarOutput) {
                value.output_elements = 65536;
                value.output_bytes = 1048576;
            }
            message_round_trip(value, wire(value), sentinel,
                               write_execution_dnn_backend_preparation,
                               read_execution_dnn_backend_preparation);
        }
    }
    const Preparation partial{1, Direction::kScalarOutput, 0, 1, 8, 4};
    message_round_trip(partial, wire(partial), sentinel, write_execution_dnn_backend_preparation,
                       read_execution_dnn_backend_preparation);
    const std::array invalid{Preparation{0, Direction::kNone, 0, 0, 0},
                             Preparation{1, static_cast<Direction>(6), 0, 0, 0},
                             Preparation{1, Direction::kNone, 1, 0, 0},
                             Preparation{1, Direction::kNone, 0, 1, 0},
                             Preparation{1, Direction::kInput, 0, 1, 0},
                             Preparation{1, Direction::kDescriptorOutput, 0, 1, 0},
                             Preparation{1, Direction::kScalarOutput, 1, 0, 0},
                             Preparation{1, Direction::kInput, 65537, 0, 0},
                             Preparation{1, Direction::kScalarOutput, 0, 65537, 0},
                             Preparation{1, Direction::kNone, 0, 0, 2},
                             Preparation{1, Direction::kNone, 0, 0, 4},
                             Preparation{1, Direction::kNone, 0, 0, UINT32_MAX},
                             Preparation{1, Direction::kNone, 0, 0, 0, 1},
                             Preparation{1, Direction::kInput, 1, 0, 0, 1},
                             Preparation{1, Direction::kDescriptorOutput, 1, 0, 0, 1},
                             Preparation{1, Direction::kScalarOutput, 0, 1, 0, 0},
                             Preparation{1, Direction::kScalarOutput, 0, 0, 0, 1},
                             Preparation{1, Direction::kScalarOutput, 0, 1, 0, 2097153}};
    for (const auto& value : invalid) {
        reject_message_value(value, write_execution_dnn_backend_preparation);
        reject_message(wire(value), sentinel, read_execution_dnn_backend_preparation);
    }
}

void metadata_final_values() {
    using Attribute = ExecutionDnnBackendAttribute;
    using Result = ExecutionDnnBackendGetResult;
    const Attribute sentinel{{91, 906, 3, 1, 1}, 71, sample(3, 1)};
    for (const auto type : {3U, 4U, 15U, 23U, 26U}) {
        // Materialized length is intentionally independent of signed native
        // count: this includes eight-byte Matmul padding with FLOAT count1.
        Attribute value{{UINT64_MAX, 1503, type, 1, 1}, UINT64_MAX, sample(type, 2)};
        auto bytes = header_wire(value.header);
        append<8>(bytes, value.generation);
        const auto payload = expected_wire(value.values);
        bytes.insert(bytes.end(), payload.begin(), payload.end());
        message_round_trip(value, bytes, sentinel, write_execution_dnn_backend_attribute,
                           read_execution_dnn_backend_attribute);
        auto malformed = bytes;
        replace<8, 28>(malformed, 0);
        reject_message(malformed, sentinel, read_execution_dnn_backend_attribute);
        malformed = bytes;
        replace<4, 36>(malformed, type == 3 ? 5 : 3);
        reject_message(malformed, sentinel, read_execution_dnn_backend_attribute);
        malformed = bytes;
        replace<4, 24>(malformed, 0);
        reject_message(malformed, sentinel, read_execution_dnn_backend_attribute);
        auto invalid = value;
        invalid.generation = 0;
        reject_message_value(invalid, write_execution_dnn_backend_attribute);
        invalid = value;
        invalid.header.type = 2;
        reject_message_value(invalid, write_execution_dnn_backend_attribute);
        invalid = value;
        invalid.header.presence = 0;
        reject_message_value(invalid, write_execution_dnn_backend_attribute);
    }
    const Attribute empty{{0, UINT32_MAX, 3, INT64_MIN, 0}, 1, sample(3, 0)};
    auto empty_bytes = header_wire(empty.header);
    append<8>(empty_bytes, 1);
    const auto empty_value = expected_wire(empty.values);
    empty_bytes.insert(empty_bytes.end(), empty_value.begin(), empty_value.end());
    message_round_trip(empty, empty_bytes, sentinel, write_execution_dnn_backend_attribute,
                       read_execution_dnn_backend_attribute);

    const Result result_sentinel{8, 71, sample(3, 2)};
    for (const auto count_bytes : {0U, 1U, 8U}) {
        const Result value{count_bytes,
                           count_bytes == 8   ? UINT64_MAX
                           : count_bytes == 1 ? 255U
                                              : 0U,
                           sample(4, 2)};
        std::vector<std::byte> bytes;
        append<4>(bytes, count_bytes);
        append<8>(bytes, value.count_bits);
        const auto data = expected_wire(value.values);
        bytes.insert(bytes.end(), data.begin(), data.end());
        message_round_trip(value, bytes, result_sentinel, write_execution_dnn_backend_get_result,
                           read_execution_dnn_backend_get_result);
        for (const auto width : {2U, 4U, 16U, UINT32_MAX}) {
            auto invalid = value;
            invalid.count_bytes = width;
            reject_message_value(invalid, write_execution_dnn_backend_get_result);
            auto malformed = bytes;
            replace<4, 0>(malformed, width);
            reject_message(malformed, result_sentinel, read_execution_dnn_backend_get_result);
        }
        if (count_bytes < 8) {
            auto invalid = value;
            invalid.count_bits = count_bytes ? 256 : 1;
            reject_message_value(invalid, write_execution_dnn_backend_get_result);
            replace<8, 4>(bytes, invalid.count_bits);
            reject_message(bytes, result_sentinel, read_execution_dnn_backend_get_result);
        }
    }
}
}  // namespace

int main() {
    try {
        type_table_and_all_values();
        explicit_wire_and_bits();
        malformed_values();
        quota_boundaries();
        composition();
        metadata_requests();
        metadata_preparation_results();
        metadata_final_values();
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "backend attribute value protocol: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
