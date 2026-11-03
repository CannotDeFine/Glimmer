#include "glimmer/control/execution_blas_protocol.h"

#include <array>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
using namespace glimmer::control;
static_assert(static_cast<std::uint32_t>(ExecutionGeamType::kFloat) == 1);
static_assert(static_cast<std::uint32_t>(ExecutionGeamType::kDouble) == 2);
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
bool same(const ExecutionGeam& left, const ExecutionGeam& right) {
    if (left.type != right.type || left.transpose_a != right.transpose_a ||
        left.transpose_b != right.transpose_b || left.rows != right.rows ||
        left.columns != right.columns || left.leading_a != right.leading_a ||
        left.leading_b != right.leading_b || left.leading_c != right.leading_c ||
        left.handle != right.handle || left.pointer_mode != right.pointer_mode ||
        left.presence != right.presence || left.alpha != right.alpha || left.beta != right.beta)
        return false;
    for (std::size_t index = 0; index != left.buffers.size(); ++index)
        if (left.buffers[index].allocation != right.buffers[index].allocation ||
            left.buffers[index].offset != right.buffers[index].offset)
            return false;
    return true;
}
void word(std::vector<std::byte>& bytes, std::size_t offset, std::uint32_t value) {
    expect(offset <= bytes.size() && bytes.size() - offset >= 4, "bounded literal mutation");
    for (std::size_t part = 0; part != 4; ++part)
        bytes[offset + part] = static_cast<std::byte>(value >> (8 * part));
}
std::vector<std::byte> literal() {
    // Independent wire oracle: every word is specified in protocol order, not
    // obtained from production serialization or the native C++ object layout.
    constexpr std::array<std::uint32_t, 40> words{
        2,          3,          0xfedcba98, 0x80000000, 0x7fffffff, 0xffffffff, 0,
        17,         0x05060708, 0x01020304, 0,          31,         0x55667788, 0x11223344,
        0xddeeff00, 0x99aabbcc, 7,          0,          8,          0,          0xffffffff,
        0xffffffff, 9,          0,          0,          0,          0,          0,
        0,          0,          0,          0,          0x04030201, 0x08070605, 0,
        0,          0x83828180, 0x87868584, 0,          0};
    std::vector<std::byte> bytes(160);
    for (std::size_t index = 0; index != words.size(); ++index)
        word(bytes, index * 4, words[index]);
    return bytes;
}
ExecutionGeam fixture() {
    ExecutionGeam value;
    value.type = ExecutionGeamType::kDouble;
    value.transpose_a = 3;
    value.transpose_b = 0xfedcba98;
    value.rows = INT32_MIN;
    value.columns = INT32_MAX;
    value.leading_a = -1;
    value.leading_c = 17;
    value.handle = 0x0102030405060708;
    value.presence = 31;
    value.buffers = {{{0x1122334455667788, 0x99aabbccddeeff00}, {7, 8}, {UINT64_MAX, 9}, {}, {}}};
    for (std::size_t index = 0; index != 8; ++index) {
        value.alpha[index] = static_cast<std::byte>(index + 1);
        value.beta[index] = static_cast<std::byte>(index + 0x80);
    }
    return value;
}
std::vector<std::byte> encode(const ExecutionGeam& value) {
    ExecutionPayloadWriter output;
    expect(write_execution_geam(output, value), "encode structurally valid GEAM");
    auto bytes = output.take();
    expect(bytes.size() == 160 && kExecutionGeamBytes == 160,
           "fixed 160-byte GEAM, no blob headers");
    return bytes;
}
void rejects(const std::vector<std::byte>& bytes) {
    ExecutionPayloadReader input(bytes);
    auto output = fixture();
    output.handle = 42;
    const auto before = output;
    expect(!read_execution_geam(input, output) && same(output, before),
           "malformed packet preserves every destination field");
    std::uint32_t first{};
    if (bytes.size() >= 4) {
        const auto expected = std::to_integer<std::uint32_t>(bytes[0]) |
                              (std::to_integer<std::uint32_t>(bytes[1]) << 8) |
                              (std::to_integer<std::uint32_t>(bytes[2]) << 16) |
                              (std::to_integer<std::uint32_t>(bytes[3]) << 24);
        expect(input.read_u32(first) && first == expected,
               "decode failure preserves reader cursor");
    } else {
        expect(!input.read_u32(first), "incomplete first word remains incomplete after rejection");
    }
}
void rejects(const ExecutionGeam& value) {
    ExecutionPayloadWriter output;
    expect(output.write_u32(0x12345678), "write bounded prefix before invalid value");
    expect(!write_execution_geam(output, value), "reject contradictory value before writing");
    expect(output.take() == std::vector<std::byte>{std::byte{0x78}, std::byte{0x56},
                                                   std::byte{0x34}, std::byte{0x12}},
           "invalid value cannot modify existing writer content");
}
void literal_and_boundaries() {
    const auto expected = literal();
    const auto value = fixture();
    expect(encode(value) == expected, "exact independent little-endian GEAM packet");
    ExecutionPayloadReader input(expected);
    ExecutionGeam decoded;
    expect(
        read_execution_geam(input, decoded) && input.empty() && same(decoded, value),
        "literal signed fields, raw operation words and scalar bits decode without normalization");
    for (std::size_t size = 0; size != expected.size(); ++size) {
        rejects(std::vector<std::byte>(expected.begin(),
                                       expected.begin() + static_cast<std::ptrdiff_t>(size)));
        ExecutionPayloadWriter limited(size + 4);
        expect(limited.write_u32(0x12345678) && !write_execution_geam(limited, value) &&
                   limited.take() == std::vector<std::byte>{std::byte{0x78}, std::byte{0x56},
                                                            std::byte{0x34}, std::byte{0x12}},
               "writer exhaustion preserves the existing prefix");
    }
    ExecutionPayloadWriter output;
    expect(output.write_u32(0xaabbccdd) && write_execution_geam(output, value) &&
               output.write_u64(UINT64_MAX),
           "GEAM composes between independently encoded fields");
    const auto composed = output.take();
    ExecutionPayloadReader reader(composed);
    std::uint32_t prefix{};
    std::uint64_t suffix{};
    expect(reader.read_u32(prefix) && prefix == 0xaabbccdd &&
               read_execution_geam(reader, decoded) && same(decoded, value) && !reader.empty() &&
               reader.read_u64(suffix) && suffix == UINT64_MAX && reader.empty(),
           "only the enclosing message owns the trailing-data rule");
}
void presence_and_bits() {
    constexpr std::array<std::uint32_t, 5> flags{4, 8, 16, 1, 2};
    for (const auto type : {ExecutionGeamType::kFloat, ExecutionGeamType::kDouble})
        for (const auto mode : {0U, 1U})
            for (std::uint32_t mask = 0; mask != 32; ++mask) {
                auto value = fixture();
                value.type = type;
                value.pointer_mode = mode;
                value.presence = mask;
                value.alpha = value.beta = {};
                for (std::size_t index = 0; index != flags.size(); ++index)
                    value.buffers[index] = (mask & flags[index]) && (index < 3 || mode)
                                               ? ExecutionGeamBuffer{index + 1, UINT64_MAX - index}
                                               : ExecutionGeamBuffer{};
                if (!mode) {
                    const auto width = type == ExecutionGeamType::kFloat ? 4U : 8U;
                    for (std::size_t index = 0; index != width; ++index) {
                        if (mask & 1U) value.alpha[index] = std::byte{0xff};
                        if (mask & 2U)
                            value.beta[index] = index + 1 == width ? std::byte{0x80} : std::byte{};
                    }
                }
                const auto bytes = encode(value);
                ExecutionPayloadReader input(bytes);
                ExecutionGeam decoded;
                expect(read_execution_geam(input, decoded) && input.empty() && same(value, decoded),
                       "all independent nullable masks preserve signed-zero and nonfinite bits");
                value.buffers = {};
                value.handle = 0;
                expect(
                    encode(value).size() == 160,
                    "present unresolved ranges and zero logical IDs are structural, not authority");
            }
}
void malformed_values() {
    for (const auto type : {0U, 3U, UINT32_MAX}) {
        auto value = fixture();
        value.type = static_cast<ExecutionGeamType>(type);
        rejects(value);
        auto packet = literal();
        word(packet, 0, type);
        rejects(packet);
    }
    for (const auto bad : {2U, UINT32_MAX}) {
        auto value = fixture();
        value.pointer_mode = bad;
        rejects(value);
        auto packet = literal();
        word(packet, 40, bad);
        rejects(packet);
    }
    for (const auto bad : {32U, UINT32_MAX}) {
        auto value = fixture();
        value.presence = bad;
        rejects(value);
        auto packet = literal();
        word(packet, 44, bad);
        rejects(packet);
    }
    constexpr std::array<std::uint32_t, 5> bits{4, 8, 16, 1, 2};
    for (std::size_t index = 0; index != 5; ++index) {
        auto value = fixture();
        value.alpha = value.beta = {};
        value.pointer_mode = 1;
        value.buffers[index] = {0, 1};
        rejects(value);
        auto packet = literal();
        word(packet, 40, 1);
        for (std::size_t i = 128; i != packet.size(); ++i) packet[i] = std::byte{};
        word(packet, 48 + 16 * index, 0);
        word(packet, 52 + 16 * index, 0);
        word(packet, 56 + 16 * index, 1);
        rejects(packet);
        value.buffers[index] = {23, 0};
        value.presence &= ~bits[index];
        rejects(value);
        word(packet, 48 + 16 * index, 23);
        word(packet, 44, value.presence);
        rejects(packet);
    }
    for (const auto type : {ExecutionGeamType::kFloat, ExecutionGeamType::kDouble})
        for (const auto scalar : {0U, 1U})
            for (std::size_t index = 0; index != 16; ++index) {
                ExecutionGeam value;
                value.type = type;
                value.pointer_mode = 1;
                value.presence = 31;
                auto packet = encode(value);
                (scalar ? value.beta : value.alpha)[index] = std::byte{1};
                packet[128 + scalar * 16 + index] = std::byte{1};
                rejects(value);
                rejects(packet);
                value.pointer_mode = 0;
                word(packet, 40, 0);
                const auto width = type == ExecutionGeamType::kFloat ? 4U : 8U;
                if (index >= width) {
                    rejects(value);
                    rejects(packet);
                }
                value.presence = 0;
                word(packet, 44, 0);
                rejects(value);
                rejects(packet);
            }
    for (const auto slot : {3U, 4U}) {
        auto value = fixture();
        value.buffers[slot] = {1, 0};
        rejects(value);
        auto packet = literal();
        word(packet, 48 + 16 * slot, 1);
        rejects(packet);
    }
}
}  // namespace
int main() {
    try {
        static_assert(static_cast<std::uint32_t>(ExecutionOpcode::kFunctionControl) == 128);
        static_assert(static_cast<std::uint32_t>(ExecutionOpcode::kGeam) == 129);
        literal_and_boundaries();
        presence_and_bits();
        malformed_values();
        std::cout << "Execution BLAS GEAM protocol checks passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
