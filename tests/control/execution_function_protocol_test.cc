#include "glimmer/control/execution_function_protocol.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
using namespace glimmer::control;
void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
bool equal(const ExecutionFunctionAttributes& left, const ExecutionFunctionAttributes& right) {
    return left.sizes == right.sizes && left.values == right.values;
}
ExecutionFunctionAttributes example() {
    return {{0x0102030405060708ULL, UINT64_MAX, 0},
            {-1, INT32_MIN, INT32_MAX, 0, 1, 2, -2, 0x12345678, -0x01234567, 9, 10, 11, 12}};
}
std::vector<std::byte> golden() {
    constexpr std::array<unsigned char, 76> bytes{
        0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff,
        0xff, 0xff, 0x00, 0x00, 0x00, 0x80, 0xff, 0xff, 0xff, 0x7f, 0x00, 0x00, 0x00,
        0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0xfe, 0xff, 0xff, 0xff,
        0x78, 0x56, 0x34, 0x12, 0x99, 0xba, 0xdc, 0xfe, 0x09, 0x00, 0x00, 0x00, 0x0a,
        0x00, 0x00, 0x00, 0x0b, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00};
    std::vector<std::byte> result;
    result.reserve(bytes.size());
    for (const auto byte : bytes) result.push_back(static_cast<std::byte>(byte));
    return result;
}
void exact_wire() {
    static_assert(kExecutionFunctionAttributesBytes == 76);
    ExecutionPayloadWriter output(76);
    expect(write_execution_function_attributes(output, example()), "encode all sixteen fields");
    const auto bytes = output.take();
    expect(bytes == golden(), "wire must match independent literal little-endian76-byte record");
    ExecutionPayloadReader input(bytes);
    ExecutionFunctionAttributes value;
    expect(read_execution_function_attributes(input, value) && input.empty() &&
               equal(value, example()),
           "decode full unsigned sizes and bit-exact signed metadata without native padding");
}
void transactional_boundaries() {
    const auto bytes = golden();
    ExecutionFunctionAttributes sentinel{{9, 8, 7}, {6, 5, 4}};
    for (std::size_t length = 0; length < bytes.size(); ++length) {
        ExecutionPayloadReader input{std::span{bytes}.first(length)};
        auto before = input;
        auto value = sentinel;
        expect(!read_execution_function_attributes(input, value) && equal(value, sentinel),
               "every truncated aggregate leaves the complete destination unchanged");
        std::uint32_t actual{}, expected{};
        expect(input.read_u32(actual) == before.read_u32(expected) && actual == expected,
               "every truncated aggregate leaves the enclosing reader position unchanged");
        ExecutionPayloadWriter output(length);
        expect(!write_execution_function_attributes(output, example()) && output.take().empty(),
               "insufficient writer capacity must not publish even its first size field");
    }
    for (std::size_t capacity = 4; capacity < 80; ++capacity) {
        ExecutionPayloadWriter output(capacity);
        expect(output.write_u32(0x76543210) &&
                   !write_execution_function_attributes(output, example()) &&
                   output.take() == std::vector<std::byte>{std::byte{0x10}, std::byte{0x32},
                                                           std::byte{0x54}, std::byte{0x76}},
               "aggregate capacity preflight includes and preserves an existing prefix");
    }
}
void enclosing_composition() {
    ExecutionPayloadWriter output(88);
    expect(output.write_u32(37) && write_execution_function_attributes(output, example()) &&
               output.write_u64(UINT64_MAX),
           "aggregate appends between independently framed fields");
    const auto bytes = output.take();
    ExecutionPayloadReader input(bytes);
    std::uint32_t prefix{};
    std::uint64_t suffix{};
    ExecutionFunctionAttributes value;
    expect(input.read_u32(prefix) && prefix == 37 &&
               read_execution_function_attributes(input, value) && equal(value, example()) &&
               !input.empty() && input.read_u64(suffix) && suffix == UINT64_MAX && input.empty(),
           "aggregate consumes exactly76 bytes rather than requiring or swallowing the suffix");
    const auto sentinel = value;
    expect(!read_execution_function_attributes(input, value) && equal(value, sentinel) &&
               input.empty(),
           "failed second aggregate leaves a completed enclosing message unchanged");
}
}  // namespace
int main() {
    try {
        exact_wire();
        transactional_boundaries();
        enclosing_composition();
        std::cout << "Runtime function attribute protocol checks passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
