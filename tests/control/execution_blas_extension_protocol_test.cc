#include "glimmer/control/execution_blas_protocol.h"

#include <array>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace {
using namespace glimmer::control;
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void word(std::vector<std::byte>& bytes, std::size_t offset, std::uint32_t value) {
    expect(offset <= bytes.size() && bytes.size() - offset >= 4, "bounded literal mutation");
    for (std::size_t part = 0; part != 4; ++part)
        bytes[offset + part] = static_cast<std::byte>(value >> (8 * part));
}
template <typename T>
constexpr bool kVector = std::is_same_v<T, ExecutionSgemv>;
template <typename T>
constexpr std::size_t kBytes = kVector<T> ? 152 : 172;
template <typename T>
constexpr std::size_t kMode = kVector<T> ? 32 : 52;
template <typename T>
constexpr std::size_t kBuffers = kVector<T> ? 40 : 60;
template <typename T>
bool write(ExecutionPayloadWriter& output, const T& value) {
    if constexpr (kVector<T>)
        return write_execution_sgemv(output, value);
    else
        return write_execution_sgemm_ex(output, value);
}
template <typename T>
bool read(ExecutionPayloadReader& input, T& value) {
    if constexpr (kVector<T>)
        return read_execution_sgemv(input, value);
    else
        return read_execution_sgemm_ex(input, value);
}
template <typename T>
T fixture() {
    T value;
    if constexpr (kVector<T>) {
        value.transpose = 0xfedcba98;
        value.increment_x = INT32_MIN;
        value.increment_y = -3;
    } else {
        value.transpose_a = 3;
        value.transpose_b = 0xfedcba98;
        value.type_a = UINT32_MAX;
        value.type_b = 0xabcdef01;
        value.type_c = 17;
        value.depth = -7;
        value.leading_b = 0;
        value.leading_c = 19;
    }
    value.rows = INT32_MIN;
    value.columns = INT32_MAX;
    value.leading_a = -1;
    value.handle = UINT64_C(0x0102030405060708);
    value.presence = 31;
    value.buffers = {{{UINT64_C(0x1122334455667788), UINT64_C(0x99aabbccddeeff00)},
                      {7, 8},
                      {UINT64_MAX, 9},
                      {},
                      {}}};
    for (std::size_t index = 0; index != 4; ++index) {
        value.alpha[index] = static_cast<std::byte>(index + 1);
        value.beta[index] = static_cast<std::byte>(index + 0x80);
    }
    return value;
}
template <typename T>
bool same(const T& left, const T& right) {
    if constexpr (kVector<T>) {
        if (left.transpose != right.transpose || left.increment_x != right.increment_x ||
            left.increment_y != right.increment_y)
            return false;
    } else {
        if (left.transpose_a != right.transpose_a || left.transpose_b != right.transpose_b ||
            left.type_a != right.type_a || left.type_b != right.type_b ||
            left.type_c != right.type_c || left.depth != right.depth ||
            left.leading_b != right.leading_b || left.leading_c != right.leading_c)
            return false;
    }
    if (left.rows != right.rows || left.columns != right.columns ||
        left.leading_a != right.leading_a || left.handle != right.handle ||
        left.pointer_mode != right.pointer_mode || left.presence != right.presence ||
        left.alpha != right.alpha || left.beta != right.beta)
        return false;
    for (std::size_t index = 0; index != 5; ++index)
        if (left.buffers[index].allocation != right.buffers[index].allocation ||
            left.buffers[index].offset != right.buffers[index].offset)
            return false;
    return true;
}
template <typename T>
std::vector<std::byte> literal() {
    // Every word is independent of the serializer and native object layout.
    constexpr std::array<std::uint32_t, 38> vector_words{
        0xfedcba98, 0x80000000, 0x7fffffff, 0xffffffff, 0x80000000, 0xfffffffd, 0x05060708,
        0x01020304, 0,          31,         0x55667788, 0x11223344, 0xddeeff00, 0x99aabbcc,
        7,          0,          8,          0,          0xffffffff, 0xffffffff, 9,
        0,          0,          0,          0,          0,          0,          0,
        0,          0,          0x04030201, 0,          0,          0,          0x83828180,
        0,          0,          0};
    constexpr std::array<std::uint32_t, 43> matrix_words{
        3,          0xfedcba98, 0xffffffff, 0xabcdef01, 17,         0x80000000, 0x7fffffff,
        0xfffffff9, 0xffffffff, 0,          19,         0x05060708, 0x01020304, 0,
        31,         0x55667788, 0x11223344, 0xddeeff00, 0x99aabbcc, 7,          0,
        8,          0,          0xffffffff, 0xffffffff, 9,          0,          0,
        0,          0,          0,          0,          0,          0,          0,
        0x04030201, 0,          0,          0,          0x83828180, 0,          0,
        0};
    std::vector<std::byte> bytes(kBytes<T>);
    for (std::size_t index = 0; index != bytes.size() / 4; ++index) {
        if constexpr (kVector<T>)
            word(bytes, index * 4, vector_words[index]);
        else
            word(bytes, index * 4, matrix_words[index]);
    }
    return bytes;
}
template <typename T>
std::vector<std::byte> encode(const T& value) {
    ExecutionPayloadWriter output;
    expect(write(output, value), "encode structurally valid original BLAS packet");
    auto bytes = output.take();
    expect(bytes.size() == kBytes<T>, "fixed wire length without native padding or blobs");
    return bytes;
}
template <typename T>
void rejects(const std::vector<std::byte>& bytes) {
    ExecutionPayloadReader input(bytes);
    auto output = fixture<T>();
    output.handle = 42;
    const auto before = output;
    expect(!read(input, output) && same(output, before),
           "failure preserves every destination field");
    std::uint32_t first{};
    if (bytes.size() >= 4) {
        const auto expected = std::to_integer<std::uint32_t>(bytes[0]) |
                              (std::to_integer<std::uint32_t>(bytes[1]) << 8) |
                              (std::to_integer<std::uint32_t>(bytes[2]) << 16) |
                              (std::to_integer<std::uint32_t>(bytes[3]) << 24);
        expect(input.read_u32(first) && first == expected, "failure preserves the input cursor");
    } else
        expect(!input.read_u32(first), "incomplete initial word remains incomplete");
}
template <typename T>
void rejects(const T& value) {
    ExecutionPayloadWriter output;
    expect(output.write_u32(0x12345678), "writer prefix");
    expect(!write(output, value), "reject noncanonical in-memory representation");
    expect(output.take() == std::vector<std::byte>{std::byte{0x78}, std::byte{0x56},
                                                   std::byte{0x34}, std::byte{0x12}},
           "noncanonical values cannot partially mutate the writer");
}
template <typename T>
void boundaries() {
    const auto bytes = literal<T>();
    const auto value = fixture<T>();
    expect(encode(value) == bytes, "independent exact little-endian ABI oracle");
    ExecutionPayloadReader input(bytes);
    T decoded;
    expect(read(input, decoded) && input.empty() && same(decoded, value),
           "unknown enums and signed fields preserve their raw bits");
    for (std::size_t size = 0; size != bytes.size(); ++size) {
        rejects<T>(std::vector<std::byte>(bytes.begin(),
                                          bytes.begin() + static_cast<std::ptrdiff_t>(size)));
        ExecutionPayloadWriter output(size + 4);
        expect(output.write_u32(0x12345678) && !write(output, value) &&
                   output.take() == std::vector<std::byte>{std::byte{0x78}, std::byte{0x56},
                                                           std::byte{0x34}, std::byte{0x12}},
               "writer exhaustion preserves the existing prefix");
    }
    ExecutionPayloadWriter output;
    expect(output.write_u32(0xaabbccdd) && write(output, value) && output.write_u64(UINT64_MAX),
           "composed prefix, packet and suffix");
    const auto combined = output.take();
    ExecutionPayloadReader cursor(combined);
    std::uint32_t prefix{};
    std::uint64_t suffix{};
    expect(cursor.read_u32(prefix) && prefix == 0xaabbccdd && read(cursor, decoded) &&
               same(decoded, value) && !cursor.empty() && cursor.read_u64(suffix) &&
               suffix == UINT64_MAX && cursor.empty(),
           "enclosing service owns trailing-data rejection");
}
template <typename T>
void presence() {
    constexpr std::array<std::uint32_t, 5> bits{4, 8, 16, 1, 2};
    for (const auto mode : {0U, 1U})
        for (std::uint32_t mask = 0; mask != 32; ++mask) {
            auto value = fixture<T>();
            value.pointer_mode = mode;
            value.presence = mask;
            value.alpha = value.beta = {};
            for (std::size_t index = 0; index != bits.size(); ++index)
                value.buffers[index] = (mask & bits[index]) && (index < 3 || mode)
                                           ? ExecutionGeamBuffer{index + 1, UINT64_MAX - index}
                                           : ExecutionGeamBuffer{};
            if (!mode) {
                for (std::size_t index = 0; index != 4; ++index)
                    if (mask & 1U) value.alpha[index] = std::byte{0xff};
                if (mask & 2U) value.beta[3] = std::byte{0x80};
            }
            const auto bytes = encode(value);
            ExecutionPayloadReader input(bytes);
            T decoded;
            expect(read(input, decoded) && input.empty() && same(decoded, value),
                   "all nullable masks preserve NaN and signed-zero scalar bits");
            value.buffers = {};
            value.handle = 0;
            expect(encode(value).size() == kBytes<T>,
                   "zero logical identities are not native authority");
        }
}
template <typename T>
void malformed() {
    for (const auto bad : {2U, UINT32_MAX}) {
        auto value = fixture<T>();
        value.pointer_mode = bad;
        rejects(value);
        auto bytes = literal<T>();
        word(bytes, kMode<T>, bad);
        rejects<T>(bytes);
    }
    for (const auto bad : {32U, UINT32_MAX}) {
        auto value = fixture<T>();
        value.presence = bad;
        rejects(value);
        auto bytes = literal<T>();
        word(bytes, kMode<T> + 4, bad);
        rejects<T>(bytes);
    }
    constexpr std::array<std::uint32_t, 5> bits{4, 8, 16, 1, 2};
    for (std::size_t index = 0; index != 5; ++index) {
        auto value = fixture<T>();
        value.alpha = value.beta = {};
        value.pointer_mode = 1;
        auto bytes = encode(value);
        value.buffers[index] = {0, 1};
        word(bytes, kBuffers<T> + 16 * index, 0);
        word(bytes, kBuffers<T> + 16 * index + 4, 0);
        word(bytes, kBuffers<T> + 16 * index + 8, 1);
        rejects(value);
        rejects<T>(bytes);
        value.buffers[index] = {23, 0};
        value.presence &= ~bits[index];
        word(bytes, kBuffers<T> + 16 * index, 23);
        word(bytes, kMode<T> + 4, value.presence);
        rejects(value);
        rejects<T>(bytes);
    }
    for (const auto scalar : {0U, 1U})
        for (std::size_t index = 0; index != 16; ++index) {
            T value;
            value.pointer_mode = 1;
            value.presence = 31;
            auto bytes = encode(value);
            (scalar ? value.beta : value.alpha)[index] = std::byte{1};
            bytes[kBytes<T> - 32 + scalar * 16 + index] = std::byte{1};
            rejects(value);
            rejects<T>(bytes);
            value.pointer_mode = 0;
            word(bytes, kMode<T>, 0);
            if (index >= 4) {
                rejects(value);
                rejects<T>(bytes);
            }
            value.presence = 0;
            word(bytes, kMode<T> + 4, 0);
            rejects(value);
            rejects<T>(bytes);
        }
    for (const auto slot : {3U, 4U}) {
        auto value = fixture<T>();
        value.buffers[slot] = {1, 0};
        rejects(value);
        auto bytes = literal<T>();
        word(bytes, kBuffers<T> + 16 * slot, 1);
        rejects<T>(bytes);
    }
}
}  // namespace
int main() {
    try {
        static_assert(kExecutionSgemvBytes == 152 && kExecutionSgemmExBytes == 172);
        static_assert(static_cast<std::uint32_t>(ExecutionOpcode::kQueryAsyncFrontiers) == 137);
        static_assert(static_cast<std::uint32_t>(ExecutionOpcode::kReleaseAsyncFrontiers) == 138);
        static_assert(static_cast<std::uint32_t>(ExecutionOpcode::kSgemv) == 139);
        static_assert(static_cast<std::uint32_t>(ExecutionOpcode::kSgemmEx) == 140);
        boundaries<ExecutionSgemv>();
        boundaries<ExecutionSgemmEx>();
        presence<ExecutionSgemv>();
        presence<ExecutionSgemmEx>();
        malformed<ExecutionSgemv>();
        malformed<ExecutionSgemmEx>();
        std::cout << "Execution original SGEMV/SgemmEx protocol checks passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
