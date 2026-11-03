#include "glimmer/control/execution_process_memory_protocol.h"

#include <array>
#include <cstdlib>
#include <iostream>
#include <span>
#include <stdexcept>
#include <vector>

namespace {
using namespace glimmer::control;
static_assert(static_cast<std::uint32_t>(ExecutionProcessMemoryOperation::kGetDefaultPool) == 1);
static_assert(static_cast<std::uint32_t>(ExecutionProcessMemoryOperation::kGetCurrentPool) == 2);
static_assert(static_cast<std::uint32_t>(ExecutionProcessMemoryOperation::kSetCurrentPool) == 3);
static_assert(static_cast<std::uint32_t>(ExecutionProcessMemoryOperation::kAllocateAsync) == 4);
static_assert(static_cast<std::uint32_t>(ExecutionProcessMemoryOperation::kAllocateFromPoolAsync) ==
              5);
static_assert(static_cast<std::uint32_t>(ExecutionProcessMemoryOperation::kTrimPool) == 6);
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
std::vector<std::byte> literal(const std::array<unsigned char, Size>& bytes) {
    std::vector<std::byte> result;
    result.reserve(Size);
    for (const auto byte : bytes) result.push_back(static_cast<std::byte>(byte));
    return result;
}
template <class Value, class Write, class Read>
void boundaries(const Value& expected, const std::vector<std::byte>& bytes, Write write,
                Read read) {
    ExecutionPayloadWriter output(bytes.size());
    expect(write(output, expected) && output.take() == bytes,
           "producer matches independent literal, not native structure layout");
    ExecutionPayloadReader input(bytes);
    Value actual{};
    expect(read(input, actual) && input.empty() && actual == expected,
           "independent literal decodes exactly");
    for (std::size_t size = 0; size < bytes.size(); ++size) {
        ExecutionPayloadReader truncated{std::span{bytes}.first(size)};
        const auto cursor = truncated;
        actual = expected;
        expect(!read(truncated, actual) && actual == expected && same_cursor(cursor, truncated),
               "all truncations preserve destination and cursor");
        ExecutionPayloadWriter limited(size + 4);
        expect(limited.write_u32(7) && !write(limited, expected) &&
                   limited.take() == literal(std::array<unsigned char, 4>{7, 0, 0, 0}),
               "all capacity failures preserve existing prefix");
    }
    ExecutionPayloadWriter composed(bytes.size() + 12);
    expect(composed.write_u32(11) && write(composed, expected) && composed.write_u64(UINT64_MAX),
           "record composes with a prefix and suffix");
    const auto wire = composed.take();
    ExecutionPayloadReader cursor(wire);
    std::uint32_t prefix{};
    std::uint64_t suffix{};
    expect(cursor.read_u32(prefix) && prefix == 11 && read(cursor, actual) && actual == expected &&
               !cursor.empty() && cursor.read_u64(suffix) && suffix == UINT64_MAX && cursor.empty(),
           "caller retains responsibility for exact enclosing message extent");
}
void requests() {
    using Operation = ExecutionProcessMemoryOperation;
    const ExecutionProcessMemoryRequest allocation{Operation::kAllocateFromPoolAsync,
                                                   0,
                                                   true,
                                                   0x0807060504030201ULL,
                                                   UINT64_MAX,
                                                   0x1817161514131211ULL,
                                                   0x2827262524232221ULL};
    const auto bytes = literal(std::array<unsigned char, 48>{
        1,  0,  0,  0,  5,  0,  0,  0,  0,   0,   0,   0,   1,   0,   0,   0,
        1,  2,  3,  4,  5,  6,  7,  8,  255, 255, 255, 255, 255, 255, 255, 255,
        17, 18, 19, 20, 21, 22, 23, 24, 33,  34,  35,  36,  37,  38,  39,  40});
    boundaries(allocation, bytes, write_execution_process_memory_request,
               read_execution_process_memory_request);
    for (const auto operation :
         {Operation::kGetDefaultPool, Operation::kGetCurrentPool, Operation::kSetCurrentPool,
          Operation::kAllocateAsync, Operation::kAllocateFromPoolAsync, Operation::kTrimPool}) {
        for (const bool has_output : {false, true}) {
            const bool can_output =
                operation != Operation::kSetCurrentPool && operation != Operation::kTrimPool;
            if (has_output && !can_output) continue;
            ExecutionProcessMemoryRequest value;
            value.operation = operation;
            value.has_output = has_output;
            value.output_seed = has_output ? UINT64_MAX : 0;
            if (operation == Operation::kGetDefaultPool ||
                operation == Operation::kGetCurrentPool || operation == Operation::kSetCurrentPool)
                value.device = INT32_MIN;
            ExecutionPayloadWriter output;
            expect(write_execution_process_memory_request(output, value),
                   "null output, null pool and negative native device remain representable");
            const auto wire = output.take();
            ExecutionPayloadReader input(wire);
            ExecutionProcessMemoryRequest actual;
            expect(read_execution_process_memory_request(input, actual) && input.empty() &&
                       actual == value,
                   "each operation preserves native validation inputs and zero bytes");
        }
    }
    for (const auto [offset, replacement] : std::array<std::array<unsigned int, 2>, 6>{
             {{0, 0}, {0, 2}, {4, 0}, {4, 7}, {8, 1}, {12, 2}}}) {
        auto wire = bytes;
        wire[offset] = static_cast<std::byte>(replacement);
        ExecutionPayloadReader input(wire);
        const auto before = input;
        auto actual = allocation;
        expect(!read_execution_process_memory_request(input, actual) && actual == allocation &&
                   same_cursor(input, before),
               "version, discriminator, canonical device and Boolean corruption reject atomically");
    }
    const std::array invalid{
        ExecutionProcessMemoryRequest{Operation::kGetDefaultPool, 0, false, 1},
        ExecutionProcessMemoryRequest{Operation::kGetCurrentPool, 0, false, 0, 1},
        ExecutionProcessMemoryRequest{Operation::kGetCurrentPool, 0, false, 0, 0, 1},
        ExecutionProcessMemoryRequest{Operation::kSetCurrentPool, 0, true},
        ExecutionProcessMemoryRequest{Operation::kSetCurrentPool, 0, false, 0, 1},
        ExecutionProcessMemoryRequest{Operation::kSetCurrentPool, 0, false, 0, 0, 1},
        ExecutionProcessMemoryRequest{Operation::kAllocateAsync, 0, false, 1},
        ExecutionProcessMemoryRequest{Operation::kAllocateAsync, 1},
        ExecutionProcessMemoryRequest{Operation::kAllocateFromPoolAsync, 1},
        ExecutionProcessMemoryRequest{Operation::kTrimPool, 1},
        ExecutionProcessMemoryRequest{Operation::kTrimPool, 0, true},
        ExecutionProcessMemoryRequest{Operation::kTrimPool, 0, false, 0, 0, 1},
        ExecutionProcessMemoryRequest{Operation::kAllocateAsync, 0, false, 0, 0, 0, 1},
        ExecutionProcessMemoryRequest{static_cast<Operation>(UINT32_MAX)}};
    for (const auto& value : invalid) {
        ExecutionPayloadWriter output;
        expect(output.write_u32(17) && !write_execution_process_memory_request(output, value) &&
                   output.take() == literal(std::array<unsigned char, 4>{17, 0, 0, 0}),
               "unused-field authority smuggling fails before publishing a prefix");
        ExecutionPayloadWriter malformed;
        expect(malformed.write_u32(1) &&
                   malformed.write_u32(static_cast<std::uint32_t>(value.operation)) &&
                   malformed.write_u32(static_cast<std::uint32_t>(value.device)) &&
                   malformed.write_u32(value.has_output ? 1 : 0) &&
                   malformed.write_u64(value.pool) && malformed.write_u64(value.bytes) &&
                   malformed.write_u64(value.stream) && malformed.write_u64(value.output_seed),
               "construct independent noncanonical record");
        const auto wire = malformed.take();
        ExecutionPayloadReader input(wire);
        const auto before = input;
        auto actual = allocation;
        expect(!read_execution_process_memory_request(input, actual) && actual == allocation &&
                   same_cursor(input, before),
               "decoder rejects the same canonical violations without partial output");
    }
}
void observations() {
    for (const bool written : {false, true}) {
        const ExecutionProcessMemoryObservation value{written};
        const auto bytes = literal(std::array<unsigned char, 8>{
            1, 0, 0, 0, static_cast<unsigned char>(written ? 1 : 0), 0, 0, 0});
        boundaries(value, bytes, write_execution_process_memory_observation,
                   read_execution_process_memory_observation);
        for (const std::size_t offset : {0U, 4U}) {
            auto wire = bytes;
            wire[offset] = std::byte{2};
            ExecutionPayloadReader input(wire);
            const auto before = input;
            auto actual = value;
            expect(!read_execution_process_memory_observation(input, actual) && actual == value &&
                       same_cursor(input, before),
                   "unknown observation version/effect rejects atomically");
        }
        for (const auto original : {0, 1, 719, 10000, INT32_MIN, INT32_MAX}) {
            ExecutionWireReply reply;
            reply.disposition = ExecutionWireDisposition::kComplete;
            reply.original_status = original;
            reply.address = UINT64_MAX;
            reply.data = bytes;
            const auto encoded = encode_execution_reply(reply);
            if (!encoded)
                throw std::runtime_error("error-side output is independent of original SDK status");
            const auto decoded = decode_execution_reply(*encoded);
            expect(decoded && decoded->original_status == original &&
                       decoded->address == UINT64_MAX && decoded->data == bytes && !decoded->id,
                   "raw Runtime/Driver error and written bits confer no allocation identity");
        }
    }
}
}  // namespace

int main() {
    try {
        static_assert(kExecutionProcessMemoryRequestBytes == 48);
        static_assert(kExecutionProcessMemoryObservationBytes == 8);
        static_assert(!execution_is_process_allocation(0));
        static_assert(!execution_is_process_allocation(kExecutionProcessAllocationTag));
        static_assert(!execution_is_process_allocation(kExecutionProcessAllocationTag - 1));
        static_assert(execution_is_process_allocation(kExecutionProcessAllocationTag | 1));
        static_assert(execution_is_process_allocation(UINT64_MAX));
        requests();
        observations();
        std::cout << "Process memory wire controls passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
