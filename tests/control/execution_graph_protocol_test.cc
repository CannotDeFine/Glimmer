#include "glimmer/control/execution_graph_protocol.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
using namespace glimmer::control;
static_assert(static_cast<std::uint32_t>(ExecutionGraphAction::kBegin) == 1);
static_assert(static_cast<std::uint32_t>(ExecutionGraphAction::kEnd) == 2);
static_assert(static_cast<std::uint32_t>(ExecutionGraphAction::kState) == 3);
static_assert(static_cast<std::uint32_t>(ExecutionGraphAction::kInstantiate) == 4);
static_assert(static_cast<std::uint32_t>(ExecutionGraphAction::kDestroyGraph) == 5);
static_assert(static_cast<std::uint32_t>(ExecutionGraphAction::kDestroyExecutable) == 6);
static_assert(static_cast<std::uint32_t>(ExecutionGraphAction::kCaptureKernel) == 7);
static_assert(static_cast<std::uint32_t>(ExecutionGraphAction::kCaptureMemory) == 8);
static_assert(static_cast<std::uint32_t>(ExecutionGraphAction::kInvalidate) == 9);
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
bool equal(const ExecutionGraphRequest& left, const ExecutionGraphRequest& right) {
    return left.action == right.action && left.has_output == right.has_output &&
           left.object == right.object && left.stream == right.stream &&
           left.caller == right.caller && left.flags == right.flags &&
           left.output_seed == right.output_seed && std::ranges::equal(left.payload, right.payload);
}
template <class Value>
bool equal(const Value& left, const Value& right) {
    return left == right;
}
template <class Value, class Write, class Read>
void boundaries(const Value& value, const std::vector<std::byte>& bytes, Write write, Read read) {
    ExecutionPayloadWriter writer(bytes.size());
    expect(write(writer, value) && writer.take() == bytes, "exact independent wire bytes");
    ExecutionPayloadReader reader(bytes);
    Value decoded{};
    expect(read(reader, decoded) && reader.empty() && equal(value, decoded), "exact decode");
    for (std::size_t size = 0; size < bytes.size(); ++size) {
        ExecutionPayloadWriter limited(size + 4);
        expect(limited.write_u32(42) && !write(limited, value) &&
                   limited.take() == std::vector<std::byte>{std::byte{42}, {}, {}, {}},
               "capacity rejection cannot damage prior prefix");
        ExecutionPayloadReader truncated{std::span{bytes}.first(size)};
        decoded = value;
        expect(!read(truncated, decoded) && equal(value, decoded), "truncated output unchanged");
        // Every whole byte, including an odd trailing suffix, must remain unread.
        ExecutionPayloadReader expected{std::span{bytes}.first(size)};
        std::uint32_t actual_word{}, expected_word{};
        while (expected.read_u32(expected_word))
            expect(truncated.read_u32(actual_word) && actual_word == expected_word,
                   "truncated read cursor unchanged");
        expect(!truncated.read_u32(actual_word) && expected.empty() == truncated.empty(),
               "truncated tail unchanged");
    }
    ExecutionPayloadWriter composed(bytes.size() + 8);
    expect(write(composed, value) && composed.write_u64(UINT64_MAX), "composed record");
    const auto composed_bytes = composed.take();
    ExecutionPayloadReader cursor(composed_bytes);
    std::uint64_t tail{};
    expect(read(cursor, decoded) && equal(value, decoded) && cursor.read_u64(tail) &&
               tail == UINT64_MAX && cursor.empty(),
           "enclosing caller retains extent responsibility");
}
void requests() {
    using Action = ExecutionGraphAction;
    const std::array data{std::byte{9}, std::byte{8}, std::byte{7}};
    ExecutionGraphRequest kernel{.action = Action::kCaptureKernel,
                                 .stream = 0x0807060504030201ULL,
                                 .caller = 0x1817161514131211ULL,
                                 .payload = data};
    std::vector<std::byte> expected(63);
    expected[0] = std::byte{1};
    expected[4] = std::byte{7};
    for (unsigned int index = 0; index < 8; ++index) {
        expected[24 + index] = static_cast<std::byte>(index + 1);
        expected[32 + index] = static_cast<std::byte>(index + 17);
    }
    expected[56] = std::byte{3};
    std::ranges::copy(data, expected.begin() + 60);
    boundaries(kernel, expected, write_execution_graph_request, read_execution_graph_request);
    for (const auto action :
         {Action::kBegin, Action::kEnd, Action::kState, Action::kInstantiate, Action::kDestroyGraph,
          Action::kDestroyExecutable, Action::kCaptureKernel, Action::kCaptureMemory,
          Action::kInvalidate}) {
        ExecutionGraphRequest value{.action = action};
        if (action == Action::kInstantiate) {
            value.has_output = true;
            value.output_seed = UINT64_MAX;
            value.flags = UINT64_MAX;  // SDK, not carrier, validates native options.
        } else if (action != Action::kDestroyGraph && action != Action::kDestroyExecutable) {
            value.caller = 1;
            if (action == Action::kCaptureKernel || action == Action::kCaptureMemory)
                value.payload = data;
        }
        ExecutionPayloadWriter output;
        expect(write_execution_graph_request(output, value), "all actions and null SDK inputs");
        const auto bytes = output.take();
        ExecutionPayloadReader input(bytes);
        ExecutionGraphRequest actual;
        expect(read_execution_graph_request(input, actual) && equal(actual, value) && input.empty(),
               "all action values preserved");
    }
    for (const auto offset : {0U, 4U, 8U, 12U, 16U, 40U, 48U, 59U}) {
        auto bad = expected;
        bad[offset] = std::byte{255};
        ExecutionPayloadReader input(bad);
        auto retained = kernel;
        expect(!read_execution_graph_request(input, retained) && equal(retained, kernel),
               "invalid version/action/presence/reserved/irrelevant fields/blob rejected");
    }
    auto absent = kernel;
    absent.caller = 0;
    ExecutionPayloadWriter output;
    expect(!write_execution_graph_request(output, absent) && output.take().empty(),
           "construction caller cannot be zero");
    absent = kernel;
    absent.payload = {};
    expect(!write_execution_graph_request(output, absent) && output.take().empty(),
           "empty captured node cannot construct work");

    for (const auto action : {Action::kState, Action::kInvalidate}) {
        ExecutionGraphRequest canonical{.action = action, .caller = 1};
        ExecutionPayloadWriter encoded;
        expect(write_execution_graph_request(encoded, canonical), "encode canonical request");
        auto bad = encoded.take();
        auto invalid = canonical;
        if (action == Action::kState) {
            invalid.has_output = true;
            invalid.output_seed = 1;
            bad[8] = std::byte{1};
            bad[48] = std::byte{1};
        } else {
            invalid.flags = 1;
            bad[40] = std::byte{1};
        }
        ExecutionPayloadWriter guarded;
        expect(guarded.write_u32(42) && !write_execution_graph_request(guarded, invalid) &&
                   guarded.take() == std::vector<std::byte>{std::byte{42}, {}, {}, {}},
               "irrelevant fields cannot be encoded or damage prefix");
        ExecutionPayloadReader input(bad);
        auto retained = kernel;
        std::uint32_t version{};
        expect(!read_execution_graph_request(input, retained) && equal(retained, kernel) &&
                   input.read_u32(version) && version == 1,
               "irrelevant fields cannot be decoded or advance cursor");
    }
    ExecutionGraphRequest invalidate{.action = Action::kInvalidate, .stream = 7, .caller = 1};
    std::vector<std::byte> invalidate_bytes(60);
    invalidate_bytes[0] = std::byte{1};
    invalidate_bytes[4] = std::byte{9};
    invalidate_bytes[24] = std::byte{7};
    invalidate_bytes[32] = std::byte{1};
    boundaries(invalidate, invalidate_bytes, write_execution_graph_request,
               read_execution_graph_request);
}
void replies_and_launch() {
    const ExecutionGraphObservation observation{true, true, 2, 0x0807060504030201ULL};
    std::vector<std::byte> expected(24);
    expected[0] = std::byte{1};
    expected[4] = std::byte{3};
    expected[8] = std::byte{2};
    for (unsigned int index = 0; index < 8; ++index)
        expected[16 + index] = static_cast<std::byte>(index + 1);
    boundaries(observation, expected, write_execution_graph_observation,
               read_execution_graph_observation);
    for (const auto offset : {0U, 4U, 8U, 12U}) {
        auto bad = expected;
        bad[offset] = std::byte{255};
        ExecutionPayloadReader input(bad);
        auto retained = observation;
        expect(!read_execution_graph_observation(input, retained) && retained == observation,
               "invalid observation grammar rejected");
    }
    ExecutionPayloadWriter output;
    expect(
        !write_execution_graph_observation(output, {false, false, 0, 1}) && output.take().empty(),
        "unwritten nonzero output cannot be claimed");
    const ExecutionGraphLaunch launch{0x0807060504030201ULL, 0x1817161514131211ULL};
    expected.resize(16);
    for (unsigned int index = 0; index < 8; ++index) {
        expected[index] = static_cast<std::byte>(index + 1);
        expected[8 + index] = static_cast<std::byte>(index + 17);
    }
    boundaries(launch, expected, write_execution_graph_launch, read_execution_graph_launch);
}
}  // namespace

int main() {
    try {
        requests();
        replies_and_launch();
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
