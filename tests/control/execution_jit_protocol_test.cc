#include "glimmer/control/execution_jit_protocol.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace {
using namespace glimmer::control;
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
std::vector<ExecutionJitOption> example() {
    return {{7, 0x0102030405060708, {}},
            {2, 0, {std::byte{0}, std::byte{0}, std::byte{0x40}, std::byte{0x3f}}}};
}
void literal_wire_and_bounds() {
    constexpr std::array<unsigned char, 44> golden{
        0x4a, 0x49, 0x54, 0x31, 2, 0, 0, 0, 7, 0, 0, 0, 8, 7, 6, 5, 4, 3, 2, 1, 0,    0,
        0,    0,    2,    0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0x40, 0x3f};
    const auto encoded = encode_execution_jit_options(example());
    if (!encoded) throw std::runtime_error("example JIT options encode");
    expect(encoded->size() == golden.size() &&
               std::equal(
                   encoded->begin(), encoded->end(), golden.begin(),
                   [](std::byte left, unsigned char right) { return left == std::byte{right}; }),
           "independent literal wire has no pointer/native padding");
    expect(decode_execution_jit_options(*encoded) == example(),
           "all scalar and float bits round trip");
    for (std::size_t size = 0; size < encoded->size(); ++size)
        expect(!decode_execution_jit_options(std::span(*encoded).first(size)),
               "every truncation rejects");
    auto hostile = *encoded;
    hostile.push_back(std::byte{});
    expect(!decode_execution_jit_options(hostile), "trailing data rejects");
    hostile = *encoded;
    hostile[3] = std::byte{'2'};
    expect(!decode_execution_jit_options(hostile), "unrecognized version rejects");
    hostile = *encoded;
    hostile[4] = std::byte{35};
    expect(!decode_execution_jit_options(hostile), "count bound precedes storage allocation");
    hostile = *encoded;
    hostile[8] = std::byte{17};
    expect(!decode_execution_jit_options(hostile), "symbol pointers cannot become scalar slots");
    expect(encode_execution_jit_options({}).has_value() && valid_execution_jit_input({}),
           "empty codec does not invent an option");
}
void shapes_and_capacities() {
    std::vector<ExecutionJitOption> options{{3, 0, std::vector<std::byte>(65536, std::byte{0xa5})},
                                            {4, 65536, {}},
                                            {5, 0, std::vector<std::byte>(65536, std::byte{0x6b})},
                                            {6, 65536, {}},
                                            {2, 0, std::vector<std::byte>(4)}};
    expect(valid_execution_jit_input(options), "both full log budgets and float fit");
    const auto encoded = encode_execution_jit_options(options);
    expect(encoded && encoded->size() <= kExecutionMaxJitWireBytes &&
               decode_execution_jit_options(*encoded) == options,
           "full bounded input survives exact serialization");
    options[0].data.push_back(std::byte{});
    expect(!encode_execution_jit_options(options), "oversized log rejects");
    options[0].data.pop_back();
    options[1].value = 65535;
    expect(!valid_execution_jit_input(options), "capacity must match full owned input");
    options[1].value = 65536;
    options.erase(options.begin() + 1);
    expect(!valid_execution_jit_input(options), "log requires its own size pair");
    for (const auto id : {15U, 16U, 17U, 18U, 19U, 34U, UINT32_MAX}) {
        const std::array unsupported{ExecutionJitOption{id, 0, {}}};
        expect(
            !encode_execution_jit_options(unsupported) && !valid_execution_jit_input(unsupported),
            "unsupported pointer/internal/unknown option is explicit rejection");
    }
    auto duplicate = example();
    duplicate.push_back(duplicate.front());
    expect(!encode_execution_jit_options(duplicate), "duplicate keys reject");
    auto malformed = example();
    malformed[0].data = {std::byte{1}};
    expect(!encode_execution_jit_options(malformed), "scalar never carries pointer bytes");
    malformed = example();
    malformed[1].value = 1;
    expect(!encode_execution_jit_options(malformed),
           "owned byte payload requires canonical zero slot");
}
void exact_output_correspondence() {
    const std::vector<ExecutionJitOption> requested{
        {1, 128, {}},
        {7, 4, {}},
        {3, 0, {std::byte{0xa5}, std::byte{0xa5}, std::byte{0xa5}}},
        {4, 3, {}},
        {2, 0, std::vector<std::byte>(4)}};
    auto actual = requested;
    actual[0].value = 256;
    actual[2].data = {std::byte{'x'}, std::byte{0}, std::byte{0xa5}};
    actual[3].value = 1024;  // SDK may report required size beyond supplied storage.
    actual[4].data = {std::byte{0}, std::byte{0}, std::byte{0x50}, std::byte{0x40}};
    expect(execution_jit_output_matches(requested, actual),
           "exact full output storage and legal SDK output slots");
    for (unsigned int mode = 0; mode < 6; ++mode) {
        auto hostile = actual;
        if (mode == 0) hostile.pop_back();
        if (mode == 1) std::swap(hostile[0], hostile[1]);
        if (mode == 2) hostile[1].value = 3;
        if (mode == 3) hostile[2].data.pop_back();
        if (mode == 4) hostile[0].value = std::uint64_t{1} << 32;
        if (mode == 5) hostile[2].value = 0x10000;
        expect(!execution_jit_output_matches(requested, hostile),
               "partial/foreign/changed input-only outputs reject");
    }
}
void referenced_name_arrays() {
    // Independent names frame: count1, length7 INCLUDING terminating NUL.
    const std::vector<std::byte> names{
        std::byte{1},   std::byte{},    std::byte{},    std::byte{},    std::byte{7},
        std::byte{},    std::byte{},    std::byte{},    std::byte{'k'}, std::byte{'e'},
        std::byte{'r'}, std::byte{'n'}, std::byte{'e'}, std::byte{'l'}, std::byte{}};
    std::vector<ExecutionJitOption> options{{25, 1, names}, {26, 1, {}}};
    const auto encoded = encode_execution_jit_options(options);
    expect(encoded && encoded->size() == 55 && valid_execution_jit_input(options) &&
               decode_execution_jit_options(*encoded) == options,
           "input-only referenced names preserve exact literal pointer-free bytes/count");
    expect(execution_jit_output_matches(options, options),
           "names remain immutable output provenance");
    for (unsigned int mode = 0; mode < 7; ++mode) {
        auto malformed = options;
        if (mode == 0) malformed[0].value = 2;
        if (mode == 1) malformed[0].data.back() = std::byte{'x'};
        if (mode == 2) malformed[0].data[8] = std::byte{};
        if (mode == 3) malformed[0].data.push_back(std::byte{});
        if (mode == 4) malformed[1].value = 2;
        if (mode == 5) malformed[1].value = 33;
        if (mode == 6) malformed[0].value = 0;
        expect(!valid_execution_jit_input(malformed),
               "unsafe presence/name/count/tail shape rejects");
    }
    auto changed = options;
    changed[0].data[8] = std::byte{'K'};
    expect(valid_execution_jit_input(changed) && !execution_jit_output_matches(options, changed),
           "input-only names cannot change in a valid-shaped server reply");
    const std::vector<std::byte> zero(4);
    for (const auto presence : {0U, 1U}) {
        const std::vector<ExecutionJitOption> empty{{27, presence, zero}, {28, 0, {}}};
        expect(valid_execution_jit_input(empty) && encode_execution_jit_options(empty),
               "zero-count NULL/non-NULL presence stays distinct without a host pointer");
    }
    const std::array no_names{ExecutionJitOption{26, 0, {}}};
    expect(valid_execution_jit_input(no_names), "count-only zero has no implied pointer access");
    std::swap(options[0], options[1]);
    expect(valid_execution_jit_input(options), "count/name option order remains application-owned");
}
void maximum_names_and_logs_share_one_bounded_frame() {
    std::vector<std::byte> names(4);
    names[0] = std::byte{32};
    for (unsigned int index = 0; index < 32; ++index) {
        names.insert(names.end(), {std::byte{1}, std::byte{4}, std::byte{}, std::byte{}});
        names.insert(names.end(), 1024, std::byte{'x'});
        names.push_back(std::byte{});
    }
    std::vector<ExecutionJitOption> input{{25, 1, names},
                                          {26, 32, {}},
                                          {27, 1, names},
                                          {28, 32, {}},
                                          {3, 0, std::vector<std::byte>(65536)},
                                          {4, 65536, {}},
                                          {5, 0, std::vector<std::byte>(65536)},
                                          {6, 65536, {}},
                                          {2, 0, std::vector<std::byte>(4)}};
    const auto encoded = encode_execution_jit_options(input);
    expect(valid_execution_jit_input(input) && encoded && encoded->size() == 197092 &&
               encoded->size() <= kExecutionMaxJitWireBytes &&
               decode_execution_jit_options(*encoded) == input,
           "both32-name/1025-byte families coexist with both full64KiB logs and float storage");
    input[0].data[0] = std::byte{33};
    expect(!encode_execution_jit_options(input), "name count bounds precede any array access");
}
}  // namespace
int main() {
    try {
        literal_wire_and_bounds();
        shapes_and_capacities();
        exact_output_correspondence();
        referenced_name_arrays();
        maximum_names_and_logs_share_one_bounded_frame();
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
