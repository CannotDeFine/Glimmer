#include "glimmer/control/execution_dnn_protocol.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <stdexcept>
#include <vector>

namespace {
using namespace glimmer::control;
void expect(bool condition) {
    if (!condition) throw std::runtime_error("DNN profile codec contract failed");
}
void check() {
    static_assert(static_cast<std::uint32_t>(ExecutionOpcode::kGraphLaunch) == 146);
    static_assert(static_cast<std::uint32_t>(ExecutionOpcode::kBindDnnProfile) == 147);
    for (const auto version : {8907U, 91002U, 8906U, UINT32_MAX}) {
        ExecutionPayloadWriter output;
        expect(write_execution_dnn_profile(output, version));
        const auto bytes = output.take();
        expect(bytes.size() == 4);
        std::uint32_t observed{17};
        ExecutionPayloadReader input(bytes);
        expect(read_execution_dnn_profile(input, observed) && observed == version && input.empty());
        for (std::size_t count = 0; count < bytes.size(); ++count) {
            ExecutionPayloadReader short_input{std::span{bytes}.first(count)};
            observed = 17;
            expect(!read_execution_dnn_profile(short_input, observed) && observed == 17);
        }
        auto trailing = bytes;
        trailing.push_back(std::byte{});
        ExecutionPayloadReader extra(trailing);
        observed = 17;
        expect(!read_execution_dnn_profile(extra, observed) && observed == 17);
    }
    ExecutionPayloadWriter output;
    expect(!write_execution_dnn_profile(output, 0) && output.take().empty());
    const std::array<std::byte, 4> zero{};
    ExecutionPayloadReader input(zero);
    std::uint32_t observed{17};
    expect(!read_execution_dnn_profile(input, observed) && observed == 17);
    for (std::uint32_t value = 1; value <= 150; ++value) {
        const auto operation = static_cast<ExecutionOpcode>(value);
        const bool wanted =
            (value >= static_cast<std::uint32_t>(ExecutionOpcode::kCreateDnnMetadata) &&
             value <= static_cast<std::uint32_t>(ExecutionOpcode::kDnnBackendExecute)) ||
            (value >= 148 && value <= 150);
        expect(execution_opcode_is_dnn(operation) == wanted);
    }
}
}  // namespace

int main() {
    try {
        check();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
