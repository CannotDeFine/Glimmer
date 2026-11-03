#include "glimmer/control/execution_dnn_legacy_protocol.h"

#include <bit>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace {
using namespace glimmer::control;
static_assert(static_cast<std::uint32_t>(ExecutionDnnLegacyMetadataKind::kVersionCheck) == 1);
static_assert(static_cast<std::uint32_t>(ExecutionDnnLegacyMetadataKind::kSetAlgorithm) == 2);
static_assert(static_cast<std::uint32_t>(ExecutionDnnLegacyMetadataKind::kSetMath) == 3);
static_assert(static_cast<std::uint32_t>(ExecutionDnnLegacyMetadataKind::kGetMath) == 4);
static_assert(static_cast<std::uint32_t>(ExecutionDnnLegacyMetadataKind::kSetBias) == 5);
static_assert(static_cast<std::uint32_t>(ExecutionDnnLegacyMetadataKind::kGetBias) == 6);
static_assert(static_cast<std::uint32_t>(ExecutionDnnLegacyVersionCheck::kOpsInfer) == 1);
static_assert(static_cast<std::uint32_t>(ExecutionDnnLegacyVersionCheck::kOpsTrain) == 2);
static_assert(static_cast<std::uint32_t>(ExecutionDnnLegacyVersionCheck::kCnnInfer) == 3);
static_assert(static_cast<std::uint32_t>(ExecutionDnnLegacyVersionCheck::kCnnTrain) == 4);
static_assert(static_cast<std::uint32_t>(ExecutionDnnLegacyVersionCheck::kAdvInfer) == 5);
static_assert(static_cast<std::uint32_t>(ExecutionDnnLegacyVersionCheck::kAdvTrain) == 6);
static_assert(static_cast<std::uint32_t>(ExecutionDnnLegacyControlKind::kSetup) == 1);
static_assert(static_cast<std::uint32_t>(ExecutionDnnLegacyControlKind::kClip) == 2);
static_assert(static_cast<std::uint32_t>(ExecutionDnnLegacyControlKind::kWorkspaceSize) == 3);
static_assert(static_cast<std::uint32_t>(ExecutionDnnLegacyControlKind::kReserveSize) == 4);
static_assert(static_cast<std::uint32_t>(ExecutionDnnLegacyControlKind::kParameterSize) == 5);
static_assert(static_cast<std::uint32_t>(ExecutionDnnLegacyControlKind::kMatrixParameter) == 6);
static_assert(static_cast<std::uint32_t>(ExecutionDnnLegacyControlKind::kBiasParameter) == 7);
static_assert(static_cast<std::uint32_t>(ExecutionDnnLegacyControlKind::kSetAlgorithm) == 8);
static_assert(static_cast<std::uint32_t>(ExecutionDnnLegacyControlKind::kInferenceMaxCount) == 9);
static_assert(static_cast<std::uint32_t>(ExecutionDnnLegacyComputeKind::kForwardTraining) == 1);
static_assert(static_cast<std::uint32_t>(ExecutionDnnLegacyComputeKind::kBackwardData) == 2);
static_assert(static_cast<std::uint32_t>(ExecutionDnnLegacyComputeKind::kBackwardWeights) == 3);
void expect(bool condition) {
    if (!condition) throw std::runtime_error("legacy DNN codec contract failed");
}
template <typename Value, typename Write>
auto encode(const Value& value, Write write) {
    ExecutionPayloadWriter writer;
    expect(write(writer, value));
    return writer.take();
}
template <typename Value, typename Write, typename Read>
void round_trip(const Value& value, Write write, Read read) {
    const auto bytes = encode(value, write);
    Value decoded;
    ExecutionPayloadReader input(bytes);
    expect(read(input, decoded) && input.empty() && encode(decoded, write) == bytes);
    for (std::size_t count = 0; count != bytes.size(); ++count) {
        Value untouched;
        if constexpr (requires { untouched.kind; }) {
            untouched.kind = static_cast<decltype(untouched.kind)>(991);
        } else {
            untouched.written = 991;
        }
        ExecutionPayloadReader truncated{std::span{bytes}.first(count)};
        expect(!read(truncated, untouched));
        ExecutionPayloadWriter bounded(count);
        expect(!write(bounded, value));
        if constexpr (requires { untouched.kind; }) {
            expect(static_cast<std::uint32_t>(untouched.kind) == 991);
        } else {
            expect(untouched.written == 991);
        }
    }
    auto trailing = bytes;
    trailing.push_back(std::byte{1});
    ExecutionPayloadReader extra(trailing);
    expect(read(extra, decoded) && !extra.empty());
}
void metadata() {
    static_assert(static_cast<unsigned int>(ExecutionOpcode::kBindDnnProfile) == 147);
    static_assert(static_cast<unsigned int>(ExecutionOpcode::kDnnLegacyMetadata) == 148);
    static_assert(static_cast<unsigned int>(ExecutionOpcode::kDnnLegacyControl) == 149);
    static_assert(static_cast<unsigned int>(ExecutionOpcode::kDnnLegacyCompute) == 150);
    static_assert(static_cast<unsigned int>(ExecutionDnnObjectKind::kBackend) == 15);
    static_assert(static_cast<unsigned int>(ExecutionDnnObjectKind::kAlgorithm) == 16);
    for (std::uint32_t selector = 1; selector <= 6; ++selector)
        round_trip(
            ExecutionDnnLegacyMetadata{ExecutionDnnLegacyMetadataKind::kVersionCheck, 0, selector},
            write_execution_dnn_legacy_metadata, read_execution_dnn_legacy_metadata);
    for (std::uint32_t kind = 2; kind <= 6; ++kind) {
        const bool is_get = kind == 4 || kind == 6;
        ExecutionDnnLegacyMetadata value{static_cast<ExecutionDnnLegacyMetadataKind>(kind), 19,
                                         is_get ? 0U : 3U, is_get ? 1U : 0U,
                                         is_get ? 0xfedcba98U : 0U};
        expect(encode(value, write_execution_dnn_legacy_metadata).size() == 24);
        round_trip(value, write_execution_dnn_legacy_metadata, read_execution_dnn_legacy_metadata);
    }
    for (const auto selector : {0U, 7U, UINT32_MAX}) {
        ExecutionPayloadWriter writer;
        expect(!write_execution_dnn_legacy_metadata(
            writer, {ExecutionDnnLegacyMetadataKind::kVersionCheck, 0, selector}));
    }
}
void controls() {
    for (std::uint32_t kind = 1; kind <= 9; ++kind) {
        ExecutionDnnLegacyControl value;
        value.kind = static_cast<ExecutionDnnLegacyControlKind>(kind);
        value.handle = 11;
        value.metadata = 12;
        value.descriptor = 13;
        value.setup = {-7, 3, 19, 1, 2, 3, 4, 5};
        value.clip = {1, 0, 0x123456789abcdef0ULL, 0xfedcba9876543210ULL};
        value.sequence_length = 3;
        value.sequence = {1, {31, 32, 31}};
        value.input = 41;
        value.data_type = 2;
        value.parameters = {51, 52, 52, -3, 7, {61, 63}};
        value.algorithm = 71;
        value.has_output = kind >= 3 && kind != 8 ? 1U : 0U;
        value.seed = kind == 6 || kind == 7 || !value.has_output ? 0U : 0x87654321U;
        round_trip(value, write_execution_dnn_legacy_control, read_execution_dnn_legacy_control);
    }
    // Nonnull empty arrays and signed negative API counts remain distinguishable.
    for (const auto count : {-7, 0}) {
        ExecutionDnnLegacyControl value;
        value.kind = ExecutionDnnLegacyControlKind::kWorkspaceSize;
        value.sequence_length = count;
        value.sequence.has_descriptors = 1;
        round_trip(value, write_execution_dnn_legacy_control, read_execution_dnn_legacy_control);
    }
    ExecutionDnnLegacyControl value;
    value.kind = ExecutionDnnLegacyControlKind::kReserveSize;
    value.sequence_length = 2;
    value.sequence = {1, {1}};
    ExecutionPayloadWriter invalid;
    expect(!write_execution_dnn_legacy_control(invalid, value));
    value.sequence = {0, {1, 2}};
    expect(!write_execution_dnn_legacy_control(invalid, value));
    value.kind = ExecutionDnnLegacyControlKind::kInferenceMaxCount;
    value.has_output = 1;
    value.seed = std::uint64_t{1} << 32;
    expect(!write_execution_dnn_legacy_control(invalid, value));
}
void computations() {
    for (std::uint32_t kind = 1; kind <= 3; ++kind) {
        ExecutionDnnLegacyCompute value;
        value.kind = static_cast<ExecutionDnnLegacyComputeKind>(kind);
        value.handle = 11;
        value.metadata = 12;
        value.descriptor = 13;
        value.sequence_length = 2;
        value.sequences[0] = {1, {21, 22}};
        value.sequences[1] = {1, {23, 21}};
        if (kind == 2) value.sequences[2] = {1, {25, 26}};
        value.states[0] = 31;
        if (kind != 3) {
            value.states[1] = 32;
            value.states[2] = 33;
            value.states[3] = 34;
        }
        if (kind == 2) {
            value.states[4] = 35;
            value.states[5] = 36;
        }
        value.filter = 41;
        const std::uint32_t active = kind == 1 ? 0x1cf3U : kind == 2 ? 0x1ffeU : 0x1c13U;
        for (std::size_t index = 0; index != value.buffers.size(); ++index)
            if (active & (1U << index)) value.buffers[index] = {51 + index, 3 + index};
        value.workspace_bytes = 123;
        value.reserve_bytes = 456;
        const auto bytes = encode(value, write_execution_dnn_legacy_compute);
        expect(bytes.size() == (kind == 2 ? 384U : 368U));
        round_trip(value, write_execution_dnn_legacy_compute, read_execution_dnn_legacy_compute);
        for (std::size_t index = 0; index != value.buffers.size(); ++index) {
            if (active & (1U << index)) continue;
            auto invalid = value;
            invalid.buffers[index] = {9, 0};
            ExecutionPayloadWriter writer;
            expect(!write_execution_dnn_legacy_compute(writer, invalid));
        }
        auto wrong_presence = value;
        wrong_presence.sequences[0].has_descriptors = 0;
        ExecutionPayloadWriter writer;
        expect(!write_execution_dnn_legacy_compute(writer, wrong_presence));
        ExecutionPayloadWriter limited(64);
        expect(!write_execution_dnn_legacy_compute(limited, value));
    }
}
void observations() {
    for (const auto value :
         {ExecutionDnnLegacyOutput{}, ExecutionDnnLegacyOutput{1, UINT64_MAX},
          ExecutionDnnLegacyOutput{2, 0, 0, 0, 0}, ExecutionDnnLegacyOutput{2, 0, 1, 128, 64}}) {
        expect(encode(value, write_execution_dnn_legacy_output).size() == 32);
        round_trip(value, write_execution_dnn_legacy_output, read_execution_dnn_legacy_output);
    }
    for (const auto value : {ExecutionDnnLegacyOutput{4}, ExecutionDnnLegacyOutput{0, 1},
                             ExecutionDnnLegacyOutput{2, 0, 0, 1, 0},
                             ExecutionDnnLegacyOutput{2, 0, 1, UINT64_MAX, 1}}) {
        ExecutionPayloadWriter writer;
        expect(!write_execution_dnn_legacy_output(writer, value));
    }
    ExecutionSourceContinuation scope{0, {1, 2}, 3, 4, 5, 0};
    for (const auto operation :
         {ExecutionOpcode::kDnnLegacyMetadata, ExecutionOpcode::kDnnLegacyControl,
          ExecutionOpcode::kDnnLegacyCompute}) {
        expect(execution_opcode_is_dnn(operation));
        expect(encode_execution_source_continuation(scope, operation, {}).has_value());
    }
    expect(!encode_execution_source_continuation(scope, ExecutionOpcode::kBindDnnProfile, {}));
}
}  // namespace
int main() {
    try {
        metadata();
        controls();
        computations();
        observations();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
