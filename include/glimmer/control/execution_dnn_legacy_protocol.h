#pragma once

#include "glimmer/control/execution_dnn_protocol.h"

namespace glimmer::control {

// SDK8-only semantics over SDK-independent, authenticated Control messages.
// Counts retain their original signed bits; array presence is independent of
// captured length. Existing sequence and aggregate transport bounds apply.
struct ExecutionDnnLegacyDescriptorArray {
    std::uint32_t has_descriptors{};
    std::vector<std::uint64_t> descriptors;
};
// Fixed underlying type and values preserve the wire contract.
// NOLINTNEXTLINE(performance-enum-size)
enum class ExecutionDnnLegacyMetadataKind : std::uint32_t {
    kVersionCheck = 1,
    kSetAlgorithm = 2,
    kSetMath = 3,
    kGetMath = 4,
    kSetBias = 5,
    kGetBias = 6
};
// Fixed underlying type and values preserve the wire contract.
// NOLINTNEXTLINE(performance-enum-size)
enum class ExecutionDnnLegacyVersionCheck : std::uint32_t {
    kOpsInfer = 1,
    kOpsTrain = 2,
    kCnnInfer = 3,
    kCnnTrain = 4,
    kAdvInfer = 5,
    kAdvTrain = 6
};
struct ExecutionDnnLegacyMetadata {
    ExecutionDnnLegacyMetadataKind kind{};
    std::uint64_t object{};
    // Version selector, raw algorithm union bits or raw setter enum bits.
    std::uint32_t value{}, has_output{}, seed{};
};
struct ExecutionDnnLegacySetup {
    std::int32_t hidden_size{}, layers{};
    std::uint64_t dropout{};
    std::uint32_t input_mode{}, direction{}, cell_mode{}, algorithm{}, math_precision{};
};
struct ExecutionDnnLegacyParameterQuery {
    std::uint64_t input{}, filter{}, destination{};
    std::int32_t pseudo_layer{}, linear_layer{};
    ExecutionDnnBuffer weights{};
};
// Fixed underlying type and values preserve the wire contract.
// NOLINTNEXTLINE(performance-enum-size)
enum class ExecutionDnnLegacyControlKind : std::uint32_t {
    kSetup = 1,
    kClip = 2,
    kWorkspaceSize = 3,
    kReserveSize = 4,
    kParameterSize = 5,
    kMatrixParameter = 6,
    kBiasParameter = 7,
    kSetAlgorithm = 8,
    kInferenceMaxCount = 9
};
struct ExecutionDnnLegacyControl {
    ExecutionDnnLegacyControlKind kind{};
    std::uint64_t handle{}, metadata{}, descriptor{};
    std::uint32_t has_output{};
    // Original scalar output bytes, preserving failure-side no-write/updates.
    // Count uses only low32 bits. Pointer outputs never carry caller addresses.
    std::uint64_t seed{};
    ExecutionDnnLegacySetup setup{};
    ExecutionDnnRnnClip clip{};
    std::int32_t sequence_length{};
    ExecutionDnnLegacyDescriptorArray sequence{};
    std::uint64_t input{};
    std::uint32_t data_type{};
    ExecutionDnnLegacyParameterQuery parameters{};
    std::uint64_t algorithm{};
};
// Used only after a requested original actually executes. Scalar writeback=1;
// pointer writeback=2, including a native-written null. Unchanged pointer output
// has no writeback. Nonnull pointers require a separately proved bounded offset
// and actual destination-filter extent, never a remote pointer capability.
struct ExecutionDnnLegacyOutput {
    std::uint32_t written{};
    std::uint64_t scalar{};
    std::uint32_t non_null{};
    std::uint64_t offset{}, extent{};
};
// Fixed underlying type and values preserve the wire contract.
// NOLINTNEXTLINE(performance-enum-size)
enum class ExecutionDnnLegacyComputeKind : std::uint32_t {
    kForwardTraining = 1,
    kBackwardData = 2,
    kBackwardWeights = 3
};
struct ExecutionDnnLegacyCompute {
    ExecutionDnnLegacyComputeKind kind{};
    std::uint64_t handle{}, metadata{}, descriptor{};
    std::int32_t sequence_length{};
    // Forward/weights: x,y,unused. Backward-data: y,dy,dx. Preserve aliases/order.
    std::array<ExecutionDnnLegacyDescriptorArray, 3> sequences;
    // hx,cx,hy-or-dhy,cy-or-dcy,dhx,dcx. Unused positions must remain zero.
    std::array<std::uint64_t, 6> states{};
    std::uint64_t filter{};
    // x,y,dy,dx,hx,cx,hy-or-dhy,cy-or-dcy,dhx,dcx,weight-or-dweight,work,reserve.
    std::array<ExecutionDnnBuffer, 13> buffers{};
    // Exact API arguments, never the size of the containing allocation.
    std::uint64_t workspace_bytes{}, reserve_bytes{};
};

// Failure preserves the destination but may consume reader or writer bytes;
// discard the partial operation. Success consumes only this value, so the
// enclosing message must check for trailing bytes when required.
[[nodiscard]] bool write_execution_dnn_legacy_metadata(ExecutionPayloadWriter&,
                                                       const ExecutionDnnLegacyMetadata&) noexcept;
[[nodiscard]] bool read_execution_dnn_legacy_metadata(ExecutionPayloadReader&,
                                                      ExecutionDnnLegacyMetadata&) noexcept;
[[nodiscard]] bool write_execution_dnn_legacy_control(ExecutionPayloadWriter&,
                                                      const ExecutionDnnLegacyControl&) noexcept;
[[nodiscard]] bool read_execution_dnn_legacy_control(ExecutionPayloadReader&,
                                                     ExecutionDnnLegacyControl&) noexcept;
[[nodiscard]] bool write_execution_dnn_legacy_output(ExecutionPayloadWriter&,
                                                     const ExecutionDnnLegacyOutput&) noexcept;
[[nodiscard]] bool read_execution_dnn_legacy_output(ExecutionPayloadReader&,
                                                    ExecutionDnnLegacyOutput&) noexcept;
[[nodiscard]] bool write_execution_dnn_legacy_compute(ExecutionPayloadWriter&,
                                                      const ExecutionDnnLegacyCompute&) noexcept;
[[nodiscard]] bool read_execution_dnn_legacy_compute(ExecutionPayloadReader&,
                                                     ExecutionDnnLegacyCompute&) noexcept;
}  // namespace glimmer::control
