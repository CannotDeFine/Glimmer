#pragma once

#include "glimmer/control/execution_protocol.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace glimmer::control {

constexpr bool execution_opcode_is_dnn(ExecutionOpcode operation) noexcept {
    return (operation >= ExecutionOpcode::kCreateDnnMetadata &&
            operation <= ExecutionOpcode::kDnnBackendExecute) ||
           (operation >= ExecutionOpcode::kDnnLegacyMetadata &&
            operation <= ExecutionOpcode::kDnnLegacyCompute);
}

inline bool write_execution_dnn_profile(ExecutionPayloadWriter& output,
                                        std::uint32_t header_version) noexcept {
    return header_version && output.write_u32(header_version);
}
inline bool read_execution_dnn_profile(ExecutionPayloadReader& input,
                                       std::uint32_t& header_version) noexcept {
    std::uint32_t decoded{};
    if (!input.read_u32(decoded) || !decoded || !input.empty()) return false;
    header_version = decoded;
    return true;
}

inline constexpr std::size_t kExecutionDnnMaxDimensions = 8;
// Fixed underlying type and values preserve the wire contract.
// NOLINTNEXTLINE(performance-enum-size)
enum class ExecutionDnnLibraryInfoKind : std::uint32_t {
    kVersion = 1,
    kCudartVersion = 2,
    kMaxDeviceVersion = 3,
    kProperty = 4
};
struct ExecutionDnnLibraryInfo {
    ExecutionDnnLibraryInfoKind kind{};
    // Only Property carries a raw selector and independently nullable output.
    std::uint32_t property{}, has_output{};
};
// Fixed underlying type and values preserve the wire contract.
// NOLINTNEXTLINE(performance-enum-size)
enum class ExecutionDnnObjectKind : std::uint32_t {
    kTensor = 1,
    kActivation = 2,
    kFilter = 3,
    kConvolution = 4,
    kPooling = 5,
    kLrn = 6,
    kReduction = 7,
    kDropout = 8,
    kOpTensor = 9,
    kTensorTransform = 10,
    kRnnData = 11,
    kSeqData = 12,
    kRnn = 13,
    kAttn = 14,
    kBackend = 15,
    kAlgorithm = 16
};
struct ExecutionDnnRnnState {
    std::uint32_t algorithm{}, cell_mode{}, bias_mode{}, direction{}, input_mode{}, type{},
        math_precision{}, math_type{};
    std::int32_t input_size{}, hidden_size{}, projection_size{}, layers{};
    std::uint64_t dropout{};
    std::uint32_t flags{};
};
struct ExecutionDnnAttnState {
    std::uint32_t mode{};
    std::int32_t heads{};
    std::uint64_t scale_bits{};
    std::uint32_t type{}, precision{}, math_type{};
    // Preserve both original positions, including nulls or the same identity.
    std::array<std::uint64_t, 2> dropouts{};
    std::int32_t query_size{}, key_size{}, value_size{}, query_projection{}, key_projection{},
        value_projection{}, output_projection{}, max_query_sequence{}, max_key_sequence{},
        max_batch{}, max_beam{};
};
// Fixed underlying type and values preserve the wire contract.
// NOLINTNEXTLINE(performance-enum-size)
enum class ExecutionDnnRnnQueryKind : std::uint32_t {
    kWeightSpace = 1,
    kTempSpace = 2,
    kAttentionBuffers = 3
};
struct ExecutionDnnRnnQuery {
    ExecutionDnnRnnQueryKind kind{};
    std::uint64_t handle{}, metadata{}, descriptor{}, input{};
    // Only TempSpace uses input and mode. Output bits: weight=1, work=2, reserve=4.
    std::uint32_t mode{}, outputs{};
};
struct ExecutionDnnRnnQueryResult {
    // Fixed global slot order; absent outputs must remain zero.
    std::array<std::uint64_t, 3> sizes{};
};
struct ExecutionDnnRnnClip {
    std::uint32_t mode{}, nan{};
    std::uint64_t lower_bits{}, upper_bits{};
};
inline constexpr std::size_t kExecutionDnnMaxSequenceLengths = std::size_t{1} << 20;
struct ExecutionDnnRnnDataState {
    std::uint32_t type{}, layout{};
    std::int32_t max_sequence_length{}, batch_size{}, vector_size{};
    // Lengths=1, padding=2. Presence is independent of captured byte count.
    std::uint32_t presence{};
    std::vector<std::int32_t> lengths;
    std::uint32_t padding_bytes{};
    std::array<std::byte, 8> padding{};
};
struct ExecutionDnnSeqDataState {
    std::uint32_t type{};
    std::int32_t dimensions{};
    // Sizes=1, axes=2, lengths=4, padding=8. Getter replies map output bits here.
    std::uint32_t presence{};
    std::uint64_t sequence_count{};
    std::array<std::int32_t, 4> sizes{};
    std::array<std::uint32_t, 4> axes{};
    std::vector<std::int32_t> lengths;
    std::uint32_t padding_bytes{};
    std::array<std::byte, 8> padding{};
};
struct ExecutionDnnSeqDataQuery {
    std::int32_t dimensions_requested{};
    std::uint64_t lengths_requested{};
    // Type=1, rank=2, sizes=4, axes=8, count=16, lengths=32, padding=64.
    std::uint32_t outputs{};
};
// Fixed underlying type and values preserve the wire contract.
// NOLINTNEXTLINE(performance-enum-size)
enum class ExecutionDnnTensorSetter : std::uint32_t {
    kFour = 1,
    kFourStrided = 2,
    kNd = 3,
    kNdPacked = 4
};
// Fixed underlying type and values preserve the wire contract.
// NOLINTNEXTLINE(performance-enum-size)
enum class ExecutionDnnComputeKind : std::uint32_t { kActivationForward = 1, kAddTensor = 2 };

struct ExecutionDnnTensor {
    // Raw vendor enum values; the owning backend validates them before creating
    // SDK enum objects. A format ignored by one setter must not change its result.
    std::uint32_t type{}, format{};
    std::int32_t dimensions{};
    std::array<std::int32_t, kExecutionDnnMaxDimensions> sizes{}, strides{};
};
struct ExecutionDnnActivation {
    std::uint32_t mode{}, nan{};
    std::uint64_t coefficient_bits{};
};
struct ExecutionDnnOpTensor {
    std::uint32_t operation{}, type{}, nan{};
};
struct ExecutionDnnTensorTransform {
    // Presence bits select before/after/fold pointers independently. Only rank
    // padding elements and max(rank - 2, 0) folding elements are native inputs.
    std::uint32_t dimensions{}, format{}, direction{}, presence{};
    std::array<std::int32_t, kExecutionDnnMaxDimensions> before{}, after{};
    std::array<std::uint32_t, kExecutionDnnMaxDimensions> fold{};
};
struct ExecutionDnnBuffer {
    std::uint64_t allocation{}, offset{};
};
struct ExecutionDnnRnnWeightParams {
    std::uint64_t handle{}, metadata{}, descriptor{};
    std::int32_t pseudo_layer{}, linear_layer{};
    ExecutionDnnBuffer weights{};
    std::uint64_t weight_bytes{};
    std::array<std::uint64_t, 2> destinations{};
    // Matrix=1, bias=2; independent of optional destination presence.
    std::uint32_t outputs{};
};
struct ExecutionDnnRnnWeightParamsResult {
    std::uint32_t non_null{};
    // Relative to the passed owned pointer, or numerical offsets when input
    // was null. No new allocation authority is conveyed by these values.
    std::array<std::uint64_t, 2> offsets{};
};
// Address-only layout observation. A caller's host/device base stays local;
// presence never establishes device-allocation authority or a raw-offset mode.
struct ExecutionDnnAttnWeights {
    std::uint64_t handle{}, metadata{}, descriptor{};
    std::uint32_t kind{}, has_base{};
    std::uint64_t weight_bytes{}, destination{};
    std::uint32_t has_output{};
};
struct ExecutionDnnAttnWeightsResult {
    std::uint32_t non_null{};
    std::uint64_t offset{};
};
inline constexpr std::size_t kExecutionDnnMaxAttentionWindows = std::size_t{1} << 20;
// Fixed underlying type and values preserve the wire contract.
// NOLINTNEXTLINE(performance-enum-size)
enum class ExecutionDnnAttnComputeKind : std::uint32_t {
    kForward = 1,
    kBackwardData = 2,
    kBackwardWeights = 3
};
struct ExecutionDnnAttnCompute {
    ExecutionDnnAttnComputeKind kind{};
    // BackwardWeights preserves ADD=0/SET=1; other kinds use canonical zero.
    std::uint32_t mode{};
    std::int32_t current{};
    // Independent lower/upper pointer presence, including unread early errors.
    std::uint32_t window_presence{};
    std::uint64_t handle{}, metadata{};
    // Attn, Q, K, V, O. Gradient descriptors occupy their corresponding slot.
    std::array<std::uint64_t, 5> descriptors{};
    // devQO, devKV, Q, K, V, residual, O, dO, dQ, dK, dV,
    // weights, dweights, workspace, reserve. Preserve all positional aliases.
    std::array<ExecutionDnnBuffer, 15> buffers{};
    std::uint64_t weight_bytes{}, workspace_bytes{}, reserve_bytes{};
    // Preparation sends empty vectors. Final admission reobserves the exact
    // interval; these compact snapshots never authorize a caller address.
    std::vector<std::int32_t> lower, upper;
};
struct ExecutionDnnAttnWindowRange {
    std::uint32_t first{}, count{};
};
// Fixed underlying type and values preserve the wire contract.
// NOLINTNEXTLINE(performance-enum-size)
enum class ExecutionDnnRnnComputeKind : std::uint32_t {
    kForward = 1,
    kBackwardData = 2,
    kBackwardWeights = 3
};
struct ExecutionDnnRnnCompute {
    ExecutionDnnRnnComputeKind kind{};
    // Original forward/add-gradient mode; BackwardData uses canonical zero.
    std::uint32_t mode{};
    std::uint64_t handle{}, metadata{};
    // RNN, x-or-dx data, y data, hidden tensor, cell tensor. Preserve aliases.
    std::array<std::uint64_t, 5> descriptors{};
    // devSeqLengths, x, y, dy, dx, hx, hy, dhy, dhx, cx, cy, dcy, dcx,
    // weight-or-dweight, workspace, reserve. Unused per-kind slots are absent.
    std::array<ExecutionDnnBuffer, 16> buffers{};
    // Null pointers and byte counts remain independent native inputs.
    std::uint64_t weight_bytes{}, workspace_bytes{}, reserve_bytes{};
};
struct ExecutionDnnBuildRnnDynamic {
    std::uint64_t handle{}, metadata{}, descriptor{};
    std::int32_t mini_batch{};
};
struct ExecutionDnnReorderFilterAndBias {
    std::uint64_t handle{}, metadata{}, filter{};
    std::uint32_t reorder{};
    std::int32_t reorder_bias{};
    // Filter input/output, then bias input/output. Only a zero signed flag
    // ignores bias: both ignored identities must be absent, never caller pointers.
    std::array<ExecutionDnnBuffer, 4> buffers{};
};
// Fixed underlying type and values preserve the wire contract.
// NOLINTNEXTLINE(performance-enum-size)
enum class ExecutionDnnDropoutKind : std::uint32_t { kSet = 1, kRestore = 2 };
struct ExecutionDnnDropout {
    std::uint64_t metadata{}, descriptor{}, handle{};
    ExecutionDnnBuffer states;
    std::uint64_t state_bytes{}, seed{};
    std::uint32_t probability_bits{};
    ExecutionDnnDropoutKind kind{};
};
struct ExecutionDnnCompute {
    ExecutionDnnComputeKind kind{};
    std::uint64_t handle{}, metadata{}, activation{};
    std::array<std::uint64_t, 2> tensors{};
    std::array<ExecutionDnnBuffer, 2> buffers{};
    std::uint32_t scalar_bytes{};
    std::array<std::byte, 8> alpha{}, beta{};
};
struct ExecutionDnnFilter {
    std::uint32_t type{}, format{};
    std::int32_t dimensions{};
    std::array<std::int32_t, kExecutionDnnMaxDimensions> sizes{};
};
struct ExecutionDnnConvolutionState {
    std::uint32_t mode{}, type{};
    std::int32_t dimensions{};
    std::array<std::int32_t, kExecutionDnnMaxDimensions> padding{}, strides{}, dilation{};
};
// Fixed underlying type and values preserve the wire contract.
// NOLINTNEXTLINE(performance-enum-size)
enum class ExecutionDnnConvolutionAttribute : std::uint32_t { kGroup = 1, kMath = 2, kReorder = 3 };
// Fixed underlying type and values preserve the wire contract.
// NOLINTNEXTLINE(performance-enum-size)
enum class ExecutionDnnConvolutionQueryKind : std::uint32_t {
    kWorkspace = 1,
    kHeuristic = 2,
    kMaxCount = 3
};
struct ExecutionDnnConvolutionQuery {
    ExecutionDnnConvolutionQueryKind kind{};
    std::uint64_t handle{}, metadata{};
    // Input tensor, filter, convolution and output tensor, in this order.
    std::array<std::uint64_t, 4> descriptors{};
    std::uint32_t algorithm{}, requested{};
};
// Fixed underlying type and values preserve the wire contract.
// NOLINTNEXTLINE(performance-enum-size)
enum class ExecutionDnnBackwardConvolutionQueryKind : std::uint32_t {
    kDataMaxCount = 1,
    kFilterMaxCount = 2,
    kDataWorkspace = 3
};
struct ExecutionDnnBackwardConvolutionQuery {
    ExecutionDnnBackwardConvolutionQueryKind kind{};
    std::uint64_t handle{}, metadata{};
    // Public backward-data ABI order: filter w, tensor dy, convolution, tensor dx.
    // MaxCount has no metadata/descriptors/algorithm; has_output remains explicit.
    std::array<std::uint64_t, 4> descriptors{};
    std::uint32_t algorithm{}, has_output{};
};
struct ExecutionDnnFoldedConvBackwardDataDescriptors {
    std::uint64_t handle{}, metadata{};
    // Filter, diff tensor, convolution, grad tensor, in original ABI order.
    std::array<std::uint64_t, 4> inputs{};
    std::uint32_t format{};
    // Folded filter, padded diff, folded convolution, folded grad, filter-fold,
    // diff-pad, grad-fold and optional grad-unfold transforms. Preserve aliases.
    std::array<std::uint64_t, 8> outputs{};
};
struct ExecutionDnnConvolutionForward {
    std::uint64_t handle{}, metadata{};
    std::array<std::uint64_t, 4> descriptors{};
    // Input, filter, output and workspace. A nonzero workspace allocation with
    // zero workspace_bytes remains an owned pointer, not an absent workspace.
    std::array<ExecutionDnnBuffer, 4> buffers{};
    std::uint32_t algorithm{};
    std::uint64_t workspace_bytes{};
    std::uint32_t scalar_bytes{};
    std::array<std::byte, 8> alpha{}, beta{};
};
struct ExecutionDnnConvolutionFind {
    std::uint64_t handle{}, metadata{};
    // Input tensor, filter, convolution and output tensor. Find owns its
    // benchmark allocations; this non-Ex operation carries no device pointers.
    std::array<std::uint64_t, 4> descriptors{};
    // Preserve signed int bits, including zero/negative native error requests.
    std::uint32_t requested{};
};
struct ExecutionDnnPooling {
    std::uint32_t mode{}, nan{};
    std::int32_t dimensions{};
    std::array<std::int32_t, kExecutionDnnMaxDimensions> window{}, padding{}, strides{};
};
struct ExecutionDnnLrn {
    std::uint32_t n{};
    std::uint64_t alpha_bits{}, beta_bits{}, k_bits{};
};
struct ExecutionDnnReduction {
    std::uint32_t operation{}, type{}, nan{}, indices{}, indices_type{};
};
// Fixed underlying type and values preserve the wire contract.
// NOLINTNEXTLINE(performance-enum-size)
enum class ExecutionDnnReductionQueryKind : std::uint32_t { kIndices = 1, kWorkspace = 2 };
struct ExecutionDnnReductionQuery {
    ExecutionDnnReductionQueryKind kind{};
    std::uint64_t handle{}, metadata{};
    // Input tensor, reduction and output tensor, in this order.
    std::array<std::uint64_t, 3> descriptors{};
};
struct ExecutionDnnReductionCompute {
    std::uint64_t handle{}, metadata{};
    std::array<std::uint64_t, 3> descriptors{};
    // Input, output, indices and workspace. Optional null identities preserve
    // their byte counts; nonnull zero-byte pointers retain their exact owner.
    std::array<ExecutionDnnBuffer, 4> buffers{};
    std::uint64_t indices_bytes{}, workspace_bytes{};
    std::uint32_t scalar_bytes{};
    std::array<std::byte, 8> alpha{}, beta{};
};
// Fixed underlying type and values preserve the wire contract.
// NOLINTNEXTLINE(performance-enum-size)
enum class ExecutionDnnBatchNormKind : std::uint32_t { kForward = 1, kBackward = 2, kReserve = 3 };
struct ExecutionDnnBatchNormQuery {
    ExecutionDnnBatchNormKind kind{};
    std::uint32_t mode{}, operation{};
    std::uint64_t handle{}, metadata{};
    // x, y, dy, z-or-dz, dx, parameters, optional activation.
    std::array<std::uint64_t, 7> descriptors{};
};
struct ExecutionDnnBatchNormCompute {
    ExecutionDnnBatchNormQuery query;
    // x, y, dy, z-or-dz, dx, scale, bias, running-mean-or-dscale,
    // running-variance-or-dbias, saved-mean, saved-inverse-variance,
    // workspace, reserve.
    std::array<ExecutionDnnBuffer, 13> buffers{};
    std::uint64_t workspace_bytes{}, reserve_bytes{}, epsilon_bits{}, average_factor_bits{};
    std::uint32_t scalar_bytes{};
    // Two forward or four backward fixed-width raw coefficient payloads.
    std::array<std::array<std::byte, 8>, 4> coefficients{};
};
// Fixed underlying type and values preserve the wire contract.
// NOLINTNEXTLINE(performance-enum-size)
enum class ExecutionDnnTensorForwardKind : std::uint32_t {
    kPoolingForward = 1,
    kSoftmaxForward = 2,
    kLrnForward = 3,
    kTransformTensor = 4
};
struct ExecutionDnnTensorForward {
    ExecutionDnnTensorForwardKind kind{};
    // The operation descriptor occupies the original pooling-ID wire position.
    std::uint64_t handle{}, metadata{}, descriptor{};
    std::array<std::uint64_t, 2> tensors{};
    std::array<ExecutionDnnBuffer, 2> buffers{};
    std::uint32_t algorithm{}, mode{}, scalar_bytes{};
    std::array<std::byte, 8> alpha{}, beta{};
};
inline constexpr std::size_t kExecutionDnnMaxConvolutionAlgorithms = 64;
struct ExecutionDnnConvolutionPerformance {
    std::uint32_t algorithm{}, status{}, time_bits{};
    std::uint64_t memory{};
    std::uint32_t determinism{}, math{};
    std::array<std::int32_t, 3> reserved{};
};
struct ExecutionDnnConvolutionPerformances {
    std::uint32_t count{};
    std::array<ExecutionDnnConvolutionPerformance, kExecutionDnnMaxConvolutionAlgorithms> values{};
};

// CUDA-free little-endian field codecs. Tensor ranks must fit the fixed arrays;
// raw vendor enums, signed dimensions/strides and coefficient bits are preserved.
// Compute kind and scalar width are validated. Earlier compute layouts use
// length-prefixed coefficient blobs; BN uses two/four fixed-width payloads.
// Ownership, peer identity, allocation ranges and SDK legality are
// checked by the service/backend, never inferred from these decoded values.
//
// Independent reader/writer instances may be used concurrently. A failed read
// leaves the destination unchanged. Failure may consume reader/writer bytes:
// discard that partial operation. Success consumes only this value; the caller
// must check reader.empty() when this is the final value in its message.
[[nodiscard]] bool write_execution_dnn_rnn(ExecutionPayloadWriter& output,
                                           const ExecutionDnnRnnState& value) noexcept;
[[nodiscard]] bool read_execution_dnn_rnn(ExecutionPayloadReader& input,
                                          ExecutionDnnRnnState& value) noexcept;
[[nodiscard]] bool write_execution_dnn_attn(ExecutionPayloadWriter& output,
                                            const ExecutionDnnAttnState& value) noexcept;
[[nodiscard]] bool read_execution_dnn_attn(ExecutionPayloadReader& input,
                                           ExecutionDnnAttnState& value) noexcept;
[[nodiscard]] bool write_execution_dnn_rnn_query(ExecutionPayloadWriter& output,
                                                 const ExecutionDnnRnnQuery& value) noexcept;
[[nodiscard]] bool read_execution_dnn_rnn_query(ExecutionPayloadReader& input,
                                                ExecutionDnnRnnQuery& value) noexcept;
[[nodiscard]] bool write_execution_dnn_rnn_query_result(
    ExecutionPayloadWriter& output, const ExecutionDnnRnnQueryResult& value) noexcept;
[[nodiscard]] bool read_execution_dnn_rnn_query_result(ExecutionPayloadReader& input,
                                                       ExecutionDnnRnnQueryResult& value) noexcept;
[[nodiscard]] bool write_execution_dnn_rnn_clip(ExecutionPayloadWriter& output,
                                                const ExecutionDnnRnnClip& value) noexcept;
[[nodiscard]] bool read_execution_dnn_rnn_clip(ExecutionPayloadReader& input,
                                               ExecutionDnnRnnClip& value) noexcept;
[[nodiscard]] bool write_execution_dnn_rnn_compute(ExecutionPayloadWriter& output,
                                                   const ExecutionDnnRnnCompute& value) noexcept;
[[nodiscard]] bool read_execution_dnn_rnn_compute(ExecutionPayloadReader& input,
                                                  ExecutionDnnRnnCompute& value) noexcept;
[[nodiscard]] bool write_execution_dnn_build_rnn_dynamic(
    ExecutionPayloadWriter& output, const ExecutionDnnBuildRnnDynamic& value) noexcept;
[[nodiscard]] bool read_execution_dnn_build_rnn_dynamic(
    ExecutionPayloadReader& input, ExecutionDnnBuildRnnDynamic& value) noexcept;
[[nodiscard]] bool write_execution_dnn_rnn_weight_params(
    ExecutionPayloadWriter& output, const ExecutionDnnRnnWeightParams& value) noexcept;
[[nodiscard]] bool read_execution_dnn_rnn_weight_params(
    ExecutionPayloadReader& input, ExecutionDnnRnnWeightParams& value) noexcept;
[[nodiscard]] bool write_execution_dnn_rnn_weight_params_result(
    ExecutionPayloadWriter& output, const ExecutionDnnRnnWeightParamsResult& value) noexcept;
[[nodiscard]] bool read_execution_dnn_rnn_weight_params_result(
    ExecutionPayloadReader& input, ExecutionDnnRnnWeightParamsResult& value) noexcept;
[[nodiscard]] bool write_execution_dnn_attn_weights(ExecutionPayloadWriter& output,
                                                    const ExecutionDnnAttnWeights& value) noexcept;
[[nodiscard]] bool read_execution_dnn_attn_weights(ExecutionPayloadReader& input,
                                                   ExecutionDnnAttnWeights& value) noexcept;
[[nodiscard]] bool write_execution_dnn_attn_weights_result(
    ExecutionPayloadWriter& output, const ExecutionDnnAttnWeightsResult& value) noexcept;
[[nodiscard]] bool read_execution_dnn_attn_weights_result(
    ExecutionPayloadReader& input, ExecutionDnnAttnWeightsResult& value) noexcept;
[[nodiscard]] bool write_execution_dnn_attn_compute(ExecutionPayloadWriter& output,
                                                    const ExecutionDnnAttnCompute& value) noexcept;
[[nodiscard]] bool read_execution_dnn_attn_compute(ExecutionPayloadReader& input,
                                                   ExecutionDnnAttnCompute& value) noexcept;
[[nodiscard]] bool write_execution_dnn_attn_window_range(
    ExecutionPayloadWriter& output, const ExecutionDnnAttnWindowRange& value) noexcept;
[[nodiscard]] bool read_execution_dnn_attn_window_range(
    ExecutionPayloadReader& input, ExecutionDnnAttnWindowRange& value) noexcept;
[[nodiscard]] bool write_execution_dnn_rnn_data(ExecutionPayloadWriter& output,
                                                const ExecutionDnnRnnDataState& value) noexcept;
[[nodiscard]] bool read_execution_dnn_rnn_data(ExecutionPayloadReader& input,
                                               ExecutionDnnRnnDataState& value) noexcept;
[[nodiscard]] bool write_execution_dnn_seq_data(ExecutionPayloadWriter& output,
                                                const ExecutionDnnSeqDataState& value) noexcept;
[[nodiscard]] bool read_execution_dnn_seq_data(ExecutionPayloadReader& input,
                                               ExecutionDnnSeqDataState& value) noexcept;
[[nodiscard]] bool write_execution_dnn_seq_data_query(
    ExecutionPayloadWriter& output, const ExecutionDnnSeqDataQuery& value) noexcept;
[[nodiscard]] bool read_execution_dnn_seq_data_query(ExecutionPayloadReader& input,
                                                     ExecutionDnnSeqDataQuery& value) noexcept;
[[nodiscard]] bool write_execution_dnn_library_info(ExecutionPayloadWriter& output,
                                                    const ExecutionDnnLibraryInfo& query) noexcept;
[[nodiscard]] bool read_execution_dnn_library_info(ExecutionPayloadReader& input,
                                                   ExecutionDnnLibraryInfo& query) noexcept;
[[nodiscard]] bool write_execution_dnn_reorder_filter_and_bias(
    ExecutionPayloadWriter& output, const ExecutionDnnReorderFilterAndBias& reorder) noexcept;
[[nodiscard]] bool read_execution_dnn_reorder_filter_and_bias(
    ExecutionPayloadReader& input, ExecutionDnnReorderFilterAndBias& reorder) noexcept;
[[nodiscard]] bool write_execution_dnn_op_tensor(ExecutionPayloadWriter& output,
                                                 const ExecutionDnnOpTensor& operation) noexcept;
[[nodiscard]] bool read_execution_dnn_op_tensor(ExecutionPayloadReader& input,
                                                ExecutionDnnOpTensor& operation) noexcept;
[[nodiscard]] bool write_execution_dnn_tensor_transform(
    ExecutionPayloadWriter& output, const ExecutionDnnTensorTransform& transform) noexcept;
[[nodiscard]] bool read_execution_dnn_tensor_transform(
    ExecutionPayloadReader& input, ExecutionDnnTensorTransform& transform) noexcept;
[[nodiscard]] bool write_execution_dnn_dropout(ExecutionPayloadWriter& output,
                                               const ExecutionDnnDropout& dropout) noexcept;
[[nodiscard]] bool read_execution_dnn_dropout(ExecutionPayloadReader& input,
                                              ExecutionDnnDropout& dropout) noexcept;
[[nodiscard]] bool write_execution_dnn_tensor(ExecutionPayloadWriter& output,
                                              const ExecutionDnnTensor& tensor) noexcept;
[[nodiscard]] bool read_execution_dnn_tensor(ExecutionPayloadReader& input,
                                             ExecutionDnnTensor& tensor) noexcept;
[[nodiscard]] bool write_execution_dnn_activation(
    ExecutionPayloadWriter& output, const ExecutionDnnActivation& activation) noexcept;
[[nodiscard]] bool read_execution_dnn_activation(ExecutionPayloadReader& input,
                                                 ExecutionDnnActivation& activation) noexcept;
[[nodiscard]] bool write_execution_dnn_compute(ExecutionPayloadWriter& output,
                                               const ExecutionDnnCompute& compute) noexcept;
[[nodiscard]] bool read_execution_dnn_compute(ExecutionPayloadReader& input,
                                              ExecutionDnnCompute& compute) noexcept;
[[nodiscard]] bool write_execution_dnn_filter(ExecutionPayloadWriter& output,
                                              const ExecutionDnnFilter& filter) noexcept;
[[nodiscard]] bool read_execution_dnn_filter(ExecutionPayloadReader& input,
                                             ExecutionDnnFilter& filter) noexcept;
[[nodiscard]] bool write_execution_dnn_convolution(
    ExecutionPayloadWriter& output, const ExecutionDnnConvolutionState& convolution) noexcept;
[[nodiscard]] bool read_execution_dnn_convolution(
    ExecutionPayloadReader& input, ExecutionDnnConvolutionState& convolution) noexcept;
[[nodiscard]] bool write_execution_dnn_convolution_query(
    ExecutionPayloadWriter& output, const ExecutionDnnConvolutionQuery& query) noexcept;
[[nodiscard]] bool read_execution_dnn_convolution_query(
    ExecutionPayloadReader& input, ExecutionDnnConvolutionQuery& query) noexcept;
[[nodiscard]] bool write_execution_dnn_backward_convolution_query(
    ExecutionPayloadWriter& output, const ExecutionDnnBackwardConvolutionQuery& query) noexcept;
[[nodiscard]] bool read_execution_dnn_backward_convolution_query(
    ExecutionPayloadReader& input, ExecutionDnnBackwardConvolutionQuery& query) noexcept;
[[nodiscard]] bool write_execution_dnn_folded_conv_backward_data_descriptors(
    ExecutionPayloadWriter& output,
    const ExecutionDnnFoldedConvBackwardDataDescriptors& descriptors) noexcept;
[[nodiscard]] bool read_execution_dnn_folded_conv_backward_data_descriptors(
    ExecutionPayloadReader& input,
    ExecutionDnnFoldedConvBackwardDataDescriptors& descriptors) noexcept;
[[nodiscard]] bool write_execution_dnn_convolution_forward(
    ExecutionPayloadWriter& output, const ExecutionDnnConvolutionForward& forward) noexcept;
[[nodiscard]] bool read_execution_dnn_convolution_forward(
    ExecutionPayloadReader& input, ExecutionDnnConvolutionForward& forward) noexcept;
[[nodiscard]] bool write_execution_dnn_convolution_find(
    ExecutionPayloadWriter& output, const ExecutionDnnConvolutionFind& find) noexcept;
[[nodiscard]] bool read_execution_dnn_convolution_find(ExecutionPayloadReader& input,
                                                       ExecutionDnnConvolutionFind& find) noexcept;
[[nodiscard]] bool write_execution_dnn_pooling(ExecutionPayloadWriter& output,
                                               const ExecutionDnnPooling& pooling) noexcept;
[[nodiscard]] bool read_execution_dnn_pooling(ExecutionPayloadReader& input,
                                              ExecutionDnnPooling& pooling) noexcept;
[[nodiscard]] bool write_execution_dnn_lrn(ExecutionPayloadWriter& output,
                                           const ExecutionDnnLrn& lrn) noexcept;
[[nodiscard]] bool read_execution_dnn_lrn(ExecutionPayloadReader& input,
                                          ExecutionDnnLrn& lrn) noexcept;
[[nodiscard]] bool write_execution_dnn_reduction(ExecutionPayloadWriter& output,
                                                 const ExecutionDnnReduction& reduction) noexcept;
[[nodiscard]] bool read_execution_dnn_reduction(ExecutionPayloadReader& input,
                                                ExecutionDnnReduction& reduction) noexcept;
[[nodiscard]] bool write_execution_dnn_reduction_query(
    ExecutionPayloadWriter& output, const ExecutionDnnReductionQuery& query) noexcept;
[[nodiscard]] bool read_execution_dnn_reduction_query(ExecutionPayloadReader& input,
                                                      ExecutionDnnReductionQuery& query) noexcept;
[[nodiscard]] bool write_execution_dnn_reduction_compute(
    ExecutionPayloadWriter& output, const ExecutionDnnReductionCompute& compute) noexcept;
[[nodiscard]] bool read_execution_dnn_reduction_compute(
    ExecutionPayloadReader& input, ExecutionDnnReductionCompute& compute) noexcept;
[[nodiscard]] bool write_execution_dnn_batch_norm_query(
    ExecutionPayloadWriter& output, const ExecutionDnnBatchNormQuery& query) noexcept;
[[nodiscard]] bool read_execution_dnn_batch_norm_query(ExecutionPayloadReader& input,
                                                       ExecutionDnnBatchNormQuery& query) noexcept;
[[nodiscard]] bool write_execution_dnn_batch_norm_compute(
    ExecutionPayloadWriter& output, const ExecutionDnnBatchNormCompute& compute) noexcept;
[[nodiscard]] bool read_execution_dnn_batch_norm_compute(
    ExecutionPayloadReader& input, ExecutionDnnBatchNormCompute& compute) noexcept;
[[nodiscard]] bool write_execution_dnn_tensor_forward(
    ExecutionPayloadWriter& output, const ExecutionDnnTensorForward& forward) noexcept;
[[nodiscard]] bool read_execution_dnn_tensor_forward(ExecutionPayloadReader& input,
                                                     ExecutionDnnTensorForward& forward) noexcept;
[[nodiscard]] bool write_execution_dnn_convolution_performances(
    ExecutionPayloadWriter& output,
    const ExecutionDnnConvolutionPerformances& performances) noexcept;
[[nodiscard]] bool read_execution_dnn_convolution_performances(
    ExecutionPayloadReader& input, ExecutionDnnConvolutionPerformances& performances) noexcept;

}  // namespace glimmer::control
