#include "glimmer/control/execution_dnn_protocol.h"

#include <algorithm>
#include <bit>

namespace glimmer::control {
namespace {

bool valid_rank(std::int32_t dimensions) noexcept {
    return dimensions >= 0 && static_cast<std::size_t>(dimensions) <= kExecutionDnnMaxDimensions;
}
bool valid_sequence_padding(std::uint32_t bytes) noexcept {
    return bytes == 0 || bytes == 2 || bytes == 4 || bytes == 8;
}
bool write_sequence_lengths(ExecutionPayloadWriter& output, const std::vector<std::int32_t>& values,
                            std::size_t maximum = kExecutionDnnMaxSequenceLengths) noexcept {
    if (values.size() > maximum) return false;
    ExecutionPayloadWriter encoded;
    for (auto value : values)
        if (!encoded.write_u32(std::bit_cast<std::uint32_t>(value))) return false;
    return output.write_blob(encoded.take());
}
bool read_sequence_lengths(ExecutionPayloadReader& input, std::vector<std::int32_t>& values,
                           std::size_t maximum = kExecutionDnnMaxSequenceLengths) {
    std::span<const std::byte> bytes;
    if (!input.read_blob(bytes) || bytes.size() % 4 || bytes.size() / 4 > maximum) return false;
    // read_blob proves the entire byte span before any dynamic allocation.
    values.resize(bytes.size() / 4);
    ExecutionPayloadReader elements(bytes);
    for (auto& value : values) {
        std::uint32_t bits{};
        if (!elements.read_u32(bits)) return false;
        value = std::bit_cast<std::int32_t>(bits);
    }
    return true;
}
template <typename Value>
bool write_sequence_padding(ExecutionPayloadWriter& output, const Value& value) noexcept {
    return valid_sequence_padding(value.padding_bytes) &&
           output.write_blob(std::span{value.padding}.first(value.padding_bytes));
}
template <typename Value>
bool read_sequence_padding(ExecutionPayloadReader& input, Value& value) noexcept {
    std::span<const std::byte> bytes;
    if (!input.read_blob(bytes) ||
        !valid_sequence_padding(static_cast<std::uint32_t>(bytes.size())))
        return false;
    value.padding_bytes = static_cast<std::uint32_t>(bytes.size());
    std::copy(bytes.begin(), bytes.end(), value.padding.begin());
    return true;
}
bool valid_sequence_shape(const ExecutionDnnRnnDataState& value) noexcept {
    return value.presence <= 3 && ((value.presence & 1) || value.lengths.empty()) &&
           ((value.presence & 2) || !value.padding_bytes);
}
bool valid_sequence_shape(const ExecutionDnnSeqDataState& value) noexcept {
    return value.presence <= 15 && ((value.presence & 4) || value.lengths.empty()) &&
           ((value.presence & 8) || !value.padding_bytes);
}
bool valid_library_info_shape(const ExecutionDnnLibraryInfo& query) noexcept {
    if (query.kind == ExecutionDnnLibraryInfoKind::kProperty) return query.has_output <= 1;
    return (query.kind == ExecutionDnnLibraryInfoKind::kVersion ||
            query.kind == ExecutionDnnLibraryInfoKind::kCudartVersion ||
            query.kind == ExecutionDnnLibraryInfoKind::kMaxDeviceVersion) &&
           query.property == 0 && query.has_output == 0;
}
bool valid_rnn_query_shape(const ExecutionDnnRnnQuery& query) noexcept {
    if (query.kind == ExecutionDnnRnnQueryKind::kTempSpace) return !(query.outputs & ~6U);
    if (query.input || query.mode) return false;
    return (query.kind == ExecutionDnnRnnQueryKind::kWeightSpace && !(query.outputs & ~1U)) ||
           (query.kind == ExecutionDnnRnnQueryKind::kAttentionBuffers && !(query.outputs & ~7U));
}
bool valid_rnn_compute_shape(const ExecutionDnnRnnCompute& value) noexcept {
    std::uint32_t unused{};
    switch (value.kind) {
        case ExecutionDnnRnnComputeKind::kForward:
            unused = (1U << 3) | (1U << 4) | (1U << 7) | (1U << 8) | (1U << 11) | (1U << 12);
            break;
        case ExecutionDnnRnnComputeKind::kBackwardData:
            if (value.mode) return false;
            unused = (1U << 1) | (1U << 6) | (1U << 10);
            break;
        case ExecutionDnnRnnComputeKind::kBackwardWeights:
            if (value.descriptors[4]) return false;
            unused = (1U << 3) | (1U << 4) | (1U << 6) | (1U << 7) | (1U << 8) | (1U << 9) |
                     (1U << 10) | (1U << 11) | (1U << 12);
            break;
        default:
            return false;
    }
    for (std::size_t index = 0; index < value.buffers.size(); ++index) {
        const auto& buffer = value.buffers[index];
        if ((!buffer.allocation && buffer.offset) ||
            ((unused & (1U << index)) && buffer.allocation))
            return false;
    }
    return true;
}
bool valid_attn_compute_shape(const ExecutionDnnAttnCompute& value) noexcept {
    if (value.window_presence > 3 || (!(value.window_presence & 1U) && !value.lower.empty()) ||
        (!(value.window_presence & 2U) && !value.upper.empty()) ||
        (value.window_presence == 3 && value.lower.size() != value.upper.size()) ||
        value.lower.size() > kExecutionDnnMaxAttentionWindows ||
        value.upper.size() > kExecutionDnnMaxAttentionWindows)
        return false;
    std::uint32_t unused{};
    switch (value.kind) {
        case ExecutionDnnAttnComputeKind::kForward:
            if (value.mode) return false;
            unused = (1U << 7) | (1U << 8) | (1U << 9) | (1U << 10) | (1U << 12);
            break;
        case ExecutionDnnAttnComputeKind::kBackwardData:
            if (value.mode || value.current) return false;
            unused = (1U << 5) | (1U << 6) | (1U << 12);
            break;
        case ExecutionDnnAttnComputeKind::kBackwardWeights:
            if (value.mode > 1 || value.current || value.window_presence) return false;
            unused =
                (1U << 0) | (1U << 1) | (1U << 5) | (1U << 6) | (1U << 8) | (1U << 9) | (1U << 10);
            break;
        default:
            return false;
    }
    for (std::size_t index = 0; index != value.buffers.size(); ++index) {
        const auto& buffer = value.buffers[index];
        if ((!buffer.allocation && buffer.offset) ||
            ((unused & (1U << index)) && buffer.allocation))
            return false;
    }
    return true;
}
bool valid_attn_window_range(const ExecutionDnnAttnWindowRange& value) noexcept {
    return (value.count || !value.first) && value.first <= kExecutionDnnMaxAttentionWindows &&
           value.count <= kExecutionDnnMaxAttentionWindows - value.first;
}
bool valid_backward_convolution_query_shape(
    const ExecutionDnnBackwardConvolutionQuery& query) noexcept {
    if (query.has_output > 1) return false;
    if (query.kind == ExecutionDnnBackwardConvolutionQueryKind::kDataWorkspace) return true;
    return (query.kind == ExecutionDnnBackwardConvolutionQueryKind::kDataMaxCount ||
            query.kind == ExecutionDnnBackwardConvolutionQueryKind::kFilterMaxCount) &&
           query.metadata == 0 && query.algorithm == 0 &&
           std::ranges::all_of(query.descriptors, [](auto id) { return id == 0; });
}

bool valid_compute_shape(const ExecutionDnnCompute& compute) noexcept {
    return (compute.scalar_bytes == 4 || compute.scalar_bytes == 8) &&
           (compute.kind == ExecutionDnnComputeKind::kActivationForward ||
            (compute.kind == ExecutionDnnComputeKind::kAddTensor && !compute.activation));
}
bool valid_tensor_forward_shape(const ExecutionDnnTensorForward& forward) noexcept {
    return (forward.scalar_bytes == 4 || forward.scalar_bytes == 8) &&
           ((forward.kind == ExecutionDnnTensorForwardKind::kPoolingForward &&
             forward.descriptor != 0 && forward.algorithm == 0 && forward.mode == 0) ||
            (forward.kind == ExecutionDnnTensorForwardKind::kSoftmaxForward &&
             forward.descriptor == 0) ||
            (forward.kind == ExecutionDnnTensorForwardKind::kLrnForward &&
             forward.descriptor != 0 && forward.algorithm == 0) ||
            (forward.kind == ExecutionDnnTensorForwardKind::kTransformTensor &&
             forward.descriptor == 0 && forward.algorithm == 0 && forward.mode == 0));
}
template <std::size_t Size>
bool write_signed_array(ExecutionPayloadWriter& output,
                        const std::array<std::int32_t, Size>& values) noexcept {
    for (const auto value : values)
        if (!output.write_u32(std::bit_cast<std::uint32_t>(value))) return false;
    return true;
}
template <std::size_t Size>
bool read_signed_array(ExecutionPayloadReader& input,
                       std::array<std::int32_t, Size>& values) noexcept {
    for (auto& value : values) {
        std::uint32_t bits{};
        if (!input.read_u32(bits)) return false;
        value = std::bit_cast<std::int32_t>(bits);
    }
    return true;
}
bool valid_query_kind(ExecutionDnnConvolutionQueryKind kind) noexcept {
    return kind == ExecutionDnnConvolutionQueryKind::kWorkspace ||
           kind == ExecutionDnnConvolutionQueryKind::kHeuristic ||
           kind == ExecutionDnnConvolutionQueryKind::kMaxCount;
}
bool valid_reduction_query_kind(ExecutionDnnReductionQueryKind kind) noexcept {
    return kind == ExecutionDnnReductionQueryKind::kIndices ||
           kind == ExecutionDnnReductionQueryKind::kWorkspace;
}
bool valid_reduction_compute_shape(const ExecutionDnnReductionCompute& compute) noexcept {
    return (compute.scalar_bytes == 4 || compute.scalar_bytes == 8) &&
           std::ranges::all_of(compute.buffers, [](const auto& buffer) {
               return buffer.allocation != 0 || buffer.offset == 0;
           });
}
bool valid_batch_norm_query_shape(const ExecutionDnnBatchNormQuery& query) noexcept {
    if (query.kind == ExecutionDnnBatchNormKind::kBackward) return true;
    if (query.kind == ExecutionDnnBatchNormKind::kForward)
        return !query.descriptors[2] && !query.descriptors[4];
    return query.kind == ExecutionDnnBatchNormKind::kReserve &&
           std::all_of(query.descriptors.begin() + 1, query.descriptors.end() - 1,
                       [](auto id) { return id == 0; });
}
bool valid_batch_norm_compute_shape(const ExecutionDnnBatchNormCompute& compute) noexcept {
    if (!valid_batch_norm_query_shape(compute.query) ||
        compute.query.kind == ExecutionDnnBatchNormKind::kReserve ||
        (compute.scalar_bytes != 4 && compute.scalar_bytes != 8) ||
        !std::ranges::all_of(compute.buffers, [](const auto& buffer) {
            return buffer.allocation != 0 || buffer.offset == 0;
        }))
        return false;
    return compute.query.kind == ExecutionDnnBatchNormKind::kBackward ||
           (!compute.buffers[2].allocation && !compute.buffers[4].allocation &&
            compute.coefficients[2] == std::array<std::byte, 8>{} &&
            compute.coefficients[3] == std::array<std::byte, 8>{});
}
bool valid_dropout_shape(const ExecutionDnnDropout& dropout) noexcept {
    return (dropout.kind == ExecutionDnnDropoutKind::kSet ||
            dropout.kind == ExecutionDnnDropoutKind::kRestore) &&
           (dropout.states.allocation != 0 || dropout.states.offset == 0);
}
bool valid_reorder_shape(const ExecutionDnnReorderFilterAndBias& reorder) noexcept {
    return std::ranges::all_of(
               reorder.buffers,
               [](const auto& buffer) { return buffer.allocation != 0 || buffer.offset == 0; }) &&
           (reorder.reorder_bias != 0 ||
            (!reorder.buffers[2].allocation && !reorder.buffers[3].allocation));
}

}  // namespace

bool write_execution_dnn_rnn(ExecutionPayloadWriter& output,
                             const ExecutionDnnRnnState& value) noexcept {
    for (const auto field : {value.algorithm, value.cell_mode, value.bias_mode, value.direction,
                             value.input_mode, value.type, value.math_precision, value.math_type})
        if (!output.write_u32(field)) return false;
    return write_signed_array(output, std::array{value.input_size, value.hidden_size,
                                                 value.projection_size, value.layers}) &&
           output.write_u64(value.dropout) && output.write_u32(value.flags);
}
bool read_execution_dnn_rnn(ExecutionPayloadReader& input, ExecutionDnnRnnState& output) noexcept {
    ExecutionDnnRnnState value;
    for (auto* field : {&value.algorithm, &value.cell_mode, &value.bias_mode, &value.direction,
                        &value.input_mode, &value.type, &value.math_precision, &value.math_type})
        if (!input.read_u32(*field)) return false;
    std::array<std::int32_t, 4> sizes{};
    if (!read_signed_array(input, sizes) || !input.read_u64(value.dropout) ||
        !input.read_u32(value.flags))
        return false;
    value.input_size = sizes[0];
    value.hidden_size = sizes[1];
    value.projection_size = sizes[2];
    value.layers = sizes[3];
    output = value;
    return true;
}
bool write_execution_dnn_attn(ExecutionPayloadWriter& output,
                              const ExecutionDnnAttnState& value) noexcept {
    if (!output.write_u32(value.mode) ||
        !output.write_u32(std::bit_cast<std::uint32_t>(value.heads)) ||
        !output.write_u64(value.scale_bits) || !output.write_u32(value.type) ||
        !output.write_u32(value.precision) || !output.write_u32(value.math_type) ||
        !output.write_u64(value.dropouts[0]) || !output.write_u64(value.dropouts[1]))
        return false;
    return write_signed_array(
        output, std::array{value.query_size, value.key_size, value.value_size,
                           value.query_projection, value.key_projection, value.value_projection,
                           value.output_projection, value.max_query_sequence,
                           value.max_key_sequence, value.max_batch, value.max_beam});
}
bool read_execution_dnn_attn(ExecutionPayloadReader& input,
                             ExecutionDnnAttnState& output) noexcept {
    ExecutionDnnAttnState value;
    std::uint32_t heads{};
    if (!input.read_u32(value.mode) || !input.read_u32(heads) ||
        !input.read_u64(value.scale_bits) || !input.read_u32(value.type) ||
        !input.read_u32(value.precision) || !input.read_u32(value.math_type) ||
        !input.read_u64(value.dropouts[0]) || !input.read_u64(value.dropouts[1]))
        return false;
    value.heads = std::bit_cast<std::int32_t>(heads);
    for (auto* field :
         {&value.query_size, &value.key_size, &value.value_size, &value.query_projection,
          &value.key_projection, &value.value_projection, &value.output_projection,
          &value.max_query_sequence, &value.max_key_sequence, &value.max_batch, &value.max_beam}) {
        std::uint32_t bits{};
        if (!input.read_u32(bits)) return false;
        *field = std::bit_cast<std::int32_t>(bits);
    }
    output = value;
    return true;
}

bool write_execution_dnn_rnn_query(ExecutionPayloadWriter& output,
                                   const ExecutionDnnRnnQuery& value) noexcept {
    return valid_rnn_query_shape(value) &&
           output.write_u32(static_cast<std::uint32_t>(value.kind)) &&
           output.write_u64(value.handle) && output.write_u64(value.metadata) &&
           output.write_u64(value.descriptor) && output.write_u64(value.input) &&
           output.write_u32(value.mode) && output.write_u32(value.outputs);
}
bool read_execution_dnn_rnn_query(ExecutionPayloadReader& input,
                                  ExecutionDnnRnnQuery& output) noexcept {
    ExecutionDnnRnnQuery value;
    std::uint32_t kind{};
    if (!input.read_u32(kind) || !input.read_u64(value.handle) || !input.read_u64(value.metadata) ||
        !input.read_u64(value.descriptor) || !input.read_u64(value.input) ||
        !input.read_u32(value.mode) || !input.read_u32(value.outputs))
        return false;
    value.kind = static_cast<ExecutionDnnRnnQueryKind>(kind);
    if (!valid_rnn_query_shape(value)) return false;
    output = value;
    return true;
}
bool write_execution_dnn_rnn_query_result(ExecutionPayloadWriter& output,
                                          const ExecutionDnnRnnQueryResult& value) noexcept {
    for (const auto size : value.sizes)
        if (!output.write_u64(size)) return false;
    return true;
}
bool read_execution_dnn_rnn_query_result(ExecutionPayloadReader& input,
                                         ExecutionDnnRnnQueryResult& output) noexcept {
    ExecutionDnnRnnQueryResult value;
    for (auto& size : value.sizes)
        if (!input.read_u64(size)) return false;
    output = value;
    return true;
}
bool write_execution_dnn_rnn_clip(ExecutionPayloadWriter& output,
                                  const ExecutionDnnRnnClip& value) noexcept {
    return output.write_u32(value.mode) && output.write_u32(value.nan) &&
           output.write_u64(value.lower_bits) && output.write_u64(value.upper_bits);
}
bool read_execution_dnn_rnn_clip(ExecutionPayloadReader& input,
                                 ExecutionDnnRnnClip& output) noexcept {
    ExecutionDnnRnnClip value;
    if (!input.read_u32(value.mode) || !input.read_u32(value.nan) ||
        !input.read_u64(value.lower_bits) || !input.read_u64(value.upper_bits))
        return false;
    output = value;
    return true;
}

bool write_execution_dnn_rnn_compute(ExecutionPayloadWriter& output,
                                     const ExecutionDnnRnnCompute& value) noexcept {
    if (!valid_rnn_compute_shape(value) ||
        !output.write_u32(static_cast<std::uint32_t>(value.kind)) ||
        !output.write_u32(value.mode) || !output.write_u64(value.handle) ||
        !output.write_u64(value.metadata))
        return false;
    for (const auto descriptor : value.descriptors)
        if (!output.write_u64(descriptor)) return false;
    for (const auto& buffer : value.buffers)
        if (!output.write_u64(buffer.allocation) || !output.write_u64(buffer.offset)) return false;
    return output.write_u64(value.weight_bytes) && output.write_u64(value.workspace_bytes) &&
           output.write_u64(value.reserve_bytes);
}
bool read_execution_dnn_rnn_compute(ExecutionPayloadReader& input,
                                    ExecutionDnnRnnCompute& output) noexcept {
    ExecutionDnnRnnCompute value;
    std::uint32_t kind{};
    if (!input.read_u32(kind) || !input.read_u32(value.mode) || !input.read_u64(value.handle) ||
        !input.read_u64(value.metadata))
        return false;
    value.kind = static_cast<ExecutionDnnRnnComputeKind>(kind);
    for (auto& descriptor : value.descriptors)
        if (!input.read_u64(descriptor)) return false;
    for (auto& buffer : value.buffers)
        if (!input.read_u64(buffer.allocation) || !input.read_u64(buffer.offset)) return false;
    if (!input.read_u64(value.weight_bytes) || !input.read_u64(value.workspace_bytes) ||
        !input.read_u64(value.reserve_bytes) || !valid_rnn_compute_shape(value))
        return false;
    output = value;
    return true;
}
bool write_execution_dnn_build_rnn_dynamic(ExecutionPayloadWriter& output,
                                           const ExecutionDnnBuildRnnDynamic& value) noexcept {
    return output.write_u64(value.handle) && output.write_u64(value.metadata) &&
           output.write_u64(value.descriptor) &&
           output.write_u32(std::bit_cast<std::uint32_t>(value.mini_batch));
}
bool read_execution_dnn_build_rnn_dynamic(ExecutionPayloadReader& input,
                                          ExecutionDnnBuildRnnDynamic& output) noexcept {
    ExecutionDnnBuildRnnDynamic value;
    std::uint32_t batch{};
    if (!input.read_u64(value.handle) || !input.read_u64(value.metadata) ||
        !input.read_u64(value.descriptor) || !input.read_u32(batch))
        return false;
    value.mini_batch = std::bit_cast<std::int32_t>(batch);
    output = value;
    return true;
}

bool write_execution_dnn_rnn_weight_params(ExecutionPayloadWriter& output,
                                           const ExecutionDnnRnnWeightParams& value) noexcept {
    return value.outputs <= 3 && (value.weights.allocation || !value.weights.offset) &&
           output.write_u64(value.handle) && output.write_u64(value.metadata) &&
           output.write_u64(value.descriptor) &&
           output.write_u32(std::bit_cast<std::uint32_t>(value.pseudo_layer)) &&
           output.write_u32(std::bit_cast<std::uint32_t>(value.linear_layer)) &&
           output.write_u64(value.weights.allocation) && output.write_u64(value.weights.offset) &&
           output.write_u64(value.weight_bytes) && output.write_u64(value.destinations[0]) &&
           output.write_u64(value.destinations[1]) && output.write_u32(value.outputs);
}
bool read_execution_dnn_rnn_weight_params(ExecutionPayloadReader& input,
                                          ExecutionDnnRnnWeightParams& output) noexcept {
    ExecutionDnnRnnWeightParams value;
    std::uint32_t pseudo{}, linear{};
    if (!input.read_u64(value.handle) || !input.read_u64(value.metadata) ||
        !input.read_u64(value.descriptor) || !input.read_u32(pseudo) || !input.read_u32(linear) ||
        !input.read_u64(value.weights.allocation) || !input.read_u64(value.weights.offset) ||
        !input.read_u64(value.weight_bytes) || !input.read_u64(value.destinations[0]) ||
        !input.read_u64(value.destinations[1]) || !input.read_u32(value.outputs) ||
        value.outputs > 3 || (!value.weights.allocation && value.weights.offset))
        return false;
    value.pseudo_layer = std::bit_cast<std::int32_t>(pseudo);
    value.linear_layer = std::bit_cast<std::int32_t>(linear);
    output = value;
    return true;
}
bool write_execution_dnn_rnn_weight_params_result(
    ExecutionPayloadWriter& output, const ExecutionDnnRnnWeightParamsResult& value) noexcept {
    if (value.non_null > 3) return false;
    for (std::size_t i = 0; i != value.offsets.size(); ++i)
        if (!(value.non_null & (1U << i)) && value.offsets[i]) return false;
    return output.write_u32(value.non_null) && output.write_u64(value.offsets[0]) &&
           output.write_u64(value.offsets[1]);
}
bool read_execution_dnn_rnn_weight_params_result(
    ExecutionPayloadReader& input, ExecutionDnnRnnWeightParamsResult& output) noexcept {
    ExecutionDnnRnnWeightParamsResult value;
    if (!input.read_u32(value.non_null) || !input.read_u64(value.offsets[0]) ||
        !input.read_u64(value.offsets[1]) || value.non_null > 3)
        return false;
    for (std::size_t i = 0; i != value.offsets.size(); ++i)
        if (!(value.non_null & (1U << i)) && value.offsets[i]) return false;
    output = value;
    return true;
}

bool write_execution_dnn_attn_weights(ExecutionPayloadWriter& output,
                                      const ExecutionDnnAttnWeights& value) noexcept {
    return value.kind <= 7 && value.has_base <= 1 && value.has_output <= 1 &&
           output.write_u64(value.handle) && output.write_u64(value.metadata) &&
           output.write_u64(value.descriptor) && output.write_u32(value.kind) &&
           output.write_u32(value.has_base) && output.write_u64(value.weight_bytes) &&
           output.write_u64(value.destination) && output.write_u32(value.has_output);
}
bool read_execution_dnn_attn_weights(ExecutionPayloadReader& input,
                                     ExecutionDnnAttnWeights& output) noexcept {
    ExecutionDnnAttnWeights value;
    if (!input.read_u64(value.handle) || !input.read_u64(value.metadata) ||
        !input.read_u64(value.descriptor) || !input.read_u32(value.kind) ||
        !input.read_u32(value.has_base) || !input.read_u64(value.weight_bytes) ||
        !input.read_u64(value.destination) || !input.read_u32(value.has_output) || value.kind > 7 ||
        value.has_base > 1 || value.has_output > 1)
        return false;
    output = value;
    return true;
}
bool write_execution_dnn_attn_weights_result(ExecutionPayloadWriter& output,
                                             const ExecutionDnnAttnWeightsResult& value) noexcept {
    return value.non_null <= 1 && (value.non_null || !value.offset) &&
           output.write_u32(value.non_null) && output.write_u64(value.offset);
}
bool read_execution_dnn_attn_weights_result(ExecutionPayloadReader& input,
                                            ExecutionDnnAttnWeightsResult& output) noexcept {
    ExecutionDnnAttnWeightsResult value;
    if (!input.read_u32(value.non_null) || !input.read_u64(value.offset) || value.non_null > 1 ||
        (!value.non_null && value.offset))
        return false;
    output = value;
    return true;
}
bool write_execution_dnn_attn_compute(ExecutionPayloadWriter& output,
                                      const ExecutionDnnAttnCompute& value) noexcept {
    if (!valid_attn_compute_shape(value) ||
        !output.write_u32(static_cast<std::uint32_t>(value.kind)) ||
        !output.write_u32(value.mode) ||
        !output.write_u32(std::bit_cast<std::uint32_t>(value.current)) ||
        !output.write_u32(value.window_presence) || !output.write_u64(value.handle) ||
        !output.write_u64(value.metadata))
        return false;
    for (const auto descriptor : value.descriptors)
        if (!output.write_u64(descriptor)) return false;
    for (const auto& buffer : value.buffers)
        if (!output.write_u64(buffer.allocation) || !output.write_u64(buffer.offset)) return false;
    return output.write_u64(value.weight_bytes) && output.write_u64(value.workspace_bytes) &&
           output.write_u64(value.reserve_bytes) &&
           write_sequence_lengths(output, value.lower, kExecutionDnnMaxAttentionWindows) &&
           write_sequence_lengths(output, value.upper, kExecutionDnnMaxAttentionWindows);
}
bool read_execution_dnn_attn_compute(ExecutionPayloadReader& input,
                                     ExecutionDnnAttnCompute& output) noexcept {
    try {
        ExecutionDnnAttnCompute value;
        std::uint32_t kind{}, current{};
        if (!input.read_u32(kind) || !input.read_u32(value.mode) || !input.read_u32(current) ||
            !input.read_u32(value.window_presence) || !input.read_u64(value.handle) ||
            !input.read_u64(value.metadata))
            return false;
        value.kind = static_cast<ExecutionDnnAttnComputeKind>(kind);
        value.current = std::bit_cast<std::int32_t>(current);
        for (auto& descriptor : value.descriptors)
            if (!input.read_u64(descriptor)) return false;
        for (auto& buffer : value.buffers)
            if (!input.read_u64(buffer.allocation) || !input.read_u64(buffer.offset)) return false;
        if (!input.read_u64(value.weight_bytes) || !input.read_u64(value.workspace_bytes) ||
            !input.read_u64(value.reserve_bytes) ||
            !read_sequence_lengths(input, value.lower, kExecutionDnnMaxAttentionWindows) ||
            !read_sequence_lengths(input, value.upper, kExecutionDnnMaxAttentionWindows) ||
            !valid_attn_compute_shape(value))
            return false;
        output = std::move(value);
        return true;
    } catch (...) {
        return false;
    }
}
bool write_execution_dnn_attn_window_range(ExecutionPayloadWriter& output,
                                           const ExecutionDnnAttnWindowRange& value) noexcept {
    return valid_attn_window_range(value) && output.write_u32(value.first) &&
           output.write_u32(value.count);
}
bool read_execution_dnn_attn_window_range(ExecutionPayloadReader& input,
                                          ExecutionDnnAttnWindowRange& output) noexcept {
    ExecutionDnnAttnWindowRange value;
    if (!input.read_u32(value.first) || !input.read_u32(value.count) ||
        !valid_attn_window_range(value))
        return false;
    output = value;
    return true;
}

bool write_execution_dnn_rnn_data(ExecutionPayloadWriter& output,
                                  const ExecutionDnnRnnDataState& value) noexcept {
    return valid_sequence_shape(value) && output.write_u32(value.type) &&
           output.write_u32(value.layout) &&
           output.write_u32(std::bit_cast<std::uint32_t>(value.max_sequence_length)) &&
           output.write_u32(std::bit_cast<std::uint32_t>(value.batch_size)) &&
           output.write_u32(std::bit_cast<std::uint32_t>(value.vector_size)) &&
           output.write_u32(value.presence) && write_sequence_lengths(output, value.lengths) &&
           write_sequence_padding(output, value);
}
bool read_execution_dnn_rnn_data(ExecutionPayloadReader& input,
                                 ExecutionDnnRnnDataState& output) noexcept {
    try {
        ExecutionDnnRnnDataState value;
        std::uint32_t maximum{}, batch{}, size{};
        if (!input.read_u32(value.type) || !input.read_u32(value.layout) ||
            !input.read_u32(maximum) || !input.read_u32(batch) || !input.read_u32(size) ||
            !input.read_u32(value.presence) || !read_sequence_lengths(input, value.lengths) ||
            !read_sequence_padding(input, value) || !valid_sequence_shape(value))
            return false;
        value.max_sequence_length = std::bit_cast<std::int32_t>(maximum);
        value.batch_size = std::bit_cast<std::int32_t>(batch);
        value.vector_size = std::bit_cast<std::int32_t>(size);
        output = std::move(value);
        return true;
    } catch (...) {
        return false;
    }
}
bool write_execution_dnn_seq_data(ExecutionPayloadWriter& output,
                                  const ExecutionDnnSeqDataState& value) noexcept {
    if (!valid_sequence_shape(value) || !output.write_u32(value.type) ||
        !output.write_u32(std::bit_cast<std::uint32_t>(value.dimensions)) ||
        !output.write_u32(value.presence) || !output.write_u64(value.sequence_count) ||
        !write_signed_array(output, value.sizes))
        return false;
    for (const auto axis : value.axes)
        if (!output.write_u32(axis)) return false;
    return write_sequence_lengths(output, value.lengths) && write_sequence_padding(output, value);
}
bool read_execution_dnn_seq_data(ExecutionPayloadReader& input,
                                 ExecutionDnnSeqDataState& output) noexcept {
    try {
        ExecutionDnnSeqDataState value;
        std::uint32_t rank{};
        if (!input.read_u32(value.type) || !input.read_u32(rank) ||
            !input.read_u32(value.presence) || !input.read_u64(value.sequence_count) ||
            !read_signed_array(input, value.sizes))
            return false;
        for (auto& axis : value.axes)
            if (!input.read_u32(axis)) return false;
        if (!read_sequence_lengths(input, value.lengths) || !read_sequence_padding(input, value) ||
            !valid_sequence_shape(value))
            return false;
        value.dimensions = std::bit_cast<std::int32_t>(rank);
        output = std::move(value);
        return true;
    } catch (...) {
        return false;
    }
}
bool write_execution_dnn_seq_data_query(ExecutionPayloadWriter& output,
                                        const ExecutionDnnSeqDataQuery& value) noexcept {
    return value.outputs <= 127 &&
           output.write_u32(std::bit_cast<std::uint32_t>(value.dimensions_requested)) &&
           output.write_u64(value.lengths_requested) && output.write_u32(value.outputs);
}
bool read_execution_dnn_seq_data_query(ExecutionPayloadReader& input,
                                       ExecutionDnnSeqDataQuery& output) noexcept {
    ExecutionDnnSeqDataQuery value;
    std::uint32_t rank{};
    if (!input.read_u32(rank) || !input.read_u64(value.lengths_requested) ||
        !input.read_u32(value.outputs) || value.outputs > 127)
        return false;
    value.dimensions_requested = std::bit_cast<std::int32_t>(rank);
    output = value;
    return true;
}

bool write_execution_dnn_folded_conv_backward_data_descriptors(
    ExecutionPayloadWriter& output,
    const ExecutionDnnFoldedConvBackwardDataDescriptors& descriptors) noexcept {
    if (!output.write_u64(descriptors.handle) || !output.write_u64(descriptors.metadata))
        return false;
    for (const auto id : descriptors.inputs)
        if (!output.write_u64(id)) return false;
    if (!output.write_u32(descriptors.format)) return false;
    for (const auto id : descriptors.outputs)
        if (!output.write_u64(id)) return false;
    return true;
}
bool read_execution_dnn_folded_conv_backward_data_descriptors(
    ExecutionPayloadReader& input,
    ExecutionDnnFoldedConvBackwardDataDescriptors& descriptors) noexcept {
    ExecutionDnnFoldedConvBackwardDataDescriptors result;
    if (!input.read_u64(result.handle) || !input.read_u64(result.metadata)) return false;
    for (auto& id : result.inputs)
        if (!input.read_u64(id)) return false;
    if (!input.read_u32(result.format)) return false;
    for (auto& id : result.outputs)
        if (!input.read_u64(id)) return false;
    descriptors = result;
    return true;
}

bool write_execution_dnn_backward_convolution_query(
    ExecutionPayloadWriter& output, const ExecutionDnnBackwardConvolutionQuery& query) noexcept {
    if (!valid_backward_convolution_query_shape(query) ||
        !output.write_u32(static_cast<std::uint32_t>(query.kind)) ||
        !output.write_u64(query.handle) || !output.write_u64(query.metadata))
        return false;
    for (const auto id : query.descriptors)
        if (!output.write_u64(id)) return false;
    return output.write_u32(query.algorithm) && output.write_u32(query.has_output);
}
bool read_execution_dnn_backward_convolution_query(
    ExecutionPayloadReader& input, ExecutionDnnBackwardConvolutionQuery& query) noexcept {
    ExecutionDnnBackwardConvolutionQuery result;
    std::uint32_t kind{};
    if (!input.read_u32(kind) || !input.read_u64(result.handle) || !input.read_u64(result.metadata))
        return false;
    for (auto& id : result.descriptors)
        if (!input.read_u64(id)) return false;
    if (!input.read_u32(result.algorithm) || !input.read_u32(result.has_output)) return false;
    result.kind = static_cast<ExecutionDnnBackwardConvolutionQueryKind>(kind);
    if (!valid_backward_convolution_query_shape(result)) return false;
    query = result;
    return true;
}

bool write_execution_dnn_library_info(ExecutionPayloadWriter& output,
                                      const ExecutionDnnLibraryInfo& query) noexcept {
    return valid_library_info_shape(query) &&
           output.write_u32(static_cast<std::uint32_t>(query.kind)) &&
           output.write_u32(query.property) && output.write_u32(query.has_output);
}
bool read_execution_dnn_library_info(ExecutionPayloadReader& input,
                                     ExecutionDnnLibraryInfo& query) noexcept {
    ExecutionDnnLibraryInfo result;
    std::uint32_t kind{};
    if (!input.read_u32(kind) || !input.read_u32(result.property) ||
        !input.read_u32(result.has_output))
        return false;
    result.kind = static_cast<ExecutionDnnLibraryInfoKind>(kind);
    if (!valid_library_info_shape(result)) return false;
    query = result;
    return true;
}

bool write_execution_dnn_reorder_filter_and_bias(
    ExecutionPayloadWriter& output, const ExecutionDnnReorderFilterAndBias& reorder) noexcept {
    if (!valid_reorder_shape(reorder) || !output.write_u64(reorder.handle) ||
        !output.write_u64(reorder.metadata) || !output.write_u64(reorder.filter) ||
        !output.write_u32(reorder.reorder) ||
        !output.write_u32(std::bit_cast<std::uint32_t>(reorder.reorder_bias)))
        return false;
    for (const auto& buffer : reorder.buffers)
        if (!output.write_u64(buffer.allocation) || !output.write_u64(buffer.offset)) return false;
    return true;
}
bool read_execution_dnn_reorder_filter_and_bias(
    ExecutionPayloadReader& input, ExecutionDnnReorderFilterAndBias& reorder) noexcept {
    ExecutionDnnReorderFilterAndBias result;
    std::uint32_t flag{};
    if (!input.read_u64(result.handle) || !input.read_u64(result.metadata) ||
        !input.read_u64(result.filter) || !input.read_u32(result.reorder) || !input.read_u32(flag))
        return false;
    result.reorder_bias = std::bit_cast<std::int32_t>(flag);
    for (auto& buffer : result.buffers)
        if (!input.read_u64(buffer.allocation) || !input.read_u64(buffer.offset)) return false;
    if (!valid_reorder_shape(result)) return false;
    reorder = result;
    return true;
}

bool write_execution_dnn_op_tensor(ExecutionPayloadWriter& output,
                                   const ExecutionDnnOpTensor& operation) noexcept {
    return output.write_u32(operation.operation) && output.write_u32(operation.type) &&
           output.write_u32(operation.nan);
}
bool read_execution_dnn_op_tensor(ExecutionPayloadReader& input,
                                  ExecutionDnnOpTensor& operation) noexcept {
    ExecutionDnnOpTensor result;
    if (!input.read_u32(result.operation) || !input.read_u32(result.type) ||
        !input.read_u32(result.nan))
        return false;
    operation = result;
    return true;
}
bool write_execution_dnn_tensor_transform(ExecutionPayloadWriter& output,
                                          const ExecutionDnnTensorTransform& transform) noexcept {
    if (transform.dimensions > kExecutionDnnMaxDimensions || transform.presence > 7 ||
        !output.write_u32(transform.dimensions) || !output.write_u32(transform.format) ||
        !output.write_u32(transform.direction) || !output.write_u32(transform.presence) ||
        !write_signed_array(output, transform.before) ||
        !write_signed_array(output, transform.after))
        return false;
    for (const auto value : transform.fold)
        if (!output.write_u32(value)) return false;
    return true;
}
bool read_execution_dnn_tensor_transform(ExecutionPayloadReader& input,
                                         ExecutionDnnTensorTransform& transform) noexcept {
    ExecutionDnnTensorTransform result;
    if (!input.read_u32(result.dimensions) || result.dimensions > kExecutionDnnMaxDimensions ||
        !input.read_u32(result.format) || !input.read_u32(result.direction) ||
        !input.read_u32(result.presence) || result.presence > 7 ||
        !read_signed_array(input, result.before) || !read_signed_array(input, result.after))
        return false;
    for (auto& value : result.fold)
        if (!input.read_u32(value)) return false;
    transform = result;
    return true;
}

bool write_execution_dnn_dropout(ExecutionPayloadWriter& output,
                                 const ExecutionDnnDropout& dropout) noexcept {
    return valid_dropout_shape(dropout) && output.write_u64(dropout.metadata) &&
           output.write_u64(dropout.descriptor) && output.write_u64(dropout.handle) &&
           output.write_u64(dropout.states.allocation) && output.write_u64(dropout.states.offset) &&
           output.write_u64(dropout.state_bytes) && output.write_u64(dropout.seed) &&
           output.write_u32(dropout.probability_bits) &&
           output.write_u32(static_cast<std::uint32_t>(dropout.kind));
}
bool read_execution_dnn_dropout(ExecutionPayloadReader& input,
                                ExecutionDnnDropout& dropout) noexcept {
    ExecutionDnnDropout result;
    std::uint32_t kind{};
    if (!input.read_u64(result.metadata) || !input.read_u64(result.descriptor) ||
        !input.read_u64(result.handle) || !input.read_u64(result.states.allocation) ||
        !input.read_u64(result.states.offset) || !input.read_u64(result.state_bytes) ||
        !input.read_u64(result.seed) || !input.read_u32(result.probability_bits) ||
        !input.read_u32(kind))
        return false;
    result.kind = static_cast<ExecutionDnnDropoutKind>(kind);
    if (!valid_dropout_shape(result)) return false;
    dropout = result;
    return true;
}

bool write_execution_dnn_batch_norm_query(ExecutionPayloadWriter& output,
                                          const ExecutionDnnBatchNormQuery& query) noexcept {
    if (!valid_batch_norm_query_shape(query) ||
        !output.write_u32(static_cast<std::uint32_t>(query.kind)) ||
        !output.write_u32(query.mode) || !output.write_u32(query.operation) ||
        !output.write_u64(query.handle) || !output.write_u64(query.metadata))
        return false;
    for (const auto descriptor : query.descriptors)
        if (!output.write_u64(descriptor)) return false;
    return true;
}
bool read_execution_dnn_batch_norm_query(ExecutionPayloadReader& input,
                                         ExecutionDnnBatchNormQuery& query) noexcept {
    ExecutionDnnBatchNormQuery result;
    std::uint32_t kind{};
    if (!input.read_u32(kind) || kind < 1 || kind > 3 || !input.read_u32(result.mode) ||
        !input.read_u32(result.operation) || !input.read_u64(result.handle) ||
        !input.read_u64(result.metadata))
        return false;
    result.kind = static_cast<ExecutionDnnBatchNormKind>(kind);
    for (auto& descriptor : result.descriptors)
        if (!input.read_u64(descriptor)) return false;
    if (!valid_batch_norm_query_shape(result)) return false;
    query = result;
    return true;
}
bool write_execution_dnn_batch_norm_compute(ExecutionPayloadWriter& output,
                                            const ExecutionDnnBatchNormCompute& compute) noexcept {
    if (!valid_batch_norm_compute_shape(compute) ||
        !write_execution_dnn_batch_norm_query(output, compute.query))
        return false;
    for (const auto buffer : compute.buffers)
        if (!output.write_u64(buffer.allocation) || !output.write_u64(buffer.offset)) return false;
    if (!output.write_u64(compute.workspace_bytes) || !output.write_u64(compute.reserve_bytes) ||
        !output.write_u64(compute.epsilon_bits) || !output.write_u64(compute.average_factor_bits) ||
        !output.write_u32(compute.scalar_bytes))
        return false;
    const std::size_t count = compute.query.kind == ExecutionDnnBatchNormKind::kForward ? 2 : 4;
    for (std::size_t i = 0; i != count; ++i) {
        for (std::size_t offset = 0; offset != compute.scalar_bytes; offset += 4) {
            std::uint32_t bits{};
            for (std::size_t byte = 0; byte != 4; ++byte)
                bits |= std::to_integer<std::uint32_t>(compute.coefficients[i][offset + byte])
                        << (8 * byte);
            if (!output.write_u32(bits)) return false;
        }
    }
    return true;
}
bool read_execution_dnn_batch_norm_compute(ExecutionPayloadReader& input,
                                           ExecutionDnnBatchNormCompute& compute) noexcept {
    ExecutionDnnBatchNormCompute result;
    if (!read_execution_dnn_batch_norm_query(input, result.query) ||
        result.query.kind == ExecutionDnnBatchNormKind::kReserve)
        return false;
    for (auto& buffer : result.buffers)
        if (!input.read_u64(buffer.allocation) || !input.read_u64(buffer.offset)) return false;
    if (!input.read_u64(result.workspace_bytes) || !input.read_u64(result.reserve_bytes) ||
        !input.read_u64(result.epsilon_bits) || !input.read_u64(result.average_factor_bits) ||
        !input.read_u32(result.scalar_bytes) || !valid_batch_norm_compute_shape(result))
        return false;
    const std::size_t count = result.query.kind == ExecutionDnnBatchNormKind::kForward ? 2 : 4;
    for (std::size_t i = 0; i != count; ++i) {
        for (std::size_t offset = 0; offset != result.scalar_bytes; offset += 4) {
            std::uint32_t bits{};
            if (!input.read_u32(bits)) return false;
            for (std::size_t byte = 0; byte != 4; ++byte)
                result.coefficients[i][offset + byte] = static_cast<std::byte>(bits >> (8 * byte));
        }
    }
    compute = result;
    return true;
}

bool write_execution_dnn_reduction(ExecutionPayloadWriter& output,
                                   const ExecutionDnnReduction& reduction) noexcept {
    return output.write_u32(reduction.operation) && output.write_u32(reduction.type) &&
           output.write_u32(reduction.nan) && output.write_u32(reduction.indices) &&
           output.write_u32(reduction.indices_type);
}
bool read_execution_dnn_reduction(ExecutionPayloadReader& input,
                                  ExecutionDnnReduction& reduction) noexcept {
    ExecutionDnnReduction result;
    if (!input.read_u32(result.operation) || !input.read_u32(result.type) ||
        !input.read_u32(result.nan) || !input.read_u32(result.indices) ||
        !input.read_u32(result.indices_type))
        return false;
    reduction = result;
    return true;
}
bool write_execution_dnn_reduction_query(ExecutionPayloadWriter& output,
                                         const ExecutionDnnReductionQuery& query) noexcept {
    if (!valid_reduction_query_kind(query.kind) ||
        !output.write_u32(static_cast<std::uint32_t>(query.kind)) ||
        !output.write_u64(query.handle) || !output.write_u64(query.metadata))
        return false;
    for (const auto descriptor : query.descriptors)
        if (!output.write_u64(descriptor)) return false;
    return true;
}
bool read_execution_dnn_reduction_query(ExecutionPayloadReader& input,
                                        ExecutionDnnReductionQuery& query) noexcept {
    ExecutionDnnReductionQuery result;
    std::uint32_t kind{};
    if (!input.read_u32(kind) ||
        (kind != static_cast<std::uint32_t>(ExecutionDnnReductionQueryKind::kIndices) &&
         kind != static_cast<std::uint32_t>(ExecutionDnnReductionQueryKind::kWorkspace)))
        return false;
    result.kind = static_cast<ExecutionDnnReductionQueryKind>(kind);
    if (!input.read_u64(result.handle) || !input.read_u64(result.metadata)) return false;
    for (auto& descriptor : result.descriptors)
        if (!input.read_u64(descriptor)) return false;
    query = result;
    return true;
}
bool write_execution_dnn_reduction_compute(ExecutionPayloadWriter& output,
                                           const ExecutionDnnReductionCompute& compute) noexcept {
    if (!valid_reduction_compute_shape(compute) || !output.write_u64(compute.handle) ||
        !output.write_u64(compute.metadata))
        return false;
    for (const auto descriptor : compute.descriptors)
        if (!output.write_u64(descriptor)) return false;
    for (const auto buffer : compute.buffers)
        if (!output.write_u64(buffer.allocation) || !output.write_u64(buffer.offset)) return false;
    return output.write_u64(compute.indices_bytes) && output.write_u64(compute.workspace_bytes) &&
           output.write_u32(compute.scalar_bytes) &&
           output.write_blob(std::span{compute.alpha}.first(compute.scalar_bytes)) &&
           output.write_blob(std::span{compute.beta}.first(compute.scalar_bytes));
}
bool read_execution_dnn_reduction_compute(ExecutionPayloadReader& input,
                                          ExecutionDnnReductionCompute& compute) noexcept {
    ExecutionDnnReductionCompute result;
    if (!input.read_u64(result.handle) || !input.read_u64(result.metadata)) return false;
    for (auto& descriptor : result.descriptors)
        if (!input.read_u64(descriptor)) return false;
    for (auto& buffer : result.buffers)
        if (!input.read_u64(buffer.allocation) || !input.read_u64(buffer.offset)) return false;
    if (!input.read_u64(result.indices_bytes) || !input.read_u64(result.workspace_bytes) ||
        !input.read_u32(result.scalar_bytes) || !valid_reduction_compute_shape(result))
        return false;
    std::span<const std::byte> alpha, beta;
    if (!input.read_blob(alpha) || alpha.size() != result.scalar_bytes || !input.read_blob(beta) ||
        beta.size() != result.scalar_bytes)
        return false;
    std::copy(alpha.begin(), alpha.end(), result.alpha.begin());
    std::copy(beta.begin(), beta.end(), result.beta.begin());
    compute = result;
    return true;
}

bool write_execution_dnn_tensor(ExecutionPayloadWriter& output,
                                const ExecutionDnnTensor& tensor) noexcept {
    if (!valid_rank(tensor.dimensions) || !output.write_u32(tensor.type) ||
        !output.write_u32(tensor.format) ||
        !output.write_u32(std::bit_cast<std::uint32_t>(tensor.dimensions)))
        return false;
    for (const auto size : tensor.sizes)
        if (!output.write_u32(std::bit_cast<std::uint32_t>(size))) return false;
    for (const auto stride : tensor.strides)
        if (!output.write_u32(std::bit_cast<std::uint32_t>(stride))) return false;
    return true;
}

bool read_execution_dnn_tensor(ExecutionPayloadReader& input, ExecutionDnnTensor& tensor) noexcept {
    ExecutionDnnTensor result;
    std::uint32_t field{};
    if (!input.read_u32(result.type) || !input.read_u32(result.format) || !input.read_u32(field))
        return false;
    result.dimensions = std::bit_cast<std::int32_t>(field);
    if (!valid_rank(result.dimensions)) return false;
    for (auto& size : result.sizes) {
        if (!input.read_u32(field)) return false;
        size = std::bit_cast<std::int32_t>(field);
    }
    for (auto& stride : result.strides) {
        if (!input.read_u32(field)) return false;
        stride = std::bit_cast<std::int32_t>(field);
    }
    tensor = result;
    return true;
}

bool write_execution_dnn_activation(ExecutionPayloadWriter& output,
                                    const ExecutionDnnActivation& activation) noexcept {
    return output.write_u32(activation.mode) && output.write_u32(activation.nan) &&
           output.write_u64(activation.coefficient_bits);
}

bool read_execution_dnn_activation(ExecutionPayloadReader& input,
                                   ExecutionDnnActivation& activation) noexcept {
    ExecutionDnnActivation result;
    if (!input.read_u32(result.mode) || !input.read_u32(result.nan) ||
        !input.read_u64(result.coefficient_bits))
        return false;
    activation = result;
    return true;
}

bool write_execution_dnn_compute(ExecutionPayloadWriter& output,
                                 const ExecutionDnnCompute& compute) noexcept {
    if (!valid_compute_shape(compute) ||
        !output.write_u32(static_cast<std::uint32_t>(compute.kind)) ||
        !output.write_u64(compute.handle) || !output.write_u64(compute.metadata) ||
        !output.write_u64(compute.activation))
        return false;
    for (const auto tensor : compute.tensors)
        if (!output.write_u64(tensor)) return false;
    for (const auto buffer : compute.buffers)
        if (!output.write_u64(buffer.allocation) || !output.write_u64(buffer.offset)) return false;
    return output.write_u32(compute.scalar_bytes) &&
           output.write_blob(std::span{compute.alpha}.first(compute.scalar_bytes)) &&
           output.write_blob(std::span{compute.beta}.first(compute.scalar_bytes));
}

bool read_execution_dnn_compute(ExecutionPayloadReader& input,
                                ExecutionDnnCompute& compute) noexcept {
    ExecutionDnnCompute result;
    std::uint32_t kind{};
    if (!input.read_u32(kind) ||
        (kind != static_cast<std::uint32_t>(ExecutionDnnComputeKind::kActivationForward) &&
         kind != static_cast<std::uint32_t>(ExecutionDnnComputeKind::kAddTensor)) ||
        !input.read_u64(result.handle) || !input.read_u64(result.metadata) ||
        !input.read_u64(result.activation))
        return false;
    result.kind = static_cast<ExecutionDnnComputeKind>(kind);
    for (auto& tensor : result.tensors)
        if (!input.read_u64(tensor)) return false;
    for (auto& buffer : result.buffers)
        if (!input.read_u64(buffer.allocation) || !input.read_u64(buffer.offset)) return false;
    if (!input.read_u32(result.scalar_bytes) || !valid_compute_shape(result)) return false;
    std::span<const std::byte> alpha, beta;
    if (!input.read_blob(alpha) || alpha.size() != result.scalar_bytes || !input.read_blob(beta) ||
        beta.size() != result.scalar_bytes)
        return false;
    std::copy(alpha.begin(), alpha.end(), result.alpha.begin());
    std::copy(beta.begin(), beta.end(), result.beta.begin());
    compute = result;
    return true;
}

bool write_execution_dnn_filter(ExecutionPayloadWriter& output,
                                const ExecutionDnnFilter& filter) noexcept {
    return valid_rank(filter.dimensions) && output.write_u32(filter.type) &&
           output.write_u32(filter.format) &&
           output.write_u32(std::bit_cast<std::uint32_t>(filter.dimensions)) &&
           write_signed_array(output, filter.sizes);
}
bool read_execution_dnn_filter(ExecutionPayloadReader& input, ExecutionDnnFilter& filter) noexcept {
    ExecutionDnnFilter result;
    std::uint32_t rank{};
    if (!input.read_u32(result.type) || !input.read_u32(result.format) || !input.read_u32(rank))
        return false;
    result.dimensions = std::bit_cast<std::int32_t>(rank);
    if (!valid_rank(result.dimensions) || !read_signed_array(input, result.sizes)) return false;
    filter = result;
    return true;
}
bool write_execution_dnn_convolution(ExecutionPayloadWriter& output,
                                     const ExecutionDnnConvolutionState& value) noexcept {
    return valid_rank(value.dimensions) && output.write_u32(value.mode) &&
           output.write_u32(value.type) &&
           output.write_u32(std::bit_cast<std::uint32_t>(value.dimensions)) &&
           write_signed_array(output, value.padding) && write_signed_array(output, value.strides) &&
           write_signed_array(output, value.dilation);
}
bool read_execution_dnn_convolution(ExecutionPayloadReader& input,
                                    ExecutionDnnConvolutionState& convolution) noexcept {
    ExecutionDnnConvolutionState result;
    std::uint32_t rank{};
    if (!input.read_u32(result.mode) || !input.read_u32(result.type) || !input.read_u32(rank))
        return false;
    result.dimensions = std::bit_cast<std::int32_t>(rank);
    if (!valid_rank(result.dimensions) || !read_signed_array(input, result.padding) ||
        !read_signed_array(input, result.strides) || !read_signed_array(input, result.dilation))
        return false;
    convolution = result;
    return true;
}
bool write_execution_dnn_pooling(ExecutionPayloadWriter& output,
                                 const ExecutionDnnPooling& pooling) noexcept {
    return valid_rank(pooling.dimensions) && output.write_u32(pooling.mode) &&
           output.write_u32(pooling.nan) &&
           output.write_u32(std::bit_cast<std::uint32_t>(pooling.dimensions)) &&
           write_signed_array(output, pooling.window) &&
           write_signed_array(output, pooling.padding) &&
           write_signed_array(output, pooling.strides);
}
bool read_execution_dnn_pooling(ExecutionPayloadReader& input,
                                ExecutionDnnPooling& pooling) noexcept {
    ExecutionDnnPooling result;
    std::uint32_t rank{};
    if (!input.read_u32(result.mode) || !input.read_u32(result.nan) || !input.read_u32(rank))
        return false;
    result.dimensions = std::bit_cast<std::int32_t>(rank);
    if (!valid_rank(result.dimensions) || !read_signed_array(input, result.window) ||
        !read_signed_array(input, result.padding) || !read_signed_array(input, result.strides))
        return false;
    pooling = result;
    return true;
}
bool write_execution_dnn_tensor_forward(ExecutionPayloadWriter& output,
                                        const ExecutionDnnTensorForward& forward) noexcept {
    if (!valid_tensor_forward_shape(forward) ||
        !output.write_u32(static_cast<std::uint32_t>(forward.kind)) ||
        !output.write_u64(forward.handle) || !output.write_u64(forward.metadata) ||
        !output.write_u64(forward.descriptor))
        return false;
    for (const auto tensor : forward.tensors)
        if (!output.write_u64(tensor)) return false;
    for (const auto buffer : forward.buffers)
        if (!output.write_u64(buffer.allocation) || !output.write_u64(buffer.offset)) return false;
    return output.write_u32(forward.algorithm) && output.write_u32(forward.mode) &&
           output.write_u32(forward.scalar_bytes) &&
           output.write_blob(std::span{forward.alpha}.first(forward.scalar_bytes)) &&
           output.write_blob(std::span{forward.beta}.first(forward.scalar_bytes));
}
bool read_execution_dnn_tensor_forward(ExecutionPayloadReader& input,
                                       ExecutionDnnTensorForward& forward) noexcept {
    ExecutionDnnTensorForward result;
    std::uint32_t kind{};
    if (!input.read_u32(kind) ||
        (kind != static_cast<std::uint32_t>(ExecutionDnnTensorForwardKind::kPoolingForward) &&
         kind != static_cast<std::uint32_t>(ExecutionDnnTensorForwardKind::kSoftmaxForward) &&
         kind != static_cast<std::uint32_t>(ExecutionDnnTensorForwardKind::kLrnForward) &&
         kind != static_cast<std::uint32_t>(ExecutionDnnTensorForwardKind::kTransformTensor)) ||
        !input.read_u64(result.handle) || !input.read_u64(result.metadata) ||
        !input.read_u64(result.descriptor))
        return false;
    result.kind = static_cast<ExecutionDnnTensorForwardKind>(kind);
    for (auto& tensor : result.tensors)
        if (!input.read_u64(tensor)) return false;
    for (auto& buffer : result.buffers)
        if (!input.read_u64(buffer.allocation) || !input.read_u64(buffer.offset)) return false;
    if (!input.read_u32(result.algorithm) || !input.read_u32(result.mode) ||
        !input.read_u32(result.scalar_bytes) || !valid_tensor_forward_shape(result))
        return false;
    std::span<const std::byte> alpha, beta;
    if (!input.read_blob(alpha) || alpha.size() != result.scalar_bytes || !input.read_blob(beta) ||
        beta.size() != result.scalar_bytes)
        return false;
    std::copy(alpha.begin(), alpha.end(), result.alpha.begin());
    std::copy(beta.begin(), beta.end(), result.beta.begin());
    forward = result;
    return true;
}
bool write_execution_dnn_lrn(ExecutionPayloadWriter& output, const ExecutionDnnLrn& lrn) noexcept {
    return output.write_u32(lrn.n) && output.write_u64(lrn.alpha_bits) &&
           output.write_u64(lrn.beta_bits) && output.write_u64(lrn.k_bits);
}
bool read_execution_dnn_lrn(ExecutionPayloadReader& input, ExecutionDnnLrn& lrn) noexcept {
    ExecutionDnnLrn result;
    if (!input.read_u32(result.n) || !input.read_u64(result.alpha_bits) ||
        !input.read_u64(result.beta_bits) || !input.read_u64(result.k_bits))
        return false;
    lrn = result;
    return true;
}
bool write_execution_dnn_convolution_query(ExecutionPayloadWriter& output,
                                           const ExecutionDnnConvolutionQuery& query) noexcept {
    if (!valid_query_kind(query.kind) ||
        !output.write_u32(static_cast<std::uint32_t>(query.kind)) ||
        !output.write_u64(query.handle) || !output.write_u64(query.metadata))
        return false;
    for (const auto descriptor : query.descriptors)
        if (!output.write_u64(descriptor)) return false;
    return output.write_u32(query.algorithm) && output.write_u32(query.requested);
}
bool read_execution_dnn_convolution_query(ExecutionPayloadReader& input,
                                          ExecutionDnnConvolutionQuery& query) noexcept {
    ExecutionDnnConvolutionQuery result;
    std::uint32_t kind{};
    if (!input.read_u32(kind)) return false;
    result.kind = static_cast<ExecutionDnnConvolutionQueryKind>(kind);
    if (!valid_query_kind(result.kind) || !input.read_u64(result.handle) ||
        !input.read_u64(result.metadata))
        return false;
    for (auto& descriptor : result.descriptors)
        if (!input.read_u64(descriptor)) return false;
    if (!input.read_u32(result.algorithm) || !input.read_u32(result.requested)) return false;
    query = result;
    return true;
}
bool write_execution_dnn_convolution_forward(
    ExecutionPayloadWriter& output, const ExecutionDnnConvolutionForward& forward) noexcept {
    if ((forward.scalar_bytes != 4 && forward.scalar_bytes != 8) ||
        !output.write_u64(forward.handle) || !output.write_u64(forward.metadata))
        return false;
    for (const auto descriptor : forward.descriptors)
        if (!output.write_u64(descriptor)) return false;
    for (const auto& buffer : forward.buffers)
        if (!output.write_u64(buffer.allocation) || !output.write_u64(buffer.offset)) return false;
    return output.write_u32(forward.algorithm) && output.write_u64(forward.workspace_bytes) &&
           output.write_u32(forward.scalar_bytes) &&
           output.write_blob(std::span{forward.alpha}.first(forward.scalar_bytes)) &&
           output.write_blob(std::span{forward.beta}.first(forward.scalar_bytes));
}
bool read_execution_dnn_convolution_forward(ExecutionPayloadReader& input,
                                            ExecutionDnnConvolutionForward& forward) noexcept {
    ExecutionDnnConvolutionForward result;
    if (!input.read_u64(result.handle) || !input.read_u64(result.metadata)) return false;
    for (auto& descriptor : result.descriptors)
        if (!input.read_u64(descriptor)) return false;
    for (auto& buffer : result.buffers)
        if (!input.read_u64(buffer.allocation) || !input.read_u64(buffer.offset)) return false;
    if (!input.read_u32(result.algorithm) || !input.read_u64(result.workspace_bytes) ||
        !input.read_u32(result.scalar_bytes) ||
        (result.scalar_bytes != 4 && result.scalar_bytes != 8))
        return false;
    std::span<const std::byte> alpha, beta;
    if (!input.read_blob(alpha) || alpha.size() != result.scalar_bytes || !input.read_blob(beta) ||
        beta.size() != result.scalar_bytes)
        return false;
    std::copy(alpha.begin(), alpha.end(), result.alpha.begin());
    std::copy(beta.begin(), beta.end(), result.beta.begin());
    forward = result;
    return true;
}
bool write_execution_dnn_convolution_find(ExecutionPayloadWriter& output,
                                          const ExecutionDnnConvolutionFind& find) noexcept {
    if (!output.write_u64(find.handle) || !output.write_u64(find.metadata)) return false;
    for (const auto descriptor : find.descriptors)
        if (!output.write_u64(descriptor)) return false;
    return output.write_u32(find.requested);
}
bool read_execution_dnn_convolution_find(ExecutionPayloadReader& input,
                                         ExecutionDnnConvolutionFind& find) noexcept {
    ExecutionDnnConvolutionFind result;
    if (!input.read_u64(result.handle) || !input.read_u64(result.metadata)) return false;
    for (auto& descriptor : result.descriptors)
        if (!input.read_u64(descriptor)) return false;
    if (!input.read_u32(result.requested)) return false;
    find = result;
    return true;
}
bool write_execution_dnn_convolution_performances(
    ExecutionPayloadWriter& output,
    const ExecutionDnnConvolutionPerformances& performances) noexcept {
    if (performances.count > performances.values.size() || !output.write_u32(performances.count))
        return false;
    for (const auto& value : std::span{performances.values}.first(performances.count))
        if (!output.write_u32(value.algorithm) || !output.write_u32(value.status) ||
            !output.write_u32(value.time_bits) || !output.write_u64(value.memory) ||
            !output.write_u32(value.determinism) || !output.write_u32(value.math) ||
            !write_signed_array(output, value.reserved))
            return false;
    return true;
}
bool read_execution_dnn_convolution_performances(
    ExecutionPayloadReader& input, ExecutionDnnConvolutionPerformances& performances) noexcept {
    ExecutionDnnConvolutionPerformances result;
    if (!input.read_u32(result.count) || result.count > result.values.size()) return false;
    for (auto& value : std::span{result.values}.first(result.count))
        if (!input.read_u32(value.algorithm) || !input.read_u32(value.status) ||
            !input.read_u32(value.time_bits) || !input.read_u64(value.memory) ||
            !input.read_u32(value.determinism) || !input.read_u32(value.math) ||
            !read_signed_array(input, value.reserved))
            return false;
    performances = result;
    return true;
}

}  // namespace glimmer::control
