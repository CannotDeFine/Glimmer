// Included after rnn_data_frontend.inc for strict metadata mutation receipts.
namespace {
using DnnAttnHandle = cudnnAttnStruct*;

template <typename Function>
cudnnStatus_t with_dnn_dropout_dependencies(const void* parent, ExecutionDnnObjectKind kind,
                                            const std::array<cudnnDropoutDescriptor_t, 2>& dropouts,
                                            Function&& function) noexcept {
    return dnn_boundary([&](State& shared, Exclusion& exclusion) {
        auto* metadata = shared.dnn_metadata_context;
        if (!metadata || !metadata->connection || !metadata->is_alive || metadata->is_uncertain ||
            metadata->is_closing)
            return CUDNN_STATUS_NOT_INITIALIZED;
        if (metadata->active_calls >= kMaxHandles - 1 ||
            metadata->active_mutations >= kMaxHandles - 1)
            return dnn_profile::host_allocation_status();
        CounterPin owner(metadata->active_calls, exclusion);
        CounterPin work(metadata->active_mutations, exclusion);
        const std::array<const void*, 3> handles{parent, dropouts[0], dropouts[1]};
        std::array<std::shared_ptr<DnnEntry>, 3> descriptors;
        for (;;) {
            bool is_waiting{};
            for (std::size_t i = 0; i != descriptors.size(); ++i) {
                if (i && !handles[i]) {
                    descriptors[i].reset();
                    continue;
                }
                auto& descriptor = descriptors[i];
                descriptor = find_dnn_metadata(shared, handles[i],
                                               i ? ExecutionDnnObjectKind::kDropout : kind);
                if (!descriptor || descriptor->context != metadata ||
                    !valid_dnn_metadata(*descriptor))
                    return CUDNN_STATUS_NOT_INITIALIZED;
                if (descriptor->active_calls >= kMaxHandles - 1)
                    return dnn_profile::host_allocation_status();
                is_waiting |= descriptor->is_retiring || (i == 0 && descriptor->active_calls);
            }
            if (!is_waiting) break;
            // Acquire no partial object pins/flags while waiting. Revalidate
            // every dependency and its owner after the frontend lock is yielded.
            if (!yield_frontend(exclusion)) return CUDNN_STATUS_NOT_INITIALIZED;
        }
        FlagPin mutation(descriptors[0]->is_retiring, exclusion);
        std::array<std::optional<CounterPin>, 3> pins;
        for (std::size_t i = 0; i != descriptors.size(); ++i) {
            if (!descriptors[i] || (i == 2 && descriptors[2] == descriptors[1])) continue;
            pins[i].emplace(descriptors[i]->active_calls, exclusion);
        }
        // These borrows last only through the RPC. Durable dependency edges
        // belong to the registry and must not freeze dropout configuration.
        return function(shared, exclusion, descriptors);
    });
}
}  // namespace

extern "C" cudnnStatus_t CUDNNWINAPI cudnnCreateRNNDescriptor(cudnnRNNDescriptor_t* output) {
    return create_dnn_metadata(output, ExecutionDnnObjectKind::kRnn, true);
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnDestroyRNNDescriptor(cudnnRNNDescriptor_t descriptor) {
    return destroy_dnn_metadata(descriptor, ExecutionDnnObjectKind::kRnn, true);
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnCreateAttnDescriptor(DnnAttnHandle* output) {
    return create_dnn_metadata(output, ExecutionDnnObjectKind::kAttn, true);
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnDestroyAttnDescriptor(DnnAttnHandle descriptor) {
    return destroy_dnn_metadata(descriptor, ExecutionDnnObjectKind::kAttn, true);
}
// NOLINTBEGIN(bugprone-easily-swappable-parameters): exact public SDK ABIs.
extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetRNNDescriptor_v8(
    cudnnRNNDescriptor_t descriptor, cudnnRNNAlgo_t algorithm, cudnnRNNMode_t cell_mode,
    cudnnRNNBiasMode_t bias_mode, cudnnDirectionMode_t direction, cudnnRNNInputMode_t input_mode,
    cudnnDataType_t type, cudnnDataType_t math_precision, cudnnMathType_t math_type,
    std::int32_t input_size, std::int32_t hidden_size, std::int32_t projection_size,
    std::int32_t layers, cudnnDropoutDescriptor_t dropout, std::uint32_t flags) {
    return with_dnn_dropout_dependencies(
        descriptor, ExecutionDnnObjectKind::kRnn, {dropout, nullptr},
        [&](State& shared, Exclusion& exclusion, const auto& descriptors) {
            ExecutionDnnRnnState value;
            value.algorithm = dnn_enum_bits(algorithm);
            value.cell_mode = dnn_enum_bits(cell_mode);
            value.bias_mode = dnn_enum_bits(bias_mode);
            value.direction = dnn_enum_bits(direction);
            value.input_mode = dnn_enum_bits(input_mode);
            value.type = dnn_enum_bits(type);
            value.math_precision = dnn_enum_bits(math_precision);
            value.math_type = dnn_enum_bits(math_type);
            value.input_size = input_size;
            value.hidden_size = hidden_size;
            value.projection_size = projection_size;
            value.layers = layers;
            value.dropout = descriptors[1] ? descriptors[1]->remote : 0;
            value.flags = flags;
            return send_sequence_set(shared, exclusion, *descriptors[0],
                                     ExecutionOpcode::kSetDnnRnn, value, write_execution_dnn_rnn);
        });
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetAttnDescriptor(
    DnnAttnHandle descriptor, unsigned int mode, int heads, double scale, cudnnDataType_t type,
    cudnnDataType_t precision, cudnnMathType_t math_type,
    cudnnDropoutDescriptor_t attention_dropout, cudnnDropoutDescriptor_t post_dropout,
    int query_size, int key_size, int value_size, int query_projection, int key_projection,
    int value_projection, int output_projection, int max_query_sequence, int max_key_sequence,
    int max_batch, int max_beam) {
    return with_dnn_dropout_dependencies(
        descriptor, ExecutionDnnObjectKind::kAttn, {attention_dropout, post_dropout},
        [&](State& shared, Exclusion& exclusion, const auto& descriptors) {
            ExecutionDnnAttnState value;
            value.mode = mode;
            value.heads = heads;
            value.scale_bits = std::bit_cast<std::uint64_t>(scale);
            value.type = dnn_enum_bits(type);
            value.precision = dnn_enum_bits(precision);
            value.math_type = dnn_enum_bits(math_type);
            value.dropouts = {descriptors[1] ? descriptors[1]->remote : 0,
                              descriptors[2] ? descriptors[2]->remote : 0};
            value.query_size = query_size;
            value.key_size = key_size;
            value.value_size = value_size;
            value.query_projection = query_projection;
            value.key_projection = key_projection;
            value.value_projection = value_projection;
            value.output_projection = output_projection;
            value.max_query_sequence = max_query_sequence;
            value.max_key_sequence = max_key_sequence;
            value.max_batch = max_batch;
            value.max_beam = max_beam;
            return send_sequence_set(shared, exclusion, *descriptors[0],
                                     ExecutionOpcode::kSetDnnAttn, value, write_execution_dnn_attn);
        });
}
// Pinned-reference stubs cannot receive remote tokens in a caller-native SDK.
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetRNNDescriptor_v8(
    cudnnRNNDescriptor_t, cudnnRNNAlgo_t*, cudnnRNNMode_t*, cudnnRNNBiasMode_t*,
    cudnnDirectionMode_t*, cudnnRNNInputMode_t*, cudnnDataType_t*, cudnnDataType_t*,
    cudnnMathType_t*, std::int32_t*, std::int32_t*, std::int32_t*, std::int32_t*,
    cudnnDropoutDescriptor_t*, std::uint32_t*) {
    return dnn_unsupported();
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetAttnDescriptor(
    DnnAttnHandle, unsigned int*, int*, double*, cudnnDataType_t*, cudnnDataType_t*,
    cudnnMathType_t*, cudnnDropoutDescriptor_t*, cudnnDropoutDescriptor_t*, int*, int*, int*, int*,
    int*, int*, int*, int*, int*, int*, int*) {
    return dnn_unsupported();
}
// NOLINTEND(bugprone-easily-swappable-parameters)
