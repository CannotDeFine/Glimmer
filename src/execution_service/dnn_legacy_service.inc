// Included inside ExecutionService::Implementation. All vendor types remain
// behind the optional backend boundary; Control carries no native pointers.
static ExecutionWireReply project_dnn_legacy(const CudaDnnLegacyResult& source,
                                             ExecutionPayloadWriter& output) {
    auto reply = project(source.result);
    reply.parameter_bytes =
        (source.has_metadata_quarantine ? 1U : 0U) | (source.has_context_loss ? 2U : 0U);
    if (source.result.original_status) {
        const auto& value = source.output;
        if (!write_execution_dnn_legacy_output(output, {value.written, value.scalar, value.non_null,
                                                        value.offset, value.extent})) {
            // An invalid internal receipt is never authority to publish success.
            reply.disposition = ExecutionWireDisposition::kRejected;
            reply.status = CUDA_ERROR_UNKNOWN;
            reply.parameter_bytes |= 1U;
        } else {
            reply.data = output.take();
        }
    }
    return reply;
}

std::optional<ExecutionWireReply> execute_dnn_legacy_metadata(Slot& slot,
                                                              ExecutionPayloadReader& input) {
    ExecutionDnnLegacyMetadata wire;
    if (!read_execution_dnn_legacy_metadata(input, wire) || !input.empty())
        return rejection(CUDA_ERROR_INVALID_VALUE);
    ExecutionPayloadWriter output(32);
    if (!output.reserve_capacity(32)) return rejection(CUDA_ERROR_OUT_OF_MEMORY);
    const auto result =
        slot.dnn_metadata->legacy_metadata({static_cast<CudaDnnLegacyMetadataKind>(wire.kind),
                                            wire.object, wire.value, wire.has_output, wire.seed});
    if (result.result.disposition == CudaDnnDisposition::kDeferred) return std::nullopt;
    auto reply = project_dnn_legacy(result, output);
    if (result.result.original_status && reply.data.empty()) slot.dnn_metadata->close();
    return reply;
}

static std::optional<CudaDnnLegacyControl> legacy_control(ExecutionDnnLegacyControl wire) {
    if (wire.parameters.weights.offset > SIZE_MAX) return std::nullopt;
    CudaDnnLegacyControl value;
    value.kind = static_cast<CudaDnnLegacyControlKind>(wire.kind);
    value.handle = wire.handle;
    value.metadata = wire.metadata;
    value.descriptor = wire.descriptor;
    value.has_output = wire.has_output;
    value.seed = wire.seed;
    const auto& setup = wire.setup;
    value.setup = {setup.hidden_size, setup.layers,    setup.dropout,   setup.input_mode,
                   setup.direction,   setup.cell_mode, setup.algorithm, setup.math_precision};
    value.clip = {wire.clip.mode, wire.clip.nan, std::bit_cast<double>(wire.clip.lower_bits),
                  std::bit_cast<double>(wire.clip.upper_bits)};
    value.sequence_length = wire.sequence_length;
    value.sequence = {wire.sequence.has_descriptors, std::move(wire.sequence.descriptors)};
    value.input = wire.input;
    value.data_type = wire.data_type;
    const auto& parameters = wire.parameters;
    value.parameters = {
        parameters.input,
        parameters.filter,
        parameters.destination,
        parameters.pseudo_layer,
        parameters.linear_layer,
        {parameters.weights.allocation, static_cast<std::size_t>(parameters.weights.offset)}};
    value.algorithm = wire.algorithm;
    return value;
}

std::optional<ExecutionWireReply> execute_dnn_legacy_control(Slot& slot,
                                                             ExecutionPayloadReader& input) {
    ExecutionDnnLegacyControl wire;
    if (!read_execution_dnn_legacy_control(input, wire) || !input.empty())
        return rejection(CUDA_ERROR_INVALID_VALUE);
    auto decoded = legacy_control(std::move(wire));
    if (!decoded) return rejection(CUDA_ERROR_INVALID_VALUE);
    const auto owner = std::ranges::find(slots, decoded->metadata, &Slot::connection);
    if (owner == slots.end() || !owner->dnn_metadata || owner->peer != slot.peer ||
        !owner->is_initialized || owner->is_retiring || owner->is_closing ||
        owner->has_dnn_profile_mismatch || owner->dnn_profile_version != CUDNN_VERSION)
        return rejection(CUDA_ERROR_INVALID_HANDLE);
    // Reserve all primitive/pointer-observation output before original mutation.
    ExecutionPayloadWriter output(32);
    if (!output.reserve_capacity(32)) return rejection(CUDA_ERROR_OUT_OF_MEMORY);
    const auto activated = activate(slot);
    if (activated.status != CUDA_SUCCESS) return rejection(activated.status);
    const auto result =
        slot.session->dnn_legacy_control(decoded->handle, *owner->dnn_metadata, *decoded);
    if (result.result.disposition == CudaDnnDisposition::kDeferred) return std::nullopt;
    auto reply = project_dnn_legacy(result, output);
    reply.parameter_bytes |= (owner->dnn_metadata->snapshot().is_quarantined ? 1U : 0U) |
                             (slot.session->snapshot().is_quarantined ? 2U : 0U);
    if (result.result.original_status && reply.data.empty()) {
        owner->dnn_metadata->close();
        slot.session->close();
        reply.parameter_bytes |= 3U;
    }
    return reply;
}

CUresult decode_dnn_legacy_compute(Slot& slot, ExecutionPayloadReader& input,
                                   CudaSessionDnnLegacyRnn& output, std::uint64_t& metadata) {
    ExecutionDnnLegacyCompute wire;
    if (!read_execution_dnn_legacy_compute(input, wire) || !input.empty() ||
        wire.workspace_bytes > SIZE_MAX || wire.reserve_bytes > SIZE_MAX)
        return CUDA_ERROR_INVALID_VALUE;
    const auto owner = std::ranges::find(slots, wire.metadata, &Slot::connection);
    if (owner == slots.end() || !owner->dnn_metadata || owner->peer != slot.peer ||
        !owner->is_initialized || owner->is_retiring || owner->is_closing ||
        owner->has_dnn_profile_mismatch || owner->dnn_profile_version != CUDNN_VERSION)
        return CUDA_ERROR_INVALID_HANDLE;
    CudaSessionDnnLegacyRnn value;
    value.kind = static_cast<CudaDnnLegacyComputeKind>(wire.kind);
    value.handle = wire.handle;
    value.metadata = wire.metadata;
    value.descriptor = wire.descriptor;
    value.sequence_length = wire.sequence_length;
    for (std::size_t index = 0; index != value.sequences.size(); ++index)
        value.sequences[index] = {wire.sequences[index].has_descriptors,
                                  std::move(wire.sequences[index].descriptors)};
    value.states = wire.states;
    value.filter = wire.filter;
    for (std::size_t index = 0; index != value.buffers.size(); ++index) {
        const auto& buffer = wire.buffers[index];
        if (buffer.offset > SIZE_MAX) return CUDA_ERROR_INVALID_VALUE;
        value.buffers[index] = {buffer.allocation, static_cast<std::size_t>(buffer.offset)};
    }
    value.workspace_bytes = wire.workspace_bytes;
    value.reserve_bytes = wire.reserve_bytes;
    metadata = wire.metadata;
    output = std::move(value);
    return CUDA_SUCCESS;
}
