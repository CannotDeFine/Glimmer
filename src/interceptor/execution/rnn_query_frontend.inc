// Owned-context observations, not framework computation or implicit drains.
namespace {
cudnnStatus_t rnn_query_status(const std::optional<ExecutionWireReply>& reply, Entry& handle,
                               Entry& metadata, const ExecutionDnnRnnQuery& query,
                               ExecutionDnnRnnQueryResult& output) {
    auto& context = *handle.context;
    const auto fault_context = [&] {
        context.is_uncertain = true;
        handle.is_uncertain = true;
    };
    const auto invalid = [&] {
        metadata.is_uncertain = true;
        fault_context();
        return CUDNN_STATUS_INTERNAL_ERROR;
    };
    if (!reply || reply->id || reply->address || !reply->arguments.empty() ||
        reply->parameter_bytes > 3)
        return invalid();
    const auto flags = reply->parameter_bytes;
    const auto original = reply->original_status;
    if (reply->disposition == ExecutionWireDisposition::kRejected) {
        constexpr auto driver_max =
            std::bit_ceil(static_cast<std::uint32_t>(CUDA_ERROR_UNKNOWN) + 1U) - 1U;
        if (!reply->data.empty() || reply->status <= 0 ||
            static_cast<std::uint32_t>(reply->status) > driver_max ||
            (original && (!(flags & 2U) || !dnn_profile::status_is_representable(*original))))
            return invalid();
        if (flags & 1U) metadata.is_uncertain = true;
        if (flags & 2U) fault_context();
        return dnn_error(static_cast<CUresult>(reply->status));
    }
    if (reply->disposition != ExecutionWireDisposition::kComplete || !original ||
        !dnn_profile::status_is_representable(*original) || reply->status != *original || flags)
        return invalid();
    if (*original != CUDNN_STATUS_SUCCESS)
        return reply->data.empty() ? static_cast<cudnnStatus_t>(*original) : invalid();
    if (!context.is_alive || context.is_uncertain || !handle.is_alive || handle.is_uncertain ||
        !metadata.is_alive || metadata.is_uncertain)
        return invalid();
    ExecutionDnnRnnQueryResult value;
    ExecutionPayloadReader input(reply->data);
    if (!read_execution_dnn_rnn_query_result(input, value) || !input.empty()) return invalid();
    for (std::size_t i = 0; i != value.sizes.size(); ++i)
        if (value.sizes[i] > SIZE_MAX || (!(query.outputs & (1U << i)) && value.sizes[i]))
            return invalid();
    output = value;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t query_dnn_rnn(cudnnHandle_t handle, const void* parent, const void* input,
                            ExecutionDnnRnnQueryKind kind, std::uint32_t mode,
                            const std::array<std::size_t*, 3>& outputs) noexcept {
    return dnn_handle(handle, false, false, [&](State& shared, Exclusion& exclusion, Entry& entry) {
        for (auto* output : outputs)
            if (output && !writable_host_range(output, sizeof(*output)))
                return CUDNN_STATUS_BAD_PARAM;
        const bool is_temp = kind == ExecutionDnnRnnQueryKind::kTempSpace;
        const std::array<const void*, 2> handles{parent, input};
        const std::array kinds{kind == ExecutionDnnRnnQueryKind::kAttentionBuffers
                                   ? ExecutionDnnObjectKind::kAttn
                                   : ExecutionDnnObjectKind::kRnn,
                               ExecutionDnnObjectKind::kRnnData};
        const auto count = is_temp ? 2U : 1U;
        const auto epoch = entry.context->epoch;
        // All explicit descriptor borrows are acquired together. The service
        // additionally pins the parent's private dropout dependency edges.
        return with_dnn_descriptors(
            shared, exclusion, std::span{handles}.first(count), std::span{kinds}.first(count),
            [&](Entry& metadata, const auto& descriptors) {
                if (!entry.is_alive || entry.is_uncertain || !entry.context->is_alive ||
                    entry.context->is_uncertain || entry.context->is_closing ||
                    entry.context->epoch != epoch)
                    return CUDNN_STATUS_NOT_INITIALIZED;
                ExecutionDnnRnnQuery query;
                query.kind = kind;
                query.handle = entry.remote;
                query.metadata = metadata.connection->session();
                query.descriptor = descriptors[0]->remote;
                query.input = is_temp ? descriptors[1]->remote : 0;
                query.mode = mode;
                for (std::size_t i = 0; i != outputs.size(); ++i)
                    if (outputs[i]) query.outputs |= 1U << i;
                ExecutionPayloadWriter payload;
                if (!write_execution_dnn_rnn_query(payload, query))
                    return dnn_profile::host_allocation_status();
                const auto reply = shared.call(exclusion, *entry.context,
                                               ExecutionOpcode::kDnnRnnQuery, std::move(payload));
                ExecutionDnnRnnQueryResult value;
                const auto status = rnn_query_status(reply, entry, metadata, query, value);
                if (status != CUDNN_STATUS_SUCCESS) return status;
                for (std::size_t i = 0; i != count; ++i)
                    if (!valid_dnn_metadata(*descriptors[i]))
                        return dnn_invalid(metadata, &descriptors[i]->is_uncertain);
                // Validate every full span before the first write. The original
                // SDK's successful alias order differs between the two queries.
                for (auto* output : outputs)
                    if (output && !writable_host_range(output, sizeof(*output)))
                        return CUDNN_STATUS_BAD_PARAM;
                const auto order = kind == ExecutionDnnRnnQueryKind::kAttentionBuffers
                                       ? std::array<std::size_t, 3>{0, 2, 1}
                                       : std::array<std::size_t, 3>{0, 1, 2};
                for (auto i : order)
                    if (outputs[i] &&
                        !publish(outputs[i], static_cast<std::size_t>(value.sizes[i])))
                        return CUDNN_STATUS_BAD_PARAM;
                return CUDNN_STATUS_SUCCESS;
            });
    });
}
}  // namespace

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetRNNWeightSpaceSize(cudnnHandle_t handle,
                                                                cudnnRNNDescriptor_t descriptor,
                                                                std::size_t* bytes) {
    return query_dnn_rnn(handle, descriptor, nullptr, ExecutionDnnRnnQueryKind::kWeightSpace, 0,
                         {bytes, nullptr, nullptr});
}
// NOLINTBEGIN(bugprone-easily-swappable-parameters): exact installed public SDK ABIs.
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetRNNTempSpaceSizes(
    cudnnHandle_t handle, cudnnRNNDescriptor_t descriptor, cudnnForwardMode_t mode,
    cudnnRNNDataDescriptor_t input, std::size_t* workspace, std::size_t* reserve) {
    return query_dnn_rnn(handle, descriptor, input, ExecutionDnnRnnQueryKind::kTempSpace,
                         dnn_enum_bits(mode), {nullptr, workspace, reserve});
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetMultiHeadAttnBuffers(cudnnHandle_t handle,
                                                                  const DnnAttnHandle descriptor,
                                                                  std::size_t* weight,
                                                                  std::size_t* workspace,
                                                                  std::size_t* reserve) {
    return query_dnn_rnn(handle, descriptor, nullptr, ExecutionDnnRnnQueryKind::kAttentionBuffers,
                         0, {weight, workspace, reserve});
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnRNNSetClip_v8(cudnnRNNDescriptor_t descriptor,
                                                        cudnnRNNClipMode_t mode,
                                                        cudnnNanPropagation_t nan, double lower,
                                                        double upper) {
    return dnn_metadata(descriptor, ExecutionDnnObjectKind::kRnn, true,
                        [&](State& shared, Exclusion& exclusion, DnnEntry& entry) {
                            const ExecutionDnnRnnClip value{dnn_enum_bits(mode), dnn_enum_bits(nan),
                                                            std::bit_cast<std::uint64_t>(lower),
                                                            std::bit_cast<std::uint64_t>(upper)};
                            return send_sequence_set(shared, exclusion, entry,
                                                     ExecutionOpcode::kSetDnnRnnClip, value,
                                                     write_execution_dnn_rnn_clip);
                        });
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnRNNGetClip_v8(cudnnRNNDescriptor_t, cudnnRNNClipMode_t*,
                                                        cudnnNanPropagation_t*, double*, double*) {
    return dnn_unsupported();
}
// NOLINTEND(bugprone-easily-swappable-parameters)
