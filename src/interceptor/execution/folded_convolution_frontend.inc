// One synchronous original mutation, with no queue submission or device drain.
namespace {
constexpr std::array kFoldedDescriptorKinds{
    ExecutionDnnObjectKind::kFilter,          ExecutionDnnObjectKind::kTensor,
    ExecutionDnnObjectKind::kConvolution,     ExecutionDnnObjectKind::kTensor,
    ExecutionDnnObjectKind::kFilter,          ExecutionDnnObjectKind::kTensor,
    ExecutionDnnObjectKind::kConvolution,     ExecutionDnnObjectKind::kTensor,
    ExecutionDnnObjectKind::kTensorTransform, ExecutionDnnObjectKind::kTensorTransform,
    ExecutionDnnObjectKind::kTensorTransform, ExecutionDnnObjectKind::kTensorTransform};
using FoldedEntries = std::array<std::shared_ptr<DnnEntry>, 12>;

cudnnStatus_t folded_descriptor_status(const std::optional<ExecutionWireReply>& reply,
                                       Entry& handle, Entry& metadata,
                                       const FoldedEntries& descriptors) {
    auto& context = *handle.context;
    const auto fault_metadata = [&] {
        metadata.is_uncertain = true;
        for (std::size_t index = 4; index != descriptors.size(); ++index)
            if (descriptors[index]) descriptors[index]->is_uncertain = true;
    };
    const auto fault_context = [&] {
        context.is_uncertain = true;
        handle.is_uncertain = true;
    };
    const auto invalid = [&] {
        fault_metadata();
        fault_context();
        return CUDNN_STATUS_INTERNAL_ERROR;
    };
    if (!reply || reply->id || reply->address || !reply->data.empty() || reply->parameter_bytes > 3)
        return invalid();
    const auto flags = reply->parameter_bytes;
    const auto original = reply->original_status;
    if (reply->disposition == ExecutionWireDisposition::kComplete) {
        if (!original || !dnn_profile::status_is_representable(*original) ||
            reply->status != *original)
            return invalid();
        if (*original != CUDNN_STATUS_SUCCESS) {
            // An issued mutation error has no established rollback. Its native
            // status is retained while the execution handle remains usable.
            if (flags != 1) return invalid();
            fault_metadata();
            return static_cast<cudnnStatus_t>(*original);
        }
        if (flags || context.is_uncertain || handle.is_uncertain || metadata.is_uncertain ||
            std::ranges::any_of(descriptors, [](const auto& descriptor) {
                return descriptor && descriptor->is_uncertain;
            }))
            return invalid();
        return CUDNN_STATUS_SUCCESS;
    }
    constexpr auto driver_max =
        std::bit_ceil(static_cast<std::uint32_t>(CUDA_ERROR_UNKNOWN) + 1U) - 1U;
    if (reply->disposition != ExecutionWireDisposition::kRejected || reply->status <= 0 ||
        static_cast<std::uint32_t>(reply->status) > driver_max ||
        (original && (flags != 3 || !dnn_profile::status_is_representable(*original))))
        return invalid();
    if (flags & 1U) fault_metadata();
    if (flags & 2U) fault_context();
    return dnn_error(static_cast<CUresult>(reply->status));
}

cudnnStatus_t fold_backward_convolution(cudnnHandle_t handle,
                                        const std::array<const void*, 12>& handles,
                                        std::uint32_t format) noexcept {
    return dnn_handle(handle, false, false, [&](State& shared, Exclusion& exclusion, Entry& entry) {
        auto* metadata = shared.dnn_metadata_context;
        if (!metadata || !metadata->connection || !metadata->is_alive || metadata->is_uncertain ||
            metadata->is_closing)
            return CUDNN_STATUS_NOT_INITIALIZED;
        if (metadata->active_calls >= kMaxHandles - 1 ||
            metadata->active_mutations >= kMaxHandles - 1)
            return dnn_profile::host_allocation_status();
        CounterPin owner(metadata->active_calls, exclusion);
        CounterPin work(metadata->active_mutations, exclusion);
        auto& context = *entry.context;
        const auto epoch = context.epoch;
        FoldedEntries descriptors;
        for (;;) {
            if (!entry.is_alive || entry.is_uncertain || !context.is_alive || context.is_closing ||
                context.is_uncertain || context.epoch != epoch)
                return CUDNN_STATUS_NOT_INITIALIZED;
            bool is_waiting{};
            for (std::size_t index = 0; index != descriptors.size(); ++index) {
                auto& descriptor = descriptors[index];
                if (index == descriptors.size() - 1 && !handles[index]) {
                    descriptor.reset();
                    continue;
                }
                descriptor =
                    find_dnn_metadata(shared, handles[index], kFoldedDescriptorKinds[index]);
                if (!descriptor || descriptor->context != metadata ||
                    !valid_dnn_metadata(*descriptor))
                    return CUDNN_STATUS_NOT_INITIALIZED;
                if (descriptor->active_calls >= kMaxHandles - 1)
                    return dnn_profile::host_allocation_status();
                is_waiting |= descriptor->is_retiring || (index >= 4 && descriptor->active_calls);
            }
            if (!is_waiting) break;
            // No object pins or retiring flags until the entire set is ready.
            // Reciprocal mutations and source/output aliases must not self-lock.
            if (!yield_frontend(exclusion)) return CUDNN_STATUS_NOT_INITIALIZED;
        }
        std::array<std::optional<FlagPin>, 8> mutations;
        for (std::size_t index = 4; index != descriptors.size(); ++index) {
            if (!descriptors[index]) continue;
            bool is_duplicate{};
            for (std::size_t previous = 4; previous != index; ++previous)
                is_duplicate |= descriptors[index] == descriptors[previous];
            if (!is_duplicate)
                mutations[index - 4].emplace(descriptors[index]->is_retiring, exclusion);
        }
        std::array<std::optional<CounterPin>, 12> pins;
        for (std::size_t index = 0; index != descriptors.size(); ++index) {
            if (!descriptors[index]) continue;
            bool is_duplicate{};
            for (std::size_t previous = 0; previous != index; ++previous)
                is_duplicate |= descriptors[index] == descriptors[previous];
            if (!is_duplicate) pins[index].emplace(descriptors[index]->active_calls, exclusion);
        }
        ExecutionDnnFoldedConvBackwardDataDescriptors value;
        value.handle = entry.remote;
        value.metadata = metadata->connection->session();
        value.format = format;
        for (std::size_t index = 0; index != value.inputs.size(); ++index)
            value.inputs[index] = descriptors[index]->remote;
        for (std::size_t index = 0; index != value.outputs.size(); ++index)
            if (descriptors[index + 4]) value.outputs[index] = descriptors[index + 4]->remote;
        ExecutionPayloadWriter payload;
        if (!write_execution_dnn_folded_conv_backward_data_descriptors(payload, value))
            return dnn_profile::host_allocation_status();
        return folded_descriptor_status(
            shared.call(exclusion, context, ExecutionOpcode::kDnnFoldedConvBackwardDataDescriptors,
                        std::move(payload)),
            entry, *metadata, descriptors);
    });
}
}  // namespace

// NOLINTBEGIN(bugprone-easily-swappable-parameters): exact installed public SDK ABI.
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetFoldedConvBackwardDataDescriptors(
    const cudnnHandle_t handle, const cudnnFilterDescriptor_t filter,
    const cudnnTensorDescriptor_t diff, const cudnnConvolutionDescriptor_t convolution,
    const cudnnTensorDescriptor_t grad, const cudnnTensorFormat_t format,
    cudnnFilterDescriptor_t folded_filter, cudnnTensorDescriptor_t padded_diff,
    cudnnConvolutionDescriptor_t folded_convolution, cudnnTensorDescriptor_t folded_grad,
    cudnnTensorTransformDescriptor_t filter_fold, cudnnTensorTransformDescriptor_t diff_pad,
    cudnnTensorTransformDescriptor_t grad_fold, cudnnTensorTransformDescriptor_t grad_unfold) {
    return fold_backward_convolution(
        handle,
        {filter, diff, convolution, grad, folded_filter, padded_diff, folded_convolution,
         folded_grad, filter_fold, diff_pad, grad_fold, grad_unfold},
        dnn_enum_bits(format));
}
// NOLINTEND(bugprone-easily-swappable-parameters)
