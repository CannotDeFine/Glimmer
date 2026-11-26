// Context-independent descriptor routes use the authenticated DNM1 owner.
// The configured SDK's InitTransformDest classification is a service gate.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
namespace {
cudnnStatus_t transform_destination_status(const std::optional<ExecutionWireReply>& reply,
                                           Entry& metadata, DnnEntry& destination, bool has_size) {
    const auto invalid = [&] { return dnn_invalid(metadata, &destination.is_uncertain); };
    if (!reply || reply->id || !reply->data.empty() || reply->parameter_bytes > 1) return invalid();
    if (reply->disposition == ExecutionWireDisposition::kRejected) {
        if (reply->original_status || reply->address) return invalid();
        return dnn_status(reply, metadata, &destination.is_uncertain);
    }
    if (reply->disposition != ExecutionWireDisposition::kComplete) return invalid();
    const auto status = dnn_status(reply, metadata, &destination.is_uncertain);
    if (status != CUDNN_STATUS_SUCCESS)
        return reply->address || reply->parameter_bytes != 1 ? invalid() : status;
    if (reply->parameter_bytes || metadata.is_uncertain || destination.is_uncertain ||
        reply->address > SIZE_MAX || (!has_size && reply->address))
        return invalid();
    return CUDNN_STATUS_SUCCESS;
}
cudnnStatus_t initialize_transform_destination(cudnnTensorTransformDescriptor_t transform,
                                               cudnnTensorDescriptor_t source,
                                               cudnnTensorDescriptor_t destination,
                                               std::size_t* bytes) noexcept {
    return dnn_boundary([&](State& shared, Exclusion& exclusion) {
        if (bytes && !writable_host_range(bytes, sizeof(*bytes))) return CUDNN_STATUS_BAD_PARAM;
        auto* metadata = shared.dnn_metadata_context;
        if (!metadata || !metadata->connection || !metadata->is_alive || metadata->is_uncertain ||
            metadata->is_closing)
            return CUDNN_STATUS_NOT_INITIALIZED;
        if (metadata->active_calls >= kMaxHandles - 1 ||
            metadata->active_mutations >= kMaxHandles - 1)
            return dnn_profile::host_allocation_status();
        CounterPin owner(metadata->active_calls, exclusion);
        CounterPin work(metadata->active_mutations, exclusion);
        const std::array<const void*, 3> handles{transform, source, destination};
        constexpr std::array kinds{ExecutionDnnObjectKind::kTensorTransform,
                                   ExecutionDnnObjectKind::kTensor,
                                   ExecutionDnnObjectKind::kTensor};
        std::array<std::shared_ptr<DnnEntry>, 3> descriptors;
        for (;;) {
            bool is_waiting{};
            for (std::size_t i = 0; i != descriptors.size(); ++i) {
                auto& descriptor = descriptors[i];
                descriptor = find_dnn_metadata(shared, handles[i], kinds[i]);
                if (!descriptor || descriptor->context != metadata ||
                    !valid_dnn_metadata(*descriptor))
                    return CUDNN_STATUS_NOT_INITIALIZED;
                if (descriptor->active_calls >= kMaxHandles - 1)
                    return dnn_profile::host_allocation_status();
                is_waiting |= descriptor->is_retiring;
            }
            is_waiting |= descriptors[2]->active_calls != 0;
            if (!is_waiting) break;
            // Hold no object flags or borrows while waiting: reciprocal
            // source/destination mutations must not wait on one another.
            if (!yield_frontend(exclusion)) return CUDNN_STATUS_NOT_INITIALIZED;
        }
        FlagPin mutation(descriptors[2]->is_retiring, exclusion);
        std::array<std::optional<CounterPin>, 3> pins;
        for (std::size_t i = 0; i != descriptors.size(); ++i) {
            if (i == 2 && descriptors[2] == descriptors[1]) continue;
            pins[i].emplace(descriptors[i]->active_calls, exclusion);
        }
        ExecutionPayloadWriter payload;
        for (const auto& descriptor : descriptors)
            if (!payload.write_u64(descriptor->remote))
                return dnn_profile::host_allocation_status();
        if (!payload.write_u32(bytes ? 1 : 0)) return dnn_profile::host_allocation_status();
        const auto reply = shared.call(exclusion, *metadata, ExecutionOpcode::kDnnInitTransformDest,
                                       std::move(payload));
        const auto result =
            transform_destination_status(reply, *metadata, *descriptors[2], bytes != nullptr);
        if (result != CUDNN_STATUS_SUCCESS) return result;
        if (!bytes) return CUDNN_STATUS_SUCCESS;
        if (!writable_host_range(bytes, sizeof(*bytes))) return CUDNN_STATUS_BAD_PARAM;
        return publish(bytes, static_cast<std::size_t>(reply->address)) ? CUDNN_STATUS_SUCCESS
                                                                        : CUDNN_STATUS_BAD_PARAM;
    });
}
}  // namespace

extern "C" cudnnStatus_t CUDNNWINAPI
cudnnCreateOpTensorDescriptor(cudnnOpTensorDescriptor_t* output) {
    return create_dnn_metadata(output, ExecutionDnnObjectKind::kOpTensor);
}
extern "C" cudnnStatus_t CUDNNWINAPI
cudnnDestroyOpTensorDescriptor(cudnnOpTensorDescriptor_t descriptor) {
    return destroy_dnn_metadata(descriptor, ExecutionDnnObjectKind::kOpTensor);
}
extern "C" cudnnStatus_t CUDNNWINAPI
cudnnCreateTensorTransformDescriptor(cudnnTensorTransformDescriptor_t* output) {
    return create_dnn_metadata(output, ExecutionDnnObjectKind::kTensorTransform);
}
extern "C" cudnnStatus_t CUDNNWINAPI
cudnnDestroyTensorTransformDescriptor(cudnnTensorTransformDescriptor_t descriptor) {
    return destroy_dnn_metadata(descriptor, ExecutionDnnObjectKind::kTensorTransform);
}
// NOLINTBEGIN(bugprone-easily-swappable-parameters): exact public SDK ABIs.
extern "C" cudnnStatus_t CUDNNWINAPI
cudnnSetOpTensorDescriptor(cudnnOpTensorDescriptor_t descriptor, cudnnOpTensorOp_t operation,
                           cudnnDataType_t type, cudnnNanPropagation_t nan) {
    return dnn_metadata(
        descriptor, ExecutionDnnObjectKind::kOpTensor, true,
        [&](State& shared, Exclusion& exclusion, DnnEntry& entry) {
            const ExecutionDnnOpTensor value{dnn_enum_bits(operation), dnn_enum_bits(type),
                                             dnn_enum_bits(nan)};
            ExecutionPayloadWriter payload;
            if (!payload.write_u64(entry.remote) || !write_execution_dnn_op_tensor(payload, value))
                return dnn_profile::host_allocation_status();
            return dnn_status(shared.call(exclusion, *entry.context,
                                          ExecutionOpcode::kSetDnnOpTensor, std::move(payload)),
                              *entry.context, &entry.is_uncertain);
        });
}
extern "C" cudnnStatus_t CUDNNWINAPI
cudnnGetOpTensorDescriptor(const cudnnOpTensorDescriptor_t descriptor, cudnnOpTensorOp_t* operation,
                           cudnnDataType_t* type, cudnnNanPropagation_t* nan) {
    return dnn_metadata(
        descriptor, ExecutionDnnObjectKind::kOpTensor, false,
        [&](State& shared, Exclusion& exclusion, DnnEntry& entry) {
            const auto writable = [&] {
                return (!operation || writable_host_range(operation, sizeof(*operation))) &&
                       (!type || writable_host_range(type, sizeof(*type))) &&
                       (!nan || writable_host_range(nan, sizeof(*nan)));
            };
            if (!writable()) return CUDNN_STATUS_BAD_PARAM;
            const std::uint32_t mask = (operation ? 1U : 0U) | (type ? 2U : 0U) | (nan ? 4U : 0U);
            ExecutionPayloadWriter payload;
            if (!payload.write_u64(entry.remote) || !payload.write_u32(mask))
                return dnn_profile::host_allocation_status();
            const auto reply = shared.call(exclusion, *entry.context,
                                           ExecutionOpcode::kGetDnnOpTensor, std::move(payload));
            const auto result = dnn_status(reply, *entry.context, &entry.is_uncertain);
            if (result != CUDNN_STATUS_SUCCESS) return result;
            ExecutionDnnOpTensor value;
            ExecutionPayloadReader input(reply->data);
            if (!read_execution_dnn_op_tensor(input, value) || !input.empty() ||
                (operation && value.operation > 7) ||
                (type && value.type > dnn_profile::data_type_enum_max()) || (nan && value.nan > 1))
                return dnn_invalid(*entry.context, &entry.is_uncertain);
            // Validate the entire result and every present output before
            // publishing any field. Null outputs remain native null pointers.
            if (!writable()) return CUDNN_STATUS_BAD_PARAM;
            if ((operation &&
                 !publish(operation, static_cast<cudnnOpTensorOp_t>(value.operation))) ||
                (type && !publish(type, static_cast<cudnnDataType_t>(value.type))) ||
                (nan && !publish(nan, static_cast<cudnnNanPropagation_t>(value.nan))))
                return CUDNN_STATUS_BAD_PARAM;
            return CUDNN_STATUS_SUCCESS;
        });
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetTensorTransformDescriptor(
    cudnnTensorTransformDescriptor_t descriptor, const std::uint32_t dimensions,
    const cudnnTensorFormat_t format, const std::int32_t before[], const std::int32_t after[],
    const std::uint32_t fold[], const cudnnFoldingDirection_t direction) {
    if (::getpid() != kProcess) return CUDNN_STATUS_NOT_INITIALIZED;
    if (dimensions > kExecutionDnnMaxDimensions) return CUDNN_STATUS_BAD_PARAM;
    ExecutionDnnTensorTransform value;
    value.dimensions = dimensions;
    value.format = dnn_enum_bits(format);
    value.direction = dnn_enum_bits(direction);
    value.presence = (before ? 1U : 0U) | (after ? 2U : 0U) | (fold ? 4U : 0U);
    const auto padding_bytes = static_cast<std::size_t>(dimensions) * sizeof(std::int32_t);
    const auto folding_bytes =
        static_cast<std::size_t>(dimensions > 2 ? dimensions - 2 : 0) * sizeof(std::uint32_t);
    if ((before && !self_copy(value.before.data(), before, padding_bytes, false)) ||
        (after && !self_copy(value.after.data(), after, padding_bytes, false)) ||
        (fold && !self_copy(value.fold.data(), fold, folding_bytes, false)))
        return CUDNN_STATUS_BAD_PARAM;
    return dnn_metadata(descriptor, ExecutionDnnObjectKind::kTensorTransform, true,
                        [&](State& shared, Exclusion& exclusion, DnnEntry& entry) {
                            ExecutionPayloadWriter payload;
                            if (!payload.write_u64(entry.remote) ||
                                !write_execution_dnn_tensor_transform(payload, value))
                                return dnn_profile::host_allocation_status();
                            return dnn_status(shared.call(exclusion, *entry.context,
                                                          ExecutionOpcode::kSetDnnTensorTransform,
                                                          std::move(payload)),
                                              *entry.context, &entry.is_uncertain);
                        });
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnInitTransformDest(
    const cudnnTensorTransformDescriptor_t transform, const cudnnTensorDescriptor_t source,
    cudnnTensorDescriptor_t destination, std::size_t* bytes) {
    return initialize_transform_destination(transform, source, destination, bytes);
}
// Pinned reference stubs; TransformTensorEx keeps its existing typed guard.
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetTensorTransformDescriptor(
    cudnnTensorTransformDescriptor_t, std::uint32_t, cudnnTensorFormat_t*, std::int32_t[],
    std::int32_t[], std::uint32_t[], cudnnFoldingDirection_t*) {
    return dnn_unsupported();
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnOpTensor(cudnnHandle_t, const cudnnOpTensorDescriptor_t,
                                                   const void*, const cudnnTensorDescriptor_t,
                                                   const void*, const void*,
                                                   const cudnnTensorDescriptor_t, const void*,
                                                   const void*, const cudnnTensorDescriptor_t,
                                                   void*) {
    return dnn_unsupported();
}
// NOLINTEND(bugprone-easily-swappable-parameters)
#pragma GCC diagnostic pop
