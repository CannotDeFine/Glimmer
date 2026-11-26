// Included after the other DNN frontends; all SDK work stays in the service.
namespace {
using DnnReductionHandle = cudnnReduceTensorStruct*;
constexpr std::array kDnnReductionKinds{ExecutionDnnObjectKind::kTensor,
                                        ExecutionDnnObjectKind::kReduction,
                                        ExecutionDnnObjectKind::kTensor};

cudnnStatus_t query_dnn_reduction_size(cudnnHandle_t handle, DnnReductionHandle reduction,
                                       cudnnTensorDescriptor_t input,
                                       cudnnTensorDescriptor_t output, std::size_t* bytes,
                                       ExecutionDnnReductionQueryKind kind) noexcept {
    return dnn_handle(handle, false, false, [&](State& shared, Exclusion& exclusion, Entry& entry) {
        if (!writable_host_range(bytes, sizeof(*bytes))) return CUDNN_STATUS_BAD_PARAM;
        const std::array<const void*, 3> handles{input, reduction, output};
        return with_dnn_descriptors(
            shared, exclusion, handles, kDnnReductionKinds,
            [&](Entry& metadata, const auto& descriptors) {
                ExecutionDnnReductionQuery query;
                query.kind = kind;
                query.handle = entry.remote;
                query.metadata = metadata.connection->session();
                for (std::size_t i = 0; i != query.descriptors.size(); ++i)
                    query.descriptors[i] = descriptors[i]->remote;
                ExecutionPayloadWriter payload;
                if (!write_execution_dnn_reduction_query(payload, query))
                    return dnn_profile::host_allocation_status();
                const auto reply =
                    shared.call(exclusion, *entry.context, ExecutionOpcode::kDnnReductionQuery,
                                std::move(payload));
                const auto result = dnn_status(reply, *entry.context, &entry.is_uncertain);
                if (result != CUDNN_STATUS_SUCCESS) return result;
                if (!reply->data.empty() || reply->address > SIZE_MAX)
                    return dnn_invalid(*entry.context, &entry.is_uncertain);
                if (!writable_host_range(bytes, sizeof(*bytes))) return CUDNN_STATUS_BAD_PARAM;
                return publish(bytes, static_cast<std::size_t>(reply->address))
                           ? CUDNN_STATUS_SUCCESS
                           : CUDNN_STATUS_BAD_PARAM;
            });
    });
}
}  // namespace

// Reduction remains a supported legacy SDK ABI despite its deprecation.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
extern "C" cudnnStatus_t CUDNNWINAPI cudnnCreateReduceTensorDescriptor(DnnReductionHandle* output) {
    return create_dnn_metadata(output, ExecutionDnnObjectKind::kReduction);
}
extern "C" cudnnStatus_t CUDNNWINAPI
cudnnDestroyReduceTensorDescriptor(DnnReductionHandle descriptor) {
    return destroy_dnn_metadata(descriptor, ExecutionDnnObjectKind::kReduction);
}
// NOLINTBEGIN(bugprone-easily-swappable-parameters): exact public SDK ABIs.
extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetReduceTensorDescriptor(
    DnnReductionHandle descriptor, cudnnReduceTensorOp_t operation, cudnnDataType_t type,
    cudnnNanPropagation_t nan, cudnnReduceTensorIndices_t indices,
    cudnnIndicesType_t indices_type) {
    return dnn_metadata(
        descriptor, ExecutionDnnObjectKind::kReduction, true,
        [&](State& shared, Exclusion& exclusion, DnnEntry& entry) {
            const ExecutionDnnReduction value{dnn_enum_bits(operation), dnn_enum_bits(type),
                                              dnn_enum_bits(nan), dnn_enum_bits(indices),
                                              dnn_enum_bits(indices_type)};
            ExecutionPayloadWriter input;
            if (!input.write_u64(entry.remote) || !write_execution_dnn_reduction(input, value))
                return dnn_profile::host_allocation_status();
            return dnn_status(shared.call(exclusion, *entry.context,
                                          ExecutionOpcode::kSetDnnReduction, std::move(input)),
                              *entry.context, &entry.is_uncertain);
        });
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetReductionIndicesSize(cudnnHandle_t handle,
                                                                  DnnReductionHandle reduction,
                                                                  cudnnTensorDescriptor_t input,
                                                                  cudnnTensorDescriptor_t output,
                                                                  std::size_t* bytes) {
    return query_dnn_reduction_size(handle, reduction, input, output, bytes,
                                    ExecutionDnnReductionQueryKind::kIndices);
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetReductionWorkspaceSize(cudnnHandle_t handle,
                                                                    DnnReductionHandle reduction,
                                                                    cudnnTensorDescriptor_t input,
                                                                    cudnnTensorDescriptor_t output,
                                                                    std::size_t* bytes) {
    return query_dnn_reduction_size(handle, reduction, input, output, bytes,
                                    ExecutionDnnReductionQueryKind::kWorkspace);
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnReduceTensor(
    cudnnHandle_t handle, DnnReductionHandle reduction, void* indices, std::size_t indices_bytes,
    void* workspace, std::size_t workspace_bytes, const void* alpha, cudnnTensorDescriptor_t input,
    const void* x, const void* beta, cudnnTensorDescriptor_t output, void* y) {
    return dnn_handle(handle, false, true, [&](State& shared, Exclusion& exclusion, Entry& entry) {
        const std::array<const void*, 3> handles{input, reduction, output};
        return with_dnn_descriptors(
            shared, exclusion, handles, kDnnReductionKinds,
            [&](Entry& metadata, const auto& descriptors) {
                if (entry.context->submissions == UINT64_MAX)
                    return dnn_profile::host_allocation_status();
                // Coefficients follow the input tensor, not the accumulator or
                // output storage type. All descriptor pins predate this query.
                ExecutionDnnTensor tensor;
                const auto queried = query_dnn_tensor(shared, exclusion, *descriptors[0],
                                                      CUDNN_DIM_MAX, false, tensor);
                if (queried != CUDNN_STATUS_SUCCESS) return queried;
                ExecutionDnnReductionCompute compute;
                compute.handle = entry.remote;
                compute.metadata = metadata.connection->session();
                for (std::size_t i = 0; i != compute.descriptors.size(); ++i)
                    compute.descriptors[i] = descriptors[i]->remote;
                compute.indices_bytes = indices_bytes;
                compute.workspace_bytes = workspace_bytes;
                compute.scalar_bytes =
                    tensor.type == CUDNN_DATA_DOUBLE ? sizeof(double) : sizeof(float);
                if (!self_copy(compute.alpha.data(), alpha, compute.scalar_bytes, false) ||
                    !self_copy(compute.beta.data(), beta, compute.scalar_bytes, false))
                    return CUDNN_STATUS_BAD_PARAM;
                const std::array<const void*, 4> pointers{x, y, indices, workspace};
                const std::array<std::size_t, 4> extents{1, 1, indices_bytes, workspace_bytes};
                std::array<std::optional<CounterPin>, 4> buffer_pins, module_pins;
                for (std::size_t i = 0; i != pointers.size(); ++i) {
                    const auto address = reinterpret_cast<CUdeviceptr>(pointers[i]);
                    // Native may ignore optional null pointers even with a
                    // nonzero declared size. Never fabricate device storage.
                    if (i >= 2 && !address) continue;
                    auto* allocation = shared.range(address, extents[i]);
                    if (!allocation || allocation->context != entry.context ||
                        shared.range_is_retiring(*allocation))
                        return CUDNN_STATUS_BAD_PARAM;
                    if (allocation->active_calls > kMaxHandles - pointers.size())
                        return dnn_profile::host_allocation_status();
                    buffer_pins[i].emplace(allocation->active_calls, exclusion);
                    if (allocation->kind == Kind::kModuleGlobal) {
                        const auto parent =
                            std::ranges::find_if(shared.entries, [&](const auto& module) {
                                return module->kind == Kind::kModule &&
                                       module->context == entry.context &&
                                       module->remote == allocation->parent && module->is_alive &&
                                       !module->is_retiring;
                            });
                        if (parent == shared.entries.end()) return CUDNN_STATUS_BAD_PARAM;
                        if ((*parent)->active_calls > kMaxHandles - pointers.size())
                            return dnn_profile::host_allocation_status();
                        module_pins[i].emplace((*parent)->active_calls, exclusion);
                    }
                    compute.buffers[i] = {allocation->remote, address - allocation->address};
                }
                ExecutionPayloadWriter payload;
                if (!write_execution_dnn_reduction_compute(payload, compute))
                    return dnn_profile::host_allocation_status();
                ++entry.context->submissions;
                return dnn_status(shared.call(exclusion, *entry.context,
                                              ExecutionOpcode::kDnnReduction, std::move(payload)),
                                  *entry.context, &entry.is_uncertain);
            });
    });
}
// The pinned reference getter is a stub, not part of its implemented surface.
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetReduceTensorDescriptor(
    DnnReductionHandle, cudnnReduceTensorOp_t*, cudnnDataType_t*, cudnnNanPropagation_t*,
    cudnnReduceTensorIndices_t*, cudnnIndicesType_t*) {
    return dnn_unsupported();
}
// NOLINTEND(bugprone-easily-swappable-parameters)
#pragma GCC diagnostic pop
