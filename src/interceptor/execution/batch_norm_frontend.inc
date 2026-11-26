// Included after the DNN metadata frontends. No caller-native SDK work occurs.
namespace {
template <typename Function>
cudnnStatus_t with_batch_norm_descriptors(State& shared, Exclusion& exclusion,
                                          const std::array<const void*, 7>& handles,
                                          Function&& function) {
    auto* metadata = shared.dnn_metadata_context;
    if (!metadata || !metadata->connection || !metadata->is_alive || metadata->is_uncertain ||
        metadata->is_closing)
        return CUDNN_STATUS_NOT_INITIALIZED;
    std::array<std::shared_ptr<DnnEntry>, 7> descriptors;
    for (;;) {
        bool waiting{};
        for (std::size_t i = 0; i != handles.size(); ++i) {
            if (!handles[i]) continue;
            auto& descriptor = descriptors[i];
            descriptor = find_dnn_metadata(
                shared, handles[i],
                i == 6 ? ExecutionDnnObjectKind::kActivation : ExecutionDnnObjectKind::kTensor);
            if (!descriptor || descriptor->context != metadata || !valid_dnn_metadata(*descriptor))
                return CUDNN_STATUS_NOT_INITIALIZED;
            if (descriptor->active_calls > kMaxHandles - descriptors.size())
                return dnn_profile::host_allocation_status();
            waiting |= descriptor->is_retiring;
        }
        if (!waiting) break;
        if (!yield_frontend(exclusion) || metadata->is_uncertain || metadata->is_closing)
            return CUDNN_STATUS_NOT_INITIALIZED;
    }
    if (metadata->active_calls >= kMaxHandles - 1) return dnn_profile::host_allocation_status();
    CounterPin owner(metadata->active_calls, exclusion);
    std::array<std::optional<CounterPin>, 7> pins;
    for (std::size_t i = 0; i != handles.size(); ++i)
        if (descriptors[i]) pins[i].emplace(descriptors[i]->active_calls, exclusion);
    return function(*metadata, descriptors);
}
cudnnStatus_t query_batch_norm_size(cudnnHandle_t handle, ExecutionDnnBatchNormQuery query,
                                    const std::array<const void*, 7>& handles,
                                    std::size_t* bytes) noexcept {
    return dnn_handle(handle, false, false, [&](State& shared, Exclusion& exclusion, Entry& entry) {
        if (!writable_host_range(bytes, sizeof(*bytes))) return CUDNN_STATUS_BAD_PARAM;
        return with_batch_norm_descriptors(
            shared, exclusion, handles, [&](Entry& metadata, const auto& descriptors) {
                query.handle = entry.remote;
                query.metadata = metadata.connection->session();
                for (std::size_t i = 0; i != descriptors.size(); ++i)
                    query.descriptors[i] = descriptors[i] ? descriptors[i]->remote : 0;
                ExecutionPayloadWriter payload;
                if (!write_execution_dnn_batch_norm_query(payload, query))
                    return dnn_profile::host_allocation_status();
                const auto reply =
                    shared.call(exclusion, *entry.context, ExecutionOpcode::kDnnBatchNormQuery,
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
struct BatchNormArguments {
    ExecutionDnnBatchNormQuery query;
    std::array<const void*, 7> descriptors{};
    std::array<const void*, 13> buffers{};
    std::array<const void*, 4> coefficients{};
    std::size_t workspace_bytes{}, reserve_bytes{};
    double epsilon{}, average_factor{};
};
cudnnStatus_t submit_batch_norm(cudnnHandle_t handle,
                                const BatchNormArguments& arguments) noexcept {
    return dnn_handle(handle, false, true, [&](State& shared, Exclusion& exclusion, Entry& entry) {
        return with_batch_norm_descriptors(
            shared, exclusion, arguments.descriptors,
            [&](Entry& metadata, const auto& descriptors) {
                if (!descriptors[0]) return CUDNN_STATUS_BAD_PARAM;
                if (entry.context->submissions == UINT64_MAX)
                    return dnn_profile::host_allocation_status();
                // Pin every present descriptor, including ignored optional
                // ones, before the type query temporarily releases exclusion.
                ExecutionDnnTensor tensor;
                const auto queried = query_dnn_tensor(shared, exclusion, *descriptors[0],
                                                      CUDNN_DIM_MAX, false, tensor);
                if (queried != CUDNN_STATUS_SUCCESS) return queried;
                ExecutionDnnBatchNormCompute compute;
                compute.query = arguments.query;
                compute.query.handle = entry.remote;
                compute.query.metadata = metadata.connection->session();
                for (std::size_t i = 0; i != descriptors.size(); ++i)
                    compute.query.descriptors[i] = descriptors[i] ? descriptors[i]->remote : 0;
                compute.workspace_bytes = arguments.workspace_bytes;
                compute.reserve_bytes = arguments.reserve_bytes;
                compute.epsilon_bits = std::bit_cast<std::uint64_t>(arguments.epsilon);
                compute.average_factor_bits =
                    std::bit_cast<std::uint64_t>(arguments.average_factor);
                compute.scalar_bytes =
                    tensor.type == CUDNN_DATA_DOUBLE ? sizeof(double) : sizeof(float);
                const std::size_t coefficient_count =
                    compute.query.kind == ExecutionDnnBatchNormKind::kForward ? 2 : 4;
                for (std::size_t i = 0; i != coefficient_count; ++i)
                    if (!self_copy(compute.coefficients[i].data(), arguments.coefficients[i],
                                   compute.scalar_bytes, false))
                        return CUDNN_STATUS_BAD_PARAM;
                std::array<std::optional<CounterPin>, 13> buffer_pins, module_pins;
                for (std::size_t i = 0; i != arguments.buffers.size(); ++i) {
                    const auto address = reinterpret_cast<CUdeviceptr>(arguments.buffers[i]);
                    // Optional null identities remain independent of other
                    // statistics pointers and workspace/reserve byte counts.
                    if (!address) continue;
                    const auto bytes = i == 11   ? arguments.workspace_bytes
                                       : i == 12 ? arguments.reserve_bytes
                                                 : std::size_t{1};
                    auto* allocation = shared.range(address, bytes);
                    if (!allocation || allocation->context != entry.context ||
                        shared.range_is_retiring(*allocation))
                        return CUDNN_STATUS_BAD_PARAM;
                    if (allocation->active_calls > kMaxHandles - arguments.buffers.size())
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
                        if ((*parent)->active_calls > kMaxHandles - arguments.buffers.size())
                            return dnn_profile::host_allocation_status();
                        module_pins[i].emplace((*parent)->active_calls, exclusion);
                    }
                    compute.buffers[i] = {allocation->remote, address - allocation->address};
                }
                ExecutionPayloadWriter payload;
                if (!write_execution_dnn_batch_norm_compute(payload, compute))
                    return dnn_profile::host_allocation_status();
                ++entry.context->submissions;
                return dnn_status(shared.call(exclusion, *entry.context,
                                              ExecutionOpcode::kDnnBatchNorm, std::move(payload)),
                                  *entry.context, &entry.is_uncertain);
            });
    });
}
}  // namespace

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
// NOLINTBEGIN(bugprone-easily-swappable-parameters): exact public SDK ABIs.
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t operation,
    cudnnTensorDescriptor_t x, cudnnTensorDescriptor_t z, cudnnTensorDescriptor_t y,
    cudnnTensorDescriptor_t parameters, cudnnActivationDescriptor_t activation,
    std::size_t* bytes) {
    return query_batch_norm_size(
        handle,
        {ExecutionDnnBatchNormKind::kForward, dnn_enum_bits(mode), dnn_enum_bits(operation)},
        {x, y, nullptr, z, nullptr, parameters, activation}, bytes);
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetBatchNormalizationBackwardExWorkspaceSize(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t operation,
    cudnnTensorDescriptor_t x, cudnnTensorDescriptor_t y, cudnnTensorDescriptor_t dy,
    cudnnTensorDescriptor_t dz, cudnnTensorDescriptor_t dx, cudnnTensorDescriptor_t parameters,
    cudnnActivationDescriptor_t activation, std::size_t* bytes) {
    return query_batch_norm_size(
        handle,
        {ExecutionDnnBatchNormKind::kBackward, dnn_enum_bits(mode), dnn_enum_bits(operation)},
        {x, y, dy, dz, dx, parameters, activation}, bytes);
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t operation,
    cudnnActivationDescriptor_t activation, cudnnTensorDescriptor_t x, std::size_t* bytes) {
    return query_batch_norm_size(
        handle,
        {ExecutionDnnBatchNormKind::kReserve, dnn_enum_bits(mode), dnn_enum_bits(operation)},
        {x, nullptr, nullptr, nullptr, nullptr, nullptr, activation}, bytes);
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnBatchNormalizationForwardTrainingEx(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t operation,
    const void* alpha, const void* beta, cudnnTensorDescriptor_t x_descriptor, const void* x,
    cudnnTensorDescriptor_t z_descriptor, const void* z, cudnnTensorDescriptor_t y_descriptor,
    void* y, cudnnTensorDescriptor_t parameters, const void* scale, const void* bias,
    double average_factor, void* running_mean, void* running_variance, double epsilon,
    void* saved_mean, void* saved_inverse_variance, cudnnActivationDescriptor_t activation,
    void* workspace, std::size_t workspace_bytes, void* reserve, std::size_t reserve_bytes) {
    return submit_batch_norm(
        handle,
        {{ExecutionDnnBatchNormKind::kForward, dnn_enum_bits(mode), dnn_enum_bits(operation)},
         {x_descriptor, y_descriptor, nullptr, z_descriptor, nullptr, parameters, activation},
         {x, y, nullptr, z, nullptr, scale, bias, running_mean, running_variance, saved_mean,
          saved_inverse_variance, workspace, reserve},
         {alpha, beta, nullptr, nullptr},
         workspace_bytes,
         reserve_bytes,
         epsilon,
         average_factor});
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnBatchNormalizationBackwardEx(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t operation,
    const void* alpha_data, const void* beta_data, const void* alpha_parameter,
    const void* beta_parameter, cudnnTensorDescriptor_t x_descriptor, const void* x,
    cudnnTensorDescriptor_t y_descriptor, const void* y, cudnnTensorDescriptor_t dy_descriptor,
    const void* dy, cudnnTensorDescriptor_t dz_descriptor, void* dz,
    cudnnTensorDescriptor_t dx_descriptor, void* dx, cudnnTensorDescriptor_t parameters,
    const void* scale, const void* bias, void* dscale, void* dbias, double epsilon,
    const void* saved_mean, const void* saved_inverse_variance,
    cudnnActivationDescriptor_t activation, void* workspace, std::size_t workspace_bytes,
    void* reserve, std::size_t reserve_bytes) {
    return submit_batch_norm(handle, {{ExecutionDnnBatchNormKind::kBackward, dnn_enum_bits(mode),
                                       dnn_enum_bits(operation)},
                                      {x_descriptor, y_descriptor, dy_descriptor, dz_descriptor,
                                       dx_descriptor, parameters, activation},
                                      {x, y, dy, dz, dx, scale, bias, dscale, dbias, saved_mean,
                                       saved_inverse_variance, workspace, reserve},
                                      {alpha_data, beta_data, alpha_parameter, beta_parameter},
                                      workspace_bytes,
                                      reserve_bytes,
                                      epsilon,
                                      0});
}
// These four pinned reference handlers are stubs, not supported computation.
extern "C" cudnnStatus_t CUDNNWINAPI cudnnDeriveBNTensorDescriptor(cudnnTensorDescriptor_t,
                                                                   cudnnTensorDescriptor_t,
                                                                   cudnnBatchNormMode_t) {
    return dnn_unsupported();
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnBatchNormalizationForwardTraining(
    cudnnHandle_t, cudnnBatchNormMode_t, const void*, const void*, cudnnTensorDescriptor_t,
    const void*, cudnnTensorDescriptor_t, void*, cudnnTensorDescriptor_t, const void*, const void*,
    double, void*, void*, double, void*, void*) {
    return dnn_unsupported();
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnBatchNormalizationForwardInference(
    cudnnHandle_t, cudnnBatchNormMode_t, const void*, const void*, cudnnTensorDescriptor_t,
    const void*, cudnnTensorDescriptor_t, void*, cudnnTensorDescriptor_t, const void*, const void*,
    const void*, const void*, double) {
    return dnn_unsupported();
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnBatchNormalizationBackward(
    cudnnHandle_t, cudnnBatchNormMode_t, const void*, const void*, const void*, const void*,
    cudnnTensorDescriptor_t, const void*, cudnnTensorDescriptor_t, const void*,
    cudnnTensorDescriptor_t, void*, cudnnTensorDescriptor_t, const void*, void*, void*, double,
    const void*, const void*) {
    return dnn_unsupported();
}
// NOLINTEND(bugprone-easily-swappable-parameters)
#pragma GCC diagnostic pop
