// Original RNN calls execute only on the service owner. Local pins cover the
// RPC; admitted backend leases survive host results until physical completion.
namespace {
using RnnEntries = std::array<std::shared_ptr<DnnEntry>, 5>;
constexpr std::array kRnnDescriptorKinds{
    ExecutionDnnObjectKind::kRnn, ExecutionDnnObjectKind::kRnnData,
    ExecutionDnnObjectKind::kRnnData, ExecutionDnnObjectKind::kTensor,
    ExecutionDnnObjectKind::kTensor};

template <typename Function>
cudnnStatus_t with_rnn_descriptors(State& shared, Exclusion& exclusion, Entry& handle,
                                   const std::array<const void*, 5>& handles, bool is_build,
                                   Function&& function) {
    auto* metadata = shared.dnn_metadata_context;
    if (!metadata || !metadata->connection || !metadata->is_alive || metadata->is_uncertain ||
        metadata->is_closing)
        return CUDNN_STATUS_NOT_INITIALIZED;
    if (metadata->active_calls >= kMaxHandles - 1 ||
        (is_build && metadata->active_mutations >= kMaxHandles - 1))
        return dnn_profile::host_allocation_status();
    CounterPin owner(metadata->active_calls, exclusion);
    std::optional<CounterPin> work;
    if (is_build) work.emplace(metadata->active_mutations, exclusion);
    auto& context = *handle.context;
    const auto epoch = context.epoch;
    const auto count = is_build ? 1U : handles.size();
    RnnEntries descriptors;
    for (;;) {
        if (!handle.is_alive || handle.is_uncertain || !context.is_alive || context.is_closing ||
            context.is_uncertain || context.epoch != epoch || metadata->is_closing ||
            metadata->is_uncertain)
            return CUDNN_STATUS_NOT_INITIALIZED;
        bool is_waiting{};
        for (std::size_t i = 0; i != count; ++i) {
            auto& descriptor = descriptors[i];
            if (i >= 3 && !handles[i]) {
                descriptor.reset();
                continue;
            }
            descriptor = find_dnn_metadata(shared, handles[i], kRnnDescriptorKinds[i]);
            if (!descriptor || descriptor->context != metadata || !valid_dnn_metadata(*descriptor))
                return CUDNN_STATUS_NOT_INITIALIZED;
            if (descriptor->active_calls >= kMaxHandles - 1)
                return dnn_profile::host_allocation_status();
            is_waiting |= descriptor->is_retiring || (is_build && descriptor->active_calls);
        }
        if (!is_waiting) break;
        // No partial object pins or exclusive flags while waiting. Reciprocal
        // callers can finish, and every identity is revalidated after yielding.
        if (!yield_frontend(exclusion)) return CUDNN_STATUS_NOT_INITIALIZED;
    }
    std::optional<FlagPin> mutation;
    if (is_build) mutation.emplace(descriptors[0]->is_retiring, exclusion);
    std::array<std::optional<CounterPin>, 5> pins;
    for (std::size_t i = 0; i != count; ++i) {
        if (!descriptors[i]) continue;
        bool is_duplicate{};
        for (std::size_t previous = 0; previous != i; ++previous)
            is_duplicate |= descriptors[previous] == descriptors[i];
        if (!is_duplicate) pins[i].emplace(descriptors[i]->active_calls, exclusion);
    }
    return function(*metadata, descriptors);
}

cudnnStatus_t rnn_execution_status(const std::optional<ExecutionWireReply>& reply, Entry& handle,
                                   Entry& metadata, const RnnEntries& descriptors, bool is_build) {
    auto& context = *handle.context;
    const auto fault_metadata = [&] {
        metadata.is_uncertain = true;
        descriptors[0]->is_uncertain = true;
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
    if (!reply || reply->address || !reply->data.empty() || !reply->arguments.empty() ||
        reply->parameter_bytes > 3 || (is_build && reply->id))
        return invalid();
    const auto flags = reply->parameter_bytes;
    const auto original = reply->original_status;
    if (reply->disposition == ExecutionWireDisposition::kRejected) {
        constexpr auto driver_max =
            std::bit_ceil(static_cast<std::uint32_t>(CUDA_ERROR_UNKNOWN) + 1U) - 1U;
        if (reply->status <= 0 || static_cast<std::uint32_t>(reply->status) > driver_max ||
            (original && (!dnn_profile::status_is_representable(*original) || !(flags & 2U) ||
                          (!is_build && !reply->id))))
            return invalid();
        if (flags & 1U) fault_metadata();
        if (flags & 2U) fault_context();
        return dnn_error(static_cast<CUresult>(reply->status));
    }
    if (reply->disposition != ExecutionWireDisposition::kComplete || !original ||
        !dnn_profile::status_is_representable(*original) || reply->status != *original ||
        (!is_build && !reply->id))
        return invalid();
    if (*original != CUDNN_STATUS_SUCCESS) {
        // Build may have partially populated native library state. Ordinary
        // computation errors with complete physical/stream proof remain healthy.
        if (flags != (is_build ? 1U : 0U)) return invalid();
        if (is_build) fault_metadata();
        return static_cast<cudnnStatus_t>(*original);
    }
    if (flags || !context.is_alive || context.is_uncertain || !handle.is_alive ||
        handle.is_uncertain || !metadata.is_alive || metadata.is_uncertain ||
        std::ranges::any_of(descriptors, [](const auto& descriptor) {
            return descriptor && !valid_dnn_metadata(*descriptor);
        }))
        return invalid();
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t submit_dnn_rnn(cudnnHandle_t handle, ExecutionDnnRnnComputeKind kind,
                             std::uint32_t mode, const std::array<const void*, 5>& handles,
                             const std::array<const void*, 16>& pointers,
                             const std::array<std::size_t, 3>& sizes) noexcept {
    return dnn_handle(handle, false, true, [&](State& shared, Exclusion& exclusion, Entry& entry) {
        return with_rnn_descriptors(
            shared, exclusion, entry, handles, false,
            [&](Entry& metadata, const RnnEntries& descriptors) {
                auto& context = *entry.context;
                if (context.submissions == UINT64_MAX) return dnn_profile::host_allocation_status();
                ExecutionDnnRnnCompute value;
                value.kind = kind;
                value.mode = mode;
                value.handle = entry.remote;
                value.metadata = metadata.connection->session();
                for (std::size_t i = 0; i != descriptors.size(); ++i)
                    if (descriptors[i]) value.descriptors[i] = descriptors[i]->remote;
                value.weight_bytes = sizes[0];
                value.workspace_bytes = sizes[1];
                value.reserve_bytes = sizes[2];
                std::array<std::optional<CounterPin>, 16> buffer_pins, module_pins;
                for (std::size_t i = 0; i != pointers.size(); ++i) {
                    const auto address = reinterpret_cast<CUdeviceptr>(pointers[i]);
                    if (!address) continue;
                    // The backend derives complete tensor/sequence extents from
                    // original metadata. Explicit byte ranges are proved here,
                    // including nonnull zero-size one-past-end owned pointers.
                    const auto bytes = i >= 13 ? sizes[i - 13] : 1;
                    auto* allocation = shared.range(address, bytes);
                    if (!allocation || allocation->context != &context ||
                        shared.range_is_retiring(*allocation))
                        return CUDNN_STATUS_BAD_PARAM;
                    if (allocation->active_calls >= kMaxHandles - 1)
                        return dnn_profile::host_allocation_status();
                    buffer_pins[i].emplace(allocation->active_calls, exclusion);
                    if (allocation->kind == Kind::kModuleGlobal) {
                        const auto parent =
                            std::ranges::find_if(shared.entries, [&](const auto& module) {
                                return module->kind == Kind::kModule &&
                                       module->context == &context &&
                                       module->remote == allocation->parent && module->is_alive &&
                                       !module->is_retiring;
                            });
                        if (parent == shared.entries.end()) return CUDNN_STATUS_BAD_PARAM;
                        if ((*parent)->active_calls >= kMaxHandles - 1)
                            return dnn_profile::host_allocation_status();
                        module_pins[i].emplace((*parent)->active_calls, exclusion);
                    }
                    value.buffers[i] = {allocation->remote, address - allocation->address};
                }
                ExecutionPayloadWriter payload;
                if (!write_execution_dnn_rnn_compute(payload, value))
                    return dnn_profile::host_allocation_status();
                ++context.submissions;
                return rnn_execution_status(
                    shared.call(exclusion, context, ExecutionOpcode::kDnnRnnCompute,
                                std::move(payload)),
                    entry, metadata, descriptors, false);
            });
    });
}
}  // namespace

// NOLINTBEGIN(bugprone-easily-swappable-parameters): exact original public SDK ABIs.
extern "C" cudnnStatus_t CUDNNWINAPI cudnnRNNForward(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnn, cudnnForwardMode_t mode,
    const std::int32_t lengths[], cudnnRNNDataDescriptor_t x_descriptor, const void* x,
    cudnnRNNDataDescriptor_t y_descriptor, void* y, cudnnTensorDescriptor_t h_descriptor,
    const void* hx, void* hy, cudnnTensorDescriptor_t c_descriptor, const void* cx, void* cy,
    std::size_t weight_bytes, const void* weight, std::size_t workspace_bytes, void* workspace,
    std::size_t reserve_bytes, void* reserve) {
    return submit_dnn_rnn(handle, ExecutionDnnRnnComputeKind::kForward, dnn_enum_bits(mode),
                          {rnn, x_descriptor, y_descriptor, h_descriptor, c_descriptor},
                          {lengths, x, y, nullptr, nullptr, hx, hy, nullptr, nullptr, cx, cy,
                           nullptr, nullptr, weight, workspace, reserve},
                          {weight_bytes, workspace_bytes, reserve_bytes});
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnRNNBackwardData_v8(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnn, const std::int32_t lengths[],
    cudnnRNNDataDescriptor_t y_descriptor, const void* y, const void* dy,
    cudnnRNNDataDescriptor_t x_descriptor, void* dx, cudnnTensorDescriptor_t h_descriptor,
    const void* hx, const void* dhy, void* dhx, cudnnTensorDescriptor_t c_descriptor,
    const void* cx, const void* dcy, void* dcx, std::size_t weight_bytes, const void* weight,
    std::size_t workspace_bytes, void* workspace, std::size_t reserve_bytes, void* reserve) {
    return submit_dnn_rnn(handle, ExecutionDnnRnnComputeKind::kBackwardData, 0,
                          {rnn, x_descriptor, y_descriptor, h_descriptor, c_descriptor},
                          {lengths, nullptr, y, dy, dx, hx, nullptr, dhy, dhx, cx, nullptr, dcy,
                           dcx, weight, workspace, reserve},
                          {weight_bytes, workspace_bytes, reserve_bytes});
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnRNNBackwardWeights_v8(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnn, cudnnWgradMode_t mode,
    const std::int32_t lengths[], cudnnRNNDataDescriptor_t x_descriptor, const void* x,
    cudnnTensorDescriptor_t h_descriptor, const void* hx, cudnnRNNDataDescriptor_t y_descriptor,
    const void* y, std::size_t weight_bytes, void* weight, std::size_t workspace_bytes,
    void* workspace, std::size_t reserve_bytes, void* reserve) {
    return submit_dnn_rnn(handle, ExecutionDnnRnnComputeKind::kBackwardWeights, dnn_enum_bits(mode),
                          {rnn, x_descriptor, y_descriptor, h_descriptor, nullptr},
                          {lengths, x, y, nullptr, nullptr, hx, nullptr, nullptr, nullptr, nullptr,
                           nullptr, nullptr, nullptr, weight, workspace, reserve},
                          {weight_bytes, workspace_bytes, reserve_bytes});
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnBuildRNNDynamic(cudnnHandle_t handle,
                                                          cudnnRNNDescriptor_t rnn,
                                                          int mini_batch) {
    return dnn_handle(handle, false, false, [&](State& shared, Exclusion& exclusion, Entry& entry) {
        return with_rnn_descriptors(
            shared, exclusion, entry, {rnn, nullptr, nullptr, nullptr, nullptr}, true,
            [&](Entry& metadata, const RnnEntries& descriptors) {
                const ExecutionDnnBuildRnnDynamic value{entry.remote,
                                                        metadata.connection->session(),
                                                        descriptors[0]->remote, mini_batch};
                ExecutionPayloadWriter payload;
                if (!write_execution_dnn_build_rnn_dynamic(payload, value))
                    return dnn_profile::host_allocation_status();
                return rnn_execution_status(
                    shared.call(exclusion, *entry.context, ExecutionOpcode::kDnnBuildRnnDynamic,
                                std::move(payload)),
                    entry, metadata, descriptors, true);
            });
    });
}
// NOLINTEND(bugprone-easily-swappable-parameters)
