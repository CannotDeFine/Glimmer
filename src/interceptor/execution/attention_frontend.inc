// Window preparation observes owned native metadata, not a caller-side cache.
// Pins span preparation, bounded host capture and queued original submission.
namespace {
using AttnEntries = std::array<std::shared_ptr<DnnEntry>, 5>;

template <typename Function>
cudnnStatus_t with_attn_descriptors(State& shared, Exclusion& exclusion, Entry& handle,
                                    const std::array<const void*, 5>& handles,
                                    Function&& function) {
    auto* metadata = shared.dnn_metadata_context;
    if (!metadata || !metadata->connection || !metadata->is_alive || metadata->is_uncertain ||
        metadata->is_closing)
        return CUDNN_STATUS_NOT_INITIALIZED;
    if (metadata->active_calls >= kMaxHandles - 1) return dnn_profile::host_allocation_status();
    CounterPin owner(metadata->active_calls, exclusion);
    auto& context = *handle.context;
    const auto epoch = context.epoch;
    AttnEntries descriptors;
    for (;;) {
        if (!handle.is_alive || handle.is_uncertain || !context.is_alive || context.is_closing ||
            context.is_uncertain || context.epoch != epoch || !metadata->is_alive ||
            metadata->is_closing || metadata->is_uncertain)
            return CUDNN_STATUS_NOT_INITIALIZED;
        bool is_waiting{};
        for (std::size_t index = 0; index != descriptors.size(); ++index) {
            auto& descriptor = descriptors[index];
            if (index && !handles[index]) {
                descriptor.reset();
                continue;
            }
            descriptor = find_dnn_metadata(
                shared, handles[index],
                index ? ExecutionDnnObjectKind::kSeqData : ExecutionDnnObjectKind::kAttn);
            if (!descriptor || descriptor->context != metadata || !valid_dnn_metadata(*descriptor))
                return CUDNN_STATUS_NOT_INITIALIZED;
            if (descriptor->active_calls >= kMaxHandles - 1)
                return dnn_profile::host_allocation_status();
            is_waiting |= descriptor->is_retiring;
        }
        if (!is_waiting) break;
        // Recheck all positions after every yield, without a partial object pin.
        if (!yield_frontend(exclusion)) return CUDNN_STATUS_NOT_INITIALIZED;
    }
    std::array<std::optional<CounterPin>, 5> pins;
    for (std::size_t index = 0; index != descriptors.size(); ++index) {
        if (!descriptors[index]) continue;
        bool duplicate{};
        for (std::size_t previous = 0; previous != index; ++previous)
            duplicate |= descriptors[previous] == descriptors[index];
        if (!duplicate) pins[index].emplace(descriptors[index]->active_calls, exclusion);
    }
    return function(*metadata, descriptors);
}

cudnnStatus_t attn_preparation_status(const std::optional<ExecutionWireReply>& reply, Entry& handle,
                                      Entry& metadata, const AttnEntries& descriptors,
                                      const ExecutionDnnAttnCompute& request,
                                      ExecutionDnnAttnWindowRange& output) {
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
    if (!reply || reply->id || reply->address || reply->original_status ||
        !reply->arguments.empty() || reply->parameter_bytes > 3)
        return invalid();
    const auto flags = reply->parameter_bytes;
    if (reply->disposition == ExecutionWireDisposition::kRejected) {
        constexpr auto maximum =
            std::bit_ceil(static_cast<std::uint32_t>(CUDA_ERROR_UNKNOWN) + 1U) - 1U;
        if (!reply->data.empty() || reply->status <= 0 ||
            static_cast<std::uint32_t>(reply->status) > maximum)
            return invalid();
        if (flags & 1U) fault_metadata();
        if (flags & 2U) fault_context();
        return dnn_error(static_cast<CUresult>(reply->status));
    }
    if (reply->disposition != ExecutionWireDisposition::kComplete || reply->status || flags ||
        !context.is_alive || context.is_uncertain || !handle.is_alive || handle.is_uncertain ||
        !metadata.is_alive || metadata.is_uncertain ||
        std::ranges::any_of(descriptors, [](const auto& descriptor) {
            return descriptor && !valid_dnn_metadata(*descriptor);
        }))
        return invalid();
    ExecutionDnnAttnWindowRange range;
    ExecutionPayloadReader input(reply->data);
    if (!read_execution_dnn_attn_window_range(input, range) || !input.empty()) return invalid();
    if (range.count) {
        if (request.window_presence != 3 ||
            request.kind == ExecutionDnnAttnComputeKind::kBackwardWeights)
            return invalid();
        if (request.kind == ExecutionDnnAttnComputeKind::kForward && request.current >= 0) {
            if (range.first != static_cast<std::uint32_t>(request.current) || range.count != 1)
                return invalid();
        } else if (range.first) {
            return invalid();
        }
    }
    output = range;
    return CUDNN_STATUS_SUCCESS;
}

bool capture_attn_window(std::vector<std::int32_t>& output, const int* source,
                         const ExecutionDnnAttnWindowRange& range) {
    if (!range.count) return true;  // No arithmetic, probing or read of an ignored caller pointer.
    const auto base = reinterpret_cast<std::uintptr_t>(source);
    const auto offset = static_cast<std::size_t>(range.first) * sizeof(int);
    const auto bytes = static_cast<std::size_t>(range.count) * sizeof(int);
    if (!source || offset > UINTPTR_MAX - base || bytes > UINTPTR_MAX - (base + offset))
        return false;
    output.resize(range.count);
    return self_copy(output.data(), reinterpret_cast<const void*>(base + offset), bytes, false);
}

cudnnStatus_t submit_dnn_attn(cudnnHandle_t handle, ExecutionDnnAttnComputeKind kind,
                              std::uint32_t mode, int current,
                              const std::array<const void*, 5>& handles,
                              const std::array<const int*, 2>& windows,
                              const std::array<const void*, 15>& pointers,
                              const std::array<std::size_t, 3>& sizes) noexcept {
    return dnn_handle(handle, false, true, [&](State& shared, Exclusion& exclusion, Entry& entry) {
        if (kind == ExecutionDnnAttnComputeKind::kBackwardWeights && mode > 1)
            return CUDNN_STATUS_BAD_PARAM;
        return with_attn_descriptors(
            shared, exclusion, entry, handles,
            [&](Entry& metadata, const AttnEntries& descriptors) {
                auto& context = *entry.context;
                if (context.submissions == UINT64_MAX) return dnn_profile::host_allocation_status();
                ExecutionDnnAttnCompute value;
                value.kind = kind;
                value.mode = mode;
                value.current = current;
                value.window_presence = (windows[0] ? 1U : 0U) | (windows[1] ? 2U : 0U);
                value.handle = entry.remote;
                value.metadata = metadata.connection->session();
                for (std::size_t index = 0; index != descriptors.size(); ++index)
                    if (descriptors[index]) value.descriptors[index] = descriptors[index]->remote;
                value.weight_bytes = sizes[0];
                value.workspace_bytes = sizes[1];
                value.reserve_bytes = sizes[2];
                std::array<std::optional<CounterPin>, 15> buffer_pins, module_pins;
                for (std::size_t index = 0; index != pointers.size(); ++index) {
                    const auto address = reinterpret_cast<CUdeviceptr>(pointers[index]);
                    if (!address) continue;
                    // Ordinary pointer extents are freshly derived by the backend.
                    // Explicit spans preserve nonnull zero-size owned endpoints.
                    const auto bytes = index >= 13 ? sizes[index - 12] : index >= 11 ? sizes[0] : 1;
                    auto* allocation = shared.range(address, bytes);
                    if (!allocation || allocation->context != &context ||
                        shared.range_is_retiring(*allocation))
                        return CUDNN_STATUS_BAD_PARAM;
                    if (allocation->active_calls >= kMaxHandles - 1)
                        return dnn_profile::host_allocation_status();
                    buffer_pins[index].emplace(allocation->active_calls, exclusion);
                    if (allocation->kind == Kind::kModuleGlobal) {
                        const auto module =
                            std::ranges::find_if(shared.entries, [&](const auto& candidate) {
                                return candidate->kind == Kind::kModule &&
                                       candidate->context == &context &&
                                       candidate->remote == allocation->parent &&
                                       candidate->is_alive && !candidate->is_retiring;
                            });
                        if (module == shared.entries.end()) return CUDNN_STATUS_BAD_PARAM;
                        if ((*module)->active_calls >= kMaxHandles - 1)
                            return dnn_profile::host_allocation_status();
                        module_pins[index].emplace((*module)->active_calls, exclusion);
                    }
                    value.buffers[index] = {allocation->remote, address - allocation->address};
                }
                if (kind != ExecutionDnnAttnComputeKind::kBackwardWeights) {
                    ExecutionPayloadWriter preparation;
                    if (!write_execution_dnn_attn_compute(preparation, value))
                        return dnn_profile::host_allocation_status();
                    ExecutionDnnAttnWindowRange range;
                    const auto status = attn_preparation_status(
                        shared.call(exclusion, context, ExecutionOpcode::kPrepareDnnAttn,
                                    std::move(preparation)),
                        entry, metadata, descriptors, value, range);
                    if (status != CUDNN_STATUS_SUCCESS) return status;
                    if (!capture_attn_window(value.lower, windows[0], range) ||
                        !capture_attn_window(value.upper, windows[1], range))
                        return CUDNN_STATUS_BAD_PARAM;
                }
                ExecutionPayloadWriter payload;
                if (!write_execution_dnn_attn_compute(payload, value))
                    return dnn_profile::host_allocation_status();
                ++context.submissions;
                // Both original families use the same admitted-command capsule:
                // no output data, exact original status and two fault domains.
                return rnn_execution_status(
                    shared.call(exclusion, context, ExecutionOpcode::kDnnAttnCompute,
                                std::move(payload)),
                    entry, metadata, descriptors, false);
            });
    });
}
}  // namespace

// NOLINTBEGIN(bugprone-easily-swappable-parameters): exact original SDK ABIs.
extern "C" cudnnStatus_t CUDNNWINAPI cudnnMultiHeadAttnForward(
    cudnnHandle_t handle, const cudnnAttnDescriptor_t attention, int current, const int lower[],
    const int upper[], const int lengths_qo[], const int lengths_kv[],
    const cudnnSeqDataDescriptor_t q_descriptor, const void* queries, const void* residual,
    const cudnnSeqDataDescriptor_t k_descriptor, const void* keys,
    const cudnnSeqDataDescriptor_t v_descriptor, const void* values,
    const cudnnSeqDataDescriptor_t o_descriptor, void* output, std::size_t weight_bytes,
    const void* weight, std::size_t workspace_bytes, void* workspace, std::size_t reserve_bytes,
    void* reserve) {
    return submit_dnn_attn(
        handle, ExecutionDnnAttnComputeKind::kForward, 0, current,
        {attention, q_descriptor, k_descriptor, v_descriptor, o_descriptor}, {lower, upper},
        {lengths_qo, lengths_kv, queries, keys, values, residual, output, nullptr, nullptr, nullptr,
         nullptr, weight, nullptr, workspace, reserve},
        {weight_bytes, workspace_bytes, reserve_bytes});
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnMultiHeadAttnBackwardData(
    cudnnHandle_t handle, const cudnnAttnDescriptor_t attention, const int lower[],
    const int upper[], const int lengths_qo[], const int lengths_kv[],
    const cudnnSeqDataDescriptor_t o_descriptor, const void* doutput,
    const cudnnSeqDataDescriptor_t q_descriptor, void* dqueries, const void* queries,
    const cudnnSeqDataDescriptor_t k_descriptor, void* dkeys, const void* keys,
    const cudnnSeqDataDescriptor_t v_descriptor, void* dvalues, const void* values,
    std::size_t weight_bytes, const void* weight, std::size_t workspace_bytes, void* workspace,
    std::size_t reserve_bytes, void* reserve) {
    return submit_dnn_attn(handle, ExecutionDnnAttnComputeKind::kBackwardData, 0, 0,
                           {attention, q_descriptor, k_descriptor, v_descriptor, o_descriptor},
                           {lower, upper},
                           {lengths_qo, lengths_kv, queries, keys, values, nullptr, nullptr,
                            doutput, dqueries, dkeys, dvalues, weight, nullptr, workspace, reserve},
                           {weight_bytes, workspace_bytes, reserve_bytes});
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnMultiHeadAttnBackwardWeights(
    cudnnHandle_t handle, const cudnnAttnDescriptor_t attention, cudnnWgradMode_t mode,
    const cudnnSeqDataDescriptor_t q_descriptor, const void* queries,
    const cudnnSeqDataDescriptor_t k_descriptor, const void* keys,
    const cudnnSeqDataDescriptor_t v_descriptor, const void* values,
    const cudnnSeqDataDescriptor_t o_descriptor, const void* doutput, std::size_t weight_bytes,
    const void* weight, void* dweight, std::size_t workspace_bytes, void* workspace,
    std::size_t reserve_bytes, void* reserve) {
    return submit_dnn_attn(
        handle, ExecutionDnnAttnComputeKind::kBackwardWeights, dnn_enum_bits(mode), 0,
        {attention, q_descriptor, k_descriptor, v_descriptor, o_descriptor}, {nullptr, nullptr},
        {nullptr, nullptr, queries, keys, values, nullptr, nullptr, doutput, nullptr, nullptr,
         nullptr, weight, dweight, workspace, reserve},
        {weight_bytes, workspace_bytes, reserve_bytes});
}
// NOLINTEND(bugprone-easily-swappable-parameters)
