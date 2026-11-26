// Original filter/bias work runs only in the service-owned context and stream.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
// NOLINTBEGIN(bugprone-easily-swappable-parameters): exact public SDK ABI.
extern "C" cudnnStatus_t CUDNNWINAPI cudnnReorderFilterAndBias(
    cudnnHandle_t handle, const cudnnFilterDescriptor_t filter, cudnnReorderType_t reorder,
    const void* input, void* output, int reorder_bias, const void* bias, void* reordered_bias) {
    static_assert(sizeof(reorder_bias) == sizeof(std::int32_t));
    return dnn_handle(handle, false, true, [&](State& shared, Exclusion& exclusion, Entry& entry) {
        const std::array<const void*, 1> handles{filter};
        constexpr std::array kinds{ExecutionDnnObjectKind::kFilter};
        return with_dnn_descriptors(
            shared, exclusion, handles, kinds, [&](Entry& metadata, const auto& descriptors) {
                if (entry.context->submissions == UINT64_MAX)
                    return dnn_profile::host_allocation_status();
                ExecutionDnnReorderFilterAndBias compute;
                compute.handle = entry.remote;
                compute.metadata = metadata.connection->session();
                compute.filter = descriptors[0]->remote;
                compute.reorder = dnn_enum_bits(reorder);
                compute.reorder_bias = reorder_bias;
                const std::array<const void*, 4> pointers{input, output, bias, reordered_bias};
                std::array<std::optional<CounterPin>, 4> buffer_pins, module_pins;
                for (std::size_t i = 0; i != pointers.size(); ++i) {
                    // An ignored argument may be any caller pointer. Do not
                    // resolve it, retain an unrelated allocation or send its bits.
                    // Negative flags are active in the qualified SDK implementation.
                    if (i >= 2 && reorder_bias == 0) continue;
                    const auto address = reinterpret_cast<CUdeviceptr>(pointers[i]);
                    auto* allocation = shared.range(address, 1);
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
                if (!write_execution_dnn_reorder_filter_and_bias(payload, compute))
                    return dnn_profile::host_allocation_status();
                ++entry.context->submissions;
                const auto reply =
                    shared.call(exclusion, *entry.context,
                                ExecutionOpcode::kDnnReorderFilterAndBias, std::move(payload));
                const auto status = dnn_status(reply, *entry.context, &entry.is_uncertain);
                if (status == CUDNN_STATUS_SUCCESS && !reply->data.empty())
                    return dnn_invalid(*entry.context, &entry.is_uncertain);
                // Local pins cover this RPC. The queued service retains complete
                // ranges and descriptor/handle leases until physical retirement.
                return status;
            });
    });
}
// NOLINTEND(bugprone-easily-swappable-parameters)
#pragma GCC diagnostic pop
