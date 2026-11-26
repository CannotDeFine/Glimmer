// Original, context-authorized descriptor mutation and bounded pointer mapping.
// Returned offsets never create a new device allocation or an execution lease.
namespace {
using RnnWeightEntries = std::array<std::shared_ptr<DnnEntry>, 3>;

cudnnStatus_t rnn_weight_params_status(const std::optional<ExecutionWireReply>& reply,
                                       Entry& handle, Entry& metadata,
                                       const RnnWeightEntries& descriptors,
                                       const ExecutionDnnRnnWeightParams& request,
                                       std::uint64_t address, std::array<void*, 2>& output) {
    auto& context = *handle.context;
    const auto fault_metadata = [&] {
        metadata.is_uncertain = true;
        for (std::size_t i = 1; i != descriptors.size(); ++i)
            if (descriptors[i]) descriptors[i]->is_uncertain = true;
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
            (original && (!dnn_profile::status_is_representable(*original) ||
                          (flags != 3 && !(flags == 1 && *original == CUDNN_STATUS_SUCCESS)))))
            return invalid();
        if (flags & 1U) fault_metadata();
        if (flags & 2U) fault_context();
        return dnn_error(static_cast<CUresult>(reply->status));
    }
    if (reply->disposition != ExecutionWireDisposition::kComplete || !original ||
        !dnn_profile::status_is_representable(*original) || reply->status != *original)
        return invalid();
    if (*original != CUDNN_STATUS_SUCCESS) {
        if (flags != 1 || !reply->data.empty()) return invalid();
        fault_metadata();
        return static_cast<cudnnStatus_t>(*original);
    }
    if (flags || !context.is_alive || context.is_uncertain || !handle.is_alive ||
        handle.is_uncertain || !metadata.is_alive || metadata.is_uncertain ||
        std::ranges::any_of(descriptors, [](const auto& descriptor) {
            return descriptor && !valid_dnn_metadata(*descriptor);
        }))
        return invalid();
    ExecutionDnnRnnWeightParamsResult value;
    ExecutionPayloadReader input(reply->data);
    if (!read_execution_dnn_rnn_weight_params_result(input, value) || !input.empty() ||
        (value.non_null & ~request.outputs))
        return invalid();
    std::array<void*, 2> mapped{};
    for (std::size_t i = 0; i != mapped.size(); ++i) {
        if (!(value.non_null & (1U << i))) continue;
        const auto offset = value.offsets[i];
        if (offset >= request.weight_bytes || offset > UINTPTR_MAX - address ||
            (!request.weights.allocation && !offset))
            return invalid();
        // The request decides owned-pointer versus raw numeric-offset mode.
        // NOLINTNEXTLINE(performance-no-int-to-ptr): validated original pointer bits.
        mapped[i] = reinterpret_cast<void*>(static_cast<std::uintptr_t>(address + offset));
    }
    output = mapped;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t query_dnn_rnn_weight_params(cudnnHandle_t handle, const void* parent,
                                          std::int32_t pseudo_layer, std::size_t weight_bytes,
                                          const void* weights, std::int32_t linear_layer,
                                          const std::array<const void*, 2>& destinations,
                                          const std::array<void**, 2>& outputs) noexcept {
    return dnn_handle(handle, false, false, [&](State& shared, Exclusion& exclusion, Entry& entry) {
        for (std::size_t i = 0; i != outputs.size(); ++i)
            if ((destinations[i] && !outputs[i]) ||
                (outputs[i] && !writable_host_range(outputs[i], sizeof(*outputs[i]))))
                return CUDNN_STATUS_BAD_PARAM;
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
        const std::array handles{parent, destinations[0], destinations[1]};
        RnnWeightEntries descriptors;
        for (;;) {
            if (!entry.is_alive || entry.is_uncertain || !context.is_alive || context.is_closing ||
                context.is_uncertain || context.epoch != epoch || metadata->is_closing ||
                metadata->is_uncertain)
                return CUDNN_STATUS_NOT_INITIALIZED;
            bool is_waiting{};
            for (std::size_t i = 0; i != descriptors.size(); ++i) {
                auto& descriptor = descriptors[i];
                if (i && !handles[i]) {
                    descriptor.reset();
                    continue;
                }
                descriptor = find_dnn_metadata(
                    shared, handles[i],
                    i ? ExecutionDnnObjectKind::kTensor : ExecutionDnnObjectKind::kRnn);
                if (!descriptor || descriptor->context != metadata ||
                    !valid_dnn_metadata(*descriptor))
                    return CUDNN_STATUS_NOT_INITIALIZED;
                if (descriptor->active_calls >= kMaxHandles - 1)
                    return dnn_profile::host_allocation_status();
                is_waiting |= descriptor->is_retiring || (i && descriptor->active_calls);
            }
            if (!is_waiting) break;
            // Acquire no partial descriptor pins or destination flags while
            // yielding. In particular, aliased outputs must not self-deadlock.
            if (!yield_frontend(exclusion)) return CUDNN_STATUS_NOT_INITIALIZED;
        }
        std::array<std::optional<FlagPin>, 2> mutations;
        std::array<std::optional<CounterPin>, 3> pins;
        for (std::size_t i = 0; i != descriptors.size(); ++i) {
            if (!descriptors[i]) continue;
            bool duplicate{};
            for (std::size_t previous = 0; previous != i; ++previous)
                duplicate |= descriptors[i] == descriptors[previous];
            if (duplicate) continue;
            if (i) mutations[i - 1].emplace(descriptors[i]->is_retiring, exclusion);
            pins[i].emplace(descriptors[i]->active_calls, exclusion);
        }
        ExecutionDnnRnnWeightParams value;
        value.handle = entry.remote;
        value.metadata = metadata->connection->session();
        value.descriptor = descriptors[0]->remote;
        value.pseudo_layer = pseudo_layer;
        value.linear_layer = linear_layer;
        value.weight_bytes = weight_bytes;
        for (std::size_t i = 0; i != outputs.size(); ++i) {
            if (descriptors[i + 1]) value.destinations[i] = descriptors[i + 1]->remote;
            if (outputs[i]) value.outputs |= 1U << i;
        }
        const auto address = reinterpret_cast<CUdeviceptr>(weights);
        Entry* allocation{};
        Entry* module_owner{};
        std::optional<CounterPin> allocation_pin, module_pin;
        if (address) {
            allocation = shared.range(address, weight_bytes);
            if (!allocation || allocation->context != &context || allocation->is_uncertain ||
                shared.range_is_retiring(*allocation))
                return CUDNN_STATUS_BAD_PARAM;
            if (allocation->active_calls >= kMaxHandles - 1)
                return dnn_profile::host_allocation_status();
            allocation_pin.emplace(allocation->active_calls, exclusion);
            if (allocation->kind == Kind::kModuleGlobal) {
                const auto module =
                    std::ranges::find_if(shared.entries, [&](const auto& candidate) {
                        return candidate->kind == Kind::kModule && candidate->context == &context &&
                               candidate->remote == allocation->parent && candidate->is_alive &&
                               !candidate->is_retiring && !candidate->is_uncertain;
                    });
                if (module == shared.entries.end()) return CUDNN_STATUS_BAD_PARAM;
                if ((*module)->active_calls >= kMaxHandles - 1)
                    return dnn_profile::host_allocation_status();
                module_owner = module->get();
                module_pin.emplace((*module)->active_calls, exclusion);
            }
            value.weights = {allocation->remote, address - allocation->address};
        }
        ExecutionPayloadWriter payload;
        if (!write_execution_dnn_rnn_weight_params(payload, value))
            return dnn_profile::host_allocation_status();
        const auto reply = shared.call(exclusion, context, ExecutionOpcode::kDnnRnnWeightParams,
                                       std::move(payload));
        std::array<void*, 2> mapped{};
        const auto status =
            rnn_weight_params_status(reply, entry, *metadata, descriptors, value, address, mapped);
        if (status != CUDNN_STATUS_SUCCESS) return status;
        if ((allocation && (!allocation->is_alive || allocation->is_uncertain ||
                            allocation->context != &context)) ||
            (module_owner && (!module_owner->is_alive || module_owner->is_uncertain ||
                              module_owner->context != &context))) {
            metadata->is_uncertain = context.is_uncertain = entry.is_uncertain = true;
            return CUDNN_STATUS_INTERNAL_ERROR;
        }
        // Host mappings can change while RPC releases the frontend mutex. Check
        // BOTH spans before publishing either; exact aliases retain native order.
        for (auto* output : outputs)
            if (output && !writable_host_range(output, sizeof(*output)))
                return CUDNN_STATUS_BAD_PARAM;
        for (std::size_t i = 0; i != outputs.size(); ++i)
            if (outputs[i] && !publish(outputs[i], mapped[i])) return CUDNN_STATUS_BAD_PARAM;
        return CUDNN_STATUS_SUCCESS;
    });
}
}  // namespace

// NOLINTBEGIN(bugprone-easily-swappable-parameters): exact installed public SDK ABI.
extern "C" cudnnStatus_t CUDNNWINAPI
cudnnGetRNNWeightParams(cudnnHandle_t handle, cudnnRNNDescriptor_t descriptor,
                        std::int32_t pseudo_layer, std::size_t weight_bytes, const void* weights,
                        std::int32_t linear_layer, cudnnTensorDescriptor_t matrix,
                        void** matrix_address, cudnnTensorDescriptor_t bias, void** bias_address) {
    return query_dnn_rnn_weight_params(handle, descriptor, pseudo_layer, weight_bytes, weights,
                                       linear_layer, {matrix, bias},
                                       {matrix_address, bias_address});
}
// NOLINTEND(bugprone-easily-swappable-parameters)
