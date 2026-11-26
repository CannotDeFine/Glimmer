// Dropout initialization executes only in the owning service context. Descriptor
// lifecycle stays on DNM1; a configuration is an exclusive queued mutation.
namespace {
cudnnStatus_t configure_dnn_dropout(cudnnDropoutDescriptor_t descriptor, cudnnHandle_t handle,
                                    float probability, void* states, std::size_t state_bytes,
                                    unsigned long long seed,
                                    ExecutionDnnDropoutKind kind) noexcept {
    static_assert(sizeof(seed) == sizeof(std::uint64_t));
    static_assert(sizeof(probability) == sizeof(std::uint32_t));
    return dnn_handle(handle, false, true, [&](State& shared, Exclusion& exclusion, Entry& entry) {
        const auto object = find_dnn_metadata(shared, descriptor, ExecutionDnnObjectKind::kDropout);
        if (!object || !valid_dnn_metadata(*object)) return CUDNN_STATUS_NOT_INITIALIZED;
        auto& metadata = *object->context;
        auto& context = *entry.context;
        const auto epoch = context.epoch;
        const auto valid = [&] {
            return valid_dnn_metadata(*object) && entry.is_alive && !entry.is_uncertain &&
                   context.is_alive && !context.is_closing && !context.is_uncertain &&
                   context.epoch == epoch;
        };
        if (metadata.active_calls >= kMaxHandles - 1 ||
            metadata.active_mutations >= kMaxHandles - 1 || context.submissions == UINT64_MAX)
            return dnn_profile::host_allocation_status();
        CounterPin owner(metadata.active_calls, exclusion);
        CounterPin work(metadata.active_mutations, exclusion);
        while (object->is_retiring)
            if (!yield_frontend(exclusion) || !valid()) return CUDNN_STATUS_NOT_INITIALIZED;
        FlagPin retiring(object->is_retiring, exclusion);
        while (object->active_calls)
            if (!yield_frontend(exclusion) || !valid()) return CUDNN_STATUS_NOT_INITIALIZED;
        CounterPin mutation(object->active_calls, exclusion);

        ExecutionDnnDropout configuration;
        configuration.metadata = metadata.connection->session();
        configuration.descriptor = object->remote;
        configuration.handle = entry.remote;
        configuration.state_bytes = state_bytes;
        configuration.seed = seed;
        configuration.probability_bits = std::bit_cast<std::uint32_t>(probability);
        configuration.kind = kind;
        std::optional<CounterPin> buffer_pin, module_pin;
        const auto address = reinterpret_cast<CUdeviceptr>(states);
        if (address) {
            // Zero bytes still identifies an owned allocation, including its
            // one-past-end pointer. A null pointer preserves its separate count.
            auto* allocation = shared.range(address, state_bytes);
            if (!allocation || allocation->context != &context ||
                shared.range_is_retiring(*allocation))
                return CUDNN_STATUS_BAD_PARAM;
            if (allocation->active_calls >= kMaxHandles - 1)
                return dnn_profile::host_allocation_status();
            buffer_pin.emplace(allocation->active_calls, exclusion);
            if (allocation->kind == Kind::kModuleGlobal) {
                const auto parent = std::ranges::find_if(shared.entries, [&](const auto& module) {
                    return module->kind == Kind::kModule && module->context == &context &&
                           module->remote == allocation->parent && module->is_alive &&
                           !module->is_retiring;
                });
                if (parent == shared.entries.end()) return CUDNN_STATUS_BAD_PARAM;
                if ((*parent)->active_calls >= kMaxHandles - 1)
                    return dnn_profile::host_allocation_status();
                module_pin.emplace((*parent)->active_calls, exclusion);
            }
            configuration.states = {allocation->remote, address - allocation->address};
        }
        ExecutionPayloadWriter payload;
        if (!write_execution_dnn_dropout(payload, configuration))
            return dnn_profile::host_allocation_status();
        // Waiting for an earlier mutation can release the frontend lock.
        if (context.submissions == UINT64_MAX) return dnn_profile::host_allocation_status();
        ++context.submissions;
        const auto reply =
            shared.call(exclusion, context, ExecutionOpcode::kDnnDropout, std::move(payload));
        auto result = dnn_status(reply, context, &entry.is_uncertain);
        if (result == CUDNN_STATUS_SUCCESS && !reply->data.empty())
            result = dnn_invalid(context, &entry.is_uncertain);
        if (context.is_uncertain) {
            metadata.is_uncertain = true;
            object->is_uncertain = true;
        }
        // These local pins cover the RPC. The service's reservation and physical
        // leases outlive a host result or acknowledgment when initialization is pending.
        return result;
    });
}
}  // namespace

extern "C" cudnnStatus_t CUDNNWINAPI
cudnnCreateDropoutDescriptor(cudnnDropoutDescriptor_t* output) {
    return create_dnn_metadata(output, ExecutionDnnObjectKind::kDropout);
}
extern "C" cudnnStatus_t CUDNNWINAPI
cudnnDestroyDropoutDescriptor(cudnnDropoutDescriptor_t descriptor) {
    return destroy_dnn_metadata(descriptor, ExecutionDnnObjectKind::kDropout);
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnDropoutGetStatesSize(cudnnHandle_t handle,
                                                               std::size_t* bytes) {
    return dnn_handle(handle, false, false, [&](State& shared, Exclusion& exclusion, Entry& entry) {
        if (!writable_host_range(bytes, sizeof(*bytes))) return CUDNN_STATUS_BAD_PARAM;
        ExecutionPayloadWriter payload;
        if (!payload.write_u64(entry.remote)) return dnn_profile::host_allocation_status();
        const auto reply = shared.call(exclusion, *entry.context,
                                       ExecutionOpcode::kDnnDropoutStatesSize, std::move(payload));
        const auto result = dnn_status(reply, *entry.context, &entry.is_uncertain);
        if (result != CUDNN_STATUS_SUCCESS) return result;
        if (!reply->data.empty() || reply->address > SIZE_MAX)
            return dnn_invalid(*entry.context, &entry.is_uncertain);
        if (!writable_host_range(bytes, sizeof(*bytes))) return CUDNN_STATUS_BAD_PARAM;
        return publish(bytes, static_cast<std::size_t>(reply->address)) ? CUDNN_STATUS_SUCCESS
                                                                        : CUDNN_STATUS_BAD_PARAM;
    });
}
// NOLINTBEGIN(bugprone-easily-swappable-parameters): exact public SDK ABIs.
extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetDropoutDescriptor(cudnnDropoutDescriptor_t descriptor,
                                                               cudnnHandle_t handle,
                                                               float probability, void* states,
                                                               std::size_t state_bytes,
                                                               unsigned long long seed) {
    return configure_dnn_dropout(descriptor, handle, probability, states, state_bytes, seed,
                                 ExecutionDnnDropoutKind::kSet);
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnRestoreDropoutDescriptor(
    cudnnDropoutDescriptor_t descriptor, cudnnHandle_t handle, float probability, void* states,
    std::size_t state_bytes, unsigned long long seed) {
    return configure_dnn_dropout(descriptor, handle, probability, states, state_bytes, seed,
                                 ExecutionDnnDropoutKind::kRestore);
}
// These four handlers are explicit stubs in the pinned implementation. Keep
// their exact typed entries from escaping into caller-local vendor state.
extern "C" cudnnStatus_t CUDNNWINAPI cudnnDropoutGetReserveSpaceSize(cudnnTensorDescriptor_t,
                                                                     std::size_t*) {
    return dnn_unsupported();
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetDropoutDescriptor(cudnnDropoutDescriptor_t,
                                                               cudnnHandle_t, float*, void**,
                                                               unsigned long long*) {
    return dnn_unsupported();
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnDropoutForward(cudnnHandle_t,
                                                         const cudnnDropoutDescriptor_t,
                                                         const cudnnTensorDescriptor_t, const void*,
                                                         const cudnnTensorDescriptor_t, void*,
                                                         void*, std::size_t) {
    return dnn_unsupported();
}
extern "C" cudnnStatus_t CUDNNWINAPI
cudnnDropoutBackward(cudnnHandle_t, const cudnnDropoutDescriptor_t, const cudnnTensorDescriptor_t,
                     const void*, const cudnnTensorDescriptor_t, void*, void*, std::size_t) {
    return dnn_unsupported();
}
// NOLINTEND(bugprone-easily-swappable-parameters)
