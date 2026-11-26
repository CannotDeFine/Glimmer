// Original attention layout arithmetic never reads a caller weight buffer.
// Preserve its numeric base locally; returned offsets confer no device authority.
namespace {
using AttnWeightEntries = std::array<std::shared_ptr<DnnEntry>, 2>;

cudnnStatus_t attn_weights_status(const std::optional<ExecutionWireReply>& reply, Entry& handle,
                                  Entry& metadata, const AttnWeightEntries& descriptors,
                                  const ExecutionDnnAttnWeights& request, std::uintptr_t base,
                                  void*& output) {
    auto& context = *handle.context;
    const auto fault_metadata = [&] {
        metadata.is_uncertain = true;
        if (descriptors[1]) descriptors[1]->is_uncertain = true;
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
    if (flags || !request.destination || !request.has_output ||
        (!request.has_base && request.weight_bytes) || !context.is_alive || context.is_uncertain ||
        !handle.is_alive || handle.is_uncertain || !metadata.is_alive || metadata.is_uncertain ||
        std::ranges::any_of(descriptors, [](const auto& descriptor) {
            return descriptor && !valid_dnn_metadata(*descriptor);
        }))
        return invalid();
    ExecutionDnnAttnWeightsResult value;
    ExecutionPayloadReader input(reply->data);
    if (!read_execution_dnn_attn_weights_result(input, value) || !input.empty()) return invalid();
    if (value.non_null &&
        (!request.has_base || !request.has_output || value.offset >= request.weight_bytes ||
         value.offset > UINTPTR_MAX - base))
        return invalid();
    // A null original result stays null, even when the supplied base was nonnull.
    // NOLINTNEXTLINE(performance-no-int-to-ptr): checked original address arithmetic.
    output = value.non_null ? reinterpret_cast<void*>(base + value.offset) : nullptr;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t query_dnn_attn_weights(cudnnHandle_t handle, const void* parent, std::uint32_t kind,
                                     std::size_t weight_bytes, const void* weights,
                                     const void* destination, void** output) noexcept {
    return dnn_handle(handle, false, false, [&](State& shared, Exclusion& exclusion, Entry& entry) {
        const auto base = reinterpret_cast<std::uintptr_t>(weights);
        if (kind > 7 || weight_bytes > UINTPTR_MAX - base ||
            (output && !writable_host_range(output, sizeof(*output))))
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
        const std::array handles{parent, destination};
        AttnWeightEntries descriptors;
        for (;;) {
            if (!entry.is_alive || entry.is_uncertain || !context.is_alive || context.is_closing ||
                context.is_uncertain || context.epoch != epoch || !metadata->is_alive ||
                metadata->is_closing || metadata->is_uncertain)
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
                    i ? ExecutionDnnObjectKind::kTensor : ExecutionDnnObjectKind::kAttn);
                if (!descriptor || descriptor->context != metadata ||
                    !valid_dnn_metadata(*descriptor))
                    return CUDNN_STATUS_NOT_INITIALIZED;
                if (descriptor->active_calls >= kMaxHandles - 1)
                    return dnn_profile::host_allocation_status();
                is_waiting |= descriptor->is_retiring || (i && descriptor->active_calls);
            }
            if (!is_waiting) break;
            // Never hold a partial descriptor pin or an exclusive destination
            // flag while waiting for a previous user of that destination.
            if (!yield_frontend(exclusion)) return CUDNN_STATUS_NOT_INITIALIZED;
        }
        std::optional<FlagPin> mutation;
        if (descriptors[1]) mutation.emplace(descriptors[1]->is_retiring, exclusion);
        CounterPin parent_pin(descriptors[0]->active_calls, exclusion);
        std::optional<CounterPin> destination_pin;
        if (descriptors[1]) destination_pin.emplace(descriptors[1]->active_calls, exclusion);
        const ExecutionDnnAttnWeights value{entry.remote,
                                            metadata->connection->session(),
                                            descriptors[0]->remote,
                                            kind,
                                            weights ? 1U : 0U,
                                            weight_bytes,
                                            descriptors[1] ? descriptors[1]->remote : 0,
                                            output ? 1U : 0U};
        ExecutionPayloadWriter payload;
        if (!write_execution_dnn_attn_weights(payload, value))
            return dnn_profile::host_allocation_status();
        const auto reply =
            shared.call(exclusion, context, ExecutionOpcode::kDnnAttnWeights, std::move(payload));
        void* mapped{};
        const auto status =
            attn_weights_status(reply, entry, *metadata, descriptors, value, base, mapped);
        if (status != CUDNN_STATUS_SUCCESS) return status;
        // BAD_PARAM here is a local publication policy result, not original
        // status. Proven successful server metadata need not be quarantined.
        if (output && (!writable_host_range(output, sizeof(*output)) || !publish(output, mapped)))
            return CUDNN_STATUS_BAD_PARAM;
        return CUDNN_STATUS_SUCCESS;
    });
}
}  // namespace

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetMultiHeadAttnWeights(
    cudnnHandle_t handle, const cudnnAttnDescriptor_t descriptor,
    cudnnMultiHeadAttnWeightKind_t kind, std::size_t weight_bytes, const void* weights,
    cudnnTensorDescriptor_t destination, void** output) {
    return query_dnn_attn_weights(handle, descriptor, dnn_enum_bits(kind), weight_bytes, weights,
                                  destination, output);
}
