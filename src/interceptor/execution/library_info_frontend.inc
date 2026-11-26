// Library observations use DNM1 without creating a CUDA context or SDK handle.
namespace {
cudnnStatus_t query_dnn_library_info(State& shared, Exclusion& exclusion,
                                     const ExecutionDnnLibraryInfo& query, std::uint64_t& value) {
    const auto opened = open_dnn_metadata(shared, exclusion);
    if (opened != CUDA_SUCCESS) return dnn_error(opened);
    auto& context = *shared.dnn_metadata_context;
    if (context.active_calls >= kMaxHandles - 1) return dnn_profile::host_allocation_status();
    CounterPin origin(context.active_calls, exclusion);
    ExecutionPayloadWriter payload;
    if (!write_execution_dnn_library_info(payload, query))
        return dnn_profile::host_allocation_status();
    const auto reply =
        shared.call(exclusion, context, ExecutionOpcode::kDnnLibraryInfo, std::move(payload));
    if (!reply || reply->id || !reply->data.empty() || reply->parameter_bytes > 1)
        return dnn_invalid(context);
    if (reply->disposition == ExecutionWireDisposition::kRejected) {
        // A clean policy rejection is not corrupt metadata. Unlike native
        // Property errors, these replies never invent an original SDK status.
        if (reply->original_status || reply->address) return dnn_invalid(context);
        return dnn_status(reply, context);
    }
    if (reply->disposition != ExecutionWireDisposition::kComplete) return dnn_invalid(context);
    if (query.kind != ExecutionDnnLibraryInfoKind::kProperty) {
        // These three native ABIs return size_t directly. Passing this success
        // through dnn_status would incorrectly require an original status.
        if (reply->status || reply->original_status || reply->parameter_bytes ||
            reply->address > SIZE_MAX || context.is_uncertain)
            return dnn_invalid(context);
        value = reply->address;
        return CUDNN_STATUS_SUCCESS;
    }
    const auto result = dnn_status(reply, context);
    if (result != CUDNN_STATUS_SUCCESS) return reply->address ? dnn_invalid(context) : result;
    if (context.is_uncertain || reply->parameter_bytes || reply->address > UINT32_MAX ||
        (!query.has_output && reply->address))
        return dnn_invalid(context);
    value = reply->address;
    return CUDNN_STATUS_SUCCESS;
}
std::size_t dnn_library_value(ExecutionDnnLibraryInfoKind kind) noexcept {
    std::uint64_t value{};
    const auto result = dnn_boundary([&](State& shared, Exclusion& exclusion) {
        return query_dnn_library_info(shared, exclusion, {kind, 0, 0}, value);
    });
    // The public size_t ABI has no error channel. Zero is the explicit Glimmer
    // failure convention, not a fabricated native error or caller-SDK fallback.
    return result == CUDNN_STATUS_SUCCESS ? static_cast<std::size_t>(value) : 0;
}
}  // namespace
extern "C" std::size_t CUDNNWINAPI cudnnGetVersion() {
    return dnn_library_value(ExecutionDnnLibraryInfoKind::kVersion);
}
extern "C" std::size_t CUDNNWINAPI cudnnGetCudartVersion() {
    return dnn_library_value(ExecutionDnnLibraryInfoKind::kCudartVersion);
}
extern "C" std::size_t CUDNNWINAPI cudnnGetMaxDeviceVersion() {
    return dnn_library_value(ExecutionDnnLibraryInfoKind::kMaxDeviceVersion);
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetProperty(libraryPropertyType property, int* output) {
    return dnn_boundary([&](State& shared, Exclusion& exclusion) {
        if (output && !writable_host_range(output, sizeof(*output))) return CUDNN_STATUS_BAD_PARAM;
        const ExecutionDnnLibraryInfo query{ExecutionDnnLibraryInfoKind::kProperty,
                                            dnn_enum_bits(property), output ? 1U : 0U};
        std::uint64_t value{};
        const auto result = query_dnn_library_info(shared, exclusion, query, value);
        if (result != CUDNN_STATUS_SUCCESS || !output) return result;
        if (!writable_host_range(output, sizeof(*output))) return CUDNN_STATUS_BAD_PARAM;
        const auto signed_value = std::bit_cast<std::int32_t>(static_cast<std::uint32_t>(value));
        return publish(output, static_cast<int>(signed_value)) ? CUDNN_STATUS_SUCCESS
                                                               : CUDNN_STATUS_BAD_PARAM;
    });
}
