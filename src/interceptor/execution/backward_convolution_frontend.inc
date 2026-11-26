// Scalar observations on an owned CUDA context; no submission or queue drain.
namespace {
constexpr std::array kBackwardQueryDescriptorKinds{
    ExecutionDnnObjectKind::kFilter, ExecutionDnnObjectKind::kTensor,
    ExecutionDnnObjectKind::kConvolution, ExecutionDnnObjectKind::kTensor};

cudnnStatus_t backward_convolution_query_status(const std::optional<ExecutionWireReply>& reply,
                                                Entry& entry,
                                                const ExecutionDnnBackwardConvolutionQuery& query,
                                                std::uint64_t& value) {
    auto& context = *entry.context;
    if (!reply || reply->id || !reply->data.empty() || reply->parameter_bytes > 1)
        return dnn_invalid(context, &entry.is_uncertain);
    if (reply->disposition == ExecutionWireDisposition::kRejected) {
        // Issued queries can lose context after receiving an original status.
        // That provenance is valid only on a quarantined, nonpublishing reply.
        if (reply->address || (reply->original_status &&
                               (!reply->parameter_bytes ||
                                !dnn_profile::status_is_representable(*reply->original_status))))
            return dnn_invalid(context, &entry.is_uncertain);
        return dnn_status(reply, context, &entry.is_uncertain);
    }
    if (reply->disposition != ExecutionWireDisposition::kComplete)
        return dnn_invalid(context, &entry.is_uncertain);
    const auto status = dnn_status(reply, context, &entry.is_uncertain);
    if (status != CUDNN_STATUS_SUCCESS)
        return reply->address ? dnn_invalid(context, &entry.is_uncertain) : status;
    const auto maximum = query.kind == ExecutionDnnBackwardConvolutionQueryKind::kDataWorkspace
                             ? static_cast<std::uint64_t>(SIZE_MAX)
                             : static_cast<std::uint64_t>(INT32_MAX);
    if (reply->parameter_bytes || context.is_uncertain || entry.is_uncertain ||
        reply->address > maximum || (!query.has_output && reply->address))
        return dnn_invalid(context, &entry.is_uncertain);
    value = reply->address;
    return CUDNN_STATUS_SUCCESS;
}

template <typename Output>
cudnnStatus_t query_dnn_backward_convolution(cudnnHandle_t handle,
                                             const std::array<const void*, 4>& descriptors,
                                             ExecutionDnnBackwardConvolutionQueryKind kind,
                                             std::uint32_t algorithm, Output* output) noexcept {
    return dnn_handle(handle, false, false, [&](State& shared, Exclusion& exclusion, Entry& entry) {
        const bool is_workspace = kind == ExecutionDnnBackwardConvolutionQueryKind::kDataWorkspace;
        // The configured native MaxCount functions do not check null before a
        // store. Reject locally, not by substituting a successful temporary.
        if ((!is_workspace && !output) || (output && !writable_host_range(output, sizeof(*output))))
            return CUDNN_STATUS_BAD_PARAM;
        ExecutionDnnBackwardConvolutionQuery query;
        query.kind = kind;
        query.handle = entry.remote;
        query.algorithm = algorithm;
        query.has_output = output ? 1U : 0U;
        const auto execute = [&] {
            ExecutionPayloadWriter input;
            if (!write_execution_dnn_backward_convolution_query(input, query))
                return dnn_profile::host_allocation_status();
            const auto reply =
                shared.call(exclusion, *entry.context,
                            ExecutionOpcode::kDnnBackwardConvolutionQuery, std::move(input));
            std::uint64_t value{};
            const auto status = backward_convolution_query_status(reply, entry, query, value);
            if (status != CUDNN_STATUS_SUCCESS || !output) return status;
            if (!writable_host_range(output, sizeof(*output))) return CUDNN_STATUS_BAD_PARAM;
            return publish(output, static_cast<Output>(value)) ? CUDNN_STATUS_SUCCESS
                                                               : CUDNN_STATUS_BAD_PARAM;
        };
        if (!is_workspace) return execute();
        return with_dnn_descriptors(shared, exclusion, descriptors, kBackwardQueryDescriptorKinds,
                                    [&](Entry& metadata, const auto& objects) {
                                        query.metadata = metadata.connection->session();
                                        for (std::size_t i = 0; i != objects.size(); ++i)
                                            query.descriptors[i] = objects[i]->remote;
                                        return execute();
                                    });
    });
}
}  // namespace

extern "C" cudnnStatus_t CUDNNWINAPI
cudnnGetConvolutionBackwardDataAlgorithmMaxCount(cudnnHandle_t handle, int* count) {
    return query_dnn_backward_convolution(
        handle, {}, ExecutionDnnBackwardConvolutionQueryKind::kDataMaxCount, 0, count);
}
extern "C" cudnnStatus_t CUDNNWINAPI
cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(cudnnHandle_t handle, int* count) {
    return query_dnn_backward_convolution(
        handle, {}, ExecutionDnnBackwardConvolutionQueryKind::kFilterMaxCount, 0, count);
}
// NOLINTBEGIN(bugprone-easily-swappable-parameters): exact installed public SDK ABI.
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle_t handle, const cudnnFilterDescriptor_t filter, const cudnnTensorDescriptor_t dy,
    const cudnnConvolutionDescriptor_t convolution, const cudnnTensorDescriptor_t dx,
    cudnnConvolutionBwdDataAlgo_t algorithm, std::size_t* bytes) {
    return query_dnn_backward_convolution(handle, {filter, dy, convolution, dx},
                                          ExecutionDnnBackwardConvolutionQueryKind::kDataWorkspace,
                                          dnn_enum_bits(algorithm), bytes);
}
// NOLINTEND(bugprone-easily-swappable-parameters)
