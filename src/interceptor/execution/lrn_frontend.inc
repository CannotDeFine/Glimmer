// Included after pooling_frontend.inc; tensor-forward retains typed metadata,
// handle and allocation ownership without running a caller-native SDK.
extern "C" cudnnStatus_t CUDNNWINAPI cudnnCreateLRNDescriptor(cudnnLRNDescriptor_t* output) {
    // Native creation establishes its documented defaults. Do not add a setter.
    return create_dnn_metadata(output, ExecutionDnnObjectKind::kLrn);
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnDestroyLRNDescriptor(cudnnLRNDescriptor_t handle) {
    return destroy_dnn_metadata(handle, ExecutionDnnObjectKind::kLrn);
}
// NOLINTBEGIN(bugprone-easily-swappable-parameters): exact public SDK ABIs.
extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetLRNDescriptor(cudnnLRNDescriptor_t handle, unsigned n,
                                                           double alpha, double beta, double k) {
    return dnn_metadata(
        handle, ExecutionDnnObjectKind::kLrn, true,
        [&](State& shared, Exclusion& exclusion, DnnEntry& entry) {
            const ExecutionDnnLrn value{n, std::bit_cast<std::uint64_t>(alpha),
                                        std::bit_cast<std::uint64_t>(beta),
                                        std::bit_cast<std::uint64_t>(k)};
            ExecutionPayloadWriter input;
            if (!input.write_u64(entry.remote) || !write_execution_dnn_lrn(input, value))
                return dnn_profile::host_allocation_status();
            return dnn_status(shared.call(exclusion, *entry.context, ExecutionOpcode::kSetDnnLrn,
                                          std::move(input)),
                              *entry.context, &entry.is_uncertain);
        });
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnLRNCrossChannelForward(
    cudnnHandle_t handle, cudnnLRNDescriptor_t descriptor, cudnnLRNMode_t mode, const void* alpha,
    cudnnTensorDescriptor_t input, const void* x, const void* beta, cudnnTensorDescriptor_t output,
    void* y) {
    return submit_dnn_tensor_forward(handle, {ExecutionDnnTensorForwardKind::kLrnForward,
                                              descriptor,
                                              {input, output},
                                              {x, y},
                                              alpha,
                                              beta,
                                              0,
                                              dnn_enum_bits(mode)});
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnTransformTensor(cudnnHandle_t handle, const void* alpha,
                                                          cudnnTensorDescriptor_t input,
                                                          const void* x, const void* beta,
                                                          cudnnTensorDescriptor_t output, void* y) {
    return submit_dnn_tensor_forward(handle, {ExecutionDnnTensorForwardKind::kTransformTensor,
                                              nullptr,
                                              {input, output},
                                              {x, y},
                                              alpha,
                                              beta,
                                              0,
                                              0});
}
// The fixed reference has no implementations for these consumers. Typed guards
// prevent a caller-native SDK from dereferencing an owned remote descriptor.
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetLRNDescriptor(cudnnLRNDescriptor_t, unsigned*, double*,
                                                           double*, double*) {
    return dnn_unsupported();
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnLRNCrossChannelBackward(
    cudnnHandle_t, cudnnLRNDescriptor_t, cudnnLRNMode_t, const void*, cudnnTensorDescriptor_t,
    const void*, cudnnTensorDescriptor_t, const void*, cudnnTensorDescriptor_t, const void*,
    const void*, cudnnTensorDescriptor_t, void*) {
    return dnn_unsupported();
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnDivisiveNormalizationForward(
    cudnnHandle_t, cudnnLRNDescriptor_t, cudnnDivNormMode_t, const void*, cudnnTensorDescriptor_t,
    const void*, const void*, void*, void*, const void*, cudnnTensorDescriptor_t, void*) {
    return dnn_unsupported();
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnDivisiveNormalizationBackward(
    cudnnHandle_t, cudnnLRNDescriptor_t, cudnnDivNormMode_t, const void*, cudnnTensorDescriptor_t,
    const void*, const void*, const void*, void*, void*, const void*, cudnnTensorDescriptor_t,
    void*, void*) {
    return dnn_unsupported();
}
// NOLINTEND(bugprone-easily-swappable-parameters)
