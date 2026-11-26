// Included after convolution_frontend.inc; no caller-native SDK work is issued.
namespace {
using DnnPoolingHandle = cudnnPoolingStruct*;

struct DnnTensorForwardArguments {
    ExecutionDnnTensorForwardKind kind{};
    const void* descriptor{};
    std::array<cudnnTensorDescriptor_t, 2> tensors{};
    std::array<const void*, 2> buffers{};
    const void* alpha{};
    const void* beta{};
    std::uint32_t algorithm{}, mode{};
};

cudnnStatus_t submit_dnn_tensor_forward(cudnnHandle_t handle,
                                        const DnnTensorForwardArguments& arguments) noexcept {
    return dnn_handle(handle, false, true, [&](State& shared, Exclusion& exclusion, Entry& entry) {
        const std::array<const void*, 3> handles{arguments.tensors[0], arguments.tensors[1],
                                                 arguments.descriptor};
        const std::array kinds{ExecutionDnnObjectKind::kTensor, ExecutionDnnObjectKind::kTensor,
                               arguments.kind == ExecutionDnnTensorForwardKind::kLrnForward
                                   ? ExecutionDnnObjectKind::kLrn
                                   : ExecutionDnnObjectKind::kPooling};
        const std::size_t count =
            arguments.kind == ExecutionDnnTensorForwardKind::kPoolingForward ||
                    arguments.kind == ExecutionDnnTensorForwardKind::kLrnForward
                ? 3
                : 2;
        return with_dnn_descriptors(
            shared, exclusion, std::span{handles}.first(count), std::span{kinds}.first(count),
            [&](Entry& metadata, const auto& descriptors) {
                if (entry.context->submissions == UINT64_MAX)
                    return dnn_profile::host_allocation_status();
                // All descriptors are immutable before the type query releases
                // exclusion. In-place softmax deliberately keeps both leases.
                ExecutionDnnTensor tensor;
                const auto queried = query_dnn_tensor(shared, exclusion, *descriptors[0],
                                                      CUDNN_DIM_MAX, false, tensor);
                if (queried != CUDNN_STATUS_SUCCESS) return queried;
                ExecutionDnnTensorForward forward;
                forward.kind = arguments.kind;
                forward.handle = entry.remote;
                forward.metadata = metadata.connection->session();
                forward.descriptor = count == 3 ? descriptors[2]->remote : 0;
                forward.tensors = {descriptors[0]->remote, descriptors[1]->remote};
                forward.algorithm = arguments.algorithm;
                forward.mode = arguments.mode;
                forward.scalar_bytes =
                    tensor.type == CUDNN_DATA_DOUBLE ? sizeof(double) : sizeof(float);
                if (!self_copy(forward.alpha.data(), arguments.alpha, forward.scalar_bytes,
                               false) ||
                    !self_copy(forward.beta.data(), arguments.beta, forward.scalar_bytes, false))
                    return CUDNN_STATUS_BAD_PARAM;
                std::array<std::optional<CounterPin>, 2> buffer_pins, module_pins;
                for (std::size_t i = 0; i != forward.buffers.size(); ++i) {
                    const auto address = reinterpret_cast<CUdeviceptr>(arguments.buffers[i]);
                    auto* allocation = shared.range(address, 1);
                    if (!allocation || allocation->context != entry.context ||
                        shared.range_is_retiring(*allocation))
                        return CUDNN_STATUS_BAD_PARAM;
                    if (allocation->active_calls > kMaxHandles - forward.buffers.size())
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
                        if ((*parent)->active_calls > kMaxHandles - forward.buffers.size())
                            return dnn_profile::host_allocation_status();
                        module_pins[i].emplace((*parent)->active_calls, exclusion);
                    }
                    forward.buffers[i] = {allocation->remote, address - allocation->address};
                }
                ExecutionPayloadWriter input;
                if (!write_execution_dnn_tensor_forward(input, forward))
                    return dnn_profile::host_allocation_status();
                ++entry.context->submissions;
                return dnn_status(shared.call(exclusion, *entry.context,
                                              ExecutionOpcode::kDnnTensorForward, std::move(input)),
                                  *entry.context, &entry.is_uncertain);
            });
    });
}
}  // namespace

extern "C" cudnnStatus_t CUDNNWINAPI cudnnCreatePoolingDescriptor(DnnPoolingHandle* output) {
    return create_dnn_metadata(output, ExecutionDnnObjectKind::kPooling);
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnDestroyPoolingDescriptor(DnnPoolingHandle handle) {
    return destroy_dnn_metadata(handle, ExecutionDnnObjectKind::kPooling);
}
// The legacy pooling enum is deprecated in SDK 9.10.2 but remains its exact ABI.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
// NOLINTBEGIN(bugprone-easily-swappable-parameters): exact public SDK ABIs.
extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetPoolingNdDescriptor(
    DnnPoolingHandle handle, cudnnPoolingMode_t mode, cudnnNanPropagation_t nan, int dimensions,
    const int* window, const int* padding, const int* strides) {
    return dnn_metadata(
        handle, ExecutionDnnObjectKind::kPooling, true,
        [&](State& shared, Exclusion& exclusion, DnnEntry& entry) {
            if (dimensions < 0 || dimensions > CUDNN_DIM_MAX) return CUDNN_STATUS_BAD_PARAM;
            ExecutionDnnPooling value;
            value.mode = dnn_enum_bits(mode);
            value.nan = dnn_enum_bits(nan);
            value.dimensions = dimensions;
            const auto bytes = static_cast<std::size_t>(dimensions) * sizeof(int);
            if (!self_copy(value.window.data(), window, bytes, false) ||
                !self_copy(value.padding.data(), padding, bytes, false) ||
                !self_copy(value.strides.data(), strides, bytes, false))
                return CUDNN_STATUS_BAD_PARAM;
            ExecutionPayloadWriter input;
            if (!input.write_u64(entry.remote) || !write_execution_dnn_pooling(input, value))
                return dnn_profile::host_allocation_status();
            return dnn_status(shared.call(exclusion, *entry.context,
                                          ExecutionOpcode::kSetDnnPooling, std::move(input)),
                              *entry.context, &entry.is_uncertain);
        });
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetPoolingNdDescriptor(
    DnnPoolingHandle handle, int requested, cudnnPoolingMode_t* mode, cudnnNanPropagation_t* nan,
    int* dimensions, int* window, int* padding, int* strides) {
    return dnn_metadata(
        handle, ExecutionDnnObjectKind::kPooling, false,
        [&](State& shared, Exclusion& exclusion, DnnEntry& entry) {
            const auto valid_outputs = [&] {
                return writable_host_range(mode, sizeof(*mode)) &&
                       writable_host_range(nan, sizeof(*nan)) &&
                       writable_host_range(dimensions, sizeof(*dimensions));
            };
            if (!valid_outputs()) return CUDNN_STATUS_BAD_PARAM;
            ExecutionPayloadWriter input;
            // A capacity above eight is a native error, not a smaller query.
            if (!input.write_u64(entry.remote) ||
                !input.write_u32(std::bit_cast<std::uint32_t>(requested)))
                return dnn_profile::host_allocation_status();
            const auto reply = shared.call(exclusion, *entry.context,
                                           ExecutionOpcode::kGetDnnPooling, std::move(input));
            const auto result = dnn_status(reply, *entry.context, &entry.is_uncertain);
            if (result != CUDNN_STATUS_SUCCESS) return result;
            ExecutionPayloadReader reader(reply->data);
            ExecutionDnnPooling value;
            if (!read_execution_dnn_pooling(reader, value) || !reader.empty() ||
                value.dimensions <= 0 || value.dimensions > CUDNN_DIM_MAX - 2 || value.mode > 3 ||
                value.nan > 1 || requested < 0 || requested > CUDNN_DIM_MAX)
                return dnn_invalid(*entry.context, &entry.is_uncertain);
            const auto bytes =
                static_cast<std::size_t>(std::min(requested, value.dimensions)) * sizeof(int);
            // Validate the entire bounded publication after transport, before
            // changing any caller field. Native errors never publish data.
            if (!valid_outputs() || !writable_host_range(window, bytes) ||
                !writable_host_range(padding, bytes) || !writable_host_range(strides, bytes))
                return CUDNN_STATUS_BAD_PARAM;
            return publish(mode, static_cast<cudnnPoolingMode_t>(value.mode)) &&
                           publish(nan, static_cast<cudnnNanPropagation_t>(value.nan)) &&
                           publish(dimensions, value.dimensions) &&
                           self_copy(window, value.window.data(), bytes, true) &&
                           self_copy(padding, value.padding.data(), bytes, true) &&
                           self_copy(strides, value.strides.data(), bytes, true)
                       ? CUDNN_STATUS_SUCCESS
                       : CUDNN_STATUS_BAD_PARAM;
        });
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetPoolingNdForwardOutputDim(
    DnnPoolingHandle pooling, cudnnTensorDescriptor_t tensor, int requested, int* output) {
    return dnn_boundary([&](State& shared, Exclusion& exclusion) {
        const bool is_bounded = requested >= 0 && requested <= CUDNN_DIM_MAX;
        const auto bytes = is_bounded ? static_cast<std::size_t>(requested) * sizeof(int) : 0;
        if (is_bounded && !writable_host_range(output, bytes)) return CUDNN_STATUS_BAD_PARAM;
        const std::array<const void*, 2> handles{pooling, tensor};
        constexpr std::array kinds{ExecutionDnnObjectKind::kPooling,
                                   ExecutionDnnObjectKind::kTensor};
        return with_dnn_descriptors(
            shared, exclusion, handles, kinds, [&](Entry& metadata, const auto& descriptors) {
                ExecutionPayloadWriter input;
                if (!input.write_u64(descriptors[0]->remote) ||
                    !input.write_u64(descriptors[1]->remote) ||
                    !input.write_u32(std::bit_cast<std::uint32_t>(requested)))
                    return dnn_profile::host_allocation_status();
                const auto reply = shared.call(
                    exclusion, metadata, ExecutionOpcode::kDnnPoolingOutput, std::move(input));
                const auto result = dnn_status(reply, metadata);
                if (result != CUDNN_STATUS_SUCCESS) return result;
                ExecutionPayloadReader reader(reply->data);
                ExecutionDnnTensor value;
                if (!read_execution_dnn_tensor(reader, value) || !reader.empty() || !is_bounded ||
                    value.dimensions != requested)
                    return dnn_invalid(metadata);
                if (!writable_host_range(output, bytes)) return CUDNN_STATUS_BAD_PARAM;
                return self_copy(output, value.sizes.data(), bytes, true) ? CUDNN_STATUS_SUCCESS
                                                                          : CUDNN_STATUS_BAD_PARAM;
            });
    });
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnPoolingForward(cudnnHandle_t handle,
                                                         DnnPoolingHandle pooling,
                                                         const void* alpha,
                                                         cudnnTensorDescriptor_t input,
                                                         const void* x, const void* beta,
                                                         cudnnTensorDescriptor_t output, void* y) {
    return submit_dnn_tensor_forward(handle, {ExecutionDnnTensorForwardKind::kPoolingForward,
                                              pooling,
                                              {input, output},
                                              {x, y},
                                              alpha,
                                              beta,
                                              0,
                                              0});
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnSoftmaxForward(cudnnHandle_t handle,
                                                         cudnnSoftmaxAlgorithm_t algorithm,
                                                         cudnnSoftmaxMode_t mode, const void* alpha,
                                                         cudnnTensorDescriptor_t input,
                                                         const void* x, const void* beta,
                                                         cudnnTensorDescriptor_t output, void* y) {
    return submit_dnn_tensor_forward(handle, {ExecutionDnnTensorForwardKind::kSoftmaxForward,
                                              nullptr,
                                              {input, output},
                                              {x, y},
                                              alpha,
                                              beta,
                                              dnn_enum_bits(algorithm),
                                              dnn_enum_bits(mode)});
}
// These five pinned-reference handlers are explicit stubs. Prevent local SDK
// use of remote tokens without claiming implementation of their semantics.
extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetPooling2dDescriptor(DnnPoolingHandle,
                                                                 cudnnPoolingMode_t,
                                                                 cudnnNanPropagation_t, int, int,
                                                                 int, int, int, int) {
    return dnn_unsupported();
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetPooling2dDescriptor(DnnPoolingHandle,
                                                                 cudnnPoolingMode_t*,
                                                                 cudnnNanPropagation_t*, int*, int*,
                                                                 int*, int*, int*, int*) {
    return dnn_unsupported();
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetPooling2dForwardOutputDim(DnnPoolingHandle,
                                                                       cudnnTensorDescriptor_t,
                                                                       int*, int*, int*, int*) {
    return dnn_unsupported();
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnPoolingBackward(cudnnHandle_t, DnnPoolingHandle,
                                                          const void*, cudnnTensorDescriptor_t,
                                                          const void*, cudnnTensorDescriptor_t,
                                                          const void*, cudnnTensorDescriptor_t,
                                                          const void*, const void*,
                                                          cudnnTensorDescriptor_t, void*) {
    return dnn_unsupported();
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnSoftmaxBackward(cudnnHandle_t, cudnnSoftmaxAlgorithm_t,
                                                          cudnnSoftmaxMode_t, const void*,
                                                          cudnnTensorDescriptor_t, const void*,
                                                          cudnnTensorDescriptor_t, const void*,
                                                          const void*, cudnnTensorDescriptor_t,
                                                          void*) {
    return dnn_unsupported();
}
// NOLINTEND(bugprone-easily-swappable-parameters)
#pragma GCC diagnostic pop
