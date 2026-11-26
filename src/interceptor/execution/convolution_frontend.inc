// Included after dnn_frontend.inc: all SDK effects stay on the service owner.
namespace {
using DnnFilterHandle = cudnnFilterStruct*;
using DnnConvolutionHandle = cudnnConvolutionStruct*;

template <typename Function>
cudnnStatus_t with_dnn_descriptors(State& shared, Exclusion& exclusion,
                                   std::span<const void* const> handles,
                                   std::span<const ExecutionDnnObjectKind> kinds,
                                   Function&& function) {
    auto* metadata = shared.dnn_metadata_context;
    if (!metadata || !metadata->connection || !metadata->is_alive || metadata->is_uncertain ||
        metadata->is_closing || handles.size() != kinds.size() || handles.size() > 4)
        return CUDNN_STATUS_NOT_INITIALIZED;
    std::array<std::shared_ptr<DnnEntry>, 4> descriptors;
    for (;;) {
        bool waiting{};
        for (std::size_t index = 0; index < handles.size(); ++index) {
            auto& descriptor = descriptors[index];
            descriptor = find_dnn_metadata(shared, handles[index], kinds[index]);
            if (!descriptor || descriptor->context != metadata || !valid_dnn_metadata(*descriptor))
                return CUDNN_STATUS_NOT_INITIALIZED;
            if (descriptor->active_calls > kMaxHandles - descriptors.size())
                return dnn_profile::host_allocation_status();
            waiting |= descriptor->is_retiring;
        }
        if (!waiting) break;
        if (!yield_frontend(exclusion) || metadata->is_uncertain || metadata->is_closing)
            return CUDNN_STATUS_NOT_INITIALIZED;
    }
    if (metadata->active_calls >= kMaxHandles - 1) return dnn_profile::host_allocation_status();
    CounterPin owner(metadata->active_calls, exclusion);
    std::array<std::optional<CounterPin>, 4> pins;
    for (std::size_t index = 0; index < handles.size(); ++index)
        pins[index].emplace(descriptors[index]->active_calls, exclusion);
    return function(*metadata, descriptors);
}

cudnnStatus_t set_dnn_filter(DnnFilterHandle handle, const ExecutionDnnFilter& value,
                             bool is_four) noexcept {
    return dnn_metadata(
        handle, ExecutionDnnObjectKind::kFilter, true,
        [&](State& shared, Exclusion& exclusion, DnnEntry& entry) {
            ExecutionPayloadWriter input;
            if (!input.write_u64(entry.remote) || !input.write_u32(is_four ? 1 : 0) ||
                !write_execution_dnn_filter(input, value))
                return dnn_profile::host_allocation_status();
            return dnn_status(shared.call(exclusion, *entry.context, ExecutionOpcode::kSetDnnFilter,
                                          std::move(input)),
                              *entry.context, &entry.is_uncertain);
        });
}
cudnnStatus_t query_dnn_filter(State& shared, Exclusion& exclusion, DnnEntry& entry, int requested,
                               bool is_four, ExecutionDnnFilter& output) {
    ExecutionPayloadWriter input;
    if (!input.write_u64(entry.remote) ||
        !input.write_u32(std::bit_cast<std::uint32_t>(requested)) ||
        !input.write_u32(is_four ? 1 : 0))
        return dnn_profile::host_allocation_status();
    const auto reply =
        shared.call(exclusion, *entry.context, ExecutionOpcode::kGetDnnFilter, std::move(input));
    const auto result = dnn_status(reply, *entry.context, &entry.is_uncertain);
    if (result != CUDNN_STATUS_SUCCESS) return result;
    ExecutionPayloadReader reader(reply->data);
    ExecutionDnnFilter value;
    if (!read_execution_dnn_filter(reader, value) || !reader.empty() ||
        value.type > dnn_profile::data_type_enum_max() || value.format > 3 ||
        (is_four && value.dimensions != 4))
        return dnn_invalid(*entry.context, &entry.is_uncertain);
    output = value;
    return CUDNN_STATUS_SUCCESS;
}
cudnnStatus_t set_dnn_convolution(DnnConvolutionHandle handle,
                                  const ExecutionDnnConvolutionState& value, bool is_two) noexcept {
    return dnn_metadata(
        handle, ExecutionDnnObjectKind::kConvolution, true,
        [&](State& shared, Exclusion& exclusion, DnnEntry& entry) {
            ExecutionPayloadWriter input;
            if (!input.write_u64(entry.remote) || !input.write_u32(is_two ? 1 : 0) ||
                !write_execution_dnn_convolution(input, value))
                return dnn_profile::host_allocation_status();
            return dnn_status(shared.call(exclusion, *entry.context,
                                          ExecutionOpcode::kSetDnnConvolution, std::move(input)),
                              *entry.context, &entry.is_uncertain);
        });
}
cudnnStatus_t query_dnn_convolution(State& shared, Exclusion& exclusion, DnnEntry& entry,
                                    int requested, bool is_two,
                                    ExecutionDnnConvolutionState& output) {
    ExecutionPayloadWriter input;
    // Unlike tensor/filter capacity, the convolution request is itself subject
    // to a native upper bound. Preserve its signed bits; never clamp to six.
    if (!input.write_u64(entry.remote) ||
        !input.write_u32(std::bit_cast<std::uint32_t>(requested)) ||
        !input.write_u32(is_two ? 1 : 0))
        return dnn_profile::host_allocation_status();
    const auto reply = shared.call(exclusion, *entry.context, ExecutionOpcode::kGetDnnConvolution,
                                   std::move(input));
    const auto result = dnn_status(reply, *entry.context, &entry.is_uncertain);
    if (result != CUDNN_STATUS_SUCCESS) return result;
    if (reply->parameter_bytes || !valid_dnn_metadata(entry) || reply->id || reply->address)
        return dnn_invalid(*entry.context, &entry.is_uncertain);
    ExecutionPayloadReader reader(reply->data);
    ExecutionDnnConvolutionState value;
    if (!read_execution_dnn_convolution(reader, value) || !reader.empty() || value.mode > 1 ||
        value.type > dnn_profile::data_type_enum_max() || value.dimensions > CUDNN_DIM_MAX - 2 ||
        requested < 0 || requested > CUDNN_DIM_MAX - 2 || (is_two && value.dimensions != 2))
        return dnn_invalid(*entry.context, &entry.is_uncertain);
    output = value;
    return CUDNN_STATUS_SUCCESS;
}
cudnnStatus_t set_dnn_convolution_attribute(DnnConvolutionHandle handle,
                                            ExecutionDnnConvolutionAttribute attribute,
                                            std::uint32_t value) noexcept {
    return dnn_metadata(
        handle, ExecutionDnnObjectKind::kConvolution, true,
        [&](State& shared, Exclusion& exclusion, DnnEntry& entry) {
            ExecutionPayloadWriter input;
            if (!input.write_u64(entry.remote) ||
                !input.write_u32(static_cast<std::uint32_t>(attribute)) || !input.write_u32(value))
                return dnn_profile::host_allocation_status();
            return dnn_status(
                shared.call(exclusion, *entry.context, ExecutionOpcode::kSetDnnConvolutionAttribute,
                            std::move(input)),
                *entry.context, &entry.is_uncertain);
        });
}
template <typename Value>
cudnnStatus_t get_dnn_convolution_attribute(DnnConvolutionHandle handle,
                                            ExecutionDnnConvolutionAttribute attribute,
                                            Value* output, std::uint32_t maximum) noexcept {
    return dnn_metadata(
        handle, ExecutionDnnObjectKind::kConvolution, false,
        [&](State& shared, Exclusion& exclusion, DnnEntry& entry) {
            if (!writable_host_range(output, sizeof(*output))) return CUDNN_STATUS_BAD_PARAM;
            ExecutionPayloadWriter input;
            if (!input.write_u64(entry.remote) ||
                !input.write_u32(static_cast<std::uint32_t>(attribute)))
                return dnn_profile::host_allocation_status();
            const auto reply =
                shared.call(exclusion, *entry.context, ExecutionOpcode::kGetDnnConvolutionAttribute,
                            std::move(input));
            const auto result = dnn_status(reply, *entry.context, &entry.is_uncertain);
            if (result != CUDNN_STATUS_SUCCESS) return result;
            if (reply->address > maximum || !reply->data.empty())
                return dnn_invalid(*entry.context, &entry.is_uncertain);
            // Group count is a signed SDK int, not a nonnegative enum.
            // Enum callers retain their separate representation bounds.
            const auto value = std::bit_cast<Value>(static_cast<std::uint32_t>(reply->address));
            return publish(output, value) ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_BAD_PARAM;
        });
}
}  // namespace

extern "C" cudnnStatus_t CUDNNWINAPI cudnnCreateFilterDescriptor(DnnFilterHandle* output) {
    return create_dnn_metadata(output, ExecutionDnnObjectKind::kFilter);
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnDestroyFilterDescriptor(DnnFilterHandle handle) {
    return destroy_dnn_metadata(handle, ExecutionDnnObjectKind::kFilter);
}
extern "C" cudnnStatus_t CUDNNWINAPI
cudnnCreateConvolutionDescriptor(DnnConvolutionHandle* output) {
    return create_dnn_metadata(output, ExecutionDnnObjectKind::kConvolution);
}
extern "C" cudnnStatus_t CUDNNWINAPI
cudnnDestroyConvolutionDescriptor(DnnConvolutionHandle handle) {
    return destroy_dnn_metadata(handle, ExecutionDnnObjectKind::kConvolution);
}
// NOLINTBEGIN(bugprone-easily-swappable-parameters): exact public SDK ABIs.
extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetFilter4dDescriptor(DnnFilterHandle handle,
                                                                cudnnDataType_t type,
                                                                cudnnTensorFormat_t format, int k,
                                                                int c, int h, int w) {
    return set_dnn_filter(handle, {dnn_enum_bits(type), dnn_enum_bits(format), 4, {k, c, h, w}},
                          true);
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetFilterNdDescriptor(DnnFilterHandle handle,
                                                                cudnnDataType_t type,
                                                                cudnnTensorFormat_t format,
                                                                int rank, const int* sizes) {
    if (::getpid() != kProcess) return CUDNN_STATUS_NOT_INITIALIZED;
    if (rank < 0 || rank > CUDNN_DIM_MAX) return CUDNN_STATUS_BAD_PARAM;
    ExecutionDnnFilter value{dnn_enum_bits(type), dnn_enum_bits(format), rank};
    if (!self_copy(value.sizes.data(), sizes, static_cast<std::size_t>(rank) * sizeof(int), false))
        return CUDNN_STATUS_BAD_PARAM;
    return set_dnn_filter(handle, value, false);
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetFilter4dDescriptor(DnnFilterHandle handle,
                                                                cudnnDataType_t* type,
                                                                cudnnTensorFormat_t* format, int* k,
                                                                int* c, int* h, int* w) {
    return dnn_metadata(handle, ExecutionDnnObjectKind::kFilter, false,
                        [&](State& shared, Exclusion& exclusion, DnnEntry& entry) {
                            if (!writable_host_range(type, sizeof(*type)) ||
                                !writable_host_range(format, sizeof(*format)))
                                return CUDNN_STATUS_BAD_PARAM;
                            const std::array outputs{k, c, h, w};
                            for (auto* output : outputs)
                                if (!writable_host_range(output, sizeof(int)))
                                    return CUDNN_STATUS_BAD_PARAM;
                            ExecutionDnnFilter value;
                            const auto result =
                                query_dnn_filter(shared, exclusion, entry, 4, true, value);
                            if (result != CUDNN_STATUS_SUCCESS) return result;
                            if (!publish(type, static_cast<cudnnDataType_t>(value.type)) ||
                                !publish(format, static_cast<cudnnTensorFormat_t>(value.format)))
                                return CUDNN_STATUS_BAD_PARAM;
                            for (std::size_t i = 0; i != outputs.size(); ++i)
                                if (!publish(outputs[i], value.sizes[i]))
                                    return CUDNN_STATUS_BAD_PARAM;
                            return CUDNN_STATUS_SUCCESS;
                        });
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetFilterNdDescriptor(DnnFilterHandle handle,
                                                                int requested,
                                                                cudnnDataType_t* type,
                                                                cudnnTensorFormat_t* format,
                                                                int* rank, int* sizes) {
    return dnn_metadata(
        handle, ExecutionDnnObjectKind::kFilter, false,
        [&](State& shared, Exclusion& exclusion, DnnEntry& entry) {
            if (requested < 0 || !writable_host_range(type, sizeof(*type)) ||
                !writable_host_range(format, sizeof(*format)) ||
                !writable_host_range(rank, sizeof(*rank)))
                return CUDNN_STATUS_BAD_PARAM;
            ExecutionDnnFilter value;
            const auto result = query_dnn_filter(shared, exclusion, entry,
                                                 std::min(requested, CUDNN_DIM_MAX), false, value);
            if (result != CUDNN_STATUS_SUCCESS) return result;
            const auto bytes =
                static_cast<std::size_t>(std::min(requested, value.dimensions)) * sizeof(int);
            if (!writable_host_range(sizes, bytes)) return CUDNN_STATUS_BAD_PARAM;
            return publish(type, static_cast<cudnnDataType_t>(value.type)) &&
                           publish(format, static_cast<cudnnTensorFormat_t>(value.format)) &&
                           publish(rank, value.dimensions) &&
                           self_copy(sizes, value.sizes.data(), bytes, true)
                       ? CUDNN_STATUS_SUCCESS
                       : CUDNN_STATUS_BAD_PARAM;
        });
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetFilterSizeInBytes(DnnFilterHandle handle,
                                                               std::size_t* output) {
    return dnn_metadata(
        handle, ExecutionDnnObjectKind::kFilter, false,
        [&](State& shared, Exclusion& exclusion, DnnEntry& entry) {
            if (!writable_host_range(output, sizeof(*output))) return CUDNN_STATUS_BAD_PARAM;
            ExecutionPayloadWriter input;
            if (!input.write_u64(entry.remote)) return dnn_profile::host_allocation_status();
            const auto reply = shared.call(exclusion, *entry.context,
                                           ExecutionOpcode::kDnnFilterSize, std::move(input));
            const auto result = dnn_status(reply, *entry.context, &entry.is_uncertain);
            if (result != CUDNN_STATUS_SUCCESS) return result;
            if (reply->address > SIZE_MAX || !reply->data.empty())
                return dnn_invalid(*entry.context, &entry.is_uncertain);
            return publish(output, static_cast<std::size_t>(reply->address))
                       ? CUDNN_STATUS_SUCCESS
                       : CUDNN_STATUS_BAD_PARAM;
        });
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetConvolution2dDescriptor(DnnConvolutionHandle handle,
                                                                     int ph, int pw, int sh, int sw,
                                                                     int dh, int dw,
                                                                     cudnnConvolutionMode_t mode,
                                                                     cudnnDataType_t type) {
    return set_dnn_convolution(
        handle, {dnn_enum_bits(mode), dnn_enum_bits(type), 2, {ph, pw}, {sh, sw}, {dh, dw}}, true);
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetConvolutionNdDescriptor(
    DnnConvolutionHandle handle, int rank, const int* padding, const int* strides,
    const int* dilation, cudnnConvolutionMode_t mode, cudnnDataType_t type) {
    if (::getpid() != kProcess) return CUDNN_STATUS_NOT_INITIALIZED;
    if (rank < 0 || rank > CUDNN_DIM_MAX) return CUDNN_STATUS_BAD_PARAM;
    ExecutionDnnConvolutionState value{dnn_enum_bits(mode), dnn_enum_bits(type), rank};
    const auto bytes = static_cast<std::size_t>(rank) * sizeof(int);
    if (!self_copy(value.padding.data(), padding, bytes, false) ||
        !self_copy(value.strides.data(), strides, bytes, false) ||
        !self_copy(value.dilation.data(), dilation, bytes, false))
        return CUDNN_STATUS_BAD_PARAM;
    return set_dnn_convolution(handle, value, false);
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetConvolution2dDescriptor(DnnConvolutionHandle handle,
                                                                     int* ph, int* pw, int* sh,
                                                                     int* sw, int* dh, int* dw,
                                                                     cudnnConvolutionMode_t* mode,
                                                                     cudnnDataType_t* type) {
    return dnn_metadata(handle, ExecutionDnnObjectKind::kConvolution, false,
                        [&](State& shared, Exclusion& exclusion, DnnEntry& entry) {
                            if (!writable_host_range(mode, sizeof(*mode)) ||
                                !writable_host_range(type, sizeof(*type)))
                                return CUDNN_STATUS_BAD_PARAM;
                            const std::array outputs{ph, pw, sh, sw, dh, dw};
                            for (auto* output : outputs)
                                if (!writable_host_range(output, sizeof(int)))
                                    return CUDNN_STATUS_BAD_PARAM;
                            ExecutionDnnConvolutionState value;
                            const auto result =
                                query_dnn_convolution(shared, exclusion, entry, 2, true, value);
                            if (result != CUDNN_STATUS_SUCCESS) return result;
                            if (!publish(mode, static_cast<cudnnConvolutionMode_t>(value.mode)) ||
                                !publish(type, static_cast<cudnnDataType_t>(value.type)))
                                return CUDNN_STATUS_BAD_PARAM;
                            for (std::size_t i = 0; i != 2; ++i)
                                if (!publish(outputs[i], value.padding[i]) ||
                                    !publish(outputs[i + 2], value.strides[i]) ||
                                    !publish(outputs[i + 4], value.dilation[i]))
                                    return CUDNN_STATUS_BAD_PARAM;
                            return CUDNN_STATUS_SUCCESS;
                        });
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionNdDescriptor(
    DnnConvolutionHandle handle, int requested, int* rank, int* padding, int* strides,
    int* dilation, cudnnConvolutionMode_t* mode, cudnnDataType_t* type) {
    return dnn_metadata(
        handle, ExecutionDnnObjectKind::kConvolution, false,
        [&](State& shared, Exclusion& exclusion, DnnEntry& entry) {
            if (!writable_host_range(rank, sizeof(*rank)) ||
                !writable_host_range(mode, sizeof(*mode)) ||
                !writable_host_range(type, sizeof(*type)))
                return CUDNN_STATUS_BAD_PARAM;
            ExecutionDnnConvolutionState value;
            const auto result =
                query_dnn_convolution(shared, exclusion, entry, requested, false, value);
            if (result != CUDNN_STATUS_SUCCESS) return result;
            const auto bytes = static_cast<std::size_t>(requested) * sizeof(int);
            if (!writable_host_range(padding, bytes) || !writable_host_range(strides, bytes) ||
                !writable_host_range(dilation, bytes))
                return CUDNN_STATUS_BAD_PARAM;
            return publish(rank, value.dimensions) &&
                           publish(mode, static_cast<cudnnConvolutionMode_t>(value.mode)) &&
                           publish(type, static_cast<cudnnDataType_t>(value.type)) &&
                           self_copy(padding, value.padding.data(), bytes, true) &&
                           self_copy(strides, value.strides.data(), bytes, true) &&
                           self_copy(dilation, value.dilation.data(), bytes, true)
                       ? CUDNN_STATUS_SUCCESS
                       : CUDNN_STATUS_BAD_PARAM;
        });
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetConvolutionGroupCount(DnnConvolutionHandle handle,
                                                                   int value) {
    return set_dnn_convolution_attribute(handle, ExecutionDnnConvolutionAttribute::kGroup,
                                         std::bit_cast<std::uint32_t>(value));
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionGroupCount(DnnConvolutionHandle handle,
                                                                   int* value) {
    return get_dnn_convolution_attribute(handle, ExecutionDnnConvolutionAttribute::kGroup, value,
                                         UINT32_MAX);
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetConvolutionMathType(DnnConvolutionHandle handle,
                                                                 cudnnMathType_t value) {
    return set_dnn_convolution_attribute(handle, ExecutionDnnConvolutionAttribute::kMath,
                                         dnn_enum_bits(value));
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionMathType(DnnConvolutionHandle handle,
                                                                 cudnnMathType_t* value) {
    return get_dnn_convolution_attribute(handle, ExecutionDnnConvolutionAttribute::kMath, value, 3);
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetConvolutionReorderType(DnnConvolutionHandle handle,
                                                                    cudnnReorderType_t value) {
    return set_dnn_convolution_attribute(handle, ExecutionDnnConvolutionAttribute::kReorder,
                                         dnn_enum_bits(value));
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionReorderType(DnnConvolutionHandle handle,
                                                                    cudnnReorderType_t* value) {
    return get_dnn_convolution_attribute(handle, ExecutionDnnConvolutionAttribute::kReorder, value,
                                         1);
}
// NOLINTEND(bugprone-easily-swappable-parameters)

namespace {
template <typename Function>
cudnnStatus_t query_dnn_output(DnnConvolutionHandle convolution, cudnnTensorDescriptor_t tensor,
                               DnnFilterHandle filter, int requested, bool is_two,
                               Function&& publish_output) noexcept {
    return dnn_boundary([&](State& shared, Exclusion& exclusion) {
        if (requested < 0 || requested > CUDNN_DIM_MAX) return CUDNN_STATUS_BAD_PARAM;
        const std::array<const void*, 3> handles{convolution, tensor, filter};
        constexpr std::array kinds{ExecutionDnnObjectKind::kConvolution,
                                   ExecutionDnnObjectKind::kTensor,
                                   ExecutionDnnObjectKind::kFilter};
        return with_dnn_descriptors(
            shared, exclusion, handles, kinds, [&](Entry& metadata, const auto& descriptors) {
                ExecutionPayloadWriter input;
                for (std::size_t i = 0; i != handles.size(); ++i)
                    if (!input.write_u64(descriptors[i]->remote))
                        return dnn_profile::host_allocation_status();
                if (!input.write_u32(std::bit_cast<std::uint32_t>(requested)) ||
                    !input.write_u32(is_two ? 1 : 0))
                    return dnn_profile::host_allocation_status();
                const auto reply = shared.call(
                    exclusion, metadata, ExecutionOpcode::kDnnConvolutionOutput, std::move(input));
                const auto result = dnn_status(reply, metadata);
                if (result != CUDNN_STATUS_SUCCESS) return result;
                ExecutionPayloadReader reader(reply->data);
                std::uint32_t rank{}, bits{};
                std::array<std::int32_t, kExecutionDnnMaxDimensions> sizes{};
                if (!reader.read_u32(rank) || rank != static_cast<std::uint32_t>(requested))
                    return dnn_invalid(metadata);
                for (auto& size : sizes) {
                    if (!reader.read_u32(bits)) return dnn_invalid(metadata);
                    size = std::bit_cast<std::int32_t>(bits);
                }
                if (!reader.empty()) return dnn_invalid(metadata);
                return publish_output(sizes) ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_BAD_PARAM;
            });
    });
}
constexpr std::array kForwardDescriptorKinds{
    ExecutionDnnObjectKind::kTensor, ExecutionDnnObjectKind::kFilter,
    ExecutionDnnObjectKind::kConvolution, ExecutionDnnObjectKind::kTensor};
template <typename Function>
cudnnStatus_t query_dnn_forward(cudnnHandle_t handle, const std::array<const void*, 4>& handles,
                                ExecutionDnnConvolutionQueryKind kind, std::uint32_t algorithm,
                                int requested, Function&& publish_result) noexcept {
    return dnn_handle(handle, false, false, [&](State& shared, Exclusion& exclusion, Entry& entry) {
        ExecutionDnnConvolutionQuery query;
        query.kind = kind;
        query.handle = entry.remote;
        query.algorithm = algorithm;
        query.requested = std::bit_cast<std::uint32_t>(requested);
        const auto execute = [&] {
            ExecutionPayloadWriter input;
            if (!write_execution_dnn_convolution_query(input, query))
                return dnn_profile::host_allocation_status();
            const auto reply = shared.call(exclusion, *entry.context,
                                           ExecutionOpcode::kDnnConvolutionQuery, std::move(input));
            const auto result = dnn_status(reply, *entry.context, &entry.is_uncertain);
            return result == CUDNN_STATUS_SUCCESS ? publish_result(*reply, entry) : result;
        };
        if (kind == ExecutionDnnConvolutionQueryKind::kMaxCount) return execute();
        return with_dnn_descriptors(shared, exclusion, handles, kForwardDescriptorKinds,
                                    [&](Entry& metadata, const auto& descriptors) {
                                        query.metadata = metadata.connection->session();
                                        for (std::size_t i = 0; i != query.descriptors.size(); ++i)
                                            query.descriptors[i] = descriptors[i]->remote;
                                        return execute();
                                    });
    });
}
struct DnnConvolutionArguments {
    std::array<const void*, 4> descriptors{};
    std::array<const void*, 4> buffers{};
    const void* alpha{};
    const void* beta{};
    std::uint32_t algorithm{};
    std::size_t workspace_bytes{};
};
cudnnStatus_t publish_dnn_performances(const ExecutionWireReply& reply, Entry& entry, int requested,
                                       int* returned,
                                       cudnnConvolutionFwdAlgoPerfStruct* performance) {
    ExecutionPayloadReader reader(reply.data);
    ExecutionDnnConvolutionPerformances values;
    if (!read_execution_dnn_convolution_performances(reader, values) || !reader.empty() ||
        requested < 0 || values.count > static_cast<std::uint32_t>(requested))
        return dnn_invalid(*entry.context, &entry.is_uncertain);
    std::array<cudnnConvolutionFwdAlgoPerfStruct, kExecutionDnnMaxConvolutionAlgorithms> native{};
    for (std::size_t i = 0; i != values.count; ++i) {
        const auto& value = values.values[i];
        if (value.algorithm > 15 || value.status > dnn_profile::status_enum_max() ||
            value.memory > SIZE_MAX || value.determinism > 1 || value.math > 3)
            return dnn_invalid(*entry.context, &entry.is_uncertain);
        native[i].algo = static_cast<cudnnConvolutionFwdAlgo_t>(value.algorithm);
        native[i].status = static_cast<cudnnStatus_t>(value.status);
        native[i].time = std::bit_cast<float>(value.time_bits);
        native[i].memory = static_cast<std::size_t>(value.memory);
        native[i].determinism = static_cast<cudnnDeterminism_t>(value.determinism);
        native[i].mathType = static_cast<cudnnMathType_t>(value.math);
        std::copy(value.reserved.begin(), value.reserved.end(), native[i].reserved);
    }
    const auto bytes = static_cast<std::size_t>(values.count) * sizeof(native[0]);
    // Decode and validate the entire result before publishing any caller field.
    // Revalidate both output ranges after transport released frontend exclusion.
    if (!writable_host_range(returned, sizeof(*returned)) ||
        !writable_host_range(performance, bytes))
        return CUDNN_STATUS_BAD_PARAM;
    return self_copy(performance, native.data(), bytes, true) &&
                   publish(returned, static_cast<int>(values.count))
               ? CUDNN_STATUS_SUCCESS
               : CUDNN_STATUS_BAD_PARAM;
}
cudnnStatus_t submit_dnn_convolution(cudnnHandle_t handle,
                                     const DnnConvolutionArguments& arguments) noexcept {
    return dnn_handle(handle, false, true, [&](State& shared, Exclusion& exclusion, Entry& entry) {
        return with_dnn_descriptors(
            shared, exclusion, arguments.descriptors, kForwardDescriptorKinds,
            [&](Entry& metadata, const auto& descriptors) {
                if (entry.context->submissions == UINT64_MAX)
                    return dnn_profile::host_allocation_status();
                ExecutionDnnTensor tensor;
                const auto queried = query_dnn_tensor(shared, exclusion, *descriptors[0],
                                                      CUDNN_DIM_MAX, false, tensor);
                if (queried != CUDNN_STATUS_SUCCESS) return queried;
                ExecutionDnnConvolutionForward forward;
                forward.handle = entry.remote;
                forward.metadata = metadata.connection->session();
                forward.algorithm = arguments.algorithm;
                forward.workspace_bytes = arguments.workspace_bytes;
                forward.scalar_bytes =
                    tensor.type == CUDNN_DATA_DOUBLE ? sizeof(double) : sizeof(float);
                for (std::size_t i = 0; i != forward.descriptors.size(); ++i)
                    forward.descriptors[i] = descriptors[i]->remote;
                if (!self_copy(forward.alpha.data(), arguments.alpha, forward.scalar_bytes,
                               false) ||
                    !self_copy(forward.beta.data(), arguments.beta, forward.scalar_bytes, false))
                    return CUDNN_STATUS_BAD_PARAM;
                std::array<std::optional<CounterPin>, 4> buffer_pins, module_pins;
                for (std::size_t i = 0; i != forward.buffers.size(); ++i) {
                    const bool is_workspace = i == 3;
                    const auto address = reinterpret_cast<CUdeviceptr>(arguments.buffers[i]);
                    if (is_workspace && !address && !arguments.workspace_bytes) continue;
                    // Nonnull+zero retains the exact owned identity, including
                    // an allocation's one-past-end pointer, without inventing
                    // a byte of access. The service revalidates every extent.
                    const auto bytes = is_workspace ? arguments.workspace_bytes : std::size_t{1};
                    auto* allocation = shared.range(address, bytes);
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
                if (!write_execution_dnn_convolution_forward(input, forward))
                    return dnn_profile::host_allocation_status();
                ++entry.context->submissions;
                return dnn_status(
                    shared.call(exclusion, *entry.context, ExecutionOpcode::kDnnConvolutionForward,
                                std::move(input)),
                    *entry.context, &entry.is_uncertain);
            });
    });
}
}  // namespace
// NOLINTBEGIN(bugprone-easily-swappable-parameters): exact public SDK ABIs.
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetConvolution2dForwardOutputDim(
    DnnConvolutionHandle convolution, cudnnTensorDescriptor_t tensor, DnnFilterHandle filter,
    int* n, int* c, int* h, int* w) {
    const std::array outputs{n, c, h, w};
    for (auto* output : outputs)
        if (!writable_host_range(output, sizeof(int))) return CUDNN_STATUS_BAD_PARAM;
    return query_dnn_output(convolution, tensor, filter, 4, true, [&](const auto& sizes) {
        for (std::size_t i = 0; i != outputs.size(); ++i)
            if (!publish(outputs[i], sizes[i])) return false;
        return true;
    });
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionNdForwardOutputDim(
    DnnConvolutionHandle convolution, cudnnTensorDescriptor_t tensor, DnnFilterHandle filter,
    int requested, int* output) {
    if (requested < 0 || requested > CUDNN_DIM_MAX) return CUDNN_STATUS_BAD_PARAM;
    const auto bytes = static_cast<std::size_t>(requested) * sizeof(int);
    if (!writable_host_range(output, bytes)) return CUDNN_STATUS_BAD_PARAM;
    return query_dnn_output(convolution, tensor, filter, requested, false, [&](const auto& sizes) {
        return self_copy(output, sizes.data(), bytes, true);
    });
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionForwardWorkspaceSize(
    cudnnHandle_t handle, cudnnTensorDescriptor_t input, DnnFilterHandle filter,
    DnnConvolutionHandle convolution, cudnnTensorDescriptor_t output,
    cudnnConvolutionFwdAlgo_t algorithm, std::size_t* bytes) {
    if (!writable_host_range(bytes, sizeof(*bytes))) return CUDNN_STATUS_BAD_PARAM;
    return query_dnn_forward(
        handle, {input, filter, convolution, output}, ExecutionDnnConvolutionQueryKind::kWorkspace,
        dnn_enum_bits(algorithm), 0, [&](const ExecutionWireReply& reply, Entry& entry) {
            if (reply.address > SIZE_MAX || !reply.data.empty())
                return dnn_invalid(*entry.context, &entry.is_uncertain);
            return publish(bytes, static_cast<std::size_t>(reply.address)) ? CUDNN_STATUS_SUCCESS
                                                                           : CUDNN_STATUS_BAD_PARAM;
        });
}
extern "C" cudnnStatus_t CUDNNWINAPI
cudnnGetConvolutionForwardAlgorithmMaxCount(cudnnHandle_t handle, int* count) {
    if (!writable_host_range(count, sizeof(*count))) return CUDNN_STATUS_BAD_PARAM;
    return query_dnn_forward(
        handle, {}, ExecutionDnnConvolutionQueryKind::kMaxCount, 0, 0,
        [&](const ExecutionWireReply& reply, Entry& entry) {
            if (reply.address > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
                !reply.data.empty())
                return dnn_invalid(*entry.context, &entry.is_uncertain);
            return publish(count, static_cast<int>(reply.address)) ? CUDNN_STATUS_SUCCESS
                                                                   : CUDNN_STATUS_BAD_PARAM;
        });
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionForwardAlgorithm_v7(
    cudnnHandle_t handle, cudnnTensorDescriptor_t input, DnnFilterHandle filter,
    DnnConvolutionHandle convolution, cudnnTensorDescriptor_t output, int requested, int* returned,
    cudnnConvolutionFwdAlgoPerfStruct* performance) {
    if (!writable_host_range(returned, sizeof(*returned))) return CUDNN_STATUS_BAD_PARAM;
    return query_dnn_forward(
        handle, {input, filter, convolution, output}, ExecutionDnnConvolutionQueryKind::kHeuristic,
        0, requested, [&](const ExecutionWireReply& reply, Entry& entry) {
            return publish_dnn_performances(reply, entry, requested, returned, performance);
        });
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnConvolutionForward(
    cudnnHandle_t handle, const void* alpha, cudnnTensorDescriptor_t input, const void* x,
    DnnFilterHandle filter, const void* w, DnnConvolutionHandle convolution,
    cudnnConvolutionFwdAlgo_t algorithm, void* workspace, std::size_t workspace_bytes,
    const void* beta, cudnnTensorDescriptor_t output, void* y) {
    return submit_dnn_convolution(handle, {{input, filter, convolution, output},
                                           {x, w, y, workspace},
                                           alpha,
                                           beta,
                                           dnn_enum_bits(algorithm),
                                           workspace_bytes});
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnFindConvolutionForwardAlgorithm(
    cudnnHandle_t handle, cudnnTensorDescriptor_t input, DnnFilterHandle filter,
    DnnConvolutionHandle convolution, cudnnTensorDescriptor_t output, int requested, int* returned,
    cudnnConvolutionFwdAlgoPerfStruct* performance) {
    return dnn_handle(handle, false, true, [&](State& shared, Exclusion& exclusion, Entry& entry) {
        if (!writable_host_range(returned, sizeof(*returned))) return CUDNN_STATUS_BAD_PARAM;
        // Reject an obviously invalid result before launching benchmark work.
        // The exact returned extent is checked again after IPC; a large caller
        // capacity need not describe more storage than the SDK actually returns.
        if (requested > 0 && !writable_host_range(performance, sizeof(*performance)))
            return CUDNN_STATUS_BAD_PARAM;
        const std::array<const void*, 4> handles{input, filter, convolution, output};
        return with_dnn_descriptors(
            shared, exclusion, handles, kForwardDescriptorKinds,
            [&](Entry& metadata, const auto& descriptors) {
                if (entry.context->submissions == UINT64_MAX)
                    return dnn_profile::host_allocation_status();
                ExecutionDnnConvolutionFind find;
                find.handle = entry.remote;
                find.metadata = metadata.connection->session();
                find.requested = std::bit_cast<std::uint32_t>(requested);
                for (std::size_t i = 0; i != find.descriptors.size(); ++i)
                    find.descriptors[i] = descriptors[i]->remote;
                ExecutionPayloadWriter payload;
                if (!write_execution_dnn_convolution_find(payload, find))
                    return dnn_profile::host_allocation_status();
                // This is real benchmark execution. Admission/SDK status/result
                // publication are separate from service-owned physical proof.
                ++entry.context->submissions;
                const auto reply =
                    shared.call(exclusion, *entry.context, ExecutionOpcode::kDnnConvolutionFind,
                                std::move(payload));
                const auto result = dnn_status(reply, *entry.context, &entry.is_uncertain);
                if (result != CUDNN_STATUS_SUCCESS) return result;
                if (!reply->id) return dnn_invalid(*entry.context, &entry.is_uncertain);
                return publish_dnn_performances(*reply, entry, requested, returned, performance);
            });
    });
}
// The pinned reference leaves FindEx unimplemented. Keep its client-native
// opaque-descriptor boundary closed until its owned-buffer execution is added.
extern "C" cudnnStatus_t CUDNNWINAPI cudnnFindConvolutionForwardAlgorithmEx(
    cudnnHandle_t, cudnnTensorDescriptor_t, const void*, DnnFilterHandle, const void*,
    DnnConvolutionHandle, cudnnTensorDescriptor_t, void*, int, int*,
    cudnnConvolutionFwdAlgoPerfStruct*, void*, std::size_t) {
    return dnn_unsupported();
}
// NOLINTEND(bugprone-easily-swappable-parameters)
