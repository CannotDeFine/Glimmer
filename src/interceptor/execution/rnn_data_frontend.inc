// Host-only sequence metadata uses the authenticated DNM1 owner, never a
// caller-native descriptor. Variable inputs are bounded before copying.
namespace {
using DnnSeqDataHandle = cudnnSeqDataStruct*;

cudnnStatus_t sequence_metadata_status(const std::optional<ExecutionWireReply>& reply,
                                       DnnEntry& entry, bool is_mutating) {
    const auto invalid = [&] { return dnn_invalid(*entry.context, &entry.is_uncertain); };
    if (!reply || reply->id || reply->address || !reply->arguments.empty() ||
        reply->parameter_bytes > 1)
        return invalid();
    if (reply->disposition == ExecutionWireDisposition::kRejected) {
        if (reply->original_status || !reply->data.empty()) return invalid();
        return dnn_status(reply, *entry.context, &entry.is_uncertain);
    }
    if (reply->disposition != ExecutionWireDisposition::kComplete || !reply->original_status ||
        reply->status != *reply->original_status)
        return invalid();
    const auto status = dnn_status(reply, *entry.context, &entry.is_uncertain);
    if (status != CUDNN_STATUS_SUCCESS) {
        if (!reply->data.empty() || (is_mutating && reply->parameter_bytes != 1)) return invalid();
        return status;
    }
    if (reply->parameter_bytes || !valid_dnn_metadata(entry) ||
        (is_mutating && !reply->data.empty()))
        return invalid();
    return CUDNN_STATUS_SUCCESS;
}

std::uint32_t sequence_padding_width(std::uint32_t type, bool is_seq_setter = false) noexcept {
    if (type == 0) return 4;
    if (type == 1) return 8;
    if (type == 2 || (is_seq_setter && type == 9)) return 2;
    return 0;
}
template <typename Value>
bool capture_sequence_padding(Value& value, const void* padding, bool is_seq_setter) {
    if (!padding) return true;
    value.padding_bytes = sequence_padding_width(value.type, is_seq_setter);
    return !value.padding_bytes ||
           self_copy(value.padding.data(), padding, value.padding_bytes, false);
}
bool capture_sequence_lengths(std::vector<std::int32_t>& output, const int* source,
                              std::uint64_t count) {
    if (!source || count > kExecutionDnnMaxSequenceLengths) return false;
    output.resize(static_cast<std::size_t>(count));
    return self_copy(output.data(), source, output.size() * sizeof(int), false);
}
template <typename Value, typename Writer>
cudnnStatus_t send_sequence_set(State& shared, Exclusion& exclusion, DnnEntry& entry,
                                ExecutionOpcode operation, const Value& value, Writer writer) {
    ExecutionPayloadWriter payload;
    if (!payload.write_u64(entry.remote) || !writer(payload, value))
        return dnn_profile::host_allocation_status();
    return sequence_metadata_status(
        shared.call(exclusion, *entry.context, operation, std::move(payload)), entry, true);
}

bool valid_sequence_result(const ExecutionDnnSeqDataState& value,
                           const ExecutionDnnSeqDataQuery& query) {
    if (query.dimensions_requested < 0 || value.type > dnn_profile::data_type_enum_max() ||
        value.dimensions != 4 || value.sequence_count > kExecutionDnnMaxSequenceLengths)
        return false;
    const auto presence = ((query.outputs & 4) ? 1U : 0U) | ((query.outputs & 8) ? 2U : 0U) |
                          ((query.outputs & 32) ? 4U : 0U) | ((query.outputs & 64) ? 8U : 0U);
    const auto copied = static_cast<std::size_t>(std::min(query.dimensions_requested, 4));
    const auto lengths =
        (query.outputs & 32) ? std::min(query.lengths_requested, value.sequence_count) : 0;
    const auto padding = (query.outputs & 64) ? sequence_padding_width(value.type) : 0;
    if (value.presence != presence || value.lengths.size() != lengths ||
        value.padding_bytes != padding || ((query.outputs & 64) && !padding))
        return false;
    for (std::size_t i = 0; i != value.sizes.size(); ++i) {
        if (((query.outputs & 4) == 0 || i >= copied) && value.sizes[i] != 0) return false;
        if (value.axes[i] > 3 || (((query.outputs & 8) == 0 || i >= copied) && value.axes[i] != 0))
            return false;
    }
    return true;
}
}  // namespace

// NOLINTBEGIN(bugprone-easily-swappable-parameters): exact public SDK ABIs.
extern "C" cudnnStatus_t CUDNNWINAPI
cudnnCreateRNNDataDescriptor(cudnnRNNDataDescriptor_t* output) {
    return create_dnn_metadata(output, ExecutionDnnObjectKind::kRnnData, true);
}
extern "C" cudnnStatus_t CUDNNWINAPI
cudnnDestroyRNNDataDescriptor(cudnnRNNDataDescriptor_t descriptor) {
    return destroy_dnn_metadata(descriptor, ExecutionDnnObjectKind::kRnnData, true);
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetRNNDataDescriptor(cudnnRNNDataDescriptor_t descriptor,
                                                               cudnnDataType_t type,
                                                               cudnnRNNDataLayout_t layout,
                                                               int maximum, int batch, int size,
                                                               const int* lengths, void* padding) {
    return dnn_metadata(descriptor, ExecutionDnnObjectKind::kRnnData, true,
                        [&](State& shared, Exclusion& exclusion, DnnEntry& entry) {
                            ExecutionDnnRnnDataState value;
                            value.type = dnn_enum_bits(type);
                            value.layout = dnn_enum_bits(layout);
                            value.max_sequence_length = maximum;
                            value.batch_size = batch;
                            value.vector_size = size;
                            value.presence = (lengths ? 1U : 0U) | (padding ? 2U : 0U);
                            if (value.type > dnn_profile::data_type_enum_max() || value.layout > 3)
                                return CUDNN_STATUS_BAD_PARAM;
                            // Original validation rejects these scalar/null errors before reading
                            // arrays. Preserve presence but never inspect ignored caller pointers.
                            if (lengths && value.type <= 2 && value.layout <= 2 && maximum > 0 &&
                                maximum <= 65535 && batch > 0 && size > 0) {
                                if (!capture_sequence_lengths(value.lengths, lengths,
                                                              static_cast<std::uint64_t>(batch)) ||
                                    !capture_sequence_padding(value, padding, false))
                                    return CUDNN_STATUS_BAD_PARAM;
                            }
                            return send_sequence_set(shared, exclusion, entry,
                                                     ExecutionOpcode::kSetDnnRnnData, value,
                                                     write_execution_dnn_rnn_data);
                        });
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnCreateSeqDataDescriptor(DnnSeqDataHandle* output) {
    return create_dnn_metadata(output, ExecutionDnnObjectKind::kSeqData, true);
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnDestroySeqDataDescriptor(DnnSeqDataHandle descriptor) {
    return destroy_dnn_metadata(descriptor, ExecutionDnnObjectKind::kSeqData, true);
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetSeqDataDescriptor(
    DnnSeqDataHandle descriptor, cudnnDataType_t type, int dimensions, const int* sizes,
    const cudnnSeqDataAxis_t* axes, std::size_t count, const int* lengths, void* padding) {
    return dnn_metadata(
        descriptor, ExecutionDnnObjectKind::kSeqData, true,
        [&](State& shared, Exclusion& exclusion, DnnEntry& entry) {
            ExecutionDnnSeqDataState value;
            value.type = dnn_enum_bits(type);
            value.dimensions = dimensions;
            value.sequence_count = count;
            value.presence =
                (sizes ? 1U : 0U) | (axes ? 2U : 0U) | (lengths ? 4U : 0U) | (padding ? 8U : 0U);
            if (value.type > dnn_profile::data_type_enum_max()) return CUDNN_STATUS_BAD_PARAM;
            if (value.type <= 16 && dimensions == 4) {
                if (!sizes || !self_copy(value.sizes.data(), sizes, sizeof(value.sizes), false))
                    return CUDNN_STATUS_BAD_PARAM;
                if (std::ranges::all_of(value.sizes, [](auto size) { return size > 0; })) {
                    const auto product = static_cast<std::uint64_t>(value.sizes[1]) *
                                         static_cast<std::uint64_t>(value.sizes[2]);
                    if (product > INT32_MAX) return CUDNN_STATUS_BAD_PARAM;
                    if (product == value.sequence_count) {
                        if (!capture_sequence_lengths(value.lengths, lengths, product) || !axes ||
                            !self_copy(value.axes.data(), axes, sizeof(value.axes), false) ||
                            std::ranges::any_of(value.axes, [](auto axis) { return axis > 3; }) ||
                            !capture_sequence_padding(value, padding, true))
                            return CUDNN_STATUS_BAD_PARAM;
                    }
                }
            }
            return send_sequence_set(shared, exclusion, entry, ExecutionOpcode::kSetDnnSeqData,
                                     value, write_execution_dnn_seq_data);
        });
}
extern "C" cudnnStatus_t CUDNNWINAPI
cudnnGetSeqDataDescriptor(const DnnSeqDataHandle descriptor, cudnnDataType_t* type, int* dimensions,
                          int requested, int* sizes, cudnnSeqDataAxis_t* axes, std::size_t* count,
                          std::size_t lengths_requested, int* lengths, void* padding) {
    return dnn_metadata(
        descriptor, ExecutionDnnObjectKind::kSeqData, false,
        [&](State& shared, Exclusion& exclusion, DnnEntry& entry) {
            const auto outputs = (type ? 1U : 0U) | (dimensions ? 2U : 0U) | (sizes ? 4U : 0U) |
                                 (axes ? 8U : 0U) | (count ? 16U : 0U) | (lengths ? 32U : 0U) |
                                 (padding ? 64U : 0U);
            const ExecutionDnnSeqDataQuery query{requested, lengths_requested, outputs};
            ExecutionPayloadWriter payload;
            if (!payload.write_u64(entry.remote) ||
                !write_execution_dnn_seq_data_query(payload, query))
                return dnn_profile::host_allocation_status();
            const auto reply = shared.call(exclusion, *entry.context,
                                           ExecutionOpcode::kGetDnnSeqData, std::move(payload));
            const auto status = sequence_metadata_status(reply, entry, false);
            if (status != CUDNN_STATUS_SUCCESS) return status;
            ExecutionDnnSeqDataState value;
            ExecutionPayloadReader reader(reply->data);
            if (!read_execution_dnn_seq_data(reader, value) || !reader.empty() ||
                !valid_sequence_result(value, query))
                return dnn_invalid(*entry.context, &entry.is_uncertain);
            const auto dimension_bytes =
                static_cast<std::size_t>(std::min(requested, 4)) * sizeof(int);
            const auto length_bytes = value.lengths.size() * sizeof(int);
            if ((type && !writable_host_range(type, sizeof(*type))) ||
                (dimensions && !writable_host_range(dimensions, sizeof(*dimensions))) ||
                (count && !writable_host_range(count, sizeof(*count))) ||
                (sizes && !writable_host_range(sizes, dimension_bytes)) ||
                (axes && !writable_host_range(axes, dimension_bytes)) ||
                (lengths && !writable_host_range(lengths, length_bytes)) ||
                (padding && !writable_host_range(padding, value.padding_bytes)))
                return CUDNN_STATUS_BAD_PARAM;
            return (!type || publish(type, static_cast<cudnnDataType_t>(value.type))) &&
                           (!dimensions || publish(dimensions, value.dimensions)) &&
                           (!count ||
                            publish(count, static_cast<std::size_t>(value.sequence_count))) &&
                           (!sizes ||
                            self_copy(sizes, value.sizes.data(), dimension_bytes, true)) &&
                           (!axes || self_copy(axes, value.axes.data(), dimension_bytes, true)) &&
                           (!lengths ||
                            self_copy(lengths, value.lengths.data(), length_bytes, true)) &&
                           (!padding ||
                            self_copy(padding, value.padding.data(), value.padding_bytes, true))
                       ? CUDNN_STATUS_SUCCESS
                       : CUDNN_STATUS_BAD_PARAM;
        });
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetRNNDataDescriptor(cudnnRNNDataDescriptor_t,
                                                               cudnnDataType_t*,
                                                               cudnnRNNDataLayout_t*, int*, int*,
                                                               int*, int, int*, void*) {
    return dnn_unsupported();
}
// NOLINTEND(bugprone-easily-swappable-parameters)
