#include "internal/cuda_execution_session_state.h"

// cuDNN operations, compiled only with GLIMMER_HAS_CUDNN.
namespace glimmer::backend::cuda {
#ifdef GLIMMER_HAS_CUDNN
CudaDnnResult CudaExecutionSession::create_dnn() noexcept {
    auto& state = *implementation_;
    const auto gate = state.cold_gate(true);
    if (gate.status != CUDA_SUCCESS) return dnn_gate(gate);
    const auto& sdk = state.dispatch.dnn;
    if (!sdk || !sdk->create || !sdk->destroy || !sdk->set_stream || !sdk->get_stream)
        return dnn_gate(rejected(CUDA_ERROR_NOT_SUPPORTED));
    if (!state.has_id_space || state.storage->dnn.size() == state.limits.max_objects)
        return dnn_gate(rejected(CUDA_ERROR_OUT_OF_MEMORY));
    std::optional<NativeExecutionScope> proxy_scope;
    if (state.stream_config.uses_default_proxy) proxy_scope.emplace(state.is_native_issue_active);
    Implementation::Dnn handle{.id = state.next_id};
    auto result = dnn_completed(state.invoke(sdk->create, &handle.native));
    const auto after_create = state.check_context(state.stream_config.uses_default_proxy);
    if (result.original_status != CUDNN_STATUS_SUCCESS && !handle.native) {
        if (after_create != CUDA_SUCCESS) {
            state.fault();
            result.disposition = CudaDnnDisposition::kRejected;
            result.status = after_create;
        }
        return result;
    }
    (void)state.consume_id();
    handle.is_uncertain = result.original_status != CUDNN_STATUS_SUCCESS || !handle.native ||
                          after_create != CUDA_SUCCESS;
    for (auto& existing : state.storage->dnn)
        if (existing.native == handle.native) existing.is_uncertain = handle.is_uncertain = true;
    state.storage->dnn.push_back(handle);
    if (!handle.is_uncertain && state.storage->default_proxy) {
        const auto configured =
            state.invoke(sdk->set_stream, handle.native,
                         reinterpret_cast<cudaStream_t>(state.storage->default_proxy));
        const auto proof = state.check_context(true);
        handle.is_uncertain = configured != CUDNN_STATUS_SUCCESS || proof != CUDA_SUCCESS;
    }
    if (!handle.is_uncertain) {
        cudaStream_t observed{};
        const auto queried = state.invoke(sdk->get_stream, handle.native, &observed);
        handle.is_uncertain = queried != CUDNN_STATUS_SUCCESS ||
                              reinterpret_cast<CUstream>(observed) != state.storage->default_proxy;
    }
    const auto final_proof = state.stream_config.uses_default_proxy || !handle.is_uncertain
                                 ? state.check_context(state.stream_config.uses_default_proxy)
                                 : CUDA_SUCCESS;
    if (handle.is_uncertain || final_proof != CUDA_SUCCESS) {
        state.storage->dnn.back().is_uncertain = true;
        state.fault();
        result.disposition = CudaDnnDisposition::kRejected;
        result.status = CUDA_ERROR_UNKNOWN;
    } else
        result.id = handle.id;
    return result;
}
CudaDnnResult CudaExecutionSession::destroy_dnn(std::uint64_t id) noexcept {
    auto& state = *implementation_;
    const auto gate = state.cold_gate();
    if (gate.status != CUDA_SUCCESS) return dnn_gate(gate);
    const auto found = std::ranges::find(state.storage->dnn, id, &Implementation::Dnn::id);
    if (found == state.storage->dnn.end()) return dnn_gate(rejected(CUDA_ERROR_INVALID_HANDLE));
    if (found->is_uncertain || found->has_destroy_attempt)
        return dnn_gate(rejected(CUDA_ERROR_UNKNOWN));
    found->has_destroy_attempt = true;
    auto result = dnn_completed(state.invoke(state.dispatch.dnn->destroy, found->native));
    if (result.original_status == CUDNN_STATUS_SUCCESS)
        state.storage->dnn.erase(found);
    else {
        found->is_uncertain = true;
        state.fault();
    }
    if (state.check_context() != CUDA_SUCCESS) {
        state.fault();
        result.disposition = CudaDnnDisposition::kRejected;
        result.status = CUDA_ERROR_UNKNOWN;
    }
    return result;
}
// NOLINTBEGIN(bugprone-easily-swappable-parameters): logical handle and stream wire IDs.
CudaDnnResult CudaExecutionSession::set_dnn_stream(std::uint64_t id,
                                                   std::uint64_t stream_id) noexcept {
    // NOLINTEND(bugprone-easily-swappable-parameters)
    auto& state = *implementation_;
    const auto gate = state.cold_gate();
    if (gate.status != CUDA_SUCCESS) return dnn_gate(gate);
    const auto found = std::ranges::find(state.storage->dnn, id, &Implementation::Dnn::id);
    if (found == state.storage->dnn.end()) return dnn_gate(rejected(CUDA_ERROR_INVALID_HANDLE));
    if (found->is_uncertain || found->has_destroy_attempt)
        return dnn_gate(rejected(CUDA_ERROR_UNKNOWN));
    CUstream stream = state.storage->default_proxy;
    if (stream_id) {
        const auto owned =
            std::ranges::find(state.storage->streams, stream_id, &Implementation::Stream::id);
        if (owned == state.storage->streams.end())
            return dnn_gate(rejected(CUDA_ERROR_INVALID_HANDLE));
        stream = owned->native;
    }
    std::optional<NativeExecutionScope> proxy_scope;
    if (state.stream_config.uses_default_proxy) proxy_scope.emplace(state.is_native_issue_active);
    auto result = dnn_completed(state.invoke(state.dispatch.dnn->set_stream, found->native,
                                             reinterpret_cast<cudaStream_t>(stream)));
    if (result.original_status == CUDNN_STATUS_SUCCESS)
        found->stream = stream_id;
    else {
        // A failed native mutation does not prove the old stream is still set.
        found->is_uncertain = true;
        state.fault();
    }
    if (state.check_context(state.stream_config.uses_default_proxy) != CUDA_SUCCESS) {
        state.fault();
        result.disposition = CudaDnnDisposition::kRejected;
        result.status = CUDA_ERROR_UNKNOWN;
    }
    return result;
}
CudaDnnResult CudaExecutionSession::get_dnn_stream(std::uint64_t id) noexcept {
    auto& state = *implementation_;
    if (!state.owns() || state.is_faulted) return dnn_gate(rejected(CUDA_ERROR_UNKNOWN));
    if (state.is_native_issue_active || state.is_folded_active || state.is_rnn_query_active)
        return {.disposition = CudaDnnDisposition::kDeferred, .status = CUDA_ERROR_NOT_READY};
    struct ActiveCall {
        bool& active;
        ~ActiveCall() {
            active = false;
        }
    };
    state.is_rnn_query_active = true;
    const ActiveCall active{state.is_rnn_query_active};
    return state.observe_dnn_stream(id);
}
CudaDnnResult CudaExecutionSession::query_dnn_dropout_states_size(std::uint64_t id) noexcept {
    auto& state = *implementation_;
    if (!state.owns() || !state.executor || state.is_faulted ||
        state.executor->snapshot().is_quarantined)
        return dnn_gate(rejected(CUDA_ERROR_UNKNOWN));
    if (state.is_closing) return dnn_gate(rejected(CUDA_ERROR_DEINITIALIZED));
    if (const auto status = state.check_context(); status != CUDA_SUCCESS)
        return dnn_gate(rejected(status));
    const auto& sdk = state.dispatch.dnn;
    const auto handle = std::ranges::find(state.storage->dnn, id, &Implementation::Dnn::id);
    if (!sdk || handle == state.storage->dnn.end() || handle->is_uncertain ||
        handle->has_destroy_attempt)
        return dnn_gate(rejected(CUDA_ERROR_INVALID_HANDLE));
    if (!sdk->dropout_states_size) return dnn_gate(rejected(CUDA_ERROR_NOT_SUPPORTED));
    // This describes storage only and must not drain previously accepted work.
    std::size_t bytes{};
    auto result = dnn_completed(state.invoke(sdk->dropout_states_size, handle->native, &bytes));
    if (const auto status = state.check_context(); status != CUDA_SUCCESS) {
        state.fault();
        result.disposition = CudaDnnDisposition::kRejected;
        result.status = status;
    } else if (result.original_status == CUDNN_STATUS_SUCCESS) {
        result.value = bytes;
    }
    return result;
}
std::optional<CudaDeviceMemoryView> CudaExecutionSession::resolve_dnn_dropout(
    const CudaDnnDropoutView& dropout) const noexcept {
    const auto& state = *implementation_;
    if (!state.owns() || !state.executor || state.is_closing || state.is_faulted ||
        state.executor->snapshot().is_quarantined || state.check_context() != CUDA_SUCCESS ||
        dropout.object.kind != CudaDnnObjectKind::kDropout || !dropout.object.native ||
        !dropout.object.lifetime)
        return std::nullopt;
    const auto& binding = dropout.binding;
    // A newly created/null-only descriptor has no proven state allocation.
    // Preserve that null state for the original consumer's legality decision.
    if (!binding.allocation) return CudaDeviceMemoryView{};
    if (binding.context.lock() != state.storage) return std::nullopt;
    return device_memory({binding.allocation, binding.offset}, binding.bytes);
}
CudaDnnComputeAcceptance CudaExecutionSession::accept_dnn_dropout(
    std::uint64_t sequence, const CudaSessionDnnDropout& problem) noexcept {
    auto& state = *implementation_;
    if (!state.owns() || !state.executor || state.is_faulted ||
        state.executor->snapshot().is_quarantined)
        return {{.status = ExecutionAcceptStatus::kQuarantined, .receipt = {}}, {}};
    if (state.is_closing) return {{.status = ExecutionAcceptStatus::kClosed, .receipt = {}}, {}};
    if (state.check_context() != CUDA_SUCCESS) return {};
    const auto& sdk = state.dispatch.dnn;
    const auto handle =
        std::ranges::find(state.storage->dnn, problem.handle, &Implementation::Dnn::id);
    const bool is_set = problem.kind == CudaDnnDropoutKind::kSet;
    const bool is_restore = problem.kind == CudaDnnDropoutKind::kRestore;
    if (!sdk || (!is_set && !is_restore) || (is_set && !sdk->dropout_set) ||
        (is_restore && !sdk->dropout_restore) || !state.dispatch.context_synchronize ||
        handle == state.storage->dnn.end() || handle->is_uncertain || handle->has_destroy_attempt ||
        !problem.mutation || !problem.mutation->native())
        return {};
    const auto stream = get_dnn_stream(problem.handle);
    if (stream.status != CUDA_SUCCESS)
        return {{.status = state.is_faulted ? ExecutionAcceptStatus::kQuarantined
                                            : ExecutionAcceptStatus::kInvalid,
                 .receipt = {}},
                stream.original_status};
    if (stream.original_status != CUDNN_STATUS_SUCCESS) return {{}, stream.original_status};
    CudaDeviceMemoryView states;
    auto binding = problem.mutation->old_binding();
    if (problem.states.allocation) {
        const auto memory = device_memory(problem.states, problem.state_bytes);
        if (!memory) return {};
        states = *memory;
        binding = {state.storage, problem.states.allocation, problem.states.offset,
                   problem.state_bytes};
    } else if (problem.states.offset) {
        return {};
    }
    // Null Set updates only metadata; null Restore has no documented detach
    // contract. Both forward the original count and preserve prior provenance.
    try {
        auto call = make_cuda_dnn_dropout_call(
            sdk, handle->native, problem, std::move(states), std::move(binding),
            {state.storage->context, state.dispatch.execution.context_get_current,
             state.dispatch.context_synchronize});
        return {
            state.executor->accept_library(
                sequence, {.stream = handle->stream, .parameters = {}, .call = std::move(call)}),
            {}};
    } catch (...) {
        return {{.status = ExecutionAcceptStatus::kFull, .receipt = {}}, {}};
    }
}
CudaDnnComputeAcceptance CudaExecutionSession::accept_dnn_reorder(
    std::uint64_t sequence, const CudaSessionDnnReorder& problem) noexcept {
    auto& state = *implementation_;
    if (!state.owns() || !state.executor || state.is_faulted ||
        state.executor->snapshot().is_quarantined)
        return {{.status = ExecutionAcceptStatus::kQuarantined, .receipt = {}}, {}};
    if (state.is_closing) return {{.status = ExecutionAcceptStatus::kClosed, .receipt = {}}, {}};
    if (state.check_context() != CUDA_SUCCESS) return {};
    const auto& sdk = state.dispatch.dnn;
    const auto handle =
        std::ranges::find(state.storage->dnn, problem.handle, &Implementation::Dnn::id);
    if (!sdk || handle == state.storage->dnn.end() || handle->is_uncertain ||
        handle->has_destroy_attempt || !valid_cuda_dnn_reorder_descriptors(problem) ||
        problem.reorder > 1)
        return {};
    if (!sdk->version || !sdk->reorder_filter_bias || !sdk->filter_get_nd || !sdk->filter_size)
        return {{}, {}, CUDA_ERROR_NOT_SUPPORTED};
    // The two native branches have different bias footprints. This exact
    // version gate does not replace runtime qualification of mapped libraries.
    const auto version = state.invoke(sdk->version);
    if (state.check_context() != CUDA_SUCCESS) {
        state.fault();
        return {{.status = ExecutionAcceptStatus::kQuarantined, .receipt = {}}, {}};
    }
    if (version != 91002) return {{}, {}, CUDA_ERROR_NOT_SUPPORTED};
    for (std::size_t index = 0; index != problem.buffers.size(); ++index) {
        const auto& range = problem.buffers[index];
        if (index >= 2 && !problem.reorder_bias) {
            if (range.allocation || range.offset) return {};
        } else if (!range.allocation) {
            // Negative flags also access bias in the inspected native binary.
            // Nonzero/null rejection is an explicit compatibility restriction.
            return {};
        }
    }
    const auto stream = get_dnn_stream(problem.handle);
    if (stream.status != CUDA_SUCCESS)
        return {{.status = state.is_faulted ? ExecutionAcceptStatus::kQuarantined
                                            : ExecutionAcceptStatus::kInvalid,
                 .receipt = {}},
                stream.original_status};
    if (stream.original_status != CUDNN_STATUS_SUCCESS) return {{}, stream.original_status};
    const auto descriptor = static_cast<CudaDnnFilterHandle>(problem.filter.native);
    cudnnDataType_t type{};
    cudnnTensorFormat_t format{};
    int dimensions{};
    std::array<int, CUDNN_DIM_MAX> sizes{};
    auto status = state.invoke(sdk->filter_get_nd, descriptor, CUDNN_DIM_MAX, &type, &format,
                               &dimensions, sizes.data());
    if (state.check_context() != CUDA_SUCCESS) {
        state.fault();
        return {{.status = ExecutionAcceptStatus::kQuarantined, .receipt = {}}, status};
    }
    if (status != CUDNN_STATUS_SUCCESS) return {{}, status};
    std::uint32_t type_bits{}, format_bits{};
    static_assert(sizeof(type) == sizeof(type_bits) && sizeof(format) == sizeof(format_bits));
    std::memcpy(&type_bits, &type, sizeof(type_bits));
    std::memcpy(&format_bits, &format, sizeof(format_bits));
    if (dimensions <= 0 || dimensions > CUDNN_DIM_MAX ||
        type_bits > dnn_profile::data_type_enum_max() || format_bits > 3 ||
        std::ranges::any_of(std::span{sizes.data(), static_cast<std::size_t>(dimensions)},
                            [](int size) { return size <= 0; })) {
        state.fault();
        return {{.status = ExecutionAcceptStatus::kQuarantined, .receipt = {}}, status};
    }
    std::size_t filter_bytes{};
    status = state.invoke(sdk->filter_size, descriptor, &filter_bytes);
    if (state.check_context() != CUDA_SUCCESS) {
        state.fault();
        return {{.status = ExecutionAcceptStatus::kQuarantined, .receipt = {}}, status};
    }
    if (status != CUDNN_STATUS_SUCCESS) return {{}, status};
    if (!filter_bytes) return {};
    // Native FilterNd expands vectorized C, while FilterSize already includes
    // the vector width. Never multiply logical INT8x32 K*C*R*S by 32 again.
    constexpr std::array<std::size_t, 17> native_widths{4, 8, 2, 1, 4, 4, 1, 4, 32,
                                                        2, 8, 1, 1, 1, 4, 1, 1};
    std::size_t bias_bytes{};
    if (problem.reorder == 1 && type_bits >= native_widths.size())
        return {{}, {}, CUDA_ERROR_NOT_SUPPORTED};
    if (problem.reorder_bias) {
        const auto width = problem.reorder == 0 ? sizeof(float) : native_widths[type_bits];
        const auto channels = static_cast<std::size_t>(sizes[0]);
        if (channels > std::numeric_limits<std::size_t>::max() / width) return {};
        bias_bytes = channels * width;
    }
    std::array<CudaDeviceMemoryView, 4> buffers{};
    for (std::size_t index = 0; index != buffers.size(); ++index) {
        if (index >= 2 && !problem.reorder_bias) continue;
        const auto memory =
            device_memory(problem.buffers[index], index < 2 ? filter_bytes : bias_bytes);
        if (!memory) return {};
        buffers[index] = *memory;
    }
    for (std::size_t output : {std::size_t{1}, std::size_t{3}}) {
        const auto& destination = buffers[output];
        if (!destination.bytes) continue;
        for (std::size_t index = 0; index != buffers.size(); ++index) {
            const auto& other = buffers[index];
            if (index == output || !other.bytes) continue;
            // General in-place reordering is not qualified. Read/read aliases
            // and disjoint views of one allocation remain valid.
            const bool is_disjoint = destination.address <= other.address
                                         ? other.address - destination.address >= destination.bytes
                                         : destination.address - other.address >= other.bytes;
            if (!is_disjoint) return {};
        }
    }
    if (state.check_context() != CUDA_SUCCESS) {
        state.fault();
        return {{.status = ExecutionAcceptStatus::kQuarantined, .receipt = {}}, {}};
    }
    try {
        return {state.executor->accept_library(
                    sequence, {.stream = handle->stream,
                               .parameters = {},
                               .call = make_cuda_dnn_reorder_call(sdk, handle->native, problem,
                                                                  std::move(buffers))}),
                {}};
    } catch (...) {
        return {{.status = ExecutionAcceptStatus::kFull, .receipt = {}}, {}};
    }
}

CudaDnnResult CudaExecutionSession::query_dnn_reduction(const CudaSessionDnnReduction& problem,
                                                        CudaDnnReductionQueryKind kind) noexcept {
    auto& state = *implementation_;
    if (!state.owns() || !state.executor || state.is_faulted ||
        state.executor->snapshot().is_quarantined)
        return dnn_gate(rejected(CUDA_ERROR_UNKNOWN));
    if (state.is_closing) return dnn_gate(rejected(CUDA_ERROR_DEINITIALIZED));
    if (const auto status = state.check_context(); status != CUDA_SUCCESS)
        return dnn_gate(rejected(status));
    const auto& sdk = state.dispatch.dnn;
    const auto handle =
        std::ranges::find(state.storage->dnn, problem.handle, &Implementation::Dnn::id);
    if (!sdk || handle == state.storage->dnn.end() || handle->is_uncertain ||
        handle->has_destroy_attempt || !valid_cuda_dnn_reduction_descriptors(problem))
        return dnn_gate(rejected(CUDA_ERROR_INVALID_HANDLE));
    if (kind != CudaDnnReductionQueryKind::kIndices &&
        kind != CudaDnnReductionQueryKind::kWorkspace)
        return dnn_gate(rejected(CUDA_ERROR_INVALID_VALUE));
    const auto query = kind == CudaDnnReductionQueryKind::kIndices ? sdk->reduction_indices_size
                                                                   : sdk->reduction_workspace_size;
    if (!query) return dnn_gate(rejected(CUDA_ERROR_NOT_SUPPORTED));
    // Both queries describe storage; neither issues computation nor needs to
    // drain unrelated accepted work. Immutable descriptor borrows cover them.
    std::size_t bytes{};
    auto result = dnn_completed(state.invoke(
        query, handle->native, static_cast<CudaDnnReduction>(problem.descriptors[1].native),
        static_cast<cudnnTensorDescriptor_t>(problem.descriptors[0].native),
        static_cast<cudnnTensorDescriptor_t>(problem.descriptors[2].native), &bytes));
    if (state.check_context() != CUDA_SUCCESS) {
        state.fault();
        result.disposition = CudaDnnDisposition::kRejected;
        result.status = CUDA_ERROR_UNKNOWN;
    } else if (result.original_status == CUDNN_STATUS_SUCCESS)
        result.value = bytes;
    return result;
}

CudaDnnComputeAcceptance CudaExecutionSession::accept_dnn_reduction(
    std::uint64_t sequence, const CudaSessionDnnReduction& problem) noexcept {
    auto& state = *implementation_;
    if (!state.owns() || !state.executor || state.is_faulted ||
        state.executor->snapshot().is_quarantined)
        return {{.status = ExecutionAcceptStatus::kQuarantined, .receipt = {}}, {}};
    if (state.is_closing) return {{.status = ExecutionAcceptStatus::kClosed, .receipt = {}}, {}};
    if (state.check_context() != CUDA_SUCCESS) return {};
    const auto& sdk = state.dispatch.dnn;
    const auto handle =
        std::ranges::find(state.storage->dnn, problem.handle, &Implementation::Dnn::id);
    if (!sdk || !sdk->reduce_tensor || !sdk->tensor_get_nd || !sdk->tensor_size ||
        handle == state.storage->dnn.end() || handle->is_uncertain || handle->has_destroy_attempt ||
        !valid_cuda_dnn_reduction_descriptors(problem))
        return {};
    const auto stream = get_dnn_stream(problem.handle);
    if (stream.status != CUDA_SUCCESS)
        return {{.status = state.is_faulted ? ExecutionAcceptStatus::kQuarantined
                                            : ExecutionAcceptStatus::kInvalid,
                 .receipt = {}},
                stream.original_status};
    if (stream.original_status != CUDNN_STATUS_SUCCESS) return {{}, stream.original_status};
    std::array<CudaDeviceMemoryView, 4> buffers{};
    for (std::size_t index = 0; index != 2; ++index) {
        const auto descriptor =
            static_cast<cudnnTensorDescriptor_t>(problem.descriptors[index * 2].native);
        cudnnDataType_t type{};
        int rank{};
        std::array<int, CUDNN_DIM_MAX> sizes{}, strides{};
        auto status = state.invoke(sdk->tensor_get_nd, descriptor, CUDNN_DIM_MAX, &type, &rank,
                                   sizes.data(), strides.data());
        if (state.check_context() != CUDA_SUCCESS) {
            state.fault();
            return {{.status = ExecutionAcceptStatus::kQuarantined, .receipt = {}}, status};
        }
        if (status != CUDNN_STATUS_SUCCESS) return {{}, status};
        std::uint32_t type_bits{};
        std::memcpy(&type_bits, &type, sizeof(type_bits));
        if (rank <= 0 || rank > CUDNN_DIM_MAX || type_bits > dnn_profile::data_type_enum_max()) {
            state.fault();
            return {{.status = ExecutionAcceptStatus::kQuarantined, .receipt = {}}, status};
        }
        // Reduction permits mixed non-double storage types, all with float
        // coefficients. The native SDK validates the actual combination.
        const auto width = type_bits == CUDNN_DATA_DOUBLE ? sizeof(double) : sizeof(float);
        if (problem.scalar_bytes != width) return {};
        std::size_t bytes{};
        status = state.invoke(sdk->tensor_size, descriptor, &bytes);
        if (state.check_context() != CUDA_SUCCESS) {
            state.fault();
            return {{.status = ExecutionAcceptStatus::kQuarantined, .receipt = {}}, status};
        }
        if (status != CUDNN_STATUS_SUCCESS) return {{}, status};
        if (!bytes) return {};
        const auto memory = device_memory(problem.buffers[index], bytes);
        if (!memory) return {};
        buffers[index] = *memory;
    }
    for (std::size_t index = 2; index != buffers.size(); ++index) {
        const auto& range = problem.buffers[index];
        const auto bytes = index == 2 ? problem.indices_bytes : problem.workspace_bytes;
        if (range.allocation) {
            const auto memory = device_memory(range, bytes);
            if (!memory) return {};
            buffers[index] = *memory;
        } else if (range.offset) {
            return {};
        }
        // A null optional pointer preserves its original count. The SDK
        // decides whether the space is ignored or insufficient, not the wire.
    }
    try {
        return {state.executor->accept_library(
                    sequence, {.stream = handle->stream,
                               .parameters = {},
                               .call = make_cuda_dnn_reduction_call(sdk, handle->native, problem,
                                                                    std::move(buffers))}),
                {}};
    } catch (...) {
        return {{.status = ExecutionAcceptStatus::kFull, .receipt = {}}, {}};
    }
}

CudaDnnResult CudaExecutionSession::query_dnn_batch_norm(
    const CudaSessionDnnBatchNorm& problem) noexcept {
    auto& state = *implementation_;
    if (!state.owns() || !state.executor || state.is_faulted ||
        state.executor->snapshot().is_quarantined)
        return dnn_gate(rejected(CUDA_ERROR_UNKNOWN));
    if (state.is_closing) return dnn_gate(rejected(CUDA_ERROR_DEINITIALIZED));
    if (const auto status = state.check_context(); status != CUDA_SUCCESS)
        return dnn_gate(rejected(status));
    const auto& sdk = state.dispatch.dnn;
    const auto handle =
        std::ranges::find(state.storage->dnn, problem.handle, &Implementation::Dnn::id);
    if (!sdk || handle == state.storage->dnn.end() || handle->is_uncertain ||
        handle->has_destroy_attempt || !valid_cuda_dnn_batch_norm_descriptors(problem))
        return dnn_gate(rejected(CUDA_ERROR_INVALID_HANDLE));
    if (problem.mode > 3 || problem.operation > 3)
        return dnn_gate(rejected(CUDA_ERROR_INVALID_VALUE));
    const auto tensor = [&](std::size_t index) {
        return static_cast<cudnnTensorDescriptor_t>(problem.descriptors[index].native);
    };
    const auto activation = static_cast<CudaDnnActivation>(problem.descriptors[6].native);
    const auto mode = static_cast<decltype(CUDNN_BATCHNORM_SPATIAL)>(problem.mode);
    const auto operation = static_cast<decltype(CUDNN_BATCHNORM_OPS_BN)>(problem.operation);
    std::size_t bytes{};
    cudnnStatus_t status{};
    // These original storage queries issue no computation and do not acquire
    // the physical-quiescence gate used by handle mutation.
    switch (problem.kind) {
        case CudaDnnBatchNormKind::kForward:
            if (!sdk->batch_norm_forward_workspace)
                return dnn_gate(rejected(CUDA_ERROR_NOT_SUPPORTED));
            status =
                state.invoke(sdk->batch_norm_forward_workspace, handle->native, mode, operation,
                             tensor(0), tensor(3), tensor(1), tensor(5), activation, &bytes);
            break;
        case CudaDnnBatchNormKind::kBackward:
            if (!sdk->batch_norm_backward_workspace)
                return dnn_gate(rejected(CUDA_ERROR_NOT_SUPPORTED));
            status = state.invoke(sdk->batch_norm_backward_workspace, handle->native, mode,
                                  operation, tensor(0), tensor(1), tensor(2), tensor(3), tensor(4),
                                  tensor(5), activation, &bytes);
            break;
        case CudaDnnBatchNormKind::kReserve:
            if (!sdk->batch_norm_reserve_size) return dnn_gate(rejected(CUDA_ERROR_NOT_SUPPORTED));
            status = state.invoke(sdk->batch_norm_reserve_size, handle->native, mode, operation,
                                  activation, tensor(0), &bytes);
            break;
        default:
            return dnn_gate(rejected(CUDA_ERROR_INVALID_VALUE));
    }
    auto result = dnn_completed(status);
    if (state.check_context() != CUDA_SUCCESS) {
        state.fault();
        result.disposition = CudaDnnDisposition::kRejected;
        result.status = CUDA_ERROR_UNKNOWN;
    } else if (status == CUDNN_STATUS_SUCCESS)
        result.value = bytes;
    return result;
}

CudaDnnComputeAcceptance CudaExecutionSession::accept_dnn_batch_norm(
    std::uint64_t sequence, const CudaSessionDnnBatchNorm& problem) noexcept {
    auto& state = *implementation_;
    if (!state.owns() || !state.executor || state.is_faulted ||
        state.executor->snapshot().is_quarantined)
        return {{.status = ExecutionAcceptStatus::kQuarantined, .receipt = {}}, {}};
    if (state.is_closing) return {{.status = ExecutionAcceptStatus::kClosed, .receipt = {}}, {}};
    if (state.check_context() != CUDA_SUCCESS) return {};
    const bool is_forward = problem.kind == CudaDnnBatchNormKind::kForward;
    if (!is_forward && problem.kind != CudaDnnBatchNormKind::kBackward) return {};
    if (problem.mode > 3 || problem.operation > 3) return {};
    const auto& sdk = state.dispatch.dnn;
    const auto handle =
        std::ranges::find(state.storage->dnn, problem.handle, &Implementation::Dnn::id);
    if (!sdk || !sdk->tensor_get_nd || !sdk->tensor_size ||
        (is_forward ? !sdk->batch_norm_forward : !sdk->batch_norm_backward) ||
        handle == state.storage->dnn.end() || handle->is_uncertain || handle->has_destroy_attempt ||
        !valid_cuda_dnn_batch_norm_descriptors(problem))
        return {};
    if (is_forward)
        for (const auto index : {2U, 4U})
            if (problem.buffers[index].allocation || problem.buffers[index].offset) return {};
    const auto stream = get_dnn_stream(problem.handle);
    if (stream.status != CUDA_SUCCESS)
        return {{.status = state.is_faulted ? ExecutionAcceptStatus::kQuarantined
                                            : ExecutionAcceptStatus::kInvalid,
                 .receipt = {}},
                stream.original_status};
    if (stream.original_status != CUDNN_STATUS_SUCCESS) return {{}, stream.original_status};
    std::array<std::size_t, 6> extents{};
    for (std::size_t index = 0; index != extents.size(); ++index) {
        if (!problem.descriptors[index].native) continue;
        const auto descriptor =
            static_cast<cudnnTensorDescriptor_t>(problem.descriptors[index].native);
        cudnnDataType_t type{};
        int rank{};
        std::array<int, CUDNN_DIM_MAX> sizes{}, strides{};
        auto status = state.invoke(sdk->tensor_get_nd, descriptor, CUDNN_DIM_MAX, &type, &rank,
                                   sizes.data(), strides.data());
        if (state.check_context() != CUDA_SUCCESS) {
            state.fault();
            return {{.status = ExecutionAcceptStatus::kQuarantined, .receipt = {}}, status};
        }
        if (status != CUDNN_STATUS_SUCCESS) return {{}, status};
        std::uint32_t type_bits{};
        std::memcpy(&type_bits, &type, sizeof(type_bits));
        if (rank <= 0 || rank > CUDNN_DIM_MAX || type_bits > dnn_profile::data_type_enum_max()) {
            state.fault();
            return {{.status = ExecutionAcceptStatus::kQuarantined, .receipt = {}}, status};
        }
        if (index == 0 && problem.scalar_bytes !=
                              (type_bits == CUDNN_DATA_DOUBLE ? sizeof(double) : sizeof(float)))
            return {};
        status = state.invoke(sdk->tensor_size, descriptor, &extents[index]);
        if (state.check_context() != CUDA_SUCCESS) {
            state.fault();
            return {{.status = ExecutionAcceptStatus::kQuarantined, .receipt = {}}, status};
        }
        if (status != CUDNN_STATUS_SUCCESS) return {{}, status};
        if (!extents[index]) return {};
    }
    std::array<CudaDeviceMemoryView, 13> buffers{};
    for (std::size_t index = 0; index != buffers.size(); ++index) {
        const auto& range = problem.buffers[index];
        const bool is_required =
            index == 0 || index == 5 ||
            (is_forward ? index == 1 || index == 6
                        : index == 2 || index == 4 || index == 7 || index == 8);
        if (!range.allocation) {
            if (range.offset || is_required) return {};
            continue;
        }
        // An ignored optional tensor may lack its descriptor. Still retain its
        // owned address; active tensor shapes and optional pair legality remain
        // original SDK decisions. Parameter tensors have independent extents.
        const auto bytes = index == 11   ? problem.workspace_bytes
                           : index == 12 ? problem.reserve_bytes
                                         : std::max<std::size_t>(1, extents[index < 5 ? index : 5]);
        const auto memory = device_memory(range, bytes);
        if (!memory) return {};
        buffers[index] = *memory;
    }
    try {
        return {state.executor->accept_library(
                    sequence, {.stream = handle->stream,
                               .parameters = {},
                               .call = make_cuda_dnn_batch_norm_call(sdk, handle->native, problem,
                                                                     std::move(buffers))}),
                {}};
    } catch (...) {
        return {{.status = ExecutionAcceptStatus::kFull, .receipt = {}}, {}};
    }
}

CudaDnnConvolutionQueryResult CudaExecutionSession::query_dnn_convolution(
    const CudaSessionDnnConvolution& problem, CudaDnnConvolutionQueryKind kind,
    int requested) noexcept {
    auto& state = *implementation_;
    CudaDnnConvolutionQueryResult output;
    auto reject = [&](CUresult status) {
        output.result = dnn_gate(rejected(status));
        return output;
    };
    const auto quarantine = [&] {
        state.fault();
        output.result.disposition = CudaDnnDisposition::kRejected;
        output.result.status = CUDA_ERROR_UNKNOWN;
        return output;  // Retain the successful native call's original status.
    };
    if (!state.owns() || !state.executor || state.is_faulted) return reject(CUDA_ERROR_UNKNOWN);
    if (state.is_closing) return reject(CUDA_ERROR_DEINITIALIZED);
    if (const auto status = state.check_context(); status != CUDA_SUCCESS) return reject(status);
    const auto& sdk = state.dispatch.dnn;
    const auto handle =
        std::ranges::find(state.storage->dnn, problem.handle, &Implementation::Dnn::id);
    if (!sdk || handle == state.storage->dnn.end() || handle->is_uncertain ||
        handle->has_destroy_attempt)
        return reject(CUDA_ERROR_INVALID_HANDLE);
    // These are original metadata/heuristic queries, not an exhaustive Find
    // benchmark. They do not drain unrelated accepted GPU work.
    const auto finished = [&](cudnnStatus_t status) {
        output.result = dnn_completed(status);
        if (state.check_context() != CUDA_SUCCESS) {
            state.fault();
            output.result.disposition = CudaDnnDisposition::kRejected;
            output.result.status = CUDA_ERROR_UNKNOWN;
        }
        return output.result.status == CUDA_SUCCESS && status == CUDNN_STATUS_SUCCESS;
    };
    if (kind == CudaDnnConvolutionQueryKind::kMaxCount) {
        if (!sdk->convolution_forward_max_count) return reject(CUDA_ERROR_NOT_SUPPORTED);
        int count{};
        if (!finished(state.invoke(sdk->convolution_forward_max_count, handle->native, &count)))
            return output;
        if (count < 0) return quarantine();
        output.result.value = static_cast<std::uint64_t>(count);
        return output;
    }
    if (!valid_cuda_dnn_convolution_descriptors(problem)) return reject(CUDA_ERROR_INVALID_HANDLE);
    const auto x = static_cast<cudnnTensorDescriptor_t>(problem.descriptors[0].native);
    const auto w = static_cast<CudaDnnFilterHandle>(problem.descriptors[1].native);
    const auto convolution = static_cast<CudaDnnConvolution>(problem.descriptors[2].native);
    const auto y = static_cast<cudnnTensorDescriptor_t>(problem.descriptors[3].native);
    if (kind == CudaDnnConvolutionQueryKind::kWorkspace) {
        if (!sdk->convolution_forward_workspace) return reject(CUDA_ERROR_NOT_SUPPORTED);
        constexpr auto algorithm_max =
            std::bit_ceil(static_cast<std::uint32_t>(CUDNN_CONVOLUTION_FWD_ALGO_COUNT) + 1U) - 1U;
        if (problem.algorithm > algorithm_max) return reject(CUDA_ERROR_INVALID_VALUE);
        std::size_t bytes{};
        if (finished(
                state.invoke(sdk->convolution_forward_workspace, handle->native, x, w, convolution,
                             y, static_cast<cudnnConvolutionFwdAlgo_t>(problem.algorithm), &bytes)))
            output.result.value = bytes;
        return output;
    }
    if (kind != CudaDnnConvolutionQueryKind::kHeuristic) return reject(CUDA_ERROR_INVALID_VALUE);
    if (!sdk->convolution_forward_heuristic || !sdk->convolution_forward_max_count)
        return reject(CUDA_ERROR_NOT_SUPPORTED);
    int maximum{};
    if (!finished(state.invoke(sdk->convolution_forward_max_count, handle->native, &maximum)))
        return output;
    if (maximum < 0) return quarantine();
    if (static_cast<std::size_t>(maximum) > kCudaDnnMaxConvolutionAlgorithms)
        return reject(CUDA_ERROR_NOT_SUPPORTED);
    // Preserve zero/negative request errors. Larger capacities need no larger
    // native buffer than the SDK's actual maximum (including math variants).
    const auto capacity = std::min(requested, maximum);
    std::array<cudnnConvolutionFwdAlgoPerfStruct, kCudaDnnMaxConvolutionAlgorithms> native{};
    int count{};
    if (!finished(state.invoke(sdk->convolution_forward_heuristic, handle->native, x, w,
                               convolution, y, capacity, &count, native.data())))
        return output;
    if (count < 0 || count > capacity) return quarantine();
    decltype(output.performance) candidate{};
    for (int index = 0; index < count; ++index) {
        const auto& value = native[static_cast<std::size_t>(index)];
        auto& result = candidate[static_cast<std::size_t>(index)];
        std::memcpy(&result.algorithm, &value.algo, sizeof(result.algorithm));
        std::memcpy(&result.status, &value.status, sizeof(result.status));
        std::memcpy(&result.determinism, &value.determinism, sizeof(result.determinism));
        std::memcpy(&result.math, &value.mathType, sizeof(result.math));
        result.time_bits = std::bit_cast<std::uint32_t>(value.time);
        result.memory = value.memory;
        std::copy(std::begin(value.reserved), std::end(value.reserved), result.reserved.begin());
        if (result.algorithm > 15 || result.status > dnn_profile::status_enum_max() ||
            result.determinism > 1 || result.math > 3)
            return quarantine();
    }
    output.count = static_cast<std::uint32_t>(count);
    output.performance = candidate;
    return output;
}

CudaDnnResult CudaExecutionSession::query_dnn_backward_convolution(
    const CudaSessionDnnConvolution& problem, CudaDnnBackwardConvolutionQueryKind kind,
    bool has_output) noexcept {
    auto& state = *implementation_;
    const auto reject = [](CUresult status) { return dnn_gate(rejected(status)); };
    if (!state.owns() || !state.executor || state.is_faulted ||
        state.executor->snapshot().is_quarantined)
        return reject(CUDA_ERROR_UNKNOWN);
    if (state.is_closing) return reject(CUDA_ERROR_DEINITIALIZED);
    if (kind != CudaDnnBackwardConvolutionQueryKind::kDataMaxCount &&
        kind != CudaDnnBackwardConvolutionQueryKind::kFilterMaxCount &&
        kind != CudaDnnBackwardConvolutionQueryKind::kDataWorkspace)
        return reject(CUDA_ERROR_INVALID_VALUE);
    // The configured native MaxCount implementations write through count
    // without checking it. Reject absence instead of fabricating a temporary
    // output for a call that would otherwise crash.
    if (kind != CudaDnnBackwardConvolutionQueryKind::kDataWorkspace && !has_output)
        return reject(CUDA_ERROR_INVALID_VALUE);
    if (const auto status = state.check_context(); status != CUDA_SUCCESS) return reject(status);
    const auto& sdk = state.dispatch.dnn;
    const auto handle =
        std::ranges::find(state.storage->dnn, problem.handle, &Implementation::Dnn::id);
    if (!sdk || handle == state.storage->dnn.end() || handle->is_uncertain ||
        handle->has_destroy_attempt)
        return reject(CUDA_ERROR_INVALID_HANDLE);

    CudaDnnResult output;
    const auto finished = [&](cudnnStatus_t status) {
        output = dnn_completed(status);
        if (const auto checked = state.check_context(); checked != CUDA_SUCCESS) {
            state.fault();
            output.disposition = CudaDnnDisposition::kRejected;
            output.status = checked;
        }
        return output.status == CUDA_SUCCESS && status == CUDNN_STATUS_SUCCESS;
    };
    // These original host-planning queries do not drain accepted work or
    // record events. Vendor loading/planning may still perform Driver work.
    if (kind != CudaDnnBackwardConvolutionQueryKind::kDataWorkspace) {
        const auto query = kind == CudaDnnBackwardConvolutionQueryKind::kDataMaxCount
                               ? sdk->convolution_backward_data_max_count
                               : sdk->convolution_backward_filter_max_count;
        if (!query) return reject(CUDA_ERROR_NOT_SUPPORTED);
        int count{};
        if (!finished(state.invoke(query, handle->native, &count))) return output;
        if (count < 0) {
            state.fault();
            output.disposition = CudaDnnDisposition::kRejected;
            output.status = CUDA_ERROR_UNKNOWN;
            return output;  // Preserve native SUCCESS but publish no count.
        }
        output.value = static_cast<std::uint64_t>(count);
        return output;
    }
    if (!sdk->convolution_backward_data_workspace) return reject(CUDA_ERROR_NOT_SUPPORTED);
    if (!valid_cuda_dnn_convolution_descriptors(problem)) return reject(CUDA_ERROR_INVALID_HANDLE);
    constexpr auto algorithm_max =
        std::bit_ceil(static_cast<std::uint32_t>(CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT) + 1U) - 1U;
    if (problem.algorithm > algorithm_max) return reject(CUDA_ERROR_INVALID_VALUE);
    const auto dy = static_cast<cudnnTensorDescriptor_t>(problem.descriptors[0].native);
    const auto w = static_cast<CudaDnnFilterHandle>(problem.descriptors[1].native);
    const auto convolution = static_cast<CudaDnnConvolution>(problem.descriptors[2].native);
    const auto dx = static_cast<cudnnTensorDescriptor_t>(problem.descriptors[3].native);
    std::size_t bytes{};
    if (finished(state.invoke(sdk->convolution_backward_data_workspace, handle->native, w, dy,
                              convolution, dx,
                              static_cast<cudnnConvolutionBwdDataAlgo_t>(problem.algorithm),
                              has_output ? &bytes : nullptr)) &&
        has_output)
        output.value = bytes;
    return output;
}

CudaDnnBackendContextResult CudaExecutionSession::backend_context(
    std::uint64_t owner, std::uint64_t id, CudaDnnRegistry& registry,
    const CudaDnnBackendContextRequest& request) noexcept {
    auto& state = *implementation_;
    CudaDnnBackendContextResult output;
    const auto reject = [&](CUresult status) {
        output.result = dnn_gate(rejected(status));
        return output;
    };
    if (!owner || !state.owns() || !state.executor || state.is_faulted ||
        state.executor->snapshot().is_quarantined)
        return reject(CUDA_ERROR_UNKNOWN);
    if (state.is_native_issue_active || state.is_folded_active || state.is_rnn_query_active) {
        output.result.disposition = CudaDnnDisposition::kDeferred;
        output.result.status = CUDA_ERROR_NOT_READY;
        return output;
    }
    if (state.is_closing) return reject(CUDA_ERROR_DEINITIALIZED);
    cudnnHandle_t native{};
    if (id) {
        const auto handle = std::ranges::find(state.storage->dnn, id, &Implementation::Dnn::id);
        if (handle == state.storage->dnn.end() || !handle->native || handle->is_uncertain ||
            handle->has_destroy_attempt)
            return reject(CUDA_ERROR_INVALID_HANDLE);
        native = handle->native;
    }
    struct ActiveCall {
        bool& active;
        ~ActiveCall() {
            active = false;
        }
    };
    if (const auto status = state.domain_cold_status(); status != CUDA_SUCCESS)
        return reject(status);
    state.is_rnn_query_active = true;
    const ActiveCall active{state.is_rnn_query_active};
    const auto check = [](void* opaque) noexcept {
        auto& session = *static_cast<Implementation*>(opaque);
        const auto status = session.check_context(true);
        if (status != CUDA_SUCCESS) session.fault();
        return status;
    };
    if (const auto status = check(&state); status != CUDA_SUCCESS) return reject(status);
    const CudaDnnBackendContext context{
        native,
        owner,
        id,
        state.storage,
        check,
        &state,
        state.device_ordinal,
        [](std::uint64_t handle_id, void* opaque) noexcept -> cudnnHandle_t {
            const auto& session = *static_cast<const Implementation*>(opaque);
            const auto found =
                std::ranges::find(session.storage->dnn, handle_id, &Implementation::Dnn::id);
            return found != session.storage->dnn.end() && !found->is_uncertain &&
                           !found->has_destroy_attempt
                       ? found->native
                       : nullptr;
        }};
    switch (request.operation) {
        case CudaDnnBackendContextOperation::kPrepare:
            output.preparation = registry.prepare_backend_attribute(request.attribute_operation,
                                                                    request.header, &context);
            output.result = output.preparation.result;
            break;
        case CudaDnnBackendContextOperation::kSet:
            output.attribute = registry.set_backend_attribute(request.header, request.generation,
                                                              request.values, &context);
            output.result = output.attribute.result;
            break;
        case CudaDnnBackendContextOperation::kGet:
            output.attribute = registry.get_backend_attribute(request.header, request.generation,
                                                              request.values, &context);
            output.result = output.attribute.result;
            break;
        case CudaDnnBackendContextOperation::kFinalize:
            output.result = registry.finalize_backend(request.id, &context);
            break;
        case CudaDnnBackendContextOperation::kInitialize:
            output.result = registry.initialize_backend(request.id);
            break;
        case CudaDnnBackendContextOperation::kDestroy:
            if (!registry.borrow(request.id, CudaDnnObjectKind::kBackend))
                return reject(CUDA_ERROR_INVALID_HANDLE);
            output.result = registry.destroy(request.id);
            break;
        default:
            return reject(CUDA_ERROR_INVALID_VALUE);
    }
    if (const auto status = check(&state); status != CUDA_SUCCESS) {
        output.result.disposition = CudaDnnDisposition::kRejected;
        output.result.status = status;
    }
    return output;
}
CudaDnnBackendExecutePreparation CudaExecutionSession::prepare_dnn_backend_execute(
    std::uint64_t owner, std::uint64_t handle, CudaDnnRegistry& registry, std::uint64_t plan_id,
    std::uint64_t variant_id) noexcept {
    CudaDnnBackendExecutePreparation output;
    output.result = dnn_gate(rejected(CUDA_ERROR_INVALID_VALUE));
    auto& state = *implementation_;
    if (!state.owns() || !state.executor || state.is_faulted || state.is_closing) return output;
    try {
        const auto plan = registry.borrow_backend(plan_id);
        const auto variant = registry.borrow_backend(variant_id);
        if (!plan || !variant || plan->snapshot->kind != 5 || variant->snapshot->kind != 16 ||
            !plan->snapshot->is_valid || !plan->snapshot->is_finalized ||
            !variant->snapshot->is_valid || !variant->snapshot->is_finalized ||
            !plan->snapshot->context ||
            plan->snapshot->context->device_ordinal != state.device_ordinal ||
            variant->snapshot->variant_uids.size() != variant->snapshot->pointer_slots.size() ||
            variant->snapshot->variant_uids.size() > 100)
            return output;
        const auto tensors = cuda_dnn_backend_tensor_requirements(*plan->snapshot);
        if (tensors.status != CUDA_SUCCESS) {
            output.result.status = tensors.status;
            return output;
        }
        CudaDnnBackendContextRequest request;
        request.operation = CudaDnnBackendContextOperation::kGet;
        request.header = {plan_id, 402, 3, 1, 3};
        request.generation = plan->snapshot->generation;
        const auto workspace = backend_context(owner, handle, registry, request);
        if (workspace.result.disposition == CudaDnnDisposition::kDeferred) {
            output.result = workspace.result;
            return output;
        }
        if (workspace.result.status != CUDA_SUCCESS ||
            workspace.result.original_status != CUDNN_STATUS_SUCCESS ||
            workspace.attribute.values.scalars.size() != sizeof(std::int64_t)) {
            // The private workspace query is never requested Execute provenance.
            output.result.status = workspace.result.status != CUDA_SUCCESS ? workspace.result.status
                                                                           : CUDA_ERROR_UNKNOWN;
            return output;
        }
        std::int64_t bytes{};
        std::memcpy(&bytes, workspace.attribute.values.scalars.data(), sizeof(bytes));
        if (bytes < 0 || static_cast<std::uint64_t>(bytes) > SIZE_MAX) return output;
        output.plan_generation = plan->snapshot->generation;
        output.variant_generation = variant->snapshot->generation;
        for (std::size_t i = 0; i != variant->snapshot->variant_uids.size(); ++i) {
            const auto uid = variant->snapshot->variant_uids[i];
            CudaDnnBackendExecuteRequirement requirement{static_cast<std::uint32_t>(i),
                                                         CudaDnnBackendBindingClass::kIgnored,
                                                         uid,
                                                         variant->snapshot->pointer_slots[i],
                                                         0,
                                                         0};
            const auto tensor =
                std::ranges::find(tensors.tensors, uid, &CudaDnnBackendTensorRequirement::uid);
            if (tensor != tensors.tensors.end()) {
                requirement.value_class = tensor->is_by_value ? CudaDnnBackendBindingClass::kByValue
                                                              : CudaDnnBackendBindingClass::kDevice;
                requirement.bytes = tensor->bytes;
                requirement.alignment = tensor->alignment;
            }
            output.requirements.push_back(requirement);
        }
        for (const auto& tensor : tensors.tensors)
            if (std::ranges::find(variant->snapshot->variant_uids, tensor.uid) ==
                variant->snapshot->variant_uids.end()) {
                output.requirements.clear();
                return output;
            }
        output.requirements.push_back({UINT32_MAX, CudaDnnBackendBindingClass::kDevice, 0,
                                       variant->snapshot->workspace_slot,
                                       static_cast<std::size_t>(bytes), 1});
        output.result.disposition = CudaDnnDisposition::kComplete;
        output.result.status = CUDA_SUCCESS;
    } catch (...) {
        output.result.status = CUDA_ERROR_OUT_OF_MEMORY;
        output.requirements.clear();
    }
    return output;
}
CudaDnnBackendExecuteAcceptance CudaExecutionSession::accept_dnn_backend_execute(
    std::uint64_t sequence, CudaDnnRegistry& registry, std::uint64_t owner,
    const CudaSessionDnnBackendExecute& problem) noexcept {
    auto& state = *implementation_;
    CudaDnnBackendExecuteAcceptance output;
    const auto reject = [&](CUresult status) {
        output.native.rejection_status = status;
        return std::move(output);
    };
    const auto prepared =
        prepare_dnn_backend_execute(owner, problem.handle, registry, problem.plan, problem.variant);
    if (prepared.result.status != CUDA_SUCCESS) return reject(prepared.result.status);
    if (prepared.plan_generation != problem.plan_generation ||
        prepared.variant_generation != problem.variant_generation ||
        prepared.requirements.size() != problem.bindings.size())
        return reject(CUDA_ERROR_INVALID_VALUE);
    try {
        auto plan = registry.borrow_backend_call(problem.plan);
        const auto variant = registry.borrow_backend(problem.variant);
        if (!plan || !variant || plan->descriptor.snapshot->generation != problem.plan_generation ||
            variant->snapshot->generation != problem.variant_generation)
            return reject(CUDA_ERROR_INVALID_VALUE);
        const auto handle =
            std::ranges::find(state.storage->dnn, problem.handle, &Implementation::Dnn::id);
        if (handle == state.storage->dnn.end() || handle->is_uncertain ||
            handle->has_destroy_attempt)
            return reject(CUDA_ERROR_INVALID_HANDLE);
        CudaDnnBackendExecutionArguments arguments;
        arguments.plan = std::move(*plan);
        arguments.uids = variant->snapshot->variant_uids;
        arguments.pointers.resize(arguments.uids.size());
        std::size_t retained{};
        for (std::size_t i = 0; i != prepared.requirements.size(); ++i) {
            const auto& requirement = prepared.requirements[i];
            const auto& binding = problem.bindings[i];
            if (binding.index != requirement.index || binding.slot != requirement.slot ||
                binding.value_class != requirement.value_class)
                return reject(CUDA_ERROR_INVALID_VALUE);
            if (!binding.slot) {
                if (binding.device.allocation || binding.device.offset ||
                    !binding.host_bytes.empty())
                    return reject(CUDA_ERROR_INVALID_VALUE);
                continue;
            }
            if (requirement.value_class == CudaDnnBackendBindingClass::kDevice) {
                if (!binding.host_bytes.empty()) return reject(CUDA_ERROR_INVALID_VALUE);
                const auto memory = device_memory(binding.device, requirement.bytes);
                if (!memory || !memory->address || memory->address % requirement.alignment ||
                    requirement.bytes > std::numeric_limits<CUdeviceptr>::max() - memory->address)
                    return reject(CUDA_ERROR_INVALID_VALUE);
                if (requirement.index == UINT32_MAX)
                    arguments.workspace = *memory;
                else
                    arguments.pointers[requirement.index] = {
                        CudaDnnBackendPointerKind::kDevice, *memory, {}, requirement.alignment};
            } else {
                if (binding.device.allocation || binding.device.offset ||
                    (requirement.value_class == CudaDnnBackendBindingClass::kByValue
                         ? binding.host_bytes.size() != requirement.bytes
                         : !binding.host_bytes.empty()))
                    return reject(CUDA_ERROR_INVALID_VALUE);
                auto& pointer = arguments.pointers[requirement.index];
                pointer.kind = CudaDnnBackendPointerKind::kHostValue;
                pointer.alignment = requirement.alignment ? requirement.alignment : 1;
                pointer.value = requirement.value_class == CudaDnnBackendBindingClass::kByValue
                                    ? binding.host_bytes
                                    : std::vector<std::byte>{std::byte{}};
                // Zero-size live pointers still preserve non-NULL identity.
                if (pointer.value.empty()) pointer.value.push_back(std::byte{});
                if (pointer.alignment > state.limits.submissions.max_parameter_bytes ||
                    pointer.value.size() >
                        state.limits.submissions.max_parameter_bytes - (pointer.alignment - 1))
                    return reject(CUDA_ERROR_OUT_OF_MEMORY);
                const auto bytes = pointer.value.size() + pointer.alignment - 1;
                if (bytes > state.limits.submissions.max_parameter_bytes - retained)
                    return reject(CUDA_ERROR_OUT_OF_MEMORY);
                retained += bytes;
            }
        }
        const auto metadata_call = arguments.plan.call;
        if (!metadata_call->begin_call()) return reject(CUDA_ERROR_NOT_READY);
        struct ActiveCall {
            bool& active;
            CudaDnnRnnLease& metadata;
            ~ActiveCall() {
                active = false;
                metadata.end_call();
            }
        };
        if (const auto status = state.domain_cold_status(); status != CUDA_SUCCESS)
            return reject(status);
        state.is_rnn_query_active = true;
        const ActiveCall active{state.is_rnn_query_active, *metadata_call};
        const auto stream = state.observe_dnn_stream(problem.handle);
        if (stream.status != CUDA_SUCCESS || stream.original_status != CUDNN_STATUS_SUCCESS)
            return reject(stream.status != CUDA_SUCCESS ? stream.status : CUDA_ERROR_UNKNOWN);
        auto result = std::make_shared<CudaDnnBackendExecutionCompletion>();
        auto call = make_cuda_dnn_backend_call(
            state.dispatch.dnn, handle->native, std::move(arguments),
            {state.storage->context, state.dispatch.execution.context_get_current,
             state.dispatch.context_synchronize},
            result);
        // Account retained host snapshots in the submission byte limit too.
        const std::vector<std::byte> charge(retained);
        output.native.acceptance = state.executor->accept_library(
            sequence, {.stream = handle->stream, .parameters = charge, .call = std::move(call)});
        if (output.native.acceptance.accepted()) output.result = std::move(result);
        return output;
    } catch (...) {
        return reject(CUDA_ERROR_OUT_OF_MEMORY);
    }
}
CudaDnnResult CudaExecutionSession::fold_dnn_backward_convolution(
    std::uint64_t id, CudaDnnRegistry& metadata,
    const CudaDnnFoldedDescriptors& descriptors) noexcept {
    auto& state = *implementation_;
    const auto reject = [](CUresult status) { return dnn_gate(rejected(status)); };
    if (!state.owns() || !state.executor || state.is_faulted ||
        state.executor->snapshot().is_quarantined)
        return reject(CUDA_ERROR_UNKNOWN);
    if (state.is_folded_active || state.is_rnn_query_active || state.is_native_issue_active)
        return {.disposition = CudaDnnDisposition::kDeferred, .status = CUDA_ERROR_NOT_READY};
    if (state.is_closing) return reject(CUDA_ERROR_DEINITIALIZED);
    const auto handle = std::ranges::find(state.storage->dnn, id, &Implementation::Dnn::id);
    if (!state.dispatch.dnn || handle == state.storage->dnn.end() || !handle->native ||
        handle->is_uncertain || handle->has_destroy_attempt)
        return reject(CUDA_ERROR_INVALID_HANDLE);
    struct ActiveCall {
        bool& active;
        ~ActiveCall() {
            active = false;
        }
    };
    if (const auto status = state.domain_cold_status(); status != CUDA_SUCCESS)
        return reject(status);
    state.is_folded_active = true;
    const ActiveCall active{state.is_folded_active};
    if (const auto status = state.check_context(true); status != CUDA_SUCCESS)
        return reject(status);
    // The registry holds every unique descriptor owner and excludes new reads
    // through getter, original call and both post-call context checks. Session
    // lifecycle reentry cannot destroy/rebind this handle during those calls.
    const auto result = metadata.folded_backward_data(
        handle->native, descriptors,
        [](const void* opaque) noexcept {
            return static_cast<const Implementation*>(opaque)->check_context(true);
        },
        &state);
    if (result.has_context_loss) state.fault();
    // Metadata-only failure does not erase known handle cleanup ownership.
    // The helper's diagnostic status is not the requested Folded's provenance.
    return result.result;
}

CudaDnnRnnQueryResult CudaExecutionSession::query_dnn_rnn(std::uint64_t id,
                                                          CudaDnnRegistry& metadata,
                                                          const CudaDnnRnnQuery& query) noexcept {
    auto& state = *implementation_;
    const auto reject = [](CUresult status) {
        CudaDnnRnnQueryResult output;
        output.result = dnn_gate(rejected(status));
        if (status == CUDA_ERROR_NOT_READY)
            output.result.disposition = CudaDnnDisposition::kDeferred;
        return output;
    };
    if (!state.owns() || !state.executor || state.is_faulted ||
        state.executor->snapshot().is_quarantined)
        return reject(CUDA_ERROR_UNKNOWN);
    if (state.is_folded_active || state.is_rnn_query_active || state.is_native_issue_active)
        return reject(CUDA_ERROR_NOT_READY);
    if (state.is_closing) return reject(CUDA_ERROR_DEINITIALIZED);
    const auto handle = std::ranges::find(state.storage->dnn, id, &Implementation::Dnn::id);
    if (!state.dispatch.dnn || handle == state.storage->dnn.end() || !handle->native ||
        handle->is_uncertain || handle->has_destroy_attempt)
        return reject(CUDA_ERROR_INVALID_HANDLE);
    struct ActiveCall {
        bool& active;
        ~ActiveCall() {
            active = false;
        }
    };
    if (const auto status = state.domain_cold_status(); status != CUDA_SUCCESS)
        return reject(status);
    state.is_rnn_query_active = true;
    const ActiveCall active{state.is_rnn_query_active};
    if (const auto status = state.check_context(true); status != CUDA_SUCCESS)
        return reject(status);
    // Do not call cold_gate: pending physical work need not finish for these
    // original planning queries. Their vendor loader/device behavior remains
    // context-owned and may block internally, without a framework queue drain.
    auto result = metadata.query_rnn_sizes(
        handle->native, query,
        [](const void* opaque) noexcept {
            return static_cast<const Implementation*>(opaque)->check_context(true);
        },
        &state);
    if (result.has_context_loss) state.fault();
    // Metadata-only uncertainty does not erase proven native handle ownership.
    return std::move(result.result);
}

CudaDnnRnnBuildResult CudaExecutionSession::build_dnn_rnn(std::uint64_t id,
                                                          CudaDnnRegistry& metadata,
                                                          std::uint64_t descriptor,
                                                          int mini_batch) noexcept {
    auto& state = *implementation_;
    const auto reject = [](CUresult status) {
        CudaDnnRnnBuildResult result{dnn_gate(rejected(status))};
        if (status == CUDA_ERROR_NOT_READY)
            result.result.disposition = CudaDnnDisposition::kDeferred;
        return result;
    };
    if (!state.owns() || !state.executor || state.is_faulted ||
        state.executor->snapshot().is_quarantined)
        return reject(CUDA_ERROR_UNKNOWN);
    if (state.is_folded_active || state.is_rnn_query_active || state.is_native_issue_active)
        return reject(CUDA_ERROR_NOT_READY);
    if (state.is_closing) return reject(CUDA_ERROR_DEINITIALIZED);
    const auto handle = std::ranges::find(state.storage->dnn, id, &Implementation::Dnn::id);
    if (!state.dispatch.dnn || handle == state.storage->dnn.end() || !handle->native ||
        handle->is_uncertain || handle->has_destroy_attempt)
        return reject(CUDA_ERROR_INVALID_HANDLE);
    struct ActiveCall {
        bool& active;
        ~ActiveCall() {
            active = false;
        }
    };
    if (const auto status = state.domain_cold_status(); status != CUDA_SUCCESS)
        return reject(status);
    state.is_rnn_query_active = true;
    const ActiveCall active{state.is_rnn_query_active};
    if (const auto status = state.check_context(true); status != CUDA_SUCCESS)
        return reject(status);
    auto result = metadata.build_rnn_dynamic(
        handle->native, descriptor, mini_batch,
        [](const void* opaque) noexcept {
            return static_cast<const Implementation*>(opaque)->check_context(true);
        },
        &state);
    if (result.has_context_loss) state.fault();
    return result;
}

CudaDnnRnnWeightParamsResult CudaExecutionSession::query_dnn_rnn_weight_params(
    std::uint64_t id, CudaDnnRegistry& metadata,
    const CudaSessionDnnRnnWeightParams& problem) noexcept {
    auto& state = *implementation_;
    const auto reject = [](CUresult status) {
        CudaDnnRnnWeightParamsResult output{dnn_gate(rejected(status))};
        if (status == CUDA_ERROR_NOT_READY)
            output.result.disposition = CudaDnnDisposition::kDeferred;
        return output;
    };
    if (!state.owns() || !state.executor || state.is_faulted ||
        state.executor->snapshot().is_quarantined)
        return reject(CUDA_ERROR_UNKNOWN);
    if (state.is_folded_active || state.is_rnn_query_active || state.is_native_issue_active)
        return reject(CUDA_ERROR_NOT_READY);
    if (state.is_closing) return reject(CUDA_ERROR_DEINITIALIZED);
    const auto handle = std::ranges::find(state.storage->dnn, id, &Implementation::Dnn::id);
    if (!state.dispatch.dnn || handle == state.storage->dnn.end() || !handle->native ||
        handle->is_uncertain || handle->has_destroy_attempt)
        return reject(CUDA_ERROR_INVALID_HANDLE);
    struct ActiveCall {
        bool& active;
        ~ActiveCall() {
            active = false;
        }
    };
    if (const auto status = state.domain_cold_status(); status != CUDA_SUCCESS)
        return reject(status);
    state.is_rnn_query_active = true;
    const ActiveCall active{state.is_rnn_query_active};
    if (const auto proof = state.check_context(true); proof != CUDA_SUCCESS) {
        state.fault();
        auto output = reject(proof);
        output.has_context_loss = true;
        return output;
    }
    std::optional<CudaDeviceMemoryView> weights;
    if (problem.weights.allocation) {
        weights = device_memory(problem.weights, problem.weight_bytes);
        if (!weights || !weights->address ||
            problem.weight_bytes > std::numeric_limits<CUdeviceptr>::max() - weights->address)
            return reject(CUDA_ERROR_INVALID_VALUE);
    } else if (problem.weights.offset) {
        return reject(CUDA_ERROR_INVALID_VALUE);
    }
    // This host metadata query neither consumes the weight payload nor drains
    // pending physical readers. The scoped session guard prevents explicit
    // free/context teardown while the complete declared range is observed.
    auto output = metadata.rnn_weight_params(
        handle->native, problem.parameters, weights ? weights->address : 0, problem.weight_bytes,
        [](const void* opaque) noexcept {
            return static_cast<const Implementation*>(opaque)->check_context(true);
        },
        &state);
    if (output.has_context_loss) state.fault();
    return output;
}

CudaDnnAttnWeightsResult CudaExecutionSession::query_dnn_attn_weights(
    std::uint64_t id, CudaDnnRegistry& metadata, const CudaDnnAttnWeights& parameters) noexcept {
    auto& state = *implementation_;
    const auto reject = [](CUresult status) {
        CudaDnnAttnWeightsResult output{dnn_gate(rejected(status))};
        if (status == CUDA_ERROR_NOT_READY)
            output.result.disposition = CudaDnnDisposition::kDeferred;
        return output;
    };
    if (!state.owns() || !state.executor || state.is_faulted ||
        state.executor->snapshot().is_quarantined)
        return reject(CUDA_ERROR_UNKNOWN);
    if (state.is_folded_active || state.is_rnn_query_active || state.is_native_issue_active)
        return reject(CUDA_ERROR_NOT_READY);
    if (state.is_closing) return reject(CUDA_ERROR_DEINITIALIZED);
    const auto handle = std::ranges::find(state.storage->dnn, id, &Implementation::Dnn::id);
    if (!state.dispatch.dnn || handle == state.storage->dnn.end() || !handle->native ||
        handle->is_uncertain || handle->has_destroy_attempt)
        return reject(CUDA_ERROR_INVALID_HANDLE);
    struct ActiveCall {
        bool& active;
        ~ActiveCall() {
            active = false;
        }
    };
    if (const auto status = state.domain_cold_status(); status != CUDA_SUCCESS)
        return reject(status);
    state.is_rnn_query_active = true;
    const ActiveCall active{state.is_rnn_query_active};
    if (const auto proof = state.check_context(true); proof != CUDA_SUCCESS) {
        state.fault();
        auto output = reject(proof);
        output.has_context_loss = true;
        return output;
    }
    // The original only calculates addresses. Neither GPU allocation ownership
    // nor RNG-state resolution is required, and pending work is not drained.
    auto output = metadata.attn_weights(
        handle->native, parameters,
        [](const void* opaque) noexcept {
            return static_cast<const Implementation*>(opaque)->check_context(true);
        },
        &state);
    if (output.has_context_loss) state.fault();
    return output;
}

CudaExecutionSession::Implementation::AttnOperation
CudaExecutionSession::Implementation::observe_attn(CudaExecutionSession& session,
                                                   const CudaSessionDnnAttn& problem,
                                                   bool is_preparation,
                                                   std::uint64_t sequence) noexcept {
    AttnOperation output;
    const auto reject = [&](CUresult status,
                            ExecutionAcceptStatus disposition = ExecutionAcceptStatus::kInvalid) {
        output.preparation.result.disposition = CudaDnnDisposition::kRejected;
        output.preparation.result.status = status;
        output.preparation.first = output.preparation.count = 0;
        output.acceptance.native.rejection_status = status;
        output.acceptance.native.acceptance.status = disposition;
        return std::move(output);
    };
    if (!owns() || !executor || is_faulted || executor->snapshot().is_quarantined)
        return reject(CUDA_ERROR_UNKNOWN, ExecutionAcceptStatus::kQuarantined);
    if (is_closing) return reject(CUDA_ERROR_DEINITIALIZED, ExecutionAcceptStatus::kClosed);
    if (is_folded_active || is_rnn_query_active || is_native_issue_active)
        return reject(CUDA_ERROR_NOT_READY, ExecutionAcceptStatus::kFull);
    if (!valid_cuda_dnn_attn_descriptors(problem) ||
        (is_preparation && (!problem.windows[0].empty() || !problem.windows[1].empty())))
        return reject(CUDA_ERROR_INVALID_VALUE);
    const auto& sdk = dispatch.dnn;
    const auto handle = std::ranges::find(storage->dnn, problem.handle, &Dnn::id);
    if (!sdk || handle == storage->dnn.end() || !handle->native || handle->is_uncertain ||
        handle->has_destroy_attempt)
        return reject(CUDA_ERROR_INVALID_HANDLE);
    const bool forward = problem.kind == CudaDnnAttnComputeKind::kForward;
    const bool data = problem.kind == CudaDnnAttnComputeKind::kBackwardData;
    const bool weights = problem.kind == CudaDnnAttnComputeKind::kBackwardWeights;
    if (!sdk->version || !sdk->get_stream || !dispatch.context_synchronize ||
        (forward && !sdk->attn_forward) || (data && !sdk->attn_backward_data) ||
        (weights && !sdk->attn_backward_weights))
        return reject(CUDA_ERROR_NOT_SUPPORTED);
    if (const auto status = domain_cold_status(); status != CUDA_SUCCESS)
        return reject(status, status == CUDA_ERROR_NOT_READY ? ExecutionAcceptStatus::kFull
                                                             : ExecutionAcceptStatus::kQuarantined);
    if (!problem.attn.lease->begin_call())
        return reject(CUDA_ERROR_NOT_READY, ExecutionAcceptStatus::kFull);
    struct ActiveCall {
        bool& active;
        CudaDnnRnnLease& lease;
        ~ActiveCall() {
            active = false;
            lease.end_call();
        }
    };
    is_rnn_query_active = true;
    const ActiveCall active{is_rnn_query_active, *problem.attn.lease};
    const auto context = [&]() {
        const auto status = check_context(true);
        if (status != CUDA_SUCCESS) {
            fault();
            output.preparation.has_context_loss = true;
        }
        return status;
    };
    const auto observed_status = [&](cudnnStatus_t status) {
        output.preparation.preflight_status = status;
        output.acceptance.preflight_status = status;
        const auto proof = context();
        return proof != CUDA_SUCCESS            ? proof
               : status == CUDNN_STATUS_SUCCESS ? CUDA_SUCCESS
                                                : CUDA_ERROR_UNKNOWN;
    };
    const auto malformed = [&]() {
        problem.attn.lease->quarantine();
        return CUDA_ERROR_UNKNOWN;
    };
    if (const auto proof = context(); proof != CUDA_SUCCESS)
        return reject(proof, ExecutionAcceptStatus::kQuarantined);
    const auto version = invoke(sdk->version);
    if (const auto proof = context(); proof != CUDA_SUCCESS)
        return reject(proof, ExecutionAcceptStatus::kQuarantined);
    if (version != 91002) return reject(CUDA_ERROR_NOT_SUPPORTED);
    const auto stream = observe_dnn_stream(problem.handle);
    if (stream.status != CUDA_SUCCESS || stream.original_status != CUDNN_STATUS_SUCCESS) {
        output.preparation.preflight_status = output.acceptance.preflight_status =
            stream.original_status;
        if (const auto proof = context(); proof != CUDA_SUCCESS)
            return reject(proof, ExecutionAcceptStatus::kQuarantined);
        output.preparation.has_context_loss = is_faulted;
        return reject(
            stream.status == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : stream.status,
            is_faulted ? ExecutionAcceptStatus::kQuarantined : ExecutionAcceptStatus::kInvalid);
    }
    const auto present = [&](std::size_t index) { return problem.buffers[index].allocation != 0; };
    bool is_early_error = std::ranges::any_of(
        problem.descriptors, [](const auto& descriptor) { return !descriptor.native; });
    is_early_error |= !present(2) || !present(3) || !present(4) || !present(forward ? 6 : 7);
    is_early_error |= (problem.weight_bytes && !present(11)) ||
                      (problem.workspace_bytes && !present(13)) ||
                      (problem.reserve_bytes && !present(14));
    if (!weights) is_early_error |= problem.window_presence != 3 || !present(0) || !present(1);
    if (forward)
        is_early_error |= problem.current >= 0 && present(14);
    else {
        is_early_error |= !present(14) || !problem.reserve_bytes;
        if (data) is_early_error |= !present(8) || !present(9) || !present(10);
        if (weights) is_early_error |= problem.weight_bytes && !present(12);
    }
    const auto multiply = [](std::size_t& product, std::size_t value) {
        if (value && product > std::numeric_limits<std::size_t>::max() / value) return false;
        product *= value;
        return true;
    };
    const auto width = [](std::uint32_t type) -> std::size_t {
        switch (type) {
            case 0:
                return 4;
            case 1:
                return 8;
            case 2:
            case 9:
                return 2;
            default:
                return 0;
        }
    };
    struct SeqObservation {
        std::uint32_t type{};
        std::array<int, 4> sizes{};
        std::array<std::uint32_t, 4> axes{};
        std::size_t count{}, bytes{};
    };
    std::array<SeqObservation, 4> observed{};
    const auto observe_sequence = [&](std::size_t index, SeqObservation& value,
                                      std::size_t capacity, int* lengths) -> CUresult {
        cudnnDataType_t type{};
        int rank{};
        std::array<cudnnSeqDataAxis_t, 4> axes{};
        const auto status =
            invoke(sdk->seq_data_get,
                   static_cast<CudaDnnSeqData>(problem.descriptors[index].native), &type, &rank, 4,
                   value.sizes.data(), axes.data(), &value.count, capacity, lengths, nullptr);
        if (const auto proof = observed_status(status); proof != CUDA_SUCCESS) return proof;
        std::memcpy(&value.type, &type, sizeof(value.type));
        for (std::size_t axis = 0; axis < axes.size(); ++axis)
            std::memcpy(&value.axes[axis], &axes[axis], sizeof(value.axes[axis]));
        if (rank != 4 || value.type > dnn_profile::data_type_enum_max()) return malformed();
        // Fresh descriptors have rank four but zero dimensions/axes/count.
        // This is an unusable extent, not proof of metadata corruption.
        if (std::ranges::any_of(value.sizes, [](int size) { return size <= 0; }))
            return CUDA_ERROR_INVALID_VALUE;
        if (!width(value.type)) return CUDA_ERROR_NOT_SUPPORTED;
        auto sorted = value.axes;
        std::sort(sorted.begin(), sorted.begin() + 3);
        if (sorted != std::array<std::uint32_t, 4>{0, 1, 2, 3}) return malformed();
        auto count = static_cast<std::size_t>(value.sizes[1]);
        if (!multiply(count, static_cast<std::size_t>(value.sizes[2])) ||
            count > kCudaDnnMaxSequenceLengths)
            return CUDA_ERROR_INVALID_VALUE;
        if (value.count != count) return malformed();
        return CUDA_SUCCESS;
    };
    std::array<std::size_t, 15> sizes{};
    sizes.fill(1);
    sizes[11] = sizes[12] = problem.weight_bytes;
    sizes[13] = problem.workspace_bytes;
    sizes[14] = problem.reserve_bytes;
    bool is_noop{};
    try {
        if (!is_early_error) {
            if (!sdk->seq_data_get) return reject(CUDA_ERROR_NOT_SUPPORTED);
            if (const auto status = observe_sequence(0, observed[0], 0, nullptr);
                status != CUDA_SUCCESS)
                return reject(status, is_faulted ? ExecutionAcceptStatus::kQuarantined
                                                 : ExecutionAcceptStatus::kInvalid);
            // This original check precedes every host-window payload access.
            is_early_error = forward && problem.current >= observed[0].sizes[0];
        }
        if (!is_early_error) {
            if (!sdk->attn_get) return reject(CUDA_ERROR_NOT_SUPPORTED);
            cudnnDataType_t type{};
            int maximum_query{};
            const auto native =
                invoke(sdk->attn_get, static_cast<CudaDnnAttn>(problem.attn.object.native), nullptr,
                       nullptr, nullptr, &type, nullptr, nullptr, nullptr, nullptr, nullptr,
                       nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &maximum_query,
                       nullptr, nullptr, nullptr);
            if (const auto status = observed_status(native); status != CUDA_SUCCESS)
                return reject(status, is_faulted ? ExecutionAcceptStatus::kQuarantined
                                                 : ExecutionAcceptStatus::kInvalid);
            std::uint32_t type_bits{};
            std::memcpy(&type_bits, &type, sizeof(type_bits));
            if (type_bits > dnn_profile::data_type_enum_max()) return reject(malformed());
            const auto parent_width = width(type_bits);
            if (!parent_width) return reject(CUDA_ERROR_NOT_SUPPORTED);
            for (std::size_t index = 1; index < observed.size(); ++index)
                if (const auto status = observe_sequence(index, observed[index], 0, nullptr);
                    status != CUDA_SUCCESS)
                    return reject(status, is_faulted ? ExecutionAcceptStatus::kQuarantined
                                                     : ExecutionAcceptStatus::kInvalid);
            for (auto& value : observed) {
                value.bytes = std::max(parent_width, width(value.type));
                for (const int size : value.sizes)
                    if (!multiply(value.bytes, static_cast<std::size_t>(size)))
                        return reject(CUDA_ERROR_INVALID_VALUE);
            }
            sizes[0] = observed[0].count * sizeof(int);
            sizes[1] = observed[1].count * sizeof(int);
            sizes[2] = sizes[8] = observed[0].bytes;
            sizes[3] = sizes[9] = observed[1].bytes;
            sizes[4] = sizes[10] = observed[2].bytes;
            sizes[6] = sizes[7] = observed[3].bytes;
            sizes[5] = std::max(observed[0].bytes, observed[3].bytes);
            if (!weights) {
                if (forward && problem.current >= 0) {
                    std::vector<int> lengths(observed[0].count);
                    SeqObservation again{};
                    if (const auto status =
                            observe_sequence(0, again, lengths.size(), lengths.data());
                        status != CUDA_SUCCESS)
                        return reject(status, is_faulted ? ExecutionAcceptStatus::kQuarantined
                                                         : ExecutionAcceptStatus::kInvalid);
                    if (again.type != observed[0].type || again.sizes != observed[0].sizes ||
                        again.axes != observed[0].axes || again.count != observed[0].count)
                        return reject(malformed());
                    int maximum_length{};
                    for (const int length : lengths) {
                        if (length < 0 || length > observed[0].sizes[0]) return reject(malformed());
                        maximum_length = std::max(maximum_length, length);
                    }
                    is_noop = problem.current >= maximum_length;
                    if (!is_noop) {
                        output.preparation.first = static_cast<std::uint32_t>(problem.current);
                        output.preparation.count = 1;
                    }
                } else {
                    if (maximum_query <= 0 ||
                        static_cast<std::size_t>(maximum_query) > kCudaDnnMaxSequenceLengths)
                        return reject(CUDA_ERROR_INVALID_VALUE);
                    output.preparation.count = static_cast<std::uint32_t>(maximum_query);
                }
            }
        }
        const auto first = static_cast<std::size_t>(output.preparation.first);
        const auto count = static_cast<std::size_t>(output.preparation.count);
        if (first > kCudaDnnMaxSequenceLengths || count > kCudaDnnMaxSequenceLengths - first)
            return reject(CUDA_ERROR_INVALID_VALUE);
        if (!is_preparation)
            for (std::size_t index = 0; index < problem.windows.size(); ++index)
                if (problem.windows[index].size() !=
                    ((problem.window_presence & (1U << index)) ? count : 0))
                    return reject(CUDA_ERROR_INVALID_VALUE);
        std::array<CudaDeviceMemoryView, 15> buffers{};
        for (std::size_t index = 0; index < buffers.size(); ++index) {
            if (!present(index)) continue;
            const auto memory = session.device_memory(problem.buffers[index], sizes[index]);
            if (!memory || !memory->address ||
                sizes[index] > std::numeric_limits<CUdeviceptr>::max() - memory->address)
                return reject(CUDA_ERROR_INVALID_VALUE);
            buffers[index] = *memory;
        }
        std::array<CudaDeviceMemoryView, 4> dropout_states{};
        if (forward && present(14) && !is_early_error && !is_noop)
            for (std::size_t index = 0; index < dropout_states.size(); ++index) {
                const auto& dropout = problem.attn.dropouts[index];
                if (!dropout.object.native) continue;
                const auto memory = session.resolve_dnn_dropout(dropout);
                if (!memory) return reject(CUDA_ERROR_INVALID_VALUE);
                dropout_states[index] = *memory;
            }
        if (const auto proof = context(); proof != CUDA_SUCCESS)
            return reject(proof, ExecutionAcceptStatus::kQuarantined);
        output.preparation.result.disposition = CudaDnnDisposition::kComplete;
        output.preparation.result.status = CUDA_SUCCESS;
        if (is_preparation) return output;
        auto result = std::make_shared<CudaDnnAttnCompletion>();
        auto call =
            make_cuda_dnn_attn_call(sdk, handle->native, problem, std::move(buffers),
                                    {storage->context, dispatch.execution.context_get_current,
                                     dispatch.context_synchronize},
                                    result, std::move(dropout_states), output.preparation.first);
        output.acceptance.native.acceptance = executor->accept_library(
            sequence, {.stream = handle->stream, .parameters = {}, .call = std::move(call)});
        if (output.acceptance.native.acceptance.accepted())
            output.acceptance.result = std::move(result);
        return output;
    } catch (...) {
        return reject(CUDA_ERROR_OUT_OF_MEMORY, ExecutionAcceptStatus::kFull);
    }
}

CudaDnnAttnPreparation CudaExecutionSession::prepare_dnn_attn(
    const CudaSessionDnnAttn& problem) noexcept {
    return implementation_->observe_attn(*this, problem, true, 0).preparation;
}
CudaDnnAttnAcceptance CudaExecutionSession::accept_dnn_attn(
    std::uint64_t sequence, const CudaSessionDnnAttn& problem) noexcept {
    return implementation_->observe_attn(*this, problem, false, sequence).acceptance;
}

CudaDnnRnnAcceptance CudaExecutionSession::accept_dnn_rnn(
    std::uint64_t sequence, const CudaSessionDnnRnn& problem) noexcept {
    auto& state = *implementation_;
    CudaDnnRnnAcceptance output;
    const auto reject = [&](CUresult status,
                            ExecutionAcceptStatus disposition = ExecutionAcceptStatus::kInvalid) {
        output.native.rejection_status = status;
        output.native.acceptance.status = disposition;
        return output;
    };
    if (!state.owns() || !state.executor || state.is_faulted ||
        state.executor->snapshot().is_quarantined)
        return reject(CUDA_ERROR_UNKNOWN, ExecutionAcceptStatus::kQuarantined);
    if (state.is_closing) return reject(CUDA_ERROR_DEINITIALIZED, ExecutionAcceptStatus::kClosed);
    if (state.is_folded_active || state.is_rnn_query_active || state.is_native_issue_active)
        return reject(CUDA_ERROR_NOT_READY, ExecutionAcceptStatus::kFull);
    if (!valid_cuda_dnn_rnn_descriptors(problem)) return reject(CUDA_ERROR_INVALID_VALUE);
    const auto& sdk = state.dispatch.dnn;
    const auto handle =
        std::ranges::find(state.storage->dnn, problem.handle, &Implementation::Dnn::id);
    if (!sdk || handle == state.storage->dnn.end() || !handle->native || handle->is_uncertain ||
        handle->has_destroy_attempt)
        return reject(CUDA_ERROR_INVALID_HANDLE);
    if (!sdk->get_stream || !sdk->rnn_data_get || !state.dispatch.context_synchronize ||
        (problem.kind == CudaDnnRnnComputeKind::kForward && !sdk->rnn_forward) ||
        (problem.kind == CudaDnnRnnComputeKind::kBackwardData && !sdk->rnn_backward_data) ||
        (problem.kind == CudaDnnRnnComputeKind::kBackwardWeights && !sdk->rnn_backward_weights))
        return reject(CUDA_ERROR_NOT_SUPPORTED);
    if (!problem.rnn.lease->begin_call())
        return reject(CUDA_ERROR_NOT_READY, ExecutionAcceptStatus::kFull);
    struct ActiveCall {
        bool& active;
        CudaDnnRnnLease& lease;
        ~ActiveCall() {
            active = false;
            lease.end_call();
        }
    };
    if (const auto status = state.domain_cold_status(); status != CUDA_SUCCESS)
        return reject(status);
    state.is_rnn_query_active = true;
    const ActiveCall active{state.is_rnn_query_active, *problem.rnn.lease};
    const auto context = [&]() {
        const auto status = state.check_context(true);
        if (status != CUDA_SUCCESS) state.fault();
        return status;
    };
    if (const auto proof = context(); proof != CUDA_SUCCESS)
        return reject(proof, ExecutionAcceptStatus::kQuarantined);
    const auto stream = state.observe_dnn_stream(problem.handle);
    if (stream.status != CUDA_SUCCESS || stream.original_status != CUDNN_STATUS_SUCCESS) {
        output.preflight_status = stream.original_status;
        if (const auto proof = context(); proof != CUDA_SUCCESS)
            return reject(proof, ExecutionAcceptStatus::kQuarantined);
        return reject(stream.status == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : stream.status,
                      state.is_faulted ? ExecutionAcceptStatus::kQuarantined
                                       : ExecutionAcceptStatus::kInvalid);
    }
    const auto malformed = [&]() {
        problem.rnn.lease->quarantine();
        return CUDA_ERROR_UNKNOWN;
    };
    const auto multiply = [](std::size_t& product, std::size_t value) {
        if (value && product > std::numeric_limits<std::size_t>::max() / value) return false;
        product *= value;
        return true;
    };
    struct DataExtent {
        std::size_t bytes{}, batch{}, width{};
    };
    std::array<DataExtent, 2> data{};
    const auto data_extent = [&](std::size_t index) -> CUresult {
        cudnnDataType_t type{};
        cudnnRNNDataLayout_t layout{};
        int maximum{}, batch{}, vector{};
        alignas(double) std::array<std::byte, 8> padding{};
        const auto native = static_cast<CudaDnnRnnData>(problem.descriptors[index].native);
        auto status = state.invoke(sdk->rnn_data_get, native, &type, &layout, &maximum, &batch,
                                   &vector, 0, nullptr, padding.data());
        output.preflight_status = status;
        if (const auto proof = context(); proof != CUDA_SUCCESS) return proof;
        if (status != CUDNN_STATUS_SUCCESS) return CUDA_ERROR_UNKNOWN;
        std::uint32_t type_bits{}, layout_bits{};
        std::memcpy(&type_bits, &type, sizeof(type_bits));
        std::memcpy(&layout_bits, &layout, sizeof(layout_bits));
        if (type_bits > dnn_profile::data_type_enum_max() || layout_bits > 3) return malformed();
        if (type_bits > 2 || layout_bits > 2) return CUDA_ERROR_NOT_SUPPORTED;
        if (maximum <= 0 || batch <= 0 || vector <= 0 ||
            static_cast<std::size_t>(batch) > kCudaDnnMaxSequenceLengths)
            return CUDA_ERROR_INVALID_VALUE;
        std::size_t elements = static_cast<std::size_t>(maximum);
        if (!multiply(elements, static_cast<std::size_t>(batch))) return CUDA_ERROR_INVALID_VALUE;
        if (layout_bits == 1) {
            std::vector<int> lengths(static_cast<std::size_t>(batch));
            cudnnDataType_t again_type{};
            cudnnRNNDataLayout_t again_layout{};
            int again_maximum{}, again_batch{}, again_vector{};
            status =
                state.invoke(sdk->rnn_data_get, native, &again_type, &again_layout, &again_maximum,
                             &again_batch, &again_vector, batch, lengths.data(), padding.data());
            output.preflight_status = status;
            if (const auto proof = context(); proof != CUDA_SUCCESS) return proof;
            if (status != CUDNN_STATUS_SUCCESS) return CUDA_ERROR_UNKNOWN;
            std::uint32_t again_type_bits{}, again_layout_bits{};
            std::memcpy(&again_type_bits, &again_type, sizeof(again_type_bits));
            std::memcpy(&again_layout_bits, &again_layout, sizeof(again_layout_bits));
            if (again_type_bits != type_bits || again_layout_bits != layout_bits ||
                again_maximum != maximum || again_batch != batch || again_vector != vector)
                return malformed();
            elements = 0;
            for (const int length : lengths) {
                if (length < 0 || length > maximum) return malformed();
                const auto value = static_cast<std::size_t>(length);
                if (elements > std::numeric_limits<std::size_t>::max() - value)
                    return CUDA_ERROR_INVALID_VALUE;
                elements += value;
            }
        }
        constexpr std::array<std::size_t, 3> widths{4, 8, 2};
        if (!multiply(elements, static_cast<std::size_t>(vector)) ||
            !multiply(elements, widths[type_bits]))
            return CUDA_ERROR_INVALID_VALUE;
        data[index] = {elements, static_cast<std::size_t>(batch), widths[type_bits]};
        return CUDA_SUCCESS;
    };
    try {
        for (std::size_t index = 0; index < data.size(); ++index)
            if (const auto status = data_extent(index); status != CUDA_SUCCESS)
                return reject(status, state.is_faulted ? ExecutionAcceptStatus::kQuarantined
                                                       : ExecutionAcceptStatus::kInvalid);
        std::array<std::size_t, 2> state_bytes{};
        for (std::size_t index = 0; index < state_bytes.size(); ++index) {
            const auto begin = index == 0 ? std::size_t{5} : std::size_t{9};
            bool needed{};
            for (auto slot = begin; slot < begin + 4; ++slot)
                needed |= problem.buffers[slot].allocation != 0;
            if (!needed) continue;
            if (!sdk->tensor_get_nd || !sdk->tensor_size) return reject(CUDA_ERROR_NOT_SUPPORTED);
            const auto descriptor =
                static_cast<cudnnTensorDescriptor_t>(problem.descriptors[index + 2].native);
            cudnnDataType_t type{};
            int rank{};
            std::array<int, CUDNN_DIM_MAX> sizes{}, strides{};
            auto status = state.invoke(sdk->tensor_get_nd, descriptor, CUDNN_DIM_MAX, &type, &rank,
                                       sizes.data(), strides.data());
            output.preflight_status = status;
            if (const auto proof = context(); proof != CUDA_SUCCESS)
                return reject(proof, ExecutionAcceptStatus::kQuarantined);
            if (status != CUDNN_STATUS_SUCCESS) return reject(CUDA_ERROR_UNKNOWN);
            std::uint32_t type_bits{};
            std::memcpy(&type_bits, &type, sizeof(type_bits));
            if (rank < 0 || rank > CUDNN_DIM_MAX || type_bits > dnn_profile::data_type_enum_max())
                return reject(malformed());
            // Native RNN reads three dimension slots; smaller ranks can hide
            // historical tail dimensions that the public getter cannot prove.
            if (rank < 3) return reject(CUDA_ERROR_INVALID_VALUE);
            if (type_bits > 2) return reject(CUDA_ERROR_NOT_SUPPORTED);
            std::size_t hull = 1;
            std::size_t dense = 1;
            for (int dim = 0; dim < rank; ++dim) {
                if (sizes[dim] <= 0 || strides[dim] <= 0) return reject(CUDA_ERROR_INVALID_VALUE);
                auto offset = static_cast<std::size_t>(sizes[dim] - 1);
                if (!multiply(dense, static_cast<std::size_t>(sizes[dim])) ||
                    !multiply(offset, static_cast<std::size_t>(strides[dim])) ||
                    hull > std::numeric_limits<std::size_t>::max() - offset)
                    return reject(CUDA_ERROR_INVALID_VALUE);
                hull += offset;
            }
            constexpr std::array<std::size_t, 3> widths{4, 8, 2};
            // Native packed checks use narrow arithmetic, and state accesses
            // follow RNNData type dispatch rather than every state descriptor.
            const auto compute_width = std::max({widths[type_bits], data[0].width, data[1].width});
            if (!multiply(hull, widths[type_bits]) || !multiply(dense, compute_width))
                return reject(CUDA_ERROR_INVALID_VALUE);
            status = state.invoke(sdk->tensor_size, descriptor, &state_bytes[index]);
            output.preflight_status = status;
            if (const auto proof = context(); proof != CUDA_SUCCESS)
                return reject(proof, ExecutionAcceptStatus::kQuarantined);
            if (status != CUDNN_STATUS_SUCCESS) return reject(CUDA_ERROR_UNKNOWN);
            if (state_bytes[index] < hull) return reject(malformed());
            state_bytes[index] = std::max(state_bytes[index], dense);
        }
        std::array<std::size_t, 16> sizes{};
        sizes[0] = std::max(data[0].batch, data[1].batch) * sizeof(std::int32_t);
        sizes[1] = sizes[4] = data[0].bytes;
        sizes[2] = sizes[3] = data[1].bytes;
        std::fill(sizes.begin() + 5, sizes.begin() + 9, state_bytes[0]);
        std::fill(sizes.begin() + 9, sizes.begin() + 13, state_bytes[1]);
        sizes[13] = problem.weight_bytes;
        sizes[14] = problem.workspace_bytes;
        sizes[15] = problem.reserve_bytes;
        std::array<CudaDeviceMemoryView, 16> buffers{};
        for (std::size_t index = 0; index < buffers.size(); ++index) {
            if (!problem.buffers[index].allocation) continue;
            const auto memory = device_memory(problem.buffers[index], sizes[index]);
            if (!memory) return reject(CUDA_ERROR_INVALID_VALUE);
            buffers[index] = *memory;
        }
        std::array<CudaDeviceMemoryView, 4> dropout_states{};
        for (std::size_t index = 0; index < dropout_states.size(); ++index) {
            const auto& dropout = problem.rnn.dropouts[index];
            if (!dropout.object.native) continue;
            const auto memory = resolve_dnn_dropout(dropout);
            if (!memory) return reject(CUDA_ERROR_INVALID_VALUE);
            dropout_states[index] = *memory;
        }
        if (const auto proof = context(); proof != CUDA_SUCCESS)
            return reject(proof, ExecutionAcceptStatus::kQuarantined);
        auto result = std::make_shared<CudaDnnRnnCompletion>();
        auto call = make_cuda_dnn_rnn_call(
            sdk, handle->native, problem, std::move(buffers),
            {state.storage->context, state.dispatch.execution.context_get_current,
             state.dispatch.context_synchronize},
            result, std::move(dropout_states));
        output.native.acceptance = state.executor->accept_library(
            sequence, {.stream = handle->stream, .parameters = {}, .call = std::move(call)});
        if (output.native.acceptance.accepted()) output.result = std::move(result);
        return output;
    } catch (...) {
        return reject(CUDA_ERROR_OUT_OF_MEMORY, ExecutionAcceptStatus::kFull);
    }
}

CudaDnnFindAcceptance CudaExecutionSession::accept_dnn_find(
    std::uint64_t sequence, const CudaSessionDnnConvolution& problem, int requested) noexcept {
    auto& state = *implementation_;
    CudaDnnFindAcceptance output;
    if (!state.owns() || !state.executor || state.is_faulted ||
        state.executor->snapshot().is_quarantined) {
        output.native.acceptance.status = ExecutionAcceptStatus::kQuarantined;
        return output;
    }
    if (state.is_closing) {
        output.native.acceptance.status = ExecutionAcceptStatus::kClosed;
        return output;
    }
    if (state.check_context() != CUDA_SUCCESS) return output;
    const auto& sdk = state.dispatch.dnn;
    const auto handle =
        std::ranges::find(state.storage->dnn, problem.handle, &Implementation::Dnn::id);
    if (!sdk || !sdk->convolution_forward_find || !state.dispatch.context_synchronize ||
        handle == state.storage->dnn.end() || handle->is_uncertain || handle->has_destroy_attempt ||
        !valid_cuda_dnn_convolution_descriptors(problem))
        return output;
    const auto stream = get_dnn_stream(problem.handle);
    if (stream.status != CUDA_SUCCESS || stream.original_status != CUDNN_STATUS_SUCCESS) {
        output.native.original_status = stream.original_status;
        if (state.is_faulted) output.native.acceptance.status = ExecutionAcceptStatus::kQuarantined;
        return output;
    }
    // The SDK maximum includes math variants, not just the algorithm enums.
    // This metadata query does not execute/replay the benchmark itself.
    const auto maximum = query_dnn_convolution(problem, CudaDnnConvolutionQueryKind::kMaxCount);
    if (maximum.result.status != CUDA_SUCCESS ||
        maximum.result.original_status != CUDNN_STATUS_SUCCESS) {
        output.native.original_status = maximum.result.original_status;
        if (state.is_faulted) output.native.acceptance.status = ExecutionAcceptStatus::kQuarantined;
        return output;
    }
    if (maximum.result.value > kCudaDnnMaxConvolutionAlgorithms) return output;
    const auto capacity = std::min(requested, static_cast<int>(maximum.result.value));
    try {
        auto result = std::make_shared<CudaDnnConvolutionFindResult>();
        auto call = make_cuda_dnn_convolution_find_call(
            sdk, handle->native, problem, capacity,
            {state.storage->context, state.dispatch.execution.context_get_current,
             state.dispatch.context_synchronize},
            result);
        output.native.acceptance = state.executor->accept_library(
            sequence, {.stream = handle->stream, .parameters = {}, .call = std::move(call)});
        if (output.native.acceptance.accepted()) output.result = std::move(result);
        return output;
    } catch (...) {
        output.native.acceptance.status = ExecutionAcceptStatus::kFull;
        return output;
    }
}

CudaDnnComputeAcceptance CudaExecutionSession::accept_dnn_convolution(
    std::uint64_t sequence, const CudaSessionDnnConvolution& problem) noexcept {
    auto& state = *implementation_;
    if (!state.owns() || !state.executor || state.is_faulted)
        return {{.status = ExecutionAcceptStatus::kQuarantined, .receipt = {}}, {}};
    if (state.is_closing) return {{.status = ExecutionAcceptStatus::kClosed, .receipt = {}}, {}};
    if (state.check_context() != CUDA_SUCCESS) return {};
    const auto& sdk = state.dispatch.dnn;
    const auto handle =
        std::ranges::find(state.storage->dnn, problem.handle, &Implementation::Dnn::id);
    constexpr auto algorithm_max =
        std::bit_ceil(static_cast<std::uint32_t>(CUDNN_CONVOLUTION_FWD_ALGO_COUNT) + 1U) - 1U;
    if (!sdk || !sdk->convolution_forward || !sdk->filter_size || !sdk->tensor_get_nd ||
        !sdk->tensor_size || handle == state.storage->dnn.end() || handle->is_uncertain ||
        handle->has_destroy_attempt || !valid_cuda_dnn_convolution_descriptors(problem) ||
        problem.algorithm > algorithm_max)
        return {};
    const auto stream = get_dnn_stream(problem.handle);
    if (stream.status != CUDA_SUCCESS)
        return {{.status = state.is_faulted ? ExecutionAcceptStatus::kQuarantined
                                            : ExecutionAcceptStatus::kInvalid,
                 .receipt = {}},
                stream.original_status};
    if (stream.original_status != CUDNN_STATUS_SUCCESS) return {{}, stream.original_status};
    std::array<CudaDeviceMemoryView, 4> buffers{};
    for (std::size_t index = 0; index != 3; ++index) {
        const auto descriptor = problem.descriptors[index == 2 ? 3 : index].native;
        std::size_t bytes{};
        cudnnStatus_t status{};
        if (index == 1) {
            status = state.invoke(sdk->filter_size, static_cast<CudaDnnFilterHandle>(descriptor),
                                  &bytes);
        } else {
            if (!index) {
                cudnnDataType_t type{};
                int rank{};
                std::array<int, CUDNN_DIM_MAX> sizes{}, strides{};
                status = state.invoke(sdk->tensor_get_nd,
                                      static_cast<cudnnTensorDescriptor_t>(descriptor),
                                      CUDNN_DIM_MAX, &type, &rank, sizes.data(), strides.data());
                if (state.check_context() != CUDA_SUCCESS) {
                    state.fault();
                    return {{.status = ExecutionAcceptStatus::kQuarantined, .receipt = {}}, status};
                }
                if (status != CUDNN_STATUS_SUCCESS) return {{}, status};
                std::uint32_t type_bits{};
                std::memcpy(&type_bits, &type, sizeof(type_bits));
                if (rank <= 0 || rank > CUDNN_DIM_MAX ||
                    type_bits > dnn_profile::data_type_enum_max()) {
                    state.fault();
                    return {{.status = ExecutionAcceptStatus::kQuarantined, .receipt = {}}, status};
                }
                const auto width = type_bits == CUDNN_DATA_DOUBLE ? sizeof(double) : sizeof(float);
                if (problem.scalar_bytes != width) return {};
            }
            status = state.invoke(sdk->tensor_size,
                                  static_cast<cudnnTensorDescriptor_t>(descriptor), &bytes);
        }
        if (state.check_context() != CUDA_SUCCESS) {
            state.fault();
            return {{.status = ExecutionAcceptStatus::kQuarantined, .receipt = {}}, status};
        }
        if (status != CUDNN_STATUS_SUCCESS) return {{}, status};
        if (!bytes) return {};
        const auto memory = device_memory(problem.buffers[index], bytes);
        if (!memory) return {};
        buffers[index] = *memory;
    }
    const auto& workspace = problem.buffers[3];
    if (workspace.allocation) {
        const auto memory = device_memory(workspace, problem.workspace_bytes);
        if (!memory) return {};
        buffers[3] = *memory;
    } else if (workspace.offset || problem.workspace_bytes)
        return {};
    try {
        return {state.executor->accept_library(
                    sequence, {.stream = handle->stream,
                               .parameters = {},
                               .call = make_cuda_dnn_convolution_call(sdk, handle->native, problem,
                                                                      std::move(buffers))}),
                {}};
    } catch (...) {
        return {{.status = ExecutionAcceptStatus::kFull, .receipt = {}}, {}};
    }
}

CudaDnnComputeAcceptance CudaExecutionSession::accept_dnn_compute(
    std::uint64_t sequence, const CudaSessionDnnCompute& problem) noexcept {
    auto& state = *implementation_;
    if (!state.owns() || !state.executor || state.is_faulted)
        return {{.status = ExecutionAcceptStatus::kQuarantined, .receipt = {}}, {}};
    if (state.is_closing) return {{.status = ExecutionAcceptStatus::kClosed, .receipt = {}}, {}};
    if (state.check_context() != CUDA_SUCCESS) return {};
    const auto& sdk = state.dispatch.dnn;
    const auto handle =
        std::ranges::find(state.storage->dnn, problem.handle, &Implementation::Dnn::id);
    if (!sdk || !sdk->tensor_get_nd || !sdk->tensor_size || handle == state.storage->dnn.end() ||
        handle->is_uncertain || handle->has_destroy_attempt)
        return {};
    if (problem.kind == CudaDnnComputeKind::kActivationForward) {
        if (!sdk->activation_forward || problem.activation.kind != CudaDnnObjectKind::kActivation ||
            !problem.activation.native || !problem.activation.lifetime)
            return {};
    } else if (problem.kind == CudaDnnComputeKind::kPoolingForward) {
        if (!sdk->pooling_forward || problem.pooling.kind != CudaDnnObjectKind::kPooling ||
            !problem.pooling.native || !problem.pooling.lifetime || problem.algorithm ||
            problem.mode)
            return {};
    } else if (problem.kind == CudaDnnComputeKind::kSoftmaxForward) {
        // The unfixed SDK algorithm enum represents [0,3]. Unknown but
        // representable value 3 still reaches the original native error path.
        if (!sdk->softmax_forward || problem.algorithm > 3 || problem.mode > 1 ||
            problem.pooling.native || problem.pooling.lifetime)
            return {};
    } else if (problem.kind == CudaDnnComputeKind::kLrnForward) {
        if (!sdk->lrn_forward || problem.lrn.kind != CudaDnnObjectKind::kLrn ||
            !problem.lrn.native || !problem.lrn.lifetime || problem.algorithm || problem.mode > 1)
            return {};
    } else if (problem.kind == CudaDnnComputeKind::kTransformTensor) {
        if (!sdk->transform_tensor || problem.algorithm || problem.mode ||
            problem.activation.native || problem.activation.lifetime || problem.pooling.native ||
            problem.pooling.lifetime || problem.lrn.native || problem.lrn.lifetime)
            return {};
    } else if (problem.kind != CudaDnnComputeKind::kAddTensor || !sdk->add_tensor) {
        return {};
    }
    const auto stream = get_dnn_stream(problem.handle);
    if (stream.status != CUDA_SUCCESS)
        return {{.status = state.is_faulted ? ExecutionAcceptStatus::kQuarantined
                                            : ExecutionAcceptStatus::kInvalid,
                 .receipt = {}},
                stream.original_status};
    if (stream.original_status != CUDNN_STATUS_SUCCESS) return {{}, stream.original_status};
    std::array<CudaDeviceMemoryView, 2> buffers;
    for (std::size_t index = 0; index < buffers.size(); ++index) {
        const auto& descriptor = problem.tensors[index];
        if (descriptor.kind != CudaDnnObjectKind::kTensor || !descriptor.native ||
            !descriptor.lifetime)
            return {};
        const auto native = static_cast<cudnnTensorDescriptor_t>(descriptor.native);
        cudnnDataType_t type{};
        int dimensions{};
        std::array<int, CUDNN_DIM_MAX> sizes{}, strides{};
        auto status = state.invoke(sdk->tensor_get_nd, native, CUDNN_DIM_MAX, &type, &dimensions,
                                   sizes.data(), strides.data());
        if (state.check_context() != CUDA_SUCCESS) {
            state.fault();
            return {{.status = ExecutionAcceptStatus::kQuarantined, .receipt = {}}, status};
        }
        if (status != CUDNN_STATUS_SUCCESS) return {{}, status};
        std::uint32_t type_bits{};
        static_assert(sizeof(type) == sizeof(type_bits));
        std::memcpy(&type_bits, &type, sizeof(type_bits));
        if (dimensions <= 0 || dimensions > CUDNN_DIM_MAX ||
            type_bits > dnn_profile::data_type_enum_max()) {
            state.fault();
            return {{.status = ExecutionAcceptStatus::kQuarantined, .receipt = {}}, status};
        }
        const auto scalar_bytes = type_bits == CUDNN_DATA_DOUBLE ? sizeof(double) : sizeof(float);
        if (problem.scalar_bytes != scalar_bytes) return {};
        std::size_t bytes{};
        status = state.invoke(sdk->tensor_size, native, &bytes);
        if (state.check_context() != CUDA_SUCCESS) {
            state.fault();
            return {{.status = ExecutionAcceptStatus::kQuarantined, .receipt = {}}, status};
        }
        if (status != CUDNN_STATUS_SUCCESS) return {{}, status};
        if (!bytes) return {};
        const auto memory = device_memory(problem.buffers[index], bytes);
        if (!memory) return {};
        buffers[index] = *memory;
    }
    if (problem.kind == CudaDnnComputeKind::kTransformTensor) {
        const auto& x = buffers[0];
        const auto& y = buffers[1];
        // Unlike softmax, the original transformation contract forbids
        // overlapping tensor storage. Subtraction avoids address-end overflow
        // and still permits adjacent/disjoint views of the same allocation.
        const bool is_disjoint = x.address <= y.address ? y.address - x.address >= x.bytes
                                                        : x.address - y.address >= y.bytes;
        if (!is_disjoint) return {};
    }
    if (state.check_context() != CUDA_SUCCESS) {
        state.fault();
        return {{.status = ExecutionAcceptStatus::kQuarantined, .receipt = {}}, {}};
    }
    try {
        return {state.executor->accept_library(
                    sequence,
                    {.stream = handle->stream,
                     .parameters = {},
                     .call = make_cuda_dnn_call(sdk, handle->native, problem, std::move(buffers))}),
                {}};
    } catch (...) {
        return {{.status = ExecutionAcceptStatus::kFull, .receipt = {}}, {}};
    }
}
#include "internal/cuda_dnn_legacy_session.inc"
#endif
}  // namespace glimmer::backend::cuda
