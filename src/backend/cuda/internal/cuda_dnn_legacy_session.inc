// Included by the Session translation unit after its private owner definition.
CudaDnnLegacyResult CudaExecutionSession::dnn_legacy_control(
    std::uint64_t id, CudaDnnRegistry& metadata, const CudaDnnLegacyControl& input) noexcept {
    auto& state = *implementation_;
    const auto reject = [](CUresult proof) {
        CudaDnnLegacyResult result;
        result.result = dnn_gate(rejected(proof));
        if (proof == CUDA_ERROR_NOT_READY)
            result.result.disposition = CudaDnnDisposition::kDeferred;
        return result;
    };
    if (!state.owns() || !state.executor || state.is_faulted ||
        state.executor->snapshot().is_quarantined)
        return reject(CUDA_ERROR_UNKNOWN);
    if (state.is_folded_active || state.is_rnn_query_active || state.is_native_issue_active)
        return reject(CUDA_ERROR_NOT_READY);
    if (state.is_closing) return reject(CUDA_ERROR_DEINITIALIZED);
#if CUDNN_MAJOR != 8
    (void)id;
    (void)metadata;
    (void)input;
    return reject(CUDA_ERROR_NOT_SUPPORTED);
#else
    const auto handle = std::ranges::find(state.storage->dnn, id, &Implementation::Dnn::id);
    if (!state.dispatch.dnn || handle == state.storage->dnn.end() || !handle->native ||
        handle->is_uncertain || handle->has_destroy_attempt || input.handle != id)
        return reject(CUDA_ERROR_INVALID_HANDLE);
    if (const auto proof = state.domain_cold_status(); proof != CUDA_SUCCESS) return reject(proof);
    struct Active {
        bool& value;
        explicit Active(bool& flag) : value(flag) {
            value = true;
        }
        ~Active() {
            value = false;
        }
    } active{state.is_rnn_query_active};
    if (const auto proof = state.check_context(true); proof != CUDA_SUCCESS) return reject(proof);
    auto result = metadata.legacy_control(
        handle->native, input,
        {[](const void* opaque) noexcept {
             return static_cast<const Implementation*>(opaque)->check_context(true);
         },
         &state, this});
    if (result.has_context_loss) state.fault();
    return result;
#endif
}

CudaDnnRnnAcceptance CudaExecutionSession::accept_dnn_legacy_rnn(
    std::uint64_t sequence, CudaDnnRegistry& metadata,
    const CudaSessionDnnLegacyRnn& input) noexcept {
    auto& state = *implementation_;
    CudaDnnRnnAcceptance output;
    const auto reject = [&](CUresult proof,
                            ExecutionAcceptStatus why = ExecutionAcceptStatus::kInvalid) {
        output.native.rejection_status = proof;
        output.native.acceptance.status = why;
        return output;
    };
    if (!state.owns() || !state.executor || state.is_faulted ||
        state.executor->snapshot().is_quarantined)
        return reject(CUDA_ERROR_UNKNOWN, ExecutionAcceptStatus::kQuarantined);
    if (state.is_closing) return reject(CUDA_ERROR_DEINITIALIZED, ExecutionAcceptStatus::kClosed);
    if (state.is_folded_active || state.is_rnn_query_active || state.is_native_issue_active)
        return reject(CUDA_ERROR_NOT_READY, ExecutionAcceptStatus::kFull);
#if CUDNN_MAJOR != 8
    (void)sequence;
    (void)metadata;
    (void)input;
    return reject(CUDA_ERROR_NOT_SUPPORTED);
#else
    using Kind = CudaDnnLegacyComputeKind;
    const bool forward = input.kind == Kind::kForwardTraining;
    const bool backward = input.kind == Kind::kBackwardData;
    const bool weights = input.kind == Kind::kBackwardWeights;
    if (!forward && !backward && !weights) return reject(CUDA_ERROR_INVALID_VALUE);
    // Exact selected-ABI fields: unused identities cannot conceal a resource.
    if (!backward &&
        (input.sequences[2].has_descriptors || !input.sequences[2].descriptors.empty()))
        return reject(CUDA_ERROR_INVALID_VALUE);
    const auto absent = [&](std::size_t index) {
        return !input.buffers[index].allocation && !input.buffers[index].offset;
    };
    if ((forward && (!absent(2) || !absent(3) || !absent(8) || !absent(9) || input.states[4] ||
                     input.states[5])) ||
        (backward && !absent(0)) ||
        (weights && (!absent(2) || !absent(3) || !absent(5) || !absent(6) || !absent(7) ||
                     !absent(8) || !absent(9) ||
                     std::any_of(input.states.begin() + 1, input.states.end(),
                                 [](auto id) { return id != 0; }))))
        return reject(CUDA_ERROR_INVALID_VALUE);
    for (const auto& range : input.buffers)
        if (!range.allocation && range.offset) return reject(CUDA_ERROR_INVALID_VALUE);
    const auto& sdk = state.dispatch.dnn;
    const auto handle =
        std::ranges::find(state.storage->dnn, input.handle, &Implementation::Dnn::id);
    if (!sdk || handle == state.storage->dnn.end() || !handle->native || handle->is_uncertain ||
        handle->has_destroy_attempt)
        return reject(CUDA_ERROR_INVALID_HANDLE);
    if (!sdk->get_stream || !sdk->tensor_get_nd || !sdk->tensor_size || !sdk->filter_size ||
        !sdk->filter_get_nd || !state.dispatch.context_synchronize ||
        (forward && !sdk->legacy.forward) || (backward && !sdk->legacy.backward_data) ||
        (weights && !sdk->legacy.backward_weights))
        return reject(CUDA_ERROR_NOT_SUPPORTED);
    if (const auto proof = state.domain_cold_status(); proof != CUDA_SUCCESS) return reject(proof);
    const auto bytes = cuda_dnn_legacy_storage_bytes(input);
    if (!bytes) return reject(CUDA_ERROR_INVALID_VALUE);
    // Keep a local storage owner until preflight PMR vectors have unwound, even
    // when the moved capsule is destroyed by a later admission rejection.
    auto storage = state.executor->allocate_host_storage(*bytes);
    if (storage.status != ExecutionAcceptStatus::kAccepted) {
        const auto proof =
            storage.status == ExecutionAcceptStatus::kFull          ? CUDA_ERROR_NOT_READY
            : storage.status == ExecutionAcceptStatus::kOutOfMemory ? CUDA_ERROR_OUT_OF_MEMORY
            : storage.status == ExecutionAcceptStatus::kClosed      ? CUDA_ERROR_DEINITIALIZED
            : storage.status == ExecutionAcceptStatus::kInvalid     ? CUDA_ERROR_INVALID_VALUE
                                                                    : CUDA_ERROR_UNKNOWN;
        return reject(proof, storage.status);
    }
    auto borrowed = metadata.borrow_legacy_rnn(input, storage.storage);
    if (borrowed.result.disposition != CudaDnnDisposition::kComplete ||
        borrowed.result.status != CUDA_SUCCESS)
        return reject(borrowed.result.status,
                      borrowed.result.disposition == CudaDnnDisposition::kDeferred
                          ? ExecutionAcceptStatus::kFull
                      : borrowed.result.status == CUDA_ERROR_OUT_OF_MEMORY
                          ? ExecutionAcceptStatus::kOutOfMemory
                          : ExecutionAcceptStatus::kInvalid);
    // Keep the guard's referenced lease alive even if queue rejection destroys
    // the moved call capsule before this function unwinds.
    const auto call_lease = borrowed.view.rnn.lease;
    if (!call_lease->begin_call())
        return reject(CUDA_ERROR_NOT_READY, ExecutionAcceptStatus::kFull);
    struct Active {
        bool& flag;
        CudaDnnRnnLease& lease;
        ~Active() {
            flag = false;
            lease.end_call();
        }
    } active{state.is_rnn_query_active, *call_lease};
    state.is_rnn_query_active = true;
    const auto context = [&]() {
        const auto proof = state.check_context(true);
        if (proof != CUDA_SUCCESS) state.fault();
        return proof;
    };
    if (const auto proof = context(); proof != CUDA_SUCCESS)
        return reject(proof, ExecutionAcceptStatus::kQuarantined);
    const auto stream = state.observe_dnn_stream(input.handle);
    if (stream.status != CUDA_SUCCESS || stream.original_status != CUDNN_STATUS_SUCCESS) {
        output.preflight_status = stream.original_status;
        return reject(stream.status == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : stream.status);
    }
    const auto malformed = [&]() {
        borrowed.view.rnn.lease->quarantine();
        return CUDA_ERROR_UNKNOWN;
    };
    const auto multiply = [](std::size_t& value, std::size_t factor) {
        if (factor && value > std::numeric_limits<std::size_t>::max() / factor) return false;
        value *= factor;
        return true;
    };
    using Extent = CudaDnnLegacyExtent;
    std::size_t widest{};
    const auto tensor = [&](const CudaDnnObjectView& descriptor, Extent& extent) -> CUresult {
        if (!descriptor.native) return CUDA_SUCCESS;
        cudnnDataType_t type{};
        int rank{};
        std::array<int, CUDNN_DIM_MAX> sizes{}, strides{};
        const auto native = static_cast<cudnnTensorDescriptor_t>(descriptor.native);
        auto status = state.invoke(sdk->tensor_get_nd, native, CUDNN_DIM_MAX, &type, &rank,
                                   sizes.data(), strides.data());
        output.preflight_status = status;
        if (const auto proof = context(); proof != CUDA_SUCCESS) return proof;
        if (status != CUDNN_STATUS_SUCCESS) return CUDA_ERROR_INVALID_VALUE;
        std::uint32_t bits{};
        std::memcpy(&bits, &type, sizeof(bits));
        if (bits > dnn_profile::data_type_enum_max() || rank < 0 || rank > CUDNN_DIM_MAX)
            return malformed();
        // Old RNN consumes three tensor dimensions. A shorter public descriptor
        // cannot prove historical tail dimensions, even if its size getter is small.
        if (rank < 3) return CUDA_ERROR_INVALID_VALUE;
        if (bits > 2) return CUDA_ERROR_NOT_SUPPORTED;
        constexpr std::array<std::size_t, 3> widths{4, 8, 2};
        extent.width = widths[bits];
        widest = std::max(widest, extent.width);
        std::size_t hull = 1;
        extent.elements = 1;
        for (int i = 0; i < rank; ++i) {
            if (sizes[i] <= 0 || strides[i] <= 0) return CUDA_ERROR_INVALID_VALUE;
            auto step = static_cast<std::size_t>(sizes[i] - 1);
            if (!multiply(extent.elements, static_cast<std::size_t>(sizes[i])) ||
                !multiply(step, static_cast<std::size_t>(strides[i])) ||
                hull > std::numeric_limits<std::size_t>::max() - step)
                return CUDA_ERROR_INVALID_VALUE;
            hull += step;
        }
        if (!multiply(hull, extent.width)) return CUDA_ERROR_INVALID_VALUE;
        status = state.invoke(sdk->tensor_size, native, &extent.bytes);
        output.preflight_status = status;
        if (const auto proof = context(); proof != CUDA_SUCCESS) return proof;
        if (status != CUDNN_STATUS_SUCCESS) return CUDA_ERROR_INVALID_VALUE;
        return extent.bytes < hull ? malformed() : CUDA_SUCCESS;
    };
    try {
        std::array<std::pmr::vector<Extent>, 3> arrays{
            std::pmr::vector<Extent>{storage.storage->resource()},
            std::pmr::vector<Extent>{storage.storage->resource()},
            std::pmr::vector<Extent>{storage.storage->resource()}};
        std::array<Extent, 6> states{};
        for (std::size_t i = 0; i < arrays.size(); ++i) {
            arrays[i].resize(borrowed.view.sequences[i].size());
            for (std::size_t j = 0; j < arrays[i].size(); ++j)
                if (const auto proof = tensor(borrowed.view.sequences[i][j], arrays[i][j]);
                    proof != CUDA_SUCCESS)
                    return reject(proof);
        }
        for (std::size_t i = 0; i < states.size(); ++i)
            if (const auto proof = tensor(borrowed.view.states[i], states[i]);
                proof != CUDA_SUCCESS)
                return reject(proof);
        std::array<std::size_t, 13> sizes{};
        for (std::size_t i = 0; i < arrays.size(); ++i) {
            const std::size_t slot = backward ? i + 1 : i;
            if (!backward && i == 2) continue;
            for (const auto& extent : arrays[i]) {
                auto dense = extent.elements;
                if (!multiply(dense, widest)) return reject(CUDA_ERROR_INVALID_VALUE);
                const auto bytes = std::max(dense, extent.bytes);
                if (sizes[slot] > std::numeric_limits<std::size_t>::max() - bytes)
                    return reject(CUDA_ERROR_INVALID_VALUE);
                sizes[slot] += bytes;
            }
            if (input.buffers[slot].allocation &&
                (!input.sequences[i].has_descriptors ||
                 std::ranges::any_of(borrowed.view.sequences[i],
                                     [](const auto& view) { return !view.native; })))
                return reject(CUDA_ERROR_INVALID_VALUE);
        }
        for (std::size_t i = 0; i < states.size(); ++i) {
            auto dense = states[i].elements;
            if (!multiply(dense, widest)) return reject(CUDA_ERROR_INVALID_VALUE);
            sizes[i + 4] = std::max(dense, states[i].bytes);
            if (input.buffers[i + 4].allocation && !borrowed.view.states[i].native)
                return reject(CUDA_ERROR_INVALID_VALUE);
        }
        if (borrowed.view.filter.native) {
            cudnnDataType_t type{};
            cudnnTensorFormat_t format{};
            int rank{};
            std::array<int, CUDNN_DIM_MAX> dimensions{};
            const auto filter = static_cast<CudaDnnFilterHandle>(borrowed.view.filter.native);
            auto native = state.invoke(sdk->filter_get_nd, filter, CUDNN_DIM_MAX, &type, &format,
                                       &rank, dimensions.data());
            output.preflight_status = native;
            if (const auto proof = context(); proof != CUDA_SUCCESS) return reject(proof);
            if (native != CUDNN_STATUS_SUCCESS) return reject(CUDA_ERROR_INVALID_VALUE);
            std::uint32_t bits{}, format_bits{};
            std::memcpy(&bits, &type, sizeof(bits));
            std::memcpy(&format_bits, &format, sizeof(format_bits));
            if (bits > dnn_profile::data_type_enum_max() || format_bits > 3 || rank < 0 ||
                rank > CUDNN_DIM_MAX)
                return reject(malformed());
            if (bits > 2 || format_bits > 1) return reject(CUDA_ERROR_NOT_SUPPORTED);
            std::size_t dense = 1;
            for (int i = 0; i < rank; ++i)
                if (dimensions[i] <= 0 || !multiply(dense, static_cast<std::size_t>(dimensions[i])))
                    return reject(CUDA_ERROR_INVALID_VALUE);
            constexpr std::array<std::size_t, 3> widths{4, 8, 2};
            if (!multiply(dense, std::max(widest, widths[bits])))
                return reject(CUDA_ERROR_INVALID_VALUE);
            native = state.invoke(sdk->filter_size, filter, &sizes[10]);
            output.preflight_status = native;
            if (const auto proof = context(); proof != CUDA_SUCCESS) return reject(proof);
            if (native != CUDNN_STATUS_SUCCESS) return reject(CUDA_ERROR_INVALID_VALUE);
            sizes[10] = std::max(sizes[10], dense);
        } else if (input.buffers[10].allocation)
            return reject(CUDA_ERROR_INVALID_VALUE);
        sizes[11] = static_cast<std::size_t>(input.workspace_bytes);
        sizes[12] = static_cast<std::size_t>(input.reserve_bytes);
        std::array<CudaDeviceMemoryView, 13> buffers{};
        for (std::size_t i = 0; i < buffers.size(); ++i) {
            if (!input.buffers[i].allocation) continue;
            auto memory = device_memory(input.buffers[i], sizes[i]);
            if (!memory) return reject(CUDA_ERROR_INVALID_VALUE);
            buffers[i] = std::move(*memory);
        }
        std::array<CudaDeviceMemoryView, 4> dropout{};
        for (std::size_t i = 0; i < dropout.size(); ++i) {
            if (!borrowed.view.rnn.dropouts[i].object.native) continue;
            auto memory = resolve_dnn_dropout(borrowed.view.rnn.dropouts[i]);
            if (!memory) return reject(CUDA_ERROR_INVALID_VALUE);
            dropout[i] = std::move(*memory);
        }
        if (const auto proof = context(); proof != CUDA_SUCCESS) return reject(proof);
        auto completion = std::make_shared<CudaDnnRnnCompletion>();
        auto call = make_cuda_dnn_legacy_call(
            sdk, handle->native, input, std::move(borrowed.view), std::move(buffers),
            {state.storage->context, state.dispatch.execution.context_get_current,
             state.dispatch.context_synchronize},
            completion, std::move(dropout));
        output.native.acceptance = state.executor->accept_library(
            sequence, {.stream = handle->stream, .parameters = {}, .call = std::move(call)});
        if (output.native.acceptance.accepted()) output.result = std::move(completion);
        return output;
    } catch (...) {
        return reject(CUDA_ERROR_OUT_OF_MEMORY, ExecutionAcceptStatus::kOutOfMemory);
    }
#endif
}
