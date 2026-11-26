#include "internal/cuda_execution_session_state.h"

// cuRAND, cuBLAS and cuBLASLt operations.
namespace glimmer::backend::cuda {

CudaSessionRandResult CudaExecutionSession::create_random(std::uint32_t type) noexcept {
    auto& state = *implementation_;
    const auto gate = state.cold_gate(true);
    if (gate.status != CUDA_SUCCESS) return random_gate(gate);
    const auto& sdk = state.dispatch.random;
    if (!sdk.create || !sdk.set_seed || !sdk.destroy)
        return random_gate(rejected(CUDA_ERROR_NOT_SUPPORTED));
    // CUDA 12.6's unfixed enum has the C++ value range [0,255]. Unknown
    // representable values still reach the SDK; arbitrary wire bits must not
    // become an invalid enum load before that boundary.
    if (type > 255) return random_gate(rejected(CUDA_ERROR_INVALID_VALUE));
    if (!state.has_id_space || state.storage->random.size() == state.limits.max_objects)
        return random_gate(rejected(CUDA_ERROR_OUT_OF_MEMORY));
    Implementation::Random handle{.id = state.next_id};
    curandRngType_t native_type{};
    static_assert(sizeof(native_type) == sizeof(type));
    std::memcpy(&native_type, &type, sizeof(type));
    auto result = random_completed(state.invoke(sdk.create, &handle.native, native_type));
    if (*result.original_status != CURAND_STATUS_SUCCESS && !handle.native) return result;
    (void)state.consume_id();
    handle.is_uncertain = *result.original_status != CURAND_STATUS_SUCCESS || !handle.native;
    for (auto& existing : state.storage->random)
        if (existing.native == handle.native) existing.is_uncertain = handle.is_uncertain = true;
    state.storage->random.push_back(handle);  // Capacity reserved before any SDK call.
    if (handle.is_uncertain) {
        state.fault();
        result.disposition = CudaSessionDisposition::kRejected;
        result.status = CUDA_ERROR_UNKNOWN;
    } else
        result.id = handle.id;
    return result;
}
// NOLINTBEGIN(bugprone-easily-swappable-parameters): logical generator ID and SDK seed ABI.
CudaSessionRandResult CudaExecutionSession::set_random_seed(std::uint64_t id,
                                                            std::uint64_t seed) noexcept {
    // NOLINTEND(bugprone-easily-swappable-parameters)
    auto& state = *implementation_;
    const auto gate = state.cold_gate(true);
    if (gate.status != CUDA_SUCCESS) return random_gate(gate);
    const auto found = std::ranges::find(state.storage->random, id, &Implementation::Random::id);
    if (found == state.storage->random.end())
        return random_gate(rejected(CUDA_ERROR_INVALID_HANDLE));
    if (found->is_uncertain || found->has_destroy_attempt)
        return random_gate(rejected(CUDA_ERROR_UNKNOWN));
    return random_completed(state.invoke(state.dispatch.random.set_seed, found->native, seed));
}
CudaSessionRandResult CudaExecutionSession::destroy_random(std::uint64_t id) noexcept {
    auto& state = *implementation_;
    const auto gate = state.cold_gate();
    if (gate.status != CUDA_SUCCESS) return random_gate(gate);
    const auto found = std::ranges::find(state.storage->random, id, &Implementation::Random::id);
    if (found == state.storage->random.end())
        return random_gate(rejected(CUDA_ERROR_INVALID_HANDLE));
    if (found->is_uncertain || found->has_destroy_attempt)
        return random_gate(rejected(CUDA_ERROR_UNKNOWN));
    found->has_destroy_attempt = true;
    const auto result =
        random_completed(state.invoke(state.dispatch.random.destroy, found->native));
    if (*result.original_status == CURAND_STATUS_SUCCESS)
        state.storage->random.erase(found);
    else {
        found->is_uncertain = true;
        state.fault();
    }
    return result;
}

CudaSessionBlasResult CudaExecutionSession::create_blas() noexcept {
    // Existing cuBLAS preparation and callback identity remain unchanged.
    auto& state = *implementation_;
    const auto gate = state.cold_gate(true);
    if (gate.status != CUDA_SUCCESS) return blas_gate(gate);
    const auto& dispatch = state.dispatch.blas;
    if (!dispatch.create || !dispatch.destroy || !dispatch.set_stream || !dispatch.sgemm)
        return blas_gate(rejected(CUDA_ERROR_NOT_SUPPORTED));
    if (state.stream_config.uses_default_proxy && !dispatch.get_stream)
        return blas_gate(rejected(CUDA_ERROR_NOT_SUPPORTED));
    if (!state.has_id_space || state.storage->blas.size() == state.limits.max_objects)
        return blas_gate(rejected(CUDA_ERROR_OUT_OF_MEMORY));
    const auto prepared = state.prepare_sgemm();
    if (prepared != CUDA_SUCCESS) return blas_gate(rejected(prepared));
    std::optional<NativeExecutionScope> proxy_scope;
    if (state.stream_config.uses_default_proxy) proxy_scope.emplace(state.is_native_issue_active);
    Implementation::Blas handle{.id = state.next_id};
    auto result = blas_completed(state.invoke(dispatch.create, &handle.native));
    const auto proof =
        state.stream_config.uses_default_proxy ? state.check_context(true) : CUDA_SUCCESS;
    if (*result.original_status != CUBLAS_STATUS_SUCCESS && !handle.native) {
        if (proof != CUDA_SUCCESS) {
            state.fault();
            result.status = proof;
            result.disposition = CudaSessionDisposition::kRejected;
        }
        return result;
    }
    (void)state.consume_id();
    handle.is_uncertain =
        *result.original_status != CUBLAS_STATUS_SUCCESS || !handle.native || proof != CUDA_SUCCESS;
    handle.lt.id = handle.id;
    handle.lt.native = reinterpret_cast<cublasLtHandle_t>(handle.native);
    for (auto& existing : state.storage->blas)
        if (existing.native == handle.native) {
            existing.is_uncertain = existing.lt.is_uncertain = handle.is_uncertain = true;
        }
    for (auto& existing : state.storage->lt)
        if (static_cast<void*>(existing.native) == static_cast<void*>(handle.native))
            existing.is_uncertain = handle.is_uncertain = true;
    handle.lt.is_uncertain = handle.is_uncertain;
    state.storage->blas.push_back(handle);
    if (!handle.is_uncertain && state.storage->default_proxy) {
        const auto configured =
            state.invoke(dispatch.set_stream, handle.native,
                         reinterpret_cast<cudaStream_t>(state.storage->default_proxy));
        const auto configured_proof = state.check_context(true);
        handle.is_uncertain =
            configured != CUBLAS_STATUS_SUCCESS || configured_proof != CUDA_SUCCESS;
        if (!handle.is_uncertain) {
            cudaStream_t observed{};
            const auto queried = state.invoke(dispatch.get_stream, handle.native, &observed);
            const auto queried_proof = state.check_context(true);
            handle.is_uncertain =
                queried != CUBLAS_STATUS_SUCCESS || queried_proof != CUDA_SUCCESS ||
                reinterpret_cast<CUstream>(observed) != state.storage->default_proxy;
        }
        state.storage->blas.back().is_uncertain = handle.is_uncertain;
        state.storage->blas.back().lt.is_uncertain = handle.is_uncertain;
    }
    if (handle.is_uncertain) {
        state.fault();
        result.disposition = CudaSessionDisposition::kRejected;
        result.status = CUDA_ERROR_UNKNOWN;
    } else {
        result.id = handle.id;
    }
    return result;
}
CudaSessionBlasResult CudaExecutionSession::destroy_blas(std::uint64_t id) noexcept {
    auto& state = *implementation_;
    const auto gate = state.cold_gate();
    if (gate.status != CUDA_SUCCESS) return blas_gate(gate);
    const auto found = std::ranges::find(state.storage->blas, id, &Implementation::Blas::id);
    if (found == state.storage->blas.end()) return blas_gate(rejected(CUDA_ERROR_INVALID_HANDLE));
    if (found->is_uncertain || found->has_destroy_attempt)
        return blas_gate(rejected(CUDA_ERROR_UNKNOWN));
    found->has_destroy_attempt = true;
    auto result = blas_completed(state.invoke(state.dispatch.blas.destroy, found->native));
    if (*result.original_status != CUBLAS_STATUS_SUCCESS) {
        found->is_uncertain = true;
        state.fault();
    } else {
        state.storage->blas.erase(found);
    }
    return result;
}
CudaSessionBlasResult CudaExecutionSession::create_lt() noexcept {
    auto& state = *implementation_;
    const auto gate = state.cold_gate(true);
    if (gate.status != CUDA_SUCCESS) return blas_gate(gate);
    const auto& dispatch = state.dispatch.lt;
    if (!dispatch.create || !dispatch.destroy) return blas_gate(rejected(CUDA_ERROR_NOT_SUPPORTED));
    if (!state.has_id_space || state.storage->lt.size() == state.limits.max_objects)
        return blas_gate(rejected(CUDA_ERROR_OUT_OF_MEMORY));
    const auto prepared = state.prepare_lt();
    if (prepared != CUDA_SUCCESS) return blas_gate(rejected(prepared));
    Implementation::Lt handle{.id = state.next_id};
    try {
        handle.algorithms.reserve(state.limits.max_objects);
    } catch (...) {
        return blas_gate(rejected(CUDA_ERROR_OUT_OF_MEMORY));
    }
    auto result = blas_completed(state.invoke(dispatch.create, &handle.native));
    if (*result.original_status != CUBLAS_STATUS_SUCCESS && !handle.native) return result;
    (void)state.consume_id();
    handle.is_uncertain =
        *result.original_status != CUBLAS_STATUS_SUCCESS || !handle.native || state.is_faulted;
    for (auto& existing : state.storage->lt)
        if (existing.native == handle.native) existing.is_uncertain = handle.is_uncertain = true;
    for (auto& existing : state.storage->blas)
        if (static_cast<void*>(existing.native) == static_cast<void*>(handle.native))
            existing.is_uncertain = existing.lt.is_uncertain = handle.is_uncertain = true;
    state.storage->lt.push_back(std::move(handle));
    const auto& retained = state.storage->lt.back();
    if (retained.is_uncertain) {
        state.fault();
        result.disposition = CudaSessionDisposition::kRejected;
        result.status = CUDA_ERROR_UNKNOWN;
    } else
        result.id = retained.id;
    return result;
}
CudaSessionLtAlgorithms CudaExecutionSession::query_lt_algorithms(
    const CudaSessionLtQuery& query) noexcept {
    auto& state = *implementation_;
    CudaSessionLtAlgorithms output;
    const auto reject = [&](CUresult status) {
        output.result.status = status;
        output.result.disposition = CudaSessionDisposition::kRejected;
        output.algorithms.clear();
        return std::move(output);
    };
    if (!state.owns()) return reject(CUDA_ERROR_INVALID_CONTEXT);
    if (state.is_native_issue_active || state.is_folded_active || state.is_rnn_query_active)
        return reject(CUDA_ERROR_NOT_READY);
    const NativeExecutionScope active(state.is_native_issue_active);
    const auto context_status = state.check_context(true);
    if (context_status != CUDA_SUCCESS) return reject(context_status);
    if (state.is_faulted || state.is_closing || !state.executor ||
        state.executor->snapshot().is_quarantined)
        return reject(CUDA_ERROR_UNKNOWN);
    auto* handle = state.lt_handle(query.handle);
    if (!handle || handle->is_uncertain || handle->has_destroy_attempt)
        return reject(CUDA_ERROR_INVALID_HANDLE);
    if (state.persistent.replace_sgemm && !state.storage->lt_ids.back()) {
        // The outer native scope already excludes reentry. Only the physical
        // quiescence part of cold_gate applies to this internal preparation.
        if (!state.executor->resources_quiescent()) {
            output.result = blas_gate(rejected(CUDA_ERROR_NOT_READY));
            output.result.disposition = CudaSessionDisposition::kDeferred;
            return output;
        }
        const auto prepared = state.prepare_lt();
        if (prepared != CUDA_SUCCESS) return reject(prepared);
    }
    const auto proof = [&] {
        const auto status = state.check_context(true);
        if (status != CUDA_SUCCESS) state.fault();
        return status;
    };
    if (const auto status = proof(); status != CUDA_SUCCESS) return reject(status);
    const auto valid_view = [](const CudaLtObjectView& view, CudaLtObjectKind kind) {
        return view.kind == kind && view.native && view.lifetime;
    };
    if (!valid_view(query.operation, CudaLtObjectKind::kOperation) ||
        !std::ranges::all_of(query.matrices, [&](const auto& view) {
            return valid_view(view, CudaLtObjectKind::kLayout);
        }))
        return reject(CUDA_ERROR_INVALID_VALUE);
    const auto& sdk = state.dispatch.lt;
    if (!sdk.version || (query.algorithm ? !sdk.algorithm_check : !sdk.heuristic))
        return reject(CUDA_ERROR_NOT_SUPPORTED);
    if (!query.algorithm && (query.requested <= 0 || query.requested > 64 || !query.preference ||
                             !valid_view(*query.preference, CudaLtObjectKind::kPreference)))
        return reject(CUDA_ERROR_INVALID_VALUE);
    const auto version = state.invoke(sdk.version);
    if (const auto status = proof(); status != CUDA_SUCCESS) return reject(status);
    if (!version || (handle->sdk_version && handle->sdk_version != version)) {
        state.fault();
        return reject(CUDA_ERROR_UNKNOWN);
    }
    const auto same_algorithm = [](const auto& left, const auto& right) {
        static_assert(sizeof(cublasLtMatmulAlgo_t) == 64);
        return std::memcmp(&left, &right, sizeof(left)) == 0;
    };
    if (query.algorithm && !std::ranges::any_of(handle->algorithms, [&](const auto& known) {
            return same_algorithm(known, *query.algorithm);
        }))
        return reject(CUDA_ERROR_NOT_SUPPORTED);
    try {
        output.algorithms.resize(query.algorithm ? 1 : static_cast<std::size_t>(query.requested));
        const auto operation = static_cast<cublasLtMatmulDesc_t>(query.operation.native);
        std::array<cublasLtMatrixLayout_t, 4> layouts{};
        for (std::size_t index = 0; index < layouts.size(); ++index)
            layouts[index] = static_cast<cublasLtMatrixLayout_t>(query.matrices[index].native);
        if (!query.has_c_layout) layouts[2] = nullptr;
        int count{};
        if (query.algorithm) {
            output.algorithms[0].algo = *query.algorithm;
            output.result = blas_completed(
                state.invoke(sdk.algorithm_check, handle->native, operation, layouts[0], layouts[1],
                             layouts[2], layouts[3], &*query.algorithm, output.algorithms.data()));
            count = 1;
        } else {
            // Limited-by-algorithm search consumes caller algorithm state and
            // requires its own validated initialization/configuration contract.
            if (!sdk.preference_get) return reject(CUDA_ERROR_NOT_SUPPORTED);
            std::uint32_t mode{};
            std::size_t written{};
            const auto status =
                state.invoke(sdk.preference_get,
                             static_cast<cublasLtMatmulPreference_t>(query.preference->native),
                             CUBLASLT_MATMUL_PREF_SEARCH_MODE, &mode, sizeof(mode), &written);
            if (const auto context = proof(); context != CUDA_SUCCESS) return reject(context);
            if (status != CUBLAS_STATUS_SUCCESS || written != sizeof(mode))
                return reject(CUDA_ERROR_UNKNOWN);
            if (mode != 0) return reject(CUDA_ERROR_NOT_SUPPORTED);
            output.result = blas_completed(state.invoke(
                sdk.heuristic, handle->native, operation, layouts[0], layouts[1], layouts[2],
                layouts[3], static_cast<cublasLtMatmulPreference_t>(query.preference->native),
                query.requested, output.algorithms.data(), &count));
        }
        if (const auto status = proof(); status != CUDA_SUCCESS) return reject(status);
        output.sdk_version = version;
        if (output.result.original_status != CUBLAS_STATUS_SUCCESS) {
            output.algorithms.clear();
            return output;
        }
        if (count < 0 || static_cast<std::size_t>(count) > output.algorithms.size()) {
            state.fault();
            output.result.status = CUDA_ERROR_UNKNOWN;
            output.result.disposition = CudaSessionDisposition::kRejected;
            output.algorithms.clear();
            return output;
        }
        output.algorithms.resize(static_cast<std::size_t>(count));
        // Reserve all provenance records before publishing any algorithm. The
        // query performs no application GPU work and may reject a full cache.
        auto fresh = handle->algorithms;
        for (const auto& item : output.algorithms) {
            if (item.state != CUBLAS_STATUS_SUCCESS ||
                std::ranges::any_of(
                    fresh, [&](const auto& known) { return same_algorithm(known, item.algo); }))
                continue;
            if (fresh.size() == state.limits.max_objects) return reject(CUDA_ERROR_OUT_OF_MEMORY);
            fresh.push_back(item.algo);
        }
        handle->algorithms = std::move(fresh);
        handle->sdk_version = version;
        return output;
    } catch (...) {
        return reject(CUDA_ERROR_OUT_OF_MEMORY);
    }
}
CudaSessionBlasResult CudaExecutionSession::destroy_lt(std::uint64_t id) noexcept {
    auto& state = *implementation_;
    const auto gate = state.cold_gate();
    if (gate.status != CUDA_SUCCESS) return blas_gate(gate);
    const auto found = std::ranges::find(state.storage->lt, id, &Implementation::Lt::id);
    if (found == state.storage->lt.end()) return blas_gate(rejected(CUDA_ERROR_INVALID_HANDLE));
    if (found->is_uncertain || found->has_destroy_attempt)
        return blas_gate(rejected(CUDA_ERROR_UNKNOWN));
    found->has_destroy_attempt = true;
    auto result = blas_completed(state.invoke(state.dispatch.lt.destroy, found->native));
    if (*result.original_status != CUBLAS_STATUS_SUCCESS) {
        found->is_uncertain = true;
        state.fault();
    } else
        state.storage->lt.erase(found);
    return result;
}
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): distinct scoped resource IDs.
CudaSessionBlasResult CudaExecutionSession::set_blas_stream(std::uint64_t id,
                                                            std::uint64_t stream_id) noexcept {
    auto& state = *implementation_;
    const auto gate = state.cold_gate();
    if (gate.status != CUDA_SUCCESS) return blas_gate(gate);
    const auto handle = std::ranges::find(state.storage->blas, id, &Implementation::Blas::id);
    if (handle == state.storage->blas.end()) return blas_gate(rejected(CUDA_ERROR_INVALID_HANDLE));
    CUstream stream = state.storage->default_proxy;
    if (stream_id) {
        const auto found =
            std::ranges::find(state.storage->streams, stream_id, &Implementation::Stream::id);
        if (found == state.storage->streams.end())
            return blas_gate(rejected(CUDA_ERROR_INVALID_HANDLE));
        stream = found->native;
    }
    std::optional<NativeExecutionScope> proxy_scope;
    if (state.stream_config.uses_default_proxy) proxy_scope.emplace(state.is_native_issue_active);
    auto result = blas_completed(state.invoke(state.dispatch.blas.set_stream, handle->native,
                                              reinterpret_cast<cudaStream_t>(stream)));
    if (*result.original_status == CUBLAS_STATUS_SUCCESS) {
        handle->stream = stream_id;
        // The SDK resets workspace even when the stream identity is unchanged.
        handle->has_workspace = false;
        handle->workspace = {};
        handle->workspace_bytes = 0;
    } else
        state.fault();
    if (state.stream_config.uses_default_proxy && state.check_context(true) != CUDA_SUCCESS) {
        state.fault();
        result.status = CUDA_ERROR_UNKNOWN;
        result.disposition = CudaSessionDisposition::kRejected;
    }
    return result;
}
CudaSessionBlasResult CudaExecutionSession::set_blas_workspace(std::uint64_t id,
                                                               CudaAllocationRange workspace,
                                                               std::size_t bytes) noexcept {
    auto& state = *implementation_;
    const auto gate = state.cold_gate();
    if (gate.status != CUDA_SUCCESS) return blas_gate(gate);
    const auto handle = std::ranges::find(state.storage->blas, id, &Implementation::Blas::id);
    if (handle == state.storage->blas.end()) return blas_gate(rejected(CUDA_ERROR_INVALID_HANDLE));
    if (handle->is_uncertain || handle->has_destroy_attempt)
        return blas_gate(rejected(CUDA_ERROR_UNKNOWN));
    if (!state.dispatch.blas.set_workspace) return blas_gate(rejected(CUDA_ERROR_NOT_SUPPORTED));
    CUdeviceptr address{};
    if (workspace.allocation) {
        const auto memory = device_memory(workspace, bytes ? bytes : 1);
        if (!memory) return blas_gate(rejected(CUDA_ERROR_INVALID_VALUE));
        address = memory->address;
    } else if (bytes || workspace.offset)
        return blas_gate(rejected(CUDA_ERROR_INVALID_VALUE));
    // Let the original SDK reject misalignment and preserve its own status.
    // NOLINTBEGIN(performance-no-int-to-ptr): exact SDK device-address ABI, never a host access.
    auto result = blas_completed(state.invoke(state.dispatch.blas.set_workspace, handle->native,
                                              reinterpret_cast<void*>(address), bytes));
    // NOLINTEND(performance-no-int-to-ptr)
    if (*result.original_status == CUBLAS_STATUS_SUCCESS) {
        handle->has_workspace = true;
        handle->workspace = workspace;
        handle->workspace_bytes = bytes;
    } else if (*result.original_status != CUBLAS_STATUS_INVALID_VALUE &&
               *result.original_status != CUBLAS_STATUS_NOT_SUPPORTED) {
        state.fault();
    }
    return result;
}
CudaSessionBlasResult CudaExecutionSession::set_blas_attribute(std::uint64_t id,
                                                               CudaBlasAttribute attribute,
                                                               std::uint64_t value) noexcept {
    if (attribute == CudaBlasAttribute::kStream) return set_blas_stream(id, value);
    auto& state = *implementation_;
    const auto gate = state.cold_gate();
    if (gate.status != CUDA_SUCCESS) return blas_gate(gate);
    const auto handle = std::ranges::find(state.storage->blas, id, &Implementation::Blas::id);
    if (handle == state.storage->blas.end()) return blas_gate(rejected(CUDA_ERROR_INVALID_HANDLE));
    cublasStatus_t status{};
    if (attribute == CudaBlasAttribute::kPointerMode) {
        if (value > 1) return blas_gate(rejected(CUDA_ERROR_INVALID_VALUE));
        if (!state.dispatch.blas.set_pointer_mode)
            return blas_gate(rejected(CUDA_ERROR_NOT_SUPPORTED));
        const auto mode = static_cast<cublasPointerMode_t>(value);
        status = state.invoke(state.dispatch.blas.set_pointer_mode, handle->native, mode);
        if (status == CUBLAS_STATUS_SUCCESS) handle->pointer_mode = mode;
    } else if (attribute == CudaBlasAttribute::kMathMode) {
        // SDK enum values, including combined reduction flags, fit 0..31.
        // The actual library rejects unsupported combinations in that range.
        if (value > 31) return blas_gate(rejected(CUDA_ERROR_INVALID_VALUE));
        if (!state.dispatch.blas.set_math_mode)
            return blas_gate(rejected(CUDA_ERROR_NOT_SUPPORTED));
        status = state.invoke(state.dispatch.blas.set_math_mode, handle->native,
                              static_cast<cublasMath_t>(value));
        if (status == CUBLAS_STATUS_SUCCESS) handle->math_mode = value;
    } else if (attribute == CudaBlasAttribute::kAtomicsMode) {
        if (value > 1) return blas_gate(rejected(CUDA_ERROR_INVALID_VALUE));
        if (!state.dispatch.blas.set_atomics_mode)
            return blas_gate(rejected(CUDA_ERROR_NOT_SUPPORTED));
        const auto mode = static_cast<cublasAtomicsMode_t>(value);
        status = state.invoke(state.dispatch.blas.set_atomics_mode, handle->native, mode);
        if (status == CUBLAS_STATUS_SUCCESS) handle->atomics_mode = mode;
    } else if (attribute == CudaBlasAttribute::kSmCountTarget) {
        if (value > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
            return blas_gate(rejected(CUDA_ERROR_INVALID_VALUE));
        if (!state.dispatch.blas.set_sm_count_target)
            return blas_gate(rejected(CUDA_ERROR_NOT_SUPPORTED));
        const auto count = static_cast<int>(value);
        status = state.invoke(state.dispatch.blas.set_sm_count_target, handle->native, count);
        if (status == CUBLAS_STATUS_SUCCESS) handle->sm_count_target = count;
    } else
        return blas_gate(rejected(CUDA_ERROR_INVALID_VALUE));
    if (status != CUBLAS_STATUS_SUCCESS && status != CUBLAS_STATUS_INVALID_VALUE &&
        status != CUBLAS_STATUS_NOT_SUPPORTED)
        state.fault();
    return blas_completed(status);
}
CudaSessionBlasResult CudaExecutionSession::get_blas_attribute(
    std::uint64_t id, CudaBlasAttribute attribute) noexcept {
    auto& state = *implementation_;
    const auto checked = state.check_context();
    if (checked != CUDA_SUCCESS) return blas_gate(rejected(checked));
    if (state.is_faulted) return blas_gate(rejected(CUDA_ERROR_UNKNOWN));
    const auto handle = std::ranges::find(state.storage->blas, id, &Implementation::Blas::id);
    if (handle == state.storage->blas.end()) return blas_gate(rejected(CUDA_ERROR_INVALID_HANDLE));
    CudaSessionBlasResult result;
    const auto invalid_observation = [&] {
        state.fault();
        result.status = CUDA_ERROR_UNKNOWN;
        result.disposition = CudaSessionDisposition::kRejected;
        return result;  // Preserve the actual successful SDK return separately.
    };
    if (attribute == CudaBlasAttribute::kStream) {
        if (!state.dispatch.blas.get_stream) return blas_gate(rejected(CUDA_ERROR_NOT_SUPPORTED));
        std::optional<NativeExecutionScope> proxy_scope;
        if (state.stream_config.uses_default_proxy)
            proxy_scope.emplace(state.is_native_issue_active);
        cudaStream_t stream{};
        result =
            blas_completed(state.invoke(state.dispatch.blas.get_stream, handle->native, &stream));
        if (state.stream_config.uses_default_proxy && state.check_context(true) != CUDA_SUCCESS)
            return invalid_observation();
        if (result.original_status != CUBLAS_STATUS_SUCCESS) return result;
        if (!handle->stream)
            return reinterpret_cast<CUstream>(stream) == state.storage->default_proxy
                       ? result
                       : invalid_observation();
        if (!stream) return handle->stream ? invalid_observation() : result;
        const auto found =
            std::ranges::find(state.storage->streams, reinterpret_cast<CUstream>(stream),
                              &Implementation::Stream::native);
        if (found == state.storage->streams.end() || found->id != handle->stream)
            return invalid_observation();
        result.value = found->id;
    } else if (attribute == CudaBlasAttribute::kPointerMode) {
        if (!state.dispatch.blas.get_pointer_mode)
            return blas_gate(rejected(CUDA_ERROR_NOT_SUPPORTED));
        cublasPointerMode_t mode{};
        result = blas_completed(
            state.invoke(state.dispatch.blas.get_pointer_mode, handle->native, &mode));
        if (result.original_status == CUBLAS_STATUS_SUCCESS) {
            std::uint32_t bits{};
            static_assert(sizeof(mode) == sizeof(bits));
            std::memcpy(&bits, &mode, sizeof(bits));
            if (bits > 1 || bits != static_cast<std::uint32_t>(handle->pointer_mode))
                return invalid_observation();
            result.value = bits;
        }
    } else if (attribute == CudaBlasAttribute::kMathMode) {
        if (!state.dispatch.blas.get_math_mode)
            return blas_gate(rejected(CUDA_ERROR_NOT_SUPPORTED));
        cublasMath_t mode{};
        result =
            blas_completed(state.invoke(state.dispatch.blas.get_math_mode, handle->native, &mode));
        if (result.original_status == CUBLAS_STATUS_SUCCESS) {
            std::uint32_t bits{};
            static_assert(sizeof(mode) == sizeof(bits));
            std::memcpy(&bits, &mode, sizeof(bits));
            if (bits > 31 || bits != handle->math_mode) return invalid_observation();
            result.value = bits;
        }
    } else if (attribute == CudaBlasAttribute::kAtomicsMode) {
        if (!state.dispatch.blas.get_atomics_mode)
            return blas_gate(rejected(CUDA_ERROR_NOT_SUPPORTED));
        cublasAtomicsMode_t mode{};
        result = blas_completed(
            state.invoke(state.dispatch.blas.get_atomics_mode, handle->native, &mode));
        if (result.original_status == CUBLAS_STATUS_SUCCESS) {
            std::uint32_t bits{};
            static_assert(sizeof(mode) == sizeof(bits));
            std::memcpy(&bits, &mode, sizeof(bits));
            if (bits > 1 || bits != static_cast<std::uint32_t>(handle->atomics_mode))
                return invalid_observation();
            result.value = bits;
        }
    } else if (attribute == CudaBlasAttribute::kSmCountTarget) {
        if (!state.dispatch.blas.get_sm_count_target)
            return blas_gate(rejected(CUDA_ERROR_NOT_SUPPORTED));
        int count{};
        result = blas_completed(
            state.invoke(state.dispatch.blas.get_sm_count_target, handle->native, &count));
        if (result.original_status == CUBLAS_STATUS_SUCCESS) {
            if (count < 0 || count != handle->sm_count_target) return invalid_observation();
            result.value = static_cast<std::uint64_t>(count);
        }
    } else if (attribute == CudaBlasAttribute::kVersion) {
        if (!state.dispatch.blas.get_version) return blas_gate(rejected(CUDA_ERROR_NOT_SUPPORTED));
        int version{};
        result =
            blas_completed(state.invoke(state.dispatch.blas.get_version, handle->native, &version));
        if (result.original_status == CUBLAS_STATUS_SUCCESS) {
            if (version < 0) return invalid_observation();
            result.value = static_cast<std::uint64_t>(version);
        }
    } else
        return blas_gate(rejected(CUDA_ERROR_INVALID_VALUE));
    return result;
}
ExecutionAcceptance CudaExecutionSession::accept_sgemm(std::uint64_t sequence,
                                                       const CudaSessionSgemm& problem) noexcept {
    return implementation_->accept_gemm(sequence, problem, nullptr);
}
ExecutionAcceptance CudaExecutionSession::accept_geam(std::uint64_t sequence,
                                                      const CudaSessionGeam& problem) noexcept {
    auto& state = *implementation_;
    auto invalid = ExecutionAcceptance{.status = ExecutionAcceptStatus::kInvalid, .receipt = {}};
    if (!state.owns() || !state.executor || state.is_faulted)
        return {.status = ExecutionAcceptStatus::kQuarantined, .receipt = {}};
    if (state.is_native_issue_active || state.is_folded_active || state.is_rnn_query_active)
        return invalid;
    if (state.is_closing) return {.status = ExecutionAcceptStatus::kClosed, .receipt = {}};
    const NativeExecutionScope active(state.is_native_issue_active);
    if (state.check_context(true) != CUDA_SUCCESS) {
        state.fault();
        return invalid;
    }
    const auto& sdk = state.dispatch.blas;
    if ((problem.type != CUDA_R_32F && problem.type != CUDA_R_64F) || problem.transpose_a > 3 ||
        problem.transpose_b > 3 || problem.pointer_mode > 1 || problem.presence > 31 ||
        !sdk.get_stream || !sdk.get_pointer_mode ||
        (problem.type == CUDA_R_32F ? !sdk.sgeam : !sdk.dgeam))
        return invalid;
    const auto width = problem.type == CUDA_R_32F ? sizeof(float) : sizeof(double);
    const auto is_zero = [](auto bytes) {
        return std::ranges::all_of(bytes, [](std::byte value) { return value == std::byte{}; });
    };
    const bool is_device = problem.pointer_mode == CUBLAS_POINTER_MODE_DEVICE;
    const std::array<std::span<const std::byte>, 2> coefficients{problem.alpha, problem.beta};
    for (std::size_t index = 0; index < coefficients.size(); ++index) {
        const auto bytes = coefficients[index];
        const auto used = !is_device && (problem.presence & (1U << index)) ? width : 0;
        if (!is_zero(bytes.subspan(used))) return invalid;
        if (!is_device &&
            (problem.buffers[index + 3].allocation || problem.buffers[index + 3].offset))
            return invalid;
    }
    const auto handle =
        std::ranges::find(state.storage->blas, problem.handle, &Implementation::Blas::id);
    if (handle == state.storage->blas.end() || handle->is_uncertain ||
        handle->has_destroy_attempt || !state.workspace_is_valid(*handle) ||
        static_cast<std::uint32_t>(handle->pointer_mode) != problem.pointer_mode)
        return invalid;
    if (handle->stream &&
        std::ranges::find(state.storage->streams, handle->stream, &Implementation::Stream::id) ==
            state.storage->streams.end())
        return invalid;
    std::array<CUdeviceptr, 5> addresses{};
    std::array<std::size_t, 5> remaining{};
    constexpr std::array<unsigned int, 5> presence_bits{2, 3, 4, 0, 1};
    for (std::size_t index = 0; index < problem.buffers.size(); ++index) {
        const auto range = problem.buffers[index];
        if ((!range.allocation && range.offset) ||
            (!(problem.presence & (1U << presence_bits[index])) &&
             (range.allocation || range.offset)))
            return invalid;
        if (!range.allocation) continue;
        const auto allocation = state.find_allocation(range.allocation);
        if (!allocation || range.offset > allocation->bytes ||
            range.offset > std::numeric_limits<CUdeviceptr>::max() - allocation->address)
            return invalid;
        addresses[index] = allocation->address + range.offset;
        remaining[index] = allocation->bytes - range.offset;
    }
    const bool is_a_transposed = problem.transpose_a != CUBLAS_OP_N;
    const bool is_b_transposed = problem.transpose_b != CUBLAS_OP_N;
    const bool has_valid_shape =
        problem.transpose_a <= CUBLAS_OP_C && problem.transpose_b <= CUBLAS_OP_C &&
        problem.rows >= 0 && problem.columns >= 0 &&
        problem.leading_a >= std::max(1, is_a_transposed ? problem.columns : problem.rows) &&
        problem.leading_b >= std::max(1, is_b_transposed ? problem.columns : problem.rows) &&
        problem.leading_c >= std::max(1, problem.rows);
    const bool is_empty = has_valid_shape && (!problem.rows || !problem.columns);
    const bool has_invalid_alias =
        addresses[2] && ((addresses[2] == addresses[0] &&
                          (is_a_transposed || problem.leading_a != problem.leading_c)) ||
                         (addresses[2] == addresses[1] &&
                          (is_b_transposed || problem.leading_b != problem.leading_c)));
    // These exact SDK paths reject or return before reading any scalar/matrix.
    // Unknown client addresses stay NULL; resolved exact aliases stay intact.
    const bool has_no_reads = !has_valid_shape || is_empty || has_invalid_alias ||
                              (is_device && (problem.presence & 3U) != 3U);
    bool requires_version = has_no_reads;
    bool stages_scalars{};
    std::array<bool, 2> safe_inputs{};
    if (!has_no_reads) {
        const std::array extents{
            matrix_bytes({is_a_transposed ? problem.columns : problem.rows,
                          is_a_transposed ? problem.rows : problem.columns, problem.leading_a},
                         width),
            matrix_bytes({is_b_transposed ? problem.columns : problem.rows,
                          is_b_transposed ? problem.rows : problem.columns, problem.leading_b},
                         width),
            matrix_bytes({problem.rows, problem.columns, problem.leading_c}, width)};
        const auto covers = [&](std::size_t index, std::size_t bytes) {
            return addresses[index] && addresses[index] % width == 0 && bytes <= remaining[index] &&
                   bytes <= std::numeric_limits<CUdeviceptr>::max() - addresses[index];
        };
        if (!extents[2] || !covers(2, *extents[2])) return invalid;
        for (std::size_t index = 0; index < safe_inputs.size(); ++index) {
            if (!extents[index] || !covers(index, *extents[index])) continue;
            const bool overlaps = addresses[index] < addresses[2] + *extents[2] &&
                                  addresses[2] < addresses[index] + *extents[index];
            // Invalid exact aliases took the native no-read path above. All
            // remaining equal addresses obey the documented in-place rule.
            safe_inputs[index] = !overlaps || addresses[index] == addresses[2];
        }
        if (is_device) {
            if (!covers(3, width) || !covers(4, width)) return invalid;
            stages_scalars = !safe_inputs[0] || !safe_inputs[1];
            requires_version = stages_scalars;
            if (stages_scalars &&
                (!state.dispatch.download || !state.dispatch.execution.stream_synchronize ||
                 !sdk.set_pointer_mode))
                return invalid;
        } else {
            for (std::size_t index = 0; index < safe_inputs.size(); ++index) {
                const auto& value = index == 0 ? problem.alpha : problem.beta;
                if (!safe_inputs[index] && !geam_scalar_is_zero(problem.type, value))
                    return invalid;
                if (!safe_inputs[index] || !(problem.presence & (1U << index)))
                    requires_version = true;
            }
        }
    }
    if (requires_version && !sdk.get_version) return invalid;
    try {
        auto call = std::make_shared<Implementation::GeamCall>(
            state, problem, addresses, safe_inputs, stages_scalars, requires_version);
        // Session cold gates retain every referenced allocation/module, handle,
        // stream and workspace until this record's real physical retirement.
        return state.executor->accept_library(
            sequence, {.stream = handle->stream, .parameters = {}, .call = std::move(call)});
    } catch (...) {
        return {.status = ExecutionAcceptStatus::kFull, .receipt = {}};
    }
}
template <typename Problem>
ExecutionAcceptance CudaExecutionSession::Implementation::accept_blas_extension(
    std::uint64_t sequence, const Problem& problem) noexcept {
    auto invalid = ExecutionAcceptance{.status = ExecutionAcceptStatus::kInvalid, .receipt = {}};
    if (!owns() || !executor || is_faulted)
        return {.status = ExecutionAcceptStatus::kQuarantined, .receipt = {}};
    if (is_native_issue_active || is_folded_active || is_rnn_query_active) return invalid;
    if (is_closing) return {.status = ExecutionAcceptStatus::kClosed, .receipt = {}};
    const NativeExecutionScope active(is_native_issue_active);
    if (check_context(true) != CUDA_SUCCESS) {
        fault();
        return invalid;
    }
    if (problem.pointer_mode > 1 || problem.presence > 31 || !dispatch.blas.get_stream ||
        !dispatch.blas.get_pointer_mode)
        return invalid;
    const auto handle = std::ranges::find(storage->blas, problem.handle, &Blas::id);
    if (handle == storage->blas.end() || handle->is_uncertain || handle->has_destroy_attempt ||
        !workspace_is_valid(*handle) ||
        static_cast<std::uint32_t>(handle->pointer_mode) != problem.pointer_mode ||
        (handle->stream && std::ranges::find(storage->streams, handle->stream, &Stream::id) ==
                               storage->streams.end()))
        return invalid;
    const bool is_device = problem.pointer_mode == CUBLAS_POINTER_MODE_DEVICE;
    const auto is_zero = [](std::span<const std::byte> bytes) {
        return std::ranges::all_of(bytes, [](std::byte value) { return value == std::byte{}; });
    };
    for (std::size_t index = 0; index != 2; ++index) {
        const auto& scalar = index ? problem.beta : problem.alpha;
        const auto used = !is_device && (problem.presence & (1U << index)) ? sizeof(float) : 0;
        if (!is_zero(std::span{scalar}.subspan(used)) ||
            (!is_device &&
             (problem.buffers[index + 3].allocation || problem.buffers[index + 3].offset)))
            return invalid;
    }
    // Optional operands do not turn stale or foreign identities into NULL.
    // Resolve every supplied allocation before applying access exceptions.
    std::array<CUdeviceptr, 5> addresses{};
    std::array<std::size_t, 5> remaining{};
    constexpr std::array<std::uint32_t, 5> bits{4, 8, 16, 1, 2};
    for (std::size_t index = 0; index != problem.buffers.size(); ++index) {
        const auto range = problem.buffers[index];
        if ((!range.allocation &&
             (range.offset || ((problem.presence & bits[index]) && (index < 3 || is_device)))) ||
            (!(problem.presence & bits[index]) && (range.allocation || range.offset)))
            return invalid;
        if (!range.allocation) continue;
        const auto allocation = find_allocation(range.allocation);
        if (!allocation || range.offset > allocation->bytes ||
            range.offset > std::numeric_limits<CUdeviceptr>::max() - allocation->address)
            return invalid;
        addresses[index] = allocation->address + range.offset;
        remaining[index] = allocation->bytes - range.offset;
    }
    // Candidate predicates require independent API/mode/pointer-pattern access
    // qualification, not GEAM's no-read rules or merely an unchanged output.
    bool metadata_only{}, zero_depth{};
    std::array<std::size_t, 5> alignments{4, 4, 4, 4, 4};
    std::array<std::optional<std::size_t>, 5> extents;
    extents.fill(0);
    if constexpr (std::is_same_v<Problem, CudaSessionSgemv>) {
        if (!dispatch.blas.sgemv || problem.transpose > CUBLAS_OP_CONJG) return invalid;
        metadata_only = problem.rows <= 0 || problem.columns <= 0;
        if (!metadata_only) {
            // Actual SDK access controls refute a no-read permission for these
            // positive-dimension errors; they must retain strict geometry.
            if (problem.transpose > CUBLAS_OP_C || problem.leading_a < std::max(1, problem.rows) ||
                !problem.increment_x || !problem.increment_y)
                return invalid;
            extents[0] = matrix_bytes({problem.rows, problem.columns, problem.leading_a});
            const bool is_transposed = problem.transpose != CUBLAS_OP_N;
            extents[1] =
                vector_bytes(is_transposed ? problem.rows : problem.columns, problem.increment_x);
            extents[2] =
                vector_bytes(is_transposed ? problem.columns : problem.rows, problem.increment_y);
        }
    } else {
        if (!dispatch.blas.sgemm_ex || problem.transpose_a > CUBLAS_OP_CONJG ||
            problem.transpose_b > CUBLAS_OP_CONJG || problem.type_a > CUDA_R_8F_E5M2 ||
            problem.type_b > CUDA_R_8F_E5M2 || problem.type_c > CUDA_R_8F_E5M2)
            return invalid;
        const bool transposed_a = problem.transpose_a != CUBLAS_OP_N;
        const bool transposed_b = problem.transpose_b != CUBLAS_OP_N;
        metadata_only =
            problem.rows <= 0 || problem.columns <= 0 || problem.depth < 0 ||
            problem.transpose_a == CUBLAS_OP_CONJG || problem.transpose_b == CUBLAS_OP_CONJG ||
            problem.leading_a < std::max(1, transposed_a ? problem.depth : problem.rows) ||
            problem.leading_b < std::max(1, transposed_b ? problem.columns : problem.depth) ||
            problem.leading_c < std::max(1, problem.rows);
        zero_depth = !metadata_only && !problem.depth;
        if (!metadata_only) {
            const std::array types{static_cast<cudaDataType>(problem.type_a),
                                   static_cast<cudaDataType>(problem.type_b),
                                   static_cast<cudaDataType>(problem.type_c)};
            std::array<std::size_t, 3> widths{};
            for (std::size_t index = 0; index < types.size(); ++index) {
                widths[index] = gemm_element_bytes(types[index]);
                if (!widths[index]) return invalid;
                alignments[index] = widths[index] / (gemm_is_complex(types[index]) ? 2U : 1U);
            }
            if (!zero_depth) {
                extents[0] =
                    matrix_bytes({transposed_a ? problem.depth : problem.rows,
                                  transposed_a ? problem.rows : problem.depth, problem.leading_a},
                                 widths[0]);
                extents[1] = matrix_bytes(
                    {transposed_b ? problem.columns : problem.depth,
                     transposed_b ? problem.depth : problem.columns, problem.leading_b},
                    widths[1]);
            }
            extents[2] =
                matrix_bytes({problem.rows, problem.columns, problem.leading_c}, widths[2]);
        }
    }
    // Native access qualification covers grouped scalar/data presence only.
    // Mixed presence remains a policy gap, not an invented original SDK error.
    if (metadata_only && problem.presence != 0 && problem.presence != 3 && problem.presence != 28 &&
        problem.presence != 31)
        return invalid;
    const bool requires_version = metadata_only || zero_depth;
    if (requires_version && !dispatch.blas.get_version) return invalid;
    if (!metadata_only) {
        if (zero_depth && problem.presence != 19 && problem.presence != 31) return invalid;
        const auto required_presence = zero_depth ? 19U : 31U;
        if ((problem.presence & required_presence) != required_presence) return invalid;
        extents[3] = extents[4] = is_device ? sizeof(float) : 0;
    }
    for (std::size_t index = 0; index != extents.size(); ++index) {
        if (!extents[index]) return invalid;
        if (!*extents[index]) continue;
        if (!addresses[index] || *extents[index] > remaining[index] ||
            addresses[index] % alignments[index] ||
            *extents[index] > std::numeric_limits<CUdeviceptr>::max() - addresses[index])
            return invalid;
    }
    const auto overlaps = [](CUdeviceptr first, std::size_t first_bytes, CUdeviceptr second,
                             std::size_t second_bytes) {
        return first_bytes && second_bytes && first < second + second_bytes &&
               second < first + first_bytes;
    };
    for (std::size_t index = 0; index < addresses.size(); ++index)
        if (index != 2 && overlaps(addresses[2], *extents[2], addresses[index], *extents[index]))
            return invalid;
    if (handle->has_workspace && handle->workspace_bytes) {
        const auto allocation = find_allocation(handle->workspace.allocation);
        if (!allocation) return invalid;
        const auto address = allocation->address + handle->workspace.offset;
        if (handle->workspace_bytes > std::numeric_limits<CUdeviceptr>::max() - address)
            return invalid;
        for (std::size_t index = 0; index < addresses.size(); ++index)
            if (overlaps(address, handle->workspace_bytes, addresses[index], *extents[index]))
                return invalid;
    }
    try {
        // Fixed-size snapshots are bounded by max_commands. The dispatcher
        // captures logical dependencies and retains Storage through physical
        // retirement; ACK cannot open cold resource/mode/workspace mutation.
        return executor->accept_library(sequence,
                                        {.stream = handle->stream,
                                         .parameters = {},
                                         .call = std::make_shared<BlasExtensionCall<Problem>>(
                                             *this, problem, addresses, requires_version)});
    } catch (...) {
        return {.status = ExecutionAcceptStatus::kOutOfMemory, .receipt = {}};
    }
}
ExecutionAcceptance CudaExecutionSession::accept_sgemv(std::uint64_t sequence,
                                                       const CudaSessionSgemv& problem) noexcept {
    return implementation_->accept_blas_extension(sequence, problem);
}
ExecutionAcceptance CudaExecutionSession::accept_sgemm_ex(
    std::uint64_t sequence, const CudaSessionSgemmEx& problem) noexcept {
    return implementation_->accept_blas_extension(sequence, problem);
}
ExecutionAcceptance CudaExecutionSession::accept_lt_matmul(
    std::uint64_t sequence, const CudaSessionLtMatmul& problem) noexcept {
    auto& state = *implementation_;
    auto invalid = ExecutionAcceptance{.status = ExecutionAcceptStatus::kInvalid, .receipt = {}};
    if (!state.owns() || !state.executor || state.is_faulted)
        return {.status = ExecutionAcceptStatus::kQuarantined, .receipt = {}};
    if (state.is_closing) return {.status = ExecutionAcceptStatus::kClosed, .receipt = {}};
    if (state.is_native_issue_active || state.is_folded_active || state.is_rnn_query_active)
        return invalid;
    const NativeExecutionScope active(state.is_native_issue_active);
    if (state.check_context(true) != CUDA_SUCCESS) return invalid;
    bool has_context_loss{};
    const auto proof = [&] {
        if (has_context_loss) return false;
        has_context_loss = state.check_context(true) != CUDA_SUCCESS;
        if (has_context_loss) state.fault();
        return !has_context_loss;
    };
    const auto& descriptors = problem.descriptors;
    auto* handle = state.lt_handle(descriptors.handle);
    const auto& sdk = state.dispatch.lt;
    if (!handle || handle->is_uncertain || handle->has_destroy_attempt || !sdk.matmul ||
        !sdk.version || !descriptors.operation.native || !descriptors.operation.lifetime ||
        descriptors.operation.kind != CudaLtObjectKind::kOperation)
        return invalid;
    if (descriptors.algorithm) {
        if (!proof()) return invalid;
        const auto version = state.invoke(sdk.version);
        if (!proof() || handle->sdk_version != version ||
            !std::ranges::any_of(handle->algorithms, [&](const auto& known) {
                return std::memcmp(&known, &*descriptors.algorithm, sizeof(known)) == 0;
            }))
            return invalid;
    }
    std::array<LtMatrixState, 4> matrices;
    std::array<CUdeviceptr, 7> addresses{};
    for (std::size_t index = 0; index < matrices.size(); ++index) {
        if (!lt_matrix_state(sdk, descriptors.matrices[index], index == 3, matrices[index], proof))
            return invalid;
        const auto memory = device_memory(problem.matrices[index], matrices[index].bytes);
        if (!memory) return invalid;
        addresses[index] = memory->address;
    }
    cudaDataType scale{};
    cublasLtPointerMode_t pointer_mode{};
    cublasLtEpilogue_t epilogue{};
    if (!lt_property(sdk, descriptors.operation, CUBLASLT_MATMUL_DESC_SCALE_TYPE, scale, proof) ||
        !lt_property(sdk, descriptors.operation, CUBLASLT_MATMUL_DESC_POINTER_MODE, pointer_mode,
                     proof) ||
        !lt_property(sdk, descriptors.operation, CUBLASLT_MATMUL_DESC_EPILOGUE, epilogue, proof))
        return invalid;
    const auto width = gemm_element_bytes(scale);
    if (!width || width > problem.alpha.size() || problem.scalar_bytes != width ||
        (pointer_mode != CUBLASLT_POINTER_MODE_HOST &&
         pointer_mode != CUBLASLT_POINTER_MODE_DEVICE) ||
        problem.uses_device_scalars != (pointer_mode == CUBLASLT_POINTER_MODE_DEVICE))
        return invalid;
    if (problem.uses_device_scalars) {
        for (std::size_t index = 0; index < 2; ++index) {
            const auto memory = device_memory(problem.device_scalars[index], width);
            if (!memory) return invalid;
            addresses[4 + index] = memory->address;
        }
    }
    if (problem.workspace_bytes) {
        const auto memory = device_memory(problem.workspace, problem.workspace_bytes);
        if (!memory) return invalid;
        addresses[6] = memory->address;
    } else if (problem.workspace.allocation) {
        const auto memory = device_memory(problem.workspace, 0);
        if (!memory) return invalid;
        addresses[6] = memory->address;
    }
    if (epilogue != CUBLASLT_EPILOGUE_DEFAULT && epilogue != CUBLASLT_EPILOGUE_BIAS &&
        epilogue != CUBLASLT_EPILOGUE_RELU && epilogue != CUBLASLT_EPILOGUE_RELU_BIAS &&
        epilogue != CUBLASLT_EPILOGUE_GELU && epilogue != CUBLASLT_EPILOGUE_GELU_BIAS)
        return invalid;
    CUdeviceptr bias_address{};
    cudaDataType bias_type = matrices[3].type;
    std::int64_t bias_stride{};
    if ((static_cast<unsigned int>(epilogue) & CUBLASLT_EPILOGUE_BIAS) != 0) {
        const auto& bias = descriptors.operation.bias;
        std::int32_t type{};
        std::int64_t stride{};
        CUdeviceptr pointer{};
        if (!bias || bias->session != state.identity.client_id ||
            !lt_property(sdk, descriptors.operation, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, type,
                         proof) ||
            !lt_property(sdk, descriptors.operation, CUBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE, stride,
                         proof) ||
            !lt_property(sdk, descriptors.operation, CUBLASLT_MATMUL_DESC_BIAS_POINTER, pointer,
                         proof) ||
            pointer != bias->address || stride < 0)
            return invalid;
        bias_type = type == -1 ? matrices[3].type : static_cast<cudaDataType>(type);
        if (type == -1 && matrices[3].type == CUDA_R_8I && scale == CUDA_R_32F)
            bias_type = CUDA_R_32F;
        const auto bias_width = gemm_element_bytes(bias_type);
        const auto count = static_cast<std::uint64_t>(matrices[3].batches - 1);
        if (!bias_width || matrices[3].rows > SIZE_MAX / bias_width ||
            (count && static_cast<std::uint64_t>(stride) >
                          (SIZE_MAX / bias_width - matrices[3].rows) / count))
            return invalid;
        const auto bytes =
            (matrices[3].rows + count * static_cast<std::uint64_t>(stride)) * bias_width;
        const auto memory =
            device_memory({bias->allocation, static_cast<std::size_t>(bias->offset)}, bytes);
        if (!memory || memory->address != pointer) return invalid;
        bias_address = pointer;
        bias_stride = stride;
    }
    try {
        cublasComputeType_t compute{};
        bool has_direct_reduction = !descriptors.algorithm;
        bool has_split_reduction{};
        std::uint32_t split{}, reduction{};
        if (descriptors.algorithm && sdk.configuration_get) {
            std::size_t split_bytes{}, reduction_bytes{};
            const bool has_configuration =
                state.invoke(sdk.configuration_get, &*descriptors.algorithm,
                             CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &split, sizeof(split),
                             &split_bytes) == CUBLAS_STATUS_SUCCESS &&
                proof() &&
                state.invoke(sdk.configuration_get, &*descriptors.algorithm,
                             CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reduction, sizeof(reduction),
                             &reduction_bytes) == CUBLAS_STATUS_SUCCESS &&
                proof() && split_bytes == sizeof(split) && reduction_bytes == sizeof(reduction);
            has_direct_reduction = has_configuration && split <= 1 && !reduction;
            has_split_reduction = has_configuration && (split == 3 || split == 6) &&
                                  reduction == CUBLASLT_REDUCTION_SCHEME_INPLACE;
        }
        if ((has_direct_reduction || has_split_reduction) && descriptors.algorithm) {
            cublasLtMatmulHeuristicResult_t checked{};
            checked.algo = *descriptors.algorithm;
            const auto layout = [&](std::size_t index) {
                return static_cast<cublasLtMatrixLayout_t>(descriptors.matrices[index].native);
            };
            const bool has_valid_algorithm =
                sdk.algorithm_check &&
                state.invoke(sdk.algorithm_check, handle->native,
                             static_cast<cublasLtMatmulDesc_t>(descriptors.operation.native),
                             layout(0), layout(1), descriptors.has_c_layout ? layout(2) : nullptr,
                             layout(3), &*descriptors.algorithm,
                             &checked) == CUBLAS_STATUS_SUCCESS &&
                proof() && checked.state == CUBLAS_STATUS_SUCCESS &&
                checked.workspaceSize <= problem.workspace_bytes;
            has_direct_reduction &= has_valid_algorithm;
            has_split_reduction &= has_valid_algorithm;
        }
        std::array<cublasOperation_t, 3> transposes{};
        const bool has_operations =
            lt_property(sdk, descriptors.operation, CUBLASLT_MATMUL_DESC_COMPUTE_TYPE, compute,
                        proof) &&
            lt_property(sdk, descriptors.operation, CUBLASLT_MATMUL_DESC_TRANSA, transposes[0],
                        proof) &&
            lt_property(sdk, descriptors.operation, CUBLASLT_MATMUL_DESC_TRANSB, transposes[1],
                        proof) &&
            lt_property(sdk, descriptors.operation, CUBLASLT_MATMUL_DESC_TRANSC, transposes[2],
                        proof);
        if (!proof()) return invalid;
        const bool is_float = matrices[0].type == CUDA_R_32F && matrices[1].type == CUDA_R_32F &&
                              matrices[3].type == CUDA_R_32F;
        const bool is_half = matrices[0].type == CUDA_R_16F && matrices[1].type == CUDA_R_16F &&
                             (matrices[3].type == CUDA_R_16F || matrices[3].type == CUDA_R_32F);
        std::uint32_t reduction_depth{};
        float host_alpha{}, host_beta{};
        std::memcpy(&host_alpha, problem.alpha.data(), sizeof(host_alpha));
        std::memcpy(&host_beta, problem.beta.data(), sizeof(host_beta));
        // This arithmetic is pinned to these exact BERT forward
        // profiles, not a promise about arbitrary split-K SDK algorithms.
        const bool has_pinned_split_profile =
            has_split_reduction && handle->sdk_version == 120604 && is_float && has_operations &&
            !problem.uses_device_scalars && std::isfinite(host_alpha) && host_alpha != 0.0F &&
            host_beta == 0.0F && epilogue == CUBLASLT_EPILOGUE_BIAS && bias_type == CUDA_R_32F &&
            bias_address && bias_stride == 0 && transposes[0] == CUBLAS_OP_T &&
            transposes[1] == CUBLAS_OP_N && transposes[2] == CUBLAS_OP_N &&
            matrices[3].rows == 768 && matrices[3].columns == 768 &&
            ((matrices[0].rows == 768 && split == 6) || (matrices[0].rows == 3072 && split == 3)) &&
            std::ranges::all_of(matrices, [](const auto& matrix) {
                return matrix.order == CUBLASLT_ORDER_COL && matrix.batches == 1 &&
                       matrix.leading == static_cast<std::int64_t>(matrix.rows);
            });
        if (has_pinned_split_profile)
            reduction_depth = static_cast<std::uint32_t>(matrices[0].rows / split);
        const auto disjoint = [](CUdeviceptr left, std::size_t left_bytes, CUdeviceptr right,
                                 std::size_t right_bytes) {
            return left <= right ? right - left >= left_bytes : left - right >= right_bytes;
        };
        const bool safe_output =
            disjoint(addresses[3], matrices[3].bytes, addresses[0], matrices[0].bytes) &&
            disjoint(addresses[3], matrices[3].bytes, addresses[1], matrices[1].bytes) &&
            (disjoint(addresses[3], matrices[3].bytes, addresses[2], matrices[2].bytes) ||
             (addresses[2] == addresses[3] && matrices[2].order == matrices[3].order &&
              matrices[2].leading == matrices[3].leading &&
              matrices[2].stride == matrices[3].stride && transposes[2] == CUBLAS_OP_N));
        bool has_safe_alignment = !addresses[6] || addresses[6] % 256 == 0;
        for (std::size_t operand = 0; operand < 4; ++operand)
            has_safe_alignment =
                has_safe_alignment &&
                addresses[operand] % gemm_element_bytes(matrices[operand].type) == 0;
        if (bias_address)
            has_safe_alignment =
                has_safe_alignment && bias_address % gemm_element_bytes(bias_type) == 0;
        if (problem.uses_device_scalars)
            has_safe_alignment =
                has_safe_alignment && addresses[4] % width == 0 && addresses[5] % width == 0;
        if (state.persistent.replace_sgemm && state.storage->lt_ids.back() && has_operations &&
            (has_direct_reduction || has_pinned_split_profile) && (is_float || is_half) &&
            matrices[2].type == matrices[3].type && matrices[0].leading > 0 &&
            matrices[1].leading > 0 &&
            (epilogue != CUBLASLT_EPILOGUE_DEFAULT || !descriptors.operation.bias) &&
            (epilogue == CUBLASLT_EPILOGUE_DEFAULT || matrices[3].order == CUBLASLT_ORDER_COL) &&
            has_safe_alignment && compute == CUBLAS_COMPUTE_32F && scale == CUDA_R_32F &&
            safe_output &&
            (epilogue == CUBLASLT_EPILOGUE_DEFAULT || epilogue == CUBLASLT_EPILOGUE_BIAS) &&
            (bias_type == matrices[3].type || bias_type == CUDA_R_32F) &&
            std::ranges::all_of(transposes,
                                [](auto value) {
                                    return value == CUBLAS_OP_N || value == CUBLAS_OP_T ||
                                           value == CUBLAS_OP_C;
                                }) &&
            std::ranges::all_of(matrices, [&](const auto& matrix) {
                return matrix.leading <= INT_MAX && matrix.rows <= INT_MAX - 64 &&
                       matrix.columns <= INT_MAX - 64 &&
                       (matrix.batches == 1 || matrix.batches == matrices[3].batches);
            })) {
            const auto rows = transposes[0] == CUBLAS_OP_N ? matrices[0].rows : matrices[0].columns;
            const auto depth =
                transposes[0] == CUBLAS_OP_N ? matrices[0].columns : matrices[0].rows;
            const auto columns =
                transposes[1] == CUBLAS_OP_N ? matrices[1].columns : matrices[1].rows;
            const auto b_depth =
                transposes[1] == CUBLAS_OP_N ? matrices[1].rows : matrices[1].columns;
            const auto c_rows =
                transposes[2] == CUBLAS_OP_N ? matrices[2].rows : matrices[2].columns;
            const auto c_columns =
                transposes[2] == CUBLAS_OP_N ? matrices[2].columns : matrices[2].rows;
            const auto tiles =
                cuda_gemm_matrix_tiles(rows, columns, state.persistent.gemm_tile_side);
            const auto batches = static_cast<std::uint64_t>(matrices[3].batches);
            if (rows != matrices[3].rows || columns != matrices[3].columns || b_depth != depth ||
                c_rows != rows || c_columns != columns || !tiles || tiles > UINT32_MAX / batches)
                return invalid;
            const auto total = tiles * batches;
            const auto free_slot = std::ranges::find(state.storage->occupied, false);
            if (free_slot == state.storage->occupied.end())
                return {.status = ExecutionAcceptStatus::kFull, .receipt = {}};
            const auto variant = is_float ? 0U : matrices[3].type == CUDA_R_16F ? 1U : 2U;
            int capacity{};
            if (state.invoke(state.dispatch.occupancy, &capacity,
                             state.storage->lt_functions[variant], 128, 0) != CUDA_SUCCESS ||
                capacity <= 0)
                return invalid;
            if (!proof()) return invalid;
            const auto index =
                static_cast<std::size_t>(free_slot - state.storage->occupied.begin());
            auto progress = std::make_shared<Implementation::Progress>(
                state.storage, state.dispatch,
                Implementation::ProgressParameters{
                    .slot = index,
                    .sequence = sequence,
                    .total = total,
                    .workers = state.persistent.workers,
                    .observes_block_completion = state.persistent.observes_block_completion,
                    .uses_sharded_host_retreat = state.persistent.uses_sharded_host_retreat},
                std::make_shared<CudaSessionLtMatmul>(problem));
            *free_slot = true;
            std::array<std::byte, 176> parameters{};
            const auto put = [&](std::size_t offset, const auto& value) {
                std::memcpy(parameters.data() + offset, &value, sizeof(value));
            };
            for (std::size_t operand = 0; operand < 4; ++operand)
                put(operand * 8, addresses[operand]);
            put(32, bias_address);
            const std::array dimensions{static_cast<int>(rows),
                                        static_cast<int>(columns),
                                        static_cast<int>(depth),
                                        static_cast<int>(matrices[0].leading),
                                        static_cast<int>(matrices[1].leading),
                                        static_cast<int>(matrices[2].leading),
                                        static_cast<int>(matrices[3].leading)};
            for (std::size_t field = 0; field < dimensions.size(); ++field)
                put(40 + field * 4, dimensions[field]);
            std::uint32_t flags{};
            for (std::size_t operand = 0; operand < 3; ++operand)
                if ((transposes[operand] != CUBLAS_OP_N) !=
                    (matrices[operand].order == CUBLASLT_ORDER_ROW))
                    flags |= 1U << operand;
            if (matrices[3].order == CUBLASLT_ORDER_ROW) flags |= 8U;
            put(68, flags);
            std::memcpy(parameters.data() + 72, problem.alpha.data(), 4);
            std::memcpy(parameters.data() + 76, problem.beta.data(), 4);
            put(80, addresses[4]);
            put(88, addresses[5]);
            put(96, std::uint32_t{problem.uses_device_scalars ? 1U : 0U});
            put(100, std::uint32_t{bias_type == CUDA_R_32F ? 1U : 0U});
            put(104, progress->counter());
            put(112, progress->retreat());
            for (std::size_t operand = 0; operand < 4; ++operand)
                put(120 + operand * 8,
                    matrices[operand].batches == 1 ? std::int64_t{0} : matrices[operand].stride);
            put(152, bias_stride);
            put(160, matrices[3].batches);
            put(164, reduction_depth);
            if (state.persistent.observes_block_completion) put(168, progress->block_completion());
            std::erase_if(state.progress, [](const auto& entry) { return entry.expired(); });
            auto accepted = state.executor->accept_persistent_kernel(
                sequence,
                {.function = state.storage->lt_ids[variant],
                 .stream = problem.stream,
                 .grid = {static_cast<unsigned int>(total), 1, 1},
                 .block = {128, 1, 1},
                 .parameters = {}},
                {state.storage->lt_functions[variant], state.persistent.workers,
                 std::span{parameters}.first(state.persistent.observes_block_completion ? 176
                                                                                        : 168),
                 progress});
            if (accepted.accepted()) {
                state.progress.push_back(progress);
                if (state.persistent_accepted != UINT64_MAX) ++state.persistent_accepted;
            }
            return accepted;
        }
        if (!proof()) return invalid;
        return state.executor->accept_library(
            sequence, {.stream = problem.stream,
                       .parameters = {},
                       .call = std::make_shared<LtMatmulCall>(sdk.matmul, handle->native, problem,
                                                              addresses)});
    } catch (...) {
        return {.status = ExecutionAcceptStatus::kFull, .receipt = {}};
    }
}
ExecutionAcceptance CudaExecutionSession::accept_gemm_ex(
    std::uint64_t sequence, const CudaSessionGemmEx& problem) noexcept {
    return implementation_->accept_gemm(sequence, problem.operation, &problem);
}
ExecutionAcceptance CudaExecutionSession::accept_strided_gemm_ex(
    std::uint64_t sequence, const CudaSessionStridedGemmEx& problem) noexcept {
    return implementation_->accept_gemm(sequence, problem.operation.operation, &problem.operation,
                                        &problem);
}
ExecutionAcceptance CudaExecutionSession::accept_strided_sgemm(
    std::uint64_t sequence, const CudaSessionStridedGemmEx& problem) noexcept {
    const auto& typed = problem.operation;
    if (typed.a_type != CUDA_R_32F || typed.b_type != CUDA_R_32F || typed.c_type != CUDA_R_32F ||
        typed.compute_type != CUBLAS_COMPUTE_32F || typed.algorithm != CUBLAS_GEMM_DEFAULT)
        return {.status = ExecutionAcceptStatus::kInvalid, .receipt = {}};
    return implementation_->accept_gemm(sequence, typed.operation, &typed, &problem, true);
}
ExecutionAcceptance CudaExecutionSession::Implementation::accept_gemm(
    std::uint64_t sequence, const CudaSessionSgemm& problem, const CudaSessionGemmEx* extended,
    const CudaSessionStridedGemmEx* batch, bool uses_sgemm_batch) noexcept {
    auto& state = *this;
    auto invalid = ExecutionAcceptance{.status = ExecutionAcceptStatus::kInvalid, .receipt = {}};
    if (!state.owns() || !state.executor || state.is_faulted)
        return {.status = ExecutionAcceptStatus::kQuarantined, .receipt = {}};
    if (state.is_native_issue_active || state.is_folded_active || state.is_rnn_query_active)
        return invalid;
    if (state.is_closing) return {.status = ExecutionAcceptStatus::kClosed, .receipt = {}};
    const NativeExecutionScope active(state.is_native_issue_active);
    if (state.check_context(true) != CUDA_SUCCESS) return invalid;
    const auto proof = [&] {
        if (state.check_context(true) == CUDA_SUCCESS) return true;
        state.fault();
        return false;
    };
    const auto handle =
        std::ranges::find(state.storage->blas, problem.handle, &Implementation::Blas::id);
    if (handle == state.storage->blas.end() || handle->is_uncertain ||
        !state.workspace_is_valid(*handle) ||
        (batch      ? (uses_sgemm_batch ? !state.dispatch.blas.strided_sgemm
                                        : !state.dispatch.blas.strided_gemm_ex)
         : extended ? !state.dispatch.blas.gemm_ex
                    : !state.dispatch.blas.sgemm) ||
        (batch && (batch->batches <= 0 ||
                   (batch->batches > 1 &&
                    std::ranges::any_of(batch->strides, [](auto stride) { return stride < 0; })))))
        return invalid;
    const std::array types{extended ? extended->a_type : CUDA_R_32F,
                           extended ? extended->b_type : CUDA_R_32F,
                           extended ? extended->c_type : CUDA_R_32F};
    const std::array widths{gemm_element_bytes(types[0]), gemm_element_bytes(types[1]),
                            gemm_element_bytes(types[2])};
    const auto scalar_bytes =
        extended ? gemm_scalar_bytes(extended->compute_type, extended->c_type) : sizeof(float);
    if (!scalar_bytes || std::ranges::any_of(widths, [](auto width) { return width == 0; }) ||
        (extended && (extended->algorithm < CUBLAS_GEMM_DEFAULT || extended->algorithm > 127)))
        return invalid;
    const auto valid_transpose = [](auto value) {
        return value == CUBLAS_OP_N || value == CUBLAS_OP_T || value == CUBLAS_OP_C;
    };
    if (!valid_transpose(problem.transpose_a) || !valid_transpose(problem.transpose_b))
        return invalid;
    const bool is_a_transposed = problem.transpose_a != CUBLAS_OP_N;
    const bool is_b_transposed = problem.transpose_b != CUBLAS_OP_N;
    const std::array extents{
        matrix_bytes({is_a_transposed ? problem.depth : problem.rows,
                      is_a_transposed ? problem.rows : problem.depth, problem.leading_a},
                     widths[0]),
        matrix_bytes({is_b_transposed ? problem.columns : problem.depth,
                      is_b_transposed ? problem.depth : problem.columns, problem.leading_b},
                     widths[1]),
        matrix_bytes({problem.rows, problem.columns, problem.leading_c}, widths[2])};
    const std::array ranges{problem.a, problem.b, problem.c};
    if (problem.uses_device_scalars != (handle->pointer_mode == CUBLAS_POINTER_MODE_DEVICE))
        return invalid;
    std::array<CUdeviceptr, 5> addresses{};
    bool is_disjoint_batch = true;
    bool is_signed_indexable = true;
    for (std::size_t index = 0; index < ranges.size(); ++index) {
        if (!extents[index]) return invalid;
        const auto found = state.find_allocation(ranges[index].allocation);
        if (!found || ranges[index].offset > found->bytes ||
            *extents[index] > found->bytes - ranges[index].offset ||
            ranges[index].offset > std::numeric_limits<CUdeviceptr>::max() - found->address)
            return invalid;
        addresses[index] = found->address + ranges[index].offset;
        const auto alignment = widths[index] / (gemm_is_complex(types[index]) ? 2U : 1U);
        if (addresses[index] % alignment) return invalid;
        if (batch && batch->batches > 1) {
            const auto stride = batch->strides[index];
            // This unsigned magnitude also handles INT64_MIN without signed overflow.
            const auto magnitude = stride < 0 ? static_cast<std::uint64_t>(-(stride + 1)) + 1
                                              : static_cast<std::uint64_t>(stride);
            const auto intervals = static_cast<std::uint64_t>(batch->batches - 1);
            const auto maximum = std::numeric_limits<std::uint64_t>::max();
            if (magnitude > maximum / widths[index]) return invalid;
            const auto stride_bytes = magnitude * widths[index];
            if (stride_bytes > maximum / intervals) return invalid;
            const auto span = stride_bytes * intervals;
            if (stride < 0 ? span > ranges[index].offset
                           : span > found->bytes - ranges[index].offset - *extents[index])
                return invalid;
            is_signed_indexable =
                is_signed_indexable &&
                magnitude <= static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) /
                                 intervals;
            if (index == 2) is_disjoint_batch = stride_bytes >= *extents[index];
        }
    }
    if (problem.uses_device_scalars) {
        const std::array scalars{problem.device_alpha, problem.device_beta};
        for (std::size_t index = 0; index < scalars.size(); ++index) {
            const auto found = state.find_allocation(scalars[index].allocation);
            if (!found || scalars[index].offset > found->bytes ||
                scalar_bytes > found->bytes - scalars[index].offset ||
                scalars[index].offset > std::numeric_limits<CUdeviceptr>::max() - found->address)
                return invalid;
            addresses[index + 3] = found->address + scalars[index].offset;
            const auto alignment = scalar_bytes / (gemm_is_complex(types[2]) ? 2U : 1U);
            if (addresses[index + 3] % alignment) return invalid;
        }
    } else if (problem.device_alpha.allocation || problem.device_alpha.offset ||
               problem.device_beta.allocation || problem.device_beta.offset) {
        return invalid;
    }
    try {
        // Selection precedes queue admission. No original library call is made
        // for this path; the returned host result is the actual Driver launch.
        const auto matrix_tiles = cuda_gemm_matrix_tiles(
            static_cast<std::uint64_t>(problem.rows), static_cast<std::uint64_t>(problem.columns),
            state.persistent.gemm_tile_side);
        const auto batches = batch ? static_cast<std::uint64_t>(batch->batches) : 1;
        if (matrix_tiles > std::numeric_limits<std::uint64_t>::max() / batches) return invalid;
        const auto total = matrix_tiles * batches;
        const bool is_default_ex =
            !extended || (extended->compute_type == CUBLAS_COMPUTE_32F &&
                          (extended->algorithm == CUBLAS_GEMM_DEFAULT ||
                           extended->algorithm == CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        const bool is_float =
            types[0] == CUDA_R_32F && types[1] == CUDA_R_32F && types[2] == CUDA_R_32F;
        const bool is_half = types[0] == CUDA_R_16F && types[1] == CUDA_R_16F &&
                             (types[2] == CUDA_R_16F || types[2] == CUDA_R_32F);
        bool is_indexable = true;
        for (std::size_t index = 0; index < extents.size(); ++index)
            is_indexable =
                is_indexable && *extents[index] / widths[index] <=
                                    static_cast<std::size_t>(std::numeric_limits<int>::max());
        bool is_replaced =
            is_default_ex && (is_float || is_half) && is_disjoint_batch && is_signed_indexable &&
            // Zero input strides reuse read-only matrices. Disjoint output
            // hulls above still exclude overlapping C from private execution.
            (!batch || batch->batches == 1 ||
             std::ranges::all_of(batch->strides, [](auto stride) { return stride >= 0; })) &&
            state.persistent.replace_sgemm && handle->math_mode == 0 &&
            handle->sm_count_target == 0 &&
            (!handle->has_workspace || handle->workspace_bytes >= std::size_t{16} * 1024) &&
            state.storage->gemm_ids.back() &&
            problem.rows <= std::numeric_limits<int>::max() - 64 &&
            problem.columns <= std::numeric_limits<int>::max() - 64 &&
            problem.depth <= std::numeric_limits<int>::max() - 8 &&
            total <= std::numeric_limits<std::uint32_t>::max() && is_indexable;
        float alpha = problem.alpha, beta = problem.beta;
        if (extended && is_float && is_default_ex) {
            std::memcpy(&alpha, extended->alpha.data(), sizeof(alpha));
            std::memcpy(&beta, extended->beta.data(), sizeof(beta));
        }
        const bool is_input_gradient =
            !batch && is_float && is_default_ex && !problem.uses_device_scalars &&
            problem.transpose_a == CUBLAS_OP_N && problem.transpose_b == CUBLAS_OP_N &&
            problem.rows == 768 && problem.columns == 768 && problem.depth == 3072 &&
            problem.leading_a == 768 && problem.leading_b == 3072 && problem.leading_c == 768 &&
            alpha == 1.0F && beta == 0.0F;
        bool uses_split_reduction{};
        if (is_replaced && is_input_gradient) {
            // This arithmetic is qualified for the pinned SDK, not for
            // every library algorithm. A private query failure keeps the
            // original route and never becomes an original compute result.
            if (state.dispatch.blas.get_version) {
                if (!proof()) return invalid;
                int version{};
                const auto status =
                    state.invoke(state.dispatch.blas.get_version, handle->native, &version);
                if (!proof()) return invalid;
                uses_split_reduction = status == CUBLAS_STATUS_SUCCESS && version == 120604;
            }
            is_replaced = uses_split_reduction;
        }
        if (is_replaced) {
            const auto free_slot = std::ranges::find(state.storage->occupied, false);
            if (free_slot == state.storage->occupied.end())
                return {.status = ExecutionAcceptStatus::kFull, .receipt = {}};
            const auto variant = uses_split_reduction
                                     ? 24U
                                     : (is_a_transposed ? 2U : 0U) + (is_b_transposed ? 1U : 0U) +
                                           (is_half ? (types[2] == CUDA_R_16F ? 4U : 8U) : 0U) +
                                           (batch ? 12U : 0U);
            if (!proof()) return invalid;
            int capacity{};
            const auto occupancy = state.invoke(state.dispatch.occupancy, &capacity,
                                                state.storage->gemm_functions[variant], 128, 0);
            if (!proof() || occupancy != CUDA_SUCCESS || capacity <= 0) return invalid;
            const auto index =
                static_cast<std::size_t>(free_slot - state.storage->occupied.begin());
            auto progress = std::make_shared<Implementation::Progress>(
                state.storage, state.dispatch,
                Implementation::ProgressParameters{
                    .slot = index,
                    .sequence = sequence,
                    .total = total,
                    .workers = state.persistent.workers,
                    .observes_block_completion = state.persistent.observes_block_completion,
                    .uses_sharded_host_retreat = state.persistent.uses_sharded_host_retreat});
            *free_slot = true;
            std::array<std::byte, 136> parameters{};
            const auto put = [&parameters](std::size_t offset, const auto& value) {
                std::memcpy(parameters.data() + offset, &value, sizeof(value));
            };
            for (std::size_t operand = 0; operand < 3; ++operand)
                put(operand * 8, addresses[operand]);
            const std::array dimensions{problem.rows,      problem.columns,   problem.depth,
                                        problem.leading_a, problem.leading_b, problem.leading_c};
            for (std::size_t axis = 0; axis < dimensions.size(); ++axis)
                put(24 + axis * 4, dimensions[axis]);
            put(48, problem.alpha);
            put(52, problem.beta);
            if (extended) {
                std::memcpy(parameters.data() + 48, extended->alpha.data(), sizeof(float));
                std::memcpy(parameters.data() + 52, extended->beta.data(), sizeof(float));
            }
            put(56, addresses[3]);
            put(64, addresses[4]);
            const std::uint32_t device_scalars = problem.uses_device_scalars ? 1U : 0U;
            put(72, device_scalars);
            put(80, progress->counter());
            put(88, progress->retreat());
            if (batch) {
                for (std::size_t operand = 0; operand < batch->strides.size(); ++operand)
                    put(96 + operand * 8, batch->strides[operand]);
                put(120, batch->batches);
            } else if (uses_split_reduction) {
                put(96, std::uint32_t{768});
            }
            const std::size_t ordinary_bytes = batch ? 128 : uses_split_reduction ? 104 : 96;
            if (state.persistent.observes_block_completion)
                put(ordinary_bytes, progress->block_completion());
            std::erase_if(state.progress, [](const auto& entry) { return entry.expired(); });
            auto accepted = state.executor->accept_persistent_kernel(
                sequence,
                {.function = state.storage->gemm_ids[variant],
                 .stream = handle->stream,
                 .grid = {static_cast<unsigned int>(total), 1, 1},
                 .block = {128, 1, 1},
                 .parameters = {}},
                {state.storage->gemm_functions[variant], state.persistent.workers,
                 std::span{parameters}.first(ordinary_bytes +
                                             (state.persistent.observes_block_completion ? 8 : 0)),
                 progress});
            if (accepted.accepted()) {
                state.progress.push_back(progress);
                if (state.persistent_accepted != std::numeric_limits<std::uint64_t>::max())
                    ++state.persistent_accepted;
            }
            return accepted;
        }
        if (batch && uses_sgemm_batch)
            return state.executor->accept_library(
                sequence,
                {.stream = handle->stream,
                 .parameters = {},
                 .call = std::make_shared<StridedSgemmCall>(state.dispatch.blas.strided_sgemm,
                                                            handle->native, *batch, addresses)});
        if (batch)
            return state.executor->accept_library(
                sequence,
                {.stream = handle->stream,
                 .parameters = {},
                 .call = std::make_shared<StridedGemmExCall>(state.dispatch.blas.strided_gemm_ex,
                                                             handle->native, *batch, addresses)});
        if (extended)
            return state.executor->accept_library(
                sequence, {.stream = handle->stream,
                           .parameters = {},
                           .call = std::make_shared<GemmExCall>(
                               state.dispatch.blas.gemm_ex, handle->native, *extended, addresses)});
        return state.executor->accept_library(
            sequence, {.stream = handle->stream,
                       .parameters = {},
                       .call = std::make_shared<SgemmCall>(state.dispatch.blas.sgemm,
                                                           handle->native, problem, addresses)});
    } catch (...) {
        return {.status = ExecutionAcceptStatus::kOutOfMemory, .receipt = {}};
    }
}

}  // namespace glimmer::backend::cuda
