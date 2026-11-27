#include "internal/driver_api_interceptor_state.h"

namespace glimmer::interceptor {

CUresult allocate_cooperative_progress(CUdeviceptr* pointer, std::size_t memory_bytes) noexcept {
    if (!pointer || memory_bytes == 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *pointer = 0;
    InterceptorState* state = nullptr;
    try {
        state = &get_state();
        return intercept_tracked_allocation(
            pointer, memory_bytes, [&] { return state->driver.mem_alloc(pointer, memory_bytes); },
            true, AllocationProvenance::kIndependentDevice);
    } catch (...) {
        if (state) {
            state->allocations.mark_accounting_degraded();
        }
        return CUDA_ERROR_UNKNOWN;
    }
}

CUresult cleanup_async_allocation(InterceptorState& state, CUdeviceptr device_pointer,
                                  CUstream stream, bool per_thread_default_stream) {
    const CUresult free_result = per_thread_default_stream
                                     ? state.driver.mem_free_async_ptsz(device_pointer, stream)
                                     : state.driver.mem_free_async(device_pointer, stream);
    if (free_result != CUDA_SUCCESS) {
        return free_result;
    }
    return per_thread_default_stream ? state.driver.stream_synchronize_ptsz(stream)
                                     : state.driver.stream_synchronize(stream);
}

template <typename AllocateFunction>
CUresult intercept_async_tracked_allocation(CUdeviceptr* device_pointer, MemoryBytes memory_bytes,
                                            CUstream stream, bool per_thread_default_stream,
                                            AllocateFunction&& allocate) {
    InterceptorState& state = get_state();
    if (device_pointer == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (glimmer::interceptor::is_inside_driver_call() ||
        glimmer::interceptor::is_inside_runtime_call()) {
        return std::forward<AllocateFunction>(allocate)();
    }
    if (!ensure_initialized(state)) {
        return CUDA_ERROR_UNKNOWN;
    }

    if (state.quota_mode == QuotaMode::kDisabled) {
        return std::forward<AllocateFunction>(allocate)();
    }
    if (state.quota_mode == QuotaMode::kInvalidConfiguration) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (state.allocations.is_accounting_degraded() || state.quota == nullptr ||
        !state.quota->is_healthy()) {
        return CUDA_ERROR_UNKNOWN;
    }
    if (per_thread_default_stream ? !state.driver.has_mem_free_async_ptsz()
                                  : !state.driver.has_mem_free_async()) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    const std::optional<ContextIdentity> stream_identity =
        capture_stream_identity(state, stream, per_thread_default_stream);
    if (!stream_identity.has_value()) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }

    auto reservation =
        state.quota->try_reserve(static_cast<DeviceId>(stream_identity->device), memory_bytes);
    if (!reservation.has_value()) {
        return state.quota->is_healthy() ? CUDA_ERROR_OUT_OF_MEMORY : CUDA_ERROR_UNKNOWN;
    }

    const CUresult allocation_result = std::forward<AllocateFunction>(allocate)();
    if (allocation_result != CUDA_SUCCESS) {
        return allocation_result;
    }
    if (device_pointer == nullptr || *device_pointer == 0) {
        // A successful external allocation with no pointer is ambiguous. The
        // reservation must remain charged because there is no safe pointer to
        // pass to the asynchronous free path.
        reservation->abandon();
        state.allocations.mark_accounting_degraded();
        return CUDA_ERROR_UNKNOWN;
    }

    if (!reservation->commit()) {
        const CUresult cleanup_result =
            cleanup_async_allocation(state, *device_pointer, stream, per_thread_default_stream);
        if (cleanup_result != CUDA_SUCCESS) {
            reservation->abandon();
        }
        state.allocations.mark_accounting_degraded();
        return cleanup_result != CUDA_SUCCESS ? cleanup_result : CUDA_ERROR_UNKNOWN;
    }

    const auto identity = make_allocation_identity(*device_pointer, *stream_identity);
    if (!state.allocations.record(identity, memory_bytes, AllocationScope::kContextIndependent)) {
        const CUresult cleanup_result =
            cleanup_async_allocation(state, *device_pointer, stream, per_thread_default_stream);
        state.allocations.mark_accounting_degraded();
        if (cleanup_result != CUDA_SUCCESS) {
            return cleanup_result;
        }
        if (state.quota == nullptr ||
            !state.quota->release(static_cast<DeviceId>(stream_identity->device), memory_bytes)) {
            return CUDA_ERROR_UNKNOWN;
        }
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    return CUDA_SUCCESS;
}

void finalize_async_release(InterceptorState& state, DeviceId device, MemoryBytes released_bytes) {
    if (released_bytes == 0 || state.allocations.is_accounting_degraded()) {
        return;
    }
    if (state.quota == nullptr || !state.quota->release(device, released_bytes)) {
        state.allocations.mark_accounting_degraded();
    }
}

void finalize_async_releases(
    InterceptorState& state,
    const std::vector<AllocationRegistry::DeviceReleaseSummary>& summaries) {
    for (const auto& summary : summaries) {
        finalize_async_release(state, static_cast<DeviceId>(summary.device), summary.memory_bytes);
    }
}

[[nodiscard]] AsyncStreamIdentity make_runtime_async_stream_identity(
    cudaStream_t stream, const ContextIdentity& context) {
    const CUstream driver_stream = reinterpret_cast<CUstream>(stream);
    // Runtime does not expose whether a null stream is legacy or per-thread.
    // Treat it as per-thread to keep records from different calling threads
    // isolated; explicit streams are identified by their owning context.
    return make_async_stream_identity(driver_stream, context, stream == nullptr);
}

[[nodiscard]] cudaError_t cleanup_runtime_async_allocation(
    void* device_pointer, cudaStream_t stream, RuntimeFreeAsyncFunction release,
    RuntimeDeviceSynchronizeFunction synchronize) {
    if (release == nullptr) {
        return cudaErrorNotSupported;
    }
    const cudaError_t free_result = release(device_pointer, stream);
    if (free_result != cudaSuccess) {
        return free_result;
    }
    return synchronize == nullptr ? cudaErrorUnknown : synchronize();
}

template <typename Allocate>
cudaError_t intercept_runtime_malloc_async_impl(void** device_pointer, std::size_t memory_bytes,
                                                cudaStream_t stream, Allocate allocate,
                                                RuntimeFreeAsyncFunction release,
                                                RuntimeDeviceSynchronizeFunction synchronize,
                                                RuntimeGetDeviceFunction get_device) {
    InterceptorState& state = get_state();
    if (device_pointer == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (!ensure_initialized(state)) {
        return std::getenv("GLIMMER_MEMORY_LIMIT_BYTES") == nullptr ? allocate() : cudaErrorUnknown;
    }
    if (state.quota_mode == QuotaMode::kDisabled) {
        return allocate();
    }
    if (state.quota_mode == QuotaMode::kInvalidConfiguration) {
        return cudaErrorInvalidValue;
    }
    if (state.allocations.is_accounting_degraded() || state.quota == nullptr ||
        !state.quota->is_healthy()) {
        return cudaErrorUnknown;
    }

    DeviceId device = QuotaStore::default_device;
    bool device_resolved = false;
    if (get_device != nullptr) {
        int runtime_device = 0;
        if (get_device(&runtime_device) == cudaSuccess && runtime_device >= 0) {
            device = static_cast<DeviceId>(runtime_device);
            device_resolved = true;
        }
    }
    const CUstream driver_stream = reinterpret_cast<CUstream>(stream);
    const std::optional<ContextIdentity> stream_identity =
        capture_stream_identity(state, driver_stream, false);
    if (stream_identity.has_value()) {
        device = static_cast<DeviceId>(stream_identity->device);
        device_resolved = true;
    } else if (state.uses_shared_quota && !device_resolved) {
        return cudaErrorInvalidValue;
    }

    auto reservation = state.quota->try_reserve(device, memory_bytes);
    if (!reservation.has_value()) {
        return state.quota->is_healthy() ? cudaErrorMemoryAllocation : cudaErrorUnknown;
    }
    const cudaError_t allocation_result = allocate();
    if (allocation_result != cudaSuccess) {
        return allocation_result;
    }
    if (*device_pointer == nullptr) {
        // A successful external allocation with no pointer is ambiguous. The
        // reservation must remain charged because there is no safe pointer to
        // pass to the asynchronous free path.
        reservation->abandon();
        state.allocations.mark_accounting_degraded();
        return cudaErrorUnknown;
    }

    const std::optional<ContextIdentity> final_identity =
        capture_stream_identity(state, driver_stream, false);
    if (!final_identity.has_value() || static_cast<DeviceId>(final_identity->device) != device) {
        const cudaError_t cleanup_result =
            cleanup_runtime_async_allocation(*device_pointer, stream, release, synchronize);
        if (cleanup_result != cudaSuccess) {
            reservation->abandon();
        }
        state.allocations.mark_accounting_degraded();
        return cleanup_result != cudaSuccess ? cleanup_result : cudaErrorUnknown;
    }

    if (!reservation->commit()) {
        const cudaError_t cleanup_result =
            cleanup_runtime_async_allocation(*device_pointer, stream, release, synchronize);
        if (cleanup_result != cudaSuccess) {
            reservation->abandon();
        }
        state.allocations.mark_accounting_degraded();
        return cleanup_result != cudaSuccess ? cleanup_result : cudaErrorUnknown;
    }

    if (!state.allocations.record(make_allocation_identity(*device_pointer, *final_identity),
                                  memory_bytes, AllocationScope::kContextIndependent)) {
        const cudaError_t cleanup_result =
            cleanup_runtime_async_allocation(*device_pointer, stream, release, synchronize);
        state.allocations.mark_accounting_degraded();
        if (cleanup_result != cudaSuccess) {
            return cleanup_result;
        }
        if (state.quota == nullptr ||
            !state.quota->release(static_cast<DeviceId>(final_identity->device), memory_bytes)) {
            return cudaErrorUnknown;
        }
        return cudaErrorMemoryAllocation;
    }
    return cudaSuccess;
}

cudaError_t intercept_runtime_malloc_async(void** device_pointer, std::size_t memory_bytes,
                                           cudaStream_t stream, RuntimeMallocAsyncFunction allocate,
                                           RuntimeFreeAsyncFunction release,
                                           RuntimeDeviceSynchronizeFunction synchronize,
                                           RuntimeGetDeviceFunction get_device) {
    if (allocate == nullptr) {
        return cudaErrorNotSupported;
    }
    return intercept_runtime_malloc_async_impl(
        device_pointer, memory_bytes, stream,
        [device_pointer, memory_bytes, stream, allocate] {
            return allocate(device_pointer, memory_bytes, stream);
        },
        release, synchronize, get_device);
}

cudaError_t intercept_runtime_malloc_from_pool_async(
    void** device_pointer, std::size_t memory_bytes, cudaMemPool_t pool, cudaStream_t stream,
    RuntimeMallocFromPoolAsyncFunction allocate, RuntimeFreeAsyncFunction release,
    RuntimeDeviceSynchronizeFunction synchronize, RuntimeGetDeviceFunction get_device) {
    if (allocate == nullptr) {
        return cudaErrorNotSupported;
    }
    return intercept_runtime_malloc_async_impl(
        device_pointer, memory_bytes, stream,
        [device_pointer, memory_bytes, pool, stream, allocate] {
            return allocate(device_pointer, memory_bytes, pool, stream);
        },
        release, synchronize, get_device);
}

cudaError_t intercept_runtime_mem_pool_import_from_shareable_handle(
    cudaMemPool_t* pool_out, void* handle, cudaMemAllocationHandleType handle_type,
    unsigned long long flags, RuntimeMemPoolImportFromShareableHandleFunction import_pool) {
    InterceptorState& state = get_state();
    if (pool_out == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (import_pool == nullptr) {
        return cudaErrorNotSupported;
    }
    if (!ensure_initialized(state)) {
        return std::getenv("GLIMMER_MEMORY_LIMIT_BYTES") == nullptr
                   ? import_pool(pool_out, handle, handle_type, flags)
                   : cudaErrorUnknown;
    }
    if (state.quota_mode == QuotaMode::kEnabled) {
        return cudaErrorNotSupported;
    }
    if (state.quota_mode == QuotaMode::kInvalidConfiguration) {
        return cudaErrorInvalidValue;
    }
    return import_pool(pool_out, handle, handle_type, flags);
}

cudaError_t intercept_runtime_mem_pool_import_pointer(
    void** pointer_out, cudaMemPool_t pool, cudaMemPoolPtrExportData* share_data,
    RuntimeMemPoolImportPointerFunction import_pointer) {
    InterceptorState& state = get_state();
    if (pointer_out == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (import_pointer == nullptr) {
        return cudaErrorNotSupported;
    }
    if (!ensure_initialized(state)) {
        return std::getenv("GLIMMER_MEMORY_LIMIT_BYTES") == nullptr
                   ? import_pointer(pointer_out, pool, share_data)
                   : cudaErrorUnknown;
    }
    if (state.quota_mode == QuotaMode::kEnabled) {
        return cudaErrorNotSupported;
    }
    if (state.quota_mode == QuotaMode::kInvalidConfiguration) {
        return cudaErrorInvalidValue;
    }
    return import_pointer(pointer_out, pool, share_data);
}

cudaError_t intercept_runtime_free_async(void* device_pointer, cudaStream_t stream,
                                         RuntimeFreeAsyncFunction release) {
    InterceptorState& state = get_state();
    if (release == nullptr) {
        return cudaErrorNotSupported;
    }
    const auto ordered_release = [&] {
        return intercept_cooperative_runtime_native_boundary(
            [&] { return release(device_pointer, stream); });
    };
    if (!ensure_initialized(state)) {
        return std::getenv("GLIMMER_MEMORY_LIMIT_BYTES") == nullptr
                   ? release(device_pointer, stream)
                   : cudaErrorUnknown;
    }
    if (state.quota_mode == QuotaMode::kDisabled) {
        return ordered_release();
    }
    if (state.quota_mode == QuotaMode::kInvalidConfiguration) {
        return ordered_release();
    }
    if (state.allocations.is_accounting_degraded() || state.quota == nullptr ||
        !state.quota->is_healthy()) {
        return cudaErrorUnknown;
    }

    const CUstream driver_stream = reinterpret_cast<CUstream>(stream);
    const std::optional<ContextIdentity> stream_identity =
        capture_stream_identity(state, driver_stream, false);
    if (!stream_identity.has_value()) {
        const cudaError_t free_result = ordered_release();
        if (free_result == cudaSuccess) {
            state.allocations.mark_accounting_degraded();
            return cudaErrorUnknown;
        }
        return free_result;
    }
    const AsyncStreamIdentity async_stream =
        make_runtime_async_stream_identity(stream, *stream_identity);
    auto release_state = state.allocations.begin_async_release(
        make_allocation_identity(device_pointer, *stream_identity), async_stream);
    if (release_state.first == AllocationRegistry::ReleaseStatus::kUnknown) {
        release_state = state.allocations.begin_async_release_by_pointer(
            static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(device_pointer)),
            stream_identity->device, async_stream);
    }
    const auto [release_status, release_ticket] = release_state;
    if (release_status == AllocationRegistry::ReleaseStatus::kInProgress) {
        return cudaErrorInvalidValue;
    }
    if (release_status == AllocationRegistry::ReleaseStatus::kUnknown ||
        !release_ticket.has_value()) {
        return ordered_release();
    }
    RuntimeQueuedRelease pending{*release_ticket};
    const ReleaseExceptionGuard exception_guard(state.allocations, pending.ticket,
                                                pending.has_owned_release);
    cudaError_t free_result = cudaErrorUnknown;
    if (cooperative_runtime_preview_requested() &&
        can_queue_cooperative_release(state, *release_ticket)) {
        std::shared_ptr<CudaCooperativeFrontend> frontend;
        const auto ready = cooperative_frontend_current(state, frontend);
        if (ready != CUDA_SUCCESS) {
            state.allocations.cancel_release(*release_ticket);
            return cudaErrorUnknown;
        }
        const RuntimeQueuedReleaseScope scope(pending);
        free_result = release(device_pointer, stream);
        if (free_result != cudaSuccess && pending.has_owned_release) {
            // Native validation cannot undo a release already accepted by the queue.
            state.allocations.mark_accounting_degraded();
            return cudaErrorUnknown;
        }
        if (free_result == cudaSuccess && (!pending.was_consumed || !pending.has_owned_release)) {
            state.allocations.mark_accounting_degraded();
            report_diagnostic("[glimmer] Runtime async free bypassed its owned Driver release\n");
            return cudaErrorUnknown;
        }
    } else {
        free_result = ordered_release();
    }
    if (free_result != cudaSuccess) {
        state.allocations.cancel_release(*release_ticket);
        return free_result;
    }
    if (!state.allocations.commit_async_release(*release_ticket)) {
        state.allocations.mark_accounting_degraded();
        return cudaErrorUnknown;
    }
    return cudaSuccess;
}

cudaError_t intercept_runtime_device_synchronize(RuntimeDeviceSynchronizeFunction synchronize) {
    InterceptorState& state = get_state();
    if (synchronize == nullptr) {
        return cudaErrorNotSupported;
    }
    if (!ensure_initialized(state)) {
        return std::getenv("GLIMMER_MEMORY_LIMIT_BYTES") == nullptr ? synchronize()
                                                                    : cudaErrorUnknown;
    }
    const cudaError_t sync_result = synchronize();
    if (sync_result != cudaSuccess || state.quota_mode != QuotaMode::kEnabled) {
        return sync_result;
    }
    if (state.allocations.is_accounting_degraded()) {
        return cudaErrorUnknown;
    }
    const std::optional<ContextIdentity> context = capture_context_identity(state);
    if (!context.has_value()) {
        state.allocations.mark_accounting_degraded();
        return cudaErrorUnknown;
    }
    const auto summaries =
        state.allocations.complete_async_releases_for_context_by_device(context->context);
    if (!summaries.has_value()) {
        state.allocations.mark_accounting_degraded();
        return cudaErrorUnknown;
    }
    finalize_async_releases(state, *summaries);
    return state.allocations.is_accounting_degraded() ? cudaErrorUnknown : sync_result;
}

cudaError_t intercept_runtime_stream_synchronize(cudaStream_t stream,
                                                 RuntimeStreamSynchronizeFunction synchronize) {
    InterceptorState& state = get_state();
    if (synchronize == nullptr) {
        return cudaErrorNotSupported;
    }
    if (!ensure_initialized(state)) {
        return std::getenv("GLIMMER_MEMORY_LIMIT_BYTES") == nullptr ? synchronize(stream)
                                                                    : cudaErrorUnknown;
    }
    const cudaError_t sync_result = synchronize(stream);
    if (sync_result != cudaSuccess || state.quota_mode != QuotaMode::kEnabled) {
        return sync_result;
    }
    if (state.allocations.is_accounting_degraded()) {
        return cudaErrorUnknown;
    }
    const std::optional<ContextIdentity> context =
        capture_stream_identity(state, reinterpret_cast<CUstream>(stream), false);
    if (!context.has_value()) {
        state.allocations.mark_accounting_degraded();
        return cudaErrorUnknown;
    }
    const auto summaries = state.allocations.complete_async_releases_for_stream_by_device(
        make_runtime_async_stream_identity(stream, *context));
    if (!summaries.has_value()) {
        state.allocations.mark_accounting_degraded();
        return cudaErrorUnknown;
    }
    finalize_async_releases(state, *summaries);
    return state.allocations.is_accounting_degraded() ? cudaErrorUnknown : sync_result;
}

cudaError_t intercept_runtime_stream_query(cudaStream_t stream, RuntimeStreamQueryFunction query) {
    InterceptorState& state = get_state();
    if (query == nullptr) {
        return cudaErrorNotSupported;
    }
    if (!ensure_initialized(state)) {
        return std::getenv("GLIMMER_MEMORY_LIMIT_BYTES") == nullptr ? query(stream)
                                                                    : cudaErrorUnknown;
    }
    const cudaError_t query_result = query(stream);
    if (query_result != cudaSuccess || state.quota_mode != QuotaMode::kEnabled) {
        return query_result;
    }
    if (state.allocations.is_accounting_degraded()) {
        return cudaErrorUnknown;
    }
    const std::optional<ContextIdentity> context =
        capture_stream_identity(state, reinterpret_cast<CUstream>(stream), false);
    if (!context.has_value()) {
        state.allocations.mark_accounting_degraded();
        return cudaErrorUnknown;
    }
    const auto summaries = state.allocations.complete_async_releases_for_stream_by_device(
        make_runtime_async_stream_identity(stream, *context));
    if (!summaries.has_value()) {
        state.allocations.mark_accounting_degraded();
        return cudaErrorUnknown;
    }
    finalize_async_releases(state, *summaries);
    return state.allocations.is_accounting_degraded() ? cudaErrorUnknown : query_result;
}

std::optional<cudaError_t> intercept_normal_runtime_device_completion(
    const std::function<cudaError_t()>& original,
    const RuntimeContextCompletionExecution& epoch_completion) {
    if (!original || !cooperative_runtime_preview_requested() || is_inside_driver_call() ||
        g_is_inside_cooperative_boundary ||
        cooperative_process_call_gate().is_native_only_on_current_thread()) {
        return std::nullopt;
    }
    auto& state = get_state();
    if (!ensure_initialized(state) || !state.cooperative_router) {
        return std::nullopt;
    }
    const CooperativeBoundaryScope scope;
    auto& router = *state.cooperative_router;
    if (router.normal_owner_status() != CUDA_SUCCESS) {
        return cudaErrorUnknown;
    }
    auto submitted = state.allocations.submitted_async_releases(true);
    if (!submitted) {
        return cudaErrorUnknown;  // No SDK issue or accounting mutation.
    }
    std::optional<cudaError_t> recovered;
    if (epoch_completion || router.has_native_epoch_policy()) {
        recovered = invoke_runtime_context_completion_original(
            [&](const auto& completion) {
                const auto original_completion = [&] {
                    // Include only submissions joined before the owned SDK boundary.
                    // No future release may be finalized by an earlier proof.
                    submitted = state.allocations.submitted_async_releases(true);
                    return submitted ? completion() : CUDA_ERROR_UNKNOWN;
                };
                return epoch_completion ? epoch_completion(original_completion)
                                        : router.recover_current_native_epoch(
                                              cooperative_process_call_gate(), original_completion);
            },
            original, [&] { router.quarantine_normal_launch(nullptr, CUDA_ERROR_UNKNOWN); });
    }
    const auto native_status = recovered ? *recovered : original();
    if (native_status != cudaSuccess) {
        return native_status;
    }
    CUcontext context{};
    CUresult status{};
    {
        const NativeDriverCallScope native;
        status = state.driver.context_get_current(&context);
    }
    if (status != CUDA_SUCCESS || !context) {
        return cudaErrorUnknown;
    }
    std::shared_ptr<CudaCooperativeFrontend> owner;
    status = router.existing_frontend(context, owner);
    if (status == CUDA_SUCCESS && owner && !recovered) {
        status = router.with_context(context, [&] { return owner->synchronize_context(); });
    }
    if (status != CUDA_SUCCESS) {
        return cudaErrorUnknown;
    }
    if (state.quota_mode != QuotaMode::kEnabled) {
        return native_status;
    }
    if (state.allocations.is_accounting_degraded()) {
        return cudaErrorUnknown;
    }
    const auto summaries =
        state.allocations.complete_submitted_context_releases(*submitted, context);
    if (!summaries) {
        state.allocations.mark_accounting_degraded();
        return cudaErrorUnknown;
    }
    finalize_async_releases(state, *summaries);
    return state.allocations.is_accounting_degraded() ? cudaErrorUnknown : native_status;
}

std::optional<cudaError_t> intercept_normal_runtime_stream_completion(
    cudaStream_t stream, bool is_ptds, bool should_wait,
    const std::function<cudaError_t()>& original) {
    static_assert(std::is_same_v<cudaStream_t, CUstream>);
    if (!original || !cooperative_runtime_preview_requested() || is_inside_driver_call() ||
        g_is_inside_cooperative_boundary ||
        cooperative_process_call_gate().is_native_only_on_current_thread()) {
        return std::nullopt;
    }
    auto& state = get_state();
    if (!state.cooperative_router) {
        return std::nullopt;
    }
    const CooperativeBoundaryScope scope;
    auto& router = *state.cooperative_router;
    if (router.normal_owner_status() != CUDA_SUCCESS) {
        return cudaErrorUnknown;
    }
    const auto submitted = state.allocations.submitted_async_releases();
    if (!submitted) {
        return cudaErrorUnknown;  // No original issue or accounting mutation.
    }
    const auto native_status = original();
    if (native_status != cudaSuccess) {
        return native_status;  // Pending query remains nonblocking, with native TLS.
    }
    CUcontext context{};
    CUresult status{};
    {
        const NativeDriverCallScope native;
        status = is_ptds ? state.driver.stream_get_context_ptsz(stream, &context)
                         : state.driver.stream_get_context(stream, &context);
    }
    if (status != CUDA_SUCCESS || !context) {
        return cudaErrorUnknown;
    }
    std::shared_ptr<CudaCooperativeFrontend> owner;
    status = router.existing_frontend(context, owner);
    bool has_binding{};
    if (status == CUDA_SUCCESS && owner) {
        status = router.with_context(context, [&] {
            return owner->complete_stream_snapshot(stream, is_ptds, should_wait, has_binding);
        });
    }
    if (status == CUDA_ERROR_NOT_READY) {
        return cudaErrorNotReady;
    }
    if (status != CUDA_SUCCESS) {
        return cudaErrorUnknown;
    }
    if (state.quota_mode != QuotaMode::kEnabled) {
        return native_status;
    }
    if (state.allocations.is_accounting_degraded()) {
        return cudaErrorUnknown;
    }
    if (submitted->empty()) {
        return native_status;
    }
    status = router.with_context(context, [&] {
        if (state.allocations.is_accounting_degraded()) {
            return CUDA_ERROR_UNKNOWN;
        }
        const auto identity = capture_stream_identity(state, stream, is_ptds);
        if (!identity || identity->context != context) {
            return CUDA_ERROR_UNKNOWN;
        }
        const auto retire = [&](const AsyncStreamIdentity& proven) {
            const auto summaries =
                state.allocations.complete_submitted_async_releases(*submitted, proven);
            if (!summaries) {
                state.allocations.mark_accounting_degraded();
                return CUDA_ERROR_UNKNOWN;
            }
            finalize_async_releases(state, *summaries);
            return state.allocations.is_accounting_degraded() ? CUDA_ERROR_UNKNOWN : CUDA_SUCCESS;
        };
        auto completed = retire(make_async_stream_identity(stream, *identity, is_ptds));
        if (completed != CUDA_SUCCESS || stream != nullptr || is_ptds) {
            return completed;
        }
        // Runtime's existing null release namespace is caller-scoped even for
        // legacy aliases. Do not let legacy readiness retire pending PTDS frees.
        {
            const NativeDriverCallScope native;
            completed = state.driver.stream_query_ptsz(nullptr);
        }
        if (completed == CUDA_ERROR_NOT_READY) {
            return CUDA_SUCCESS;  // Retain the ambiguous namespace without blocking this query.
        }
        if (completed != CUDA_SUCCESS) {
            return completed;
        }
        if (owner) {
            bool has_thread_binding{};
            completed = owner->complete_stream_snapshot(nullptr, true, false, has_thread_binding);
            if (completed == CUDA_ERROR_NOT_READY) {
                return CUDA_SUCCESS;
            }
            if (completed != CUDA_SUCCESS) {
                return completed;
            }
        }
        return retire(make_runtime_async_stream_identity(stream, *identity));
    });
    return status == CUDA_SUCCESS ? native_status : cudaErrorUnknown;
}

cudaError_t intercept_runtime_stream_destroy(cudaStream_t stream,
                                             RuntimeStreamDestroyFunction destroy) {
    InterceptorState& state = get_state();
    if (destroy == nullptr) {
        return cudaErrorNotSupported;
    }
    if (!ensure_initialized(state)) {
        return std::getenv("GLIMMER_MEMORY_LIMIT_BYTES") == nullptr ? destroy(stream)
                                                                    : cudaErrorUnknown;
    }
    std::optional<AsyncStreamIdentity> async_stream;
    if (state.quota_mode == QuotaMode::kEnabled && !state.allocations.is_accounting_degraded()) {
        const std::optional<ContextIdentity> context =
            capture_stream_identity(state, reinterpret_cast<CUstream>(stream), false);
        if (!context.has_value()) {
            state.allocations.mark_accounting_degraded();
        } else {
            async_stream = make_runtime_async_stream_identity(stream, *context);
        }
    }
    const cudaError_t destroy_result = destroy(stream);
    if (destroy_result != cudaSuccess || state.quota_mode != QuotaMode::kEnabled) {
        return destroy_result;
    }
    if (state.allocations.is_accounting_degraded()) {
        return cudaErrorUnknown;
    }
    if (async_stream.has_value() &&
        !state.allocations.detach_async_releases_for_stream(*async_stream)) {
        state.allocations.mark_accounting_degraded();
        return cudaErrorUnknown;
    }
    return cudaSuccess;
}

CUresult intercept_mem_alloc_async(CUdeviceptr* device_pointer, std::size_t memory_bytes,
                                   CUstream stream, bool per_thread_default_stream) {
    InterceptorState& state = get_state();
    if (device_pointer == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (glimmer::interceptor::is_inside_driver_call() ||
        glimmer::interceptor::is_inside_runtime_call()) {
        return per_thread_default_stream
                   ? state.driver.mem_alloc_async_ptsz(device_pointer, memory_bytes, stream)
                   : state.driver.mem_alloc_async(device_pointer, memory_bytes, stream);
    }
    if (!ensure_initialized(state) ||
        (per_thread_default_stream ? !state.driver.has_mem_alloc_async_ptsz()
                                   : !state.driver.has_mem_alloc_async())) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return intercept_async_tracked_allocation(
        device_pointer, memory_bytes, stream, per_thread_default_stream,
        [&state, device_pointer, memory_bytes, stream, per_thread_default_stream] {
            return per_thread_default_stream
                       ? state.driver.mem_alloc_async_ptsz(device_pointer, memory_bytes, stream)
                       : state.driver.mem_alloc_async(device_pointer, memory_bytes, stream);
        });
}

CUresult intercept_mem_alloc_from_pool_async(CUdeviceptr* device_pointer, std::size_t memory_bytes,
                                             CUmemoryPool pool, CUstream stream,
                                             bool per_thread_default_stream) {
    InterceptorState& state = get_state();
    if (device_pointer == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (glimmer::interceptor::is_inside_driver_call() ||
        glimmer::interceptor::is_inside_runtime_call()) {
        return per_thread_default_stream ? state.driver.mem_alloc_from_pool_async_ptsz(
                                               device_pointer, memory_bytes, pool, stream)
                                         : state.driver.mem_alloc_from_pool_async(
                                               device_pointer, memory_bytes, pool, stream);
    }
    if (!ensure_initialized(state) ||
        (per_thread_default_stream ? !state.driver.has_mem_alloc_from_pool_async_ptsz()
                                   : !state.driver.has_mem_alloc_from_pool_async())) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return intercept_async_tracked_allocation(
        device_pointer, memory_bytes, stream, per_thread_default_stream,
        [&state, device_pointer, memory_bytes, pool, stream, per_thread_default_stream] {
            return per_thread_default_stream ? state.driver.mem_alloc_from_pool_async_ptsz(
                                                   device_pointer, memory_bytes, pool, stream)
                                             : state.driver.mem_alloc_from_pool_async(
                                                   device_pointer, memory_bytes, pool, stream);
        });
}

CUresult intercept_mem_create(CUmemGenericAllocationHandle* handle, std::size_t memory_bytes,
                              const CUmemAllocationProp* prop, unsigned long long flags) {
    InterceptorState& state = get_state();
    if (handle == nullptr || prop == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (glimmer::interceptor::is_inside_driver_call() ||
        glimmer::interceptor::is_inside_runtime_call()) {
        return state.driver.mem_create(handle, memory_bytes, prop, flags);
    }
    if (!ensure_initialized(state) || !state.driver.has_mem_create()) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    // VMM can also create host allocations. Only device-resident physical
    // allocations consume the CUDA device quota; address reservation and
    // non-device VMM paths remain delegated to the real driver.
    if (prop->location.type != CU_MEM_LOCATION_TYPE_DEVICE) {
        return state.driver.mem_create(handle, memory_bytes, prop, flags);
    }
    if (prop->location.id < 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (state.quota_mode == QuotaMode::kDisabled) {
        return state.driver.mem_create(handle, memory_bytes, prop, flags);
    }
    if (!state.driver.has_mem_release()) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (state.quota_mode == QuotaMode::kInvalidConfiguration ||
        state.allocations.is_accounting_degraded() || state.quota == nullptr ||
        !state.quota->is_healthy()) {
        return state.quota_mode == QuotaMode::kInvalidConfiguration ? CUDA_ERROR_INVALID_VALUE
                                                                    : CUDA_ERROR_UNKNOWN;
    }

    const DeviceId device = static_cast<DeviceId>(prop->location.id);
    auto reservation = state.quota->try_reserve(device, memory_bytes);
    if (!reservation.has_value()) {
        return state.quota->is_healthy() ? CUDA_ERROR_OUT_OF_MEMORY : CUDA_ERROR_UNKNOWN;
    }

    const CUresult create_result = state.driver.mem_create(handle, memory_bytes, prop, flags);
    if (create_result != CUDA_SUCCESS) {
        return create_result;
    }
    if (*handle == 0) {
        reservation->abandon();
        state.allocations.mark_accounting_degraded();
        return CUDA_ERROR_UNKNOWN;
    }

    if (!reservation->commit()) {
        const CUresult cleanup_result = state.driver.mem_release(*handle);
        if (cleanup_result != CUDA_SUCCESS) {
            reservation->abandon();
        }
        state.allocations.mark_accounting_degraded();
        return cleanup_result != CUDA_SUCCESS ? cleanup_result : CUDA_ERROR_UNKNOWN;
    }

    const VmmAllocationIdentity identity{.handle = *handle, .device = prop->location.id};
    if (state.allocations.record_vmm(identity, memory_bytes)) {
        return CUDA_SUCCESS;
    }

    state.allocations.mark_accounting_degraded();
    const CUresult cleanup_result = state.driver.mem_release(*handle);
    if (cleanup_result != CUDA_SUCCESS) {
        return cleanup_result;
    }
    if (!state.quota->release(device, memory_bytes)) {
        return CUDA_ERROR_UNKNOWN;
    }
    return CUDA_ERROR_OUT_OF_MEMORY;
}

CUresult intercept_mem_release(CUmemGenericAllocationHandle handle) {
    InterceptorState& state = get_state();
    if (glimmer::interceptor::is_inside_driver_call() ||
        glimmer::interceptor::is_inside_runtime_call()) {
        return state.driver.mem_release(handle);
    }
    if (!ensure_initialized(state) || !state.driver.has_mem_release()) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (state.quota_mode != QuotaMode::kEnabled) {
        return state.driver.mem_release(handle);
    }
    if (state.allocations.is_accounting_degraded() || state.quota == nullptr ||
        !state.quota->is_healthy()) {
        return CUDA_ERROR_UNKNOWN;
    }

    // Handles created outside this interceptor (for example imported handles)
    // are delegated because their owning allocation is not locally trackable.
    const auto release_state = state.allocations.begin_vmm_release_by_handle(handle);
    if (release_state.first == AllocationRegistry::ReleaseStatus::kUnknown) {
        return state.driver.mem_release(handle);
    }
    if (release_state.first == AllocationRegistry::ReleaseStatus::kInProgress ||
        !release_state.second.has_value()) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    const AllocationRegistry::VmmReleaseTicket& ticket = *release_state.second;
    const CUresult release_result = state.driver.mem_release(handle);
    if (release_result != CUDA_SUCCESS) {
        state.allocations.cancel_vmm_release(ticket);
        return release_result;
    }

    const bool is_released =
        !ticket.is_last_reference ||
        state.quota->release(static_cast<DeviceId>(ticket.identity.device), ticket.memory_bytes);
    const bool is_completed = state.allocations.complete_vmm_release(ticket);
    if (!is_released || !is_completed) {
        state.allocations.mark_accounting_degraded();
        return CUDA_ERROR_UNKNOWN;
    }
    return CUDA_SUCCESS;
}

CUresult intercept_mem_address_reserve(CUdeviceptr* device_pointer, std::size_t memory_bytes,
                                       std::size_t alignment, CUdeviceptr requested_address,
                                       unsigned long long flags) {
    InterceptorState& state = get_state();
    if (device_pointer == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (glimmer::interceptor::is_inside_driver_call() ||
        glimmer::interceptor::is_inside_runtime_call()) {
        return state.driver.mem_address_reserve(device_pointer, memory_bytes, alignment,
                                                requested_address, flags);
    }
    if (!ensure_initialized(state) || !state.driver.has_mem_address_reserve()) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return state.driver.mem_address_reserve(device_pointer, memory_bytes, alignment,
                                            requested_address, flags);
}

CUresult intercept_mem_address_free(CUdeviceptr device_pointer, std::size_t memory_bytes) {
    InterceptorState& state = get_state();
    if (glimmer::interceptor::is_inside_driver_call() ||
        glimmer::interceptor::is_inside_runtime_call()) {
        return state.driver.mem_address_free(device_pointer, memory_bytes);
    }
    if (!ensure_initialized(state) || !state.driver.has_mem_address_free()) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return state.driver.mem_address_free(device_pointer, memory_bytes);
}

CUresult intercept_mem_map(CUdeviceptr device_pointer, std::size_t memory_bytes, std::size_t offset,
                           CUmemGenericAllocationHandle handle, unsigned long long flags) {
    InterceptorState& state = get_state();
    if (glimmer::interceptor::is_inside_driver_call() ||
        glimmer::interceptor::is_inside_runtime_call()) {
        return state.driver.mem_map(device_pointer, memory_bytes, offset, handle, flags);
    }
    if (!ensure_initialized(state) || !state.driver.has_mem_map()) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return state.driver.mem_map(device_pointer, memory_bytes, offset, handle, flags);
}

CUresult intercept_mem_map_array_async(CUarrayMapInfo* map_info_list, unsigned int count,
                                       CUstream stream) {
    if (count != 0 && map_info_list == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    if (glimmer::interceptor::is_inside_driver_call() ||
        glimmer::interceptor::is_inside_runtime_call()) {
        return state.driver.mem_map_array_async(map_info_list, count, stream);
    }
    if (!ensure_initialized(state) || !state.driver.has_mem_map_array_async()) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (state.quota_mode == QuotaMode::kEnabled) {
        // Sparse and deferred-mapping arrays can map tile-pool memory through
        // this API without exposing a stable allocation size at this boundary.
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (state.quota_mode == QuotaMode::kInvalidConfiguration) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return state.driver.mem_map_array_async(map_info_list, count, stream);
}

CUresult intercept_mem_unmap(CUdeviceptr device_pointer, std::size_t memory_bytes) {
    InterceptorState& state = get_state();
    if (glimmer::interceptor::is_inside_driver_call() ||
        glimmer::interceptor::is_inside_runtime_call()) {
        return state.driver.mem_unmap(device_pointer, memory_bytes);
    }
    if (!ensure_initialized(state) || !state.driver.has_mem_unmap()) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return state.driver.mem_unmap(device_pointer, memory_bytes);
}

CUresult intercept_mem_set_access(CUdeviceptr device_pointer, std::size_t memory_bytes,
                                  const CUmemAccessDesc* access_descriptors,
                                  std::size_t descriptor_count) {
    InterceptorState& state = get_state();
    if (descriptor_count != 0 && access_descriptors == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (glimmer::interceptor::is_inside_driver_call() ||
        glimmer::interceptor::is_inside_runtime_call()) {
        return state.driver.mem_set_access(device_pointer, memory_bytes, access_descriptors,
                                           descriptor_count);
    }
    if (!ensure_initialized(state) || !state.driver.has_mem_set_access()) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return state.driver.mem_set_access(device_pointer, memory_bytes, access_descriptors,
                                       descriptor_count);
}

CUresult intercept_mem_get_address_range(CUdeviceptr* base_pointer, std::size_t* memory_bytes,
                                         CUdeviceptr device_pointer) {
    if (base_pointer == nullptr || memory_bytes == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    return forward_optional_driver_call(
        state, [&state] { return state.driver.has_mem_get_address_range(); },
        [&state, base_pointer, memory_bytes, device_pointer] {
            return state.driver.mem_get_address_range(base_pointer, memory_bytes, device_pointer);
        });
}

CUresult intercept_mem_get_access(unsigned long long* flags, const CUmemLocation* location,
                                  CUdeviceptr device_pointer) {
    if (flags == nullptr || location == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    return forward_optional_driver_call(
        state, [&state] { return state.driver.has_mem_get_access(); },
        [&state, flags, location, device_pointer] {
            return state.driver.mem_get_access(flags, location, device_pointer);
        });
}

CUresult intercept_mem_export_to_shareable_handle(void* shareable_handle,
                                                  CUmemGenericAllocationHandle handle,
                                                  CUmemAllocationHandleType handle_type,
                                                  unsigned long long flags) {
    if (shareable_handle == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    return forward_optional_driver_call(
        state, [&state] { return state.driver.has_mem_export_to_shareable_handle(); },
        [&state, shareable_handle, handle, handle_type, flags] {
            return state.driver.mem_export_to_shareable_handle(shareable_handle, handle,
                                                               handle_type, flags);
        });
}

CUresult intercept_mem_import_from_shareable_handle(CUmemGenericAllocationHandle* handle,
                                                    void* os_handle,
                                                    CUmemAllocationHandleType handle_type) {
    if (handle == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    if (glimmer::interceptor::is_inside_driver_call() ||
        glimmer::interceptor::is_inside_runtime_call()) {
        return state.driver.mem_import_from_shareable_handle(handle, os_handle, handle_type);
    }
    if (!ensure_initialized(state) || !state.driver.has_mem_import_from_shareable_handle()) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (state.quota_mode == QuotaMode::kEnabled) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (state.quota_mode == QuotaMode::kInvalidConfiguration) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return state.driver.mem_import_from_shareable_handle(handle, os_handle, handle_type);
}

CUresult intercept_ipc_get_mem_handle(CUipcMemHandle* handle, CUdeviceptr device_pointer) {
    if (handle == nullptr || device_pointer == 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    return forward_optional_driver_call(
        state, [&state] { return state.driver.has_ipc_get_mem_handle(); },
        [&state, handle, device_pointer] {
            return state.driver.ipc_get_mem_handle(handle, device_pointer);
        });
}

CUresult intercept_ipc_open_mem_handle(CUdeviceptr* device_pointer, CUipcMemHandle handle,
                                       unsigned int flags) {
    if (device_pointer == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    if (glimmer::interceptor::is_inside_driver_call() ||
        glimmer::interceptor::is_inside_runtime_call()) {
        return state.driver.ipc_open_mem_handle(device_pointer, handle, flags);
    }
    if (!ensure_initialized(state) || !state.driver.has_ipc_open_mem_handle()) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (state.quota_mode == QuotaMode::kEnabled) {
        // An IPC handle refers to physical memory owned and accounted by
        // another process. Without a trusted cross-process allocation record,
        // accepting it would let the importing process bypass its quota.
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (state.quota_mode == QuotaMode::kInvalidConfiguration) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return state.driver.ipc_open_mem_handle(device_pointer, handle, flags);
}

CUresult intercept_ipc_close_mem_handle(CUdeviceptr device_pointer) {
    if (device_pointer == 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    return forward_optional_driver_call(
        state, [&state] { return state.driver.has_ipc_close_mem_handle(); },
        [&state, device_pointer] { return state.driver.ipc_close_mem_handle(device_pointer); });
}

CUresult intercept_import_external_memory(CUexternalMemory* external_memory,
                                          const CUDA_EXTERNAL_MEMORY_HANDLE_DESC* handle_desc) {
    if (external_memory == nullptr || handle_desc == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    if (glimmer::interceptor::is_inside_driver_call() ||
        glimmer::interceptor::is_inside_runtime_call()) {
        return state.driver.import_external_memory(external_memory, handle_desc);
    }
    if (!ensure_initialized(state) || !state.driver.has_import_external_memory()) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (state.quota_mode == QuotaMode::kEnabled) {
        // External memory is allocated and owned by another API or process.
        // Its size and tenant ownership are not authenticated at this ABI.
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (state.quota_mode == QuotaMode::kInvalidConfiguration) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return state.driver.import_external_memory(external_memory, handle_desc);
}

CUresult intercept_external_memory_get_mapped_buffer(
    CUdeviceptr* device_pointer, CUexternalMemory external_memory,
    const CUDA_EXTERNAL_MEMORY_BUFFER_DESC* buffer_desc) {
    if (device_pointer == nullptr || external_memory == nullptr || buffer_desc == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    if (glimmer::interceptor::is_inside_driver_call() ||
        glimmer::interceptor::is_inside_runtime_call()) {
        return state.driver.external_memory_get_mapped_buffer(device_pointer, external_memory,
                                                              buffer_desc);
    }
    if (!ensure_initialized(state) || !state.driver.has_external_memory_get_mapped_buffer()) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (state.quota_mode == QuotaMode::kEnabled) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (state.quota_mode == QuotaMode::kInvalidConfiguration) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return state.driver.external_memory_get_mapped_buffer(device_pointer, external_memory,
                                                          buffer_desc);
}

CUresult intercept_external_memory_get_mapped_mipmapped_array(
    CUmipmappedArray* mipmap, CUexternalMemory external_memory,
    const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC* mipmap_desc) {
    if (mipmap == nullptr || external_memory == nullptr || mipmap_desc == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    if (glimmer::interceptor::is_inside_driver_call() ||
        glimmer::interceptor::is_inside_runtime_call()) {
        return state.driver.external_memory_get_mapped_mipmapped_array(mipmap, external_memory,
                                                                       mipmap_desc);
    }
    if (!ensure_initialized(state) ||
        !state.driver.has_external_memory_get_mapped_mipmapped_array()) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (state.quota_mode == QuotaMode::kEnabled) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (state.quota_mode == QuotaMode::kInvalidConfiguration) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return state.driver.external_memory_get_mapped_mipmapped_array(mipmap, external_memory,
                                                                   mipmap_desc);
}

CUresult intercept_destroy_external_memory(CUexternalMemory external_memory) {
    if (external_memory == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    return forward_optional_driver_call(
        state, [&state] { return state.driver.has_destroy_external_memory(); },
        [&state, external_memory] {
            return state.driver.destroy_external_memory(external_memory);
        });
}

CUresult intercept_array_create(CUarray* array, const CUDA_ARRAY_DESCRIPTOR* descriptor) {
    if (array == nullptr || descriptor == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    if (glimmer::interceptor::is_inside_driver_call() ||
        glimmer::interceptor::is_inside_runtime_call()) {
        return state.driver.array_create(array, descriptor);
    }
    if (!ensure_initialized(state) || !state.driver.has_array_create()) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (state.quota_mode == QuotaMode::kEnabled) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (state.quota_mode == QuotaMode::kInvalidConfiguration) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return state.driver.array_create(array, descriptor);
}

CUresult intercept_array_3d_create(CUarray* array, const CUDA_ARRAY3D_DESCRIPTOR* descriptor) {
    if (array == nullptr || descriptor == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    if (glimmer::interceptor::is_inside_driver_call() ||
        glimmer::interceptor::is_inside_runtime_call()) {
        return state.driver.array_3d_create(array, descriptor);
    }
    if (!ensure_initialized(state) || !state.driver.has_array_3d_create()) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (state.quota_mode == QuotaMode::kEnabled) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (state.quota_mode == QuotaMode::kInvalidConfiguration) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return state.driver.array_3d_create(array, descriptor);
}

CUresult intercept_array_destroy(CUarray array) {
    if (array == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    return forward_optional_driver_call(
        state, [&state] { return state.driver.has_array_destroy(); },
        [&state, array] { return state.driver.array_destroy(array); });
}

CUresult intercept_mipmapped_array_create(CUmipmappedArray* mipmap,
                                          const CUDA_ARRAY3D_DESCRIPTOR* descriptor,
                                          unsigned int level_count) {
    if (mipmap == nullptr || descriptor == nullptr || level_count == 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    if (glimmer::interceptor::is_inside_driver_call() ||
        glimmer::interceptor::is_inside_runtime_call()) {
        return state.driver.mipmapped_array_create(mipmap, descriptor, level_count);
    }
    if (!ensure_initialized(state) || !state.driver.has_mipmapped_array_create()) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (state.quota_mode == QuotaMode::kEnabled) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (state.quota_mode == QuotaMode::kInvalidConfiguration) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return state.driver.mipmapped_array_create(mipmap, descriptor, level_count);
}

CUresult intercept_mipmapped_array_destroy(CUmipmappedArray mipmap) {
    if (mipmap == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    return forward_optional_driver_call(
        state, [&state] { return state.driver.has_mipmapped_array_destroy(); },
        [&state, mipmap] { return state.driver.mipmapped_array_destroy(mipmap); });
}

CUresult intercept_graphics_unregister_resource(CUgraphicsResource resource) {
    if (resource == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    return forward_optional_driver_call(
        state, [&state] { return state.driver.has_graphics_unregister_resource(); },
        [&state, resource] { return state.driver.graphics_unregister_resource(resource); });
}

CUresult intercept_graphics_subresource_get_mapped_array(CUarray* array,
                                                         CUgraphicsResource resource,
                                                         unsigned int array_index,
                                                         unsigned int mip_level) {
    if (array == nullptr || resource == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    if (is_inside_driver_call() || is_inside_runtime_call()) {
        return state.driver.graphics_subresource_get_mapped_array(array, resource, array_index,
                                                                  mip_level);
    }
    if (!ensure_initialized(state) || !state.driver.has_graphics_subresource_get_mapped_array()) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (state.quota_mode == QuotaMode::kEnabled) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (state.quota_mode == QuotaMode::kInvalidConfiguration) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return state.driver.graphics_subresource_get_mapped_array(array, resource, array_index,
                                                              mip_level);
}

CUresult intercept_graphics_resource_get_mapped_mipmapped_array(CUmipmappedArray* mipmap,
                                                                CUgraphicsResource resource) {
    if (mipmap == nullptr || resource == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    if (is_inside_driver_call() || is_inside_runtime_call()) {
        return state.driver.graphics_resource_get_mapped_mipmapped_array(mipmap, resource);
    }
    if (!ensure_initialized(state) ||
        !state.driver.has_graphics_resource_get_mapped_mipmapped_array()) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (state.quota_mode == QuotaMode::kEnabled) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (state.quota_mode == QuotaMode::kInvalidConfiguration) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return state.driver.graphics_resource_get_mapped_mipmapped_array(mipmap, resource);
}

CUresult intercept_graphics_resource_get_mapped_pointer(CUdeviceptr* device_pointer,
                                                        std::size_t* size,
                                                        CUgraphicsResource resource) {
    if (device_pointer == nullptr || size == nullptr || resource == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    if (is_inside_driver_call() || is_inside_runtime_call()) {
        return state.driver.graphics_resource_get_mapped_pointer(device_pointer, size, resource);
    }
    if (!ensure_initialized(state) || !state.driver.has_graphics_resource_get_mapped_pointer()) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (state.quota_mode == QuotaMode::kEnabled) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (state.quota_mode == QuotaMode::kInvalidConfiguration) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return state.driver.graphics_resource_get_mapped_pointer(device_pointer, size, resource);
}

CUresult intercept_graphics_resource_set_map_flags(CUgraphicsResource resource,
                                                   unsigned int flags) {
    if (resource == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    return forward_optional_driver_call(
        state, [&state] { return state.driver.has_graphics_resource_set_map_flags(); },
        [&state, resource, flags] {
            return state.driver.graphics_resource_set_map_flags(resource, flags);
        });
}

CUresult intercept_graphics_map_resources(unsigned int count, CUgraphicsResource* resources,
                                          CUstream stream) {
    if (count != 0 && resources == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    if (is_inside_driver_call() || is_inside_runtime_call()) {
        return state.driver.graphics_map_resources(count, resources, stream);
    }
    if (!ensure_initialized(state) || !state.driver.has_graphics_map_resources()) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (state.quota_mode == QuotaMode::kEnabled) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (state.quota_mode == QuotaMode::kInvalidConfiguration) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return state.driver.graphics_map_resources(count, resources, stream);
}

CUresult intercept_graphics_unmap_resources(unsigned int count, CUgraphicsResource* resources,
                                            CUstream stream) {
    if (count != 0 && resources == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    return forward_optional_driver_call(
        state, [&state] { return state.driver.has_graphics_unmap_resources(); },
        [&state, count, resources, stream] {
            return state.driver.graphics_unmap_resources(count, resources, stream);
        });
}

CUresult intercept_mem_get_allocation_granularity(std::size_t* granularity,
                                                  const CUmemAllocationProp* prop,
                                                  CUmemAllocationGranularity_flags option) {
    if (granularity == nullptr || prop == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    return forward_optional_driver_call(
        state, [&state] { return state.driver.has_mem_get_allocation_granularity(); },
        [&state, granularity, prop, option] {
            return state.driver.mem_get_allocation_granularity(granularity, prop, option);
        });
}

CUresult intercept_mem_get_allocation_properties(CUmemAllocationProp* prop,
                                                 CUmemGenericAllocationHandle handle) {
    if (prop == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    return forward_optional_driver_call(
        state, [&state] { return state.driver.has_mem_get_allocation_properties(); },
        [&state, prop, handle] {
            return state.driver.mem_get_allocation_properties(prop, handle);
        });
}

CUresult intercept_mem_retain_allocation_handle(CUmemGenericAllocationHandle* handle,
                                                void* device_pointer) {
    if (handle == nullptr || device_pointer == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    if (glimmer::interceptor::is_inside_driver_call() ||
        glimmer::interceptor::is_inside_runtime_call()) {
        return state.driver.mem_retain_allocation_handle(handle, device_pointer);
    }
    if (!ensure_initialized(state) || !state.driver.has_mem_retain_allocation_handle()) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (state.quota_mode == QuotaMode::kDisabled) {
        return state.driver.mem_retain_allocation_handle(handle, device_pointer);
    }
    if (state.quota_mode == QuotaMode::kInvalidConfiguration ||
        state.allocations.is_accounting_degraded() || state.quota == nullptr ||
        !state.quota->is_healthy()) {
        return state.quota_mode == QuotaMode::kInvalidConfiguration ? CUDA_ERROR_INVALID_VALUE
                                                                    : CUDA_ERROR_UNKNOWN;
    }

    const CUresult retain_result =
        state.driver.mem_retain_allocation_handle(handle, device_pointer);
    if (retain_result != CUDA_SUCCESS) {
        return retain_result;
    }
    if (*handle == 0) {
        state.allocations.mark_accounting_degraded();
        return CUDA_ERROR_UNKNOWN;
    }
    if (state.allocations.retain_vmm_handle(*handle)) {
        return CUDA_SUCCESS;
    }

    const CUresult cleanup_result = state.driver.mem_release(*handle);
    if (cleanup_result != CUDA_SUCCESS) {
        state.allocations.mark_accounting_degraded();
        return cleanup_result;
    }
    state.allocations.mark_accounting_degraded();
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult intercept_mem_pool_trim_to(CUmemoryPool pool, std::size_t min_bytes_to_keep) {
    if (pool == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    return forward_optional_driver_call(
        state, [&state] { return state.driver.has_mem_pool_trim_to(); },
        [&state, pool, min_bytes_to_keep] {
            return state.driver.mem_pool_trim_to(pool, min_bytes_to_keep);
        });
}

CUresult intercept_mem_pool_set_attribute(CUmemoryPool pool, CUmemPool_attribute attribute,
                                          void* value) {
    if (pool == nullptr || value == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    return forward_optional_driver_call(
        state, [&state] { return state.driver.has_mem_pool_set_attribute(); },
        [&state, pool, attribute, value] {
            return state.driver.mem_pool_set_attribute(pool, attribute, value);
        });
}

CUresult intercept_mem_pool_get_attribute(CUmemoryPool pool, CUmemPool_attribute attribute,
                                          void* value) {
    if (pool == nullptr || value == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    return forward_optional_driver_call(
        state, [&state] { return state.driver.has_mem_pool_get_attribute(); },
        [&state, pool, attribute, value] {
            return state.driver.mem_pool_get_attribute(pool, attribute, value);
        });
}

CUresult intercept_mem_pool_set_access(CUmemoryPool pool, const CUmemAccessDesc* access_descriptors,
                                       std::size_t descriptor_count) {
    if (pool == nullptr || (descriptor_count != 0 && access_descriptors == nullptr)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    return forward_optional_driver_call(
        state, [&state] { return state.driver.has_mem_pool_set_access(); },
        [&state, pool, access_descriptors, descriptor_count] {
            return state.driver.mem_pool_set_access(pool, access_descriptors, descriptor_count);
        });
}

CUresult intercept_mem_pool_get_access(CUmemAccess_flags* flags, CUmemoryPool pool,
                                       CUmemLocation* location) {
    if (flags == nullptr || pool == nullptr || location == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    return forward_optional_driver_call(
        state, [&state] { return state.driver.has_mem_pool_get_access(); },
        [&state, flags, pool, location] {
            return state.driver.mem_pool_get_access(flags, pool, location);
        });
}

CUresult intercept_mem_pool_create(CUmemoryPool* pool, const CUmemPoolProps* properties) {
    if (pool == nullptr || properties == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    return forward_optional_driver_call(
        state, [&state] { return state.driver.has_mem_pool_create(); },
        [&state, pool, properties] { return state.driver.mem_pool_create(pool, properties); });
}

CUresult intercept_mem_pool_destroy(CUmemoryPool pool) {
    if (pool == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    return forward_optional_driver_call(
        state, [&state] { return state.driver.has_mem_pool_destroy(); },
        [&state, pool] { return state.driver.mem_pool_destroy(pool); });
}

CUresult intercept_device_get_mem_pool(CUmemoryPool* pool, CUdevice device) {
    if (pool == nullptr || device < 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    return forward_optional_driver_call(
        state, [&state] { return state.driver.has_device_get_mem_pool(); },
        [&state, pool, device] { return state.driver.device_get_mem_pool(pool, device); });
}

CUresult intercept_device_set_mem_pool(CUdevice device, CUmemoryPool pool) {
    if (device < 0 || pool == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    return forward_optional_driver_call(
        state, [&state] { return state.driver.has_device_set_mem_pool(); },
        [&state, device, pool] { return state.driver.device_set_mem_pool(device, pool); });
}

CUresult intercept_device_get_default_mem_pool(CUmemoryPool* pool, CUdevice device) {
    if (pool == nullptr || device < 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    return forward_optional_driver_call(
        state, [&state] { return state.driver.has_device_get_default_mem_pool(); },
        [&state, pool, device] { return state.driver.device_get_default_mem_pool(pool, device); });
}

CUresult intercept_mem_get_default_mem_pool(CUmemoryPool* pool, CUmemLocation* location,
                                            CUmemAllocationType allocation_type) {
    if (pool == nullptr || location == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    return forward_optional_driver_call(
        state, [&state] { return state.driver.has_mem_get_default_mem_pool(); },
        [&state, pool, location, allocation_type] {
            return state.driver.mem_get_default_mem_pool(pool, location, allocation_type);
        });
}

CUresult intercept_mem_get_mem_pool(CUmemoryPool* pool, CUmemLocation* location,
                                    CUmemAllocationType allocation_type) {
    if (pool == nullptr || location == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    return forward_optional_driver_call(
        state, [&state] { return state.driver.has_mem_get_mem_pool(); },
        [&state, pool, location, allocation_type] {
            return state.driver.mem_get_mem_pool(pool, location, allocation_type);
        });
}

CUresult intercept_mem_set_mem_pool(CUmemLocation* location, CUmemAllocationType allocation_type,
                                    CUmemoryPool pool) {
    if (location == nullptr || pool == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    return forward_optional_driver_call(
        state, [&state] { return state.driver.has_mem_set_mem_pool(); },
        [&state, location, allocation_type, pool] {
            return state.driver.mem_set_mem_pool(location, allocation_type, pool);
        });
}

CUresult intercept_mem_pool_export_to_shareable_handle(void* handle_out, CUmemoryPool pool,
                                                       CUmemAllocationHandleType handle_type,
                                                       unsigned long long flags) {
    if (handle_out == nullptr || pool == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    return forward_optional_driver_call(
        state, [&state] { return state.driver.has_mem_pool_export_to_shareable_handle(); },
        [&state, handle_out, pool, handle_type, flags] {
            return state.driver.mem_pool_export_to_shareable_handle(handle_out, pool, handle_type,
                                                                    flags);
        });
}

CUresult intercept_mem_pool_import_from_shareable_handle(CUmemoryPool* pool_out, void* handle,
                                                         CUmemAllocationHandleType handle_type,
                                                         unsigned long long flags) {
    if (pool_out == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    if (glimmer::interceptor::is_inside_driver_call() ||
        glimmer::interceptor::is_inside_runtime_call()) {
        return state.driver.mem_pool_import_from_shareable_handle(pool_out, handle, handle_type,
                                                                  flags);
    }
    if (!ensure_initialized(state) || !state.driver.has_mem_pool_import_from_shareable_handle()) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (state.quota_mode == QuotaMode::kEnabled) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (state.quota_mode == QuotaMode::kInvalidConfiguration) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return state.driver.mem_pool_import_from_shareable_handle(pool_out, handle, handle_type, flags);
}

CUresult intercept_mem_pool_export_pointer(CUmemPoolPtrExportData* share_data_out,
                                           CUdeviceptr device_pointer) {
    if (share_data_out == nullptr || device_pointer == 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    return forward_optional_driver_call(
        state, [&state] { return state.driver.has_mem_pool_export_pointer(); },
        [&state, share_data_out, device_pointer] {
            return state.driver.mem_pool_export_pointer(share_data_out, device_pointer);
        });
}

CUresult intercept_mem_pool_import_pointer(CUdeviceptr* pointer_out, CUmemoryPool pool,
                                           CUmemPoolPtrExportData* share_data) {
    if (pointer_out == nullptr || pool == nullptr || share_data == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    if (glimmer::interceptor::is_inside_driver_call() ||
        glimmer::interceptor::is_inside_runtime_call()) {
        return state.driver.mem_pool_import_pointer(pointer_out, pool, share_data);
    }
    if (!ensure_initialized(state) || !state.driver.has_mem_pool_import_pointer()) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (state.quota_mode == QuotaMode::kEnabled) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (state.quota_mode == QuotaMode::kInvalidConfiguration) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return state.driver.mem_pool_import_pointer(pointer_out, pool, share_data);
}

namespace {

[[nodiscard]] nvmlReturn_t initialize_nvml(InterceptorState& state) {
    return state.nvml.initialize() ? NVML_SUCCESS : NVML_ERROR_LIBRARY_NOT_FOUND;
}

// NOLINTBEGIN(bugprone-easily-swappable-parameters): output pointers are paired by name.
[[nodiscard]] nvmlReturn_t virtualize_nvml_memory(InterceptorState& state, nvmlDevice_t device,
                                                  unsigned long long physical_total,
                                                  unsigned long long physical_free,
                                                  unsigned long long* visible_total,
                                                  unsigned long long* visible_free) {
    ensure_state_initialized(state);
    if (state.quota_mode == QuotaMode::kDisabled) {
        *visible_total = physical_total;
        *visible_free = physical_free;
        return NVML_SUCCESS;
    }
    if (state.quota_mode == QuotaMode::kInvalidConfiguration ||
        state.allocations.is_accounting_degraded() || state.quota == nullptr ||
        !state.quota->is_healthy()) {
        return state.quota_mode == QuotaMode::kInvalidConfiguration ? NVML_ERROR_INVALID_ARGUMENT
                                                                    : NVML_ERROR_UNKNOWN;
    }

    unsigned int device_index = 0;
    if (!state.nvml.has_device_get_index() ||
        state.nvml.device_get_index(device, &device_index) != NVML_SUCCESS) {
        return NVML_ERROR_UNKNOWN;
    }
    const glimmer::control::MemoryInfo memory_info = state.quota->get_memory_info(
        static_cast<DeviceId>(device_index), static_cast<MemoryBytes>(physical_total),
        static_cast<MemoryBytes>(physical_free));
    if (!state.quota->is_healthy()) {
        return NVML_ERROR_UNKNOWN;
    }
    *visible_total = static_cast<unsigned long long>(memory_info.total_bytes);
    *visible_free = static_cast<unsigned long long>(memory_info.free_bytes);
    return NVML_SUCCESS;
}
// NOLINTEND(bugprone-easily-swappable-parameters)

}  // namespace

nvmlReturn_t intercept_nvml_init() {
    InterceptorState& state = get_state();
    const nvmlReturn_t initialize_result = initialize_nvml(state);
    return initialize_result == NVML_SUCCESS ? state.nvml.init() : initialize_result;
}

nvmlReturn_t intercept_nvml_init_with_flags(unsigned int flags) {
    InterceptorState& state = get_state();
    const nvmlReturn_t initialize_result = initialize_nvml(state);
    if (initialize_result != NVML_SUCCESS) {
        return initialize_result;
    }
    return state.nvml.init_with_flags(flags);
}

nvmlReturn_t intercept_nvml_shutdown() {
    InterceptorState& state = get_state();
    const nvmlReturn_t initialize_result = initialize_nvml(state);
    return initialize_result == NVML_SUCCESS ? state.nvml.shutdown() : initialize_result;
}

nvmlReturn_t intercept_nvml_device_get_count(unsigned int* device_count) {
    if (device_count == nullptr) {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    InterceptorState& state = get_state();
    const nvmlReturn_t initialize_result = initialize_nvml(state);
    return initialize_result == NVML_SUCCESS ? state.nvml.device_get_count(device_count)
                                             : initialize_result;
}

nvmlReturn_t intercept_nvml_device_get_handle_by_index(unsigned int index, nvmlDevice_t* device) {
    if (device == nullptr) {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    InterceptorState& state = get_state();
    const nvmlReturn_t initialize_result = initialize_nvml(state);
    return initialize_result == NVML_SUCCESS ? state.nvml.device_get_handle_by_index(index, device)
                                             : initialize_result;
}

nvmlReturn_t intercept_nvml_device_get_index(nvmlDevice_t device, unsigned int* index) {
    if (device == nullptr || index == nullptr) {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    InterceptorState& state = get_state();
    const nvmlReturn_t initialize_result = initialize_nvml(state);
    return initialize_result == NVML_SUCCESS ? state.nvml.device_get_index(device, index)
                                             : initialize_result;
}

nvmlReturn_t intercept_nvml_device_get_memory_info(nvmlDevice_t device, nvmlMemory_t* memory) {
    if (device == nullptr || memory == nullptr) {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    InterceptorState& state = get_state();
    const nvmlReturn_t initialize_result = initialize_nvml(state);
    if (initialize_result != NVML_SUCCESS) {
        return initialize_result;
    }
    const nvmlReturn_t query_result = state.nvml.device_get_memory_info(device, memory);
    if (query_result != NVML_SUCCESS) {
        return query_result;
    }
    unsigned long long visible_total = memory->total;
    unsigned long long visible_free = memory->free;
    const nvmlReturn_t virtualize_result = virtualize_nvml_memory(
        state, device, memory->total, memory->free, &visible_total, &visible_free);
    if (virtualize_result != NVML_SUCCESS) {
        return virtualize_result;
    }
    if (state.quota_mode != QuotaMode::kEnabled) {
        return NVML_SUCCESS;
    }
    memory->total = visible_total;
    memory->free = visible_free;
    memory->used = visible_total >= visible_free ? visible_total - visible_free : 0;
    return NVML_SUCCESS;
}

nvmlReturn_t intercept_nvml_device_get_memory_info_v2(nvmlDevice_t device,
                                                      nvmlMemory_v2_t* memory) {
    if (device == nullptr || memory == nullptr) {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    InterceptorState& state = get_state();
    const nvmlReturn_t initialize_result = initialize_nvml(state);
    if (initialize_result != NVML_SUCCESS) {
        return initialize_result;
    }
    if (!state.nvml.has_device_get_memory_info_v2()) {
        return NVML_ERROR_FUNCTION_NOT_FOUND;
    }
    const nvmlReturn_t query_result = state.nvml.device_get_memory_info_v2(device, memory);
    if (query_result != NVML_SUCCESS) {
        return query_result;
    }
    unsigned long long visible_total = memory->total;
    unsigned long long visible_free = memory->free;
    const nvmlReturn_t virtualize_result = virtualize_nvml_memory(
        state, device, memory->total, memory->free, &visible_total, &visible_free);
    if (virtualize_result != NVML_SUCCESS) {
        return virtualize_result;
    }
    if (state.quota_mode != QuotaMode::kEnabled) {
        return NVML_SUCCESS;
    }
    memory->total = visible_total;
    memory->free = visible_free;
    memory->reserved = 0;
    memory->used = visible_total >= visible_free ? visible_total - visible_free : 0;
    return NVML_SUCCESS;
}

}  // namespace glimmer::interceptor
