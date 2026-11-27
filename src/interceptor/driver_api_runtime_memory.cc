#include "internal/driver_api_interceptor_state.h"

namespace glimmer::interceptor {

template <typename Allocate>
cudaError_t intercept_runtime_malloc_impl(void** device_pointer, std::size_t memory_bytes,
                                          Allocate&& allocate, RuntimeFreeFunction release,
                                          RuntimeGetDeviceFunction get_device,
                                          AllocationProvenance provenance) {
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

    if (state.allocations.is_accounting_degraded()) {
        return cudaErrorUnknown;
    }

    if (state.quota == nullptr) {
        return cudaErrorUnknown;
    }
    if (!state.quota->is_healthy()) {
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

    const std::optional<ContextIdentity> context_before_allocation =
        capture_context_identity(state);
    if (context_before_allocation.has_value()) {
        device = static_cast<DeviceId>(context_before_allocation->device);
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
        reservation->abandon();
        state.allocations.mark_accounting_degraded();
        return cudaErrorUnknown;
    }
    const std::optional<ContextIdentity> context = capture_context_identity(state);
    if (!context.has_value() || static_cast<DeviceId>(context->device) != device) {
        const cudaError_t cleanup_result =
            release == nullptr ? cudaErrorUnknown : release(*device_pointer);
        if (cleanup_result != cudaSuccess) {
            reservation->abandon();
            state.allocations.mark_accounting_degraded();
            return cleanup_result;
        }
        return cudaErrorUnknown;
    }
    if (!reservation->commit()) {
        const cudaError_t cleanup_result =
            release == nullptr ? cudaErrorUnknown : release(*device_pointer);
        if (cleanup_result != cudaSuccess) {
            reservation->abandon();
        }
        state.allocations.mark_accounting_degraded();
        return cleanup_result != cudaSuccess ? cleanup_result : cudaErrorUnknown;
    }

    if (!state.allocations.record(make_allocation_identity(*device_pointer, *context), memory_bytes,
                                  AllocationScope::kContextBound, provenance)) {
        state.allocations.mark_accounting_degraded();
        const cudaError_t cleanup_result =
            release == nullptr ? cudaErrorUnknown : release(*device_pointer);
        if (cleanup_result != cudaSuccess) {
            state.allocations.mark_accounting_degraded();
            return cleanup_result;
        }
        if (state.quota == nullptr ||
            !state.quota->release(static_cast<DeviceId>(context->device), memory_bytes)) {
            state.allocations.mark_accounting_degraded();
            return cudaErrorUnknown;
        }
        return cudaErrorMemoryAllocation;
    }
    return cudaSuccess;
}

cudaError_t intercept_runtime_malloc(void** device_pointer, std::size_t memory_bytes,
                                     RuntimeMallocFunction allocate, RuntimeFreeFunction release,
                                     RuntimeGetDeviceFunction get_device) {
    if (allocate == nullptr) {
        return cudaErrorNotSupported;
    }
    return intercept_runtime_malloc_impl(
        device_pointer, memory_bytes,
        [device_pointer, memory_bytes, allocate] {
            mark_runtime_source_memory_issue();
            return allocate(device_pointer, memory_bytes);
        },
        release, get_device, AllocationProvenance::kIndependentDevice);
}

cudaError_t intercept_runtime_malloc_managed(void** device_pointer, std::size_t memory_bytes,
                                             unsigned int flags,
                                             RuntimeMallocManagedFunction allocate,
                                             RuntimeFreeFunction release,
                                             RuntimeGetDeviceFunction get_device) {
    if (allocate == nullptr) {
        return cudaErrorNotSupported;
    }
    return intercept_runtime_malloc_impl(
        device_pointer, memory_bytes,
        [device_pointer, memory_bytes, flags, allocate] {
            mark_runtime_source_memory_issue();
            return allocate(device_pointer, memory_bytes, flags);
        },
        release, get_device, AllocationProvenance::kUnknown);
}

template <typename Allocate>
cudaError_t intercept_runtime_malloc_pitch_impl(void** device_pointer, std::size_t* pitch,
                                                std::size_t width_bytes, std::size_t height,
                                                Allocate&& allocate, RuntimeFreeFunction release,
                                                RuntimeGetDeviceFunction get_device) {
    InterceptorState& state = get_state();
    if (device_pointer == nullptr || pitch == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (!ensure_initialized(state)) {
        return std::getenv("GLIMMER_MEMORY_LIMIT_BYTES") == nullptr
                   ? allocate(device_pointer, pitch, width_bytes, height)
                   : cudaErrorUnknown;
    }

    if (state.quota_mode == QuotaMode::kDisabled) {
        return allocate(device_pointer, pitch, width_bytes, height);
    }
    if (state.quota_mode == QuotaMode::kInvalidConfiguration) {
        return cudaErrorInvalidValue;
    }
    if (state.allocations.is_accounting_degraded()) {
        return cudaErrorUnknown;
    }
    if (state.quota == nullptr || !state.quota->is_healthy()) {
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

    const std::optional<ContextIdentity> context_before_allocation =
        capture_context_identity(state);
    if (context_before_allocation.has_value()) {
        device = static_cast<DeviceId>(context_before_allocation->device);
        device_resolved = true;
    } else if (state.uses_shared_quota && !device_resolved) {
        return cudaErrorInvalidValue;
    }

    const std::optional<MemoryBytes> requested_bytes =
        checked_multiply(static_cast<MemoryBytes>(width_bytes), static_cast<MemoryBytes>(height));
    if (!requested_bytes.has_value()) {
        return cudaErrorInvalidValue;
    }
    auto reservation = state.quota->try_reserve(device, *requested_bytes);
    if (!reservation.has_value()) {
        return state.quota->is_healthy() ? cudaErrorMemoryAllocation : cudaErrorUnknown;
    }

    const cudaError_t allocation_result = allocate(device_pointer, pitch, width_bytes, height);
    if (allocation_result != cudaSuccess) {
        return allocation_result;
    }
    if (*device_pointer == nullptr) {
        reservation->abandon();
        state.allocations.mark_accounting_degraded();
        return cudaErrorUnknown;
    }
    const auto cleanup_allocation = [device_pointer, release] {
        return release == nullptr ? cudaErrorUnknown : release(*device_pointer);
    };

    const std::optional<ContextIdentity> context = capture_context_identity(state);
    if (!context.has_value() || static_cast<DeviceId>(context->device) != device) {
        const cudaError_t cleanup_result = cleanup_allocation();
        if (cleanup_result != cudaSuccess) {
            reservation->abandon();
            state.allocations.mark_accounting_degraded();
            return cleanup_result;
        }
        return cudaErrorUnknown;
    }

    const std::optional<MemoryBytes> actual_bytes =
        checked_multiply(static_cast<MemoryBytes>(*pitch), static_cast<MemoryBytes>(height));
    if (!actual_bytes.has_value()) {
        const cudaError_t cleanup_result = cleanup_allocation();
        if (cleanup_result != cudaSuccess) {
            reservation->abandon();
            state.allocations.mark_accounting_degraded();
            return cleanup_result;
        }
        return cudaErrorInvalidValue;
    }

    std::optional<glimmer::control::MemoryReservation> adjustment;
    if (*actual_bytes > *requested_bytes) {
        adjustment = state.quota->try_reserve(device, *actual_bytes - *requested_bytes);
        if (!adjustment.has_value()) {
            const cudaError_t cleanup_result = cleanup_allocation();
            if (cleanup_result != cudaSuccess) {
                reservation->abandon();
                state.allocations.mark_accounting_degraded();
                return cleanup_result;
            }
            return state.quota->is_healthy() ? cudaErrorMemoryAllocation : cudaErrorUnknown;
        }
    }

    if (!reservation->commit()) {
        const cudaError_t cleanup_result = cleanup_allocation();
        if (cleanup_result != cudaSuccess) {
            reservation->abandon();
            if (adjustment.has_value()) {
                adjustment->abandon();
            }
        }
        state.allocations.mark_accounting_degraded();
        return cleanup_result != cudaSuccess ? cleanup_result : cudaErrorUnknown;
    }

    MemoryBytes committed_bytes = *requested_bytes;
    if (adjustment.has_value()) {
        if (!adjustment->commit()) {
            const cudaError_t cleanup_result = cleanup_allocation();
            if (cleanup_result != cudaSuccess) {
                state.allocations.mark_accounting_degraded();
                return cleanup_result;
            }
            if (!state.quota->release(device, committed_bytes)) {
                state.allocations.mark_accounting_degraded();
            }
            state.allocations.mark_accounting_degraded();
            return cudaErrorUnknown;
        }
        committed_bytes = *actual_bytes;
    } else if (*actual_bytes < *requested_bytes) {
        if (!state.quota->release(device, *requested_bytes - *actual_bytes)) {
            const cudaError_t cleanup_result = cleanup_allocation();
            state.allocations.mark_accounting_degraded();
            return cleanup_result == cudaSuccess ? cudaErrorUnknown : cleanup_result;
        }
        committed_bytes = *actual_bytes;
    }

    if (!state.allocations.record(make_allocation_identity(*device_pointer, *context),
                                  *actual_bytes, AllocationScope::kContextBound,
                                  AllocationProvenance::kIndependentDevice)) {
        state.allocations.mark_accounting_degraded();
        const cudaError_t cleanup_result = cleanup_allocation();
        if (cleanup_result != cudaSuccess) {
            state.allocations.mark_accounting_degraded();
            return cleanup_result;
        }
        if (!state.quota->release(device, committed_bytes)) {
            state.allocations.mark_accounting_degraded();
            return cudaErrorUnknown;
        }
        return cudaErrorMemoryAllocation;
    }
    return cudaSuccess;
}

cudaError_t intercept_runtime_malloc_pitch(void** device_pointer, std::size_t* pitch,
                                           std::size_t width_bytes, std::size_t height,
                                           RuntimeMallocPitchFunction allocate,
                                           RuntimeFreeFunction release,
                                           RuntimeGetDeviceFunction get_device) {
    if (allocate == nullptr) {
        return cudaErrorNotSupported;
    }
    return intercept_runtime_malloc_pitch_impl(
        device_pointer, pitch, width_bytes, height,
        [device_pointer, pitch, width_bytes, height, allocate](void**, std::size_t*, std::size_t,
                                                               std::size_t) {
            mark_runtime_source_memory_issue();
            return allocate(device_pointer, pitch, width_bytes, height);
        },
        release, get_device);
}

cudaError_t intercept_runtime_malloc_3d(struct cudaPitchedPtr* pitched_device_pointer,
                                        struct cudaExtent extent, RuntimeMalloc3DFunction allocate,
                                        RuntimeFreeFunction release,
                                        RuntimeGetDeviceFunction get_device) {
    if (pitched_device_pointer == nullptr) {
        return cudaErrorInvalidValue;
    }
    *pitched_device_pointer = {};
    if (allocate == nullptr) {
        return cudaErrorNotSupported;
    }

    const std::optional<MemoryBytes> row_count = checked_multiply(
        static_cast<MemoryBytes>(extent.height), static_cast<MemoryBytes>(extent.depth));
    if (!row_count.has_value()) {
        return cudaErrorInvalidValue;
    }

    cudaPitchedPtr allocation{};
    void* device_pointer = nullptr;
    std::size_t pitch = 0;
    const cudaError_t result = intercept_runtime_malloc_pitch_impl(
        &device_pointer, &pitch, extent.width, *row_count,
        [&allocation, &device_pointer, &pitch, extent, allocate](void**, std::size_t*, std::size_t,
                                                                 std::size_t) {
            mark_runtime_source_memory_issue();
            const cudaError_t allocation_result = allocate(&allocation, extent);
            if (allocation_result == cudaSuccess) {
                device_pointer = allocation.ptr;
                pitch = allocation.pitch;
            }
            return allocation_result;
        },
        release, get_device);
    if (result == cudaSuccess) {
        *pitched_device_pointer = allocation;
    }
    return result;
}

cudaError_t intercept_runtime_free(void* device_pointer, RuntimeFreeFunction release) {
    InterceptorState& state = get_state();
    if (release == nullptr) {
        return cudaErrorNotSupported;
    }
    const auto invoke = [&] {
        mark_runtime_source_memory_issue();
        return release(device_pointer);
    };
    if (!ensure_initialized(state)) {
        return std::getenv("GLIMMER_MEMORY_LIMIT_BYTES") == nullptr ? invoke() : cudaErrorUnknown;
    }

    // A null release owns no allocation, even when the SDK binds its first
    // context here. Keep the original call and its nested memory boundary.
    if (state.quota_mode != QuotaMode::kEnabled || device_pointer == nullptr) {
        return invoke();
    }

    const std::optional<ContextIdentity> context = capture_context_identity(state);
    if (!context.has_value()) {
        const cudaError_t free_result = invoke();
        if (free_result == cudaSuccess) {
            state.allocations.mark_accounting_degraded();
            return cudaErrorUnknown;
        }
        return free_result;
    }

    auto release_state =
        state.allocations.begin_release(make_allocation_identity(device_pointer, *context));
    if (release_state.first == AllocationRegistry::ReleaseStatus::kUnknown) {
        release_state = state.allocations.begin_release_by_pointer(
            static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(device_pointer)),
            context->device);
    }
    const auto [release_status, release_ticket] = release_state;
    if (release_status == AllocationRegistry::ReleaseStatus::kInProgress) {
        return cudaErrorInvalidValue;
    }
    if (release_status == AllocationRegistry::ReleaseStatus::kUnknown ||
        !release_ticket.has_value()) {
        return invoke();
    }

    const cudaError_t free_result = invoke();
    if (free_result != cudaSuccess) {
        state.allocations.cancel_release(*release_ticket);
        return free_result;
    }

    const bool is_released =
        state.quota != nullptr &&
        state.quota->release(static_cast<DeviceId>(release_ticket->identity.device),
                             release_ticket->memory_bytes);
    const bool is_completed = state.allocations.complete_release(*release_ticket);
    if (!is_released || !is_completed) {
        state.allocations.mark_accounting_degraded();
        return cudaErrorUnknown;
    }
    return cudaSuccess;
}

cudaError_t intercept_runtime_ipc_get_mem_handle(cudaIpcMemHandle_t* handle, void* device_pointer,
                                                 RuntimeIpcGetMemHandleFunction get_handle) {
    if (handle == nullptr || device_pointer == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (get_handle == nullptr) {
        return cudaErrorNotSupported;
    }
    return get_handle(handle, device_pointer);
}

cudaError_t intercept_runtime_ipc_open_mem_handle(void** device_pointer, cudaIpcMemHandle_t handle,
                                                  unsigned int flags,
                                                  RuntimeIpcOpenMemHandleFunction open_handle) {
    if (device_pointer == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (open_handle == nullptr) {
        return cudaErrorNotSupported;
    }
    InterceptorState& state = get_state();
    return forward_untracked_runtime_call(
        state, [&] { return open_handle(device_pointer, handle, flags); });
}

cudaError_t intercept_runtime_ipc_close_mem_handle(void* device_pointer,
                                                   RuntimeIpcCloseMemHandleFunction close_handle) {
    if (device_pointer == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (close_handle == nullptr) {
        return cudaErrorNotSupported;
    }
    return close_handle(device_pointer);
}

cudaError_t intercept_runtime_import_external_memory(
    cudaExternalMemory_t* external_memory, const struct cudaExternalMemoryHandleDesc* handle_desc,
    RuntimeImportExternalMemoryFunction import_memory) {
    if (external_memory == nullptr || handle_desc == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (import_memory == nullptr) {
        return cudaErrorNotSupported;
    }
    InterceptorState& state = get_state();
    return forward_untracked_runtime_call(
        state, [&] { return import_memory(external_memory, handle_desc); });
}

cudaError_t intercept_runtime_external_memory_get_mapped_buffer(
    void** device_pointer, cudaExternalMemory_t external_memory,
    const struct cudaExternalMemoryBufferDesc* buffer_desc,
    RuntimeExternalMemoryGetMappedBufferFunction get_buffer) {
    if (device_pointer == nullptr || external_memory == nullptr || buffer_desc == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (get_buffer == nullptr) {
        return cudaErrorNotSupported;
    }
    InterceptorState& state = get_state();
    return forward_untracked_runtime_call(
        state, [&] { return get_buffer(device_pointer, external_memory, buffer_desc); });
}

cudaError_t intercept_runtime_external_memory_get_mapped_mipmapped_array(
    cudaMipmappedArray_t* mipmap, cudaExternalMemory_t external_memory,
    const struct cudaExternalMemoryMipmappedArrayDesc* mipmap_desc,
    RuntimeExternalMemoryGetMappedMipmappedArrayFunction get_mipmap) {
    if (mipmap == nullptr || external_memory == nullptr || mipmap_desc == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (get_mipmap == nullptr) {
        return cudaErrorNotSupported;
    }
    InterceptorState& state = get_state();
    return forward_untracked_runtime_call(
        state, [&] { return get_mipmap(mipmap, external_memory, mipmap_desc); });
}

cudaError_t intercept_runtime_destroy_external_memory(
    cudaExternalMemory_t external_memory, RuntimeDestroyExternalMemoryFunction destroy_memory) {
    if (external_memory == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (destroy_memory == nullptr) {
        return cudaErrorNotSupported;
    }
    return destroy_memory(external_memory);
}

cudaError_t intercept_runtime_malloc_array(cudaArray_t* array,
                                           const struct cudaChannelFormatDesc* descriptor,
                                           std::size_t width, std::size_t height,
                                           unsigned int flags,
                                           RuntimeMallocArrayFunction allocate) {
    if (array == nullptr || descriptor == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (allocate == nullptr) {
        return cudaErrorNotSupported;
    }
    InterceptorState& state = get_state();
    return forward_untracked_runtime_call(
        state, [&] { return allocate(array, descriptor, width, height, flags); });
}

cudaError_t intercept_runtime_malloc_3d_array(cudaArray_t* array,
                                              const struct cudaChannelFormatDesc* descriptor,
                                              struct cudaExtent extent, unsigned int flags,
                                              RuntimeMalloc3DArrayFunction allocate) {
    if (array == nullptr || descriptor == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (allocate == nullptr) {
        return cudaErrorNotSupported;
    }
    InterceptorState& state = get_state();
    return forward_untracked_runtime_call(
        state, [&] { return allocate(array, descriptor, extent, flags); });
}

cudaError_t intercept_runtime_malloc_mipmapped_array(cudaMipmappedArray_t* mipmap,
                                                     const struct cudaChannelFormatDesc* descriptor,
                                                     struct cudaExtent extent,
                                                     unsigned int level_count, unsigned int flags,
                                                     RuntimeMallocMipmappedArrayFunction allocate) {
    if (mipmap == nullptr || descriptor == nullptr || level_count == 0) {
        return cudaErrorInvalidValue;
    }
    if (allocate == nullptr) {
        return cudaErrorNotSupported;
    }
    InterceptorState& state = get_state();
    return forward_untracked_runtime_call(
        state, [&] { return allocate(mipmap, descriptor, extent, level_count, flags); });
}

cudaError_t intercept_runtime_free_array(cudaArray_t array, RuntimeFreeArrayFunction release) {
    if (array == nullptr) {
        return cudaErrorInvalidValue;
    }
    return release == nullptr ? cudaErrorNotSupported : release(array);
}

cudaError_t intercept_runtime_free_mipmapped_array(cudaMipmappedArray_t mipmap,
                                                   RuntimeFreeMipmappedArrayFunction release) {
    if (mipmap == nullptr) {
        return cudaErrorInvalidValue;
    }
    return release == nullptr ? cudaErrorNotSupported : release(mipmap);
}

cudaError_t intercept_runtime_graph_add_mem_alloc_node(
    cudaGraphNode_t* graph_node, cudaGraph_t graph, const cudaGraphNode_t* dependencies,
    std::size_t dependency_count, struct cudaMemAllocNodeParams* parameters,
    RuntimeGraphAddMemAllocNodeFunction add_node) {
    if (graph_node == nullptr || graph == nullptr || parameters == nullptr ||
        (dependency_count != 0 && dependencies == nullptr)) {
        return cudaErrorInvalidValue;
    }
    if (add_node == nullptr) {
        return cudaErrorNotSupported;
    }
    InterceptorState& state = get_state();
    // Graph memory nodes allocate outside the ordinary malloc/free ABI. We
    // reject them until their lifetime can be charged across graph launch,
    // auto-free, update, and process-exit transitions.
    return forward_untracked_runtime_call(state, [&] {
        return add_node(graph_node, graph, dependencies, dependency_count, parameters);
    });
}

cudaError_t intercept_runtime_graph_instantiation(
    cudaGraphExec_t* output, cudaGraph_t graph, bool should_prepare_effects,
    const std::function<cudaError_t(cudaGraphExec_t*)>& original) {
    static_assert(std::is_same_v<cudaGraph_t, CUgraph>);
    static_assert(std::is_same_v<cudaGraphExec_t, CUgraphExec>);
    if (!original) {
        return cudaErrorNotSupported;
    }
    if (!cooperative_runtime_preview_requested() || is_inside_driver_call() ||
        g_is_inside_cooperative_boundary) {
        return original(output);
    }
    auto& gate = cooperative_process_call_gate();
    if (!gate.is_entered_by_current_thread() || gate.is_native_only_on_current_thread()) {
        return cudaErrorNotSupported;
    }
    auto& state = get_state();
    if (!ensure_initialized(state) || !state.cooperative_router) {
        return cudaErrorNotSupported;
    }
    const CooperativeBoundaryScope boundary;
    auto& router = *state.cooperative_router;
    if (router.normal_owner_status() != CUDA_SUCCESS) {
        return cudaErrorUnknown;
    }
    CudaNativeGraphReservation reservation;
    if (output) {
        const auto prepared =
            router.prepare_graph(nullptr, graph, reservation, should_prepare_effects);
        if (prepared != CUDA_SUCCESS) {
            return prepared == CUDA_ERROR_OUT_OF_MEMORY ? cudaErrorMemoryAllocation
                                                        : cudaErrorUnknown;
        }
    }
    cudaError_t native_status{cudaErrorUnknown};
    bool was_invoked{};
    const auto owned = router.native_boundary(
        [&] {
            cudaGraphExec_t staged{};
            {
                const NativeDriverCallScope native;
                was_invoked = true;
                try {
                    native_status = original(output ? &staged : nullptr);
                } catch (...) {
                    // A staged handle may already be an actual SDK effect.
                    // Retain it as unprepared; never replay or guess cleanup.
                    native_status = cudaErrorUnknown;
                }
            }
            CUresult published{CUDA_SUCCESS};
            if (output) {
                std::optional<CUcontext> actual_context;
                if (staged) {
                    CUcontext current{};
                    const auto queried = state.driver.context_get_current(&current);
                    if (queried == CUDA_SUCCESS) {
                        actual_context = current;
                    }
                }
                published = router.graphs().publish(
                    reservation, staged,
                    native_status == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN,
                    actual_context);
                if (staged || native_status == cudaSuccess) {
                    *output = staged;
                }
            }
            return published;
        },
        [] { return intercept_context_synchronize(); });
    return was_invoked && native_status != cudaSuccess ? native_status
           : owned == CUDA_SUCCESS                     ? native_status
                                                       : cudaErrorUnknown;
}

cudaError_t intercept_runtime_graph_exec_destroy(cudaGraphExec_t executable,
                                                 const std::function<cudaError_t()>& original) {
    static_assert(std::is_same_v<cudaGraphExec_t, CUgraphExec>);
    if (!original) {
        return cudaErrorNotSupported;
    }
    if (!cooperative_runtime_preview_requested() || is_inside_driver_call() ||
        g_is_inside_cooperative_boundary) {
        return original();
    }
    auto& gate = cooperative_process_call_gate();
    if (!gate.is_entered_by_current_thread() || gate.is_native_only_on_current_thread()) {
        return cudaErrorNotSupported;
    }
    auto& state = get_state();
    if (!ensure_initialized(state) || !state.cooperative_router) {
        return cudaErrorNotSupported;
    }
    const CooperativeBoundaryScope boundary;
    auto& graphs = state.cooperative_router->graphs();
    graphs.invalidate(executable);
    const NativeDriverCallScope native;
    const auto status = original();
    if (status == cudaSuccess) {
        graphs.forget(executable);
    }
    return status;
}

cudaError_t intercept_runtime_graph_event_update(cudaGraphExec_t executable, cudaGraphNode_t node,
                                                 cudaEvent_t event, bool is_record,
                                                 const std::function<cudaError_t()>& original) {
    static_assert(std::is_same_v<cudaGraphExec_t, CUgraphExec>);
    static_assert(std::is_same_v<cudaGraphNode_t, CUgraphNode>);
    static_assert(std::is_same_v<cudaEvent_t, CUevent>);
    if (!original) return cudaErrorNotSupported;
    if (!cooperative_runtime_preview_requested() || is_inside_driver_call() ||
        g_is_inside_cooperative_boundary)
        return original();
    auto& gate = cooperative_process_call_gate();
    if (!gate.is_entered_by_current_thread() || gate.is_native_only_on_current_thread()) {
        return cudaErrorNotSupported;
    }
    auto& state = get_state();
    if (!ensure_initialized(state) || !state.cooperative_router) return cudaErrorNotSupported;
    const CooperativeBoundaryScope boundary;
    auto& router = *state.cooperative_router;
    CudaNativeGraphReservation replacement;
    if (executable) {
        const auto prepared =
            router.prepare_graph_event_update(executable, node, event, is_record, replacement);
        if (prepared != CUDA_SUCCESS && prepared != CUDA_ERROR_INVALID_HANDLE) {
            return prepared == CUDA_ERROR_OUT_OF_MEMORY ? cudaErrorMemoryAllocation
                                                        : cudaErrorUnknown;
        }
    }
    const NativeDriverCallScope native;
    const auto status = original();
    const auto published =
        replacement ? router.graphs().publish_event_update(
                          replacement, status == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN)
                    : CUDA_SUCCESS;
    return status != cudaSuccess || published == CUDA_SUCCESS ? status : cudaErrorUnknown;
}

cudaError_t intercept_runtime_graph_node_update(cudaGraphExec_t executable, cudaGraphNode_t node,
                                                std::optional<CUgraphNodeType> expected_type,
                                                const std::function<cudaError_t()>& original) {
    static_assert(std::is_same_v<cudaGraphExec_t, CUgraphExec>);
    static_assert(std::is_same_v<cudaGraphNode_t, CUgraphNode>);
    if (!original) return cudaErrorNotSupported;
    if (!cooperative_runtime_preview_requested() || is_inside_driver_call() ||
        g_is_inside_cooperative_boundary)
        return original();
    auto& gate = cooperative_process_call_gate();
    if (!gate.is_entered_by_current_thread() || gate.is_native_only_on_current_thread())
        return cudaErrorNotSupported;
    auto& state = get_state();
    if (!ensure_initialized(state) || !state.cooperative_router) return cudaErrorNotSupported;
    const CooperativeBoundaryScope boundary;
    auto& graphs = state.cooperative_router->graphs();
    CudaNativeGraphReservation replacement;
    if (executable) {
        const auto prepared =
            graphs.prepare_node_update(executable, node, expected_type, replacement);
        if (prepared != CUDA_SUCCESS && prepared != CUDA_ERROR_INVALID_HANDLE)
            return prepared == CUDA_ERROR_OUT_OF_MEMORY ? cudaErrorMemoryAllocation
                                                        : cudaErrorUnknown;
    }
    const NativeDriverCallScope native;
    const auto status = original();
    const auto published =
        replacement ? graphs.publish_event_update(
                          replacement, status == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN)
                    : CUDA_SUCCESS;
    return status != cudaSuccess || published == CUDA_SUCCESS ? status : cudaErrorUnknown;
}

cudaError_t intercept_runtime_graph_update(cudaGraphExec_t executable, cudaGraph_t graph,
                                           std::optional<cudaGraphNode_t> child_node,
                                           const std::function<cudaError_t()>& original) {
    static_assert(std::is_same_v<cudaGraphExec_t, CUgraphExec>);
    static_assert(std::is_same_v<cudaGraph_t, CUgraph>);
    static_assert(std::is_same_v<cudaGraphNode_t, CUgraphNode>);
    if (!original) return cudaErrorNotSupported;
    if (!cooperative_runtime_preview_requested() || is_inside_driver_call() ||
        g_is_inside_cooperative_boundary)
        return original();
    auto& gate = cooperative_process_call_gate();
    if (!gate.is_entered_by_current_thread() || gate.is_native_only_on_current_thread())
        return cudaErrorNotSupported;
    auto& state = get_state();
    if (!ensure_initialized(state) || !state.cooperative_router) return cudaErrorNotSupported;
    const CooperativeBoundaryScope boundary;
    auto& router = *state.cooperative_router;
    CudaNativeGraphReservation replacement;
    if (executable) {
        const auto prepared =
            router.prepare_graph_update(executable, graph, child_node, replacement);
        if (prepared != CUDA_SUCCESS && prepared != CUDA_ERROR_INVALID_HANDLE)
            return prepared == CUDA_ERROR_OUT_OF_MEMORY ? cudaErrorMemoryAllocation
                                                        : cudaErrorUnknown;
    }
    const NativeDriverCallScope native;
    const auto status = original();
    const auto published =
        replacement ? router.graphs().publish_event_update(
                          replacement, status == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN)
                    : CUDA_SUCCESS;
    return status != cudaSuccess || published == CUDA_SUCCESS ? status : cudaErrorUnknown;
}

CUresult intercept_driver_capture(CUstream stream, bool is_ptds, CudaSourceCaptureKind kind,
                                  const std::function<CUresult()>& original) {
    if (!original) return CUDA_ERROR_NOT_SUPPORTED;
    if (!cooperative_driver_preview_requested() || is_inside_driver_call() ||
        is_inside_runtime_call())
        return original();
    auto& gate = cooperative_process_call_gate();
    auto& state = get_state();
    if (!gate.is_entered_by_current_thread() || !ensure_initialized(state) ||
        !state.cooperative_router)
        return CUDA_ERROR_NOT_SUPPORTED;
    const CooperativeBoundaryScope boundary;
    return state.cooperative_router->source_capture_root(gate, kind, [&](auto& dispatch) {
        CUcontext context{};
        CUresult queried{};
        {
            const NativeDriverCallScope native;
            queried = is_ptds ? state.driver.stream_get_context_ptsz(stream, &context)
                              : state.driver.stream_get_context(stream, &context);
        }
        if (queried == CUDA_ERROR_STREAM_CAPTURE_INVALIDATED &&
            kind == CudaSourceCaptureKind::kEnd) {
            context = nullptr;
            queried = dispatch.lookup_origin_context(stream, is_ptds, context);
        }
        if (queried != CUDA_SUCCESS || !context)
            return queried == CUDA_SUCCESS ? CUDA_ERROR_INVALID_CONTEXT : queried;
        return dispatch.invoke(context, stream, is_ptds, kind, original);
    });
}

cudaError_t intercept_runtime_capture(CudaSourceCaptureKind kind,
                                      const std::function<cudaError_t()>& original) noexcept {
    try {
        if (!original) return cudaErrorNotSupported;
        if (is_inside_runtime_call() || is_inside_driver_call()) return original();
        if (!cooperative_runtime_preview_requested())
            return cooperative_driver_preview_requested() ? cudaErrorNotSupported : original();
        auto& gate = cooperative_process_call_gate();
        auto& state = get_state();
        if (!gate.is_entered_by_current_thread() || !ensure_initialized(state) ||
            !state.cooperative_router)
            return cudaErrorNotSupported;
        const CooperativeBoundaryScope boundary;
        RuntimeCaptureObservation observation;
        cudaError_t native_status{cudaErrorUnknown};
        const auto owned =
            state.cooperative_router->source_capture_root(gate, kind, [&](auto& dispatch) {
                return invoke_runtime_capture_original(
                    [&](CUcontext context, CUstream stream, bool is_ptds,
                        CudaSourceCaptureKind actual_kind, const auto& native) {
                        return dispatch.invoke(context, stream, is_ptds, actual_kind, native);
                    },
                    original, observation, native_status,
                    [&](CUstream stream, bool is_ptds, CUcontext& context) {
                        return dispatch.lookup_origin_context(stream, is_ptds, context);
                    });
            });
        if (observation.was_runtime_invoked && native_status != cudaSuccess) return native_status;
        return owned == CUDA_SUCCESS && observation.was_runtime_invoked ? cudaSuccess
                                                                        : cudaErrorUnknown;
    } catch (...) {
        return cudaErrorUnknown;
    }
}

cudaError_t execute_owned_runtime_graph(cudaGraphExec_t executable, cudaStream_t stream,
                                        bool is_ptds,
                                        const std::function<cudaError_t()>& original) noexcept {
    static_assert(std::is_same_v<cudaGraphExec_t, CUgraphExec>);
    static_assert(std::is_same_v<cudaStream_t, CUstream>);
    try {
        auto& gate = cooperative_process_call_gate();
        if (!executable || !original) {
            return cudaErrorInvalidValue;
        }
        if (!cooperative_runtime_preview_requested() || !gate.is_entered_by_current_thread() ||
            is_inside_driver_call() || g_is_inside_cooperative_boundary) {
            return cudaErrorNotSupported;
        }
        auto& state = get_state();
        if (!ensure_initialized(state) || !state.cooperative_router) {
            return cudaErrorNotSupported;
        }
        const CooperativeBoundaryScope boundary;
        auto& router = *state.cooperative_router;
        if (!gate.is_native_only_on_current_thread() &&
            router.normal_owner_status() != CUDA_SUCCESS) {
            return cudaErrorUnknown;
        }
        return invoke_runtime_graph_original(
            [&](const auto& issue) {
                return execute_cooperative_graph(state, executable, stream, is_ptds, issue);
            },
            original);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

cudaError_t intercept_runtime_graph_launch(const char* api_name, cudaGraphExec_t graph_exec,
                                           cudaStream_t stream, RuntimeGraphLaunchFunction launch,
                                           bool per_thread_default_stream) {
    if (cooperative_driver_preview_requested()) {
        if (!cooperative_runtime_preview_requested() || !launch) return cudaErrorNotSupported;
        return execute_owned_runtime_graph(graph_exec, stream, per_thread_default_stream,
                                           [&] { return launch(graph_exec, stream); });
    }
    if (graph_exec == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (launch == nullptr) {
        return cudaErrorNotSupported;
    }
    const bool scheduling_enforced = launch_scheduling_is_enforced();
    const bool trace_launch_timings = launch_timing_is_enabled();
    glimmer::control::LaunchGateTiming gate_timing;
    std::optional<glimmer::core::TaskId> launch_task;
    if (scheduling_enforced) {
        const auto capturing =
            runtime_launch_stream_is_capturing(stream, per_thread_default_stream);
        if (!capturing.has_value()) {
            return cudaErrorNotSupported;
        }
        if (capturing.value()) {
            return launch(graph_exec, stream);
        }
        launch_task = acquire_launch_slot(trace_launch_timings ? &gate_timing : nullptr);
        if (!launch_task.has_value()) {
            return cudaErrorNotSupported;
        }
    }

    const auto cuda_launch_started = trace_launch_timings ? std::chrono::steady_clock::now()
                                                          : std::chrono::steady_clock::time_point{};
    const cudaError_t launch_result = launch(graph_exec, stream);
    const auto cuda_launch_finished = trace_launch_timings
                                          ? std::chrono::steady_clock::now()
                                          : std::chrono::steady_clock::time_point{};
    if (launch_task.has_value() && launch_result != cudaSuccess) {
        fail_launch_slot(launch_task.value());
    }

    bool event_tracking_succeeded = false;
    bool batch_call_succeeded = true;
    std::uint64_t event_tracking_nanoseconds = 0;
    if (launch_task.has_value() && launch_result == cudaSuccess) {
        const auto event_tracking_started = trace_launch_timings
                                                ? std::chrono::steady_clock::now()
                                                : std::chrono::steady_clock::time_point{};
        event_tracking_succeeded = track_launch_completion(
            launch_task.value(), per_thread_default_stream && stream == nullptr
                                     ? CU_STREAM_PER_THREAD
                                     : reinterpret_cast<CUstream>(stream));
        if (trace_launch_timings) {
            event_tracking_nanoseconds =
                elapsed_nanoseconds(event_tracking_started, std::chrono::steady_clock::now());
        }
        if (event_tracking_succeeded) {
            batch_call_succeeded = finish_launch_batch_call(launch_task.value());
        }
    }
    if (trace_launch_timings) {
        report_launch_timing_observed(
            launch_task.value_or(0), gate_timing,
            elapsed_nanoseconds(cuda_launch_started, cuda_launch_finished),
            event_tracking_nanoseconds, static_cast<std::uint32_t>(launch_result),
            event_tracking_succeeded);
    }
    if (launch_task.has_value() && launch_result == cudaSuccess &&
        (!event_tracking_succeeded || !batch_call_succeeded)) {
        return cudaErrorUnknown;
    }
    if (launch_result == cudaSuccess) {
        report_graph_launch_observed(api_name, reinterpret_cast<const void*>(stream));
    }
    return launch_result;
}

cudaError_t intercept_runtime_graphics_unregister_resource(
    cudaGraphicsResource_t resource, RuntimeGraphicsUnregisterResourceFunction unregister) {
    if (resource == nullptr) {
        return cudaErrorInvalidValue;
    }
    return unregister == nullptr ? cudaErrorNotSupported : unregister(resource);
}

cudaError_t intercept_runtime_graphics_resource_set_map_flags(
    cudaGraphicsResource_t resource, unsigned int flags,
    RuntimeGraphicsResourceSetMapFlagsFunction set_flags) {
    if (resource == nullptr) {
        return cudaErrorInvalidValue;
    }
    return set_flags == nullptr ? cudaErrorNotSupported : set_flags(resource, flags);
}

cudaError_t intercept_runtime_graphics_map_resources(
    int count, cudaGraphicsResource_t* resources, cudaStream_t stream,
    RuntimeGraphicsMapResourcesFunction map_resources) {
    if (count < 0 || (count != 0 && resources == nullptr)) {
        return cudaErrorInvalidValue;
    }
    if (map_resources == nullptr) {
        return cudaErrorNotSupported;
    }
    InterceptorState& state = get_state();
    return forward_untracked_runtime_call(state,
                                          [&] { return map_resources(count, resources, stream); });
}

cudaError_t intercept_runtime_graphics_unmap_resources(
    int count, cudaGraphicsResource_t* resources, cudaStream_t stream,
    RuntimeGraphicsUnmapResourcesFunction unmap_resources) {
    if (count < 0 || (count != 0 && resources == nullptr)) {
        return cudaErrorInvalidValue;
    }
    return unmap_resources == nullptr ? cudaErrorNotSupported
                                      : unmap_resources(count, resources, stream);
}

cudaError_t intercept_runtime_graphics_resource_get_mapped_pointer(
    void** device_pointer, std::size_t* size, cudaGraphicsResource_t resource,
    RuntimeGraphicsResourceGetMappedPointerFunction get_pointer) {
    if (device_pointer == nullptr || size == nullptr || resource == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (get_pointer == nullptr) {
        return cudaErrorNotSupported;
    }
    InterceptorState& state = get_state();
    return forward_untracked_runtime_call(
        state, [&] { return get_pointer(device_pointer, size, resource); });
}

cudaError_t intercept_runtime_graphics_subresource_get_mapped_array(
    cudaArray_t* array, cudaGraphicsResource_t resource, unsigned int array_index,
    unsigned int mip_level, RuntimeGraphicsSubResourceGetMappedArrayFunction get_array) {
    if (array == nullptr || resource == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (get_array == nullptr) {
        return cudaErrorNotSupported;
    }
    InterceptorState& state = get_state();
    return forward_untracked_runtime_call(
        state, [&] { return get_array(array, resource, array_index, mip_level); });
}

cudaError_t intercept_runtime_graphics_resource_get_mapped_mipmapped_array(
    cudaMipmappedArray_t* mipmap, cudaGraphicsResource_t resource,
    RuntimeGraphicsResourceGetMappedMipmappedArrayFunction get_mipmap) {
    if (mipmap == nullptr || resource == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (get_mipmap == nullptr) {
        return cudaErrorNotSupported;
    }
    InterceptorState& state = get_state();
    return forward_untracked_runtime_call(state, [&] { return get_mipmap(mipmap, resource); });
}

cudaError_t intercept_runtime_mem_get_info(std::size_t* free_bytes, std::size_t* total_bytes,
                                           RuntimeMemGetInfoFunction query) {
    InterceptorState& state = get_state();
    if (query == nullptr) {
        return cudaErrorNotSupported;
    }
    if (free_bytes == nullptr || total_bytes == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (!ensure_initialized(state)) {
        return std::getenv("GLIMMER_MEMORY_LIMIT_BYTES") == nullptr ? query(free_bytes, total_bytes)
                                                                    : cudaErrorUnknown;
    }

    if (state.allocations.is_accounting_degraded()) {
        return cudaErrorUnknown;
    }

    const cudaError_t info_result = query(free_bytes, total_bytes);
    if (info_result != cudaSuccess || state.quota_mode != QuotaMode::kEnabled) {
        return info_result;
    }

    if (state.quota == nullptr) {
        return cudaErrorUnknown;
    }
    const std::optional<ContextIdentity> context = capture_context_identity(state);
    if (!context.has_value() && state.uses_shared_quota) {
        return cudaErrorInvalidValue;
    }
    const DeviceId device =
        context.has_value() ? static_cast<DeviceId>(context->device) : QuotaStore::default_device;
    const std::size_t physical_total_bytes = *total_bytes;
    const std::size_t physical_free_bytes = *free_bytes;
    const glimmer::control::MemoryInfo memory_info =
        state.quota->get_memory_info(device, *total_bytes, *free_bytes);
    if (!state.quota->is_healthy()) {
        return cudaErrorUnknown;
    }
    *total_bytes = static_cast<std::size_t>(memory_info.total_bytes);
    *free_bytes = static_cast<std::size_t>(memory_info.free_bytes);
    report_memory_info_observed("cudaMemGetInfo", device, static_cast<std::uint64_t>(*total_bytes),
                                static_cast<std::uint64_t>(*free_bytes), physical_total_bytes,
                                physical_free_bytes);
    return cudaSuccess;
}

}  // namespace glimmer::interceptor
