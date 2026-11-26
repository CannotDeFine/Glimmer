#include "internal/runtime_api_bridge.h"

#include "internal/driver_api_interceptor.h"
#include "internal/driver_dispatch.h"
#include "internal/cuda_native_graph_effects.h"

#include "glimmer/control/launch_gate.h"

#include <array>
#include <atomic>
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <optional>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#ifdef cudaMemcpy
#undef cudaMemcpy
#endif
#ifdef cudaMemcpy_ptds
#undef cudaMemcpy_ptds
#endif
#ifdef cudaMemcpyAsync
#undef cudaMemcpyAsync
#endif
#ifdef cudaMemcpyAsync_ptsz
#undef cudaMemcpyAsync_ptsz
#endif
#ifdef cudaMemset
#undef cudaMemset
#endif
#ifdef cudaMemset_ptds
#undef cudaMemset_ptds
#endif
#ifdef cudaMemsetAsync
#undef cudaMemsetAsync
#endif
#ifdef cudaMemsetAsync_ptsz
#undef cudaMemsetAsync_ptsz
#endif
#ifdef cudaMallocHost
#undef cudaMallocHost
#endif
#ifdef cudaHostAlloc
#undef cudaHostAlloc
#endif
#ifdef cudaFreeHost
#undef cudaFreeHost
#endif
#ifdef cudaHostRegister
#undef cudaHostRegister
#endif
#ifdef cudaHostUnregister
#undef cudaHostUnregister
#endif
#ifdef cudaHostGetDevicePointer
#undef cudaHostGetDevicePointer
#endif
#ifdef cudaStreamCreate
#undef cudaStreamCreate
#endif
#ifdef cudaStreamCreateWithFlags
#undef cudaStreamCreateWithFlags
#endif
#ifdef cudaStreamCreateWithPriority
#undef cudaStreamCreateWithPriority
#endif
#ifdef cudaStreamGetFlags
#undef cudaStreamGetFlags
#endif
#ifdef cudaStreamGetPriority
#undef cudaStreamGetPriority
#endif

#ifdef cudaEventRecord
#undef cudaEventRecord
#endif

#ifdef cudaEventRecordWithFlags
#undef cudaEventRecordWithFlags
#endif

#ifdef cudaStreamWaitEvent
#undef cudaStreamWaitEvent
#endif
#ifdef cudaStreamBeginCapture
#undef cudaStreamBeginCapture
#endif
#ifdef cudaStreamEndCapture
#undef cudaStreamEndCapture
#endif

#ifdef cudaMallocFromPoolAsync
#undef cudaMallocFromPoolAsync
#endif
#ifdef cudaLaunchKernel
#undef cudaLaunchKernel
#endif
#ifdef cudaGraphLaunch
#undef cudaGraphLaunch
#endif
#ifdef cudaGraphInstantiateWithParams
#undef cudaGraphInstantiateWithParams
#endif
#ifdef cudaLaunchKernelExC
#undef cudaLaunchKernelExC
#endif

namespace {

thread_local bool g_is_inside_runtime_call = false;
thread_local bool g_should_route_runtime_driver = false;
struct PendingRuntimeCapture {
    const glimmer::interceptor::RuntimeCaptureDispatch& dispatch;
    glimmer::interceptor::RuntimeCaptureObservation& observation;
    const glimmer::interceptor::RuntimeCaptureContextLookup& lookup;
};
thread_local PendingRuntimeCapture* g_pending_runtime_capture{};
struct PendingRuntimeSourceKernel {
    const glimmer::interceptor::RuntimeSourceKernelShape& shape;
    bool has_dispatched{};
    bool has_rejected{};
    CUresult driver_status{CUDA_ERROR_UNKNOWN};
};
thread_local PendingRuntimeSourceKernel* g_pending_runtime_source_kernel{};
thread_local bool* g_runtime_source_memory_issue{};
class RuntimeSourceMemoryScope final {
   public:
    explicit RuntimeSourceMemoryScope(bool& has_native_issue) noexcept {
        g_runtime_source_memory_issue = &has_native_issue;
    }
    ~RuntimeSourceMemoryScope() {
        g_runtime_source_memory_issue = nullptr;
    }
    RuntimeSourceMemoryScope(const RuntimeSourceMemoryScope&) = delete;
    RuntimeSourceMemoryScope& operator=(const RuntimeSourceMemoryScope&) = delete;
};
class RuntimeSourceKernelScope final {
   public:
    explicit RuntimeSourceKernelScope(PendingRuntimeSourceKernel& pending) noexcept {
        g_pending_runtime_source_kernel = &pending;
    }
    ~RuntimeSourceKernelScope() {
        g_pending_runtime_source_kernel = nullptr;
    }
    RuntimeSourceKernelScope(const RuntimeSourceKernelScope&) = delete;
    RuntimeSourceKernelScope& operator=(const RuntimeSourceKernelScope&) = delete;
};

class RuntimeCaptureScope final {
   public:
    explicit RuntimeCaptureScope(PendingRuntimeCapture& pending) noexcept {
        g_pending_runtime_capture = &pending;
    }
    ~RuntimeCaptureScope() {
        g_pending_runtime_capture = nullptr;
    }
    RuntimeCaptureScope(const RuntimeCaptureScope&) = delete;
    RuntimeCaptureScope& operator=(const RuntimeCaptureScope&) = delete;
};
struct PendingRuntimeCooperativeLaunch {
    glimmer::interceptor::RuntimeCooperativeLaunch plan;
    const bool is_ptds{};
    const glimmer::interceptor::RuntimeNativeKernelExecution& native_execution;
    bool was_consumed = false;
    bool has_returned{};
    bool has_rejected{};
    CUresult driver_status{CUDA_ERROR_UNKNOWN};
};
thread_local PendingRuntimeCooperativeLaunch* g_pending_runtime_cooperative = nullptr;

class RuntimeCooperativeLaunchScope final {
   public:
    explicit RuntimeCooperativeLaunchScope(PendingRuntimeCooperativeLaunch& pending)
        : previous_(g_pending_runtime_cooperative) {
        g_pending_runtime_cooperative = &pending;
    }
    ~RuntimeCooperativeLaunchScope() {
        g_pending_runtime_cooperative = previous_;
    }
    RuntimeCooperativeLaunchScope(const RuntimeCooperativeLaunchScope&) = delete;
    RuntimeCooperativeLaunchScope& operator=(const RuntimeCooperativeLaunchScope&) = delete;

   private:
    PendingRuntimeCooperativeLaunch* previous_;
};

class RuntimeNativeDriverScope final {
   public:
    RuntimeNativeDriverScope() : previous_(g_should_route_runtime_driver) {
        g_should_route_runtime_driver = false;
    }
    ~RuntimeNativeDriverScope() {
        g_should_route_runtime_driver = previous_;
    }
    RuntimeNativeDriverScope(const RuntimeNativeDriverScope&) = delete;
    RuntimeNativeDriverScope& operator=(const RuntimeNativeDriverScope&) = delete;

   private:
    bool previous_;
};
struct PendingRuntimeReplacement {
    glimmer::interceptor::RuntimePersistentLaunch variant;
    void** arguments = nullptr;
    bool was_consumed = false;
};
thread_local PendingRuntimeReplacement* g_pending_runtime_replacement = nullptr;

class RuntimeReplacementScope {
   public:
    explicit RuntimeReplacementScope(PendingRuntimeReplacement& replacement)
        : previous_(g_pending_runtime_replacement) {
        g_pending_runtime_replacement = &replacement;
    }
    ~RuntimeReplacementScope() {
        g_pending_runtime_replacement = previous_;
    }

   private:
    PendingRuntimeReplacement* previous_;
};

class RuntimeCallScope {
   public:
    explicit RuntimeCallScope(bool should_route_driver = false)
        : previous_state_(g_is_inside_runtime_call),
          previous_route_(g_should_route_runtime_driver) {
        if (!previous_state_) {
            g_should_route_runtime_driver =
                should_route_driver &&
                glimmer::interceptor::cooperative_runtime_preview_requested();
        }
        g_is_inside_runtime_call = true;
    }

    ~RuntimeCallScope() {
        g_is_inside_runtime_call = previous_state_;
        g_should_route_runtime_driver = previous_route_;
    }

   private:
    glimmer::interceptor::CooperativeCallScope cooperative_call_;
    bool previous_state_;
    bool previous_route_;
};

// One known Runtime event original owns source metadata. Cached native SDK
// functions may bypass Driver wrappers; covered callbacks must not retire again
// or enter another transition. Keep the Runtime frame visible, not a Driver scope.
class RuntimeEventOriginalScope final {
   public:
    RuntimeEventOriginalScope() noexcept : previous_(g_should_route_runtime_driver) {
        g_should_route_runtime_driver = false;
    }
    ~RuntimeEventOriginalScope() {
        g_should_route_runtime_driver = previous_;
    }
    RuntimeEventOriginalScope(const RuntimeEventOriginalScope&) = delete;
    RuntimeEventOriginalScope& operator=(const RuntimeEventOriginalScope&) = delete;

   private:
    bool previous_;
};

using glimmer::interceptor::DlsymFunction;
using glimmer::interceptor::RuntimeDestroyExternalMemoryFunction;
using glimmer::interceptor::RuntimeDeviceGetDefaultMemPoolFunction;
using glimmer::interceptor::RuntimeDeviceGetMemPoolFunction;
using glimmer::interceptor::RuntimeDeviceSetMemPoolFunction;
using glimmer::interceptor::RuntimeDeviceSynchronizeFunction;
using glimmer::interceptor::RuntimeExternalMemoryGetMappedBufferFunction;
using glimmer::interceptor::RuntimeExternalMemoryGetMappedMipmappedArrayFunction;
using glimmer::interceptor::RuntimeFreeArrayFunction;
using glimmer::interceptor::RuntimeFreeAsyncFunction;
using glimmer::interceptor::RuntimeFreeFunction;
using glimmer::interceptor::RuntimeFreeMipmappedArrayFunction;
using glimmer::interceptor::RuntimeGetDeviceFunction;
using glimmer::interceptor::RuntimeGraphAddMemAllocNodeFunction;
using glimmer::interceptor::RuntimeGraphicsMapResourcesFunction;
using glimmer::interceptor::RuntimeGraphicsResourceGetMappedMipmappedArrayFunction;
using glimmer::interceptor::RuntimeGraphicsResourceGetMappedPointerFunction;
using glimmer::interceptor::RuntimeGraphicsResourceSetMapFlagsFunction;
using glimmer::interceptor::RuntimeGraphicsSubResourceGetMappedArrayFunction;
using glimmer::interceptor::RuntimeGraphicsUnmapResourcesFunction;
using glimmer::interceptor::RuntimeGraphicsUnregisterResourceFunction;
using glimmer::interceptor::RuntimeGraphLaunchFunction;
using glimmer::interceptor::RuntimeImportExternalMemoryFunction;
using glimmer::interceptor::RuntimeIpcCloseMemHandleFunction;
using glimmer::interceptor::RuntimeIpcGetMemHandleFunction;
using glimmer::interceptor::RuntimeIpcOpenMemHandleFunction;
using glimmer::interceptor::RuntimeMalloc3DArrayFunction;
using glimmer::interceptor::RuntimeMalloc3DFunction;
using glimmer::interceptor::RuntimeMallocArrayFunction;
using glimmer::interceptor::RuntimeMallocAsyncFunction;
using glimmer::interceptor::RuntimeMallocFromPoolAsyncFunction;
using glimmer::interceptor::RuntimeMallocFunction;
using glimmer::interceptor::RuntimeMallocManagedFunction;
using glimmer::interceptor::RuntimeMallocMipmappedArrayFunction;
using glimmer::interceptor::RuntimeMallocPitchFunction;
using RuntimeLaunchKernelFunction = cudaError_t (*)(const void* function, dim3 grid_dim,
                                                    dim3 block_dim, void** arguments,
                                                    std::size_t shared_memory_bytes,
                                                    cudaStream_t stream);
using RuntimeInternalLaunchKernelFunction = cudaError_t (*)(cudaKernel_t kernel, dim3 grid_dim,
                                                            dim3 block_dim, void** arguments,
                                                            std::size_t shared_memory_bytes,
                                                            cudaStream_t stream);
using RuntimeLaunchKernelExFunction = cudaError_t (*)(const cudaLaunchConfig_t* config,
                                                      const void* function, void** arguments);
using glimmer::interceptor::RuntimeMemGetDefaultMemPoolFunction;
using glimmer::interceptor::RuntimeMemGetInfoFunction;
using glimmer::interceptor::RuntimeMemGetMemPoolFunction;
using glimmer::interceptor::RuntimeMemPoolCreateFunction;
using glimmer::interceptor::RuntimeMemPoolDestroyFunction;
using glimmer::interceptor::RuntimeMemPoolExportPointerFunction;
using glimmer::interceptor::RuntimeMemPoolExportToShareableHandleFunction;
using glimmer::interceptor::RuntimeMemPoolGetAccessFunction;
using glimmer::interceptor::RuntimeMemPoolGetAttributeFunction;
using glimmer::interceptor::RuntimeMemPoolImportFromShareableHandleFunction;
using glimmer::interceptor::RuntimeMemPoolImportPointerFunction;
using glimmer::interceptor::RuntimeMemPoolSetAccessFunction;
using glimmer::interceptor::RuntimeMemPoolSetAttributeFunction;
using glimmer::interceptor::RuntimeMemPoolTrimToFunction;
using glimmer::interceptor::RuntimeMemSetMemPoolFunction;
using glimmer::interceptor::RuntimeStreamDestroyFunction;
using glimmer::interceptor::RuntimeStreamQueryFunction;
using glimmer::interceptor::RuntimeStreamSynchronizeFunction;

[[nodiscard]] std::uint64_t elapsed_nanoseconds(
    std::chrono::steady_clock::time_point started,
    std::chrono::steady_clock::time_point finished) noexcept {
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(finished - started);
    return elapsed.count() < 0 ? std::uint64_t{0} : static_cast<std::uint64_t>(elapsed.count());
}

template <typename Invoke>
[[nodiscard]] cudaError_t intercept_runtime_kernel_launch(const char* api_name, bool is_reentrant,
                                                          dim3 grid_dim, dim3 block_dim,
                                                          std::size_t shared_memory_bytes,
                                                          cudaStream_t stream, Invoke&& invoke,
                                                          bool per_thread_default_stream = false) {
    if (is_reentrant) {
        return invoke();
    }

    // The Driver preview does not yet own Runtime parameters, handles or
    // last-error state. Never silently run an unadmitted Runtime kernel.
    if (glimmer::interceptor::cooperative_driver_preview_requested()) {
        const auto name = std::string_view(api_name);
        if (!glimmer::interceptor::cooperative_runtime_preview_requested() ||
            (name != "cudaLaunchKernel" && name != "cudaLaunchKernel_ptsz")) {
            return cudaErrorNotSupported;
        }
        return invoke();
    }

    const bool scheduling_enforced = glimmer::interceptor::launch_scheduling_is_enforced();
    const bool trace_launch_timings = glimmer::interceptor::launch_timing_is_enabled();
    glimmer::control::LaunchGateTiming gate_timing;
    std::optional<glimmer::core::TaskId> launch_task;
    if (scheduling_enforced) {
        const auto capturing = glimmer::interceptor::runtime_launch_stream_is_capturing(
            stream, per_thread_default_stream);
        if (!capturing.has_value()) {
            return cudaErrorNotSupported;
        }
        if (capturing.value()) {
            return invoke();
        }
        launch_task = glimmer::interceptor::acquire_launch_slot(trace_launch_timings ? &gate_timing
                                                                                     : nullptr);
        if (!launch_task.has_value()) {
            return cudaErrorNotSupported;
        }
    }

    const auto cuda_launch_started = trace_launch_timings ? std::chrono::steady_clock::now()
                                                          : std::chrono::steady_clock::time_point{};
    const cudaError_t result = invoke();
    const auto cuda_launch_finished = trace_launch_timings
                                          ? std::chrono::steady_clock::now()
                                          : std::chrono::steady_clock::time_point{};
    if (launch_task.has_value() && result != cudaSuccess) {
        glimmer::interceptor::fail_launch_slot(launch_task.value());
    }

    bool event_tracking_succeeded = false;
    bool batch_call_succeeded = true;
    std::uint64_t event_tracking_nanoseconds = 0;
    if (launch_task.has_value() && result == cudaSuccess) {
        const auto event_tracking_started = trace_launch_timings
                                                ? std::chrono::steady_clock::now()
                                                : std::chrono::steady_clock::time_point{};
        event_tracking_succeeded = glimmer::interceptor::track_launch_completion(
            launch_task.value(), per_thread_default_stream && stream == nullptr
                                     ? CU_STREAM_PER_THREAD
                                     : reinterpret_cast<CUstream>(stream));
        if (trace_launch_timings) {
            event_tracking_nanoseconds =
                elapsed_nanoseconds(event_tracking_started, std::chrono::steady_clock::now());
        }
        if (event_tracking_succeeded) {
            batch_call_succeeded =
                glimmer::interceptor::finish_launch_batch_call(launch_task.value());
        }
    }
    if (trace_launch_timings) {
        glimmer::interceptor::report_launch_timing_observed(
            launch_task.value_or(0), gate_timing,
            elapsed_nanoseconds(cuda_launch_started, cuda_launch_finished),
            event_tracking_nanoseconds, static_cast<std::uint32_t>(result),
            event_tracking_succeeded);
    }
    if (launch_task.has_value() && result == cudaSuccess &&
        (!event_tracking_succeeded || !batch_call_succeeded)) {
        return cudaErrorUnknown;
    }
    if (result == cudaSuccess) {
        glimmer::interceptor::report_kernel_launch_observed({
            .api_name = api_name,
            .grid_dim_x = grid_dim.x,
            .grid_dim_y = grid_dim.y,
            .grid_dim_z = grid_dim.z,
            .block_dim_x = block_dim.x,
            .block_dim_y = block_dim.y,
            .block_dim_z = block_dim.z,
            .shared_memory_bytes = shared_memory_bytes,
            .stream = reinterpret_cast<const void*>(stream),
        });
    }
    return result;
}

[[nodiscard]] RuntimeMallocFunction resolve_runtime_malloc() noexcept {
    static RuntimeMallocFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr
                   ? nullptr
                   : reinterpret_cast<RuntimeMallocFunction>(real_dlsym(RTLD_NEXT, "cudaMalloc"));
    }();
    return function;
}

[[nodiscard]] RuntimeMallocManagedFunction resolve_runtime_malloc_managed() noexcept {
    static RuntimeMallocManagedFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeMallocManagedFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaMallocManaged"));
    }();
    return function;
}

[[nodiscard]] RuntimeMallocPitchFunction resolve_runtime_malloc_pitch() noexcept {
    static RuntimeMallocPitchFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeMallocPitchFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaMallocPitch"));
    }();
    return function;
}

[[nodiscard]] RuntimeMalloc3DFunction resolve_runtime_malloc_3d() noexcept {
    static RuntimeMalloc3DFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeMalloc3DFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaMalloc3D"));
    }();
    return function;
}

[[nodiscard]] RuntimeFreeFunction resolve_runtime_free() noexcept {
    static RuntimeFreeFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr
                   ? nullptr
                   : reinterpret_cast<RuntimeFreeFunction>(real_dlsym(RTLD_NEXT, "cudaFree"));
    }();
    return function;
}

[[nodiscard]] RuntimeIpcGetMemHandleFunction resolve_runtime_ipc_get_mem_handle() noexcept {
    static RuntimeIpcGetMemHandleFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeIpcGetMemHandleFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaIpcGetMemHandle"));
    }();
    return function;
}

[[nodiscard]] RuntimeIpcOpenMemHandleFunction resolve_runtime_ipc_open_mem_handle() noexcept {
    static RuntimeIpcOpenMemHandleFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeIpcOpenMemHandleFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaIpcOpenMemHandle"));
    }();
    return function;
}

[[nodiscard]] RuntimeIpcCloseMemHandleFunction resolve_runtime_ipc_close_mem_handle() noexcept {
    static RuntimeIpcCloseMemHandleFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeIpcCloseMemHandleFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaIpcCloseMemHandle"));
    }();
    return function;
}

[[nodiscard]] RuntimeImportExternalMemoryFunction
resolve_runtime_import_external_memory() noexcept {
    static RuntimeImportExternalMemoryFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeImportExternalMemoryFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaImportExternalMemory"));
    }();
    return function;
}

[[nodiscard]] RuntimeExternalMemoryGetMappedBufferFunction
resolve_runtime_external_memory_get_mapped_buffer() noexcept {
    static RuntimeExternalMemoryGetMappedBufferFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr
                   ? nullptr
                   : reinterpret_cast<RuntimeExternalMemoryGetMappedBufferFunction>(
                         real_dlsym(RTLD_NEXT, "cudaExternalMemoryGetMappedBuffer"));
    }();
    return function;
}

[[nodiscard]] RuntimeExternalMemoryGetMappedMipmappedArrayFunction
resolve_runtime_external_memory_get_mapped_mipmapped_array() noexcept {
    static RuntimeExternalMemoryGetMappedMipmappedArrayFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr
                   ? nullptr
                   : reinterpret_cast<RuntimeExternalMemoryGetMappedMipmappedArrayFunction>(
                         real_dlsym(RTLD_NEXT, "cudaExternalMemoryGetMappedMipmappedArray"));
    }();
    return function;
}

[[nodiscard]] RuntimeDestroyExternalMemoryFunction
resolve_runtime_destroy_external_memory() noexcept {
    static RuntimeDestroyExternalMemoryFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeDestroyExternalMemoryFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaDestroyExternalMemory"));
    }();
    return function;
}

[[nodiscard]] RuntimeMallocArrayFunction resolve_runtime_malloc_array() noexcept {
    static RuntimeMallocArrayFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeMallocArrayFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaMallocArray"));
    }();
    return function;
}

[[nodiscard]] RuntimeMalloc3DArrayFunction resolve_runtime_malloc_3d_array() noexcept {
    static RuntimeMalloc3DArrayFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeMalloc3DArrayFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaMalloc3DArray"));
    }();
    return function;
}

[[nodiscard]] RuntimeMallocMipmappedArrayFunction
resolve_runtime_malloc_mipmapped_array() noexcept {
    static RuntimeMallocMipmappedArrayFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeMallocMipmappedArrayFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaMallocMipmappedArray"));
    }();
    return function;
}

[[nodiscard]] RuntimeFreeArrayFunction resolve_runtime_free_array() noexcept {
    static RuntimeFreeArrayFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeFreeArrayFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaFreeArray"));
    }();
    return function;
}

[[nodiscard]] RuntimeFreeMipmappedArrayFunction resolve_runtime_free_mipmapped_array() noexcept {
    static RuntimeFreeMipmappedArrayFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeFreeMipmappedArrayFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaFreeMipmappedArray"));
    }();
    return function;
}

[[nodiscard]] RuntimeGraphicsUnregisterResourceFunction
resolve_runtime_graphics_unregister_resource() noexcept {
    static RuntimeGraphicsUnregisterResourceFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeGraphicsUnregisterResourceFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaGraphicsUnregisterResource"));
    }();
    return function;
}

[[nodiscard]] RuntimeGraphicsResourceSetMapFlagsFunction
resolve_runtime_graphics_resource_set_map_flags() noexcept {
    static RuntimeGraphicsResourceSetMapFlagsFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr
                   ? nullptr
                   : reinterpret_cast<RuntimeGraphicsResourceSetMapFlagsFunction>(
                         real_dlsym(RTLD_NEXT, "cudaGraphicsResourceSetMapFlags"));
    }();
    return function;
}

[[nodiscard]] RuntimeGraphicsMapResourcesFunction
resolve_runtime_graphics_map_resources() noexcept {
    static RuntimeGraphicsMapResourcesFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeGraphicsMapResourcesFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaGraphicsMapResources"));
    }();
    return function;
}

[[nodiscard]] RuntimeGraphicsUnmapResourcesFunction
resolve_runtime_graphics_unmap_resources() noexcept {
    static RuntimeGraphicsUnmapResourcesFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeGraphicsUnmapResourcesFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaGraphicsUnmapResources"));
    }();
    return function;
}

[[nodiscard]] RuntimeGraphicsResourceGetMappedPointerFunction
resolve_runtime_graphics_resource_get_mapped_pointer() noexcept {
    static RuntimeGraphicsResourceGetMappedPointerFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr
                   ? nullptr
                   : reinterpret_cast<RuntimeGraphicsResourceGetMappedPointerFunction>(
                         real_dlsym(RTLD_NEXT, "cudaGraphicsResourceGetMappedPointer"));
    }();
    return function;
}

[[nodiscard]] RuntimeGraphicsSubResourceGetMappedArrayFunction
resolve_runtime_graphics_subresource_get_mapped_array() noexcept {
    static RuntimeGraphicsSubResourceGetMappedArrayFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr
                   ? nullptr
                   : reinterpret_cast<RuntimeGraphicsSubResourceGetMappedArrayFunction>(
                         real_dlsym(RTLD_NEXT, "cudaGraphicsSubResourceGetMappedArray"));
    }();
    return function;
}

[[nodiscard]] RuntimeGraphicsResourceGetMappedMipmappedArrayFunction
resolve_runtime_graphics_resource_get_mapped_mipmapped_array() noexcept {
    static RuntimeGraphicsResourceGetMappedMipmappedArrayFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr
                   ? nullptr
                   : reinterpret_cast<RuntimeGraphicsResourceGetMappedMipmappedArrayFunction>(
                         real_dlsym(RTLD_NEXT, "cudaGraphicsResourceGetMappedMipmappedArray"));
    }();
    return function;
}

[[nodiscard]] RuntimeGraphAddMemAllocNodeFunction
resolve_runtime_graph_add_mem_alloc_node() noexcept {
    static RuntimeGraphAddMemAllocNodeFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeGraphAddMemAllocNodeFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaGraphAddMemAllocNode"));
    }();
    return function;
}

[[nodiscard]] RuntimeGraphLaunchFunction resolve_runtime_graph_launch() noexcept {
    static RuntimeGraphLaunchFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeGraphLaunchFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaGraphLaunch"));
    }();
    return function;
}

[[nodiscard]] RuntimeGraphLaunchFunction resolve_runtime_graph_launch_ptsz() noexcept {
    static RuntimeGraphLaunchFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeGraphLaunchFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaGraphLaunch_ptsz"));
    }();
    return function;
}

[[nodiscard]] RuntimeGetDeviceFunction resolve_runtime_get_device() noexcept {
    static RuntimeGetDeviceFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeGetDeviceFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaGetDevice"));
    }();
    return function;
}

[[nodiscard]] RuntimeMemGetInfoFunction resolve_runtime_mem_get_info() noexcept {
    static RuntimeMemGetInfoFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeMemGetInfoFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaMemGetInfo"));
    }();
    return function;
}

[[nodiscard]] RuntimeLaunchKernelFunction resolve_runtime_launch_kernel() noexcept {
    static RuntimeLaunchKernelFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeLaunchKernelFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaLaunchKernel"));
    }();
    return function;
}

[[nodiscard]] RuntimeLaunchKernelExFunction resolve_runtime_launch_kernel_ex(
    bool per_thread_default_stream) noexcept {
    static const auto functions = [] {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return std::array<RuntimeLaunchKernelExFunction, 2>{
            real_dlsym == nullptr ? nullptr
                                  : reinterpret_cast<RuntimeLaunchKernelExFunction>(
                                        real_dlsym(RTLD_NEXT, "cudaLaunchKernelExC")),
            real_dlsym == nullptr ? nullptr
                                  : reinterpret_cast<RuntimeLaunchKernelExFunction>(
                                        real_dlsym(RTLD_NEXT, "cudaLaunchKernelExC_ptsz"))};
    }();
    return functions[per_thread_default_stream ? 1 : 0];
}

[[nodiscard]] cudaError_t invoke_runtime_source_kernel(
    const glimmer::interceptor::RuntimeSourceKernelShape& shape,
    const std::function<cudaError_t()>& original) {
    if (!glimmer::interceptor::cooperative_source_router()) {
        return cudaErrorUnknown;
    }
    return glimmer::interceptor::invoke_runtime_source_kernel_original(shape, original, [] {
        const auto status =
            glimmer::interceptor::quarantine_cooperative_source_call(CUDA_ERROR_UNKNOWN);
        if (status == CUDA_SUCCESS || status == CUDA_ERROR_INVALID_VALUE ||
            status == CUDA_ERROR_NOT_SUPPORTED) {
            glimmer::interceptor::report_diagnostic(
                "[glimmer] source kernel uncertainty was not retained\n");
        }
    });
}

[[nodiscard]] cudaError_t invoke_runtime_source_memory(
    const std::function<cudaError_t()>& original) {
    if (!glimmer::interceptor::cooperative_runtime_preview_requested() ||
        !glimmer::interceptor::cooperative_process_call_gate().is_native_only_on_current_thread() ||
        glimmer::interceptor::is_inside_driver_call()) {
        return original();
    }
    if (g_runtime_source_memory_issue != nullptr) {
        return cudaErrorUnknown;
    }
    bool has_native_issue{};
    const RuntimeSourceMemoryScope scope(has_native_issue);
    return glimmer::interceptor::intercept_cooperative_runtime_memory_boundary(original,
                                                                               has_native_issue);
}

[[nodiscard]] cudaError_t intercept_runtime_kernel_launch_ex(const cudaLaunchConfig_t* config,
                                                             const void* function, void** arguments,
                                                             bool per_thread_default_stream) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call() ||
                              glimmer::interceptor::is_inside_driver_call();
    try {
        RuntimeCallScope scope(true);
        if (config == nullptr) {
            return cudaErrorInvalidValue;
        }
        const auto real_launch = resolve_runtime_launch_kernel_ex(per_thread_default_stream);
        if (real_launch == nullptr) {
            return cudaErrorNotSupported;
        }
        if (!is_reentrant && glimmer::interceptor::cooperative_runtime_preview_requested() &&
            glimmer::interceptor::cooperative_process_call_gate()
                .is_native_only_on_current_thread()) {
            return invoke_runtime_source_kernel(
                {{config->gridDim.x, config->gridDim.y, config->gridDim.z},
                 {config->blockDim.x, config->blockDim.y, config->blockDim.z},
                 config->dynamicSmemBytes,
                 per_thread_default_stream},
                [&] { return real_launch(config, function, arguments); });
        }
        return intercept_runtime_kernel_launch(
            per_thread_default_stream ? "cudaLaunchKernelExC_ptsz" : "cudaLaunchKernelExC",
            is_reentrant, config->gridDim, config->blockDim, config->dynamicSmemBytes,
            config->stream, [&] { return real_launch(config, function, arguments); },
            per_thread_default_stream);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

[[nodiscard]] RuntimeLaunchKernelFunction resolve_runtime_launch_kernel_ptsz() noexcept {
    static RuntimeLaunchKernelFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeLaunchKernelFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaLaunchKernel_ptsz"));
    }();
    return function;
}

[[nodiscard]] RuntimeInternalLaunchKernelFunction
resolve_runtime_internal_launch_kernel() noexcept {
    static RuntimeInternalLaunchKernelFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeInternalLaunchKernelFunction>(
                                           real_dlsym(RTLD_NEXT, "__cudaLaunchKernel"));
    }();
    return function;
}

[[nodiscard]] RuntimeInternalLaunchKernelFunction
resolve_runtime_internal_launch_kernel_ptsz() noexcept {
    static RuntimeInternalLaunchKernelFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeInternalLaunchKernelFunction>(
                                           real_dlsym(RTLD_NEXT, "__cudaLaunchKernel_ptsz"));
    }();
    return function;
}

[[nodiscard]] RuntimeMallocAsyncFunction resolve_runtime_malloc_async() noexcept {
    static RuntimeMallocAsyncFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeMallocAsyncFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaMallocAsync"));
    }();
    return function;
}

[[nodiscard]] RuntimeMallocAsyncFunction resolve_runtime_malloc_async_ptsz() noexcept {
    static RuntimeMallocAsyncFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeMallocAsyncFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaMallocAsync_ptsz"));
    }();
    return function;
}

[[nodiscard]] RuntimeMallocFromPoolAsyncFunction resolve_runtime_malloc_from_pool_async() noexcept {
    static RuntimeMallocFromPoolAsyncFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeMallocFromPoolAsyncFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaMallocFromPoolAsync"));
    }();
    return function;
}

[[nodiscard]] RuntimeMallocFromPoolAsyncFunction
resolve_runtime_malloc_from_pool_async_ptsz() noexcept {
    static RuntimeMallocFromPoolAsyncFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeMallocFromPoolAsyncFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaMallocFromPoolAsync_ptsz"));
    }();
    return function;
}

[[nodiscard]] RuntimeFreeAsyncFunction resolve_runtime_free_async() noexcept {
    static RuntimeFreeAsyncFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeFreeAsyncFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaFreeAsync"));
    }();
    return function;
}

[[nodiscard]] RuntimeFreeAsyncFunction resolve_runtime_free_async_ptsz() noexcept {
    static RuntimeFreeAsyncFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeFreeAsyncFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaFreeAsync_ptsz"));
    }();
    return function;
}

[[nodiscard]] RuntimeDeviceSynchronizeFunction resolve_runtime_device_synchronize() noexcept {
    static RuntimeDeviceSynchronizeFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeDeviceSynchronizeFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaDeviceSynchronize"));
    }();
    return function;
}

[[nodiscard]] RuntimeStreamSynchronizeFunction resolve_runtime_stream_synchronize() noexcept {
    static RuntimeStreamSynchronizeFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeStreamSynchronizeFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaStreamSynchronize"));
    }();
    return function;
}

[[nodiscard]] RuntimeStreamSynchronizeFunction resolve_runtime_stream_synchronize_ptsz() noexcept {
    static RuntimeStreamSynchronizeFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeStreamSynchronizeFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaStreamSynchronize_ptsz"));
    }();
    return function;
}

[[nodiscard]] RuntimeStreamQueryFunction resolve_runtime_stream_query() noexcept {
    static RuntimeStreamQueryFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeStreamQueryFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaStreamQuery"));
    }();
    return function;
}

[[nodiscard]] RuntimeStreamQueryFunction resolve_runtime_stream_query_ptsz() noexcept {
    static RuntimeStreamQueryFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeStreamQueryFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaStreamQuery_ptsz"));
    }();
    return function;
}

[[nodiscard]] RuntimeStreamDestroyFunction resolve_runtime_stream_destroy() noexcept {
    static RuntimeStreamDestroyFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeStreamDestroyFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaStreamDestroy"));
    }();
    return function;
}

[[nodiscard]] RuntimeDeviceGetDefaultMemPoolFunction
resolve_runtime_device_get_default_mem_pool() noexcept {
    static RuntimeDeviceGetDefaultMemPoolFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeDeviceGetDefaultMemPoolFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaDeviceGetDefaultMemPool"));
    }();
    return function;
}

[[nodiscard]] RuntimeDeviceSetMemPoolFunction resolve_runtime_device_set_mem_pool() noexcept {
    static RuntimeDeviceSetMemPoolFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeDeviceSetMemPoolFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaDeviceSetMemPool"));
    }();
    return function;
}

[[nodiscard]] RuntimeDeviceGetMemPoolFunction resolve_runtime_device_get_mem_pool() noexcept {
    static RuntimeDeviceGetMemPoolFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeDeviceGetMemPoolFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaDeviceGetMemPool"));
    }();
    return function;
}

[[nodiscard]] RuntimeMemPoolTrimToFunction resolve_runtime_mem_pool_trim_to() noexcept {
    static RuntimeMemPoolTrimToFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeMemPoolTrimToFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaMemPoolTrimTo"));
    }();
    return function;
}

[[nodiscard]] RuntimeMemPoolSetAttributeFunction resolve_runtime_mem_pool_set_attribute() noexcept {
    static RuntimeMemPoolSetAttributeFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeMemPoolSetAttributeFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaMemPoolSetAttribute"));
    }();
    return function;
}

[[nodiscard]] RuntimeMemPoolGetAttributeFunction resolve_runtime_mem_pool_get_attribute() noexcept {
    static RuntimeMemPoolGetAttributeFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeMemPoolGetAttributeFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaMemPoolGetAttribute"));
    }();
    return function;
}

[[nodiscard]] RuntimeMemPoolSetAccessFunction resolve_runtime_mem_pool_set_access() noexcept {
    static RuntimeMemPoolSetAccessFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeMemPoolSetAccessFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaMemPoolSetAccess"));
    }();
    return function;
}

[[nodiscard]] RuntimeMemPoolGetAccessFunction resolve_runtime_mem_pool_get_access() noexcept {
    static RuntimeMemPoolGetAccessFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeMemPoolGetAccessFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaMemPoolGetAccess"));
    }();
    return function;
}

[[nodiscard]] RuntimeMemPoolCreateFunction resolve_runtime_mem_pool_create() noexcept {
    static RuntimeMemPoolCreateFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeMemPoolCreateFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaMemPoolCreate"));
    }();
    return function;
}

[[nodiscard]] RuntimeMemPoolDestroyFunction resolve_runtime_mem_pool_destroy() noexcept {
    static RuntimeMemPoolDestroyFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeMemPoolDestroyFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaMemPoolDestroy"));
    }();
    return function;
}

[[nodiscard]] RuntimeMemGetDefaultMemPoolFunction
resolve_runtime_mem_get_default_mem_pool() noexcept {
    static RuntimeMemGetDefaultMemPoolFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeMemGetDefaultMemPoolFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaMemGetDefaultMemPool"));
    }();
    return function;
}

[[nodiscard]] RuntimeMemGetMemPoolFunction resolve_runtime_mem_get_mem_pool() noexcept {
    static RuntimeMemGetMemPoolFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeMemGetMemPoolFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaMemGetMemPool"));
    }();
    return function;
}

[[nodiscard]] RuntimeMemSetMemPoolFunction resolve_runtime_mem_set_mem_pool() noexcept {
    static RuntimeMemSetMemPoolFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeMemSetMemPoolFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaMemSetMemPool"));
    }();
    return function;
}

[[nodiscard]] RuntimeMemPoolExportToShareableHandleFunction
resolve_runtime_mem_pool_export_to_shareable_handle() noexcept {
    static RuntimeMemPoolExportToShareableHandleFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr
                   ? nullptr
                   : reinterpret_cast<RuntimeMemPoolExportToShareableHandleFunction>(
                         real_dlsym(RTLD_NEXT, "cudaMemPoolExportToShareableHandle"));
    }();
    return function;
}

[[nodiscard]] RuntimeMemPoolImportFromShareableHandleFunction
resolve_runtime_mem_pool_import_from_shareable_handle() noexcept {
    static RuntimeMemPoolImportFromShareableHandleFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr
                   ? nullptr
                   : reinterpret_cast<RuntimeMemPoolImportFromShareableHandleFunction>(
                         real_dlsym(RTLD_NEXT, "cudaMemPoolImportFromShareableHandle"));
    }();
    return function;
}

[[nodiscard]] RuntimeMemPoolExportPointerFunction
resolve_runtime_mem_pool_export_pointer() noexcept {
    static RuntimeMemPoolExportPointerFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeMemPoolExportPointerFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaMemPoolExportPointer"));
    }();
    return function;
}

[[nodiscard]] RuntimeMemPoolImportPointerFunction
resolve_runtime_mem_pool_import_pointer() noexcept {
    static RuntimeMemPoolImportPointerFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeMemPoolImportPointerFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaMemPoolImportPointer"));
    }();
    return function;
}

[[nodiscard]] glimmer::interceptor::RuntimeFuncGetNameFunction
resolve_runtime_func_get_name() noexcept {
    static const auto function = []() noexcept {
        const auto real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr
                   ? nullptr
                   : reinterpret_cast<glimmer::interceptor::RuntimeFuncGetNameFunction>(
                         real_dlsym(RTLD_NEXT, "cudaFuncGetName"));
    }();
    return function;
}

[[nodiscard]] glimmer::interceptor::RuntimeDeviceResetFunction
resolve_runtime_device_reset() noexcept {
    static const auto function = []() noexcept {
        const auto real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr
                   ? nullptr
                   : reinterpret_cast<glimmer::interceptor::RuntimeDeviceResetFunction>(
                         real_dlsym(RTLD_NEXT, "cudaDeviceReset"));
    }();
    return function;
}

cudaError_t invoke_runtime_persistent_or_native(const void* function, dim3 grid, dim3 block,
                                                void** arguments, std::size_t shared_memory_bytes,
                                                cudaStream_t stream,
                                                RuntimeLaunchKernelFunction native,
                                                bool is_reentrant, bool per_thread_default_stream) {
    if (!is_reentrant && glimmer::interceptor::cooperative_runtime_preview_requested() &&
        glimmer::interceptor::cooperative_process_call_gate().is_native_only_on_current_thread()) {
        return invoke_runtime_source_kernel(
            {{grid.x, grid.y, grid.z},
             {block.x, block.y, block.z},
             shared_memory_bytes,
             per_thread_default_stream},
            [&] { return native(function, grid, block, arguments, shared_memory_bytes, stream); });
    }
    static const auto get_function = []() noexcept {
        const auto real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr
                   ? nullptr
                   : reinterpret_cast<glimmer::interceptor::RuntimeGetFuncBySymbolFunction>(
                         real_dlsym(RTLD_NEXT, "cudaGetFuncBySymbol"));
    }();
    if (!is_reentrant && glimmer::interceptor::cooperative_runtime_preview_requested()) {
        const auto capturing = glimmer::interceptor::runtime_launch_stream_is_capturing(
            stream, per_thread_default_stream);
        if (!capturing || *capturing) {
            return cudaErrorNotSupported;
        }
        const auto profiling = glimmer::interceptor::runtime_profiling_boundary_mode();
        if (profiling == glimmer::interceptor::RuntimeProfilingBoundaryMode::kInvalid) {
            return cudaErrorInvalidValue;
        }
        if (profiling == glimmer::interceptor::RuntimeProfilingBoundaryMode::kEnabled) {
            // Select before any original issue. Tool injection may hide the SDK
            // leaf; never retry it or fabricate lexical permission consumption.
            return glimmer::interceptor::intercept_cooperative_runtime_native_boundary([&] {
                return native(function, grid, block, arguments, shared_memory_bytes, stream);
            });
        }
        std::optional<glimmer::interceptor::RuntimeCooperativeLaunch> plan;
        {
            const RuntimeNativeDriverScope native_scope;
            plan = glimmer::interceptor::prepare_runtime_cooperative_launch(
                function, grid, block, shared_memory_bytes, resolve_runtime_func_get_name(),
                get_function);
        }
        if (!plan) {
            // Preserve real Runtime validation on an ordered, original caller
            // boundary. No cooperative/native launch has been attempted yet.
            return glimmer::interceptor::intercept_cooperative_runtime_native_boundary([&] {
                return native(function, grid, block, arguments, shared_memory_bytes, stream);
            });
        }
        const auto context = plan->context;
        return glimmer::interceptor::invoke_runtime_cooperative_launch_original(
            std::move(*plan), per_thread_default_stream,
            [&] { return native(function, grid, block, arguments, shared_memory_bytes, stream); },
            [context] { glimmer::interceptor::quarantine_runtime_cooperative_launch(context); });
    }
    std::optional<glimmer::interceptor::RuntimePersistentLaunch> variant;
    std::vector<void*> private_arguments;
    if (!is_reentrant && get_function != nullptr) {
        try {
            variant = glimmer::interceptor::prepare_runtime_persistent_launch(
                function, grid, block, shared_memory_bytes, stream, per_thread_default_stream,
                resolve_runtime_func_get_name(), get_function);
            if (variant.has_value() && (arguments != nullptr || variant->parameter_count == 0)) {
                private_arguments.reserve(variant->parameter_count + 3);
                for (std::size_t index = 0; index < variant->parameter_count; ++index) {
                    private_arguments.push_back(arguments[index]);
                }
                private_arguments.push_back(&grid.x);
                if (variant->execution_kind ==
                    glimmer::interceptor::PtxExecutionKind::kBlockStatic) {
                    private_arguments.push_back(&grid.y);
                    private_arguments.push_back(&grid.z);
                }
            } else {
                variant.reset();
            }
        } catch (...) {
            variant.reset();
        }
    }
    if (!variant.has_value()) {
        return native(function, grid, block, arguments, shared_memory_bytes, stream);
    }
    // Preserve Runtime validation/error state by replacing only its matched Driver launch.
    PendingRuntimeReplacement pending{*variant, private_arguments.data(), false};
    RuntimeReplacementScope replacement_scope(pending);
    const auto result = native(function, grid, block, arguments, shared_memory_bytes, stream);
    if (result == cudaSuccess && pending.was_consumed) {
        glimmer::interceptor::report_runtime_persistent_launch(variant->logical_blocks,
                                                               variant->worker_blocks);
    }
    return result;
}

}  // namespace

namespace glimmer::interceptor {

void mark_runtime_source_memory_issue() noexcept {
    if (g_runtime_source_memory_issue != nullptr) {
        *g_runtime_source_memory_issue = true;
    }
}

cudaError_t invoke_runtime_source_kernel_original(
    const RuntimeSourceKernelShape& shape, const std::function<cudaError_t()>& original,
    const std::function<void()>& quarantine) noexcept {
    auto& gate = cooperative_process_call_gate();
    if (!original || !quarantine || g_pending_runtime_source_kernel || is_inside_driver_call() ||
        !gate.is_entered_by_current_thread() || !gate.is_native_only_on_current_thread()) {
        return cudaErrorUnknown;
    }
    cudaError_t native_status{cudaErrorUnknown};
    bool has_uncertainty = true;
    try {
        RuntimeCallScope runtime(true);
        if (!runtime_cooperative_driver_routing_is_enabled()) {
            return cudaErrorUnknown;
        }
        PendingRuntimeSourceKernel pending{shape};
        const RuntimeSourceKernelScope permission(pending);
        native_status = original();
        has_uncertainty = !pending.has_dispatched || pending.has_rejected ||
                          (native_status == cudaSuccess && pending.driver_status != CUDA_SUCCESS);
    } catch (...) {
        has_uncertainty = true;
    }
    if (has_uncertainty) {
        try {
            quarantine();
        } catch (...) {
            return cudaErrorUnknown;
        }
        return native_status == cudaSuccess ? cudaErrorUnknown : native_status;
    }
    return native_status;
}

std::optional<CUresult> dispatch_runtime_source_kernel(
    CUfunction function, const RuntimeSourceKernelShape& shape,
    const std::function<CUresult()>& execute) noexcept {
    if (!g_pending_runtime_source_kernel || !is_inside_runtime_call() || is_inside_driver_call()) {
        return std::nullopt;
    }
    auto& pending = *g_pending_runtime_source_kernel;
    if (!function || !execute || pending.has_dispatched || pending.has_rejected ||
        pending.shape != shape) {
        pending.has_rejected = true;
        return CUDA_ERROR_INVALID_VALUE;
    }
    pending.has_dispatched = true;
    try {
        pending.driver_status = execute();
    } catch (...) {
        pending.has_rejected = true;
        pending.driver_status = CUDA_ERROR_UNKNOWN;
    }
    return pending.driver_status;
}

cudaError_t invoke_runtime_graph_original(const RuntimeGraphExecution& execution,
                                          const std::function<cudaError_t()>& original) noexcept {
    if (!execution || !original) {
        return cudaErrorNotSupported;
    }
    cudaError_t native_status{cudaErrorUnknown};
    bool was_invoked{};
    std::atomic<bool> has_rejected{};
    const auto caller = std::this_thread::get_id();
    try {
        const auto owned = execution([&] {
            if (std::this_thread::get_id() != caller || !is_inside_driver_call() || was_invoked) {
                has_rejected = true;
                return CUDA_ERROR_INVALID_VALUE;
            }
            was_invoked = true;
            try {
                const RuntimeCallScope runtime;
                native_status = original();
            } catch (...) {
                native_status = cudaErrorUnknown;
            }
            return native_status == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
        });
        if (was_invoked && native_status != cudaSuccess) {
            return native_status;
        }
        return owned == CUDA_SUCCESS && was_invoked && !has_rejected ? cudaSuccess
                                                                     : cudaErrorUnknown;
    } catch (...) {
        return was_invoked && native_status != cudaSuccess ? native_status : cudaErrorUnknown;
    }
}

std::optional<cudaError_t> invoke_runtime_context_completion_original(
    const RuntimeContextCompletionExecution& execution,
    const std::function<cudaError_t()>& original,
    const std::function<void()>& quarantine) noexcept {
    auto& gate = cooperative_process_call_gate();
    if (!execution || !original || !quarantine || is_inside_driver_call() ||
        g_pending_runtime_cooperative || g_pending_runtime_capture ||
        g_pending_runtime_source_kernel || g_pending_runtime_replacement ||
        !gate.is_entered_by_current_thread() || gate.is_native_only_on_current_thread())
        return cudaErrorUnknown;
    cudaError_t native_status{cudaErrorUnknown};
    bool was_invoked{}, has_returned{}, has_execution_return{};
    std::atomic<bool> has_rejected{};
    const auto caller = std::this_thread::get_id();
    std::optional<CUresult> owned;
    try {
        owned = execution([&] {
            if (std::this_thread::get_id() != caller || !is_inside_driver_call() ||
                !gate.is_native_only_on_current_thread() || was_invoked) {
                has_rejected = true;
                return CUDA_ERROR_INVALID_VALUE;
            }
            was_invoked = true;
            try {
                const RuntimeCallScope runtime;
                const RuntimeEventOriginalScope sdk;
                native_status = original();
                has_returned = true;
            } catch (...) {
                native_status = cudaErrorUnknown;
            }
            return has_returned && native_status == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
        });
        has_execution_return = true;
    } catch (...) {
        // Unknown owner effects are not permission to retry the application SDK.
    }
    const bool has_uncertainty =
        has_rejected || !has_execution_return || (was_invoked && !has_returned) ||
        (!owned && was_invoked) ||
        (owned && *owned == CUDA_SUCCESS && (!was_invoked || native_status != cudaSuccess)) ||
        (owned && *owned != CUDA_SUCCESS && was_invoked && native_status == cudaSuccess);
    if (has_uncertainty) {
        try {
            quarantine();
        } catch (...) {
            // Keep the original enum; a CPU quarantine exception cannot legalize replay.
        }
    }
    if (was_invoked && has_returned && native_status != cudaSuccess) return native_status;
    if (has_uncertainty) return cudaErrorUnknown;
    if (!owned) return std::nullopt;  // decline before any original effect only
    return *owned == CUDA_SUCCESS && was_invoked && has_returned ? cudaSuccess : cudaErrorUnknown;
}

CUresult invoke_runtime_capture_original(const RuntimeCaptureDispatch& dispatch,
                                         const std::function<cudaError_t()>& original,
                                         RuntimeCaptureObservation& observation,
                                         cudaError_t& native_status,
                                         const RuntimeCaptureContextLookup& lookup) noexcept {
    try {
        auto& gate = cooperative_process_call_gate();
        if (!dispatch || !original || g_pending_runtime_capture || is_inside_runtime_call() ||
            is_inside_driver_call() || !gate.is_entered_by_current_thread() ||
            !gate.is_native_only_on_current_thread()) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        observation = {};
        native_status = cudaErrorUnknown;
        PendingRuntimeCapture pending{dispatch, observation, lookup};
        RuntimeCallScope runtime;
        const RuntimeCaptureScope permission(pending);
        observation.was_runtime_invoked = true;
        native_status = original();
        return native_status == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
    } catch (const std::bad_alloc&) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}

std::optional<CUresult> dispatch_runtime_capture_origin(CUstream stream, bool is_ptds,
                                                        CudaSourceCaptureKind kind,
                                                        const std::function<CUresult()>& original,
                                                        CUgraph* native_graph) noexcept {
    if (!g_pending_runtime_capture || !is_inside_runtime_call() || is_inside_driver_call()) {
        return std::nullopt;
    }
    try {
        if (!original) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        using GetContext = CUresult(CUDAAPI*)(CUstream, CUcontext*);
        // At 12050 the base resolver name selects the green-context ABI with
        // three arguments. Request the exact two-argument stream-context ABI.
        const auto get_context = reinterpret_cast<GetContext>(
            resolve_native_driver_function("cuStreamGetCtx", is_ptds, 9020));
        CUcontext context{};
        CUresult queried{CUDA_ERROR_NOT_SUPPORTED};
        if (get_context) {
            const NativeDriverCallScope scope;
            queried = get_context(stream, &context);
        }
        // An invalidated capture can reject stream-context metadata before the
        // actual end call. Only its recorded typed origin may supply identity.
        if (queried == CUDA_ERROR_STREAM_CAPTURE_INVALIDATED &&
            kind == CudaSourceCaptureKind::kEnd && g_pending_runtime_capture->lookup) {
            context = nullptr;
            queried = g_pending_runtime_capture->lookup(stream, is_ptds, context);
        }
        if (queried != CUDA_SUCCESS || !context) {
            return queried == CUDA_SUCCESS ? CUDA_ERROR_INVALID_CONTEXT : queried;
        }
        auto& pending = *g_pending_runtime_capture;
        struct ActualCaptureCall {
            RuntimeCaptureObservation& observation;
            const std::function<CUresult()>& original;
            CUcontext context;
            CUstream stream;
            bool is_ptds;
            CUgraph* graph;
        } call{pending.observation, original, context, stream, is_ptds, native_graph};
        return pending.dispatch(context, stream, is_ptds, kind, [call = &call] {
            // The owner accepts this original at most once. Retain the first
            // actual leaf identity, not an opaque Runtime handle or later alias.
            auto& observed = call->observation;
            observed.context = call->context;
            observed.stream = call->stream;
            observed.is_ptds = call->is_ptds;
            ++observed.original_driver_calls;
            const auto result = call->original();
            if (result == CUDA_SUCCESS && call->graph) {
                observed.graph = *call->graph;
            }
            return result;
        });
    } catch (const std::bad_alloc&) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}

bool cooperative_runtime_preview_requested() noexcept {
    const char* value = std::getenv("GLIMMER_COOPERATIVE_RUNTIME_PREVIEW");
    return value != nullptr && std::strcmp(value, "1") == 0;
}

bool runtime_cooperative_driver_routing_is_enabled() noexcept {
    return g_is_inside_runtime_call && g_should_route_runtime_driver;
}

RuntimeProfilingBoundaryMode runtime_profiling_boundary_mode() noexcept {
    static const auto mode = [] {
        const auto* value = std::getenv("GLIMMER_PROFILE_RUNTIME_NATIVE_BOUNDARY");
        if (!value || std::strcmp(value, "0") == 0) return RuntimeProfilingBoundaryMode::kDisabled;
        if (std::strcmp(value, "1") != 0) return RuntimeProfilingBoundaryMode::kInvalid;
        report_diagnostic(
            "[glimmer] profiling Runtime launches use synchronous native boundaries; "
            "application timings are not performance evidence\n");
        return RuntimeProfilingBoundaryMode::kEnabled;
    }();
    return mode;
}

bool runtime_cooperative_launch_is_pending() noexcept {
    return g_pending_runtime_cooperative != nullptr;
}

std::optional<RuntimeCooperativeLaunch> consume_runtime_cooperative_launch(
    CUfunction function, const RuntimeSourceKernelShape& shape) noexcept {
    if (!g_pending_runtime_cooperative) return std::nullopt;
    auto& pending = *g_pending_runtime_cooperative;
    if (!function || pending.was_consumed || pending.has_rejected ||
        pending.plan.original != function || pending.plan.grid != shape.grid ||
        pending.plan.block != shape.block ||
        pending.plan.shared_memory_bytes != shape.shared_memory_bytes ||
        pending.is_ptds != shape.is_ptds) {
        pending.has_rejected = true;
        return std::nullopt;
    }
    pending.was_consumed = true;
    return std::move(pending.plan);
}

void complete_runtime_cooperative_launch(CUresult status) noexcept {
    if (!g_pending_runtime_cooperative) return;
    auto& pending = *g_pending_runtime_cooperative;
    if (!pending.was_consumed || pending.has_returned) {
        pending.has_rejected = true;
        return;
    }
    pending.has_returned = true;
    pending.driver_status = status;
}

bool runtime_native_kernel_execution_is_pending() noexcept {
    return g_pending_runtime_cooperative &&
           static_cast<bool>(g_pending_runtime_cooperative->native_execution);
}

std::optional<CUresult> dispatch_runtime_native_kernel(
    CUfunction function, CUstream stream, const RuntimeSourceKernelShape& shape,
    const std::function<CUresult()>& original) noexcept {
    if (!runtime_native_kernel_execution_is_pending() || !is_inside_runtime_call() ||
        is_inside_driver_call())
        return std::nullopt;
    auto& pending = *g_pending_runtime_cooperative;
    if (!original || !consume_runtime_cooperative_launch(function, shape)) {
        pending.has_rejected = true;
        return CUDA_ERROR_INVALID_VALUE;
    }
    CUresult status = CUDA_ERROR_UNKNOWN;
    const auto caller = std::this_thread::get_id();
    std::atomic<bool> was_invoked{};
    std::atomic<bool> has_rejected{};
    try {
        status = pending.native_execution(function, stream, shape.is_ptds, [&] {
            bool expected = false;
            if (std::this_thread::get_id() != caller || !is_inside_driver_call() ||
                !is_inside_runtime_call() || !was_invoked.compare_exchange_strong(expected, true)) {
                has_rejected = true;
                return CUDA_ERROR_INVALID_VALUE;
            }
            return original();
        });
    } catch (...) {
        pending.has_rejected = true;
    }
    if (has_rejected || (status == CUDA_SUCCESS && !was_invoked)) pending.has_rejected = true;
    complete_runtime_cooperative_launch(status);
    return status;
}

cudaError_t invoke_runtime_cooperative_launch_original(
    RuntimeCooperativeLaunch plan, bool is_ptds, const std::function<cudaError_t()>& original,
    const std::function<void()>& quarantine,
    const RuntimeNativeKernelExecution& native_execution) noexcept {
    auto& gate = cooperative_process_call_gate();
    if (!plan.original || !original || !quarantine || g_pending_runtime_cooperative ||
        g_pending_runtime_capture || g_pending_runtime_source_kernel ||
        g_pending_runtime_replacement || is_inside_driver_call() ||
        !gate.is_entered_by_current_thread() || gate.is_native_only_on_current_thread())
        return cudaErrorUnknown;
    if (native_execution && (plan.variant || plan.binary_variant || plan.gemm_variant))
        return cudaErrorUnknown;  // A native-only permission cannot discard a selected variant.
    cudaError_t native_status = cudaErrorUnknown;
    bool has_uncertainty = true;
    try {
        RuntimeCallScope runtime(true);
        if (!runtime_cooperative_driver_routing_is_enabled()) return cudaErrorUnknown;
        PendingRuntimeCooperativeLaunch pending{std::move(plan), is_ptds, native_execution};
        const RuntimeCooperativeLaunchScope permission(pending);
        native_status = original();
        // Qualify only zero-sized launch rejection before an observed leaf.
        // An arbitrary SDK error does not prove that no original was issued.
        const auto is_empty = [](const auto& dimensions) {
            return std::ranges::find(dimensions, 0U) != dimensions.end();
        };
        const bool is_validation_rejection =
            native_status == cudaErrorInvalidConfiguration && !pending.was_consumed &&
            (is_empty(pending.plan.grid) || is_empty(pending.plan.block));
        has_uncertainty = pending.has_rejected ||
                          (!pending.was_consumed && !is_validation_rejection) ||
                          (pending.was_consumed && !pending.has_returned) ||
                          (native_status == cudaSuccess &&
                           (!pending.was_consumed || pending.driver_status != CUDA_SUCCESS));
    } catch (...) {
        has_uncertainty = true;
    }
    if (has_uncertainty) {
        try {
            quarantine();
        } catch (...) {
            return native_status == cudaSuccess ? cudaErrorUnknown : native_status;
        }
        return native_status == cudaSuccess ? cudaErrorUnknown : native_status;
    }
    return native_status;
}

std::optional<bool> runtime_launch_stream_is_capturing(cudaStream_t stream,
                                                       bool per_thread_default_stream) noexcept {
    using CaptureQuery = cudaError_t (*)(cudaStream_t, cudaStreamCaptureStatus*);
    static const CaptureQuery legacy_query = [] {
        const auto resolver = resolve_real_dlsym();
        return resolver == nullptr
                   ? nullptr
                   : reinterpret_cast<CaptureQuery>(resolver(RTLD_NEXT, "cudaStreamIsCapturing"));
    }();
    static const CaptureQuery ptds_query = [] {
        const auto resolver = resolve_real_dlsym();
        return resolver == nullptr ? nullptr
                                   : reinterpret_cast<CaptureQuery>(
                                         resolver(RTLD_NEXT, "cudaStreamIsCapturing_ptsz"));
    }();
    const auto query = per_thread_default_stream ? ptds_query : legacy_query;
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    // Runtime owns lazy primary-context binding, including on autograd threads.
    // A Driver query before that binding can reject otherwise valid launches.
    if (query == nullptr || query(stream, &status) != cudaSuccess) {
        report_diagnostic("[glimmer] launch admission rejected: Runtime capture query failed\n");
        return std::nullopt;
    }
    switch (status) {
        case cudaStreamCaptureStatusNone:
            return false;
        case cudaStreamCaptureStatusActive:
        case cudaStreamCaptureStatusInvalidated:
            return true;
        default:
            return std::nullopt;
    }
}

std::optional<RuntimeDriverReplacement> consume_runtime_driver_replacement(
    CUfunction function) noexcept {
    if (g_pending_runtime_replacement == nullptr || g_pending_runtime_replacement->was_consumed ||
        g_pending_runtime_replacement->variant.original != function) {
        return std::nullopt;
    }
    auto& pending = *g_pending_runtime_replacement;
    pending.was_consumed = true;
    return RuntimeDriverReplacement{pending.variant.persistent, pending.variant.worker_blocks,
                                    pending.arguments};
}

bool is_inside_runtime_call() noexcept {
    return g_is_inside_runtime_call;
}

bool runtime_persistent_lookup_is_enabled() noexcept {
    if (cooperative_runtime_preview_requested()) {
        return true;
    }
    const auto* setting = std::getenv("GLIMMER_PERSISTENT_RUNTIME");
    return setting != nullptr && std::strcmp(setting, "1") == 0 &&
           std::getenv("GLIMMER_PERSISTENT_WORKER_BLOCKS") != nullptr;
}

CUresult runtime_driver_get_proc_address(const char* symbol, void** address, int version,
                                         cuuint64_t flags) {
    try {
        const CooperativeCallScope cooperative_call;
        if (!initialize_runtime_driver_lookup()) {
            return CUDA_ERROR_NOT_SUPPORTED;
        }
        RuntimeCallScope scope;
        return intercept_get_proc_address(symbol, address, version, flags);
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}

CUresult runtime_driver_get_proc_address_v2(const char* symbol, void** address, int version,
                                            cuuint64_t flags,
                                            CUdriverProcAddressQueryResult* status) {
    try {
        const CooperativeCallScope cooperative_call;
        if (!initialize_runtime_driver_lookup()) {
            return CUDA_ERROR_NOT_SUPPORTED;
        }
        RuntimeCallScope scope;
        return intercept_get_proc_address_v2(symbol, address, version, flags, status);
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}

}  // namespace glimmer::interceptor

extern "C" cudaError_t CUDARTAPI cudaMalloc(void** device_pointer, std::size_t memory_bytes) {
    if (device_pointer == nullptr) {
        return cudaErrorInvalidValue;
    }
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    try {
        RuntimeCallScope scope(true);
        const RuntimeMallocFunction real_allocate = resolve_runtime_malloc();
        if (is_reentrant) {
            return real_allocate == nullptr ? cudaErrorNotSupported
                                            : real_allocate(device_pointer, memory_bytes);
        }
        return invoke_runtime_source_memory([&] {
            return glimmer::interceptor::intercept_runtime_malloc(
                device_pointer, memory_bytes, real_allocate, resolve_runtime_free(),
                resolve_runtime_get_device());
        });
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMallocManaged(void** device_pointer, std::size_t memory_bytes,
                                                   unsigned int flags) {
    if (device_pointer == nullptr) {
        return cudaErrorInvalidValue;
    }
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    try {
        RuntimeCallScope scope(true);
        const RuntimeMallocManagedFunction real_allocate = resolve_runtime_malloc_managed();
        if (is_reentrant) {
            return real_allocate == nullptr ? cudaErrorNotSupported
                                            : real_allocate(device_pointer, memory_bytes, flags);
        }
        return invoke_runtime_source_memory([&] {
            return glimmer::interceptor::intercept_runtime_malloc_managed(
                device_pointer, memory_bytes, flags, real_allocate, resolve_runtime_free(),
                resolve_runtime_get_device());
        });
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMallocPitch(void** device_pointer, std::size_t* pitch,
                                                 std::size_t width_bytes, std::size_t height) {
    if (device_pointer == nullptr || pitch == nullptr) {
        return cudaErrorInvalidValue;
    }
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    try {
        RuntimeCallScope scope(true);
        const RuntimeMallocPitchFunction real_allocate = resolve_runtime_malloc_pitch();
        if (is_reentrant) {
            return real_allocate == nullptr
                       ? cudaErrorNotSupported
                       : real_allocate(device_pointer, pitch, width_bytes, height);
        }
        return invoke_runtime_source_memory([&] {
            return glimmer::interceptor::intercept_runtime_malloc_pitch(
                device_pointer, pitch, width_bytes, height, real_allocate, resolve_runtime_free(),
                resolve_runtime_get_device());
        });
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMalloc3D(struct cudaPitchedPtr* pitched_device_pointer,
                                              struct cudaExtent extent) {
    if (pitched_device_pointer == nullptr) {
        return cudaErrorInvalidValue;
    }
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    try {
        RuntimeCallScope scope(true);
        const RuntimeMalloc3DFunction real_allocate = resolve_runtime_malloc_3d();
        if (is_reentrant) {
            return real_allocate == nullptr ? cudaErrorNotSupported
                                            : real_allocate(pitched_device_pointer, extent);
        }
        return invoke_runtime_source_memory([&] {
            return glimmer::interceptor::intercept_runtime_malloc_3d(
                pitched_device_pointer, extent, real_allocate, resolve_runtime_free(),
                resolve_runtime_get_device());
        });
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaFree(void* device_pointer) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    try {
        RuntimeCallScope scope(true);
        const RuntimeFreeFunction real_release = resolve_runtime_free();
        if (is_reentrant) {
            return real_release == nullptr ? cudaErrorNotSupported : real_release(device_pointer);
        }
        return invoke_runtime_source_memory([&] {
            return glimmer::interceptor::intercept_runtime_free(device_pointer, real_release);
        });
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaIpcGetMemHandle(cudaIpcMemHandle_t* handle,
                                                     void* device_pointer) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    try {
        RuntimeCallScope scope(true);
        const RuntimeIpcGetMemHandleFunction real_get_handle = resolve_runtime_ipc_get_mem_handle();
        if (is_reentrant) {
            return real_get_handle == nullptr ? cudaErrorNotSupported
                                              : real_get_handle(handle, device_pointer);
        }
        return glimmer::interceptor::intercept_runtime_ipc_get_mem_handle(handle, device_pointer,
                                                                          real_get_handle);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaIpcOpenMemHandle(void** device_pointer,
                                                      cudaIpcMemHandle_t handle,
                                                      unsigned int flags) {
    if (device_pointer == nullptr) {
        return cudaErrorInvalidValue;
    }
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    try {
        RuntimeCallScope scope(true);
        const RuntimeIpcOpenMemHandleFunction real_open_handle =
            resolve_runtime_ipc_open_mem_handle();
        if (is_reentrant) {
            return real_open_handle == nullptr ? cudaErrorNotSupported
                                               : real_open_handle(device_pointer, handle, flags);
        }
        return glimmer::interceptor::intercept_runtime_ipc_open_mem_handle(device_pointer, handle,
                                                                           flags, real_open_handle);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaIpcCloseMemHandle(void* device_pointer) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    try {
        RuntimeCallScope scope(true);
        const RuntimeIpcCloseMemHandleFunction real_close_handle =
            resolve_runtime_ipc_close_mem_handle();
        if (is_reentrant) {
            return real_close_handle == nullptr ? cudaErrorNotSupported
                                                : real_close_handle(device_pointer);
        }
        return glimmer::interceptor::intercept_runtime_ipc_close_mem_handle(device_pointer,
                                                                            real_close_handle);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaImportExternalMemory(
    cudaExternalMemory_t* external_memory, const struct cudaExternalMemoryHandleDesc* handle_desc) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    try {
        RuntimeCallScope scope(true);
        const RuntimeImportExternalMemoryFunction real_import =
            resolve_runtime_import_external_memory();
        if (is_reentrant) {
            return real_import == nullptr ? cudaErrorNotSupported
                                          : real_import(external_memory, handle_desc);
        }
        return glimmer::interceptor::intercept_runtime_import_external_memory(
            external_memory, handle_desc, real_import);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI
cudaExternalMemoryGetMappedBuffer(void** device_pointer, cudaExternalMemory_t external_memory,
                                  const struct cudaExternalMemoryBufferDesc* buffer_desc) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    try {
        RuntimeCallScope scope(true);
        const RuntimeExternalMemoryGetMappedBufferFunction real_get_buffer =
            resolve_runtime_external_memory_get_mapped_buffer();
        if (is_reentrant) {
            return real_get_buffer == nullptr
                       ? cudaErrorNotSupported
                       : real_get_buffer(device_pointer, external_memory, buffer_desc);
        }
        return glimmer::interceptor::intercept_runtime_external_memory_get_mapped_buffer(
            device_pointer, external_memory, buffer_desc, real_get_buffer);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaExternalMemoryGetMappedMipmappedArray(
    cudaMipmappedArray_t* mipmap, cudaExternalMemory_t external_memory,
    const struct cudaExternalMemoryMipmappedArrayDesc* mipmap_desc) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    try {
        RuntimeCallScope scope(true);
        const RuntimeExternalMemoryGetMappedMipmappedArrayFunction real_get_mipmap =
            resolve_runtime_external_memory_get_mapped_mipmapped_array();
        if (is_reentrant) {
            return real_get_mipmap == nullptr
                       ? cudaErrorNotSupported
                       : real_get_mipmap(mipmap, external_memory, mipmap_desc);
        }
        return glimmer::interceptor::intercept_runtime_external_memory_get_mapped_mipmapped_array(
            mipmap, external_memory, mipmap_desc, real_get_mipmap);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaDestroyExternalMemory(cudaExternalMemory_t external_memory) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    try {
        RuntimeCallScope scope(true);
        const RuntimeDestroyExternalMemoryFunction real_destroy =
            resolve_runtime_destroy_external_memory();
        if (is_reentrant) {
            return real_destroy == nullptr ? cudaErrorNotSupported : real_destroy(external_memory);
        }
        return glimmer::interceptor::intercept_runtime_destroy_external_memory(external_memory,
                                                                               real_destroy);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMallocArray(cudaArray_t* array,
                                                 const struct cudaChannelFormatDesc* descriptor,
                                                 std::size_t width, std::size_t height,
                                                 unsigned int flags) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    try {
        RuntimeCallScope scope(true);
        const RuntimeMallocArrayFunction real_allocate = resolve_runtime_malloc_array();
        if (is_reentrant) {
            return real_allocate == nullptr
                       ? cudaErrorNotSupported
                       : real_allocate(array, descriptor, width, height, flags);
        }
        return glimmer::interceptor::intercept_runtime_malloc_array(array, descriptor, width,
                                                                    height, flags, real_allocate);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMalloc3DArray(cudaArray_t* array,
                                                   const struct cudaChannelFormatDesc* descriptor,
                                                   struct cudaExtent extent, unsigned int flags) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    try {
        RuntimeCallScope scope(true);
        const RuntimeMalloc3DArrayFunction real_allocate = resolve_runtime_malloc_3d_array();
        if (is_reentrant) {
            return real_allocate == nullptr ? cudaErrorNotSupported
                                            : real_allocate(array, descriptor, extent, flags);
        }
        return glimmer::interceptor::intercept_runtime_malloc_3d_array(array, descriptor, extent,
                                                                       flags, real_allocate);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMallocMipmappedArray(
    cudaMipmappedArray_t* mipmap, const struct cudaChannelFormatDesc* descriptor,
    struct cudaExtent extent, unsigned int level_count, unsigned int flags) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    try {
        RuntimeCallScope scope(true);
        const RuntimeMallocMipmappedArrayFunction real_allocate =
            resolve_runtime_malloc_mipmapped_array();
        if (is_reentrant) {
            return real_allocate == nullptr
                       ? cudaErrorNotSupported
                       : real_allocate(mipmap, descriptor, extent, level_count, flags);
        }
        return glimmer::interceptor::intercept_runtime_malloc_mipmapped_array(
            mipmap, descriptor, extent, level_count, flags, real_allocate);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaFreeArray(cudaArray_t array) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    try {
        RuntimeCallScope scope(true);
        const RuntimeFreeArrayFunction real_release = resolve_runtime_free_array();
        if (is_reentrant) {
            return real_release == nullptr ? cudaErrorNotSupported : real_release(array);
        }
        return glimmer::interceptor::intercept_runtime_free_array(array, real_release);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaFreeMipmappedArray(cudaMipmappedArray_t mipmap) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    try {
        RuntimeCallScope scope(true);
        const RuntimeFreeMipmappedArrayFunction real_release =
            resolve_runtime_free_mipmapped_array();
        if (is_reentrant) {
            return real_release == nullptr ? cudaErrorNotSupported : real_release(mipmap);
        }
        return glimmer::interceptor::intercept_runtime_free_mipmapped_array(mipmap, real_release);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaGraphicsUnregisterResource(cudaGraphicsResource_t resource) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    try {
        RuntimeCallScope scope(true);
        const RuntimeGraphicsUnregisterResourceFunction real_unregister =
            resolve_runtime_graphics_unregister_resource();
        if (is_reentrant) {
            return real_unregister == nullptr ? cudaErrorNotSupported : real_unregister(resource);
        }
        return glimmer::interceptor::intercept_runtime_graphics_unregister_resource(
            resource, real_unregister);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaGraphicsResourceSetMapFlags(cudaGraphicsResource_t resource,
                                                                 unsigned int flags) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    try {
        RuntimeCallScope scope(true);
        const RuntimeGraphicsResourceSetMapFlagsFunction real_set_flags =
            resolve_runtime_graphics_resource_set_map_flags();
        if (is_reentrant) {
            return real_set_flags == nullptr ? cudaErrorNotSupported
                                             : real_set_flags(resource, flags);
        }
        return glimmer::interceptor::intercept_runtime_graphics_resource_set_map_flags(
            resource, flags, real_set_flags);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaGraphicsMapResources(int count,
                                                          cudaGraphicsResource_t* resources,
                                                          cudaStream_t stream) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    try {
        RuntimeCallScope scope(true);
        const RuntimeGraphicsMapResourcesFunction real_map =
            resolve_runtime_graphics_map_resources();
        if (is_reentrant) {
            return real_map == nullptr ? cudaErrorNotSupported : real_map(count, resources, stream);
        }
        return glimmer::interceptor::intercept_runtime_graphics_map_resources(count, resources,
                                                                              stream, real_map);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaGraphicsUnmapResources(int count,
                                                            cudaGraphicsResource_t* resources,
                                                            cudaStream_t stream) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    try {
        RuntimeCallScope scope(true);
        const RuntimeGraphicsUnmapResourcesFunction real_unmap =
            resolve_runtime_graphics_unmap_resources();
        if (is_reentrant) {
            return real_unmap == nullptr ? cudaErrorNotSupported
                                         : real_unmap(count, resources, stream);
        }
        return glimmer::interceptor::intercept_runtime_graphics_unmap_resources(count, resources,
                                                                                stream, real_unmap);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaGraphicsResourceGetMappedPointer(
    void** device_pointer, std::size_t* size, cudaGraphicsResource_t resource) {
    if (device_pointer == nullptr || size == nullptr) {
        return cudaErrorInvalidValue;
    }
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    try {
        RuntimeCallScope scope(true);
        const RuntimeGraphicsResourceGetMappedPointerFunction real_get_pointer =
            resolve_runtime_graphics_resource_get_mapped_pointer();
        if (is_reentrant) {
            return real_get_pointer == nullptr ? cudaErrorNotSupported
                                               : real_get_pointer(device_pointer, size, resource);
        }
        return glimmer::interceptor::intercept_runtime_graphics_resource_get_mapped_pointer(
            device_pointer, size, resource, real_get_pointer);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI
cudaGraphicsSubResourceGetMappedArray(cudaArray_t* array, cudaGraphicsResource_t resource,
                                      unsigned int array_index, unsigned int mip_level) {
    if (array == nullptr) {
        return cudaErrorInvalidValue;
    }
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    try {
        RuntimeCallScope scope(true);
        const RuntimeGraphicsSubResourceGetMappedArrayFunction real_get_array =
            resolve_runtime_graphics_subresource_get_mapped_array();
        if (is_reentrant) {
            return real_get_array == nullptr
                       ? cudaErrorNotSupported
                       : real_get_array(array, resource, array_index, mip_level);
        }
        return glimmer::interceptor::intercept_runtime_graphics_subresource_get_mapped_array(
            array, resource, array_index, mip_level, real_get_array);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaGraphicsResourceGetMappedMipmappedArray(
    cudaMipmappedArray_t* mipmap, cudaGraphicsResource_t resource) {
    if (mipmap == nullptr) {
        return cudaErrorInvalidValue;
    }
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    try {
        RuntimeCallScope scope(true);
        const RuntimeGraphicsResourceGetMappedMipmappedArrayFunction real_get_mipmap =
            resolve_runtime_graphics_resource_get_mapped_mipmapped_array();
        if (is_reentrant) {
            return real_get_mipmap == nullptr ? cudaErrorNotSupported
                                              : real_get_mipmap(mipmap, resource);
        }
        return glimmer::interceptor::intercept_runtime_graphics_resource_get_mapped_mipmapped_array(
            mipmap, resource, real_get_mipmap);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaGraphAddMemAllocNode(
    cudaGraphNode_t* graph_node, cudaGraph_t graph, const cudaGraphNode_t* dependencies,
    std::size_t dependency_count, struct cudaMemAllocNodeParams* parameters) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    try {
        RuntimeCallScope scope(true);
        const RuntimeGraphAddMemAllocNodeFunction real_add_node =
            resolve_runtime_graph_add_mem_alloc_node();
        if (is_reentrant) {
            return real_add_node == nullptr ? cudaErrorNotSupported
                                            : real_add_node(graph_node, graph, dependencies,
                                                            dependency_count, parameters);
        }
        return glimmer::interceptor::intercept_runtime_graph_add_mem_alloc_node(
            graph_node, graph, dependencies, dependency_count, parameters, real_add_node);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI glimmer_runtime_graph_instantiate_legacy(
    cudaGraphExec_t* output, cudaGraph_t graph, cudaGraphNode_t* error_node, char* log,
    std::size_t log_bytes) {
    using Function = cudaError_t(CUDARTAPI*)(cudaGraphExec_t*, cudaGraph_t, cudaGraphNode_t*, char*,
                                             std::size_t);
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    try {
        const RuntimeCallScope runtime(true);
        static const auto original = reinterpret_cast<Function>(
            dlvsym(RTLD_NEXT, "cudaGraphInstantiate", "libcudart.so.11.0"));
        if (!original) return cudaErrorNotSupported;
        const auto invoke = [&](cudaGraphExec_t* native_output) {
            const RuntimeEventOriginalScope scope;
            return original(native_output, graph, error_node, log, log_bytes);
        };
        return is_reentrant ? invoke(output)
                            : glimmer::interceptor::intercept_runtime_graph_instantiation(
                                  output, graph, true, invoke);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI glimmer_runtime_graph_instantiate_modern(
    cudaGraphExec_t* output, cudaGraph_t graph, unsigned long long flags) {
    using Function = cudaError_t(CUDARTAPI*)(cudaGraphExec_t*, cudaGraph_t, unsigned long long);
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    try {
        const RuntimeCallScope runtime(true);
        static const auto original = reinterpret_cast<Function>(
            dlvsym(RTLD_NEXT, "cudaGraphInstantiate", "libcudart.so.12"));
        if (!original) return cudaErrorNotSupported;
        const auto invoke = [&](cudaGraphExec_t* native_output) {
            const RuntimeEventOriginalScope scope;
            return original(native_output, graph, flags);
        };
        return is_reentrant
                   ? invoke(output)
                   : glimmer::interceptor::intercept_runtime_graph_instantiation(
                         output, graph,
                         glimmer::interceptor::native_graph_flags_allow_effect_preparation(flags),
                         invoke);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

#if CUDART_VERSION >= 12000
asm(".symver glimmer_runtime_graph_instantiate_legacy,cudaGraphInstantiate@libcudart.so.11.0");
asm(".symver glimmer_runtime_graph_instantiate_modern,cudaGraphInstantiate@@libcudart.so.12");
#else
asm(".symver glimmer_runtime_graph_instantiate_legacy,cudaGraphInstantiate@@libcudart.so.11.0");
asm(".symver glimmer_runtime_graph_instantiate_modern,cudaGraphInstantiate@libcudart.so.12");
#endif

namespace glimmer::interceptor {
void* runtime_graph_constructor_version(const char* version) noexcept {
    if (!version) return nullptr;
    if (std::string_view(version) == "libcudart.so.11.0") {
        return reinterpret_cast<void*>(&glimmer_runtime_graph_instantiate_legacy);
    }
    if (std::string_view(version) == "libcudart.so.12") {
        return reinterpret_cast<void*>(&glimmer_runtime_graph_instantiate_modern);
    }
    return nullptr;
}

void* runtime_graph_abi_lookup(void* handle, const char* name, bool is_update) noexcept {
    const auto wrapper_version =
        is_update ? runtime_graph_update_version : runtime_graph_constructor_version;
    const auto resolve = resolve_real_dlsym();
    auto original = resolve ? resolve(handle == RTLD_DEFAULT ? RTLD_NEXT : handle, name) : nullptr;
    Dl_info owner{};
    if (!original || dladdr(original, &owner) == 0 || !owner.dli_fname) return nullptr;
    void* sdk = dlopen(owner.dli_fname, RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
    if (!sdk) return nullptr;
    // Old glibc dlsym ABI may choose an oldest RTLD_NEXT version. Once its
    // owning SDK is known, explicit lookup supplies that SDK's default ABI.
    if (handle == RTLD_DEFAULT) original = resolve(sdk, name);
    const auto legacy = dlvsym(sdk, name, "libcudart.so.11.0");
    const auto modern = dlvsym(sdk, name, "libcudart.so.12");
    void* result{};
    if (original == legacy && original != modern) {
        result = wrapper_version("libcudart.so.11.0");
    } else if (original == modern && original != legacy) {
        result = wrapper_version("libcudart.so.12");
    }
    if (!result && !cooperative_runtime_preview_requested()) {
        // Native passthrough needs no inferred prototype: return the SDK's
        // pointer unchanged. Only owned routing requires a known version.
        result = original;
    }
    if (dlclose(sdk) != 0) return nullptr;
    // A missing alternate SDK version is an internal probe, not a failed
    // application lookup. Do not return a valid pointer with a spurious error.
    if (result) (void)dlerror();
    return result;
}
void* runtime_graph_constructor_lookup(void* handle) noexcept {
    return runtime_graph_abi_lookup(handle, "cudaGraphInstantiate", false);
}
void* runtime_graph_update_lookup(void* handle) noexcept {
    return runtime_graph_abi_lookup(handle, "cudaGraphExecUpdate", true);
}
}  // namespace glimmer::interceptor

extern "C" cudaError_t CUDARTAPI glimmer_runtime_graph_update_legacy(
    cudaGraphExec_t executable, cudaGraph_t graph, cudaGraphNode_t* error_node,
    cudaGraphExecUpdateResult* result) {
    using Function = cudaError_t(CUDARTAPI*)(cudaGraphExec_t, cudaGraph_t, cudaGraphNode_t*,
                                             cudaGraphExecUpdateResult*);
    try {
        const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
        const RuntimeCallScope runtime(true);
        static const auto original = reinterpret_cast<Function>(
            dlvsym(RTLD_NEXT, "cudaGraphExecUpdate", "libcudart.so.11.0"));
        if (!original) return cudaErrorNotSupported;
        const auto invoke = [&] {
            const RuntimeEventOriginalScope scope;
            return original(executable, graph, error_node, result);
        };
        return is_reentrant ? invoke()
                            : glimmer::interceptor::intercept_runtime_graph_update(
                                  executable, graph, std::nullopt, invoke);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

#if CUDART_VERSION >= 12000
extern "C" cudaError_t CUDARTAPI glimmer_runtime_graph_update_modern(
    cudaGraphExec_t executable, cudaGraph_t graph, cudaGraphExecUpdateResultInfo* result) {
    using Function =
        cudaError_t(CUDARTAPI*)(cudaGraphExec_t, cudaGraph_t, cudaGraphExecUpdateResultInfo*);
    try {
        const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
        const RuntimeCallScope runtime(true);
        static const auto original =
            reinterpret_cast<Function>(dlvsym(RTLD_NEXT, "cudaGraphExecUpdate", "libcudart.so.12"));
        if (!original) return cudaErrorNotSupported;
        const auto invoke = [&] {
            const RuntimeEventOriginalScope scope;
            return original(executable, graph, result);
        };
        return is_reentrant ? invoke()
                            : glimmer::interceptor::intercept_runtime_graph_update(
                                  executable, graph, std::nullopt, invoke);
    } catch (...) {
        return cudaErrorUnknown;
    }
}
asm(".symver glimmer_runtime_graph_update_legacy,cudaGraphExecUpdate@libcudart.so.11.0");
asm(".symver glimmer_runtime_graph_update_modern,cudaGraphExecUpdate@@libcudart.so.12");
#else
asm(".symver glimmer_runtime_graph_update_legacy,cudaGraphExecUpdate@@libcudart.so.11.0");
#endif
namespace glimmer::interceptor {
void* runtime_graph_update_version(const char* version) noexcept {
    if (!version) return nullptr;
    if (std::string_view(version) == "libcudart.so.11.0")
        return reinterpret_cast<void*>(&glimmer_runtime_graph_update_legacy);
#if CUDART_VERSION >= 12000
    if (std::string_view(version) == "libcudart.so.12")
        return reinterpret_cast<void*>(&glimmer_runtime_graph_update_modern);
#endif
    return nullptr;
}
}  // namespace glimmer::interceptor

extern "C" cudaError_t CUDARTAPI cudaGraphExecChildGraphNodeSetParams(cudaGraphExec_t executable,
                                                                      cudaGraphNode_t node,
                                                                      cudaGraph_t child) {
    try {
        const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
        const RuntimeCallScope runtime(true);
        static const auto original = [] {
            const auto resolve = glimmer::interceptor::resolve_real_dlsym();
            return resolve ? reinterpret_cast<decltype(&cudaGraphExecChildGraphNodeSetParams)>(
                                 resolve(RTLD_NEXT, "cudaGraphExecChildGraphNodeSetParams"))
                           : nullptr;
        }();
        if (!original) return cudaErrorNotSupported;
        const auto invoke = [&] {
            const RuntimeEventOriginalScope scope;
            return original(executable, node, child);
        };
        return is_reentrant ? invoke()
                            : glimmer::interceptor::intercept_runtime_graph_update(
                                  executable, child, node, invoke);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaGraphInstantiateWithFlags(cudaGraphExec_t* output,
                                                               cudaGraph_t graph,
                                                               unsigned long long flags) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    try {
        const RuntimeCallScope runtime(true);
        static const auto original = [] {
            const auto resolver = glimmer::interceptor::resolve_real_dlsym();
            return resolver ? reinterpret_cast<decltype(&cudaGraphInstantiateWithFlags)>(
                                  resolver(RTLD_NEXT, "cudaGraphInstantiateWithFlags"))
                            : nullptr;
        }();
        if (!original) {
            return cudaErrorNotSupported;
        }
        const auto invoke = [&](cudaGraphExec_t* native_output) {
            const RuntimeEventOriginalScope scope;
            return original(native_output, graph, flags);
        };
        return is_reentrant
                   ? invoke(output)
                   : glimmer::interceptor::intercept_runtime_graph_instantiation(
                         output, graph,
                         glimmer::interceptor::native_graph_flags_allow_effect_preparation(flags),
                         invoke);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

#if CUDART_VERSION >= 12000
namespace {
cudaError_t instantiate_runtime_graph_params(cudaGraphExec_t* output, cudaGraph_t graph,
                                             cudaGraphInstantiateParams* parameters, bool is_ptds) {
    using Function = decltype(&cudaGraphInstantiateWithParams);
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    const RuntimeCallScope runtime(true);
    const auto resolve = [](const char* name) {
        const auto resolver = glimmer::interceptor::resolve_real_dlsym();
        return resolver ? reinterpret_cast<Function>(resolver(RTLD_NEXT, name)) : nullptr;
    };
    static const auto legacy = resolve("cudaGraphInstantiateWithParams");
    static const auto ptds = resolve("cudaGraphInstantiateWithParams_ptsz");
    const auto original = is_ptds ? ptds : legacy;
    if (!original) {
        return cudaErrorNotSupported;
    }
    const auto invoke = [&](cudaGraphExec_t* native_output) {
        const RuntimeEventOriginalScope scope;
        return original(native_output, graph, parameters);
    };
    return is_reentrant ? invoke(output)
                        : glimmer::interceptor::intercept_runtime_graph_instantiation(
                              output, graph, false, invoke);
}
}  // namespace

extern "C" cudaError_t CUDARTAPI cudaGraphInstantiateWithParams(
    cudaGraphExec_t* output, cudaGraph_t graph, cudaGraphInstantiateParams* parameters) {
    try {
        return instantiate_runtime_graph_params(output, graph, parameters, false);
    } catch (...) {
        return cudaErrorUnknown;
    }
}
extern "C" cudaError_t CUDARTAPI cudaGraphInstantiateWithParams_ptsz(
    cudaGraphExec_t* output, cudaGraph_t graph, cudaGraphInstantiateParams* parameters) {
    try {
        return instantiate_runtime_graph_params(output, graph, parameters, true);
    } catch (...) {
        return cudaErrorUnknown;
    }
}
#endif

extern "C" cudaError_t CUDARTAPI cudaGraphExecDestroy(cudaGraphExec_t executable) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    try {
        const RuntimeCallScope runtime(true);
        static const auto original = [] {
            const auto resolver = glimmer::interceptor::resolve_real_dlsym();
            return resolver ? reinterpret_cast<decltype(&cudaGraphExecDestroy)>(
                                  resolver(RTLD_NEXT, "cudaGraphExecDestroy"))
                            : nullptr;
        }();
        if (!original) {
            return cudaErrorNotSupported;
        }
        const auto invoke = [&] {
            const RuntimeEventOriginalScope scope;
            return original(executable);
        };
        return is_reentrant
                   ? invoke()
                   : glimmer::interceptor::intercept_runtime_graph_exec_destroy(executable, invoke);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

namespace {
template <bool kIsRecord>
cudaError_t update_runtime_graph_event(cudaGraphExec_t executable, cudaGraphNode_t node,
                                       cudaEvent_t event) {
    using Function = decltype(&cudaGraphExecEventRecordNodeSetEvent);
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    const RuntimeCallScope runtime(true);
    static const auto original = [] {
        const auto resolver = glimmer::interceptor::resolve_real_dlsym();
        return resolver ? reinterpret_cast<Function>(
                              resolver(RTLD_NEXT, kIsRecord ? "cudaGraphExecEventRecordNodeSetEvent"
                                                            : "cudaGraphExecEventWaitNodeSetEvent"))
                        : nullptr;
    }();
    if (!original) return cudaErrorNotSupported;
    const auto invoke = [&] {
        const RuntimeEventOriginalScope scope;
        return original(executable, node, event);
    };
    return is_reentrant ? invoke()
                        : glimmer::interceptor::intercept_runtime_graph_event_update(
                              executable, node, event, kIsRecord, invoke);
}
}  // namespace

extern "C" cudaError_t CUDARTAPI cudaGraphExecEventRecordNodeSetEvent(cudaGraphExec_t executable,
                                                                      cudaGraphNode_t node,
                                                                      cudaEvent_t event) {
    try {
        return update_runtime_graph_event<true>(executable, node, event);
    } catch (...) {
        return cudaErrorUnknown;
    }
}
extern "C" cudaError_t CUDARTAPI cudaGraphExecEventWaitNodeSetEvent(cudaGraphExec_t executable,
                                                                    cudaGraphNode_t node,
                                                                    cudaEvent_t event) {
    try {
        return update_runtime_graph_event<false>(executable, node, event);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

#if CUDART_VERSION >= 12020
extern "C" cudaError_t CUDARTAPI cudaGraphExecNodeSetParams(cudaGraphExec_t executable,
                                                            cudaGraphNode_t node,
                                                            cudaGraphNodeParams* params) {
    try {
        const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
        const RuntimeCallScope runtime(true);
        static const auto original = [] {
            const auto resolver = glimmer::interceptor::resolve_real_dlsym();
            return resolver ? reinterpret_cast<decltype(&cudaGraphExecNodeSetParams)>(
                                  resolver(RTLD_NEXT, "cudaGraphExecNodeSetParams"))
                            : nullptr;
        }();
        if (!original) return cudaErrorNotSupported;
        const auto invoke = [&] {
            const RuntimeEventOriginalScope scope;
            return original(executable, node, params);
        };
        if (is_reentrant) return invoke();
        if (params && params->type == cudaGraphNodeTypeGraph)
            return glimmer::interceptor::intercept_runtime_graph_update(
                executable, params->graph.graph, node, invoke);
        if (params && (params->type == cudaGraphNodeTypeEventRecord ||
                       params->type == cudaGraphNodeTypeWaitEvent)) {
            const bool is_record = params->type == cudaGraphNodeTypeEventRecord;
            return glimmer::interceptor::intercept_runtime_graph_event_update(
                executable, node, is_record ? params->eventRecord.event : params->eventWait.event,
                is_record, invoke);
        }
        auto expected = CU_GRAPH_NODE_TYPE_GRAPH;
        if (params) {
            switch (params->type) {
                case cudaGraphNodeTypeKernel:
                    expected = CU_GRAPH_NODE_TYPE_KERNEL;
                    break;
                case cudaGraphNodeTypeMemcpy:
                    expected = CU_GRAPH_NODE_TYPE_MEMCPY;
                    break;
                case cudaGraphNodeTypeMemset:
                    expected = CU_GRAPH_NODE_TYPE_MEMSET;
                    break;
                default:
                    break;
            }
        }
        return glimmer::interceptor::intercept_runtime_graph_node_update(executable, node, expected,
                                                                         invoke);
    } catch (...) {
        return cudaErrorUnknown;
    }
}
#endif

extern "C" cudaError_t CUDARTAPI cudaGraphNodeSetEnabled(cudaGraphExec_t executable,
                                                         cudaGraphNode_t node,
                                                         unsigned int is_enabled) {
    try {
        const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
        const RuntimeCallScope runtime(true);
        static const auto original = [] {
            const auto resolver = glimmer::interceptor::resolve_real_dlsym();
            return resolver ? reinterpret_cast<decltype(&cudaGraphNodeSetEnabled)>(
                                  resolver(RTLD_NEXT, "cudaGraphNodeSetEnabled"))
                            : nullptr;
        }();
        if (!original) return cudaErrorNotSupported;
        const auto invoke = [&] {
            const RuntimeEventOriginalScope scope;
            return original(executable, node, is_enabled);
        };
        return is_reentrant ? invoke()
                            : glimmer::interceptor::intercept_runtime_graph_node_update(
                                  executable, node, std::nullopt, invoke);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaGraphLaunch(cudaGraphExec_t graph_exec, cudaStream_t stream) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call() ||
                              glimmer::interceptor::is_inside_driver_call();
    try {
        RuntimeCallScope scope(true);
        const RuntimeGraphLaunchFunction real_launch = resolve_runtime_graph_launch();
        if (is_reentrant) {
            return real_launch == nullptr ? cudaErrorNotSupported : real_launch(graph_exec, stream);
        }
        return glimmer::interceptor::intercept_runtime_graph_launch("cudaGraphLaunch", graph_exec,
                                                                    stream, real_launch);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaGraphLaunch_ptsz(cudaGraphExec_t graph_exec,
                                                      cudaStream_t stream) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call() ||
                              glimmer::interceptor::is_inside_driver_call();
    try {
        RuntimeCallScope scope(true);
        const RuntimeGraphLaunchFunction real_launch = resolve_runtime_graph_launch_ptsz();
        if (is_reentrant) {
            return real_launch == nullptr ? cudaErrorNotSupported : real_launch(graph_exec, stream);
        }
        return glimmer::interceptor::intercept_runtime_graph_launch(
            "cudaGraphLaunch_ptsz", graph_exec, stream, real_launch, true);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMemGetInfo(std::size_t* free_bytes, std::size_t* total_bytes) {
    if (free_bytes == nullptr || total_bytes == nullptr) {
        return cudaErrorInvalidValue;
    }
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    try {
        RuntimeCallScope scope(true);
        const RuntimeMemGetInfoFunction real_query = resolve_runtime_mem_get_info();
        if (is_reentrant) {
            return real_query == nullptr ? cudaErrorNotSupported
                                         : real_query(free_bytes, total_bytes);
        }
        return glimmer::interceptor::intercept_runtime_mem_get_info(free_bytes, total_bytes,
                                                                    real_query);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMallocAsync(void** device_pointer, std::size_t memory_bytes,
                                                 cudaStream_t stream) {
    if (device_pointer == nullptr) {
        return cudaErrorInvalidValue;
    }
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    try {
        RuntimeCallScope scope(true);
        const RuntimeMallocAsyncFunction real_allocate = resolve_runtime_malloc_async();
        if (is_reentrant) {
            return real_allocate == nullptr ? cudaErrorNotSupported
                                            : real_allocate(device_pointer, memory_bytes, stream);
        }
        return glimmer::interceptor::intercept_runtime_malloc_async(
            device_pointer, memory_bytes, stream, real_allocate, resolve_runtime_free_async(),
            resolve_runtime_device_synchronize(), resolve_runtime_get_device());
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMallocAsync_ptsz(void** device_pointer,
                                                      std::size_t memory_bytes,
                                                      cudaStream_t stream) {
    if (device_pointer == nullptr) {
        return cudaErrorInvalidValue;
    }
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    try {
        RuntimeCallScope scope(true);
        const RuntimeMallocAsyncFunction real_allocate = resolve_runtime_malloc_async_ptsz();
        if (is_reentrant) {
            return real_allocate == nullptr ? cudaErrorNotSupported
                                            : real_allocate(device_pointer, memory_bytes, stream);
        }
        return glimmer::interceptor::intercept_runtime_malloc_async(
            device_pointer, memory_bytes, stream, real_allocate, resolve_runtime_free_async_ptsz(),
            resolve_runtime_device_synchronize(), resolve_runtime_get_device());
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMallocFromPoolAsync(void** device_pointer,
                                                         std::size_t memory_bytes,
                                                         cudaMemPool_t pool, cudaStream_t stream) {
    if (device_pointer == nullptr) {
        return cudaErrorInvalidValue;
    }
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    try {
        RuntimeCallScope scope(true);
        const RuntimeMallocFromPoolAsyncFunction real_allocate =
            resolve_runtime_malloc_from_pool_async();
        if (is_reentrant) {
            return real_allocate == nullptr
                       ? cudaErrorNotSupported
                       : real_allocate(device_pointer, memory_bytes, pool, stream);
        }
        return glimmer::interceptor::intercept_runtime_malloc_from_pool_async(
            device_pointer, memory_bytes, pool, stream, real_allocate, resolve_runtime_free_async(),
            resolve_runtime_device_synchronize(), resolve_runtime_get_device());
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMallocFromPoolAsync_ptsz(void** device_pointer,
                                                              std::size_t memory_bytes,
                                                              cudaMemPool_t pool,
                                                              cudaStream_t stream) {
    if (device_pointer == nullptr) {
        return cudaErrorInvalidValue;
    }
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    try {
        RuntimeCallScope scope(true);
        const RuntimeMallocFromPoolAsyncFunction real_allocate =
            resolve_runtime_malloc_from_pool_async_ptsz();
        if (is_reentrant) {
            return real_allocate == nullptr
                       ? cudaErrorNotSupported
                       : real_allocate(device_pointer, memory_bytes, pool, stream);
        }
        return glimmer::interceptor::intercept_runtime_malloc_from_pool_async(
            device_pointer, memory_bytes, pool, stream, real_allocate,
            resolve_runtime_free_async_ptsz(), resolve_runtime_device_synchronize(),
            resolve_runtime_get_device());
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaDeviceGetDefaultMemPool(cudaMemPool_t* pool, int device) {
    if (pool == nullptr) {
        return cudaErrorInvalidValue;
    }
    try {
        RuntimeCallScope scope(true);
        const RuntimeDeviceGetDefaultMemPoolFunction real_query =
            resolve_runtime_device_get_default_mem_pool();
        return real_query == nullptr ? cudaErrorNotSupported : real_query(pool, device);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaDeviceSetMemPool(int device, cudaMemPool_t pool) {
    if (pool == nullptr) {
        return cudaErrorInvalidValue;
    }
    try {
        RuntimeCallScope scope(true);
        const RuntimeDeviceSetMemPoolFunction real_set = resolve_runtime_device_set_mem_pool();
        return real_set == nullptr ? cudaErrorNotSupported : real_set(device, pool);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaDeviceGetMemPool(cudaMemPool_t* pool, int device) {
    if (pool == nullptr) {
        return cudaErrorInvalidValue;
    }
    try {
        RuntimeCallScope scope(true);
        const RuntimeDeviceGetMemPoolFunction real_query = resolve_runtime_device_get_mem_pool();
        return real_query == nullptr ? cudaErrorNotSupported : real_query(pool, device);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMemPoolTrimTo(cudaMemPool_t pool,
                                                   std::size_t min_bytes_to_keep) {
    if (pool == nullptr) {
        return cudaErrorInvalidValue;
    }
    try {
        RuntimeCallScope scope(true);
        const RuntimeMemPoolTrimToFunction real_trim = resolve_runtime_mem_pool_trim_to();
        return real_trim == nullptr ? cudaErrorNotSupported : real_trim(pool, min_bytes_to_keep);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMemPoolSetAttribute(cudaMemPool_t pool,
                                                         cudaMemPoolAttr attribute, void* value) {
    if (pool == nullptr || value == nullptr) {
        return cudaErrorInvalidValue;
    }
    try {
        RuntimeCallScope scope(true);
        const RuntimeMemPoolSetAttributeFunction real_set =
            resolve_runtime_mem_pool_set_attribute();
        return real_set == nullptr ? cudaErrorNotSupported : real_set(pool, attribute, value);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMemPoolGetAttribute(cudaMemPool_t pool,
                                                         cudaMemPoolAttr attribute, void* value) {
    if (pool == nullptr || value == nullptr) {
        return cudaErrorInvalidValue;
    }
    try {
        RuntimeCallScope scope(true);
        const RuntimeMemPoolGetAttributeFunction real_get =
            resolve_runtime_mem_pool_get_attribute();
        return real_get == nullptr ? cudaErrorNotSupported : real_get(pool, attribute, value);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMemPoolSetAccess(cudaMemPool_t pool,
                                                      const cudaMemAccessDesc* descriptors,
                                                      std::size_t descriptor_count) {
    if (pool == nullptr || (descriptor_count != 0 && descriptors == nullptr)) {
        return cudaErrorInvalidValue;
    }
    try {
        RuntimeCallScope scope(true);
        const RuntimeMemPoolSetAccessFunction real_set = resolve_runtime_mem_pool_set_access();
        return real_set == nullptr ? cudaErrorNotSupported
                                   : real_set(pool, descriptors, descriptor_count);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMemPoolGetAccess(cudaMemAccessFlags* flags, cudaMemPool_t pool,
                                                      cudaMemLocation* location) {
    if (flags == nullptr || pool == nullptr || location == nullptr) {
        return cudaErrorInvalidValue;
    }
    try {
        RuntimeCallScope scope(true);
        const RuntimeMemPoolGetAccessFunction real_get = resolve_runtime_mem_pool_get_access();
        return real_get == nullptr ? cudaErrorNotSupported : real_get(flags, pool, location);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMemPoolCreate(cudaMemPool_t* pool,
                                                   const cudaMemPoolProps* properties) {
    if (pool == nullptr || properties == nullptr) {
        return cudaErrorInvalidValue;
    }
    try {
        RuntimeCallScope scope(true);
        const RuntimeMemPoolCreateFunction real_create = resolve_runtime_mem_pool_create();
        return real_create == nullptr ? cudaErrorNotSupported : real_create(pool, properties);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMemPoolDestroy(cudaMemPool_t pool) {
    if (pool == nullptr) {
        return cudaErrorInvalidValue;
    }
    try {
        RuntimeCallScope scope(true);
        const RuntimeMemPoolDestroyFunction real_destroy = resolve_runtime_mem_pool_destroy();
        return real_destroy == nullptr ? cudaErrorNotSupported : real_destroy(pool);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMemGetDefaultMemPool(cudaMemPool_t* pool,
                                                          cudaMemLocation* location,
                                                          cudaMemAllocationType allocation_type) {
    if (pool == nullptr || location == nullptr) {
        return cudaErrorInvalidValue;
    }
    try {
        RuntimeCallScope scope(true);
        const RuntimeMemGetDefaultMemPoolFunction real_get =
            resolve_runtime_mem_get_default_mem_pool();
        return real_get == nullptr ? cudaErrorNotSupported
                                   : real_get(pool, location, allocation_type);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMemGetMemPool(cudaMemPool_t* pool, cudaMemLocation* location,
                                                   cudaMemAllocationType allocation_type) {
    if (pool == nullptr || location == nullptr) {
        return cudaErrorInvalidValue;
    }
    try {
        RuntimeCallScope scope(true);
        const RuntimeMemGetMemPoolFunction real_get = resolve_runtime_mem_get_mem_pool();
        return real_get == nullptr ? cudaErrorNotSupported
                                   : real_get(pool, location, allocation_type);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMemSetMemPool(cudaMemLocation* location,
                                                   cudaMemAllocationType allocation_type,
                                                   cudaMemPool_t pool) {
    if (location == nullptr || pool == nullptr) {
        return cudaErrorInvalidValue;
    }
    try {
        RuntimeCallScope scope(true);
        const RuntimeMemSetMemPoolFunction real_set = resolve_runtime_mem_set_mem_pool();
        return real_set == nullptr ? cudaErrorNotSupported
                                   : real_set(location, allocation_type, pool);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMemPoolExportToShareableHandle(
    void* handle_out, cudaMemPool_t pool, enum cudaMemAllocationHandleType handle_type,
    unsigned int flags) {
    if (handle_out == nullptr || pool == nullptr) {
        return cudaErrorInvalidValue;
    }
    try {
        RuntimeCallScope scope(true);
        const RuntimeMemPoolExportToShareableHandleFunction real_export =
            resolve_runtime_mem_pool_export_to_shareable_handle();
        return real_export == nullptr ? cudaErrorNotSupported
                                      : real_export(handle_out, pool, handle_type, flags);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMemPoolImportFromShareableHandle(
    cudaMemPool_t* pool_out, void* handle, enum cudaMemAllocationHandleType handle_type,
    unsigned int flags) {
    if (pool_out == nullptr) {
        return cudaErrorInvalidValue;
    }
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    try {
        RuntimeCallScope scope(true);
        const RuntimeMemPoolImportFromShareableHandleFunction real_import =
            resolve_runtime_mem_pool_import_from_shareable_handle();
        if (is_reentrant) {
            return real_import == nullptr ? cudaErrorNotSupported
                                          : real_import(pool_out, handle, handle_type, flags);
        }
        return glimmer::interceptor::intercept_runtime_mem_pool_import_from_shareable_handle(
            pool_out, handle, handle_type, flags, real_import);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMemPoolExportPointer(cudaMemPoolPtrExportData* share_data_out,
                                                          void* device_pointer) {
    if (share_data_out == nullptr || device_pointer == nullptr) {
        return cudaErrorInvalidValue;
    }
    try {
        RuntimeCallScope scope(true);
        const RuntimeMemPoolExportPointerFunction real_export =
            resolve_runtime_mem_pool_export_pointer();
        return real_export == nullptr ? cudaErrorNotSupported
                                      : real_export(share_data_out, device_pointer);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMemPoolImportPointer(void** pointer_out, cudaMemPool_t pool,
                                                          cudaMemPoolPtrExportData* share_data) {
    if (pointer_out == nullptr || pool == nullptr || share_data == nullptr) {
        return cudaErrorInvalidValue;
    }
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    try {
        RuntimeCallScope scope(true);
        const RuntimeMemPoolImportPointerFunction real_import =
            resolve_runtime_mem_pool_import_pointer();
        if (is_reentrant) {
            return real_import == nullptr ? cudaErrorNotSupported
                                          : real_import(pointer_out, pool, share_data);
        }
        return glimmer::interceptor::intercept_runtime_mem_pool_import_pointer(
            pointer_out, pool, share_data, real_import);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaFreeAsync(void* device_pointer, cudaStream_t stream) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    try {
        RuntimeCallScope scope(true);
        const RuntimeFreeAsyncFunction real_release = resolve_runtime_free_async();
        if (is_reentrant) {
            return real_release == nullptr ? cudaErrorNotSupported
                                           : real_release(device_pointer, stream);
        }
        return glimmer::interceptor::intercept_runtime_free_async(device_pointer, stream,
                                                                  real_release);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaFreeAsync_ptsz(void* device_pointer, cudaStream_t stream) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    try {
        RuntimeCallScope scope(true);
        const RuntimeFreeAsyncFunction real_release = resolve_runtime_free_async_ptsz();
        if (is_reentrant) {
            return real_release == nullptr ? cudaErrorNotSupported
                                           : real_release(device_pointer, stream);
        }
        return glimmer::interceptor::intercept_runtime_free_async(device_pointer, stream,
                                                                  real_release);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaDeviceSynchronize() {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    try {
        RuntimeCallScope scope(true);
        const RuntimeDeviceSynchronizeFunction real_synchronize =
            resolve_runtime_device_synchronize();
        if (is_reentrant) {
            return real_synchronize == nullptr ? cudaErrorNotSupported : real_synchronize();
        }
        const auto completed =
            glimmer::interceptor::intercept_normal_runtime_device_completion([&] {
                const RuntimeEventOriginalScope original;
                return real_synchronize == nullptr ? cudaErrorNotSupported : real_synchronize();
            });
        if (completed) {
            return *completed;
        }
        return glimmer::interceptor::intercept_runtime_device_synchronize(real_synchronize);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaDeviceReset() {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    try {
        RuntimeCallScope scope(true);
        const auto reset = resolve_runtime_device_reset();
        if (is_reentrant) {
            return reset == nullptr ? cudaErrorNotSupported : reset();
        }
        const RuntimeEventOriginalScope original;
        return glimmer::interceptor::intercept_runtime_device_reset(reset);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaEventCreate(cudaEvent_t* event) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    try {
        RuntimeCallScope scope(true);
        using Function = cudaError_t (*)(cudaEvent_t* event);
        static const Function native = []() noexcept {
            const auto lookup = glimmer::interceptor::resolve_real_dlsym();
            return lookup == nullptr
                       ? nullptr
                       : reinterpret_cast<Function>(lookup(RTLD_NEXT, "cudaEventCreate"));
        }();
        if (!is_reentrant && glimmer::interceptor::cooperative_runtime_preview_requested() &&
            glimmer::interceptor::cooperative_process_call_gate()
                .is_native_only_on_current_thread() &&
            !glimmer::interceptor::is_inside_driver_call()) {
            return glimmer::interceptor::intercept_cooperative_runtime_event_create(
                event, cudaEventDefault, [&] {
                    const RuntimeEventOriginalScope original_scope;
                    return native == nullptr ? cudaErrorNotSupported : native(event);
                });
        }
        if (!is_reentrant) {
            const auto accepted = glimmer::interceptor::intercept_normal_runtime_event_create(
                event, cudaEventDefault, [&] {
                    const RuntimeEventOriginalScope original_scope;
                    return native == nullptr ? cudaErrorNotSupported : native(event);
                });
            if (accepted) {
                return *accepted;
            }
        }
        return native == nullptr ? cudaErrorNotSupported : native(event);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    try {
        RuntimeCallScope scope(true);
        using Function = cudaError_t (*)(cudaEvent_t* event, unsigned int flags);
        static const Function native = []() noexcept {
            const auto lookup = glimmer::interceptor::resolve_real_dlsym();
            return lookup == nullptr
                       ? nullptr
                       : reinterpret_cast<Function>(lookup(RTLD_NEXT, "cudaEventCreateWithFlags"));
        }();
        if (!is_reentrant && glimmer::interceptor::cooperative_runtime_preview_requested() &&
            glimmer::interceptor::cooperative_process_call_gate()
                .is_native_only_on_current_thread() &&
            !glimmer::interceptor::is_inside_driver_call()) {
            return glimmer::interceptor::intercept_cooperative_runtime_event_create(
                event, flags, [&] {
                    const RuntimeEventOriginalScope original_scope;
                    return native == nullptr ? cudaErrorNotSupported : native(event, flags);
                });
        }
        if (!is_reentrant) {
            const auto accepted =
                glimmer::interceptor::intercept_normal_runtime_event_create(event, flags, [&] {
                    const RuntimeEventOriginalScope original_scope;
                    return native == nullptr ? cudaErrorNotSupported : native(event, flags);
                });
            if (accepted) {
                return *accepted;
            }
        }
        return native == nullptr ? cudaErrorNotSupported : native(event, flags);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    try {
        RuntimeCallScope scope(true);
        using Function = cudaError_t (*)(cudaEvent_t event, cudaStream_t stream);
        static const Function native = []() noexcept {
            const auto lookup = glimmer::interceptor::resolve_real_dlsym();
            return lookup == nullptr
                       ? nullptr
                       : reinterpret_cast<Function>(lookup(RTLD_NEXT, "cudaEventRecord"));
        }();
        if (!is_reentrant && glimmer::interceptor::cooperative_runtime_preview_requested() &&
            glimmer::interceptor::cooperative_process_call_gate()
                .is_native_only_on_current_thread() &&
            !glimmer::interceptor::is_inside_driver_call()) {
            return glimmer::interceptor::intercept_cooperative_runtime_event_record(
                event, stream, false, 0, [&] {
                    const RuntimeEventOriginalScope original_scope;
                    return native == nullptr ? cudaErrorNotSupported : native(event, stream);
                });
        }
        if (!is_reentrant) {
            const auto recorded = glimmer::interceptor::intercept_normal_runtime_event_record(
                event, stream, false, [&] {
                    const RuntimeEventOriginalScope original_scope;
                    return native == nullptr ? cudaErrorNotSupported : native(event, stream);
                });
            if (recorded) {
                return *recorded;
            }
        }
        return native == nullptr ? cudaErrorNotSupported : native(event, stream);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaEventRecord_ptsz(cudaEvent_t event, cudaStream_t stream) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    try {
        RuntimeCallScope scope(true);
        using Function = cudaError_t (*)(cudaEvent_t event, cudaStream_t stream);
        static const Function native = []() noexcept {
            const auto lookup = glimmer::interceptor::resolve_real_dlsym();
            return lookup == nullptr
                       ? nullptr
                       : reinterpret_cast<Function>(lookup(RTLD_NEXT, "cudaEventRecord_ptsz"));
        }();
        if (!is_reentrant && glimmer::interceptor::cooperative_runtime_preview_requested() &&
            glimmer::interceptor::cooperative_process_call_gate()
                .is_native_only_on_current_thread() &&
            !glimmer::interceptor::is_inside_driver_call()) {
            return glimmer::interceptor::intercept_cooperative_runtime_event_record(
                event, stream, true, 0, [&] {
                    const RuntimeEventOriginalScope original_scope;
                    return native == nullptr ? cudaErrorNotSupported : native(event, stream);
                });
        }
        if (!is_reentrant) {
            const auto recorded = glimmer::interceptor::intercept_normal_runtime_event_record(
                event, stream, true, [&] {
                    const RuntimeEventOriginalScope original_scope;
                    return native == nullptr ? cudaErrorNotSupported : native(event, stream);
                });
            if (recorded) {
                return *recorded;
            }
        }
        return native == nullptr ? cudaErrorNotSupported : native(event, stream);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream,
                                                          unsigned int flags) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    try {
        RuntimeCallScope scope(true);
        using Function =
            cudaError_t (*)(cudaEvent_t event, cudaStream_t stream, unsigned int flags);
        static const Function native = []() noexcept {
            const auto lookup = glimmer::interceptor::resolve_real_dlsym();
            return lookup == nullptr
                       ? nullptr
                       : reinterpret_cast<Function>(lookup(RTLD_NEXT, "cudaEventRecordWithFlags"));
        }();
        if (!is_reentrant && glimmer::interceptor::cooperative_runtime_preview_requested() &&
            glimmer::interceptor::cooperative_process_call_gate()
                .is_native_only_on_current_thread() &&
            !glimmer::interceptor::is_inside_driver_call()) {
            return glimmer::interceptor::intercept_cooperative_runtime_event_record(
                event, stream, false, flags, [&] {
                    const RuntimeEventOriginalScope original_scope;
                    return native == nullptr ? cudaErrorNotSupported : native(event, stream, flags);
                });
        }
        if (!is_reentrant) {
            const auto recorded = glimmer::interceptor::intercept_normal_runtime_event_record(
                event, stream, false, [&] {
                    const RuntimeEventOriginalScope original_scope;
                    return native == nullptr ? cudaErrorNotSupported : native(event, stream, flags);
                });
            if (recorded) {
                return *recorded;
            }
        }
        return native == nullptr ? cudaErrorNotSupported : native(event, stream, flags);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaEventRecordWithFlags_ptsz(cudaEvent_t event,
                                                               cudaStream_t stream,
                                                               unsigned int flags) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    try {
        RuntimeCallScope scope(true);
        using Function =
            cudaError_t (*)(cudaEvent_t event, cudaStream_t stream, unsigned int flags);
        static const Function native = []() noexcept {
            const auto lookup = glimmer::interceptor::resolve_real_dlsym();
            return lookup == nullptr ? nullptr
                                     : reinterpret_cast<Function>(
                                           lookup(RTLD_NEXT, "cudaEventRecordWithFlags_ptsz"));
        }();
        if (!is_reentrant && glimmer::interceptor::cooperative_runtime_preview_requested() &&
            glimmer::interceptor::cooperative_process_call_gate()
                .is_native_only_on_current_thread() &&
            !glimmer::interceptor::is_inside_driver_call()) {
            return glimmer::interceptor::intercept_cooperative_runtime_event_record(
                event, stream, true, flags, [&] {
                    const RuntimeEventOriginalScope original_scope;
                    return native == nullptr ? cudaErrorNotSupported : native(event, stream, flags);
                });
        }
        if (!is_reentrant) {
            const auto recorded = glimmer::interceptor::intercept_normal_runtime_event_record(
                event, stream, true, [&] {
                    const RuntimeEventOriginalScope original_scope;
                    return native == nullptr ? cudaErrorNotSupported : native(event, stream, flags);
                });
            if (recorded) {
                return *recorded;
            }
        }
        return native == nullptr ? cudaErrorNotSupported : native(event, stream, flags);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaEventQuery(cudaEvent_t event) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    try {
        RuntimeCallScope scope(true);
        using Function = cudaError_t (*)(cudaEvent_t event);
        static const Function native = []() noexcept {
            const auto lookup = glimmer::interceptor::resolve_real_dlsym();
            return lookup == nullptr
                       ? nullptr
                       : reinterpret_cast<Function>(lookup(RTLD_NEXT, "cudaEventQuery"));
        }();
        if (!is_reentrant) {
            const auto completed =
                glimmer::interceptor::intercept_normal_runtime_event_completion(event, false, [&] {
                    const RuntimeEventOriginalScope original_scope;
                    return native == nullptr ? cudaErrorNotSupported : native(event);
                });
            if (completed) {
                return *completed;
            }
        }
        return native == nullptr ? cudaErrorNotSupported : native(event);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaEventSynchronize(cudaEvent_t event) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    try {
        RuntimeCallScope scope(true);
        using Function = cudaError_t (*)(cudaEvent_t event);
        static const Function native = []() noexcept {
            const auto lookup = glimmer::interceptor::resolve_real_dlsym();
            return lookup == nullptr
                       ? nullptr
                       : reinterpret_cast<Function>(lookup(RTLD_NEXT, "cudaEventSynchronize"));
        }();
        if (!is_reentrant) {
            const auto completed =
                glimmer::interceptor::intercept_normal_runtime_event_completion(event, true, [&] {
                    const RuntimeEventOriginalScope original_scope;
                    return native == nullptr ? cudaErrorNotSupported : native(event);
                });
            if (completed) {
                return *completed;
            }
        }
        return native == nullptr ? cudaErrorNotSupported : native(event);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaEventDestroy(cudaEvent_t event) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    try {
        RuntimeCallScope scope(true);
        using Function = cudaError_t (*)(cudaEvent_t event);
        static const Function native = []() noexcept {
            const auto lookup = glimmer::interceptor::resolve_real_dlsym();
            return lookup == nullptr
                       ? nullptr
                       : reinterpret_cast<Function>(lookup(RTLD_NEXT, "cudaEventDestroy"));
        }();
        if (!is_reentrant && glimmer::interceptor::cooperative_runtime_preview_requested() &&
            glimmer::interceptor::cooperative_process_call_gate()
                .is_native_only_on_current_thread() &&
            !glimmer::interceptor::is_inside_driver_call()) {
            return glimmer::interceptor::intercept_cooperative_runtime_event_destroy(event, [&] {
                const RuntimeEventOriginalScope original_scope;
                return native == nullptr ? cudaErrorNotSupported : native(event);
            });
        }
        if (!is_reentrant) {
            const auto retired =
                glimmer::interceptor::intercept_normal_runtime_event_destroy(event, [&] {
                    const RuntimeEventOriginalScope original_scope;
                    return native == nullptr ? cudaErrorNotSupported : native(event);
                });
            if (retired) {
                return *retired;
            }
        }
        return native == nullptr ? cudaErrorNotSupported : native(event);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaEventElapsedTime(float* milliseconds, cudaEvent_t start,
                                                      cudaEvent_t end) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    try {
        RuntimeCallScope scope(true);
        using Function = cudaError_t (*)(float* milliseconds, cudaEvent_t start, cudaEvent_t end);
        static const Function native = []() noexcept {
            const auto lookup = glimmer::interceptor::resolve_real_dlsym();
            return lookup == nullptr
                       ? nullptr
                       : reinterpret_cast<Function>(lookup(RTLD_NEXT, "cudaEventElapsedTime"));
        }();
        if (!is_reentrant) {
            const auto measured = glimmer::interceptor::intercept_normal_runtime_event_elapsed_time(
                milliseconds, start, end, [&](float* output) {
                    const RuntimeEventOriginalScope original_scope;
                    return native == nullptr ? cudaErrorNotSupported : native(output, start, end);
                });
            if (measured) {
                return *measured;
            }
        }
        return native == nullptr ? cudaErrorNotSupported : native(milliseconds, start, end);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

namespace {
template <typename Operation>
cudaError_t runtime_capture_call(glimmer::interceptor::CudaSourceCaptureKind kind,
                                 Operation&& original) noexcept {
    try {
        // Capture root must remain outside Runtime scope until the retained
        // process transition lends its one actual Driver-origin callback.
        const glimmer::interceptor::CooperativeCallScope call;
        return glimmer::interceptor::intercept_runtime_capture(kind, [&] {
            const RuntimeCallScope runtime;
            return original();
        });
    } catch (...) {
        return cudaErrorUnknown;
    }
}
}  // namespace

extern "C" cudaError_t CUDARTAPI cudaStreamBeginCapture(cudaStream_t stream,
                                                        cudaStreamCaptureMode mode) {
    static const auto original = [] {
        const auto lookup = glimmer::interceptor::resolve_real_dlsym();
        return lookup ? reinterpret_cast<decltype(&cudaStreamBeginCapture)>(
                            lookup(RTLD_NEXT, "cudaStreamBeginCapture"))
                      : nullptr;
    }();
    if (!original) return cudaErrorNotSupported;
    return runtime_capture_call(glimmer::interceptor::CudaSourceCaptureKind::kBegin,
                                [&] { return original(stream, mode); });
}

extern "C" cudaError_t CUDARTAPI cudaStreamBeginCapture_ptsz(cudaStream_t stream,
                                                             cudaStreamCaptureMode mode) {
    static const auto original = [] {
        const auto lookup = glimmer::interceptor::resolve_real_dlsym();
        return lookup ? reinterpret_cast<decltype(&cudaStreamBeginCapture)>(
                            lookup(RTLD_NEXT, "cudaStreamBeginCapture_ptsz"))
                      : nullptr;
    }();
    if (!original) return cudaErrorNotSupported;
    return runtime_capture_call(glimmer::interceptor::CudaSourceCaptureKind::kBegin,
                                [&] { return original(stream, mode); });
}

extern "C" cudaError_t CUDARTAPI cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t* graph) {
    static const auto original = [] {
        const auto lookup = glimmer::interceptor::resolve_real_dlsym();
        return lookup ? reinterpret_cast<decltype(&cudaStreamEndCapture)>(
                            lookup(RTLD_NEXT, "cudaStreamEndCapture"))
                      : nullptr;
    }();
    if (!original) return cudaErrorNotSupported;
    return runtime_capture_call(glimmer::interceptor::CudaSourceCaptureKind::kEnd,
                                [&] { return original(stream, graph); });
}

extern "C" cudaError_t CUDARTAPI cudaStreamEndCapture_ptsz(cudaStream_t stream,
                                                           cudaGraph_t* graph) {
    static const auto original = [] {
        const auto lookup = glimmer::interceptor::resolve_real_dlsym();
        return lookup ? reinterpret_cast<decltype(&cudaStreamEndCapture)>(
                            lookup(RTLD_NEXT, "cudaStreamEndCapture_ptsz"))
                      : nullptr;
    }();
    if (!original) return cudaErrorNotSupported;
    return runtime_capture_call(glimmer::interceptor::CudaSourceCaptureKind::kEnd,
                                [&] { return original(stream, graph); });
}

extern "C" cudaError_t CUDARTAPI cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event,
                                                     unsigned int flags) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    try {
        RuntimeCallScope scope(true);
        using Function =
            cudaError_t (*)(cudaStream_t stream, cudaEvent_t event, unsigned int flags);
        static const Function native = []() noexcept {
            const auto lookup = glimmer::interceptor::resolve_real_dlsym();
            return lookup == nullptr
                       ? nullptr
                       : reinterpret_cast<Function>(lookup(RTLD_NEXT, "cudaStreamWaitEvent"));
        }();
        if (!is_reentrant) {
            const auto waited = glimmer::interceptor::intercept_normal_runtime_event_wait(
                stream, event, false, flags, [&] {
                    const RuntimeEventOriginalScope original_scope;
                    return native == nullptr ? cudaErrorNotSupported : native(stream, event, flags);
                });
            if (waited) {
                return *waited;
            }
        }
        return native == nullptr ? cudaErrorNotSupported : native(stream, event, flags);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaStreamWaitEvent_ptsz(cudaStream_t stream, cudaEvent_t event,
                                                          unsigned int flags) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    try {
        RuntimeCallScope scope(true);
        using Function =
            cudaError_t (*)(cudaStream_t stream, cudaEvent_t event, unsigned int flags);
        static const Function native = []() noexcept {
            const auto lookup = glimmer::interceptor::resolve_real_dlsym();
            return lookup == nullptr
                       ? nullptr
                       : reinterpret_cast<Function>(lookup(RTLD_NEXT, "cudaStreamWaitEvent_ptsz"));
        }();
        if (!is_reentrant) {
            const auto waited = glimmer::interceptor::intercept_normal_runtime_event_wait(
                stream, event, true, flags, [&] {
                    const RuntimeEventOriginalScope original_scope;
                    return native == nullptr ? cudaErrorNotSupported : native(stream, event, flags);
                });
            if (waited) {
                return *waited;
            }
        }
        return native == nullptr ? cudaErrorNotSupported : native(stream, event, flags);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaStreamSynchronize(cudaStream_t stream) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    try {
        RuntimeCallScope scope(true);
        const RuntimeStreamSynchronizeFunction real_synchronize =
            resolve_runtime_stream_synchronize();
        if (is_reentrant) {
            return real_synchronize == nullptr ? cudaErrorNotSupported : real_synchronize(stream);
        }
        const auto completed = glimmer::interceptor::intercept_normal_runtime_stream_completion(
            stream, false, true, [&] {
                const RuntimeEventOriginalScope original;
                return real_synchronize == nullptr ? cudaErrorNotSupported
                                                   : real_synchronize(stream);
            });
        return completed ? *completed
                         : glimmer::interceptor::intercept_runtime_stream_synchronize(
                               stream, real_synchronize);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaStreamSynchronize_ptsz(cudaStream_t stream) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    try {
        RuntimeCallScope scope(true);
        const RuntimeStreamSynchronizeFunction real_synchronize =
            resolve_runtime_stream_synchronize_ptsz();
        if (is_reentrant) {
            return real_synchronize == nullptr ? cudaErrorNotSupported : real_synchronize(stream);
        }
        const auto completed = glimmer::interceptor::intercept_normal_runtime_stream_completion(
            stream, true, true, [&] {
                const RuntimeEventOriginalScope original;
                return real_synchronize == nullptr ? cudaErrorNotSupported
                                                   : real_synchronize(stream);
            });
        return completed ? *completed
                         : glimmer::interceptor::intercept_runtime_stream_synchronize(
                               stream, real_synchronize);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaStreamQuery(cudaStream_t stream) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    try {
        RuntimeCallScope scope(true);
        const RuntimeStreamQueryFunction real_query = resolve_runtime_stream_query();
        if (is_reentrant) {
            return real_query == nullptr ? cudaErrorNotSupported : real_query(stream);
        }
        const auto completed = glimmer::interceptor::intercept_normal_runtime_stream_completion(
            stream, false, false, [&] {
                const RuntimeEventOriginalScope original;
                return real_query == nullptr ? cudaErrorNotSupported : real_query(stream);
            });
        return completed ? *completed
                         : glimmer::interceptor::intercept_runtime_stream_query(stream, real_query);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaStreamQuery_ptsz(cudaStream_t stream) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    try {
        RuntimeCallScope scope(true);
        const RuntimeStreamQueryFunction real_query = resolve_runtime_stream_query_ptsz();
        if (is_reentrant) {
            return real_query == nullptr ? cudaErrorNotSupported : real_query(stream);
        }
        const auto completed = glimmer::interceptor::intercept_normal_runtime_stream_completion(
            stream, true, false, [&] {
                const RuntimeEventOriginalScope original;
                return real_query == nullptr ? cudaErrorNotSupported : real_query(stream);
            });
        return completed ? *completed
                         : glimmer::interceptor::intercept_runtime_stream_query(stream, real_query);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaStreamDestroy(cudaStream_t stream) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    try {
        RuntimeCallScope scope(true);
        const RuntimeStreamDestroyFunction real_destroy = resolve_runtime_stream_destroy();
        if (is_reentrant || glimmer::interceptor::is_inside_driver_call()) {
            return real_destroy == nullptr ? cudaErrorNotSupported : real_destroy(stream);
        }
        if (glimmer::interceptor::cooperative_runtime_preview_requested()) {
            return glimmer::interceptor::intercept_cooperative_runtime_stream_destroy(stream, [&] {
                // Reuse the lexical SDK-original route exclusion only here;
                // the Runtime root remains visible and validates its caller.
                const RuntimeEventOriginalScope original_scope;
                return real_destroy == nullptr ? cudaErrorNotSupported : real_destroy(stream);
            });
        }
        return glimmer::interceptor::intercept_runtime_stream_destroy(stream, real_destroy);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaLaunchKernel(const void* function, dim3 grid_dim,
                                                  dim3 block_dim, void** arguments,
                                                  std::size_t shared_memory_bytes,
                                                  cudaStream_t stream) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call() ||
                              glimmer::interceptor::is_inside_driver_call();
    try {
        RuntimeCallScope scope(true);
        const RuntimeLaunchKernelFunction real_launch = resolve_runtime_launch_kernel();
        if (real_launch == nullptr) {
            return cudaErrorNotSupported;
        }
        return intercept_runtime_kernel_launch(
            "cudaLaunchKernel", is_reentrant, grid_dim, block_dim, shared_memory_bytes, stream,
            [real_launch, function, grid_dim, block_dim, arguments, shared_memory_bytes, stream,
             is_reentrant] {
                return invoke_runtime_persistent_or_native(function, grid_dim, block_dim, arguments,
                                                           shared_memory_bytes, stream, real_launch,
                                                           is_reentrant, false);
            });
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaLaunchKernelExC(const cudaLaunchConfig_t* config,
                                                     const void* function, void** arguments) {
    return intercept_runtime_kernel_launch_ex(config, function, arguments, false);
}

extern "C" cudaError_t CUDARTAPI cudaLaunchKernelExC_ptsz(const cudaLaunchConfig_t* config,
                                                          const void* function, void** arguments) {
    return intercept_runtime_kernel_launch_ex(config, function, arguments, true);
}

extern "C" cudaError_t CUDARTAPI cudaLaunchKernel_ptsz(const void* function, dim3 grid_dim,
                                                       dim3 block_dim, void** arguments,
                                                       std::size_t shared_memory_bytes,
                                                       cudaStream_t stream) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call() ||
                              glimmer::interceptor::is_inside_driver_call();
    try {
        RuntimeCallScope scope(true);
        const RuntimeLaunchKernelFunction real_launch = resolve_runtime_launch_kernel_ptsz();
        if (real_launch == nullptr) {
            return cudaErrorNotSupported;
        }
        return intercept_runtime_kernel_launch(
            "cudaLaunchKernel_ptsz", is_reentrant, grid_dim, block_dim, shared_memory_bytes, stream,
            [real_launch, function, grid_dim, block_dim, arguments, shared_memory_bytes, stream,
             is_reentrant] {
                return invoke_runtime_persistent_or_native(function, grid_dim, block_dim, arguments,
                                                           shared_memory_bytes, stream, real_launch,
                                                           is_reentrant, true);
            },
            true);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

// NOLINTBEGIN(bugprone-reserved-identifier, readability-identifier-naming): preserve CUDA compiler
// ABI names.
extern "C" cudaError_t CUDARTAPI __cudaLaunchKernel(cudaKernel_t kernel, dim3 grid_dim,
                                                    dim3 block_dim, void** arguments,
                                                    std::size_t shared_memory_bytes,
                                                    cudaStream_t stream) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call() ||
                              glimmer::interceptor::is_inside_driver_call();
    try {
        RuntimeCallScope scope(true);
        const RuntimeInternalLaunchKernelFunction real_launch =
            resolve_runtime_internal_launch_kernel();
        if (real_launch == nullptr) {
            return cudaErrorNotSupported;
        }
        return intercept_runtime_kernel_launch(
            "__cudaLaunchKernel", is_reentrant, grid_dim, block_dim, shared_memory_bytes, stream,
            [real_launch, kernel, grid_dim, block_dim, arguments, shared_memory_bytes, stream] {
                return real_launch(kernel, grid_dim, block_dim, arguments, shared_memory_bytes,
                                   stream);
            });
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI __cudaLaunchKernel_ptsz(cudaKernel_t kernel, dim3 grid_dim,
                                                         dim3 block_dim, void** arguments,
                                                         std::size_t shared_memory_bytes,
                                                         cudaStream_t stream) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call() ||
                              glimmer::interceptor::is_inside_driver_call();
    try {
        RuntimeCallScope scope(true);
        const RuntimeInternalLaunchKernelFunction real_launch =
            resolve_runtime_internal_launch_kernel_ptsz();
        if (real_launch == nullptr) {
            return cudaErrorNotSupported;
        }
        return intercept_runtime_kernel_launch(
            "__cudaLaunchKernel_ptsz", is_reentrant, grid_dim, block_dim, shared_memory_bytes,
            stream,
            [real_launch, kernel, grid_dim, block_dim, arguments, shared_memory_bytes, stream] {
                return real_launch(kernel, grid_dim, block_dim, arguments, shared_memory_bytes,
                                   stream);
            },
            true);
    } catch (...) {
        return cudaErrorUnknown;
    }
}
// NOLINTEND(bugprone-reserved-identifier, readability-identifier-naming)
extern "C" cudaError_t CUDARTAPI cudaMemcpy(void* destination, const void* source,
                                            std::size_t memory_bytes, cudaMemcpyKind kind) {
    try {
        RuntimeCallScope scope(true);
        using Function = cudaError_t (*)(void* destination, const void* source,
                                         std::size_t memory_bytes, cudaMemcpyKind kind);
        static const Function native = []() noexcept {
            const auto lookup = glimmer::interceptor::resolve_real_dlsym();
            return lookup == nullptr ? nullptr
                                     : reinterpret_cast<Function>(lookup(RTLD_NEXT, "cudaMemcpy"));
        }();
        return native == nullptr ? cudaErrorNotSupported
                                 : native(destination, source, memory_bytes, kind);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMemcpy_ptds(void* destination, const void* source,
                                                 std::size_t memory_bytes, cudaMemcpyKind kind) {
    try {
        RuntimeCallScope scope(true);
        using Function = cudaError_t (*)(void* destination, const void* source,
                                         std::size_t memory_bytes, cudaMemcpyKind kind);
        static const Function native = []() noexcept {
            const auto lookup = glimmer::interceptor::resolve_real_dlsym();
            return lookup == nullptr
                       ? nullptr
                       : reinterpret_cast<Function>(lookup(RTLD_NEXT, "cudaMemcpy_ptds"));
        }();
        return native == nullptr ? cudaErrorNotSupported
                                 : native(destination, source, memory_bytes, kind);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMemcpyAsync(void* destination, const void* source,
                                                 std::size_t memory_bytes, cudaMemcpyKind kind,
                                                 cudaStream_t stream) {
    try {
        RuntimeCallScope scope(true);
        using Function =
            cudaError_t (*)(void* destination, const void* source, std::size_t memory_bytes,
                            cudaMemcpyKind kind, cudaStream_t stream);
        static const Function native = []() noexcept {
            const auto lookup = glimmer::interceptor::resolve_real_dlsym();
            return lookup == nullptr
                       ? nullptr
                       : reinterpret_cast<Function>(lookup(RTLD_NEXT, "cudaMemcpyAsync"));
        }();
        return native == nullptr ? cudaErrorNotSupported
                                 : native(destination, source, memory_bytes, kind, stream);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMemcpyAsync_ptsz(void* destination, const void* source,
                                                      std::size_t memory_bytes, cudaMemcpyKind kind,
                                                      cudaStream_t stream) {
    try {
        RuntimeCallScope scope(true);
        using Function =
            cudaError_t (*)(void* destination, const void* source, std::size_t memory_bytes,
                            cudaMemcpyKind kind, cudaStream_t stream);
        static const Function native = []() noexcept {
            const auto lookup = glimmer::interceptor::resolve_real_dlsym();
            return lookup == nullptr
                       ? nullptr
                       : reinterpret_cast<Function>(lookup(RTLD_NEXT, "cudaMemcpyAsync_ptsz"));
        }();
        return native == nullptr ? cudaErrorNotSupported
                                 : native(destination, source, memory_bytes, kind, stream);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMemset(void* pointer, int value, std::size_t memory_bytes) {
    try {
        RuntimeCallScope scope(true);
        using Function = cudaError_t (*)(void* pointer, int value, std::size_t memory_bytes);
        static const Function native = []() noexcept {
            const auto lookup = glimmer::interceptor::resolve_real_dlsym();
            return lookup == nullptr ? nullptr
                                     : reinterpret_cast<Function>(lookup(RTLD_NEXT, "cudaMemset"));
        }();
        return native == nullptr ? cudaErrorNotSupported : native(pointer, value, memory_bytes);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMemset_ptds(void* pointer, int value,
                                                 std::size_t memory_bytes) {
    try {
        RuntimeCallScope scope(true);
        using Function = cudaError_t (*)(void* pointer, int value, std::size_t memory_bytes);
        static const Function native = []() noexcept {
            const auto lookup = glimmer::interceptor::resolve_real_dlsym();
            return lookup == nullptr
                       ? nullptr
                       : reinterpret_cast<Function>(lookup(RTLD_NEXT, "cudaMemset_ptds"));
        }();
        return native == nullptr ? cudaErrorNotSupported : native(pointer, value, memory_bytes);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMemsetAsync(void* pointer, int value, std::size_t memory_bytes,
                                                 cudaStream_t stream) {
    try {
        RuntimeCallScope scope(true);
        using Function = cudaError_t (*)(void* pointer, int value, std::size_t memory_bytes,
                                         cudaStream_t stream);
        static const Function native = []() noexcept {
            const auto lookup = glimmer::interceptor::resolve_real_dlsym();
            return lookup == nullptr
                       ? nullptr
                       : reinterpret_cast<Function>(lookup(RTLD_NEXT, "cudaMemsetAsync"));
        }();
        return native == nullptr ? cudaErrorNotSupported
                                 : native(pointer, value, memory_bytes, stream);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMemsetAsync_ptsz(void* pointer, int value,
                                                      std::size_t memory_bytes,
                                                      cudaStream_t stream) {
    try {
        RuntimeCallScope scope(true);
        using Function = cudaError_t (*)(void* pointer, int value, std::size_t memory_bytes,
                                         cudaStream_t stream);
        static const Function native = []() noexcept {
            const auto lookup = glimmer::interceptor::resolve_real_dlsym();
            return lookup == nullptr
                       ? nullptr
                       : reinterpret_cast<Function>(lookup(RTLD_NEXT, "cudaMemsetAsync_ptsz"));
        }();
        return native == nullptr ? cudaErrorNotSupported
                                 : native(pointer, value, memory_bytes, stream);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMallocHost(void** pointer, std::size_t memory_bytes) {
    try {
        RuntimeCallScope scope(true);
        using Function = cudaError_t (*)(void** pointer, std::size_t memory_bytes);
        static const Function native = []() noexcept {
            const auto lookup = glimmer::interceptor::resolve_real_dlsym();
            return lookup == nullptr
                       ? nullptr
                       : reinterpret_cast<Function>(lookup(RTLD_NEXT, "cudaMallocHost"));
        }();
        return native == nullptr ? cudaErrorNotSupported : native(pointer, memory_bytes);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaHostAlloc(void** pointer, std::size_t memory_bytes,
                                               unsigned int flags) {
    try {
        RuntimeCallScope scope(true);
        using Function =
            cudaError_t (*)(void** pointer, std::size_t memory_bytes, unsigned int flags);
        static const Function native = []() noexcept {
            const auto lookup = glimmer::interceptor::resolve_real_dlsym();
            return lookup == nullptr
                       ? nullptr
                       : reinterpret_cast<Function>(lookup(RTLD_NEXT, "cudaHostAlloc"));
        }();
        return native == nullptr ? cudaErrorNotSupported : native(pointer, memory_bytes, flags);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaFreeHost(void* pointer) {
    try {
        RuntimeCallScope scope(true);
        using Function = cudaError_t (*)(void* pointer);
        static const Function native = []() noexcept {
            const auto lookup = glimmer::interceptor::resolve_real_dlsym();
            return lookup == nullptr
                       ? nullptr
                       : reinterpret_cast<Function>(lookup(RTLD_NEXT, "cudaFreeHost"));
        }();
        return native == nullptr ? cudaErrorNotSupported : native(pointer);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaHostRegister(void* pointer, std::size_t memory_bytes,
                                                  unsigned int flags) {
    try {
        RuntimeCallScope scope(true);
        using Function =
            cudaError_t (*)(void* pointer, std::size_t memory_bytes, unsigned int flags);
        static const Function native = []() noexcept {
            const auto lookup = glimmer::interceptor::resolve_real_dlsym();
            return lookup == nullptr
                       ? nullptr
                       : reinterpret_cast<Function>(lookup(RTLD_NEXT, "cudaHostRegister"));
        }();
        return native == nullptr ? cudaErrorNotSupported : native(pointer, memory_bytes, flags);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaHostUnregister(void* pointer) {
    try {
        RuntimeCallScope scope(true);
        using Function = cudaError_t (*)(void* pointer);
        static const Function native = []() noexcept {
            const auto lookup = glimmer::interceptor::resolve_real_dlsym();
            return lookup == nullptr
                       ? nullptr
                       : reinterpret_cast<Function>(lookup(RTLD_NEXT, "cudaHostUnregister"));
        }();
        return native == nullptr ? cudaErrorNotSupported : native(pointer);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaHostGetDevicePointer(void** device_pointer, void* host_pointer,
                                                          unsigned int flags) {
    try {
        RuntimeCallScope scope(true);
        using Function =
            cudaError_t (*)(void** device_pointer, void* host_pointer, unsigned int flags);
        static const Function native = []() noexcept {
            const auto lookup = glimmer::interceptor::resolve_real_dlsym();
            return lookup == nullptr
                       ? nullptr
                       : reinterpret_cast<Function>(lookup(RTLD_NEXT, "cudaHostGetDevicePointer"));
        }();
        return native == nullptr ? cudaErrorNotSupported
                                 : native(device_pointer, host_pointer, flags);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaStreamCreate(cudaStream_t* stream) {
    try {
        RuntimeCallScope scope(true);
        using Function = cudaError_t (*)(cudaStream_t* stream);
        static const Function native = []() noexcept {
            const auto lookup = glimmer::interceptor::resolve_real_dlsym();
            return lookup == nullptr
                       ? nullptr
                       : reinterpret_cast<Function>(lookup(RTLD_NEXT, "cudaStreamCreate"));
        }();
        return native == nullptr ? cudaErrorNotSupported : native(stream);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaStreamCreateWithFlags(cudaStream_t* stream,
                                                           unsigned int flags) {
    try {
        RuntimeCallScope scope(true);
        using Function = cudaError_t (*)(cudaStream_t* stream, unsigned int flags);
        static const Function native = []() noexcept {
            const auto lookup = glimmer::interceptor::resolve_real_dlsym();
            return lookup == nullptr
                       ? nullptr
                       : reinterpret_cast<Function>(lookup(RTLD_NEXT, "cudaStreamCreateWithFlags"));
        }();
        return native == nullptr ? cudaErrorNotSupported : native(stream, flags);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaStreamCreateWithPriority(cudaStream_t* stream,
                                                              unsigned int flags, int priority) {
    try {
        RuntimeCallScope scope(true);
        using Function = cudaError_t (*)(cudaStream_t* stream, unsigned int flags, int priority);
        static const Function native = []() noexcept {
            const auto lookup = glimmer::interceptor::resolve_real_dlsym();
            return lookup == nullptr ? nullptr
                                     : reinterpret_cast<Function>(
                                           lookup(RTLD_NEXT, "cudaStreamCreateWithPriority"));
        }();
        return native == nullptr ? cudaErrorNotSupported : native(stream, flags, priority);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaStreamGetFlags(cudaStream_t stream, unsigned int* flags) {
    try {
        RuntimeCallScope scope(true);
        using Function = cudaError_t (*)(cudaStream_t stream, unsigned int* flags);
        static const Function native = []() noexcept {
            const auto lookup = glimmer::interceptor::resolve_real_dlsym();
            return lookup == nullptr
                       ? nullptr
                       : reinterpret_cast<Function>(lookup(RTLD_NEXT, "cudaStreamGetFlags"));
        }();
        return native == nullptr ? cudaErrorNotSupported : native(stream, flags);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaStreamGetPriority(cudaStream_t stream, int* priority) {
    try {
        RuntimeCallScope scope(true);
        using Function = cudaError_t (*)(cudaStream_t stream, int* priority);
        static const Function native = []() noexcept {
            const auto lookup = glimmer::interceptor::resolve_real_dlsym();
            return lookup == nullptr
                       ? nullptr
                       : reinterpret_cast<Function>(lookup(RTLD_NEXT, "cudaStreamGetPriority"));
        }();
        return native == nullptr ? cudaErrorNotSupported : native(stream, priority);
    } catch (...) {
        return cudaErrorUnknown;
    }
}
