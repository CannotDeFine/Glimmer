#pragma once

#include "ptx_transform.h"
#include "persistent_kernel.h"
#include "cuda_source_capture.h"

#include <cuda_runtime_api.h>
#include <cuda.h>

#include <cstddef>
#include <optional>
#include <functional>
#include <array>

namespace glimmer::interceptor {

// Diagnostic-only conventional Runtime routing. The first use pins selection;
// this does not authorize a launch or relax ordinary Driver-leaf permissions.
enum class RuntimeProfilingBoundaryMode { kDisabled, kEnabled, kInvalid };
[[nodiscard]] RuntimeProfilingBoundaryMode runtime_profiling_boundary_mode() noexcept;

// Select a base constructor only from the chosen original SDK's ELF version.
// Default lookup examines RTLD_NEXT, not our compile-time default alias.
[[nodiscard]] void* runtime_graph_constructor_lookup(void* handle) noexcept;
[[nodiscard]] void* runtime_graph_constructor_version(const char* version) noexcept;
[[nodiscard]] void* runtime_graph_update_lookup(void* handle) noexcept;
[[nodiscard]] void* runtime_graph_update_version(const char* version) noexcept;

using RuntimeCaptureDispatch = std::function<CUresult(
    CUcontext, CUstream, bool, CudaSourceCaptureKind, const std::function<CUresult()>&)>;
using RuntimeCaptureContextLookup = std::function<CUresult(CUstream, bool, CUcontext&)>;
[[nodiscard]] cudaError_t intercept_runtime_capture(
    CudaSourceCaptureKind kind, const std::function<cudaError_t()>& original) noexcept;
struct RuntimeSourceKernelShape {
    std::array<unsigned int, 3> grid;
    std::array<unsigned int, 3> block;
    std::size_t shared_memory_bytes{};
    bool is_ptds{};
    bool operator==(const RuntimeSourceKernelShape&) const = default;
};
// Lexical source permission: no Runtime handle conversion or deferred arguments.
[[nodiscard]] cudaError_t invoke_runtime_source_kernel_original(
    const RuntimeSourceKernelShape& shape, const std::function<cudaError_t()>& original,
    const std::function<void()>& quarantine) noexcept;
[[nodiscard]] std::optional<CUresult> dispatch_runtime_source_kernel(
    CUfunction function, const RuntimeSourceKernelShape& shape,
    const std::function<CUresult()>& execute) noexcept;
// Actual Driver leaf observation only; never constructed from Runtime handles.
struct RuntimeCaptureObservation {
    CUcontext context{};
    CUstream stream{};
    CUgraph graph{};
    bool is_ptds{};
    bool was_runtime_invoked{};
    unsigned int original_driver_calls{};
};
// Internal root callback assembly. Requires the admitted shared gate's native
// phase and lends one caller-bound callback only during real Runtime invocation.
// Runtime status stays separate from the CUresult owner signal; no enum casts.
[[nodiscard]] CUresult invoke_runtime_capture_original(
    const RuntimeCaptureDispatch& dispatch, const std::function<cudaError_t()>& original,
    RuntimeCaptureObservation& observation, cudaError_t& native_status,
    const RuntimeCaptureContextLookup& lookup = {}) noexcept;
// Missing permission returns no value, retaining ordinary wrapper behavior.
[[nodiscard]] std::optional<CUresult> dispatch_runtime_capture_origin(
    CUstream stream, bool is_ptds, CudaSourceCaptureKind kind,
    const std::function<CUresult()>& original, CUgraph* native_graph = nullptr) noexcept;

// Synchronous, caller-bound ownership adapter. Execution must guard original
// issue with its physically owned native Driver scope and may not retain it.
// Runtime errors remain distinct from the CUresult ownership signal.
using RuntimeGraphExecution = std::function<CUresult(const std::function<CUresult()>&)>;
[[nodiscard]] cudaError_t invoke_runtime_graph_original(
    const RuntimeGraphExecution& execution, const std::function<cudaError_t()>& original) noexcept;
// Context-wide completion only. Null ownership may decline before any effect.
// An accepted original is caller-bound, native-scoped and exactly once under
// the admitted process transition; actual Runtime enums remain authoritative.
using RuntimeContextCompletionExecution =
    std::function<std::optional<CUresult>(const std::function<CUresult()>&)>;
[[nodiscard]] std::optional<cudaError_t> invoke_runtime_context_completion_original(
    const RuntimeContextCompletionExecution& execution,
    const std::function<cudaError_t()>& original, const std::function<void()>& quarantine) noexcept;
// Registered executable and bound caller. Normal replay retains exact event
// dependencies/epochs; source replay currently requires event-free effects.
[[nodiscard]] cudaError_t execute_owned_runtime_graph(
    cudaGraphExec_t executable, cudaStream_t stream, bool is_ptds,
    const std::function<cudaError_t()>& original) noexcept;
// Stable constructor/destructor roots; original invocation owns Runtime scope.
// Construction stages only the write-only executable and observes SDK context.
[[nodiscard]] cudaError_t intercept_runtime_graph_instantiation(
    cudaGraphExec_t* output, cudaGraph_t graph, bool should_prepare_effects,
    const std::function<cudaError_t(cudaGraphExec_t*)>& original);
[[nodiscard]] cudaError_t intercept_runtime_graph_exec_destroy(
    cudaGraphExec_t executable, const std::function<cudaError_t()>& original);
[[nodiscard]] cudaError_t intercept_runtime_graph_event_update(
    cudaGraphExec_t executable, cudaGraphNode_t node, cudaEvent_t event, bool is_record,
    const std::function<cudaError_t()>& original);
[[nodiscard]] cudaError_t intercept_runtime_graph_node_update(
    cudaGraphExec_t executable, cudaGraphNode_t node, std::optional<CUgraphNodeType> expected_type,
    const std::function<cudaError_t()>& original);

[[nodiscard]] cudaError_t intercept_runtime_graph_update(
    cudaGraphExec_t executable, cudaGraph_t graph, std::optional<cudaGraphNode_t> child_node,
    const std::function<cudaError_t()>& original);

// Use Runtime's native query to preserve lazy context binding on new threads.
[[nodiscard]] std::optional<bool> runtime_launch_stream_is_capturing(
    cudaStream_t stream, bool per_thread_default_stream) noexcept;
[[nodiscard]] bool cooperative_runtime_preview_requested() noexcept;
[[nodiscard]] bool runtime_cooperative_driver_routing_is_enabled() noexcept;
[[nodiscard]] cudaError_t intercept_cooperative_runtime_native_boundary(
    const std::function<cudaError_t()>& operation);
[[nodiscard]] cudaError_t intercept_cooperative_runtime_memory_boundary(
    const std::function<cudaError_t()>& operation, const bool& has_native_issue);
// SDK-documented identical event type, exact source retirement and separate
// Runtime status. Original runs once under lexical Driver-routing exclusion.
[[nodiscard]] cudaError_t intercept_cooperative_runtime_event_destroy(
    cudaEvent_t event, const std::function<cudaError_t()>& original);
[[nodiscard]] cudaError_t intercept_cooperative_runtime_event_create(
    cudaEvent_t* event, unsigned int flags, const std::function<cudaError_t()>& original);
[[nodiscard]] std::optional<cudaError_t> intercept_normal_runtime_event_create(
    cudaEvent_t* event, unsigned int flags, const std::function<cudaError_t()>& original);
[[nodiscard]] cudaError_t intercept_cooperative_runtime_event_record(
    cudaEvent_t event, cudaStream_t stream, bool is_ptds, unsigned int flags,
    const std::function<cudaError_t()>& original);
// Normal root: metadata-only rows retire without CUDA; logical actors use an
// exact drained process boundary. No source transition permission is inferred.
[[nodiscard]] std::optional<cudaError_t> intercept_normal_runtime_event_destroy(
    cudaEvent_t event, const std::function<cudaError_t()>& original);
[[nodiscard]] std::optional<cudaError_t> intercept_normal_runtime_event_completion(
    cudaEvent_t event, bool should_wait, const std::function<cudaError_t()>& original);
// Read-only timing: check logical readiness before the SDK samples timestamps;
// stage its write-only output until native validation and both observations pass.
// Shared status/output contract; callbacks own readiness and original dispatch.
[[nodiscard]] cudaError_t measure_runtime_event_timing(
    float* milliseconds, cudaEvent_t start, cudaEvent_t end,
    const std::function<CUresult(CUevent)>& observe,
    const std::function<cudaError_t(float*)>& original);
[[nodiscard]] std::optional<cudaError_t> intercept_normal_runtime_event_elapsed_time(
    float* milliseconds, cudaEvent_t start, cudaEvent_t end,
    const std::function<cudaError_t(float*)>& original);
[[nodiscard]] std::optional<cudaError_t> intercept_normal_runtime_event_record(
    cudaEvent_t event, cudaStream_t stream, bool is_ptds,
    const std::function<cudaError_t()>& original);
[[nodiscard]] std::optional<cudaError_t> intercept_normal_runtime_event_wait(
    cudaStream_t stream, cudaEvent_t event, bool is_ptds, unsigned int flags,
    const std::function<cudaError_t()>& original);
[[nodiscard]] std::optional<cudaError_t> intercept_normal_runtime_stream_completion(
    cudaStream_t stream, bool is_ptds, bool should_wait,
    const std::function<cudaError_t()>& original);
[[nodiscard]] std::optional<cudaError_t> intercept_normal_runtime_device_completion(
    const std::function<cudaError_t()>& original,
    const RuntimeContextCompletionExecution& epoch_completion = {});
// Source uses exact retained pause; normal uses the drained process boundary.
// Caller-thread SDK effect/context proof, without cached Driver callback reliance.
[[nodiscard]] cudaError_t intercept_cooperative_runtime_stream_destroy(
    cudaStream_t stream, const std::function<cudaError_t()>& original);
void mark_runtime_source_memory_issue() noexcept;

using RuntimeMallocFunction = cudaError_t (*)(void** device_pointer, std::size_t memory_bytes);
using RuntimeMallocManagedFunction = cudaError_t (*)(void** device_pointer,
                                                     std::size_t memory_bytes, unsigned int flags);
using RuntimeMallocPitchFunction = cudaError_t (*)(void** device_pointer, std::size_t* pitch,
                                                   std::size_t width_bytes, std::size_t height);
using RuntimeMalloc3DFunction = cudaError_t (*)(struct cudaPitchedPtr* pitched_device_pointer,
                                                struct cudaExtent extent);
using RuntimeGetDeviceFunction = cudaError_t (*)(int* device);
using RuntimeFreeFunction = cudaError_t (*)(void* device_pointer);
using RuntimeIpcGetMemHandleFunction = cudaError_t (*)(cudaIpcMemHandle_t* handle,
                                                       void* device_pointer);
using RuntimeIpcOpenMemHandleFunction = cudaError_t (*)(void** device_pointer,
                                                        cudaIpcMemHandle_t handle,
                                                        unsigned int flags);
using RuntimeIpcCloseMemHandleFunction = cudaError_t (*)(void* device_pointer);
using RuntimeImportExternalMemoryFunction = cudaError_t (*)(
    cudaExternalMemory_t* external_memory, const struct cudaExternalMemoryHandleDesc* handle_desc);
using RuntimeExternalMemoryGetMappedBufferFunction =
    cudaError_t (*)(void** device_pointer, cudaExternalMemory_t external_memory,
                    const struct cudaExternalMemoryBufferDesc* buffer_desc);
using RuntimeExternalMemoryGetMappedMipmappedArrayFunction =
    cudaError_t (*)(cudaMipmappedArray_t* mipmap, cudaExternalMemory_t external_memory,
                    const struct cudaExternalMemoryMipmappedArrayDesc* mipmap_desc);
using RuntimeDestroyExternalMemoryFunction = cudaError_t (*)(cudaExternalMemory_t external_memory);
using RuntimeMallocArrayFunction = cudaError_t (*)(cudaArray_t* array,
                                                   const struct cudaChannelFormatDesc* descriptor,
                                                   std::size_t width, std::size_t height,
                                                   unsigned int flags);
using RuntimeMalloc3DArrayFunction = cudaError_t (*)(cudaArray_t* array,
                                                     const struct cudaChannelFormatDesc* descriptor,
                                                     struct cudaExtent extent, unsigned int flags);
using RuntimeMallocMipmappedArrayFunction =
    cudaError_t (*)(cudaMipmappedArray_t* mipmap, const struct cudaChannelFormatDesc* descriptor,
                    struct cudaExtent extent, unsigned int level_count, unsigned int flags);
using RuntimeFreeArrayFunction = cudaError_t (*)(cudaArray_t array);
using RuntimeFreeMipmappedArrayFunction = cudaError_t (*)(cudaMipmappedArray_t mipmap);
using RuntimeGraphicsUnregisterResourceFunction = cudaError_t (*)(cudaGraphicsResource_t resource);
using RuntimeGraphicsResourceSetMapFlagsFunction = cudaError_t (*)(cudaGraphicsResource_t resource,
                                                                   unsigned int flags);
using RuntimeGraphicsMapResourcesFunction = cudaError_t (*)(int count,
                                                            cudaGraphicsResource_t* resources,
                                                            cudaStream_t stream);
using RuntimeGraphicsUnmapResourcesFunction = cudaError_t (*)(int count,
                                                              cudaGraphicsResource_t* resources,
                                                              cudaStream_t stream);
using RuntimeGraphicsResourceGetMappedPointerFunction =
    cudaError_t (*)(void** device_pointer, std::size_t* size, cudaGraphicsResource_t resource);
using RuntimeGraphicsSubResourceGetMappedArrayFunction =
    cudaError_t (*)(cudaArray_t* array, cudaGraphicsResource_t resource, unsigned int array_index,
                    unsigned int mip_level);
using RuntimeGraphicsResourceGetMappedMipmappedArrayFunction =
    cudaError_t (*)(cudaMipmappedArray_t* mipmap, cudaGraphicsResource_t resource);
using RuntimeGraphAddMemAllocNodeFunction = cudaError_t (*)(
    cudaGraphNode_t* graph_node, cudaGraph_t graph, const cudaGraphNode_t* dependencies,
    std::size_t dependency_count, struct cudaMemAllocNodeParams* parameters);
using RuntimeGraphLaunchFunction = cudaError_t (*)(cudaGraphExec_t graph_exec, cudaStream_t stream);
using RuntimeMemGetInfoFunction = cudaError_t (*)(std::size_t* free_bytes,
                                                  std::size_t* total_bytes);
using RuntimeMallocAsyncFunction = cudaError_t (*)(void** device_pointer, std::size_t memory_bytes,
                                                   cudaStream_t stream);
using RuntimeMallocFromPoolAsyncFunction = cudaError_t (*)(void** device_pointer,
                                                           std::size_t memory_bytes,
                                                           cudaMemPool_t pool, cudaStream_t stream);
using RuntimeFreeAsyncFunction = cudaError_t (*)(void* device_pointer, cudaStream_t stream);
using RuntimeDeviceSynchronizeFunction = cudaError_t (*)();
using RuntimeDeviceResetFunction = cudaError_t (*)();
using RuntimeFuncGetNameFunction = cudaError_t (*)(const char**, const void*);
using RuntimeGetFuncBySymbolFunction = cudaError_t (*)(cudaFunction_t*, const void*);

struct RuntimeCooperativeLaunch {
    CUfunction original = nullptr;
    std::optional<CooperativeKernelVariant> variant;
    std::array<unsigned int, 3> grid{};
    std::array<unsigned int, 3> block{};
    std::optional<MappedBinaryKernelVariant> binary_variant{};
    std::optional<MappedGemmKernelVariant> gemm_variant{};
    CUcontext context{};
    std::size_t shared_memory_bytes{};
};
// Borrowed normal permission on the shared process Call. SDK Driver leaves
// supply typed streams; neither Runtime handles nor arguments are converted.
using RuntimeNativeKernelExecution =
    std::function<CUresult(CUfunction, CUstream, bool, const std::function<CUresult()>&)>;
[[nodiscard]] cudaError_t invoke_runtime_cooperative_launch_original(
    RuntimeCooperativeLaunch plan, bool is_ptds, const std::function<cudaError_t()>& original,
    const std::function<void()>& quarantine,
    const RuntimeNativeKernelExecution& native_execution = {}) noexcept;
void quarantine_runtime_cooperative_launch(CUcontext context) noexcept;
[[nodiscard]] bool runtime_native_kernel_execution_is_pending() noexcept;
[[nodiscard]] std::optional<CUresult> dispatch_runtime_native_kernel(
    CUfunction function, CUstream stream, const RuntimeSourceKernelShape& shape,
    const std::function<CUresult()>& original) noexcept;
[[nodiscard]] std::optional<RuntimeCooperativeLaunch> prepare_runtime_cooperative_launch(
    const void* function, dim3 grid, dim3 block, std::size_t shared_memory_bytes,
    RuntimeFuncGetNameFunction get_name, RuntimeGetFuncBySymbolFunction get_function);
[[nodiscard]] bool runtime_cooperative_launch_is_pending() noexcept;
[[nodiscard]] std::optional<RuntimeCooperativeLaunch> consume_runtime_cooperative_launch(
    CUfunction function, const RuntimeSourceKernelShape& shape) noexcept;
void complete_runtime_cooperative_launch(CUresult status) noexcept;

struct RuntimePersistentLaunch {
    CUfunction original = nullptr;
    CUfunction persistent = nullptr;
    unsigned int worker_blocks = 0;
    std::size_t parameter_count = 0;
    PtxExecutionKind execution_kind = PtxExecutionKind::kStatic;
    unsigned int logical_blocks = 0;
};

[[nodiscard]] std::optional<RuntimePersistentLaunch> prepare_runtime_persistent_launch(
    const void* function, dim3 grid, dim3 block, std::size_t shared_memory_bytes,
    cudaStream_t stream, bool per_thread_default_stream, RuntimeFuncGetNameFunction get_name,
    RuntimeGetFuncBySymbolFunction get_function);
struct RuntimeDriverReplacement {
    CUfunction persistent = nullptr;
    unsigned int worker_blocks = 0;
    void** arguments = nullptr;
};
[[nodiscard]] std::optional<RuntimeDriverReplacement> consume_runtime_driver_replacement(
    CUfunction function) noexcept;
void report_runtime_persistent_launch(unsigned int logical_blocks,
                                      unsigned int worker_blocks) noexcept;
[[nodiscard]] cudaError_t intercept_runtime_device_reset(RuntimeDeviceResetFunction reset);
using RuntimeStreamSynchronizeFunction = cudaError_t (*)(cudaStream_t stream);
using RuntimeStreamQueryFunction = cudaError_t (*)(cudaStream_t stream);
using RuntimeStreamDestroyFunction = cudaError_t (*)(cudaStream_t stream);
using RuntimeDeviceGetDefaultMemPoolFunction = cudaError_t (*)(cudaMemPool_t* pool, int device);
using RuntimeDeviceSetMemPoolFunction = cudaError_t (*)(int device, cudaMemPool_t pool);
using RuntimeDeviceGetMemPoolFunction = cudaError_t (*)(cudaMemPool_t* pool, int device);
using RuntimeMemPoolTrimToFunction = cudaError_t (*)(cudaMemPool_t pool,
                                                     std::size_t min_bytes_to_keep);
using RuntimeMemPoolSetAttributeFunction = cudaError_t (*)(cudaMemPool_t pool,
                                                           cudaMemPoolAttr attribute, void* value);
using RuntimeMemPoolGetAttributeFunction = cudaError_t (*)(cudaMemPool_t pool,
                                                           cudaMemPoolAttr attribute, void* value);
using RuntimeMemPoolSetAccessFunction = cudaError_t (*)(cudaMemPool_t pool,
                                                        const cudaMemAccessDesc* descriptors,
                                                        std::size_t descriptor_count);
using RuntimeMemPoolGetAccessFunction = cudaError_t (*)(cudaMemAccessFlags* flags,
                                                        cudaMemPool_t pool,
                                                        cudaMemLocation* location);
using RuntimeMemPoolCreateFunction = cudaError_t (*)(cudaMemPool_t* pool,
                                                     const cudaMemPoolProps* properties);
using RuntimeMemPoolDestroyFunction = cudaError_t (*)(cudaMemPool_t pool);
using RuntimeMemGetDefaultMemPoolFunction = cudaError_t (*)(cudaMemPool_t* pool,
                                                            cudaMemLocation* location,
                                                            cudaMemAllocationType allocation_type);
using RuntimeMemGetMemPoolFunction = cudaError_t (*)(cudaMemPool_t* pool, cudaMemLocation* location,
                                                     cudaMemAllocationType allocation_type);
using RuntimeMemSetMemPoolFunction = cudaError_t (*)(cudaMemLocation* location,
                                                     cudaMemAllocationType allocation_type,
                                                     cudaMemPool_t pool);
using RuntimeMemPoolExportToShareableHandleFunction =
    cudaError_t (*)(void* handle_out, cudaMemPool_t pool,
                    enum cudaMemAllocationHandleType handle_type, unsigned int flags);
using RuntimeMemPoolImportFromShareableHandleFunction =
    cudaError_t (*)(cudaMemPool_t* pool_out, void* handle,
                    enum cudaMemAllocationHandleType handle_type, unsigned int flags);
using RuntimeMemPoolExportPointerFunction =
    cudaError_t (*)(cudaMemPoolPtrExportData* share_data_out, void* device_pointer);
using RuntimeMemPoolImportPointerFunction = cudaError_t (*)(void** pointer_out, cudaMemPool_t pool,
                                                            cudaMemPoolPtrExportData* share_data);

[[nodiscard]] bool is_inside_runtime_call() noexcept;
[[nodiscard]] bool runtime_persistent_lookup_is_enabled() noexcept;
[[nodiscard]] bool initialize_runtime_driver_lookup();
[[nodiscard]] CUresult runtime_driver_get_proc_address(const char* symbol, void** address,
                                                       int version, cuuint64_t flags);
[[nodiscard]] CUresult runtime_driver_get_proc_address_v2(const char* symbol, void** address,
                                                          int version, cuuint64_t flags,
                                                          CUdriverProcAddressQueryResult* status);

[[nodiscard]] cudaError_t intercept_runtime_malloc(void** device_pointer, std::size_t memory_bytes,
                                                   RuntimeMallocFunction allocate,
                                                   RuntimeFreeFunction release,
                                                   RuntimeGetDeviceFunction get_device);
[[nodiscard]] cudaError_t intercept_runtime_malloc_managed(void** device_pointer,
                                                           std::size_t memory_bytes,
                                                           unsigned int flags,
                                                           RuntimeMallocManagedFunction allocate,
                                                           RuntimeFreeFunction release,
                                                           RuntimeGetDeviceFunction get_device);
[[nodiscard]] cudaError_t intercept_runtime_malloc_pitch(void** device_pointer, std::size_t* pitch,
                                                         std::size_t width_bytes,
                                                         std::size_t height,
                                                         RuntimeMallocPitchFunction allocate,
                                                         RuntimeFreeFunction release,
                                                         RuntimeGetDeviceFunction get_device);
[[nodiscard]] cudaError_t intercept_runtime_malloc_3d(struct cudaPitchedPtr* pitched_device_pointer,
                                                      struct cudaExtent extent,
                                                      RuntimeMalloc3DFunction allocate,
                                                      RuntimeFreeFunction release,
                                                      RuntimeGetDeviceFunction get_device);
[[nodiscard]] cudaError_t intercept_runtime_free(void* device_pointer, RuntimeFreeFunction release);
[[nodiscard]] cudaError_t intercept_runtime_ipc_get_mem_handle(
    cudaIpcMemHandle_t* handle, void* device_pointer, RuntimeIpcGetMemHandleFunction get_handle);
[[nodiscard]] cudaError_t intercept_runtime_ipc_open_mem_handle(
    void** device_pointer, cudaIpcMemHandle_t handle, unsigned int flags,
    RuntimeIpcOpenMemHandleFunction open_handle);
[[nodiscard]] cudaError_t intercept_runtime_ipc_close_mem_handle(
    void* device_pointer, RuntimeIpcCloseMemHandleFunction close_handle);
[[nodiscard]] cudaError_t intercept_runtime_import_external_memory(
    cudaExternalMemory_t* external_memory, const struct cudaExternalMemoryHandleDesc* handle_desc,
    RuntimeImportExternalMemoryFunction import_memory);
[[nodiscard]] cudaError_t intercept_runtime_external_memory_get_mapped_buffer(
    void** device_pointer, cudaExternalMemory_t external_memory,
    const struct cudaExternalMemoryBufferDesc* buffer_desc,
    RuntimeExternalMemoryGetMappedBufferFunction get_buffer);
[[nodiscard]] cudaError_t intercept_runtime_external_memory_get_mapped_mipmapped_array(
    cudaMipmappedArray_t* mipmap, cudaExternalMemory_t external_memory,
    const struct cudaExternalMemoryMipmappedArrayDesc* mipmap_desc,
    RuntimeExternalMemoryGetMappedMipmappedArrayFunction get_mipmap);
[[nodiscard]] cudaError_t intercept_runtime_destroy_external_memory(
    cudaExternalMemory_t external_memory, RuntimeDestroyExternalMemoryFunction destroy_memory);
[[nodiscard]] cudaError_t intercept_runtime_malloc_array(
    cudaArray_t* array, const struct cudaChannelFormatDesc* descriptor, std::size_t width,
    std::size_t height, unsigned int flags, RuntimeMallocArrayFunction allocate);
[[nodiscard]] cudaError_t intercept_runtime_malloc_3d_array(
    cudaArray_t* array, const struct cudaChannelFormatDesc* descriptor, struct cudaExtent extent,
    unsigned int flags, RuntimeMalloc3DArrayFunction allocate);
[[nodiscard]] cudaError_t intercept_runtime_malloc_mipmapped_array(
    cudaMipmappedArray_t* mipmap, const struct cudaChannelFormatDesc* descriptor,
    struct cudaExtent extent, unsigned int level_count, unsigned int flags,
    RuntimeMallocMipmappedArrayFunction allocate);
[[nodiscard]] cudaError_t intercept_runtime_free_array(cudaArray_t array,
                                                       RuntimeFreeArrayFunction release);
[[nodiscard]] cudaError_t intercept_runtime_free_mipmapped_array(
    cudaMipmappedArray_t mipmap, RuntimeFreeMipmappedArrayFunction release);
[[nodiscard]] cudaError_t intercept_runtime_graphics_unregister_resource(
    cudaGraphicsResource_t resource, RuntimeGraphicsUnregisterResourceFunction unregister);
[[nodiscard]] cudaError_t intercept_runtime_graphics_resource_set_map_flags(
    cudaGraphicsResource_t resource, unsigned int flags,
    RuntimeGraphicsResourceSetMapFlagsFunction set_flags);
[[nodiscard]] cudaError_t intercept_runtime_graphics_map_resources(
    int count, cudaGraphicsResource_t* resources, cudaStream_t stream,
    RuntimeGraphicsMapResourcesFunction map_resources);
[[nodiscard]] cudaError_t intercept_runtime_graphics_unmap_resources(
    int count, cudaGraphicsResource_t* resources, cudaStream_t stream,
    RuntimeGraphicsUnmapResourcesFunction unmap_resources);
[[nodiscard]] cudaError_t intercept_runtime_graphics_resource_get_mapped_pointer(
    void** device_pointer, std::size_t* size, cudaGraphicsResource_t resource,
    RuntimeGraphicsResourceGetMappedPointerFunction get_pointer);
[[nodiscard]] cudaError_t intercept_runtime_graphics_subresource_get_mapped_array(
    cudaArray_t* array, cudaGraphicsResource_t resource, unsigned int array_index,
    unsigned int mip_level, RuntimeGraphicsSubResourceGetMappedArrayFunction get_array);
[[nodiscard]] cudaError_t intercept_runtime_graphics_resource_get_mapped_mipmapped_array(
    cudaMipmappedArray_t* mipmap, cudaGraphicsResource_t resource,
    RuntimeGraphicsResourceGetMappedMipmappedArrayFunction get_mipmap);
[[nodiscard]] cudaError_t intercept_runtime_graph_add_mem_alloc_node(
    cudaGraphNode_t* graph_node, cudaGraph_t graph, const cudaGraphNode_t* dependencies,
    std::size_t dependency_count, struct cudaMemAllocNodeParams* parameters,
    RuntimeGraphAddMemAllocNodeFunction add_node);
[[nodiscard]] cudaError_t intercept_runtime_graph_launch(const char* api_name,
                                                         cudaGraphExec_t graph_exec,
                                                         cudaStream_t stream,
                                                         RuntimeGraphLaunchFunction launch,
                                                         bool per_thread_default_stream = false);
[[nodiscard]] cudaError_t intercept_runtime_mem_get_info(std::size_t* free_bytes,
                                                         std::size_t* total_bytes,
                                                         RuntimeMemGetInfoFunction query);
[[nodiscard]] cudaError_t intercept_runtime_malloc_async(
    void** device_pointer, std::size_t memory_bytes, cudaStream_t stream,
    RuntimeMallocAsyncFunction allocate, RuntimeFreeAsyncFunction release,
    RuntimeDeviceSynchronizeFunction synchronize, RuntimeGetDeviceFunction get_device);
[[nodiscard]] cudaError_t intercept_runtime_malloc_from_pool_async(
    void** device_pointer, std::size_t memory_bytes, cudaMemPool_t pool, cudaStream_t stream,
    RuntimeMallocFromPoolAsyncFunction allocate, RuntimeFreeAsyncFunction release,
    RuntimeDeviceSynchronizeFunction synchronize, RuntimeGetDeviceFunction get_device);
[[nodiscard]] cudaError_t intercept_runtime_mem_pool_import_from_shareable_handle(
    cudaMemPool_t* pool_out, void* handle, cudaMemAllocationHandleType handle_type,
    unsigned long long flags, RuntimeMemPoolImportFromShareableHandleFunction import_pool);
[[nodiscard]] cudaError_t intercept_runtime_mem_pool_import_pointer(
    void** pointer_out, cudaMemPool_t pool, cudaMemPoolPtrExportData* share_data,
    RuntimeMemPoolImportPointerFunction import_pointer);
[[nodiscard]] cudaError_t intercept_runtime_free_async(void* device_pointer, cudaStream_t stream,
                                                       RuntimeFreeAsyncFunction release);
[[nodiscard]] cudaError_t intercept_runtime_device_synchronize(
    RuntimeDeviceSynchronizeFunction synchronize);
[[nodiscard]] cudaError_t intercept_runtime_stream_synchronize(
    cudaStream_t stream, RuntimeStreamSynchronizeFunction synchronize);
[[nodiscard]] cudaError_t intercept_runtime_stream_query(cudaStream_t stream,
                                                         RuntimeStreamQueryFunction query);
[[nodiscard]] cudaError_t intercept_runtime_stream_destroy(cudaStream_t stream,
                                                           RuntimeStreamDestroyFunction destroy);

}  // namespace glimmer::interceptor
