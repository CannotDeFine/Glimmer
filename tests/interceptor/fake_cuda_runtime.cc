#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <cuda.h>
#include "event_audit_fixture.h"
#include <cstdlib>

#ifdef cuEventDestroy
#undef cuEventDestroy
#endif

#ifdef cuEventRecord
#undef cuEventRecord
#endif

#ifdef cuEventRecordWithFlags
#undef cuEventRecordWithFlags
#endif

#ifdef cuStreamWaitEvent
#undef cuStreamWaitEvent
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
#include <cuda_runtime_api.h>

#include <dlfcn.h>

#include <cstddef>
#include <atomic>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>

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
glimmer::tests::EventAudit g_event_audit;

using DriverAllocFunction = CUresult (*)(CUdeviceptr* device_pointer, std::size_t memory_bytes);
using DriverManagedAllocFunction = CUresult (*)(CUdeviceptr* device_pointer,
                                                std::size_t memory_bytes, unsigned int flags);
using DriverPitchAllocFunction = CUresult (*)(CUdeviceptr* device_pointer, std::size_t* pitch,
                                              std::size_t width_bytes, std::size_t height,
                                              unsigned int element_size_bytes);
using DriverAllocAsyncFunction = CUresult (*)(CUdeviceptr* device_pointer, std::size_t memory_bytes,
                                              CUstream stream);
using DriverPoolAllocFunction = CUresult (*)(CUdeviceptr* device_pointer, std::size_t memory_bytes,
                                             CUmemoryPool pool, CUstream stream);
using DriverFreeFunction = CUresult (*)(CUdeviceptr device_pointer);
using DriverFreeAsyncFunction = CUresult (*)(CUdeviceptr device_pointer, CUstream stream);
using DriverContextSynchronizeFunction = CUresult (*)();
using DriverStreamSynchronizeFunction = CUresult (*)(CUstream stream);
using DriverStreamQueryFunction = CUresult (*)(CUstream stream);
using DriverStreamDestroyFunction = CUresult (*)(CUstream stream);
using DriverMemGetInfoFunction = CUresult (*)(std::size_t* free_bytes, std::size_t* total_bytes);

std::uint8_t g_pool_token = 0;
std::uint8_t g_external_memory_token = 0;
std::uint8_t g_array_token = 0;
std::uint8_t g_graphics_resource_token = 0;

cudaMemPool_t fake_pool() noexcept {
    return reinterpret_cast<cudaMemPool_t>(&g_pool_token);
}

template <typename Function>
Function resolve_driver_function(const char* name) {
    return reinterpret_cast<Function>(dlsym(RTLD_DEFAULT, name));
}

void* to_runtime_pointer(CUdeviceptr device_pointer) noexcept {
    void* runtime_pointer = nullptr;
    static_assert(sizeof(runtime_pointer) == sizeof(device_pointer));
    std::memcpy(reinterpret_cast<void*>(&runtime_pointer),
                reinterpret_cast<const void*>(&device_pointer), sizeof(runtime_pointer));
    return runtime_pointer;
}

CUdeviceptr to_device_pointer(void* runtime_pointer) noexcept {
    CUdeviceptr device_pointer = 0;
    static_assert(sizeof(runtime_pointer) == sizeof(device_pointer));
    std::memcpy(reinterpret_cast<void*>(&device_pointer),
                reinterpret_cast<const void*>(&runtime_pointer), sizeof(device_pointer));
    return device_pointer;
}

}  // namespace

extern "C" glimmer::tests::EventAudit* glimmer_fake_event_audit() {
    return &g_event_audit;
}

extern "C" cudaError_t CUDARTAPI cudaEventCreate(cudaEvent_t* event) {
    glimmer::tests::record_event_call(g_event_audit, glimmer::tests::EventCall::kCreate, event,
                                      nullptr, nullptr, nullptr, 0);
    return static_cast<cudaError_t>(g_event_audit.result);
}

extern "C" cudaError_t CUDARTAPI cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags) {
    glimmer::tests::record_event_call(g_event_audit, glimmer::tests::EventCall::kCreateWithFlags,
                                      event, nullptr, nullptr, nullptr, flags);
    return static_cast<cudaError_t>(g_event_audit.result);
}

extern "C" cudaError_t CUDARTAPI cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    glimmer::tests::record_event_call(g_event_audit, glimmer::tests::EventCall::kRecord, nullptr,
                                      event, stream, nullptr, 0);
    return static_cast<cudaError_t>(g_event_audit.result);
}

#ifndef GLIMMER_FAKE_NO_EVENT_PTDS
extern "C" cudaError_t CUDARTAPI cudaEventRecord_ptsz(cudaEvent_t event, cudaStream_t stream) {
    glimmer::tests::record_event_call(g_event_audit, glimmer::tests::EventCall::kRecordPtds,
                                      nullptr, event, stream, nullptr, 0);
    return static_cast<cudaError_t>(g_event_audit.result);
}
#endif

extern "C" cudaError_t CUDARTAPI cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream,
                                                          unsigned int flags) {
    glimmer::tests::record_event_call(g_event_audit, glimmer::tests::EventCall::kRecordFlags,
                                      nullptr, event, stream, nullptr, flags);
    return static_cast<cudaError_t>(g_event_audit.result);
}

#ifndef GLIMMER_FAKE_NO_EVENT_PTDS
extern "C" cudaError_t CUDARTAPI cudaEventRecordWithFlags_ptsz(cudaEvent_t event,
                                                               cudaStream_t stream,
                                                               unsigned int flags) {
    glimmer::tests::record_event_call(g_event_audit, glimmer::tests::EventCall::kRecordFlagsPtds,
                                      nullptr, event, stream, nullptr, flags);
    return static_cast<cudaError_t>(g_event_audit.result);
}
#endif

extern "C" cudaError_t CUDARTAPI cudaEventQuery(cudaEvent_t event) {
    glimmer::tests::record_event_call(g_event_audit, glimmer::tests::EventCall::kQuery, nullptr,
                                      event, nullptr, nullptr, 0);
    return static_cast<cudaError_t>(g_event_audit.result);
}

extern "C" cudaError_t CUDARTAPI cudaEventSynchronize(cudaEvent_t event) {
    glimmer::tests::record_event_call(g_event_audit, glimmer::tests::EventCall::kSynchronize,
                                      nullptr, event, nullptr, nullptr, 0);
    return static_cast<cudaError_t>(g_event_audit.result);
}

extern "C" cudaError_t CUDARTAPI cudaEventDestroy(cudaEvent_t event) {
    glimmer::tests::record_event_call(g_event_audit, glimmer::tests::EventCall::kDestroy, nullptr,
                                      event, nullptr, nullptr, 0);
    return static_cast<cudaError_t>(g_event_audit.result);
}

extern "C" cudaError_t CUDARTAPI cudaEventElapsedTime(float* milliseconds, cudaEvent_t start,
                                                      cudaEvent_t end) {
    glimmer::tests::record_event_call(g_event_audit, glimmer::tests::EventCall::kElapsed,
                                      milliseconds, start, nullptr, end, 0);
    return static_cast<cudaError_t>(g_event_audit.result);
}

extern "C" cudaError_t CUDARTAPI cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event,
                                                     unsigned int flags) {
    glimmer::tests::record_event_call(g_event_audit, glimmer::tests::EventCall::kWait, nullptr,
                                      event, stream, nullptr, flags);
    return static_cast<cudaError_t>(g_event_audit.result);
}

#ifndef GLIMMER_FAKE_NO_EVENT_PTDS
extern "C" cudaError_t CUDARTAPI cudaStreamWaitEvent_ptsz(cudaStream_t stream, cudaEvent_t event,
                                                          unsigned int flags) {
    glimmer::tests::record_event_call(g_event_audit, glimmer::tests::EventCall::kWaitPtds, nullptr,
                                      event, stream, nullptr, flags);
    return static_cast<cudaError_t>(g_event_audit.result);
}
#endif

extern "C" cudaError_t CUDARTAPI cudaMalloc(void** device_pointer, std::size_t memory_bytes) {
    if (device_pointer == nullptr) {
        return cudaErrorInvalidValue;
    }
    const DriverAllocFunction allocate =
        resolve_driver_function<DriverAllocFunction>("cuMemAlloc_v2");
    if (allocate == nullptr) {
        return cudaErrorNotSupported;
    }

    CUdeviceptr driver_pointer = 0;
    const CUresult result = allocate(&driver_pointer, memory_bytes);
    if (result != CUDA_SUCCESS) {
        return cudaErrorMemoryAllocation;
    }
    *device_pointer = to_runtime_pointer(driver_pointer);
    return cudaSuccess;
}

extern "C" bool glimmer_fake_runtime_capacity_available() {
    return true;
}

extern "C" cudaError_t CUDARTAPI cudaMallocManaged(void** device_pointer, std::size_t memory_bytes,
                                                   unsigned int flags) {
    if (device_pointer == nullptr) {
        return cudaErrorInvalidValue;
    }
    const DriverManagedAllocFunction allocate =
        resolve_driver_function<DriverManagedAllocFunction>("cuMemAllocManaged");
    if (allocate == nullptr) {
        return cudaErrorNotSupported;
    }

    CUdeviceptr driver_pointer = 0;
    const CUresult result = allocate(&driver_pointer, memory_bytes, flags);
    if (result != CUDA_SUCCESS) {
        return cudaErrorMemoryAllocation;
    }
    *device_pointer = to_runtime_pointer(driver_pointer);
    return cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaIpcGetMemHandle(cudaIpcMemHandle_t* handle,
                                                     void* device_pointer) {
    if (handle == nullptr || device_pointer == nullptr) {
        return cudaErrorInvalidValue;
    }
    *handle = {};
    return cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaIpcOpenMemHandle(void** device_pointer, cudaIpcMemHandle_t,
                                                      unsigned int) {
    if (device_pointer == nullptr) {
        return cudaErrorInvalidValue;
    }
    *device_pointer = reinterpret_cast<void*>(0x71000000U);
    return cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaIpcCloseMemHandle(void* device_pointer) {
    return device_pointer == nullptr ? cudaErrorInvalidValue : cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaImportExternalMemory(
    cudaExternalMemory_t* external_memory, const struct cudaExternalMemoryHandleDesc* handle_desc) {
    if (external_memory == nullptr || handle_desc == nullptr) {
        return cudaErrorInvalidValue;
    }
    *external_memory = reinterpret_cast<cudaExternalMemory_t>(&g_external_memory_token);
    return cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI
cudaExternalMemoryGetMappedBuffer(void** device_pointer, cudaExternalMemory_t external_memory,
                                  const struct cudaExternalMemoryBufferDesc* buffer_desc) {
    if (device_pointer == nullptr || external_memory == nullptr || buffer_desc == nullptr) {
        return cudaErrorInvalidValue;
    }
    *device_pointer = reinterpret_cast<void*>(0x72000000U);
    return cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaExternalMemoryGetMappedMipmappedArray(
    cudaMipmappedArray_t* mipmap, cudaExternalMemory_t external_memory,
    const struct cudaExternalMemoryMipmappedArrayDesc* mipmap_desc) {
    if (mipmap == nullptr || external_memory == nullptr || mipmap_desc == nullptr) {
        return cudaErrorInvalidValue;
    }
    *mipmap = reinterpret_cast<cudaMipmappedArray_t>(&g_external_memory_token);
    return cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaDestroyExternalMemory(cudaExternalMemory_t external_memory) {
    return external_memory == nullptr ? cudaErrorInvalidResourceHandle : cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaMallocArray(cudaArray_t* array,
                                                 const struct cudaChannelFormatDesc* descriptor,
                                                 std::size_t width, std::size_t height,
                                                 unsigned int) {
    if (array == nullptr || descriptor == nullptr || width == 0 || height == 0) {
        return cudaErrorInvalidValue;
    }
    *array = reinterpret_cast<cudaArray_t>(&g_array_token);
    return cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaMalloc3DArray(cudaArray_t* array,
                                                   const struct cudaChannelFormatDesc* descriptor,
                                                   struct cudaExtent extent, unsigned int) {
    if (array == nullptr || descriptor == nullptr || extent.width == 0 || extent.height == 0 ||
        extent.depth == 0) {
        return cudaErrorInvalidValue;
    }
    *array = reinterpret_cast<cudaArray_t>(&g_array_token);
    return cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaMallocMipmappedArray(
    cudaMipmappedArray_t* mipmap, const struct cudaChannelFormatDesc* descriptor,
    struct cudaExtent extent, unsigned int level_count, unsigned int) {
    if (mipmap == nullptr || descriptor == nullptr || extent.width == 0 || extent.height == 0 ||
        extent.depth == 0 || level_count == 0) {
        return cudaErrorInvalidValue;
    }
    *mipmap = reinterpret_cast<cudaMipmappedArray_t>(&g_array_token);
    return cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaFreeArray(cudaArray_t array) {
    return array == nullptr ? cudaErrorInvalidValue : cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaFreeMipmappedArray(cudaMipmappedArray_t mipmap) {
    return mipmap == nullptr ? cudaErrorInvalidValue : cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaGraphicsUnregisterResource(cudaGraphicsResource_t resource) {
    return resource == nullptr ? cudaErrorInvalidResourceHandle : cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaGraphicsResourceSetMapFlags(cudaGraphicsResource_t resource,
                                                                 unsigned int) {
    return resource == nullptr ? cudaErrorInvalidResourceHandle : cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaGraphicsMapResources(int count,
                                                          cudaGraphicsResource_t* resources,
                                                          cudaStream_t) {
    return count < 0 || (count != 0 && resources == nullptr) ? cudaErrorInvalidValue : cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaGraphicsUnmapResources(int count,
                                                            cudaGraphicsResource_t* resources,
                                                            cudaStream_t) {
    return count < 0 || (count != 0 && resources == nullptr) ? cudaErrorInvalidValue : cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaGraphicsResourceGetMappedPointer(
    void** device_pointer, std::size_t* size, cudaGraphicsResource_t resource) {
    if (device_pointer == nullptr || size == nullptr || resource == nullptr) {
        return cudaErrorInvalidValue;
    }
    *device_pointer = reinterpret_cast<void*>(0x73000000U);
    *size = 4096;
    return cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaGraphicsSubResourceGetMappedArray(
    cudaArray_t* array, cudaGraphicsResource_t resource, unsigned int, unsigned int) {
    if (array == nullptr || resource == nullptr) {
        return cudaErrorInvalidValue;
    }
    *array = reinterpret_cast<cudaArray_t>(&g_graphics_resource_token);
    return cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaGraphicsResourceGetMappedMipmappedArray(
    cudaMipmappedArray_t* mipmap, cudaGraphicsResource_t resource) {
    if (mipmap == nullptr || resource == nullptr) {
        return cudaErrorInvalidValue;
    }
    *mipmap = reinterpret_cast<cudaMipmappedArray_t>(&g_graphics_resource_token);
    return cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaGraphAddMemAllocNode(cudaGraphNode_t* graph_node,
                                                          cudaGraph_t graph, const cudaGraphNode_t*,
                                                          std::size_t,
                                                          struct cudaMemAllocNodeParams*) {
    if (graph_node == nullptr || graph == nullptr) {
        return cudaErrorInvalidValue;
    }
    *graph_node = reinterpret_cast<cudaGraphNode_t>(&g_array_token);
    return cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaMallocPitch(void** device_pointer, std::size_t* pitch,
                                                 std::size_t width_bytes, std::size_t height) {
    if (device_pointer == nullptr || pitch == nullptr) {
        return cudaErrorInvalidValue;
    }
    const DriverPitchAllocFunction allocate =
        resolve_driver_function<DriverPitchAllocFunction>("cuMemAllocPitch_v2");
    if (allocate == nullptr) {
        return cudaErrorNotSupported;
    }

    CUdeviceptr driver_pointer = 0;
    const CUresult result = allocate(&driver_pointer, pitch, width_bytes, height, 1);
    if (result != CUDA_SUCCESS) {
        return cudaErrorMemoryAllocation;
    }
    *device_pointer = to_runtime_pointer(driver_pointer);
    return cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaMalloc3D(struct cudaPitchedPtr* pitched_device_pointer,
                                              struct cudaExtent extent) {
    if (pitched_device_pointer == nullptr || extent.width == 0 || extent.height == 0 ||
        extent.depth == 0) {
        return cudaErrorInvalidValue;
    }
    if (extent.depth > std::numeric_limits<std::size_t>::max() / extent.height) {
        return cudaErrorInvalidValue;
    }

    const DriverPitchAllocFunction allocate =
        resolve_driver_function<DriverPitchAllocFunction>("cuMemAllocPitch_v2");
    if (allocate == nullptr) {
        return cudaErrorNotSupported;
    }

    CUdeviceptr driver_pointer = 0;
    std::size_t pitch = 0;
    const CUresult result =
        allocate(&driver_pointer, &pitch, extent.width, extent.height * extent.depth, 1);
    if (result != CUDA_SUCCESS) {
        return cudaErrorMemoryAllocation;
    }
    pitched_device_pointer->ptr = to_runtime_pointer(driver_pointer);
    pitched_device_pointer->pitch = pitch;
    pitched_device_pointer->xsize = extent.width;
    pitched_device_pointer->ysize = extent.height;
    return cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaGetDevice(int* device) {
    if (device == nullptr) {
        return cudaErrorInvalidValue;
    }
    *device = 0;
    return cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaLaunchKernel(const void*, dim3, dim3, void**, std::size_t,
                                                  cudaStream_t) {
    return cudaSuccess;
}

#ifndef GLIMMER_FAKE_NO_CAPTURE
namespace {

cudaError_t fake_runtime_capture_query(cudaStream_t stream, cudaStreamCaptureStatus* status,
                                       const char* alias) {
    using DriverQuery = CUresult (*)(CUstream, CUstreamCaptureStatus*);
    if (status == nullptr) {
        return cudaErrorInvalidValue;
    }
    // Existing preload fixtures load the native Driver locally. Capture APIs
    // are not exported interceptor wrappers, so RTLD_DEFAULT cannot find them.
    void* driver = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (driver == nullptr) {
        return cudaErrorNotSupported;
    }
    const auto query = reinterpret_cast<DriverQuery>(dlsym(driver, alias));
    CUstreamCaptureStatus driver_status{};
    const CUresult result = query == nullptr
                                ? CUDA_ERROR_NOT_SUPPORTED
                                : query(reinterpret_cast<CUstream>(stream), &driver_status);
    const bool library_closed = dlclose(driver) == 0;
    if (result != CUDA_SUCCESS || !library_closed) {
        return cudaErrorInitializationError;
    }
    *status = static_cast<cudaStreamCaptureStatus>(driver_status);
    return cudaSuccess;
}

}  // namespace

extern "C" cudaError_t CUDARTAPI cudaStreamIsCapturing(cudaStream_t stream,
                                                       cudaStreamCaptureStatus* status) {
    return fake_runtime_capture_query(stream, status, "cuStreamIsCapturing");
}

extern "C" cudaError_t CUDARTAPI cudaStreamIsCapturing_ptsz(cudaStream_t stream,
                                                            cudaStreamCaptureStatus* status) {
    return fake_runtime_capture_query(stream, status, "cuStreamIsCapturing_ptsz");
}
#endif

extern "C" cudaError_t CUDARTAPI cudaLaunchKernel_ptsz(const void* function, dim3 grid_dim,
                                                       dim3 block_dim, void** arguments,
                                                       std::size_t shared_memory_bytes,
                                                       cudaStream_t stream) {
    return cudaLaunchKernel(function, grid_dim, block_dim, arguments, shared_memory_bytes, stream);
}

extern "C" cudaError_t CUDARTAPI cudaGraphLaunch(cudaGraphExec_t graph_exec, cudaStream_t) {
    return graph_exec == nullptr ? cudaErrorInvalidValue : cudaSuccess;
}

namespace {
std::atomic<unsigned int> g_graph_constructor_calls{};
std::atomic<unsigned int> g_graph_destroy_calls{};
std::array<std::atomic<unsigned int>, 2> g_graph_event_update_calls{};
std::array<std::atomic<unsigned int>, 2> g_graph_node_update_calls{};
std::array<std::atomic<unsigned int>, 2> g_graph_update_calls{};
std::atomic<unsigned int> g_graph_child_update_calls{};
#ifdef GLIMMER_FAKE_GRAPH_ABI_VERSIONS
std::array<std::atomic<unsigned int>, 2> g_graph_base_calls{};
#endif
cudaError_t construct_fake_runtime_graph(cudaGraphExec_t* output, cudaGraph_t graph,
                                         unsigned long long flags) {
    ++g_graph_constructor_calls;
    if (!output || graph != reinterpret_cast<cudaGraph_t>(0xfa020)) {
        return cudaErrorInvalidValue;
    }
    if (flags == 10000) {
        return cudaErrorApiFailureBase;
    }
    if (flags == 10002) {
        *output = nullptr;
        return cudaSuccess;
    }
    *output = reinterpret_cast<cudaGraphExec_t>(0xfa010);
    return flags == 10001 ? cudaErrorApiFailureBase : cudaSuccess;
}
}  // namespace
extern "C" unsigned int glimmer_fake_runtime_graph_constructor_calls() {
    return g_graph_constructor_calls.load();
}
extern "C" unsigned int glimmer_fake_runtime_graph_destroy_calls() {
    return g_graph_destroy_calls.load();
}
extern "C" unsigned int glimmer_fake_runtime_graph_event_update_calls(bool is_record) {
    return g_graph_event_update_calls[is_record ? 0 : 1].load();
}
extern "C" unsigned int glimmer_fake_runtime_graph_node_update_calls(bool is_generic) {
    return g_graph_node_update_calls[is_generic ? 0 : 1].load();
}
extern "C" unsigned int glimmer_fake_runtime_graph_update_calls(bool is_legacy) {
    return g_graph_update_calls[is_legacy ? 0 : 1].load();
}
extern "C" unsigned int glimmer_fake_runtime_graph_child_update_calls() {
    return g_graph_child_update_calls.load();
}
extern "C" cudaError_t CUDARTAPI cudaGraphExecChildGraphNodeSetParams(cudaGraphExec_t executable,
                                                                      cudaGraphNode_t node,
                                                                      cudaGraph_t graph) {
    ++g_graph_child_update_calls;
    if (executable != reinterpret_cast<cudaGraphExec_t>(0xfa010) || !node || !graph)
        return cudaErrorInvalidValue;
    return graph == reinterpret_cast<cudaGraph_t>(0xfa099) ? cudaErrorApiFailureBase : cudaSuccess;
}
extern "C" cudaError_t CUDARTAPI glimmer_runtime_graph_update_legacy(
    cudaGraphExec_t executable, cudaGraph_t graph, cudaGraphNode_t* error_node,
    cudaGraphExecUpdateResult* result) {
    ++g_graph_update_calls[0];
    if (!executable || !graph || !result) return cudaErrorInvalidValue;
    if (error_node) *error_node = reinterpret_cast<cudaGraphNode_t>(0xfa052);
    *result = cudaGraphExecUpdateSuccess;
    return graph == reinterpret_cast<cudaGraph_t>(0xfa099) ? cudaErrorApiFailureBase : cudaSuccess;
}
#if CUDART_VERSION >= 12000
extern "C" cudaError_t CUDARTAPI glimmer_runtime_graph_update_modern(
    cudaGraphExec_t executable, cudaGraph_t graph, cudaGraphExecUpdateResultInfo* result) {
    ++g_graph_update_calls[1];
    if (!executable || !graph || !result) return cudaErrorInvalidValue;
    result->result = cudaGraphExecUpdateSuccess;
    result->errorNode = reinterpret_cast<cudaGraphNode_t>(0xfa053);
    result->errorFromNode = reinterpret_cast<cudaGraphNode_t>(0xfa054);
    return graph == reinterpret_cast<cudaGraph_t>(0xfa099) ? cudaErrorApiFailureBase : cudaSuccess;
}
#endif
#ifdef GLIMMER_FAKE_GRAPH_ABI_VERSIONS
#if CUDART_VERSION >= 12000
asm(".symver glimmer_runtime_graph_update_legacy,cudaGraphExecUpdate@libcudart.so.11.0");
asm(".symver glimmer_runtime_graph_update_modern,cudaGraphExecUpdate@@libcudart.so.12");
#else
asm(".symver glimmer_runtime_graph_update_legacy,cudaGraphExecUpdate@@libcudart.so.11.0");
#endif
#else
#if CUDART_VERSION >= 12000
extern "C" cudaError_t CUDARTAPI cudaGraphExecUpdate(cudaGraphExec_t executable, cudaGraph_t graph,
                                                     cudaGraphExecUpdateResultInfo* result) {
    return glimmer_runtime_graph_update_modern(executable, graph, result);
}
#else
extern "C" cudaError_t CUDARTAPI cudaGraphExecUpdate(cudaGraphExec_t executable, cudaGraph_t graph,
                                                     cudaGraphNode_t* error_node,
                                                     cudaGraphExecUpdateResult* result) {
    return glimmer_runtime_graph_update_legacy(executable, graph, error_node, result);
}
#endif
#endif
extern "C" cudaError_t CUDARTAPI cudaGraphNodeSetEnabled(cudaGraphExec_t executable,
                                                         cudaGraphNode_t node,
                                                         unsigned int is_enabled) {
    ++g_graph_node_update_calls[1];
    if (executable != reinterpret_cast<cudaGraphExec_t>(0xfa010) || !node)
        return cudaErrorInvalidValue;
    return is_enabled == 10000 ? cudaErrorApiFailureBase : cudaSuccess;
}
#if CUDART_VERSION >= 12020
extern "C" cudaError_t CUDARTAPI cudaGraphExecNodeSetParams(cudaGraphExec_t executable,
                                                            cudaGraphNode_t node,
                                                            cudaGraphNodeParams* params) {
    ++g_graph_node_update_calls[0];
    if (executable != reinterpret_cast<cudaGraphExec_t>(0xfa010) || !node || !params)
        return cudaErrorInvalidValue;
    return params->reserved0[0] == 10000 ? cudaErrorApiFailureBase : cudaSuccess;
}
#endif
namespace {
cudaError_t update_fake_graph_event(cudaGraphExec_t executable, cudaGraphNode_t node,
                                    cudaEvent_t event, bool is_record) {
    ++g_graph_event_update_calls[is_record ? 0 : 1];
    if (executable != reinterpret_cast<cudaGraphExec_t>(0xfa010) || !node || !event) {
        return cudaErrorInvalidValue;
    }
    return event == reinterpret_cast<cudaEvent_t>(0xfa099) ? cudaErrorApiFailureBase : cudaSuccess;
}
}  // namespace
extern "C" cudaError_t CUDARTAPI cudaGraphExecEventRecordNodeSetEvent(cudaGraphExec_t executable,
                                                                      cudaGraphNode_t node,
                                                                      cudaEvent_t event) {
    return update_fake_graph_event(executable, node, event, true);
}
extern "C" cudaError_t CUDARTAPI cudaGraphExecEventWaitNodeSetEvent(cudaGraphExec_t executable,
                                                                    cudaGraphNode_t node,
                                                                    cudaEvent_t event) {
    return update_fake_graph_event(executable, node, event, false);
}
#ifdef GLIMMER_FAKE_GRAPH_ABI_VERSIONS
extern "C" unsigned int glimmer_fake_runtime_graph_base_calls(bool is_legacy) {
    return g_graph_base_calls[is_legacy ? 0 : 1].load();
}
extern "C" cudaError_t CUDARTAPI glimmer_runtime_graph_instantiate_legacy(
    cudaGraphExec_t* output, cudaGraph_t graph, cudaGraphNode_t* error_node, char* log,
    std::size_t log_bytes) {
    ++g_graph_base_calls[0];
    if (error_node) *error_node = reinterpret_cast<cudaGraphNode_t>(0xfa060);
    if (log && log_bytes) {
        log[0] = 'L';
        log[log_bytes - 1] = '!';
    }
    return construct_fake_runtime_graph(output, graph, 0);
}
extern "C" cudaError_t CUDARTAPI glimmer_runtime_graph_instantiate_modern(
    cudaGraphExec_t* output, cudaGraph_t graph, unsigned long long flags) {
    ++g_graph_base_calls[1];
    return construct_fake_runtime_graph(output, graph, flags);
}
#if CUDART_VERSION >= 12000
asm(".symver glimmer_runtime_graph_instantiate_legacy,cudaGraphInstantiate@libcudart.so.11.0");
asm(".symver glimmer_runtime_graph_instantiate_modern,cudaGraphInstantiate@@libcudart.so.12");
#else
asm(".symver glimmer_runtime_graph_instantiate_legacy,cudaGraphInstantiate@@libcudart.so.11.0");
asm(".symver glimmer_runtime_graph_instantiate_modern,cudaGraphInstantiate@libcudart.so.12");
#endif
#endif
extern "C" cudaError_t CUDARTAPI cudaGraphInstantiateWithFlags(cudaGraphExec_t* output,
                                                               cudaGraph_t graph,
                                                               unsigned long long flags) {
    return construct_fake_runtime_graph(output, graph, flags);
}
#if CUDART_VERSION >= 12000
namespace {
cudaError_t construct_fake_runtime_graph_params(cudaGraphExec_t* output, cudaGraph_t graph,
                                                cudaGraphInstantiateParams* parameters,
                                                bool is_ptds) {
    if (!parameters || parameters->uploadStream != reinterpret_cast<cudaStream_t>(0xfa040)) {
        return cudaErrorInvalidValue;
    }
    const auto legacy_node = reinterpret_cast<cudaGraphNode_t>(0xfa050);
    const auto ptds_node = reinterpret_cast<cudaGraphNode_t>(0xfa051);
    parameters->errNode_out = is_ptds ? ptds_node : legacy_node;
    parameters->result_out = cudaGraphInstantiateSuccess;
    return construct_fake_runtime_graph(output, graph, parameters->flags);
}
}  // namespace
extern "C" cudaError_t CUDARTAPI cudaGraphInstantiateWithParams(
    cudaGraphExec_t* output, cudaGraph_t graph, cudaGraphInstantiateParams* parameters) {
    return construct_fake_runtime_graph_params(output, graph, parameters, false);
}
extern "C" cudaError_t CUDARTAPI cudaGraphInstantiateWithParams_ptsz(
    cudaGraphExec_t* output, cudaGraph_t graph, cudaGraphInstantiateParams* parameters) {
    return construct_fake_runtime_graph_params(output, graph, parameters, true);
}
#endif
extern "C" cudaError_t CUDARTAPI cudaGraphExecDestroy(cudaGraphExec_t executable) {
    ++g_graph_destroy_calls;
    return executable == reinterpret_cast<cudaGraphExec_t>(0xfa099) ? cudaErrorApiFailureBase
           : executable                                             ? cudaSuccess
                                                                    : cudaErrorInvalidValue;
}

extern "C" cudaError_t CUDARTAPI cudaLaunchKernelExC(const cudaLaunchConfig_t* config,
                                                     const void* function, void** arguments) {
    if (config == nullptr || function == nullptr || arguments == nullptr ||
        arguments[0] != config || arguments[1] != config->attrs || config->numAttrs != 1 ||
        config->gridDim.x != 3 || config->blockDim.x != 32 || config->dynamicSmemBytes != 64) {
        return cudaErrorInvalidValue;
    }
    return *static_cast<cudaError_t*>(arguments[2]);
}

extern "C" cudaError_t CUDARTAPI cudaLaunchKernelExC_ptsz(const cudaLaunchConfig_t* config,
                                                          const void* function, void** arguments) {
    return cudaLaunchKernelExC(config, function, arguments);
}

extern "C" cudaError_t CUDARTAPI cudaGraphLaunch_ptsz(cudaGraphExec_t graph_exec,
                                                      cudaStream_t stream) {
    return cudaGraphLaunch(graph_exec, stream);
}

// NOLINTBEGIN(bugprone-reserved-identifier, readability-identifier-naming): preserve CUDA
// compiler ABI names.
extern "C" cudaError_t CUDARTAPI __cudaLaunchKernel(cudaKernel_t, dim3, dim3, void**, std::size_t,
                                                    cudaStream_t) {
    return cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI __cudaLaunchKernel_ptsz(cudaKernel_t kernel, dim3 grid_dim,
                                                         dim3 block_dim, void** arguments,
                                                         std::size_t shared_memory_bytes,
                                                         cudaStream_t stream) {
    return __cudaLaunchKernel(kernel, grid_dim, block_dim, arguments, shared_memory_bytes, stream);
}
// NOLINTEND(bugprone-reserved-identifier, readability-identifier-naming)

extern "C" cudaError_t CUDARTAPI cudaFree(void* device_pointer) {
    const DriverFreeFunction release = resolve_driver_function<DriverFreeFunction>("cuMemFree_v2");
    if (release == nullptr) {
        return cudaErrorNotSupported;
    }
    const CUresult result = release(to_device_pointer(device_pointer));
    return result == CUDA_SUCCESS ? cudaSuccess : cudaErrorInvalidValue;
}

extern "C" cudaError_t CUDARTAPI cudaMallocAsync(void** device_pointer, std::size_t memory_bytes,
                                                 cudaStream_t stream) {
    if (device_pointer == nullptr) {
        return cudaErrorInvalidValue;
    }
    const DriverAllocAsyncFunction allocate =
        resolve_driver_function<DriverAllocAsyncFunction>("cuMemAllocAsync");
    if (allocate == nullptr) {
        return cudaErrorNotSupported;
    }

    CUdeviceptr driver_pointer = 0;
    const CUresult result = allocate(&driver_pointer, memory_bytes, stream);
    if (result != CUDA_SUCCESS) {
        return cudaErrorMemoryAllocation;
    }
    *device_pointer = to_runtime_pointer(driver_pointer);
    return cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaMallocAsync_ptsz(void** device_pointer,
                                                      std::size_t memory_bytes,
                                                      cudaStream_t stream) {
    return cudaMallocAsync(device_pointer, memory_bytes, stream);
}

extern "C" cudaError_t CUDARTAPI cudaMallocFromPoolAsync(void** device_pointer,
                                                         std::size_t memory_bytes,
                                                         cudaMemPool_t pool, cudaStream_t stream) {
    if (device_pointer == nullptr) {
        return cudaErrorInvalidValue;
    }
    const DriverPoolAllocFunction allocate =
        resolve_driver_function<DriverPoolAllocFunction>("cuMemAllocFromPoolAsync");
    if (allocate == nullptr) {
        return cudaErrorNotSupported;
    }
    CUdeviceptr driver_pointer = 0;
    const CUresult result =
        allocate(&driver_pointer, memory_bytes, reinterpret_cast<CUmemoryPool>(pool), stream);
    if (result != CUDA_SUCCESS) {
        return cudaErrorMemoryAllocation;
    }
    *device_pointer = to_runtime_pointer(driver_pointer);
    return cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaMallocFromPoolAsync_ptsz(void** device_pointer,
                                                              std::size_t memory_bytes,
                                                              cudaMemPool_t pool,
                                                              cudaStream_t stream) {
    return cudaMallocFromPoolAsync(device_pointer, memory_bytes, pool, stream);
}

extern "C" cudaError_t CUDARTAPI cudaDeviceGetDefaultMemPool(cudaMemPool_t* pool, int device) {
    if (pool == nullptr || device < 0) {
        return cudaErrorInvalidValue;
    }
    *pool = fake_pool();
    return cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaDeviceSetMemPool(int device, cudaMemPool_t pool) {
    return device < 0 || pool == nullptr ? cudaErrorInvalidValue : cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaDeviceGetMemPool(cudaMemPool_t* pool, int device) {
    return cudaDeviceGetDefaultMemPool(pool, device);
}

extern "C" cudaError_t CUDARTAPI cudaMemPoolTrimTo(cudaMemPool_t pool, std::size_t) {
    return pool == nullptr ? cudaErrorInvalidValue : cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaMemPoolSetAttribute(cudaMemPool_t pool, cudaMemPoolAttr,
                                                         void* value) {
    return pool == nullptr || value == nullptr ? cudaErrorInvalidValue : cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaMemPoolGetAttribute(cudaMemPool_t pool, cudaMemPoolAttr,
                                                         void* value) {
    return pool == nullptr || value == nullptr ? cudaErrorInvalidValue : cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaMemPoolSetAccess(cudaMemPool_t pool,
                                                      const cudaMemAccessDesc* descriptors,
                                                      std::size_t descriptor_count) {
    return pool == nullptr || (descriptor_count != 0 && descriptors == nullptr)
               ? cudaErrorInvalidValue
               : cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaMemPoolGetAccess(cudaMemAccessFlags* flags, cudaMemPool_t pool,
                                                      cudaMemLocation* location) {
    if (flags == nullptr || pool == nullptr || location == nullptr) {
        return cudaErrorInvalidValue;
    }
    *flags = cudaMemAccessFlagsProtReadWrite;
    return cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaMemPoolCreate(cudaMemPool_t* pool,
                                                   const cudaMemPoolProps* properties) {
    if (pool == nullptr || properties == nullptr) {
        return cudaErrorInvalidValue;
    }
    *pool = fake_pool();
    return cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaMemPoolDestroy(cudaMemPool_t pool) {
    return pool == nullptr ? cudaErrorInvalidValue : cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaMemGetDefaultMemPool(cudaMemPool_t* pool,
                                                          cudaMemLocation* location,
                                                          cudaMemAllocationType) {
    return pool == nullptr || location == nullptr ? cudaErrorInvalidValue
                                                  : (*pool = fake_pool(), cudaSuccess);
}

extern "C" cudaError_t CUDARTAPI cudaMemGetMemPool(cudaMemPool_t* pool, cudaMemLocation* location,
                                                   cudaMemAllocationType type) {
    return cudaMemGetDefaultMemPool(pool, location, type);
}

extern "C" cudaError_t CUDARTAPI cudaMemSetMemPool(cudaMemLocation* location, cudaMemAllocationType,
                                                   cudaMemPool_t pool) {
    return location == nullptr || pool == nullptr ? cudaErrorInvalidValue : cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaMemPoolExportToShareableHandle(
    void* handle_out, cudaMemPool_t pool, enum cudaMemAllocationHandleType, unsigned int) {
    return handle_out == nullptr || pool == nullptr ? cudaErrorInvalidValue : cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaMemPoolImportFromShareableHandle(
    cudaMemPool_t* pool_out, void*, enum cudaMemAllocationHandleType, unsigned int) {
    return pool_out == nullptr ? cudaErrorInvalidValue : (*pool_out = fake_pool(), cudaSuccess);
}

extern "C" cudaError_t CUDARTAPI cudaMemPoolExportPointer(cudaMemPoolPtrExportData* share_data_out,
                                                          void* device_pointer) {
    return share_data_out == nullptr || device_pointer == nullptr ? cudaErrorInvalidValue
                                                                  : cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaMemPoolImportPointer(void** pointer_out, cudaMemPool_t pool,
                                                          cudaMemPoolPtrExportData* share_data) {
    if (pointer_out == nullptr || pool == nullptr || share_data == nullptr) {
        return cudaErrorInvalidValue;
    }
    *pointer_out = reinterpret_cast<void*>(0x70000000U);
    return cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaFreeAsync(void* device_pointer, cudaStream_t stream) {
    const DriverFreeAsyncFunction release =
        resolve_driver_function<DriverFreeAsyncFunction>("cuMemFreeAsync");
    if (release == nullptr) {
        return cudaErrorNotSupported;
    }
    const CUresult result = release(to_device_pointer(device_pointer), stream);
    return result == CUDA_SUCCESS ? cudaSuccess : cudaErrorInvalidValue;
}

extern "C" cudaError_t CUDARTAPI cudaFreeAsync_ptsz(void* device_pointer, cudaStream_t stream) {
    return cudaFreeAsync(device_pointer, stream);
}

extern "C" cudaError_t CUDARTAPI cudaDeviceSynchronize() {
    const DriverContextSynchronizeFunction synchronize =
        resolve_driver_function<DriverContextSynchronizeFunction>("cuCtxSynchronize");
    if (synchronize == nullptr) {
        return cudaErrorNotSupported;
    }
    return synchronize() == CUDA_SUCCESS ? cudaSuccess : cudaErrorUnknown;
}

extern "C" cudaError_t CUDARTAPI cudaStreamSynchronize(cudaStream_t stream) {
    const DriverStreamSynchronizeFunction synchronize =
        resolve_driver_function<DriverStreamSynchronizeFunction>("cuStreamSynchronize");
    if (synchronize == nullptr) {
        return cudaErrorNotSupported;
    }
    return synchronize(reinterpret_cast<CUstream>(stream)) == CUDA_SUCCESS ? cudaSuccess
                                                                           : cudaErrorUnknown;
}

extern "C" cudaError_t CUDARTAPI cudaStreamSynchronize_ptsz(cudaStream_t stream) {
    return cudaStreamSynchronize(stream);
}

extern "C" cudaError_t CUDARTAPI cudaStreamQuery(cudaStream_t stream) {
    const DriverStreamQueryFunction query =
        resolve_driver_function<DriverStreamQueryFunction>("cuStreamQuery");
    if (query == nullptr) {
        return cudaErrorNotSupported;
    }
    return query(reinterpret_cast<CUstream>(stream)) == CUDA_SUCCESS ? cudaSuccess
                                                                     : cudaErrorUnknown;
}

extern "C" cudaError_t CUDARTAPI cudaStreamQuery_ptsz(cudaStream_t stream) {
    return cudaStreamQuery(stream);
}

extern "C" cudaError_t CUDARTAPI cudaStreamDestroy(cudaStream_t stream) {
    const DriverStreamDestroyFunction destroy =
        resolve_driver_function<DriverStreamDestroyFunction>("cuStreamDestroy");
    if (destroy == nullptr) {
        return cudaErrorNotSupported;
    }
    return destroy(reinterpret_cast<CUstream>(stream)) == CUDA_SUCCESS ? cudaSuccess
                                                                       : cudaErrorUnknown;
}

extern "C" cudaError_t CUDARTAPI cudaMemGetInfo(std::size_t* free_bytes, std::size_t* total_bytes) {
    const DriverMemGetInfoFunction query =
        resolve_driver_function<DriverMemGetInfoFunction>("cuMemGetInfo_v2");
    if (query == nullptr) {
        return cudaErrorNotSupported;
    }
    const CUresult result = query(free_bytes, total_bytes);
    return result == CUDA_SUCCESS ? cudaSuccess : cudaErrorInvalidValue;
}

extern "C" cudaError_t CUDARTAPI cudaStreamGetFlags(cudaStream_t stream, unsigned int* flags) {
    const auto query = resolve_driver_function<decltype(&cuStreamGetFlags)>("cuStreamGetFlags");
    if (!query) {
        return cudaErrorNotSupported;
    }
    return query(reinterpret_cast<CUstream>(stream), flags) == CUDA_SUCCESS ? cudaSuccess
                                                                            : cudaErrorUnknown;
}
