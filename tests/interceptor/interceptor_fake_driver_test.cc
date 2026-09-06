#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nvml.h>

#include "glimmer/control/shared_memory_quota.h"

#include <cerrno>
#include <dlfcn.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <bit>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <sys/wait.h>
#include <unistd.h>

namespace {

constexpr std::size_t kQuotaBytes = 4096;
constexpr std::size_t kFakePhysicalMemoryBytes = std::size_t{64} * 1024 * 1024;
constexpr std::size_t kDirectAllocationBytes = 3072;
constexpr std::size_t kRuntimeAllocationBytes = 1024;
constexpr std::size_t kAsyncAllocationBytes = 1024;
constexpr std::size_t kRejectedAllocationBytes = 2048;
constexpr std::size_t kForcedAllocationFailureBytes = 1536;
constexpr std::size_t kTaskLimitBytes = 1024;

using InitFunction = CUresult (*)(unsigned int flags);
using LaunchKernelFunction = CUresult (*)(CUfunction function, unsigned int grid_dim_x,
                                          unsigned int grid_dim_y, unsigned int grid_dim_z,
                                          unsigned int block_dim_x, unsigned int block_dim_y,
                                          unsigned int block_dim_z,
                                          unsigned int shared_memory_bytes, CUstream stream,
                                          void** kernel_parameters, void** extra);
using GraphLaunchFunction = CUresult (*)(CUgraphExec graph_exec, CUstream stream);
using AllocFunction = CUresult (*)(CUdeviceptr* device_pointer, std::size_t memory_bytes);
using AsyncAllocFunction = CUresult (*)(CUdeviceptr* device_pointer, std::size_t memory_bytes,
                                        CUstream stream);
using PoolAsyncAllocFunction = CUresult (*)(CUdeviceptr* device_pointer, std::size_t memory_bytes,
                                            CUmemoryPool pool, CUstream stream);
using VmmCreateFunction = CUresult (*)(CUmemGenericAllocationHandle* handle,
                                       std::size_t memory_bytes, const CUmemAllocationProp* prop,
                                       unsigned long long flags);
using VmmReleaseFunction = CUresult (*)(CUmemGenericAllocationHandle handle);
using AddressReserveFunction = CUresult (*)(CUdeviceptr* device_pointer, std::size_t memory_bytes,
                                            std::size_t alignment, CUdeviceptr requested_address,
                                            unsigned long long flags);
using AddressFreeFunction = CUresult (*)(CUdeviceptr device_pointer, std::size_t memory_bytes);
using MapFunction = CUresult (*)(CUdeviceptr device_pointer, std::size_t memory_bytes,
                                 std::size_t offset, CUmemGenericAllocationHandle handle,
                                 unsigned long long flags);
using MapArrayAsyncFunction = CUresult (*)(CUarrayMapInfo* map_info_list, unsigned int count,
                                           CUstream stream);
using UnmapFunction = CUresult (*)(CUdeviceptr device_pointer, std::size_t memory_bytes);
using SetAccessFunction = CUresult (*)(CUdeviceptr device_pointer, std::size_t memory_bytes,
                                       const CUmemAccessDesc* access_descriptors,
                                       std::size_t descriptor_count);
using GetAddressRangeFunction = CUresult (*)(CUdeviceptr* base_pointer, std::size_t* memory_bytes,
                                             CUdeviceptr device_pointer);
using GetAccessFunction = CUresult (*)(unsigned long long* flags, const CUmemLocation* location,
                                       CUdeviceptr device_pointer);
using ExportHandleFunction = CUresult (*)(void* shareable_handle,
                                          CUmemGenericAllocationHandle handle,
                                          CUmemAllocationHandleType handle_type,
                                          unsigned long long flags);
using ImportHandleFunction = CUresult (*)(CUmemGenericAllocationHandle* handle, void* os_handle,
                                          CUmemAllocationHandleType handle_type);
using IpcGetMemHandleFunction = CUresult (*)(CUipcMemHandle* handle, CUdeviceptr device_pointer);
using IpcOpenMemHandleFunction = CUresult (*)(CUdeviceptr* device_pointer, CUipcMemHandle handle,
                                              unsigned int flags);
using IpcCloseMemHandleFunction = CUresult (*)(CUdeviceptr device_pointer);
using ImportExternalMemoryFunction = CUresult (*)(
    CUexternalMemory* external_memory, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC* handle_desc);
using ExternalMemoryGetMappedBufferFunction =
    CUresult (*)(CUdeviceptr* device_pointer, CUexternalMemory external_memory,
                 const CUDA_EXTERNAL_MEMORY_BUFFER_DESC* buffer_desc);
using ExternalMemoryGetMappedMipmappedArrayFunction =
    CUresult (*)(CUmipmappedArray* mipmap, CUexternalMemory external_memory,
                 const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC* mipmap_desc);
using DestroyExternalMemoryFunction = CUresult (*)(CUexternalMemory external_memory);
using ArrayCreateFunction = CUresult (*)(CUarray* array, const CUDA_ARRAY_DESCRIPTOR* descriptor);
using Array3DCreateFunction = CUresult (*)(CUarray* array,
                                           const CUDA_ARRAY3D_DESCRIPTOR* descriptor);
using ArrayDestroyFunction = CUresult (*)(CUarray array);
using MipmappedArrayCreateFunction = CUresult (*)(CUmipmappedArray* mipmap,
                                                  const CUDA_ARRAY3D_DESCRIPTOR* descriptor,
                                                  unsigned int level_count);
using MipmappedArrayDestroyFunction = CUresult (*)(CUmipmappedArray mipmap);
using GraphicsUnregisterResourceFunction = CUresult (*)(CUgraphicsResource resource);
using GraphicsSubResourceGetMappedArrayFunction = CUresult (*)(CUarray* array,
                                                               CUgraphicsResource resource,
                                                               unsigned int array_index,
                                                               unsigned int mip_level);
using GraphicsResourceGetMappedMipmappedArrayFunction = CUresult (*)(CUmipmappedArray* mipmap,
                                                                     CUgraphicsResource resource);
using GraphicsResourceGetMappedPointerFunction = CUresult (*)(CUdeviceptr* device_pointer,
                                                              std::size_t* size,
                                                              CUgraphicsResource resource);
using GraphicsResourceSetMapFlagsFunction = CUresult (*)(CUgraphicsResource resource,
                                                         unsigned int flags);
using GraphicsMapResourcesFunction = CUresult (*)(unsigned int count, CUgraphicsResource* resources,
                                                  CUstream stream);
using GraphicsUnmapResourcesFunction = CUresult (*)(unsigned int count,
                                                    CUgraphicsResource* resources, CUstream stream);
using GetGranularityFunction = CUresult (*)(std::size_t* granularity,
                                            const CUmemAllocationProp* prop,
                                            CUmemAllocationGranularity_flags option);
using GetPropertiesFunction = CUresult (*)(CUmemAllocationProp* prop,
                                           CUmemGenericAllocationHandle handle);
using RetainHandleFunction = CUresult (*)(CUmemGenericAllocationHandle* handle,
                                          void* device_pointer);
using PoolTrimFunction = CUresult (*)(CUmemoryPool pool, std::size_t min_bytes_to_keep);
using PoolSetAttributeFunction = CUresult (*)(CUmemoryPool pool, CUmemPool_attribute attribute,
                                              void* value);
using PoolGetAttributeFunction = CUresult (*)(CUmemoryPool pool, CUmemPool_attribute attribute,
                                              void* value);
using PoolSetAccessFunction = CUresult (*)(CUmemoryPool pool,
                                           const CUmemAccessDesc* access_descriptors,
                                           std::size_t descriptor_count);
using PoolGetAccessFunction = CUresult (*)(CUmemAccess_flags* flags, CUmemoryPool pool,
                                           CUmemLocation* location);
using PoolCreateFunction = CUresult (*)(CUmemoryPool* pool, const CUmemPoolProps* properties);
using PoolDestroyFunction = CUresult (*)(CUmemoryPool pool);
using DeviceGetPoolFunction = CUresult (*)(CUmemoryPool* pool, CUdevice device);
using DeviceSetPoolFunction = CUresult (*)(CUdevice device, CUmemoryPool pool);
using DeviceGetDefaultPoolFunction = CUresult (*)(CUmemoryPool* pool, CUdevice device);
using GetDefaultPoolFunction = CUresult (*)(CUmemoryPool* pool, CUmemLocation* location,
                                            CUmemAllocationType type);
using GetPoolFunction = CUresult (*)(CUmemoryPool* pool, CUmemLocation* location,
                                     CUmemAllocationType type);
using SetPoolFunction = CUresult (*)(CUmemLocation* location, CUmemAllocationType type,
                                     CUmemoryPool pool);
using PoolExportHandleFunction = CUresult (*)(void* handle_out, CUmemoryPool pool,
                                              CUmemAllocationHandleType handle_type,
                                              unsigned long long flags);
using PoolImportHandleFunction = CUresult (*)(CUmemoryPool* pool_out, void* handle,
                                              CUmemAllocationHandleType handle_type,
                                              unsigned long long flags);
using PoolExportPointerFunction = CUresult (*)(CUmemPoolPtrExportData* share_data_out,
                                               CUdeviceptr device_pointer);
using PoolImportPointerFunction = CUresult (*)(CUdeviceptr* pointer_out, CUmemoryPool pool,
                                               CUmemPoolPtrExportData* share_data);
using FreeFunction = CUresult (*)(CUdeviceptr device_pointer);
using AsyncFreeFunction = CUresult (*)(CUdeviceptr device_pointer, CUstream stream);
using StreamGetDeviceFunction = CUresult (*)(CUstream stream, CUdevice* device);
using StreamGetContextFunction = CUresult (*)(CUstream stream, CUcontext* context);
using StreamQueryFunction = CUresult (*)(CUstream stream);
using StreamSynchronizeFunction = CUresult (*)(CUstream stream);
using StreamDestroyFunction = CUresult (*)(CUstream stream);
using ContextSynchronizeFunction = CUresult (*)();
using MemGetInfoFunction = CUresult (*)(std::size_t* free_bytes, std::size_t* total_bytes);
using DeviceTotalMemFunction = CUresult (*)(std::size_t* total_bytes, CUdevice device);
using ContextGetCurrentFunction = CUresult (*)(CUcontext* context);
using ContextDestroyFunction = CUresult (*)(CUcontext context);
using LegacyGetProcAddressFunction = CUresult (*)(const char* symbol, void** function_pointer,
                                                  int cuda_version, cuuint64_t flags);
using GetProcAddressV2Function = CUresult (*)(const char* symbol, void** function_pointer,
                                              int cuda_version, cuuint64_t flags,
                                              CUdriverProcAddressQueryResult* symbol_status);
using RuntimeMallocFunction = cudaError_t (*)(void** device_pointer, std::size_t memory_bytes);
using RuntimeMallocManagedFunction = cudaError_t (*)(void** device_pointer,
                                                     std::size_t memory_bytes, unsigned int flags);
using RuntimeMallocPitchFunction = cudaError_t (*)(void** device_pointer, std::size_t* pitch,
                                                   std::size_t width_bytes, std::size_t height);
using RuntimeMalloc3DFunction = cudaError_t (*)(struct cudaPitchedPtr* pitched_device_pointer,
                                                struct cudaExtent extent);
using RuntimeMallocAsyncFunction = cudaError_t (*)(void** device_pointer, std::size_t memory_bytes,
                                                   cudaStream_t stream);
using RuntimeLaunchKernelFunction = cudaError_t (*)(const void* function, dim3 grid_dim,
                                                    dim3 block_dim, void** arguments,
                                                    std::size_t shared_memory_bytes,
                                                    cudaStream_t stream);
using RuntimeGraphLaunchFunction = cudaError_t (*)(cudaGraphExec_t graph_exec, cudaStream_t stream);
using RuntimeInternalLaunchKernelFunction = cudaError_t (*)(cudaKernel_t kernel, dim3 grid_dim,
                                                            dim3 block_dim, void** arguments,
                                                            std::size_t shared_memory_bytes,
                                                            cudaStream_t stream);
using RuntimeMallocFromPoolAsyncFunction = cudaError_t (*)(void** device_pointer,
                                                           std::size_t memory_bytes,
                                                           cudaMemPool_t pool, cudaStream_t stream);
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
using RuntimeFreeAsyncFunction = cudaError_t (*)(void* device_pointer, cudaStream_t stream);
using RuntimeDeviceSynchronizeFunction = cudaError_t (*)();
using RuntimeStreamSynchronizeFunction = cudaError_t (*)(cudaStream_t stream);
using RuntimeStreamQueryFunction = cudaError_t (*)(cudaStream_t stream);
using RuntimeStreamDestroyFunction = cudaError_t (*)(cudaStream_t stream);
using RuntimeMemGetInfoFunction = cudaError_t (*)(std::size_t* free_bytes,
                                                  std::size_t* total_bytes);
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
using NvmlInitFunction = nvmlReturn_t (*)();
using NvmlShutdownFunction = nvmlReturn_t (*)();
using NvmlDeviceGetCountFunction = nvmlReturn_t (*)(unsigned int* device_count);
using NvmlDeviceGetHandleByIndexFunction = nvmlReturn_t (*)(unsigned int index,
                                                            nvmlDevice_t* device);
using NvmlDeviceGetIndexFunction = nvmlReturn_t (*)(nvmlDevice_t device, unsigned int* index);
using NvmlDeviceGetMemoryInfoFunction = nvmlReturn_t (*)(nvmlDevice_t device, nvmlMemory_t* memory);
using NvmlDeviceGetMemoryInfoV2Function = nvmlReturn_t (*)(nvmlDevice_t device,
                                                           nvmlMemory_v2_t* memory);

template <typename Function>
Function resolve_default(const char* name) {
    return reinterpret_cast<Function>(dlsym(RTLD_DEFAULT, name));
}

bool expect(bool condition, std::string_view message) {
    if (condition) {
        return true;
    }
    std::cerr << "test assertion failed: " << message << '\n';
    return false;
}

void write_result(int file_descriptor, int result) {
    const char* data = reinterpret_cast<const char*>(&result);
    std::size_t written = 0;
    while (written < sizeof(result)) {
        const ssize_t count = ::write(file_descriptor, data + written, sizeof(result) - written);
        if (count > 0) {
            written += static_cast<std::size_t>(count);
            continue;
        }
        if (count < 0 && errno == EINTR) {
            continue;
        }
        _exit(EXIT_FAILURE);
    }
}

[[nodiscard]] bool read_result(int file_descriptor, int* result) {
    if (result == nullptr) {
        return false;
    }
    char* data = reinterpret_cast<char*>(result);
    std::size_t received = 0;
    while (received < sizeof(*result)) {
        const ssize_t count = ::read(file_descriptor, data + received, sizeof(*result) - received);
        if (count > 0) {
            received += static_cast<std::size_t>(count);
            continue;
        }
        if (count < 0 && errno == EINTR) {
            continue;
        }
        return false;
    }
    return true;
}

int run_duplicate_record_failure_test() {
    const InitFunction init = resolve_default<InitFunction>("cuInit");
    const AllocFunction allocate = resolve_default<AllocFunction>("cuMemAlloc_v2");
    const MemGetInfoFunction get_info = resolve_default<MemGetInfoFunction>("cuMemGetInfo_v2");
    if (!expect(init != nullptr && allocate != nullptr && get_info != nullptr,
                "duplicate-record test symbols were not exported")) {
        return EXIT_FAILURE;
    }
    if (!expect(init(0) == CUDA_SUCCESS, "duplicate-record test cuInit failed")) {
        return EXIT_FAILURE;
    }

    CUdeviceptr first_pointer = 0;
    if (!expect(allocate(&first_pointer, 128) == CUDA_SUCCESS,
                "initial duplicate-record allocation failed")) {
        return EXIT_FAILURE;
    }

    CUdeviceptr duplicate_pointer = 0;
    if (!expect(allocate(&duplicate_pointer, 128) == CUDA_ERROR_OUT_OF_MEMORY,
                "duplicate allocation was not rejected")) {
        return EXIT_FAILURE;
    }
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    if (!expect(get_info(&free_bytes, &total_bytes) == CUDA_ERROR_UNKNOWN,
                "duplicate record failure did not enter degraded mode")) {
        return EXIT_FAILURE;
    }
    return expect(allocate(&duplicate_pointer, 128) == CUDA_ERROR_UNKNOWN,
                  "degraded mode accepted a later allocation")
               ? EXIT_SUCCESS
               : EXIT_FAILURE;
}

int run_task_memory_limit_test() {
    const char* configured_tenant = std::getenv("GLIMMER_QUOTA_TENANT_ID");
    if (!expect(configured_tenant != nullptr && *configured_tenant != '\0',
                "task-limit test tenant was not configured")) {
        return EXIT_FAILURE;
    }
    const std::string tenant_id =
        std::string(configured_tenant) + "-" + std::to_string(static_cast<long long>(::getpid()));
    if (!expect(::setenv("GLIMMER_QUOTA_TENANT_ID", tenant_id.c_str(), 1) == 0,
                "task-limit test tenant could not be isolated") ||
        !expect(glimmer::control::SharedMemoryQuota::remove_region(tenant_id),
                "stale task-limit test region could not be removed")) {
        return EXIT_FAILURE;
    }

    const InitFunction init = resolve_default<InitFunction>("cuInit");
    const AllocFunction allocate = resolve_default<AllocFunction>("cuMemAlloc_v2");
    const FreeFunction release = resolve_default<FreeFunction>("cuMemFree_v2");
    const MemGetInfoFunction get_info = resolve_default<MemGetInfoFunction>("cuMemGetInfo_v2");
    if (!expect(init != nullptr && allocate != nullptr && release != nullptr && get_info != nullptr,
                "task-limit test symbols were not exported")) {
        return EXIT_FAILURE;
    }

    CUdeviceptr pointer = 0;
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    const bool allocated =
        init(0) == CUDA_SUCCESS && allocate(&pointer, kTaskLimitBytes) == CUDA_SUCCESS;
    const bool rejected = allocate(&pointer, 1) == CUDA_ERROR_OUT_OF_MEMORY && pointer != 0;
    const bool virtualized = get_info(&free_bytes, &total_bytes) == CUDA_SUCCESS &&
                             total_bytes == kTaskLimitBytes && free_bytes == 0;
    const bool released = release(pointer) == CUDA_SUCCESS &&
                          get_info(&free_bytes, &total_bytes) == CUDA_SUCCESS &&
                          free_bytes == kTaskLimitBytes;
    const bool removed = glimmer::control::SharedMemoryQuota::remove_region(tenant_id);
    return expect(allocated, "task-limit allocation was rejected") &&
                   expect(rejected, "task-limit allocation was not rejected") &&
                   expect(virtualized, "task-limit memory view was not virtualized") &&
                   expect(released, "task-limit release did not restore the task quota") &&
                   expect(removed, "task-limit region could not be removed")
               ? EXIT_SUCCESS
               : EXIT_FAILURE;
}

int run_physical_capacity_test() {
    const InitFunction init = resolve_default<InitFunction>("cuInit");
    const AllocFunction allocate = resolve_default<AllocFunction>("cuMemAlloc_v2");
    const FreeFunction release = resolve_default<FreeFunction>("cuMemFree_v2");
    const MemGetInfoFunction get_info = resolve_default<MemGetInfoFunction>("cuMemGetInfo_v2");
    const DeviceTotalMemFunction get_total =
        resolve_default<DeviceTotalMemFunction>("cuDeviceTotalMem_v2");
    if (!expect(init != nullptr && allocate != nullptr && release != nullptr &&
                    get_info != nullptr && get_total != nullptr,
                "physical-capacity test symbols were not exported")) {
        return EXIT_FAILURE;
    }

    std::size_t total_bytes = 0;
    std::size_t free_bytes = 0;
    CUdeviceptr pointer = 0;
    const bool initialized = init(0) == CUDA_SUCCESS;
    const bool total_is_physical = initialized && get_total(&total_bytes, 0) == CUDA_SUCCESS &&
                                   total_bytes == kFakePhysicalMemoryBytes;
    const bool initial_view_is_physical = get_info(&free_bytes, &total_bytes) == CUDA_SUCCESS &&
                                          total_bytes == kFakePhysicalMemoryBytes &&
                                          free_bytes == kFakePhysicalMemoryBytes;
    const bool allocated = allocate(&pointer, kFakePhysicalMemoryBytes) == CUDA_SUCCESS;
    const bool exhausted_view = get_info(&free_bytes, &total_bytes) == CUDA_SUCCESS &&
                                total_bytes == kFakePhysicalMemoryBytes && free_bytes == 0;
    CUdeviceptr rejected_pointer = 0x1234;
    const bool rejected =
        allocate(&rejected_pointer, 1) == CUDA_ERROR_OUT_OF_MEMORY && rejected_pointer == 0x1234;
    const bool released = release(pointer) == CUDA_SUCCESS &&
                          get_info(&free_bytes, &total_bytes) == CUDA_SUCCESS &&
                          free_bytes == kFakePhysicalMemoryBytes;
    return expect(total_is_physical, "configured quota was not clamped to physical total") &&
                   expect(initial_view_is_physical,
                          "initial memory view did not expose physical capacity") &&
                   expect(allocated, "allocation at the physical capacity was rejected") &&
                   expect(exhausted_view, "physical exhaustion was not reflected in the view") &&
                   expect(rejected, "allocation beyond physical capacity was not rejected") &&
                   expect(released, "physical-capacity release did not restore the view")
               ? EXIT_SUCCESS
               : EXIT_FAILURE;
}

int run_nvml_only_virtualization_test() {
    const NvmlInitFunction nvml_init = resolve_default<NvmlInitFunction>("nvmlInit_v2");
    const NvmlShutdownFunction nvml_shutdown =
        resolve_default<NvmlShutdownFunction>("nvmlShutdown");
    const NvmlDeviceGetHandleByIndexFunction get_handle =
        resolve_default<NvmlDeviceGetHandleByIndexFunction>("nvmlDeviceGetHandleByIndex_v2");
    const NvmlDeviceGetMemoryInfoFunction get_memory_info =
        resolve_default<NvmlDeviceGetMemoryInfoFunction>("nvmlDeviceGetMemoryInfo");
    if (!expect(nvml_init != nullptr && nvml_shutdown != nullptr && get_handle != nullptr &&
                    get_memory_info != nullptr,
                "NVML-only test symbols were not exported")) {
        return EXIT_FAILURE;
    }

    nvmlDevice_t device = nullptr;
    nvmlMemory_t memory{};
    const bool virtualized =
        nvml_init() == NVML_SUCCESS && get_handle(0, &device) == NVML_SUCCESS &&
        get_memory_info(device, &memory) == NVML_SUCCESS && memory.total == kQuotaBytes &&
        memory.free == kQuotaBytes && memory.used == 0;
    const bool shutdown = nvml_shutdown() == NVML_SUCCESS;
    return expect(virtualized, "NVML-only virtualization required CUDA Driver initialization") &&
                   expect(shutdown, "NVML-only test shutdown failed")
               ? EXIT_SUCCESS
               : EXIT_FAILURE;
}

int run_async_null_pointer_test(bool runtime) {
    const char* configured_tenant = std::getenv("GLIMMER_QUOTA_TENANT_ID");
    if (!expect(configured_tenant != nullptr && *configured_tenant != '\0',
                "null-pointer test tenant was not configured")) {
        return EXIT_FAILURE;
    }

    const std::string tenant_id =
        std::string(configured_tenant) + "-" + std::to_string(static_cast<long long>(::getpid()));
    if (!expect(::setenv("GLIMMER_QUOTA_TENANT_ID", tenant_id.c_str(), 1) == 0,
                "null-pointer test tenant could not be isolated") ||
        !expect(glimmer::control::SharedMemoryQuota::remove_region(tenant_id),
                "stale null-pointer test region could not be removed")) {
        return EXIT_FAILURE;
    }

    int result_pipe[2] = {-1, -1};
    int release_pipe[2] = {-1, -1};
    if (!expect(::pipe(result_pipe) == 0 && ::pipe(release_pipe) == 0,
                "null-pointer test pipes could not be created")) {
        if (result_pipe[0] >= 0) {
            (void)::close(result_pipe[0]);
            (void)::close(result_pipe[1]);
        }
        return EXIT_FAILURE;
    }

    const pid_t child_pid = ::fork();
    if (!expect(child_pid >= 0, "null-pointer test child could not be created")) {
        (void)::close(result_pipe[0]);
        (void)::close(result_pipe[1]);
        (void)::close(release_pipe[0]);
        (void)::close(release_pipe[1]);
        return EXIT_FAILURE;
    }

    if (child_pid == 0) {
        (void)::close(result_pipe[0]);
        (void)::close(release_pipe[1]);
        int child_result = EXIT_FAILURE;
        if (runtime) {
            void* runtime_handle = dlopen("libcudart.so", RTLD_NOW | RTLD_GLOBAL);
            const RuntimeMallocAsyncFunction allocate =
                resolve_default<RuntimeMallocAsyncFunction>("cudaMallocAsync");
            void* device_pointer = nullptr;
            child_result = allocate != nullptr && runtime_handle != nullptr &&
                                   allocate(&device_pointer, 128, nullptr) == cudaErrorUnknown &&
                                   device_pointer == nullptr
                               ? EXIT_SUCCESS
                               : EXIT_FAILURE;
        } else {
            const InitFunction init = resolve_default<InitFunction>("cuInit");
            const AsyncAllocFunction allocate =
                resolve_default<AsyncAllocFunction>("cuMemAllocAsync");
            CUdeviceptr device_pointer = 0;
            child_result = init != nullptr && allocate != nullptr && init(0) == CUDA_SUCCESS &&
                                   allocate(&device_pointer, 128, nullptr) == CUDA_ERROR_UNKNOWN &&
                                   device_pointer == 0
                               ? EXIT_SUCCESS
                               : EXIT_FAILURE;
        }
        write_result(result_pipe[1], child_result);
        char release_signal = 0;
        if (::read(release_pipe[0], &release_signal, sizeof(release_signal)) !=
            static_cast<ssize_t>(sizeof(release_signal))) {
            _exit(EXIT_FAILURE);
        }
        _exit(child_result);
    }

    (void)::close(result_pipe[1]);
    (void)::close(release_pipe[0]);
    int child_result = EXIT_FAILURE;
    const bool child_reported = read_result(result_pipe[0], &child_result);
    (void)::close(result_pipe[0]);

    auto quota =
        glimmer::control::SharedMemoryQuota::open(glimmer::control::SharedMemoryQuotaConfig{
            .tenant_id = tenant_id, .device = 0, .limit_bytes = kQuotaBytes});
    const bool reservation_retained = quota != nullptr && quota->usage(0).reserved_bytes == 128 &&
                                      quota->usage(0).allocated_bytes == 0;

    const char release_signal = 0;
    const bool release_sent = ::write(release_pipe[1], &release_signal, sizeof(release_signal)) ==
                              static_cast<ssize_t>(sizeof(release_signal));
    (void)::close(release_pipe[1]);
    int child_status = 0;
    const bool child_exited = ::waitpid(child_pid, &child_status, 0) == child_pid;

    quota.reset();
    auto recovered_quota =
        glimmer::control::SharedMemoryQuota::open(glimmer::control::SharedMemoryQuotaConfig{
            .tenant_id = tenant_id, .device = 0, .limit_bytes = kQuotaBytes});
    const bool reservation_recovered =
        recovered_quota != nullptr && recovered_quota->usage(0).reserved_bytes == 0;
    recovered_quota.reset();
    const bool region_removed = glimmer::control::SharedMemoryQuota::remove_region(tenant_id);

    return expect(child_reported && child_result == EXIT_SUCCESS,
                  "async null-pointer fault injection did not return the expected error") &&
                   expect(reservation_retained,
                          "async null-pointer cleanup released an ambiguous reservation") &&
                   expect(release_sent && child_exited && WIFEXITED(child_status),
                          "null-pointer test child did not exit cleanly") &&
                   expect(reservation_recovered,
                          "dead null-pointer test process reservation was not recovered") &&
                   expect(region_removed, "null-pointer test region could not be removed")
               ? EXIT_SUCCESS
               : EXIT_FAILURE;
}

int run_runtime_untracked_initialization_test() {
    void* runtime_handle = dlopen("libcudart.so", RTLD_NOW | RTLD_GLOBAL);
    const RuntimeIpcOpenMemHandleFunction open_handle =
        resolve_default<RuntimeIpcOpenMemHandleFunction>("cudaIpcOpenMemHandle");
    const RuntimeMallocArrayFunction allocate_array =
        resolve_default<RuntimeMallocArrayFunction>("cudaMallocArray");
    if (!expect(runtime_handle != nullptr && open_handle != nullptr && allocate_array != nullptr,
                "initial Runtime untracked-path test symbols were not exported")) {
        if (runtime_handle != nullptr) {
            dlclose(runtime_handle);
        }
        return EXIT_FAILURE;
    }

    void* imported_pointer = nullptr;
    cudaIpcMemHandle_t handle{};
    const bool ipc_rejected = open_handle(&imported_pointer, handle, 0) == cudaErrorNotSupported &&
                              imported_pointer == nullptr;
    cudaChannelFormatDesc descriptor{};
    cudaArray_t array = nullptr;
    const bool array_rejected =
        allocate_array(&array, &descriptor, 1, 1, 0) == cudaErrorNotSupported && array == nullptr;
    dlclose(runtime_handle);
    return expect(ipc_rejected && array_rejected,
                  "Runtime untracked paths bypassed quota before initialization")
               ? EXIT_SUCCESS
               : EXIT_FAILURE;
}

}  // namespace

int main() {
    if (std::getenv("GLIMMER_FAKE_NVML_ONLY") != nullptr) {
        return run_nvml_only_virtualization_test();
    }
    if (std::getenv("GLIMMER_FAKE_PHYSICAL_CAP") != nullptr) {
        return run_physical_capacity_test();
    }
    if (std::getenv("GLIMMER_FAKE_TASK_LIMIT") != nullptr) {
        return run_task_memory_limit_test();
    }
    if (std::getenv("GLIMMER_FAKE_DUPLICATE_POINTER") != nullptr) {
        return run_duplicate_record_failure_test();
    }
    if (std::getenv("GLIMMER_FAKE_NULL_SUCCESS_POINTER") != nullptr) {
        return run_async_null_pointer_test(std::getenv("GLIMMER_NULL_SUCCESS_RUNTIME") != nullptr);
    }
    if (std::getenv("GLIMMER_FAKE_RUNTIME_UNTRACKED_FIRST") != nullptr) {
        return run_runtime_untracked_initialization_test();
    }

    bool all_passed = true;
    const bool shared_mode = [] {
        const char* value = std::getenv("GLIMMER_QUOTA_MODE");
        return value != nullptr && std::strcmp(value, "shared") == 0;
    }();
    std::string shared_tenant_id;
    if (shared_mode) {
        const char* configured_tenant = std::getenv("GLIMMER_QUOTA_TENANT_ID");
        all_passed &= expect(configured_tenant != nullptr && *configured_tenant != '\0',
                             "shared test tenant was not configured");
        if (configured_tenant != nullptr && *configured_tenant != '\0') {
            shared_tenant_id = std::string(configured_tenant) + "-" +
                               std::to_string(static_cast<long long>(::getpid()));
            all_passed &=
                expect(::setenv("GLIMMER_QUOTA_TENANT_ID", shared_tenant_id.c_str(), 1) == 0,
                       "shared test tenant could not be isolated");
        }
    }
    const InitFunction init = resolve_default<InitFunction>("cuInit");
    const LaunchKernelFunction launch_kernel =
        resolve_default<LaunchKernelFunction>("cuLaunchKernel");
    const LaunchKernelFunction launch_kernel_ptsz =
        resolve_default<LaunchKernelFunction>("cuLaunchKernel_ptsz");
    const GraphLaunchFunction graph_launch = resolve_default<GraphLaunchFunction>("cuGraphLaunch");
    const GraphLaunchFunction graph_launch_ptsz =
        resolve_default<GraphLaunchFunction>("cuGraphLaunch_ptsz");
    const AllocFunction allocate = resolve_default<AllocFunction>("cuMemAlloc_v2");
    const FreeFunction release = resolve_default<FreeFunction>("cuMemFree_v2");
    const AsyncAllocFunction async_allocate =
        resolve_default<AsyncAllocFunction>("cuMemAllocAsync");
    const AsyncAllocFunction async_allocate_ptsz =
        resolve_default<AsyncAllocFunction>("cuMemAllocAsync_ptsz");
    const PoolAsyncAllocFunction pool_async_allocate =
        resolve_default<PoolAsyncAllocFunction>("cuMemAllocFromPoolAsync");
    const PoolAsyncAllocFunction pool_async_allocate_ptsz =
        resolve_default<PoolAsyncAllocFunction>("cuMemAllocFromPoolAsync_ptsz");
    const VmmCreateFunction vmm_create = resolve_default<VmmCreateFunction>("cuMemCreate");
    const VmmReleaseFunction vmm_release = resolve_default<VmmReleaseFunction>("cuMemRelease");
    const AddressReserveFunction address_reserve =
        resolve_default<AddressReserveFunction>("cuMemAddressReserve");
    const AddressFreeFunction address_free =
        resolve_default<AddressFreeFunction>("cuMemAddressFree");
    const MapFunction map = resolve_default<MapFunction>("cuMemMap");
    const MapArrayAsyncFunction map_array_async =
        resolve_default<MapArrayAsyncFunction>("cuMemMapArrayAsync");
    const UnmapFunction unmap = resolve_default<UnmapFunction>("cuMemUnmap");
    const SetAccessFunction set_access = resolve_default<SetAccessFunction>("cuMemSetAccess");
    const GetAddressRangeFunction get_address_range =
        resolve_default<GetAddressRangeFunction>("cuMemGetAddressRange_v2");
    const GetAccessFunction get_access = resolve_default<GetAccessFunction>("cuMemGetAccess");
    const ExportHandleFunction export_handle =
        resolve_default<ExportHandleFunction>("cuMemExportToShareableHandle");
    const ImportHandleFunction import_handle =
        resolve_default<ImportHandleFunction>("cuMemImportFromShareableHandle");
    const IpcGetMemHandleFunction ipc_get_handle =
        resolve_default<IpcGetMemHandleFunction>("cuIpcGetMemHandle");
    const IpcOpenMemHandleFunction ipc_open_handle =
        resolve_default<IpcOpenMemHandleFunction>("cuIpcOpenMemHandle");
    const IpcOpenMemHandleFunction ipc_open_handle_v2 =
        resolve_default<IpcOpenMemHandleFunction>("cuIpcOpenMemHandle_v2");
    const IpcCloseMemHandleFunction ipc_close_handle =
        resolve_default<IpcCloseMemHandleFunction>("cuIpcCloseMemHandle");
    const ImportExternalMemoryFunction import_external_memory =
        resolve_default<ImportExternalMemoryFunction>("cuImportExternalMemory");
    const ExternalMemoryGetMappedBufferFunction get_external_buffer =
        resolve_default<ExternalMemoryGetMappedBufferFunction>("cuExternalMemoryGetMappedBuffer");
    const ExternalMemoryGetMappedMipmappedArrayFunction get_external_mipmap =
        resolve_default<ExternalMemoryGetMappedMipmappedArrayFunction>(
            "cuExternalMemoryGetMappedMipmappedArray");
    const DestroyExternalMemoryFunction destroy_external_memory =
        resolve_default<DestroyExternalMemoryFunction>("cuDestroyExternalMemory");
    const ArrayCreateFunction array_create = resolve_default<ArrayCreateFunction>("cuArrayCreate");
    const ArrayCreateFunction array_create_v2 =
        resolve_default<ArrayCreateFunction>("cuArrayCreate_v2");
    const Array3DCreateFunction array_3d_create =
        resolve_default<Array3DCreateFunction>("cuArray3DCreate");
    const ArrayDestroyFunction array_destroy =
        resolve_default<ArrayDestroyFunction>("cuArrayDestroy");
    const MipmappedArrayCreateFunction mipmapped_array_create =
        resolve_default<MipmappedArrayCreateFunction>("cuMipmappedArrayCreate");
    const MipmappedArrayDestroyFunction mipmapped_array_destroy =
        resolve_default<MipmappedArrayDestroyFunction>("cuMipmappedArrayDestroy");
    const GraphicsUnregisterResourceFunction graphics_unregister_resource =
        resolve_default<GraphicsUnregisterResourceFunction>("cuGraphicsUnregisterResource");
    const GraphicsSubResourceGetMappedArrayFunction graphics_get_mapped_array =
        resolve_default<GraphicsSubResourceGetMappedArrayFunction>(
            "cuGraphicsSubResourceGetMappedArray");
    const GraphicsResourceGetMappedMipmappedArrayFunction graphics_get_mapped_mipmap =
        resolve_default<GraphicsResourceGetMappedMipmappedArrayFunction>(
            "cuGraphicsResourceGetMappedMipmappedArray");
    const GraphicsResourceGetMappedPointerFunction graphics_get_mapped_pointer =
        resolve_default<GraphicsResourceGetMappedPointerFunction>(
            "cuGraphicsResourceGetMappedPointer_v2");
    const GraphicsResourceSetMapFlagsFunction graphics_set_map_flags =
        resolve_default<GraphicsResourceSetMapFlagsFunction>("cuGraphicsResourceSetMapFlags_v2");
    const GraphicsMapResourcesFunction graphics_map_resources =
        resolve_default<GraphicsMapResourcesFunction>("cuGraphicsMapResources");
    const GraphicsUnmapResourcesFunction graphics_unmap_resources =
        resolve_default<GraphicsUnmapResourcesFunction>("cuGraphicsUnmapResources");
    const GetGranularityFunction get_granularity =
        resolve_default<GetGranularityFunction>("cuMemGetAllocationGranularity");
    const GetPropertiesFunction get_properties =
        resolve_default<GetPropertiesFunction>("cuMemGetAllocationPropertiesFromHandle");
    const RetainHandleFunction retain_handle =
        resolve_default<RetainHandleFunction>("cuMemRetainAllocationHandle");
    const PoolTrimFunction pool_trim = resolve_default<PoolTrimFunction>("cuMemPoolTrimTo");
    const PoolSetAttributeFunction pool_set_attribute =
        resolve_default<PoolSetAttributeFunction>("cuMemPoolSetAttribute");
    const PoolGetAttributeFunction pool_get_attribute =
        resolve_default<PoolGetAttributeFunction>("cuMemPoolGetAttribute");
    const PoolSetAccessFunction pool_set_access =
        resolve_default<PoolSetAccessFunction>("cuMemPoolSetAccess");
    const PoolGetAccessFunction pool_get_access =
        resolve_default<PoolGetAccessFunction>("cuMemPoolGetAccess");
    const PoolCreateFunction pool_create = resolve_default<PoolCreateFunction>("cuMemPoolCreate");
    const PoolDestroyFunction pool_destroy =
        resolve_default<PoolDestroyFunction>("cuMemPoolDestroy");
    const DeviceGetPoolFunction device_get_pool =
        resolve_default<DeviceGetPoolFunction>("cuDeviceGetMemPool");
    const DeviceSetPoolFunction device_set_pool =
        resolve_default<DeviceSetPoolFunction>("cuDeviceSetMemPool");
    const DeviceGetDefaultPoolFunction device_get_default_pool =
        resolve_default<DeviceGetDefaultPoolFunction>("cuDeviceGetDefaultMemPool");
    const GetDefaultPoolFunction get_default_pool =
        resolve_default<GetDefaultPoolFunction>("cuMemGetDefaultMemPool");
    const GetPoolFunction get_pool = resolve_default<GetPoolFunction>("cuMemGetMemPool");
    const SetPoolFunction set_pool = resolve_default<SetPoolFunction>("cuMemSetMemPool");
    const PoolExportHandleFunction pool_export_handle =
        resolve_default<PoolExportHandleFunction>("cuMemPoolExportToShareableHandle");
    const PoolImportHandleFunction pool_import_handle =
        resolve_default<PoolImportHandleFunction>("cuMemPoolImportFromShareableHandle");
    const PoolExportPointerFunction pool_export_pointer =
        resolve_default<PoolExportPointerFunction>("cuMemPoolExportPointer");
    const PoolImportPointerFunction pool_import_pointer =
        resolve_default<PoolImportPointerFunction>("cuMemPoolImportPointer");
    const AsyncFreeFunction async_release = resolve_default<AsyncFreeFunction>("cuMemFreeAsync");
    const AsyncFreeFunction async_release_ptsz =
        resolve_default<AsyncFreeFunction>("cuMemFreeAsync_ptsz");
    const StreamGetDeviceFunction stream_get_device =
        resolve_default<StreamGetDeviceFunction>("cuStreamGetDevice");
    const StreamGetDeviceFunction stream_get_device_ptsz =
        resolve_default<StreamGetDeviceFunction>("cuStreamGetDevice_ptsz");
    const StreamGetContextFunction stream_get_context =
        resolve_default<StreamGetContextFunction>("cuStreamGetCtx");
    const StreamGetContextFunction stream_get_context_ptsz =
        resolve_default<StreamGetContextFunction>("cuStreamGetCtx_ptsz");
    const StreamQueryFunction stream_query = resolve_default<StreamQueryFunction>("cuStreamQuery");
    const StreamQueryFunction stream_query_ptsz =
        resolve_default<StreamQueryFunction>("cuStreamQuery_ptsz");
    const StreamSynchronizeFunction stream_synchronize =
        resolve_default<StreamSynchronizeFunction>("cuStreamSynchronize");
    const StreamSynchronizeFunction stream_synchronize_ptsz =
        resolve_default<StreamSynchronizeFunction>("cuStreamSynchronize_ptsz");
    const StreamDestroyFunction stream_destroy =
        resolve_default<StreamDestroyFunction>("cuStreamDestroy_v2");
    const ContextSynchronizeFunction context_synchronize =
        resolve_default<ContextSynchronizeFunction>("cuCtxSynchronize");
    const MemGetInfoFunction get_info = resolve_default<MemGetInfoFunction>("cuMemGetInfo_v2");
    const DeviceTotalMemFunction get_total =
        resolve_default<DeviceTotalMemFunction>("cuDeviceTotalMem_v2");
    const ContextGetCurrentFunction get_current =
        resolve_default<ContextGetCurrentFunction>("cuCtxGetCurrent");
    const ContextDestroyFunction destroy_context =
        resolve_default<ContextDestroyFunction>("cuCtxDestroy_v2");
    const NvmlInitFunction nvml_init = resolve_default<NvmlInitFunction>("nvmlInit_v2");
    const NvmlShutdownFunction nvml_shutdown =
        resolve_default<NvmlShutdownFunction>("nvmlShutdown");
    const NvmlDeviceGetCountFunction nvml_get_count =
        resolve_default<NvmlDeviceGetCountFunction>("nvmlDeviceGetCount_v2");
    const NvmlDeviceGetHandleByIndexFunction nvml_get_handle =
        resolve_default<NvmlDeviceGetHandleByIndexFunction>("nvmlDeviceGetHandleByIndex_v2");
    const NvmlDeviceGetIndexFunction nvml_get_index =
        resolve_default<NvmlDeviceGetIndexFunction>("nvmlDeviceGetIndex");
    const NvmlDeviceGetMemoryInfoFunction nvml_get_memory_info =
        resolve_default<NvmlDeviceGetMemoryInfoFunction>("nvmlDeviceGetMemoryInfo");
    const NvmlDeviceGetMemoryInfoV2Function nvml_get_memory_info_v2 =
        resolve_default<NvmlDeviceGetMemoryInfoV2Function>("nvmlDeviceGetMemoryInfo_v2");
    all_passed &= expect(
        init != nullptr && launch_kernel != nullptr && launch_kernel_ptsz != nullptr &&
            graph_launch != nullptr && graph_launch_ptsz != nullptr && allocate != nullptr &&
            release != nullptr && async_allocate != nullptr && async_allocate_ptsz != nullptr &&
            pool_async_allocate != nullptr && pool_async_allocate_ptsz != nullptr &&
            vmm_create != nullptr && vmm_release != nullptr && address_reserve != nullptr &&
            address_free != nullptr && map != nullptr && map_array_async != nullptr &&
            unmap != nullptr && set_access != nullptr && get_address_range != nullptr &&
            get_access != nullptr && export_handle != nullptr && import_handle != nullptr &&
            ipc_get_handle != nullptr && ipc_open_handle != nullptr &&
            ipc_open_handle_v2 != nullptr && ipc_close_handle != nullptr &&
            get_granularity != nullptr && import_external_memory != nullptr &&
            get_external_buffer != nullptr && get_external_mipmap != nullptr &&
            destroy_external_memory != nullptr && array_create != nullptr &&
            array_create_v2 != nullptr && array_3d_create != nullptr && array_destroy != nullptr &&
            mipmapped_array_create != nullptr && mipmapped_array_destroy != nullptr &&
            graphics_unregister_resource != nullptr && graphics_get_mapped_array != nullptr &&
            graphics_get_mapped_mipmap != nullptr && graphics_get_mapped_pointer != nullptr &&
            graphics_set_map_flags != nullptr && graphics_map_resources != nullptr &&
            graphics_unmap_resources != nullptr && get_properties != nullptr &&
            retain_handle != nullptr && pool_trim != nullptr && pool_set_attribute != nullptr &&
            pool_get_attribute != nullptr && pool_set_access != nullptr &&
            pool_get_access != nullptr && pool_create != nullptr && pool_destroy != nullptr &&
            device_get_pool != nullptr && device_set_pool != nullptr &&
            device_get_default_pool != nullptr && get_default_pool != nullptr &&
            get_pool != nullptr && set_pool != nullptr && pool_export_handle != nullptr &&
            pool_import_handle != nullptr && pool_export_pointer != nullptr &&
            pool_import_pointer != nullptr && async_release != nullptr &&
            async_release_ptsz != nullptr && stream_get_device != nullptr &&
            stream_get_device_ptsz != nullptr && stream_get_context != nullptr &&
            stream_get_context_ptsz != nullptr && stream_query != nullptr &&
            stream_query_ptsz != nullptr && stream_synchronize != nullptr &&
            stream_synchronize_ptsz != nullptr && stream_destroy != nullptr &&
            context_synchronize != nullptr && get_info != nullptr && get_total != nullptr &&
            get_current != nullptr && destroy_context != nullptr && nvml_init != nullptr &&
            nvml_shutdown != nullptr && nvml_get_count != nullptr && nvml_get_handle != nullptr &&
            nvml_get_index != nullptr && nvml_get_memory_info != nullptr &&
            nvml_get_memory_info_v2 != nullptr,
        "interceptor symbols were not exported");
    if (!all_passed) {
        return EXIT_FAILURE;
    }

    all_passed &= expect(init(0) == CUDA_SUCCESS, "fake cuInit failed");
    all_passed &= expect(
        launch_kernel(nullptr, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, nullptr) == CUDA_SUCCESS,
        "fake cuLaunchKernel forwarding failed");
    all_passed &= expect(
        launch_kernel_ptsz(nullptr, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, nullptr) == CUDA_SUCCESS,
        "fake cuLaunchKernel_ptsz forwarding failed");
    all_passed &=
        expect(graph_launch(reinterpret_cast<CUgraphExec>(0x76000000U), nullptr) == CUDA_SUCCESS,
               "fake cuGraphLaunch forwarding failed");
    all_passed &= expect(
        graph_launch_ptsz(reinterpret_cast<CUgraphExec>(0x76000000U), nullptr) == CUDA_SUCCESS,
        "fake cuGraphLaunch_ptsz forwarding failed");
    all_passed &= expect(graph_launch(nullptr, nullptr) == CUDA_ERROR_INVALID_VALUE &&
                             graph_launch_ptsz(nullptr, nullptr) == CUDA_ERROR_INVALID_VALUE,
                         "null graph launch handles were not rejected");

    std::size_t total_bytes = 0;
    all_passed &= expect(allocate(nullptr, 1) == CUDA_ERROR_INVALID_VALUE,
                         "null Driver allocation output was not rejected");
    all_passed &= expect(async_allocate(nullptr, 1, nullptr) == CUDA_ERROR_INVALID_VALUE,
                         "null async Driver allocation output was not rejected");
    all_passed &= expect(async_allocate_ptsz(nullptr, 1, nullptr) == CUDA_ERROR_INVALID_VALUE,
                         "null PTDS async Driver allocation output was not rejected");
    all_passed &=
        expect(pool_async_allocate(nullptr, 1, nullptr, nullptr) == CUDA_ERROR_INVALID_VALUE,
               "null memory-pool allocation output was not rejected");
    all_passed &=
        expect(pool_async_allocate_ptsz(nullptr, 1, nullptr, nullptr) == CUDA_ERROR_INVALID_VALUE,
               "null PTDS memory-pool allocation output was not rejected");
    all_passed &= expect(get_info(nullptr, &total_bytes) == CUDA_ERROR_INVALID_VALUE,
                         "null Driver free-memory output was not rejected");
    all_passed &= expect(get_total(nullptr, 0) == CUDA_ERROR_INVALID_VALUE,
                         "null Driver total-memory output was not rejected");
    all_passed &= expect(stream_get_device(nullptr, nullptr) == CUDA_ERROR_INVALID_VALUE,
                         "null stream device output was not rejected");
    all_passed &= expect(stream_get_context(nullptr, nullptr) == CUDA_ERROR_INVALID_VALUE,
                         "null stream context output was not rejected");
    all_passed &= expect(get_current(nullptr) == CUDA_ERROR_INVALID_VALUE,
                         "null current-context output was not rejected");
    all_passed &= expect(get_total(&total_bytes, 0) == CUDA_SUCCESS && total_bytes == kQuotaBytes,
                         "device total memory was not virtualized without a current-context query");

    std::size_t free_bytes = 0;
    all_passed &= expect(get_info(&free_bytes, &total_bytes) == CUDA_SUCCESS &&
                             free_bytes == kQuotaBytes && total_bytes == kQuotaBytes,
                         "initial memory info was incorrect");

    CUipcMemHandle ipc_handle{};
    all_passed &= expect(ipc_get_handle(&ipc_handle, 0x100000U) == CUDA_SUCCESS,
                         "IPC handle export was not forwarded");
    CUdeviceptr imported_ipc_pointer = 0;
    all_passed &=
        expect(ipc_open_handle(&imported_ipc_pointer, ipc_handle, 0) == CUDA_ERROR_NOT_SUPPORTED &&
                   imported_ipc_pointer == 0,
               "IPC memory import was not rejected under quota");
    all_passed &=
        expect(ipc_close_handle(0x60000000U) == CUDA_SUCCESS, "IPC memory close was not forwarded");

    CUDA_EXTERNAL_MEMORY_HANDLE_DESC external_handle_desc{};
    CUexternalMemory external_memory = nullptr;
    all_passed &= expect(import_external_memory(&external_memory, &external_handle_desc) ==
                                 CUDA_ERROR_NOT_SUPPORTED &&
                             external_memory == nullptr,
                         "external memory import was not rejected under quota");
    CUDA_EXTERNAL_MEMORY_BUFFER_DESC external_buffer_desc{};
    CUdeviceptr external_device_pointer = 0;
    all_passed &= expect(get_external_buffer(&external_device_pointer,
                                             reinterpret_cast<CUexternalMemory>(0x62000000U),
                                             &external_buffer_desc) == CUDA_ERROR_NOT_SUPPORTED &&
                             external_device_pointer == 0,
                         "external memory buffer mapping was not rejected under quota");
    CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC external_mipmap_desc{};
    CUmipmappedArray external_mipmap = nullptr;
    all_passed &= expect(
        get_external_mipmap(&external_mipmap, reinterpret_cast<CUexternalMemory>(0x62000000U),
                            &external_mipmap_desc) == CUDA_ERROR_NOT_SUPPORTED &&
            external_mipmap == nullptr,
        "external memory mipmap mapping was not rejected under quota");
    all_passed &= expect(
        destroy_external_memory(reinterpret_cast<CUexternalMemory>(0x62000000U)) == CUDA_SUCCESS,
        "external memory destruction was not forwarded");
    CUDA_ARRAY_DESCRIPTOR array_descriptor{};
    CUarray array = nullptr;
    all_passed &= expect(
        array_create(&array, &array_descriptor) == CUDA_ERROR_NOT_SUPPORTED && array == nullptr,
        "CUDA array allocation was not rejected under quota");
    CUDA_ARRAY3D_DESCRIPTOR array_3d_descriptor{};
    CUarray array_3d = nullptr;
    all_passed &=
        expect(array_3d_create(&array_3d, &array_3d_descriptor) == CUDA_ERROR_NOT_SUPPORTED &&
                   array_3d == nullptr,
               "CUDA 3D array allocation was not rejected under quota");
    CUmipmappedArray mipmapped_array = nullptr;
    all_passed &= expect(mipmapped_array_create(&mipmapped_array, &array_3d_descriptor, 1) ==
                                 CUDA_ERROR_NOT_SUPPORTED &&
                             mipmapped_array == nullptr,
                         "CUDA mipmapped array allocation was not rejected under quota");
    all_passed &= expect(array_destroy(reinterpret_cast<CUarray>(0x63000000U)) == CUDA_SUCCESS,
                         "CUDA array destruction was not forwarded");
    all_passed &= expect(
        mipmapped_array_destroy(reinterpret_cast<CUmipmappedArray>(0x63000000U)) == CUDA_SUCCESS,
        "CUDA mipmapped array destruction was not forwarded");

    CUarrayMapInfo sparse_map_info{};
    all_passed &= expect(map_array_async(&sparse_map_info, 1, nullptr) == CUDA_ERROR_NOT_SUPPORTED,
                         "CUDA sparse array mapping was not rejected under quota");

    CUgraphicsResource graphics_resource = reinterpret_cast<CUgraphicsResource>(0x64000000U);
    CUdeviceptr graphics_pointer = 0;
    std::size_t graphics_size = 0;
    all_passed &=
        expect(graphics_map_resources(1, &graphics_resource, nullptr) == CUDA_ERROR_NOT_SUPPORTED,
               "CUDA graphics mapping was not rejected under quota");
    all_passed &=
        expect(graphics_get_mapped_pointer(&graphics_pointer, &graphics_size, graphics_resource) ==
                       CUDA_ERROR_NOT_SUPPORTED &&
                   graphics_pointer == 0 && graphics_size == 0,
               "CUDA graphics mapped pointer was not rejected under quota");
    CUarray graphics_array = nullptr;
    all_passed &= expect(graphics_get_mapped_array(&graphics_array, graphics_resource, 0, 0) ==
                                 CUDA_ERROR_NOT_SUPPORTED &&
                             graphics_array == nullptr,
                         "CUDA graphics mapped array was not rejected under quota");
    CUmipmappedArray graphics_mipmap = nullptr;
    all_passed &= expect(graphics_get_mapped_mipmap(&graphics_mipmap, graphics_resource) ==
                                 CUDA_ERROR_NOT_SUPPORTED &&
                             graphics_mipmap == nullptr,
                         "CUDA graphics mapped mipmap was not rejected under quota");
    all_passed &= expect(graphics_set_map_flags(graphics_resource, 0) == CUDA_SUCCESS,
                         "CUDA graphics map flags were not forwarded");
    all_passed &= expect(graphics_unmap_resources(1, &graphics_resource, nullptr) == CUDA_SUCCESS,
                         "CUDA graphics unmapping was not forwarded");
    all_passed &= expect(graphics_unregister_resource(graphics_resource) == CUDA_SUCCESS,
                         "CUDA graphics resource destruction was not forwarded");

    nvmlDevice_t nvml_device = nullptr;
    unsigned int nvml_count = 0;
    unsigned int nvml_index = 0;
    nvmlMemory_t nvml_memory{};
    nvmlMemory_v2_t nvml_memory_v2{};
    nvml_memory_v2.version = nvmlMemory_v2;
    const bool nvml_initialized = nvml_init() == NVML_SUCCESS;
    const bool nvml_visible =
        nvml_initialized && nvml_get_count(&nvml_count) == NVML_SUCCESS && nvml_count == 2 &&
        nvml_get_handle(0, &nvml_device) == NVML_SUCCESS &&
        nvml_get_index(nvml_device, &nvml_index) == NVML_SUCCESS && nvml_index == 0 &&
        nvml_get_memory_info(nvml_device, &nvml_memory) == NVML_SUCCESS &&
        nvml_memory.total == kQuotaBytes && nvml_memory.free == kQuotaBytes &&
        nvml_memory.used == 0 &&
        nvml_get_memory_info_v2(nvml_device, &nvml_memory_v2) == NVML_SUCCESS &&
        nvml_memory_v2.total == kQuotaBytes && nvml_memory_v2.free == kQuotaBytes &&
        nvml_memory_v2.used == 0 && nvml_memory_v2.reserved == 0;
    all_passed &= expect(nvml_visible, "NVML memory view was not virtualized");

    CUdeviceptr direct_pointer = 0;
    all_passed &= expect(allocate(&direct_pointer, kDirectAllocationBytes) == CUDA_SUCCESS,
                         "direct allocation was rejected");
    all_passed &= expect(get_info(&free_bytes, &total_bytes) == CUDA_SUCCESS &&
                             free_bytes == kQuotaBytes - kDirectAllocationBytes,
                         "direct allocation was not accounted once");
    all_passed &= expect(nvml_get_memory_info(nvml_device, &nvml_memory) == NVML_SUCCESS &&
                             nvml_memory.total == kQuotaBytes &&
                             nvml_memory.free == kQuotaBytes - kDirectAllocationBytes &&
                             nvml_memory.used == kDirectAllocationBytes,
                         "NVML memory view did not follow quota usage");

    CUdeviceptr rejected_pointer = 0;
    all_passed &=
        expect(allocate(&rejected_pointer, kRejectedAllocationBytes) == CUDA_ERROR_OUT_OF_MEMORY &&
                   rejected_pointer == 0,
               "quota rejection did not preserve the pointer and usage");
    all_passed &=
        expect(release(direct_pointer) == CUDA_SUCCESS, "direct allocation was not freed");
    all_passed &=
        expect(get_info(&free_bytes, &total_bytes) == CUDA_SUCCESS && free_bytes == kQuotaBytes,
               "direct release did not restore quota");

    CUdeviceptr failed_pointer = 0;
    all_passed &= expect(
        allocate(&failed_pointer, kForcedAllocationFailureBytes) == CUDA_ERROR_INVALID_VALUE &&
            get_info(&free_bytes, &total_bytes) == CUDA_SUCCESS && free_bytes == kQuotaBytes,
        "real Driver allocation failure did not roll back the reservation");

    CUmemAllocationProp vmm_prop{};
    vmm_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    vmm_prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;
    vmm_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    vmm_prop.location.id = 0;
    CUmemGenericAllocationHandle vmm_handle = 0;
    all_passed &= expect(vmm_create(nullptr, 1, &vmm_prop, 0) == CUDA_ERROR_INVALID_VALUE,
                         "null VMM handle output was not rejected");
    all_passed &= expect(vmm_create(&vmm_handle, 1, nullptr, 0) == CUDA_ERROR_INVALID_VALUE,
                         "null VMM allocation properties were not rejected");
    all_passed &=
        expect(vmm_create(&vmm_handle, kAsyncAllocationBytes, &vmm_prop, 0) == CUDA_SUCCESS &&
                   get_info(&free_bytes, &total_bytes) == CUDA_SUCCESS &&
                   free_bytes == kQuotaBytes - kAsyncAllocationBytes,
               "VMM allocation was not admitted and accounted");
    CUdeviceptr virtual_address = 0;
    all_passed &=
        expect(address_reserve(nullptr, kAsyncAllocationBytes, 0, 0, 0) == CUDA_ERROR_INVALID_VALUE,
               "null VMM address output was not rejected");
    all_passed &=
        expect(address_reserve(&virtual_address, kAsyncAllocationBytes, 0, 0, 0) == CUDA_SUCCESS &&
                   virtual_address != 0,
               "VMM address reservation was not forwarded");
    all_passed &=
        expect(map(virtual_address, kAsyncAllocationBytes, 0, vmm_handle, 0) == CUDA_SUCCESS,
               "VMM address mapping was not forwarded");
    CUmemAccessDesc access_descriptor{};
    access_descriptor.location = vmm_prop.location;
    access_descriptor.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    all_passed &= expect(
        set_access(virtual_address, kAsyncAllocationBytes, &access_descriptor, 1) == CUDA_SUCCESS,
        "VMM access permissions were not forwarded");
    all_passed &= expect(
        set_access(virtual_address, kAsyncAllocationBytes, nullptr, 1) == CUDA_ERROR_INVALID_VALUE,
        "null VMM access descriptors were not rejected");
    CUdeviceptr address_base = 0;
    std::size_t address_size = 0;
    all_passed &=
        expect(get_address_range(&address_base, &address_size, virtual_address) == CUDA_SUCCESS &&
                   address_base == virtual_address && address_size != 0,
               "VMM address range query was not forwarded");
    unsigned long long access_flags = 0;
    all_passed &= expect(
        get_access(&access_flags, &access_descriptor.location, virtual_address) == CUDA_SUCCESS &&
            access_flags == CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
        "VMM access query was not forwarded");
    unsigned long long exported_handle = 0;
    all_passed &= expect(
        export_handle(&exported_handle, vmm_handle, CU_MEM_HANDLE_TYPE_NONE, 0) == CUDA_SUCCESS,
        "VMM shareable-handle export was not forwarded");
    CUmemGenericAllocationHandle imported_handle = 0;
    all_passed &= expect(import_handle(&imported_handle, &exported_handle,
                                       CU_MEM_HANDLE_TYPE_NONE) == CUDA_ERROR_NOT_SUPPORTED &&
                             imported_handle == 0,
                         "VMM shareable-handle import was not rejected under quota");
    std::size_t granularity = 0;
    all_passed &= expect(get_granularity(&granularity, &vmm_prop,
                                         CU_MEM_ALLOC_GRANULARITY_MINIMUM) == CUDA_SUCCESS &&
                             granularity != 0,
                         "VMM allocation granularity query was not forwarded");
    CUmemAllocationProp queried_properties{};
    all_passed &= expect(get_properties(&queried_properties, vmm_handle) == CUDA_SUCCESS &&
                             queried_properties.location.type == CU_MEM_LOCATION_TYPE_DEVICE,
                         "VMM allocation properties query was not forwarded");
    CUmemGenericAllocationHandle retained_handle = 0;
    void* retained_pointer = std::bit_cast<void*>(virtual_address);
    all_passed &= expect(retain_handle(&retained_handle, retained_pointer) == CUDA_SUCCESS &&
                             retained_handle == vmm_handle,
                         "VMM allocation-handle retain was not forwarded and accounted");

    CUmemPoolProps pool_properties{};
    pool_properties.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
    pool_properties.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    pool_properties.location.id = 0;
    CUmemoryPool pool = nullptr;
    all_passed &= expect(pool_create(&pool, &pool_properties) == CUDA_SUCCESS && pool != nullptr,
                         "memory-pool creation was not forwarded");
    std::uint64_t release_threshold = 128;
    all_passed &= expect(pool_set_attribute(pool, CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                                            &release_threshold) == CUDA_SUCCESS,
                         "memory-pool attribute update was not forwarded");
    release_threshold = 0;
    all_passed &= expect(pool_get_attribute(pool, CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                                            &release_threshold) == CUDA_SUCCESS &&
                             release_threshold == 128,
                         "memory-pool attribute query was not forwarded");
    all_passed &= expect(pool_set_access(pool, &access_descriptor, 1) == CUDA_SUCCESS,
                         "memory-pool access update was not forwarded");
    CUmemAccess_flags pool_access_flags = CU_MEM_ACCESS_FLAGS_PROT_NONE;
    CUmemLocation pool_location = pool_properties.location;
    all_passed &=
        expect(pool_get_access(&pool_access_flags, pool, &pool_location) == CUDA_SUCCESS &&
                   pool_access_flags == CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
               "memory-pool access query was not forwarded");
    all_passed &= expect(pool_trim(pool, 0) == CUDA_SUCCESS, "memory-pool trim was not forwarded");
    CUmemoryPool queried_pool = nullptr;
    all_passed &=
        expect(device_get_pool(&queried_pool, 0) == CUDA_SUCCESS && queried_pool != nullptr,
               "device memory-pool query was not forwarded");
    all_passed &= expect(device_set_pool(0, pool) == CUDA_SUCCESS,
                         "device memory-pool update was not forwarded");
    all_passed &=
        expect(device_get_default_pool(&queried_pool, 0) == CUDA_SUCCESS && queried_pool != nullptr,
               "device default memory-pool query was not forwarded");
    all_passed &= expect(get_default_pool(&queried_pool, &pool_location,
                                          CU_MEM_ALLOCATION_TYPE_PINNED) == CUDA_SUCCESS,
                         "default memory-pool query was not forwarded");
    all_passed &= expect(
        get_pool(&queried_pool, &pool_location, CU_MEM_ALLOCATION_TYPE_PINNED) == CUDA_SUCCESS,
        "pointer memory-pool query was not forwarded");
    all_passed &=
        expect(set_pool(&pool_location, CU_MEM_ALLOCATION_TYPE_PINNED, pool) == CUDA_SUCCESS,
               "memory-pool selection was not forwarded");
    unsigned long long pool_exported_handle = 0;
    all_passed &= expect(
        pool_export_handle(&pool_exported_handle, pool, CU_MEM_HANDLE_TYPE_NONE, 0) == CUDA_SUCCESS,
        "memory-pool handle export was not forwarded");
    CUmemoryPool imported_pool = nullptr;
    all_passed &=
        expect(pool_import_handle(&imported_pool, &pool_exported_handle, CU_MEM_HANDLE_TYPE_NONE,
                                  0) == CUDA_ERROR_NOT_SUPPORTED &&
                   imported_pool == nullptr,
               "memory-pool handle import was not rejected under quota");
    CUmemPoolPtrExportData pointer_export_data{};
    all_passed &= expect(pool_export_pointer(&pointer_export_data, virtual_address) == CUDA_SUCCESS,
                         "memory-pool pointer export was not forwarded");
    CUdeviceptr imported_pointer = 0;
    all_passed &= expect(pool_import_pointer(&imported_pointer, pool, &pointer_export_data) ==
                                 CUDA_ERROR_NOT_SUPPORTED &&
                             imported_pointer == 0,
                         "memory-pool pointer import was not rejected under quota");
    all_passed &=
        expect(pool_destroy(pool) == CUDA_SUCCESS, "memory-pool destruction was not forwarded");
    all_passed &= expect(unmap(virtual_address, kAsyncAllocationBytes) == CUDA_SUCCESS,
                         "VMM address unmapping was not forwarded");
    all_passed &= expect(address_free(virtual_address, kAsyncAllocationBytes) == CUDA_SUCCESS,
                         "VMM address free was not forwarded");
    all_passed &= expect(vmm_release(retained_handle) == CUDA_SUCCESS &&
                             get_info(&free_bytes, &total_bytes) == CUDA_SUCCESS &&
                             free_bytes == kQuotaBytes - kAsyncAllocationBytes,
                         "first VMM release did not preserve shared allocation quota");
    all_passed &=
        expect(vmm_release(vmm_handle) == CUDA_SUCCESS &&
                   get_info(&free_bytes, &total_bytes) == CUDA_SUCCESS && free_bytes == kQuotaBytes,
               "final VMM release did not restore quota");
    vmm_handle = 0;
    all_passed &=
        expect(vmm_create(&vmm_handle, kForcedAllocationFailureBytes, &vmm_prop, 0) ==
                       CUDA_ERROR_INVALID_VALUE &&
                   get_info(&free_bytes, &total_bytes) == CUDA_SUCCESS && free_bytes == kQuotaBytes,
               "real VMM allocation failure did not roll back the reservation");
    vmm_handle = 0;
    all_passed &=
        expect(vmm_create(&vmm_handle, kQuotaBytes + 1, &vmm_prop, 0) == CUDA_ERROR_OUT_OF_MEMORY &&
                   vmm_handle == 0 && get_info(&free_bytes, &total_bytes) == CUDA_SUCCESS &&
                   free_bytes == kQuotaBytes,
               "VMM quota rejection did not preserve usage");
    vmm_prop.location.type = CU_MEM_LOCATION_TYPE_HOST;
    all_passed &=
        expect(vmm_create(&vmm_handle, kAsyncAllocationBytes, &vmm_prop, 0) == CUDA_SUCCESS &&
                   get_info(&free_bytes, &total_bytes) == CUDA_SUCCESS &&
                   free_bytes == kQuotaBytes && vmm_release(vmm_handle) == CUDA_SUCCESS,
               "host VMM allocation was incorrectly charged by the device quota");

    void* cuda_handle = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    all_passed &= expect(cuda_handle != nullptr, "fake CUDA driver could not be loaded");
    if (cuda_handle != nullptr) {
        all_passed &=
            expect(dlsym(cuda_handle, "cuMemAlloc_v2") == reinterpret_cast<void*>(allocate),
                   "explicit CUDA handle did not return the interceptor wrapper");
        all_passed &=
            expect(dlsym(cuda_handle, "cuMemCreate") == reinterpret_cast<void*>(vmm_create),
                   "explicit CUDA handle did not return the VMM wrapper");
        all_passed &=
            expect(dlsym(cuda_handle, "cuGraphLaunch") == reinterpret_cast<void*>(graph_launch),
                   "explicit CUDA handle did not return the graph launch wrapper");
        all_passed &= expect(
            dlsym(cuda_handle, "cuMemAddressReserve") == reinterpret_cast<void*>(address_reserve),
            "explicit CUDA handle did not return the VMM address wrapper");
        all_passed &=
            expect(dlsym(cuda_handle, "cuMemPoolCreate") == reinterpret_cast<void*>(pool_create),
                   "explicit CUDA handle did not return the memory-pool wrapper");
        all_passed &= expect(dlsym(cuda_handle, "cuDeviceGetDefaultMemPool") ==
                                 reinterpret_cast<void*>(device_get_default_pool),
                             "explicit CUDA handle did not return the default-pool wrapper");
        dlclose(cuda_handle);
    }
    void* nvml_handle = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
    all_passed &= expect(nvml_handle != nullptr, "fake NVML could not be loaded");
    if (nvml_handle != nullptr) {
        all_passed &= expect(dlsym(nvml_handle, "nvmlDeviceGetMemoryInfo") ==
                                 reinterpret_cast<void*>(nvml_get_memory_info),
                             "explicit NVML handle did not return the memory wrapper");
        dlclose(nvml_handle);
    }

    void* runtime_handle = dlopen("libcudart.so", RTLD_NOW | RTLD_GLOBAL);
    all_passed &= expect(runtime_handle != nullptr, "fake CUDA runtime could not be loaded");
    const RuntimeMallocFunction runtime_allocate =
        resolve_default<RuntimeMallocFunction>("cudaMalloc");
    const RuntimeMallocManagedFunction runtime_managed_allocate =
        resolve_default<RuntimeMallocManagedFunction>("cudaMallocManaged");
    const RuntimeMallocPitchFunction runtime_pitch_allocate =
        resolve_default<RuntimeMallocPitchFunction>("cudaMallocPitch");
    const RuntimeMalloc3DFunction runtime_3d_allocate =
        resolve_default<RuntimeMalloc3DFunction>("cudaMalloc3D");
    const RuntimeLaunchKernelFunction runtime_launch_kernel =
        resolve_default<RuntimeLaunchKernelFunction>("cudaLaunchKernel");
    const RuntimeLaunchKernelFunction runtime_launch_kernel_ptsz =
        resolve_default<RuntimeLaunchKernelFunction>("cudaLaunchKernel_ptsz");
    const RuntimeGraphLaunchFunction runtime_graph_launch =
        resolve_default<RuntimeGraphLaunchFunction>("cudaGraphLaunch");
    const RuntimeGraphLaunchFunction runtime_graph_launch_ptsz =
        resolve_default<RuntimeGraphLaunchFunction>("cudaGraphLaunch_ptsz");
    // NOLINTBEGIN(bugprone-reserved-identifier, readability-identifier-naming): preserve CUDA
    // compiler ABI names.
    const RuntimeInternalLaunchKernelFunction runtime_internal_launch_kernel =
        resolve_default<RuntimeInternalLaunchKernelFunction>("__cudaLaunchKernel");
    const RuntimeInternalLaunchKernelFunction runtime_internal_launch_kernel_ptsz =
        resolve_default<RuntimeInternalLaunchKernelFunction>("__cudaLaunchKernel_ptsz");
    // NOLINTEND(bugprone-reserved-identifier, readability-identifier-naming)
    const RuntimeMallocAsyncFunction runtime_async_allocate =
        resolve_default<RuntimeMallocAsyncFunction>("cudaMallocAsync");
    const RuntimeMallocAsyncFunction runtime_async_allocate_ptsz =
        resolve_default<RuntimeMallocAsyncFunction>("cudaMallocAsync_ptsz");
    const RuntimeMallocFromPoolAsyncFunction runtime_pool_async_allocate =
        resolve_default<RuntimeMallocFromPoolAsyncFunction>("cudaMallocFromPoolAsync");
    const RuntimeMallocFromPoolAsyncFunction runtime_pool_async_allocate_ptsz =
        resolve_default<RuntimeMallocFromPoolAsyncFunction>("cudaMallocFromPoolAsync_ptsz");
    const RuntimeFreeFunction runtime_release = resolve_default<RuntimeFreeFunction>("cudaFree");
    const RuntimeIpcGetMemHandleFunction runtime_ipc_get_handle =
        resolve_default<RuntimeIpcGetMemHandleFunction>("cudaIpcGetMemHandle");
    const RuntimeIpcOpenMemHandleFunction runtime_ipc_open_handle =
        resolve_default<RuntimeIpcOpenMemHandleFunction>("cudaIpcOpenMemHandle");
    const RuntimeIpcCloseMemHandleFunction runtime_ipc_close_handle =
        resolve_default<RuntimeIpcCloseMemHandleFunction>("cudaIpcCloseMemHandle");
    const RuntimeImportExternalMemoryFunction runtime_import_external_memory =
        resolve_default<RuntimeImportExternalMemoryFunction>("cudaImportExternalMemory");
    const RuntimeExternalMemoryGetMappedBufferFunction runtime_get_external_buffer =
        resolve_default<RuntimeExternalMemoryGetMappedBufferFunction>(
            "cudaExternalMemoryGetMappedBuffer");
    const RuntimeExternalMemoryGetMappedMipmappedArrayFunction runtime_get_external_mipmap =
        resolve_default<RuntimeExternalMemoryGetMappedMipmappedArrayFunction>(
            "cudaExternalMemoryGetMappedMipmappedArray");
    const RuntimeDestroyExternalMemoryFunction runtime_destroy_external_memory =
        resolve_default<RuntimeDestroyExternalMemoryFunction>("cudaDestroyExternalMemory");
    const RuntimeMallocArrayFunction runtime_array_create =
        resolve_default<RuntimeMallocArrayFunction>("cudaMallocArray");
    const RuntimeMalloc3DArrayFunction runtime_array_3d_create =
        resolve_default<RuntimeMalloc3DArrayFunction>("cudaMalloc3DArray");
    const RuntimeMallocMipmappedArrayFunction runtime_mipmapped_array_create =
        resolve_default<RuntimeMallocMipmappedArrayFunction>("cudaMallocMipmappedArray");
    const RuntimeFreeArrayFunction runtime_array_destroy =
        resolve_default<RuntimeFreeArrayFunction>("cudaFreeArray");
    const RuntimeFreeMipmappedArrayFunction runtime_mipmapped_array_destroy =
        resolve_default<RuntimeFreeMipmappedArrayFunction>("cudaFreeMipmappedArray");
    const RuntimeGraphicsUnregisterResourceFunction runtime_graphics_unregister_resource =
        resolve_default<RuntimeGraphicsUnregisterResourceFunction>(
            "cudaGraphicsUnregisterResource");
    const RuntimeGraphicsResourceSetMapFlagsFunction runtime_graphics_set_map_flags =
        resolve_default<RuntimeGraphicsResourceSetMapFlagsFunction>(
            "cudaGraphicsResourceSetMapFlags");
    const RuntimeGraphicsMapResourcesFunction runtime_graphics_map_resources =
        resolve_default<RuntimeGraphicsMapResourcesFunction>("cudaGraphicsMapResources");
    const RuntimeGraphicsUnmapResourcesFunction runtime_graphics_unmap_resources =
        resolve_default<RuntimeGraphicsUnmapResourcesFunction>("cudaGraphicsUnmapResources");
    const RuntimeGraphicsResourceGetMappedPointerFunction runtime_graphics_get_mapped_pointer =
        resolve_default<RuntimeGraphicsResourceGetMappedPointerFunction>(
            "cudaGraphicsResourceGetMappedPointer");
    const RuntimeGraphicsSubResourceGetMappedArrayFunction runtime_graphics_get_mapped_array =
        resolve_default<RuntimeGraphicsSubResourceGetMappedArrayFunction>(
            "cudaGraphicsSubResourceGetMappedArray");
    const RuntimeGraphicsResourceGetMappedMipmappedArrayFunction
        runtime_graphics_get_mapped_mipmap =
            resolve_default<RuntimeGraphicsResourceGetMappedMipmappedArrayFunction>(
                "cudaGraphicsResourceGetMappedMipmappedArray");
    const RuntimeGraphAddMemAllocNodeFunction runtime_graph_add_mem_alloc_node =
        resolve_default<RuntimeGraphAddMemAllocNodeFunction>("cudaGraphAddMemAllocNode");
    const RuntimeFreeAsyncFunction runtime_async_release =
        resolve_default<RuntimeFreeAsyncFunction>("cudaFreeAsync");
    const RuntimeFreeAsyncFunction runtime_async_release_ptsz =
        resolve_default<RuntimeFreeAsyncFunction>("cudaFreeAsync_ptsz");
    const RuntimeDeviceSynchronizeFunction runtime_device_synchronize =
        resolve_default<RuntimeDeviceSynchronizeFunction>("cudaDeviceSynchronize");
    const RuntimeStreamSynchronizeFunction runtime_stream_synchronize =
        resolve_default<RuntimeStreamSynchronizeFunction>("cudaStreamSynchronize");
    const RuntimeStreamSynchronizeFunction runtime_stream_synchronize_ptsz =
        resolve_default<RuntimeStreamSynchronizeFunction>("cudaStreamSynchronize_ptsz");
    const RuntimeStreamQueryFunction runtime_stream_query =
        resolve_default<RuntimeStreamQueryFunction>("cudaStreamQuery");
    const RuntimeStreamQueryFunction runtime_stream_query_ptsz =
        resolve_default<RuntimeStreamQueryFunction>("cudaStreamQuery_ptsz");
    const RuntimeStreamDestroyFunction runtime_stream_destroy =
        resolve_default<RuntimeStreamDestroyFunction>("cudaStreamDestroy");
    const RuntimeMemGetInfoFunction runtime_get_info =
        resolve_default<RuntimeMemGetInfoFunction>("cudaMemGetInfo");
    const RuntimeDeviceGetDefaultMemPoolFunction runtime_device_get_default_pool =
        resolve_default<RuntimeDeviceGetDefaultMemPoolFunction>("cudaDeviceGetDefaultMemPool");
    const RuntimeDeviceSetMemPoolFunction runtime_device_set_pool =
        resolve_default<RuntimeDeviceSetMemPoolFunction>("cudaDeviceSetMemPool");
    const RuntimeDeviceGetMemPoolFunction runtime_device_get_pool =
        resolve_default<RuntimeDeviceGetMemPoolFunction>("cudaDeviceGetMemPool");
    const RuntimeMemPoolTrimToFunction runtime_pool_trim =
        resolve_default<RuntimeMemPoolTrimToFunction>("cudaMemPoolTrimTo");
    const RuntimeMemPoolSetAttributeFunction runtime_pool_set_attribute =
        resolve_default<RuntimeMemPoolSetAttributeFunction>("cudaMemPoolSetAttribute");
    const RuntimeMemPoolGetAttributeFunction runtime_pool_get_attribute =
        resolve_default<RuntimeMemPoolGetAttributeFunction>("cudaMemPoolGetAttribute");
    const RuntimeMemPoolSetAccessFunction runtime_pool_set_access =
        resolve_default<RuntimeMemPoolSetAccessFunction>("cudaMemPoolSetAccess");
    const RuntimeMemPoolGetAccessFunction runtime_pool_get_access =
        resolve_default<RuntimeMemPoolGetAccessFunction>("cudaMemPoolGetAccess");
    const RuntimeMemPoolCreateFunction runtime_pool_create =
        resolve_default<RuntimeMemPoolCreateFunction>("cudaMemPoolCreate");
    const RuntimeMemPoolDestroyFunction runtime_pool_destroy =
        resolve_default<RuntimeMemPoolDestroyFunction>("cudaMemPoolDestroy");
    const RuntimeMemGetDefaultMemPoolFunction runtime_get_default_pool =
        resolve_default<RuntimeMemGetDefaultMemPoolFunction>("cudaMemGetDefaultMemPool");
    const RuntimeMemGetMemPoolFunction runtime_get_pool =
        resolve_default<RuntimeMemGetMemPoolFunction>("cudaMemGetMemPool");
    const RuntimeMemSetMemPoolFunction runtime_set_pool =
        resolve_default<RuntimeMemSetMemPoolFunction>("cudaMemSetMemPool");
    const RuntimeMemPoolExportToShareableHandleFunction runtime_pool_export_handle =
        resolve_default<RuntimeMemPoolExportToShareableHandleFunction>(
            "cudaMemPoolExportToShareableHandle");
    const RuntimeMemPoolImportFromShareableHandleFunction runtime_pool_import_handle =
        resolve_default<RuntimeMemPoolImportFromShareableHandleFunction>(
            "cudaMemPoolImportFromShareableHandle");
    const RuntimeMemPoolExportPointerFunction runtime_pool_export_pointer =
        resolve_default<RuntimeMemPoolExportPointerFunction>("cudaMemPoolExportPointer");
    const RuntimeMemPoolImportPointerFunction runtime_pool_import_pointer =
        resolve_default<RuntimeMemPoolImportPointerFunction>("cudaMemPoolImportPointer");
    void* runtime_pointer = nullptr;
    all_passed &= expect(
        runtime_allocate != nullptr && runtime_managed_allocate != nullptr &&
            runtime_pitch_allocate != nullptr && runtime_3d_allocate != nullptr &&
            runtime_async_allocate != nullptr && runtime_launch_kernel != nullptr &&
            runtime_launch_kernel_ptsz != nullptr && runtime_internal_launch_kernel != nullptr &&
            runtime_internal_launch_kernel_ptsz != nullptr && runtime_graph_launch != nullptr &&
            runtime_graph_launch_ptsz != nullptr && runtime_async_allocate_ptsz != nullptr &&
            runtime_release != nullptr && runtime_pool_async_allocate != nullptr &&
            runtime_pool_async_allocate_ptsz != nullptr && runtime_async_release != nullptr &&
            runtime_async_release_ptsz != nullptr && runtime_ipc_get_handle != nullptr &&
            runtime_ipc_open_handle != nullptr && runtime_ipc_close_handle != nullptr &&
            runtime_device_synchronize != nullptr && runtime_import_external_memory != nullptr &&
            runtime_get_external_buffer != nullptr && runtime_get_external_mipmap != nullptr &&
            runtime_destroy_external_memory != nullptr && runtime_array_create != nullptr &&
            runtime_array_3d_create != nullptr && runtime_mipmapped_array_create != nullptr &&
            runtime_array_destroy != nullptr && runtime_mipmapped_array_destroy != nullptr &&
            runtime_graphics_unregister_resource != nullptr &&
            runtime_graphics_set_map_flags != nullptr &&
            runtime_graphics_map_resources != nullptr &&
            runtime_graphics_unmap_resources != nullptr &&
            runtime_graphics_get_mapped_pointer != nullptr &&
            runtime_graphics_get_mapped_array != nullptr &&
            runtime_graphics_get_mapped_mipmap != nullptr &&
            runtime_graph_add_mem_alloc_node != nullptr && runtime_stream_synchronize != nullptr &&
            runtime_stream_synchronize_ptsz != nullptr && runtime_stream_query != nullptr &&
            runtime_stream_query_ptsz != nullptr && runtime_stream_destroy != nullptr &&
            runtime_get_info != nullptr && runtime_device_get_default_pool != nullptr &&
            runtime_device_set_pool != nullptr && runtime_device_get_pool != nullptr &&
            runtime_pool_trim != nullptr && runtime_pool_set_attribute != nullptr &&
            runtime_pool_get_attribute != nullptr && runtime_pool_set_access != nullptr &&
            runtime_pool_get_access != nullptr && runtime_pool_create != nullptr &&
            runtime_pool_destroy != nullptr && runtime_get_default_pool != nullptr &&
            runtime_get_pool != nullptr && runtime_set_pool != nullptr &&
            runtime_pool_export_handle != nullptr && runtime_pool_import_handle != nullptr &&
            runtime_pool_export_pointer != nullptr && runtime_pool_import_pointer != nullptr,
        "runtime interceptor symbols were not exported");
    if (runtime_handle != nullptr) {
        all_passed &=
            expect(dlsym(runtime_handle, "cudaMalloc") == reinterpret_cast<void*>(runtime_allocate),
                   "explicit CUDA Runtime handle did not return the interceptor wrapper");
        all_passed &= expect(dlsym(runtime_handle, "cudaMallocManaged") ==
                                 reinterpret_cast<void*>(runtime_managed_allocate),
                             "explicit CUDA Runtime handle did not return the managed wrapper");
        all_passed &= expect(dlsym(runtime_handle, "cudaMallocPitch") ==
                                 reinterpret_cast<void*>(runtime_pitch_allocate),
                             "explicit CUDA Runtime handle did not return the pitch wrapper");
        all_passed &= expect(
            dlsym(runtime_handle, "cudaMalloc3D") == reinterpret_cast<void*>(runtime_3d_allocate),
            "explicit CUDA Runtime handle did not return the 3D wrapper");
        all_passed &= expect(dlsym(runtime_handle, "cudaLaunchKernel") ==
                                 reinterpret_cast<void*>(runtime_launch_kernel),
                             "explicit CUDA Runtime handle did not return the launch wrapper");
        all_passed &=
            expect(dlsym(runtime_handle, "cudaGraphLaunch") ==
                       reinterpret_cast<void*>(runtime_graph_launch),
                   "explicit CUDA Runtime handle did not return the graph launch wrapper");
        all_passed &=
            expect(dlsym(runtime_handle, "__cudaLaunchKernel_ptsz") ==
                       reinterpret_cast<void*>(runtime_internal_launch_kernel_ptsz),
                   "explicit CUDA Runtime handle did not return the compiler launch wrapper");
        all_passed &=
            expect(dlsym(runtime_handle, "cudaMallocAsync_ptsz") ==
                       reinterpret_cast<void*>(runtime_async_allocate_ptsz),
                   "explicit CUDA Runtime handle did not return the PTDS interceptor wrapper");
    }
    all_passed &= expect(runtime_allocate(&runtime_pointer, kRuntimeAllocationBytes) == cudaSuccess,
                         "runtime allocation was rejected");
    cudaIpcMemHandle_t runtime_ipc_handle{};
    all_passed &=
        expect(runtime_ipc_get_handle(&runtime_ipc_handle, runtime_pointer) == cudaSuccess,
               "Runtime IPC handle export was not forwarded");
    void* runtime_ipc_pointer = nullptr;
    all_passed &= expect(runtime_ipc_open_handle(&runtime_ipc_pointer, runtime_ipc_handle, 0) ==
                                 cudaErrorNotSupported &&
                             runtime_ipc_pointer == nullptr,
                         "Runtime IPC import was not rejected under quota");
    all_passed &=
        expect(runtime_ipc_close_handle(reinterpret_cast<void*>(0x71000000U)) == cudaSuccess,
               "Runtime IPC close was not forwarded");
    cudaExternalMemory_t runtime_external_memory = nullptr;
    cudaExternalMemoryHandleDesc runtime_external_handle_desc{};
    all_passed &= expect(
        runtime_import_external_memory(&runtime_external_memory, &runtime_external_handle_desc) ==
                cudaErrorNotSupported &&
            runtime_external_memory == nullptr,
        "Runtime external memory import was not rejected under quota");
    cudaExternalMemoryBufferDesc runtime_external_buffer_desc{};
    void* runtime_external_pointer = nullptr;
    all_passed &=
        expect(runtime_get_external_buffer(
                   &runtime_external_pointer, reinterpret_cast<cudaExternalMemory_t>(0x72000000U),
                   &runtime_external_buffer_desc) == cudaErrorNotSupported &&
                   runtime_external_pointer == nullptr,
               "Runtime external buffer mapping was not rejected under quota");
    cudaExternalMemoryMipmappedArrayDesc runtime_external_mipmap_desc{};
    cudaMipmappedArray_t runtime_external_mipmap = nullptr;
    all_passed &=
        expect(runtime_get_external_mipmap(
                   &runtime_external_mipmap, reinterpret_cast<cudaExternalMemory_t>(0x72000000U),
                   &runtime_external_mipmap_desc) == cudaErrorNotSupported &&
                   runtime_external_mipmap == nullptr,
               "Runtime external mipmap mapping was not rejected under quota");
    all_passed &= expect(runtime_destroy_external_memory(
                             reinterpret_cast<cudaExternalMemory_t>(0x72000000U)) == cudaSuccess,
                         "Runtime external memory destruction was not forwarded");
    cudaChannelFormatDesc runtime_array_descriptor{};
    cudaArray_t runtime_array = nullptr;
    all_passed &= expect(runtime_array_create(&runtime_array, &runtime_array_descriptor, 1, 1, 0) ==
                                 cudaErrorNotSupported &&
                             runtime_array == nullptr,
                         "Runtime CUDA array allocation was not rejected under quota");
    cudaExtent runtime_array_extent{1, 1, 1};
    cudaArray_t runtime_array_3d = nullptr;
    all_passed &=
        expect(runtime_array_3d_create(&runtime_array_3d, &runtime_array_descriptor,
                                       runtime_array_extent, 0) == cudaErrorNotSupported &&
                   runtime_array_3d == nullptr,
               "Runtime CUDA 3D array allocation was not rejected under quota");
    cudaMipmappedArray_t runtime_mipmapped_array = nullptr;
    all_passed &= expect(
        runtime_mipmapped_array_create(&runtime_mipmapped_array, &runtime_array_descriptor,
                                       runtime_array_extent, 1, 0) == cudaErrorNotSupported &&
            runtime_mipmapped_array == nullptr,
        "Runtime CUDA mipmapped array allocation was not rejected under quota");
    all_passed &=
        expect(runtime_array_destroy(reinterpret_cast<cudaArray_t>(0x73000000U)) == cudaSuccess,
               "Runtime CUDA array destruction was not forwarded");
    all_passed &= expect(runtime_mipmapped_array_destroy(
                             reinterpret_cast<cudaMipmappedArray_t>(0x73000000U)) == cudaSuccess,
                         "Runtime CUDA mipmapped array destruction was not forwarded");
    cudaGraphicsResource_t runtime_graphics_resource =
        reinterpret_cast<cudaGraphicsResource_t>(0x75000000U);
    all_passed &= expect(runtime_graphics_map_resources(1, &runtime_graphics_resource, nullptr) ==
                             cudaErrorNotSupported,
                         "Runtime graphics mapping was not rejected under quota");
    void* runtime_graphics_pointer = nullptr;
    std::size_t runtime_graphics_size = 0;
    all_passed &= expect(
        runtime_graphics_get_mapped_pointer(&runtime_graphics_pointer, &runtime_graphics_size,
                                            runtime_graphics_resource) == cudaErrorNotSupported &&
            runtime_graphics_pointer == nullptr && runtime_graphics_size == 0,
        "Runtime graphics mapped pointer was not rejected under quota");
    cudaArray_t runtime_graphics_array = nullptr;
    all_passed &=
        expect(runtime_graphics_get_mapped_array(&runtime_graphics_array, runtime_graphics_resource,
                                                 0, 0) == cudaErrorNotSupported &&
                   runtime_graphics_array == nullptr,
               "Runtime graphics mapped array was not rejected under quota");
    cudaMipmappedArray_t runtime_graphics_mipmap = nullptr;
    all_passed &=
        expect(runtime_graphics_get_mapped_mipmap(
                   &runtime_graphics_mipmap, runtime_graphics_resource) == cudaErrorNotSupported &&
                   runtime_graphics_mipmap == nullptr,
               "Runtime graphics mapped mipmap was not rejected under quota");
    all_passed &=
        expect(runtime_graphics_set_map_flags(runtime_graphics_resource, 0) == cudaSuccess,
               "Runtime graphics map flags were not forwarded");
    all_passed &= expect(
        runtime_graphics_unmap_resources(1, &runtime_graphics_resource, nullptr) == cudaSuccess,
        "Runtime graphics unmapping was not forwarded");
    all_passed &=
        expect(runtime_graphics_unregister_resource(runtime_graphics_resource) == cudaSuccess,
               "Runtime graphics resource destruction was not forwarded");
    cudaGraphNode_t graph_node = nullptr;
    cudaMemAllocNodeParams graph_parameters{};
    all_passed &= expect(
        runtime_graph_add_mem_alloc_node(&graph_node, reinterpret_cast<cudaGraph_t>(0x74000000U),
                                         nullptr, 0, &graph_parameters) == cudaErrorNotSupported &&
            graph_node == nullptr,
        "CUDA graph memory node was not rejected under quota");
    all_passed &= expect(runtime_launch_kernel(nullptr, dim3{1, 1, 1}, dim3{1, 1, 1}, nullptr, 0,
                                               nullptr) == cudaSuccess,
                         "Runtime cudaLaunchKernel forwarding failed");
    all_passed &= expect(runtime_launch_kernel_ptsz(nullptr, dim3{1, 1, 1}, dim3{1, 1, 1}, nullptr,
                                                    0, nullptr) == cudaSuccess,
                         "Runtime cudaLaunchKernel_ptsz forwarding failed");
    all_passed &= expect(runtime_graph_launch(reinterpret_cast<cudaGraphExec_t>(0x76000000U),
                                              nullptr) == cudaSuccess,
                         "Runtime cudaGraphLaunch forwarding failed");
    all_passed &= expect(runtime_graph_launch_ptsz(reinterpret_cast<cudaGraphExec_t>(0x76000000U),
                                                   nullptr) == cudaSuccess,
                         "Runtime cudaGraphLaunch_ptsz forwarding failed");
    all_passed &= expect(runtime_graph_launch(nullptr, nullptr) == cudaErrorInvalidValue &&
                             runtime_graph_launch_ptsz(nullptr, nullptr) == cudaErrorInvalidValue,
                         "null Runtime graph launch handles were not rejected");
    all_passed &= expect(runtime_internal_launch_kernel(nullptr, dim3{1, 1, 1}, dim3{1, 1, 1},
                                                        nullptr, 0, nullptr) == cudaSuccess,
                         "Runtime __cudaLaunchKernel forwarding failed");
    all_passed &= expect(runtime_internal_launch_kernel_ptsz(nullptr, dim3{1, 1, 1}, dim3{1, 1, 1},
                                                             nullptr, 0, nullptr) == cudaSuccess,
                         "Runtime __cudaLaunchKernel_ptsz forwarding failed");
    all_passed &= expect(runtime_get_info(&free_bytes, &total_bytes) == cudaSuccess &&
                             free_bytes == kQuotaBytes - kRuntimeAllocationBytes,
                         "runtime allocation was double-accounted or not accounted");
    all_passed &=
        expect(runtime_release(runtime_pointer) == cudaSuccess, "runtime allocation was not freed");
    all_passed &= expect(
        runtime_get_info(&free_bytes, &total_bytes) == cudaSuccess && free_bytes == kQuotaBytes,
        "runtime release did not restore quota");

    void* runtime_managed_pointer = nullptr;
    all_passed &= expect(runtime_managed_allocate(&runtime_managed_pointer, kRuntimeAllocationBytes,
                                                  cudaMemAttachGlobal) == cudaSuccess,
                         "Runtime managed allocation was rejected");
    all_passed &= expect(runtime_get_info(&free_bytes, &total_bytes) == cudaSuccess &&
                             free_bytes == kQuotaBytes - kRuntimeAllocationBytes,
                         "Runtime managed allocation was not accounted");
    all_passed &= expect(runtime_release(runtime_managed_pointer) == cudaSuccess,
                         "Runtime managed allocation was not freed");
    void* rejected_managed_pointer = nullptr;
    all_passed &=
        expect(runtime_managed_allocate(&rejected_managed_pointer, kQuotaBytes + 1,
                                        cudaMemAttachGlobal) == cudaErrorMemoryAllocation &&
                   rejected_managed_pointer == nullptr,
               "Runtime managed quota rejection was not enforced");

    constexpr std::size_t k_runtime_pitch_width_bytes = 500;
    constexpr std::size_t k_runtime_pitch_height = 2;
    constexpr std::size_t k_runtime_pitch_bytes = 512 * k_runtime_pitch_height;
    void* runtime_pitch_pointer = nullptr;
    std::size_t runtime_pitch = 0;
    const cudaError_t runtime_pitch_result =
        runtime_pitch_allocate(&runtime_pitch_pointer, &runtime_pitch, k_runtime_pitch_width_bytes,
                               k_runtime_pitch_height);
    all_passed &= expect(runtime_pitch_result == cudaSuccess && runtime_pitch == 512,
                         "Runtime pitched allocation was rejected");
    all_passed &= expect(runtime_get_info(&free_bytes, &total_bytes) == cudaSuccess &&
                             free_bytes == kQuotaBytes - k_runtime_pitch_bytes,
                         "Runtime pitched allocation was not accounted using the physical pitch");
    all_passed &= expect(runtime_release(runtime_pitch_pointer) == cudaSuccess,
                         "Runtime pitched allocation was not freed");
    void* rejected_pitch_pointer = nullptr;
    std::size_t rejected_pitch = 0;
    all_passed &= expect(runtime_pitch_allocate(&rejected_pitch_pointer, &rejected_pitch,
                                                kQuotaBytes + 1, 1) == cudaErrorMemoryAllocation &&
                             rejected_pitch_pointer == nullptr,
                         "Runtime pitched quota rejection was not enforced");
    all_passed &= expect(
        runtime_get_info(&free_bytes, &total_bytes) == cudaSuccess && free_bytes == kQuotaBytes,
        "Runtime managed or pitched release did not restore quota");

    constexpr cudaExtent k_runtime_3d_extent{500, 2, 2};
    constexpr std::size_t k_runtime_3d_bytes =
        512 * k_runtime_3d_extent.height * k_runtime_3d_extent.depth;
    cudaPitchedPtr runtime_3d_allocation{};
    all_passed &=
        expect(runtime_3d_allocate(&runtime_3d_allocation, k_runtime_3d_extent) == cudaSuccess &&
                   runtime_3d_allocation.pitch == 512 && runtime_3d_allocation.ptr != nullptr,
               "Runtime 3D pitched allocation was rejected");
    all_passed &= expect(runtime_get_info(&free_bytes, &total_bytes) == cudaSuccess &&
                             free_bytes == kQuotaBytes - k_runtime_3d_bytes,
                         "Runtime 3D allocation was not accounted using physical pitch");
    all_passed &= expect(runtime_release(runtime_3d_allocation.ptr) == cudaSuccess,
                         "Runtime 3D allocation was not freed");
    cudaPitchedPtr rejected_3d_allocation{};
    constexpr cudaExtent k_rejected_3d_extent{kQuotaBytes + 1, 1, 1};
    all_passed &= expect(runtime_3d_allocate(&rejected_3d_allocation, k_rejected_3d_extent) ==
                                 cudaErrorMemoryAllocation &&
                             rejected_3d_allocation.ptr == nullptr,
                         "Runtime 3D quota rejection was not enforced");
    cudaPitchedPtr overflow_3d_allocation{};
    const cudaExtent overflow_3d_extent{1, std::numeric_limits<std::size_t>::max(), 2};
    all_passed &= expect(
        runtime_3d_allocate(&overflow_3d_allocation, overflow_3d_extent) == cudaErrorInvalidValue &&
            overflow_3d_allocation.ptr == nullptr,
        "Runtime 3D extent overflow was not rejected");
    all_passed &= expect(
        runtime_get_info(&free_bytes, &total_bytes) == cudaSuccess && free_bytes == kQuotaBytes,
        "Runtime 3D release did not restore quota");

    void* runtime_async_pointer = nullptr;
    all_passed &= expect(runtime_async_allocate(&runtime_async_pointer, kAsyncAllocationBytes,
                                                nullptr) == cudaSuccess,
                         "Runtime async allocation was rejected");
    all_passed &= expect(runtime_async_release(runtime_async_pointer, nullptr) == cudaSuccess,
                         "Runtime async release was rejected");
    all_passed &= expect(runtime_stream_query(nullptr) == cudaSuccess,
                         "Runtime stream query did not complete the release");
    all_passed &= expect(
        runtime_get_info(&free_bytes, &total_bytes) == cudaSuccess && free_bytes == kQuotaBytes,
        "Runtime stream query did not restore quota");

    void* runtime_ptsz_pointer = nullptr;
    all_passed &= expect(runtime_async_allocate_ptsz(&runtime_ptsz_pointer, kAsyncAllocationBytes,
                                                     nullptr) == cudaSuccess,
                         "Runtime PTDS async allocation was rejected");
    all_passed &= expect(runtime_async_release_ptsz(runtime_ptsz_pointer, nullptr) == cudaSuccess,
                         "Runtime PTDS async release was rejected");
    all_passed &= expect(runtime_stream_query_ptsz(nullptr) == cudaSuccess,
                         "Runtime PTDS stream query did not complete the release");
    all_passed &= expect(runtime_stream_synchronize_ptsz(nullptr) == cudaSuccess,
                         "Runtime PTDS stream synchronization was rejected");
    all_passed &= expect(
        runtime_get_info(&free_bytes, &total_bytes) == cudaSuccess && free_bytes == kQuotaBytes,
        "Runtime PTDS stream query did not restore quota");

    cudaMemPool_t runtime_pool = nullptr;
    cudaMemPoolProps runtime_pool_properties{};
    all_passed &=
        expect(runtime_pool_create(&runtime_pool, &runtime_pool_properties) == cudaSuccess &&
                   runtime_pool != nullptr,
               "Runtime memory-pool creation was rejected");
    cudaMemPool_t queried_runtime_pool = nullptr;
    cudaMemLocation runtime_pool_location{};
    all_passed &= expect(runtime_device_get_default_pool(&queried_runtime_pool, 0) == cudaSuccess &&
                             queried_runtime_pool != nullptr &&
                             runtime_device_get_pool(&queried_runtime_pool, 0) == cudaSuccess &&
                             runtime_device_set_pool(0, runtime_pool) == cudaSuccess,
                         "Runtime device memory-pool operations were rejected");
    all_passed &= expect(runtime_get_default_pool(&queried_runtime_pool, &runtime_pool_location,
                                                  cudaMemAllocationTypePinned) == cudaSuccess &&
                             runtime_get_pool(&queried_runtime_pool, &runtime_pool_location,
                                              cudaMemAllocationTypePinned) == cudaSuccess &&
                             runtime_set_pool(&runtime_pool_location, cudaMemAllocationTypePinned,
                                              runtime_pool) == cudaSuccess,
                         "Runtime memory-pool selection operations were rejected");
    std::uint64_t runtime_release_threshold = 128;
    cudaMemAccessFlags runtime_access_flags = cudaMemAccessFlagsProtNone;
    all_passed &=
        expect(runtime_pool_set_attribute(runtime_pool, cudaMemPoolAttrReleaseThreshold,
                                          &runtime_release_threshold) == cudaSuccess &&
                   runtime_pool_get_attribute(runtime_pool, cudaMemPoolAttrReleaseThreshold,
                                              &runtime_release_threshold) == cudaSuccess &&
                   runtime_pool_set_access(runtime_pool, nullptr, 0) == cudaSuccess &&
                   runtime_pool_get_access(&runtime_access_flags, runtime_pool,
                                           &runtime_pool_location) == cudaSuccess &&
                   runtime_pool_trim(runtime_pool, 0) == cudaSuccess,
               "Runtime memory-pool management operations were rejected");
    void* runtime_pool_pointer = nullptr;
    all_passed &= expect(runtime_pool_async_allocate(&runtime_pool_pointer, kAsyncAllocationBytes,
                                                     runtime_pool, nullptr) == cudaSuccess,
                         "Runtime memory-pool allocation was rejected");
    all_passed &= expect(runtime_get_info(&free_bytes, &total_bytes) == cudaSuccess &&
                             free_bytes == kQuotaBytes - kAsyncAllocationBytes,
                         "Runtime memory-pool allocation was not accounted");
    cudaMemPoolPtrExportData runtime_pointer_export_data{};
    all_passed &= expect(runtime_pool_export_pointer(&runtime_pointer_export_data,
                                                     runtime_pool_pointer) == cudaSuccess,
                         "Runtime memory-pool pointer export was rejected");
    unsigned long long runtime_pool_exported_handle = 0;
    all_passed &= expect(
        runtime_pool_export_handle(&runtime_pool_exported_handle, runtime_pool,
                                   static_cast<cudaMemAllocationHandleType>(0), 0) == cudaSuccess,
        "Runtime memory-pool handle export was rejected");
    cudaMemPool_t imported_runtime_pool = nullptr;
    all_passed &=
        expect(runtime_pool_import_handle(&imported_runtime_pool, &runtime_pool_exported_handle,
                                          static_cast<cudaMemAllocationHandleType>(0),
                                          0) == cudaErrorNotSupported &&
                   imported_runtime_pool == nullptr,
               "Runtime memory-pool handle import was not rejected under quota");
    void* imported_runtime_pointer = nullptr;
    all_passed &=
        expect(runtime_pool_import_pointer(&imported_runtime_pointer, runtime_pool,
                                           &runtime_pointer_export_data) == cudaErrorNotSupported &&
                   imported_runtime_pointer == nullptr,
               "Runtime memory-pool pointer import was not rejected under quota");
    all_passed &= expect(runtime_async_release(runtime_pool_pointer, nullptr) == cudaSuccess &&
                             runtime_device_synchronize() == cudaSuccess &&
                             runtime_get_info(&free_bytes, &total_bytes) == cudaSuccess &&
                             free_bytes == kQuotaBytes,
                         "Runtime memory-pool release did not restore quota");
    void* runtime_pool_ptsz_pointer = nullptr;
    all_passed &= expect(
        runtime_pool_async_allocate_ptsz(&runtime_pool_ptsz_pointer, kAsyncAllocationBytes,
                                         runtime_pool, nullptr) == cudaSuccess &&
            runtime_async_release_ptsz(runtime_pool_ptsz_pointer, nullptr) == cudaSuccess &&
            runtime_device_synchronize() == cudaSuccess &&
            runtime_get_info(&free_bytes, &total_bytes) == cudaSuccess && free_bytes == kQuotaBytes,
        "Runtime PTDS memory-pool path did not restore quota");
    all_passed &= expect(runtime_pool_destroy(runtime_pool) == cudaSuccess,
                         "Runtime memory-pool destruction was rejected");

    void* runtime_device_async_pointer = nullptr;
    all_passed &= expect(runtime_async_allocate(&runtime_device_async_pointer,
                                                kAsyncAllocationBytes, nullptr) == cudaSuccess,
                         "second Runtime async allocation was rejected");
    all_passed &=
        expect(runtime_async_release(runtime_device_async_pointer, nullptr) == cudaSuccess,
               "second Runtime async release was rejected");
    all_passed &= expect(runtime_device_synchronize() == cudaSuccess,
                         "Runtime device synchronization did not complete the release");
    all_passed &= expect(
        runtime_get_info(&free_bytes, &total_bytes) == cudaSuccess && free_bytes == kQuotaBytes,
        "Runtime device synchronization did not restore quota");

    const cudaStream_t runtime_stream = reinterpret_cast<cudaStream_t>(0x2345);
    void* runtime_stream_pointer = nullptr;
    all_passed &= expect(runtime_async_allocate(&runtime_stream_pointer, kAsyncAllocationBytes,
                                                runtime_stream) == cudaSuccess,
                         "Runtime explicit-stream allocation was rejected");
    all_passed &=
        expect(runtime_async_release(runtime_stream_pointer, runtime_stream) == cudaSuccess,
               "Runtime explicit-stream release was rejected");
    all_passed &= expect(runtime_stream_synchronize(runtime_stream) == cudaSuccess,
                         "Runtime stream synchronization did not complete the release");
    all_passed &= expect(
        runtime_get_info(&free_bytes, &total_bytes) == cudaSuccess && free_bytes == kQuotaBytes,
        "Runtime stream synchronization did not restore quota");

    void* runtime_destroyed_stream_pointer = nullptr;
    all_passed &=
        expect(runtime_async_allocate(&runtime_destroyed_stream_pointer, kAsyncAllocationBytes,
                                      runtime_stream) == cudaSuccess,
               "Runtime stream-destroy allocation was rejected");
    all_passed &= expect(
        runtime_async_release(runtime_destroyed_stream_pointer, runtime_stream) == cudaSuccess,
        "Runtime stream-destroy release was rejected");
    all_passed &= expect(runtime_stream_destroy(runtime_stream) == cudaSuccess,
                         "Runtime stream destruction failed");
    all_passed &= expect(runtime_get_info(&free_bytes, &total_bytes) == cudaSuccess &&
                             free_bytes == kQuotaBytes - kAsyncAllocationBytes,
                         "Runtime stream destruction completed the release too early");
    all_passed &= expect(runtime_device_synchronize() == cudaSuccess,
                         "Runtime device synchronization did not recover destroyed stream");
    all_passed &= expect(
        runtime_get_info(&free_bytes, &total_bytes) == cudaSuccess && free_bytes == kQuotaBytes,
        "Runtime destroyed-stream release did not restore quota");

    CUdeviceptr async_pointer = 0;
    all_passed &=
        expect(async_allocate(&async_pointer, kAsyncAllocationBytes, nullptr) == CUDA_SUCCESS,
               "stream-ordered allocation was rejected");
    all_passed &= expect(get_info(&free_bytes, &total_bytes) == CUDA_SUCCESS &&
                             free_bytes == kQuotaBytes - kAsyncAllocationBytes,
                         "stream-ordered allocation was not accounted");
    CUdevice stream_device = 0;
    CUcontext stream_context = nullptr;
    all_passed &= expect(stream_get_device(nullptr, &stream_device) == CUDA_SUCCESS,
                         "stream device query was rejected");
    all_passed &= expect(stream_get_device_ptsz(nullptr, &stream_device) == CUDA_SUCCESS,
                         "PTDS stream device query was rejected");
    all_passed &= expect(stream_get_context(nullptr, &stream_context) == CUDA_SUCCESS,
                         "stream context query was rejected");
    all_passed &= expect(stream_get_context_ptsz(nullptr, &stream_context) == CUDA_SUCCESS,
                         "PTDS stream context query was rejected");
    all_passed &= expect(async_release(async_pointer, nullptr) == CUDA_SUCCESS,
                         "stream-ordered release was rejected");
    all_passed &= expect(async_release(async_pointer, nullptr) == CUDA_ERROR_INVALID_VALUE,
                         "duplicate stream-ordered release was accepted");
    all_passed &= expect(get_info(&free_bytes, &total_bytes) == CUDA_SUCCESS &&
                             free_bytes == kQuotaBytes - kAsyncAllocationBytes,
                         "pending stream release was charged too early");
    all_passed &=
        expect(stream_query(nullptr) == CUDA_SUCCESS, "stream query did not complete the release");
    all_passed &=
        expect(get_info(&free_bytes, &total_bytes) == CUDA_SUCCESS && free_bytes == kQuotaBytes,
               "stream query did not restore quota");

    CUdeviceptr ptsz_pointer = 0;
    all_passed &=
        expect(async_allocate_ptsz(&ptsz_pointer, kAsyncAllocationBytes, nullptr) == CUDA_SUCCESS,
               "PTDS stream-ordered allocation was rejected");
    all_passed &= expect(async_release_ptsz(ptsz_pointer, nullptr) == CUDA_SUCCESS,
                         "PTDS stream-ordered release was rejected");
    all_passed &= expect(stream_query_ptsz(nullptr) == CUDA_SUCCESS,
                         "PTDS stream query did not complete the release");
    all_passed &=
        expect(get_info(&free_bytes, &total_bytes) == CUDA_SUCCESS && free_bytes == kQuotaBytes,
               "PTDS stream query did not restore quota");

    CUdeviceptr pool_async_pointer = 0;
    all_passed &= expect(pool_async_allocate_ptsz(&pool_async_pointer, kAsyncAllocationBytes,
                                                  nullptr, nullptr) == CUDA_SUCCESS,
                         "memory-pool stream-ordered allocation was rejected");
    all_passed &= expect(async_release_ptsz(pool_async_pointer, nullptr) == CUDA_SUCCESS,
                         "memory-pool stream-ordered release was rejected");
    all_passed &= expect(stream_synchronize_ptsz(nullptr) == CUDA_SUCCESS,
                         "stream synchronization did not complete the pool release");

    CUdeviceptr device_async_pointer = 0;
    all_passed &= expect(
        async_allocate(&device_async_pointer, kAsyncAllocationBytes, nullptr) == CUDA_SUCCESS,
        "device-synchronized stream allocation was rejected");
    all_passed &= expect(async_release(device_async_pointer, nullptr) == CUDA_SUCCESS,
                         "device-synchronized stream release was rejected");
    all_passed &= expect(context_synchronize() == CUDA_SUCCESS,
                         "context synchronization did not complete the release");
    all_passed &=
        expect(get_info(&free_bytes, &total_bytes) == CUDA_SUCCESS && free_bytes == kQuotaBytes,
               "device synchronization did not restore quota");

    const CUstream destroyed_stream = reinterpret_cast<CUstream>(0x1234);
    CUdeviceptr detached_pointer = 0;
    all_passed &= expect(
        async_allocate(&detached_pointer, kAsyncAllocationBytes, destroyed_stream) == CUDA_SUCCESS,
        "stream-destroy allocation was rejected");
    all_passed &= expect(async_release(detached_pointer, destroyed_stream) == CUDA_SUCCESS,
                         "stream-destroy release was rejected");
    all_passed &=
        expect(stream_destroy(destroyed_stream) == CUDA_SUCCESS, "stream destruction failed");
    all_passed &= expect(get_info(&free_bytes, &total_bytes) == CUDA_SUCCESS &&
                             free_bytes == kQuotaBytes - kAsyncAllocationBytes,
                         "destroyed stream release was completed too early");
    all_passed &= expect(context_synchronize() == CUDA_SUCCESS,
                         "context synchronization did not complete a destroyed-stream release");
    all_passed &=
        expect(get_info(&free_bytes, &total_bytes) == CUDA_SUCCESS && free_bytes == kQuotaBytes,
               "destroyed-stream release was not recovered by context synchronization");

    CUcontext context = nullptr;
    all_passed &= expect(get_current(&context) == CUDA_SUCCESS && context != nullptr,
                         "current context was not available");
    CUdeviceptr context_pointer = 0;
    all_passed &= expect(allocate(&context_pointer, kRuntimeAllocationBytes) == CUDA_SUCCESS,
                         "context cleanup allocation was rejected");
    all_passed &= expect(destroy_context(context) == CUDA_SUCCESS, "context destruction failed");
    const CUresult post_destroy_info_result = get_info(&free_bytes, &total_bytes);
    all_passed &=
        expect(shared_mode ? post_destroy_info_result == CUDA_ERROR_INVALID_CONTEXT
                           : post_destroy_info_result == CUDA_SUCCESS && free_bytes == kQuotaBytes,
               shared_mode ? "shared memory info should require a current context"
                           : "context destruction did not release accounted bytes");
    all_passed &= expect(get_total(&total_bytes, 0) == CUDA_SUCCESS && total_bytes == kQuotaBytes,
                         "device total memory incorrectly depended on a current context");

    const LegacyGetProcAddressFunction legacy_get_proc =
        resolve_default<LegacyGetProcAddressFunction>("cuGetProcAddress");
    const GetProcAddressV2Function get_proc_v2 =
        resolve_default<GetProcAddressV2Function>("cuGetProcAddress_v2");
    void* queried_symbol = nullptr;
    all_passed &= expect(
        legacy_get_proc != nullptr &&
            legacy_get_proc(nullptr, &queried_symbol, CUDA_VERSION, 0) ==
                CUDA_ERROR_INVALID_VALUE &&
            legacy_get_proc("cuMemAlloc_v2", nullptr, CUDA_VERSION, 0) == CUDA_ERROR_INVALID_VALUE,
        "legacy cuGetProcAddress invalid arguments were not rejected");
    all_passed &= expect(get_proc_v2 != nullptr &&
                             get_proc_v2(nullptr, &queried_symbol, CUDA_VERSION, 0, nullptr) ==
                                 CUDA_ERROR_INVALID_VALUE &&
                             get_proc_v2("cuMemAlloc_v2", nullptr, CUDA_VERSION, 0, nullptr) ==
                                 CUDA_ERROR_INVALID_VALUE,
                         "v2 cuGetProcAddress invalid arguments were not rejected");
    all_passed &= expect(
        legacy_get_proc != nullptr &&
            legacy_get_proc("cuMemAlloc_v2", &queried_symbol, CUDA_VERSION, 0) == CUDA_SUCCESS &&
            queried_symbol == reinterpret_cast<void*>(allocate),
        "legacy cuGetProcAddress did not return the interceptor wrapper");
    queried_symbol = nullptr;
    all_passed &= expect(
        legacy_get_proc != nullptr &&
            legacy_get_proc("cuMemCreate", &queried_symbol, CUDA_VERSION, 0) == CUDA_SUCCESS &&
            queried_symbol == reinterpret_cast<void*>(vmm_create),
        "legacy cuGetProcAddress did not return the VMM wrapper");
    queried_symbol = nullptr;
    all_passed &=
        expect(legacy_get_proc != nullptr &&
                   legacy_get_proc("cuMemMap", &queried_symbol, CUDA_VERSION, 0) == CUDA_SUCCESS &&
                   queried_symbol == reinterpret_cast<void*>(map),
               "legacy cuGetProcAddress did not return the VMM address wrapper");
    queried_symbol = nullptr;
    all_passed &= expect(
        legacy_get_proc != nullptr &&
            legacy_get_proc("cuMemPoolCreate", &queried_symbol, CUDA_VERSION, 0) == CUDA_SUCCESS &&
            queried_symbol == reinterpret_cast<void*>(pool_create),
        "legacy cuGetProcAddress did not return the memory-pool wrapper");
    queried_symbol = nullptr;
    all_passed &= expect(legacy_get_proc != nullptr &&
                             legacy_get_proc("cuDeviceGetDefaultMemPool", &queried_symbol,
                                             CUDA_VERSION, 0) == CUDA_SUCCESS &&
                             queried_symbol == reinterpret_cast<void*>(device_get_default_pool),
                         "legacy cuGetProcAddress did not return the default-pool wrapper");
    CUdriverProcAddressQueryResult query_status{};
    all_passed &= expect(get_proc_v2 != nullptr &&
                             get_proc_v2("cuMemAllocAsync", &queried_symbol, CUDA_VERSION, 0,
                                         &query_status) == CUDA_SUCCESS &&
                             queried_symbol == reinterpret_cast<void*>(async_allocate) &&
                             query_status == CU_GET_PROC_ADDRESS_SUCCESS,
                         "v2 cuGetProcAddress did not return the async interceptor wrapper");
    queried_symbol = nullptr;
    query_status = {};
    all_passed &= expect(get_proc_v2 != nullptr &&
                             get_proc_v2("cuMemAllocAsync_ptsz", &queried_symbol, CUDA_VERSION, 0,
                                         &query_status) == CUDA_SUCCESS &&
                             queried_symbol == reinterpret_cast<void*>(async_allocate_ptsz) &&
                             query_status == CU_GET_PROC_ADDRESS_SUCCESS,
                         "v2 cuGetProcAddress did not return the PTDS async wrapper");
    queried_symbol = nullptr;
    all_passed &=
        expect(legacy_get_proc != nullptr &&
                   legacy_get_proc("cuLaunchKernel", &queried_symbol, CUDA_VERSION,
                                   CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM) == CUDA_SUCCESS &&
                   queried_symbol == reinterpret_cast<void*>(launch_kernel_ptsz),
               "cuGetProcAddress did not return the PTDS kernel-launch wrapper");
    queried_symbol = nullptr;
    all_passed &= expect(
        legacy_get_proc != nullptr &&
            legacy_get_proc("cuGraphLaunch", &queried_symbol, CUDA_VERSION, 0) == CUDA_SUCCESS &&
            queried_symbol == reinterpret_cast<void*>(graph_launch),
        "legacy cuGetProcAddress did not return the graph-launch wrapper");
    queried_symbol = nullptr;
    query_status = {};
    all_passed &= expect(get_proc_v2 != nullptr &&
                             get_proc_v2("cuGraphLaunch_ptsz", &queried_symbol, CUDA_VERSION, 0,
                                         &query_status) == CUDA_SUCCESS &&
                             queried_symbol == reinterpret_cast<void*>(graph_launch_ptsz) &&
                             query_status == CU_GET_PROC_ADDRESS_SUCCESS,
                         "v2 cuGetProcAddress did not return the PTDS graph-launch wrapper");
    queried_symbol = nullptr;
    query_status = {};
    all_passed &= expect(get_proc_v2 != nullptr &&
                             get_proc_v2("cuMemSetAccess", &queried_symbol, CUDA_VERSION, 0,
                                         &query_status) == CUDA_SUCCESS &&
                             queried_symbol == reinterpret_cast<void*>(set_access) &&
                             query_status == CU_GET_PROC_ADDRESS_SUCCESS,
                         "v2 cuGetProcAddress did not return the VMM access wrapper");

    queried_symbol = nullptr;
    query_status = {};
    all_passed &= expect(get_proc_v2 != nullptr &&
                             get_proc_v2("cuMemMapArrayAsync", &queried_symbol, CUDA_VERSION, 0,
                                         &query_status) == CUDA_SUCCESS &&
                             queried_symbol == reinterpret_cast<void*>(map_array_async) &&
                             query_status == CU_GET_PROC_ADDRESS_SUCCESS,
                         "v2 cuGetProcAddress did not return the sparse array mapping wrapper");

    queried_symbol = nullptr;
    query_status = {};
    all_passed &= expect(get_proc_v2 != nullptr &&
                             get_proc_v2("cuMemGetAllocationGranularity", &queried_symbol,
                                         CUDA_VERSION, 0, &query_status) == CUDA_SUCCESS &&
                             queried_symbol == reinterpret_cast<void*>(get_granularity) &&
                             query_status == CU_GET_PROC_ADDRESS_SUCCESS,
                         "v2 cuGetProcAddress did not return the VMM query wrapper");

    queried_symbol = nullptr;
    query_status = {};
    all_passed &= expect(get_proc_v2 != nullptr &&
                             get_proc_v2("cuIpcOpenMemHandle_v2", &queried_symbol, CUDA_VERSION, 0,
                                         &query_status) == CUDA_SUCCESS &&
                             queried_symbol == reinterpret_cast<void*>(ipc_open_handle_v2) &&
                             query_status == CU_GET_PROC_ADDRESS_SUCCESS,
                         "v2 cuGetProcAddress did not return the IPC wrapper");
    queried_symbol = nullptr;
    query_status = {};
    all_passed &= expect(legacy_get_proc != nullptr &&
                             legacy_get_proc("cuImportExternalMemory", &queried_symbol,
                                             CUDA_VERSION, 0) == CUDA_SUCCESS &&
                             queried_symbol == reinterpret_cast<void*>(import_external_memory),
                         "legacy cuGetProcAddress did not return the external-memory wrapper");
    queried_symbol = nullptr;
    query_status = {};
    all_passed &= expect(get_proc_v2 != nullptr &&
                             get_proc_v2("cuArrayCreate_v2", &queried_symbol, CUDA_VERSION, 0,
                                         &query_status) == CUDA_SUCCESS &&
                             queried_symbol == reinterpret_cast<void*>(array_create_v2) &&
                             query_status == CU_GET_PROC_ADDRESS_SUCCESS,
                         "v2 cuGetProcAddress did not return the array wrapper");
    queried_symbol = nullptr;
    query_status = {};
    all_passed &= expect(get_proc_v2 != nullptr &&
                             get_proc_v2("cuGraphicsMapResources", &queried_symbol, CUDA_VERSION, 0,
                                         &query_status) == CUDA_SUCCESS &&
                             queried_symbol == reinterpret_cast<void*>(graphics_map_resources) &&
                             query_status == CU_GET_PROC_ADDRESS_SUCCESS,
                         "v2 cuGetProcAddress did not return the graphics mapping wrapper");
    queried_symbol = nullptr;
    query_status = {};
    all_passed &= expect(legacy_get_proc != nullptr &&
                             legacy_get_proc("cuGraphicsResourceGetMappedPointer_v2",
                                             &queried_symbol, CUDA_VERSION, 0) == CUDA_SUCCESS &&
                             queried_symbol == reinterpret_cast<void*>(graphics_get_mapped_pointer),
                         "legacy cuGetProcAddress did not return the graphics pointer wrapper");

    if (runtime_handle != nullptr) {
        dlclose(runtime_handle);
    }
    all_passed &= expect(nvml_shutdown() == NVML_SUCCESS, "NVML shutdown failed");

    if (shared_mode && !shared_tenant_id.empty()) {
        all_passed &= expect(glimmer::control::SharedMemoryQuota::remove_region(shared_tenant_id),
                             "shared test region could not be removed");
    }
    return all_passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
