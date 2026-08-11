#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "internal/driver_api_interceptor.h"

#include <cuda.h>

#ifdef cuGetProcAddress
#undef cuGetProcAddress
#endif
#ifdef cuDeviceTotalMem
#undef cuDeviceTotalMem
#endif
#ifdef cuMemAlloc
#undef cuMemAlloc
#endif
#ifdef cuMemAllocPitch
#undef cuMemAllocPitch
#endif
#ifdef cuMemFree
#undef cuMemFree
#endif
#ifdef cuMemGetInfo
#undef cuMemGetInfo
#endif
#ifdef cuCtxDestroy
#undef cuCtxDestroy
#endif
#ifdef cuCtxSynchronize
#undef cuCtxSynchronize
#endif
#ifdef cuMemAllocAsync
#undef cuMemAllocAsync
#endif
#ifdef cuMemAllocFromPoolAsync
#undef cuMemAllocFromPoolAsync
#endif
#ifdef cuMemCreate
#undef cuMemCreate
#endif
#ifdef cuMemRelease
#undef cuMemRelease
#endif
#ifdef cuMemAddressReserve
#undef cuMemAddressReserve
#endif
#ifdef cuMemAddressFree
#undef cuMemAddressFree
#endif
#ifdef cuMemMap
#undef cuMemMap
#endif
#ifdef cuMemUnmap
#undef cuMemUnmap
#endif
#ifdef cuMemSetAccess
#undef cuMemSetAccess
#endif
#ifdef cuMemGetAddressRange
#undef cuMemGetAddressRange
#endif
#ifdef cuMemFreeAsync
#undef cuMemFreeAsync
#endif
#ifdef cuStreamGetDevice
#undef cuStreamGetDevice
#endif
#ifdef cuStreamGetCtx
#undef cuStreamGetCtx
#endif
#ifdef cuStreamGetCtx_v2
#undef cuStreamGetCtx_v2
#endif
#ifdef cuStreamQuery
#undef cuStreamQuery
#endif
#ifdef cuStreamSynchronize
#undef cuStreamSynchronize
#endif
#ifdef cuStreamDestroy
#undef cuStreamDestroy
#endif

namespace {

template <typename Function>
CUresult guard_cuda_boundary(Function&& function) noexcept {
    try {
        return function();
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}

}  // namespace

extern "C" CUresult CUDAAPI cuMemAlloc_v2(CUdeviceptr* device_pointer, std::size_t memory_bytes) {
    return guard_cuda_boundary([device_pointer, memory_bytes] {
        return glimmer::interceptor::intercept_mem_alloc(device_pointer, memory_bytes);
    });
}

extern "C" CUresult CUDAAPI cuMemAlloc(CUdeviceptr* device_pointer, std::size_t memory_bytes) {
    return cuMemAlloc_v2(device_pointer, memory_bytes);
}

extern "C" CUresult CUDAAPI cuInit(unsigned int flags) {
    return guard_cuda_boundary([flags] { return glimmer::interceptor::intercept_init(flags); });
}

extern "C" CUresult CUDAAPI cuMemAllocManaged(CUdeviceptr* device_pointer, std::size_t memory_bytes,
                                              unsigned int flags) {
    return guard_cuda_boundary([device_pointer, memory_bytes, flags] {
        return glimmer::interceptor::intercept_mem_alloc_managed(device_pointer, memory_bytes,
                                                                 flags);
    });
}

extern "C" CUresult CUDAAPI cuMemAllocPitch_v2(CUdeviceptr* device_pointer, std::size_t* pitch,
                                               std::size_t width_bytes, std::size_t height,
                                               unsigned int element_size_bytes) {
    return guard_cuda_boundary([device_pointer, pitch, width_bytes, height, element_size_bytes] {
        return glimmer::interceptor::intercept_mem_alloc_pitch(device_pointer, pitch, width_bytes,
                                                               height, element_size_bytes);
    });
}

extern "C" CUresult CUDAAPI cuMemAllocPitch(CUdeviceptr* device_pointer, std::size_t* pitch,
                                            std::size_t width_bytes, std::size_t height,
                                            unsigned int element_size_bytes) {
    return cuMemAllocPitch_v2(device_pointer, pitch, width_bytes, height, element_size_bytes);
}

extern "C" CUresult CUDAAPI cuMemAllocAsync(CUdeviceptr* device_pointer, std::size_t memory_bytes,
                                            CUstream stream) {
    return guard_cuda_boundary([device_pointer, memory_bytes, stream] {
        return glimmer::interceptor::intercept_mem_alloc_async(device_pointer, memory_bytes, stream,
                                                               false);
    });
}

extern "C" CUresult CUDAAPI cuMemAllocAsync_ptsz(CUdeviceptr* device_pointer,
                                                 std::size_t memory_bytes, CUstream stream) {
    return guard_cuda_boundary([device_pointer, memory_bytes, stream] {
        return glimmer::interceptor::intercept_mem_alloc_async(device_pointer, memory_bytes, stream,
                                                               true);
    });
}

extern "C" CUresult CUDAAPI cuMemAllocFromPoolAsync(CUdeviceptr* device_pointer,
                                                    std::size_t memory_bytes, CUmemoryPool pool,
                                                    CUstream stream) {
    return guard_cuda_boundary([device_pointer, memory_bytes, pool, stream] {
        return glimmer::interceptor::intercept_mem_alloc_from_pool_async(
            device_pointer, memory_bytes, pool, stream, false);
    });
}

extern "C" CUresult CUDAAPI cuMemAllocFromPoolAsync_ptsz(CUdeviceptr* device_pointer,
                                                         std::size_t memory_bytes,
                                                         CUmemoryPool pool, CUstream stream) {
    return guard_cuda_boundary([device_pointer, memory_bytes, pool, stream] {
        return glimmer::interceptor::intercept_mem_alloc_from_pool_async(
            device_pointer, memory_bytes, pool, stream, true);
    });
}

extern "C" CUresult CUDAAPI cuMemCreate(CUmemGenericAllocationHandle* handle,
                                        std::size_t memory_bytes, const CUmemAllocationProp* prop,
                                        unsigned long long flags) {
    return guard_cuda_boundary([handle, memory_bytes, prop, flags] {
        return glimmer::interceptor::intercept_mem_create(handle, memory_bytes, prop, flags);
    });
}

extern "C" CUresult CUDAAPI cuMemRelease(CUmemGenericAllocationHandle handle) {
    return guard_cuda_boundary(
        [handle] { return glimmer::interceptor::intercept_mem_release(handle); });
}

extern "C" CUresult CUDAAPI cuMemAddressReserve(CUdeviceptr* device_pointer,
                                                std::size_t memory_bytes, std::size_t alignment,
                                                CUdeviceptr requested_address,
                                                unsigned long long flags) {
    return guard_cuda_boundary([device_pointer, memory_bytes, alignment, requested_address, flags] {
        return glimmer::interceptor::intercept_mem_address_reserve(
            device_pointer, memory_bytes, alignment, requested_address, flags);
    });
}

extern "C" CUresult CUDAAPI cuMemAddressFree(CUdeviceptr device_pointer, std::size_t memory_bytes) {
    return guard_cuda_boundary([device_pointer, memory_bytes] {
        return glimmer::interceptor::intercept_mem_address_free(device_pointer, memory_bytes);
    });
}

extern "C" CUresult CUDAAPI cuMemMap(CUdeviceptr device_pointer, std::size_t memory_bytes,
                                     std::size_t offset, CUmemGenericAllocationHandle handle,
                                     unsigned long long flags) {
    return guard_cuda_boundary([device_pointer, memory_bytes, offset, handle, flags] {
        return glimmer::interceptor::intercept_mem_map(device_pointer, memory_bytes, offset, handle,
                                                       flags);
    });
}

extern "C" CUresult CUDAAPI cuMemUnmap(CUdeviceptr device_pointer, std::size_t memory_bytes) {
    return guard_cuda_boundary([device_pointer, memory_bytes] {
        return glimmer::interceptor::intercept_mem_unmap(device_pointer, memory_bytes);
    });
}

extern "C" CUresult CUDAAPI cuMemSetAccess(CUdeviceptr device_pointer, std::size_t memory_bytes,
                                           const CUmemAccessDesc* access_descriptors,
                                           std::size_t descriptor_count) {
    return guard_cuda_boundary(
        [device_pointer, memory_bytes, access_descriptors, descriptor_count] {
            return glimmer::interceptor::intercept_mem_set_access(
                device_pointer, memory_bytes, access_descriptors, descriptor_count);
        });
}

extern "C" CUresult CUDAAPI cuMemGetAddressRange_v2(CUdeviceptr* base_pointer,
                                                    std::size_t* memory_bytes,
                                                    CUdeviceptr device_pointer) {
    return guard_cuda_boundary([base_pointer, memory_bytes, device_pointer] {
        return glimmer::interceptor::intercept_mem_get_address_range(base_pointer, memory_bytes,
                                                                     device_pointer);
    });
}

extern "C" CUresult CUDAAPI cuMemGetAddressRange(CUdeviceptr* base_pointer,
                                                 std::size_t* memory_bytes,
                                                 CUdeviceptr device_pointer) {
    return cuMemGetAddressRange_v2(base_pointer, memory_bytes, device_pointer);
}

extern "C" CUresult CUDAAPI cuMemGetAccess(unsigned long long* flags, const CUmemLocation* location,
                                           CUdeviceptr device_pointer) {
    return guard_cuda_boundary([flags, location, device_pointer] {
        return glimmer::interceptor::intercept_mem_get_access(flags, location, device_pointer);
    });
}

extern "C" CUresult CUDAAPI cuMemExportToShareableHandle(void* shareable_handle,
                                                         CUmemGenericAllocationHandle handle,
                                                         CUmemAllocationHandleType handle_type,
                                                         unsigned long long flags) {
    return guard_cuda_boundary([shareable_handle, handle, handle_type, flags] {
        return glimmer::interceptor::intercept_mem_export_to_shareable_handle(
            shareable_handle, handle, handle_type, flags);
    });
}

extern "C" CUresult CUDAAPI cuMemImportFromShareableHandle(CUmemGenericAllocationHandle* handle,
                                                           void* os_handle,
                                                           CUmemAllocationHandleType handle_type) {
    return guard_cuda_boundary([handle, os_handle, handle_type] {
        return glimmer::interceptor::intercept_mem_import_from_shareable_handle(handle, os_handle,
                                                                                handle_type);
    });
}

extern "C" CUresult CUDAAPI cuMemGetAllocationGranularity(std::size_t* granularity,
                                                          const CUmemAllocationProp* prop,
                                                          CUmemAllocationGranularity_flags option) {
    return guard_cuda_boundary([granularity, prop, option] {
        return glimmer::interceptor::intercept_mem_get_allocation_granularity(granularity, prop,
                                                                              option);
    });
}

extern "C" CUresult CUDAAPI cuMemGetAllocationPropertiesFromHandle(
    CUmemAllocationProp* prop, CUmemGenericAllocationHandle handle) {
    return guard_cuda_boundary([prop, handle] {
        return glimmer::interceptor::intercept_mem_get_allocation_properties(prop, handle);
    });
}

extern "C" CUresult CUDAAPI cuMemRetainAllocationHandle(CUmemGenericAllocationHandle* handle,
                                                        void* device_pointer) {
    return guard_cuda_boundary([handle, device_pointer] {
        return glimmer::interceptor::intercept_mem_retain_allocation_handle(handle, device_pointer);
    });
}

extern "C" CUresult CUDAAPI cuMemPoolTrimTo(CUmemoryPool pool, std::size_t min_bytes_to_keep) {
    return guard_cuda_boundary([pool, min_bytes_to_keep] {
        return glimmer::interceptor::intercept_mem_pool_trim_to(pool, min_bytes_to_keep);
    });
}

extern "C" CUresult CUDAAPI cuMemPoolSetAttribute(CUmemoryPool pool, CUmemPool_attribute attribute,
                                                  void* value) {
    return guard_cuda_boundary([pool, attribute, value] {
        return glimmer::interceptor::intercept_mem_pool_set_attribute(pool, attribute, value);
    });
}

extern "C" CUresult CUDAAPI cuMemPoolGetAttribute(CUmemoryPool pool, CUmemPool_attribute attribute,
                                                  void* value) {
    return guard_cuda_boundary([pool, attribute, value] {
        return glimmer::interceptor::intercept_mem_pool_get_attribute(pool, attribute, value);
    });
}

extern "C" CUresult CUDAAPI cuMemPoolSetAccess(CUmemoryPool pool,
                                               const CUmemAccessDesc* access_descriptors,
                                               std::size_t descriptor_count) {
    return guard_cuda_boundary([pool, access_descriptors, descriptor_count] {
        return glimmer::interceptor::intercept_mem_pool_set_access(pool, access_descriptors,
                                                                   descriptor_count);
    });
}

extern "C" CUresult CUDAAPI cuMemPoolGetAccess(CUmemAccess_flags* flags, CUmemoryPool pool,
                                               CUmemLocation* location) {
    return guard_cuda_boundary([flags, pool, location] {
        return glimmer::interceptor::intercept_mem_pool_get_access(flags, pool, location);
    });
}

extern "C" CUresult CUDAAPI cuMemPoolCreate(CUmemoryPool* pool, const CUmemPoolProps* properties) {
    return guard_cuda_boundary([pool, properties] {
        return glimmer::interceptor::intercept_mem_pool_create(pool, properties);
    });
}

extern "C" CUresult CUDAAPI cuMemPoolDestroy(CUmemoryPool pool) {
    return guard_cuda_boundary(
        [pool] { return glimmer::interceptor::intercept_mem_pool_destroy(pool); });
}

extern "C" CUresult CUDAAPI cuDeviceGetMemPool(CUmemoryPool* pool, CUdevice device) {
    return guard_cuda_boundary([pool, device] {
        return glimmer::interceptor::intercept_device_get_mem_pool(pool, device);
    });
}

extern "C" CUresult CUDAAPI cuDeviceSetMemPool(CUdevice device, CUmemoryPool pool) {
    return guard_cuda_boundary([device, pool] {
        return glimmer::interceptor::intercept_device_set_mem_pool(device, pool);
    });
}

extern "C" CUresult CUDAAPI cuDeviceGetDefaultMemPool(CUmemoryPool* pool, CUdevice device) {
    return guard_cuda_boundary([pool, device] {
        return glimmer::interceptor::intercept_device_get_default_mem_pool(pool, device);
    });
}

extern "C" CUresult CUDAAPI cuMemGetDefaultMemPool(CUmemoryPool* pool, CUmemLocation* location,
                                                   CUmemAllocationType allocation_type) {
    return guard_cuda_boundary([pool, location, allocation_type] {
        return glimmer::interceptor::intercept_mem_get_default_mem_pool(pool, location,
                                                                        allocation_type);
    });
}

extern "C" CUresult CUDAAPI cuMemGetMemPool(CUmemoryPool* pool, CUmemLocation* location,
                                            CUmemAllocationType allocation_type) {
    return guard_cuda_boundary([pool, location, allocation_type] {
        return glimmer::interceptor::intercept_mem_get_mem_pool(pool, location, allocation_type);
    });
}

extern "C" CUresult CUDAAPI cuMemSetMemPool(CUmemLocation* location,
                                            CUmemAllocationType allocation_type,
                                            CUmemoryPool pool) {
    return guard_cuda_boundary([location, allocation_type, pool] {
        return glimmer::interceptor::intercept_mem_set_mem_pool(location, allocation_type, pool);
    });
}

extern "C" CUresult CUDAAPI cuMemPoolExportToShareableHandle(void* handle_out, CUmemoryPool pool,
                                                             CUmemAllocationHandleType handle_type,
                                                             unsigned long long flags) {
    return guard_cuda_boundary([handle_out, pool, handle_type, flags] {
        return glimmer::interceptor::intercept_mem_pool_export_to_shareable_handle(
            handle_out, pool, handle_type, flags);
    });
}

extern "C" CUresult CUDAAPI cuMemPoolImportFromShareableHandle(
    CUmemoryPool* pool_out, void* handle, CUmemAllocationHandleType handle_type,
    unsigned long long flags) {
    return guard_cuda_boundary([pool_out, handle, handle_type, flags] {
        return glimmer::interceptor::intercept_mem_pool_import_from_shareable_handle(
            pool_out, handle, handle_type, flags);
    });
}

extern "C" CUresult CUDAAPI cuMemPoolExportPointer(CUmemPoolPtrExportData* share_data_out,
                                                   CUdeviceptr device_pointer) {
    return guard_cuda_boundary([share_data_out, device_pointer] {
        return glimmer::interceptor::intercept_mem_pool_export_pointer(share_data_out,
                                                                       device_pointer);
    });
}

extern "C" CUresult CUDAAPI cuMemPoolImportPointer(CUdeviceptr* pointer_out, CUmemoryPool pool,
                                                   CUmemPoolPtrExportData* share_data) {
    return guard_cuda_boundary([pointer_out, pool, share_data] {
        return glimmer::interceptor::intercept_mem_pool_import_pointer(pointer_out, pool,
                                                                       share_data);
    });
}

extern "C" CUresult CUDAAPI cuMemFree_v2(CUdeviceptr device_pointer) {
    return guard_cuda_boundary(
        [device_pointer] { return glimmer::interceptor::intercept_mem_free(device_pointer); });
}

extern "C" CUresult CUDAAPI cuMemFree(CUdeviceptr device_pointer) {
    return cuMemFree_v2(device_pointer);
}

extern "C" CUresult CUDAAPI cuMemFreeAsync(CUdeviceptr device_pointer, CUstream stream) {
    return guard_cuda_boundary([device_pointer, stream] {
        return glimmer::interceptor::intercept_mem_free_async(device_pointer, stream, false);
    });
}

extern "C" CUresult CUDAAPI cuMemFreeAsync_ptsz(CUdeviceptr device_pointer, CUstream stream) {
    return guard_cuda_boundary([device_pointer, stream] {
        return glimmer::interceptor::intercept_mem_free_async(device_pointer, stream, true);
    });
}

extern "C" CUresult CUDAAPI cuMemGetInfo_v2(std::size_t* free_bytes, std::size_t* total_bytes) {
    return guard_cuda_boundary([free_bytes, total_bytes] {
        return glimmer::interceptor::intercept_mem_get_info(free_bytes, total_bytes);
    });
}

extern "C" CUresult CUDAAPI cuMemGetInfo(std::size_t* free_bytes, std::size_t* total_bytes) {
    return cuMemGetInfo_v2(free_bytes, total_bytes);
}

extern "C" CUresult CUDAAPI cuDeviceTotalMem_v2(std::size_t* total_bytes, CUdevice device) {
    return guard_cuda_boundary([total_bytes, device] {
        return glimmer::interceptor::intercept_device_total_mem(total_bytes, device);
    });
}

extern "C" CUresult CUDAAPI cuDeviceTotalMem(std::size_t* total_bytes, CUdevice device) {
    return cuDeviceTotalMem_v2(total_bytes, device);
}

extern "C" CUresult CUDAAPI cuCtxGetCurrent(CUcontext* context) {
    return guard_cuda_boundary(
        [context] { return glimmer::interceptor::intercept_context_get_current(context); });
}

extern "C" CUresult CUDAAPI cuCtxGetDevice(CUdevice* device) {
    return guard_cuda_boundary(
        [device] { return glimmer::interceptor::intercept_context_get_device(device); });
}

extern "C" CUresult CUDAAPI cuCtxDestroy_v2(CUcontext context) {
    return guard_cuda_boundary(
        [context] { return glimmer::interceptor::intercept_context_destroy(context); });
}

extern "C" CUresult CUDAAPI cuCtxDestroy(CUcontext context) {
    return cuCtxDestroy_v2(context);
}

extern "C" CUresult CUDAAPI cuStreamGetDevice(CUstream stream, CUdevice* device) {
    return guard_cuda_boundary([stream, device] {
        return glimmer::interceptor::intercept_stream_get_device(stream, device, false);
    });
}

extern "C" CUresult CUDAAPI cuStreamGetDevice_ptsz(CUstream stream, CUdevice* device) {
    return guard_cuda_boundary([stream, device] {
        return glimmer::interceptor::intercept_stream_get_device(stream, device, true);
    });
}

extern "C" CUresult CUDAAPI cuStreamGetCtx(CUstream stream, CUcontext* context) {
    return guard_cuda_boundary([stream, context] {
        return glimmer::interceptor::intercept_stream_get_context(stream, context, false);
    });
}

extern "C" CUresult CUDAAPI cuStreamGetCtx_ptsz(CUstream stream, CUcontext* context) {
    return guard_cuda_boundary([stream, context] {
        return glimmer::interceptor::intercept_stream_get_context(stream, context, true);
    });
}

extern "C" CUresult CUDAAPI cuStreamQuery(CUstream stream) {
    return guard_cuda_boundary(
        [stream] { return glimmer::interceptor::intercept_stream_query(stream, false); });
}

extern "C" CUresult CUDAAPI cuStreamQuery_ptsz(CUstream stream) {
    return guard_cuda_boundary(
        [stream] { return glimmer::interceptor::intercept_stream_query(stream, true); });
}

extern "C" CUresult CUDAAPI cuStreamSynchronize(CUstream stream) {
    return guard_cuda_boundary(
        [stream] { return glimmer::interceptor::intercept_stream_synchronize(stream, false); });
}

extern "C" CUresult CUDAAPI cuStreamSynchronize_ptsz(CUstream stream) {
    return guard_cuda_boundary(
        [stream] { return glimmer::interceptor::intercept_stream_synchronize(stream, true); });
}

extern "C" CUresult CUDAAPI cuStreamDestroy_v2(CUstream stream) {
    return guard_cuda_boundary(
        [stream] { return glimmer::interceptor::intercept_stream_destroy(stream); });
}

extern "C" CUresult CUDAAPI cuStreamDestroy(CUstream stream) {
    return cuStreamDestroy_v2(stream);
}

extern "C" CUresult CUDAAPI cuCtxSynchronize() {
    return guard_cuda_boundary(
        [] { return glimmer::interceptor::intercept_context_synchronize(); });
}

extern "C" CUresult CUDAAPI cuGetProcAddress(const char* symbol, void** function_pointer,
                                             int cuda_version, cuuint64_t flags) {
    return guard_cuda_boundary([symbol, function_pointer, cuda_version, flags] {
        return glimmer::interceptor::intercept_get_proc_address(symbol, function_pointer,
                                                                cuda_version, flags);
    });
}

extern "C" CUresult CUDAAPI cuGetProcAddress_v2(const char* symbol, void** function_pointer,
                                                int cuda_version, cuuint64_t flags,
                                                CUdriverProcAddressQueryResult* symbol_status) {
    return guard_cuda_boundary([symbol, function_pointer, cuda_version, flags, symbol_status] {
        return glimmer::interceptor::intercept_get_proc_address_v2(
            symbol, function_pointer, cuda_version, flags, symbol_status);
    });
}
