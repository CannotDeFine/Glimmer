#include "internal/driver_dispatch.h"

#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <string_view>

namespace {

using glimmer::interceptor::DriverDispatch;
using glimmer::interceptor::DriverFunctionTable;
using glimmer::interceptor::is_inside_driver_call;

bool g_guard_failed = false;

bool expect(bool condition, std::string_view message) {
    if (condition) {
        return true;
    }

    std::cerr << message << '\n';
    return false;
}

void check_driver_guard() {
    if (!is_inside_driver_call()) {
        g_guard_failed = true;
    }
}

void CUDAAPI fake_symbol() {}

CUresult CUDAAPI fake_init(unsigned int) {
    check_driver_guard();
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_launch_kernel(CUfunction, unsigned int, unsigned int, unsigned int,
                                    unsigned int, unsigned int, unsigned int, unsigned int,
                                    CUstream, void**, void**) {
    check_driver_guard();
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_mem_alloc(CUdeviceptr* device_pointer, std::size_t) {
    check_driver_guard();
    if (device_pointer == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *device_pointer = 0x1234;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_mem_alloc_managed(CUdeviceptr* device_pointer, std::size_t, unsigned int) {
    return fake_mem_alloc(device_pointer, 0);
}

CUresult CUDAAPI fake_mem_alloc_pitch(CUdeviceptr* device_pointer, std::size_t* pitch, std::size_t,
                                      std::size_t, unsigned int) {
    check_driver_guard();
    if (device_pointer == nullptr || pitch == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *device_pointer = 0x5678;
    *pitch = 64;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_mem_alloc_async(CUdeviceptr* device_pointer, std::size_t memory_bytes,
                                      CUstream) {
    return fake_mem_alloc(device_pointer, memory_bytes);
}

CUresult CUDAAPI fake_mem_alloc_async_ptsz(CUdeviceptr* device_pointer, std::size_t memory_bytes,
                                           CUstream stream) {
    return fake_mem_alloc_async(device_pointer, memory_bytes, stream);
}

CUresult CUDAAPI fake_mem_alloc_from_pool_async(CUdeviceptr* device_pointer,
                                                std::size_t memory_bytes, CUmemoryPool, CUstream) {
    return fake_mem_alloc(device_pointer, memory_bytes);
}

CUresult CUDAAPI fake_mem_alloc_from_pool_async_ptsz(CUdeviceptr* device_pointer,
                                                     std::size_t memory_bytes, CUmemoryPool pool,
                                                     CUstream stream) {
    return fake_mem_alloc_from_pool_async(device_pointer, memory_bytes, pool, stream);
}

CUresult CUDAAPI fake_mem_create(CUmemGenericAllocationHandle* handle, std::size_t,
                                 const CUmemAllocationProp*, unsigned long long) {
    check_driver_guard();
    if (handle == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *handle = 0x42;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_mem_release(CUmemGenericAllocationHandle) {
    check_driver_guard();
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_mem_address_reserve(CUdeviceptr* device_pointer, std::size_t memory_bytes,
                                          std::size_t, CUdeviceptr, unsigned long long) {
    check_driver_guard();
    if (device_pointer == nullptr || memory_bytes == 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *device_pointer = 0x4000;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_mem_address_free(CUdeviceptr device_pointer, std::size_t memory_bytes) {
    check_driver_guard();
    return device_pointer == 0 || memory_bytes == 0 ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
}

CUresult CUDAAPI fake_mem_map(CUdeviceptr device_pointer, std::size_t memory_bytes, std::size_t,
                              CUmemGenericAllocationHandle handle, unsigned long long) {
    check_driver_guard();
    return device_pointer == 0 || memory_bytes == 0 || handle == 0 ? CUDA_ERROR_INVALID_VALUE
                                                                   : CUDA_SUCCESS;
}

CUresult CUDAAPI fake_mem_unmap(CUdeviceptr device_pointer, std::size_t memory_bytes) {
    check_driver_guard();
    return device_pointer == 0 || memory_bytes == 0 ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
}

CUresult CUDAAPI fake_mem_set_access(CUdeviceptr device_pointer, std::size_t memory_bytes,
                                     const CUmemAccessDesc* access_descriptors,
                                     std::size_t descriptor_count) {
    check_driver_guard();
    if (device_pointer == 0 || memory_bytes == 0 ||
        (descriptor_count != 0 && access_descriptors == nullptr)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_mem_get_address_range(CUdeviceptr* base_pointer, std::size_t* memory_bytes,
                                            CUdeviceptr device_pointer) {
    check_driver_guard();
    if (base_pointer == nullptr || memory_bytes == nullptr || device_pointer == 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *base_pointer = device_pointer;
    *memory_bytes = 4096;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_mem_get_access(unsigned long long* flags, const CUmemLocation* location,
                                     CUdeviceptr device_pointer) {
    check_driver_guard();
    if (flags == nullptr || location == nullptr || device_pointer == 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_mem_export(void* shareable_handle, CUmemGenericAllocationHandle handle,
                                 CUmemAllocationHandleType, unsigned long long) {
    check_driver_guard();
    return shareable_handle == nullptr || handle == 0 ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
}

CUresult CUDAAPI fake_mem_import(CUmemGenericAllocationHandle* handle, void* os_handle,
                                 CUmemAllocationHandleType) {
    check_driver_guard();
    if (handle == nullptr || os_handle == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *handle = 0x43;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_mem_granularity(std::size_t* granularity, const CUmemAllocationProp* prop,
                                      CUmemAllocationGranularity_flags) {
    check_driver_guard();
    if (granularity == nullptr || prop == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *granularity = 65536;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_mem_properties(CUmemAllocationProp* prop,
                                     CUmemGenericAllocationHandle handle) {
    check_driver_guard();
    if (prop == nullptr || handle == 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *prop = {};
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_mem_retain(CUmemGenericAllocationHandle* handle, void* device_pointer) {
    check_driver_guard();
    if (handle == nullptr || device_pointer == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *handle = 0x44;
    return CUDA_SUCCESS;
}

std::uint8_t g_pool_token = 0;

CUresult CUDAAPI fake_pool_trim(CUmemoryPool pool, std::size_t) {
    check_driver_guard();
    return pool == nullptr ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
}

CUresult CUDAAPI fake_pool_set_attribute(CUmemoryPool pool, CUmemPool_attribute, void* value) {
    check_driver_guard();
    return pool == nullptr || value == nullptr ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
}

CUresult CUDAAPI fake_pool_get_attribute(CUmemoryPool pool, CUmemPool_attribute, void* value) {
    check_driver_guard();
    return pool == nullptr || value == nullptr ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
}

CUresult CUDAAPI fake_pool_set_access(CUmemoryPool pool, const CUmemAccessDesc* descriptors,
                                      std::size_t count) {
    check_driver_guard();
    return pool == nullptr || (count != 0 && descriptors == nullptr) ? CUDA_ERROR_INVALID_VALUE
                                                                     : CUDA_SUCCESS;
}

CUresult CUDAAPI fake_pool_get_access(CUmemAccess_flags* flags, CUmemoryPool pool,
                                      CUmemLocation* location) {
    check_driver_guard();
    if (flags == nullptr || pool == nullptr || location == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_pool_create(CUmemoryPool* pool, const CUmemPoolProps* properties) {
    check_driver_guard();
    if (pool == nullptr || properties == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *pool = reinterpret_cast<CUmemoryPool>(&g_pool_token);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_pool_destroy(CUmemoryPool pool) {
    check_driver_guard();
    return pool == nullptr ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
}

CUresult CUDAAPI fake_device_get_pool(CUmemoryPool* pool, CUdevice device) {
    check_driver_guard();
    if (pool == nullptr || device < 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *pool = reinterpret_cast<CUmemoryPool>(&g_pool_token);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_device_set_pool(CUdevice device, CUmemoryPool pool) {
    check_driver_guard();
    return device < 0 || pool == nullptr ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
}

CUresult CUDAAPI fake_device_get_default_pool(CUmemoryPool* pool, CUdevice device) {
    check_driver_guard();
    if (pool == nullptr || device < 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *pool = reinterpret_cast<CUmemoryPool>(&g_pool_token);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_get_pool(CUmemoryPool* pool, CUmemLocation* location, CUmemAllocationType) {
    check_driver_guard();
    if (pool == nullptr || location == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *pool = reinterpret_cast<CUmemoryPool>(&g_pool_token);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_set_pool(CUmemLocation* location, CUmemAllocationType, CUmemoryPool pool) {
    check_driver_guard();
    return location == nullptr || pool == nullptr ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
}

CUresult CUDAAPI fake_pool_export(void* handle_out, CUmemoryPool pool, CUmemAllocationHandleType,
                                  unsigned long long) {
    check_driver_guard();
    return handle_out == nullptr || pool == nullptr ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
}

CUresult CUDAAPI fake_pool_import(CUmemoryPool* pool_out, void* handle, CUmemAllocationHandleType,
                                  unsigned long long) {
    check_driver_guard();
    if (pool_out == nullptr || handle == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *pool_out = reinterpret_cast<CUmemoryPool>(&g_pool_token);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_pool_export_pointer(CUmemPoolPtrExportData* share_data_out,
                                          CUdeviceptr device_pointer) {
    check_driver_guard();
    return share_data_out == nullptr || device_pointer == 0 ? CUDA_ERROR_INVALID_VALUE
                                                            : CUDA_SUCCESS;
}

CUresult CUDAAPI fake_pool_import_pointer(CUdeviceptr* pointer_out, CUmemoryPool pool,
                                          CUmemPoolPtrExportData* share_data) {
    check_driver_guard();
    if (pointer_out == nullptr || pool == nullptr || share_data == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *pointer_out = 0x7000;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_mem_free(CUdeviceptr) {
    check_driver_guard();
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_mem_free_async(CUdeviceptr device_pointer, CUstream) {
    return fake_mem_free(device_pointer);
}

CUresult CUDAAPI fake_mem_free_async_ptsz(CUdeviceptr device_pointer, CUstream stream) {
    return fake_mem_free_async(device_pointer, stream);
}

CUresult CUDAAPI fake_mem_get_info(std::size_t* free_bytes, std::size_t* total_bytes) {
    check_driver_guard();
    if (free_bytes == nullptr || total_bytes == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *free_bytes = 100;
    *total_bytes = 200;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_device_total_mem(std::size_t* total_bytes, CUdevice) {
    check_driver_guard();
    if (total_bytes == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *total_bytes = 200;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_context_get_current(CUcontext* context) {
    check_driver_guard();
    if (context == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *context = nullptr;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_context_get_device(CUdevice* device) {
    check_driver_guard();
    if (device == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *device = 0;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_context_destroy(CUcontext) {
    check_driver_guard();
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_context_synchronize() {
    check_driver_guard();
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_stream_get_device(CUstream, CUdevice* device) {
    check_driver_guard();
    if (device == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *device = 0;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_stream_get_device_ptsz(CUstream stream, CUdevice* device) {
    return fake_stream_get_device(stream, device);
}

CUresult CUDAAPI fake_stream_get_context(CUstream, CUcontext* context) {
    check_driver_guard();
    if (context == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *context = nullptr;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_stream_get_context_ptsz(CUstream stream, CUcontext* context) {
    return fake_stream_get_context(stream, context);
}

CUresult CUDAAPI fake_stream_query(CUstream) {
    check_driver_guard();
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_stream_query_ptsz(CUstream stream) {
    return fake_stream_query(stream);
}

CUresult CUDAAPI fake_stream_synchronize(CUstream) {
    check_driver_guard();
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_stream_synchronize_ptsz(CUstream stream) {
    return fake_stream_synchronize(stream);
}

CUresult CUDAAPI fake_stream_destroy(CUstream) {
    check_driver_guard();
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_get_proc_address(const char*, void** function_pointer, int, cuuint64_t) {
    check_driver_guard();
    if (function_pointer == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *function_pointer = reinterpret_cast<void*>(&fake_symbol);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_get_proc_address_v2(const char*, void** function_pointer, int, cuuint64_t,
                                          CUdriverProcAddressQueryResult* symbol_status) {
    const CUresult result = fake_get_proc_address(nullptr, function_pointer, 0, 0);
    if (symbol_status != nullptr) {
        *symbol_status = CU_GET_PROC_ADDRESS_SUCCESS;
    }
    return result;
}

}  // namespace

int main() {
    const glimmer::interceptor::DriverFunctionTable functions{
        .init = &fake_init,
        .launch_kernel = &fake_launch_kernel,
        .launch_kernel_ptsz = &fake_launch_kernel,
        .mem_alloc = &fake_mem_alloc,
        .mem_alloc_managed = &fake_mem_alloc_managed,
        .mem_alloc_pitch = &fake_mem_alloc_pitch,
        .mem_alloc_async = &fake_mem_alloc_async,
        .mem_alloc_async_ptsz = &fake_mem_alloc_async_ptsz,
        .mem_alloc_from_pool_async = &fake_mem_alloc_from_pool_async,
        .mem_alloc_from_pool_async_ptsz = &fake_mem_alloc_from_pool_async_ptsz,
        .mem_create = &fake_mem_create,
        .mem_release = &fake_mem_release,
        .mem_address_reserve = &fake_mem_address_reserve,
        .mem_address_free = &fake_mem_address_free,
        .mem_map = &fake_mem_map,
        .mem_unmap = &fake_mem_unmap,
        .mem_set_access = &fake_mem_set_access,
        .mem_get_address_range = &fake_mem_get_address_range,
        .mem_get_access = &fake_mem_get_access,
        .mem_export_to_shareable_handle = &fake_mem_export,
        .mem_import_from_shareable_handle = &fake_mem_import,
        .mem_get_allocation_granularity = &fake_mem_granularity,
        .mem_get_allocation_properties = &fake_mem_properties,
        .mem_retain_allocation_handle = &fake_mem_retain,
        .mem_pool_trim_to = &fake_pool_trim,
        .mem_pool_set_attribute = &fake_pool_set_attribute,
        .mem_pool_get_attribute = &fake_pool_get_attribute,
        .mem_pool_set_access = &fake_pool_set_access,
        .mem_pool_get_access = &fake_pool_get_access,
        .mem_pool_create = &fake_pool_create,
        .mem_pool_destroy = &fake_pool_destroy,
        .device_get_mem_pool = &fake_device_get_pool,
        .device_set_mem_pool = &fake_device_set_pool,
        .device_get_default_mem_pool = &fake_device_get_default_pool,
        .mem_get_default_mem_pool = &fake_get_pool,
        .mem_get_mem_pool = &fake_get_pool,
        .mem_set_mem_pool = &fake_set_pool,
        .mem_pool_export_to_shareable_handle = &fake_pool_export,
        .mem_pool_import_from_shareable_handle = &fake_pool_import,
        .mem_pool_export_pointer = &fake_pool_export_pointer,
        .mem_pool_import_pointer = &fake_pool_import_pointer,
        .mem_free = &fake_mem_free,
        .mem_free_async = &fake_mem_free_async,
        .mem_free_async_ptsz = &fake_mem_free_async_ptsz,
        .mem_get_info = &fake_mem_get_info,
        .device_total_mem = &fake_device_total_mem,
        .context_synchronize = &fake_context_synchronize,
        .context_get_current = &fake_context_get_current,
        .context_get_device = &fake_context_get_device,
        .context_destroy = &fake_context_destroy,
        .stream_get_device = &fake_stream_get_device,
        .stream_get_device_ptsz = &fake_stream_get_device_ptsz,
        .stream_get_context = &fake_stream_get_context,
        .stream_get_context_ptsz = &fake_stream_get_context_ptsz,
        .stream_query = &fake_stream_query,
        .stream_query_ptsz = &fake_stream_query_ptsz,
        .stream_synchronize = &fake_stream_synchronize,
        .stream_synchronize_ptsz = &fake_stream_synchronize_ptsz,
        .stream_destroy = &fake_stream_destroy,
        .get_proc_address = &fake_get_proc_address,
        .get_proc_address_v2 = &fake_get_proc_address_v2,
    };
    const DriverDispatch dispatch(functions);
    bool all_passed = true;

    all_passed &= expect(dispatch.has_get_proc_address(), "legacy resolver was not available");
    all_passed &= expect(dispatch.has_get_proc_address_v2(), "v2 resolver was not available");
    all_passed &= expect(dispatch.has_launch_kernel(), "kernel launch was not available");
    all_passed &= expect(dispatch.has_launch_kernel_ptsz(), "PTDS kernel launch was not available");
    all_passed &= expect(dispatch.has_mem_alloc_managed(), "managed allocator was not available");
    all_passed &= expect(dispatch.has_mem_alloc_pitch(), "pitched allocator was not available");
    all_passed &= expect(dispatch.has_mem_alloc_async(), "async allocator was not available");
    all_passed &=
        expect(dispatch.has_mem_alloc_async_ptsz(), "PTDS async allocator was not available");
    all_passed &=
        expect(dispatch.has_mem_alloc_from_pool_async(), "pool async allocator was not available");
    all_passed &= expect(dispatch.has_mem_alloc_from_pool_async_ptsz(),
                         "PTDS pool async allocator was not available");
    all_passed &= expect(dispatch.has_mem_create(), "VMM allocator was not available");
    all_passed &= expect(dispatch.has_mem_release(), "VMM release was not available");
    all_passed &=
        expect(dispatch.has_mem_address_reserve(), "VMM address reserve was not available");
    all_passed &= expect(dispatch.has_mem_address_free(), "VMM address free was not available");
    all_passed &= expect(dispatch.has_mem_map(), "VMM map was not available");
    all_passed &= expect(dispatch.has_mem_unmap(), "VMM unmap was not available");
    all_passed &= expect(dispatch.has_mem_set_access(), "VMM access update was not available");
    all_passed &=
        expect(dispatch.has_mem_get_address_range(), "VMM address query was not available");
    all_passed &= expect(dispatch.has_mem_get_access(), "VMM access query was not available");
    all_passed &= expect(dispatch.has_mem_export_to_shareable_handle(),
                         "VMM handle export was not available");
    all_passed &= expect(dispatch.has_mem_import_from_shareable_handle(),
                         "VMM handle import was not available");
    all_passed &= expect(dispatch.has_mem_get_allocation_granularity(),
                         "VMM granularity query was not available");
    all_passed &= expect(dispatch.has_mem_get_allocation_properties(),
                         "VMM properties query was not available");
    all_passed &=
        expect(dispatch.has_mem_retain_allocation_handle(), "VMM handle retain was not available");
    all_passed &= expect(dispatch.has_mem_pool_trim_to(), "memory-pool trim was not available");
    all_passed &= expect(dispatch.has_mem_pool_set_attribute(),
                         "memory-pool attribute update was not available");
    all_passed &= expect(dispatch.has_mem_pool_get_attribute(),
                         "memory-pool attribute query was not available");
    all_passed &=
        expect(dispatch.has_mem_pool_set_access(), "memory-pool access update was not available");
    all_passed &=
        expect(dispatch.has_mem_pool_get_access(), "memory-pool access query was not available");
    all_passed &= expect(dispatch.has_mem_pool_create(), "memory-pool creation was not available");
    all_passed &=
        expect(dispatch.has_mem_pool_destroy(), "memory-pool destruction was not available");
    all_passed &=
        expect(dispatch.has_device_get_mem_pool(), "device memory-pool query was not available");
    all_passed &=
        expect(dispatch.has_device_set_mem_pool(), "device memory-pool update was not available");
    all_passed &= expect(dispatch.has_device_get_default_mem_pool(),
                         "device default memory-pool query was not available");
    all_passed &= expect(dispatch.has_mem_get_default_mem_pool(),
                         "default memory-pool query was not available");
    all_passed &= expect(dispatch.has_mem_get_mem_pool(), "memory-pool lookup was not available");
    all_passed &=
        expect(dispatch.has_mem_set_mem_pool(), "memory-pool selection was not available");
    all_passed &= expect(dispatch.has_mem_pool_export_to_shareable_handle(),
                         "memory-pool handle export was not available");
    all_passed &= expect(dispatch.has_mem_pool_import_from_shareable_handle(),
                         "memory-pool handle import was not available");
    all_passed &= expect(dispatch.has_mem_pool_export_pointer(),
                         "memory-pool pointer export was not available");
    all_passed &= expect(dispatch.has_mem_pool_import_pointer(),
                         "memory-pool pointer import was not available");
    all_passed &= expect(dispatch.has_mem_free_async(), "async free was not available");
    all_passed &= expect(dispatch.has_mem_free_async_ptsz(), "PTDS async free was not available");
    all_passed &= expect(dispatch.has_device_total_mem(), "capacity query was not available");
    all_passed &=
        expect(dispatch.has_context_synchronize(), "context synchronization was not available");
    all_passed &= expect(dispatch.has_context_queries(), "context queries were not available");
    all_passed &= expect(dispatch.has_context_destroy(), "context destruction was not available");
    all_passed &= expect(dispatch.has_stream_identity(), "stream identity was not available");
    all_passed &=
        expect(dispatch.has_stream_identity_ptsz(), "PTDS stream identity was not available");
    all_passed &= expect(dispatch.has_stream_query(), "stream query was not available");
    all_passed &= expect(dispatch.has_stream_query_ptsz(), "PTDS stream query was not available");
    all_passed &=
        expect(dispatch.has_stream_synchronize(), "stream synchronization was not available");
    all_passed &= expect(dispatch.has_stream_synchronize_ptsz(),
                         "PTDS stream synchronization was not available");
    all_passed &= expect(dispatch.has_stream_destroy(), "stream destruction was not available");

    CUdeviceptr device_pointer{};
    std::size_t pitch{};
    std::size_t free_bytes{};
    std::size_t total_bytes{};
    std::size_t address_size{};
    std::size_t granularity{};
    CUdeviceptr address_base{};
    unsigned long long access_flags{};
    CUmemAllocationProp allocation_properties{};
    CUmemAllocationProp allocation_prop{};
    allocation_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    allocation_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    allocation_prop.location.id = 0;
    CUmemAllocationHandleType handle_type = CU_MEM_HANDLE_TYPE_NONE;
    unsigned long long shareable_handle{};
    CUmemGenericAllocationHandle imported_handle{};
    CUmemGenericAllocationHandle retained_handle{};
    CUmemPoolProps pool_properties{};
    pool_properties.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
    pool_properties.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    CUmemoryPool pool{};
    CUmemoryPool queried_pool{};
    CUmemLocation pool_location = pool_properties.location;
    std::uint64_t pool_attribute_value{};
    CUmemAccess_flags pool_access_flags = CU_MEM_ACCESS_FLAGS_PROT_NONE;
    CUmemAccessDesc pool_access_descriptor{};
    CUmemPoolPtrExportData pool_pointer_data{};
    CUdeviceptr imported_pointer{};
    void* function_pointer = nullptr;
    CUdriverProcAddressQueryResult symbol_status{};
    CUcontext context{};
    CUdevice device{};
    all_passed &= expect(dispatch.init(0) == CUDA_SUCCESS, "fake cuInit failed");
    all_passed &= expect(dispatch.launch_kernel(nullptr, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr,
                                                nullptr) == CUDA_SUCCESS,
                         "fake cuLaunchKernel failed");
    all_passed &= expect(dispatch.launch_kernel_ptsz(nullptr, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr,
                                                     nullptr) == CUDA_SUCCESS,
                         "fake cuLaunchKernel_ptsz failed");
    all_passed &=
        expect(dispatch.mem_alloc(&device_pointer, 16) == CUDA_SUCCESS, "fake cuMemAlloc failed");
    all_passed &= expect(dispatch.mem_alloc_managed(&device_pointer, 16, 0) == CUDA_SUCCESS,
                         "fake cuMemAllocManaged failed");
    all_passed &= expect(dispatch.mem_alloc_async(&device_pointer, 16, nullptr) == CUDA_SUCCESS,
                         "fake cuMemAllocAsync failed");
    all_passed &=
        expect(dispatch.mem_alloc_async_ptsz(&device_pointer, 16, nullptr) == CUDA_SUCCESS,
               "fake cuMemAllocAsync_ptsz failed");
    all_passed &= expect(
        dispatch.mem_alloc_from_pool_async(&device_pointer, 16, nullptr, nullptr) == CUDA_SUCCESS,
        "fake cuMemAllocFromPoolAsync failed");
    all_passed &= expect(dispatch.mem_alloc_from_pool_async_ptsz(&device_pointer, 16, nullptr,
                                                                 nullptr) == CUDA_SUCCESS,
                         "fake cuMemAllocFromPoolAsync_ptsz failed");
    CUmemGenericAllocationHandle vmm_handle = 0;
    all_passed &= expect(
        dispatch.mem_create(&vmm_handle, 16, nullptr, 0) == CUDA_SUCCESS && vmm_handle == 0x42,
        "fake cuMemCreate failed");
    all_passed &=
        expect(dispatch.mem_release(vmm_handle) == CUDA_SUCCESS, "fake cuMemRelease failed");
    CUdeviceptr virtual_address = 0;
    all_passed &=
        expect(dispatch.mem_address_reserve(&virtual_address, 16, 0, 0, 0) == CUDA_SUCCESS &&
                   virtual_address != 0,
               "fake cuMemAddressReserve failed");
    all_passed &= expect(dispatch.mem_map(virtual_address, 16, 0, 1, 0) == CUDA_SUCCESS,
                         "fake cuMemMap failed");
    all_passed &= expect(dispatch.mem_set_access(virtual_address, 16, nullptr, 0) == CUDA_SUCCESS,
                         "fake cuMemSetAccess failed");
    all_passed &= expect(dispatch.mem_get_address_range(&address_base, &address_size,
                                                        virtual_address) == CUDA_SUCCESS &&
                             address_base == virtual_address && address_size == 4096,
                         "fake cuMemGetAddressRange failed");
    all_passed &= expect(dispatch.mem_get_access(&access_flags, &allocation_prop.location,
                                                 virtual_address) == CUDA_SUCCESS &&
                             access_flags == CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
                         "fake cuMemGetAccess failed");
    all_passed &= expect(dispatch.mem_export_to_shareable_handle(&shareable_handle, vmm_handle,
                                                                 handle_type, 0) == CUDA_SUCCESS,
                         "fake cuMemExportToShareableHandle failed");
    all_passed &= expect(dispatch.mem_import_from_shareable_handle(
                             &imported_handle, &shareable_handle, handle_type) == CUDA_SUCCESS &&
                             imported_handle != 0,
                         "fake cuMemImportFromShareableHandle failed");
    all_passed &= expect(
        dispatch.mem_get_allocation_granularity(&granularity, &allocation_prop,
                                                CU_MEM_ALLOC_GRANULARITY_MINIMUM) == CUDA_SUCCESS &&
            granularity == 65536,
        "fake cuMemGetAllocationGranularity failed");
    all_passed &= expect(
        dispatch.mem_get_allocation_properties(&allocation_properties, vmm_handle) == CUDA_SUCCESS,
        "fake cuMemGetAllocationPropertiesFromHandle failed");
    all_passed &= expect(
        dispatch.mem_retain_allocation_handle(&retained_handle, &virtual_address) == CUDA_SUCCESS &&
            retained_handle != 0,
        "fake cuMemRetainAllocationHandle failed");
    all_passed &=
        expect(dispatch.mem_pool_create(&pool, &pool_properties) == CUDA_SUCCESS && pool != nullptr,
               "fake cuMemPoolCreate failed");
    all_passed &= expect(dispatch.mem_pool_set_attribute(pool, CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                                                         &pool_attribute_value) == CUDA_SUCCESS,
                         "fake cuMemPoolSetAttribute failed");
    all_passed &= expect(dispatch.mem_pool_get_attribute(pool, CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                                                         &pool_attribute_value) == CUDA_SUCCESS,
                         "fake cuMemPoolGetAttribute failed");
    all_passed &=
        expect(dispatch.mem_pool_set_access(pool, &pool_access_descriptor, 1) == CUDA_SUCCESS,
               "fake cuMemPoolSetAccess failed");
    all_passed &= expect(
        dispatch.mem_pool_get_access(&pool_access_flags, pool, &pool_location) == CUDA_SUCCESS,
        "fake cuMemPoolGetAccess failed");
    all_passed &=
        expect(dispatch.mem_pool_trim_to(pool, 0) == CUDA_SUCCESS, "fake cuMemPoolTrimTo failed");
    all_passed &= expect(
        dispatch.device_get_mem_pool(&queried_pool, 0) == CUDA_SUCCESS && queried_pool != nullptr,
        "fake cuDeviceGetMemPool failed");
    all_passed &= expect(dispatch.device_set_mem_pool(0, pool) == CUDA_SUCCESS,
                         "fake cuDeviceSetMemPool failed");
    all_passed &= expect(dispatch.device_get_default_mem_pool(&queried_pool, 0) == CUDA_SUCCESS &&
                             queried_pool != nullptr,
                         "fake cuDeviceGetDefaultMemPool failed");
    all_passed &=
        expect(dispatch.mem_get_default_mem_pool(&queried_pool, &pool_location,
                                                 CU_MEM_ALLOCATION_TYPE_PINNED) == CUDA_SUCCESS,
               "fake cuMemGetDefaultMemPool failed");
    all_passed &= expect(dispatch.mem_get_mem_pool(&queried_pool, &pool_location,
                                                   CU_MEM_ALLOCATION_TYPE_PINNED) == CUDA_SUCCESS,
                         "fake cuMemGetMemPool failed");
    all_passed &= expect(dispatch.mem_set_mem_pool(&pool_location, CU_MEM_ALLOCATION_TYPE_PINNED,
                                                   pool) == CUDA_SUCCESS,
                         "fake cuMemSetMemPool failed");
    all_passed &= expect(dispatch.mem_pool_export_to_shareable_handle(
                             &shareable_handle, pool, handle_type, 0) == CUDA_SUCCESS,
                         "fake cuMemPoolExportToShareableHandle failed");
    all_passed &= expect(dispatch.mem_pool_import_from_shareable_handle(
                             &queried_pool, &shareable_handle, handle_type, 0) == CUDA_SUCCESS,
                         "fake cuMemPoolImportFromShareableHandle failed");
    all_passed &= expect(
        dispatch.mem_pool_export_pointer(&pool_pointer_data, virtual_address) == CUDA_SUCCESS,
        "fake cuMemPoolExportPointer failed");
    all_passed &= expect(dispatch.mem_pool_import_pointer(&imported_pointer, pool,
                                                          &pool_pointer_data) == CUDA_SUCCESS,
                         "fake cuMemPoolImportPointer failed");
    all_passed &=
        expect(dispatch.mem_pool_destroy(pool) == CUDA_SUCCESS, "fake cuMemPoolDestroy failed");
    all_passed &=
        expect(dispatch.mem_unmap(virtual_address, 16) == CUDA_SUCCESS, "fake cuMemUnmap failed");
    all_passed &= expect(dispatch.mem_address_free(virtual_address, 16) == CUDA_SUCCESS,
                         "fake cuMemAddressFree failed");
    all_passed &=
        expect(dispatch.mem_alloc_pitch(&device_pointer, &pitch, 16, 16, 4) == CUDA_SUCCESS,
               "fake cuMemAllocPitch failed");
    all_passed &=
        expect(dispatch.mem_free(device_pointer) == CUDA_SUCCESS, "fake cuMemFree failed");
    all_passed &= expect(dispatch.mem_free_async(device_pointer, nullptr) == CUDA_SUCCESS,
                         "fake cuMemFreeAsync failed");
    all_passed &= expect(dispatch.mem_free_async_ptsz(device_pointer, nullptr) == CUDA_SUCCESS,
                         "fake cuMemFreeAsync_ptsz failed");
    all_passed &= expect(dispatch.mem_get_info(&free_bytes, &total_bytes) == CUDA_SUCCESS &&
                             free_bytes == 100 && total_bytes == 200,
                         "fake cuMemGetInfo failed");
    all_passed &=
        expect(dispatch.device_total_mem(&total_bytes, 0) == CUDA_SUCCESS && total_bytes == 200,
               "fake cuDeviceTotalMem failed");
    all_passed &= expect(dispatch.context_get_current(&context) == CUDA_SUCCESS,
                         "fake cuCtxGetCurrent failed");
    all_passed &= expect(dispatch.context_get_device(&device) == CUDA_SUCCESS && device == 0,
                         "fake cuCtxGetDevice failed");
    all_passed &=
        expect(dispatch.context_destroy(context) == CUDA_SUCCESS, "fake cuCtxDestroy failed");
    all_passed &=
        expect(dispatch.context_synchronize() == CUDA_SUCCESS, "fake cuCtxSynchronize failed");
    all_passed &= expect(dispatch.stream_get_device(nullptr, &device) == CUDA_SUCCESS,
                         "fake cuStreamGetDevice failed");
    all_passed &= expect(dispatch.stream_get_device_ptsz(nullptr, &device) == CUDA_SUCCESS,
                         "fake cuStreamGetDevice_ptsz failed");
    all_passed &= expect(dispatch.stream_get_context(nullptr, &context) == CUDA_SUCCESS,
                         "fake cuStreamGetCtx failed");
    all_passed &= expect(dispatch.stream_get_context_ptsz(nullptr, &context) == CUDA_SUCCESS,
                         "fake cuStreamGetCtx_ptsz failed");
    all_passed &=
        expect(dispatch.stream_query(nullptr) == CUDA_SUCCESS, "fake cuStreamQuery failed");
    all_passed &= expect(dispatch.stream_query_ptsz(nullptr) == CUDA_SUCCESS,
                         "fake cuStreamQuery_ptsz failed");
    all_passed &= expect(dispatch.stream_synchronize(nullptr) == CUDA_SUCCESS,
                         "fake cuStreamSynchronize failed");
    all_passed &= expect(dispatch.stream_synchronize_ptsz(nullptr) == CUDA_SUCCESS,
                         "fake cuStreamSynchronize_ptsz failed");
    all_passed &=
        expect(dispatch.stream_destroy(nullptr) == CUDA_SUCCESS, "fake cuStreamDestroy failed");
    all_passed &=
        expect(dispatch.get_proc_address("fake", &function_pointer, 0, 0) == CUDA_SUCCESS &&
                   function_pointer != nullptr,
               "fake legacy resolver failed");
    function_pointer = nullptr;
    all_passed &=
        expect(dispatch.get_proc_address_v2("fake", &function_pointer, 0, 0, &symbol_status) ==
                       CUDA_SUCCESS &&
                   function_pointer != nullptr && symbol_status == CU_GET_PROC_ADDRESS_SUCCESS,
               "fake v2 resolver failed");
    all_passed &= expect(!g_guard_failed, "Driver call guard was not active");

    DriverFunctionTable legacy_only_functions = functions;
    legacy_only_functions.mem_alloc_async_ptsz = nullptr;
    legacy_only_functions.mem_alloc_from_pool_async_ptsz = nullptr;
    legacy_only_functions.mem_free_async_ptsz = nullptr;
    legacy_only_functions.mem_address_reserve = nullptr;
    legacy_only_functions.mem_address_free = nullptr;
    legacy_only_functions.mem_map = nullptr;
    legacy_only_functions.mem_unmap = nullptr;
    legacy_only_functions.mem_set_access = nullptr;
    legacy_only_functions.mem_get_address_range = nullptr;
    legacy_only_functions.mem_get_access = nullptr;
    legacy_only_functions.mem_pool_create = nullptr;
    legacy_only_functions.mem_pool_destroy = nullptr;
    legacy_only_functions.device_get_default_mem_pool = nullptr;
    legacy_only_functions.stream_get_device_ptsz = nullptr;
    legacy_only_functions.stream_get_context_ptsz = nullptr;
    legacy_only_functions.stream_query_ptsz = nullptr;
    legacy_only_functions.stream_synchronize_ptsz = nullptr;
    const DriverDispatch legacy_only_dispatch(legacy_only_functions);
    all_passed &= expect(!legacy_only_dispatch.has_mem_alloc_async_ptsz(),
                         "legacy allocator was reported as a PTDS allocator");
    all_passed &= expect(!legacy_only_dispatch.has_mem_alloc_from_pool_async_ptsz(),
                         "legacy pool allocator was reported as a PTDS allocator");
    all_passed &= expect(!legacy_only_dispatch.has_mem_free_async_ptsz(),
                         "legacy free was reported as a PTDS free");
    all_passed &= expect(!legacy_only_dispatch.has_mem_address_reserve(),
                         "missing VMM address reserve was reported as available");
    all_passed &= expect(!legacy_only_dispatch.has_mem_address_free(),
                         "missing VMM address free was reported as available");
    all_passed &=
        expect(!legacy_only_dispatch.has_mem_map(), "missing VMM map was reported as available");
    all_passed &= expect(!legacy_only_dispatch.has_mem_unmap(),
                         "missing VMM unmap was reported as available");
    all_passed &= expect(!legacy_only_dispatch.has_mem_set_access(),
                         "missing VMM access update was reported as available");
    all_passed &= expect(!legacy_only_dispatch.has_mem_get_address_range(),
                         "missing VMM address query was reported as available");
    all_passed &= expect(!legacy_only_dispatch.has_mem_get_access(),
                         "missing VMM access query was reported as available");
    all_passed &= expect(!legacy_only_dispatch.has_mem_pool_create(),
                         "missing memory-pool creation was reported as available");
    all_passed &= expect(!legacy_only_dispatch.has_mem_pool_destroy(),
                         "missing memory-pool destruction was reported as available");
    all_passed &= expect(!legacy_only_dispatch.has_device_get_default_mem_pool(),
                         "missing device default memory-pool query was reported as available");
    all_passed &= expect(!legacy_only_dispatch.has_stream_identity_ptsz(),
                         "legacy stream identity was reported as PTDS identity");
    all_passed &= expect(!legacy_only_dispatch.has_stream_query_ptsz(),
                         "legacy stream query was reported as PTDS query");
    all_passed &= expect(!legacy_only_dispatch.has_stream_synchronize_ptsz(),
                         "legacy stream synchronization was reported as PTDS synchronization");
    all_passed &= expect(legacy_only_dispatch.mem_alloc_async_ptsz(&device_pointer, 16, nullptr) ==
                             CUDA_ERROR_NOT_SUPPORTED,
                         "missing PTDS allocator did not fail closed");
    all_passed &= expect(legacy_only_dispatch.mem_free_async_ptsz(device_pointer, nullptr) ==
                             CUDA_ERROR_NOT_SUPPORTED,
                         "missing PTDS free did not fail closed");
    all_passed &= expect(legacy_only_dispatch.mem_address_reserve(&virtual_address, 16, 0, 0, 0) ==
                             CUDA_ERROR_NOT_SUPPORTED,
                         "missing VMM address reserve did not fail closed");
    all_passed &= expect(
        legacy_only_dispatch.mem_address_free(virtual_address, 16) == CUDA_ERROR_NOT_SUPPORTED,
        "missing VMM address free did not fail closed");
    all_passed &= expect(
        legacy_only_dispatch.mem_map(virtual_address, 16, 0, 1, 0) == CUDA_ERROR_NOT_SUPPORTED,
        "missing VMM map did not fail closed");
    all_passed &=
        expect(legacy_only_dispatch.mem_unmap(virtual_address, 16) == CUDA_ERROR_NOT_SUPPORTED,
               "missing VMM unmap did not fail closed");
    all_passed &= expect(legacy_only_dispatch.mem_set_access(virtual_address, 16, nullptr, 0) ==
                             CUDA_ERROR_NOT_SUPPORTED,
                         "missing VMM access update did not fail closed");
    all_passed &=
        expect(legacy_only_dispatch.mem_get_address_range(
                   &address_base, &address_size, virtual_address) == CUDA_ERROR_NOT_SUPPORTED,
               "missing VMM address query did not fail closed");
    all_passed &= expect(
        legacy_only_dispatch.mem_pool_create(&pool, &pool_properties) == CUDA_ERROR_NOT_SUPPORTED,
        "missing memory-pool creation did not fail closed");
    all_passed &= expect(legacy_only_dispatch.device_get_default_mem_pool(&queried_pool, 0) ==
                             CUDA_ERROR_NOT_SUPPORTED,
                         "missing device default memory-pool query did not fail closed");
    all_passed &=
        expect(legacy_only_dispatch.stream_query_ptsz(nullptr) == CUDA_ERROR_NOT_SUPPORTED,
               "missing PTDS query did not fail closed");
    all_passed &=
        expect(legacy_only_dispatch.stream_synchronize_ptsz(nullptr) == CUDA_ERROR_NOT_SUPPORTED,
               "missing PTDS synchronization did not fail closed");
    return all_passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
