#include "internal/driver_dispatch.h"

#include <cstdlib>
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
        .mem_alloc = &fake_mem_alloc,
        .mem_alloc_managed = &fake_mem_alloc_managed,
        .mem_alloc_pitch = &fake_mem_alloc_pitch,
        .mem_alloc_async = &fake_mem_alloc_async,
        .mem_alloc_async_ptsz = &fake_mem_alloc_async_ptsz,
        .mem_alloc_from_pool_async = &fake_mem_alloc_from_pool_async,
        .mem_alloc_from_pool_async_ptsz = &fake_mem_alloc_from_pool_async_ptsz,
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
    all_passed &= expect(dispatch.has_mem_alloc_managed(), "managed allocator was not available");
    all_passed &= expect(dispatch.has_mem_alloc_pitch(), "pitched allocator was not available");
    all_passed &= expect(dispatch.has_mem_alloc_async(), "async allocator was not available");
    all_passed &=
        expect(dispatch.has_mem_alloc_async_ptsz(), "PTDS async allocator was not available");
    all_passed &=
        expect(dispatch.has_mem_alloc_from_pool_async(), "pool async allocator was not available");
    all_passed &= expect(dispatch.has_mem_alloc_from_pool_async_ptsz(),
                         "PTDS pool async allocator was not available");
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
    void* function_pointer = nullptr;
    CUdriverProcAddressQueryResult symbol_status{};
    CUcontext context{};
    CUdevice device{};
    all_passed &= expect(dispatch.init(0) == CUDA_SUCCESS, "fake cuInit failed");
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
    all_passed &=
        expect(legacy_only_dispatch.stream_query_ptsz(nullptr) == CUDA_ERROR_NOT_SUPPORTED,
               "missing PTDS query did not fail closed");
    all_passed &=
        expect(legacy_only_dispatch.stream_synchronize_ptsz(nullptr) == CUDA_ERROR_NOT_SUPPORTED,
               "missing PTDS synchronization did not fail closed");
    return all_passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
