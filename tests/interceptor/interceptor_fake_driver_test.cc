#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <dlfcn.h>

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <string_view>

namespace {

constexpr std::size_t kQuotaBytes = 4096;
constexpr std::size_t kDirectAllocationBytes = 3072;
constexpr std::size_t kRuntimeAllocationBytes = 1024;
constexpr std::size_t kAsyncAllocationBytes = 1024;
constexpr std::size_t kRejectedAllocationBytes = 2048;
constexpr std::size_t kForcedAllocationFailureBytes = 1536;

using InitFunction = CUresult (*)(unsigned int flags);
using AllocFunction = CUresult (*)(CUdeviceptr* device_pointer, std::size_t memory_bytes);
using AsyncAllocFunction = CUresult (*)(CUdeviceptr* device_pointer, std::size_t memory_bytes,
                                        CUstream stream);
using PoolAsyncAllocFunction = CUresult (*)(CUdeviceptr* device_pointer, std::size_t memory_bytes,
                                            CUmemoryPool pool, CUstream stream);
using FreeFunction = CUresult (*)(CUdeviceptr device_pointer);
using AsyncFreeFunction = CUresult (*)(CUdeviceptr device_pointer, CUstream stream);
using StreamGetDeviceFunction = CUresult (*)(CUstream stream, CUdevice* device);
using StreamGetContextFunction = CUresult (*)(CUstream stream, CUcontext* context);
using StreamQueryFunction = CUresult (*)(CUstream stream);
using StreamSynchronizeFunction = CUresult (*)(CUstream stream);
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
using RuntimeMallocAsyncFunction = cudaError_t (*)(void** device_pointer, std::size_t memory_bytes,
                                                   cudaStream_t stream);
using RuntimeFreeFunction = cudaError_t (*)(void* device_pointer);
using RuntimeFreeAsyncFunction = cudaError_t (*)(void* device_pointer, cudaStream_t stream);
using RuntimeDeviceSynchronizeFunction = cudaError_t (*)();
using RuntimeMemGetInfoFunction = cudaError_t (*)(std::size_t* free_bytes,
                                                  std::size_t* total_bytes);

template <typename Function>
Function resolve_default(const char* name) {
    return reinterpret_cast<Function>(dlsym(RTLD_DEFAULT, name));
}

bool expect(bool condition, std::string_view message) {
    if (condition) {
        return true;
    }
    std::cerr << message << '\n';
    return false;
}

}  // namespace

int main() {
    bool all_passed = true;
    const InitFunction init = resolve_default<InitFunction>("cuInit");
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
    const ContextSynchronizeFunction context_synchronize =
        resolve_default<ContextSynchronizeFunction>("cuCtxSynchronize");
    const MemGetInfoFunction get_info = resolve_default<MemGetInfoFunction>("cuMemGetInfo_v2");
    const DeviceTotalMemFunction get_total =
        resolve_default<DeviceTotalMemFunction>("cuDeviceTotalMem_v2");
    const ContextGetCurrentFunction get_current =
        resolve_default<ContextGetCurrentFunction>("cuCtxGetCurrent");
    const ContextDestroyFunction destroy_context =
        resolve_default<ContextDestroyFunction>("cuCtxDestroy_v2");
    all_passed &=
        expect(init != nullptr && allocate != nullptr && release != nullptr &&
                   async_allocate != nullptr && async_allocate_ptsz != nullptr &&
                   pool_async_allocate != nullptr && pool_async_allocate_ptsz != nullptr &&
                   async_release != nullptr && async_release_ptsz != nullptr &&
                   stream_get_device != nullptr && stream_get_device_ptsz != nullptr &&
                   stream_get_context != nullptr && stream_get_context_ptsz != nullptr &&
                   stream_query != nullptr && stream_query_ptsz != nullptr &&
                   stream_synchronize != nullptr && stream_synchronize_ptsz != nullptr &&
                   context_synchronize != nullptr && get_info != nullptr && get_total != nullptr &&
                   get_current != nullptr && destroy_context != nullptr,
               "interceptor symbols were not exported");
    if (!all_passed) {
        return EXIT_FAILURE;
    }

    all_passed &= expect(init(0) == CUDA_SUCCESS, "fake cuInit failed");

    std::size_t total_bytes = 0;
    all_passed &= expect(get_total(&total_bytes, 0) == CUDA_SUCCESS && total_bytes == kQuotaBytes,
                         "device total memory was not virtualized without a current-context query");

    std::size_t free_bytes = 0;
    all_passed &= expect(get_info(&free_bytes, &total_bytes) == CUDA_SUCCESS &&
                             free_bytes == kQuotaBytes && total_bytes == kQuotaBytes,
                         "initial memory info was incorrect");

    CUdeviceptr direct_pointer = 0;
    all_passed &= expect(allocate(&direct_pointer, kDirectAllocationBytes) == CUDA_SUCCESS,
                         "direct allocation was rejected");
    all_passed &= expect(get_info(&free_bytes, &total_bytes) == CUDA_SUCCESS &&
                             free_bytes == kQuotaBytes - kDirectAllocationBytes,
                         "direct allocation was not accounted once");

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

    void* cuda_handle = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    all_passed &= expect(cuda_handle != nullptr, "fake CUDA driver could not be loaded");
    if (cuda_handle != nullptr) {
        all_passed &=
            expect(dlsym(cuda_handle, "cuMemAlloc_v2") == reinterpret_cast<void*>(allocate),
                   "explicit CUDA handle did not return the interceptor wrapper");
        dlclose(cuda_handle);
    }

    void* runtime_handle = dlopen("libcudart.so", RTLD_NOW | RTLD_GLOBAL);
    all_passed &= expect(runtime_handle != nullptr, "fake CUDA runtime could not be loaded");
    const RuntimeMallocFunction runtime_allocate =
        resolve_default<RuntimeMallocFunction>("cudaMalloc");
    const RuntimeMallocAsyncFunction runtime_async_allocate =
        resolve_default<RuntimeMallocAsyncFunction>("cudaMallocAsync");
    const RuntimeFreeFunction runtime_release = resolve_default<RuntimeFreeFunction>("cudaFree");
    const RuntimeFreeAsyncFunction runtime_async_release =
        resolve_default<RuntimeFreeAsyncFunction>("cudaFreeAsync");
    const RuntimeDeviceSynchronizeFunction runtime_device_synchronize =
        resolve_default<RuntimeDeviceSynchronizeFunction>("cudaDeviceSynchronize");
    const RuntimeMemGetInfoFunction runtime_get_info =
        resolve_default<RuntimeMemGetInfoFunction>("cudaMemGetInfo");
    void* runtime_pointer = nullptr;
    all_passed &= expect(runtime_allocate != nullptr && runtime_async_allocate != nullptr &&
                             runtime_release != nullptr && runtime_async_release != nullptr &&
                             runtime_device_synchronize != nullptr && runtime_get_info != nullptr,
                         "runtime interceptor symbols were not exported");
    all_passed &= expect(runtime_allocate(&runtime_pointer, kRuntimeAllocationBytes) == cudaSuccess,
                         "runtime allocation was rejected");
    all_passed &= expect(runtime_get_info(&free_bytes, &total_bytes) == cudaSuccess &&
                             free_bytes == kQuotaBytes - kRuntimeAllocationBytes,
                         "runtime allocation was double-accounted or not accounted");
    all_passed &=
        expect(runtime_release(runtime_pointer) == cudaSuccess, "runtime allocation was not freed");
    all_passed &= expect(
        runtime_get_info(&free_bytes, &total_bytes) == cudaSuccess && free_bytes == kQuotaBytes,
        "runtime release did not restore quota");

    void* runtime_async_pointer = nullptr;
    all_passed &= expect(runtime_async_allocate(&runtime_async_pointer, kAsyncAllocationBytes,
                                                nullptr) == cudaSuccess,
                         "Runtime async allocation was rejected");
    all_passed &= expect(runtime_async_release(runtime_async_pointer, nullptr) == cudaSuccess,
                         "Runtime async release was rejected");
    all_passed &= expect(runtime_device_synchronize() == cudaSuccess,
                         "Runtime device synchronization did not complete the release");
    all_passed &= expect(
        runtime_get_info(&free_bytes, &total_bytes) == cudaSuccess && free_bytes == kQuotaBytes,
        "Runtime async release did not restore quota");

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

    CUcontext context = nullptr;
    all_passed &= expect(get_current(&context) == CUDA_SUCCESS && context != nullptr,
                         "current context was not available");
    CUdeviceptr context_pointer = 0;
    all_passed &= expect(allocate(&context_pointer, kRuntimeAllocationBytes) == CUDA_SUCCESS,
                         "context cleanup allocation was rejected");
    all_passed &= expect(destroy_context(context) == CUDA_SUCCESS, "context destruction failed");
    all_passed &=
        expect(get_info(&free_bytes, &total_bytes) == CUDA_SUCCESS && free_bytes == kQuotaBytes,
               "context destruction did not release accounted bytes");
    all_passed &= expect(get_total(&total_bytes, 0) == CUDA_SUCCESS && total_bytes == kQuotaBytes,
                         "device total memory incorrectly depended on a current context");

    const LegacyGetProcAddressFunction legacy_get_proc =
        resolve_default<LegacyGetProcAddressFunction>("cuGetProcAddress");
    const GetProcAddressV2Function get_proc_v2 =
        resolve_default<GetProcAddressV2Function>("cuGetProcAddress_v2");
    void* queried_symbol = nullptr;
    all_passed &= expect(
        legacy_get_proc != nullptr &&
            legacy_get_proc("cuMemAlloc_v2", &queried_symbol, CUDA_VERSION, 0) == CUDA_SUCCESS &&
            queried_symbol == reinterpret_cast<void*>(allocate),
        "legacy cuGetProcAddress did not return the interceptor wrapper");
    queried_symbol = nullptr;
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

    if (runtime_handle != nullptr) {
        dlclose(runtime_handle);
    }

    return all_passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
