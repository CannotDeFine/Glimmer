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
constexpr std::size_t kRejectedAllocationBytes = 2048;
constexpr std::size_t kForcedAllocationFailureBytes = 1536;

using InitFunction = CUresult (*)(unsigned int flags);
using AllocFunction = CUresult (*)(CUdeviceptr* device_pointer, std::size_t memory_bytes);
using FreeFunction = CUresult (*)(CUdeviceptr device_pointer);
using MemGetInfoFunction = CUresult (*)(std::size_t* free_bytes, std::size_t* total_bytes);
using DeviceTotalMemFunction = CUresult (*)(std::size_t* total_bytes, CUdevice device);
using ContextGetCurrentFunction = CUresult (*)(CUcontext* context);
using ContextDestroyFunction = CUresult (*)(CUcontext context);
using LegacyGetProcAddressFunction = CUresult (*)(const char* symbol, void** function_pointer,
                                                  int cuda_version, cuuint64_t flags);
using RuntimeMallocFunction = cudaError_t (*)(void** device_pointer, std::size_t memory_bytes);
using RuntimeFreeFunction = cudaError_t (*)(void* device_pointer);
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
    const MemGetInfoFunction get_info = resolve_default<MemGetInfoFunction>("cuMemGetInfo_v2");
    const DeviceTotalMemFunction get_total =
        resolve_default<DeviceTotalMemFunction>("cuDeviceTotalMem_v2");
    const ContextGetCurrentFunction get_current =
        resolve_default<ContextGetCurrentFunction>("cuCtxGetCurrent");
    const ContextDestroyFunction destroy_context =
        resolve_default<ContextDestroyFunction>("cuCtxDestroy_v2");
    all_passed &= expect(init != nullptr && allocate != nullptr && release != nullptr &&
                             get_info != nullptr && get_total != nullptr &&
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
    const RuntimeFreeFunction runtime_release = resolve_default<RuntimeFreeFunction>("cudaFree");
    const RuntimeMemGetInfoFunction runtime_get_info =
        resolve_default<RuntimeMemGetInfoFunction>("cudaMemGetInfo");
    void* runtime_pointer = nullptr;
    all_passed &= expect(
        runtime_allocate != nullptr && runtime_release != nullptr && runtime_get_info != nullptr,
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
    void* queried_symbol = nullptr;
    all_passed &= expect(
        legacy_get_proc != nullptr &&
            legacy_get_proc("cuMemAlloc_v2", &queried_symbol, CUDA_VERSION, 0) == CUDA_SUCCESS &&
            queried_symbol == reinterpret_cast<void*>(allocate),
        "legacy cuGetProcAddress did not return the interceptor wrapper");

    if (runtime_handle != nullptr) {
        dlclose(runtime_handle);
    }

    return all_passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
