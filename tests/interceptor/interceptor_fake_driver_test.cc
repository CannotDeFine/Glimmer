#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "glimmer/control/shared_memory_quota.h"

#include <cerrno>
#include <dlfcn.h>

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <string_view>
#include <sys/wait.h>
#include <unistd.h>

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
using RuntimeMallocAsyncFunction = cudaError_t (*)(void** device_pointer, std::size_t memory_bytes,
                                                   cudaStream_t stream);
using RuntimeFreeFunction = cudaError_t (*)(void* device_pointer);
using RuntimeFreeAsyncFunction = cudaError_t (*)(void* device_pointer, cudaStream_t stream);
using RuntimeDeviceSynchronizeFunction = cudaError_t (*)();
using RuntimeStreamSynchronizeFunction = cudaError_t (*)(cudaStream_t stream);
using RuntimeStreamQueryFunction = cudaError_t (*)(cudaStream_t stream);
using RuntimeStreamDestroyFunction = cudaError_t (*)(cudaStream_t stream);
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

}  // namespace

int main() {
    if (std::getenv("GLIMMER_FAKE_DUPLICATE_POINTER") != nullptr) {
        return run_duplicate_record_failure_test();
    }
    if (std::getenv("GLIMMER_FAKE_NULL_SUCCESS_POINTER") != nullptr) {
        return run_async_null_pointer_test(std::getenv("GLIMMER_NULL_SUCCESS_RUNTIME") != nullptr);
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
    all_passed &= expect(
        init != nullptr && allocate != nullptr && release != nullptr && async_allocate != nullptr &&
            async_allocate_ptsz != nullptr && pool_async_allocate != nullptr &&
            pool_async_allocate_ptsz != nullptr && async_release != nullptr &&
            async_release_ptsz != nullptr && stream_get_device != nullptr &&
            stream_get_device_ptsz != nullptr && stream_get_context != nullptr &&
            stream_get_context_ptsz != nullptr && stream_query != nullptr &&
            stream_query_ptsz != nullptr && stream_synchronize != nullptr &&
            stream_synchronize_ptsz != nullptr && stream_destroy != nullptr &&
            context_synchronize != nullptr && get_info != nullptr && get_total != nullptr &&
            get_current != nullptr && destroy_context != nullptr,
        "interceptor symbols were not exported");
    if (!all_passed) {
        return EXIT_FAILURE;
    }

    all_passed &= expect(init(0) == CUDA_SUCCESS, "fake cuInit failed");

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
    const RuntimeMallocAsyncFunction runtime_async_allocate_ptsz =
        resolve_default<RuntimeMallocAsyncFunction>("cudaMallocAsync_ptsz");
    const RuntimeFreeFunction runtime_release = resolve_default<RuntimeFreeFunction>("cudaFree");
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
    void* runtime_pointer = nullptr;
    all_passed &=
        expect(runtime_allocate != nullptr && runtime_async_allocate != nullptr &&
                   runtime_async_allocate_ptsz != nullptr && runtime_release != nullptr &&
                   runtime_async_release != nullptr && runtime_async_release_ptsz != nullptr &&
                   runtime_device_synchronize != nullptr && runtime_stream_synchronize != nullptr &&
                   runtime_stream_synchronize_ptsz != nullptr && runtime_stream_query != nullptr &&
                   runtime_stream_query_ptsz != nullptr && runtime_stream_destroy != nullptr &&
                   runtime_get_info != nullptr,
               "runtime interceptor symbols were not exported");
    if (runtime_handle != nullptr) {
        all_passed &=
            expect(dlsym(runtime_handle, "cudaMalloc") == reinterpret_cast<void*>(runtime_allocate),
                   "explicit CUDA Runtime handle did not return the interceptor wrapper");
        all_passed &=
            expect(dlsym(runtime_handle, "cudaMallocAsync_ptsz") ==
                       reinterpret_cast<void*>(runtime_async_allocate_ptsz),
                   "explicit CUDA Runtime handle did not return the PTDS interceptor wrapper");
    }
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

    if (shared_mode && !shared_tenant_id.empty()) {
        all_passed &= expect(glimmer::control::SharedMemoryQuota::remove_region(shared_tenant_id),
                             "shared test region could not be removed");
    }
    return all_passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
