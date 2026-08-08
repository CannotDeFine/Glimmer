#include <cuda.h>

#include <cstdlib>
#include <iostream>
#include <string_view>

namespace {

constexpr std::size_t kQuotaBytes = std::size_t{8} * 1024 * 1024;
constexpr std::size_t kSuccessfulRequestBytes = std::size_t{1} * 1024 * 1024;
constexpr std::size_t kRequestedBytes = std::size_t{16} * 1024 * 1024;

bool check(CUresult result, std::string_view operation) {
    if (result == CUDA_SUCCESS) {
        return true;
    }

    std::cerr << operation << " failed with CUresult " << static_cast<int>(result) << '\n';
    return false;
}

}  // namespace

int main() {
    if (!check(cuInit(0), "cuInit")) {
        return EXIT_FAILURE;
    }

    CUdevice device{};
    if (!check(cuDeviceGet(&device, 0), "cuDeviceGet")) {
        return EXIT_FAILURE;
    }

    CUcontext context{};
    if (!check(cuCtxCreate(&context, nullptr, 0, device), "cuCtxCreate")) {
        return EXIT_FAILURE;
    }

    std::size_t free_bytes{};
    std::size_t total_bytes{};
    const bool has_memory_info =
        check(cuMemGetInfo_v2(&free_bytes, &total_bytes), "cuMemGetInfo_v2");
    const bool has_expected_total = total_bytes == kQuotaBytes;
    const bool has_expected_initial_free = free_bytes == kQuotaBytes;

    CUdeviceptr successful_pointer{};
    const bool allocated_within_quota =
        check(cuMemAlloc_v2(&successful_pointer, kSuccessfulRequestBytes), "cuMemAlloc_v2");
    const bool has_reduced_free =
        check(cuMemGetInfo_v2(&free_bytes, &total_bytes), "cuMemGetInfo_v2") &&
        free_bytes == kQuotaBytes - kSuccessfulRequestBytes;
    const bool released_within_quota =
        !allocated_within_quota || check(cuMemFree_v2(successful_pointer), "cuMemFree_v2");
    const bool has_restored_free =
        check(cuMemGetInfo_v2(&free_bytes, &total_bytes), "cuMemGetInfo_v2") &&
        free_bytes == kQuotaBytes;

    const CUresult invalid_allocation_result = cuMemAlloc_v2(nullptr, kSuccessfulRequestBytes);
    const bool propagated_driver_error = invalid_allocation_result != CUDA_SUCCESS;
    const CUresult unknown_free_result = cuMemFree_v2(static_cast<CUdeviceptr>(1));
    const bool unknown_free_preserved_usage =
        unknown_free_result != CUDA_SUCCESS &&
        check(cuMemGetInfo_v2(&free_bytes, &total_bytes), "cuMemGetInfo_v2") &&
        free_bytes == kQuotaBytes;

    CUdeviceptr device_pointer{};
    const CUresult allocation_result = cuMemAlloc_v2(&device_pointer, kRequestedBytes);
    const bool was_rejected = allocation_result == CUDA_ERROR_OUT_OF_MEMORY;
    const bool released_unexpected_allocation =
        allocation_result != CUDA_SUCCESS || check(cuMemFree_v2(device_pointer), "cuMemFree_v2");
    const bool destroyed_context = check(cuCtxDestroy(context), "cuCtxDestroy");

    if (!has_memory_info || !has_expected_total || !has_expected_initial_free ||
        !allocated_within_quota || !has_reduced_free || !released_within_quota ||
        !has_restored_free || !propagated_driver_error || !unknown_free_preserved_usage ||
        !was_rejected || !released_unexpected_allocation || !destroyed_context) {
        std::cerr << "CUDA interceptor GPU test failed\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
