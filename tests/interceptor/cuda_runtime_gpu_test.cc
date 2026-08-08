#include <cuda_runtime_api.h>

#include <cstdlib>
#include <iostream>
#include <string_view>

namespace {

constexpr std::size_t kQuotaBytes = std::size_t{8} * 1024 * 1024;
constexpr std::size_t kAllocationBytes = std::size_t{1} * 1024 * 1024;
constexpr std::size_t kRejectedAllocationBytes = kQuotaBytes;

bool check(cudaError_t result, std::string_view operation) {
    if (result == cudaSuccess) {
        return true;
    }

    std::cerr << operation << " failed with cudaError " << static_cast<int>(result) << '\n';
    return false;
}

}  // namespace

int main() {
    std::size_t free_bytes{};
    std::size_t total_bytes{};
    const bool has_initial_info =
        check(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo initial") &&
        total_bytes == kQuotaBytes && free_bytes == kQuotaBytes;

    void* device_pointer = nullptr;
    const bool allocated = check(cudaMalloc(&device_pointer, kAllocationBytes), "cudaMalloc");
    const bool has_reduced_free =
        allocated && check(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo allocated") &&
        free_bytes == kQuotaBytes - kAllocationBytes;

    void* rejected_pointer = nullptr;
    const cudaError_t rejected_result = cudaMalloc(&rejected_pointer, kRejectedAllocationBytes);
    const bool rejected =
        rejected_result == cudaErrorMemoryAllocation && rejected_pointer == nullptr;
    const bool has_unchanged_free_after_rejection =
        check(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo rejected") &&
        free_bytes == kQuotaBytes - kAllocationBytes;

    const bool released = !allocated || check(cudaFree(device_pointer), "cudaFree");
    const bool has_restored_free =
        check(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo restored") &&
        free_bytes == kQuotaBytes;

    if (!has_initial_info || !allocated || !has_reduced_free || !rejected ||
        !has_unchanged_free_after_rejection || !released || !has_restored_free) {
        std::cerr << "CUDA Runtime interceptor GPU test failed\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
