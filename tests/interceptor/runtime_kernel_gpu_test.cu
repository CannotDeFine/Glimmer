#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string_view>

namespace {

constexpr std::uint32_t kExpectedValue = 42;
constexpr std::size_t kQuotaBytes = std::size_t{8} * 1024 * 1024;

__global__ void write_value(std::uint32_t* output) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *output = kExpectedValue;
    }
}

bool check(cudaError_t result, std::string_view operation) {
    if (result == cudaSuccess) {
        return true;
    }

    std::cerr << operation << " failed with cudaError " << static_cast<int>(result) << ": "
              << cudaGetErrorString(result) << '\n';
    return false;
}

}  // namespace

int main() {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    bool workload_succeeded =
        check(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo initial") &&
        total_bytes == kQuotaBytes && free_bytes == kQuotaBytes;

    std::uint32_t* device_output = nullptr;
    workload_succeeded = check(cudaMalloc(&device_output, sizeof(kExpectedValue)), "cudaMalloc") &&
                         workload_succeeded;

    if (workload_succeeded) {
        workload_succeeded =
            check(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo allocated") &&
            total_bytes == kQuotaBytes && free_bytes == kQuotaBytes - sizeof(kExpectedValue) &&
            workload_succeeded;
    }

    if (workload_succeeded) {
        write_value<<<1, 1>>>(device_output);
        workload_succeeded = check(cudaGetLastError(), "kernel launch") && workload_succeeded;
        workload_succeeded =
            check(cudaDeviceSynchronize(), "cudaDeviceSynchronize") && workload_succeeded;
    }

    std::uint32_t host_output = 0;
    if (workload_succeeded) {
        const cudaError_t copy_result =
            cudaMemcpy(&host_output, device_output, sizeof(host_output), cudaMemcpyDeviceToHost);
        workload_succeeded =
            check(copy_result, "cudaMemcpy") && host_output == kExpectedValue && workload_succeeded;
    }

    const bool released = device_output == nullptr || check(cudaFree(device_output), "cudaFree");
    const bool restored =
        released && check(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo restored") &&
        total_bytes == kQuotaBytes && free_bytes == kQuotaBytes;
    return workload_succeeded && released && restored ? EXIT_SUCCESS : EXIT_FAILURE;
}
