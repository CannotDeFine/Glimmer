#include "workload_common.h"

#include <cuda_runtime.h>

#include <cstddef>

namespace {

constexpr std::size_t kBufferBytes = std::size_t{1} * 1024 * 1024;
constexpr unsigned int kThreadsPerBlock = 256;
constexpr unsigned char kExpectedValue = 0xA5;

__global__ void fill_managed_buffer(unsigned char* buffer, std::size_t bytes) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t offset = index; offset < bytes; offset += stride) {
        buffer[offset] = kExpectedValue;
    }
}

}  // namespace

int main() {
    using glimmer::cuda_workloads::allocation_delta_matches;
    using glimmer::cuda_workloads::check_cuda;
    using glimmer::cuda_workloads::MemoryInfo;
    using glimmer::cuda_workloads::print_summary;
    using glimmer::cuda_workloads::read_memory;

    MemoryInfo before;
    MemoryInfo after_alloc;
    MemoryInfo after_release;
    unsigned char* managed_buffer = nullptr;
    bool succeeded = read_memory(&before, "cudaMemGetInfo before");

    if (succeeded) {
        succeeded =
            check_cuda(cudaMallocManaged(&managed_buffer, kBufferBytes), "cudaMallocManaged") &&
            succeeded;
    }
    if (succeeded) {
        succeeded =
            read_memory(&after_alloc, "cudaMemGetInfo after managed allocation") && succeeded;
    }

    const unsigned int blocks =
        static_cast<unsigned int>((kBufferBytes + kThreadsPerBlock - 1) / kThreadsPerBlock);
    if (succeeded) {
        fill_managed_buffer<<<blocks, kThreadsPerBlock>>>(managed_buffer, kBufferBytes);
        succeeded = check_cuda(cudaGetLastError(), "managed kernel launch") && succeeded;
        succeeded =
            check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize managed") && succeeded;
    }

    const bool contents_valid = succeeded && managed_buffer != nullptr &&
                                managed_buffer[0] == kExpectedValue &&
                                managed_buffer[kBufferBytes - 1] == kExpectedValue;
    succeeded = contents_valid && succeeded;
    if (managed_buffer != nullptr) {
        const bool released = check_cuda(cudaFree(managed_buffer), "cudaFree managed");
        succeeded = released && succeeded;
        if (released) {
            managed_buffer = nullptr;
        }
    }
    if (succeeded) {
        succeeded =
            read_memory(&after_release, "cudaMemGetInfo after managed release") && succeeded;
    }

    const bool validation = succeeded && contents_valid &&
                            allocation_delta_matches(before, after_alloc, kBufferBytes) &&
                            after_release.free_bytes >= after_alloc.free_bytes;
    print_summary("glimmer_cuda_managed_workload", kBufferBytes, before, after_alloc, after_release,
                  validation, succeeded && validation);
    return succeeded && validation ? 0 : 1;
}
