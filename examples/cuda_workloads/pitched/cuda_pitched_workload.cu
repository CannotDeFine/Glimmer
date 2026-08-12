#include "workload_common.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <vector>

namespace {

constexpr std::size_t kWidthBytes = 1024;
constexpr std::size_t kHeight = 1024;
constexpr unsigned int kThreadsPerBlock = 256;

__global__ void fill_pitched_buffer(unsigned char* buffer, std::size_t pitch,
                                    std::size_t width_bytes, std::size_t height) {
    const std::size_t row = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= height) {
        return;
    }
    unsigned char* row_begin = buffer + row * pitch;
    for (std::size_t column = 0; column < width_bytes; ++column) {
        row_begin[column] = static_cast<unsigned char>((row + column) & 0xFFU);
    }
}

}  // namespace

int main() {
    using glimmer::cuda_workloads::allocation_delta_matches;
    using glimmer::cuda_workloads::check_cuda;
    using glimmer::cuda_workloads::MemoryInfo;
    using glimmer::cuda_workloads::print_summary;
    using glimmer::cuda_workloads::read_memory;

    constexpr std::size_t kLogicalBytes = kWidthBytes * kHeight;
    MemoryInfo before;
    MemoryInfo after_alloc;
    MemoryInfo after_release;
    unsigned char* device_buffer = nullptr;
    std::size_t pitch = 0;
    std::vector<unsigned char> host_buffer(kLogicalBytes, 0);
    bool succeeded = read_memory(&before, "cudaMemGetInfo before");

    if (succeeded) {
        void* allocation = nullptr;
        succeeded = check_cuda(cudaMallocPitch(&allocation, &pitch, kWidthBytes, kHeight),
                               "cudaMallocPitch") &&
                    succeeded;
        device_buffer = static_cast<unsigned char*>(allocation);
    }
    const std::size_t charged_bytes = pitch * kHeight;
    if (succeeded) {
        succeeded =
            read_memory(&after_alloc, "cudaMemGetInfo after pitched allocation") && succeeded;
    }
    if (succeeded) {
        const unsigned int blocks =
            static_cast<unsigned int>((kHeight + kThreadsPerBlock - 1) / kThreadsPerBlock);
        fill_pitched_buffer<<<blocks, kThreadsPerBlock>>>(device_buffer, pitch, kWidthBytes,
                                                          kHeight);
        succeeded = check_cuda(cudaGetLastError(), "pitched kernel launch") && succeeded;
        succeeded =
            check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize pitched") && succeeded;
    }
    if (succeeded) {
        succeeded = check_cuda(cudaMemcpy2D(host_buffer.data(), kWidthBytes, device_buffer, pitch,
                                            kWidthBytes, kHeight, cudaMemcpyDeviceToHost),
                               "cudaMemcpy2D") &&
                    succeeded;
    }

    bool contents_valid = succeeded;
    if (contents_valid) {
        for (std::size_t row = 0; row < kHeight && contents_valid; ++row) {
            for (std::size_t column = 0; column < kWidthBytes; ++column) {
                if (host_buffer[row * kWidthBytes + column] !=
                    static_cast<unsigned char>((row + column) & 0xFFU)) {
                    contents_valid = false;
                    break;
                }
            }
        }
    }
    succeeded = contents_valid && succeeded;
    if (device_buffer != nullptr) {
        const bool released = check_cuda(cudaFree(device_buffer), "cudaFree pitched");
        succeeded = released && succeeded;
        if (released) {
            device_buffer = nullptr;
        }
    }
    if (succeeded) {
        succeeded =
            read_memory(&after_release, "cudaMemGetInfo after pitched release") && succeeded;
    }

    const bool validation = succeeded && contents_valid &&
                            allocation_delta_matches(before, after_alloc, charged_bytes) &&
                            after_release.free_bytes >= after_alloc.free_bytes;
    print_summary("glimmer_cuda_pitched_workload", kLogicalBytes, before, after_alloc,
                  after_release, validation, succeeded && validation);
    return succeeded && validation ? 0 : 1;
}
