#include "workload_common.h"

#include <cuda_runtime.h>

#include <cstddef>

namespace {

constexpr std::size_t kBufferBytes = std::size_t{1} * 1024 * 1024;
constexpr unsigned int kThreadsPerBlock = 256;

__global__ void touch_buffer(unsigned char* buffer, std::size_t bytes, unsigned char value) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t offset = index; offset < bytes; offset += stride) {
        buffer[offset] = value;
    }
}

}  // namespace

int main() {
    using glimmer::cuda_workloads::allocation_delta_matches;
    using glimmer::cuda_workloads::check_cuda;
    using glimmer::cuda_workloads::MemoryInfo;
    using glimmer::cuda_workloads::print_summary;
    using glimmer::cuda_workloads::read_memory;

    constexpr std::size_t kRequestedBytes = 2 * kBufferBytes;
    MemoryInfo before;
    MemoryInfo after_alloc;
    MemoryInfo after_release;
    cudaStream_t stream = nullptr;
    cudaMemPool_t pool = nullptr;
    unsigned char* async_buffer = nullptr;
    unsigned char* pool_buffer = nullptr;
    bool async_allocated = false;
    bool pool_allocated = false;
    bool succeeded = read_memory(&before, "cudaMemGetInfo before");

    succeeded = check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate") && succeeded;
    succeeded = check_cuda(cudaDeviceGetDefaultMemPool(&pool, 0), "cudaDeviceGetDefaultMemPool") &&
                succeeded;
    if (succeeded) {
        succeeded =
            check_cuda(cudaMallocAsync(&async_buffer, kBufferBytes, stream), "cudaMallocAsync") &&
            succeeded;
        async_allocated = succeeded;
    }
    if (succeeded) {
        succeeded = check_cuda(cudaMallocFromPoolAsync(&pool_buffer, kBufferBytes, pool, stream),
                               "cudaMallocFromPoolAsync") &&
                    succeeded;
        pool_allocated = succeeded;
    }
    if (succeeded) {
        succeeded = check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize allocate") &&
                    succeeded;
    }
    if (succeeded) {
        succeeded = read_memory(&after_alloc, "cudaMemGetInfo after async allocation") && succeeded;
    }

    const unsigned int blocks =
        static_cast<unsigned int>((kBufferBytes + kThreadsPerBlock - 1) / kThreadsPerBlock);
    if (succeeded) {
        touch_buffer<<<blocks, kThreadsPerBlock, 0, stream>>>(async_buffer, kBufferBytes, 0x2A);
        succeeded = check_cuda(cudaGetLastError(), "async kernel launch") && succeeded;
    }
    if (succeeded) {
        touch_buffer<<<blocks, kThreadsPerBlock, 0, stream>>>(pool_buffer, kBufferBytes, 0x55);
        succeeded = check_cuda(cudaGetLastError(), "pool kernel launch") && succeeded;
    }
    if (succeeded) {
        succeeded =
            check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize kernel") && succeeded;
    }
    if (succeeded) {
        const bool async_released =
            check_cuda(cudaFreeAsync(async_buffer, stream), "cudaFreeAsync");
        succeeded = async_released && succeeded;
        if (async_released) {
            async_allocated = false;
            async_buffer = nullptr;
        }
        const bool pool_released =
            check_cuda(cudaFreeAsync(pool_buffer, stream), "cudaFreeAsync pool");
        succeeded = pool_released && succeeded;
        if (pool_released) {
            pool_allocated = false;
            pool_buffer = nullptr;
        }
    }
    if (stream != nullptr) {
        succeeded =
            check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize release") && succeeded;
    }
    if (succeeded) {
        succeeded = read_memory(&after_release, "cudaMemGetInfo after async release") && succeeded;
    }

    if (async_allocated && async_buffer != nullptr) {
        succeeded =
            check_cuda(cudaFreeAsync(async_buffer, stream), "cleanup cudaFreeAsync") && succeeded;
    }
    if (pool_allocated && pool_buffer != nullptr) {
        succeeded = check_cuda(cudaFreeAsync(pool_buffer, stream), "cleanup cudaFreeAsync pool") &&
                    succeeded;
    }
    if (stream != nullptr) {
        succeeded =
            check_cuda(cudaStreamSynchronize(stream), "cleanup cudaStreamSynchronize") && succeeded;
        succeeded = check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy") && succeeded;
    }

    const bool validation = succeeded &&
                            allocation_delta_matches(before, after_alloc, kRequestedBytes) &&
                            after_release.free_bytes >= after_alloc.free_bytes;
    print_summary("glimmer_cuda_async_workload", kRequestedBytes, before, after_alloc,
                  after_release, validation, succeeded && validation);
    return succeeded && validation ? 0 : 1;
}
