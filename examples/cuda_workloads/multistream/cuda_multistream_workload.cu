#include "workload_common.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>

namespace {

constexpr std::size_t kBufferBytes = std::size_t{512} * 1024;
constexpr unsigned int kThreadsPerBlock = 256;
constexpr std::array<unsigned char, 2> kExpectedValues{0x3C, 0xC3};

__global__ void fill_stream_buffer(unsigned char* buffer, std::size_t bytes, unsigned char value) {
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

    constexpr std::size_t kRequestedBytes = kBufferBytes * kExpectedValues.size();
    MemoryInfo before;
    MemoryInfo after_alloc;
    MemoryInfo after_release;
    std::array<cudaStream_t, 2> streams{};
    std::array<cudaEvent_t, 2> events{};
    std::array<unsigned char*, 2> device_buffers{};
    std::array<std::array<unsigned char, 1>, 2> host_checks{};
    bool succeeded = read_memory(&before, "cudaMemGetInfo before");

    for (std::size_t index = 0; index < streams.size() && succeeded; ++index) {
        succeeded = check_cuda(cudaStreamCreate(&streams[index]), "cudaStreamCreate") && succeeded;
        succeeded = check_cuda(cudaEventCreate(&events[index]), "cudaEventCreate") && succeeded;
        succeeded =
            check_cuda(cudaMalloc(&device_buffers[index], kBufferBytes), "cudaMalloc") && succeeded;
    }
    if (succeeded) {
        succeeded =
            read_memory(&after_alloc, "cudaMemGetInfo after multi-stream allocation") && succeeded;
    }

    const unsigned int blocks =
        static_cast<unsigned int>((kBufferBytes + kThreadsPerBlock - 1) / kThreadsPerBlock);
    for (std::size_t index = 0; index < streams.size() && succeeded; ++index) {
        fill_stream_buffer<<<blocks, kThreadsPerBlock, 0, streams[index]>>>(
            device_buffers[index], kBufferBytes, kExpectedValues[index]);
        succeeded = check_cuda(cudaGetLastError(), "multi-stream kernel launch") && succeeded;
        succeeded = check_cuda(cudaEventRecord(events[index], streams[index]), "cudaEventRecord") &&
                    succeeded;
        succeeded = check_cuda(cudaMemcpyAsync(host_checks[index].data(), device_buffers[index],
                                               host_checks[index].size(), cudaMemcpyDeviceToHost,
                                               streams[index]),
                               "cudaMemcpyAsync") &&
                    succeeded;
    }
    for (cudaStream_t stream : streams) {
        succeeded = check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize") && succeeded;
    }
    bool contents_valid = succeeded;
    for (std::size_t index = 0; index < host_checks.size(); ++index) {
        contents_valid = contents_valid && host_checks[index][0] == kExpectedValues[index];
    }
    succeeded = contents_valid && succeeded;

    for (unsigned char*& device_buffer : device_buffers) {
        if (device_buffer != nullptr) {
            const bool released = check_cuda(cudaFree(device_buffer), "cudaFree multi-stream");
            succeeded = released && succeeded;
            if (released) {
                device_buffer = nullptr;
            }
        }
    }
    for (cudaEvent_t& event : events) {
        if (event != nullptr) {
            const bool destroyed = check_cuda(cudaEventDestroy(event), "cudaEventDestroy");
            succeeded = destroyed && succeeded;
            if (destroyed) {
                event = nullptr;
            }
        }
    }
    for (cudaStream_t& stream : streams) {
        if (stream != nullptr) {
            const bool destroyed = check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
            succeeded = destroyed && succeeded;
            if (destroyed) {
                stream = nullptr;
            }
        }
    }
    if (succeeded) {
        succeeded =
            read_memory(&after_release, "cudaMemGetInfo after multi-stream release") && succeeded;
    }

    const bool validation = succeeded && contents_valid &&
                            allocation_delta_matches(before, after_alloc, kRequestedBytes) &&
                            after_release.free_bytes >= after_alloc.free_bytes;
    print_summary("glimmer_cuda_multistream_workload", kRequestedBytes, before, after_alloc,
                  after_release, validation, succeeded && validation);
    return succeeded && validation ? 0 : 1;
}
