#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <iostream>
#include <string_view>

namespace glimmer::cuda_workloads {

struct MemoryInfo {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
};

inline bool check_cuda(cudaError_t status, std::string_view operation) {
    if (status == cudaSuccess) {
        return true;
    }

    std::cerr << operation << " failed with cudaError " << static_cast<int>(status) << ": "
              << cudaGetErrorString(status) << '\n';
    return false;
}

inline bool read_memory(MemoryInfo* memory, std::string_view operation) {
    if (memory == nullptr) {
        return false;
    }
    return check_cuda(cudaMemGetInfo(&memory->free_bytes, &memory->total_bytes), operation);
}

inline bool allocation_delta_matches(const MemoryInfo& before, const MemoryInfo& after,
                                     std::size_t allocated_bytes) {
    return before.free_bytes >= after.free_bytes &&
           before.free_bytes - after.free_bytes >= allocated_bytes;
}

inline void print_summary(std::string_view workload, std::size_t requested_bytes,
                          const MemoryInfo& before, const MemoryInfo& after_alloc,
                          const MemoryInfo& after_release, bool validation, bool succeeded) {
    std::cout << "workload=" << workload << " status=" << (succeeded ? "ok" : "failed")
              << " requested_bytes=" << requested_bytes
              << " visible_total_bytes=" << before.total_bytes
              << " visible_free_before_bytes=" << before.free_bytes
              << " visible_free_after_alloc_bytes=" << after_alloc.free_bytes
              << " visible_free_after_release_bytes=" << after_release.free_bytes
              << " validation=" << (validation ? 1 : 0) << '\n';
}

}  // namespace glimmer::cuda_workloads
