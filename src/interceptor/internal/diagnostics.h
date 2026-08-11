#pragma once

#include <cstddef>
#include <cstdint>

namespace glimmer::interceptor {

struct KernelLaunchObservation {
    const char* api_name = nullptr;
    unsigned int grid_dim_x = 0;
    unsigned int grid_dim_y = 0;
    unsigned int grid_dim_z = 0;
    unsigned int block_dim_x = 0;
    unsigned int block_dim_y = 0;
    unsigned int block_dim_z = 0;
    std::size_t shared_memory_bytes = 0;
    const void* stream = nullptr;
};

struct MemoryInfoObservation {
    const char* api_name = nullptr;
    std::int32_t device = 0;
    std::uint64_t total_bytes = 0;
    std::uint64_t free_bytes = 0;
};

void report_diagnostic(const char* message) noexcept;
void report_kernel_launch_diagnostic(const KernelLaunchObservation& observation,
                                     std::uint64_t launch_count) noexcept;
void report_memory_info_diagnostic(const MemoryInfoObservation& observation) noexcept;

}  // namespace glimmer::interceptor
