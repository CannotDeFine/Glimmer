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
    std::uint64_t physical_total_bytes = 0;
    std::uint64_t physical_free_bytes = 0;
};

struct LaunchTimingObservation {
    std::uint64_t task_id = 0;
    bool remote = false;
    bool lease_reused = false;
    std::uint64_t acquire_nanoseconds = 0;
    std::uint64_t acquire_transport_nanoseconds = 0;
    std::uint32_t acquire_request_count = 0;
    std::uint32_t claim_poll_count = 0;
    std::uint32_t wait_request_count = 0;
    std::uint64_t cuda_launch_nanoseconds = 0;
    std::uint64_t event_tracking_nanoseconds = 0;
    std::uint32_t cuda_launch_status = 0;
    bool event_tracking_succeeded = false;
};

struct LaunchCompletionObservation {
    std::uint64_t task_id = 0;
    bool remote = false;
    std::uint64_t completion_nanoseconds = 0;
    std::uint64_t completion_transport_nanoseconds = 0;
    std::uint32_t completion_request_count = 0;
    bool completed = false;
};

void report_diagnostic(const char* message) noexcept;
void report_kernel_launch_diagnostic(const KernelLaunchObservation& observation,
                                     std::uint64_t launch_count) noexcept;
void report_graph_launch_diagnostic(const char* api_name, const void* stream,
                                    std::uint64_t launch_count) noexcept;
void report_memory_info_diagnostic(const MemoryInfoObservation& observation) noexcept;
void report_launch_timing_diagnostic(const LaunchTimingObservation& observation) noexcept;
void report_launch_completion_diagnostic(const LaunchCompletionObservation& observation) noexcept;

}  // namespace glimmer::interceptor
