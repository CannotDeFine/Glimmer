#include "internal/diagnostics.h"

#include <cerrno>
#include <charconv>
#include <cstring>
#include <system_error>
#include <unistd.h>

namespace glimmer::interceptor {

namespace {

void write_message(const char* message, std::size_t message_length) noexcept {
    if (message == nullptr) {
        return;
    }
    if (message_length == 0) {
        return;
    }

    // The preload library cannot use an allocating logger while resolving CUDA
    // symbols. A fixed-buffer write keeps diagnostics reentrant.
    std::size_t bytes_remaining = message_length;
    const char* next_byte = message;
    while (bytes_remaining != 0) {
        const ssize_t bytes_written = ::write(STDERR_FILENO, next_byte, bytes_remaining);
        if (bytes_written > 0) {
            next_byte += bytes_written;
            bytes_remaining -= static_cast<std::size_t>(bytes_written);
            continue;
        }
        if (bytes_written < 0 && errno == EINTR) {
            continue;
        }
        break;
    }
}

char* append_text(char* destination, char* end, const char* text) noexcept {
    if (text == nullptr) {
        return destination;
    }
    while (*text != '\0' && destination != end) {
        *destination++ = *text++;
    }
    return destination;
}

template <typename Integer>
char* append_number(char* destination, char* end, Integer value, int base = 10) noexcept {
    const auto result = std::to_chars(destination, end, value, base);
    return result.ec == std::errc{} ? result.ptr : destination;
}

}  // namespace

void report_diagnostic(const char* message) noexcept {
    if (message == nullptr) {
        return;
    }
    write_message(message, std::strlen(message));
}

void report_kernel_launch_diagnostic(const KernelLaunchObservation& observation,
                                     std::uint64_t launch_count) noexcept {
    char message[512]{};
    char* cursor = message;
    char* const end = message + sizeof(message);
    cursor = append_text(cursor, end, "[glimmer] observed CUDA kernel launch #");
    cursor = append_number(cursor, end, launch_count);
    cursor = append_text(cursor, end, " api=");
    cursor = append_text(cursor, end,
                         observation.api_name == nullptr ? "unknown" : observation.api_name);
    cursor = append_text(cursor, end, " grid=(");
    cursor = append_number(cursor, end, observation.grid_dim_x);
    cursor = append_text(cursor, end, ",");
    cursor = append_number(cursor, end, observation.grid_dim_y);
    cursor = append_text(cursor, end, ",");
    cursor = append_number(cursor, end, observation.grid_dim_z);
    cursor = append_text(cursor, end, ") block=(");
    cursor = append_number(cursor, end, observation.block_dim_x);
    cursor = append_text(cursor, end, ",");
    cursor = append_number(cursor, end, observation.block_dim_y);
    cursor = append_text(cursor, end, ",");
    cursor = append_number(cursor, end, observation.block_dim_z);
    cursor = append_text(cursor, end, ") shared_memory_bytes=");
    cursor = append_number(cursor, end, observation.shared_memory_bytes);
    cursor = append_text(cursor, end, " stream=0x");
    cursor = append_number(cursor, end, reinterpret_cast<std::uintptr_t>(observation.stream), 16);
    cursor = append_text(cursor, end, "\n");
    write_message(message, static_cast<std::size_t>(cursor - message));
}

void report_memory_info_diagnostic(const MemoryInfoObservation& observation) noexcept {
    char message[384]{};
    char* cursor = message;
    char* const end = message + sizeof(message);
    const std::uint64_t used_bytes = observation.total_bytes >= observation.free_bytes
                                         ? observation.total_bytes - observation.free_bytes
                                         : 0;
    cursor = append_text(cursor, end, "[glimmer] memory info api=");
    cursor = append_text(cursor, end,
                         observation.api_name == nullptr ? "unknown" : observation.api_name);
    cursor = append_text(cursor, end, " device=");
    cursor = append_number(cursor, end, observation.device);
    cursor = append_text(cursor, end, " total_bytes=");
    cursor = append_number(cursor, end, observation.total_bytes);
    cursor = append_text(cursor, end, " used_bytes=");
    cursor = append_number(cursor, end, used_bytes);
    cursor = append_text(cursor, end, " free_bytes=");
    cursor = append_number(cursor, end, observation.free_bytes);
    cursor = append_text(cursor, end, " physical_total_bytes=");
    cursor = append_number(cursor, end, observation.physical_total_bytes);
    cursor = append_text(cursor, end, " physical_free_bytes=");
    cursor = append_number(cursor, end, observation.physical_free_bytes);
    cursor = append_text(cursor, end, "\n");
    write_message(message, static_cast<std::size_t>(cursor - message));
}

void report_launch_timing_diagnostic(const LaunchTimingObservation& observation) noexcept {
    char message[640]{};
    char* cursor = message;
    char* const end = message + sizeof(message);
    cursor = append_text(cursor, end, "[glimmer] launch timing task=");
    cursor = append_number(cursor, end, observation.task_id);
    cursor = append_text(cursor, end, " remote=");
    cursor = append_number(cursor, end, observation.remote ? 1U : 0U);
    cursor = append_text(cursor, end, " lease_reused=");
    cursor = append_number(cursor, end, observation.lease_reused ? 1U : 0U);
    cursor = append_text(cursor, end, " acquire_ns=");
    cursor = append_number(cursor, end, observation.acquire_nanoseconds);
    cursor = append_text(cursor, end, " acquire_transport_ns=");
    cursor = append_number(cursor, end, observation.acquire_transport_nanoseconds);
    cursor = append_text(cursor, end, " acquire_requests=");
    cursor = append_number(cursor, end, observation.acquire_request_count);
    cursor = append_text(cursor, end, " claim_polls=");
    cursor = append_number(cursor, end, observation.claim_poll_count);
    cursor = append_text(cursor, end, " wait_requests=");
    cursor = append_number(cursor, end, observation.wait_request_count);
    cursor = append_text(cursor, end, " cuda_launch_ns=");
    cursor = append_number(cursor, end, observation.cuda_launch_nanoseconds);
    cursor = append_text(cursor, end, " event_tracking_ns=");
    cursor = append_number(cursor, end, observation.event_tracking_nanoseconds);
    cursor = append_text(cursor, end, " cuda_status=");
    cursor = append_number(cursor, end, observation.cuda_launch_status);
    cursor = append_text(cursor, end, " event_tracking=");
    cursor = append_number(cursor, end, observation.event_tracking_succeeded ? 1U : 0U);
    cursor = append_text(cursor, end, "\n");
    write_message(message, static_cast<std::size_t>(cursor - message));
}

void report_launch_completion_diagnostic(const LaunchCompletionObservation& observation) noexcept {
    char message[384]{};
    char* cursor = message;
    char* const end = message + sizeof(message);
    cursor = append_text(cursor, end, "[glimmer] completion timing task=");
    cursor = append_number(cursor, end, observation.task_id);
    cursor = append_text(cursor, end, " remote=");
    cursor = append_number(cursor, end, observation.remote ? 1U : 0U);
    cursor = append_text(cursor, end, " completion_ns=");
    cursor = append_number(cursor, end, observation.completion_nanoseconds);
    cursor = append_text(cursor, end, " completion_transport_ns=");
    cursor = append_number(cursor, end, observation.completion_transport_nanoseconds);
    cursor = append_text(cursor, end, " completion_requests=");
    cursor = append_number(cursor, end, observation.completion_request_count);
    cursor = append_text(cursor, end, " completed=");
    cursor = append_number(cursor, end, observation.completed ? 1U : 0U);
    cursor = append_text(cursor, end, "\n");
    write_message(message, static_cast<std::size_t>(cursor - message));
}

}  // namespace glimmer::interceptor
