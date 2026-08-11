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
    cursor = append_text(cursor, end, "\n");
    write_message(message, static_cast<std::size_t>(cursor - message));
}

}  // namespace glimmer::interceptor
