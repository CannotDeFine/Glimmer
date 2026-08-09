#include "internal/diagnostics.h"

#include <cerrno>
#include <cstring>
#include <unistd.h>

namespace glimmer::interceptor {

void report_diagnostic(const char* message) noexcept {
    if (message == nullptr) {
        return;
    }

    const std::size_t message_length = std::strlen(message);
    if (message_length == 0) {
        return;
    }

    // The preload library cannot use an allocating logger while resolving CUDA
    // symbols. A single fixed-string write keeps diagnostics reentrant.
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

}  // namespace glimmer::interceptor
