#include "internal/diagnostics.h"

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
    const ssize_t bytes_written = ::write(STDERR_FILENO, message, message_length);
    static_cast<void>(bytes_written);
}

}  // namespace glimmer::interceptor
