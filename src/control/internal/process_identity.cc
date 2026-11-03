#include "process_identity.h"

#include <charconv>
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>

namespace glimmer::control {
bool read_process_start_time(std::uint32_t pid, std::uint64_t* start_time_ticks) noexcept {
    if (pid == 0 || start_time_ticks == nullptr) {
        return false;
    }
    try {
        const std::string path = "/proc/" + std::to_string(pid) + "/stat";
        const int file_descriptor = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (file_descriptor < 0) {
            return false;
        }
        char buffer[4096]{};
        ssize_t bytes_read = 0;
        do {
            bytes_read = ::read(file_descriptor, buffer, sizeof(buffer) - 1);
        } while (bytes_read < 0 && errno == EINTR);
        const int close_result = ::close(file_descriptor);
        if (bytes_read <= 0 || close_result != 0) {
            return false;
        }
        const std::string_view stat_line(buffer, static_cast<std::size_t>(bytes_read));
        const std::size_t command_end = stat_line.rfind(')');
        if (command_end == std::string_view::npos || command_end + 2 > stat_line.size()) {
            return false;
        }
        std::size_t offset = command_end + 2;
        for (unsigned int field = 3; field <= 22; ++field) {
            while (offset < stat_line.size() && stat_line[offset] == ' ') {
                ++offset;
            }
            if (offset == stat_line.size()) {
                return false;
            }
            const std::size_t token_start = offset;
            while (offset < stat_line.size() && stat_line[offset] != ' ') {
                ++offset;
            }
            if (field == 22) {
                std::uint64_t parsed = 0;
                const auto result = std::from_chars(stat_line.data() + token_start,
                                                    stat_line.data() + offset, parsed);
                if (result.ec != std::errc{} || result.ptr != stat_line.data() + offset ||
                    parsed == 0) {
                    return false;
                }
                *start_time_ticks = parsed;
                return true;
            }
        }
    } catch (...) {
        return false;
    }
    return false;
}

std::optional<TaskPeerIdentity> current_process_identity() noexcept {
    TaskPeerIdentity identity{static_cast<std::uint32_t>(::getpid()),
                              static_cast<std::uint32_t>(::geteuid()), 0};
    return read_process_start_time(identity.pid, &identity.start_time_ticks)
               ? std::optional{identity}
               : std::nullopt;
}
}  // namespace glimmer::control
