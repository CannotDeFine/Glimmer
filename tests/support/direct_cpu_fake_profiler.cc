// No signals or sampling: lifecycle fault injection for the preload controller.
#include <fcntl.h>
#include <gperftools/profiler.h>
#include <unistd.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {
ProfilerState current{};
int output_fd{-1};

bool mode_is(const char* expected) {
    const char* selected = std::getenv("GLIMMER_FAKE_MODE");
    return selected != nullptr && std::strcmp(selected, expected) == 0;
}
}  // namespace

// NOLINTNEXTLINE(readability-identifier-naming): vendor ABI fixture.
extern "C" int ProfilerStartWithOptions(const char* path, const ProfilerOptions* options) {
    if (mode_is("start-fail")) {
        return 0;
    }
    if (options == nullptr || options->filter_in_thread == nullptr || current.enabled != 0) {
        std::_Exit(89);
    }
    // Start runs on the controller, not the main thread; it must be excluded.
    if (options->filter_in_thread(options->filter_in_thread_arg) != 0) {
        std::_Exit(89);
    }
    output_fd = open(path, O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC, 0600);
    if (output_fd < 0) {
        return 0;
    }
    current.enabled = 1;
    std::snprintf(current.profile_name, sizeof(current.profile_name), "%s", path);
    return 1;
}

// NOLINTNEXTLINE(readability-identifier-naming): vendor ABI fixture.
extern "C" void ProfilerStop() {
    if (mode_is("foreign-profiler")) {
        std::_Exit(89);
    }
    if (output_fd >= 0) {
        const std::array<char, 80> bytes{};
        if (write(output_fd, bytes.data(), bytes.size()) != static_cast<ssize_t>(bytes.size()) ||
            close(output_fd) != 0) {
            std::_Exit(89);
        }
        output_fd = -1;
    }
    if (!mode_is("stop-fail")) {
        current = {};
    }
}

// NOLINTNEXTLINE(readability-identifier-naming): vendor ABI fixture.
extern "C" void ProfilerGetCurrentState(ProfilerState* output) {
    *output = current;
    if (mode_is("foreign-profiler")) {
        output->enabled = 1;
    }
}
