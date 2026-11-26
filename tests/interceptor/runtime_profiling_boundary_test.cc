#include "internal/runtime_api_bridge.h"

#include <cstdlib>
#include <iostream>
#include <string_view>

int main(int argc, char** argv) {
    using glimmer::interceptor::runtime_profiling_boundary_mode;
    using glimmer::interceptor::RuntimeProfilingBoundaryMode;
    if (argc != 2) return EXIT_FAILURE;
    const auto expected = std::string_view(argv[1]);
    const auto mode = runtime_profiling_boundary_mode();
    const bool matches =
        (expected == "disabled" && mode == RuntimeProfilingBoundaryMode::kDisabled) ||
        (expected == "enabled" && mode == RuntimeProfilingBoundaryMode::kEnabled) ||
        (expected == "invalid" && mode == RuntimeProfilingBoundaryMode::kInvalid);
    if (!matches) {
        std::cerr << "Profiling selection accepts only unset, zero or one\n";
        return EXIT_FAILURE;
    }
    if (setenv("GLIMMER_PROFILE_RUNTIME_NATIVE_BOUNDARY", "changed", 1) != 0 ||
        runtime_profiling_boundary_mode() != mode) {
        std::cerr << "Profiling selection must remain immutable after first use\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
