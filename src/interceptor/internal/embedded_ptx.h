#pragma once

#include <chrono>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace glimmer::interceptor {

// Parse bounded cuobjdump output, retaining separate module images.
[[nodiscard]] std::optional<std::vector<std::string>> parse_embedded_ptx(std::string_view output);

// The absolute tool path is trusted deployment configuration, never a shell command.
// Failure, timeout, excessive output, or absent PTX returns no images.
[[nodiscard]] std::optional<std::vector<std::string>> extract_embedded_ptx(
    const std::string& tool, const std::string& binary,
    std::chrono::milliseconds timeout = std::chrono::seconds{5});

}  // namespace glimmer::interceptor
