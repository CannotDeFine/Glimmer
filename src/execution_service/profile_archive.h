#pragma once

#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace glimmer::app {
// Explicit startup/shutdown I/O, never submission-path or cache-lock work.
// Absolute no-follow paths, owned private files and a protected owned parent.
[[nodiscard]] std::optional<std::string> profile_runtime_identity() noexcept;
// Shared bounded I/O for trusted preparation and measurement artifacts.
[[nodiscard]] std::optional<std::vector<std::byte>> read_execution_archive(
    std::string_view path, std::size_t maximum_bytes) noexcept;
[[nodiscard]] bool write_execution_archive(std::string_view path, std::span<const std::byte> bytes,
                                           std::size_t maximum_bytes) noexcept;
[[nodiscard]] std::optional<std::vector<std::byte>> read_profile_archive(
    std::string_view path) noexcept;
// Atomically publish a complete private file; never overwrite an existing name.
// Failure after publication may leave the complete file, never a partial target.
[[nodiscard]] bool write_profile_archive(std::string_view path,
                                         std::span<const std::byte> bytes) noexcept;
}  // namespace glimmer::app
