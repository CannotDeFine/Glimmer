#pragma once

#include <optional>
#include <string>

namespace glimmer::interceptor {

// Thread-safe metadata lookup. The caller must keep the image loaded. Returns
// an owned pathname, preserves errno, caches nothing and never invokes CUDA.
// Unproven fast lookups retain the original dladdr fallback.
[[nodiscard]] std::optional<std::string> loaded_image_path(const void* address);

}  // namespace glimmer::interceptor
