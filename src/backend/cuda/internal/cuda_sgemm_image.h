#pragma once

#include <string_view>

namespace glimmer::backend::cuda {
// Trusted immutable build input, not application-provided executable bytes.
// Empty when the optional provider was not built. NUL-terminated when present.
[[nodiscard]] std::string_view cuda_sgemm_image(unsigned int tile_side = 64) noexcept;
[[nodiscard]] std::string_view cuda_sgemm_sharded_image(unsigned int tile_side) noexcept;
}  // namespace glimmer::backend::cuda
