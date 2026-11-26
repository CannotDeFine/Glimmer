#include "internal/cuda_sgemm_image.h"

namespace glimmer::backend::cuda {
std::string_view cuda_sgemm_image(unsigned int tile_side) noexcept {
#ifdef GLIMMER_HAS_CUTLASS_SGEMM
    static constexpr char image[] = {
#include "cuda_sgemm_ptx.inc"
    };
    static constexpr char fine_image[] = {
#include "cuda_sgemm_tile32_ptx.inc"
    };
    if (tile_side == 64) return {image, sizeof(image) - 1};
    if (tile_side == 32) return {fine_image, sizeof(fine_image) - 1};
    return {};
#else
    (void)tile_side;
    return {};
#endif
}
std::string_view cuda_sgemm_sharded_image(unsigned int tile_side) noexcept {
#ifdef GLIMMER_HAS_CUTLASS_SGEMM
    static constexpr char image[] = {
#include "cuda_sgemm_tile32_sharded_ptx.inc"
    };
    if (tile_side == 32) return {image, sizeof(image) - 1};
    return {};
#else
    (void)tile_side;
    return {};
#endif
}
}  // namespace glimmer::backend::cuda
