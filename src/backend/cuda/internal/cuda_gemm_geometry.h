#pragma once

#include <cstdint>
#include <limits>

namespace glimmer::backend::cuda {
constexpr bool cuda_gemm_tile_side_is_valid(unsigned int side) noexcept {
    return side == 32 || side == 64;
}

// Zero denotes an empty/invalid/unrepresentable matrix, never wrapped progress.
// Square tiles make the two matrix axes symmetric.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
constexpr std::uint64_t cuda_gemm_matrix_tiles(std::uint64_t rows, std::uint64_t columns,
                                               unsigned int side) noexcept {
    if (!cuda_gemm_tile_side_is_valid(side)) return 0;
    const auto row_tiles = rows / side + static_cast<unsigned int>(rows % side != 0);
    const auto column_tiles = columns / side + static_cast<unsigned int>(columns % side != 0);
    if (!row_tiles || column_tiles > std::numeric_limits<std::uint64_t>::max() / row_tiles)
        return 0;
    return row_tiles * column_tiles;
}
}  // namespace glimmer::backend::cuda
