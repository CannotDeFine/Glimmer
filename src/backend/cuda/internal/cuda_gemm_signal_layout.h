#pragma once

#include <cstddef>

namespace glimmer::backend::cuda {

// Private host-mapped GEMM control ABI. Scalar providers never use this layout.
inline constexpr std::size_t kGemmSignalShards = 32;
inline constexpr std::size_t kGemmSignalShardBytes = 64;
inline constexpr std::size_t kGemmSignalSlotBytes = kGemmSignalShards * kGemmSignalShardBytes;
inline constexpr std::size_t kPersistentProgressHostSlotBytes = 64;

}  // namespace glimmer::backend::cuda
