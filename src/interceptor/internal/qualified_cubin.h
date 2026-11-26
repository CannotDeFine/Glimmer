#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace glimmer::interceptor {

using BinaryImageDigest = std::array<std::uint32_t, 8>;
inline constexpr std::size_t kMaxBinaryImageBytes = std::size_t{8} * 1024 * 1024;

enum class QualifiedAddMapping { kSingleCta, kRange };
enum class QualifiedGemmMapping { kPlane, kRange };

// SHA-256 words in digest order; bounded CPU-only content identity, not an
// authentication API. Input must remain readable and stable during the call.
[[nodiscard]] std::optional<BinaryImageDigest> binary_image_digest(
    std::span<const std::byte> image) noexcept;

// Exact original module -> privately extended mapped-add module: single-CTA
// mapping or physical-CTA-plus-private-base range mapping.
// Copies before qualification, preserves caller bytes, catches allocation errors.
// No CUDA calls or source-handle discovery; the caller still owes context and
// lifetime proof. Only compute capability 70 and the pinned input/output qualify.
[[nodiscard]] std::optional<std::vector<std::byte>> rewrite_qualified_add_cubin(
    std::span<const std::byte> original, unsigned int compute_capability,
    QualifiedAddMapping mapping = QualifiedAddMapping::kSingleCta) noexcept;

// Bounded internal block decoder. Input and output must not overlap. Failure
// may leave partial output, which must be discarded. No allocations or CUDA.
[[nodiscard]] bool decode_binary_lz4_block(std::span<const std::byte> input,
                                           std::span<std::byte> output) noexcept;

// Exact pinned fatbinary -> the selected private cubin mapping above. Whole
// input identity precedes fixed-offset decompression; other sources decline.
[[nodiscard]] std::optional<std::vector<std::byte>> rewrite_qualified_add_fatbin(
    std::span<const std::byte> original, unsigned int compute_capability,
    QualifiedAddMapping mapping = QualifiedAddMapping::kSingleCta) noexcept;

// Exact pinned sm_70 GEMM image -> its private Z-plane/range mapping. Copy, whole
// input/output identities and the fixed qualified ELF recipe precede return.
// No source registration, CUDA calls, original mutation or public selection.
[[nodiscard]] std::optional<std::vector<std::byte>> rewrite_qualified_gemm_cubin(
    std::span<const std::byte> original, unsigned int compute_capability,
    QualifiedGemmMapping mapping = QualifiedGemmMapping::kPlane) noexcept;

// Exact pinned compressed envelope; only its pinned sm_70 member.
// Full envelope identity includes the unselected sm_75 member and padding.
[[nodiscard]] std::optional<std::vector<std::byte>> rewrite_qualified_gemm_fatbin(
    std::span<const std::byte> original, unsigned int compute_capability,
    QualifiedGemmMapping mapping = QualifiedGemmMapping::kPlane) noexcept;

}  // namespace glimmer::interceptor
