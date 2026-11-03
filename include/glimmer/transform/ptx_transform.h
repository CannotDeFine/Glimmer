#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace glimmer::transform {

inline constexpr std::size_t kMaxPersistentPtxBytes = std::size_t{4} * 1024 * 1024;
inline constexpr std::size_t kMaxPtxTransformSourceBytes = std::size_t{64} * 1024 * 1024;
inline constexpr std::size_t kMaxPtxTransformOutputBytes = std::size_t{256} * 1024 * 1024;
inline constexpr std::size_t kMaxPtxTransformTokens = std::size_t{16} * 1024 * 1024;

// Trusted preparation eligibility, independent of cache/session retention.
// Token count bounds logical records, not vector capacity or peak process RSS.
// Legacy adapters retain their original source/output and token bounds.
struct PtxTransformLimits {
    std::size_t source_bytes{kMaxPersistentPtxBytes};
    std::size_t output_bytes{kMaxPersistentPtxBytes};
    std::size_t token_count{kMaxPersistentPtxBytes / 2};
    // Explicit diagnostic ABI: preemptive PTX>=6.0/Sm70 only. Adds one
    // mapped32-bit completion-word pointer, never extends the retreat pointee.
    bool observes_block_completion{};
    bool operator==(const PtxTransformLimits&) const = default;
};

[[nodiscard]] constexpr bool valid_ptx_transform_limits(PtxTransformLimits limits) noexcept {
    return limits.source_bytes && limits.source_bytes <= kMaxPtxTransformSourceBytes &&
           limits.output_bytes && limits.output_bytes <= kMaxPtxTransformOutputBytes &&
           limits.token_count && limits.token_count <= kMaxPtxTransformTokens;
}

// Keep the existing underlying type for callers compiled against this interface.
enum class PtxExecutionKind {  // NOLINT(performance-enum-size)
    kStatic,
    kBlockStatic,
    kDynamic,
    kPreemptive,
    kSliced,
};

// Original entry parameters use the same bounded opaque scalar/aligned-array
// layout in every family. Parameter text/order is preserved; native ABI checks
// still determine offsets. This does not grant cooperative body capabilities.
struct PtxParameterLayout {
    std::size_t size_bytes = 0;
    std::size_t alignment_bytes = 1;
};

struct PtxKernelVariant {
    std::string original_name;
    std::string persistent_name;
    std::size_t parameter_count = 0;
    std::vector<PtxParameterLayout> parameter_layout{};
    PtxExecutionKind execution_kind = PtxExecutionKind::kStatic;
    bool observes_block_completion{};
};

struct PersistentPtx {
    std::string image;
    std::vector<PtxKernelVariant> kernels;
    bool has_module_variables{};
};

// Pure, CUDA-independent transformation. Unsupported or malformed modules
// return no image. Unsupported entries are retained without a variant.
// Dynamic/preemptive/sliced variants are generator/backend interfaces only:
// their state and complete stream-lifetime contract must be owned by the caller.
// Stateful declarations require explicit same-native-module ownership. Separate
// module adapters must leave shares_original_module false.
// Bounded compiler debug metadata, lexical scopes and thread-local data are
// preserved; supported entry launch constraints also apply to private entries.
[[nodiscard]] std::optional<PersistentPtx> transform_persistent_ptx(
    std::string_view ptx, PtxExecutionKind kind = PtxExecutionKind::kStatic,
    bool shares_original_module = false);
[[nodiscard]] std::optional<PersistentPtx> transform_persistent_ptx(std::string_view ptx,
                                                                    PtxExecutionKind kind,
                                                                    bool shares_original_module,
                                                                    PtxTransformLimits limits);

// Prefer the existing independent-thread ABI; otherwise select a safe
// block-static variant. Both families execute completely on the caller stream.
[[nodiscard]] std::optional<PersistentPtx> transform_transparent_ptx(std::string_view ptx);

}  // namespace glimmer::transform
