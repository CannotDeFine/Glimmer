#pragma once

#include "glimmer/transform/ptx_transform.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace glimmer::backend::cuda {

inline constexpr std::uint32_t kCudaPtxGeneratorAbi = 10;
inline constexpr std::size_t kCudaPtxArchiveMaxBytes = std::size_t{128} * 1024 * 1024;
struct CudaPtxCacheLimits {
    std::size_t max_entries{128};
    // Logical source/image/metadata retention, not a process RSS measurement.
    std::size_t max_bytes{std::size_t{64} * 1024 * 1024};
};
enum class CudaPtxCacheStatus : std::uint8_t {
    kReady,
    kUnsupported,
    kBypass,
    kInvalid,
    kWrongProcess
};
struct CudaPtxCacheResult {
    CudaPtxCacheStatus status{CudaPtxCacheStatus::kBypass};
    std::shared_ptr<const transform::PersistentPtx> image;
    bool was_hit{};
};
struct CudaPtxCacheSnapshot {
    std::size_t entries{}, retained_bytes{};
    std::uint64_t lookups{}, hits{}, transforms{}, evictions{}, bypasses{};
    bool is_valid{};
    std::uint64_t loaded_entries{};
};
using CudaPtxCacheHash = std::size_t (*)(std::string_view) noexcept;

// Thread-safe, CPU-only, process-bound preparation cache. Exact bytes/kind/ABI,
// preparation limits and module-sharing contract, not hash equality, authorize reuse.
// Pure hash callbacks run before exclusion; transformation under exclusion calls only owned CPU
// code, never CUDA or IPC. Returned immutable images contain no native handles. Pinned entries
// cannot be evicted; capacity pressure safely bypasses memoization. Zero bounds disable it.
class CudaPtxCache final {
   public:
    explicit CudaPtxCache(CudaPtxCacheLimits limits = {}, CudaPtxCacheHash hash = nullptr);
    ~CudaPtxCache();
    CudaPtxCache(const CudaPtxCache&) = delete;
    CudaPtxCache& operator=(const CudaPtxCache&) = delete;
    [[nodiscard]] CudaPtxCacheResult lookup(std::string_view source,
                                            transform::PtxExecutionKind kind,
                                            bool shares_original_module = false) noexcept;
    [[nodiscard]] CudaPtxCacheResult lookup(std::string_view source,
                                            transform::PtxExecutionKind kind,
                                            bool shares_original_module,
                                            transform::PtxTransformLimits limits) noexcept;
    [[nodiscard]] CudaPtxCacheSnapshot snapshot() const noexcept;
    // Only unborrowed CPU entries are removed. No SDK resources are touched.
    [[nodiscard]] bool prune() noexcept;
    // CPU-only codec for trusted deployment artifacts, not client payloads.
    // Generated PTX is executable input: file ownership and executable/generator
    // identity are mandatory at composition. A checksum only detects corruption.
    // Restore commits all-or-none into a pristine cache, without regenerating code.
    // Version 2 records exact preparation limits; trusted token eligibility is
    // restored as generation provenance, not re-lexed or client authority.
    // Every session still owns fresh native loading/ABI validation and accounting.
    [[nodiscard]] std::optional<std::vector<std::byte>> encode(
        std::string_view runtime_identity) const noexcept;
    [[nodiscard]] bool restore(std::span<const std::byte> archive,
                               std::string_view runtime_identity) noexcept;

   private:
    struct Implementation;
    std::unique_ptr<Implementation> implementation_;
};

}  // namespace glimmer::backend::cuda
