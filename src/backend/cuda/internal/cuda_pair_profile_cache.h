#pragma once

#include "cuda_profile_cache.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace glimmer::backend::cuda {

inline constexpr std::size_t kCudaPairProfileArchiveMaxBytes = std::size_t{128} * 1024 * 1024;
inline constexpr std::uint64_t kCudaPairProfileMaxSampleNs = 3600ULL * 1000000000ULL;

struct CudaPairProfileConfig {
    std::size_t max_pairs{128}, max_candidates{32};
    std::size_t max_bytes{std::size_t{64} * 1024 * 1024};
    std::uint32_t warmup_calls{1}, samples_per_candidate{3};
};

enum class CudaPairProfileDomainKind : std::uint32_t { kSameContext = 1, kDistinctContexts = 2 };
struct CudaPairProfileDomain {
    CudaPairProfileDomainKind kind{CudaPairProfileDomainKind::kSameContext};
    // Positive, stable logical arrangement labels, NOT native pointers, source
    // connection IDs or execution authority. Equal exactly for kSameContext.
    // The execution owner must independently verify the current arrangement.
    std::array<std::uint64_t, 2> labels{};
    bool operator==(const CudaPairProfileDomain&) const = default;
};

struct CudaPairProfileLimits {
    unsigned int sm_count{}, blocks_per_sm{}, threads_per_sm{}, registers_per_sm{}, max_workers{};
    std::uint64_t shared_memory_bytes_per_sm{};
    bool operator==(const CudaPairProfileLimits&) const = default;
};
struct CudaPairProfileVariant {
    transform::PtxExecutionKind kind{transform::PtxExecutionKind::kPreemptive};
    unsigned int workers{}, registers_per_thread{}, occupancy_blocks_per_sm{};
    std::uint64_t static_shared_memory_bytes{};
    bool operator==(const CudaPairProfileVariant&) const = default;
};
struct CudaPairProfileCandidate {
    std::array<CudaPairProfileVariant, 2> legs{};
    bool operator==(const CudaPairProfileCandidate&) const = default;
};
// Pure checked resource feasibility used by both catalog construction and cache
// validation. It is not SDK metadata provenance, SM coverage or native authority.
[[nodiscard]] bool cuda_pair_profile_candidate_fits(
    const std::array<CudaProfileGeometry, 2>& geometry, CudaPairProfileLimits limits,
    const CudaPairProfileCandidate& candidate) noexcept;

struct CudaPairProfileKernelState;
class CudaPairProfileKernel final {
   public:
    [[nodiscard]] explicit operator bool() const noexcept {
        return state_ != nullptr;
    }

   private:
    friend class CudaPairProfileCache;
    std::weak_ptr<const void> owner_;
    std::shared_ptr<CudaPairProfileKernelState> state_;
};
enum class CudaPairProfileBindStatus { kBound, kInvalid, kMissingOriginal, kCapacity };
struct CudaPairProfileBindResult {
    CudaPairProfileBindStatus status{CudaPairProfileBindStatus::kInvalid};
    CudaPairProfileKernel kernel{};
};

struct CudaPairProfileMeasurement {
    std::uint64_t total_ns{};
    std::uint32_t samples{};
    bool operator==(const CudaPairProfileMeasurement&) const = default;
};
struct CudaPairProfileChoice {
    // A miss or tie is serial originals. first_original is the shorter measured
    // original; exact ties and unavailable identities deterministically use 0.
    bool is_shared{}, is_trained{};
    std::size_t first_original{};
    std::optional<std::size_t> candidate{};
    std::array<CudaPairProfileMeasurement, 2> original{}, paired{};
    CudaPairProfileCandidate configuration{};
    long double normalized_speed{};
};
// CPU value comparison only, not resource/measurement provenance or execution
// authority. Positive means first is better. Shared wins over serial; shared
// scores use exact rational arithmetic, serial choices prefer the shorter
// qualified original mean. Missing originals tie conservatively. Invalid or
// out-of-bound measurement facts return nullopt; normalized_speed is ignored.
[[nodiscard]] std::optional<int> cuda_pair_profile_compare_choices(
    const CudaPairProfileChoice& first, const CudaPairProfileChoice& second) noexcept;
struct CudaPairProfileReservation {
    std::size_t candidate{};
    CudaPairProfileCandidate configuration{};
    std::array<std::shared_ptr<CudaExecutionObserver>, 2> observations{};
    [[nodiscard]] explicit operator bool() const noexcept {
        return observations[0] != nullptr && observations[1] != nullptr;
    }
};
enum class CudaPairProfileReserveStatus { kReserved, kTrained, kBusy, kInvalid, kCapacity };
struct CudaPairProfileReserveResult {
    CudaPairProfileReserveStatus status{CudaPairProfileReserveStatus::kInvalid};
    CudaPairProfileReservation reservation{};
};
struct CudaPairProfileSnapshot {
    std::size_t kernels{}, pairs{}, retained_bytes{}, active{}, trained_pairs{};
    std::uint64_t samples{}, rejected_samples{}, cancelled{}, evictions{}, loaded_pairs{};
    bool is_valid{}, is_learning{};
};

// Private, thread-safe CPU measurement cache, not an executor or GPU-overlap
// proof. The nonnull original cache must be readonly. At most 2*max_pairs kernel
// identities are retained, including externally pinned tokens. Active samples
// and live observers pin their pair; eviction never overwrites them. Bytes
// conservatively include pinned identity capacities and metadata. Forked access
// rejects before any inherited mutex. No SDK, I/O or automatic clock calls on
// bind/choose/encode/restore; observer clocks run outside metadata locks.
class CudaPairProfileCache final {
   public:
    explicit CudaPairProfileCache(std::shared_ptr<const CudaProfileCache> originals,
                                  CudaPairProfileConfig config = {},
                                  CudaProfileClock clock = nullptr, bool is_learning = true);
    ~CudaPairProfileCache();
    CudaPairProfileCache(const CudaPairProfileCache&) = delete;
    CudaPairProfileCache& operator=(const CudaPairProfileCache&) = delete;

    [[nodiscard]] CudaPairProfileKernel bind(const CudaProfileFunction& function,
                                             CudaProfileGeometry geometry) noexcept;
    [[nodiscard]] CudaPairProfileBindResult bind_checked(const CudaProfileFunction& function,
                                                         CudaProfileGeometry geometry) noexcept;
    // Both paths validate the complete catalog before looking up or reserving a
    // case. Catalog ordering is identity. Reverse orientation swaps every leg,
    // including domain labels; identical legs prefer exact catalog orientation.
    // choose never learns or returns observers, even on a learning cache.
    [[nodiscard]] CudaPairProfileChoice choose(
        const std::array<CudaPairProfileKernel, 2>& kernels, CudaPairProfileDomain domain,
        CudaPairProfileLimits limits,
        std::span<const CudaPairProfileCandidate> candidates) const noexcept;
    // Non-inserting lookup for speculative global selection. Resolves qualified
    // originals, pins ONLY already-retained kernel identities and never binds,
    // prunes, evicts, learns or allocates cache records. A missing pair preserves
    // exact known original means for serial fallback without displacing a winner.
    [[nodiscard]] CudaPairProfileChoice lookup(
        const std::array<CudaProfileFunction, 2>& functions,
        const std::array<CudaProfileGeometry, 2>& geometry, CudaPairProfileDomain domain,
        CudaPairProfileLimits limits,
        std::span<const CudaPairProfileCandidate> candidates) const noexcept;
    // Learning-only: reserves the first incomplete candidate. Each leg needs
    // exactly one begin and one successful final retirement, with strict host
    // interval overlap. Duplicate callbacks before commit invalidate the sample;
    // callbacks after terminal commit/cancellation cannot change it. Abandoning
    // either unfinished observer cancels once. Release both terminal observer
    // handles before reserving the next sample of this pair: retained leases
    // still consume its bounded reservation storage. A finished leg may release
    // its handle before the other finishes. No lease authorizes replay or GPU cleanup.
    [[nodiscard]] CudaPairProfileReservation reserve(
        const std::array<CudaPairProfileKernel, 2>& kernels, CudaPairProfileDomain domain,
        CudaPairProfileLimits limits,
        std::span<const CudaPairProfileCandidate> candidates) noexcept;
    // Pure diagnostic distinction only, never a native status or authority.
    // Live/retained reservation storage is transient; capacity failure is not.
    [[nodiscard]] CudaPairProfileReserveResult reserve_checked(
        const std::array<CudaPairProfileKernel, 2>& kernels, CudaPairProfileDomain domain,
        CudaPairProfileLimits limits,
        std::span<const CudaPairProfileCandidate> candidates) noexcept;
    [[nodiscard]] CudaPairProfileSnapshot snapshot() const noexcept;

    // V1 initial-only, V2 canonical context and V3 complete function geometry, little-endian
    // cache_archive.h framing; final FNV64 corruption
    // checksum. Only complete inactive catalogs are encoded. Restore is atomic
    // into an empty cache (including no bound kernels), resolving existing exact
    // qualified originals and sample facts without mutating the single cache.
    [[nodiscard]] std::optional<std::vector<std::byte>> encode(
        std::string_view runtime_identity) const noexcept;
    [[nodiscard]] bool restore(std::span<const std::byte> archive,
                               std::string_view runtime_identity) noexcept;

   private:
    struct Implementation;
    std::shared_ptr<Implementation> implementation_;
};

}  // namespace glimmer::backend::cuda
