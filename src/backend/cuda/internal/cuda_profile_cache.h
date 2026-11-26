#pragma once

#include "cuda_execution_observer.h"
#include "glimmer/transform/ptx_transform.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace glimmer::backend::cuda {

inline constexpr std::size_t kCudaProfileArchiveMaxBytes = std::size_t{128} * 1024 * 1024;
struct CudaProfilePlatform {
    std::array<std::byte, 16> uuid{};
    unsigned int driver_version{}, toolkit_version{}, generator_abi{};
    bool operator==(const CudaProfilePlatform&) const = default;
};
struct CudaProfileConfig {
    std::size_t max_functions{128};
    std::size_t max_bytes{std::size_t{64} * 1024 * 1024};
    std::uint32_t warmup_calls{1}, samples_per_candidate{3};
    // Original (zero workers) is always a candidate. Nonzero entries must be
    // distinct, explicit and within the session's configured maximum.
    std::vector<unsigned int> workers{80, 160, 320};
    // Empty preserves the session's single configured family.
    std::vector<transform::PtxExecutionKind> kinds{};
};
struct CudaProfileContextConfiguration {
    // Zero denotes LEGACY INITIAL-ONLY provenance, never a verified general
    // configuration. Otherwise flags/cache/shared and supported limit fields.
    std::uint32_t available{};
    std::array<std::uint64_t, 19> values{};
    bool operator==(const CudaProfileContextConfiguration&) const = default;
};
[[nodiscard]] bool cuda_profile_context_configuration_valid(
    const CudaProfileContextConfiguration& configuration) noexcept;
struct CudaProfileFunctionConfiguration {
    // Zero is legacy INITIAL-ONLY provenance. Bits 0..5 identify original,
    // static, block-static, dynamic, preemptive and sliced native functions.
    std::uint32_t functions{};
    // 0: untouched initial preference; 1..4: successful explicit cache enum+1.
    // CACHE_MODE_CA is a different attribute, not a cache preference getter.
    std::uint32_t cache_preference{};
    std::array<std::uint32_t, 6> available{};
    std::array<std::array<std::int32_t, 16>, 6> values{};
    bool operator==(const CudaProfileFunctionConfiguration&) const = default;
};
[[nodiscard]] bool cuda_profile_function_configuration_valid(
    const CudaProfileFunctionConfiguration& configuration) noexcept;
struct CudaProfileGeometry {
    std::array<unsigned int, 3> grid{}, block{};
    unsigned int shared_memory_bytes{};
    transform::PtxExecutionKind kind{transform::PtxExecutionKind::kPreemptive};
    CudaProfileContextConfiguration context{};
    CudaProfileFunctionConfiguration function{};
    bool operator==(const CudaProfileGeometry&) const = default;
};
// Immutable code identity only: no SDK handle, generated-code authority or
// permission to execute. Shared views pin the existing cache's exact bytes.
struct CudaProfileCodeIdentity {
    CudaProfilePlatform platform;
    std::string source, binary, name;
    bool operator==(const CudaProfileCodeIdentity&) const = default;
};
struct CudaProfileOriginalMeasurement {
    std::shared_ptr<const CudaProfileCodeIdentity> identity;
    CudaProfileGeometry geometry;
    std::uint64_t total_ns{};
    std::uint32_t samples{};
};
struct CudaProfileCode;
class CudaProfileFunction final {
   public:
    [[nodiscard]] explicit operator bool() const noexcept {
        return code_ != nullptr;
    }

   private:
    friend class CudaProfileCache;
    std::weak_ptr<const void> owner_;
    std::shared_ptr<CudaProfileCode> code_;
};
struct CudaProfileChoice {
    unsigned int workers{};
    std::shared_ptr<CudaExecutionObserver> observation;
    bool was_selected{}, is_trained{};
    transform::PtxExecutionKind kind{transform::PtxExecutionKind::kPreemptive};
};
struct CudaProfileSharingLimits {
    unsigned int sm_count{}, threads_per_sm{1024}, max_workers{}, minimum_speed_percent{90};
};
struct CudaProfileSharingCandidate {
    transform::PtxExecutionKind kind{};
    unsigned int max_workers{};
};
struct CudaProfileSnapshot {
    std::size_t functions{}, retained_bytes{}, cases{}, trained_cases{};
    std::uint64_t samples{}, rejected_samples{}, cancelled{}, hits{}, bypasses{}, evictions{};
    bool is_valid{};
    std::uint64_t loaded_cases{};
    bool is_learning{};
};
// Nanoseconds from a monotonic clock; injected pure test clocks run outside locks.
using CudaProfileClock = std::uint64_t (*)() noexcept;

// Thread-safe CPU cache, bounded to 16 launch geometries per function and 36
// candidates including original. Full bytes/platform authorize reuse, not hashes
// or native handles. Function leases pin code; live observations pin their case.
// No SDK or disk operations. Fork access rejects before inherited exclusion.
class CudaProfileCache final {
   public:
    explicit CudaProfileCache(CudaProfileConfig config = {}, CudaProfileClock clock = nullptr,
                              bool is_learning = true);
    ~CudaProfileCache();
    CudaProfileCache(const CudaProfileCache&) = delete;
    CudaProfileCache& operator=(const CudaProfileCache&) = delete;
    [[nodiscard]] CudaProfileFunction bind(const CudaProfilePlatform& platform,
                                           std::string_view source,
                                           std::span<const std::byte> binary,
                                           std::string_view name) noexcept;
    // Does not execute/replay work. A missing choice preserves the caller's
    // pre-admission default. Explicit profile serialization belongs to the owner.
    [[nodiscard]] CudaProfileChoice choose(const CudaProfileFunction& function,
                                           CudaProfileGeometry geometry, unsigned int max_workers,
                                           bool can_transform) noexcept;
    [[nodiscard]] CudaProfileChoice choose(
        const CudaProfileFunction& function, CudaProfileGeometry geometry, unsigned int max_workers,
        std::span<const transform::PtxExecutionKind> eligible_kinds,
        bool requires_preemption = false) noexcept;
    // Read-only resource-constrained selection from one complete trained superset.
    // A miss means original, never a configured private default. Does not observe,
    // learn or replay work; only bounded hit/LRU bookkeeping may change.
    [[nodiscard]] CudaProfileChoice choose_sharing(
        const CudaProfileFunction& function, CudaProfileGeometry geometry,
        CudaProfileSharingLimits limits,
        std::span<const CudaProfileSharingCandidate> eligible_candidates) noexcept;
    // CPU-only readonly view from one complete inactive trained case. Does not
    // learn, create cases, update LRU or invoke a clock. A miss stays a miss.
    [[nodiscard]] std::optional<CudaProfileOriginalMeasurement> original_measurement(
        const CudaProfileFunction& function, CudaProfileGeometry geometry) const noexcept;
    // Archive resolution only finds existing fully qualified identities; it
    // never binds new code or mutates the single-cache catalog on a miss.
    [[nodiscard]] std::optional<CudaProfileOriginalMeasurement> original_measurement(
        const CudaProfileCodeIdentity& identity, CudaProfileGeometry geometry) const noexcept;
    [[nodiscard]] std::span<const transform::PtxExecutionKind> kinds() const noexcept;
    [[nodiscard]] bool is_learning() const noexcept;
    [[nodiscard]] CudaProfileSnapshot snapshot() const noexcept;
    // CPU-only codec. V1 preserves legacy initial-only records byte-for-byte;
    // V2 includes canonical context configuration; V3 adds complete function
    // configuration on every geometry. Restore
    // is all-or-none into an empty cache. Complete,
    // inactive candidate sets only. Code bytes are identity keys, never loaded
    // from an archive into the SDK; no generated artifacts or native handles.
    [[nodiscard]] std::optional<std::vector<std::byte>> encode(
        std::string_view runtime_identity) const noexcept;
    [[nodiscard]] bool restore(std::span<const std::byte> archive,
                               std::string_view runtime_identity) noexcept;

   private:
    struct Implementation;
    std::shared_ptr<Implementation> implementation_;
};

}  // namespace glimmer::backend::cuda
