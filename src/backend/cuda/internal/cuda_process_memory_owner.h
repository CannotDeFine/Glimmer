#pragma once

#include "cuda_native_output_cell.h"
#include "glimmer/core/quota_ledger.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace glimmer::backend::cuda {

// Trusted backend factory only. Success must prove the exact physical command
// tail, not receipt admission, logical Close or destruction of a CPU handle.
class CudaProcessMemoryCompletion {
   public:
    virtual ~CudaProcessMemoryCompletion() = default;
    [[nodiscard]] virtual CUresult observe() noexcept = 0;
};

struct CudaProcessMemoryConfig {
    std::uint64_t identity{};
    int device_ordinal{-1};
    std::size_t maximum_allocations{256}, maximum_pools{64}, maximum_borrows{4096};
};
struct CudaProcessMemoryResult {
    CUresult status{CUDA_ERROR_INVALID_VALUE};
    std::optional<CUresult> original_status{};
    std::uint64_t id{};
    CUdeviceptr address{};
};
struct CudaProcessMemoryObservation {
    CUresult original_status{CUDA_ERROR_UNKNOWN};
    CUresult owner_status{CUDA_ERROR_UNKNOWN};
    CudaNativeOutputObservation output{};
    CUdeviceptr address{};
    std::shared_ptr<CudaProcessMemoryCompletion> tail;
};
enum class CudaProcessMemoryFreeKind : std::uint8_t { kSynchronous, kAsynchronous };
enum class CudaProcessPoolQuery : std::uint8_t { kDefault, kCurrent };
struct CudaProcessPoolObservation {
    CUresult original_status{CUDA_ERROR_UNKNOWN};
    CUresult owner_status{CUDA_ERROR_UNKNOWN};
    CudaNativeOutputObservation output{};
    CUmemoryPool native{};
};
struct CudaProcessPoolView {
    CUresult status{CUDA_ERROR_INVALID_HANDLE};
    CUmemoryPool native{};
};
struct CudaProcessMemoryBorrow {
    CUresult status{CUDA_ERROR_INVALID_HANDLE};
    CUdeviceptr address{};
    std::size_t bytes{};
    std::shared_ptr<const void> lifetime;
};
struct CudaProcessMemoryView {
    std::uint64_t id{};
    CUdeviceptr address{};
    std::size_t bytes{};
};
struct CudaProcessMemoryLeases {
    CUresult status{CUDA_ERROR_INVALID_CONTEXT};
    std::shared_ptr<const void> lifetime;
};
struct CudaProcessMemorySnapshot {
    CUresult status{CUDA_ERROR_INVALID_CONTEXT};
    std::size_t allocations{}, pools{}, borrowed{};
    core::MemoryBytes retained_bytes{};
    std::uint64_t default_pool{}, current_pool{};
    core::QuotaUsage quota{};
    bool is_closed{}, is_quarantined{}, is_shutdown{}, has_native_call{};
};

// Private Backend process/device registry, never a transport or application
// capability. Caller supplies original SDK observations only after actual host
// return and post-call ownership proof. This class makes no SDK calls itself,
// except trusted tail observers supplied by the original backend factory.
//
// All methods/lifecycle are constructing-owner-thread/PID only. Borrow lifetimes
// may be copied/dropped elsewhere; dropping one only updates its atomic count.
// Caller retains the owner through every method/callback. Unknown effects retain
// quota/state even if the wrapper is dropped. Context Close is not owner Close.
class CudaProcessMemoryOwner final {
   public:
    CudaProcessMemoryOwner(CudaProcessMemoryConfig config,
                           std::shared_ptr<core::QuotaLedger> quota);
    ~CudaProcessMemoryOwner();
    CudaProcessMemoryOwner(const CudaProcessMemoryOwner&) = delete;
    CudaProcessMemoryOwner& operator=(const CudaProcessMemoryOwner&) = delete;

    [[nodiscard]] CudaProcessMemoryResult publish_pool(
        CudaProcessPoolQuery query, const CudaProcessPoolObservation& observation) noexcept;
    [[nodiscard]] CudaProcessPoolView pool(std::uint64_t id) const noexcept;
    // Original native setter already ran under process-wide native exclusion.
    // Failure may have unknown selection effects and quarantines this owner.
    [[nodiscard]] CudaProcessMemoryResult publish_selected_pool(std::uint64_t id,
                                                                CUresult original_status,
                                                                CUresult owner_status) noexcept;

    [[nodiscard]] CudaProcessMemoryResult reserve(std::size_t bytes, std::uint64_t pool) noexcept;
    [[nodiscard]] CUresult cancel(std::uint64_t id) noexcept;
    // Commit accounting before a possible SDK effect. Only matching publication
    // may run while this original host call is outstanding. IDs are not recycled.
    [[nodiscard]] CUresult begin_allocation(std::uint64_t id) noexcept;
    [[nodiscard]] CudaProcessMemoryResult publish_allocation(
        std::uint64_t id, CudaProcessMemoryObservation observation) noexcept;
    [[nodiscard]] CUresult allocation_ready(std::uint64_t id) noexcept;
    // Borrow pins ownership, not readiness. Same-stream FIFO and explicit event
    // dependencies govern enqueue; do not wait for the allocator's physical tail
    // before issuing a consumer. Hold the pin through actual consumer GPU stop.
    [[nodiscard]] CudaProcessMemoryBorrow borrow(std::uint64_t id, std::size_t offset,
                                                 std::size_t bytes) noexcept;
    [[nodiscard]] std::optional<CudaProcessMemoryView> allocation(std::uint64_t id) const noexcept;
    [[nodiscard]] std::vector<CudaProcessMemoryView> live_allocations() const;
    [[nodiscard]] CudaProcessMemoryLeases borrow_live() noexcept;
    // The same process budget also charges classic/context-owned storage.
    [[nodiscard]] CUresult reserve_context_bytes(std::size_t bytes) noexcept;
    [[nodiscard]] CUresult release_context_bytes(std::size_t bytes) noexcept;
    [[nodiscard]] CUresult quarantine(CUresult failure) noexcept;
    [[nodiscard]] std::optional<std::uint64_t> pool_identity(CUmemoryPool native) const noexcept;

    [[nodiscard]] CudaProcessMemoryResult begin_release(std::uint64_t id,
                                                        CudaProcessMemoryFreeKind kind) noexcept;
    [[nodiscard]] CudaProcessMemoryResult publish_release(
        std::uint64_t id, CUresult original_status, CUresult owner_status,
        std::shared_ptr<CudaProcessMemoryCompletion> tail = {}) noexcept;
    [[nodiscard]] CUresult release_ready(std::uint64_t id) noexcept;
    // Owner loop must progress source tails even without new client work, so a
    // retired source context is not pinned forever by a live process allocation.
    // Success here is not memory release, stream completion or shutdown proof.
    [[nodiscard]] CUresult progress() noexcept;
    [[nodiscard]] CudaProcessMemorySnapshot snapshot() const noexcept;
    [[nodiscard]] CUresult close() noexcept;
    [[nodiscard]] CUresult shutdown() noexcept;

   private:
    struct Implementation;
    std::unique_ptr<Implementation> implementation_;
};

}  // namespace glimmer::backend::cuda
