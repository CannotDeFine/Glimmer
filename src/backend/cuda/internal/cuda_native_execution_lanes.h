#pragma once

#include "cuda_execution_dispatcher.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace glimmer::backend::cuda {

struct CudaNativeExecutionLaneKey {
    std::uint64_t peer{};
    std::array<std::uint64_t, 2> thread{};
    bool operator==(const CudaNativeExecutionLaneKey&) const = default;
};
struct CudaNativeExecutionLaneLimits {
    std::size_t maximum_lanes{16};
    std::size_t maximum_jobs{256};
    std::size_t maximum_jobs_per_lane{32};
    std::size_t maximum_devices{64};
};
struct CudaNativeExecutionLaneDispatch {
    decltype(&cuCtxSetCurrent) context_set_current{};
    decltype(&cuCtxGetCurrent) context_get_current{};
};
struct CudaNativeExecutionCall {
    std::uint64_t sequence{};
    int device_ordinal{-1};
    CudaExecutionContext context;
    std::shared_ptr<const void> arguments_lifetime;
    // Trusted backend code only. Return carrier/proof status, NOT a native SDK
    // status cast to CUresult. Preserve the requested native status separately.
    std::function<CUresult()> invoke;
    // Empty preserves singleton admission. Otherwise a bounded unique set of
    // nonnegative ordinals INCLUDING device_ordinal. The trusted argument owner
    // also retains every participating domain. Acquire all callback gates or
    // none, before any SDK call; context selection/proof still uses context.
    std::vector<int> device_ordinals{};
};
enum class CudaNativeExecutionCallPhase : std::uint8_t {
    kPending,
    // Device admission obtained; was_invoked remains false during context proof.
    kEntered,
    kReturned,
    kCancelled,
    kUncertain,
};
struct CudaNativeExecutionCallObservation {
    CudaNativeExecutionCallPhase phase{CudaNativeExecutionCallPhase::kPending};
    CUresult status{CUDA_ERROR_NOT_READY};
    bool was_invoked{};
};
struct CudaNativeExecutionCallState;

// CPU-only observation/copy/destruction. Neither host return nor release is GPU
// completion or permission to destroy a communicator/configuration or any source
// resource still used by native asynchronous/group work.
// Empty observation is Cancelled/INVALID_VALUE/uninvoked; forked observation is
// Uncertain/INVALID_CONTEXT/uninvoked before accessing inherited synchronization.
class CudaNativeExecutionTicket final {
   public:
    CudaNativeExecutionTicket() = default;
    [[nodiscard]] CudaNativeExecutionCallObservation observe() const noexcept;

   private:
    friend class CudaNativeExecutionLanes;
    std::shared_ptr<CudaNativeExecutionCallState> state_;
};
struct CudaNativeExecutionAcceptance {
    CUresult status{CUDA_ERROR_INVALID_VALUE};
    CudaNativeExecutionTicket ticket;
};

// One constructing thread/process owns admission, release, stop and shutdown.
// It authenticates keys and retains the single hub for ALL native calls on its
// devices. Workers receive immutable leases, never sessions/frontiers. Native
// callbacks on one device cannot overlap, while independent devices may progress.
// Caller/cohort admission MUST additionally exclude unsafe open-group/async use;
// callback serialization alone does not prove NCCL thread safety or scheduling.
// Multi-device calls reserve the complete set atomically, never a partial subset.
// An entered failure quarantines its complete set. Cancelling an unentered
// intersecting superset cannot quarantine that superset's other healthy devices.
// First proof/callable failure quarantines that device; other devices progress.
// Queued/future admitted jobs for it cancel without SDK entry and remain releasable.
// First committed carrier failure is sticky, including across post-call proof.
// Malformed admission/tickets return INVALID_VALUE, quotas OUT_OF_MEMORY, foreign
// owner/process INVALID_CONTEXT and admission after stop DEINITIALIZED. Constructor
// rejects missing dispatch/zero limits with std::invalid_argument. Sequences begin
// at1 and increase exactly once for each accepted job; all retained jobs use quota.
class CudaNativeExecutionLanes final {
   public:
    explicit CudaNativeExecutionLanes(CudaNativeExecutionLaneDispatch dispatch,
                                      CudaNativeExecutionLaneLimits limits = {});
    ~CudaNativeExecutionLanes();
    CudaNativeExecutionLanes(const CudaNativeExecutionLanes&) = delete;
    CudaNativeExecutionLanes& operator=(const CudaNativeExecutionLanes&) = delete;

    [[nodiscard]] CudaNativeExecutionAcceptance submit(CudaNativeExecutionLaneKey key,
                                                       CudaNativeExecutionCall call) noexcept;
    // Explicit trusted HOST storage release after a successful returned call or
    // known unentered cancellation. Caller keeps separate durable async owners.
    // Unknown callbacks/proofs cannot release; duplicate/foreign tickets reject.
    [[nodiscard]] CUresult release(const CudaNativeExecutionTicket& ticket) noexcept;
    [[nodiscard]] CUresult request_stop() noexcept;
    // Nonblocking: NOT_READY while any callback/worker is live. Join only workers
    // already independently observed exited. Unreleased jobs remain retained.
    [[nodiscard]] CUresult shutdown() noexcept;
    [[nodiscard]] std::size_t retained_jobs() const noexcept;

   private:
    struct Implementation;
    std::unique_ptr<Implementation> implementation_;
};

}  // namespace glimmer::backend::cuda
