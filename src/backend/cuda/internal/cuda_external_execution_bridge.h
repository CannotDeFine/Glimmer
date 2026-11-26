#pragma once

#include "cuda_execution_dispatcher.h"

#include <memory>

namespace glimmer::backend::cuda {

class CudaSessionExternalCommand;
class CudaExecutionSession;
struct CudaExternalExecutionBridgeState;

struct CudaExternalExecutionBridgeDispatch {
    decltype(&cuCtxGetCurrent) context_get_current{};
    decltype(&cuEventCreate) event_create{};
    decltype(&cuEventRecord) event_record{};
    decltype(&cuEventDestroy) event_destroy{};
    decltype(&cuStreamWaitEvent) stream_wait_event{};
};
struct CudaExternalExecutionDomain {
    CudaExecutionContext context;
    int device_ordinal{-1};
    CUstream stream{};
    std::shared_ptr<const void> stream_lifetime;
};

// Copy/destruction is CPU-only. This endpoint contains immutable native resource
// owners and bridge CPU state, never a mutable session/dispatcher or frontier.
// First begin binds one nonrecycled native worker incarnation, distinct from
// the source owner (whose session exclusion must not be bypassed). Native context
// selection remains the trusted lane's responsibility, not this helper's action.
class CudaExternalExecutionWorker final {
   public:
    CudaExternalExecutionWorker() = default;
    [[nodiscard]] CUresult begin() noexcept;
    // TRUSTED: actual vendor/group asynchronous enqueue is complete. No native
    // result or timeout implies this proof. Records only the execution-stream
    // event; it does not publish a source-stream enqueue or physical completion.
    [[nodiscard]] CUresult record_enqueue() noexcept;
    [[nodiscard]] CUresult cleanup() noexcept;

   private:
    friend class CudaExternalExecutionBridge;
    std::shared_ptr<CudaExternalExecutionBridgeState> state_;
};

// Private source-owner helper. Trusted composition authenticates that source and
// tail belong to the SAME accepted external boundary, begins its barrier only
// after logical readiness, and retains every participant through uncertainty.
// No requested native NCCL call, grouping, scheduling or host reply occurs here.
// All native operations are nonblocking Driver event operations, owner-only and
// reentry-excluded. Destruction never calls SDK or invents physical stop.
class CudaExternalExecutionBridge final {
   public:
    CudaExternalExecutionBridge(CudaSessionExternalCommand source, CudaExecutionFrontier tail,
                                CudaExternalExecutionDomain execution,
                                CudaExternalExecutionBridgeDispatch dispatch);
    ~CudaExternalExecutionBridge();
    CudaExternalExecutionBridge(const CudaExternalExecutionBridge&) = delete;
    CudaExternalExecutionBridge& operator=(const CudaExternalExecutionBridge&) = delete;

    [[nodiscard]] CudaExternalExecutionWorker worker() const noexcept;
    [[nodiscard]] bool completed() const noexcept;
    [[nodiscard]] CUresult failure() const noexcept;

   private:
    friend class CudaExecutionSession;
    // Session entry also excludes every other mutable session/dispatcher path.
    [[nodiscard]] bool matches_source(CUcontext context, std::uint64_t id) const noexcept;
    [[nodiscard]] CUresult prepare_source() noexcept;
    // Waits source stream on the actual recorded execution event. Only successful
    // join authorizes the separate dispatcher record_external_enqueue transition.
    [[nodiscard]] CUresult join_source() noexcept;
    // Requires this exact immutable tail's physical SUCCESS first. Worker event
    // cleanup follows separately on its original execution-context worker.
    [[nodiscard]] CUresult cleanup_source() noexcept;
    struct Implementation;
    std::unique_ptr<Implementation> implementation_;
};

}  // namespace glimmer::backend::cuda
