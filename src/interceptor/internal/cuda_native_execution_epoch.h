#pragma once

#include "native_execution_epoch.h"
#include "glimmer/control/async_cooperative_launch_gate.h"

#include <cuda.h>

#include <cstdint>
#include <thread>
#include <functional>

namespace glimmer::interceptor {

// Native SDK pointers only, never public wrappers. The monitor owns context
// activation; original application calls remain on their actual caller.
struct CudaNativeEpochDispatch {
    decltype(&cuCtxGetCurrent) context_get_current{};
    decltype(&cuCtxPushCurrent) context_push{};
    decltype(&cuCtxPopCurrent) context_pop{};
    decltype(&cuStreamGetCtx) stream_get_context{};
    decltype(&cuStreamIsCapturing) stream_is_capturing{};
    decltype(&cuEventCreate) event_create{};
    decltype(&cuEventRecord) event_record{};
    decltype(&cuEventQuery) event_query{};
    decltype(&cuEventDestroy) event_destroy{};
    bool (*enter_native_scope)() noexcept {};
    void (*leave_native_scope)(bool) noexcept {};
    // Optional whole-context recovery proof; never a public wrapper.
    decltype(&cuCtxSynchronize) context_synchronize{};
};

enum class CudaNativeEpochPhase : std::uint8_t {
    kPreparing,
    kPending,
    kReady,
    kStopping,
    kReporting,
    kComplete,
    kRejected,
    kRetained
};
struct CudaNativeEpochOptions {
    std::chrono::microseconds poll_interval{50};
    std::chrono::microseconds control_interval{250};
    std::chrono::milliseconds shutdown_grace{100};
    // Optional trusted CPU-only construction/fault adapter. It must return the
    // exact supplied scope/authorization or throw; no CUDA/control/reentry.
    // Called outside locks once, never for original issue or physical proof.
    std::unique_ptr<NativeExecutionEpoch> (*create_execution)(NativeEpochScope,
                                                              NativeEpochAuthorization,
                                                              NativeExecutionEpoch::TimePoint){};
};
struct CudaNativeEpochSnapshot {
    CudaNativeEpochPhase phase{};
    CUresult status{CUDA_SUCCESS};
    control::NativeEpochGrant grant{};
    std::optional<NativeEpochSnapshot> execution{};
    std::uint32_t recorded_prefixes{};
    std::uint32_t proven_prefixes{};
    std::uint32_t created_events{};
    std::uint32_t destroyed_events{};
};

// Private trusted adapter, not an application API. The constructing caller and
// exact context/canonical-stream registrations must remain alive and uncaptured.
// Predecessor work must already have independent admission/queue ownership.
// No borrowed arguments survive begin/returned. Only monitor polls/records GPU
// proof for ordinary streams. PTDS uses caller-created predecessor and a caller
// trailing record on every original return; its authorized prefix width is one.
// The monitor only queries those PTDS events. Thread-safe close/snapshot;
// shutdown/destruction externally serialized
// with caller methods. Failed owners must remain quarantined through teardown.
class CudaNativeExecutionEpoch final {
   public:
    CudaNativeExecutionEpoch(CUcontext context, CUstream stream, NativeEpochScope scope,
                             control::NativeEpochLaunchRequest request,
                             std::shared_ptr<control::AsyncCooperativeLaunchGate> gate,
                             CudaNativeEpochDispatch dispatch, CudaNativeEpochOptions options = {});
    ~CudaNativeExecutionEpoch();
    CudaNativeExecutionEpoch(const CudaNativeExecutionEpoch&) = delete;
    CudaNativeExecutionEpoch& operator=(const CudaNativeExecutionEpoch&) = delete;

    // On the original caller only. A returned capability authorizes one actual
    // original entry; the caller must publish return even after concurrent close.
    [[nodiscard]] std::optional<NativeEpochIssue> begin_call() noexcept;
    [[nodiscard]] bool original_returned(const NativeEpochIssue& issue,
                                         NativeEpochCallOutcome outcome) noexcept;
    void close() noexcept;
    void shutdown() noexcept;
    // Explicit private recovery, externally serialized with caller/shutdown.
    // Joins a retained monitor, proves the exact context with the original SDK,
    // then submits one failed terminal. No failed query/report/cleanup replay.
    // Success retires physical ownership, not the original error or frontend.
    // A supplied trusted whole-context original runs once on the actual caller,
    // with the owned context current before/after return and no context push/pop.
    [[nodiscard]] CUresult recover_context_completion(
        NativeEpochScope scope, const std::function<CUresult()>& context_completion = {}) noexcept;
    [[nodiscard]] CudaNativeEpochSnapshot snapshot() const noexcept;

   private:
    struct Implementation;
    std::unique_ptr<Implementation> implementation_;
};

}  // namespace glimmer::interceptor
