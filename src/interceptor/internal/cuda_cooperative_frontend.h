#pragma once

#include "cuda_admission_coordinator.h"
#include "cuda_cooperative_pool.h"
#include "cuda_native_execution_epoch.h"

#include <functional>
#include <memory>
#include <span>

namespace glimmer::interceptor {
struct CudaQueuedReleaseAdmission {
    CUresult status{CUDA_ERROR_UNKNOWN};
    // True also covers uncertain mailbox return after submission started.
    // Only false permits cancellation of the caller's release ticket.
    bool has_owned_release{};
};

struct CudaFrontendDispatch {
    CudaQueueWorkerDispatch worker;
    CudaCooperativePoolDispatch pool;
    decltype(&cuStreamCreate) stream_create{};
    decltype(&cuStreamDestroy) stream_destroy{};
    decltype(&cuStreamGetFlags) stream_get_flags{};
    decltype(&cuStreamGetPriority) stream_get_priority{};
};

struct CudaFrontendOptions {
    std::size_t progress_slots{1024};
    std::size_t audit_block_capacity{};
    std::size_t import_events{256};
    std::size_t per_thread_streams{32};
    // Private physical-completion scans; control retry deadlines are separate.
    CudaQueueWorkerOptions worker{.poll_interval = std::chrono::microseconds{25}};
    bool should_trace_native_timings{};
    // Internal only, default off. The shared gate must independently approve
    // these bounds. No public selector enables this policy yet.
    std::optional<control::NativeEpochExtent> native_epoch_policy{};
    CudaNativeEpochDispatch native_epoch_dispatch{};
    CudaNativeEpochOptions native_epoch_monitor{};
};

// Source event ownership classification on the queue actor, outside queue
// locks. Preallocated CPU metadata only: no allocation, CUDA/control, waiting
// or frontend reentry. Logical events must belong to this frontend's context.
class CudaFrontendNativeWaitGuard {
   public:
    virtual ~CudaFrontendNativeWaitGuard() = default;
    virtual CUresult should_wait(CUevent event, bool& output) noexcept = 0;
};

// Opaque CPU-only historical prefix plus exact frontend/event identity.
// Prepared empty (unrecorded) prefixes are valid; submission checks live identity.
class CudaFrontendEventWait final {
   public:
    [[nodiscard]] explicit operator bool() const noexcept {
        return owner_ && event_;
    }

   private:
    friend class CudaCooperativeFrontend;
    std::shared_ptr<void> owner_;
    CudaQueueEventToken event_;
    CudaOperationToken prefix_;
};

// One context owner. Metadata methods are thread-safe after initialization;
// initialize/recovery/shutdown/destruction are externally serialized on the context
// owner. Native dispatch is unwrapped. Resource destructors never call CUDA.
// Supplied resource owners must likewise have metadata-only destructors; callers
// retain original native handles/buffers until successful logical close/drain.
// This is an internal frontend, not public-wrapper activation.
class CudaCooperativeFrontend final {
   public:
    using NativeOperation = std::function<CUresult()>;
    CudaCooperativeFrontend(CUcontext context, CudaFrontendDispatch dispatch,
                            std::shared_ptr<control::AsyncCooperativeLaunchGate> gate,
                            CudaFrontendOptions options = {});
    ~CudaCooperativeFrontend();
    CudaCooperativeFrontend(const CudaCooperativeFrontend&) = delete;
    CudaCooperativeFrontend& operator=(const CudaCooperativeFrontend&) = delete;

    [[nodiscard]] CUresult initialize() noexcept;
    [[nodiscard]] CUresult launch(const CooperativeKernelVariant& variant, CUstream stream,
                                  bool per_thread_default_stream, std::array<unsigned int, 3> grid,
                                  std::array<unsigned int, 3> block, void* const* arguments,
                                  std::shared_ptr<void> kernel_resources) noexcept;
    [[nodiscard]] CUresult copy(CUstream stream, bool per_thread_default_stream,
                                CudaQueuedCopy operation) noexcept;
    // Trusted already-qualified private ABI with owned arguments/resources.
    // Logical XYZ volume must fit the uint32 admission work-unit field; the
    // continuation, not this frontend, selects each physical launch geometry.
    // No original fallback is allowed after this call accepts queue ownership.
    [[nodiscard]] CUresult launch_sliced(CUcontext context, CUstream stream,
                                         bool per_thread_default_stream,
                                         CudaQueuedKernel kernel) noexcept;
    [[nodiscard]] CudaQueuedReleaseAdmission release(CUstream stream,
                                                     bool per_thread_default_stream,
                                                     CudaQueuedFree operation) noexcept;
    [[nodiscard]] CUresult bind_event(CUevent event, std::shared_ptr<void> resources) noexcept;
    [[nodiscard]] CUresult record_event(
        CUevent event, CUstream stream, bool per_thread_default_stream,
        std::shared_ptr<CudaQueueEventRecordGuard> guard = {}) noexcept;
    // Original-caller contribution across SDK event metadata and queued acceptance.
    // No locks or context mutation cover the callback. It may publish a record/wait,
    // but must not wait for GPU progress or enter a native boundary. SDK failure
    // accepts nothing; after-effect CPU failure must fail_native_boundary.
    // Same binding's Runtime roots serialize their SDK effects, without holding
    // a metadata mutex across the callback or moving it to the actor.
    [[nodiscard]] CUresult admit_event_metadata(CUevent event,
                                                const NativeOperation& operation) noexcept;
    [[nodiscard]] CUresult prepare_event_wait(CUevent event,
                                              CudaFrontendEventWait& output) noexcept;
    [[nodiscard]] CUresult wait_event_snapshot(CUstream stream, bool per_thread_default_stream,
                                               const CudaFrontendEventWait& snapshot) noexcept;
    [[nodiscard]] CUresult wait_event(CUstream stream, bool per_thread_default_stream,
                                      CUevent event) noexcept;
    [[nodiscard]] CUresult query_event(CUevent event) noexcept;
    [[nodiscard]] CUresult synchronize_event(CUevent event) noexcept;
    [[nodiscard]] CUresult query_stream(CUstream stream, bool per_thread_default_stream) noexcept;
    // Existing/implicit completion snapshots, never a stream/fence import.
    // Missing explicit bindings use checked original flags on the actor.
    [[nodiscard]] CUresult complete_stream_snapshot(CUstream stream, bool per_thread_default_stream,
                                                    bool should_wait,
                                                    bool& has_logical_prefix) noexcept;
    [[nodiscard]] CUresult synchronize_stream(CUstream stream,
                                              bool per_thread_default_stream) noexcept;
    [[nodiscard]] CUresult synchronize_context() noexcept;
    // Runs once on this caller under an empty boundary or a granted priority
    // reservation. A failed fence import keeps the frontend paused/owned.
    // A preallocated observer follows the actual imported prefix, not API return.
    // Waited event snapshots are dependencies BEFORE control admission/pause.
    // No guard means every waited handle is a context-local logical event;
    // guarded native events are left to the original operation. The caller
    // keeps the span alive through return. Non-execution boundaries reject waits.
    [[nodiscard]] CUresult native(
        CUstream stream, bool per_thread_default_stream, const NativeOperation& operation,
        bool is_execution, std::shared_ptr<CudaQueueNativeFenceObserver> completion_observer = {},
        std::span<const CUevent> waited_events = {},
        std::shared_ptr<CudaFrontendNativeWaitGuard> wait_guard = {}) noexcept;
    // Trusted original conventional launch only, uncaptured canonical stream.
    // PTDS proofs stay on the actual caller and use authorized width one.
    // Same caller lifetime/scope may join one bounded epoch.
    // No frontend reentry or borrowed data after return; no fallback after entry.
    [[nodiscard]] CUresult native_epoch(CUstream stream, const NativeOperation& original,
                                        bool per_thread_default_stream = false) noexcept;
    [[nodiscard]] std::optional<CudaNativeEpochSnapshot> native_epoch_snapshot() const noexcept;
    // CPU request/join only; the physical owner activates its own context.
    // Process scope switches can close another context without changing caller TLS.
    [[nodiscard]] CUresult close_native_epoch(bool should_wait = true) noexcept;
    // Trusted, externally serialized recovery of this exact joined epoch only.
    // A separate unknown boundary permanently invalidates recovery provenance.
    // Original SDK whole-context proof/cleanup/failed acknowledgment precede
    // one logical marker and healthy thaw; the old owner's error stays intact.
    [[nodiscard]] CUresult recover_native_epoch(
        const control::NativeEpochGrant& expected,
        const NativeOperation& context_completion = {}) noexcept;
    // Same external serialization contract; healthy state alone may decline.
    // Derives the exact old grant from the live joined owner, never a new grant.
    [[nodiscard]] std::optional<CUresult> recover_native_epoch(
        const NativeOperation& context_completion) noexcept;
    // Long process-native/capture boundary; all commands remain excluded until
    // an actual native context fence can be imported using this exact token.
    [[nodiscard]] CUresult pause(CudaNativeBoundaryToken& boundary) noexcept;
    // CPU-only token check, including while ordinary commands are excluded.
    [[nodiscard]] bool native_boundary_is_paused(
        const CudaNativeBoundaryToken& boundary) const noexcept;
    [[nodiscard]] CUresult finish_pause(const CudaNativeBoundaryToken& boundary,
                                        const NativeOperation& prove_native_context) noexcept;
    // Retain physical ownership but wake blocked callers after an uncertain
    // process-native boundary. No native cleanup or automatic replay follows.
    void fail_native_boundary(CUresult error) noexcept;
    // Closing is a synchronous, checked boundary; no native destruction is
    // performed here. The wrapper may destroy the original handle on success.
    [[nodiscard]] CUresult close_event(CUevent event) noexcept;
    // Exact drained process pause only (Source or normal handle destruction).
    // Validation precedes original destroy;
    // retirement follows success. Actor CPU metadata only, without thaw.
    [[nodiscard]] CUresult retire_paused_event(const CudaNativeBoundaryToken& boundary,
                                               CUevent event, bool should_retire) noexcept;
    [[nodiscard]] CUresult close_stream(CUstream stream) noexcept;
    // Exact drained process pause only, same CPU-only contract as event
    // retirement. Preparation leaves the association available on SDK failure.
    [[nodiscard]] CUresult retire_paused_stream(const CudaNativeBoundaryToken& boundary,
                                                CUstream stream, bool should_retire) noexcept;
    [[nodiscard]] CUresult shutdown() noexcept;
    // Trusted lifecycle owner only, externally serialized with native teardown.
    // Success drains and joins without retiring stream/event/code resources.
    [[nodiscard]] CUresult checkpoint(CudaWorkerCheckpoint& output) noexcept;
    // Requires independent proof that every original context/resource survived.
    [[nodiscard]] CUresult resume_checkpoint(const CudaWorkerCheckpoint& checkpoint) noexcept;
    // Trusted, externally serialized lifecycle only. Release the empty private
    // pool in its live target before a possibly destructive original operation.
    [[nodiscard]] CUresult release_checkpoint_progress(
        const CudaWorkerCheckpoint& checkpoint) noexcept;
    // Independent original-resource survival proof and target binding required.
    [[nodiscard]] CUresult restore_checkpoint_progress(
        const CudaWorkerCheckpoint& checkpoint) noexcept;
    // Actual original destruction proof required. Pool must already be freed;
    // retire exact joined CPU ownership without touching destroyed CUDA handles.
    [[nodiscard]] CUresult retire_checkpoint_after_context_destroy(
        const CudaWorkerCheckpoint& checkpoint) noexcept;
    [[nodiscard]] CudaCoordinationStats stats() noexcept;

   private:
    struct Implementation;
    std::unique_ptr<Implementation> implementation_;
};

}  // namespace glimmer::interceptor
