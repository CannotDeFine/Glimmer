#pragma once

#include "cuda_cooperative_frontend.h"
#include "cuda_copy_range.h"
#include "driver_dispatch.h"
#include "cuda_process_native_session.h"
#include "cuda_paused_native_executor.h"
#include "cuda_event_projection_registry.h"
#include "cuda_native_graph_registry.h"
#include "process_call_gate.h"
#include "cuda_source_capture.h"

#include <thread>

namespace glimmer::interceptor {

struct CudaContextDestroyDisposition {
    bool was_handled{true};
    bool has_uncertain_lifetime{};
};
using CudaPrimaryDestroyDisposition = CudaContextDestroyDisposition;

// Trusted physical-only qualification adapters; never queue/original dispatch.
// Their native function pointers must outlive the process owner.
struct CudaNativeEpochProjectionOverrides {
    decltype(&cuEventQuery) event_query{};
    decltype(&cuEventDestroy) event_destroy{};
    decltype(&cuCtxSynchronize) context_synchronize{};
};
struct CudaNativeEpochConfiguration {
    control::NativeEpochExtent extent;
    CudaNativeEpochProjectionOverrides projections;
};

// Process-owned dispatch/handle registry. Admission policy remains in control.
// Calls are thread-safe; destructive application handle use follows CUDA's
// ordinary external serialization contract. No native calls occur under locks.
class DriverCooperativeRouter final {
   public:
    using NativeOperation = CudaCooperativeFrontend::NativeOperation;
    using NativeIssueScope = std::function<CUresult(const NativeOperation&)>;
    // Borrowed only during one root callback. No Runtime handle conversion or
    // nested transition: the actual Driver callback supplies its typed origin.
    class SourceCaptureDispatch final {
       public:
        SourceCaptureDispatch(const SourceCaptureDispatch&) = delete;
        SourceCaptureDispatch& operator=(const SourceCaptureDispatch&) = delete;
        [[nodiscard]] CUresult invoke(CUcontext context, CUstream stream, bool is_ptds,
                                      CudaSourceCaptureKind kind,
                                      const NativeOperation& original) noexcept;
        [[nodiscard]] CUresult lookup_origin_context(CUstream stream, bool is_ptds,
                                                     CUcontext& output) noexcept;

       private:
        friend class DriverCooperativeRouter;
        SourceCaptureDispatch(DriverCooperativeRouter& router, ProcessCallGate& gate,
                              CudaSourceCaptureKind kind) noexcept;
        DriverCooperativeRouter& router_;
        ProcessCallGate& gate_;
        const CudaSourceCaptureKind root_kind;
        const std::thread::id caller;
        bool has_dispatched_{};
        bool has_origin_lookup_{};
        bool has_rejected_{};
        CUresult native_status_{CUDA_ERROR_UNKNOWN};
    };
    using RootCaptureOperation = std::function<CUresult(SourceCaptureDispatch&)>;
    DriverCooperativeRouter(const DriverDispatch& driver, control::LaunchGateOptions options,
                            bool should_trace, bool should_trace_native_timings = false,
                            std::optional<control::NativeEpochExtent> native_epoch_policy = {},
                            CudaNativeEpochProjectionOverrides projections = {},
                            std::size_t audit_block_capacity = 0,
                            bool should_owner_pump_control = false);
    ~DriverCooperativeRouter();
    DriverCooperativeRouter(const DriverCooperativeRouter&) = delete;
    DriverCooperativeRouter& operator=(const DriverCooperativeRouter&) = delete;
    // Immutable construction policy, CPU-only; ordinary roots avoid building
    // a borrowed SDK permission when epoch ownership is not enabled.
    [[nodiscard]] bool has_native_epoch_policy() const noexcept;
    [[nodiscard]] CUresult frontend(CUcontext context,
                                    std::shared_ptr<CudaCooperativeFrontend>& output) noexcept;
    // CPU-only lookup; absent/initializing contexts return an empty owner.
    [[nodiscard]] CUresult existing_frontend(
        CUcontext context, std::shared_ptr<CudaCooperativeFrontend>& output) noexcept;
    // Conventional original kernel fallback. Enabled epochs require this gate's
    // normal enclosing Call and an exact existing caller-context frontend.
    // No graph/event effects or opaque Runtime handle conversion. Policy defaults off.
    [[nodiscard]] CUresult native_execution(
        ProcessCallGate& gate, const std::shared_ptr<CudaCooperativeFrontend>& frontend,
        CUstream stream, bool is_ptds, const NativeOperation& original) noexcept;
    // Trusted exact existing target/grant, under the bound normal Call. Owns
    // process transition and private tickets before frontend recovery; unknown
    // process/source effects reject without clearing any quarantine or handle.
    [[nodiscard]] CUresult recover_native_epoch(
        ProcessCallGate& gate, CUcontext context, const control::NativeEpochGrant& expected,
        const NativeOperation& context_completion = {}) noexcept;
    // Context-wide SDK completion only. Disabled/healthy ownership may decline
    // before effects. Transition precedes actual current-context/actor lookup;
    // failed unknown owners reject rather than invoking an unowned original.
    [[nodiscard]] std::optional<CUresult> recover_current_native_epoch(
        ProcessCallGate& gate, const NativeOperation& context_completion) noexcept;
    [[nodiscard]] CUresult native_boundary(const NativeOperation& operation,
                                           const NativeOperation& context_completion) noexcept;
    // Normal Runtime destruction only: same drained process boundary, exact
    // paused CPU retirement after borrowed SDK success, never lazy actor import.
    [[nodiscard]] CUresult native_event_destroy(CUevent event, const NativeOperation& original,
                                                const NativeOperation& context_completion,
                                                const bool* has_native_effect) noexcept;
    [[nodiscard]] CUresult native_stream_destroy(
        CUstream stream, const NativeOperation& original,
        const std::function<CUresult(CUcontext)>& retire_releases,
        const NativeOperation& context_completion, const bool* has_native_effect) noexcept;
    // Session begin/finish require process-call transition exclusion; they must
    // not race frontend initialization, close or each other. While retained,
    // normal router entry points reject instead of waiting on this reservation.
    // Native-context observations serialize through the session metadata lock;
    // finish must first exclude/drain those observers at the process call gate.
    [[nodiscard]] CUresult begin_native_session(CudaProcessNativeToken& token) noexcept;
    [[nodiscard]] CUresult observe_native_session_context(CUcontext context) noexcept;
    // Source scope excludes capture-changing calls during classification/issue,
    // not during control admission waiting. Its body runs exactly once.
    [[nodiscard]] CUresult native_session_execution(
        const CudaProcessNativeToken& token, CUcontext context, CUstream stream, bool is_ptds,
        const NativeOperation& operation, const NativeIssueScope& source_scope,
        ProcessCallGate* process_gate = nullptr,
        const CudaSourceEventOperation* event_operation = nullptr,
        const CudaNativeGraphExecutable* graph_operation = nullptr) noexcept;
    [[nodiscard]] CudaPausedNativeStats native_execution_stats() const noexcept;
    [[nodiscard]] CUresult finish_native_session(const CudaProcessNativeToken& token,
                                                 const NativeOperation& verify_inactive,
                                                 const NativeOperation& context_completion,
                                                 bool should_wait_for_executions = true) noexcept;
    // Internal source owner assembly. Requires an enclosing Call on this gate;
    // original capture callbacks execute once, under its exclusive transition.
    [[nodiscard]] CUresult source_capture_transition(ProcessCallGate& gate, CUcontext context,
                                                     CUstream stream, bool is_ptds,
                                                     CudaSourceCaptureKind kind,
                                                     const NativeOperation& original) noexcept;
    // Pause before root lazy binding, then discover the actual origin through
    // one borrowed dispatch. Missing/misused dispatch retains failed ownership.
    [[nodiscard]] CUresult source_capture_root(ProcessCallGate& gate, CudaSourceCaptureKind kind,
                                               const RootCaptureOperation& original) noexcept;
    [[nodiscard]] CUresult try_finish_source_capture(
        ProcessCallGate& gate, bool should_wait_for_executions = false) noexcept;
    [[nodiscard]] CUresult quarantine_source_call(ProcessCallGate& gate, CUresult error) noexcept;
    // Covered synchronous memory metadata, not execution or stream/event/context
    // destruction. Runtime borrows caller-local issue state through this call;
    // its complete accounting frame precedes post-issue context validation.
    [[nodiscard]] CUresult source_memory_metadata(
        ProcessCallGate& gate, const NativeOperation& original,
        CudaSourceMemoryOrigin origin = CudaSourceMemoryOrigin::kDriver,
        const bool* has_native_issue = nullptr) noexcept;
    // Original Driver module load/lookup/unload and CPU retirement only.
    // Private preparation/cleanup and other handle destruction are not permitted.
    [[nodiscard]] CUresult source_module_metadata(ProcessCallGate& gate,
                                                  const NativeOperation& original) noexcept;
    // Original Driver destroy once, then exact CPU registration/actor retirement.
    // Never drains source execution, initializes a frontend or clears its pause.
    [[nodiscard]] CUresult source_event_destroy(
        ProcessCallGate& gate, CUevent event, const NativeOperation& original,
        CudaSourceMemoryOrigin origin = CudaSourceMemoryOrigin::kDriver,
        const bool* has_native_issue = nullptr) noexcept;
    // Original Runtime creation with bounded, late-context CPU registration.
    // Failed SDK output is never inspected; no private CUDA work or cleanup.
    [[nodiscard]] CUresult source_runtime_event_create(ProcessCallGate& gate, CUevent* event,
                                                       const NativeOperation& original,
                                                       const bool* has_native_issue) noexcept;
    // Normal caller SDK creation, bounded fresh-unrecorded CPU publication.
    // No frontend initialization, context override or GPU boundary.
    [[nodiscard]] CUresult normal_runtime_event_create(CUevent* event,
                                                       const NativeOperation& original) noexcept;
    // Exact source explicit-stream retirement after one original success.
    // The CPU-only release callback detaches saved identity, never completes
    // memory/control quota. No destroyed handle query or implicit stream retire.
    [[nodiscard]] CUresult source_stream_destroy(
        ProcessCallGate& gate, CUstream stream, const NativeOperation& original,
        const std::function<CUresult(CUcontext)>& retire_releases,
        CudaSourceMemoryOrigin origin = CudaSourceMemoryOrigin::kDriver,
        const bool* has_native_issue = nullptr) noexcept;
    [[nodiscard]] CUresult source_runtime_event_record(ProcessCallGate& gate, CUevent event,
                                                       CUstream stream, bool is_ptds,
                                                       unsigned int flags,
                                                       const NativeOperation& original,
                                                       const bool* has_native_issue) noexcept;
    // Read only within an admitted native-only source call on the bound gate.
    [[nodiscard]] CUresult source_native_token(ProcessCallGate& gate,
                                               CudaProcessNativeToken& output) noexcept;
    [[nodiscard]] CUresult close_context(CUcontext context,
                                         const NativeOperation& destroy_native) noexcept;
    // Existing primary actor only: measured SDK rejection cannot close it.
    // Owns transition before native transaction; unknown effects stay retained.
    [[nodiscard]] CUresult try_rejected_primary_destroy(
        ProcessCallGate& gate, CUcontext context, const NativeOperation& original,
        CudaPrimaryDestroyDisposition& disposition) noexcept;
    // Requires enclosing Call. Join before exact SDK effect; original qualified
    // legacy rejection resumes the same owner, success retires only CPU handles.
    [[nodiscard]] CUresult destroy_context(
        ProcessCallGate& gate, CUcontext context, const NativeOperation& original,
        const std::function<CUresult(CUcontext)>& retire_destroyed, bool is_modern,
        CudaContextDestroyDisposition& disposition) noexcept;
    // Requires enclosing Call; exclusive transition spans identity/native effect
    // and retirement. Never imports an actor or guesses application references.
    [[nodiscard]] CUresult release_primary_context(
        ProcessCallGate& gate, CUdevice device, const NativeOperation& original,
        const std::function<CUresult(CUcontext)>& retire_destroyed,
        bool& has_uncertain_lifetime) noexcept;
    [[nodiscard]] CUresult register_event(CUcontext context, CUevent event) noexcept;
    // Actual non-IPC Driver creation, externally serialized against destruction.
    // CPU-only observation while paused; lazy queue binding waits for native proof.
    [[nodiscard]] CUresult register_native_event(const CudaProcessNativeToken& token,
                                                 CUcontext context, CUevent event) noexcept;
    [[nodiscard]] CUresult prepare_native_event(const CudaProcessNativeToken& token,
                                                CUcontext context,
                                                CudaEventRegistrationReservation& output) noexcept;
    [[nodiscard]] CUresult publish_native_event(CudaEventRegistrationReservation& reservation,
                                                CUevent event, CUresult native_status) noexcept;
    [[nodiscard]] CUresult event_projection(CUevent event, CudaEventProjection& output) noexcept;
    [[nodiscard]] CUresult admit_event_metadata(std::span<const CudaEventProjection> owned,
                                                const NativeOperation& operation) noexcept;
    [[nodiscard]] CUresult refresh_event_projection(const CudaEventProjection& owned,
                                                    CudaEventProjection& output) noexcept;
    // CPU-only exact registration/epoch validation after a caller SDK effect.
    [[nodiscard]] CUresult matches_event_epoch(const CudaEventProjection& owned,
                                               bool& output) noexcept;
    // After actual successful normal destruction; no actor binding can exist.
    [[nodiscard]] CUresult retire_metadata_event(const CudaEventProjection& owned) noexcept;
    // Native selection leaves readiness to the actual original event query/sync.
    [[nodiscard]] CUresult logical_event_completion(CUevent event, bool should_wait,
                                                    bool& needs_native_boundary) noexcept;
    // CPU-only sticky failure gate for normal caller-issued metadata effects.
    [[nodiscard]] CUresult normal_owner_status() noexcept;
    void fail_context_owner(CUcontext context, CUresult error) noexcept;
    // CPU-only first-error publication for uncertain normal launch effects.
    // Unknown/cold actors cannot make a missing SDK leaf appear harmless.
    void quarantine_normal_launch(CUcontext context, CUresult error) noexcept;
    [[nodiscard]] CUresult publish_logical_event(const CudaEventProjection& accepted) noexcept;
    // Prepare before accepting a record; output owns CPU metadata only.
    [[nodiscard]] CUresult prepare_event_record_guard(
        const CudaEventProjection& accepted,
        std::shared_ptr<CudaQueueEventRecordGuard>& output) noexcept;
    [[nodiscard]] CUresult invalidate_native_events(CUcontext context = nullptr) noexcept;
    // A null context prepares late-bound SDK metadata only; publish must supply
    // its actual post-original context observation before replay is possible.
    [[nodiscard]] CUresult prepare_graph(CUcontext context, CUgraph graph,
                                         CudaNativeGraphReservation& output,
                                         bool should_prepare_effects) noexcept;
    [[nodiscard]] CudaNativeGraphRegistry& graphs() noexcept;
    [[nodiscard]] CUresult prepare_graph_update(CUgraphExec executable, CUgraph graph,
                                                std::optional<CUgraphNode> child_node,
                                                CudaNativeGraphReservation& output) noexcept;
    [[nodiscard]] CUresult prepare_graph_event_update(CUgraphExec executable, CUgraphNode node,
                                                      CUevent event, bool is_record,
                                                      CudaNativeGraphReservation& output) noexcept;
    // Internal normal-phase owner: original launch once, guarded graph waits
    // before grant, issued record epochs and exact imported prefix completion.
    // Ambiguous record/wait reuse rejects; no public activation implied.
    [[nodiscard]] CUresult native_graph_execution(CUgraphExec executable, CUstream stream,
                                                  bool is_ptds,
                                                  const NativeOperation& original) noexcept;
    [[nodiscard]] CUresult event_frontend(
        CUevent event, CUcontext& context,
        std::shared_ptr<CudaCooperativeFrontend>& output) noexcept;
    void forget_event(CUevent event) noexcept;
    [[nodiscard]] CUresult with_context(CUcontext context,
                                        const NativeOperation& operation) noexcept;
    [[nodiscard]] std::shared_ptr<void> resources() const noexcept;
    [[nodiscard]] decltype(&cuPointerGetAttribute) pointer_attribute() const noexcept;
    [[nodiscard]] const CudaCopyRangeDispatch& copy_range_dispatch() const noexcept;

   private:
    enum class ContextDestroyMode { kPrimaryOnly, kLegacy, kModern };
    [[nodiscard]] CUresult destroy_context_transaction(
        ProcessCallGate& gate, CUcontext context, const NativeOperation& original,
        const std::function<CUresult(CUcontext)>& retire_destroyed, ContextDestroyMode mode,
        CudaContextDestroyDisposition& disposition) noexcept;
    struct NativeHandleRetirement;
    [[nodiscard]] CUresult native_boundary_impl(const NativeOperation& operation,
                                                const NativeOperation& context_completion,
                                                const NativeHandleRetirement* retirement) noexcept;
    [[nodiscard]] CUresult source_metadata_original(ProcessCallGate& gate,
                                                    const NativeOperation& original,
                                                    CudaSourceMemoryOrigin origin,
                                                    const bool* has_native_issue) noexcept;
    [[nodiscard]] CUresult dispatch_source_capture(CUcontext context, CUstream stream, bool is_ptds,
                                                   const NativeOperation& original) noexcept;
    [[nodiscard]] CUresult finish_source_capture(ProcessCallGate::Transition& transition,
                                                 bool should_wait_for_executions = false) noexcept;
    [[nodiscard]] CUresult discover_graph(CUgraph graph, CudaNativeGraphEffects& output) noexcept;
    struct Implementation;
    std::unique_ptr<Implementation> implementation_;
};

}  // namespace glimmer::interceptor
