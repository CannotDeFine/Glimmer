#pragma once

#include "cuda_execution_observer.h"

#include "internal/execution_submission_queue.h"

#include <cuda.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace glimmer::backend::cuda {
class CudaRetirementObserver;

class CudaProcessMemoryOwner;

namespace detail {
// Diagnostics never wrap or reject an otherwise valid operation. Zero remains
// unavailable after exhaustion; it is never an execution or retirement token.
[[nodiscard]] constexpr std::uint64_t take_diagnostic_id(std::uint64_t& next) noexcept {
    const auto identity = next;
    if (next) next = next == UINT64_MAX ? 0 : next + 1;
    return identity;
}
}  // namespace detail

struct CudaExecutionDispatch {
    decltype(&cuCtxGetCurrent) context_get_current{};
    decltype(&cuLaunchKernel) launch_kernel{};
    decltype(&cuEventCreate) event_create{};
    decltype(&cuEventRecord) event_record{};
    decltype(&cuEventQuery) event_query{};
    decltype(&cuEventDestroy) event_destroy{};
    decltype(&cuStreamSynchronize) stream_synchronize{};
    decltype(&cuMemcpyDtoD) copy_device{};
    decltype(&cuMemcpyDtoDAsync) copy_device_async{};
    decltype(&cuMemsetD8) fill8{};
    decltype(&cuMemsetD8Async) fill8_async{};
    decltype(&cuMemsetD16) fill16{};
    decltype(&cuMemsetD16Async) fill16_async{};
    decltype(&cuMemsetD32) fill32{};
    decltype(&cuMemsetD32Async) fill32_async{};
    decltype(&cuMemcpyHtoDAsync) upload_async{};
    decltype(&cuMemcpyDtoHAsync) download_async{};
    decltype(&cuStreamWaitEvent) stream_wait_event{};
    decltype(&cuMemcpy) copy_unified{};
    decltype(&cuMemcpyAsync) copy_unified_async{};
};

// Optional cold source-owned binding for logical legacy stream zero. Logical
// blocking dependencies do not change when the physical stream is nonblocking.
struct CudaExecutionStreamConfig {
    CUstream default_stream{};
    std::shared_ptr<const void> default_stream_lifetime{};
};

struct CudaExecutionContext {
    std::uint64_t id{};
    CUcontext native{};
    // Trusted leases defer required SDK cleanup to their actual resource owner.
    std::shared_ptr<const void> lifetime;
    std::shared_ptr<CudaProcessMemoryOwner> process_memory{};
    // CPU-only sticky device proof. It gates issue, never physical retirement.
    std::function<CUresult()> native_issue_health{};
};

// CPU-only, original-owner-thread activity shared with an enclosing Session.
// It retains no Session, native resource or authority to issue work. Standalone
// dispatchers need no outer activity; their own callback exclusion still applies.
struct CudaExecutionOwnerActivity {
    bool is_native_issue_active{}, is_folded_active{}, is_rnn_query_active{},
        is_preparation_active{};
    // Shared only for an explicitly attached physical execution domain. Each
    // isolated Session still owns its independent activity and fault state.
    bool is_quarantined{};
};

struct CudaExecutionKernel {
    std::uint64_t function{};
    std::uint64_t stream{};
    std::array<unsigned int, 3> grid{};
    std::array<unsigned int, 3> block{};
    unsigned int shared_memory_bytes{};
    // Packed against the registered function's independently validated ABI.
    std::span<const std::byte> parameters;
    std::shared_ptr<const void> lifetime{};
    std::shared_ptr<CudaExecutionObserver> observation{};
};

struct CudaExecutionPersistentStop {
    CUresult status{CUDA_SUCCESS};
    bool is_complete{};
};

enum class CudaMemoryKind : std::uint8_t {
    kCopy,
    kFill8,
    kFill16,
    kFill32,
    kUpload,
    kDownload,
    kCopyUva
};
struct CudaExecutionMemory {
    CudaMemoryKind kind{CudaMemoryKind::kCopy};
    CUdeviceptr destination{}, source{};
    // Bytes for copy; elements for typed fill. All addresses and leases are
    // supplied by the trusted session after independent allocation validation.
    std::size_t count{};
    std::uint32_t value{};
    std::uint64_t stream{};
    bool is_asynchronous{};
    void* host{};
    std::shared_ptr<const void> host_lifetime{};
    // Unified copies may borrow two independent host owners. These leases are
    // ordered destination/source and survive SDK return until physical stop.
    std::array<std::shared_ptr<const void>, 2> endpoint_lifetimes{};
};

// Trusted backend factory only; never a callback supplied by an application or
// a wire message. issue runs once on the context owner, without metadata locks.
// Return the original vendor status, not a converted Driver error. Destruction
// is CPU-only; all native handles/buffers remain leased through physical stop.
struct CudaExecutionLibraryCompletion {
    CUresult status{CUDA_SUCCESS};
    bool is_cleanup_uncertain{};
    // Trusted calls can fail a last-moment ownership proof before invoking the
    // vendor. Such failures are preparation errors, never native SDK status.
    bool was_native_issued{true};
    // A known preparation rejection is not a physical completion failure. This
    // requires a nonzero status, no requested native call and known cleanup;
    // the dispatcher independently checks context and still fences all private
    // work before exactly-once cleanup. Contradictions retain uncertain owners.
    bool is_clean_preparation_rejection{};
};
class CudaExecutionLibraryCall {
   public:
    virtual ~CudaExecutionLibraryCall() = default;
    [[nodiscard]] virtual bool borrows_process_memory() const noexcept {
        return true;
    }
    [[nodiscard]] virtual std::int32_t issue(CUstream stream,
                                             std::span<const std::byte> parameters) noexcept = 0;
    // CPU-only post-issue information supplied by trusted backend factories.
    // Failed opaque-work physical proof is stronger than a trailing stream
    // event: uncertainty must retain the call and its leases indefinitely.
    [[nodiscard]] virtual CudaExecutionLibraryCompletion completion() const noexcept {
        return {};
    }
    // Native command-private objects may be created only inside issue(). After
    // proven physical stop, the dispatcher calls this once on the matching
    // context, before releasing any command leases. Failure retains the whole
    // record without retry; it never replaces the already published host result.
    // Queued cancellation does not call this hook. Destructors remain CPU-only.
    [[nodiscard]] virtual CUresult cleanup_after_physical_stop() noexcept {
        return CUDA_SUCCESS;
    }
};
// Trusted session-owned event operation, never a caller-provided callback.
// The shared call retains its native generation until actual retirement.
class CudaExecutionEventCall {
   public:
    virtual ~CudaExecutionEventCall() = default;
    [[nodiscard]] virtual CUresult issue(CUstream stream) noexcept = 0;
    // CPU-only trusted dependency observation on the shared resource owner.
    // NOT_READY blocks issue; failure rejects before the native wait call.
    [[nodiscard]] virtual CUresult dependency_status() const noexcept {
        return CUDA_SUCCESS;
    }
};
struct CudaExecutionLibrary {
    std::uint64_t stream{};
    std::span<const std::byte> parameters;
    std::shared_ptr<CudaExecutionLibraryCall> call;
};

// Trusted resource-factory object, never application/transport supplied. Calls
// are bounded, owner-only, nonreentrant and noexcept. It retains all private
// state and native resource leases until final retirement. prepare_launch is
// CPU-only, as is request_retreat. Readback is
// enqueued on the supplied stream before the private trailing event. Only
// observe_physical_stop may interpret that readback, after actual stop proof.
struct CudaExecutionParameterUpdate {
    std::size_t offset{};
    std::uint32_t value{};
};

class CudaExecutionPersistentProgress {
   public:
    virtual ~CudaExecutionPersistentProgress() = default;
    [[nodiscard]] virtual CUresult prepare_launch() noexcept = 0;
    [[nodiscard]] virtual CUresult request_retreat() noexcept = 0;
    // CPU-only geometry change after the dispatcher proves an incomplete
    // physical stop. Failure MUST leave progress unchanged. No counter reset,
    // SDK call, new execution permission or lifetime release is permitted.
    [[nodiscard]] virtual CUresult retarget_workers(unsigned int) noexcept {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    // Private u32 ABI fields only, prepared on the owner before each issue.
    // The dispatcher patches its charged payload without changing caller inputs.
    [[nodiscard]] virtual std::span<const CudaExecutionParameterUpdate> parameter_updates()
        const noexcept {
        return {};
    }
    // Stream-ordered private initialization before first kernel issue. Failure
    // does not prove that private GPU work stopped; the event still fences it.
    // Resumes must retain existing progress instead of resetting the counter.
    [[nodiscard]] virtual CUresult record_initialization(CUstream) noexcept {
        return CUDA_SUCCESS;
    }
    [[nodiscard]] virtual CUresult record_readback(CUstream stream) noexcept = 0;
    [[nodiscard]] virtual CudaExecutionPersistentStop observe_physical_stop() noexcept = 0;
};
struct CudaExecutionPersistentKernel {
    CUfunction function{};
    unsigned int workers{};
    // Independently validated private ABI, including owned progress addresses.
    // Copied into the same bounded queue budget instead of the original ABI.
    std::span<const std::byte> parameters;
    std::shared_ptr<CudaExecutionPersistentProgress> progress;
};

struct CudaExecutionWorkerGeometry {
    unsigned int workers{};
};

struct CudaExecutionPendingDescription {
    std::uint64_t sequence{}, acceptance_order{}, physical_generation{}, geometry_epoch{},
        command_epoch{};
    std::uint64_t function{}, stream{};
    std::array<unsigned int, 3> grid{}, block{}, physical_grid{};
    unsigned int shared_memory_bytes{};
    bool is_persistent{}, is_resume{}, requires_preparation{};
    bool operator==(const CudaExecutionPendingDescription&) const = default;
};

// CPU-only weak observation of one retained record, not a physical stop proof.
// Copies cannot keep retired native resources alive or select another command.
// Only the original dispatcher can consume it after exact phase/generation/
// geometry/readiness validation on its nonrecycled owner thread and process.
class CudaExecutionPendingCommand final {
   public:
    [[nodiscard]] explicit operator bool() const noexcept {
        return description_.sequence != 0 && !owner_.expired() && !record_.expired();
    }
    [[nodiscard]] CudaExecutionPendingDescription description() const noexcept {
        return description_;
    }

   private:
    friend class CudaExecutionDispatcher;
    CudaExecutionPendingDescription description_;
    std::weak_ptr<const void> owner_, record_;
};
struct CudaExecutionPendingResult {
    CUresult status{CUDA_SUCCESS};
    CudaExecutionPendingCommand command{};
};
struct CudaExecutionPendingKernelSelection {
    unsigned int workers{};
    bool uses_original{};
    // Initial profiling only; once any preparation has started, an existing
    // observer cannot be replaced or restarted for a partial continuation.
    std::shared_ptr<CudaExecutionObserver> observation{};
};

enum class CudaExecutionIssueState : std::uint8_t { kIdle, kIssued, kRejected, kError };
struct CudaExecutionIssueResult {
    CudaExecutionIssueState state{CudaExecutionIssueState::kIdle};
    std::uint64_t sequence{};
    CUresult native_status{CUDA_SUCCESS};
    CUresult completion_status{CUDA_SUCCESS};
    // Present only after an actual vendor call; native_status remains in the
    // Driver domain. A nonzero library status still needs physical retirement.
    std::optional<std::int32_t> library_status{};
};

struct CudaExecutionPollResult {
    CUresult status{CUDA_SUCCESS};
    std::size_t retired{};
    std::size_t retained{};
    std::size_t paused{};
};

// Immutable command membership, not a GPU event or a recyclable sequence.
// CPU-only metadata; native ownership remains in the executor until real stop.
struct CudaExecutionFrontierRetirement {
    CUresult status{CUDA_ERROR_INVALID_HANDLE};
    bool is_retired{};
};
class CudaExecutionFrontier final {
   public:
    CudaExecutionFrontier() = default;
    // Same nonrecycled owner thread/process only, independent of native current
    // context. Never calls SDK or manufactures a completion observation.
    [[nodiscard]] CUresult observe() const noexcept;
    // Reports the same first-error status, but scans every captured record for
    // retirement. An error is not stop proof. Invalid/reentrant access is false.
    [[nodiscard]] CudaExecutionFrontierRetirement retirement() const noexcept;
    // Exact immutable capture identity, not equivalent membership or a wire ID.
    // False for empty, expired, foreign-thread or inherited-process tokens.
    [[nodiscard]] bool same_capture(const CudaExecutionFrontier& other) const noexcept;
    // Owner/PID/thread/reentry-checked CPU metadata only. Copies share a stable
    // owner-scoped ID; zero is unavailable, not an empty/completed frontier.
    [[nodiscard]] std::uint64_t diagnostic_id() const noexcept;

   private:
    friend class CudaExecutionDispatcher;
    friend class CudaExecutionFrontierReservation;
    struct State;
    std::shared_ptr<const State> state_;
};
struct CudaExecutionFrontierResult {
    CUresult status{CUDA_ERROR_INVALID_HANDLE};
    CudaExecutionFrontier frontier;
};
// CPU-only, one exact future ordinary sequence. Copies share one metadata
// charge. Pending/cancelled tokens never supply a retirement observation.
class CudaExecutionFrontierReservation final {
   public:
    CudaExecutionFrontierReservation() = default;

   private:
    friend class CudaExecutionDispatcher;
    struct State;
    std::shared_ptr<State> state_;
};
struct CudaExecutionFrontierReservationResult {
    CUresult status{CUDA_ERROR_INVALID_HANDLE};
    CudaExecutionFrontierReservation reservation{};
};
struct CudaExecutionSynchronousResult {
    CUresult status{CUDA_ERROR_INVALID_HANDLE};
    std::optional<CUresult> original_status{};
    // Exact private tail, including post-admission failures. Not wire authority.
    CudaExecutionFrontier frontier{};
};
struct CudaExecutionDependencyInspection {
    // Without an observed failure, SUCCESS means satisfied and NOT_READY means
    // pending, with an optional exact foreign blocker. Failed dependencies also
    // retain that blocker when available; status alone never implies retirement.
    CUresult status{CUDA_ERROR_INVALID_HANDLE};
    std::optional<CudaExecutionFrontier> foreign_blocker{};
    // The error came from an authenticated retained observation, not a gate
    // or a callback's unrecorded return. Its native tail may still be pending.
    bool has_failure_observation{};
    // Only the exact reported failed observation is known settled. This is
    // neither whole-cut retirement nor proof for owner/gate/allocation errors;
    // implies has_failure_observation.
    bool has_retired_failure{};
    // This exact local closure contains a preparation or eligible issue/resume,
    // not merely an unretired physical tail. CPU observation only: the caller
    // still needs fresh authority and exact foreign-owner authentication.
    bool has_progress{};
    // Exact closure-local deferred input, never an ordinary suffix. Preparation
    // requires the same source-cut authority; it is not a native issue.
    std::optional<std::uint64_t> preparation_sequence{};
};
// Opaque owner/process/thread-scoped logical boundary, never wire authority or
// a native host result. Copy/destruction is CPU-only. The owner alone performs
// transitions; the accompanying frontier retains actual retirement observation.
class CudaExecutionExternalBarrier final {
   public:
    CudaExecutionExternalBarrier() = default;

   private:
    friend class CudaExecutionDispatcher;
    std::weak_ptr<const void> owner_;
    std::uint64_t id_{};
};
struct CudaExecutionExternalAcceptance {
    CUresult status{CUDA_ERROR_INVALID_HANDLE};
    CudaExecutionExternalBarrier barrier{};
    CudaExecutionFrontier frontier{};
};
class CudaNativeExecutionTicket;
struct CudaExecutionEnqueueState;
class CudaExecutionEnqueueTurn final {
   public:
    CudaExecutionEnqueueTurn() = default;

   private:
    friend class CudaExecutionDispatcher;
    std::weak_ptr<const void> owner_;
    std::uint64_t id_{};
};
// CPU-only endpoint. begin runs once on a distinct native worker, after source
// authorization. bind_ticket runs once on the original source owner and binds
// the exact admitted call; entry may race binding and never reads ticket state.
// Neither endpoint operation grants GPU completion or mutable session access.
class CudaExecutionEnqueueWorker final {
   public:
    CudaExecutionEnqueueWorker() = default;
    [[nodiscard]] CUresult begin() const noexcept;
    [[nodiscard]] CUresult bind_ticket(const CudaNativeExecutionTicket& ticket) const noexcept;

   private:
    friend class CudaExecutionDispatcher;
    std::shared_ptr<CudaExecutionEnqueueState> state_;
};
struct CudaExecutionEnqueueAcceptance {
    CUresult status{CUDA_ERROR_INVALID_HANDLE};
    CudaExecutionEnqueueTurn turn{};
    CudaExecutionEnqueueWorker worker{};
    // Exact accepted prefix only: never includes later ordinary admissions.
    CudaExecutionFrontier frontier{};
};
struct CudaExecutionExternalCohortMember {
    std::uint64_t stream{};
    CudaExecutionExternalBarrier barrier{};
    CudaExecutionFrontier frontier{};
};
struct CudaExecutionExternalCohortAcceptance {
    CUresult status{CUDA_ERROR_INVALID_HANDLE};
    std::vector<CudaExecutionExternalCohortMember> members{};
};
struct CudaExecutionEvent {
    std::uint64_t stream{};
    std::shared_ptr<CudaExecutionEventCall> call;
    std::optional<CudaExecutionFrontier> dependency;
    // Trusted Session factory only. Its call retains the source generation and
    // lifetime; this immutable token grants observation, not foreign issue.
    std::optional<CudaExecutionFrontier> foreign_dependency{};
};

struct CudaExecutionSchedulingSnapshot {
    ExecutionSubmissionSnapshot submissions;
    std::size_t external_barriers{};
    std::size_t enqueue_turns{};
    std::size_t external_work_in_flight{};
    bool has_ready_work{};
    // CPU demand only. Actual preparation must reacquire its exact token.
    bool has_pending_preparation{};
};
struct CudaExecutionEventAcceptance {
    ExecutionAcceptance acceptance{};
    CudaExecutionFrontier frontier{};
    // Actual accepted capture identity, available to the enclosing Session's
    // guarded factory without bypassing the public diagnostic reentry gate.
    std::uint64_t frontier_capture{};
};

// Trusted private server adapter, not an application API or a wire decoder.
// Except reply query/acknowledgment, calls are externally serialized on the bound
// executor. It owns private trailing events, not caller-owned native handles.
// All bound resources/streams must remain exclusive and valid until shutdown;
// registration is immutable and cannot substitute for server ABI/handle checks.
// Native dispatch must never resolve to an interceptor or reenter this adapter.
class CudaExecutionDispatcher final {
   public:
    // Optional trusted owner-only predicate for original opaque library issue.
    // Runs under callback exclusion before queue/native preparation. NOT_READY
    // preserves the queued command; other failures retain/quarantine it. This
    // predicate is not consulted for pure or transformed kernel computation.
    CudaExecutionDispatcher(ExecutionSessionIdentity session, CudaExecutionContext context,
                            CudaExecutionDispatch dispatch, ExecutionSubmissionLimits limits = {},
                            std::size_t max_bindings = 256, bool is_profile_isolated = false,
                            CudaExecutionStreamConfig streams = {},
                            std::shared_ptr<const CudaExecutionOwnerActivity> owner_activity = {},
                            std::function<CUresult()> exclusive_native_gate = {});
    ~CudaExecutionDispatcher();
    CudaExecutionDispatcher(const CudaExecutionDispatcher&) = delete;
    CudaExecutionDispatcher& operator=(const CudaExecutionDispatcher&) = delete;

    [[nodiscard]] CUresult bind_executor() noexcept;
    [[nodiscard]] bool bind_function(std::uint64_t id, CUfunction function,
                                     std::size_t parameter_bytes,
                                     std::shared_ptr<const void> lifetime) noexcept;
    // Stream zero denotes the context's legacy default stream, prebound here.
    // Client PTDS must be mapped to an actual owner-created stream, not a sentinel.
    [[nodiscard]] bool bind_stream(std::uint64_t id, CUstream stream,
                                   std::shared_ptr<const void> lifetime,
                                   bool is_nonblocking = false) noexcept;
    [[nodiscard]] bool resources_quiescent() const noexcept;
    // Physical ownership proof only; a queue fault still forbids new work.
    [[nodiscard]] bool physical_readers_quiescent() const noexcept;
    // Removal requires full quiescence; binding IDs advance monotonically and
    // cannot be reused after removal. Stream zero remains permanently bound.
    [[nodiscard]] bool unbind_function(std::uint64_t id) noexcept;
    [[nodiscard]] bool unbind_stream(std::uint64_t id) noexcept;
    [[nodiscard]] ExecutionAcceptance accept_kernel(std::uint64_t sequence,
                                                    const CudaExecutionKernel& kernel) noexcept;
    // CPU-only copied admission of an already selected bound original. Native
    // issue still needs its exact owner grant; late SDK errors remain observable
    // independently of host reply acknowledgment and physical retirement.
    [[nodiscard]] ExecutionAcceptance accept_async_original_kernel(
        std::uint64_t sequence, const CudaExecutionKernel& kernel) noexcept;
    // Pure CPU admission of an ordinary asynchronous launch. The retained
    // record is visible to captures/cleanup but cannot issue until prepared.
    [[nodiscard]] ExecutionAcceptance accept_deferred_kernel(
        std::uint64_t sequence, const CudaExecutionKernel& kernel,
        std::size_t parameter_capacity_bytes) noexcept;
    [[nodiscard]] CudaExecutionPendingResult pending_preparation(
        std::optional<std::uint64_t> sequence = std::nullopt) const noexcept;
    [[nodiscard]] CUresult copy_deferred_parameters(const CudaExecutionPendingCommand& command,
                                                    std::span<std::byte> output) const noexcept;
    [[nodiscard]] CudaExecutionPendingResult complete_kernel_preparation(
        const CudaExecutionPendingCommand& command,
        std::shared_ptr<CudaExecutionObserver> observation,
        const CudaExecutionPersistentKernel* persistent = nullptr) noexcept;
    // Exact failure observation without settling or releasing an uncertain
    // preparation. No SDK result or physical-stop proof is manufactured.
    [[nodiscard]] CUresult quarantine_deferred_kernel(const CudaExecutionPendingCommand& command,
                                                      CUresult failure) noexcept;
    [[nodiscard]] ExecutionAcceptance accept_memory(std::uint64_t sequence,
                                                    const CudaExecutionMemory& memory) noexcept;
    [[nodiscard]] ExecutionAcceptance accept_library(std::uint64_t sequence,
                                                     const CudaExecutionLibrary& library) noexcept;
    [[nodiscard]] ExecutionHostAllocation allocate_host_storage(std::size_t bytes) noexcept;
    [[nodiscard]] ExecutionAcceptance accept_event(std::uint64_t sequence,
                                                   const CudaExecutionEvent& event) noexcept;
    // Transactional exact-command capture: all frontier storage/quota precedes
    // queue acceptance. Failure neither consumes sequence nor admits a command.
    [[nodiscard]] CudaExecutionEventAcceptance accept_event_with_frontier(
        std::uint64_t sequence, const CudaExecutionEvent& event) noexcept;
    // Consume only private bounded record/frontier capacity, not an application
    // sequence/receipt. Retain exact stream/context and trusted source owners.
    [[nodiscard]] CudaExecutionExternalAcceptance accept_external_barrier(
        std::uint64_t stream, const std::shared_ptr<const void>& lifetime) noexcept;
    // Atomic within this dispatcher: bounded raw input, first-appearance stream
    // deduplication and one exact-tail frontier per member. The common lifetime
    // must retain ALL participating source capsules, including duplicate inputs.
    // Every member checks the union of pre-cohort dependencies; siblings never
    // block each other. No native call, application sequence or receipt is made.
    [[nodiscard]] CudaExecutionExternalCohortAcceptance accept_external_cohort(
        std::span<const std::uint64_t> streams,
        const std::shared_ptr<const void>& lifetime) noexcept;
    // CPU logical readiness, not physical completion or native SDK invocation.
    [[nodiscard]] CUresult external_barrier_ready(
        const CudaExecutionExternalBarrier& barrier) const noexcept;
    [[nodiscard]] CUresult begin_external_barrier(
        const CudaExecutionExternalBarrier& barrier) noexcept;
    // TRUSTED: actual enqueue is complete and every effect precedes THIS bound
    // native stream. Foreign/default work needs both source-prefix -> foreign
    // and foreign-completion -> source dependency bridges, not logical readiness
    // alone. This method creates neither bridge. Only this transition permits
    // recording/querying the trailing physical event.
    [[nodiscard]] CUresult record_external_enqueue(
        const CudaExecutionExternalBarrier& barrier) noexcept;
    // Only pre-begin known no-work cancellation. No native SDK result is minted.
    [[nodiscard]] CUresult cancel_external_barrier(
        const CudaExecutionExternalBarrier& barrier) noexcept;
    // CPU-only unknown-work retention; started/nonzero failure required. SDK
    // status publication remains in the external actor, not the ordinary queue.
    [[nodiscard]] CUresult fail_external_barrier(const CudaExecutionExternalBarrier& barrier,
                                                 CUresult failure) noexcept;
    [[nodiscard]] std::size_t external_barrier_count() const noexcept;
    // Owner-only CPU observation of retained external records whose native
    // boundary began, including enqueue turns. Accepted unstarted cuts do not
    // reserve physical work; failed/unproven started records remain counted.
    [[nodiscard]] std::size_t started_external_count() const noexcept;
    // A host-only admission cut, not a whole-group tail. Ready/begin respect
    // queued and paused predecessors. Finish requires the once-bound actual
    // native ticket's successful invoked return plus worker entry, then records
    // only the already-posted source prefix and releases host exclusion.
    [[nodiscard]] CudaExecutionEnqueueAcceptance accept_enqueue_turn(
        std::uint64_t stream, const std::shared_ptr<const void>& lifetime) noexcept;
    // One callback spanning several exact streams in this source context.
    // Predecessor union, host exclusion and terminal ticket are indivisible;
    // the posted prefix retires only after every member event stops.
    [[nodiscard]] CudaExecutionEnqueueAcceptance accept_enqueue_cohort(
        std::span<const std::uint64_t> streams,
        const std::shared_ptr<const void>& lifetime) noexcept;
    [[nodiscard]] CUresult enqueue_turn_ready(const CudaExecutionEnqueueTurn& turn) const noexcept;
    // Issue/resume at most one actual pre-cut dependency of this accepted turn.
    // Never selects unrelated/later work. Idle/NOT_READY retains unresolved
    // native/external/foreign-owner dependencies; it is not a completion proof.
    [[nodiscard]] CudaExecutionIssueResult progress_enqueue_dependency(
        const CudaExecutionEnqueueTurn& turn) noexcept;
    [[nodiscard]] CudaExecutionDependencyInspection inspect_enqueue_dependency(
        const CudaExecutionEnqueueTurn& turn) const noexcept;
    [[nodiscard]] CudaExecutionDependencyInspection inspect_frontier_dependency(
        const CudaExecutionFrontier& frontier) const noexcept;
    [[nodiscard]] bool owns_frontier(const CudaExecutionFrontier& frontier) const noexcept;
    // Includes unsettled captured roots and only their actual ancestors. The
    // trusted caller separately authenticates every cross-owner/Core grant.
    [[nodiscard]] CudaExecutionIssueResult progress_frontier_dependency(
        const CudaExecutionFrontier& frontier) noexcept;
    // Quiescence view of the SAME immutable cut. Settled historical failures
    // do not block remaining work; failure flags describe only blocking live
    // observations. SUCCESS means all target members settled, not SDK success.
    // Original observe()/retirement() errors remain unchanged. No later member
    // is added, and progress still issues/resumes at most one actual operation.
    [[nodiscard]] CudaExecutionDependencyInspection inspect_frontier_retirement(
        const CudaExecutionFrontier& frontier) const noexcept;
    [[nodiscard]] CudaExecutionIssueResult progress_frontier_retirement(
        const CudaExecutionFrontier& frontier) noexcept;
    [[nodiscard]] CUresult begin_enqueue_turn(const CudaExecutionEnqueueTurn& turn) noexcept;
    [[nodiscard]] CUresult finish_enqueue_turn(const CudaExecutionEnqueueTurn& turn) noexcept;
    [[nodiscard]] CUresult cancel_enqueue_turn(const CudaExecutionEnqueueTurn& turn) noexcept;
    [[nodiscard]] CUresult fail_enqueue_turn(const CudaExecutionEnqueueTurn& turn,
                                             CUresult failure) noexcept;
    [[nodiscard]] bool has_active_enqueue_turn() const noexcept;
    [[nodiscard]] std::size_t enqueue_turn_count() const noexcept;
    // Trusted owner-only Session adapter for original synchronous no-stream
    // callbacks. All storage/events precede the call; actual native legacy work
    // is fenced onto the default proxy before prefix publication. No wire input
    // or worker may supply this callback. Destruction never invokes it.
    [[nodiscard]] CudaExecutionSynchronousResult execute_synchronous_legacy(
        const std::function<CUresult()>& invoke,
        const std::shared_ptr<const void>& lifetime) noexcept;
    [[nodiscard]] ExecutionAcceptance accept_persistent_kernel(
        std::uint64_t sequence, const CudaExecutionKernel& logical,
        const CudaExecutionPersistentKernel& persistent) noexcept;
    [[nodiscard]] CudaExecutionIssueResult issue_next() noexcept;
    // Owner-only CPU capture: prefer a ready paused continuation, otherwise a
    // ready initial command. An explicit sequence never selects a substitute.
    // Success with an empty token means no ready command; no SDK call or quota
    // allocation occurs. Existing dependency-cut authorization remains separate.
    [[nodiscard]] CudaExecutionPendingResult pending_command(
        std::optional<std::uint64_t> sequence = std::nullopt) const noexcept;
    [[nodiscard]] CudaExecutionIssueResult issue(
        const CudaExecutionPendingCommand& command) noexcept;
    // CPU-only exact kernel preparation, not execution authority. Original
    // fallback/observer replacement is legal only before ANY initial attempt;
    // known-stopped continuation keeps its existing family, payload and counter.
    // On success return the new geometry token; old tokens cannot issue it.
    [[nodiscard]] CudaExecutionPendingResult select_pending_kernel(
        const CudaExecutionPendingCommand& command,
        const CudaExecutionPendingKernelSelection& selection) noexcept;
    // Request publication is not stop proof. A paused task stays logically
    // pending, retaining its exact frontier membership, parameters and leases.
    [[nodiscard]] CUresult request_retreat(std::uint64_t sequence) noexcept;
    // Changes only known-stopped incomplete private geometry. The same logical
    // task/parameters/frontier remain retained; this neither resumes nor grants
    // authority for a future resume. The progress factory enforces its bounds.
    [[nodiscard]] CUresult retarget_workers(std::uint64_t sequence,
                                            CudaExecutionWorkerGeometry geometry) noexcept;
    [[nodiscard]] CudaExecutionIssueResult resume(std::uint64_t sequence) noexcept;
    [[nodiscard]] CudaExecutionPollResult poll() noexcept;
    // Explicit blocking shutdown/recovery boundary; never used by issue/poll.
    [[nodiscard]] CudaExecutionPollResult drain_issued() noexcept;
    // CPU-only, owner-serialized snapshots of current accepted commands.
    // nullopt means the context; zero means only the legacy default stream.
    [[nodiscard]] CudaExecutionFrontierResult capture_frontier(
        std::optional<std::uint64_t> stream = std::nullopt,
        bool includes_submission_dependencies = false) noexcept;
    // One accepted logical command, not its context/stream prefix. Capture
    // before issue/rejection; the immutable token survives record retirement.
    [[nodiscard]] CudaExecutionFrontierResult capture_command_frontier(
        std::uint64_t sequence) noexcept;
    // Reserve every capture allocation/quota before submitting via any ordinary
    // admission API. Success atomically binds all observers of that sequence.
    // Rejection leaves tokens pending and consumes no sequence. Read/cancel of
    // existing tokens is permitted after close; bound tokens cannot be cancelled.
    [[nodiscard]] CudaExecutionFrontierReservationResult reserve_command_frontier(
        std::uint64_t sequence) noexcept;
    [[nodiscard]] CudaExecutionFrontierResult reserved_command_frontier(
        const CudaExecutionFrontierReservation& reservation) const noexcept;
    [[nodiscard]] CUresult cancel_frontier_reservation(
        const CudaExecutionFrontierReservation& reservation) noexcept;
    // Exact never-attempted target only. Any issue/preparation/private-native
    // attempt, pause or uncertainty forbids this CPU rejection path.
    [[nodiscard]] bool reject_command(const ExecutionReceipt& receipt, CUresult failure) noexcept;
    [[nodiscard]] CUresult query_frontier(const CudaExecutionFrontier& frontier) const noexcept;
    [[nodiscard]] std::size_t reject_queued(CUresult failure) noexcept;
    [[nodiscard]] std::optional<ExecutionHostReply> host_result(
        const ExecutionReceipt& receipt) const noexcept;
    [[nodiscard]] bool acknowledge_result(const ExecutionReceipt& receipt) noexcept;
    [[nodiscard]] ExecutionSubmissionSnapshot snapshot() const noexcept;
    [[nodiscard]] bool is_quarantined() const noexcept;
    [[nodiscard]] bool set_retirement_observer(
        std::shared_ptr<CudaRetirementObserver> observer) noexcept;
    // One owner/PID/thread-checked CPU projection, not physical observation or
    // an issue grant. Never cache across calls, native callbacks or owner turns.
    // Callback/active-enqueue restrictions match the individual getters.
    [[nodiscard]] CudaExecutionSchedulingSnapshot scheduling_snapshot() const noexcept;
    [[nodiscard]] bool has_ready_work() const noexcept;
    // Read-only opt-in audit aid, not stop or scheduling authority. A pair
    // exists only for exactly two issued original native kernels whose actual
    // trailing events remain pending after the owner's latest observation.
    [[nodiscard]] std::optional<std::array<std::uint64_t, 2>> native_kernel_pair() const noexcept;
    void close() noexcept;
    [[nodiscard]] bool shutdown() noexcept;

   private:
    enum class KernelAdmission { kHostResult, kDeferredAsync, kOriginalAsync };
    enum class FrontierPurpose { kExecution, kRetirement };
    [[nodiscard]] CudaExecutionDependencyInspection inspect_frontier_impl(
        const CudaExecutionFrontier& frontier, FrontierPurpose purpose) const noexcept;
    [[nodiscard]] CudaExecutionIssueResult progress_frontier_impl(
        const CudaExecutionFrontier& frontier, FrontierPurpose purpose) noexcept;
    [[nodiscard]] CudaExecutionEventAcceptance accept_event_impl(std::uint64_t sequence,
                                                                 const CudaExecutionEvent& event,
                                                                 bool captures_frontier) noexcept;
    [[nodiscard]] CudaExecutionFrontierResult capture_frontier_impl(
        std::optional<std::uint64_t> stream, std::optional<std::uint64_t> sequence,
        bool includes_submission_dependencies = false) noexcept;
    [[nodiscard]] ExecutionAcceptance accept_kernel_impl(
        std::uint64_t sequence, const CudaExecutionKernel& kernel,
        const CudaExecutionPersistentKernel* persistent,
        KernelAdmission admission = KernelAdmission::kHostResult,
        std::size_t parameter_capacity_bytes = 0) noexcept;
    struct Implementation;
    std::unique_ptr<Implementation> implementation_;
};

}  // namespace glimmer::backend::cuda
