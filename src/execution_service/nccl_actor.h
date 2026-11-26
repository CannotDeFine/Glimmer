#pragma once

#include "glimmer/control/execution_nccl_protocol.h"
#include "internal/cuda_execution_session.h"
#include "internal/cuda_nccl_coordinator.h"
#include "internal/cuda_nccl_primary_domain.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

namespace glimmer::app {

// Private version-one numeric diagnostics. These values are not wire authority,
// timestamps, native handles, or proof of entry on a worker. Inactive fields are
// zero; stages/phases/origins/dispositions encode their existing enum plus one.
enum class NcclActorAuditEvent : std::uint32_t {
    kBegin = 1,
    kAdmission,
    kGrantBegin,
    kGrantEnd,
    kState,
    kBinding,
    kSource,
    kNativeSubmit,
    kObservation,
    kGroup,
    kReply,
    kRelease,
    kRetirement,
    kParent,
    kCleanup,
    kBatch,
    kDisconnect,
    kClose,
    kStopped,
    kLost,
};
enum class NcclActorAuditSubject : std::uint32_t {
    kActor = 1,
    kBatch,
    kRequest,
    kContinuation,
    kParent,
    kCleanupTicket,
};
enum class NcclActorAuditAction : std::uint32_t {
    kNone = 0,
    kAccept,
    kBegin,
    kFinish,
    kCancel,
    kRecord,
    kReject,
    kBind,
    kObserve,
    kPublish,
    kRelease,
    kRetire,
    kRemove,
    kSeal,
    kProve,
};
enum class NcclActorAuditBinding : std::uint32_t {
    kNone = 0,
    kPrecall,
    kTarget,
    kTurn,
    kTail,
    kGroupMember,
    kPublication,
};
struct NcclActorAuditRecord {
    std::uint64_t actor_serial{};
    NcclActorAuditEvent event{NcclActorAuditEvent::kState};
    NcclActorAuditSubject subject{NcclActorAuditSubject::kActor};
    NcclActorAuditAction action{};
    NcclActorAuditBinding binding{};
    std::uint64_t id{}, peer{};
    std::uint32_t priority{};
    std::array<std::uint64_t, 2> caller{};
    std::uint64_t group{}, sequence{}, ordinary_sequence{}, cleanup_parent{}, origin_source{};
    // Claim is input only; transition fields require an actual observed pair.
    // Context kinds and signed device ordinals are encoded plus one; NULL is
    // kind1/device0 and absent observations leave all transition fields zero.
    std::uint64_t caller_primary_domain{};
    std::uint32_t before_kind{}, after_kind{}, before_device{}, after_device{};
    std::uint64_t before_identity{}, before_context{}, after_identity{}, after_context{};
    std::uint64_t reservation{}, generation{}, cutoff{};
    std::uint64_t input_reservation{}, input_generation{}, input_cutoff{};
    std::uint64_t native_sequence{}, native_operation{};
    std::uint32_t native_kind{}, ordinary_opcode{};
    // All zero until coordinator acceptance; then one cached symbolic tuple.
    // A reserved Init object proves neither a native output nor readiness.
    std::uint64_t communicator_domain{}, communicator_context{}, communicator_object{};
    // Trusted cold output contract plus one on native requests, zero otherwise.
    std::uint32_t output_contract{};
    std::uint64_t connection{}, wire_sequence{}, attempt{};
    std::uint64_t source{}, stream{}, capture{}, command_sequence{}, related{}, index{}, count{};
    std::uint64_t allocation{}, ticket{}, depth{};
    std::uint32_t stage{}, previous_stage{}, phase{}, outcome{}, disposition{}, host_origin{};
    std::int32_t status{}, carrier{}, original_status{}, async_status{};
    std::int32_t native_status{}, completion_status{}, library_status{};
    bool accepted{}, sequence_consumed{}, has_original{}, has_async{}, has_library{}, has_host{};
    bool has_output_written{}, output_written{}, reply_ready{}, reply_released{};
    bool enqueue_complete{}, physical_complete{}, has_physical_work{}, open_group{}, unknown{},
        retired{};
    bool has_context{};
    bool bound{}, consumed{}, active_call{}, ready_continuation{};
    bool operator==(const NcclActorAuditRecord&) const = default;
};
struct NcclActorObserver {
    // Nonowning; the composer keeps this context alive through actor destruction.
    // The callback must contain its own logging failures, must not call SDKs or
    // reenter/mutate the actor, and must not retain a reference to this record.
    void* context{};
    void (*emit)(void*, const NcclActorAuditRecord&) noexcept {};
};

struct NcclActorPeer {
    std::uint64_t identity{};
    std::uint32_t priority{};
};
struct NcclActorSource {
    std::uint64_t session{};
    std::shared_ptr<backend::cuda::CudaExecutionSession> owner;
    std::uint64_t stream{};
};
struct NcclActorResolver {
    // Trusted runtime authentication, never native pointer conversion. Context
    // and buffer owners resolve by peer + exact source session identity.
    std::function<std::shared_ptr<backend::cuda::CudaExecutionSession>(std::uint64_t,
                                                                       std::uint64_t)>
        session;
    // Resolve Default/Legacy/PTDS to the existing authenticated source proxy;
    // Owned must match its exact owner/object. A NULL native stream is rejected.
    std::function<NcclActorSource(std::uint64_t, const control::ExecutionNcclScope&,
                                  const control::ExecutionNcclStream&)>
        stream;
    // Optional CPU-only admission fence. Existing accepted turns are unaffected.
    std::function<CUresult(std::uint64_t, std::uint64_t)> admit_source;
    // Phase-two proof from the exact runtime-owned HostOwner, not wire flags.
    // SUCCESS requires healthy closed state, no context/host bytes/reservation.
    std::function<CUresult(std::uint64_t, std::uint64_t, const std::shared_ptr<const void>&)>
        cleanup_owner_retired;
};
struct NcclActorEffectInput {
    const control::ExecutionNcclRequest& request;
    std::int32_t original_status{};
    const std::optional<backend::cuda::CudaNcclNativeObservation>& native;
    control::ExecutionNcclCommunicator reserved_communicator;
    bool is_deferred{};
};
struct NcclActorEffectProducer {
    // Trusted native write-contract implementation ONLY. Byte differences and
    // success/error guesses are not write proof. Missing producer deliberately
    // leaves output-bearing public envelopes unavailable; private evidence stays.
    void* context{};
    CUresult (*produce)(void*, const NcclActorEffectInput&,
                        control::ExecutionNcclObservation&) noexcept {};
    std::shared_ptr<const void> lifetime;
    backend::cuda::CudaNcclNativeOutputContract output_contract{
        backend::cuda::CudaNcclNativeOutputContract::kDeferred};
};
struct NcclActorConfig {
    std::uint64_t primary_domain{1};
    int device_ordinal{};
    std::size_t maximum_requests{256}, maximum_callers{64}, maximum_sources{64};
    // Each prepared outer Start reserves one End slot through native async,
    // tails, replies and unretired ordinary work. Parent metadata is separate.
    std::size_t maximum_groups{16}, maximum_communicators{64};
    std::size_t maximum_reservations{64};
    backend::cuda::CudaNcclCoordinatorLimits coordinator{};
    // Ordinary RPCs and durable import cleanup never consume native End credit.
    std::size_t maximum_continuations{256}, maximum_cleanup_parents{256};
};
struct NcclActorPreparation {
    CUresult status{CUDA_ERROR_INVALID_VALUE};
    std::uint64_t request{};
};
struct NcclActorCleanupRetirement {
    std::uint64_t identity{};
    std::size_t parents{};
};
struct NcclActorCleanupRetirementPreparation {
    CUresult status{CUDA_ERROR_INVALID_VALUE};
    NcclActorCleanupRetirement ticket;
};
struct NcclActorSourceProgress {
    CUresult status{CUDA_ERROR_INVALID_VALUE};
    std::uint64_t source_session{};
    backend::cuda::CudaExecutionIssueResult issue{};
};
struct NcclActorSourceInspection {
    CUresult status{CUDA_ERROR_INVALID_VALUE};
    std::uint64_t source_session{};
    backend::cuda::CudaExecutionDependencyInspection dependency{};
};
struct NcclActorGrant {
    // Reservation/generation survive group inners, async state and physical tails.
    // Cutoff names one exact actor stage; it is not a caller/application sequence.
    std::uint64_t reservation{}, generation{}, cutoff{};
    bool operator==(const NcclActorGrant&) const = default;
};
struct NcclActorContinuationTarget {
    std::uint64_t connection{}, wire_sequence{};
    control::ExecutionOpcode operation{};
    std::uint64_t cleanup_allocation{};
    std::shared_ptr<const void> request_lifetime;
    // Stable runtime-owned helper identity, shared across its import/release RPCs.
    std::shared_ptr<const void> helper_owner;
    // Trusted runtime role classification, never caller/wire authority. CPU
    // metadata roles need no HostOwner; imports still require the actual owner.
    bool is_context_free_helper{};
    // Exact runtime-authenticated target sealed by its own Close admission.
    // Only normal Close with origin_source == connection may supply this lease.
    std::shared_ptr<backend::cuda::CudaExecutionSession> closing_source;
};
struct NcclActorContinuationBinding {
    std::uint64_t session{};
    std::shared_ptr<backend::cuda::CudaExecutionSession> owner;
    backend::cuda::CudaExecutionFrontier frontier;
    std::optional<backend::ExecutionReceipt> receipt{};
    // Diagnostic only; actor overwrites this at binding while the owner is live.
    std::uint64_t diagnostic_capture{};
};
struct NcclActorContinuationPermit {
    std::uint64_t continuation{}, attempt{};
    bool operator==(const NcclActorContinuationPermit&) const = default;
};
struct NcclActorContinuationAdmission {
    CUresult status{CUDA_ERROR_INVALID_VALUE};
    NcclActorContinuationPermit permit;
};
enum class NcclActorContinuationStage : std::uint8_t {
    kPrepared,
    kAttempting,
    kBlocked,
    kCommandPending,
    kHostPending,
    kWaitingRetirement,
    kComplete,
    kCancelled,
    kUncertain,
};
enum class NcclActorContinuationOutcomeKind : std::uint8_t {
    kDeferred,
    kCommandAccepted,
    kHostJobPending,
    kReturned,
    kUncertain,
};
struct NcclActorContinuationOutcome {
    NcclActorContinuationOutcomeKind kind{NcclActorContinuationOutcomeKind::kUncertain};
    // Carrier/proof status, independent of every actual ordinary SDK/wire status.
    CUresult status{CUDA_ERROR_UNKNOWN};
    std::optional<NcclActorContinuationBinding> binding{};
    std::optional<control::ExecutionWireReply> reply{};
    // Only a real accepted helper mailbox job may supply this owner/sequence.
    std::shared_ptr<const void> pending_lifetime;
    std::uint64_t pending_sequence{}, imported_allocation{};
};
struct NcclActorContinuationSnapshot {
    std::uint64_t continuation{};
    NcclActorGrant grant;
    control::ExecutionSourceContinuation scope;
    std::uint64_t connection{}, wire_sequence{};
    control::ExecutionOpcode operation{};
    NcclActorContinuationStage stage{};
    bool is_ready{}, reply_ready{}, is_retired{};
    std::vector<std::uint64_t> dependencies;
};
struct NcclActorContinuationInspection {
    CUresult status{CUDA_ERROR_INVALID_VALUE};
    NcclActorContinuationBinding binding;
    backend::cuda::CudaExecutionDependencyInspection dependency;
};
struct NcclActorContinuationReply {
    CUresult status{CUDA_ERROR_NOT_READY};
    bool ready{}, is_retired{};
    std::optional<control::ExecutionWireReply> reply{};
    std::optional<backend::ExecutionHostReply> host_reply{};
};
enum class NcclActorStage : std::uint8_t {
    kPrepared,
    kSourceBlocked,
    kNativePending,
    kFinishSources,
    kWaitingAsync,
    kRecordContinuation,
    kRecordTails,
    kWaitingTails,
    kComplete,
    kUncertain,
    kRejected,
};
struct NcclActorDependency {
    std::uint64_t session{};
    std::vector<std::uint64_t> streams;
    bool is_ready{};
};
struct NcclActorRequestSnapshot {
    std::uint64_t request{};
    NcclActorGrant grant;
    NcclActorStage stage{};
    bool is_ready{}, has_native_call{}, reply_ready{};
    std::vector<NcclActorDependency> dependencies;
};
struct NcclActorReservation {
    std::uint64_t reservation{}, generation{}, peer{};
    std::uint32_t priority{};
    bool has_physical_work{}, has_unknown_state{}, has_open_group{};
    std::vector<NcclActorRequestSnapshot> requests;
    std::vector<NcclActorContinuationSnapshot> continuations;
};
struct NcclActorSnapshot {
    CUresult status{CUDA_ERROR_UNKNOWN};
    std::vector<NcclActorReservation> reservations{};
};
struct NcclActorReply {
    CUresult carrier{CUDA_ERROR_NOT_READY};
    bool ready{};
    std::optional<control::ExecutionNcclReply> envelope{};
    // Private diagnostic evidence, never directly projected to caller storage.
    std::optional<backend::cuda::CudaNcclCoordinatorObservation> native{};
};

// Single constructing owner thread/process, retained independently of sockets.
// Resolver/effect callbacks run under the actor's reentry guard; workers receive
// immutable capsules only. Owner native entry must be NULL between actor calls.
// prepare/snapshot/observe/reply do not enqueue SDK work. Only an exact grant may
// initialize/capture/begin/finish/record. Runtime separately polls source Sessions
// and authorizes their required prefix work against this retained reservation.
class NcclActor final {
   public:
    NcclActor(NcclActorConfig config, backend::cuda::CudaNcclNativeDispatch native,
              backend::cuda::CudaNativeExecutionLaneDispatch lane,
              backend::cuda::CudaNcclPrimaryDomainDispatch primary, NcclActorResolver resolver,
              NcclActorEffectProducer effects = {});
    ~NcclActor();
    NcclActor(const NcclActor&) = delete;
    NcclActor& operator=(const NcclActor&) = delete;
    // Cold owner/process/thread installation only, before any caller exists.
    // Serial exhaustion disables diagnostics, never execution or ownership.
    [[nodiscard]] bool set_observer(NcclActorObserver observer) noexcept;
    [[nodiscard]] NcclActorPreparation prepare(NcclActorPeer peer,
                                               control::ExecutionNcclRequest request,
                                               std::uint64_t caller_primary_domain = 0) noexcept;
    [[nodiscard]] CUresult issue(std::uint64_t request, NcclActorGrant grant) noexcept;
    // Exact accepted source-turn prefix only. The returned source/sequence lets
    // the service account for actual issue; it never authorizes issue_next().
    [[nodiscard]] NcclActorSourceProgress progress_source(std::uint64_t request,
                                                          NcclActorGrant grant,
                                                          std::uint64_t source_session) noexcept;
    // CPU-only authentication of the original stage and retained source turn.
    // A dependency error is evidence, not permission to issue or retire it.
    [[nodiscard]] NcclActorSourceInspection inspect_source(
        std::uint64_t request, NcclActorGrant grant, std::uint64_t source_session) const noexcept;
    // Revalidates the exact source's settled failure under the original grant.
    // Cancels only this request's unbegun turns; never invokes NCCL, consumes a
    // native sequence or cancels other work. A supplied error is not authority.
    [[nodiscard]] CUresult reject_source(std::uint64_t request, NcclActorGrant grant,
                                         std::uint64_t source_session, CUresult failure) noexcept;
    // CPU-only ordinary bookkeeping. Runtime owns the actual handler/Core/fence
    // composition. Neither a permit nor a reply is physical completion proof.
    [[nodiscard]] NcclActorPreparation prepare_continuation(
        NcclActorPeer peer, control::ExecutionSourceContinuation scope,
        NcclActorContinuationTarget target) noexcept;
    [[nodiscard]] CUresult bind_continuation_dependency(
        std::uint64_t continuation, NcclActorContinuationBinding binding) noexcept;
    [[nodiscard]] NcclActorContinuationAdmission begin_continuation(std::uint64_t continuation,
                                                                    NcclActorGrant grant) noexcept;
    // Must finish exactly once, including exceptions/unknown outcomes. Other
    // actor entry is excluded while this synchronous runtime attempt is armed.
    [[nodiscard]] CUresult finish_continuation_attempt(
        NcclActorContinuationPermit permit, NcclActorContinuationOutcome outcome) noexcept;
    // Only consumes the exact already accepted command/mailbox result. No retry.
    [[nodiscard]] CUresult complete_continuation(std::uint64_t continuation,
                                                 NcclActorContinuationOutcome outcome) noexcept;
    [[nodiscard]] NcclActorContinuationInspection inspect_continuation_source(
        std::uint64_t continuation, NcclActorGrant grant, std::uint64_t source) const noexcept;
    [[nodiscard]] NcclActorContinuationReply continuation_reply(
        std::uint64_t continuation) const noexcept;
    [[nodiscard]] CUresult release_continuation_reply(std::uint64_t continuation) noexcept;
    // Exact never-entered cancellation only; never acknowledges a backend receipt.
    [[nodiscard]] CUresult cancel_continuation(std::uint64_t continuation) noexcept;
    // CPU-only seal of the exact helper and immutable parent set. The optional
    // existing terminal Close still requires its own fresh ordinary grant.
    [[nodiscard]] NcclActorCleanupRetirementPreparation begin_cleanup_retirement(
        NcclActorPeer peer, std::uint64_t connection, std::shared_ptr<const void> helper,
        std::uint64_t closing_continuation = 0) noexcept;
    // Only the trusted resolver's actual closed-owner proof consumes the pins.
    [[nodiscard]] CUresult complete_cleanup_retirement(NcclActorCleanupRetirement ticket) noexcept;
    [[nodiscard]] CUresult observe() noexcept;
    [[nodiscard]] NcclActorSnapshot snapshot() const noexcept;
    [[nodiscard]] NcclActorReply reply(std::uint64_t request) const noexcept;
    [[nodiscard]] CUresult release_reply(std::uint64_t request) noexcept;
    [[nodiscard]] CUresult disconnect(std::uint64_t peer) noexcept;
    [[nodiscard]] CUresult close() noexcept;
    // Never signals/cancels native calls, resets devices or destroys live TLS.
    // Unknown/disconnected native ownership remains retained, not timeout-released.
    [[nodiscard]] CUresult shutdown() noexcept;

   private:
    struct Implementation;
    std::unique_ptr<Implementation> implementation_;
};
}  // namespace glimmer::app
