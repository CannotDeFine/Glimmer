#pragma once

#include "cuda_native_execution_lanes.h"
#include "cuda_nccl_native_call.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace glimmer::backend::cuda {

struct CudaNcclCoordinatorLimits {
    CudaNativeExecutionLaneLimits lanes{};
    std::size_t maximum_communicators{64};
    std::size_t maximum_operations{256};
    std::size_t maximum_group_members{256};
    std::size_t maximum_group_depth{64};
    std::size_t maximum_tails{256};
    // Separate progress storage: normal operations cannot consume a group's
    // reserved End slot or the independent non-grouped async-query/Abort pool.
    // Total retained operation bound is the checked sum of these two limits
    // and maximum_operations. Finished replies return credit only on release.
    std::size_t maximum_retained_groups{16};
    std::size_t maximum_progress_operations{16};
    // Independent of public operation/progress storage; the checked aggregate
    // operation bound adds this housekeeping limit to the three pools above.
    // Caller domain history holds weak identities only, bounded per lane;
    // accepted identities cannot silently rebind, including after expiry.
    std::size_t maximum_housekeeping_operations{64};
    std::size_t maximum_caller_domains{64};
};
struct CudaNcclCoordinatorDomain {
    std::uint64_t identity{};
    int device_ordinal{-1};
    CudaExecutionContext context;
};
struct CudaNcclCoordinatorRequest {
    CudaNativeExecutionLaneKey caller;
    // Public caller sequence. Internal lane housekeeping has a separate sequence.
    std::uint64_t sequence{};
    CudaNcclCoordinatorDomain domain;
    std::uint64_t communicator{};
    // Raw input communicator/lifetime MUST be empty. The registry supplies it.
    // Collective extents, aliases, stream and progress-cohort authorization are
    // trusted caller responsibilities, retained by arguments.resources_lifetime.
    CudaNcclNativeArguments arguments;
    // Immutable worker endpoints only: NEVER capture a mutable session/frontier.
    // Every collective, including a group inner, REQUIRES prepare before original
    // entry: vendor fast paths may enqueue even inside an open native group.
    // Direct collective: both required. Group inner: prepare only; NEVER record.
    // Outermost End of a collective group: record only, aggregating ALL outgoing
    // events after known enqueue. Incoming waits must already precede each inner.
    // No hook may block on a tail or capture mutable session/frontier state.
    std::function<CUresult()> prepare_enqueue;
    std::function<CUresult()> record_enqueue;
    // Owner-begun source turns. Exact admitted ticket is bound once; immutable
    // endpoints are entered on the real lane before context selection/native entry.
    std::vector<CudaExecutionEnqueueWorker> source_turns{};
    // Absent preserves fixed-domain private callers. Present canonical empty
    // domain (identity/context zero, ordinal -1) selects actual native NULL.
    // Non-NULL entry is independently leased, never inferred from an opcode.
    std::optional<CudaNcclCoordinatorDomain> caller_context{};
};
enum class CudaNcclContextKind : std::uint8_t { kNull, kCaller, kDomain };
struct CudaNcclContextIdentity {
    CudaNcclContextKind kind{CudaNcclContextKind::kNull};
    std::uint64_t identity{}, context_id{};
    int device_ordinal{-1};
    bool operator==(const CudaNcclContextIdentity&) const = default;
};
struct CudaNcclContextTransition {
    CudaNcclContextIdentity before, after;
};
struct CudaNcclCoordinatorAcceptance {
    CUresult carrier{CUDA_ERROR_INVALID_VALUE};
    std::uint64_t operation{};
    std::uint64_t reserved_communicator{};
};
struct CudaNcclHousekeepingAcceptance {
    CUresult status{CUDA_ERROR_INVALID_VALUE};
    std::uint64_t job{};
};
enum class CudaNcclCoordinatorPhase : std::uint8_t {
    kPending,
    kReturned,
    kWaitingAsync,
    kWaitingTails,
    kComplete,
    kUncertain,
};
struct CudaNcclCoordinatorPublication {
    std::uint64_t operation{}, reserved_communicator{};
    std::optional<CudaNcclNativeObservation> output{};
};
struct CudaNcclCoordinatorObservation {
    CUresult carrier{CUDA_ERROR_INVALID_HANDLE};
    CudaNcclCoordinatorPhase phase{CudaNcclCoordinatorPhase::kPending};
    std::optional<std::int32_t> original_status{};
    // Private CPU receipt, NOT a public output-write oracle/native pointer token.
    std::optional<CudaNcclNativeObservation> output{};
    std::uint64_t reserved_communicator{};
    bool reply_ready{};
    bool enqueue_complete{};
    bool physical_complete{};
    // Outermost End carries deferred init projections even after each inner
    // reply was released. Private snapshots still are not arbitrary-write masks.
    std::vector<CudaNcclCoordinatorPublication> publications{};
    // Actual proved observations survive a later restoration/carrier failure.
    // This is identity evidence, not public context mutation authorization.
    std::optional<CudaNcclContextTransition> context_transition{};
};
struct CudaNcclCallerSnapshot {
    CudaNativeExecutionLaneKey caller;
    std::size_t group_depth{};
    std::uint64_t active_operation{}, active_housekeeping{};
    bool uncertain{};
};
struct CudaNcclOperationSnapshot {
    std::uint64_t operation{}, reserved_communicator{};
    CudaNativeExecutionLaneKey caller;
    CudaNcclCoordinatorPhase phase{};
    std::vector<int> devices;
    bool active_call{}, grouped{}, is_batch{}, pending_async{}, pending_tails{};
    // Attachment marker only. The source owner independently finishes each turn;
    // neither host return nor this snapshot claims that source host exclusion ended.
    bool has_source_turns{}, ready_continuation{}, uncertain{};
};
struct CudaNcclCoordinatorSnapshot {
    CUresult carrier{CUDA_ERROR_UNKNOWN};
    std::vector<CudaNcclCallerSnapshot> callers{};
    std::vector<CudaNcclOperationSnapshot> operations{};
};

// One owner thread/process. Owns the one shared native hub and bounded symbolic
// communicator/group/operation registries. Workers receive immutable packets,
// leases and receipts, never this coordinator or mutable source sessions.
// Caller authenticates peer/thread/domain identities; IDs are never raw pointers.
// Cross-caller sequential use is permitted within the same authenticated peer
// and exact retained domain. Native result, async state, enqueue, tails and
// destructive completion are distinct. No automatic public status fabrication,
// SDK teardown, replay, post-destructive polling or arbitrary completion boolean.
class CudaNcclCoordinator final {
   public:
    CudaNcclCoordinator(CudaNcclNativeDispatch native_dispatch,
                        CudaNativeExecutionLaneDispatch lane_dispatch,
                        CudaNcclCoordinatorLimits limits = {});
    ~CudaNcclCoordinator();
    CudaNcclCoordinator(const CudaNcclCoordinator&) = delete;
    CudaNcclCoordinator& operator=(const CudaNcclCoordinator&) = delete;
    [[nodiscard]] CudaNcclCoordinatorAcceptance submit(CudaNcclCoordinatorRequest request) noexcept;
    // Trusted immutable endpoints only; never capture mutable owner/session/
    // frontier state. Reuses an EXISTING authenticated caller and an exact domain
    // previously accepted there. Uses only the internal lane sequence, not a
    // public NCCL sequence or reply. Empty devices means the domain's singleton;
    // otherwise require a bounded unique set of previously accepted live devices
    // including that device. The payload structurally retains only the selected
    // domain; lifetime/callable MUST retain every additional-device domain owner.
    // Weak history is identity evidence, never a resource lease. This grants no GPU-stop
    // proof: e.g. bridge cleanup independently requires actual source cleanup.
    // Close permits this cleanup; stopped/shutdown hubs do not. All jobs retain
    // their owners until explicit release; unknown entry remains unreleasable.
    [[nodiscard]] CudaNcclHousekeepingAcceptance submit_housekeeping(
        CudaNativeExecutionLaneKey caller, CudaNcclCoordinatorDomain domain,
        std::shared_ptr<const void> lifetime, std::function<CUresult()> invoke,
        std::span<const int> device_ordinals = {}) noexcept;
    [[nodiscard]] CudaNativeExecutionCallObservation observe_housekeeping(
        std::uint64_t job) const noexcept;
    [[nodiscard]] CUresult release_housekeeping(std::uint64_t job) noexcept;
    // Bounded CPU observation and asynchronous housekeeping submission; no waits
    // and no native callbacks on the owner. Requested async queries are real calls.
    [[nodiscard]] CUresult advance() noexcept;
    // Public actor uses false: consume CPU observations/tails without issuing any
    // new lane callback. Only its exact Core grant authorizes continue_enqueue.
    [[nodiscard]] CUresult advance(bool authorize_continuations) noexcept;
    [[nodiscard]] CUresult continue_enqueue(std::uint64_t operation) noexcept;
    [[nodiscard]] CudaNcclCoordinatorSnapshot snapshot() const noexcept;
    [[nodiscard]] CudaNcclCoordinatorObservation observe(std::uint64_t operation) const noexcept;
    [[nodiscard]] CUresult release_reply(std::uint64_t operation) noexcept;
    // Once, nonempty, exact actual source-owner frontiers. These are observed on
    // the owner only; every tail must succeed before collective owners can retire.
    [[nodiscard]] CUresult bind_tails(std::uint64_t operation,
                                      std::span<const CudaExecutionFrontier> tails) noexcept;
    // Admission closure is not cancellation/Abort/Destroy. Shutdown succeeds only
    // with closed groups, no native owners, pending async work or unretired tails.
    // An unsafe destructor retains the whole hub without stopping open native TLS.
    [[nodiscard]] CUresult close() noexcept;
    [[nodiscard]] CUresult shutdown() noexcept;

   private:
    struct Implementation;
    std::unique_ptr<Implementation> implementation_;
};

}  // namespace glimmer::backend::cuda
