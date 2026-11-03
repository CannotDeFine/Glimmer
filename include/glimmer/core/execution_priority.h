#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

namespace glimmer::core {

inline constexpr std::size_t kMaxExecutionPriorityPeers = 64;
enum class ExecutionSchedulingMode : std::uint8_t {
    kPriority,
    kNaive,
    kProfile,
    kWorkloadAgnostic,
    kWorkloadAware,
    kPrioritySpaceShare,
};
// Workload-aware narrows these permissions through exact measured pair grants
// and retained arrival checkpoints in the owner, never naive/all-ready issue.
[[nodiscard]] constexpr bool is_priority_execution_mode(ExecutionSchedulingMode mode) noexcept {
    return mode == ExecutionSchedulingMode::kPriority ||
           mode == ExecutionSchedulingMode::kWorkloadAware;
}
// Trusted owner-observed dependency, never a wire/client claim. The runtime
// authenticates generation/cutoff against a retained native batch before calling
// pure policy and must revalidate it before issuing ONLY that dependency cut.
// This does not grant a new reservation, general peer work or physical stop.
struct ExecutionPriorityProgress {
    std::uint64_t reservation{}, generation{}, cutoff{};
    bool operator==(const ExecutionPriorityProgress&) const = default;
};
// Trusted runtime observation of ONE ordinary retained command. Opaque backend
// ownership remains outside Core. Initial work has physical_generation zero;
// continuation requires a positive actual generation. Shared-domain epochs are
// either all zero (isolated) or all positive. No field proves physical stop.
struct ExecutionPriorityCommand {
    std::uint64_t sequence{}, acceptance_order{}, physical_generation{}, geometry_epoch{},
        command_epoch{}, control_epoch{}, member_generation{}, configuration_epoch{},
        physical_configuration_epoch{};
    bool is_resume{};
    bool operator==(const ExecutionPriorityCommand&) const = default;
};
// Trusted backend metadata for the current exact command. Numeric labels alone
// do not prove native ownership; the adapter verifies opaque domain/configuration
// equality and the actual SDK resource catalog before proposing shared work.
struct ExecutionPairContext {
    std::uint64_t execution_domain{}, catalog_generation{};
    unsigned int workers{};
    bool is_configuration_materialized{};
};
// Owner-observed ordinary persistent physical epoch, not a stop/cleanup proof
// or a native collective reservation. Native owners revalidate it before retreat.
struct ExecutionPriorityPhysical {
    std::uint64_t sequence{}, generation{};
    bool operator==(const ExecutionPriorityPhysical&) const = default;
};
struct ExecutionPriorityPeer {
    std::uint64_t session{};
    std::uint32_t priority{};
    bool has_ready_work{}, has_physical_work{}, has_paused_work{}, has_unknown_state{};
    std::optional<ExecutionPriorityProgress> progress{};
    // Nonzero only for one exact retained native reservation represented by
    // this policy peer. Distinct simultaneous batches require distinct peers;
    // a process/connection-wide number cannot stand for several generations.
    std::uint64_t reservation_generation{};
    std::optional<ExecutionPriorityCommand> command{};
    std::optional<ExecutionPairContext> pair_context{};
    std::optional<ExecutionPriorityPhysical> physical{};
};
enum class ExecutionPriorityAction : std::uint8_t {
    kIdle,
    kWait,
    kRequestRetreat,
    kExecute,
    kQuarantined,
    // Only the exact returned progress tuple, not ordinary ready work. Needed
    // to finish a retained nonpreemptible batch without priority inversion.
    kExecuteProgress,
};
struct ExecutionPriorityDecision {
    std::uint64_t session{};
    ExecutionPriorityAction action{ExecutionPriorityAction::kIdle};
    std::optional<ExecutionPriorityProgress> progress{};
    // Present only with Execute and exactly equal to the observed command.
    // It never authorizes issue_next, resume-all or a native dependency cut.
    std::optional<ExecutionPriorityCommand> command{};
};
struct ExecutionPriorityPlan {
    std::array<ExecutionPriorityDecision, kMaxExecutionPriorityPeers> decisions{};
    std::size_t size{};
    std::optional<std::uint32_t> selected_priority;
    bool is_valid{};
};
struct ExecutionPriorityIdleConfig {
    std::uint32_t priority{};
    std::uint64_t duration_ns{};
};
// Serialized pure policy: timestamps and observations come from the trusted
// resource owner. A successful issue arms the guard even when a short command
// retires before the next snapshot. Only healthy queue/physical drain starts
// the interval. This narrows ordinary grants; it never proves physical stop.
class ExecutionPriorityIdleWindow final {
   public:
    explicit ExecutionPriorityIdleWindow(ExecutionPriorityIdleConfig config = {}) noexcept;
    [[nodiscard]] bool enabled() const noexcept;
    void note_issue(std::uint32_t priority) noexcept;
    [[nodiscard]] ExecutionPriorityPlan apply(std::span<const ExecutionPriorityPeer> peers,
                                              ExecutionSchedulingMode mode, std::uint64_t now_ns,
                                              ExecutionPriorityPlan plan) noexcept;
    [[nodiscard]] std::uint64_t remaining_ns(std::uint64_t now_ns) const noexcept;
    [[nodiscard]] bool has_idle_window() const noexcept;

   private:
    ExecutionPriorityIdleConfig config_;
    std::optional<std::uint64_t> idle_since_ns_, last_now_ns_;
    bool has_activity_{};
};
struct ExecutionPairLeg {
    std::uint64_t session{};
    ExecutionPriorityCommand command;
    std::uint64_t catalog_generation{};
    unsigned int workers{};
    bool operator==(const ExecutionPairLeg&) const = default;
};
struct ExecutionPairProposal {
    std::array<ExecutionPairLeg, 2> legs{};
    bool is_shared{};
    std::size_t first_serial{};
};
struct ExecutionPairPlan {
    ExecutionPriorityPlan priority;
    std::array<ExecutionPairLeg, 2> grants{};
    std::size_t grant_count{};
    bool is_valid{}, is_shared{};
};
struct ExecutionRebalanceTrigger {
    std::uint64_t session{};
    ExecutionPriorityCommand command;
};
struct ExecutionRebalanceLeg {
    std::uint64_t session{};
    ExecutionPriorityPhysical physical;
};
struct ExecutionRebalanceProposal {
    // Required to start a checkpoint. Absence ONLY maintains an existing
    // owner-retained checkpoint after cancellation; never mints ordinary issue.
    std::optional<ExecutionRebalanceTrigger> trigger{};
    std::array<ExecutionRebalanceLeg, 2> legs{};
    std::size_t size{};
};

// Stop-only same-priority checkpoint. Validates exact ready initial trigger and
// ordinary physical epochs, requests retreat and withholds all new ordinary
// Execute grants. Valid native dependency cuts remain exact; unknown quarantine
// and higher numeric priorities dominate. The owner alone tracks checkpoint
// lifetime/arrival consumption and actual backend physical stop/retargeting.
[[nodiscard]] ExecutionPriorityPlan plan_execution_rebalance(
    std::span<const ExecutionPriorityPeer> peers,
    const ExecutionRebalanceProposal& proposal) noexcept;

// Select only the exact two measured/shared legs or one exact serial fallback.
// Every ordinary Execute outside returned grants becomes Wait; dependency-cut
// progress, retreat and quarantine stay distinct. Any physical peer prevents a
// new pair. Shared work needs two distinct members of one materialized domain;
// serial original fallback (workers zero) is forbidden for partial continuation.
// This is not capacity, GPU-concurrency, physical-stop or native issue authority.
// The owner still revalidates its opaque commands/resources at actual issue.
[[nodiscard]] ExecutionPairPlan plan_execution_pair(
    std::span<const ExecutionPriorityPeer> peers, const ExecutionPairProposal& proposal,
    ExecutionSchedulingMode mode = ExecutionSchedulingMode::kPriority) noexcept;

// Pure policy, not execution authority. The serialized resource owner supplies
// actual stop observations (never client claims or retreat intent), revalidates
// its resources and applies decisions. No SDK, transport, callbacks or clock.
// Equal highest-priority peers may pipeline while physical work is outstanding.
// A valid progress binding can authorize only a dependency of an existing known
// reservation when ordinary policy would wait/retreat. Any unknown state still
// quarantines the entire plan. Stale/missing/idle bindings reject the snapshot;
// policy never guesses a generation or broadens a dependency to general issue.
[[nodiscard]] ExecutionPriorityPlan plan_execution_priority(
    std::span<const ExecutionPriorityPeer> peers) noexcept;
// Naive authorizes all ready/paused peers, but never bypasses unknown ownership
// or the resource owner's per-stream dependencies and physical retirement.
// Profile waits for all physical work and selects one paused/ready peer. This
// instrumented isolation mode requires independently terminating operations.
// Workload-agnostic permits concurrent ready/paused peers like naive. Its
// resource-constrained variant selection belongs to the backend; this plan
// grants neither CUDA ownership nor hardware co-residency.
// Priority-space-share permits original work without software retreat. The
// highest represented priority is retained even while idle. The serialized
// owner drains high work first and bounds each lower client to one ordinary
// issue per turn, retaining exact dependency-progress authority separately.
// This mode neither applies the preemptive idle window nor grants paired tuning.
[[nodiscard]] ExecutionPriorityPlan plan_execution_priority(
    std::span<const ExecutionPriorityPeer> peers, ExecutionSchedulingMode mode) noexcept;

}  // namespace glimmer::core
