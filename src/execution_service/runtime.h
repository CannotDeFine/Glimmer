#pragma once

#include "sgemm_issue_range.h"
#include "glimmer/control/execution_socket.h"
#include "glimmer/core/execution_priority.h"
#include "internal/cuda_execution_session.h"
#include "internal/cuda_ptx_cache.h"
#include "internal/cuda_profile_cache.h"
#include "internal/cuda_pair_profile_cache.h"
#ifdef GLIMMER_HAS_NCCL
#include "internal/cuda_nccl_native_call.h"
#include "internal/cuda_nccl_primary_domain.h"
#endif

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace glimmer::app {
class HostStageMetrics;
class IssueDepthMetrics;
#ifdef GLIMMER_HAS_NCCL
struct ExecutionNcclDispatch {
    backend::cuda::CudaNcclNativeDispatch native{};
    backend::cuda::CudaNcclPrimaryDomainDispatch primary{};
};
#endif
enum class ExecutionClientKernelMode : std::uint8_t { kAuto, kOriginal, kConfigured };
enum class IssueDepthComputeClass { kExcluded, kIncluded, kUnsupported };
// Diagnostic scope only; this does not select or authorize application work.
[[nodiscard]] constexpr IssueDepthComputeClass issue_depth_compute_class(
    control::ExecutionOpcode operation) noexcept {
    using Opcode = control::ExecutionOpcode;
    switch (operation) {
        case Opcode::kLaunch:
        case Opcode::kSgemm:
        case Opcode::kGemmEx:
        case Opcode::kStridedGemmEx:
        case Opcode::kStridedSgemm:
        case Opcode::kLtMatmul:
        case Opcode::kGeam:
        case Opcode::kSgemv:
        case Opcode::kSgemmEx:
            return IssueDepthComputeClass::kIncluded;
        case Opcode::kGraphLaunch:
        case Opcode::kDnnCompute:
        case Opcode::kDnnConvolutionForward:
        case Opcode::kDnnConvolutionFind:
        case Opcode::kDnnTensorForward:
        case Opcode::kDnnReduction:
        case Opcode::kDnnBatchNorm:
        case Opcode::kDnnDropout:
        case Opcode::kDnnRnnCompute:
        case Opcode::kDnnLegacyCompute:
        case Opcode::kDnnAttnCompute:
        case Opcode::kDnnBackendExecute:
        case Opcode::kDnnReorderFilterAndBias:
            return IssueDepthComputeClass::kUnsupported;
        default:
            return IssueDepthComputeClass::kExcluded;
    }
}

// CPU admission failures have no original SDK result. Callers retain their
// family-specific fallback and handle ordinary queue-full retry separately.
[[nodiscard]] constexpr CUresult admission_rejection_status(backend::ExecutionAcceptStatus status,
                                                            CUresult fallback) noexcept {
    switch (status) {
        case backend::ExecutionAcceptStatus::kOutOfMemory:
            return CUDA_ERROR_OUT_OF_MEMORY;
        case backend::ExecutionAcceptStatus::kClosed:
            return CUDA_ERROR_CONTEXT_IS_DESTROYED;
        case backend::ExecutionAcceptStatus::kQuarantined:
            return CUDA_ERROR_UNKNOWN;
        default:
            return fallback;
    }
}

struct ExecutionClientPriority {
    control::TaskPeerIdentity peer;
    std::uint32_t priority{};
    // Trusted startup selection, independent of numeric scheduling priority.
    ExecutionClientKernelMode kernel_mode{ExecutionClientKernelMode::kAuto};
};
// Trusted owner-side assembly, never supplied by a wire/client. The official
// workload-aware CLI creates an empty uninitialized domain; existing private
// assembly supplies an empty initialized domain. The service takes exclusive
// lifecycle responsibility. External shared views may inspect, never mutate it.
// General public context compatibility and whole-application acceptance remain gates.
struct ExecutionServicePairConfig {
    std::shared_ptr<backend::cuda::CudaExecutionDomainOwner> domain;
    backend::cuda::CudaPairProfileConfig profile;
    std::string profile_input;
    std::vector<unsigned int> workers{80, 160};
    bool is_isolated_profile{};
    std::string profile_output{};
    // Explicit trusted policy for same-priority initial PTX arrivals. Ordinary
    // workload-aware CLI mode enables it; isolated pair sampling never does.
    bool should_rebalance_on_arrival{};
    // Explicit official-service cold ownership. The supplied domain is empty
    // and uninitialized; initialize only after ALL CPU startup checks succeed.
    // Default false preserves trusted already-initialized private assembly.
    bool should_initialize_domain{};
};
// Trusted common ownership without measured-pair policy or tuning authority.
struct ExecutionServiceSharedConfig {
    std::shared_ptr<backend::cuda::CudaExecutionDomainOwner> domain{};
    bool should_initialize_domain{};
};
// Diagnostics only: the identity is scoped to one service execution.
struct ExecutionSgemmAuditTarget {
    std::uint64_t session{}, sequence{};
    // Exactly one of session or client selects the ordinary compute connection.
    std::optional<control::TaskPeerIdentity> client{};
};
struct ExecutionServiceConfig {
    control::ExecutionSocketConfig transport;
    backend::cuda::CudaSessionLimits session_limits;
    backend::cuda::CudaSessionPersistentConfig persistent;
    std::vector<ExecutionClientPriority> client_priorities;
    int device_ordinal{};
    bool exit_when_idle{};
    bool should_audit_native{};
    std::optional<ExecutionSgemmAuditTarget> sgemm_audit_target;
    std::optional<ExecutionSgemmRangeHooks> sgemm_range_hooks;
    bool should_audit_priority{};
    core::ExecutionSchedulingMode scheduling_mode{core::ExecutionSchedulingMode::kPriority};
    bool uses_original_kernels{};
    backend::cuda::CudaPtxCacheLimits ptx_cache;
    backend::cuda::CudaProfileConfig profile;
    std::string profile_input, profile_output;
    std::string ptx_cache_input, ptx_cache_output;
    bool uses_nccl{};
    // Includes staged foreign waits and accepted edges through target retirement.
    std::size_t maximum_foreign_event_edges{4096};
    std::optional<ExecutionServicePairConfig> paired;
    bool should_audit_copy_requests{};
    // Trusted policy/clock, never client input. Zero disables the idle window.
    std::uint32_t priority_idle_us{100};
    std::uint64_t (*priority_clock_ns)() noexcept {};
    std::optional<ExecutionServiceSharedConfig> shared{};
    // Optional owner-thread observation; never scheduling or backend authority.
    std::shared_ptr<HostStageMetrics> host_stage_metrics{};
    std::shared_ptr<IssueDepthMetrics> issue_depth_metrics{};
    // Opt-in local no-op elimination; never removes a peer from global policy.
    bool should_skip_idle_helper_plans{};
    // Trusted, default-off two-round intake for ordinary equal-priority ASY1.
    // At most one extra frame per eligible peer; no issue or completion grant.
    bool should_receive_async_burst{};
};
struct ExecutionBlockCompletionObservation {
    std::uint64_t session{}, sequence{};
    backend::cuda::CudaSessionPersistentProgress progress;
};

// Application composition layer: authenticated CPU transport plus the native
// GPU owner and bounded independent host-registration actors. Public methods
// and lifecycle run on the constructing thread/process; no CUDA runs under
// actor metadata locks or on transport workers.
// Trusted scheduler selection uses Core policy and backend stop proof; naive
// admission does not bypass dependencies, peer ownership or physical cleanup.
// Transport/client claims cannot mint priority or physical completion.
class ExecutionService final {
   public:
    ExecutionService(ExecutionServiceConfig config, backend::cuda::CudaSessionDispatch dispatch);
    // Explicit diagnostic ABI; the normal constructors/configuration stay
    // unchanged. The journal must be fresh and owned by this service thread.
    ExecutionService(std::shared_ptr<control::ExecutionSocketPollJournal> journal,
                     ExecutionServiceConfig config, backend::cuda::CudaSessionDispatch dispatch);
#ifdef GLIMMER_HAS_NCCL
    ExecutionService(ExecutionServiceConfig config, backend::cuda::CudaSessionDispatch dispatch,
                     ExecutionNcclDispatch nccl);
    ExecutionService(std::shared_ptr<control::ExecutionSocketPollJournal> journal,
                     ExecutionServiceConfig config, backend::cuda::CudaSessionDispatch dispatch,
                     ExecutionNcclDispatch nccl);
#endif
    ~ExecutionService();
    ExecutionService(const ExecutionService&) = delete;
    ExecutionService& operator=(const ExecutionService&) = delete;
    [[nodiscard]] bool start() noexcept;
    [[nodiscard]] bool step(std::uint32_t wait_ms = 1) noexcept;
    [[nodiscard]] bool finished() const noexcept;
    // An independent terminal receipt: a logged end marker alone cannot prove
    // that a destination accepted or flushed all diagnostics.
    [[nodiscard]] bool audit_healthy() const noexcept;
    // Owner-only aggregate of successful second-round transport transfers.
    // Disabled/foreign-owner reads have no value; UINT64_MAX means saturated.
    // This is not evidence of admission, issue or physical completion.
    [[nodiscard]] std::optional<std::uint64_t> async_intake_second_takes() const noexcept;
    [[nodiscard]] backend::cuda::CudaPtxCacheSnapshot ptx_cache_snapshot() const noexcept;
    [[nodiscard]] backend::cuda::CudaProfileSnapshot profile_snapshot() const noexcept;
    [[nodiscard]] backend::cuda::CudaPairProfileSnapshot pair_profile_snapshot() const noexcept;
    [[nodiscard]] CUresult pair_profile_status() const noexcept;
    // Owner-only CPU diagnostic: authenticated received requests, not a grant,
    // native-entry observation, or physical-stop/priority qualification.
    [[nodiscard]] std::size_t pending_priority_requests(std::uint32_t priority) const noexcept;
    // Owner-thread readonly pending ASY1 metadata, not preparation/issue/stop
    // authority. Unknown/fenced state has no count, rather than a fabricated zero.
    [[nodiscard]] std::optional<std::size_t> pending_priority_preparations(
        std::uint32_t priority) const noexcept;
    // Exact accepted ASY1 epochs still awaiting their first issue outcome.
    // Known host outcomes are excluded even while physical leases remain.
    // CPU readiness is not native configuration or execution authority.
    [[nodiscard]] std::optional<std::size_t> pending_priority_async_admissions(
        std::uint32_t priority) const noexcept;
    // CPU-only, exact trusted peer, owner-thread diagnostic. Publication is
    // independent of stop and never changes counters or physical authority.
    // Unissued/failed, retreat-requested/waiting and paused generations are excluded.
    [[nodiscard]] std::optional<ExecutionBlockCompletionObservation> block_completion_observation(
        const control::TaskPeerIdentity& peer) const noexcept;
    // Publish an already observed witness in the same ordered diagnostics FIFO.
    [[nodiscard]] bool publish_block_completion_audit(
        const control::TaskPeerIdentity& peer, std::uint32_t priority,
        const ExecutionBlockCompletionObservation& observation) noexcept;
    [[nodiscard]] bool save_profile() const noexcept;
    [[nodiscard]] bool save_pair_profile() const noexcept;
    [[nodiscard]] bool save_ptx_cache() const noexcept;
    [[nodiscard]] bool request_stop() noexcept;

   private:
    struct Implementation;
    std::unique_ptr<Implementation> implementation_;
};
}  // namespace glimmer::app
