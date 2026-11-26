#include "cuda_admission_coordinator.h"
#include "diagnostics.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <utility>

namespace glimmer::interceptor {
namespace {
template <typename Value>
bool ready(const std::future<Value>& future) {
    return future.valid() &&
           future.wait_for(std::chrono::milliseconds::zero()) == std::future_status::ready;
}
}  // namespace

struct CudaAdmissionCoordinator::Implementation {
    enum class Phase {
        kUnsubmitted,
        kSubmitting,
        kPending,
        kClaiming,
        kRunning,
        kRetreating,
        kHeartbeating,
        kSuspending,
        kCompleting,
        kFailing,
        kStopping,
        kTerminal,
        kUncertain
    };
    struct Entry {
        CudaOperationToken token;
        control::CooperativeLaunchRequest request;
        Phase phase{Phase::kUnsubmitted};
        core::TaskId task{};
        control::TaskExecutionIdentity execution{};
        std::future<control::CooperativeSubmission> submission;
        std::future<control::CooperativeClaim> claim;
        std::future<std::optional<bool>> retreat;
        std::future<bool> report;
        std::chrono::steady_clock::time_point last_heartbeat{};
        std::chrono::steady_clock::time_point next_claim{
            std::chrono::steady_clock::time_point::min()};
        std::chrono::steady_clock::time_point next_retreat{
            std::chrono::steady_clock::time_point::min()};
        bool needs_resume{};
        bool can_retreat{};
        bool has_retreat_request{};
        bool has_suspended_lease{};
        bool is_native_execution{};
        bool is_native_reservation{};
        CUresult native_status{CUDA_SUCCESS};
        CUresult stopped_token_result{CUDA_ERROR_NOT_READY};
        bool is_stopped_success{};
        bool is_stopped_failure{};
    };
    std::shared_ptr<control::AsyncCooperativeLaunchGate> gate;
    const std::size_t limit;
    const CudaAdmissionOptions options;
    std::vector<Entry> entries;
    CudaCoordinationStats stats;
    CUresult status{CUDA_SUCCESS};
    bool is_stopping{};
    bool has_physical_stop{};

    Implementation(std::shared_ptr<control::AsyncCooperativeLaunchGate> value,
                   CudaAdmissionOptions settings)
        : gate(std::move(value)), limit(settings.pending_limit), options(settings) {
        if (!gate || limit == 0 || limit > 65536 || !options.clock_now ||
            options.heartbeat_interval.count() <= 0 ||
            options.request_retry_interval.count() <= 0 ||
            options.request_retry_interval > std::chrono::seconds{1} ||
            options.native_request_retry_interval.count() <= 0 ||
            options.native_request_retry_interval > std::chrono::seconds{1}) {
            throw std::invalid_argument("invalid CUDA admission coordinator configuration");
        }
        entries.reserve(limit);
    }

    std::chrono::steady_clock::time_point retry_deadline(
        std::chrono::steady_clock::duration interval) const noexcept {
        const auto now = options.clock_now();
        const auto maximum = std::chrono::steady_clock::time_point::max();
        return now > maximum - interval ? maximum : now + interval;
    }

    void report_request(const Entry& entry, const control::CooperativeClaim& claim,
                        bool is_submission) const noexcept {
        if (!entry.request.should_measure_timing) {
            return;
        }
        const auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch());
        report_cooperative_control_timing_diagnostic(
            {.task_id = entry.task,
             .generation = claim.execution.generation,
             .is_submission = is_submission,
             .claim_status = static_cast<std::uint32_t>(claim.status),
             .enqueued_nanoseconds = claim.timing.enqueued_nanoseconds,
             .started_nanoseconds = claim.timing.started_nanoseconds,
             .finished_nanoseconds = claim.timing.finished_nanoseconds,
             .observed_nanoseconds =
                 duration.count() > 0 ? static_cast<std::uint64_t>(duration.count()) : 0});
    }

    bool accept_claim(Entry& entry, const control::CooperativeClaim& claim) {
        if (claim.status == control::CooperativeClaimStatus::kFailed) {
            return false;
        }
        if (claim.status == control::CooperativeClaimStatus::kPending) {
            entry.next_claim =
                retry_deadline(entry.is_native_reservation ? options.native_request_retry_interval
                                                           : options.request_retry_interval);
            entry.phase = Phase::kPending;
            return true;
        }
        if (claim.execution.task_id != entry.task || claim.execution.generation == 0 ||
            (entry.needs_resume && claim.execution.generation == entry.execution.generation)) {
            return false;
        }
        entry.execution = claim.execution;
        entry.has_suspended_lease = false;
        entry.phase = Phase::kRunning;
        entry.last_heartbeat = options.clock_now();
        return true;
    }

    bool publish_grant(CudaOperationQueue& queue, Entry& entry) {
        if (entry.phase != Phase::kRunning) {
            return true;
        }
        ++stats.granted_executions;
        queue.execution_granted(entry.token, {entry.execution.task_id, entry.execution.generation});
        if (entry.needs_resume) {
            entry.needs_resume = false;
            if (!queue.resume(entry.token)) {
                return false;
            }
            ++stats.logical_resumes;
        }
        return true;
    }

    bool consume(CudaOperationQueue& queue, Entry& entry) {
        switch (entry.phase) {
            case Phase::kSubmitting:
                if (ready(entry.submission)) {
                    entry.phase = Phase::kUncertain;
                    const auto submission = entry.submission.get();
                    entry.task = submission.task_id;
                    report_request(entry, submission.claim, true);
                    if (submission.task_id == 0) {
                        return false;
                    }
                    if (!accept_claim(entry, submission.claim) || !publish_grant(queue, entry)) {
                        return false;
                    }
                }
                break;
            case Phase::kClaiming:
                if (ready(entry.claim)) {
                    entry.phase = Phase::kUncertain;
                    const auto claim = entry.claim.get();
                    report_request(entry, claim, false);
                    if (!accept_claim(entry, claim) || !publish_grant(queue, entry)) {
                        return false;
                    }
                }
                break;
            case Phase::kRetreating:
                if (ready(entry.retreat)) {
                    entry.phase = Phase::kUncertain;
                    const auto requested = entry.retreat.get();
                    if (!requested) {
                        return false;
                    }
                    entry.phase = Phase::kRunning;
                    entry.next_retreat = retry_deadline(options.request_retry_interval);
                    if (*requested && entry.can_retreat && !entry.has_retreat_request &&
                        queue.state(entry.token) == OrderedOperationState::kRunning) {
                        const auto value = queue.request_retreat(entry.token, true);
                        if (value != CUDA_SUCCESS) {
                            status = value;
                            return false;
                        }
                        entry.has_retreat_request = true;
                        ++stats.retreat_requests;
                    }
                }
                break;
            case Phase::kHeartbeating:
                if (ready(entry.report)) {
                    entry.phase = Phase::kUncertain;
                    if (!entry.report.get()) {
                        return false;
                    }
                    entry.last_heartbeat = options.clock_now();
                    ++stats.acknowledged_heartbeats;
                    entry.phase = entry.needs_resume ? Phase::kPending : Phase::kRunning;
                }
                break;
            case Phase::kSuspending:
                if (ready(entry.report)) {
                    entry.phase = Phase::kUncertain;
                    if (!entry.report.get()) {
                        return false;
                    }
                    entry.phase = Phase::kPending;
                    entry.has_suspended_lease = true;
                    entry.last_heartbeat = options.clock_now();
                    entry.has_retreat_request = false;
                    ++stats.acknowledged_suspensions;
                }
                break;
            case Phase::kCompleting:
            case Phase::kFailing: {
                const auto is_failure = entry.phase == Phase::kFailing;
                if (ready(entry.report)) {
                    entry.phase = Phase::kUncertain;
                    if (!entry.report.get()) {
                        return false;
                    }
                    entry.phase = Phase::kTerminal;
                    if (is_failure) {
                        ++stats.acknowledged_failures;
                    } else {
                        ++stats.acknowledged_completions;
                    }
                }
                break;
            }
            default:
                break;
        }
        return true;
    }

    bool advance(CudaOperationQueue& queue, Entry& entry,
                 const std::vector<CudaKernelCandidate>& candidates,
                 std::vector<CudaOperationToken>& selection) {
        if (!consume(queue, entry)) {
            return false;
        }
        const auto state = queue.state(entry.token);
        const auto candidate =
            std::ranges::find(candidates, entry.token, &CudaKernelCandidate::token);
        if (candidate != candidates.end()) {
            entry.can_retreat = candidate->can_retreat;
        }
        if (entry.phase == Phase::kUnsubmitted && state == OrderedOperationState::kFailed) {
            // A failed initialization/dependency never reached control admission.
            // Retain the failed token result, not an unbounded coordinator entry.
            entry.phase = Phase::kTerminal;
            return true;
        }
        if (entry.phase == Phase::kUnsubmitted && candidate != candidates.end() &&
            state == OrderedOperationState::kQueued) {
            entry.submission = gate->submit_and_claim(entry.request);
            if (!entry.submission.valid()) {
                return false;
            }
            entry.phase = Phase::kSubmitting;
        } else if (entry.phase == Phase::kPending) {
            if (entry.needs_resume &&
                options.clock_now() - entry.last_heartbeat >= options.heartbeat_interval) {
                entry.report = gate->heartbeat(entry.execution);
                if (!entry.report.valid()) {
                    return false;
                }
                entry.phase = Phase::kHeartbeating;
            } else if (options.clock_now() >= entry.next_claim) {
                entry.claim = gate->claim(entry.task, entry.request.should_measure_timing);
                if (!entry.claim.valid()) {
                    return false;
                }
                entry.phase = Phase::kClaiming;
            }
        } else if (entry.phase == Phase::kRunning) {
            if (state == OrderedOperationState::kSucceeded) {
                entry.report = entry.native_status == CUDA_SUCCESS ? gate->complete(entry.execution)
                                                                   : gate->fail(entry.execution);
                if (!entry.report.valid()) {
                    return false;
                }
                entry.phase =
                    entry.native_status == CUDA_SUCCESS ? Phase::kCompleting : Phase::kFailing;
            } else if (state == OrderedOperationState::kSuspended) {
                if (entry.is_native_execution) {
                    return false;
                }
                entry.report = gate->suspend(entry.execution);
                if (!entry.report.valid()) {
                    return false;
                }
                entry.needs_resume = true;
                entry.phase = Phase::kSuspending;
            } else if (state == OrderedOperationState::kQueued) {
                if (entry.is_native_reservation) {
                    if (options.clock_now() - entry.last_heartbeat >= options.heartbeat_interval) {
                        entry.report = gate->heartbeat(entry.execution);
                        if (!entry.report.valid()) {
                            return false;
                        }
                        entry.phase = Phase::kHeartbeating;
                    }
                } else if (!entry.is_native_execution) {
                    selection.push_back(entry.token);
                }
            } else if (state == OrderedOperationState::kRunning) {
                if (options.clock_now() - entry.last_heartbeat >= options.heartbeat_interval) {
                    entry.report = gate->heartbeat(entry.execution);
                    if (!entry.report.valid()) {
                        return false;
                    }
                    entry.phase = Phase::kHeartbeating;
                } else if (entry.can_retreat && !entry.has_retreat_request &&
                           options.clock_now() >= entry.next_retreat) {
                    entry.retreat = gate->retreat_requested(entry.execution);
                    if (!entry.retreat.valid()) {
                        return false;
                    }
                    entry.phase = Phase::kRetreating;
                }
            } else {
                return false;
            }
        }
        return true;
    }

    CUresult stopped_step(Entry& entry) {
        // No queue/CUDA call here. Physical token metadata was classified once
        // after successful shutdown; originals retain their exact phase.
        switch (entry.phase) {
            case Phase::kUnsubmitted:
            case Phase::kTerminal:
                entry.phase = Phase::kTerminal;
                return CUDA_SUCCESS;
            case Phase::kUncertain:
                return CUDA_ERROR_UNKNOWN;
            case Phase::kSubmitting: {
                if (!ready(entry.submission)) return CUDA_ERROR_NOT_READY;
                entry.phase = Phase::kUncertain;
                const auto submitted = entry.submission.get();
                entry.task = submitted.task_id;
                if (!entry.task || !accept_claim(entry, submitted.claim)) return CUDA_ERROR_UNKNOWN;
                break;
            }
            case Phase::kClaiming: {
                if (!ready(entry.claim)) return CUDA_ERROR_NOT_READY;
                entry.phase = Phase::kUncertain;
                const auto claimed = entry.claim.get();
                if (!accept_claim(entry, claimed)) return CUDA_ERROR_UNKNOWN;
                if (claimed.status == control::CooperativeClaimStatus::kGranted)
                    entry.needs_resume = false;
                break;
            }
            case Phase::kRetreating:
                if (!ready(entry.retreat)) return CUDA_ERROR_NOT_READY;
                entry.phase = Phase::kUncertain;
                if (!entry.retreat.get()) return CUDA_ERROR_UNKNOWN;
                entry.phase = Phase::kRunning;
                break;
            case Phase::kHeartbeating:
                if (!ready(entry.report)) return CUDA_ERROR_NOT_READY;
                entry.phase = Phase::kUncertain;
                if (!entry.report.get()) return CUDA_ERROR_UNKNOWN;
                entry.phase = entry.needs_resume ? Phase::kPending : Phase::kRunning;
                break;
            case Phase::kSuspending:
                if (!ready(entry.report)) return CUDA_ERROR_NOT_READY;
                entry.phase = Phase::kUncertain;
                if (!entry.report.get()) return CUDA_ERROR_UNKNOWN;
                entry.phase = Phase::kPending;
                entry.has_suspended_lease = true;
                ++stats.acknowledged_suspensions;
                break;
            case Phase::kCompleting:
            case Phase::kFailing:
            case Phase::kStopping: {
                if (!ready(entry.report)) return CUDA_ERROR_NOT_READY;
                const bool is_success =
                    entry.phase == Phase::kCompleting || entry.is_stopped_success;
                const bool is_failure = entry.phase == Phase::kFailing || entry.is_stopped_failure;
                entry.phase = Phase::kUncertain;
                if (!entry.report.get()) return CUDA_ERROR_UNKNOWN;
                if (is_success) ++stats.acknowledged_completions;
                if (is_failure) ++stats.acknowledged_failures;
                entry.phase = Phase::kTerminal;
                return CUDA_SUCCESS;
            }
            default:
                break;
        }
        if (entry.task == 0) return CUDA_ERROR_UNKNOWN;
        if (entry.execution.generation == 0)
            entry.report = gate->cancel_pending(entry.task);
        else if (entry.has_suspended_lease)
            entry.report = gate->cancel_stopped(entry.execution);
        else {
            entry.is_stopped_success =
                entry.stopped_token_result == CUDA_SUCCESS && entry.native_status == CUDA_SUCCESS;
            entry.is_stopped_failure = !entry.is_stopped_success;
            entry.report = entry.is_stopped_success ? gate->complete(entry.execution)
                                                    : gate->fail(entry.execution);
        }
        entry.phase = entry.report.valid() ? Phase::kStopping : Phase::kUncertain;
        return entry.report.valid() ? CUDA_ERROR_NOT_READY : CUDA_ERROR_UNKNOWN;
    }

    bool finish_stopped(CudaOperationQueue& queue, Entry& entry) {
        // This may wait for bounded control I/O ONLY after physical queue stop.
        if (entry.phase == Phase::kSubmitting) {
            entry.phase = Phase::kUncertain;
            const auto submission = entry.submission.get();
            if (submission.task_id == 0) {
                return false;
            }
            entry.task = submission.task_id;
            if (!accept_claim(entry, submission.claim)) {
                return false;
            }
        } else if (entry.phase == Phase::kClaiming) {
            entry.phase = Phase::kUncertain;
            const auto claim = entry.claim.get();
            if (!accept_claim(entry, claim)) {
                return false;
            }
            if (claim.status == control::CooperativeClaimStatus::kGranted) {
                entry.needs_resume = false;
            }
        } else if (entry.phase == Phase::kRetreating) {
            static_cast<void>(entry.retreat.get());
            entry.phase = Phase::kRunning;
        } else if (entry.phase == Phase::kHeartbeating) {
            static_cast<void>(entry.report.get());
            entry.phase = entry.needs_resume ? Phase::kPending : Phase::kRunning;
        } else if (entry.phase == Phase::kSuspending || entry.phase == Phase::kCompleting ||
                   entry.phase == Phase::kFailing) {
            if (!entry.report.get()) {
                return false;
            }
            if (entry.phase == Phase::kCompleting || entry.phase == Phase::kFailing) {
                const auto is_failure = entry.phase == Phase::kFailing;
                entry.phase = Phase::kTerminal;
                if (is_failure) {
                    ++stats.acknowledged_failures;
                } else {
                    ++stats.acknowledged_completions;
                }
                return true;
            }
            entry.phase = Phase::kPending;
            entry.has_suspended_lease = true;
            ++stats.acknowledged_suspensions;
        }
        if (entry.phase == Phase::kUnsubmitted || entry.phase == Phase::kTerminal) {
            entry.phase = Phase::kTerminal;
            return true;
        }
        if (entry.task == 0) {
            return false;
        }
        std::future<bool> terminal;
        bool should_report_success = false;
        bool should_report_failure = false;
        if (entry.execution.generation == 0) {
            terminal = gate->cancel_pending(entry.task);
        } else if (entry.has_suspended_lease) {
            terminal = gate->cancel_stopped(entry.execution);
        } else {
            should_report_success =
                queue.query(entry.token) == CUDA_SUCCESS && entry.native_status == CUDA_SUCCESS;
            should_report_failure = !should_report_success;
            terminal = should_report_success ? gate->complete(entry.execution)
                                             : gate->fail(entry.execution);
        }
        if (!terminal.valid() || !terminal.get()) {
            return false;
        }
        if (should_report_success) {
            ++stats.acknowledged_completions;
        } else if (should_report_failure) {
            ++stats.acknowledged_failures;
        }
        entry.phase = Phase::kTerminal;
        return true;
    }
};

CudaAdmissionCoordinator::CudaAdmissionCoordinator(
    std::shared_ptr<control::AsyncCooperativeLaunchGate> gate, CudaAdmissionOptions options)
    : implementation_(std::make_unique<Implementation>(std::move(gate), options)) {}
CudaAdmissionCoordinator::~CudaAdmissionCoordinator() = default;
bool CudaAdmissionCoordinator::adopt_native_execution(CudaOperationQueue& queue,
                                                      const CudaOperationToken& fence,
                                                      control::TaskExecutionIdentity execution,
                                                      CUresult native_status) noexcept {
    auto& state = *implementation_;
    if (state.status != CUDA_SUCCESS || state.entries.size() >= state.limit || !execution.task_id ||
        !execution.generation || !queue.native_fence_is_ready(fence) ||
        queue.state(fence) != OrderedOperationState::kQueued ||
        std::ranges::any_of(state.entries, [&](const auto& entry) {
            return entry.token == fence || entry.task == execution.task_id;
        })) {
        return false;
    }
    try {
        Implementation::Entry entry;
        entry.token = fence;
        entry.task = execution.task_id;
        entry.execution = execution;
        entry.phase = Implementation::Phase::kRunning;
        entry.last_heartbeat = state.options.clock_now();
        entry.is_native_execution = true;
        entry.native_status = native_status;
        state.entries.push_back(std::move(entry));
        ++state.stats.granted_executions;
        return true;
    } catch (...) {
        return false;
    }
}
bool CudaAdmissionCoordinator::track(CudaOperationQueue& queue, const CudaOperationToken& kernel,
                                     control::CooperativeLaunchRequest request) noexcept {
    auto& state = *implementation_;
    if (state.status != CUDA_SUCCESS || state.entries.size() >= state.limit ||
        queue.state(kernel) != OrderedOperationState::kQueued ||
        std::ranges::find(state.entries, kernel, &Implementation::Entry::token) !=
            state.entries.end()) {
        return false;
    }
    try {
        Implementation::Entry entry;
        entry.token = kernel;
        entry.request = request;
        entry.is_native_reservation = queue.is_native_reservation(kernel);
        state.entries.push_back(std::move(entry));
        return true;
    } catch (...) {
        return false;
    }
}
std::optional<control::TaskExecutionIdentity> CudaAdmissionCoordinator::native_reservation_grant(
    CudaOperationQueue& queue, const CudaOperationToken& reservation) const noexcept {
    const auto& state = *implementation_;
    const auto entry = std::ranges::find(state.entries, reservation, &Implementation::Entry::token);
    if (state.status != CUDA_SUCCESS || entry == state.entries.end() ||
        !entry->is_native_reservation || !queue.native_reservation_is_ready(reservation) ||
        (entry->phase != Implementation::Phase::kRunning &&
         entry->phase != Implementation::Phase::kHeartbeating)) {
        return std::nullopt;
    }
    return entry->execution;
}
bool CudaAdmissionCoordinator::commit_native_reservation(CudaOperationQueue& queue,
                                                         const CudaOperationToken& reservation,
                                                         control::TaskExecutionIdentity execution,
                                                         CudaQueuedNativeFence fence,
                                                         CUresult native_status) noexcept {
    auto& state = *implementation_;
    const auto grant = native_reservation_grant(queue, reservation);
    if (!grant || grant->task_id != execution.task_id ||
        grant->generation != execution.generation ||
        !queue.replace_native_reservation(reservation, std::move(fence))) {
        return false;
    }
    auto entry = std::ranges::find(state.entries, reservation, &Implementation::Entry::token);
    entry->is_native_reservation = false;
    entry->is_native_execution = true;
    entry->native_status = native_status;
    return true;
}
CUresult CudaAdmissionCoordinator::progress(CudaOperationQueue& queue) noexcept {
    auto& state = *implementation_;
    if (state.status != CUDA_SUCCESS) {
        return state.status;
    }
    try {
        if (const auto value = queue.pump_admitted({}); value != CUDA_SUCCESS) {
            state.status = value;
            return value;
        }
        const auto candidates = queue.kernel_candidates();
        std::vector<CudaOperationToken> selection;
        selection.reserve(state.entries.size());
        for (auto& entry : state.entries) {
            if (!state.advance(queue, entry, candidates, selection)) {
                if (state.status == CUDA_SUCCESS) {
                    state.status = CUDA_ERROR_UNKNOWN;
                }
                return state.status;
            }
        }
        std::erase_if(state.entries, [](const auto& entry) {
            return entry.phase == Implementation::Phase::kTerminal;
        });
        state.status = queue.pump_admitted(selection);
    } catch (...) {
        state.status = CUDA_ERROR_UNKNOWN;
    }
    return state.status;
}
void CudaAdmissionCoordinator::stopped(CudaOperationQueue& queue, CUresult result) noexcept {
    auto& state = *implementation_;
    if (result != CUDA_SUCCESS) {
        state.status = result;
        return;
    }
    if (state.gate->owner_pump_snapshot().is_enabled) {
        state.is_stopping = true;
        state.has_physical_stop = true;
        for (auto& entry : state.entries) entry.stopped_token_result = queue.query(entry.token);
        return;
    }
    for (auto& entry : state.entries) {
        try {
            if (!state.finish_stopped(queue, entry)) {
                state.status = CUDA_ERROR_UNKNOWN;
            }
        } catch (...) {
            state.status = CUDA_ERROR_UNKNOWN;
        }
    }
    std::erase_if(state.entries, [](const auto& entry) {
        return entry.phase == Implementation::Phase::kTerminal;
    });
}
CUresult CudaAdmissionCoordinator::stopped_progress() noexcept {
    auto& state = *implementation_;
    if (!state.gate->owner_pump_snapshot().is_enabled) return state.status;
    if (!state.is_stopping || !state.has_physical_stop) return CUDA_ERROR_UNKNOWN;
    bool has_pending = false;
    try {
        for (auto& entry : state.entries) {
            const auto value = state.stopped_step(entry);
            if (value == CUDA_ERROR_NOT_READY)
                has_pending = true;
            else if (value != CUDA_SUCCESS) {
                if (state.status == CUDA_SUCCESS) state.status = value;
                state.gate->quarantine_owner_pump();
                return state.status;
            }
        }
        std::erase_if(state.entries, [](const auto& entry) {
            return entry.phase == Implementation::Phase::kTerminal;
        });
    } catch (...) {
        if (state.status == CUDA_SUCCESS) state.status = CUDA_ERROR_UNKNOWN;
        state.gate->quarantine_owner_pump();
        return state.status;
    }
    return has_pending ? CUDA_ERROR_NOT_READY : state.status;
}
CudaCoordinationStats CudaAdmissionCoordinator::stats() const noexcept {
    return implementation_->stats;
}
CUresult CudaAdmissionCoordinator::control_status() const noexcept {
    return implementation_->status;
}
CUresult CudaAdmissionCoordinator::native_boundary_status() const noexcept {
    const auto& state = *implementation_;
    return state.status != CUDA_SUCCESS ? state.status
           : state.entries.empty()      ? CUDA_SUCCESS
                                        : CUDA_ERROR_NOT_READY;
}

}  // namespace glimmer::interceptor
