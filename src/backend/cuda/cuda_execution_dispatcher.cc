#include "internal/cuda_execution_dispatcher.h"
#include "internal/cuda_process_memory_owner.h"
#include "internal/cuda_native_execution_lanes.h"
#include "internal/cuda_retirement_observer.h"

#include <algorithm>
#include <atomic>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>
#include <unistd.h>

#include <spdlog/spdlog.h>

namespace glimmer::backend::cuda {
namespace {
thread_local std::shared_ptr<const void> frontier_thread_owner;

struct CallbackScope {
    bool& active;
    explicit CallbackScope(bool& value) noexcept : active(value) {
        active = true;
    }
    ~CallbackScope() {
        active = false;
    }
};
}  // namespace

struct CudaFrontierObservation {
    CUresult failure{CUDA_SUCCESS};
    bool is_settled{};
    std::uint64_t acceptance_order{};
    std::optional<CudaExecutionFrontier> foreign_dependency{};
};
struct CudaFrontierOwner {
    bool is_callback_active{};
    std::shared_ptr<const CudaExecutionOwnerActivity> activity{};
};
struct CudaExecutionFrontier::State {
    std::weak_ptr<const CudaFrontierOwner> owner;
    std::weak_ptr<const void> thread;
    pid_t process{};
    std::uint64_t diagnostic_id{};
    std::vector<std::shared_ptr<const CudaFrontierObservation>> observations;
    // Inspection/progress ancestry is deliberately not physical membership.
    // The target can retire independently of an earlier uncertain predecessor.
    std::shared_ptr<const std::vector<std::shared_ptr<const CudaFrontierObservation>>> predecessors;
};
struct CudaExecutionFrontierReservation::State {
    std::shared_ptr<CudaExecutionFrontier::State> capture;
    std::uint64_t sequence{};
    bool is_bound{};
};
namespace {
enum class EnqueuePhase : unsigned { kAccepted, kReady, kEntered, kComplete, kCancelled, kFailed };
}
struct CudaExecutionEnqueueState {
    pid_t process{};
    std::weak_ptr<const void> owner, thread;
    std::shared_ptr<const void> lifetime;
    std::atomic<EnqueuePhase> phase{EnqueuePhase::kAccepted};
    // Only the original source owner binds/reads this cell. Worker entry never
    // touches it, including when admission completes after callback entry.
    std::optional<CudaNativeExecutionTicket> ticket;
};
CUresult CudaExecutionEnqueueWorker::begin() const noexcept {
    if (!state_) return CUDA_ERROR_INVALID_HANDLE;
    if (::getpid() != state_->process) return CUDA_ERROR_INVALID_CONTEXT;
    if (state_->owner.expired() || state_->thread.expired()) return CUDA_ERROR_INVALID_HANDLE;
    if (frontier_thread_owner && state_->thread.lock() == frontier_thread_owner)
        return CUDA_ERROR_INVALID_CONTEXT;
    auto expected = EnqueuePhase::kReady;
    return state_->phase.compare_exchange_strong(expected, EnqueuePhase::kEntered)
               ? CUDA_SUCCESS
               : CUDA_ERROR_NOT_READY;
}
CUresult CudaExecutionEnqueueWorker::bind_ticket(
    const CudaNativeExecutionTicket& ticket) const noexcept {
    if (!state_) return CUDA_ERROR_INVALID_HANDLE;
    if (::getpid() != state_->process || !frontier_thread_owner ||
        state_->thread.lock() != frontier_thread_owner)
        return CUDA_ERROR_INVALID_CONTEXT;
    if (state_->owner.expired() || state_->ticket) return CUDA_ERROR_INVALID_HANDLE;
    const auto phase = state_->phase.load();
    if (phase != EnqueuePhase::kReady && phase != EnqueuePhase::kEntered)
        return CUDA_ERROR_NOT_READY;
    const auto observed = ticket.observe();
    if (observed.phase == CudaNativeExecutionCallPhase::kCancelled &&
        observed.status == CUDA_ERROR_INVALID_VALUE && !observed.was_invoked)
        return CUDA_ERROR_INVALID_VALUE;
    state_->ticket = ticket;
    return CUDA_SUCCESS;
}
CUresult CudaExecutionFrontier::observe() const noexcept {
    if (!state_) return CUDA_ERROR_INVALID_HANDLE;
    if (::getpid() != state_->process || !frontier_thread_owner ||
        state_->thread.lock() != frontier_thread_owner)
        return CUDA_ERROR_INVALID_CONTEXT;
    if (state_->owner.expired()) return CUDA_ERROR_INVALID_HANDLE;
    bool is_pending{};
    for (const auto& observation : state_->observations) {
        if (observation->failure != CUDA_SUCCESS) return observation->failure;
        is_pending = is_pending || !observation->is_settled;
    }
    return is_pending ? CUDA_ERROR_NOT_READY : CUDA_SUCCESS;
}
CudaExecutionFrontierRetirement CudaExecutionFrontier::retirement() const noexcept {
    if (!state_) return {};
    if (::getpid() != state_->process || !frontier_thread_owner ||
        state_->thread.lock() != frontier_thread_owner)
        return {.status = CUDA_ERROR_INVALID_CONTEXT};
    const auto owner = state_->owner.lock();
    if (!owner) return {};
    if (owner->is_callback_active ||
        (owner->activity &&
         (owner->activity->is_native_issue_active || owner->activity->is_folded_active ||
          owner->activity->is_rnn_query_active || owner->activity->is_preparation_active)))
        return {.status = CUDA_ERROR_NOT_READY};
    CudaExecutionFrontierRetirement result{CUDA_SUCCESS, true};
    for (const auto& observation : state_->observations) {
        if (result.status == CUDA_SUCCESS && observation->failure != CUDA_SUCCESS)
            result.status = observation->failure;
        if (!observation->is_settled) result.is_retired = false;
    }
    if (result.status == CUDA_SUCCESS && !result.is_retired) result.status = CUDA_ERROR_NOT_READY;
    return result;
}
bool CudaExecutionFrontier::same_capture(const CudaExecutionFrontier& other) const noexcept {
    return state_ && state_ == other.state_ && ::getpid() == state_->process &&
           frontier_thread_owner && state_->thread.lock() == frontier_thread_owner &&
           !state_->owner.expired();
}
std::uint64_t CudaExecutionFrontier::diagnostic_id() const noexcept {
    if (!state_ || ::getpid() != state_->process || !frontier_thread_owner ||
        state_->thread.lock() != frontier_thread_owner)
        return 0;
    const auto owner = state_->owner.lock();
    if (!owner || owner->is_callback_active ||
        (owner->activity &&
         (owner->activity->is_native_issue_active || owner->activity->is_folded_active ||
          owner->activity->is_rnn_query_active || owner->activity->is_preparation_active)))
        return 0;
    return state_->diagnostic_id;
}

struct CudaExecutionDispatcher::Implementation {
    struct Function {
        std::uint64_t id{};
        CUfunction native{};
        std::size_t parameter_bytes{};
        std::shared_ptr<const void> lifetime;
    };
    struct Stream {
        std::uint64_t id{};
        CUstream native{};
        std::shared_ptr<const void> lifetime;
        bool is_nonblocking{};
    };
    struct ExternalCohort {
        std::uint64_t first_order{};
        std::vector<Stream> streams;
    };
    struct Record {
        std::uint64_t sequence{};
        // Private order includes external boundaries without consuming a
        // submission sequence or manufacturing an ordinary queue receipt.
        std::uint64_t acceptance_order{};
        CudaExecutionKernel kernel;
        std::optional<CudaExecutionMemory> memory;
        std::shared_ptr<CudaExecutionLibraryCall> library;
        std::optional<CudaExecutionEvent> event_call;
        Function function;
        Stream stream;
        std::shared_ptr<const void> context_lifetime;
        std::shared_ptr<const void> process_memory_lifetime;
        std::shared_ptr<const void> external_lifetime;
        std::shared_ptr<const ExternalCohort> external_cohort;
        std::shared_ptr<CudaExecutionEnqueueState> enqueue;
        ExecutionReceipt receipt;
        ExecutionIssueToken issue;
        CUfunction physical_function{};
        std::array<unsigned int, 3> physical_grid{};
        std::uint64_t geometry_epoch{1};
        std::uint64_t command_epoch{1};
        std::shared_ptr<CudaExecutionPersistentProgress> progress;
        CUevent event{};
        CUevent legacy_event{};
        // Allocated with immutable cohort streams, before any native callback.
        std::vector<CUevent> enqueue_events;
        CUresult launch_status{CUDA_SUCCESS};
        CUresult completion_status{CUDA_SUCCESS};
        bool is_native_issued{};
        bool is_event_recorded{};
        bool is_physically_stopped{};
        bool is_cleanup_uncertain{};
        bool was_library_entered{};
        bool has_library_cleanup_attempt{};
        bool has_original_parameter_prefix{};
        bool requires_preparation{}, reports_errors_asynchronously{};
        bool is_paused{};
        bool is_external{};
        bool has_external_started{};
        std::shared_ptr<CudaFrontierObservation> observation{
            std::make_shared<CudaFrontierObservation>()};
        // Immutable exact predecessor membership, not native/resource owners.
        // Failed observations survive removal of their ordinary queue records.
        std::vector<std::shared_ptr<const CudaFrontierObservation>> predecessors;
        std::shared_ptr<const std::vector<std::shared_ptr<const CudaFrontierObservation>>>
            captured_predecessors;
    };

    ExecutionSubmissionQueue queue;
    std::shared_ptr<CudaRetirementObserver> retirement_observer;
    CudaExecutionContext context;
    CudaExecutionDispatch dispatch;
    ExecutionSubmissionLimits limits;
    std::size_t max_bindings;
    const bool is_profile_isolated;
    const bool uses_default_proxy;
    const std::function<CUresult()> exclusive_native_gate;
    std::vector<Function> functions;
    std::vector<Stream> streams;
    std::vector<std::shared_ptr<Record>> records;
    std::shared_ptr<CudaFrontierOwner> frontier_owner{std::make_shared<CudaFrontierOwner>()};
    std::vector<std::weak_ptr<const CudaExecutionFrontier::State>> frontiers;
    std::vector<std::weak_ptr<CudaExecutionFrontierReservation::State>> reservations;
    std::uint64_t last_function_id{}, last_stream_id{}, last_acceptance_order{}, last_sequence{};
    std::uint64_t next_diagnostic_id{1};
    bool has_bound_executor{};
    bool& is_callback_active{frontier_owner->is_callback_active};
    std::uint64_t active_enqueue_turn{};

    Implementation(ExecutionSessionIdentity session, CudaExecutionContext context_value,
                   CudaExecutionDispatch dispatch_value, ExecutionSubmissionLimits limits_value,
                   std::size_t bindings, bool profile_isolated,
                   CudaExecutionStreamConfig stream_config,
                   std::shared_ptr<const CudaExecutionOwnerActivity> owner_activity,
                   std::function<CUresult()> native_gate)
        : queue(session, limits_value),
          context(std::move(context_value)),
          dispatch(dispatch_value),
          limits(limits_value),
          max_bindings(bindings),
          is_profile_isolated(profile_isolated),
          uses_default_proxy(stream_config.default_stream != nullptr),
          exclusive_native_gate(std::move(native_gate)) {
        frontier_owner->activity = std::move(owner_activity);
        last_sequence = session.first_sequence - 1;
        if (!context.id || !context.native || !context.lifetime || !max_bindings ||
            !dispatch.context_get_current || !dispatch.launch_kernel || !dispatch.event_create ||
            !dispatch.event_record || !dispatch.event_query || !dispatch.event_destroy ||
            !dispatch.stream_synchronize)
            throw std::invalid_argument("invalid CUDA execution context, dispatch or bounds");
        if (static_cast<bool>(stream_config.default_stream) !=
                static_cast<bool>(stream_config.default_stream_lifetime) ||
            stream_config.default_stream == CU_STREAM_LEGACY ||
            stream_config.default_stream == CU_STREAM_PER_THREAD ||
            (uses_default_proxy && !dispatch.stream_wait_event))
            throw std::invalid_argument("invalid default-stream proxy binding");
        functions.reserve(max_bindings);
        streams.reserve(max_bindings);
        records.reserve(limits.max_commands);
        frontiers.reserve(max_bindings);
        reservations.reserve(max_bindings);
        streams.push_back({0, stream_config.default_stream,
                           uses_default_proxy ? std::move(stream_config.default_stream_lifetime)
                                              : context.lifetime});
    }

    [[nodiscard]] bool has_owner_callback() const noexcept {
        const auto& activity = frontier_owner->activity;
        return is_callback_active ||
               (activity && (activity->is_native_issue_active || activity->is_folded_active ||
                             activity->is_rnn_query_active || activity->is_preparation_active));
    }

    [[nodiscard]] static CudaExecutionPendingDescription describe_pending(
        const Record& record) noexcept {
        return {.sequence = record.sequence,
                .acceptance_order = record.acceptance_order,
                .physical_generation = record.issue.generation(),
                .geometry_epoch = record.geometry_epoch,
                .command_epoch = record.command_epoch,
                .function = record.kernel.function,
                .stream = record.stream.id,
                .grid = record.kernel.grid,
                .block = record.kernel.block,
                .physical_grid = record.physical_grid,
                .shared_memory_bytes = record.kernel.shared_memory_bytes,
                .is_persistent = record.progress != nullptr,
                .is_resume = record.is_paused,
                .requires_preparation = record.requires_preparation};
    }

    [[nodiscard]] bool can_issue_pending(const Record& record) const noexcept {
        if (record.is_external || record.requires_preparation || record.is_cleanup_uncertain ||
            record.observation->is_settled)
            return false;
        if (record.is_paused)
            return record.progress && record.is_physically_stopped && !record.event &&
                   !record.legacy_event && record.launch_status == CUDA_SUCCESS &&
                   record.completion_status == CUDA_SUCCESS && is_ready(record);
        return !record.issue && is_ready(record);
    }

    [[nodiscard]] CUresult reservation_gate(
        const CudaExecutionFrontierReservation& reservation) const noexcept {
        if (!queue.owns_executor()) return CUDA_ERROR_INVALID_CONTEXT;
        if (has_owner_callback()) return CUDA_ERROR_NOT_READY;
        const auto& token = reservation.state_;
        if (!token || !token->capture || token->capture->owner.lock() != frontier_owner)
            return CUDA_ERROR_INVALID_HANDLE;
        if (::getpid() != token->capture->process || !frontier_thread_owner ||
            token->capture->thread.lock() != frontier_thread_owner)
            return CUDA_ERROR_INVALID_CONTEXT;
        return CUDA_SUCCESS;
    }

    [[nodiscard]] CUresult check_context() const noexcept {
        if (!queue.owns_executor()) return CUDA_ERROR_INVALID_CONTEXT;
        CUcontext current{};
        const auto result = dispatch.context_get_current(&current);
        if (result != CUDA_SUCCESS) return result;
        return current == context.native ? CUDA_SUCCESS : CUDA_ERROR_INVALID_CONTEXT;
    }
    [[nodiscard]] CUresult issue_health() const noexcept {
        try {
            return context.native_issue_health ? context.native_issue_health() : CUDA_SUCCESS;
        } catch (...) {
            return CUDA_ERROR_UNKNOWN;
        }
    }

    [[nodiscard]] std::shared_ptr<Record> enqueue_record(
        const CudaExecutionEnqueueTurn& turn) const noexcept {
        if (!turn.id_ || turn.owner_.lock() != frontier_owner) return {};
        const auto found = std::ranges::find_if(records, [&](const auto& record) {
            return record->enqueue && record->acceptance_order == turn.id_;
        });
        return found == records.end() ? nullptr : *found;
    }

    [[nodiscard]] CUresult wait_on(CUstream stream, CUevent event) const noexcept {
        if (const auto proof = check_context(); proof != CUDA_SUCCESS) return proof;
        const auto result = dispatch.stream_wait_event(stream, event, 0);
        const auto proof = check_context();
        return proof != CUDA_SUCCESS ? proof : result;
    }

    template <typename Visitor>
    static void each_stream(const Record& record, Visitor&& visit) {
        if (record.enqueue && record.external_cohort) {
            for (std::size_t index = 0; index < record.external_cohort->streams.size(); ++index)
                visit(record.external_cohort->streams[index],
                      index ? record.enqueue_events[index - 1] : record.event);
        } else
            visit(record.stream, record.event);
    }
    [[nodiscard]] static bool streams_depend(const Record& previous, const Stream& candidate) {
        bool depends{};
        each_stream(previous, [&](const Stream& stream, CUevent) {
            depends = depends || stream.id == candidate.id ||
                      (stream.id == 0 && !candidate.is_nonblocking) ||
                      (candidate.id == 0 && !stream.is_nonblocking);
        });
        return depends;
    }

    [[nodiscard]] CUresult prepare_dependencies(const Record& candidate,
                                                bool native_legacy = false) const noexcept {
        if (!uses_default_proxy) return CUDA_SUCCESS;
        const auto first = candidate.external_cohort ? candidate.external_cohort->first_order
                                                     : candidate.acceptance_order;
        auto status = CUDA_SUCCESS;
        each_stream(candidate, [&](const Stream& selected, CUevent) {
            const auto target = native_legacy ? nullptr : selected.native;
            for (const auto& previous : records) {
                if (status != CUDA_SUCCESS || previous->acceptance_order >= first ||
                    previous->observation->is_settled)
                    continue;
                each_stream(*previous, [&](const Stream& prior, CUevent event) {
                    if (status != CUDA_SUCCESS || prior.native == target ||
                        !(prior.id == selected.id || (prior.id == 0 && !selected.is_nonblocking) ||
                          (selected.id == 0 && !prior.is_nonblocking)))
                        return;
                    if (previous->is_cleanup_uncertain ||
                        previous->observation->failure != CUDA_SUCCESS)
                        status = previous->observation->failure == CUDA_SUCCESS
                                     ? CUDA_ERROR_UNKNOWN
                                     : previous->observation->failure;
                    else if (!previous->is_native_issued || !previous->is_event_recorded ||
                             previous->progress)
                        status = CUDA_ERROR_NOT_READY;
                    else
                        status = wait_on(target, event);
                });
            }
        });
        return status;
    }

    [[nodiscard]] CUresult create_legacy_event(const std::shared_ptr<Record>& record) noexcept {
        if (const auto proof = check_context(); proof != CUDA_SUCCESS)
            return retain_uncertain(record, proof);
        const auto status = dispatch.event_create(&record->legacy_event, CU_EVENT_DISABLE_TIMING);
        const auto proof = check_context();
        if (proof != CUDA_SUCCESS || status != CUDA_SUCCESS || !record->legacy_event)
            return retain_uncertain(record, proof != CUDA_SUCCESS    ? proof
                                            : status != CUDA_SUCCESS ? status
                                                                     : CUDA_ERROR_UNKNOWN);
        bool aliases = record->legacy_event == record->event;
        for (const auto& other : records)
            if (other != record &&
                (other->event == record->legacy_event ||
                 other->legacy_event == record->legacy_event ||
                 std::ranges::find(other->enqueue_events, record->legacy_event) !=
                     other->enqueue_events.end())) {
                aliases = true;
                (void)retain_uncertain(other, CUDA_ERROR_UNKNOWN);
            }
        if (aliases) return retain_uncertain(record, CUDA_ERROR_UNKNOWN);
        return CUDA_SUCCESS;
    }

    [[nodiscard]] CUresult join_legacy(const std::shared_ptr<Record>& record) noexcept {
        if (const auto proof = check_context(); proof != CUDA_SUCCESS)
            return retain_uncertain(record, proof);
        const auto status = dispatch.event_record(record->legacy_event, nullptr);
        const auto proof = check_context();
        if (proof != CUDA_SUCCESS || status != CUDA_SUCCESS)
            return retain_uncertain(record, proof != CUDA_SUCCESS ? proof : status);
        const auto waited = wait_on(record->stream.native, record->legacy_event);
        return waited == CUDA_SUCCESS ? CUDA_SUCCESS : retain_uncertain(record, waited);
    }

    [[nodiscard]] bool is_ready(const Record& candidate) const noexcept {
        // A cohort is one logical admission cut. Each member waits for the
        // union of earlier source dependencies, never for a sibling that needs
        // the same outer native enqueue. Ordinary candidates keep their order.
        const auto* cohort = candidate.external_cohort.get();
        const auto first_order = cohort ? cohort->first_order : candidate.acceptance_order;
        if (is_profile_isolated && std::ranges::any_of(records, [&](const auto& previous) {
                return previous->acceptance_order < first_order;
            }))
            return false;
        if (candidate.event_call &&
            candidate.event_call->call->dependency_status() == CUDA_ERROR_NOT_READY)
            return false;
        if (candidate.event_call && candidate.event_call->dependency) {
            const auto& dependency = candidate.event_call->dependency->state_;
            if (dependency && std::ranges::any_of(dependency->observations, [](const auto& item) {
                    return item->failure == CUDA_SUCCESS && !item->is_settled;
                }))
                return false;
        }
        if (candidate.event_call && candidate.event_call->foreign_dependency &&
            candidate.event_call->foreign_dependency->observe() == CUDA_ERROR_NOT_READY)
            return false;
        return !std::ranges::any_of(records, [&](const auto& previous) {
            if (previous->acceptance_order >= first_order ||
                (previous->is_native_issued && !previous->progress))
                return false;
            // Native SDK stream ordering alone cannot cover a paused logical
            // predecessor or a not-yet-issued command. Preserve both per-stream
            // order and the context's legacy/blocking-stream dependencies.
            const auto depends_on_previous = [&](const Stream& stream) {
                return streams_depend(*previous, stream);
            };
            return cohort ? std::ranges::any_of(cohort->streams, depends_on_previous)
                          : depends_on_previous(candidate.stream);
        });
    }

    // Caller has authenticated the bound executor. Dependency callbacks retain
    // the same exclusion as the standalone public readiness query.
    [[nodiscard]] bool has_ready_work() noexcept {
        if (is_callback_active || active_enqueue_turn) return false;
        const CallbackScope callback_scope{is_callback_active};
        return std::ranges::any_of(records, [&](const auto& record) {
            return !record->is_external && !record->requires_preparation && !record->issue &&
                   is_ready(*record);
        });
    }

    struct DependencyClosure {
        CUresult status{CUDA_SUCCESS};
        std::vector<unsigned char> required;
    };
    [[nodiscard]] static CudaExecutionDependencyInspection observed_failure(
        std::span<const std::shared_ptr<const CudaFrontierObservation>> observations,
        FrontierPurpose purpose = FrontierPurpose::kExecution) noexcept {
        for (const auto& observation : observations)
            if (observation->failure != CUDA_SUCCESS &&
                (purpose != FrontierPurpose::kRetirement || !observation->is_settled))
                return {.status = observation->failure,
                        .foreign_blocker = observation->foreign_dependency,
                        .has_failure_observation = true,
                        .has_retired_failure = observation->is_settled};
        return {.status = CUDA_SUCCESS};
    }
    // Serialized admission or callback-excluded progress only. Membership is an exact retained
    // capture, not a newly sampled stream. This one bounded DAG walk serves
    // inspection and both issue paths; it never recursively visits an owner.
    [[nodiscard]] DependencyClosure dependency_closure(
        const std::shared_ptr<Record>& turn, const CudaExecutionFrontier* frontier,
        FrontierPurpose purpose = FrontierPurpose::kExecution) const {
        DependencyClosure result{.required = std::vector<unsigned char>(records.size())};
        const auto mark_predecessors = [&](const Record& candidate, std::size_t prior_count) {
            const auto first = candidate.external_cohort ? candidate.external_cohort->first_order
                                                         : candidate.acceptance_order;
            for (std::size_t prior = 0; prior != prior_count; ++prior) {
                const auto& previous = *records[prior];
                if (previous.acceptance_order >= first || previous.observation->is_settled)
                    continue;
                bool depends = is_profile_isolated;
                // Issued ordinary commands require no host progression, but
                // remain exact physical-prefix/error observations until stop.
                each_stream(candidate, [&](const Stream& stream, CUevent) {
                    depends = depends || streams_depend(previous, stream);
                });
                if (candidate.event_call && candidate.event_call->dependency) {
                    const auto& dependency = candidate.event_call->dependency->state_;
                    if (dependency && dependency->owner.lock() == frontier_owner)
                        depends = depends || std::ranges::find(dependency->observations,
                                                               previous.observation) !=
                                                 dependency->observations.end();
                }
                if (depends) result.required[prior] = 1;
            }
        };
        if (turn) {
            const auto found = std::ranges::find(records, turn);
            if (found == records.end()) {
                // Private admission preflight: the candidate is not published
                // yet, but uses exactly the same predecessor relation.
                mark_predecessors(*turn, records.size());
            } else
                result.required[static_cast<std::size_t>(found - records.begin())] = 1;
        } else {
            for (const auto& root : frontier->state_->observations) {
                if (purpose == FrontierPurpose::kRetirement && root->is_settled) continue;
                if (root->failure != CUDA_SUCCESS) return {.status = root->failure, .required = {}};
                if (root->is_settled) continue;
                const auto found = std::ranges::find_if(
                    records, [&](const auto& record) { return record->observation == root; });
                if (found == records.end()) return {.status = CUDA_ERROR_UNKNOWN, .required = {}};
                result.required[static_cast<std::size_t>(found - records.begin())] = 1;
            }
        }
        for (std::size_t index = records.size(); index-- != 0;) {
            if (!result.required[index]) continue;
            mark_predecessors(*records[index], index);
        }
        return result;
    }
    [[nodiscard]] CUresult capture_predecessors(const std::shared_ptr<Record>& candidate) const {
        const auto closure = dependency_closure(candidate, nullptr);
        if (closure.status != CUDA_SUCCESS) return closure.status;
        auto& retained = candidate->predecessors;
        retained.reserve(limits.max_commands);
        const auto append = [&](const std::shared_ptr<const CudaFrontierObservation>& item) {
            if ((item->is_settled && item->failure == CUDA_SUCCESS) ||
                std::ranges::find(retained, item) != retained.end())
                return true;
            if (retained.size() == limits.max_commands) return false;
            retained.push_back(item);
            return true;
        };
        const auto append_dependency = [&](const CudaExecutionFrontier& dependency) {
            for (const auto& item : dependency.state_->observations)
                if (!append(item)) return false;
            if (dependency.state_->predecessors)
                for (const auto& item : *dependency.state_->predecessors)
                    if (!append(item)) return false;
            return true;
        };
        if (candidate->event_call && candidate->event_call->dependency &&
            !append_dependency(*candidate->event_call->dependency))
            return CUDA_ERROR_OUT_OF_MEMORY;
        for (std::size_t index = 0; index != records.size(); ++index) {
            if (!closure.required[index]) continue;
            const auto& previous = *records[index];
            if (!append(previous.observation)) return CUDA_ERROR_OUT_OF_MEMORY;
            for (const auto& ancestor : previous.predecessors)
                if (!append(ancestor)) return CUDA_ERROR_OUT_OF_MEMORY;
            if (previous.event_call && previous.event_call->dependency)
                if (!append_dependency(*previous.event_call->dependency))
                    return CUDA_ERROR_OUT_OF_MEMORY;
        }
        std::ranges::sort(retained, {}, &CudaFrontierObservation::acceptance_order);
        return CUDA_SUCCESS;
    }
    [[nodiscard]] CUresult prepare_command_capture(
        const std::shared_ptr<Record>& record,
        const std::shared_ptr<CudaExecutionFrontier::State>& explicit_capture = {}) const {
        record->acceptance_order = last_acceptance_order + 1;
        const bool reserved = std::ranges::any_of(reservations, [&](const auto& weak) {
            const auto item = weak.lock();
            return item && item->capture && !item->is_bound && item->sequence == record->sequence;
        });
        if (!reserved && !explicit_capture) return CUDA_SUCCESS;
        const auto status = capture_predecessors(record);
        if (status != CUDA_SUCCESS) return status;
        record->captured_predecessors =
            std::make_shared<const std::vector<std::shared_ptr<const CudaFrontierObservation>>>(
                record->predecessors);
        if (explicit_capture) explicit_capture->predecessors = record->captured_predecessors;
        return CUDA_SUCCESS;
    }
    // All roots, closures and vector capacity exist before queue.accept. This
    // publication cannot allocate and binds every matching observer atomically.
    void publish_command(const std::shared_ptr<Record>& record,
                         const ExecutionReceipt& receipt) noexcept {
        last_sequence = record->sequence;
        record->acceptance_order = ++last_acceptance_order;
        record->observation->acceptance_order = record->acceptance_order;
        record->receipt = receipt;
        records.push_back(record);
        for (const auto& weak : reservations) {
            const auto item = weak.lock();
            if (!item || !item->capture || item->is_bound || item->sequence != record->sequence)
                continue;
            item->capture->observations.front() = record->observation;
            item->capture->predecessors = record->captured_predecessors;
            item->is_bound = true;
        }
    }
    [[nodiscard]] CudaExecutionDependencyInspection retained_failure(
        const DependencyClosure& closure,
        FrontierPurpose purpose = FrontierPurpose::kExecution) const noexcept {
        std::shared_ptr<const CudaFrontierObservation> first;
        for (std::size_t index = 0; index != records.size(); ++index) {
            if (!closure.required[index]) continue;
            for (const auto& observation : records[index]->predecessors)
                if (observation->failure != CUDA_SUCCESS &&
                    (purpose != FrontierPurpose::kRetirement || !observation->is_settled) &&
                    (!first || observation->acceptance_order < first->acceptance_order))
                    first = observation;
        }
        return first ? observed_failure(std::span{&first, std::size_t{1}})
                     : CudaExecutionDependencyInspection{.status = CUDA_SUCCESS};
    }
    // Called only under this owner's exclusion. Local observations may be
    // inspected directly; a foreign owner must independently authorize its
    // retirement proof (including its enclosing Session callback exclusion).
    [[nodiscard]] bool is_retired_dependency_error(const CudaExecutionFrontier& dependency,
                                                   CUresult failure) const noexcept {
        if (!dependency.same_capture(dependency)) return false;
        if (dependency.state_->owner.lock() != frontier_owner) {
            const auto stopped = dependency.retirement();
            return stopped.is_retired && stopped.status == failure;
        }
        CUresult first_failure = CUDA_SUCCESS;
        for (const auto& observation : dependency.state_->observations) {
            if (!observation->is_settled) return false;
            if (first_failure == CUDA_SUCCESS && observation->failure != CUDA_SUCCESS)
                first_failure = observation->failure;
        }
        return first_failure == failure;
    }
    [[nodiscard]] bool can_progress_dependency(const Record& record) const noexcept {
        return !record.is_external && !record.requires_preparation &&
               !record.is_cleanup_uncertain && (!record.issue || record.is_paused) &&
               is_ready(record);
    }
    [[nodiscard]] CudaExecutionDependencyInspection inspect_closure(
        const DependencyClosure& closure, CUresult root_status,
        FrontierPurpose purpose = FrontierPurpose::kExecution) const noexcept {
        if (closure.status != CUDA_SUCCESS) return {.status = closure.status};
        // Retained turn roots are authoritative even when an intervening
        // ordinary event still exposes the same error through its callback.
        auto retained = retained_failure(closure, purpose);
        if (retained.status != CUDA_SUCCESS) return retained;
        CudaExecutionDependencyInspection result{.status = root_status};
        for (std::size_t index = 0; index != records.size(); ++index) {
            if (!closure.required[index]) continue;
            const auto& record = *records[index];
            if (record.observation->failure != CUDA_SUCCESS)
                return {.status = record.observation->failure,
                        .foreign_blocker = record.observation->foreign_dependency,
                        .has_failure_observation = true,
                        .has_retired_failure = record.observation->is_settled};
            result.has_progress = result.has_progress || can_progress_dependency(record);
            if (record.requires_preparation && !record.issue && !record.is_cleanup_uncertain) {
                result.has_progress = true;
                if (!result.preparation_sequence) result.preparation_sequence = record.sequence;
            }
            if (!record.event_call || record.is_native_issued) continue;
            const auto& event = *record.event_call;
            const auto checked = event.call->dependency_status();
            if (checked != CUDA_SUCCESS && checked != CUDA_ERROR_NOT_READY) {
                const bool known_retired =
                    purpose == FrontierPurpose::kRetirement &&
                    ((event.dependency &&
                      is_retired_dependency_error(*event.dependency, checked)) ||
                     (event.foreign_dependency &&
                      is_retired_dependency_error(*event.foreign_dependency, checked)));
                if (!known_retired)
                    return {.status = checked, .foreign_blocker = event.foreign_dependency};
            }
            for (const auto* dependency : {&event.dependency, &event.foreign_dependency}) {
                if (!*dependency) continue;
                const auto status = (*dependency)->observe();
                if (status != CUDA_SUCCESS && status != CUDA_ERROR_NOT_READY) {
                    if (purpose == FrontierPurpose::kRetirement &&
                        is_retired_dependency_error(**dependency, status))
                        continue;
                    return {.status = status, .foreign_blocker = event.foreign_dependency};
                }
                if (status == CUDA_ERROR_NOT_READY && dependency == &event.foreign_dependency &&
                    !result.foreign_blocker)
                    result.foreign_blocker = **dependency;
            }
        }
        return result;
    }
    [[nodiscard]] CudaExecutionIssueResult progress_closure(
        const DependencyClosure& closure, CUresult root_status,
        FrontierPurpose purpose = FrontierPurpose::kExecution) noexcept {
        if (closure.status != CUDA_SUCCESS)
            return {.state = CudaExecutionIssueState::kError, .native_status = closure.status};
        const auto failed = retained_failure(closure, purpose);
        if (failed.status != CUDA_SUCCESS)
            return {.state = CudaExecutionIssueState::kError, .native_status = failed.status};
        if (purpose == FrontierPurpose::kRetirement) {
            const auto blocking = inspect_closure(closure, root_status, purpose);
            if (blocking.status != CUDA_SUCCESS && blocking.status != CUDA_ERROR_NOT_READY)
                return {.state = CudaExecutionIssueState::kError, .native_status = blocking.status};
        }
        for (std::size_t index = 0; index != records.size(); ++index) {
            if (!closure.required[index]) continue;
            auto record = records[index];
            if (!can_progress_dependency(*record)) continue;
            return record->is_paused ? resume_record(record) : issue_record(record);
        }
        const auto observed = inspect_closure(closure, root_status, purpose);
        return {.state = observed.status == CUDA_SUCCESS || observed.status == CUDA_ERROR_NOT_READY
                             ? CudaExecutionIssueState::kIdle
                             : CudaExecutionIssueState::kError,
                .native_status = observed.status};
    }

    // Callback exclusion and owner/context proof belong to the public entry.
    // Both ordinary scheduling and retained dependency progress use the same
    // receipt, preparation, native-status and event-retirement machinery.
    [[nodiscard]] CudaExecutionIssueResult issue_record(
        const std::shared_ptr<Record>& record) noexcept {
        if (const auto status = issue_health(); status != CUDA_SUCCESS)
            return {.state = CudaExecutionIssueState::kError,
                    .sequence = record->sequence,
                    .native_status = status};
        if (record->requires_preparation)
            return {.state = CudaExecutionIssueState::kIdle,
                    .sequence = record->sequence,
                    .native_status = CUDA_ERROR_NOT_READY};
        if (record->is_cleanup_uncertain || queue.snapshot().is_quarantined)
            return {.state = CudaExecutionIssueState::kError, .native_status = CUDA_ERROR_UNKNOWN};
        // An opaque original library may synchronize the whole physical
        // context internally. Check the trusted cohort owner at dispatch, not
        // admission, before taking the issue token or preparing any event.
        // Pure/transformed kernels deliberately do not use this exclusive gate.
        if (record->library && !record->progress && exclusive_native_gate) {
            CUresult status{};
            try {
                status = exclusive_native_gate();
            } catch (...) {
                return {.state = CudaExecutionIssueState::kError,
                        .native_status = retain_uncertain(record, CUDA_ERROR_UNKNOWN)};
            }
            const auto proof = check_context();
            if (proof != CUDA_SUCCESS)
                return {.state = CudaExecutionIssueState::kError,
                        .native_status = retain_uncertain(record, proof)};
            if (status == CUDA_ERROR_NOT_READY)
                return {.state = CudaExecutionIssueState::kIdle,
                        .native_status = CUDA_ERROR_NOT_READY};
            if (status != CUDA_SUCCESS)
                return {.state = CudaExecutionIssueState::kError,
                        .native_status = retain_uncertain(record, status)};
        }
        if (record->command_epoch == UINT64_MAX)
            return {.state = CudaExecutionIssueState::kError,
                    .sequence = record->sequence,
                    .native_status = CUDA_ERROR_OUT_OF_MEMORY};
        ++record->command_epoch;
        const auto issue = queue.take_next(record->receipt);
        if (!issue) {
            queue.quarantine();
            return {.state = CudaExecutionIssueState::kError, .native_status = CUDA_ERROR_UNKNOWN};
        }
        record->issue = issue->token;
        // Retained ancestry supports inspection and explicit targeted rejection;
        // adding an observer must not poison ordinary native stream semantics.
        auto result = CUDA_SUCCESS;
        if (record->event_call && record->event_call->dependency) {
            const auto& frontier = *record->event_call->dependency;
            result = !frontier.state_ || frontier.state_->owner.lock() != frontier_owner
                         ? CUDA_ERROR_INVALID_HANDLE
                         : frontier.observe();
        }
        if (result == CUDA_SUCCESS && record->event_call)
            result = record->event_call->call->dependency_status();
        if (result == CUDA_SUCCESS && record->event_call && record->event_call->foreign_dependency)
            result = record->event_call->foreign_dependency->observe();
        if (result == CUDA_SUCCESS && record->progress) result = record->progress->prepare_launch();
        if (result == CUDA_SUCCESS) result = create_event(record);
        const bool native_legacy = uses_default_proxy && record->memory &&
                                   !record->memory->is_asynchronous &&
                                   record->memory->kind != CudaMemoryKind::kUpload &&
                                   record->memory->kind != CudaMemoryKind::kDownload;
        if (result == CUDA_SUCCESS && native_legacy) result = create_legacy_event(record);
        if (result == CUDA_SUCCESS) {
            result = prepare_dependencies(*record, native_legacy);
            if (result != CUDA_SUCCESS) (void)retain_uncertain(record, result);
        }
        if (result != CUDA_SUCCESS) {
            if (record->observation->failure == CUDA_SUCCESS) record->observation->failure = result;
            if (!queue.reject_before_native_issue(record->issue, result)) queue.quarantine();
            if (!record->is_cleanup_uncertain) (void)destroy_event(record);
            if (!record->is_cleanup_uncertain) {
                record->observation->is_settled = true;
                std::erase(records, record);
            } else if (record->observation->failure == CUDA_SUCCESS)
                record->observation->failure = CUDA_ERROR_UNKNOWN;
            return {.state = CudaExecutionIssueState::kRejected,
                    .sequence = issue->token.sequence(),
                    .native_status = result};
        }
        return launch_record(record, *issue, false);
    }
    [[nodiscard]] CudaExecutionIssueResult resume_record(
        const std::shared_ptr<Record>& record) noexcept {
        if (const auto status = issue_health(); status != CUDA_SUCCESS)
            return {.state = CudaExecutionIssueState::kError,
                    .sequence = record->sequence,
                    .native_status = status};
        if (record->command_epoch == UINT64_MAX)
            return {.state = CudaExecutionIssueState::kError,
                    .sequence = record->sequence,
                    .native_status = CUDA_ERROR_OUT_OF_MEMORY};
        ++record->command_epoch;
        auto status = record->progress->prepare_launch();
        if (status == CUDA_SUCCESS) status = create_event(record);
        if (status == CUDA_SUCCESS) {
            status = prepare_dependencies(*record);
            if (status != CUDA_SUCCESS) (void)retain_uncertain(record, status);
        }
        if (status != CUDA_SUCCESS) {
            if (record->reports_errors_asynchronously) {
                // Admission already returned, and the first SDK receipt is
                // immutable. Preserve this continuation failure in its exact
                // frontier and retain incomplete logical work without retry or
                // fabricated completion of the previously stopped generation.
                if (record->observation->failure == CUDA_SUCCESS)
                    record->observation->failure = status;
                queue.quarantine();
            }
            return {.state = CudaExecutionIssueState::kRejected,
                    .sequence = record->sequence,
                    .native_status = status};
        }
        const auto issue = queue.resume_after_physical_stop(record->issue);
        if (!issue) {
            (void)destroy_event(record);
            return {.state = CudaExecutionIssueState::kError,
                    .sequence = record->sequence,
                    .native_status = CUDA_ERROR_UNKNOWN};
        }
        return launch_record(record, *issue, true);
    }

    [[nodiscard]] std::shared_ptr<Record> external_record(
        const CudaExecutionExternalBarrier& barrier) const noexcept {
        if (!barrier.id_ || barrier.owner_.lock() != frontier_owner) return {};
        const auto found = std::ranges::find_if(records, [&](const auto& record) {
            return record->is_external && record->acceptance_order == barrier.id_;
        });
        return found == records.end() ? nullptr : *found;
    }

    [[nodiscard]] bool has_external_stream_hazard(const Record& candidate) const noexcept {
        return std::ranges::any_of(records, [&](const auto& external) {
            if (!external->is_external ||
                (external->is_event_recorded && !external->is_cleanup_uncertain))
                return false;
            bool depends{};
            each_stream(candidate, [&](const Stream& stream, CUevent) {
                depends = depends || streams_depend(*external, stream);
            });
            return depends;
        });
    }

    // No repeated event-destroy call after an ambiguous failure. Preserve both
    // the private event and all resource leases until real context recovery.
    [[nodiscard]] CUresult destroy_event(const std::shared_ptr<Record>& record) noexcept {
        for (auto& event : record->enqueue_events) {
            if (!event) continue;
            if (const auto proof = check_context(); proof != CUDA_SUCCESS)
                return retain_uncertain(record, proof);
            const auto status = dispatch.event_destroy(event);
            const auto proof = check_context();
            if (status != CUDA_SUCCESS || proof != CUDA_SUCCESS)
                return retain_uncertain(record, proof != CUDA_SUCCESS ? proof : status);
            event = nullptr;
        }
        if (record->legacy_event) {
            if (const auto proof = check_context(); proof != CUDA_SUCCESS)
                return retain_uncertain(record, proof);
            const auto result = dispatch.event_destroy(record->legacy_event);
            const auto proof = check_context();
            if (result != CUDA_SUCCESS || proof != CUDA_SUCCESS)
                return retain_uncertain(record, proof != CUDA_SUCCESS ? proof : result);
            record->legacy_event = nullptr;
        }
        if (record->event) {
            const auto result = dispatch.event_destroy(record->event);
            if (record->is_external || uses_default_proxy) {
                const auto context_status = check_context();
                if (context_status != CUDA_SUCCESS) return retain_uncertain(record, context_status);
            }
            if (result != CUDA_SUCCESS) {
                record->is_cleanup_uncertain = true;
                if (record->observation->failure == CUDA_SUCCESS)
                    record->observation->failure = result;
                queue.quarantine();
                return result;
            }
            record->event = nullptr;
        }
        return CUDA_SUCCESS;
    }

    [[nodiscard]] CUresult retain_uncertain(const std::shared_ptr<Record>& record,
                                            CUresult failure) noexcept {
        record->is_cleanup_uncertain = true;
        if (record->observation->failure == CUDA_SUCCESS) record->observation->failure = failure;
        queue.quarantine();
        return failure;
    }

    [[nodiscard]] CUresult retire(const std::shared_ptr<Record>& record) noexcept {
        // A successful stop callback can still have changed the current
        // context. Never destroy a private object under that foreign owner.
        if (const auto status = check_context(); status != CUDA_SUCCESS)
            return retain_uncertain(record, status);
        const auto cleaned = destroy_event(record);
        if (cleaned != CUDA_SUCCESS) return cleaned;
        if (const auto status = check_context(); status != CUDA_SUCCESS)
            return retain_uncertain(record, status);
        if (record->was_library_entered) {
            if (record->has_library_cleanup_attempt)
                return retain_uncertain(record, CUDA_ERROR_UNKNOWN);
            record->has_library_cleanup_attempt = true;
            const auto status = record->library->cleanup_after_physical_stop();
            const auto context_status = check_context();
            if (status != CUDA_SUCCESS || context_status != CUDA_SUCCESS)
                return retain_uncertain(record, status != CUDA_SUCCESS ? status : context_status);
        }
        if (!record->is_external && !queue.complete_after_physical_stop(record->issue)) {
            return retain_uncertain(record, CUDA_ERROR_UNKNOWN);
        }
        record->observation->is_settled = true;
        record->process_memory_lifetime.reset();
        if (record->kernel.observation)
            record->kernel.observation->finish(record->launch_status == CUDA_SUCCESS &&
                                               record->completion_status == CUDA_SUCCESS &&
                                               record->observation->failure == CUDA_SUCCESS);
        return CUDA_SUCCESS;
    }

    [[nodiscard]] CUresult create_event(const std::shared_ptr<Record>& record) noexcept {
        if (uses_default_proxy || record->enqueue) {
            const auto proof = check_context();
            if (proof != CUDA_SUCCESS) return retain_uncertain(record, proof);
        }
        auto result = dispatch.event_create(&record->event, CU_EVENT_DISABLE_TIMING);
        if (uses_default_proxy || record->enqueue) {
            const auto proof = check_context();
            if (proof != CUDA_SUCCESS) return retain_uncertain(record, proof);
            bool aliases{};
            for (const auto& other : records)
                if (record->event && other != record &&
                    (other->event == record->event || other->legacy_event == record->event ||
                     std::ranges::find(other->enqueue_events, record->event) !=
                         other->enqueue_events.end())) {
                    aliases = true;
                    (void)retain_uncertain(other, CUDA_ERROR_UNKNOWN);
                }
            if (aliases) return retain_uncertain(record, CUDA_ERROR_UNKNOWN);
        }
        if (result == CUDA_SUCCESS && record->event) return CUDA_SUCCESS;
        if (result == CUDA_SUCCESS) result = CUDA_ERROR_UNKNOWN;
        // Partial outputs must be cleaned once or retained as ambiguous.
        (void)destroy_event(record);
        return result;
    }

    [[nodiscard]] CUresult create_enqueue_events(const std::shared_ptr<Record>& record) noexcept {
        auto status = create_event(record);
        if (status != CUDA_SUCCESS) return status;
        for (auto& event : record->enqueue_events) {
            if (const auto proof = check_context(); proof != CUDA_SUCCESS)
                return retain_uncertain(record, proof);
            status = dispatch.event_create(&event, CU_EVENT_DISABLE_TIMING);
            const auto proof = check_context();
            if (proof != CUDA_SUCCESS || status != CUDA_SUCCESS || !event)
                return retain_uncertain(record, proof != CUDA_SUCCESS    ? proof
                                                : status != CUDA_SUCCESS ? status
                                                                         : CUDA_ERROR_UNKNOWN);
            bool aliases{};
            for (const auto& other : records) {
                const auto occurrences = std::ranges::count(other->enqueue_events, event);
                if (other->event == event || other->legacy_event == event ||
                    occurrences > (other == record ? 1 : 0)) {
                    aliases = true;
                    (void)retain_uncertain(other, CUDA_ERROR_UNKNOWN);
                }
            }
            if (aliases) return retain_uncertain(record, CUDA_ERROR_UNKNOWN);
        }
        return CUDA_SUCCESS;
    }
    [[nodiscard]] CUresult record_enqueue_events(const std::shared_ptr<Record>& record) noexcept {
        auto status = CUDA_SUCCESS;
        each_stream(*record, [&](const Stream& stream, CUevent event) {
            if (status != CUDA_SUCCESS) return;
            status = check_context();
            if (status != CUDA_SUCCESS) return;
            status = dispatch.event_record(event, stream.native);
            const auto proof = check_context();
            if (proof != CUDA_SUCCESS) status = proof;
        });
        return status;
    }

    [[nodiscard]] CUresult issue_memory(const Record& record) const noexcept {
        const auto& memory = *record.memory;
        const auto stream = record.stream.native;
        switch (memory.kind) {
            case CudaMemoryKind::kUpload:
                return dispatch.upload_async(memory.destination, memory.host, memory.count, stream);
            case CudaMemoryKind::kDownload:
                return dispatch.download_async(memory.host, memory.source, memory.count, stream);
            case CudaMemoryKind::kCopy:
                return memory.is_asynchronous
                           ? dispatch.copy_device_async(memory.destination, memory.source,
                                                        memory.count, stream)
                           : dispatch.copy_device(memory.destination, memory.source, memory.count);
            case CudaMemoryKind::kCopyUva:
                return memory.is_asynchronous
                           ? dispatch.copy_unified_async(memory.destination, memory.source,
                                                         memory.count, stream)
                           : dispatch.copy_unified(memory.destination, memory.source, memory.count);
            case CudaMemoryKind::kFill8:
                return memory.is_asynchronous
                           ? dispatch.fill8_async(memory.destination,
                                                  static_cast<unsigned char>(memory.value),
                                                  memory.count, stream)
                           : dispatch.fill8(memory.destination,
                                            static_cast<unsigned char>(memory.value), memory.count);
            case CudaMemoryKind::kFill16:
                return memory.is_asynchronous
                           ? dispatch.fill16_async(memory.destination,
                                                   static_cast<unsigned short>(memory.value),
                                                   memory.count, stream)
                           : dispatch.fill16(memory.destination,
                                             static_cast<unsigned short>(memory.value),
                                             memory.count);
            case CudaMemoryKind::kFill32:
                return memory.is_asynchronous
                           ? dispatch.fill32_async(memory.destination, memory.value, memory.count,
                                                   stream)
                           : dispatch.fill32(memory.destination, memory.value, memory.count);
        }
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    [[nodiscard]] CudaExecutionIssueResult launch_record(const std::shared_ptr<Record>& record,
                                                         const ExecutionIssue& issue,
                                                         bool is_continuation) noexcept {
        record->issue = issue.token;
        record->is_paused = false;
        record->is_physically_stopped = false;
        record->is_event_recorded = false;
        auto parameter_bytes = issue.parameters.size();
        void* extra[] = {CU_LAUNCH_PARAM_BUFFER_POINTER,
                         const_cast<std::byte*>(issue.parameters.data()),
                         CU_LAUNCH_PARAM_BUFFER_SIZE, &parameter_bytes, CU_LAUNCH_PARAM_END};
        const auto& kernel = record->kernel;
        const auto& grid = record->physical_grid;
        record->is_native_issued = true;
        if (!is_continuation && record->kernel.observation) record->kernel.observation->begin();
        auto prefix = CUDA_SUCCESS;
        if (record->progress) {
            for (const auto& update : record->progress->parameter_updates())
                if (!queue.patch_issue_parameters(
                        record->issue, update.offset,
                        std::as_bytes(std::span{&update.value, std::size_t{1}}))) {
                    prefix = CUDA_ERROR_INVALID_VALUE;
                    break;
                }
            if (prefix == CUDA_SUCCESS)
                prefix = record->progress->record_initialization(record->stream.native);
            if (uses_default_proxy) {
                const auto proof = check_context();
                if (proof != CUDA_SUCCESS) prefix = retain_uncertain(record, proof);
            }
        }
        std::optional<std::int32_t> library_status;
        CudaExecutionLibraryCompletion library_completion;
        bool is_clean_preparation_rejection{};
        if (record->library && prefix == CUDA_SUCCESS) {
            record->was_library_entered = true;
            library_status = record->library->issue(record->stream.native, issue.parameters);
            library_completion = record->library->completion();
            if (library_completion.is_clean_preparation_rejection) {
                is_clean_preparation_rejection = !library_completion.was_native_issued &&
                                                 !library_completion.is_cleanup_uncertain &&
                                                 library_completion.status != CUDA_SUCCESS;
                // An inconsistent trusted capsule cannot prove private-work
                // ownership. Preserve any actual SDK return, but never fence
                // or release this record based on the contradictory claim.
                if (!is_clean_preparation_rejection) library_completion.is_cleanup_uncertain = true;
            }
            if (library_completion.is_cleanup_uncertain &&
                library_completion.status == CUDA_SUCCESS)
                library_completion.status = CUDA_ERROR_UNKNOWN;
            record->is_cleanup_uncertain |= library_completion.is_cleanup_uncertain;
            if (!library_completion.was_native_issued) {
                library_status.reset();
                prefix = library_completion.status != CUDA_SUCCESS ? library_completion.status
                                                                   : CUDA_ERROR_UNKNOWN;
            }
            if (is_clean_preparation_rejection) {
                // issue/completion may have changed context even when their
                // capsule claims known ownership. This proof runs under the
                // issue callback scope, before any trailing event operation.
                if (const auto current = check_context(); current != CUDA_SUCCESS) {
                    library_completion.status = current;
                    record->is_cleanup_uncertain = true;
                    is_clean_preparation_rejection = false;
                }
            }
        }
        const auto status =
            prefix != CUDA_SUCCESS ? prefix
            : record->library      ? CUDA_SUCCESS
            : record->event_call   ? record->event_call->call->issue(record->stream.native)
            : record->memory
                ? issue_memory(*record)
                : dispatch.launch_kernel(record->physical_function, grid[0], grid[1], grid[2],
                                         kernel.block[0], kernel.block[1], kernel.block[2],
                                         kernel.shared_memory_bytes, record->stream.native, nullptr,
                                         parameter_bytes == 0 ? nullptr : extra);
        record->launch_status = status;
        if (uses_default_proxy && !record->is_cleanup_uncertain) {
            const auto proof = check_context();
            if (proof != CUDA_SUCCESS) (void)retain_uncertain(record, proof);
        }
        if (record->legacy_event && !record->is_cleanup_uncertain) {
            const auto proof = check_context();
            if (proof != CUDA_SUCCESS)
                (void)retain_uncertain(record, proof);
            else
                (void)join_legacy(record);
        }
        if (record->event_call && status != CUDA_SUCCESS) record->observation->failure = status;
        const bool published =
            is_continuation
                ? queue.finish_continuation_issue(record->issue)
                : (prefix == CUDA_SUCCESS
                       ? queue.publish_host_result(record->issue, library_status.value_or(status))
                       : queue.publish_preparation_failure(record->issue, prefix));
        if (!published) queue.quarantine();
        CUresult readback = CUDA_SUCCESS;
        if (record->progress && status == CUDA_SUCCESS && !record->is_cleanup_uncertain)
            readback = record->progress->record_readback(record->stream.native);
        if (uses_default_proxy && !record->is_cleanup_uncertain) {
            const auto proof = check_context();
            if (proof != CUDA_SUCCESS) readback = retain_uncertain(record, proof);
        }
        // Unknown opaque work or lost native context cannot be repaired by a
        // stream event. In particular, never record it under a foreign owner.
        auto event_result =
            record->is_cleanup_uncertain
                ? (record->observation->failure != CUDA_SUCCESS ? record->observation->failure
                                                                : CUDA_ERROR_UNKNOWN)
                : CUDA_SUCCESS;
        if (!record->is_cleanup_uncertain) {
            event_result = dispatch.event_record(record->event, record->stream.native);
            record->is_event_recorded = event_result == CUDA_SUCCESS;
            if (uses_default_proxy) {
                const auto proof = check_context();
                if (proof != CUDA_SUCCESS) event_result = retain_uncertain(record, proof);
            }
        }
        record->completion_status =
            !is_clean_preparation_rejection && library_completion.status != CUDA_SUCCESS
                ? library_completion.status
            : readback != CUDA_SUCCESS ? readback
                                       : event_result;
        if (record->completion_status != CUDA_SUCCESS) {
            queue.quarantine();
            record->observation->failure = record->completion_status;
        }
        // A later physical launch failure is an async logical failure, not a
        // replacement for the application's already published initial result.
        if ((is_continuation || record->reports_errors_asynchronously) && status != CUDA_SUCCESS) {
            if (is_continuation) queue.quarantine();
            record->observation->failure = status;
        }
        if (prefix != CUDA_SUCCESS && !is_clean_preparation_rejection) {
            queue.quarantine();
            if (!library_completion.is_clean_preparation_rejection ||
                record->observation->failure == CUDA_SUCCESS)
                record->observation->failure = prefix;
        }
        return {.state = prefix == CUDA_SUCCESS ? CudaExecutionIssueState::kIssued
                                                : CudaExecutionIssueState::kRejected,
                .sequence = record->sequence,
                .native_status = status,
                .completion_status = record->completion_status,
                .library_status = library_status};
    }

    [[nodiscard]] CudaExecutionPollResult collect(bool should_drain) noexcept {
        if (!queue.owns_executor()) return {.status = CUDA_ERROR_INVALID_CONTEXT};
        if (active_enqueue_turn)
            return {.status = CUDA_ERROR_NOT_READY, .retained = records.size()};
        if (is_callback_active) return {.status = CUDA_ERROR_NOT_READY};
        const CallbackScope callback_scope{is_callback_active};
        CudaExecutionPollResult result{.status = check_context()};
        if (result.status != CUDA_SUCCESS) return result;
        for (std::size_t index = 0; index < records.size();) {
            auto& record = records[index];
            // An external callback may still be running or may enqueue later.
            // Even a successful stream synchronize cannot prove that work
            // stopped before the trusted enqueue-complete transition.
            if (record->is_external && !record->is_event_recorded &&
                !record->is_cleanup_uncertain) {
                ++result.retained;
                ++index;
                continue;
            }
            if (record->is_paused && !record->is_cleanup_uncertain) {
                ++result.paused;
                ++result.retained;
                ++index;
                continue;
            }
            if (!record->is_native_issued || record->is_cleanup_uncertain) {
                if (record->is_cleanup_uncertain) {
                    ++result.retained;
                    if (result.status == CUDA_SUCCESS) result.status = CUDA_ERROR_UNKNOWN;
                }
                ++index;
                continue;
            }
            // An ordinary predecessor's stream synchronize would also wait
            // for an external suffix whose host callback may still enqueue.
            // Observe only the predecessor's recorded event in that case.
            const bool can_synchronize = should_drain && !has_external_stream_hazard(*record) &&
                                         record->enqueue_events.empty();
            // Querying a never-recorded event may report SUCCESS. It cannot
            // prove anything about a launch whose trailing record failed.
            if (!record->is_event_recorded && !can_synchronize) {
                if (record->observation->failure == CUDA_SUCCESS)
                    record->observation->failure = record->completion_status;
                if (result.status == CUDA_SUCCESS) result.status = record->completion_status;
                ++result.retained;
                ++index;
                continue;
            }
            if (const auto status = check_context(); status != CUDA_SUCCESS) {
                (void)retain_uncertain(record, status);
                if (result.status == CUDA_SUCCESS) result.status = status;
                ++result.retained;
                ++index;
                continue;
            }
            CUresult observed = CUDA_SUCCESS;
            if (!record->is_physically_stopped)
                observed = can_synchronize ? dispatch.stream_synchronize(record->stream.native)
                                           : dispatch.event_query(record->event);
            if (!record->is_physically_stopped && observed == CUDA_SUCCESS)
                for (const auto event : record->enqueue_events) {
                    const auto proof = check_context();
                    if (proof != CUDA_SUCCESS) {
                        observed = proof;
                        break;
                    }
                    observed = dispatch.event_query(event);
                    const auto after = check_context();
                    if (after != CUDA_SUCCESS) observed = after;
                    if (observed != CUDA_SUCCESS) break;
                }
            if (record->is_external || uses_default_proxy) {
                const auto context_status = check_context();
                if (context_status != CUDA_SUCCESS ||
                    (observed != CUDA_SUCCESS &&
                     (can_synchronize || observed != CUDA_ERROR_NOT_READY))) {
                    const auto failure = context_status != CUDA_SUCCESS ? context_status : observed;
                    (void)retain_uncertain(record, failure);
                    if (result.status == CUDA_SUCCESS) result.status = failure;
                    ++result.retained;
                    ++index;
                    continue;
                }
            }
            if (!can_synchronize && observed == CUDA_ERROR_NOT_READY) {
                ++result.retained;
                ++index;
                continue;
            }
            if (observed != CUDA_SUCCESS) {
                if (record->observation->failure == CUDA_SUCCESS)
                    record->observation->failure = observed;
                queue.quarantine();
                if (result.status == CUDA_SUCCESS) result.status = observed;
                ++result.retained;
                ++index;
                continue;
            }
            record->is_physically_stopped = true;
            if (record->progress && record->launch_status == CUDA_SUCCESS &&
                record->completion_status == CUDA_SUCCESS) {
                const auto progress = record->progress->observe_physical_stop();
                if (progress.status != CUDA_SUCCESS) {
                    if (record->observation->failure == CUDA_SUCCESS)
                        record->observation->failure = progress.status;
                    queue.quarantine();
                    if (result.status == CUDA_SUCCESS) result.status = progress.status;
                } else if (!progress.is_complete) {
                    const auto cleaned = destroy_event(record);
                    if (cleaned != CUDA_SUCCESS ||
                        !queue.pause_after_physical_stop(record->issue)) {
                        record->is_cleanup_uncertain = true;
                        const auto failure = cleaned != CUDA_SUCCESS ? cleaned : CUDA_ERROR_UNKNOWN;
                        if (record->observation->failure == CUDA_SUCCESS)
                            record->observation->failure = failure;
                        queue.quarantine();
                        if (result.status == CUDA_SUCCESS) result.status = failure;
                    } else {
                        record->is_paused = true;
                        ++result.paused;
                    }
                    ++result.retained;
                    ++index;
                    continue;
                }
            }
            const auto retired = retire(record);
            if (retired != CUDA_SUCCESS) {
                if (result.status == CUDA_SUCCESS) result.status = retired;
                ++result.retained;
                ++index;
                continue;
            }
            if (retirement_observer && !record->is_external && record->sequence) {
                const auto status =
                    record->observation->failure != CUDA_SUCCESS ? record->observation->failure
                    : record->launch_status != CUDA_SUCCESS      ? record->launch_status
                                                                 : record->completion_status;
                retirement_observer->retired(record->sequence, static_cast<int>(status));
            }
            records.erase(records.begin() + static_cast<std::ptrdiff_t>(index));
            ++result.retired;
        }
        return result;
    }
};

CudaExecutionDispatcher::CudaExecutionDispatcher(
    ExecutionSessionIdentity session, CudaExecutionContext context, CudaExecutionDispatch dispatch,
    ExecutionSubmissionLimits limits, std::size_t max_bindings, bool is_profile_isolated,
    CudaExecutionStreamConfig streams,
    std::shared_ptr<const CudaExecutionOwnerActivity> owner_activity,
    std::function<CUresult()> exclusive_native_gate)
    : implementation_(std::make_unique<Implementation>(
          session, std::move(context), dispatch, limits, max_bindings, is_profile_isolated,
          std::move(streams), std::move(owner_activity), std::move(exclusive_native_gate))) {}

CudaExecutionDispatcher::~CudaExecutionDispatcher() {
    auto& state = *implementation_;
    const bool is_parent = state.queue.identity().client_id != 0;
    if (is_parent && !state.has_bound_executor && state.records.empty() && state.queue.shutdown())
        return;
    if (!is_parent || !state.queue.owns_executor() || !shutdown()) {
        // In particular, never enter an SDK or inherited mutex after fork.
        const auto* retained = implementation_.release();
        if (retained->queue.identity().client_id != 0) {
            try {
                spdlog::critical("Retaining unsettled CUDA dispatcher: context={}",
                                 retained->context.id);
            } catch (...) {
            }
        }
    }
}

CUresult CudaExecutionDispatcher::bind_executor() noexcept {
    auto& state = *implementation_;
    if (!state.queue.identity().client_id) return CUDA_ERROR_INVALID_CONTEXT;
    if (state.has_bound_executor && !state.queue.owns_executor()) return CUDA_ERROR_INVALID_CONTEXT;
    if (state.is_callback_active) return CUDA_ERROR_NOT_READY;
    const CallbackScope callback_scope{state.is_callback_active};
    // Binding is a trusted cold call before this owner is published to ingress.
    CUcontext current{};
    const auto result = state.dispatch.context_get_current(&current);
    if (result != CUDA_SUCCESS) return result;
    if (current != state.context.native || !state.queue.bind_executor())
        return CUDA_ERROR_INVALID_CONTEXT;
    state.has_bound_executor = true;
    return CUDA_SUCCESS;
}

bool CudaExecutionDispatcher::bind_function(std::uint64_t id, CUfunction function,
                                            std::size_t parameter_bytes,
                                            std::shared_ptr<const void> lifetime) noexcept {
    auto& state = *implementation_;
    if (!state.queue.owns_executor() || state.is_callback_active || id <= state.last_function_id ||
        !function || !lifetime || parameter_bytes > state.limits.max_parameter_bytes ||
        state.functions.size() == state.max_bindings ||
        std::ranges::any_of(state.functions, [id](const auto& entry) { return entry.id == id; }))
        return false;
    state.functions.push_back({id, function, parameter_bytes, std::move(lifetime)});
    state.last_function_id = id;
    return true;
}

bool CudaExecutionDispatcher::bind_stream(std::uint64_t id, CUstream stream,
                                          std::shared_ptr<const void> lifetime,
                                          bool is_nonblocking) noexcept {
    auto& state = *implementation_;
    if (!state.queue.owns_executor() || state.is_callback_active || id <= state.last_stream_id ||
        !stream || stream == CU_STREAM_LEGACY || stream == CU_STREAM_PER_THREAD || !lifetime ||
        state.streams.size() == state.max_bindings ||
        std::ranges::any_of(state.streams, [id](const auto& entry) { return entry.id == id; }))
        return false;
    state.streams.push_back({id, stream, std::move(lifetime), is_nonblocking});
    state.last_stream_id = id;
    return true;
}

bool CudaExecutionDispatcher::resources_quiescent() const noexcept {
    const auto& state = *implementation_;
    if (!state.queue.owns_executor() || state.is_callback_active) return false;
    const auto snapshot = state.queue.snapshot();
    return state.records.empty() && snapshot.outstanding == 0 && !snapshot.is_quarantined;
}
bool CudaExecutionDispatcher::physical_readers_quiescent() const noexcept {
    const auto& state = *implementation_;
    return state.queue.owns_executor() && !state.is_callback_active && state.records.empty() &&
           state.queue.snapshot().outstanding == 0;
}
bool CudaExecutionDispatcher::unbind_function(std::uint64_t id) noexcept {
    if (!resources_quiescent()) return false;
    auto& functions = implementation_->functions;
    const auto found =
        std::ranges::find_if(functions, [id](const auto& entry) { return entry.id == id; });
    if (found == functions.end()) return false;
    functions.erase(found);
    return true;
}
bool CudaExecutionDispatcher::unbind_stream(std::uint64_t id) noexcept {
    if (id == 0 || !resources_quiescent()) return false;
    auto& streams = implementation_->streams;
    const auto found =
        std::ranges::find_if(streams, [id](const auto& entry) { return entry.id == id; });
    if (found == streams.end()) return false;
    streams.erase(found);
    return true;
}

ExecutionAcceptance CudaExecutionDispatcher::accept_kernel(
    std::uint64_t sequence, const CudaExecutionKernel& kernel) noexcept {
    return accept_kernel_impl(sequence, kernel, nullptr);
}
ExecutionAcceptance CudaExecutionDispatcher::accept_async_original_kernel(
    std::uint64_t sequence, const CudaExecutionKernel& kernel) noexcept {
    return accept_kernel_impl(sequence, kernel, nullptr, KernelAdmission::kOriginalAsync);
}
ExecutionAcceptance CudaExecutionDispatcher::accept_deferred_kernel(
    std::uint64_t sequence, const CudaExecutionKernel& kernel,
    std::size_t parameter_capacity_bytes) noexcept {
    if (parameter_capacity_bytes < kernel.parameters.size())
        return {.status = ExecutionAcceptStatus::kInvalid, .receipt = {}};
    return accept_kernel_impl(sequence, kernel, nullptr, KernelAdmission::kDeferredAsync,
                              parameter_capacity_bytes);
}
ExecutionAcceptance CudaExecutionDispatcher::accept_persistent_kernel(
    std::uint64_t sequence, const CudaExecutionKernel& logical,
    const CudaExecutionPersistentKernel& persistent) noexcept {
    return accept_kernel_impl(sequence, logical, &persistent);
}
ExecutionAcceptance CudaExecutionDispatcher::accept_memory(
    std::uint64_t sequence, const CudaExecutionMemory& memory) noexcept {
    auto& state = *implementation_;
    auto invalid = ExecutionAcceptance{.status = ExecutionAcceptStatus::kInvalid, .receipt = {}};
    if (!state.queue.owns_executor() || state.is_callback_active ||
        (!memory.is_asynchronous && memory.stream))
        return invalid;
    bool is_supported{};
    std::size_t width{1};
    const bool is_host =
        memory.kind == CudaMemoryKind::kUpload || memory.kind == CudaMemoryKind::kDownload;
    if (!is_host && (memory.host || memory.host_lifetime)) return invalid;
    if (memory.kind != CudaMemoryKind::kCopyUva &&
        (memory.endpoint_lifetimes[0] || memory.endpoint_lifetimes[1]))
        return invalid;
    switch (memory.kind) {
        case CudaMemoryKind::kUpload:
        case CudaMemoryKind::kDownload:
            if (!memory.is_asynchronous || memory.value ||
                (memory.count && (!memory.host || !memory.host_lifetime)) ||
                (memory.kind == CudaMemoryKind::kUpload && memory.source) ||
                (memory.kind == CudaMemoryKind::kDownload && memory.destination))
                return invalid;
            is_supported = memory.kind == CudaMemoryKind::kUpload
                               ? state.dispatch.upload_async != nullptr
                               : state.dispatch.download_async != nullptr;
            break;
        case CudaMemoryKind::kCopy:
            is_supported = memory.is_asynchronous ? state.dispatch.copy_device_async != nullptr
                                                  : state.dispatch.copy_device != nullptr;
            if (memory.value) return invalid;
            break;
        case CudaMemoryKind::kCopyUva:
            is_supported = memory.is_asynchronous ? state.dispatch.copy_unified_async != nullptr
                                                  : state.dispatch.copy_unified != nullptr;
            if (memory.value) return invalid;
            break;
        case CudaMemoryKind::kFill8:
            is_supported = memory.is_asynchronous ? state.dispatch.fill8_async != nullptr
                                                  : state.dispatch.fill8 != nullptr;
            if (memory.source || memory.value > UINT8_MAX) return invalid;
            break;
        case CudaMemoryKind::kFill16:
            width = 2;
            is_supported = memory.is_asynchronous ? state.dispatch.fill16_async != nullptr
                                                  : state.dispatch.fill16 != nullptr;
            if (memory.source || memory.value > UINT16_MAX) return invalid;
            break;
        case CudaMemoryKind::kFill32:
            width = 4;
            is_supported = memory.is_asynchronous ? state.dispatch.fill32_async != nullptr
                                                  : state.dispatch.fill32 != nullptr;
            if (memory.source) return invalid;
            break;
        default:
            return invalid;
    }
    if (!is_supported || memory.destination % width ||
        memory.count > std::numeric_limits<std::size_t>::max() / width)
        return invalid;
    const auto bytes = memory.count * width;
    const bool has_source = memory.kind == CudaMemoryKind::kCopy ||
                            memory.kind == CudaMemoryKind::kCopyUva ||
                            memory.kind == CudaMemoryKind::kDownload;
    if ((bytes && ((memory.kind != CudaMemoryKind::kDownload && !memory.destination) ||
                   (has_source && !memory.source))) ||
        bytes > std::numeric_limits<CUdeviceptr>::max() - memory.destination ||
        (has_source && bytes > std::numeric_limits<CUdeviceptr>::max() - memory.source))
        return invalid;
    const auto stream = std::ranges::find_if(
        state.streams, [&](const auto& entry) { return entry.id == memory.stream; });
    if (stream == state.streams.end()) return invalid;
    if (state.records.size() == state.limits.max_commands)
        return {.status = ExecutionAcceptStatus::kFull, .receipt = {}};
    if (state.last_acceptance_order == UINT64_MAX)
        return {.status = ExecutionAcceptStatus::kOutOfMemory, .receipt = {}};
    try {
        auto record = std::make_shared<Implementation::Record>();
        record->sequence = sequence;
        record->memory = memory;
        if (state.context.process_memory) {
            auto captured = state.context.process_memory->borrow_live();
            if (captured.status != CUDA_SUCCESS)
                return {.status = captured.status == CUDA_ERROR_OUT_OF_MEMORY
                                      ? ExecutionAcceptStatus::kOutOfMemory
                                      : ExecutionAcceptStatus::kInvalid,
                        .receipt = {}};
            record->process_memory_lifetime = std::move(captured.lifetime);
        }
        record->stream = *stream;
        record->context_lifetime = state.context.lifetime;
        // Fixed command metadata is bounded by max_commands, like a kernel's
        // geometry/stream. No caller parameter buffer or copy data is retained.
        if (state.prepare_command_capture(record) != CUDA_SUCCESS)
            return {.status = ExecutionAcceptStatus::kOutOfMemory, .receipt = {}};
        auto accepted = state.queue.accept(sequence, {.operation = 2,
                                                      .context = state.context.id,
                                                      .stream = memory.stream,
                                                      .parameters = {},
                                                      .lifetime = record});
        if (accepted.accepted()) state.publish_command(record, accepted.receipt);
        return accepted;
    } catch (...) {
        return {.status = ExecutionAcceptStatus::kOutOfMemory, .receipt = {}};
    }
}
ExecutionHostAllocation CudaExecutionDispatcher::allocate_host_storage(std::size_t bytes) noexcept {
    auto& state = *implementation_;
    if (!state.queue.owns_executor() || state.is_callback_active) return {};
    return state.queue.allocate_host_storage(bytes);
}
ExecutionAcceptance CudaExecutionDispatcher::accept_library(
    std::uint64_t sequence, const CudaExecutionLibrary& library) noexcept {
    auto& state = *implementation_;
    auto invalid = ExecutionAcceptance{.status = ExecutionAcceptStatus::kInvalid, .receipt = {}};
    if (!state.queue.owns_executor() || state.is_callback_active || !library.call ||
        (!library.parameters.empty() && !library.parameters.data()))
        return invalid;
    const auto stream = std::ranges::find_if(
        state.streams, [&](const auto& entry) { return entry.id == library.stream; });
    if (stream == state.streams.end()) return invalid;
    if (state.records.size() == state.limits.max_commands)
        return {.status = ExecutionAcceptStatus::kFull, .receipt = {}};
    if (state.last_acceptance_order == UINT64_MAX)
        return {.status = ExecutionAcceptStatus::kOutOfMemory, .receipt = {}};
    try {
        auto record = std::make_shared<Implementation::Record>();
        record->sequence = sequence;
        record->library = library.call;
        if (state.context.process_memory && library.call->borrows_process_memory()) {
            auto captured = state.context.process_memory->borrow_live();
            if (captured.status != CUDA_SUCCESS)
                return {.status = captured.status == CUDA_ERROR_OUT_OF_MEMORY
                                      ? ExecutionAcceptStatus::kOutOfMemory
                                      : ExecutionAcceptStatus::kInvalid,
                        .receipt = {}};
            record->process_memory_lifetime = std::move(captured.lifetime);
        }
        record->stream = *stream;
        record->context_lifetime = state.context.lifetime;
        if (state.prepare_command_capture(record) != CUDA_SUCCESS)
            return {.status = ExecutionAcceptStatus::kOutOfMemory, .receipt = {}};
        auto accepted = state.queue.accept(sequence, {.operation = 3,
                                                      .context = state.context.id,
                                                      .stream = library.stream,
                                                      .parameters = library.parameters,
                                                      .lifetime = record});
        if (accepted.accepted()) state.publish_command(record, accepted.receipt);
        return accepted;
    } catch (...) {
        return {.status = ExecutionAcceptStatus::kOutOfMemory, .receipt = {}};
    }
}
ExecutionAcceptance CudaExecutionDispatcher::accept_event(
    std::uint64_t sequence, const CudaExecutionEvent& event) noexcept {
    return accept_event_impl(sequence, event, false).acceptance;
}
CudaExecutionEventAcceptance CudaExecutionDispatcher::accept_event_with_frontier(
    std::uint64_t sequence, const CudaExecutionEvent& event) noexcept {
    return accept_event_impl(sequence, event, true);
}
CudaExecutionEventAcceptance CudaExecutionDispatcher::accept_event_impl(
    std::uint64_t sequence, const CudaExecutionEvent& event, bool captures_frontier) noexcept {
    auto& state = *implementation_;
    const auto failure = [](ExecutionAcceptStatus status) {
        return CudaExecutionEventAcceptance{{.status = status, .receipt = {}}, {}};
    };
    auto invalid = failure(ExecutionAcceptStatus::kInvalid);
    if (!state.queue.owns_executor() || state.is_callback_active || !event.call) return invalid;
    if (event.dependency && (!event.dependency->state_ ||
                             event.dependency->state_->owner.lock() != state.frontier_owner))
        return invalid;
    if (event.foreign_dependency) {
        if (event.dependency ||
            !event.foreign_dependency->same_capture(*event.foreign_dependency) ||
            event.foreign_dependency->state_->owner.lock() == state.frontier_owner)
            return invalid;
        const auto status = event.foreign_dependency->observe();
        if (status != CUDA_SUCCESS && status != CUDA_ERROR_NOT_READY) return invalid;
    }
    const auto stream = std::ranges::find(state.streams, event.stream, &Implementation::Stream::id);
    if (stream == state.streams.end()) return invalid;
    if (state.records.size() == state.limits.max_commands)
        return failure(ExecutionAcceptStatus::kFull);
    if (state.last_acceptance_order == UINT64_MAX)
        return failure(ExecutionAcceptStatus::kOutOfMemory);
    if (captures_frontier &&
        static_cast<std::size_t>(std::ranges::count_if(state.frontiers, [](const auto& item) {
            return !item.expired();
        })) >= state.max_bindings)
        return failure(ExecutionAcceptStatus::kOutOfMemory);
    try {
        auto record = std::make_shared<Implementation::Record>();
        record->sequence = sequence;
        record->event_call = event;
        record->observation->foreign_dependency = event.foreign_dependency;
        record->stream = *stream;
        record->context_lifetime = state.context.lifetime;
        std::shared_ptr<CudaExecutionFrontier::State> captured;
        if (captures_frontier) {
            captured = std::make_shared<CudaExecutionFrontier::State>();
            captured->diagnostic_id = detail::take_diagnostic_id(state.next_diagnostic_id);
            if (!frontier_thread_owner) frontier_thread_owner = std::make_shared<int>(0);
            captured->owner = state.frontier_owner;
            captured->thread = frontier_thread_owner;
            captured->process = ::getpid();
            captured->observations.push_back(record->observation);
        }
        if (state.prepare_command_capture(record, captured) != CUDA_SUCCESS)
            return failure(ExecutionAcceptStatus::kOutOfMemory);
        auto accepted = state.queue.accept(sequence, {.operation = 4,
                                                      .context = state.context.id,
                                                      .stream = event.stream,
                                                      .parameters = {},
                                                      .lifetime = record});
        CudaExecutionEventAcceptance result{accepted, {}};
        if (accepted.accepted()) {
            state.publish_command(record, accepted.receipt);
            if (captured) {
                // Both vectors are reserved at construction. All allocations
                // and quota checks precede queue acceptance and sequence use.
                std::erase_if(state.frontiers, [](const auto& item) { return item.expired(); });
                state.frontiers.push_back(captured);
                result.frontier_capture = captured->diagnostic_id;
                result.frontier.state_ = std::move(captured);
            }
        }
        return result;
    } catch (...) {
        return failure(ExecutionAcceptStatus::kOutOfMemory);
    }
}
CudaExecutionExternalAcceptance CudaExecutionDispatcher::accept_external_barrier(
    std::uint64_t stream_id, const std::shared_ptr<const void>& lifetime) noexcept {
    auto accepted = accept_external_cohort(std::span{&stream_id, std::size_t{1}}, lifetime);
    if (accepted.status != CUDA_SUCCESS) return {.status = accepted.status};
    auto& member = accepted.members.front();
    return {CUDA_SUCCESS, std::move(member.barrier), std::move(member.frontier)};
}
CudaExecutionExternalCohortAcceptance CudaExecutionDispatcher::accept_external_cohort(
    std::span<const std::uint64_t> stream_ids,
    const std::shared_ptr<const void>& lifetime) noexcept {
    auto& state = *implementation_;
    if (!state.queue.owns_executor()) return {.status = CUDA_ERROR_INVALID_CONTEXT};
    if (state.is_callback_active) return {.status = CUDA_ERROR_NOT_READY};
    const auto snapshot = state.queue.snapshot();
    if (snapshot.is_quarantined) return {.status = CUDA_ERROR_UNKNOWN};
    if (snapshot.is_closing) return {.status = CUDA_ERROR_DEINITIALIZED};
    if (!lifetime || stream_ids.empty() || !stream_ids.data() ||
        stream_ids.size() > state.limits.max_commands)
        return {.status = CUDA_ERROR_INVALID_VALUE};
    try {
        std::vector<Implementation::Stream> streams;
        streams.reserve(stream_ids.size());
        for (const auto id : stream_ids) {
            const auto stream = std::ranges::find(state.streams, id, &Implementation::Stream::id);
            if (stream == state.streams.end()) return {.status = CUDA_ERROR_INVALID_HANDLE};
            if (std::ranges::none_of(streams, [id](const auto& item) { return item.id == id; }))
                streams.push_back(*stream);
        }
        const auto count = streams.size();
        const auto live_frontiers = static_cast<std::size_t>(std::ranges::count_if(
            state.frontiers, [](const auto& item) { return !item.expired(); }));
        if (state.records.size() > state.limits.max_commands ||
            count > state.limits.max_commands - state.records.size() ||
            live_frontiers > state.max_bindings || count > state.max_bindings - live_frontiers ||
            count > UINT64_MAX - state.last_acceptance_order)
            return {.status = CUDA_ERROR_OUT_OF_MEMORY};
        const auto cohort = std::make_shared<const Implementation::ExternalCohort>(
            Implementation::ExternalCohort{state.last_acceptance_order + 1, std::move(streams)});
        std::vector<std::shared_ptr<Implementation::Record>> records;
        records.reserve(count);
        std::vector<std::shared_ptr<const CudaExecutionFrontier::State>> frontiers;
        frontiers.reserve(count);
        CudaExecutionExternalCohortAcceptance accepted{.status = CUDA_SUCCESS};
        accepted.members.reserve(count);
        if (!frontier_thread_owner) frontier_thread_owner = std::make_shared<int>(0);
        for (std::size_t index = 0; index != count; ++index) {
            auto record = std::make_shared<Implementation::Record>();
            record->is_external = true;
            record->acceptance_order = cohort->first_order + index;
            record->observation->acceptance_order = record->acceptance_order;
            record->stream = cohort->streams[index];
            record->context_lifetime = state.context.lifetime;
            record->external_lifetime = lifetime;
            record->external_cohort = cohort;
            auto captured = std::make_shared<CudaExecutionFrontier::State>();
            captured->diagnostic_id = detail::take_diagnostic_id(state.next_diagnostic_id);
            captured->owner = state.frontier_owner;
            captured->thread = frontier_thread_owner;
            captured->process = ::getpid();
            captured->observations.push_back(record->observation);
            CudaExecutionExternalCohortMember member{.stream = record->stream.id};
            member.barrier.owner_ = state.frontier_owner;
            member.barrier.id_ = record->acceptance_order;
            member.frontier.state_ = captured;
            accepted.members.push_back(std::move(member));
            frontiers.push_back(std::move(captured));
            records.push_back(std::move(record));
        }
        // No SDK or application callback runs here. Every allocation and bound
        // check precedes publication; these owner-serialized vectors already
        // have capacity for the entire cohort. Failed preparation changes no
        // record, frontier registration or private acceptance-order counter.
        std::erase_if(state.frontiers, [](const auto& item) { return item.expired(); });
        for (std::size_t index = 0; index != count; ++index) {
            state.frontiers.push_back(frontiers[index]);
            state.records.push_back(std::move(records[index]));
        }
        state.last_acceptance_order += count;
        return accepted;
    } catch (...) {
        return {.status = CUDA_ERROR_OUT_OF_MEMORY};
    }
}
CUresult CudaExecutionDispatcher::external_barrier_ready(
    const CudaExecutionExternalBarrier& barrier) const noexcept {
    auto& state = *implementation_;
    if (!state.queue.owns_executor()) return CUDA_ERROR_INVALID_CONTEXT;
    if (state.is_callback_active) return CUDA_ERROR_NOT_READY;
    const auto record = state.external_record(barrier);
    if (!record) return CUDA_ERROR_INVALID_HANDLE;
    if (record->is_cleanup_uncertain || state.queue.snapshot().is_quarantined)
        return CUDA_ERROR_UNKNOWN;
    if (record->has_external_started) return CUDA_ERROR_NOT_READY;
    const CallbackScope callback_scope{state.is_callback_active};
    return state.is_ready(*record) ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
}
CUresult CudaExecutionDispatcher::begin_external_barrier(
    const CudaExecutionExternalBarrier& barrier) noexcept {
    auto& state = *implementation_;
    if (!state.queue.owns_executor()) return CUDA_ERROR_INVALID_CONTEXT;
    if (const auto status = state.issue_health(); status != CUDA_SUCCESS) return status;
    if (state.is_callback_active || state.active_enqueue_turn) return CUDA_ERROR_NOT_READY;
    auto record = state.external_record(barrier);
    if (!record) return CUDA_ERROR_INVALID_HANDLE;
    if (record->has_external_started) return CUDA_ERROR_INVALID_VALUE;
    if (record->is_cleanup_uncertain || state.queue.snapshot().is_quarantined)
        return CUDA_ERROR_UNKNOWN;
    const CallbackScope callback_scope{state.is_callback_active};
    if (!state.is_ready(*record)) return CUDA_ERROR_NOT_READY;
    if (const auto status = state.check_context(); status != CUDA_SUCCESS)
        return state.retain_uncertain(record, status);
    record->has_external_started = true;
    const auto created = state.dispatch.event_create(&record->event, CU_EVENT_DISABLE_TIMING);
    if (const auto status = state.check_context(); status != CUDA_SUCCESS)
        return state.retain_uncertain(record, status);
    if (created == CUDA_SUCCESS) {
        if (!record->event) return state.retain_uncertain(record, CUDA_ERROR_UNKNOWN);
        const auto prepared = state.prepare_dependencies(*record);
        return prepared == CUDA_SUCCESS ? CUDA_SUCCESS : state.retain_uncertain(record, prepared);
    }
    // The actor has not entered its native operation. A failed event creation
    // is known no-work only after its possible output is cleaned exactly once
    // under the independently proven owner context.
    if (record->event) {
        if (const auto status = state.destroy_event(record); status != CUDA_SUCCESS) return status;
    }
    record->observation->failure = created;
    record->observation->is_settled = true;
    std::erase(state.records, record);
    return created;
}
CUresult CudaExecutionDispatcher::record_external_enqueue(
    const CudaExecutionExternalBarrier& barrier) noexcept {
    auto& state = *implementation_;
    if (!state.queue.owns_executor()) return CUDA_ERROR_INVALID_CONTEXT;
    if (state.is_callback_active || state.active_enqueue_turn) return CUDA_ERROR_NOT_READY;
    auto record = state.external_record(barrier);
    if (!record) return CUDA_ERROR_INVALID_HANDLE;
    if (!record->has_external_started || record->is_event_recorded) return CUDA_ERROR_INVALID_VALUE;
    if (record->is_cleanup_uncertain) return CUDA_ERROR_UNKNOWN;
    const CallbackScope callback_scope{state.is_callback_active};
    if (const auto status = state.check_context(); status != CUDA_SUCCESS)
        return state.retain_uncertain(record, status);
    const auto recorded = state.dispatch.event_record(record->event, record->stream.native);
    if (const auto status = state.check_context(); status != CUDA_SUCCESS)
        return state.retain_uncertain(record, status);
    if (recorded != CUDA_SUCCESS) return state.retain_uncertain(record, recorded);
    // This is an external enqueue boundary, never a requested native SDK
    // return. Subsequent stream work can now rely on native FIFO ordering.
    record->is_event_recorded = true;
    record->is_native_issued = true;
    return CUDA_SUCCESS;
}
CUresult CudaExecutionDispatcher::cancel_external_barrier(
    const CudaExecutionExternalBarrier& barrier) noexcept {
    auto& state = *implementation_;
    if (!state.queue.owns_executor()) return CUDA_ERROR_INVALID_CONTEXT;
    if (state.is_callback_active) return CUDA_ERROR_NOT_READY;
    auto record = state.external_record(barrier);
    if (!record) return CUDA_ERROR_INVALID_HANDLE;
    if (record->has_external_started) return CUDA_ERROR_NOT_READY;
    if (record->is_cleanup_uncertain) return CUDA_ERROR_UNKNOWN;
    const CallbackScope callback_scope{state.is_callback_active};
    record->observation->is_settled = true;
    std::erase(state.records, record);
    return CUDA_SUCCESS;
}
CUresult CudaExecutionDispatcher::fail_external_barrier(const CudaExecutionExternalBarrier& barrier,
                                                        CUresult failure) noexcept {
    auto& state = *implementation_;
    if (!state.queue.owns_executor()) return CUDA_ERROR_INVALID_CONTEXT;
    if (state.is_callback_active) return CUDA_ERROR_NOT_READY;
    auto record = state.external_record(barrier);
    if (!record) return CUDA_ERROR_INVALID_HANDLE;
    if (!record->has_external_started || failure == CUDA_SUCCESS || record->is_cleanup_uncertain)
        return CUDA_ERROR_INVALID_VALUE;
    (void)state.retain_uncertain(record, failure);
    return CUDA_SUCCESS;
}
std::size_t CudaExecutionDispatcher::external_barrier_count() const noexcept {
    const auto& state = *implementation_;
    if (!state.queue.owns_executor() || state.is_callback_active) return 0;
    return static_cast<std::size_t>(std::ranges::count_if(
        state.records, [](const auto& record) { return record->is_external && !record->enqueue; }));
}
std::size_t CudaExecutionDispatcher::started_external_count() const noexcept {
    const auto& state = *implementation_;
    if (!state.queue.owns_executor()) return 0;
    return static_cast<std::size_t>(std::ranges::count_if(state.records, [](const auto& record) {
        return record->is_external && record->has_external_started;
    }));
}

CudaExecutionEnqueueAcceptance CudaExecutionDispatcher::accept_enqueue_turn(
    std::uint64_t stream, const std::shared_ptr<const void>& lifetime) noexcept {
    return accept_enqueue_cohort(std::span{&stream, std::size_t{1}}, lifetime);
}
CudaExecutionEnqueueAcceptance CudaExecutionDispatcher::accept_enqueue_cohort(
    std::span<const std::uint64_t> streams, const std::shared_ptr<const void>& lifetime) noexcept {
    auto& state = *implementation_;
    if (!state.queue.owns_executor()) return {.status = CUDA_ERROR_INVALID_CONTEXT};
    if (state.is_callback_active) return {.status = CUDA_ERROR_NOT_READY};
    const auto snapshot = state.queue.snapshot();
    if (snapshot.is_quarantined) return {.status = CUDA_ERROR_UNKNOWN};
    if (snapshot.is_closing) return {.status = CUDA_ERROR_DEINITIALIZED};
    if (!lifetime || streams.empty() || !streams.data() || streams.size() > state.max_bindings)
        return {.status = CUDA_ERROR_INVALID_VALUE};
    if (state.records.size() == state.limits.max_commands ||
        state.last_acceptance_order == UINT64_MAX)
        return {.status = CUDA_ERROR_OUT_OF_MEMORY};
    const auto live_frontiers = static_cast<std::size_t>(std::ranges::count_if(
        state.frontiers, [](const auto& frontier) { return !frontier.expired(); }));
    if (live_frontiers >= state.max_bindings) return {.status = CUDA_ERROR_OUT_OF_MEMORY};
    try {
        if (!frontier_thread_owner) frontier_thread_owner = std::make_shared<int>(0);
        auto cohort = std::make_shared<Implementation::ExternalCohort>();
        cohort->first_order = state.last_acceptance_order + 1;
        cohort->streams.reserve(streams.size());
        for (const auto stream : streams) {
            const auto found =
                std::ranges::find(state.streams, stream, &Implementation::Stream::id);
            if (found == state.streams.end()) return {.status = CUDA_ERROR_INVALID_HANDLE};
            if (std::ranges::find(cohort->streams, stream, &Implementation::Stream::id) ==
                cohort->streams.end())
                cohort->streams.push_back(*found);
        }
        auto record = std::make_shared<Implementation::Record>();
        record->enqueue = std::make_shared<CudaExecutionEnqueueState>();
        record->enqueue->process = ::getpid();
        record->enqueue->owner = state.frontier_owner;
        record->enqueue->thread = frontier_thread_owner;
        record->enqueue->lifetime = lifetime;
        record->is_external = true;
        record->stream = cohort->streams.front();
        record->enqueue_events.resize(cohort->streams.size() - 1);
        record->external_cohort = std::move(cohort);
        record->context_lifetime = state.context.lifetime;
        record->external_lifetime = lifetime;
        record->acceptance_order = state.last_acceptance_order + 1;
        record->observation->acceptance_order = record->acceptance_order;
        const auto captured = state.capture_predecessors(record);
        if (captured != CUDA_SUCCESS) return {.status = captured};
        auto frontier = std::make_shared<CudaExecutionFrontier::State>();
        frontier->diagnostic_id = detail::take_diagnostic_id(state.next_diagnostic_id);
        frontier->owner = state.frontier_owner;
        frontier->thread = frontier_thread_owner;
        frontier->process = ::getpid();
        frontier->observations.push_back(record->observation);
        CudaExecutionEnqueueAcceptance accepted{.status = CUDA_SUCCESS};
        accepted.turn.owner_ = state.frontier_owner;
        accepted.turn.id_ = record->acceptance_order;
        accepted.worker.state_ = record->enqueue;
        accepted.frontier.state_ = frontier;
        std::erase_if(state.frontiers, [](const auto& old) { return old.expired(); });
        state.frontiers.push_back(frontier);
        state.records.push_back(std::move(record));
        ++state.last_acceptance_order;
        return accepted;
    } catch (...) {
        return {.status = CUDA_ERROR_OUT_OF_MEMORY};
    }
}
CUresult CudaExecutionDispatcher::enqueue_turn_ready(
    const CudaExecutionEnqueueTurn& turn) const noexcept {
    const auto& state = *implementation_;
    if (!state.queue.owns_executor()) return CUDA_ERROR_INVALID_CONTEXT;
    if (state.is_callback_active || state.active_enqueue_turn) return CUDA_ERROR_NOT_READY;
    const auto record = state.enqueue_record(turn);
    if (!record) return CUDA_ERROR_INVALID_HANDLE;
    if (record->enqueue->phase != EnqueuePhase::kAccepted) return CUDA_ERROR_INVALID_VALUE;
    if (state.queue.snapshot().is_quarantined) return CUDA_ERROR_UNKNOWN;
    const auto failure = Implementation::observed_failure(record->predecessors);
    if (failure.status != CUDA_SUCCESS) return failure.status;
    return state.is_ready(*record) ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
}
CudaExecutionIssueResult CudaExecutionDispatcher::progress_enqueue_dependency(
    const CudaExecutionEnqueueTurn& turn) noexcept {
    auto& state = *implementation_;
    const auto failure = [](CUresult status) {
        return CudaExecutionIssueResult{.state = CudaExecutionIssueState::kError,
                                        .native_status = status};
    };
    if (!state.queue.owns_executor()) return failure(CUDA_ERROR_INVALID_CONTEXT);
    if (state.is_callback_active) return failure(CUDA_ERROR_NOT_READY);
    const auto target = state.enqueue_record(turn);
    if (!target) return failure(CUDA_ERROR_INVALID_HANDLE);
    if (state.queue.snapshot().is_quarantined) return failure(CUDA_ERROR_UNKNOWN);
    if (state.active_enqueue_turn) return failure(CUDA_ERROR_NOT_READY);
    if (target->enqueue->phase != EnqueuePhase::kAccepted) return failure(CUDA_ERROR_INVALID_VALUE);
    const auto ancestor = Implementation::observed_failure(target->predecessors);
    if (ancestor.status != CUDA_SUCCESS) return failure(ancestor.status);
    const CallbackScope scope{state.is_callback_active};
    if (const auto status = state.check_context(); status != CUDA_SUCCESS) return failure(status);
    try {
        return state.progress_closure(
            state.dependency_closure(target, nullptr),
            state.is_ready(*target) ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY);
    } catch (const std::bad_alloc&) {
        return failure(CUDA_ERROR_OUT_OF_MEMORY);
    } catch (...) {
        return failure(CUDA_ERROR_UNKNOWN);
    }
}
CudaExecutionDependencyInspection CudaExecutionDispatcher::inspect_enqueue_dependency(
    const CudaExecutionEnqueueTurn& turn) const noexcept {
    auto& state = *implementation_;
    if (!state.queue.owns_executor()) return {.status = CUDA_ERROR_INVALID_CONTEXT};
    if (state.is_callback_active) return {.status = CUDA_ERROR_NOT_READY};
    const auto target = state.enqueue_record(turn);
    if (!target) return {.status = CUDA_ERROR_INVALID_HANDLE};
    if (state.queue.snapshot().is_quarantined) return {.status = CUDA_ERROR_UNKNOWN};
    if (state.active_enqueue_turn) return {.status = CUDA_ERROR_NOT_READY};
    if (target->enqueue->phase != EnqueuePhase::kAccepted)
        return {.status = CUDA_ERROR_INVALID_VALUE};
    auto ancestor = Implementation::observed_failure(target->predecessors);
    if (ancestor.status != CUDA_SUCCESS) return ancestor;
    const CallbackScope scope{state.is_callback_active};
    try {
        return state.inspect_closure(state.dependency_closure(target, nullptr),
                                     state.is_ready(*target) ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY);
    } catch (const std::bad_alloc&) {
        return {.status = CUDA_ERROR_OUT_OF_MEMORY};
    } catch (...) {
        return {.status = CUDA_ERROR_UNKNOWN};
    }
}
bool CudaExecutionDispatcher::owns_frontier(const CudaExecutionFrontier& frontier) const noexcept {
    const auto& state = *implementation_;
    return state.queue.owns_executor() && frontier.same_capture(frontier) &&
           frontier.state_->owner.lock() == state.frontier_owner;
}
CudaExecutionDependencyInspection CudaExecutionDispatcher::inspect_frontier_dependency(
    const CudaExecutionFrontier& frontier) const noexcept {
    return inspect_frontier_impl(frontier, FrontierPurpose::kExecution);
}
CudaExecutionDependencyInspection CudaExecutionDispatcher::inspect_frontier_retirement(
    const CudaExecutionFrontier& frontier) const noexcept {
    return inspect_frontier_impl(frontier, FrontierPurpose::kRetirement);
}
CudaExecutionDependencyInspection CudaExecutionDispatcher::inspect_frontier_impl(
    const CudaExecutionFrontier& frontier, FrontierPurpose purpose) const noexcept {
    auto& state = *implementation_;
    if (!state.queue.owns_executor()) return {.status = CUDA_ERROR_INVALID_CONTEXT};
    if (state.is_callback_active) return {.status = CUDA_ERROR_NOT_READY};
    if (!owns_frontier(frontier)) return {.status = CUDA_ERROR_INVALID_HANDLE};
    if (state.queue.snapshot().is_quarantined) return {.status = CUDA_ERROR_UNKNOWN};
    if (state.active_enqueue_turn) return {.status = CUDA_ERROR_NOT_READY};
    const CallbackScope scope{state.is_callback_active};
    const bool retiring = purpose == FrontierPurpose::kRetirement;
    // Target-only retirement is independent of ancestors outside membership.
    // The frontier's public observation still retains its exact original error.
    if (retiring && std::ranges::all_of(frontier.state_->observations,
                                        [](const auto& item) { return item->is_settled; }))
        return {.status = CUDA_SUCCESS};
    if (frontier.state_->predecessors) {
        auto ancestor = Implementation::observed_failure(*frontier.state_->predecessors, purpose);
        if (ancestor.has_failure_observation) return ancestor;
    }
    auto failed = Implementation::observed_failure(frontier.state_->observations, purpose);
    if (failed.has_failure_observation) return failed;
    const auto status = retiring ? CUDA_ERROR_NOT_READY : frontier.observe();
    if (status != CUDA_ERROR_NOT_READY) return {.status = status};
    try {
        return state.inspect_closure(state.dependency_closure({}, &frontier, purpose), status,
                                     purpose);
    } catch (const std::bad_alloc&) {
        return {.status = CUDA_ERROR_OUT_OF_MEMORY};
    } catch (...) {
        return {.status = CUDA_ERROR_UNKNOWN};
    }
}
CudaExecutionIssueResult CudaExecutionDispatcher::progress_frontier_dependency(
    const CudaExecutionFrontier& frontier) noexcept {
    return progress_frontier_impl(frontier, FrontierPurpose::kExecution);
}
CudaExecutionIssueResult CudaExecutionDispatcher::progress_frontier_retirement(
    const CudaExecutionFrontier& frontier) noexcept {
    return progress_frontier_impl(frontier, FrontierPurpose::kRetirement);
}
CudaExecutionIssueResult CudaExecutionDispatcher::progress_frontier_impl(
    const CudaExecutionFrontier& frontier, FrontierPurpose purpose) noexcept {
    auto& state = *implementation_;
    const auto failure = [](CUresult status) {
        return CudaExecutionIssueResult{.state = CudaExecutionIssueState::kError,
                                        .native_status = status};
    };
    if (!state.queue.owns_executor()) return failure(CUDA_ERROR_INVALID_CONTEXT);
    if (state.is_callback_active) return failure(CUDA_ERROR_NOT_READY);
    if (!owns_frontier(frontier)) return failure(CUDA_ERROR_INVALID_HANDLE);
    if (state.queue.snapshot().is_quarantined) return failure(CUDA_ERROR_UNKNOWN);
    if (state.active_enqueue_turn) return failure(CUDA_ERROR_NOT_READY);
    const CallbackScope scope{state.is_callback_active};
    const bool retiring = purpose == FrontierPurpose::kRetirement;
    if (retiring && std::ranges::all_of(frontier.state_->observations,
                                        [](const auto& item) { return item->is_settled; }))
        return {};
    if (frontier.state_->predecessors) {
        const auto ancestor =
            Implementation::observed_failure(*frontier.state_->predecessors, purpose);
        if (ancestor.has_failure_observation) return failure(ancestor.status);
    }
    const auto failed = Implementation::observed_failure(frontier.state_->observations, purpose);
    if (failed.has_failure_observation) return failure(failed.status);
    const auto observed = retiring ? CUDA_ERROR_NOT_READY : frontier.observe();
    if (observed == CUDA_SUCCESS) return {};
    if (observed != CUDA_ERROR_NOT_READY) return failure(observed);
    if (const auto status = state.check_context(); status != CUDA_SUCCESS) return failure(status);
    try {
        return state.progress_closure(state.dependency_closure({}, &frontier, purpose), observed,
                                      purpose);
    } catch (const std::bad_alloc&) {
        return failure(CUDA_ERROR_OUT_OF_MEMORY);
    } catch (...) {
        return failure(CUDA_ERROR_UNKNOWN);
    }
}
CUresult CudaExecutionDispatcher::begin_enqueue_turn(
    const CudaExecutionEnqueueTurn& turn) noexcept {
    auto& state = *implementation_;
    if (!state.queue.owns_executor()) return CUDA_ERROR_INVALID_CONTEXT;
    if (const auto status = state.issue_health(); status != CUDA_SUCCESS) return status;
    const auto ready = enqueue_turn_ready(turn);
    if (ready != CUDA_SUCCESS) return ready;
    const auto record = state.enqueue_record(turn);
    const CallbackScope scope{state.is_callback_active};
    state.active_enqueue_turn = record->acceptance_order;
    record->has_external_started = true;
    auto status = state.check_context();
    if (status == CUDA_SUCCESS) status = state.create_enqueue_events(record);
    if (status == CUDA_SUCCESS) status = state.prepare_dependencies(*record);
    if (status != CUDA_SUCCESS) {
        record->enqueue->phase = EnqueuePhase::kFailed;
        return state.retain_uncertain(record, status);
    }
    record->enqueue->phase = EnqueuePhase::kReady;
    return CUDA_SUCCESS;
}
CUresult CudaExecutionDispatcher::finish_enqueue_turn(
    const CudaExecutionEnqueueTurn& turn) noexcept {
    auto& state = *implementation_;
    if (!state.queue.owns_executor()) return CUDA_ERROR_INVALID_CONTEXT;
    if (state.is_callback_active) return CUDA_ERROR_NOT_READY;
    const auto record = state.enqueue_record(turn);
    if (!record || state.active_enqueue_turn != turn.id_) return CUDA_ERROR_INVALID_HANDLE;
    if (record->is_cleanup_uncertain) return record->observation->failure;
    if (!record->enqueue->ticket) return CUDA_ERROR_NOT_READY;
    const auto observed = record->enqueue->ticket->observe();
    if (observed.phase == CudaNativeExecutionCallPhase::kPending ||
        observed.phase == CudaNativeExecutionCallPhase::kEntered)
        return CUDA_ERROR_NOT_READY;
    const CallbackScope scope{state.is_callback_active};
    if (observed.phase != CudaNativeExecutionCallPhase::kReturned || !observed.was_invoked ||
        observed.status != CUDA_SUCCESS || record->enqueue->phase != EnqueuePhase::kEntered) {
        record->enqueue->phase = EnqueuePhase::kFailed;
        return state.retain_uncertain(
            record, observed.status != CUDA_SUCCESS ? observed.status : CUDA_ERROR_UNKNOWN);
    }
    const auto status = state.record_enqueue_events(record);
    if (status != CUDA_SUCCESS) {
        record->enqueue->phase = EnqueuePhase::kFailed;
        return state.retain_uncertain(record, status);
    }
    record->is_native_issued = record->is_event_recorded = true;
    record->enqueue->phase = EnqueuePhase::kComplete;
    record->enqueue->ticket.reset();
    state.active_enqueue_turn = 0;
    return CUDA_SUCCESS;
}
CUresult CudaExecutionDispatcher::cancel_enqueue_turn(
    const CudaExecutionEnqueueTurn& turn) noexcept {
    auto& state = *implementation_;
    if (!state.queue.owns_executor()) return CUDA_ERROR_INVALID_CONTEXT;
    if (state.is_callback_active) return CUDA_ERROR_NOT_READY;
    const auto record = state.enqueue_record(turn);
    if (!record) return CUDA_ERROR_INVALID_HANDLE;
    if (record->enqueue->ticket) {
        const auto observed = record->enqueue->ticket->observe();
        if (observed.phase != CudaNativeExecutionCallPhase::kCancelled || observed.was_invoked)
            return CUDA_ERROR_NOT_READY;
    }
    auto phase = record->enqueue->phase.load();
    if ((phase != EnqueuePhase::kAccepted && phase != EnqueuePhase::kReady) ||
        !record->enqueue->phase.compare_exchange_strong(phase, EnqueuePhase::kCancelled))
        return CUDA_ERROR_NOT_READY;
    const CallbackScope scope{state.is_callback_active};
    if (record->event) {
        const auto proof = state.check_context();
        if (proof != CUDA_SUCCESS) return state.retain_uncertain(record, proof);
        const auto cleaned = state.destroy_event(record);
        if (cleaned != CUDA_SUCCESS) return cleaned;
    }
    record->enqueue->ticket.reset();
    record->observation->is_settled = true;
    if (state.active_enqueue_turn == turn.id_) state.active_enqueue_turn = 0;
    std::erase(state.records, record);
    return CUDA_SUCCESS;
}
CUresult CudaExecutionDispatcher::fail_enqueue_turn(const CudaExecutionEnqueueTurn& turn,
                                                    CUresult failure) noexcept {
    auto& state = *implementation_;
    if (!state.queue.owns_executor()) return CUDA_ERROR_INVALID_CONTEXT;
    if (state.is_callback_active) return CUDA_ERROR_NOT_READY;
    const auto record = state.enqueue_record(turn);
    if (!record || !record->has_external_started || failure == CUDA_SUCCESS)
        return CUDA_ERROR_INVALID_VALUE;
    record->enqueue->phase = EnqueuePhase::kFailed;
    return state.retain_uncertain(record, failure);
}
bool CudaExecutionDispatcher::has_active_enqueue_turn() const noexcept {
    const auto& state = *implementation_;
    return state.queue.owns_executor() && state.active_enqueue_turn != 0;
}
std::size_t CudaExecutionDispatcher::enqueue_turn_count() const noexcept {
    const auto& state = *implementation_;
    if (!state.queue.owns_executor()) return 0;
    return static_cast<std::size_t>(std::ranges::count_if(
        state.records, [](const auto& record) { return static_cast<bool>(record->enqueue); }));
}

CudaExecutionSynchronousResult CudaExecutionDispatcher::execute_synchronous_legacy(
    const std::function<CUresult()>& invoke, const std::shared_ptr<const void>& lifetime) noexcept {
    auto& state = *implementation_;
    if (!state.queue.owns_executor()) return {.status = CUDA_ERROR_INVALID_CONTEXT};
    if (const auto status = state.issue_health(); status != CUDA_SUCCESS) return {.status = status};
    if (state.is_callback_active || state.active_enqueue_turn)
        return {.status = CUDA_ERROR_NOT_READY};
    if (!invoke || !lifetime || !state.uses_default_proxy)
        return {.status = CUDA_ERROR_INVALID_VALUE};
    if (!resources_quiescent()) return {.status = CUDA_ERROR_NOT_READY};
    const auto accepted = accept_enqueue_turn(0, lifetime);
    if (accepted.status != CUDA_SUCCESS) return {.status = accepted.status};
    const auto record = state.enqueue_record(accepted.turn);
    const CallbackScope scope{state.is_callback_active};
    state.active_enqueue_turn = record->acceptance_order;
    record->has_external_started = true;
    CudaExecutionSynchronousResult result{.status = state.check_context(),
                                          .frontier = accepted.frontier};
    if (result.status == CUDA_SUCCESS) result.status = state.create_event(record);
    if (result.status == CUDA_SUCCESS) result.status = state.create_legacy_event(record);
    if (result.status == CUDA_SUCCESS) result.status = state.prepare_dependencies(*record, true);
    if (result.status == CUDA_SUCCESS) {
        try {
            result.original_status = invoke();
        } catch (...) {
            result.status = CUDA_ERROR_UNKNOWN;
        }
        const auto proof = state.check_context();
        if (proof != CUDA_SUCCESS) result.status = proof;
    }
    if (result.status == CUDA_SUCCESS) result.status = state.join_legacy(record);
    if (result.status == CUDA_SUCCESS) {
        result.status = state.dispatch.event_record(record->event, record->stream.native);
        const auto proof = state.check_context();
        if (proof != CUDA_SUCCESS) result.status = proof;
    }
    if (result.status != CUDA_SUCCESS) {
        record->enqueue->phase = EnqueuePhase::kFailed;
        (void)state.retain_uncertain(record, result.status);
        return result;
    }
    record->is_native_issued = record->is_event_recorded = true;
    record->enqueue->phase = EnqueuePhase::kComplete;
    state.active_enqueue_turn = 0;
    return result;
}
ExecutionAcceptance CudaExecutionDispatcher::accept_kernel_impl(
    std::uint64_t sequence, const CudaExecutionKernel& kernel,
    const CudaExecutionPersistentKernel* persistent, KernelAdmission admission,
    std::size_t parameter_capacity_bytes) noexcept {
    auto& state = *implementation_;
    auto invalid = ExecutionAcceptance{.status = ExecutionAcceptStatus::kInvalid, .receipt = {}};
    if (!state.queue.owns_executor() || state.is_callback_active) return invalid;
    if (std::ranges::any_of(kernel.grid, [](auto value) { return value == 0; }) ||
        std::ranges::any_of(kernel.block, [](auto value) { return value == 0; }) ||
        (!kernel.parameters.empty() && !kernel.parameters.data()))
        return invalid;
    const auto function = std::ranges::find_if(
        state.functions, [&](const auto& entry) { return entry.id == kernel.function; });
    const auto stream = std::ranges::find_if(
        state.streams, [&](const auto& entry) { return entry.id == kernel.stream; });
    if (function == state.functions.end() || stream == state.streams.end() ||
        kernel.parameters.size() != function->parameter_bytes)
        return invalid;
    if (persistent && (!persistent->function || !persistent->workers || !persistent->progress ||
                       (!persistent->parameters.empty() && !persistent->parameters.data())))
        return invalid;
    if (state.records.size() == state.limits.max_commands)
        return {.status = ExecutionAcceptStatus::kFull, .receipt = {}};
    if (state.last_acceptance_order == UINT64_MAX)
        return {.status = ExecutionAcceptStatus::kOutOfMemory, .receipt = {}};
    try {
        auto record = std::make_shared<Implementation::Record>();
        record->sequence = sequence;
        record->requires_preparation = admission == KernelAdmission::kDeferredAsync;
        record->reports_errors_asynchronously = admission != KernelAdmission::kHostResult;
        record->kernel = kernel;
        if (state.context.process_memory) {
            auto captured = state.context.process_memory->borrow_live();
            if (captured.status != CUDA_SUCCESS)
                return {.status = captured.status == CUDA_ERROR_OUT_OF_MEMORY
                                      ? ExecutionAcceptStatus::kOutOfMemory
                                      : ExecutionAcceptStatus::kInvalid,
                        .receipt = {}};
            record->process_memory_lifetime = std::move(captured.lifetime);
        }
        record->kernel.parameters = {};  // Never retain caller storage.
        record->function = *function;
        record->stream = *stream;
        record->context_lifetime = state.context.lifetime;
        record->physical_function = persistent ? persistent->function : function->native;
        record->physical_grid = persistent ? std::array{persistent->workers, 1U, 1U} : kernel.grid;
        record->has_original_parameter_prefix =
            !persistent || (kernel.parameters.size() <= persistent->parameters.size() &&
                            std::ranges::equal(kernel.parameters, persistent->parameters.first(
                                                                      kernel.parameters.size())));
        if (persistent) record->progress = persistent->progress;
        if (state.prepare_command_capture(record) != CUDA_SUCCESS)
            return {.status = ExecutionAcceptStatus::kOutOfMemory, .receipt = {}};
        auto accepted = state.queue.accept(
            sequence, {.operation = 1,
                       .context = state.context.id,
                       .stream = kernel.stream,
                       .parameters = persistent ? persistent->parameters : kernel.parameters,
                       .lifetime = record,
                       .parameter_capacity_bytes = parameter_capacity_bytes});
        if (accepted.accepted()) state.publish_command(record, accepted.receipt);
        return accepted;
    } catch (...) {
        return {.status = ExecutionAcceptStatus::kOutOfMemory, .receipt = {}};
    }
}

CudaExecutionIssueResult CudaExecutionDispatcher::issue_next() noexcept {
    auto& state = *implementation_;
    if (!state.queue.owns_executor())
        return {.state = CudaExecutionIssueState::kError,
                .native_status = CUDA_ERROR_INVALID_CONTEXT};
    if (state.is_callback_active || state.active_enqueue_turn)
        return {.state = CudaExecutionIssueState::kError, .native_status = CUDA_ERROR_NOT_READY};
    const CallbackScope callback_scope{state.is_callback_active};
    if (const auto result = state.check_context(); result != CUDA_SUCCESS)
        return {.state = CudaExecutionIssueState::kError, .native_status = result};
    const auto found = std::ranges::find_if(state.records, [&](const auto& record) {
        return !record->is_external && !record->requires_preparation && !record->issue &&
               state.is_ready(*record);
    });
    if (found == state.records.end()) {
        if (state.queue.snapshot().is_quarantined)
            return {.state = CudaExecutionIssueState::kError, .native_status = CUDA_ERROR_UNKNOWN};
        return {};
    }
    auto record = *found;
    return state.issue_record(record);
}

CudaExecutionPendingResult CudaExecutionDispatcher::pending_command(
    std::optional<std::uint64_t> sequence) const noexcept {
    auto& state = *implementation_;
    if (!state.queue.owns_executor()) return {.status = CUDA_ERROR_INVALID_CONTEXT};
    if (state.is_callback_active || state.active_enqueue_turn)
        return {.status = CUDA_ERROR_NOT_READY};
    if (state.queue.snapshot().is_quarantined) return {.status = CUDA_ERROR_UNKNOWN};
    if (sequence && !*sequence) return {.status = CUDA_ERROR_INVALID_VALUE};
    const CallbackScope scope{state.is_callback_active};
    const auto matches = [&](const auto& record) {
        return !record->is_external && (!sequence || record->sequence == *sequence);
    };
    auto found = std::ranges::find_if(state.records, [&](const auto& record) {
        return matches(record) && record->is_paused && state.can_issue_pending(*record);
    });
    if (found == state.records.end())
        found = std::ranges::find_if(state.records, [&](const auto& record) {
            return matches(record) && state.can_issue_pending(*record);
        });
    if (found == state.records.end()) {
        if (!sequence) return {};
        const bool exists = std::ranges::any_of(state.records, matches);
        return {.status = exists ? CUDA_ERROR_NOT_READY : CUDA_ERROR_INVALID_HANDLE};
    }
    CudaExecutionPendingResult result;
    result.command.description_ = Implementation::describe_pending(**found);
    result.command.owner_ = state.frontier_owner;
    result.command.record_ = *found;
    return result;
}

CudaExecutionPendingResult CudaExecutionDispatcher::pending_preparation(
    std::optional<std::uint64_t> sequence) const noexcept {
    auto& state = *implementation_;
    if (!state.queue.owns_executor()) return {.status = CUDA_ERROR_INVALID_CONTEXT};
    if (state.has_owner_callback() || state.active_enqueue_turn)
        return {.status = CUDA_ERROR_NOT_READY};
    if (state.queue.snapshot().is_quarantined) return {.status = CUDA_ERROR_UNKNOWN};
    if (sequence && !*sequence) return {.status = CUDA_ERROR_INVALID_VALUE};
    const auto found = std::ranges::find_if(state.records, [&](const auto& record) {
        return record->requires_preparation && !record->issue &&
               (!sequence || record->sequence == *sequence);
    });
    if (found == state.records.end())
        return {.status = sequence ? CUDA_ERROR_INVALID_HANDLE : CUDA_SUCCESS};
    CudaExecutionPendingResult result;
    result.command.description_ = Implementation::describe_pending(**found);
    result.command.owner_ = state.frontier_owner;
    result.command.record_ = *found;
    return result;
}

CUresult CudaExecutionDispatcher::copy_deferred_parameters(
    const CudaExecutionPendingCommand& command, std::span<std::byte> output) const noexcept {
    const auto& state = *implementation_;
    if (!state.queue.owns_executor()) return CUDA_ERROR_INVALID_CONTEXT;
    if (state.is_callback_active || state.active_enqueue_turn) return CUDA_ERROR_NOT_READY;
    if (state.queue.snapshot().is_quarantined) return CUDA_ERROR_UNKNOWN;
    if (command.owner_.lock() != state.frontier_owner) return CUDA_ERROR_INVALID_HANDLE;
    const auto retained = command.record_.lock();
    const auto found =
        std::ranges::find_if(state.records, [&](const auto& record) { return record == retained; });
    if (!retained || found == state.records.end() ||
        Implementation::describe_pending(**found) != command.description_ ||
        !(*found)->requires_preparation || (*found)->issue)
        return CUDA_ERROR_INVALID_HANDLE;
    return state.queue.copy_unissued_parameters((*found)->receipt, output)
               ? CUDA_SUCCESS
               : CUDA_ERROR_INVALID_VALUE;
}

CudaExecutionPendingResult CudaExecutionDispatcher::complete_kernel_preparation(
    const CudaExecutionPendingCommand& command, std::shared_ptr<CudaExecutionObserver> observation,
    const CudaExecutionPersistentKernel* persistent) noexcept {
    auto& state = *implementation_;
    if (!state.queue.owns_executor()) return {.status = CUDA_ERROR_INVALID_CONTEXT};
    if (state.is_callback_active || state.active_enqueue_turn)
        return {.status = CUDA_ERROR_NOT_READY};
    if (state.queue.snapshot().is_quarantined) return {.status = CUDA_ERROR_UNKNOWN};
    if (command.owner_.lock() != state.frontier_owner) return {.status = CUDA_ERROR_INVALID_HANDLE};
    const auto retained = command.record_.lock();
    const auto found =
        std::ranges::find_if(state.records, [&](const auto& record) { return record == retained; });
    if (!retained || found == state.records.end() ||
        Implementation::describe_pending(**found) != command.description_)
        return {.status = CUDA_ERROR_INVALID_HANDLE};
    auto& record = **found;
    if (!record.requires_preparation || record.issue || record.geometry_epoch == UINT64_MAX ||
        record.is_cleanup_uncertain || record.observation->failure != CUDA_SUCCESS)
        return {.status = CUDA_ERROR_INVALID_VALUE};
    if (persistent) {
        if (!persistent->function || !persistent->workers || !persistent->progress ||
            persistent->parameters.size() < record.function.parameter_bytes ||
            (!persistent->parameters.empty() && !persistent->parameters.data()))
            return {.status = CUDA_ERROR_INVALID_VALUE};
        try {
            std::vector<std::byte> original(record.function.parameter_bytes);
            if (!state.queue.copy_unissued_parameters(record.receipt, original) ||
                !std::ranges::equal(original, persistent->parameters.first(original.size())))
                return {.status = CUDA_ERROR_INVALID_VALUE};
        } catch (...) {
            return {.status = CUDA_ERROR_OUT_OF_MEMORY};
        }
        if (!state.queue.replace_unissued_parameters(record.receipt, persistent->parameters))
            return {.status = CUDA_ERROR_INVALID_VALUE};
        record.physical_function = persistent->function;
        record.physical_grid = {persistent->workers, 1, 1};
        record.progress = persistent->progress;
    }
    record.kernel.observation = std::move(observation);
    record.requires_preparation = false;
    ++record.geometry_epoch;
    CudaExecutionPendingResult result;
    result.command.description_ = Implementation::describe_pending(record);
    result.command.owner_ = state.frontier_owner;
    result.command.record_ = *found;
    return result;
}

CUresult CudaExecutionDispatcher::quarantine_deferred_kernel(
    const CudaExecutionPendingCommand& command, CUresult failure) noexcept {
    auto& state = *implementation_;
    if (!state.queue.owns_executor()) return CUDA_ERROR_INVALID_CONTEXT;
    if (state.is_callback_active || state.active_enqueue_turn) return CUDA_ERROR_NOT_READY;
    if (failure == CUDA_SUCCESS || command.owner_.lock() != state.frontier_owner)
        return CUDA_ERROR_INVALID_VALUE;
    const auto retained = command.record_.lock();
    const auto found =
        std::ranges::find_if(state.records, [&](const auto& record) { return record == retained; });
    if (!retained || found == state.records.end() ||
        Implementation::describe_pending(**found) != command.description_ ||
        !(*found)->reports_errors_asynchronously || (*found)->issue)
        return CUDA_ERROR_INVALID_HANDLE;
    auto& record = **found;
    if (record.observation->failure == CUDA_SUCCESS) record.observation->failure = failure;
    record.is_cleanup_uncertain = true;
    state.queue.quarantine();
    return CUDA_SUCCESS;
}

CudaExecutionIssueResult CudaExecutionDispatcher::issue(
    const CudaExecutionPendingCommand& command) noexcept {
    auto& state = *implementation_;
    const auto failure = [&](CUresult status) {
        return CudaExecutionIssueResult{.state = CudaExecutionIssueState::kError,
                                        .sequence = command.description_.sequence,
                                        .native_status = status};
    };
    if (!state.queue.owns_executor()) return failure(CUDA_ERROR_INVALID_CONTEXT);
    if (state.is_callback_active || state.active_enqueue_turn) return failure(CUDA_ERROR_NOT_READY);
    if (state.queue.snapshot().is_quarantined) return failure(CUDA_ERROR_UNKNOWN);
    if (command.owner_.lock() != state.frontier_owner) return failure(CUDA_ERROR_INVALID_HANDLE);
    const auto retained = command.record_.lock();
    const auto found =
        std::ranges::find_if(state.records, [&](const auto& record) { return record == retained; });
    if (!retained || found == state.records.end() ||
        Implementation::describe_pending(**found) != command.description_)
        return failure(CUDA_ERROR_INVALID_HANDLE);
    const CallbackScope scope{state.is_callback_active};
    if (!state.can_issue_pending(**found)) return failure(CUDA_ERROR_NOT_READY);
    if (const auto proof = state.check_context(); proof != CUDA_SUCCESS) return failure(proof);
    // Reentrant callbacks are excluded through the original proof and issue.
    // No fallback to issue_next or a later command is permitted.
    auto record = *found;
    return record->is_paused ? state.resume_record(record) : state.issue_record(record);
}

CudaExecutionPendingResult CudaExecutionDispatcher::select_pending_kernel(
    const CudaExecutionPendingCommand& command,
    const CudaExecutionPendingKernelSelection& selection) noexcept {
    auto& state = *implementation_;
    if (!state.queue.owns_executor()) return {.status = CUDA_ERROR_INVALID_CONTEXT};
    if (state.is_callback_active || state.active_enqueue_turn)
        return {.status = CUDA_ERROR_NOT_READY};
    if (state.queue.snapshot().is_quarantined) return {.status = CUDA_ERROR_UNKNOWN};
    if ((selection.uses_original && selection.workers) ||
        (!selection.uses_original &&
         (!selection.workers || selection.workers > static_cast<unsigned int>(INT32_MAX))))
        return {.status = CUDA_ERROR_INVALID_VALUE};
    if (command.owner_.lock() != state.frontier_owner) return {.status = CUDA_ERROR_INVALID_HANDLE};
    const auto retained = command.record_.lock();
    const auto found =
        std::ranges::find_if(state.records, [&](const auto& record) { return record == retained; });
    if (!retained || found == state.records.end() ||
        Implementation::describe_pending(**found) != command.description_)
        return {.status = CUDA_ERROR_INVALID_HANDLE};
    auto& record = **found;
    if (record.is_external || record.memory || record.library || record.event_call ||
        !record.function.native || !record.kernel.function)
        return {.status = CUDA_ERROR_NOT_SUPPORTED};
    if (selection.uses_original && !record.has_original_parameter_prefix)
        return {.status = CUDA_ERROR_NOT_SUPPORTED};
    if (record.geometry_epoch == UINT64_MAX) return {.status = CUDA_ERROR_OUT_OF_MEMORY};
    const bool is_initial = !record.issue && record.command_epoch == 1;
    if ((selection.uses_original || selection.observation) && !is_initial)
        return {.status = CUDA_ERROR_NOT_SUPPORTED};
    if (!selection.uses_original && !record.progress) return {.status = CUDA_ERROR_NOT_SUPPORTED};
    const CallbackScope scope{state.is_callback_active};
    if (!state.can_issue_pending(record)) return {.status = CUDA_ERROR_NOT_READY};
    if (selection.uses_original) {
        if (!state.queue.limit_unissued_parameters(record.receipt,
                                                   record.function.parameter_bytes)) {
            state.queue.quarantine();
            return {.status = CUDA_ERROR_UNKNOWN};
        }
        record.physical_function = record.function.native;
        record.physical_grid = record.kernel.grid;
        record.progress.reset();
    } else {
        const auto status = record.progress->retarget_workers(selection.workers);
        if (status != CUDA_SUCCESS) return {.status = status};
        record.physical_grid = {selection.workers, 1, 1};
    }
    if (is_initial) record.kernel.observation = selection.observation;
    ++record.geometry_epoch;
    CudaExecutionPendingResult result;
    result.command.description_ = Implementation::describe_pending(record);
    result.command.owner_ = state.frontier_owner;
    result.command.record_ = *found;
    return result;
}

CUresult CudaExecutionDispatcher::request_retreat(std::uint64_t sequence) noexcept {
    auto& state = *implementation_;
    if (!state.queue.owns_executor()) return CUDA_ERROR_INVALID_CONTEXT;
    if (state.is_callback_active || state.active_enqueue_turn) return CUDA_ERROR_NOT_READY;
    const CallbackScope callback_scope{state.is_callback_active};
    if (const auto result = state.check_context(); result != CUDA_SUCCESS) return result;
    const auto found = std::ranges::find_if(state.records, [sequence](const auto& record) {
        return !record->is_external && record->sequence == sequence;
    });
    if (found == state.records.end()) return CUDA_ERROR_INVALID_HANDLE;
    const auto& record = **found;
    if (!record.progress || !record.is_native_issued) return CUDA_ERROR_NOT_SUPPORTED;
    if (record.is_paused || record.is_physically_stopped) return CUDA_SUCCESS;
    if (record.is_cleanup_uncertain || state.queue.snapshot().is_quarantined)
        return CUDA_ERROR_UNKNOWN;
    return record.progress->request_retreat();
}

CUresult CudaExecutionDispatcher::retarget_workers(std::uint64_t sequence,
                                                   CudaExecutionWorkerGeometry geometry) noexcept {
    auto& state = *implementation_;
    const auto workers = geometry.workers;
    if (!state.queue.owns_executor()) return CUDA_ERROR_INVALID_CONTEXT;
    if (state.is_callback_active || state.active_enqueue_turn) return CUDA_ERROR_NOT_READY;
    if (!workers || workers > static_cast<unsigned int>(std::numeric_limits<int>::max()))
        return CUDA_ERROR_INVALID_VALUE;
    const CallbackScope callback_scope{state.is_callback_active};
    if (const auto result = state.check_context(); result != CUDA_SUCCESS) return result;
    if (state.queue.snapshot().is_quarantined) return CUDA_ERROR_UNKNOWN;
    const auto found = std::ranges::find_if(state.records, [sequence](const auto& record) {
        return !record->is_external && record->sequence == sequence;
    });
    if (found == state.records.end()) return CUDA_ERROR_INVALID_HANDLE;
    auto& record = **found;
    if (!record.progress || record.memory || record.library || record.event_call)
        return CUDA_ERROR_NOT_SUPPORTED;
    if (record.is_cleanup_uncertain || record.observation->failure != CUDA_SUCCESS)
        return CUDA_ERROR_UNKNOWN;
    if (!record.is_paused || !record.is_physically_stopped || record.event || record.legacy_event)
        return CUDA_ERROR_NOT_READY;
    if (record.launch_status != CUDA_SUCCESS || record.completion_status != CUDA_SUCCESS)
        return CUDA_ERROR_INVALID_VALUE;
    if (record.geometry_epoch == UINT64_MAX) return CUDA_ERROR_OUT_OF_MEMORY;
    const auto result = record.progress->retarget_workers(workers);
    if (result == CUDA_SUCCESS) {
        record.physical_grid = {workers, 1, 1};
        ++record.geometry_epoch;
    }
    return result;
}

CudaExecutionIssueResult CudaExecutionDispatcher::resume(std::uint64_t sequence) noexcept {
    auto& state = *implementation_;
    const auto failure = [sequence](CUresult status) {
        return CudaExecutionIssueResult{.state = CudaExecutionIssueState::kError,
                                        .sequence = sequence,
                                        .native_status = status};
    };
    if (!state.queue.owns_executor()) return failure(CUDA_ERROR_INVALID_CONTEXT);
    if (state.is_callback_active || state.active_enqueue_turn) return failure(CUDA_ERROR_NOT_READY);
    const CallbackScope callback_scope{state.is_callback_active};
    if (const auto status = state.check_context(); status != CUDA_SUCCESS) return failure(status);
    if (state.queue.snapshot().is_quarantined) return failure(CUDA_ERROR_UNKNOWN);
    const auto found = std::ranges::find_if(state.records, [sequence](const auto& record) {
        return !record->is_external && record->sequence == sequence;
    });
    if (found == state.records.end()) return failure(CUDA_ERROR_INVALID_HANDLE);
    auto record = *found;
    if (!record->progress || !record->is_paused || record->is_cleanup_uncertain)
        return failure(CUDA_ERROR_INVALID_VALUE);
    return state.resume_record(record);
}

CudaExecutionPollResult CudaExecutionDispatcher::poll() noexcept {
    return implementation_->collect(false);
}
CudaExecutionPollResult CudaExecutionDispatcher::drain_issued() noexcept {
    return implementation_->collect(true);
}
CudaExecutionFrontierResult CudaExecutionDispatcher::capture_frontier(
    std::optional<std::uint64_t> stream, bool includes_submission_dependencies) noexcept {
    return capture_frontier_impl(stream, std::nullopt, includes_submission_dependencies);
}
CudaExecutionFrontierResult CudaExecutionDispatcher::capture_command_frontier(
    std::uint64_t sequence) noexcept {
    return capture_frontier_impl(std::nullopt, sequence);
}
CudaExecutionFrontierReservationResult CudaExecutionDispatcher::reserve_command_frontier(
    std::uint64_t sequence) noexcept {
    auto& state = *implementation_;
    if (!state.queue.owns_executor()) return {.status = CUDA_ERROR_INVALID_CONTEXT};
    if (state.has_owner_callback()) return {.status = CUDA_ERROR_NOT_READY};
    const auto snapshot = state.queue.snapshot();
    if (snapshot.is_quarantined) return {.status = CUDA_ERROR_UNKNOWN};
    if (snapshot.is_closing) return {.status = CUDA_ERROR_DEINITIALIZED};
    if (!sequence || sequence <= state.last_sequence) return {.status = CUDA_ERROR_INVALID_VALUE};
    std::erase_if(state.frontiers, [](const auto& item) { return item.expired(); });
    std::erase_if(state.reservations, [](const auto& item) {
        const auto token = item.lock();
        return !token || !token->capture;
    });
    if (state.frontiers.size() >= state.max_bindings) return {.status = CUDA_ERROR_OUT_OF_MEMORY};
    try {
        if (!frontier_thread_owner) frontier_thread_owner = std::make_shared<int>(0);
        auto token = std::make_shared<CudaExecutionFrontierReservation::State>();
        token->sequence = sequence;
        token->capture = std::make_shared<CudaExecutionFrontier::State>();
        token->capture->diagnostic_id = detail::take_diagnostic_id(state.next_diagnostic_id);
        token->capture->owner = state.frontier_owner;
        token->capture->thread = frontier_thread_owner;
        token->capture->process = ::getpid();
        token->capture->observations.resize(1);
        CudaExecutionFrontierReservationResult result{.status = CUDA_SUCCESS};
        result.reservation.state_ = token;
        state.frontiers.push_back(token->capture);
        state.reservations.push_back(token);
        return result;
    } catch (...) {
        return {.status = CUDA_ERROR_OUT_OF_MEMORY};
    }
}
CudaExecutionFrontierResult CudaExecutionDispatcher::reserved_command_frontier(
    const CudaExecutionFrontierReservation& reservation) const noexcept {
    const auto& state = *implementation_;
    if (const auto status = state.reservation_gate(reservation); status != CUDA_SUCCESS)
        return {.status = status, .frontier = {}};
    if (!reservation.state_->is_bound) return {.status = CUDA_ERROR_NOT_READY, .frontier = {}};
    CudaExecutionFrontierResult result{.status = CUDA_SUCCESS, .frontier = {}};
    result.frontier.state_ = reservation.state_->capture;
    return result;
}
CUresult CudaExecutionDispatcher::cancel_frontier_reservation(
    const CudaExecutionFrontierReservation& reservation) noexcept {
    const auto& state = *implementation_;
    if (const auto status = state.reservation_gate(reservation); status != CUDA_SUCCESS)
        return status;
    if (reservation.state_->is_bound) return CUDA_ERROR_INVALID_VALUE;
    reservation.state_->capture.reset();
    return CUDA_SUCCESS;
}
CudaExecutionFrontierResult CudaExecutionDispatcher::capture_frontier_impl(
    std::optional<std::uint64_t> stream, std::optional<std::uint64_t> sequence,
    bool includes_submission_dependencies) noexcept {
    auto& state = *implementation_;
    if (!state.queue.owns_executor()) return {.status = CUDA_ERROR_INVALID_CONTEXT, .frontier = {}};
    if (state.is_callback_active) return {.status = CUDA_ERROR_NOT_READY, .frontier = {}};
    if (stream &&
        !std::ranges::any_of(state.streams, [&](const auto& entry) { return entry.id == *stream; }))
        return {.status = CUDA_ERROR_INVALID_HANDLE, .frontier = {}};
    if (sequence && !std::ranges::any_of(state.records, [&](const auto& record) {
            return !record->is_external && record->sequence == *sequence;
        }))
        return {.status = CUDA_ERROR_INVALID_HANDLE, .frontier = {}};
    if (state.queue.snapshot().is_quarantined)
        return {.status = CUDA_ERROR_UNKNOWN, .frontier = {}};
    std::erase_if(state.frontiers, [](const auto& item) { return item.expired(); });
    if (state.frontiers.size() == state.max_bindings)
        return {.status = CUDA_ERROR_OUT_OF_MEMORY, .frontier = {}};
    try {
        const bool is_nonblocking =
            stream && std::ranges::any_of(state.streams, [&](const auto& selected) {
                return selected.id == *stream && selected.is_nonblocking;
            });
        auto captured = std::make_shared<CudaExecutionFrontier::State>();
        captured->diagnostic_id = detail::take_diagnostic_id(state.next_diagnostic_id);
        if (!frontier_thread_owner) frontier_thread_owner = std::make_shared<int>(0);
        captured->owner = state.frontier_owner;
        captured->thread = frontier_thread_owner;
        captured->process = ::getpid();
        captured->observations.reserve(sequence ? 1 : state.records.size());
        for (const auto& record : state.records) {
            bool includes = !stream;
            Implementation::each_stream(*record, [&](const Implementation::Stream& selected,
                                                     CUevent) {
                includes = includes || (stream && (selected.id == *stream ||
                                                   (includes_submission_dependencies &&
                                                    ((*stream == 0 && !selected.is_nonblocking) ||
                                                     (selected.id == 0 && !is_nonblocking)))));
            });
            if (includes &&
                (!sequence || (!record->is_external && record->sequence == *sequence))) {
                captured->observations.push_back(record->observation);
                if (sequence) captured->predecessors = record->captured_predecessors;
            }
        }
        CudaExecutionFrontier frontier;
        frontier.state_ = captured;
        state.frontiers.push_back(captured);
        return {.status = CUDA_SUCCESS, .frontier = std::move(frontier)};
    } catch (...) {
        return {.status = CUDA_ERROR_OUT_OF_MEMORY, .frontier = {}};
    }
}
CUresult CudaExecutionDispatcher::query_frontier(
    const CudaExecutionFrontier& frontier) const noexcept {
    const auto& state = *implementation_;
    if (!state.queue.owns_executor()) return CUDA_ERROR_INVALID_CONTEXT;
    if (!frontier.state_ || frontier.state_->owner.lock() != state.frontier_owner)
        return CUDA_ERROR_INVALID_HANDLE;
    return frontier.observe();
}
bool CudaExecutionDispatcher::has_ready_work() const noexcept {
    auto& state = *implementation_;
    return state.queue.owns_executor() && state.has_ready_work();
}
bool CudaExecutionDispatcher::reject_command(const ExecutionReceipt& receipt,
                                             CUresult failure) noexcept {
    auto& state = *implementation_;
    if (!state.queue.owns_executor() || state.has_owner_callback() || failure == CUDA_SUCCESS ||
        state.queue.snapshot().is_quarantined)
        return false;
    const auto found = std::ranges::find_if(state.records, [&](const auto& record) {
        return !record->is_external && record->sequence == receipt.sequence();
    });
    if (found == state.records.end()) return false;
    const auto& record = *found;
    // The issue token is assigned BEFORE dependency, preparation or private
    // event callbacks. Never infer safe cancellation from native-issued alone.
    if (record->issue || record->is_native_issued || record->is_paused ||
        record->is_cleanup_uncertain || record->was_library_entered ||
        record->has_library_cleanup_attempt || record->event || record->legacy_event)
        return false;
    const auto exact_failure =
        record->observation->failure == CUDA_SUCCESS ? failure : record->observation->failure;
    const CallbackScope scope{state.is_callback_active};
    // The supplied opaque receipt, not its numeric sequence, authenticates the
    // queue transition. A foreign same-sequence receipt cannot cancel anything.
    if (!state.queue.reject_unissued(receipt, exact_failure)) return false;
    record->observation->failure = exact_failure;
    record->observation->is_settled = true;
    state.records.erase(found);
    return true;
}
std::size_t CudaExecutionDispatcher::reject_queued(CUresult failure) noexcept {
    auto& state = *implementation_;
    if (!state.queue.owns_executor() || state.is_callback_active || failure == CUDA_SUCCESS)
        return 0;
    const CallbackScope callback_scope{state.is_callback_active};
    std::size_t rejected{};
    for (std::size_t index = 0; index < state.records.size();) {
        const auto& record = state.records[index];
        if (record->is_external || record->issue || record->is_cleanup_uncertain ||
            !state.queue.reject_unissued(record->receipt, failure)) {
            ++index;
            continue;
        }
        if (record->observation->failure == CUDA_SUCCESS) record->observation->failure = failure;
        record->observation->is_settled = true;
        state.records.erase(state.records.begin() + static_cast<std::ptrdiff_t>(index));
        ++rejected;
    }
    return rejected;
}
std::optional<ExecutionHostReply> CudaExecutionDispatcher::host_result(
    const ExecutionReceipt& receipt) const noexcept {
    return implementation_->queue.host_result(receipt);
}
bool CudaExecutionDispatcher::acknowledge_result(const ExecutionReceipt& receipt) noexcept {
    return implementation_->queue.acknowledge_result(receipt);
}
ExecutionSubmissionSnapshot CudaExecutionDispatcher::snapshot() const noexcept {
    return implementation_->queue.snapshot();
}
bool CudaExecutionDispatcher::is_quarantined() const noexcept {
    return implementation_->queue.executor_is_quarantined();
}
bool CudaExecutionDispatcher::set_retirement_observer(
    std::shared_ptr<CudaRetirementObserver> observer) noexcept {
    auto& state = *implementation_;
    if (!state.queue.owns_executor() || state.has_owner_callback() || state.active_enqueue_turn)
        return false;
    state.retirement_observer = std::move(observer);
    return true;
}
CudaExecutionSchedulingSnapshot CudaExecutionDispatcher::scheduling_snapshot() const noexcept {
    auto& state = *implementation_;
    if (!state.queue.owns_executor())
        return {.submissions = {.is_closing = true, .is_quarantined = true}};
    CudaExecutionSchedulingSnapshot result{.submissions = state.queue.snapshot()};
    for (const auto& record : state.records) {
        if (!state.is_callback_active && record->is_external && !record->enqueue)
            ++result.external_barriers;
        if (record->enqueue) ++result.enqueue_turns;
        if (record->is_external && record->has_external_started) ++result.external_work_in_flight;
        result.has_pending_preparation =
            result.has_pending_preparation || (record->requires_preparation && !record->issue);
    }
    result.has_pending_preparation = result.has_pending_preparation &&
                                     !state.has_owner_callback() && !state.active_enqueue_turn &&
                                     !result.submissions.is_quarantined;
    // Keep counts captured before dependency callbacks, as in the legacy
    // getters. Callback exclusion prevents mutation during readiness inspection.
    result.has_ready_work = state.has_ready_work();
    return result;
}
std::optional<std::array<std::uint64_t, 2>> CudaExecutionDispatcher::native_kernel_pair()
    const noexcept {
    const auto& state = *implementation_;
    if (!state.queue.owns_executor() || state.queue.snapshot().is_quarantined) return std::nullopt;
    std::array<std::uint64_t, 2> sequences{};
    std::size_t count{};
    for (const auto& record : state.records) {
        if (record->is_external || record->memory || record->library || record->event_call ||
            record->progress || !record->is_native_issued || !record->is_event_recorded ||
            record->is_physically_stopped || record->is_paused || record->is_cleanup_uncertain ||
            record->launch_status != CUDA_SUCCESS)
            continue;
        if (count == sequences.size()) return std::nullopt;
        sequences[count++] = record->sequence;
    }
    return count == sequences.size() ? std::optional{sequences} : std::nullopt;
}
void CudaExecutionDispatcher::close() noexcept {
    if (!implementation_->is_callback_active) implementation_->queue.close();
}
bool CudaExecutionDispatcher::shutdown() noexcept {
    auto& state = *implementation_;
    if (!state.queue.owns_executor() || state.is_callback_active) return false;
    return state.queue.shutdown() && state.records.empty();
}

}  // namespace glimmer::backend::cuda
