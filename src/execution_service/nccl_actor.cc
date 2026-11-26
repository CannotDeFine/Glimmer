#include "nccl_actor.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <climits>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <unistd.h>

namespace glimmer::app {
namespace {
using namespace backend::cuda;
namespace wire = control;
using Kind = CudaNcclNativeOperation;
using Stage = NcclActorStage;
using ContinuationStage = NcclActorContinuationStage;
using OutcomeKind = NcclActorContinuationOutcomeKind;
using AuditEvent = NcclActorAuditEvent;
using AuditSubject = NcclActorAuditSubject;
using AuditAction = NcclActorAuditAction;
using AuditBinding = NcclActorAuditBinding;
template <typename Enum>
std::uint32_t audit_enum(Enum value) noexcept {
    return static_cast<std::uint32_t>(value) + 1;
}
thread_local std::shared_ptr<const void> actor_thread;
// Tickets must not collide across actors on the same process owner thread.
std::atomic<std::uint64_t> next_cleanup_retirement{1};
std::uint64_t cleanup_retirement_identity() noexcept {
    auto value = next_cleanup_retirement.load(std::memory_order_relaxed);
    while (value) {
        const auto next = value == UINT64_MAX ? 0 : value + 1;
        if (next_cleanup_retirement.compare_exchange_weak(value, next, std::memory_order_relaxed))
            return value;
    }
    return 0;
}
std::weak_ptr<const void> thread_identity() {
    if (!actor_thread) actor_thread = std::make_shared<int>(0);
    return actor_thread;
}
Kind kind(const wire::ExecutionNcclRequest& request) {
    return static_cast<Kind>(request.call.index() + 1);
}
bool collective(Kind operation) {
    return operation == Kind::kAllGather || operation == Kind::kAllReduce ||
           operation == Kind::kBcast;
}
bool initialization(Kind operation) {
    return operation == Kind::kInitRank || operation == Kind::kInitRankConfig;
}
bool member(Kind operation) {
    return collective(operation) || initialization(operation) || operation == Kind::kDestroy ||
           operation == Kind::kAbort;
}
bool has_output(const wire::ExecutionNcclRequest& request) {
    if (const auto* value = std::get_if<wire::ExecutionNcclGetUniqueId>(&request.call))
        return value->has_output;
    if (const auto* value = std::get_if<wire::ExecutionNcclInitRank>(&request.call))
        return value->has_output;
    if (const auto* value = std::get_if<wire::ExecutionNcclInitRankConfig>(&request.call))
        return value->initialization.has_output;
    if (const auto* value = std::get_if<wire::ExecutionNcclGetAsyncError>(&request.call))
        return value->has_output;
    return false;
}
wire::ExecutionNcclCommunicator communicator(const wire::ExecutionNcclRequest& request) {
    if (const auto* value = std::get_if<wire::ExecutionNcclGetAsyncError>(&request.call))
        return value->communicator;
    if (const auto* value = std::get_if<wire::ExecutionNcclDestroy>(&request.call))
        return value->communicator;
    if (const auto* value = std::get_if<wire::ExecutionNcclAbort>(&request.call))
        return value->communicator;
    if (const auto* value = std::get_if<wire::ExecutionNcclAllGather>(&request.call))
        return value->communicator;
    if (const auto* value = std::get_if<wire::ExecutionNcclAllReduce>(&request.call))
        return value->collective.communicator;
    if (const auto* value = std::get_if<wire::ExecutionNcclBcast>(&request.call))
        return value->communicator;
    return {};
}
struct Bundle {
    std::vector<CudaSessionExternalCommand> commands;
    std::vector<CudaSessionExternalContext> contexts;
};
struct Source {
    std::uint64_t identity{};
    std::shared_ptr<CudaExecutionSession> owner;
    std::vector<std::uint64_t> streams;
    CudaExecutionEnqueueAcceptance turn;
    CudaExecutionExternalCohortAcceptance final_tail;
    bool accepted{}, begun{}, finished{};
    bool tail_begun{}, tail_recorded{};
    std::uint64_t audit_capture{};
};
struct Buffer {
    wire::ExecutionNcclBuffer input;
    std::shared_ptr<CudaExecutionSession> owner;
    std::size_t bytes{};
    bool receive{};
};
struct Batch {
    std::uint64_t identity{}, generation{}, peer{};
    std::uint32_t priority{};
    bool open_group{}, has_native{}, uncertain{};
    // Immutable reservation, distinct from the native TLS group's open state.
    // End7 closes TLS but cannot refund this credit while async/tails survive.
    const bool is_group{};
    std::optional<NcclActorAuditRecord> audit_previous{};
};
struct Record {
    std::uint64_t id{}, cutoff{}, operation{}, caller_index{};
    NcclActorPeer peer;
    wire::ExecutionNcclRequest request;
    Stage stage{Stage::kPrepared};
    CUresult carrier{CUDA_SUCCESS};
    std::shared_ptr<Batch> batch;
    std::shared_ptr<Bundle> bundle;
    std::shared_ptr<CudaExecutionSession> entry_owner;
    std::vector<Source> sources;
    std::vector<Buffer> buffers;
    std::vector<std::shared_ptr<Record>> members;
    CudaNcclCoordinatorRequest native_request;
    std::optional<CudaNcclCoordinatorObservation> observed;
    std::optional<wire::ExecutionNcclReply> envelope;
    bool outer_end{}, grouped{}, released{}, backend_released{}, transition_applied{};
    bool projected{}, captured{}, tails_bound{};
    bool needs_late_tails{};
    bool end_credit{}, progress_credit{};
    bool continuation_pending{}, ready_continuation{};
    std::uint64_t caller_primary_domain{};
    bool audit_active_call{};
    std::uint64_t audit_native_sequence{};
    std::uint64_t audit_communicator_domain{}, audit_communicator_context{},
        audit_communicator_object{};
    std::optional<NcclActorAuditRecord> audit_previous, audit_observation;
};
struct Caller {
    NcclActorPeer peer;
    std::array<std::uint64_t, 2> thread{};
    std::uint64_t public_sequence{}, native_sequence{}, group{}, last_group{};
    std::uint64_t ordinary_sequence{};
    std::size_t depth{};
    bool disconnected{};
    bool observed_primary{};
    std::shared_ptr<Batch> batch;
    std::vector<std::shared_ptr<Record>> members;
};
struct Comm {
    std::uint64_t object{}, peer{};
    std::int32_t ranks{};
    bool terminal{};
    std::weak_ptr<Batch> pending{};
};
struct CleanupParent {
    NcclActorPeer peer;
    wire::ExecutionSourceContinuation scope;
    std::uint64_t connection{}, allocation{}, sequence{}, active_continuation{};
    std::shared_ptr<const void> helper_owner;
    std::shared_ptr<CudaExecutionSession> origin;
    std::shared_ptr<Batch> batch;
    bool bound{}, consumed{}, uncertain{};
    std::optional<NcclActorAuditRecord> audit_previous;
};
struct Continuation {
    std::uint64_t id{}, cutoff{}, attempt{}, caller_index{};
    NcclActorPeer peer;
    wire::ExecutionSourceContinuation scope;
    NcclActorContinuationTarget target;
    std::shared_ptr<Batch> batch;
    std::shared_ptr<CudaExecutionSession> origin;
    std::shared_ptr<CleanupParent> parent;
    std::vector<NcclActorContinuationBinding> dependencies;
    std::optional<NcclActorContinuationBinding> binding;
    std::shared_ptr<const void> pending_lifetime;
    std::uint64_t pending_sequence{};
    std::optional<wire::ExecutionWireReply> reply;
    std::optional<backend::ExecutionHostReply> host_reply;
    ContinuationStage stage{ContinuationStage::kPrepared};
    CUresult status{CUDA_SUCCESS};
    bool released{}, retired{};
    std::optional<NcclActorAuditRecord> audit_previous;
    NcclActorGrant audit_input_grant;
};
struct CleanupRetirement {
    std::uint64_t identity{}, connection{};
    NcclActorPeer peer;
    std::shared_ptr<const void> helper;
    std::vector<std::shared_ptr<CleanupParent>> parents;
    std::shared_ptr<Continuation> closing;
    CUresult failure{CUDA_SUCCESS};
};
}  // namespace

struct NcclActor::Implementation {
    const pid_t process{::getpid()};
    const std::weak_ptr<const void> owner{thread_identity()};
    NcclActorConfig config;
    CudaNativeExecutionLaneDispatch lane;
    NcclActorResolver resolver;
    NcclActorEffectProducer effects;
    CudaNcclCoordinator coordinator;
    CudaNcclPrimaryDomain primary;
    CudaNcclCoordinatorDomain domain;
    std::vector<Caller> callers;
    std::vector<std::shared_ptr<Record>> records;
    std::vector<std::shared_ptr<Batch>> batches;
    std::vector<Comm> communicators;
    std::vector<std::uint64_t> disconnected_peers;
    std::vector<std::shared_ptr<Continuation>> continuations;
    std::vector<std::shared_ptr<CleanupParent>> cleanup_parents;
    std::vector<CleanupRetirement> cleanup_retirements;
    std::uint64_t next_request{1}, next_generation{1}, next_cutoff{1};
    std::uint64_t next_continuation{1}, next_attempt{1}, active_attempt{};
    bool active{}, closed{}, initialized{}, coordinator_stopped{}, stopped{}, uncertain{};
    NcclActorObserver observer;
    std::uint64_t next_audit_serial{1};
    bool observer_installed{};

    Implementation(NcclActorConfig settings, CudaNcclNativeDispatch native,
                   CudaNativeExecutionLaneDispatch driver, CudaNcclPrimaryDomainDispatch resources,
                   NcclActorResolver lookup, NcclActorEffectProducer producer)
        : config(settings),
          lane(driver),
          resolver(std::move(lookup)),
          effects(std::move(producer)),
          coordinator(native, driver, settings.coordinator),
          primary(resources, {settings.primary_domain, settings.device_ordinal, 1}) {
        if (!config.primary_domain || config.device_ordinal != 0 || !config.maximum_requests ||
            !config.maximum_callers || !config.maximum_sources || !config.maximum_groups ||
            !config.maximum_communicators || !config.maximum_reservations || !resolver.session ||
            !config.maximum_continuations || !config.maximum_cleanup_parents || !resolver.stream ||
            (effects.produce && !effects.lifetime) ||
            config.coordinator.maximum_group_members > wire::kExecutionNcclPublicationLimit ||
            config.maximum_requests > SIZE_MAX - config.maximum_groups ||
            config.maximum_requests + config.maximum_groups >
                SIZE_MAX - config.coordinator.maximum_progress_operations)
            throw std::invalid_argument("invalid NCCL actor configuration");
        callers.reserve(config.maximum_callers);
        records.reserve(config.maximum_requests + config.maximum_groups +
                        config.coordinator.maximum_progress_operations);
        batches.reserve(config.maximum_reservations);
        communicators.reserve(config.maximum_communicators);
        disconnected_peers.reserve(config.maximum_callers);
        continuations.reserve(config.maximum_continuations);
        cleanup_parents.reserve(config.maximum_cleanup_parents);
        cleanup_retirements.reserve(config.maximum_cleanup_parents);
    }
    CUresult gate() const noexcept {
        if (::getpid() != process || !actor_thread || owner.lock() != actor_thread)
            return CUDA_ERROR_INVALID_CONTEXT;
        return active || active_attempt ? CUDA_ERROR_NOT_READY : CUDA_SUCCESS;
    }
    void emit(NcclActorAuditRecord record) noexcept {
        if (!observer.emit) return;
        record.actor_serial = backend::cuda::detail::take_diagnostic_id(next_audit_serial);
        if (record.actor_serial == UINT64_MAX) {
            record = {};
            record.actor_serial = UINT64_MAX;
            record.event = AuditEvent::kLost;
            const auto sink = observer;
            observer = {};
            sink.emit(sink.context, record);
            return;
        }
        observer.emit(observer.context, record);
    }
    static void audit_batch_key(NcclActorAuditRecord& out, const Batch& batch) noexcept {
        out.peer = batch.peer;
        out.priority = batch.priority;
        out.reservation = batch.identity;
        out.generation = batch.generation;
        out.open_group = batch.open_group;
        out.unknown = batch.uncertain;
    }
    NcclActorAuditRecord audit_record(const Record& record) const noexcept {
        NcclActorAuditRecord out;
        out.subject = AuditSubject::kRequest;
        out.id = record.id;
        out.peer = record.peer.identity;
        out.priority = record.peer.priority;
        out.caller = record.request.scope.thread;
        out.group = record.request.scope.group;
        out.sequence = record.request.scope.sequence;
        out.origin_source = record.request.scope.context;
        out.caller_primary_domain = record.caller_primary_domain;
        if (record.batch) audit_batch_key(out, *record.batch);
        out.cutoff = record.cutoff;
        out.native_sequence = record.audit_native_sequence;
        out.native_operation = record.operation;
        out.native_kind = static_cast<std::uint32_t>(kind(record.request));
        out.communicator_domain = record.audit_communicator_domain;
        out.communicator_context = record.audit_communicator_context;
        out.communicator_object = record.audit_communicator_object;
        out.output_contract = audit_enum(effects.output_contract);
        out.stage = audit_enum(record.stage);
        out.carrier = record.carrier;
        out.reply_ready = record.projected;
        out.reply_released = record.released;
        out.retired = record.backend_released;
        out.active_call = record.audit_active_call;
        out.ready_continuation = record.ready_continuation;
        if (record.observed) {
            const auto& observed = *record.observed;
            out.phase = audit_enum(observed.phase);
            out.has_original = observed.original_status.has_value();
            out.original_status = observed.original_status.value_or(0);
            out.enqueue_complete = observed.enqueue_complete;
            out.physical_complete = observed.physical_complete;
            if (observed.context_transition) {
                const auto& transition = *observed.context_transition;
                out.has_context = true;
                out.before_kind = audit_enum(transition.before.kind);
                out.before_identity = transition.before.identity;
                out.before_context = transition.before.context_id;
                out.before_device = static_cast<std::uint32_t>(
                    static_cast<std::int64_t>(transition.before.device_ordinal) + 1);
                out.after_kind = audit_enum(transition.after.kind);
                out.after_identity = transition.after.identity;
                out.after_context = transition.after.context_id;
                out.after_device = static_cast<std::uint32_t>(
                    static_cast<std::int64_t>(transition.after.device_ordinal) + 1);
            }
            if (observed.output) {
                const auto& output = *observed.output;
                out.has_output_written = output.output_written.has_value();
                out.output_written = output.output_written.value_or(false);
                if (output.operation == Kind::kGetAsyncError && output.has_output) {
                    out.has_async = true;
                    out.async_status = output.async_error;
                }
            }
        }
        return out;
    }
    NcclActorAuditRecord audit_continuation(const Continuation& record) const noexcept {
        NcclActorAuditRecord out;
        out.subject = AuditSubject::kContinuation;
        out.id = record.id;
        audit_batch_key(out, *record.batch);
        out.caller = record.scope.caller;
        out.group = record.scope.group;
        out.ordinary_sequence = record.scope.ordinary_sequence;
        out.cleanup_parent = record.scope.cleanup_parent;
        out.origin_source = record.scope.origin_source;
        out.cutoff = record.cutoff;
        out.connection = record.target.connection;
        out.wire_sequence = record.target.wire_sequence;
        out.ordinary_opcode = static_cast<std::uint32_t>(record.target.operation);
        out.attempt = record.attempt;
        out.stage = audit_enum(record.stage);
        out.status = record.status;
        out.reply_ready = record.reply.has_value();
        out.reply_released = record.released;
        out.retired = record.retired;
        if (record.binding) {
            out.source = record.binding->session;
            out.capture = record.binding->diagnostic_capture;
            out.command_sequence =
                record.binding->receipt ? record.binding->receipt->sequence() : 0;
        }
        if (record.reply) {
            out.disposition = audit_enum(record.reply->disposition);
            out.native_status = record.reply->status;
            out.has_original = record.reply->original_status.has_value();
            out.original_status = record.reply->original_status.value_or(0);
        }
        if (record.host_reply) {
            out.has_host = true;
            out.host_origin = audit_enum(record.host_reply->origin);
            out.completion_status = record.host_reply->status;
        }
        return out;
    }
    NcclActorAuditRecord audit_parent(const CleanupParent& parent) const noexcept {
        NcclActorAuditRecord out;
        out.subject = AuditSubject::kParent;
        audit_batch_key(out, *parent.batch);
        out.caller = parent.scope.caller;
        out.group = parent.scope.group;
        out.ordinary_sequence = parent.scope.ordinary_sequence;
        out.origin_source = parent.scope.origin_source;
        out.connection = parent.connection;
        out.allocation = parent.allocation;
        out.sequence = parent.sequence;
        out.related = parent.active_continuation;
        out.bound = parent.bound;
        out.consumed = parent.consumed;
        out.unknown = out.unknown || parent.uncertain;
        return out;
    }
    void changed(NcclActorAuditRecord record,
                 std::optional<NcclActorAuditRecord>& previous) noexcept {
        if (!observer.emit || (previous && *previous == record)) return;
        previous = record;
        emit(record);
    }
    void audit_all() noexcept {
        if (!observer.emit) return;
        for (auto& record : records) changed(audit_record(*record), record->audit_previous);
        for (auto& record : continuations)
            changed(audit_continuation(*record), record->audit_previous);
        for (auto& parent : cleanup_parents) {
            auto out = audit_parent(*parent);
            out.event = AuditEvent::kParent;
            changed(out, parent->audit_previous);
        }
        for (auto& batch : batches) {
            NcclActorAuditRecord out;
            out.event = AuditEvent::kBatch;
            out.subject = AuditSubject::kBatch;
            out.id = batch->identity;
            audit_batch_key(out, *batch);
            out.unknown |= uncertain;
            // Same conservative reservation formula as snapshot(), without
            // callbacks, allocation, native observation or recursive actor entry.
            out.has_physical_work = batch->open_group;
            for (const auto& record : records) {
                if (record->batch != batch) continue;
                ++out.count;
                if ((record->operation &&
                     (!record->observed || !record->observed->physical_complete)) ||
                    std::ranges::any_of(record->sources, [](const auto& source) {
                        return source.accepted && !source.finished;
                    }))
                    out.has_physical_work = true;
            }
            for (const auto& record : continuations)
                if (record->batch == batch) ++out.index;
            for (const auto& parent : cleanup_parents)
                if (parent->batch == batch) {
                    ++out.related;
                    out.unknown |= parent->uncertain;
                }
            changed(out, batch->audit_previous);
        }
    }
    template <typename Entity>
    void audit_action(const Entity& record, AuditEvent event, AuditAction action,
                      CUresult status = CUDA_SUCCESS) noexcept {
        auto out = [&] {
            if constexpr (std::is_same_v<Entity, Record>)
                return audit_record(record);
            else if constexpr (std::is_same_v<Entity, Continuation>)
                return audit_continuation(record);
            else
                return audit_parent(record);
        }();
        out.event = event;
        out.action = action;
        out.status = status;
        emit(out);
    }
    void audit_binding(const NcclActorAuditRecord& key, AuditBinding binding,
                       const NcclActorContinuationBinding& target,
                       AuditAction action = AuditAction::kBind) noexcept {
        if (!observer.emit) return;
        auto out = key;
        out.event = AuditEvent::kBinding;
        out.binding = binding;
        out.action = action;
        out.source = target.session;
        out.capture =
            target.diagnostic_capture ? target.diagnostic_capture : target.frontier.diagnostic_id();
        out.command_sequence = target.receipt ? target.receipt->sequence() : 0;
        emit(out);
    }
    void release_precall_diagnostics(const Continuation& record) noexcept {
        if (!observer.emit) return;
        for (const auto& binding : record.dependencies)
            audit_binding(audit_continuation(record), AuditBinding::kPrecall, binding,
                          AuditAction::kRemove);
    }
    void audit_source(const Record& record, const Source& source, AuditAction action,
                      CUresult status = CUDA_SUCCESS) noexcept {
        if (!observer.emit) return;
        for (std::size_t index = 0; index != source.streams.size(); ++index) {
            auto out = audit_record(record);
            out.event = AuditEvent::kSource;
            out.binding = AuditBinding::kTurn;
            out.action = action;
            out.source = source.identity;
            out.stream = source.streams[index];
            out.capture = source.audit_capture;
            out.index = index;
            out.count = source.streams.size();
            out.status = status;
            emit(out);
        }
    }
    struct AuditGrant {
        Implementation& state;
        NcclActorAuditRecord key;
        const CUresult& status;
        const CudaExecutionIssueResult* issue{};
        AuditGrant(Implementation& value, NcclActorAuditRecord record, NcclActorGrant grant,
                   const CUresult& result, AuditAction action = AuditAction::kBegin)
            : state(value), key(record), status(result) {
            key.input_reservation = grant.reservation;
            key.input_generation = grant.generation;
            key.input_cutoff = grant.cutoff;
            key.event = AuditEvent::kGrantBegin;
            key.action = action;
            state.emit(key);
        }
        ~AuditGrant() {
            key.event = AuditEvent::kGrantEnd;
            key.status = status;
            // The input grant remains the original snapshot, but an accepted
            // communicator must not disappear from its later grant-return row.
            if (key.subject == AuditSubject::kRequest) {
                if (const auto record = state.find(key.id)) {
                    key.communicator_domain = record->audit_communicator_domain;
                    key.communicator_context = record->audit_communicator_context;
                    key.communicator_object = record->audit_communicator_object;
                }
            }
            if (issue) {
                key.command_sequence = issue->sequence;
                key.outcome = audit_enum(issue->state);
                key.native_status = issue->native_status;
                key.completion_status = issue->completion_status;
                key.has_library = issue->library_status.has_value();
                key.library_status = issue->library_status.value_or(0);
            }
            state.emit(key);
        }
    };
    struct Scope {
        Implementation& state;
        explicit Scope(Implementation& value) : state(value) {
            state.active = true;
        }
        ~Scope() {
            state.audit_all();
            state.active = false;
        }
    };
    std::shared_ptr<Record> find(std::uint64_t id) const {
        const auto found =
            std::ranges::find_if(records, [id](const auto& item) { return item->id == id; });
        return found == records.end() ? nullptr : *found;
    }
    std::shared_ptr<Continuation> find_continuation(std::uint64_t id) const {
        const auto found =
            std::ranges::find_if(continuations, [id](const auto& item) { return item->id == id; });
        return found == continuations.end() ? nullptr : *found;
    }
    static NcclActorGrant continuation_grant(const Continuation& record) {
        return {record.batch->identity, record.batch->generation, record.cutoff};
    }
    void fail_continuation(Continuation& record, CUresult status) noexcept {
        record.status = status == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : status;
        record.stage = ContinuationStage::kUncertain;
        record.retired = false;
        record.batch->uncertain = uncertain = true;
        if (record.parent) record.parent->uncertain = true;
        audit_action(record, AuditEvent::kState, AuditAction::kReject, record.status);
    }
    bool continuation_stage(Continuation& record, ContinuationStage stage) noexcept {
        if (record.stage == stage) return true;
        if (!next_cutoff) {
            fail_continuation(record, CUDA_ERROR_OUT_OF_MEMORY);
            return false;
        }
        const auto previous = audit_enum(record.stage);
        record.cutoff = next_cutoff++;
        record.stage = stage;
        auto out = audit_continuation(record);
        out.previous_stage = previous;
        emit(out);
        return true;
    }
    CUresult valid_binding(const Continuation& record, const NcclActorContinuationBinding& binding,
                           bool requires_healthy = true) const {
        const bool retained_cleanup_origin =
            record.scope.flags == wire::kExecutionSourceContinuationCleanup && record.parent &&
            record.parent->bound && binding.session == record.parent->scope.origin_source &&
            binding.owner == record.parent->origin;
        const bool retained_closing_source = record.target.closing_source &&
                                             binding.session == record.target.connection &&
                                             binding.owner == record.target.closing_source;
        if (!binding.session || !binding.owner ||
            (!retained_cleanup_origin && !retained_closing_source &&
             resolver.session(record.peer.identity, binding.session) != binding.owner) ||
            !binding.owner->owns_frontier(binding.frontier) ||
            (binding.receipt && !*binding.receipt))
            return CUDA_ERROR_INVALID_HANDLE;
        return requires_healthy && binding.owner->snapshot().is_quarantined ? CUDA_ERROR_UNKNOWN
                                                                            : CUDA_SUCCESS;
    }
    CUresult continuation_ready(const Continuation& record) const noexcept {
        if (record.stage != ContinuationStage::kPrepared &&
            record.stage != ContinuationStage::kBlocked)
            return CUDA_ERROR_NOT_READY;
        for (const auto& dependency : record.dependencies) {
            if (dependency.owner->snapshot().is_quarantined) return CUDA_ERROR_UNKNOWN;
            const auto retired = dependency.frontier.retirement();
            // A cold-handler cut requires quiescence, not erasure of the old
            // operation's error. The unchanged handler still checks its gate.
            if (!retired.is_retired) return CUDA_ERROR_NOT_READY;
        }
        return CUDA_SUCCESS;
    }
    void refresh_continuation(Continuation& record) noexcept {
        if (record.stage == ContinuationStage::kUncertain || record.retired) return;
        if (record.binding && record.binding->owner->snapshot().is_quarantined) {
            fail_continuation(record, CUDA_ERROR_UNKNOWN);
            return;
        }
        if (record.stage != ContinuationStage::kWaitingRetirement) return;
        if (record.binding) {
            const auto retired = record.binding->frontier.retirement();
            if (!retired.is_retired) return;
            const bool matching_rejection =
                record.host_reply &&
                record.host_reply->origin == backend::ExecutionReplyOrigin::kRejectedBeforeIssue &&
                record.host_reply->status != CUDA_SUCCESS &&
                retired.status == record.host_reply->status;
            const bool matching_synchronous_error =
                !record.binding->receipt && record.reply && record.reply->original_status &&
                *record.reply->original_status != CUDA_SUCCESS &&
                retired.status == *record.reply->original_status;
            if (retired.status != CUDA_SUCCESS && !matching_rejection &&
                !matching_synchronous_error) {
                fail_continuation(record, retired.status);
                return;
            }
        }
        record.retired = true;
        (void)continuation_stage(record, ContinuationStage::kComplete);
        audit_action(record, AuditEvent::kRetirement, AuditAction::kRetire);
    }
    CUresult apply_continuation_outcome(Continuation& record,
                                        NcclActorContinuationOutcome outcome) {
        if (observer.emit) {
            auto out = audit_continuation(record);
            out.event = AuditEvent::kObservation;
            out.outcome = audit_enum(outcome.kind);
            out.status = outcome.status;
            out.command_sequence = outcome.pending_sequence;
            emit(out);
        }
        if ((outcome.kind == OutcomeKind::kReturned || outcome.kind == OutcomeKind::kUncertain) &&
            outcome.reply && !record.reply) {
            record.reply = std::move(outcome.reply);
            audit_action(record, AuditEvent::kReply, AuditAction::kPublish);
        }
        const auto invalid = [&] {
            // This consumes an already armed/accepted attempt. Contradictory
            // bookkeeping cannot authorize retry or discard its native owners.
            fail_continuation(record, CUDA_ERROR_INVALID_VALUE);
            return CUDA_ERROR_INVALID_VALUE;
        };
        if (outcome.binding) {
            const auto valid = valid_binding(record, *outcome.binding, false);
            if (valid != CUDA_SUCCESS) {
                fail_continuation(record, valid);
                return valid;
            }
            if (record.binding &&
                (record.binding->owner != outcome.binding->owner ||
                 record.binding->session != outcome.binding->session ||
                 !record.binding->frontier.same_capture(outcome.binding->frontier) ||
                 record.binding->receipt.has_value() != outcome.binding->receipt.has_value() ||
                 (record.binding->receipt &&
                  record.binding->receipt->sequence() != outcome.binding->receipt->sequence())))
                return invalid();
        }
        if (outcome.kind == OutcomeKind::kDeferred) {
            if (outcome.status != CUDA_ERROR_NOT_READY || outcome.reply || outcome.binding ||
                outcome.pending_lifetime || outcome.pending_sequence ||
                outcome.imported_allocation || record.binding)
                return invalid();
            record.pending_lifetime.reset();
            record.pending_sequence = 0;
            return continuation_stage(record, ContinuationStage::kBlocked) ? CUDA_SUCCESS
                                                                           : record.status;
        }
        if (outcome.kind == OutcomeKind::kCommandAccepted) {
            if (outcome.status != CUDA_SUCCESS || !outcome.binding || !outcome.binding->receipt ||
                outcome.reply || outcome.pending_lifetime || outcome.pending_sequence ||
                outcome.imported_allocation || record.binding)
                return invalid();
            outcome.binding->diagnostic_capture =
                observer.emit ? outcome.binding->frontier.diagnostic_id() : 0;
            audit_binding(audit_continuation(record), AuditBinding::kTarget, *outcome.binding);
            record.binding = std::move(outcome.binding);
            release_precall_diagnostics(record);
            record.dependencies.clear();
            return continuation_stage(record, ContinuationStage::kCommandPending) ? CUDA_SUCCESS
                                                                                  : record.status;
        }
        if (outcome.kind == OutcomeKind::kHostJobPending) {
            if (outcome.status != CUDA_SUCCESS || !outcome.pending_lifetime ||
                outcome.pending_sequence != record.target.wire_sequence || outcome.reply ||
                outcome.binding || outcome.imported_allocation || !record.target.helper_owner ||
                outcome.pending_lifetime != record.target.helper_owner)
                return invalid();
            record.pending_lifetime = std::move(outcome.pending_lifetime);
            record.pending_sequence = outcome.pending_sequence;
            return continuation_stage(record, ContinuationStage::kHostPending) ? CUDA_SUCCESS
                                                                               : record.status;
        }
        if (outcome.kind != OutcomeKind::kReturned && outcome.kind != OutcomeKind::kUncertain)
            return invalid();
        if (outcome.pending_lifetime || outcome.pending_sequence) return invalid();
        if (outcome.binding && !record.binding) {
            outcome.binding->diagnostic_capture =
                observer.emit ? outcome.binding->frontier.diagnostic_id() : 0;
            audit_binding(audit_continuation(record), AuditBinding::kTarget, *outcome.binding);
            record.binding = std::move(outcome.binding);
        }
        if (record.binding && record.binding->receipt && !record.host_reply) {
            record.host_reply = record.binding->owner->host_result(*record.binding->receipt);
            if (!record.host_reply && outcome.kind != OutcomeKind::kUncertain) return invalid();
        }
        if (outcome.kind == OutcomeKind::kUncertain) {
            fail_continuation(record, outcome.status);
            return record.status;
        }
        if (outcome.status != CUDA_SUCCESS || !record.reply) return invalid();
        if (record.binding && record.binding->owner->snapshot().is_quarantined) {
            fail_continuation(record, CUDA_ERROR_UNKNOWN);
            return record.status;
        }
        const bool imported = record.target.operation == wire::ExecutionOpcode::kImportHostMemory &&
                              record.scope.flags == 0;
        const bool success = record.reply->status == CUDA_SUCCESS &&
                             record.reply->disposition == wire::ExecutionWireDisposition::kComplete;
        if (imported && success) {
            if (!record.parent || record.parent->bound || record.reply->original_status != 0 ||
                !outcome.imported_allocation || outcome.imported_allocation != record.reply->id)
                return invalid();
            record.parent->allocation = outcome.imported_allocation;
            record.parent->bound = true;
        } else if (outcome.imported_allocation) {
            return invalid();
        } else if (imported) {
            record.parent->consumed = true;
        }
        if (record.scope.flags == wire::kExecutionSourceContinuationCleanup) {
            if (!record.parent || !record.parent->bound || record.parent->consumed)
                return invalid();
            if (success) record.parent->consumed = true;
            // An error is not authority to repeat an entered native free. The
            // runtime must classify a proven no-entry retry as Deferred instead.
            else
                record.parent->uncertain = true;
        }
        record.pending_lifetime.reset();
        record.pending_sequence = 0;
        release_precall_diagnostics(record);
        record.dependencies.clear();
        if (!continuation_stage(record, ContinuationStage::kWaitingRetirement))
            return record.status;
        refresh_continuation(record);
        return record.status;
    }
    void refresh_continuations() noexcept {
        for (auto& record : continuations) refresh_continuation(*record);
        std::erase_if(continuations, [&](const auto& record) {
            if (!record->released || !record->retired) return false;
            if (record->parent && record->parent->active_continuation == record->id)
                record->parent->active_continuation = 0;
            release_precall_diagnostics(*record);
            if (record->binding)
                audit_binding(audit_continuation(*record), AuditBinding::kTarget, *record->binding,
                              AuditAction::kRemove);
            audit_action(*record, AuditEvent::kRetirement, AuditAction::kRemove);
            return true;
        });
        std::erase_if(cleanup_parents, [&](const auto& parent) {
            const bool remove =
                parent->consumed && !parent->uncertain && !parent->active_continuation;
            if (remove) audit_action(*parent, AuditEvent::kParent, AuditAction::kRemove);
            return remove;
        });
    }
    CUresult empty_context() const {
        CUcontext current{};
        const auto result = lane.context_get_current(&current);
        return result != CUDA_SUCCESS ? result
               : current              ? CUDA_ERROR_INVALID_CONTEXT
                                      : CUDA_SUCCESS;
    }
    void fail(Record& record, CUresult status) {
        record.carrier = status == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : status;
        record.stage = Stage::kUncertain;
        record.batch->uncertain = true;
        uncertain = true;
        audit_action(record, AuditEvent::kState, AuditAction::kReject, record.carrier);
    }
    bool stage(Record& record, Stage value) {
        if (record.stage == value) return true;
        if (!next_cutoff) {
            fail(record, CUDA_ERROR_OUT_OF_MEMORY);
            return false;
        }
        const auto previous = audit_enum(record.stage);
        record.stage = value;
        record.cutoff = next_cutoff++;
        auto out = audit_record(record);
        out.previous_stage = previous;
        emit(out);
        return true;
    }
    template <typename Function>
    CUresult source_call(Record& record, CudaExecutionSession& session, Function function) {
        const auto before = empty_context();
        if (before != CUDA_SUCCESS) {
            fail(record, before);
            return before;
        }
        const auto selected = session.activate();
        if (selected.status != CUDA_SUCCESS) {
            fail(record, selected.status);
            return selected.status;
        }
        CUresult result = CUDA_ERROR_UNKNOWN;
        try {
            result = function();
        } catch (const std::bad_alloc&) {
            result = CUDA_ERROR_OUT_OF_MEMORY;
        } catch (...) {
            result = CUDA_ERROR_UNKNOWN;
        }
        // Session checks exact source ownership before clearing current. Failure
        // retains state; never blindly select NULL after a lost context proof.
        const auto restored = session.deactivate();
        if (restored.status != CUDA_SUCCESS) {
            fail(record, restored.status);
            return restored.status;
        }
        const auto proof = empty_context();
        if (proof != CUDA_SUCCESS) {
            fail(record, proof);
            return proof;
        }
        return result;
    }
    CUresult add_source(Record& record, NcclActorSource source) {
        if (!source.session || !source.owner) return CUDA_ERROR_INVALID_HANDLE;
        std::size_t streams{};
        for (const auto& item : record.sources) streams += item.streams.size();
        const auto found = std::ranges::find(record.sources, source.session, &Source::identity);
        if (found != record.sources.end()) {
            if (found->owner != source.owner) return CUDA_ERROR_INVALID_HANDLE;
            if (std::ranges::find(found->streams, source.stream) == found->streams.end()) {
                if (streams == config.coordinator.maximum_tails) return CUDA_ERROR_OUT_OF_MEMORY;
                found->streams.push_back(source.stream);
            }
        } else {
            if (record.sources.size() == config.maximum_sources ||
                streams == config.coordinator.maximum_tails)
                return CUDA_ERROR_OUT_OF_MEMORY;
            Source item;
            item.identity = source.session;
            item.owner = std::move(source.owner);
            item.streams.push_back(source.stream);
            record.sources.push_back(std::move(item));
        }
        return CUDA_SUCCESS;
    }
    CUresult buffer(Record& record, wire::ExecutionNcclBuffer input, std::size_t bytes,
                    bool receive) {
        if (input.kind == wire::ExecutionNcclBufferKind::kNull) return CUDA_SUCCESS;
        if (input.kind != wire::ExecutionNcclBufferKind::kOwned) return CUDA_ERROR_INVALID_VALUE;
        auto session = resolver.session(record.peer.identity, input.source);
        if (!session) return CUDA_ERROR_INVALID_HANDLE;
        record.buffers.push_back({input, std::move(session), bytes, receive});
        return CUDA_SUCCESS;
    }
    CUresult materialize(Record& record) {
        auto& native = record.native_request;
        const auto operation = kind(record.request);
        native.caller = {record.peer.identity, record.request.scope.thread};
        native.arguments.operation = operation;
        native.arguments.has_output = has_output(record.request);
        native.arguments.output_contract = effects.output_contract;
        const auto comm = communicator(record.request);
        std::int32_t ranks{};
        if (comm.object) {
            const auto found = std::ranges::find(communicators, comm.object, &Comm::object);
            if (comm.domain != config.primary_domain || found == communicators.end() ||
                found->peer != record.peer.identity || found->terminal)
                return CUDA_ERROR_INVALID_HANDLE;
            native.communicator = comm.object;
            ranks = found->ranks;
        }
        if (record.request.scope.context) {
            record.entry_owner =
                resolver.session(record.peer.identity, record.request.scope.context);
            if (!record.entry_owner) return CUDA_ERROR_INVALID_CONTEXT;
        }
        if (const auto* value = std::get_if<wire::ExecutionNcclInitRank>(&record.request.call)) {
            native.arguments.ranks = value->ranks;
            native.arguments.rank = value->rank;
            native.arguments.unique_id = value->unique_id;
        } else if (const auto* value =
                       std::get_if<wire::ExecutionNcclInitRankConfig>(&record.request.call)) {
            native.arguments.ranks = value->initialization.ranks;
            native.arguments.rank = value->initialization.rank;
            native.arguments.unique_id = value->initialization.unique_id;
            if (value->config) {
                const auto& input = *value->config;
                native.arguments.config =
                    CudaNcclNativeConfig{input.declared_bytes, input.magic,    input.version,
                                         input.fields,         input.net_name, input.comm_name};
            }
        }
        if (record.outer_end) {
            for (const auto& item : record.members)
                for (const auto& source : item->sources)
                    for (const auto stream : source.streams) {
                        const auto result =
                            add_source(record, {source.identity, source.owner, stream});
                        if (result != CUDA_SUCCESS) return result;
                    }
        }
        if (!collective(operation)) return CUDA_SUCCESS;
        wire::ExecutionNcclStream stream;
        wire::ExecutionNcclBuffer send, receive;
        std::int32_t datatype{};
        if (const auto* broadcast = std::get_if<wire::ExecutionNcclBcast>(&record.request.call)) {
            stream = broadcast->stream;
            receive = broadcast->buffer;
            native.arguments.count = broadcast->count;
            native.arguments.root = broadcast->root;
            datatype = broadcast->datatype;
        } else {
            const auto* value = std::get_if<wire::ExecutionNcclAllGather>(&record.request.call);
            if (const auto* reduction =
                    std::get_if<wire::ExecutionNcclAllReduce>(&record.request.call)) {
                value = &reduction->collective;
                if (reduction->reduction < 0) return CUDA_ERROR_INVALID_VALUE;
                native.arguments.reduction = static_cast<ncclRedOp_t>(reduction->reduction);
            }
            if (!value) return CUDA_ERROR_INVALID_VALUE;
            stream = value->stream;
            send = value->send;
            receive = value->receive;
            native.arguments.count = value->count;
            datatype = value->datatype;
        }
        constexpr std::array<std::size_t, 12> widths{1, 1, 4, 4, 8, 8, 2, 4, 8, 2, 1, 1};
        if (datatype < 0 || static_cast<std::size_t>(datatype) >= widths.size())
            return CUDA_ERROR_INVALID_VALUE;
        const auto width = widths[static_cast<std::size_t>(datatype)];
        if (native.arguments.count > SIZE_MAX / width) return CUDA_ERROR_INVALID_VALUE;
        const auto bytes = static_cast<std::size_t>(native.arguments.count) * width;
        auto receive_bytes = bytes;
        if (operation == Kind::kAllGather && bytes) {
            if (ranks <= 0 || static_cast<std::size_t>(ranks) > SIZE_MAX / bytes)
                return CUDA_ERROR_INVALID_VALUE;
            receive_bytes *= static_cast<std::size_t>(ranks);
        }
        native.arguments.datatype = static_cast<ncclDataType_t>(datatype);
        auto source = resolver.stream(record.peer.identity, record.request.scope, stream);
        if (stream.kind == wire::ExecutionNcclStreamKind::kOwned &&
            (source.session != stream.owner || source.stream != stream.object))
            return CUDA_ERROR_INVALID_HANDLE;
        auto result = add_source(record, std::move(source));
        if (result != CUDA_SUCCESS) return result;
        result = buffer(record, send, bytes, false);
        return result == CUDA_SUCCESS ? buffer(record, receive, receive_bytes, true) : result;
    }
    CUresult capture(Record& record) {
        if (record.captured) return CUDA_SUCCESS;
        if (resolver.admit_source) {
            const auto check = [&](std::uint64_t source) {
                return resolver.admit_source(record.peer.identity, source);
            };
            if (record.entry_owner) {
                const auto status = check(record.request.scope.context);
                if (status != CUDA_SUCCESS) return status;
            }
            for (const auto& buffer : record.buffers) {
                const auto status = check(buffer.input.source);
                if (status != CUDA_SUCCESS) return status;
            }
            for (const auto& source : record.sources) {
                const auto status = check(source.identity);
                if (status != CUDA_SUCCESS) return status;
            }
        }
        if (!initialized) {
            const auto before = empty_context();
            if (before != CUDA_SUCCESS) {
                fail(record, before);
                return before;
            }
            const auto status = primary.initialize();
            if (status != CUDA_SUCCESS) {
                fail(record, status);
                return status;
            }
            const auto captured = primary.capture_context();
            if (captured.status != CUDA_SUCCESS) {
                fail(record, captured.status);
                return captured.status;
            }
            domain = {config.primary_domain, config.device_ordinal, captured.context};
            initialized = true;
        }
        record.native_request.domain = domain;
        auto bundle = std::make_shared<Bundle>();
        bundle->commands.reserve(record.buffers.size() + record.sources.size());
        bundle->contexts.reserve(record.entry_owner ? 1 : 0);
        record.native_request.caller_context = CudaNcclCoordinatorDomain{};
        if (record.entry_owner) {
            CudaSessionExternalContextCapture captured;
            const auto status = source_call(record, *record.entry_owner, [&] {
                captured = record.entry_owner->capture_external_context();
                return captured.status;
            });
            if (status != CUDA_SUCCESS) return status;
            if (captured.context.device_ordinal() != config.device_ordinal)
                return CUDA_ERROR_INVALID_CONTEXT;
            bundle->contexts.push_back(captured.context);
            auto lease = std::make_shared<const CudaSessionExternalContext>(captured.context);
            record.native_request.caller_context = CudaNcclCoordinatorDomain{
                record.request.scope.context,
                captured.context.device_ordinal(),
                {captured.context.id(), captured.context.native(), std::move(lease)}};
        }
        if (record.caller_primary_domain) record.native_request.caller_context = domain;
        for (const auto& input : record.buffers) {
            CudaSessionExternalCommandCapture captured;
            const CudaSessionExternalRangeRequest range{
                {input.input.allocation, input.input.offset}, input.bytes};
            const auto status = source_call(record, *input.owner, [&] {
                captured = input.owner->capture_external_command(0, std::span{&range, 1});
                return captured.status;
            });
            if (status != CUDA_SUCCESS) return status;
            if (captured.command.context().device_ordinal() != config.device_ordinal ||
                captured.command.ranges().size() != 1)
                return CUDA_ERROR_INVALID_CONTEXT;
            const auto address = captured.command.ranges()[0].address;
            // Native device pointer materialization is confined to this trusted
            // service composition; no caller/server address is sent on the wire.
            if (input.receive) {
                // NOLINTNEXTLINE(performance-no-int-to-ptr) -- Owned Driver address.
                record.native_request.arguments.receive = reinterpret_cast<void*>(address);
            } else {
                // NOLINTNEXTLINE(performance-no-int-to-ptr) -- Owned Driver address.
                record.native_request.arguments.send = reinterpret_cast<const void*>(address);
            }
            bundle->commands.push_back(captured.command);
        }
        for (const auto& source : record.sources) {
            for (const auto stream : source.streams) {
                CudaSessionExternalCommandCapture captured;
                const auto status = source_call(record, *source.owner, [&] {
                    captured = source.owner->capture_external_command(stream, {});
                    return captured.status;
                });
                if (status != CUDA_SUCCESS) return status;
                if (!captured.command.stream() ||
                    captured.command.context().device_ordinal() != config.device_ordinal)
                    return CUDA_ERROR_INVALID_CONTEXT;
                if (collective(kind(record.request)))
                    record.native_request.arguments.stream =
                        reinterpret_cast<cudaStream_t>(captured.command.stream());
                bundle->commands.push_back(captured.command);
            }
        }
        for (const auto& item : record.members) {
            if (!item->bundle) continue;
            bundle->commands.insert(bundle->commands.end(), item->bundle->commands.begin(),
                                    item->bundle->commands.end());
            bundle->contexts.insert(bundle->contexts.end(), item->bundle->contexts.begin(),
                                    item->bundle->contexts.end());
        }
        record.bundle = std::move(bundle);
        record.native_request.arguments.resources_lifetime = record.bundle;
        // Exact source-native streams supply FIFO. Source begin performs actual
        // dependency waits, and finish records actual posted prefixes; there is
        // no hidden worker stream or fabricated physical completion here.
        if (collective(kind(record.request)))
            record.native_request.prepare_enqueue = [] { return CUDA_SUCCESS; };
        if ((!record.grouped && collective(kind(record.request))) ||
            (record.outer_end && !record.sources.empty()))
            record.native_request.record_enqueue = [] { return CUDA_SUCCESS; };
        record.captured = true;
        return CUDA_SUCCESS;
    }
    CUresult cancel_turns(Record& record) {
        for (auto& source : record.sources) {
            if (!source.accepted || source.finished) continue;
            const auto result = source_call(record, *source.owner, [&] {
                return source.owner->cancel_enqueue_turn(source.turn.turn);
            });
            if (result != CUDA_SUCCESS) {
                audit_source(record, source, AuditAction::kCancel, result);
                fail(record, result);
                return result;
            }
            audit_source(record, source, AuditAction::kCancel);
            source.accepted = source.begun = false;
            source.turn = {};
        }
        record.native_request.source_turns.clear();
        return CUDA_SUCCESS;
    }
    CUresult accept_turns(Record& record) {
        if (resolver.admit_source)
            for (const auto& source : record.sources) {
                if (source.accepted) continue;
                const auto status = resolver.admit_source(record.peer.identity, source.identity);
                if (status != CUDA_SUCCESS) return status;
            }
        record.native_request.source_turns.clear();
        record.native_request.source_turns.reserve(record.sources.size());
        for (auto& source : record.sources) {
            if (!source.accepted) {
                source.turn = source.owner->accept_enqueue_cohort(source.streams, record.bundle);
                if (source.turn.status != CUDA_SUCCESS) {
                    const auto original = source.turn.status;
                    const auto cancelled = cancel_turns(record);
                    return cancelled == CUDA_SUCCESS ? original : cancelled;
                }
                source.accepted = true;
                source.audit_capture = observer.emit ? source.turn.frontier.diagnostic_id() : 0;
                audit_source(record, source, AuditAction::kAccept);
            }
            record.native_request.source_turns.push_back(source.turn.worker);
        }
        return CUDA_SUCCESS;
    }
    CUresult begin_turns(Record& record) {
        for (const auto& source : record.sources) {
            const auto ready = source.owner->enqueue_turn_ready(source.turn.turn);
            if (ready != CUDA_SUCCESS) return ready;
        }
        for (auto& source : record.sources) {
            const auto result = source_call(record, *source.owner, [&] {
                return source.owner->begin_enqueue_turn(source.turn.turn);
            });
            if (result != CUDA_SUCCESS) {
                audit_source(record, source, AuditAction::kBegin, result);
                fail(record, result);
                return result;
            }
            source.begun = true;
            audit_source(record, source, AuditAction::kBegin);
        }
        return CUDA_SUCCESS;
    }
    CUresult finish_sources(Record& record) {
        for (auto& source : record.sources) {
            if (source.finished) continue;
            const auto result = source_call(record, *source.owner, [&] {
                return source.owner->finish_enqueue_turn(source.turn.turn);
            });
            if (result != CUDA_SUCCESS) {
                audit_source(record, source, AuditAction::kFinish, result);
                if (result != CUDA_ERROR_NOT_READY) fail(record, result);
                return result;
            }
            source.finished = true;
            audit_source(record, source, AuditAction::kFinish);
        }
        if (record.observed && record.observed->enqueue_complete && !record.tails_bound &&
            !record.needs_late_tails) {
            std::vector<CudaExecutionFrontier> tails;
            tails.reserve(record.sources.size());
            for (const auto& source : record.sources) tails.push_back(source.turn.frontier);
            if (!tails.empty()) {
                const auto status = coordinator.bind_tails(record.operation, tails);
                if (status != CUDA_SUCCESS) {
                    fail(record, status);
                    return status;
                }
                record.tails_bound = true;
                for (const auto& source : record.sources)
                    audit_binding(audit_record(record), AuditBinding::kTail,
                                  {source.identity, source.owner, source.turn.frontier, {}});
            }
        }
        return CUDA_SUCCESS;
    }
    CUresult record_late_tails(Record& record) {
        for (auto& source : record.sources) {
            if (!source.final_tail.members.empty()) continue;
            source.final_tail = source.owner->accept_external_cohort(source.streams, record.bundle);
            if (source.final_tail.status != CUDA_SUCCESS) return source.final_tail.status;
            if (observer.emit)
                for (const auto& item : source.final_tail.members) {
                    auto out = audit_record(record);
                    out.stream = item.stream;
                    audit_binding(out, AuditBinding::kTail,
                                  {source.identity, source.owner, item.frontier, {}});
                }
        }
        for (const auto& source : record.sources)
            if (!source.tail_begun)
                for (const auto& item : source.final_tail.members) {
                    const auto ready = source.owner->external_barrier_ready(item.barrier);
                    if (ready != CUDA_SUCCESS) return ready;
                }
        std::vector<CudaExecutionFrontier> tails;
        std::size_t count{};
        for (const auto& source : record.sources) count += source.final_tail.members.size();
        tails.reserve(count);
        for (const auto& source : record.sources)
            for (const auto& item : source.final_tail.members) tails.push_back(item.frontier);
        for (auto& source : record.sources) {
            if (!source.tail_recorded) {
                const auto status = source_call(record, *source.owner, [&] {
                    for (const auto& item : source.final_tail.members) {
                        const auto begun = source.owner->begin_external_barrier(item.barrier);
                        if (observer.emit) {
                            auto out = audit_record(record);
                            out.event = AuditEvent::kSource;
                            out.binding = AuditBinding::kTail;
                            out.action = AuditAction::kBegin;
                            out.source = source.identity;
                            out.stream = item.stream;
                            out.capture = item.frontier.diagnostic_id();
                            out.count = 1;
                            out.status = begun;
                            emit(out);
                        }
                        if (begun != CUDA_SUCCESS) return begun;
                    }
                    source.tail_begun = true;
                    for (const auto& item : source.final_tail.members) {
                        const auto recorded = source.owner->record_external_enqueue(item.barrier);
                        if (observer.emit) {
                            auto out = audit_record(record);
                            out.event = AuditEvent::kSource;
                            out.binding = AuditBinding::kTail;
                            out.action = AuditAction::kRecord;
                            out.source = source.identity;
                            out.stream = item.stream;
                            out.capture = item.frontier.diagnostic_id();
                            out.count = 1;
                            out.status = recorded;
                            emit(out);
                        }
                        if (recorded != CUDA_SUCCESS) return recorded;
                    }
                    return CUDA_SUCCESS;
                });
                if (status != CUDA_SUCCESS) {
                    fail(record, status);
                    return status;
                }
                source.tail_recorded = true;
            }
        }
        const auto status = coordinator.bind_tails(record.operation, tails);
        if (status != CUDA_SUCCESS) {
            fail(record, status);
            return status;
        }
        record.tails_bound = true;
        stage(record, Stage::kWaitingTails);
        return CUDA_SUCCESS;
    }
    void apply_transition(Record& record) {
        if (record.transition_applied || !record.observed || !record.observed->original_status)
            return;
        record.transition_applied = true;
        auto& caller = callers[record.caller_index];
        const auto operation = kind(record.request);
        const auto status = *record.observed->original_status;
        const auto previous_depth = caller.depth;
        if (record.observed->context_transition &&
            record.observed->context_transition->after.kind == CudaNcclContextKind::kDomain &&
            record.observed->context_transition->after.identity == config.primary_domain)
            for (auto& item : callers)
                if (item.peer.identity == record.peer.identity) item.observed_primary = true;
        if (status == 7) record.needs_late_tails = true;
        if (operation == Kind::kGroupStart && status == 0) {
            ++caller.depth;
            caller.group = record.request.scope.group;
            caller.last_group = caller.group;
            caller.batch = record.batch;
            record.batch->open_group = true;
        }
        if (operation == Kind::kGroupEnd && caller.depth && (status == 0 || status == 7)) {
            --caller.depth;
            if (!caller.depth) {
                caller.group = 0;
                caller.batch.reset();
                caller.members.clear();
                record.batch->open_group = false;
            }
        }
        if (previous_depth != caller.depth) {
            auto out = audit_record(record);
            out.event = AuditEvent::kGroup;
            out.index = previous_depth;
            out.depth = caller.depth;
            emit(out);
        }
        if (record.grouped && member(operation)) {
            caller.members.push_back(find(record.id));
            auto out = audit_record(record);
            out.event = AuditEvent::kBinding;
            out.binding = AuditBinding::kGroupMember;
            out.related = record.id;
            out.index = caller.members.size() - 1;
            out.count = caller.members.size();
            emit(out);
        }
        const auto comm = communicator(record.request);
        const auto found = std::ranges::find(communicators, comm.object, &Comm::object);
        if (found != communicators.end() && (collective(operation) || status == 7))
            found->pending = record.batch;
        if (found != communicators.end() &&
            (operation == Kind::kDestroy || operation == Kind::kAbort) && status == 0)
            found->terminal = true;
        if (initialization(operation) && record.observed->physical_complete &&
            record.observed->output && !record.observed->output->communicator) {
            const auto reserved = std::ranges::find(
                communicators, record.observed->reserved_communicator, &Comm::object);
            if (reserved != communicators.end()) reserved->terminal = true;
        }
    }
    CUresult project_one(const Record& record,
                         const std::optional<CudaNcclNativeObservation>& native, bool deferred,
                         wire::ExecutionNcclObservation& output) const {
        if (!has_output(record.request)) return CUDA_SUCCESS;
        if (!effects.produce || !record.observed || !record.observed->original_status)
            return CUDA_ERROR_NOT_READY;
        const auto reserved =
            record.observed->reserved_communicator
                ? wire::ExecutionNcclCommunicator{config.primary_domain,
                                                  record.observed->reserved_communicator}
                : wire::ExecutionNcclCommunicator{};
        const NcclActorEffectInput input{record.request, *record.observed->original_status, native,
                                         reserved, deferred};
        const auto status = effects.produce(effects.context, input, output);
        if (status != CUDA_SUCCESS) return status;
        wire::ExecutionPayloadWriter validator;
        if (!wire::write_execution_nccl_observation(validator, output)) return CUDA_ERROR_UNKNOWN;
        const auto operation = kind(record.request);
        const auto allowed = operation == Kind::kGetUniqueId ? 1U
                             : initialization(operation)     ? 2U
                                                             : 4U;
        if (output.effects & ~allowed) return CUDA_ERROR_UNKNOWN;
        if (output.communicator.object && output.communicator != reserved)
            return CUDA_ERROR_UNKNOWN;
        return CUDA_SUCCESS;
    }
    CUresult project(Record& record) {
        if (record.projected || !record.observed || !record.observed->reply_ready)
            return CUDA_SUCCESS;
        wire::ExecutionNcclReply reply;
        reply.is_accepted = true;
        reply.operation = record.id;
        if (record.observed->reserved_communicator)
            reply.reserved_communicator = {config.primary_domain,
                                           record.observed->reserved_communicator};
        reply.original_status = record.observed->original_status;
        // Containment without original provenance cannot invent any output effect.
        if (reply.original_status) {
            const auto status =
                project_one(record, record.observed->output, record.grouped, reply.observation);
            if (status != CUDA_SUCCESS) return status;
        }
        if (const auto& transition = record.observed->context_transition) {
            if (transition->after.kind == CudaNcclContextKind::kDomain) {
                if (transition->after.identity != config.primary_domain ||
                    transition->after.context_id != domain.context.id ||
                    transition->after.device_ordinal != config.device_ordinal)
                    return CUDA_ERROR_INVALID_CONTEXT;
                reply.context_effect = {wire::ExecutionNcclContextEffectKind::kSelectPrimary,
                                        config.device_ordinal, config.primary_domain};
            } else if (transition->after.kind == CudaNcclContextKind::kNull &&
                       transition->before.kind != CudaNcclContextKind::kNull) {
                reply.context_effect.kind = wire::ExecutionNcclContextEffectKind::kClearCurrent;
            }
        }
        reply.publications.reserve(record.observed->publications.size());
        for (const auto& publication : record.observed->publications) {
            const auto found = std::ranges::find_if(record.members, [&](const auto& item) {
                return item->operation == publication.operation;
            });
            if (found == record.members.end()) return CUDA_ERROR_UNKNOWN;
            // Pinned host-stable cells have no later vendor writes. A previously
            // projected inner must not be published a second time at End.
            if ((*found)->projected &&
                effects.output_contract == CudaNcclNativeOutputContract::kPinnedHostStable)
                continue;
            wire::ExecutionNcclPublication projected;
            projected.scope = (*found)->request.scope;
            const auto status =
                project_one(**found, publication.output, false, projected.observation);
            if (status != CUDA_SUCCESS) return status;
            reply.publications.push_back(projected);
            auto out = audit_record(record);
            out.event = AuditEvent::kBinding;
            out.binding = AuditBinding::kPublication;
            out.related = (*found)->id;
            out.index = reply.publications.size() - 1;
            out.count = record.observed->publications.size();
            out.allocation = publication.reserved_communicator;
            emit(out);
        }
        wire::ExecutionPayloadWriter validation;
        if (!wire::write_execution_nccl_reply(validation, reply)) return CUDA_ERROR_UNKNOWN;
        record.envelope = std::move(reply);
        record.projected = true;
        audit_action(record, AuditEvent::kReply, AuditAction::kPublish);
        return CUDA_SUCCESS;
    }
    bool sources_finished(const Record& record) const {
        return std::ranges::all_of(record.sources, [](const auto& item) { return item.finished; });
    }
    CUresult ready(const Record& record) const {
        if (record.stage == Stage::kPrepared || record.stage == Stage::kFinishSources ||
            record.stage == Stage::kRecordContinuation)
            return CUDA_SUCCESS;
        if (record.stage == Stage::kSourceBlocked) {
            bool is_blocked = false;
            for (const auto& source : record.sources) {
                const auto inspected = source.owner->inspect_enqueue_dependency(source.turn.turn);
                if (inspected.status == CUDA_SUCCESS) continue;
                if (inspected.status == CUDA_ERROR_NOT_READY || inspected.has_failure_observation ||
                    inspected.foreign_blocker) {
                    // A genuine failed predecessor remains a dependency, even
                    // after retirement. Only grant-bound reject_source may
                    // consume its settled proof; snapshot never begins NCCL.
                    // A foreign blocker needs separate owner authentication;
                    // its raw dependency error supplies no rejection proof.
                    is_blocked = true;
                    continue;
                }
                return inspected.status;
            }
            return is_blocked ? CUDA_ERROR_NOT_READY : CUDA_SUCCESS;
        }
        if (record.stage == Stage::kRecordTails) {
            for (const auto& source : record.sources)
                for (const auto& item : source.final_tail.members) {
                    const auto status = source.owner->external_barrier_ready(item.barrier);
                    if (status != CUDA_SUCCESS) return status;
                }
            return CUDA_SUCCESS;
        }
        return CUDA_ERROR_NOT_READY;
    }
    void reject(Record& record, CUresult status) {
        record.carrier = status;
        stage(record, Stage::kRejected);
        record.envelope.emplace();
        record.projected = true;
        audit_action(record, AuditEvent::kReply, AuditAction::kReject, status);
        record.bundle.reset();
        record.native_request = {};
    }
    CUresult refresh() {
        const auto advanced = coordinator.advance(false);
        if (advanced != CUDA_SUCCESS) return advanced;
        const auto backend = coordinator.snapshot();
        if (backend.carrier != CUDA_SUCCESS) return backend.carrier;
        for (const auto& pointer : records) {
            auto& record = *pointer;
            if (!record.operation || record.backend_released) continue;
            if (record.stage == Stage::kUncertain) {
                // A later source-tail/projection fault must not erase an already
                // observed requested native status or independently proved output.
                const auto projected = project(record);
                if (projected != CUDA_SUCCESS && projected != CUDA_ERROR_NOT_READY)
                    fail(record, projected);
                continue;
            }
            auto observed = coordinator.observe(record.operation);
            record.observed = std::move(observed);
            const auto found = std::ranges::find(backend.operations, record.operation,
                                                 &CudaNcclOperationSnapshot::operation);
            record.ready_continuation =
                found != backend.operations.end() && found->ready_continuation;
            record.audit_active_call = found != backend.operations.end() && found->active_call;
            if (observer.emit) {
                auto out = audit_record(record);
                out.event = AuditEvent::kObservation;
                out.carrier = record.observed->carrier;
                out.reply_ready = record.observed->reply_ready;
                // Native observations change only with native evidence, not
                // later actor projection, ACK, stage or group bookkeeping.
                out.stage = out.previous_stage = 0;
                out.cutoff = 0;
                out.reply_released = out.open_group = out.unknown = false;
                changed(out, record.audit_observation);
            }
            apply_transition(record);
            if (!record.observed->reply_ready) continue;
            record.carrier = record.observed->carrier;
            if (record.continuation_pending && found != backend.operations.end() &&
                found->active_call) {
                continue;
            } else if (!sources_finished(record)) {
                // finish independently checks the exact lane ticket. A requested
                // native error or coordinator semantic containment is not itself
                // evidence that the host call failed to return safely.
                stage(record, Stage::kFinishSources);
            } else if (record.carrier != CUDA_SUCCESS) {
                fail(record, record.carrier);
            } else {
                record.continuation_pending = false;
                if (record.observed->phase == CudaNcclCoordinatorPhase::kUncertain) {
                    fail(record, CUDA_ERROR_UNKNOWN);
                } else if (record.observed->physical_complete ||
                           (!record.ready_continuation &&
                            record.observed->phase != CudaNcclCoordinatorPhase::kWaitingAsync &&
                            !record.observed->enqueue_complete)) {
                    // A returned grouped inner completes its host stage only;
                    // whole-group physical ownership remains in the same batch.
                    stage(record, Stage::kComplete);
                } else if (record.ready_continuation) {
                    stage(record, Stage::kRecordContinuation);
                } else if (record.observed->phase == CudaNcclCoordinatorPhase::kWaitingAsync) {
                    stage(record, Stage::kWaitingAsync);
                } else {
                    stage(record, record.needs_late_tails && !record.tails_bound
                                      ? Stage::kRecordTails
                                      : Stage::kWaitingTails);
                }
            }
            if (sources_finished(record) || record.stage == Stage::kUncertain) {
                const auto projected = project(record);
                if (projected != CUDA_SUCCESS && projected != CUDA_ERROR_NOT_READY)
                    fail(record, projected);
            }
        }
        // Keep coordinator receipts until actual physical completion. This avoids
        // mistaking an erased released group member for an invalid observation.
        for (const auto& record : records) {
            if (!record->released || record->backend_released || !record->operation ||
                !record->observed || !record->observed->physical_complete)
                continue;
            const auto status = coordinator.release_reply(record->operation);
            if (status != CUDA_SUCCESS) {
                fail(*record, status);
                continue;
            }
            record->backend_released = true;
            audit_action(*record, AuditEvent::kRetirement, AuditAction::kRetire);
            record->native_request = {};
            record->bundle.reset();
            record->sources.clear();
            record->buffers.clear();
            record->entry_owner.reset();
            record->members.clear();
        }
        std::erase_if(records, [&](const auto& record) {
            const bool remove =
                record->released && (record->stage == Stage::kRejected || record->backend_released);
            if (remove) audit_action(*record, AuditEvent::kRetirement, AuditAction::kRemove);
            return remove;
        });
        refresh_continuations();
        std::erase_if(batches, [&](const auto& batch) {
            const bool remove =
                !batch->open_group && !batch->uncertain &&
                std::ranges::none_of(records,
                                     [&](const auto& record) { return record->batch == batch; }) &&
                std::ranges::none_of(continuations,
                                     [&](const auto& record) { return record->batch == batch; }) &&
                std::ranges::none_of(cleanup_parents,
                                     [&](const auto& parent) { return parent->batch == batch; });
            if (remove) {
                NcclActorAuditRecord out;
                audit_batch_key(out, *batch);
                out.event = AuditEvent::kBatch;
                out.subject = AuditSubject::kBatch;
                out.id = batch->identity;
                out.action = AuditAction::kRemove;
                out.retired = true;
                emit(out);
            }
            return remove;
        });
        std::erase_if(communicators, [](const auto& item) { return item.terminal; });
        return CUDA_SUCCESS;
    }
};

NcclActor::NcclActor(NcclActorConfig config, CudaNcclNativeDispatch native,
                     CudaNativeExecutionLaneDispatch lane, CudaNcclPrimaryDomainDispatch primary,
                     NcclActorResolver resolver, NcclActorEffectProducer effects)
    : implementation_(std::make_unique<Implementation>(config, native, lane, primary,
                                                       std::move(resolver), std::move(effects))) {}
NcclActor::~NcclActor() {
    if (!implementation_ || implementation_->stopped) return;
    // No destructor SDK calls, worker cancellation or inferred physical stop.
    (void)implementation_.release();
}

bool NcclActor::set_observer(NcclActorObserver observer) noexcept {
    auto& state = *implementation_;
    if (state.gate() != CUDA_SUCCESS || state.closed || state.observer_installed ||
        !state.callers.empty() || state.next_request != 1 || state.next_continuation != 1)
        return false;
    if (!observer.emit) return !observer.context;
    const Implementation::Scope scope(state);
    state.observer = observer;
    state.observer_installed = true;
    NcclActorAuditRecord begin;
    begin.event = AuditEvent::kBegin;
    state.emit(begin);
    return true;
}

NcclActorPreparation NcclActor::prepare(NcclActorPeer peer, wire::ExecutionNcclRequest request,
                                        std::uint64_t caller_primary_domain) noexcept {
    auto& state = *implementation_;
    const auto gate = state.gate();
    if (gate != CUDA_SUCCESS) return {gate};
    const Implementation::Scope scope(state);
    NcclActorAuditRecord admission;
    admission.event = AuditEvent::kAdmission;
    admission.subject = AuditSubject::kRequest;
    admission.peer = peer.identity;
    admission.priority = peer.priority;
    admission.caller = request.scope.thread;
    admission.group = request.scope.group;
    admission.sequence = request.scope.sequence;
    admission.origin_source = request.scope.context;
    admission.caller_primary_domain = caller_primary_domain;
    admission.native_kind = static_cast<std::uint32_t>(kind(request));
    admission.output_contract = audit_enum(state.effects.output_contract);
    const auto run = [&]() -> NcclActorPreparation {
        try {
            wire::ExecutionPayloadWriter canonical;
            if (!peer.identity || !wire::write_execution_nccl_request(canonical, request))
                return {CUDA_ERROR_INVALID_VALUE};
            if (std::ranges::find(state.disconnected_peers, peer.identity) !=
                state.disconnected_peers.end())
                return {CUDA_ERROR_DEINITIALIZED};
            auto found = std::ranges::find_if(state.callers, [&](const auto& caller) {
                return caller.peer.identity == peer.identity &&
                       caller.thread == request.scope.thread;
            });
            if (found == state.callers.end()) {
                if (state.callers.size() == state.config.maximum_callers ||
                    request.scope.sequence != 1)
                    return {state.callers.size() == state.config.maximum_callers
                                ? CUDA_ERROR_OUT_OF_MEMORY
                                : CUDA_ERROR_INVALID_VALUE};
                Caller caller;
                caller.peer = peer;
                caller.thread = request.scope.thread;
                caller.members.reserve(state.config.coordinator.maximum_group_members);
                caller.observed_primary = std::ranges::any_of(state.callers, [&](const auto& item) {
                    return item.peer.identity == peer.identity && item.observed_primary;
                });
                state.callers.push_back(std::move(caller));
                found = std::prev(state.callers.end());
            }
            auto& caller = *found;
            if (caller.peer.priority != peer.priority || caller.public_sequence == UINT64_MAX ||
                request.scope.sequence != caller.public_sequence + 1)
                return {CUDA_ERROR_INVALID_VALUE};
            // Canonical clean rejections consume the application sequence. Native
            // coordinator sequences advance ONLY after its actual acceptance.
            ++caller.public_sequence;
            admission.sequence_consumed = true;
            if (state.closed || caller.disconnected) return {CUDA_ERROR_DEINITIALIZED};
            if (state.uncertain) return {CUDA_ERROR_UNKNOWN};
            if (caller_primary_domain &&
                (caller_primary_domain != state.config.primary_domain || !caller.observed_primary))
                return {CUDA_ERROR_INVALID_CONTEXT};
            const auto operation = kind(request);
            const bool start = operation == Kind::kGroupStart;
            const bool end = operation == Kind::kGroupEnd;
            if (caller.depth) {
                if (request.scope.group != caller.group) return {CUDA_ERROR_INVALID_VALUE};
            } else if (start) {
                if (!request.scope.group || request.scope.group <= caller.last_group)
                    return {CUDA_ERROR_INVALID_VALUE};
            } else if (request.scope.group)
                return {CUDA_ERROR_INVALID_VALUE};
            // Initialization and collectives cannot share an outer native group.
            // Reject before source capture or native sequence admission; existing
            // members and the reserved End remain available to finish that group.
            if (caller.depth && std::ranges::any_of(caller.members, [&](const auto& prior) {
                    const auto previous = kind(prior->request);
                    return (initialization(operation) && collective(previous)) ||
                           (collective(operation) && initialization(previous));
                }))
                return {CUDA_ERROR_INVALID_VALUE};
            const auto index = static_cast<std::size_t>(found - state.callers.begin());
            if (std::ranges::any_of(state.continuations, [&](const auto& item) {
                    return item->caller_index == index && !item->scope.flags && !item->reply &&
                           !item->retired;
                }))
                return {CUDA_ERROR_NOT_READY};
            if (std::ranges::any_of(state.records, [&](const auto& record) {
                    return record->caller_index == index &&
                           (record->stage == Stage::kPrepared ||
                            record->stage == Stage::kSourceBlocked ||
                            record->stage == Stage::kNativePending ||
                            record->stage == Stage::kFinishSources);
                }))
                return {CUDA_ERROR_NOT_READY};
            if (caller.depth && member(operation) &&
                caller.members.size() == state.config.coordinator.maximum_group_members)
                return {CUDA_ERROR_OUT_OF_MEMORY};
            const bool outer_end = end && caller.depth == 1;
            const bool end_credit = end && caller.depth;
            const bool progress =
                !caller.depth && (operation == Kind::kGetAsyncError || operation == Kind::kAbort);
            const auto normal_count = std::ranges::count_if(state.records, [](const auto& item) {
                return !item->end_credit && !item->progress_credit;
            });
            const auto end_count = std::ranges::count_if(
                state.records, [](const auto& item) { return item->end_credit; });
            const auto progress_count = std::ranges::count_if(
                state.records, [](const auto& item) { return item->progress_credit; });
            // Match the coordinator's one retained End record per group. Nested
            // Ends reuse their own slot after release, never another group's slot.
            if (end_credit && std::ranges::any_of(state.records, [&](const auto& item) {
                    return item->end_credit && item->batch == caller.batch;
                }))
                return {CUDA_ERROR_OUT_OF_MEMORY};
            if (state.records.size() == state.records.capacity() ||
                (!end_credit && !progress &&
                 static_cast<std::size_t>(normal_count) >= state.config.maximum_requests) ||
                (end_credit &&
                 static_cast<std::size_t>(end_count) >= state.config.maximum_groups) ||
                (progress && static_cast<std::size_t>(progress_count) >=
                                 state.config.coordinator.maximum_progress_operations) ||
                !state.next_request || !state.next_cutoff || !state.next_generation)
                return {CUDA_ERROR_OUT_OF_MEMORY};
            if (start && !caller.depth &&
                static_cast<std::size_t>(
                    std::ranges::count_if(state.batches, [&](const auto& batch) {
                        return batch->is_group &&
                               (batch->open_group || batch->uncertain ||
                                std::ranges::any_of(
                                    state.records,
                                    [&](const auto& item) { return item->batch == batch; }) ||
                                std::ranges::any_of(state.continuations, [&](const auto& item) {
                                    return item->batch == batch && !item->retired;
                                }));
                    })) >= state.config.maximum_groups)
                return {CUDA_ERROR_OUT_OF_MEMORY};
            if (initialization(operation) && has_output(request) &&
                state.communicators.size() == state.config.maximum_communicators)
                return {CUDA_ERROR_OUT_OF_MEMORY};
            auto record = std::make_shared<Record>();
            record->id = state.next_request;
            record->cutoff = state.next_cutoff;
            record->peer = peer;
            record->caller_index = index;
            record->request = std::move(request);
            record->caller_primary_domain = caller_primary_domain;
            record->outer_end = outer_end;
            record->end_credit = end_credit;
            record->progress_credit = progress;
            record->grouped = caller.depth && member(operation);
            if (outer_end) record->members = caller.members;
            record->sources.reserve(state.config.maximum_sources);
            record->buffers.reserve(2);
            const auto status = state.materialize(*record);
            if (status != CUDA_SUCCESS) return {status};
            if (caller.depth) record->batch = caller.batch;
            if (!record->batch && progress) {
                const auto comm = communicator(record->request);
                const auto item =
                    std::ranges::find(state.communicators, comm.object, &Comm::object);
                if (item != state.communicators.end()) record->batch = item->pending.lock();
            }
            if (!record->batch) {
                if (state.batches.size() == state.config.maximum_reservations)
                    return {CUDA_ERROR_OUT_OF_MEMORY};
                record->batch = std::make_shared<Batch>(Batch{.identity = record->id,
                                                              .generation = state.next_generation++,
                                                              .peer = peer.identity,
                                                              .priority = peer.priority,
                                                              .is_group = start});
                state.batches.push_back(record->batch);
            }
            state.records.push_back(record);
            ++state.next_request;
            ++state.next_cutoff;
            return {CUDA_SUCCESS, record->id};
        } catch (const std::bad_alloc&) {
            return {CUDA_ERROR_OUT_OF_MEMORY};
        } catch (...) {
            return {CUDA_ERROR_UNKNOWN};
        }
    };
    const auto result = run();
    if (result.request) {
        const auto record = state.find(result.request);
        if (record) {
            admission = state.audit_record(*record);
            admission.sequence_consumed = true;
        }
    }
    admission.event = AuditEvent::kAdmission;
    admission.status = result.status;
    admission.accepted = result.request != 0;
    admission.action = admission.accepted ? AuditAction::kAccept : AuditAction::kReject;
    state.emit(admission);
    if (result.request) {
        const auto record = state.find(result.request);
        if (record)
            for (std::size_t index = 0; index != record->members.size(); ++index) {
                auto joined = state.audit_record(*record);
                joined.event = AuditEvent::kBinding;
                joined.action = AuditAction::kBind;
                joined.binding = AuditBinding::kGroupMember;
                joined.related = record->members[index]->id;
                joined.index = index;
                joined.count = record->members.size();
                state.emit(joined);
            }
    }
    return result;
}

CUresult NcclActor::issue(std::uint64_t id, NcclActorGrant grant) noexcept {
    auto& state = *implementation_;
    const auto gate = state.gate();
    if (gate != CUDA_SUCCESS) return gate;
    const Implementation::Scope scope(state);
    auto record = state.find(id);
    if (!record ||
        grant != NcclActorGrant{record->batch->identity, record->batch->generation, record->cutoff})
        return CUDA_ERROR_INVALID_VALUE;
    if (state.uncertain) return CUDA_ERROR_UNKNOWN;
    CUresult outcome = CUDA_ERROR_UNKNOWN;
    const Implementation::AuditGrant audit(state, state.audit_record(*record), grant, outcome);
    const auto run = [&]() -> CUresult {
        try {
            if (record->stage == Stage::kPrepared) {
                auto status = state.capture(*record);
                if (status == CUDA_SUCCESS) status = state.accept_turns(*record);
                if (status != CUDA_SUCCESS) {
                    if (status == CUDA_ERROR_NOT_READY && record->stage != Stage::kUncertain)
                        return status;
                    if (record->stage != Stage::kUncertain) state.reject(*record, status);
                    return status;
                }
                // This grant authorizes this exact initial setup and submission, not
                // an unrelated later continuation. A blocked source needs a new grant.
                state.stage(*record, Stage::kSourceBlocked);
            }
            if (record->stage == Stage::kSourceBlocked) {
                if (initialization(kind(record->request)) && has_output(record->request) &&
                    state.communicators.size() == state.config.maximum_communicators) {
                    const auto cancelled = state.cancel_turns(*record);
                    if (cancelled == CUDA_SUCCESS) state.reject(*record, CUDA_ERROR_OUT_OF_MEMORY);
                    return cancelled == CUDA_SUCCESS ? CUDA_ERROR_OUT_OF_MEMORY : cancelled;
                }
                const auto ready = state.begin_turns(*record);
                if (ready != CUDA_SUCCESS) return ready;
                auto& caller = state.callers[record->caller_index];
                if (caller.native_sequence == UINT64_MAX) {
                    const auto cancelled = state.cancel_turns(*record);
                    if (cancelled == CUDA_SUCCESS) state.reject(*record, CUDA_ERROR_OUT_OF_MEMORY);
                    return cancelled == CUDA_SUCCESS ? CUDA_ERROR_OUT_OF_MEMORY : cancelled;
                }
                record->native_request.sequence = caller.native_sequence + 1;
                const auto accepted = state.coordinator.submit(record->native_request);
                if (accepted.carrier != CUDA_SUCCESS) {
                    const auto cancelled = state.cancel_turns(*record);
                    if (cancelled != CUDA_SUCCESS) return cancelled;
                    if (accepted.carrier == CUDA_ERROR_NOT_READY) {
                        state.stage(*record, Stage::kPrepared);
                        return accepted.carrier;
                    }
                    state.reject(*record, accepted.carrier);
                    return accepted.carrier;
                }
                ++caller.native_sequence;
                record->operation = accepted.operation;
                record->audit_native_sequence = record->native_request.sequence;
                const auto object = accepted.reserved_communicator
                                        ? accepted.reserved_communicator
                                        : record->native_request.communicator;
                if (object) {
                    record->audit_communicator_domain = record->native_request.domain.identity;
                    record->audit_communicator_context = record->native_request.domain.context.id;
                    record->audit_communicator_object = object;
                }
                {
                    auto out = state.audit_record(*record);
                    out.event = AuditEvent::kNativeSubmit;
                    out.accepted = true;
                    out.allocation = accepted.reserved_communicator;
                    out.input_reservation = grant.reservation;
                    out.input_generation = grant.generation;
                    out.input_cutoff = grant.cutoff;
                    state.emit(out);
                }
                record->batch->has_native = true;
                if (accepted.reserved_communicator) {
                    Comm comm{accepted.reserved_communicator, record->peer.identity,
                              record->native_request.arguments.ranks};
                    comm.pending = record->batch;
                    state.communicators.push_back(std::move(comm));
                }
                state.stage(*record, Stage::kNativePending);
                return CUDA_SUCCESS;
            }
            if (record->stage == Stage::kFinishSources) {
                const auto status = state.finish_sources(*record);
                return status == CUDA_SUCCESS ? state.refresh() : status;
            }
            if (record->stage == Stage::kRecordContinuation) {
                const auto status = state.coordinator.continue_enqueue(record->operation);
                if (status != CUDA_SUCCESS) return status;
                record->continuation_pending = true;
                state.stage(*record, Stage::kNativePending);
                return CUDA_SUCCESS;
            }
            if (record->stage == Stage::kRecordTails) return state.record_late_tails(*record);
            return CUDA_ERROR_NOT_READY;
        } catch (const std::bad_alloc&) {
            // Allocation failures after any native/submission transition are not
            // authority to cancel, replay or discard a source prefix.
            state.fail(*record, CUDA_ERROR_OUT_OF_MEMORY);
            return CUDA_ERROR_OUT_OF_MEMORY;
        } catch (...) {
            state.fail(*record, CUDA_ERROR_UNKNOWN);
            return CUDA_ERROR_UNKNOWN;
        }
    };
    outcome = run();
    return outcome;
}

NcclActorSourceInspection NcclActor::inspect_source(std::uint64_t id, NcclActorGrant grant,
                                                    std::uint64_t source_session) const noexcept {
    auto& state = *implementation_;
    NcclActorSourceInspection result;
    const auto gate = state.gate();
    if (gate != CUDA_SUCCESS) {
        result.status = gate;
        return result;
    }
    const Implementation::Scope scope(state);
    const auto record = state.find(id);
    if (!record ||
        grant != NcclActorGrant{record->batch->identity, record->batch->generation, record->cutoff})
        return result;
    if (state.uncertain) {
        result.status = CUDA_ERROR_UNKNOWN;
        return result;
    }
    if (record->stage != Stage::kSourceBlocked) {
        result.status = CUDA_ERROR_NOT_READY;
        return result;
    }
    const auto source = std::ranges::find(record->sources, source_session, &Source::identity);
    if (source == record->sources.end() || !source->accepted || source->begun) return result;
    result.source_session = source_session;
    result.dependency = source->owner->inspect_enqueue_dependency(source->turn.turn);
    result.status = CUDA_SUCCESS;
    return result;
}

NcclActorSourceProgress NcclActor::progress_source(std::uint64_t id, NcclActorGrant grant,
                                                   std::uint64_t source_session) noexcept {
    auto& state = *implementation_;
    NcclActorSourceProgress result;
    const auto gate = state.gate();
    if (gate != CUDA_SUCCESS) {
        result.status = gate;
        return result;
    }
    const Implementation::Scope scope(state);
    const auto record = state.find(id);
    if (!record ||
        grant != NcclActorGrant{record->batch->identity, record->batch->generation, record->cutoff})
        return result;
    if (state.uncertain) {
        result.status = CUDA_ERROR_UNKNOWN;
        return result;
    }
    if (record->stage != Stage::kSourceBlocked) {
        result.status = CUDA_ERROR_NOT_READY;
        return result;
    }
    const auto found = std::ranges::find(record->sources, source_session, &Source::identity);
    if (found == record->sources.end() || !found->accepted || found->begun) return result;
    result.source_session = source_session;
    auto audit_key = state.audit_record(*record);
    audit_key.source = source_session;
    audit_key.capture = found->audit_capture;
    Implementation::AuditGrant audit(state, audit_key, grant, result.status, AuditAction::kObserve);
    audit.issue = &result.issue;
    result.status = state.empty_context();
    if (result.status == CUDA_SUCCESS) result.status = found->owner->activate().status;
    if (result.status != CUDA_SUCCESS) {
        state.fail(*record, result.status);
        return result;
    }
    result.issue = found->owner->progress_enqueue_dependency(found->turn.turn);
    result.status = result.issue.completion_status != CUDA_SUCCESS ? result.issue.completion_status
                    : result.issue.state == CudaExecutionIssueState::kError
                        ? result.issue.native_status
                        : CUDA_SUCCESS;
    if (result.issue.completion_status != CUDA_SUCCESS ||
        (result.status != CUDA_SUCCESS && result.status != CUDA_ERROR_NOT_READY)) {
        // Preserve the actual issue, but do not make a restoring SDK callback
        // after the backend has lost its independent completion/context proof.
        state.fail(*record, result.status);
        return result;
    }
    const auto restored = found->owner->deactivate();
    const auto proof = restored.status == CUDA_SUCCESS ? state.empty_context() : restored.status;
    if (proof != CUDA_SUCCESS) result.status = proof;
    if (result.status != CUDA_SUCCESS && result.status != CUDA_ERROR_NOT_READY)
        state.fail(*record, result.status);
    return result;
}

CUresult NcclActor::reject_source(std::uint64_t id, NcclActorGrant grant,
                                  std::uint64_t source_session, CUresult failure) noexcept {
    auto& state = *implementation_;
    const auto gate = state.gate();
    if (gate != CUDA_SUCCESS) return gate;
    const Implementation::Scope scope(state);
    const auto record = state.find(id);
    if (!record || failure == CUDA_SUCCESS ||
        grant != NcclActorGrant{record->batch->identity, record->batch->generation, record->cutoff})
        return CUDA_ERROR_INVALID_VALUE;
    if (state.uncertain) return CUDA_ERROR_UNKNOWN;
    if (record->stage != Stage::kSourceBlocked || record->operation ||
        std::ranges::any_of(record->sources, [](const auto& source) { return source.begun; }))
        return CUDA_ERROR_NOT_READY;
    const auto matched_source =
        std::ranges::find(record->sources, source_session, &Source::identity);
    if (matched_source == record->sources.end() || !matched_source->accepted)
        return CUDA_ERROR_INVALID_VALUE;
    CUresult result = CUDA_ERROR_UNKNOWN;
    auto audit_key = state.audit_record(*record);
    audit_key.source = source_session;
    audit_key.capture = matched_source->audit_capture;
    const Implementation::AuditGrant audit(state, audit_key, grant, result, AuditAction::kReject);
    const auto run = [&]() -> CUresult {
        const auto inspected =
            matched_source->owner->inspect_enqueue_dependency(matched_source->turn.turn);
        if (inspected.status != failure || !inspected.has_failure_observation ||
            !inspected.has_retired_failure)
            return CUDA_ERROR_INVALID_VALUE;
        if (std::ranges::any_of(record->sources, [](const auto& source) {
                return source.owner->snapshot().is_quarantined;
            }))
            return CUDA_ERROR_UNKNOWN;
        try {
            const auto cancelled = state.cancel_turns(*record);
            if (cancelled != CUDA_SUCCESS) return cancelled;
            state.reject(*record, failure);
            return CUDA_SUCCESS;
        } catch (...) {
            state.fail(*record, CUDA_ERROR_UNKNOWN);
            return CUDA_ERROR_UNKNOWN;
        }
    };
    result = run();
    return result;
}

NcclActorPreparation NcclActor::prepare_continuation(NcclActorPeer peer,
                                                     wire::ExecutionSourceContinuation input,
                                                     NcclActorContinuationTarget target) noexcept {
    auto& state = *implementation_;
    const auto gate = state.gate();
    if (gate != CUDA_SUCCESS) return {gate};
    const Implementation::Scope scope(state);
    NcclActorAuditRecord admission;
    admission.event = AuditEvent::kAdmission;
    admission.subject = AuditSubject::kContinuation;
    admission.peer = peer.identity;
    admission.priority = peer.priority;
    admission.caller = input.caller;
    admission.group = input.group;
    admission.ordinary_sequence = input.ordinary_sequence;
    admission.cleanup_parent = input.cleanup_parent;
    admission.origin_source = input.origin_source;
    admission.connection = target.connection;
    admission.wire_sequence = target.wire_sequence;
    admission.ordinary_opcode = static_cast<std::uint32_t>(target.operation);
    const auto run = [&]() -> NcclActorPreparation {
        try {
            const bool cleanup = input.flags == wire::kExecutionSourceContinuationCleanup;
            const auto opcode = static_cast<std::uint32_t>(target.operation);
            if (!peer.identity || !input.caller[0] || !input.caller[1] || !input.group ||
                !input.ordinary_sequence ||
                input.flags > wire::kExecutionSourceContinuationCleanup ||
                cleanup != (input.cleanup_parent != 0) || !target.connection ||
                !target.wire_sequence || !target.request_lifetime || !opcode ||
                (opcode >= static_cast<std::uint32_t>(wire::ExecutionOpcode::kNcclCall) &&
                 target.operation != wire::ExecutionOpcode::kEnqueueHostFunction) ||
                (cleanup && (target.operation != wire::ExecutionOpcode::kReleaseHostMemory ||
                             !target.cleanup_allocation || !target.helper_owner)) ||
                (!cleanup && target.cleanup_allocation) ||
                (target.closing_source &&
                 (cleanup || target.operation != wire::ExecutionOpcode::kClose ||
                  input.origin_source != target.connection)))
                return {CUDA_ERROR_INVALID_VALUE};
            const auto caller = std::ranges::find_if(state.callers, [&](const auto& item) {
                return item.peer.identity == peer.identity && item.thread == input.caller;
            });
            if (caller == state.callers.end() || caller->peer.priority != peer.priority)
                return {CUDA_ERROR_INVALID_VALUE};
            std::shared_ptr<CleanupParent> parent;
            if (cleanup) {
                const auto found =
                    std::ranges::find_if(state.cleanup_parents, [&](const auto& item) {
                        return item->peer.identity == peer.identity &&
                               item->scope.caller == input.caller &&
                               item->scope.group == input.group &&
                               item->scope.origin_source == input.origin_source &&
                               item->scope.ordinary_sequence == input.cleanup_parent;
                    });
                if (found == state.cleanup_parents.end()) return {CUDA_ERROR_INVALID_HANDLE};
                parent = *found;
                if (!parent->bound || parent->consumed || parent->uncertain ||
                    parent->connection != target.connection ||
                    parent->helper_owner != target.helper_owner ||
                    parent->allocation != target.cleanup_allocation)
                    return {CUDA_ERROR_INVALID_HANDLE};
            }
            auto& sequence = cleanup ? parent->sequence : caller->ordinary_sequence;
            if (sequence == UINT64_MAX || input.ordinary_sequence != sequence + 1)
                return {CUDA_ERROR_INVALID_VALUE};
            ++sequence;  // Canonical policy failures consume only this ordinary namespace.
            admission.sequence_consumed = true;
            if (std::ranges::any_of(state.cleanup_retirements, [&](const auto& item) {
                    return item.connection == target.connection ||
                           (target.helper_owner && item.helper == target.helper_owner);
                }))
                return {CUDA_ERROR_DEINITIALIZED};
            if (state.closed || caller->disconnected) return {CUDA_ERROR_DEINITIALIZED};
            if (state.uncertain) return {CUDA_ERROR_UNKNOWN};
            if (!cleanup && (!caller->depth || caller->group != input.group || !caller->batch))
                return {CUDA_ERROR_INVALID_VALUE};
            if (cleanup && parent->active_continuation) return {CUDA_ERROR_NOT_READY};
            if (!input.origin_source && !cleanup &&
                target.operation != wire::ExecutionOpcode::kHello && !target.is_context_free_helper)
                return {CUDA_ERROR_INVALID_CONTEXT};
            const bool imports =
                !cleanup && target.operation == wire::ExecutionOpcode::kImportHostMemory;
            if (imports && !target.helper_owner) return {CUDA_ERROR_INVALID_HANDLE};
            const auto caller_index = static_cast<std::size_t>(caller - state.callers.begin());
            if (!cleanup && std::ranges::any_of(state.continuations, [&](const auto& item) {
                    return item->caller_index == caller_index && !item->scope.flags &&
                           !item->reply && !item->retired;
                }))
                return {CUDA_ERROR_NOT_READY};
            if (state.continuations.size() == state.config.maximum_continuations ||
                (imports && state.cleanup_parents.size() == state.config.maximum_cleanup_parents) ||
                !state.next_continuation || !state.next_cutoff)
                return {CUDA_ERROR_OUT_OF_MEMORY};
            auto record = std::make_shared<Continuation>();
            record->dependencies.reserve(state.config.maximum_sources);
            record->id = state.next_continuation;
            record->cutoff = state.next_cutoff;
            record->caller_index = caller_index;
            record->peer = peer;
            record->scope = input;
            record->target = std::move(target);
            record->batch = cleanup ? parent->batch : caller->batch;
            if (input.origin_source) {
                record->origin = cleanup ? parent->origin
                                 : record->target.closing_source
                                     ? record->target.closing_source
                                     : state.resolver.session(peer.identity, input.origin_source);
                if (!record->origin) return {CUDA_ERROR_INVALID_CONTEXT};
            }
            if (imports) {
                parent = std::make_shared<CleanupParent>();
                parent->peer = peer;
                parent->scope = input;
                parent->connection = record->target.connection;
                parent->helper_owner = record->target.helper_owner;
                parent->origin = record->origin;
                parent->batch = record->batch;
            }
            record->parent = parent;
            if (imports) state.cleanup_parents.push_back(parent);
            state.continuations.push_back(record);
            if (parent) parent->active_continuation = record->id;
            ++state.next_continuation;
            ++state.next_cutoff;
            return {CUDA_SUCCESS, record->id};
        } catch (const std::bad_alloc&) {
            return {CUDA_ERROR_OUT_OF_MEMORY};
        } catch (...) {
            return {CUDA_ERROR_UNKNOWN};
        }
    };
    const auto result = run();
    if (result.request) {
        const auto record = state.find_continuation(result.request);
        if (record) {
            admission = state.audit_continuation(*record);
            admission.sequence_consumed = true;
        }
    }
    admission.event = AuditEvent::kAdmission;
    admission.status = result.status;
    admission.accepted = result.request != 0;
    admission.action = admission.accepted ? AuditAction::kAccept : AuditAction::kReject;
    state.emit(admission);
    return result;
}

CUresult NcclActor::bind_continuation_dependency(std::uint64_t id,
                                                 NcclActorContinuationBinding binding) noexcept {
    auto& state = *implementation_;
    const auto gate = state.gate();
    if (gate != CUDA_SUCCESS) return gate;
    const Implementation::Scope scope(state);
    const auto record = state.find_continuation(id);
    if (!record || binding.receipt) return CUDA_ERROR_INVALID_VALUE;
    if (record->stage != ContinuationStage::kPrepared || record->attempt)
        return CUDA_ERROR_NOT_READY;
    try {
        const auto valid = state.valid_binding(*record, binding);
        if (valid != CUDA_SUCCESS) return valid;
        const auto found = std::ranges::find(record->dependencies, binding.session,
                                             &NcclActorContinuationBinding::session);
        if (found != record->dependencies.end())
            return found->owner == binding.owner && found->frontier.same_capture(binding.frontier)
                       ? CUDA_SUCCESS
                       : CUDA_ERROR_INVALID_VALUE;
        if (record->dependencies.size() == state.config.maximum_sources)
            return CUDA_ERROR_OUT_OF_MEMORY;
        binding.diagnostic_capture = state.observer.emit ? binding.frontier.diagnostic_id() : 0;
        record->dependencies.push_back(std::move(binding));
        state.audit_binding(state.audit_continuation(*record), AuditBinding::kPrecall,
                            record->dependencies.back());
        return CUDA_SUCCESS;
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}

NcclActorContinuationAdmission NcclActor::begin_continuation(std::uint64_t id,
                                                             NcclActorGrant grant) noexcept {
    auto& state = *implementation_;
    const auto gate = state.gate();
    if (gate != CUDA_SUCCESS) return {gate, {}};
    const Implementation::Scope scope(state);
    const auto record = state.find_continuation(id);
    if (!record || grant != state.continuation_grant(*record))
        return {CUDA_ERROR_INVALID_VALUE, {}};
    if (state.uncertain) return {CUDA_ERROR_UNKNOWN, {}};
    const auto ready = state.continuation_ready(*record);
    if (ready != CUDA_SUCCESS) return {ready, {}};
    // Reserve a future stage cutoff before external code can run. Publishing
    // the actual result must not fail merely because the counter just exhausted.
    if (!state.next_attempt || !state.next_cutoff || state.next_cutoff == UINT64_MAX)
        return {CUDA_ERROR_OUT_OF_MEMORY, {}};
    record->attempt = state.next_attempt++;
    record->audit_input_grant = grant;
    record->cutoff = state.next_cutoff++;
    record->stage = ContinuationStage::kAttempting;
    state.active_attempt = record->id;
    auto out = state.audit_continuation(*record);
    out.event = AuditEvent::kGrantBegin;
    out.input_reservation = grant.reservation;
    out.input_generation = grant.generation;
    out.input_cutoff = grant.cutoff;
    state.emit(out);
    return {CUDA_SUCCESS, {record->id, record->attempt}};
}

CUresult NcclActor::finish_continuation_attempt(NcclActorContinuationPermit permit,
                                                NcclActorContinuationOutcome outcome) noexcept {
    auto& state = *implementation_;
    if (::getpid() != state.process || !actor_thread || state.owner.lock() != actor_thread)
        return CUDA_ERROR_INVALID_CONTEXT;
    if (state.active) return CUDA_ERROR_NOT_READY;
    const auto record = state.find_continuation(permit.continuation);
    if (!record || !permit.attempt || state.active_attempt != record->id ||
        record->stage != ContinuationStage::kAttempting || record->attempt != permit.attempt)
        return CUDA_ERROR_INVALID_VALUE;
    const Implementation::Scope scope(state);
    state.active_attempt = 0;
    CUresult result = CUDA_ERROR_UNKNOWN;
    try {
        result = state.apply_continuation_outcome(*record, std::move(outcome));
    } catch (...) {
        state.fail_continuation(*record, CUDA_ERROR_UNKNOWN);
    }
    auto out = state.audit_continuation(*record);
    out.event = AuditEvent::kGrantEnd;
    out.action = AuditAction::kFinish;
    out.input_reservation = record->audit_input_grant.reservation;
    out.input_generation = record->audit_input_grant.generation;
    out.input_cutoff = record->audit_input_grant.cutoff;
    out.status = result;
    state.emit(out);
    return result;
}

CUresult NcclActor::complete_continuation(std::uint64_t id,
                                          NcclActorContinuationOutcome outcome) noexcept {
    auto& state = *implementation_;
    const auto gate = state.gate();
    if (gate != CUDA_SUCCESS) return gate;
    const Implementation::Scope scope(state);
    const auto record = state.find_continuation(id);
    if (!record) return CUDA_ERROR_INVALID_HANDLE;
    if (record->stage != ContinuationStage::kCommandPending &&
        record->stage != ContinuationStage::kHostPending)
        return CUDA_ERROR_NOT_READY;
    if (outcome.kind != OutcomeKind::kReturned && outcome.kind != OutcomeKind::kUncertain &&
        !(record->stage == ContinuationStage::kHostPending &&
          outcome.kind == OutcomeKind::kDeferred))
        return CUDA_ERROR_INVALID_VALUE;
    try {
        if (record->stage == ContinuationStage::kCommandPending && record->binding &&
            record->binding->receipt && outcome.kind == OutcomeKind::kReturned &&
            !record->binding->owner->host_result(*record->binding->receipt))
            return CUDA_ERROR_NOT_READY;
        return state.apply_continuation_outcome(*record, std::move(outcome));
    } catch (...) {
        state.fail_continuation(*record, CUDA_ERROR_UNKNOWN);
        return CUDA_ERROR_UNKNOWN;
    }
}

NcclActorContinuationInspection NcclActor::inspect_continuation_source(
    std::uint64_t id, NcclActorGrant grant, std::uint64_t source) const noexcept {
    auto& state = *implementation_;
    const auto gate = state.gate();
    if (gate != CUDA_SUCCESS) return {gate, {}, {}};
    const Implementation::Scope scope(state);
    const auto record = state.find_continuation(id);
    if (!record || grant != state.continuation_grant(*record)) return {};
    if (state.uncertain) return {CUDA_ERROR_UNKNOWN, {}, {}};
    const NcclActorContinuationBinding* binding{};
    if ((record->stage == ContinuationStage::kCommandPending ||
         record->stage == ContinuationStage::kWaitingRetirement) &&
        record->binding && record->binding->session == source) {
        binding = &*record->binding;
    } else if (record->stage == ContinuationStage::kPrepared ||
               record->stage == ContinuationStage::kBlocked) {
        const auto found =
            std::ranges::find(record->dependencies, source, &NcclActorContinuationBinding::session);
        if (found != record->dependencies.end()) binding = &*found;
    }
    if (!binding) return {};
    return {CUDA_SUCCESS, *binding,
            record->stage == ContinuationStage::kCommandPending
                ? binding->owner->inspect_frontier_dependency(binding->frontier)
                : binding->owner->inspect_frontier_retirement(binding->frontier)};
}

NcclActorContinuationReply NcclActor::continuation_reply(std::uint64_t id) const noexcept {
    auto& state = *implementation_;
    const auto gate = state.gate();
    if (gate != CUDA_SUCCESS) return {.status = gate};
    const Implementation::Scope scope(state);
    const auto record = state.find_continuation(id);
    if (!record) return {.status = CUDA_ERROR_INVALID_HANDLE};
    try {
        return {record->status, record->reply.has_value(), record->retired, record->reply,
                record->host_reply};
    } catch (const std::bad_alloc&) {
        return {.status = CUDA_ERROR_OUT_OF_MEMORY};
    } catch (...) {
        return {.status = CUDA_ERROR_UNKNOWN};
    }
}

CUresult NcclActor::release_continuation_reply(std::uint64_t id) noexcept {
    auto& state = *implementation_;
    const auto gate = state.gate();
    if (gate != CUDA_SUCCESS) return gate;
    const Implementation::Scope scope(state);
    const auto record = state.find_continuation(id);
    if (!record || record->released) return CUDA_ERROR_INVALID_HANDLE;
    if (!record->reply) return CUDA_ERROR_NOT_READY;
    record->released = true;
    state.audit_action(*record, AuditEvent::kRelease, AuditAction::kRelease);
    state.refresh_continuations();
    return CUDA_SUCCESS;
}

CUresult NcclActor::cancel_continuation(std::uint64_t id) noexcept {
    auto& state = *implementation_;
    const auto gate = state.gate();
    if (gate != CUDA_SUCCESS) return gate;
    const Implementation::Scope scope(state);
    const auto record = state.find_continuation(id);
    if (!record) return CUDA_ERROR_INVALID_HANDLE;
    if (record->stage != ContinuationStage::kPrepared &&
        record->stage != ContinuationStage::kBlocked)
        return CUDA_ERROR_NOT_READY;
    if (record->binding || record->pending_lifetime) return CUDA_ERROR_NOT_READY;
    if (record->parent && !record->scope.flags) record->parent->consumed = true;
    record->released = record->retired = true;
    record->stage = ContinuationStage::kCancelled;
    state.audit_action(*record, AuditEvent::kRetirement, AuditAction::kCancel);
    state.refresh_continuations();
    return CUDA_SUCCESS;
}

NcclActorCleanupRetirementPreparation NcclActor::begin_cleanup_retirement(
    NcclActorPeer peer, std::uint64_t connection, std::shared_ptr<const void> helper,
    std::uint64_t closing_continuation) noexcept {
    auto& state = *implementation_;
    const auto gate = state.gate();
    if (gate != CUDA_SUCCESS) return {gate, {}};
    const Implementation::Scope scope(state);
    if (!peer.identity || !connection || !helper) return {CUDA_ERROR_INVALID_VALUE, {}};
    if (state.uncertain) return {CUDA_ERROR_UNKNOWN, {}};
    try {
        if (std::ranges::any_of(state.cleanup_retirements, [&](const auto& item) {
                return item.connection == connection || item.helper == helper;
            }))
            return {CUDA_ERROR_INVALID_VALUE, {}};
        CleanupRetirement retirement;
        retirement.peer = peer;
        retirement.connection = connection;
        retirement.helper = std::move(helper);
        if (closing_continuation) {
            retirement.closing = state.find_continuation(closing_continuation);
            const auto& close = retirement.closing;
            if (!close || close->peer.identity != peer.identity ||
                close->peer.priority != peer.priority || close->target.connection != connection ||
                close->target.helper_owner != retirement.helper || close->scope.flags ||
                close->target.operation != wire::ExecutionOpcode::kClose)
                return {CUDA_ERROR_INVALID_HANDLE, {}};
            if (close->batch->uncertain) return {CUDA_ERROR_UNKNOWN, {}};
            if (close->binding || close->pending_lifetime || close->pending_sequence ||
                close->reply || close->retired ||
                (close->stage != ContinuationStage::kPrepared &&
                 close->stage != ContinuationStage::kBlocked))
                return {CUDA_ERROR_NOT_READY, {}};
            const auto ready = state.continuation_ready(*close);
            if (ready != CUDA_SUCCESS) return {ready, {}};
        }
        for (const auto& record : state.continuations) {
            if (record->target.connection != connection &&
                record->target.helper_owner != retirement.helper)
                continue;
            if (record->peer.identity != peer.identity || record->peer.priority != peer.priority ||
                record->target.connection != connection ||
                record->target.helper_owner != retirement.helper)
                return {CUDA_ERROR_INVALID_HANDLE, {}};
            if (record == retirement.closing) continue;
            if (record->stage == ContinuationStage::kUncertain) return {CUDA_ERROR_UNKNOWN, {}};
            if (!record->released || !record->retired) return {CUDA_ERROR_NOT_READY, {}};
        }
        std::size_t parent_count{};
        for (const auto& parent : state.cleanup_parents) {
            if (parent->connection != connection && parent->helper_owner != retirement.helper)
                continue;
            if (parent->peer.identity != peer.identity || parent->peer.priority != peer.priority ||
                parent->connection != connection || parent->helper_owner != retirement.helper)
                return {CUDA_ERROR_INVALID_HANDLE, {}};
            if (parent->uncertain || parent->batch->uncertain) return {CUDA_ERROR_UNKNOWN, {}};
            if (!parent->bound || parent->consumed || parent->active_continuation)
                return {CUDA_ERROR_NOT_READY, {}};
            ++parent_count;
        }
        if (!parent_count) return {CUDA_ERROR_INVALID_HANDLE, {}};
        if (state.cleanup_retirements.size() == state.config.maximum_cleanup_parents)
            return {CUDA_ERROR_OUT_OF_MEMORY, {}};
        // Each parent belongs to only one sealed owner. Retained capacity is
        // linear in the bounded parent pool, not one full pool per helper.
        retirement.parents.reserve(parent_count);
        for (const auto& parent : state.cleanup_parents)
            if (parent->connection == connection && parent->helper_owner == retirement.helper)
                retirement.parents.push_back(parent);
        retirement.identity = cleanup_retirement_identity();
        if (!retirement.identity) return {CUDA_ERROR_OUT_OF_MEMORY, {}};
        const NcclActorCleanupRetirement ticket{retirement.identity, retirement.parents.size()};
        state.cleanup_retirements.push_back(std::move(retirement));
        NcclActorAuditRecord out;
        out.event = AuditEvent::kCleanup;
        out.subject = AuditSubject::kCleanupTicket;
        out.action = AuditAction::kSeal;
        out.id = out.ticket = ticket.identity;
        out.peer = peer.identity;
        out.priority = peer.priority;
        out.connection = connection;
        out.related = closing_continuation;
        out.count = ticket.parents;
        state.emit(out);
        for (std::size_t index = 0; index != state.cleanup_retirements.back().parents.size();
             ++index) {
            auto parent = state.audit_parent(*state.cleanup_retirements.back().parents[index]);
            parent.event = AuditEvent::kParent;
            parent.action = AuditAction::kSeal;
            parent.ticket = ticket.identity;
            parent.index = index;
            parent.count = ticket.parents;
            state.emit(parent);
        }
        return {CUDA_SUCCESS, ticket};
    } catch (const std::bad_alloc&) {
        return {CUDA_ERROR_OUT_OF_MEMORY, {}};
    } catch (...) {
        return {CUDA_ERROR_UNKNOWN, {}};
    }
}

CUresult NcclActor::complete_cleanup_retirement(NcclActorCleanupRetirement ticket) noexcept {
    auto& state = *implementation_;
    const auto gate = state.gate();
    if (gate != CUDA_SUCCESS) return gate;
    const Implementation::Scope scope(state);
    const auto found =
        std::ranges::find(state.cleanup_retirements, ticket.identity, &CleanupRetirement::identity);
    if (!ticket.identity || !ticket.parents || found == state.cleanup_retirements.end() ||
        found->parents.size() != ticket.parents)
        return CUDA_ERROR_INVALID_VALUE;
    auto& retirement = *found;
    NcclActorAuditRecord audit;
    audit.event = AuditEvent::kCleanup;
    audit.subject = AuditSubject::kCleanupTicket;
    audit.action = AuditAction::kProve;
    audit.id = audit.ticket = ticket.identity;
    audit.peer = retirement.peer.identity;
    audit.priority = retirement.peer.priority;
    audit.connection = retirement.connection;
    audit.related = retirement.closing ? retirement.closing->id : 0;
    audit.count = ticket.parents;
    const auto run = [&]() -> CUresult {
        if (retirement.failure != CUDA_SUCCESS) return retirement.failure;
        if (state.uncertain) return CUDA_ERROR_UNKNOWN;
        if (retirement.closing) {
            const auto& close = *retirement.closing;
            const bool terminal = close.stage == ContinuationStage::kComplete ||
                                  (close.stage == ContinuationStage::kCancelled && close.released);
            if (!close.retired || !terminal) return CUDA_ERROR_NOT_READY;
        }
        if (!state.resolver.cleanup_owner_retired) return CUDA_ERROR_NOT_SUPPORTED;
        for (const auto& parent : retirement.parents)
            if (std::ranges::find(state.cleanup_parents, parent) == state.cleanup_parents.end() ||
                parent->peer.identity != retirement.peer.identity ||
                parent->peer.priority != retirement.peer.priority ||
                parent->connection != retirement.connection ||
                parent->helper_owner != retirement.helper || !parent->bound || parent->consumed ||
                parent->uncertain || parent->batch->uncertain || parent->active_continuation)
                return CUDA_ERROR_UNKNOWN;
        CUresult proof{CUDA_ERROR_UNKNOWN};
        try {
            proof = state.resolver.cleanup_owner_retired(retirement.peer.identity,
                                                         retirement.connection, retirement.helper);
        } catch (...) {
            proof = CUDA_ERROR_UNKNOWN;
        }
        if (proof == CUDA_ERROR_NOT_READY) return proof;
        if (proof != CUDA_SUCCESS) {
            retirement.failure = proof;
            for (auto& parent : retirement.parents) {
                parent->uncertain = true;
                parent->batch->uncertain = true;
            }
            return proof;
        }
        for (auto& parent : retirement.parents) {
            parent->consumed = true;
            auto out = state.audit_parent(*parent);
            out.event = AuditEvent::kParent;
            out.action = AuditAction::kRetire;
            out.ticket = ticket.identity;
            state.emit(out);
        }
        state.cleanup_retirements.erase(found);
        state.refresh_continuations();
        return CUDA_SUCCESS;
    };
    audit.status = run();
    audit.retired = audit.status == CUDA_SUCCESS;
    state.emit(audit);
    return static_cast<CUresult>(audit.status);
}

CUresult NcclActor::observe() noexcept {
    auto& state = *implementation_;
    const auto gate = state.gate();
    if (gate != CUDA_SUCCESS) return gate;
    const Implementation::Scope scope(state);
    try {
        return state.refresh();
    } catch (const std::bad_alloc&) {
        state.uncertain = true;
        return CUDA_ERROR_OUT_OF_MEMORY;
    } catch (...) {
        state.uncertain = true;
        return CUDA_ERROR_UNKNOWN;
    }
}

NcclActorSnapshot NcclActor::snapshot() const noexcept {
    auto& state = *implementation_;
    const auto gate = state.gate();
    if (gate != CUDA_SUCCESS) return {gate};
    const Implementation::Scope scope(state);
    try {
        NcclActorSnapshot result;
        result.reservations.reserve(state.batches.size());
        for (const auto& batch : state.batches) {
            NcclActorReservation item;
            item.reservation = batch->identity;
            item.generation = batch->generation;
            item.peer = batch->peer;
            item.priority = batch->priority;
            item.has_open_group = batch->open_group;
            item.has_unknown_state = batch->uncertain || state.uncertain;
            item.has_physical_work = batch->open_group;
            for (const auto& record : state.records) {
                if (record->batch != batch) continue;
                NcclActorRequestSnapshot request;
                request.request = record->id;
                request.grant = {batch->identity, batch->generation, record->cutoff};
                request.stage = record->stage;
                const auto ready = state.ready(*record);
                if (ready != CUDA_SUCCESS && ready != CUDA_ERROR_NOT_READY) return {ready};
                request.is_ready = ready == CUDA_SUCCESS;
                request.has_native_call = record->operation != 0;
                request.reply_ready = record->projected;
                request.dependencies.reserve(record->sources.size());
                for (const auto& source : record->sources) {
                    bool available = true;
                    if (source.accepted && !source.begun)
                        available =
                            source.owner->enqueue_turn_ready(source.turn.turn) == CUDA_SUCCESS;
                    request.dependencies.push_back({source.identity, source.streams, available});
                }
                item.has_physical_work =
                    item.has_physical_work ||
                    (record->operation &&
                     (!record->observed || !record->observed->physical_complete)) ||
                    std::ranges::any_of(record->sources, [](const auto& source) {
                        return source.accepted && !source.finished;
                    });
                item.requests.push_back(std::move(request));
            }
            for (const auto& record : state.continuations) {
                if (record->batch != batch) continue;
                NcclActorContinuationSnapshot continuation;
                continuation.continuation = record->id;
                continuation.grant = state.continuation_grant(*record);
                continuation.scope = record->scope;
                continuation.connection = record->target.connection;
                continuation.wire_sequence = record->target.wire_sequence;
                continuation.operation = record->target.operation;
                continuation.stage = record->stage;
                continuation.reply_ready = record->reply.has_value();
                continuation.is_retired = record->retired;
                continuation.is_ready = state.continuation_ready(*record) == CUDA_SUCCESS;
                if (record->stage == ContinuationStage::kPrepared ||
                    record->stage == ContinuationStage::kBlocked) {
                    continuation.dependencies.reserve(record->dependencies.size());
                    for (const auto& dependency : record->dependencies) {
                        item.has_unknown_state =
                            item.has_unknown_state || dependency.owner->snapshot().is_quarantined;
                        if (!dependency.frontier.retirement().is_retired)
                            continuation.dependencies.push_back(dependency.session);
                    }
                } else if ((record->stage == ContinuationStage::kCommandPending ||
                            record->stage == ContinuationStage::kWaitingRetirement) &&
                           record->binding && !record->binding->frontier.retirement().is_retired) {
                    continuation.dependencies.push_back(record->binding->session);
                }
                if (record->binding) {
                    item.has_unknown_state =
                        item.has_unknown_state || record->binding->owner->snapshot().is_quarantined;
                }
                // Source/HostOwner policy peers account actual work, including
                // paused kernels and private tails. Metadata is not a second
                // unpausable physical owner of the same ordinary operation.
                item.has_unknown_state =
                    item.has_unknown_state || record->stage == ContinuationStage::kUncertain;
                item.continuations.push_back(std::move(continuation));
            }
            for (const auto& parent : state.cleanup_parents)
                if (parent->batch == batch) item.has_unknown_state |= parent->uncertain;
            result.reservations.push_back(std::move(item));
        }
        result.status = CUDA_SUCCESS;
        return result;
    } catch (const std::bad_alloc&) {
        return {CUDA_ERROR_OUT_OF_MEMORY};
    } catch (...) {
        return {CUDA_ERROR_UNKNOWN};
    }
}

NcclActorReply NcclActor::reply(std::uint64_t id) const noexcept {
    auto& state = *implementation_;
    const auto gate = state.gate();
    if (gate != CUDA_SUCCESS) return {gate};
    const Implementation::Scope scope(state);
    const auto record = state.find(id);
    if (!record) return {CUDA_ERROR_INVALID_HANDLE};
    try {
        return {record->carrier, record->projected, record->envelope, record->observed};
    } catch (const std::bad_alloc&) {
        return {CUDA_ERROR_OUT_OF_MEMORY};
    } catch (...) {
        return {CUDA_ERROR_UNKNOWN};
    }
}

CUresult NcclActor::release_reply(std::uint64_t id) noexcept {
    auto& state = *implementation_;
    const auto gate = state.gate();
    if (gate != CUDA_SUCCESS) return gate;
    const Implementation::Scope scope(state);
    const auto record = state.find(id);
    if (!record || record->released) return CUDA_ERROR_INVALID_HANDLE;
    if (!record->projected) return CUDA_ERROR_NOT_READY;
    record->released = true;
    state.audit_action(*record, AuditEvent::kRelease, AuditAction::kRelease);
    try {
        return state.refresh();
    } catch (const std::bad_alloc&) {
        state.uncertain = true;
        return CUDA_ERROR_OUT_OF_MEMORY;
    } catch (...) {
        state.uncertain = true;
        return CUDA_ERROR_UNKNOWN;
    }
}

CUresult NcclActor::disconnect(std::uint64_t peer) noexcept {
    auto& state = *implementation_;
    const auto gate = state.gate();
    if (gate != CUDA_SUCCESS) return gate;
    if (!peer) return CUDA_ERROR_INVALID_VALUE;
    const Implementation::Scope scope(state);
    if (std::ranges::find(state.disconnected_peers, peer) == state.disconnected_peers.end()) {
        if (state.disconnected_peers.size() == state.config.maximum_callers)
            return CUDA_ERROR_OUT_OF_MEMORY;
        state.disconnected_peers.push_back(peer);  // Reserved at construction.
        NcclActorAuditRecord out;
        out.event = AuditEvent::kDisconnect;
        out.peer = peer;
        state.emit(out);
    }
    for (auto& caller : state.callers)
        if (caller.peer.identity == peer) caller.disconnected = true;
    // Disconnect is CPU admission closure, not native cancellation or physical
    // proof. Existing exact-grant continuations and all retained owners remain.
    return CUDA_SUCCESS;
}
CUresult NcclActor::close() noexcept {
    auto& state = *implementation_;
    const auto gate = state.gate();
    if (gate != CUDA_SUCCESS) return gate;
    const Implementation::Scope scope(state);
    if (!state.closed) {
        NcclActorAuditRecord out;
        out.event = AuditEvent::kClose;
        state.emit(out);
    }
    state.closed = true;
    return CUDA_SUCCESS;
}
CUresult NcclActor::shutdown() noexcept {
    auto& state = *implementation_;
    const auto gate = state.gate();
    if (gate != CUDA_SUCCESS) return gate;
    if (state.stopped) return CUDA_SUCCESS;
    if (!state.closed || state.uncertain || !state.records.empty() || !state.batches.empty() ||
        !state.communicators.empty() || !state.continuations.empty() ||
        !state.cleanup_parents.empty() || !state.cleanup_retirements.empty())
        return CUDA_ERROR_NOT_READY;
    const Implementation::Scope scope(state);
    if (!state.coordinator_stopped) {
        auto status = state.coordinator.close();
        if (status == CUDA_SUCCESS) status = state.coordinator.shutdown();
        if (status != CUDA_SUCCESS) return status;
        state.coordinator_stopped = true;
    }
    state.domain = {};
    auto status = state.primary.close();
    if (status == CUDA_SUCCESS) status = state.primary.shutdown();
    if (status == CUDA_SUCCESS) {
        state.stopped = true;
        NcclActorAuditRecord out;
        out.event = AuditEvent::kStopped;
        out.retired = true;
        state.emit(out);
    }
    return status;
}
}  // namespace glimmer::app
