#include "internal/cuda_nccl_coordinator.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>
#include <unistd.h>

namespace glimmer::backend::cuda {
namespace {
using Kind = CudaNcclNativeOperation;
using Phase = CudaNcclCoordinatorPhase;
thread_local std::shared_ptr<const void> coordinator_thread;
std::weak_ptr<const void> owner_identity() {
    if (!coordinator_thread) coordinator_thread = std::make_shared<int>(0);
    return coordinator_thread;
}
bool initializes(Kind kind) {
    return kind == Kind::kInitRank || kind == Kind::kInitRankConfig;
}
bool collective(Kind kind) {
    return kind == Kind::kAllGather || kind == Kind::kAllReduce || kind == Kind::kBcast;
}
bool destructive(Kind kind) {
    return kind == Kind::kDestroy || kind == Kind::kAbort;
}
bool group_member(Kind kind) {
    return initializes(kind) || collective(kind) || destructive(kind);
}
bool same_domain(const CudaNcclCoordinatorDomain& first, const CudaNcclCoordinatorDomain& second) {
    return first.identity == second.identity && first.device_ordinal == second.device_ordinal &&
           first.context.id == second.context.id && first.context.native == second.context.native &&
           !first.context.lifetime.owner_before(second.context.lifetime) &&
           !second.context.lifetime.owner_before(first.context.lifetime);
}
CUresult prove_context(CudaNativeExecutionLaneDispatch dispatch, CUcontext expected) {
    CUcontext current{};
    const auto status = dispatch.context_get_current(&current);
    return status != CUDA_SUCCESS
               ? status
               : (current == expected ? CUDA_SUCCESS : CUDA_ERROR_INVALID_CONTEXT);
}
bool valid_context(const CudaNcclCoordinatorDomain& domain, bool nullable = false) {
    if (!domain.context.native)
        return nullable && !domain.identity && domain.device_ordinal == -1 && !domain.context.id &&
               !domain.context.lifetime;
    return domain.identity && domain.device_ordinal >= 0 && domain.context.id &&
           domain.context.lifetime;
}
CudaNcclContextIdentity context_identity(const CudaNcclCoordinatorDomain& domain,
                                         CudaNcclContextKind kind) {
    return domain.context.native ? CudaNcclContextIdentity{kind, domain.identity, domain.context.id,
                                                           domain.device_ordinal}
                                 : CudaNcclContextIdentity{};
}
struct Communicator;
struct Operation;
struct GroupCredit {};
enum class OperationCredit { kNormal, kGroupEnd, kProgress };
struct NativeOwner {
    CudaNcclCoordinatorDomain domain;
    std::shared_ptr<CudaNcclNativeCall> initialization;
};
struct Receipt {
    std::optional<CudaNcclNativeObservation> output;
    std::vector<std::optional<CudaNcclNativeObservation>> members;
    bool recorded{};
    std::optional<CudaNcclContextTransition> context_transition;
};
struct Communicator {
    std::uint64_t id{}, peer{}, group_owner{}, active_operation{}, pending_batch{};
    CudaNcclCoordinatorDomain domain;
    std::shared_ptr<CudaNcclNativeCall> initialization;
    std::shared_ptr<const NativeOwner> lease;
    ncclComm_t native{};
    bool published{}, ready{}, sealed{}, terminal{}, uncertain{};
};
struct Operation {
    std::uint64_t id{}, reserved{}, caller_index{};
    CudaNcclCoordinatorRequest request;
    std::shared_ptr<CudaNcclNativeCall> packet;
    std::shared_ptr<Communicator> communicator;
    std::shared_ptr<Receipt> receipt;
    CudaNativeExecutionTicket ticket;
    CudaNcclCoordinatorObservation observation;
    std::vector<std::shared_ptr<Operation>> members;
    std::vector<std::shared_ptr<Communicator>> participants;
    std::vector<int> devices;
    std::vector<CudaExecutionFrontier> tails;
    std::vector<CudaNcclCoordinatorPublication> publications;
    std::shared_ptr<GroupCredit> group_credit;
    OperationCredit credit{OperationCredit::kNormal};
    bool has_ticket{}, continuation{}, released{}, grouped{}, outer_end{}, is_batch{};
    bool tails_bound{}, processed{}, finished{}, uncertain{}, has_destructive{};
    bool had_source_turns{};
    CUresult binding_status{CUDA_SUCCESS};
};
struct AcceptedDomain {
    std::uint64_t identity{}, context_id{};
    int device_ordinal{-1};
    CUcontext native{};
    std::weak_ptr<const void> lifetime;
    bool matches(const CudaNcclCoordinatorDomain& domain) const {
        return identity == domain.identity && context_id == domain.context.id &&
               device_ordinal == domain.device_ordinal && native == domain.context.native &&
               !lifetime.expired() && !lifetime.owner_before(domain.context.lifetime) &&
               !domain.context.lifetime.owner_before(lifetime);
    }
};
struct Caller {
    CudaNativeExecutionLaneKey key;
    std::uint64_t public_sequence{}, internal_sequence{}, active_operation{}, active_housekeeping{};
    std::size_t depth{};
    bool uncertain{};
    std::vector<std::shared_ptr<Operation>> members;
    std::shared_ptr<GroupCredit> group_credit;
    std::vector<AcceptedDomain> domains;
};
struct HousekeepingPayload {
    CudaNcclCoordinatorDomain domain;
    std::shared_ptr<const void> lifetime;
    std::function<CUresult()> invoke;
};
struct Housekeeping {
    std::uint64_t id{};
    std::size_t caller_index{};
    std::shared_ptr<const HousekeepingPayload> payload;
    CudaNativeExecutionTicket ticket;
    CudaNativeExecutionCallObservation observation;
    bool processed{}, releasable{};
};
struct CallerReservation {
    std::vector<Caller>& callers;
    bool provisional{};
    ~CallerReservation() {
        if (provisional) callers.pop_back();
    }
};
struct WorkerMember {
    Kind kind{};
    std::shared_ptr<CudaNcclNativeCall> packet;
    CudaNcclCoordinatorDomain domain;
};
struct WorkerPayload {
    std::shared_ptr<CudaNcclNativeCall> packet;
    std::vector<WorkerMember> members;
    CudaNcclCoordinatorDomain domain;
    std::optional<CudaNcclCoordinatorDomain> caller_context;
    std::vector<CudaNcclCoordinatorDomain> known_contexts;
    std::vector<CudaExecutionEnqueueWorker> source_turns;
    std::function<CUresult()> prepare, record;
    std::shared_ptr<Receipt> receipt;
    bool grouped{}, outer_end{}, batch{}, has_destructive{}, continuation{};
};
std::optional<CudaNcclContextIdentity> identify_context(const WorkerPayload& payload,
                                                        CUcontext native) {
    if (!native) return CudaNcclContextIdentity{};
    if (payload.caller_context && native == payload.caller_context->context.native)
        return context_identity(*payload.caller_context, CudaNcclContextKind::kCaller);
    for (const auto& domain : payload.known_contexts)
        if (native == domain.context.native)
            return context_identity(domain, CudaNcclContextKind::kDomain);
    return std::nullopt;
}
// Never select away from an unobserved/foreign context. Restoration is permitted
// only after an actual successful GetCurrent matches a retained gated owner.
CUresult restore_context(const WorkerPayload& payload, CudaNativeExecutionLaneDispatch driver,
                         CUcontext current) {
    if (!identify_context(payload, current)) return CUDA_ERROR_INVALID_CONTEXT;
    if (current == payload.domain.context.native) return CUDA_SUCCESS;
    const auto status = driver.context_set_current(payload.domain.context.native);
    const auto proof = prove_context(driver, payload.domain.context.native);
    return status != CUDA_SUCCESS ? status : proof;
}
CUresult observe_host_output(const WorkerPayload& payload) {
    if (!payload.packet->has_stable_output() || !payload.packet->requested_status())
        return CUDA_SUCCESS;
    payload.receipt->output = payload.packet->snapshot_after_host_return();
    return payload.receipt->output ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
}
CUresult invoke_packet(const WorkerPayload& payload, CudaNativeExecutionLaneDispatch driver) {
    if (!payload.caller_context) {
        const auto status = payload.packet->invoke();
        const auto output = observe_host_output(payload);
        const auto proof = prove_context(driver, payload.domain.context.native);
        return status != CUDA_SUCCESS ? status : (output != CUDA_SUCCESS ? output : proof);
    }
    const auto& entry = *payload.caller_context;
    const auto selected = driver.context_set_current(entry.context.native);
    CUcontext before{};
    const auto entry_proof = driver.context_get_current(&before);
    if (entry_proof != CUDA_SUCCESS) return selected != CUDA_SUCCESS ? selected : entry_proof;
    if (selected != CUDA_SUCCESS || before != entry.context.native) {
        const auto restored = restore_context(payload, driver, before);
        return selected != CUDA_SUCCESS   ? selected
               : restored != CUDA_SUCCESS ? restored
                                          : CUDA_ERROR_INVALID_CONTEXT;
    }
    const auto status = payload.packet->invoke();
    const auto output = observe_host_output(payload);
    CUcontext after{};
    const auto observed = driver.context_get_current(&after);
    if (observed != CUDA_SUCCESS) return status != CUDA_SUCCESS ? status : observed;
    const auto identity = identify_context(payload, after);
    if (!identity) return status != CUDA_SUCCESS ? status : CUDA_ERROR_INVALID_CONTEXT;
    payload.receipt->context_transition =
        CudaNcclContextTransition{context_identity(entry, CudaNcclContextKind::kCaller), *identity};
    const auto restored = restore_context(payload, driver, after);
    return status != CUDA_SUCCESS ? status : (output != CUDA_SUCCESS ? output : restored);
}
void append_participant(std::vector<std::shared_ptr<Communicator>>& list,
                        const std::shared_ptr<Communicator>& communicator) {
    if (communicator && std::ranges::find(list, communicator) == list.end())
        list.push_back(communicator);
}
void append_device(std::vector<int>& devices, int device) {
    if (std::ranges::find(devices, device) == devices.end()) devices.push_back(device);
}
}  // namespace

struct CudaNcclCoordinator::Implementation {
    const pid_t process{::getpid()};
    const std::weak_ptr<const void> owner{owner_identity()};
    const CudaNcclNativeDispatch native_dispatch;
    const CudaNativeExecutionLaneDispatch lane_dispatch;
    const CudaNcclCoordinatorLimits limits;
    CudaNativeExecutionLanes lanes;
    std::vector<Caller> callers;
    std::vector<std::shared_ptr<Communicator>> communicators;
    std::vector<std::shared_ptr<Operation>> operations;
    std::vector<std::shared_ptr<GroupCredit>> groups;
    std::vector<Housekeeping> housekeeping;
    std::uint64_t next_operation{1}, next_communicator{1}, next_housekeeping{1};
    bool closed{}, stopped{}, is_shutdown{}, active{};

    Implementation(CudaNcclNativeDispatch native, CudaNativeExecutionLaneDispatch driver,
                   CudaNcclCoordinatorLimits bounds)
        : native_dispatch(native),
          lane_dispatch(driver),
          limits(bounds),
          lanes(driver, bounds.lanes) {
        if (!bounds.maximum_communicators || !bounds.maximum_operations ||
            !bounds.maximum_group_members || !bounds.maximum_group_depth || !bounds.maximum_tails ||
            !bounds.maximum_retained_groups || !bounds.maximum_progress_operations ||
            !bounds.maximum_housekeeping_operations || !bounds.maximum_caller_domains)
            throw std::invalid_argument("zero NCCL coordinator limit");
        const auto maximum = std::numeric_limits<std::size_t>::max();
        if (bounds.maximum_operations > maximum - bounds.maximum_retained_groups ||
            bounds.maximum_operations + bounds.maximum_retained_groups >
                maximum - bounds.maximum_progress_operations ||
            bounds.maximum_operations + bounds.maximum_retained_groups +
                    bounds.maximum_progress_operations >
                maximum - bounds.maximum_housekeeping_operations ||
            bounds.maximum_caller_domains > maximum / bounds.lanes.maximum_lanes)
            throw std::invalid_argument("NCCL coordinator operation capacity overflow");
        callers.reserve(bounds.lanes.maximum_lanes);
        communicators.reserve(bounds.maximum_communicators);
        groups.reserve(bounds.maximum_retained_groups);
        operations.reserve(bounds.maximum_operations + bounds.maximum_retained_groups +
                           bounds.maximum_progress_operations);
        housekeeping.reserve(bounds.maximum_housekeeping_operations);
    }
    CUresult gate() const noexcept {
        if (::getpid() != process || !coordinator_thread || owner.lock() != coordinator_thread)
            return CUDA_ERROR_INVALID_CONTEXT;
        return active ? CUDA_ERROR_NOT_READY : CUDA_SUCCESS;
    }
    struct Scope {
        explicit Scope(Implementation& state) : state_(state) {
            state_.active = true;
        }
        ~Scope() {
            state_.active = false;
        }
        Scope(const Scope&) = delete;
        Scope& operator=(const Scope&) = delete;
        Implementation& state_;
    };
    std::shared_ptr<Operation> find_operation(std::uint64_t id) const {
        const auto found =
            std::ranges::find_if(operations, [id](const auto& item) { return item->id == id; });
        return found == operations.end() ? nullptr : *found;
    }
    std::shared_ptr<Communicator> find_communicator(std::uint64_t id) const {
        const auto found =
            std::ranges::find_if(communicators, [id](const auto& item) { return item->id == id; });
        return found == communicators.end() ? nullptr : *found;
    }
    void forget(const std::shared_ptr<Communicator>& communicator) {
        if (!communicator) return;
        communicator->terminal = true;
        communicator->sealed = true;
        std::erase(communicators, communicator);
    }
    void fail(const std::shared_ptr<Operation>& operation, CUresult status) {
        operation->uncertain = true;
        operation->observation.phase = Phase::kUncertain;
        if (operation->observation.carrier == CUDA_SUCCESS)
            operation->observation.carrier = status == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : status;
        for (const auto& communicator : operation->participants) communicator->uncertain = true;
        if (operation->communicator) operation->communicator->uncertain = true;
    }
    void finish(const std::shared_ptr<Operation>& operation) {
        operation->finished = true;
        operation->observation.phase = Phase::kComplete;
        operation->observation.physical_complete = true;
        operation->packet.reset();
        operation->request.arguments.resources_lifetime.reset();
        operation->request.prepare_enqueue = {};
        operation->request.record_enqueue = {};
        operation->request.source_turns.clear();
        operation->request.caller_context.reset();
        operation->communicator.reset();
        operation->participants.clear();
    }
    bool publish(const std::shared_ptr<Operation>& operation,
                 const std::optional<CudaNcclNativeObservation>& output, bool is_ready) {
        operation->observation.output = output;
        const auto& communicator = operation->communicator;
        if (!communicator) return true;
        if (!output || !output->has_output) {
            communicator->sealed = true;
            fail(operation, CUDA_ERROR_UNKNOWN);
            return false;
        }
        if (!output->communicator) {
            // The NULL seed is not proof that an error created no native owner
            // or that no asynchronous writer still retains this output cell.
            communicator->sealed = true;
            fail(operation, CUDA_ERROR_UNKNOWN);
            return false;
        }
        for (const auto& other : communicators) {
            if (other != communicator && other->published &&
                other->native == output->communicator) {
                other->uncertain = true;
                other->sealed = true;
                communicator->sealed = true;
                fail(operation, CUDA_ERROR_UNKNOWN);
                return false;
            }
        }
        communicator->native = output->communicator;
        communicator->published = true;
        communicator->ready = is_ready;
        if (operation->observation.original_status != 0 &&
            operation->observation.original_status != 7) {
            // An error-output address is retained evidence, not authority to
            // query/Abort an object that the native failure may have released.
            communicator->sealed = true;
            fail(operation, CUDA_ERROR_UNKNOWN);
            return false;
        }
        return true;
    }
    CUresult enqueue(const std::shared_ptr<Operation>& operation, bool continuation) {
        auto& caller = callers[operation->caller_index];
        if (caller.active_operation || caller.active_housekeeping ||
            caller.internal_sequence == UINT64_MAX)
            return CUDA_ERROR_NOT_READY;
        for (const auto& participant : operation->participants) {
            if (participant->active_operation) return CUDA_ERROR_NOT_READY;
            if (continuation &&
                (participant->sealed || participant->terminal || participant->uncertain))
                return CUDA_ERROR_INVALID_HANDLE;
        }
        auto receipt = std::make_shared<Receipt>();
        receipt->members.resize(operation->members.size());
        auto payload = std::make_shared<WorkerPayload>();
        payload->packet = operation->packet;
        payload->members.reserve(operation->members.size());
        for (const auto& member : operation->members)
            payload->members.push_back(
                {member->request.arguments.operation, member->packet, member->request.domain});
        payload->domain = operation->request.domain;
        payload->caller_context = operation->request.caller_context;
        if (!continuation) payload->source_turns = operation->request.source_turns;
        payload->known_contexts.push_back(operation->request.domain);
        for (const auto& member : operation->members) {
            payload->known_contexts.push_back(member->request.domain);
            if (member->request.caller_context && member->request.caller_context->context.native)
                payload->known_contexts.push_back(*member->request.caller_context);
        }
        payload->prepare = operation->request.prepare_enqueue;
        payload->record = operation->request.record_enqueue;
        payload->receipt = receipt;
        payload->grouped = operation->grouped;
        payload->outer_end = operation->outer_end;
        payload->batch = operation->is_batch;
        payload->has_destructive = operation->has_destructive;
        payload->continuation = continuation;
        const std::shared_ptr<const WorkerPayload> immutable = std::move(payload);
        const auto driver = lane_dispatch;
        const auto expected = operation->request.domain.context.native;
        auto invoke = [immutable, driver, expected]() -> CUresult {
            const auto& packet = immutable->packet;
            const auto& members = immutable->members;
            const auto& receipt = immutable->receipt;
            auto proof = prove_context(driver, expected);
            if (proof != CUDA_SUCCESS) return proof;
            if (immutable->continuation) {
                const auto status = immutable->record();
                proof = prove_context(driver, expected);
                if (status != CUDA_SUCCESS) return status;
                if (proof != CUDA_SUCCESS) return proof;
                receipt->recorded = true;
                return CUDA_SUCCESS;
            }
            for (const auto& turn : immutable->source_turns) {
                const auto status = turn.begin();
                if (status != CUDA_SUCCESS) return status;
            }
            if (immutable->prepare) {
                const auto status = immutable->prepare();
                proof = prove_context(driver, expected);
                if (status != CUDA_SUCCESS) return status;
                if (proof != CUDA_SUCCESS) return proof;
            }
            const auto carrier = invoke_packet(*immutable, driver);
            if (carrier != CUDA_SUCCESS) return carrier;
            const auto original = packet->requested_status();
            if (!original) return CUDA_ERROR_UNKNOWN;
            if (!immutable->grouped && !packet->has_stable_output())
                receipt->output = packet->snapshot_after_quiescence();
            if (immutable->outer_end && (*original == 0 || *original == 7)) {
                for (std::size_t index = 0; index != members.size(); ++index)
                    if (initializes(members[index].kind))
                        receipt->members[index] =
                            members[index].packet->snapshot_after_quiescence();
            }
            const bool member_errors = std::ranges::any_of(members, [](const auto& member) {
                const auto status = member.packet->requested_status();
                return !status || (*status != 0 && *status != 7);
            });
            if (immutable->batch && *original == 0 && !member_errors &&
                !immutable->has_destructive) {
                const auto status = immutable->record();
                proof = prove_context(driver, expected);
                if (status != CUDA_SUCCESS) return status;
                if (proof != CUDA_SUCCESS) return proof;
                receipt->recorded = true;
            }
            return CUDA_SUCCESS;
        };
        CudaNativeExecutionCall call;
        call.sequence = caller.internal_sequence + 1;
        call.device_ordinal = operation->request.domain.device_ordinal;
        call.context = operation->request.domain.context;
        call.arguments_lifetime = immutable;
        call.invoke = std::move(invoke);
        call.device_ordinals = operation->devices;
        auto accepted = lanes.submit(caller.key, std::move(call));
        if (accepted.status != CUDA_SUCCESS) return accepted.status;
        ++caller.internal_sequence;
        caller.active_operation = operation->id;
        operation->receipt = std::move(receipt);
        operation->ticket = std::move(accepted.ticket);
        operation->has_ticket = true;
        operation->continuation = continuation;
        if (!continuation) {
            for (const auto& turn : operation->request.source_turns) {
                const auto status = turn.bind_ticket(operation->ticket);
                if (operation->binding_status == CUDA_SUCCESS && status != CUDA_SUCCESS)
                    operation->binding_status = status;
            }
        }
        for (const auto& communicator : operation->participants)
            communicator->active_operation = operation->id;
        return CUDA_SUCCESS;
    }
    void complete_group(const std::shared_ptr<Operation>& operation, std::int32_t status) {
        auto& caller = callers[operation->caller_index];
        if (status != 0 && status != 7) {
            caller.uncertain = true;
            fail(operation, CUDA_ERROR_UNKNOWN);
            return;
        }
        --caller.depth;
        if (!operation->outer_end) {
            finish(operation);
            return;
        }
        caller.members.clear();
        caller.group_credit.reset();
        bool invalid{};
        std::size_t publication{};
        for (std::size_t index = 0; index != operation->members.size(); ++index) {
            const auto& member = operation->members[index];
            const auto member_status = member->observation.original_status;
            invalid = invalid || !member_status || (*member_status != 0 && *member_status != 7);
            if (initializes(member->request.arguments.operation)) {
                operation->publications[publication++] = {member->id, member->reserved,
                                                          operation->receipt->members[index]};
                invalid =
                    !publish(member, operation->receipt->members[index], status == 0) || invalid;
            }
        }
        operation->observation.publications.swap(operation->publications);
        for (const auto& communicator : operation->participants) communicator->group_owner = 0;
        if (invalid || (operation->has_destructive && status == 7)) {
            fail(operation, CUDA_ERROR_UNKNOWN);
            return;
        }
        if (status == 7) {
            operation->observation.phase = Phase::kWaitingAsync;
            for (const auto& communicator : operation->participants) {
                communicator->pending_batch = operation->id;
                communicator->ready = false;
            }
            return;
        }
        for (const auto& member : operation->members)
            if (destructive(member->request.arguments.operation)) forget(member->communicator);
        if (operation->is_batch) {
            operation->observation.enqueue_complete = operation->receipt->recorded;
            operation->observation.phase = Phase::kWaitingTails;
        } else {
            for (const auto& member : operation->members) finish(member);
            operation->members.clear();
            finish(operation);
        }
    }
    void consume(const std::shared_ptr<Operation>& operation) {
        const auto returned = operation->ticket.observe();
        if (returned.phase == CudaNativeExecutionCallPhase::kPending ||
            returned.phase == CudaNativeExecutionCallPhase::kEntered)
            return;
        auto& caller = callers[operation->caller_index];
        if (caller.active_operation == operation->id) caller.active_operation = 0;
        for (const auto& communicator : operation->participants)
            if (communicator->active_operation == operation->id) communicator->active_operation = 0;
        operation->has_ticket = false;
        if (!operation->continuation) {
            operation->observation.original_status = operation->packet->requested_status();
            operation->observation.reply_ready = true;
            operation->observation.carrier = returned.status;
            operation->observation.context_transition = operation->receipt->context_transition;
            if (operation->request.arguments.output_contract ==
                CudaNcclNativeOutputContract::kPinnedHostStable)
                operation->observation.output = operation->receipt->output;
        }
        if (operation->binding_status != CUDA_SUCCESS) {
            fail(operation, operation->binding_status);
            caller.uncertain = true;
            return;
        }
        if (returned.phase != CudaNativeExecutionCallPhase::kReturned ||
            returned.status != CUDA_SUCCESS) {
            if (returned.phase == CudaNativeExecutionCallPhase::kCancelled) {
                const auto released = lanes.release(operation->ticket);
                if (released != CUDA_SUCCESS) {
                    fail(operation, released);
                    return;
                }
            }
            fail(operation, returned.status);
            if (caller.depth || operation->request.arguments.operation == Kind::kGroupStart)
                caller.uncertain = true;
            return;
        }
        const auto release = lanes.release(operation->ticket);
        if (release != CUDA_SUCCESS) {
            fail(operation, release);
            return;
        }
        if (operation->continuation) {
            operation->observation.enqueue_complete = operation->receipt->recorded;
            operation->observation.phase = Phase::kWaitingTails;
            for (const auto& communicator : operation->participants)
                if (communicator->pending_batch == operation->id) communicator->pending_batch = 0;
            return;
        }
        const auto original = operation->observation.original_status;
        if (!original) {
            fail(operation, CUDA_ERROR_UNKNOWN);
            return;
        }
        const auto kind = operation->request.arguments.operation;
        operation->processed = true;
        operation->observation.phase = Phase::kReturned;
        if (!operation->grouped) operation->observation.output = operation->receipt->output;
        if (kind == Kind::kGroupStart) {
            if (*original == 0) {
                ++caller.depth;
                finish(operation);
            } else {
                caller.uncertain = true;
                fail(operation, CUDA_ERROR_UNKNOWN);
            }
            return;
        }
        if (kind == Kind::kGroupEnd && caller.depth) {
            complete_group(operation, *original);
            return;
        }
        if (operation->grouped) {
            if (initializes(kind) && operation->request.arguments.output_contract ==
                                         CudaNcclNativeOutputContract::kPinnedHostStable)
                (void)publish(operation, operation->receipt->output, false);
            return;
        }
        if (initializes(kind)) {
            if (!publish(operation, operation->receipt->output, *original == 0)) return;
            if (*original == 7 && operation->communicator) {
                operation->communicator->pending_batch = operation->id;
                operation->observation.phase = Phase::kWaitingAsync;
            } else if (*original != 0 && operation->communicator &&
                       operation->communicator->published) {
                fail(operation, CUDA_ERROR_UNKNOWN);
            } else
                finish(operation);
            return;
        }
        if (kind == Kind::kGetAsyncError && operation->communicator && *original == 0) {
            const auto& output = operation->receipt->output;
            if (output && output->has_output && output->async_error == 0 &&
                !operation->communicator->sealed)
                operation->communicator->ready = true;
        }
        if (destructive(kind)) {
            if (*original == 0) {
                forget(operation->communicator);
                finish(operation);
            } else
                fail(operation, CUDA_ERROR_UNKNOWN);
            return;
        }
        if (operation->is_batch) {
            if (*original == 0 && operation->receipt->recorded) {
                operation->observation.enqueue_complete = true;
                operation->observation.phase = Phase::kWaitingTails;
            } else if (*original == 7 && operation->communicator) {
                operation->communicator->ready = false;
                operation->communicator->pending_batch = operation->id;
                operation->observation.phase = Phase::kWaitingAsync;
            } else
                fail(operation, CUDA_ERROR_UNKNOWN);
            return;
        }
        if (*original == 7)
            fail(operation, CUDA_ERROR_UNKNOWN);
        else
            finish(operation);
    }
    void consume_housekeeping(Housekeeping& job) {
        if (job.processed) return;
        const auto observed = job.ticket.observe();
        if (observed.phase == CudaNativeExecutionCallPhase::kPending ||
            observed.phase == CudaNativeExecutionCallPhase::kEntered)
            return;
        job.observation = observed;
        job.processed = true;
        auto& caller = callers[job.caller_index];
        if (caller.active_housekeeping == job.id) caller.active_housekeeping = 0;
        if ((observed.phase == CudaNativeExecutionCallPhase::kReturned &&
             observed.status == CUDA_SUCCESS) ||
            (observed.phase == CudaNativeExecutionCallPhase::kCancelled && !observed.was_invoked)) {
            job.releasable = true;
        } else {
            caller.uncertain = true;
        }
    }
    bool quiescent() const {
        return housekeeping.empty() && communicators.empty() &&
               std::ranges::all_of(callers,
                                   [](const auto& caller) {
                                       return !caller.depth && !caller.active_operation &&
                                              !caller.active_housekeeping && !caller.uncertain &&
                                              caller.members.empty();
                                   }) &&
               std::ranges::all_of(operations,
                                   [](const auto& operation) { return operation->finished; });
    }
    bool continuation_ready(const Operation& operation) const {
        if (operation.uncertain || operation.finished || operation.has_ticket ||
            !operation.is_batch || operation.observation.phase != Phase::kWaitingAsync ||
            operation.participants.empty())
            return false;
        const auto& caller = callers[operation.caller_index];
        return !caller.active_operation && !caller.active_housekeeping && !caller.uncertain &&
               std::ranges::all_of(operation.participants, [](const auto& item) {
                   return item->ready && !item->sealed && !item->uncertain && !item->terminal &&
                          !item->active_operation;
               });
    }
};

CudaNcclCoordinator::CudaNcclCoordinator(CudaNcclNativeDispatch native,
                                         CudaNativeExecutionLaneDispatch driver,
                                         CudaNcclCoordinatorLimits limits)
    : implementation_(std::make_unique<Implementation>(native, driver, limits)) {}
CudaNcclCoordinator::~CudaNcclCoordinator() {
    if (!implementation_) return;
    if (::getpid() != implementation_->process) {
        (void)implementation_.release();
        return;
    }
    if (implementation_->is_shutdown) return;
    if (implementation_->gate() != CUDA_SUCCESS || !implementation_->quiescent()) {
        (void)implementation_.release();
        return;
    }
    if (implementation_->lanes.request_stop() != CUDA_SUCCESS ||
        implementation_->lanes.shutdown() != CUDA_SUCCESS)
        (void)implementation_.release();
}

CudaNcclCoordinatorAcceptance CudaNcclCoordinator::submit(
    CudaNcclCoordinatorRequest request) noexcept {
    auto& state = *implementation_;
    const auto gate = state.gate();
    if (gate != CUDA_SUCCESS) return {.carrier = gate};
    const Implementation::Scope scope(state);
    if (state.closed) return {.carrier = CUDA_ERROR_DEINITIALIZED};
    const auto kind = request.arguments.operation;
    if (!request.caller.peer || (!request.caller.thread[0] && !request.caller.thread[1]) ||
        !request.sequence || !request.domain.identity || request.domain.device_ordinal < 0 ||
        !request.domain.context.id || !request.domain.context.native ||
        !request.domain.context.lifetime || request.arguments.communicator.native ||
        request.arguments.communicator.lifetime ||
        request.source_turns.size() > state.limits.maximum_tails ||
        (request.caller_context && !valid_context(*request.caller_context, true)))
        return {.carrier = CUDA_ERROR_INVALID_VALUE};
    if (!state.next_operation) return {.carrier = CUDA_ERROR_OUT_OF_MEMORY};
    try {
        auto found = std::ranges::find_if(
            state.callers, [&](const auto& caller) { return caller.key == request.caller; });
        const bool new_caller = found == state.callers.end();
        if (new_caller) {
            if (state.callers.size() == state.limits.lanes.maximum_lanes)
                return {.carrier = CUDA_ERROR_OUT_OF_MEMORY};
            Caller caller;
            caller.key = request.caller;
            caller.members.reserve(state.limits.maximum_group_members);
            caller.domains.reserve(state.limits.maximum_caller_domains);
            state.callers.push_back(std::move(caller));
            found = std::prev(state.callers.end());
        }
        CallerReservation reservation{state.callers, new_caller};
        auto& caller = *found;
        if (caller.uncertain) return {.carrier = CUDA_ERROR_UNKNOWN};
        if (caller.active_operation || caller.active_housekeeping)
            return {.carrier = CUDA_ERROR_NOT_READY};
        if (caller.public_sequence == UINT64_MAX || request.sequence != caller.public_sequence + 1)
            return {.carrier = CUDA_ERROR_INVALID_VALUE};
        const auto domain = std::ranges::find_if(caller.domains, [&](const auto& item) {
            return item.identity == request.domain.identity;
        });
        const bool new_domain = domain == caller.domains.end();
        if (!new_domain && !domain->matches(request.domain))
            return {.carrier = CUDA_ERROR_INVALID_HANDLE};
        if (new_domain && caller.domains.size() == state.limits.maximum_caller_domains)
            return {.carrier = CUDA_ERROR_OUT_OF_MEMORY};
        const auto caller_index = static_cast<std::size_t>(found - state.callers.begin());
        const bool grouped = caller.depth && group_member(kind);
        const bool outer_end = kind == Kind::kGroupEnd && caller.depth == 1;
        const bool starts_group = kind == Kind::kGroupStart && !caller.depth;
        const auto credit =
            kind == Kind::kGroupEnd && caller.depth
                ? OperationCredit::kGroupEnd
                : (!grouped && (kind == Kind::kGetAsyncError || kind == Kind::kAbort)
                       ? OperationCredit::kProgress
                       : OperationCredit::kNormal);
        std::erase_if(state.groups, [](const auto& group) { return group.use_count() == 1; });
        if (starts_group && state.groups.size() == state.limits.maximum_retained_groups)
            return {.carrier = CUDA_ERROR_OUT_OF_MEMORY};
        if (credit == OperationCredit::kGroupEnd) {
            if (!caller.group_credit ||
                std::ranges::any_of(state.operations, [&](const auto& item) {
                    return item->credit == OperationCredit::kGroupEnd &&
                           item->group_credit == caller.group_credit;
                }))
                return {.carrier = CUDA_ERROR_OUT_OF_MEMORY};
        } else {
            const auto count = static_cast<std::size_t>(std::ranges::count_if(
                state.operations, [credit](const auto& item) { return item->credit == credit; }));
            const auto bound = credit == OperationCredit::kNormal
                                   ? state.limits.maximum_operations
                                   : state.limits.maximum_progress_operations;
            if (count == bound) return {.carrier = CUDA_ERROR_OUT_OF_MEMORY};
        }
        if (kind == Kind::kGroupStart && caller.depth == state.limits.maximum_group_depth)
            return {.carrier = CUDA_ERROR_OUT_OF_MEMORY};
        if (grouped && caller.members.size() == state.limits.maximum_group_members)
            return {.carrier = CUDA_ERROR_OUT_OF_MEMORY};
        auto communicator = state.find_communicator(request.communicator);
        if (request.communicator) {
            if (!communicator || communicator->peer != request.caller.peer ||
                !same_domain(communicator->domain, request.domain) || communicator->sealed ||
                communicator->terminal)
                return {.carrier = CUDA_ERROR_INVALID_HANDLE};
            if (!communicator->published || communicator->active_operation ||
                (communicator->group_owner && communicator->group_owner != caller_index + 1))
                return {.carrier = CUDA_ERROR_NOT_READY};
            if ((!communicator->ready || communicator->uncertain || communicator->pending_batch) &&
                kind != Kind::kGetAsyncError && kind != Kind::kAbort)
                return {.carrier = CUDA_ERROR_NOT_READY};
        }
        if ((initializes(kind) || kind == Kind::kGetUniqueId || kind == Kind::kGroupStart ||
             kind == Kind::kGroupEnd) &&
            request.communicator)
            return {.carrier = CUDA_ERROR_INVALID_VALUE};
        const bool has_group_collectives =
            outer_end && std::ranges::any_of(caller.members, [](const auto& member) {
                return collective(member->request.arguments.operation);
            });
        const bool batch = (collective(kind) && !grouped) || has_group_collectives;
        if (collective(kind) != static_cast<bool>(request.prepare_enqueue) ||
            batch != static_cast<bool>(request.record_enqueue) ||
            ((batch || collective(kind)) && !request.arguments.resources_lifetime))
            return {.carrier = CUDA_ERROR_INVALID_VALUE};
        const bool reserve = initializes(kind) && request.arguments.has_output;
        if (reserve && (state.communicators.size() == state.limits.maximum_communicators ||
                        !state.next_communicator))
            return {.carrier = CUDA_ERROR_OUT_OF_MEMORY};
        auto operation = std::make_shared<Operation>();
        operation->id = state.next_operation;
        operation->credit = credit;
        if (starts_group)
            operation->group_credit = std::make_shared<GroupCredit>();
        else if (credit == OperationCredit::kGroupEnd)
            operation->group_credit = caller.group_credit;
        operation->caller_index = caller_index;
        operation->request = std::move(request);
        operation->had_source_turns = !operation->request.source_turns.empty();
        operation->grouped = grouped;
        operation->outer_end = outer_end;
        operation->is_batch = batch;
        operation->observation.carrier = CUDA_SUCCESS;
        operation->devices.reserve(state.limits.lanes.maximum_devices);
        append_device(operation->devices, operation->request.domain.device_ordinal);
        if (operation->request.caller_context && operation->request.caller_context->context.native)
            append_device(operation->devices, operation->request.caller_context->device_ordinal);
        if (reserve) {
            communicator = std::make_shared<Communicator>();
            communicator->id = state.next_communicator;
            communicator->peer = operation->request.caller.peer;
            communicator->domain = operation->request.domain;
            operation->reserved = communicator->id;
            operation->observation.reserved_communicator = communicator->id;
        }
        operation->communicator = communicator;
        append_participant(operation->participants, communicator);
        if (outer_end) {
            operation->members = caller.members;
            for (const auto& member : operation->members) {
                append_participant(operation->participants, member->communicator);
                for (const auto device : member->devices) append_device(operation->devices, device);
                operation->has_destructive =
                    operation->has_destructive || destructive(member->request.arguments.operation);
            }
            operation->publications.resize(static_cast<std::size_t>(
                std::ranges::count_if(operation->members, [](const auto& member) {
                    return initializes(member->request.arguments.operation);
                })));
        }
        if (operation->devices.size() > state.limits.lanes.maximum_devices)
            return {.carrier = CUDA_ERROR_INVALID_VALUE};
        auto native_arguments = operation->request.arguments;
        if (communicator && !reserve)
            native_arguments.communicator = {communicator->native, communicator->lease};
        const auto created =
            CudaNcclNativeCall::create(state.native_dispatch, std::move(native_arguments));
        if (created.status != CUDA_SUCCESS) return {.carrier = created.status};
        operation->packet = created.call;
        if (reserve) {
            communicator->initialization = operation->packet;
            communicator->lease = std::make_shared<const NativeOwner>(
                NativeOwner{communicator->domain, operation->packet});
        }
        const auto accepted = state.enqueue(operation, false);
        if (accepted != CUDA_SUCCESS) return {.carrier = accepted};
        // All vectors were reserved/materialized before native admission.
        state.operations.push_back(operation);
        ++state.next_operation;
        ++caller.public_sequence;
        if (new_domain) {
            const auto& accepted_domain = operation->request.domain;
            caller.domains.push_back({accepted_domain.identity, accepted_domain.context.id,
                                      accepted_domain.device_ordinal,
                                      accepted_domain.context.native,
                                      accepted_domain.context.lifetime});
        }
        if (starts_group) {
            caller.group_credit = operation->group_credit;
            state.groups.push_back(operation->group_credit);
        }
        if (reserve) {
            state.communicators.push_back(communicator);
            ++state.next_communicator;
        }
        if (grouped) {
            caller.members.push_back(operation);
            if (communicator) communicator->group_owner = caller_index + 1;
        }
        if (destructive(kind) && communicator) communicator->sealed = true;
        reservation.provisional = false;
        return {CUDA_SUCCESS, operation->id, operation->reserved};
    } catch (const std::bad_alloc&) {
        return {.carrier = CUDA_ERROR_OUT_OF_MEMORY};
    } catch (...) {
        return {.carrier = CUDA_ERROR_UNKNOWN};
    }
}

CudaNcclHousekeepingAcceptance CudaNcclCoordinator::submit_housekeeping(
    CudaNativeExecutionLaneKey key, CudaNcclCoordinatorDomain domain,
    std::shared_ptr<const void> lifetime, std::function<CUresult()> invoke,
    std::span<const int> device_ordinals) noexcept {
    auto& state = *implementation_;
    const auto gate = state.gate();
    if (gate != CUDA_SUCCESS) return {.status = gate};
    const Implementation::Scope scope(state);
    if (state.stopped || state.is_shutdown) return {.status = CUDA_ERROR_DEINITIALIZED};
    if (!lifetime || !invoke || !domain.context.lifetime ||
        device_ordinals.size() > state.limits.lanes.maximum_devices)
        return {.status = CUDA_ERROR_INVALID_VALUE};
    const auto found =
        std::ranges::find_if(state.callers, [&](const auto& caller) { return caller.key == key; });
    if (found == state.callers.end()) return {.status = CUDA_ERROR_INVALID_HANDLE};
    auto& caller = *found;
    if (!std::ranges::any_of(caller.domains,
                             [&](const auto& item) { return item.matches(domain); }))
        return {.status = CUDA_ERROR_INVALID_HANDLE};
    if (caller.uncertain) return {.status = CUDA_ERROR_UNKNOWN};
    if (caller.active_operation || caller.active_housekeeping ||
        caller.internal_sequence == UINT64_MAX)
        return {.status = CUDA_ERROR_NOT_READY};
    if (!device_ordinals.empty() &&
        std::ranges::find(device_ordinals, domain.device_ordinal) == device_ordinals.end())
        return {.status = CUDA_ERROR_INVALID_VALUE};
    for (std::size_t index = 0; index != device_ordinals.size(); ++index) {
        const auto device = device_ordinals[index];
        const auto prefix = device_ordinals.first(index);
        if (device < 0 || std::ranges::find(prefix, device) != prefix.end())
            return {.status = CUDA_ERROR_INVALID_VALUE};
        if (!std::ranges::any_of(caller.domains, [device](const auto& item) {
                return item.device_ordinal == device && !item.lifetime.expired();
            }))
            return {.status = CUDA_ERROR_INVALID_HANDLE};
    }
    if (!state.next_housekeeping ||
        state.housekeeping.size() == state.limits.maximum_housekeeping_operations)
        return {.status = CUDA_ERROR_OUT_OF_MEMORY};
    try {
        const auto payload = std::make_shared<const HousekeepingPayload>(
            HousekeepingPayload{std::move(domain), std::move(lifetime), std::move(invoke)});
        CudaNativeExecutionCall call;
        call.sequence = caller.internal_sequence + 1;
        call.device_ordinal = payload->domain.device_ordinal;
        call.context = payload->domain.context;
        call.arguments_lifetime = payload;
        call.invoke = [payload] { return payload->invoke(); };
        call.device_ordinals.assign(device_ordinals.begin(), device_ordinals.end());
        Housekeeping job;
        job.id = state.next_housekeeping;
        job.caller_index = static_cast<std::size_t>(found - state.callers.begin());
        job.payload = payload;
        auto accepted = state.lanes.submit(key, std::move(call));
        if (accepted.status != CUDA_SUCCESS) return {.status = accepted.status};
        job.ticket = std::move(accepted.ticket);
        const auto id = job.id;
        state.housekeeping.push_back(std::move(job));
        ++state.next_housekeeping;
        ++caller.internal_sequence;
        caller.active_housekeeping = id;
        return {CUDA_SUCCESS, id};
    } catch (const std::bad_alloc&) {
        return {.status = CUDA_ERROR_OUT_OF_MEMORY};
    } catch (...) {
        return {.status = CUDA_ERROR_UNKNOWN};
    }
}

CudaNativeExecutionCallObservation CudaNcclCoordinator::observe_housekeeping(
    std::uint64_t id) const noexcept {
    const auto& state = *implementation_;
    const auto gate = state.gate();
    if (gate != CUDA_SUCCESS) return {CudaNativeExecutionCallPhase::kUncertain, gate, false};
    const auto found = std::ranges::find(state.housekeeping, id, &Housekeeping::id);
    if (found == state.housekeeping.end())
        return {CudaNativeExecutionCallPhase::kCancelled, CUDA_ERROR_INVALID_HANDLE, false};
    return found->processed ? found->observation : found->ticket.observe();
}

CUresult CudaNcclCoordinator::release_housekeeping(std::uint64_t id) noexcept {
    auto& state = *implementation_;
    const auto gate = state.gate();
    if (gate != CUDA_SUCCESS) return gate;
    const Implementation::Scope scope(state);
    const auto found = std::ranges::find(state.housekeeping, id, &Housekeeping::id);
    if (found == state.housekeeping.end()) return CUDA_ERROR_INVALID_HANDLE;
    state.consume_housekeeping(*found);
    if (!found->processed) return CUDA_ERROR_NOT_READY;
    if (!found->releasable) return found->observation.status;
    const auto released = state.lanes.release(found->ticket);
    if (released != CUDA_SUCCESS) {
        found->releasable = false;
        found->observation.phase = CudaNativeExecutionCallPhase::kUncertain;
        found->observation.status = released;
        state.callers[found->caller_index].uncertain = true;
        return released;
    }
    state.housekeeping.erase(found);
    return CUDA_SUCCESS;
}

CUresult CudaNcclCoordinator::advance() noexcept {
    return advance(true);
}
CUresult CudaNcclCoordinator::advance(bool authorize_continuations) noexcept {
    auto& state = *implementation_;
    const auto gate = state.gate();
    if (gate != CUDA_SUCCESS) return gate;
    const Implementation::Scope scope(state);
    try {
        for (auto& job : state.housekeeping) state.consume_housekeeping(job);
        for (const auto& operation : state.operations)
            if (operation->has_ticket) state.consume(operation);
        for (const auto& operation : state.operations) {
            if (operation->uncertain || operation->finished || operation->has_ticket) continue;
            if (operation->observation.phase == Phase::kWaitingAsync &&
                !operation->participants.empty() &&
                std::ranges::all_of(operation->participants, [](const auto& item) {
                    return item->ready && !item->sealed && !item->uncertain;
                })) {
                if (operation->is_batch) {
                    if (authorize_continuations && state.continuation_ready(*operation)) {
                        const auto status = state.enqueue(operation, true);
                        if (status != CUDA_SUCCESS && status != CUDA_ERROR_NOT_READY &&
                            status != CUDA_ERROR_OUT_OF_MEMORY)
                            state.fail(operation, status);
                    }
                } else {
                    for (const auto& communicator : operation->participants)
                        if (communicator->pending_batch == operation->id)
                            communicator->pending_batch = 0;
                    for (const auto& member : operation->members) state.finish(member);
                    operation->members.clear();
                    state.finish(operation);
                }
            }
            if (operation->observation.enqueue_complete && operation->tails_bound) {
                bool complete = true;
                for (const auto& tail : operation->tails) {
                    const auto status = tail.observe();
                    if (status != CUDA_SUCCESS && status != CUDA_ERROR_NOT_READY) {
                        state.fail(operation, status);
                        break;
                    }
                    complete = complete && status == CUDA_SUCCESS;
                }
                if (complete && !operation->uncertain) {
                    for (const auto& member : operation->members) state.finish(member);
                    operation->members.clear();
                    state.finish(operation);
                }
            }
        }
        std::erase_if(state.operations, [](const auto& operation) {
            return operation->finished && operation->released;
        });
        return CUDA_SUCCESS;
    } catch (const std::bad_alloc&) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}
CUresult CudaNcclCoordinator::continue_enqueue(std::uint64_t id) noexcept {
    auto& state = *implementation_;
    const auto gate = state.gate();
    if (gate != CUDA_SUCCESS) return gate;
    const Implementation::Scope scope(state);
    const auto operation = state.find_operation(id);
    if (!operation) return CUDA_ERROR_INVALID_HANDLE;
    if (!state.continuation_ready(*operation)) return CUDA_ERROR_NOT_READY;
    try {
        const auto status = state.enqueue(operation, true);
        if (status != CUDA_SUCCESS && status != CUDA_ERROR_NOT_READY &&
            status != CUDA_ERROR_OUT_OF_MEMORY)
            state.fail(operation, status);
        return status;
    } catch (const std::bad_alloc&) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    } catch (...) {
        state.fail(operation, CUDA_ERROR_UNKNOWN);
        return CUDA_ERROR_UNKNOWN;
    }
}
CudaNcclCoordinatorSnapshot CudaNcclCoordinator::snapshot() const noexcept {
    const auto& state = *implementation_;
    const auto gate = state.gate();
    if (gate != CUDA_SUCCESS) return {.carrier = gate};
    try {
        CudaNcclCoordinatorSnapshot result;
        result.callers.reserve(state.callers.size());
        result.operations.reserve(state.operations.size());
        for (const auto& caller : state.callers)
            result.callers.push_back({caller.key, caller.depth, caller.active_operation,
                                      caller.active_housekeeping, caller.uncertain});
        for (const auto& operation : state.operations) {
            result.operations.push_back(
                {operation->id, operation->reserved, state.callers[operation->caller_index].key,
                 operation->observation.phase, operation->devices, operation->has_ticket,
                 operation->grouped, operation->is_batch,
                 operation->observation.phase == Phase::kWaitingAsync,
                 operation->is_batch && !operation->finished, operation->had_source_turns,
                 state.continuation_ready(*operation), operation->uncertain});
        }
        result.carrier = CUDA_SUCCESS;
        return result;
    } catch (const std::bad_alloc&) {
        return {.carrier = CUDA_ERROR_OUT_OF_MEMORY};
    } catch (...) {
        return {.carrier = CUDA_ERROR_UNKNOWN};
    }
}
CudaNcclCoordinatorObservation CudaNcclCoordinator::observe(std::uint64_t id) const noexcept {
    const auto& state = *implementation_;
    const auto gate = state.gate();
    if (gate != CUDA_SUCCESS) return {.carrier = gate};
    const auto operation = state.find_operation(id);
    try {
        return operation ? operation->observation : CudaNcclCoordinatorObservation{};
    } catch (const std::bad_alloc&) {
        return {.carrier = CUDA_ERROR_OUT_OF_MEMORY};
    } catch (...) {
        return {.carrier = CUDA_ERROR_UNKNOWN};
    }
}
CUresult CudaNcclCoordinator::release_reply(std::uint64_t id) noexcept {
    auto& state = *implementation_;
    const auto gate = state.gate();
    if (gate != CUDA_SUCCESS) return gate;
    const Implementation::Scope scope(state);
    const auto operation = state.find_operation(id);
    if (!operation || operation->released) return CUDA_ERROR_INVALID_HANDLE;
    if (!operation->observation.reply_ready) return CUDA_ERROR_NOT_READY;
    operation->released = true;
    if (operation->finished) std::erase(state.operations, operation);
    return CUDA_SUCCESS;
}
CUresult CudaNcclCoordinator::bind_tails(std::uint64_t id,
                                         std::span<const CudaExecutionFrontier> tails) noexcept {
    auto& state = *implementation_;
    const auto gate = state.gate();
    if (gate != CUDA_SUCCESS) return gate;
    const Implementation::Scope scope(state);
    const auto operation = state.find_operation(id);
    if (!operation || !operation->is_batch || operation->tails_bound || tails.empty() ||
        tails.size() > state.limits.maximum_tails)
        return CUDA_ERROR_INVALID_VALUE;
    for (const auto& tail : tails) {
        const auto observed = tail.observe();
        if (observed != CUDA_SUCCESS && observed != CUDA_ERROR_NOT_READY) return observed;
    }
    try {
        operation->tails.assign(tails.begin(), tails.end());
    } catch (const std::bad_alloc&) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
    operation->tails_bound = true;
    return CUDA_SUCCESS;
}
CUresult CudaNcclCoordinator::close() noexcept {
    auto& state = *implementation_;
    const auto gate = state.gate();
    if (gate != CUDA_SUCCESS) return gate;
    state.closed = true;
    return CUDA_SUCCESS;
}
CUresult CudaNcclCoordinator::shutdown() noexcept {
    auto& state = *implementation_;
    const auto gate = state.gate();
    if (gate != CUDA_SUCCESS) return gate;
    const Implementation::Scope scope(state);
    if (state.is_shutdown) return CUDA_SUCCESS;
    if (!state.closed || !state.quiescent()) return CUDA_ERROR_NOT_READY;
    if (!state.stopped) {
        const auto status = state.lanes.request_stop();
        if (status != CUDA_SUCCESS) return status;
        state.stopped = true;
    }
    const auto status = state.lanes.shutdown();
    if (status == CUDA_SUCCESS) state.is_shutdown = true;
    return status;
}

}  // namespace glimmer::backend::cuda
