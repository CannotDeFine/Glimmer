#include "internal/cuda_event_projection_registry.h"

#include <atomic>
#include <bit>
#include <array>
#include <iostream>
#include <latch>
#include <stdexcept>
#include <thread>
#include <vector>
#include <cstdlib>
#include <new>

namespace {
thread_local bool should_reject_allocation{};
}
// Exercise the shared registry's ordinary map-node/bucket allocation path.
// The guard is caller-local; concurrent test workers retain normal allocation.
void* operator new(std::size_t bytes) {
    if (should_reject_allocation) {
        throw std::bad_alloc();
    }
    if (void* result = std::malloc(bytes ? bytes : 1)) {
        return result;
    }
    throw std::bad_alloc();
}
void operator delete(void* pointer) noexcept {
    std::free(pointer);
}
void operator delete(void* pointer, std::size_t) noexcept {
    std::free(pointer);
}

namespace {
using namespace glimmer::interceptor;
const auto kContext = reinterpret_cast<CUcontext>(0x1000);
const auto kOtherContext = reinterpret_cast<CUcontext>(0x2000);
const auto kEvent = reinterpret_cast<CUevent>(0x3000);
const auto kOtherEvent = reinterpret_cast<CUevent>(0x4000);

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}
CudaEventProjection lookup(const CudaEventProjectionRegistry& registry, CUevent event) {
    CudaEventProjection result;
    require(registry.lookup(event, result) == CUDA_SUCCESS && result,
            "checked event projection lookup");
    return result;
}
void exact_registration_retirement() {
    CudaEventProjectionRegistry registry;
    CudaEventProjectionRegistry foreign;
    require(registry.register_event(kContext, kEvent, false) == CUDA_SUCCESS &&
                foreign.register_event(kContext, kEvent, false) == CUDA_SUCCESS,
            "independent event registration owners");
    const auto retired = lookup(registry, kEvent);
    const auto other = lookup(foreign, kEvent);
    require(registry.forget_owned({}) == CUDA_ERROR_INVALID_VALUE &&
                registry.forget_owned(other) == CUDA_ERROR_INVALID_VALUE,
            "empty and foreign owners cannot retire live registration");
    should_reject_allocation = true;
    const auto forgotten = registry.forget_owned(retired);
    should_reject_allocation = false;
    require(
        forgotten == CUDA_SUCCESS && registry.forget_owned(retired) == CUDA_ERROR_INVALID_HANDLE,
        "exact retirement is allocation-free and cannot repeat");
    require(registry.register_event(kOtherContext, kEvent, false) == CUDA_SUCCESS,
            "recycled event receives independent registration");
    require(registry.forget_owned(retired) == CUDA_ERROR_INVALID_HANDLE &&
                registry.complete_native_events(std::span(&retired, 1)) == CUDA_SUCCESS,
            "old retirement and late prefix proof cannot reach recycled event");
    const auto fresh = lookup(registry, kEvent);
    require(fresh.context() == kOtherContext && !fresh.has_native_completion() &&
                retired.context() == kContext && !retired.has_native_completion(),
            "immutable historical identity never becomes fresh completion");
    require(registry.forget_owned(fresh) == CUDA_SUCCESS,
            "fresh exact registration retires independently");
}

void generations() {
    CudaEventProjectionRegistry registry;
    CudaEventProjection empty;
    require(registry.lookup(kEvent, empty) == CUDA_ERROR_INVALID_HANDLE && !empty &&
                registry.publish_logical(empty) == CUDA_ERROR_INVALID_VALUE &&
                registry.register_event(nullptr, kEvent, true) == CUDA_ERROR_INVALID_VALUE &&
                registry.register_event(kContext, nullptr, true) == CUDA_ERROR_INVALID_VALUE,
            "invalid inputs cannot publish ownership");
    require(registry.register_event(kContext, kEvent, true) == CUDA_SUCCESS &&
                registry.register_event(kOtherContext, kOtherEvent, true) == CUDA_SUCCESS &&
                registry.register_event(kContext, kEvent, false) == CUDA_ERROR_INVALID_HANDLE,
            "unique actual event registration");
    const auto old = lookup(registry, kEvent);
    require(old.context() == kContext && old.is_logical(), "initial logical binding");
    auto occupied = old;
    require(registry.lookup(kOtherEvent, occupied) == CUDA_ERROR_INVALID_VALUE &&
                occupied.context() == kContext,
            "occupied output cannot hide an earlier snapshot");
    require(registry.invalidate_native(kContext) == CUDA_SUCCESS &&
                !lookup(registry, kEvent).is_logical() &&
                lookup(registry, kOtherEvent).is_logical() && old.is_logical() &&
                registry.publish_logical(old) == CUDA_ERROR_NOT_READY,
            "context native ownership invalidates publication, not immutable snapshots");
    const auto native = lookup(registry, kEvent);
    // Rejected source acceptance deliberately does not call publish_logical.
    require(!lookup(registry, kEvent).is_logical() && !native.has_native_completion() &&
                registry.publish_logical(native) == CUDA_ERROR_NOT_READY,
            "rejection or missing physical proof preserves native routing");
    require(registry.complete_native(kOtherContext) == CUDA_SUCCESS &&
                !lookup(registry, kEvent).has_native_completion() &&
                registry.complete_native(kContext) == CUDA_SUCCESS &&
                lookup(registry, kEvent).has_native_completion() &&
                !lookup(registry, kEvent).is_logical(),
            "native proof covers its actual context without restoring ownership");
    require(registry.publish_logical(native) == CUDA_SUCCESS &&
                lookup(registry, kEvent).is_logical() &&
                registry.publish_logical(native) == CUDA_SUCCESS,
            "matching fresh acceptance restores logical routing idempotently");
    require(registry.invalidate_native() == CUDA_SUCCESS &&
                !lookup(registry, kEvent).is_logical() &&
                !lookup(registry, kOtherEvent).is_logical() &&
                registry.publish_logical(native) == CUDA_ERROR_NOT_READY,
            "all-context native entry retires every earlier epoch");
    const auto retired = lookup(registry, kEvent);
    registry.forget(kEvent);
    require(registry.publish_logical(retired) == CUDA_ERROR_INVALID_HANDLE &&
                registry.register_event(kContext, kEvent, false) == CUDA_SUCCESS &&
                registry.publish_logical(retired) == CUDA_ERROR_INVALID_HANDLE &&
                !lookup(registry, kEvent).is_logical(),
            "same native pointer reuse cannot restore a retired registration");
    CudaEventProjectionRegistry foreign;
    require(foreign.register_event(kContext, kEvent, true) == CUDA_SUCCESS &&
                registry.publish_logical(lookup(foreign, kEvent)) == CUDA_ERROR_INVALID_HANDLE,
            "foreign registration with equal pointer/context is not a capability");
    for (unsigned int iteration = 0; iteration < 10000; ++iteration) {
        const auto before = lookup(registry, kEvent);
        require(registry.invalidate_native(kContext) == CUDA_SUCCESS &&
                    registry.publish_logical(before) == CUDA_ERROR_NOT_READY &&
                    !lookup(registry, kEvent).has_native_completion() &&
                    registry.complete_native(kContext) == CUDA_SUCCESS &&
                    registry.publish_logical(lookup(registry, kEvent)) == CUDA_SUCCESS,
                "native epochs cannot alias previous accepted snapshots");
    }
    registry.forget_context(kOtherContext);
    CudaEventProjection absent;
    require(registry.lookup(kOtherEvent, absent) == CUDA_ERROR_INVALID_HANDLE &&
                lookup(registry, kEvent).context() == kContext,
            "checked context teardown retires only its registrations");
    registry.forget(kEvent);
    registry.forget(kEvent);
}
void targeted_epoch_ownership() {
    CudaEventProjectionRegistry registry;
    require(registry.register_event(kContext, kEvent, true) == CUDA_SUCCESS &&
                registry.register_event(kContext, kOtherEvent, true) == CUDA_SUCCESS,
            "register events in the same actual context");
    const auto old = lookup(registry, kEvent);
    const auto other = lookup(registry, kOtherEvent);
    bool is_current = true;
    require(registry.matches_native_epoch({}, is_current) == CUDA_ERROR_INVALID_VALUE &&
                !is_current && registry.invalidate_native_event({}) == CUDA_ERROR_INVALID_VALUE &&
                registry.matches_native_epoch(old, is_current) == CUDA_SUCCESS && is_current,
            "guard validates before granting write permission");
    require(registry.invalidate_native_event(old) == CUDA_SUCCESS &&
                registry.matches_native_epoch(old, is_current) == CUDA_SUCCESS && !is_current &&
                registry.matches_native_epoch(other, is_current) == CUDA_SUCCESS && is_current &&
                lookup(registry, kOtherEvent).is_logical() &&
                registry.invalidate_native_event(old) == CUDA_ERROR_NOT_READY,
            "targeted ownership suppresses only the older record, not unrelated logical events");
    const auto native = lookup(registry, kEvent);
    require(registry.matches_native_epoch(native, is_current) == CUDA_SUCCESS && is_current &&
                !native.has_native_completion() &&
                registry.complete_native(kContext) == CUDA_SUCCESS &&
                registry.publish_logical(native) == CUDA_SUCCESS &&
                registry.matches_native_epoch(native, is_current) == CUDA_SUCCESS && is_current,
            "epoch match is distinct from physical completion and logical publication");
    registry.forget(kEvent);
    require(registry.matches_native_epoch(native, is_current) == CUDA_ERROR_INVALID_HANDLE &&
                !is_current &&
                registry.invalidate_native_event(native) == CUDA_ERROR_INVALID_HANDLE &&
                registry.register_event(kContext, kEvent, true) == CUDA_SUCCESS &&
                registry.matches_native_epoch(native, is_current) == CUDA_ERROR_INVALID_HANDLE &&
                !is_current,
            "handle retirement and ABA never grant an old write permission");
    CudaEventProjectionRegistry foreign;
    require(foreign.register_event(kContext, kEvent, true) == CUDA_SUCCESS &&
                registry.invalidate_native_event(lookup(foreign, kEvent)) ==
                    CUDA_ERROR_INVALID_HANDLE &&
                registry.matches_native_epoch(lookup(foreign, kEvent), is_current) ==
                    CUDA_ERROR_INVALID_HANDLE &&
                !is_current && lookup(registry, kEvent).is_logical(),
            "foreign snapshots cannot retire or write a matching native handle");
}

void coordinated_callers() {
    CudaEventProjectionRegistry registry;
    require(registry.register_event(kContext, kEvent, true) == CUDA_SUCCESS,
            "register concurrent test binding");
    std::latch captured{1};
    std::latch invalidated{1};
    std::atomic<CUresult> published{CUDA_SUCCESS};
    std::atomic<bool> is_current{true};
    std::thread caller([&] {
        const auto accepted = lookup(registry, kEvent);
        captured.count_down();
        invalidated.wait();
        bool matches = true;
        require(registry.matches_native_epoch(accepted, matches) == CUDA_SUCCESS,
                "concurrent guard observation remains checked");
        is_current.store(matches);
        published.store(registry.publish_logical(accepted));
    });
    captured.wait();
    const auto result = registry.invalidate_native(kContext);
    invalidated.count_down();
    caller.join();
    require(result == CUDA_SUCCESS && !is_current.load() &&
                published.load() == CUDA_ERROR_NOT_READY && !lookup(registry, kEvent).is_logical(),
            "late source publication cannot undo a competing native epoch");
}

void exact_native_completion() {
    CudaEventProjectionRegistry registry;
    require(registry.register_event(kContext, kEvent, true) == CUDA_SUCCESS &&
                registry.register_event(kContext, kOtherEvent, true) == CUDA_SUCCESS,
            "register exact epoch completion events");
    auto first = lookup(registry, kEvent);
    auto second = lookup(registry, kOtherEvent);
    const std::array logical_proof{first};
    require(registry.complete_native_events(logical_proof) == CUDA_ERROR_INVALID_VALUE,
            "a logical record snapshot is not a native execution epoch");
    std::array duplicate{first, first};
    require(registry.invalidate_native_events(duplicate) == CUDA_ERROR_INVALID_VALUE &&
                lookup(registry, kEvent).is_logical() && duplicate[0].is_logical(),
            "duplicate rejection cannot partially advance an epoch or caller snapshot");
    std::array invalid{first, CudaEventProjection{}};
    require(registry.invalidate_native_events(invalid) == CUDA_ERROR_INVALID_VALUE &&
                lookup(registry, kEvent).is_logical() && invalid[0].is_logical(),
            "invalid later member must leave earlier members unchanged");
    CudaEventProjectionRegistry foreign;
    require(foreign.register_event(kContext, kOtherEvent, true) == CUDA_SUCCESS,
            "register foreign completion owner");
    std::array foreign_batch{first, lookup(foreign, kOtherEvent)};
    require(registry.invalidate_native_events(foreign_batch) == CUDA_ERROR_INVALID_HANDLE &&
                lookup(registry, kEvent).is_logical(),
            "foreign later registration rejects atomically");
    require(registry.invalidate_native_event(second) == CUDA_SUCCESS,
            "advance a competing native epoch");
    std::array stale{first, second};
    require(registry.invalidate_native_events(stale) == CUDA_ERROR_NOT_READY &&
                lookup(registry, kEvent).is_logical() && stale[0].is_logical(),
            "stale later epoch cannot partially invalidate another event");
    second = lookup(registry, kOtherEvent);
    std::array epochs{first, second};
    require(registry.invalidate_native_events(epochs) == CUDA_SUCCESS && !epochs[0].is_logical() &&
                !epochs[1].has_native_completion() &&
                registry.publish_logical(first) == CUDA_ERROR_NOT_READY &&
                registry.publish_logical(second) == CUDA_ERROR_NOT_READY,
            "batch atomically publishes distinct native epochs and refreshed snapshots");
    std::array foreign_proof{epochs[0], lookup(foreign, kOtherEvent)};
    require(registry.complete_native_events(foreign_proof) == CUDA_ERROR_INVALID_HANDLE &&
                !lookup(registry, kEvent).has_native_completion() &&
                registry.complete_native_events(invalid) == CUDA_ERROR_INVALID_VALUE &&
                !lookup(registry, kEvent).has_native_completion(),
            "foreign or empty later proof cannot partially publish completion");
    require(registry.invalidate_native_event(epochs[1]) == CUDA_SUCCESS &&
                registry.complete_native_events(epochs) == CUDA_SUCCESS &&
                lookup(registry, kEvent).has_native_completion() &&
                !lookup(registry, kOtherEvent).has_native_completion() &&
                !lookup(registry, kEvent).is_logical() &&
                registry.publish_logical(epochs[0]) == CUDA_SUCCESS,
            "old physical proof completes its current epoch, not a superseding one");
    registry.forget(kEvent);
    require(registry.complete_native_events(epochs) == CUDA_SUCCESS &&
                registry.register_event(kContext, kEvent, false) == CUDA_SUCCESS &&
                registry.complete_native_events(epochs) == CUDA_SUCCESS &&
                !lookup(registry, kEvent).has_native_completion() &&
                !lookup(registry, kOtherEvent).has_native_completion(),
            "retired and reused registrations ignore late proof without granting new completion");
    require(registry.invalidate_native_events({}) == CUDA_SUCCESS &&
                registry.complete_native_events({}) == CUDA_SUCCESS,
            "graphs without event effects need no special completion mutation");

    auto current = lookup(registry, kEvent);
    std::array owned{current};
    require(registry.invalidate_native_events(owned) == CUDA_SUCCESS,
            "prepare concurrent actual execution epoch");
    std::latch captured{1};
    std::latch superseded{1};
    std::atomic<CUresult> completed{CUDA_ERROR_UNKNOWN};
    std::thread late([&] {
        captured.count_down();
        superseded.wait();
        completed.store(registry.complete_native_events(owned));
    });
    captured.wait();
    const auto advanced = registry.invalidate_native_event(owned[0]);
    superseded.count_down();
    late.join();
    require(advanced == CUDA_SUCCESS && completed.load() == CUDA_SUCCESS &&
                !lookup(registry, kEvent).has_native_completion(),
            "concurrent late old completion cannot prove a later source epoch");
}
void prepared_native_registration() {
    CudaEventProjectionRegistry registry;
    CudaEventProjectionRegistry foreign;
    CudaEventRegistrationReservation first;
    require(
        registry.prepare_native_event(nullptr, first) == CUDA_ERROR_INVALID_VALUE && !first &&
            registry.publish_native_event(first, kEvent, CUDA_SUCCESS) == CUDA_ERROR_INVALID_HANDLE,
        "preparation validates before original creation");
    require(
        registry.prepare_native_event(kContext, first) == CUDA_SUCCESS && first &&
            registry.prepare_native_event(kContext, first) == CUDA_ERROR_INVALID_VALUE &&
            foreign.publish_native_event(first, kEvent, CUDA_SUCCESS) == CUDA_ERROR_INVALID_HANDLE,
        "ticket owns one exact registry slot");
    CudaEventRegistrationReservation moved(std::move(first));
    require(static_cast<bool>(moved), "move transfers preparation without duplicate cancellation");
    require(
        registry.register_event(kOtherContext, kOtherEvent, true) == CUDA_SUCCESS &&
            registry.publish_native_event(moved, kEvent, CUDA_SUCCESS) == CUDA_SUCCESS &&
            registry.publish_native_event(moved, kEvent, CUDA_SUCCESS) == CUDA_ERROR_INVALID_HANDLE,
        "intervening normal insertion preserves native node capacity and exact-once publication");
    const auto owned = lookup(registry, kEvent);
    require(owned.context() == kContext && !owned.is_logical() && !owned.has_native_completion(),
            "published native creation is neither a logical record nor completion proof");
    registry.forget(kEvent);
    CudaEventRegistrationReservation reused;
    require(registry.prepare_native_event(kContext, reused) == CUDA_SUCCESS &&
                registry.publish_native_event(reused, kEvent, CUDA_SUCCESS) == CUDA_SUCCESS &&
                registry.publish_logical(owned) == CUDA_ERROR_INVALID_HANDLE,
            "prepared pointer reuse cannot restore an old registration");

    std::array<CudaEventRegistrationReservation, 256> pending;
    for (auto& ticket : pending) {
        require(registry.prepare_native_event(kContext, ticket) == CUDA_SUCCESS,
                "prepare bounded concurrent native creations");
    }
    CudaEventRegistrationReservation overflow;
    require(
        registry.prepare_native_event(kContext, overflow) == CUDA_ERROR_OUT_OF_MEMORY && !overflow,
        "reservation exhaustion rejects before any original create");
    pending[0] = {};
    require(registry.prepare_native_event(kContext, overflow) == CUDA_SUCCESS,
            "CPU cancellation reclaims one reserved slot");
    for (std::size_t index = 1; index < pending.size(); ++index) {
        const auto event = std::bit_cast<CUevent>(std::uintptr_t{0x10000 + index * 16});
        require(registry.publish_native_event(pending[index], event, CUDA_SUCCESS) == CUDA_SUCCESS,
                "preallocated nodes publish across concurrent slots");
    }
    overflow = {};
    // The bound applies only to pending preparation, not live event rows.
    for (std::size_t index = 0; index < 300; ++index) {
        CudaEventRegistrationReservation ticket;
        const auto event = std::bit_cast<CUevent>(std::uintptr_t{0x20000 + index * 16});
        require(registry.prepare_native_event(kOtherContext, ticket) == CUDA_SUCCESS &&
                    registry.publish_native_event(ticket, event, CUDA_SUCCESS) == CUDA_SUCCESS,
                "more than 256 live original handles retain the existing map contract");
    }

    CudaEventRegistrationReservation survivor;
    {
        CudaEventProjectionRegistry temporary;
        require(temporary.prepare_native_event(kContext, survivor) == CUDA_SUCCESS,
                "prepare ticket outliving CPU registry");
    }
    survivor = {};
}

void prepared_native_errors() {
    {
        CudaEventProjectionRegistry registry;
        CudaEventRegistrationReservation ticket;
        require(registry.prepare_native_event(kContext, ticket) == CUDA_SUCCESS &&
                    registry.publish_native_event(ticket, nullptr, CUDA_ERROR_INVALID_VALUE) ==
                        CUDA_ERROR_INVALID_VALUE &&
                    registry.publish_native_event(ticket, kEvent, CUDA_SUCCESS) ==
                        CUDA_ERROR_INVALID_HANDLE,
                "unpublished native error consumes/cancels the exact ticket");
        ticket = {};
        require(registry.prepare_native_event(kContext, ticket) == CUDA_SUCCESS &&
                    registry.publish_native_event(ticket, kEvent, CUDA_ERROR_UNKNOWN) ==
                        CUDA_ERROR_UNKNOWN,
                "native error with a handle retains its unique row and original status");
        CudaEventProjection unknown;
        require(registry.lookup(kEvent, unknown) == CUDA_ERROR_UNKNOWN && !unknown &&
                    registry.complete_native(kContext) == CUDA_SUCCESS &&
                    registry.lookup(kEvent, unknown) == CUDA_ERROR_UNKNOWN,
                "context completion cannot turn uncertain creation into logical binding");
        registry.forget(kEvent);
    }
    for (bool is_duplicate : {false, true}) {
        CudaEventProjectionRegistry registry;
        CudaEventRegistrationReservation ticket;
        if (is_duplicate) {
            require(registry.register_event(kContext, kEvent, true) == CUDA_SUCCESS,
                    "existing owner before duplicate native output");
        }
        require(registry.prepare_native_event(kContext, ticket) == CUDA_SUCCESS &&
                    registry.publish_native_event(ticket, is_duplicate ? kEvent : nullptr,
                                                  CUDA_SUCCESS) == CUDA_ERROR_UNKNOWN,
                "duplicate or unwritten SUCCESS quarantines ownership without native rollback");
        ticket = {};
        require(registry.prepare_native_event(kContext, ticket) == CUDA_ERROR_UNKNOWN && !ticket,
                "quarantined preparation is sticky, not freed execution permission");
    }
}
void allocation_free_publication() {
    CudaEventProjectionRegistry registry;
    CudaEventRegistrationReservation ticket;
    require(registry.prepare_native_event(kContext, ticket) == CUDA_SUCCESS,
            "prepare every allocation before native output");
    CudaEventRegistrationReservation rejected;
    should_reject_allocation = true;
    const auto preparation = registry.prepare_native_event(kContext, rejected);
    const auto published = registry.publish_native_event(ticket, kEvent, CUDA_SUCCESS);
    should_reject_allocation = false;
    require(preparation == CUDA_ERROR_OUT_OF_MEMORY && !rejected && published == CUDA_SUCCESS &&
                lookup(registry, kEvent).context() == kContext,
            "allocation failure rejects preparation but cannot break prepared publication");
}

void captured_record_scope() {
    CudaEventProjectionRegistry registry;
    CudaEventProjectionRegistry foreign;
    auto first_owner = std::make_shared<const int>(0);
    const auto other_owner = std::make_shared<const int>(0);
    require(registry.register_event(kContext, kEvent, false) == CUDA_SUCCESS &&
                foreign.register_event(kContext, kEvent, false) == CUDA_SUCCESS,
            "independent actual record registrations");
    const auto original = lookup(registry, kEvent);
    const auto foreign_record = lookup(foreign, kEvent);
    require(registry.publish_capture_record({}, first_owner) == CUDA_ERROR_INVALID_VALUE &&
                registry.publish_capture_record(foreign_record, first_owner) ==
                    CUDA_ERROR_INVALID_VALUE,
            "empty or foreign projections cannot publish captured provenance");
    should_reject_allocation = true;
    const auto published = registry.publish_capture_record(original, first_owner);
    should_reject_allocation = false;
    auto captured = lookup(registry, kEvent);
    const std::shared_ptr<const void> forged(first_owner.get(), [](const void*) {});
    require(published == CUDA_SUCCESS && captured.is_captured_by(first_owner) &&
                !captured.is_captured_by(other_owner) && !captured.is_captured_by(forged) &&
                !original.is_captured_by(first_owner) && !captured.has_native_completion(),
            "allocation-free scope identity cannot be forged or treated as completion");
    require(registry.invalidate_native(kContext) == CUDA_SUCCESS &&
                registry.publish_capture_record(captured, other_owner) == CUDA_ERROR_NOT_READY &&
                lookup(registry, kEvent).is_captured_by(first_owner),
            "unrelated native epochs retain actual captured record but reject stale publication");
    require(registry.publish_capture_record(lookup(registry, kEvent), {}) == CUDA_SUCCESS &&
                !lookup(registry, kEvent).is_captured_by(first_owner) &&
                captured.is_captured_by(first_owner),
            "successful noncaptured rerecord clears current provenance, not historic snapshot");
    require(registry.publish_capture_record(lookup(registry, kEvent), first_owner) == CUDA_SUCCESS,
            "fresh captured rerecord uses current epoch");
    const std::weak_ptr<const void> lifetime = first_owner;
    first_owner.reset();
    require(lifetime.expired() && !captured.is_captured_by(forged),
            "weak record ownership retains no old session");
    const auto retired = lookup(registry, kEvent);
    require(
        registry.forget_owned(retired) == CUDA_SUCCESS &&
            registry.register_event(kContext, kEvent, false) == CUDA_SUCCESS &&
            registry.publish_capture_record(retired, other_owner) == CUDA_ERROR_INVALID_HANDLE &&
            !lookup(registry, kEvent).is_captured_by(other_owner),
        "retirement/recycled handles cannot inherit capture classification");
    CudaEventRegistrationReservation metadata_only;
    require(registry.prepare_record_event(kOtherContext, metadata_only) == CUDA_SUCCESS &&
                registry.publish_native_event(metadata_only, kOtherEvent, CUDA_SUCCESS) ==
                    CUDA_SUCCESS &&
                !lookup(registry, kOtherEvent).allows_logical_binding() &&
                registry.complete_native(kOtherContext) == CUDA_SUCCESS &&
                registry.publish_capture_record(lookup(registry, kOtherEvent), other_owner) ==
                    CUDA_SUCCESS &&
                lookup(registry, kOtherEvent).is_captured_by(other_owner) &&
                registry.publish_logical(lookup(registry, kOtherEvent)) == CUDA_ERROR_NOT_SUPPORTED,
            "record-only ownership proves capture context, never non-IPC logical import");
    registry.forget(kOtherEvent);
    CudaEventRegistrationReservation runtime_metadata;
    require(registry.prepare_runtime_record_event(runtime_metadata) == CUDA_SUCCESS &&
                registry.publish_native_event(runtime_metadata, kOtherEvent, CUDA_SUCCESS) ==
                    CUDA_ERROR_INVALID_VALUE,
            "unbound record-only Runtime ticket needs actual post-SDK context proof");
    should_reject_allocation = true;
    const auto runtime_published =
        registry.publish_runtime_event(runtime_metadata, kOtherEvent, kOtherContext);
    should_reject_allocation = false;
    require(runtime_published == CUDA_SUCCESS &&
                lookup(registry, kOtherEvent).context() == kOtherContext &&
                !lookup(registry, kOtherEvent).allows_logical_binding() &&
                registry.complete_native(kOtherContext) == CUDA_SUCCESS &&
                registry.publish_logical(lookup(registry, kOtherEvent)) == CUDA_ERROR_NOT_SUPPORTED,
            "allocation-free Runtime record publication cannot infer non-IPC creation flags");
}

void late_runtime_context() {
    CudaEventProjectionRegistry registry;
    CudaEventProjectionRegistry foreign;
    CudaEventRegistrationReservation ticket;
    require(
        registry.prepare_runtime_event(ticket) == CUDA_SUCCESS &&
            registry.prepare_runtime_event(ticket) == CUDA_ERROR_INVALID_VALUE &&
            registry.publish_native_event(ticket, kEvent, CUDA_SUCCESS) ==
                CUDA_ERROR_INVALID_VALUE &&
            foreign.publish_runtime_event(ticket, kEvent, kContext) == CUDA_ERROR_INVALID_HANDLE,
        "unbound Runtime ticket cannot publish through Driver or foreign registry");
    should_reject_allocation = true;
    const auto published = registry.publish_runtime_event(ticket, kEvent, kOtherContext);
    should_reject_allocation = false;
    require(
        published == CUDA_SUCCESS && lookup(registry, kEvent).context() == kOtherContext &&
            !lookup(registry, kEvent).is_logical() &&
            !lookup(registry, kEvent).has_native_completion() &&
            registry.publish_runtime_event(ticket, kEvent, kContext) == CUDA_ERROR_INVALID_HANDLE,
        "actual late context publishes allocation-free exactly once without completion");
    CudaEventRegistrationReservation driver_ticket;
    require(
        registry.prepare_native_event(kContext, driver_ticket) == CUDA_SUCCESS &&
            registry.publish_runtime_event(driver_ticket, kOtherEvent, kOtherContext) ==
                CUDA_ERROR_INVALID_VALUE &&
            registry.publish_native_event(driver_ticket, kOtherEvent, CUDA_SUCCESS) == CUDA_SUCCESS,
        "Runtime cannot rebind a prepared actual Driver context");
    for (bool has_context : {false, true}) {
        CudaEventProjectionRegistry uncertain;
        CudaEventRegistrationReservation unbound;
        require(uncertain.prepare_runtime_event(unbound) == CUDA_SUCCESS &&
                    uncertain.publish_runtime_event(unbound, has_context ? nullptr : kEvent,
                                                    has_context ? kContext : nullptr) ==
                        CUDA_ERROR_UNKNOWN,
                "successful Runtime creation without handle or context quarantines ticket");
        unbound = {};
        require(uncertain.prepare_runtime_event(unbound) == CUDA_ERROR_UNKNOWN && !unbound,
                "cancellation cannot clear uncertain native creation");
    }
    {
        CudaEventRegistrationReservation cancelled;
        require(registry.prepare_runtime_event(cancelled) == CUDA_SUCCESS,
                "failed SDK issue owns only bounded CPU preparation");
    }
    CudaEventRegistrationReservation next;
    require(registry.prepare_runtime_event(next) == CUDA_SUCCESS,
            "failed original output need not be read to cancel CPU preparation");
}

void fresh_runtime_creation() {
    CudaEventProjectionRegistry registry;
    CudaEventRegistrationReservation ticket;
    require(registry.status() == CUDA_SUCCESS &&
                registry.prepare_unrecorded_runtime_event(ticket) == CUDA_SUCCESS,
            "normal creation prepares a late-context empty-event owner");
    should_reject_allocation = true;
    const auto published = registry.publish_runtime_event(ticket, kEvent, kContext);
    should_reject_allocation = false;
    const auto fresh = lookup(registry, kEvent);
    require(published == CUDA_SUCCESS && !fresh.is_logical() && fresh.has_native_completion() &&
                fresh.allows_logical_binding() && !fresh.issued_native_prefix() &&
                registry.status() == CUDA_SUCCESS,
            "fresh unrecorded publication allocates nothing and completes no submitted prefix");
    require(registry.invalidate_native_event(fresh) == CUDA_SUCCESS &&
                !lookup(registry, kEvent).has_native_completion() &&
                registry.publish_logical(fresh) == CUDA_ERROR_NOT_READY,
            "a later native effect invalidates the empty-event proof");
    require(registry.forget_owned(fresh) == CUDA_SUCCESS,
            "fresh registration retains exact retirement ownership");
    CudaEventRegistrationReservation reused;
    require(registry.prepare_unrecorded_runtime_event(reused) == CUDA_SUCCESS &&
                registry.publish_runtime_event(reused, kEvent, kOtherContext) == CUDA_SUCCESS &&
                registry.publish_logical(fresh) == CUDA_ERROR_INVALID_HANDLE &&
                lookup(registry, kEvent).has_native_completion(),
            "recycled SDK handles receive new independent empty-event provenance");
    for (bool has_context : {false, true}) {
        CudaEventProjectionRegistry uncertain;
        CudaEventRegistrationReservation pending;
        require(uncertain.prepare_unrecorded_runtime_event(pending) == CUDA_SUCCESS &&
                    uncertain.publish_runtime_event(pending, has_context ? nullptr : kOtherEvent,
                                                    has_context ? kContext : nullptr) ==
                        CUDA_ERROR_UNKNOWN,
                "unproven native creation poisons exact pending ownership");
        pending = {};
        require(uncertain.status() == CUDA_ERROR_UNKNOWN &&
                    uncertain.prepare_unrecorded_runtime_event(pending) == CUDA_ERROR_UNKNOWN,
                "reservation cancellation cannot restore healthy fallback permission");
    }
    CudaEventProjectionRegistry no_issue;
    CudaEventRegistrationReservation failed;
    should_reject_allocation = true;
    const auto rejected = no_issue.prepare_unrecorded_runtime_event(failed);
    should_reject_allocation = false;
    require(rejected == CUDA_ERROR_OUT_OF_MEMORY && !failed && no_issue.status() == CUDA_SUCCESS,
            "pre-issue allocation failure publishes nothing and leaves healthy ownership");
}

void concurrent_preparation() {
    CudaEventProjectionRegistry registry;
    constexpr std::size_t k_callers = 4;
    std::array<std::thread, k_callers> callers;
    std::latch prepared{k_callers};
    std::latch publish{1};
    for (std::size_t index = 0; index < callers.size(); ++index) {
        callers[index] = std::thread([&, index] {
            CudaEventRegistrationReservation ticket;
            require(registry.prepare_native_event(kContext, ticket) == CUDA_SUCCESS,
                    "independent concurrent preparation");
            prepared.count_down();
            publish.wait();
            const auto event = std::bit_cast<CUevent>(std::uintptr_t{0x50000 + index * 16});
            require(registry.publish_native_event(ticket, event, CUDA_SUCCESS) == CUDA_SUCCESS &&
                        lookup(registry, event).context() == kContext,
                    "concurrent original outputs retain exact contexts and unique registrations");
        });
    }
    prepared.wait();
    require(registry.register_event(kOtherContext, kOtherEvent, true) == CUDA_SUCCESS,
            "normal row inserted while all source creations are prepared");
    publish.count_down();
    for (auto& caller : callers) {
        caller.join();
    }
}
void metadata_permissions() {
    CudaEventProjectionRegistry registry;
    require(registry.register_event(kContext, kEvent, true) == CUDA_SUCCESS &&
                registry.register_event(kContext, kOtherEvent, true) == CUDA_SUCCESS,
            "prepare independent metadata identities");
    const std::array forward{lookup(registry, kEvent), lookup(registry, kOtherEvent)};
    const std::array reverse{forward[1], forward[0]};
    unsigned int calls{};
    require(registry.admit_metadata(forward,
                                    [&] {
                                        ++calls;
                                        require(
                                            registry.admit_metadata(reverse,
                                                                    [&] {
                                                                        ++calls;
                                                                        return CUDA_SUCCESS;
                                                                    }) == CUDA_ERROR_INVALID_VALUE,
                                            "same-thread recursion rejects without a callback");
                                        return CUDA_SUCCESS;
                                    }) == CUDA_SUCCESS &&
                calls == 1,
            "two-event permission invokes once outside metadata locks");
    const std::array duplicates{forward[0], forward[0]};
    require(registry.admit_metadata(duplicates, [] { return CUDA_SUCCESS; }) == CUDA_SUCCESS,
            "same-event pair acquires one permission");
    require(registry.admit_metadata(forward,
                                    []() -> CUresult {
                                        throw std::runtime_error("metadata callback failure");
                                    }) == CUDA_ERROR_UNKNOWN &&
                registry.status() == CUDA_SUCCESS &&
                registry.admit_metadata(reverse, [] { return CUDA_SUCCESS; }) == CUDA_SUCCESS,
            "callback exception releases both CPU permissions without poisoning GPU ownership");

    std::atomic<unsigned int> active{}, violations{};
    std::latch started{2};
    const auto run = [&](const auto& pair) {
        started.count_down();
        started.wait();
        for (unsigned int iteration = 0; iteration < 2000; ++iteration) {
            if (registry.admit_metadata(pair, [&] {
                    if (active.fetch_add(1) != 0) {
                        ++violations;
                    }
                    std::this_thread::yield();
                    if (active.fetch_sub(1) != 1) {
                        ++violations;
                    }
                    return CUDA_SUCCESS;
                }) != CUDA_SUCCESS) {
                ++violations;
            }
        }
    };
    std::jthread first([&] { run(forward); });
    std::jthread second([&] { run(reverse); });
    first.join();
    second.join();
    require(violations == 0 && active == 0,
            "opposing pairs serialize 4000 callbacks without reverse-order deadlock");

    std::latch held{1}, release{1}, attempted{1};
    CUresult holder_status{}, waiter_status{};
    std::atomic<unsigned int> late_calls{};
    std::jthread holder([&] {
        holder_status = registry.admit_metadata(std::span{forward.data(), 1}, [&] {
            held.count_down();
            release.wait();
            return CUDA_SUCCESS;
        });
    });
    held.wait();
    std::jthread waiter([&] {
        attempted.count_down();
        waiter_status = registry.admit_metadata(std::span{forward.data(), 1}, [&] {
            ++late_calls;
            return CUDA_SUCCESS;
        });
    });
    attempted.wait();
    require(registry.forget_owned(forward[0]) == CUDA_SUCCESS &&
                registry.register_event(kContext, kEvent, true) == CUDA_SUCCESS,
            "retire and reuse handle while old CPU permit stays owned");
    const auto fresh = lookup(registry, kEvent);
    require(
        registry.admit_metadata(std::span{&fresh, 1}, [] { return CUDA_SUCCESS; }) == CUDA_SUCCESS,
        "recycled registration has an independent permission");
    release.count_down();
    holder.join();
    waiter.join();
    require(holder_status == CUDA_SUCCESS && waiter_status == CUDA_ERROR_INVALID_HANDLE &&
                late_calls == 0,
            "late waiter revalidates exact ownership and never calls against reused handle");
    CudaEventProjectionRegistry foreign;
    require(
        foreign.admit_metadata(reverse, [] { return CUDA_SUCCESS; }) == CUDA_ERROR_INVALID_HANDLE &&
            registry.admit_metadata({}, {}) == CUDA_ERROR_INVALID_VALUE,
        "foreign and missing callbacks reject before invocation");
}
}  // namespace

int main() {
    try {
        generations();
        exact_registration_retirement();
        targeted_epoch_ownership();
        coordinated_callers();
        exact_native_completion();
        prepared_native_registration();
        allocation_free_publication();
        prepared_native_errors();
        concurrent_preparation();
        late_runtime_context();
        fresh_runtime_creation();
        captured_record_scope();
        metadata_permissions();
        std::cout << "native event projection epochs passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
