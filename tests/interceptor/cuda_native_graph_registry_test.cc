#include "internal/cuda_native_graph_registry.h"

#include <array>
#include <barrier>
#include <cstdlib>
#include <iostream>
#include <new>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace {
thread_local std::optional<std::size_t> allocations_before_failure;
}
// Caller-local failures cover staged storage and allocation-free publication;
// concurrent registry workers keep their ordinary allocation behavior.
void* operator new(std::size_t bytes) {
    if (allocations_before_failure) {
        if (*allocations_before_failure == 0) throw std::bad_alloc();
        --*allocations_before_failure;
    }
    if (void* pointer = std::malloc(bytes ? bytes : 1)) return pointer;
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
const auto kExecutable = reinterpret_cast<CUgraphExec>(0x5000);
const auto kSource = reinterpret_cast<CUgraph>(0x6000);
const auto kRecord = reinterpret_cast<CUgraphNode>(0x7000);
const auto kWait = reinterpret_cast<CUgraphNode>(0x8000);
void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}
CUresult CUDAAPI nodes(CUgraph graph, CUgraphNode* output, std::size_t* count) {
    if (graph != kSource || !count) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (output) {
        if (*count != 2) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        output[0] = kRecord;
        output[1] = kWait;
    }
    *count = 2;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI type(CUgraphNode node, CUgraphNodeType* output) {
    if (node != kRecord && node != kWait) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    *output = node == kRecord ? CU_GRAPH_NODE_TYPE_EVENT_RECORD : CU_GRAPH_NODE_TYPE_WAIT_EVENT;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI event(CUgraphNode node, CUevent* output) {
    if (node != kRecord && node != kWait) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    *output = node == kRecord ? kEvent : kOtherEvent;
    return CUDA_SUCCESS;
}
CudaNativeGraphEffects effects() {
    CudaNativeGraphEffects result;
    require(prepare_native_graph_effects(kSource, {nodes, type, nullptr, event, event}, result) ==
                CUDA_SUCCESS,
            "original effects preparation");
    return result;
}
void register_events(CudaEventProjectionRegistry& events) {
    require(events.register_event(kContext, kEvent, true) == CUDA_SUCCESS &&
                events.register_event(kContext, kOtherEvent, true) == CUDA_SUCCESS,
            "actual original event registrations");
}

void event_mutation_replacement() {
    CudaEventProjectionRegistry events;
    register_events(events);
    const auto replacement_event = reinterpret_cast<CUevent>(0x9000);
    require(events.register_event(kContext, replacement_event, true) == CUDA_SUCCESS,
            "register exact replacement event identity");
    CudaNativeGraphRegistry graphs;
    CudaNativeGraphReservation constructor;
    const auto original = effects();
    require(graphs.prepare(kContext, original, events, constructor) == CUDA_SUCCESS &&
                graphs.publish(constructor, kExecutable, CUDA_SUCCESS) == CUDA_SUCCESS,
            "publish original executable before event replacement");
    CudaNativeGraphExecutable before;
    require(graphs.lookup(kExecutable, before) == CUDA_SUCCESS, "retain immutable prior plan");
    for (unsigned int cycle = 0; cycle < 300; ++cycle) {
        CudaNativeGraphReservation mutation;
        require(graphs.prepare_event_update(kExecutable, kRecord, replacement_event,
                                            CudaNativeGraphEventKind::kRecord, events,
                                            mutation) == CUDA_SUCCESS,
                "preallocate event replacement without another constructor slot");
        CudaNativeGraphExecutable excluded;
        require(graphs.lookup(kExecutable, excluded) == CUDA_ERROR_NOT_SUPPORTED,
                "old registration is invalid before SDK mutation");
        CudaNativeGraphRegistry foreign;
        require(foreign.publish_event_update(mutation, CUDA_SUCCESS) == CUDA_ERROR_INVALID_HANDLE &&
                    mutation,
                "foreign publication cannot consume a replacement");
        require(
            graphs.publish_event_update(mutation, CUDA_SUCCESS) == CUDA_SUCCESS && !mutation &&
                graphs.publish_event_update(mutation, CUDA_SUCCESS) == CUDA_ERROR_INVALID_HANDLE,
            "publish replacement once into exact prior slot");
        CudaNativeGraphExecutable current;
        require(graphs.lookup(kExecutable, current) == CUDA_SUCCESS &&
                    current.events()[0].effect.event == replacement_event &&
                    current.effects().events()[0].event == replacement_event &&
                    current.events()[1].effect.event == kOtherEvent,
                "new record identity changes, unrelated wait stays immutable");
    }
    require(before.effects().events()[0].event == kEvent &&
                before.events()[0].effect.event == kEvent &&
                graphs.is_current(before) == CUDA_ERROR_INVALID_HANDLE,
            "retained old snapshot never inherits mutation or new issue permission");
    CudaNativeGraphEffects wrong;
    require(original.replace_event(kRecord, replacement_event, CudaNativeGraphEventKind::kWait,
                                   wrong) == CUDA_ERROR_INVALID_HANDLE &&
                !wrong,
            "event replacement cannot guess node kind");
    CudaNativeGraphReservation failed;
    require(graphs.prepare_event_update(kExecutable, kWait, replacement_event,
                                        CudaNativeGraphEventKind::kWait, events,
                                        failed) == CUDA_SUCCESS &&
                graphs.publish_event_update(failed, CUDA_ERROR_INVALID_VALUE) ==
                    CUDA_ERROR_INVALID_VALUE,
            "original mutation failure is unchanged with no automatic metadata restore");
    CudaNativeGraphExecutable unavailable;
    require(graphs.lookup(kExecutable, unavailable) == CUDA_ERROR_NOT_SUPPORTED,
            "uncertain failed mutation retains invalidation");
    graphs.forget(kExecutable);
    CudaNativeGraphReservation reused;
    require(graphs.prepare(kContext, original, events, reused) == CUDA_SUCCESS &&
                graphs.publish(reused, kExecutable, CUDA_SUCCESS) == CUDA_SUCCESS,
            "reuse executable handle with a fresh registration");
    CudaNativeGraphReservation stale;
    require(
        graphs.prepare_event_update(kExecutable, kWait, replacement_event,
                                    CudaNativeGraphEventKind::kWait, events, stale) == CUDA_SUCCESS,
        "prepare exact association for ABA failure");
    graphs.forget(kExecutable);
    CudaNativeGraphReservation recycled;
    require(graphs.prepare(kContext, original, events, recycled) == CUDA_SUCCESS &&
                graphs.publish(recycled, kExecutable, CUDA_SUCCESS) == CUDA_SUCCESS,
            "a recycled executable cannot receive a prior mutation publication");
    require(graphs.publish_event_update(stale, CUDA_SUCCESS) == CUDA_ERROR_UNKNOWN && !stale,
            "lost post-SDK association poisons ownership without recreating a slot");
    require(graphs.lookup(kExecutable, unavailable) == CUDA_ERROR_UNKNOWN,
            "post-effect ABA cannot authorize the unrelated recycled executable");
}

void event_mutation_identity_and_rejection() {
    CudaEventProjectionRegistry events;
    register_events(events);
    const auto foreign_event = reinterpret_cast<CUevent>(0xa000);
    const auto absent_event = reinterpret_cast<CUevent>(0xb000);
    require(events.register_event(kOtherContext, foreign_event, true) == CUDA_SUCCESS,
            "register a replacement from a different context");
    const auto original = effects();
    auto occupied = original;
    require(original.replace_event(kRecord, kOtherEvent, CudaNativeGraphEventKind::kRecord,
                                   occupied) == CUDA_ERROR_INVALID_VALUE &&
                occupied.events()[0].event == kEvent,
            "occupied effects output stays unchanged");
    for (const auto event : {foreign_event, absent_event}) {
        CudaNativeGraphRegistry graphs;
        CudaNativeGraphReservation constructor;
        require(graphs.prepare(kContext, original, events, constructor) == CUDA_SUCCESS &&
                    graphs.publish(constructor, kExecutable, CUDA_SUCCESS) == CUDA_SUCCESS,
                "publish plan before unknown or foreign replacement");
        CudaNativeGraphReservation mutation;
        require(
            graphs.prepare_event_update(nullptr, kRecord, event, CudaNativeGraphEventKind::kRecord,
                                        events, mutation) == CUDA_ERROR_INVALID_VALUE &&
                graphs.prepare_event_update(reinterpret_cast<CUgraphExec>(0xc000), kRecord, event,
                                            CudaNativeGraphEventKind::kRecord, events,
                                            mutation) == CUDA_ERROR_INVALID_HANDLE,
            "preparation rejects invalid and unknown executable without a ticket");
        CudaNativeGraphExecutable before;
        require(graphs.lookup(kExecutable, before) == CUDA_SUCCESS,
                "pre-issue rejection leaves the original plan prepared");
        require(graphs.prepare_event_update(kExecutable, kRecord, event,
                                            CudaNativeGraphEventKind::kRecord, events,
                                            mutation) == CUDA_SUCCESS &&
                    graphs.prepare_event_update(kExecutable, kWait, kEvent,
                                                CudaNativeGraphEventKind::kWait, events,
                                                mutation) == CUDA_ERROR_INVALID_VALUE,
                "unknown binding preserves SDK validation and occupied ticket identity");
        auto moved = std::move(mutation);
        // NOLINTNEXTLINE(bugprone-use-after-move): ticket move emptiness is required.
        require(!mutation && moved &&
                    graphs.publish_event_update(moved, CUDA_SUCCESS) == CUDA_SUCCESS && !moved,
                "moved replacement publishes exactly once");
        CudaNativeGraphExecutable unprepared;
        require(graphs.lookup(kExecutable, unprepared) == CUDA_ERROR_NOT_SUPPORTED &&
                    before.effects().events()[0].event == kEvent,
                "SDK success cannot guess absent or foreign event ownership");
        CudaNativeGraphReservation still_unprepared;
        require(graphs.prepare_event_update(kExecutable, kRecord, kEvent,
                                            CudaNativeGraphEventKind::kRecord, events,
                                            still_unprepared) == CUDA_SUCCESS &&
                    graphs.publish_event_update(still_unprepared, CUDA_SUCCESS) == CUDA_SUCCESS &&
                    graphs.lookup(kExecutable, unprepared) == CUDA_ERROR_NOT_SUPPORTED,
                "known subsequent event cannot reconstruct an already unprepared graph");
    }
    CudaNativeGraphRegistry graphs;
    CudaNativeGraphReservation constructor;
    require(graphs.prepare(kContext, original, events, constructor) == CUDA_SUCCESS &&
                graphs.publish(constructor, kExecutable, CUDA_SUCCESS) == CUDA_SUCCESS,
            "publish graph before event ABA");
    CudaNativeGraphReservation mutation;
    require(graphs.prepare_event_update(kExecutable, kRecord, kOtherEvent,
                                        CudaNativeGraphEventKind::kRecord, events,
                                        mutation) == CUDA_SUCCESS,
            "prepare exact replacement event registration");
    events.forget(kOtherEvent);
    require(events.register_event(kContext, kOtherEvent, true) == CUDA_SUCCESS &&
                graphs.publish_event_update(mutation, CUDA_SUCCESS) == CUDA_SUCCESS,
            "same raw event handle can be recycled before publication");
    CudaNativeGraphExecutable current;
    CudaEventProjection refreshed;
    require(graphs.lookup(kExecutable, current) == CUDA_SUCCESS &&
                events.lookup_owned(current.events()[0].registration, refreshed) ==
                    CUDA_ERROR_INVALID_HANDLE &&
                !refreshed,
            "retained replacement identity cannot borrow the recycled event's epoch");
}

void event_mutation_allocation_failure() {
    CudaEventProjectionRegistry events;
    register_events(events);
    const auto original = effects();
    bool has_prepared{};
    std::size_t rejected_allocations{};
    for (std::size_t permitted = 0; permitted < 16; ++permitted) {
        CudaNativeGraphRegistry graphs;
        CudaNativeGraphReservation constructor;
        require(graphs.prepare(kContext, original, events, constructor) == CUDA_SUCCESS &&
                    graphs.publish(constructor, kExecutable, CUDA_SUCCESS) == CUDA_SUCCESS,
                "publish original plan before allocation injection");
        CudaNativeGraphExecutable before;
        require(graphs.lookup(kExecutable, before) == CUDA_SUCCESS,
                "retain pre-mutation registration identity");
        CudaNativeGraphReservation mutation;
        allocations_before_failure = permitted;
        const auto status = graphs.prepare_event_update(
            kExecutable, kRecord, kOtherEvent, CudaNativeGraphEventKind::kRecord, events, mutation);
        allocations_before_failure.reset();
        if (status == CUDA_ERROR_OUT_OF_MEMORY) {
            ++rejected_allocations;
            require(!mutation && graphs.is_current(before) == CUDA_SUCCESS &&
                        before.effects().events()[0].event == kEvent,
                    "every pre-issue allocation failure leaves the prior plan usable");
            continue;
        }
        require(status == CUDA_SUCCESS && mutation,
                "finite staged preparation eventually succeeds");
        allocations_before_failure = 0;
        const auto published = graphs.publish_event_update(mutation, CUDA_SUCCESS);
        allocations_before_failure.reset();
        CudaNativeGraphExecutable current;
        require(published == CUDA_SUCCESS && !mutation &&
                    graphs.lookup(kExecutable, current) == CUDA_SUCCESS &&
                    current.effects().events()[0].event == kOtherEvent &&
                    graphs.is_current(before) == CUDA_ERROR_INVALID_HANDLE,
                "post-SDK publication and old ownership retirement require no allocation");
        has_prepared = true;
        break;
    }
    require(has_prepared && rejected_allocations > 0,
            "fault injection traverses staged allocations and reaches publication");
}

void whole_mutation_replacement() {
    CudaEventProjectionRegistry events;
    register_events(events);
    const auto next = reinterpret_cast<CUevent>(0x9000);
    require(events.register_event(kContext, next, true) == CUDA_SUCCESS,
            "register whole update's exact new record event");
    const auto original = effects();
    CudaNativeGraphEffects replacement;
    require(original.replace_event(kRecord, next, CudaNativeGraphEventKind::kRecord, replacement) ==
                CUDA_SUCCESS,
            "prepare complete known replacement effects");
    bool has_published{};
    std::size_t failures{};
    for (std::size_t permitted = 0; permitted < 32; ++permitted) {
        CudaNativeGraphRegistry graphs;
        CudaNativeGraphReservation constructor;
        require(graphs.prepare(kContext, original, events, constructor) == CUDA_SUCCESS &&
                    graphs.publish(constructor, kExecutable, CUDA_SUCCESS) == CUDA_SUCCESS,
                "publish original whole-update plan");
        CudaNativeGraphExecutable before;
        require(graphs.lookup(kExecutable, before) == CUDA_SUCCESS, "retain old whole plan");
        CudaNativeGraphReservation mutation;
        allocations_before_failure = permitted;
        const auto prepared =
            graphs.prepare_graph_update(kExecutable, replacement, std::nullopt, events, mutation);
        allocations_before_failure.reset();
        if (prepared == CUDA_ERROR_OUT_OF_MEMORY) {
            ++failures;
            require(!mutation && graphs.is_current(before) == CUDA_SUCCESS,
                    "every whole-update allocation failure preserves prior permission");
            continue;
        }
        require(prepared == CUDA_SUCCESS && mutation, "finite whole-update preparation succeeds");
        allocations_before_failure = 0;
        const auto published = graphs.publish_event_update(mutation, CUDA_SUCCESS);
        allocations_before_failure.reset();
        CudaNativeGraphExecutable current;
        require(published == CUDA_SUCCESS && graphs.lookup(kExecutable, current) == CUDA_SUCCESS &&
                    current.events()[0].effect.event == next &&
                    current.events()[0].effect.node == kRecord &&
                    before.events()[0].effect.event == kEvent &&
                    graphs.is_current(before) == CUDA_ERROR_INVALID_HANDLE,
                "whole publication replaces all event bindings without allocation or snapshot "
                "mutation");
        graphs.invalidate(kExecutable);
        require(graphs.prepare_graph_update(kExecutable, replacement, kRecord, events, mutation) ==
                        CUDA_SUCCESS &&
                    graphs.publish_event_update(mutation, CUDA_SUCCESS) == CUDA_SUCCESS,
                "partial child success cannot reconstruct an uncertain parent");
        CudaNativeGraphExecutable unavailable;
        require(graphs.lookup(kExecutable, unavailable) == CUDA_ERROR_NOT_SUPPORTED,
                "partial metadata remains unprepared");
        require(
            graphs.prepare_graph_update(kExecutable, replacement, std::nullopt, events, mutation) ==
                    CUDA_SUCCESS &&
                graphs.publish_event_update(mutation, CUDA_SUCCESS) == CUDA_SUCCESS &&
                graphs.lookup(kExecutable, unavailable) == CUDA_SUCCESS,
            "successful complete known whole update reconstructs exact supported origin effects");
        has_published = true;
        break;
    }
    require(has_published && failures > 0,
            "whole-update allocation/publication proof is exercised");
}

const auto kParentSource = reinterpret_cast<CUgraph>(0xd000);
const auto kChildNode = reinterpret_cast<CUgraphNode>(0xe000);
CUresult CUDAAPI child_nodes(CUgraph graph, CUgraphNode* output, std::size_t* count) {
    if (graph == kSource) return nodes(graph, output, count);
    if (graph != kParentSource || !count || (output && *count != 1))
        return CUDA_ERROR_INVALID_VALUE;
    if (output) *output = kChildNode;
    *count = 1;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI child_type(CUgraphNode node, CUgraphNodeType* output) {
    if (node != kChildNode) return type(node, output);
    *output = CU_GRAPH_NODE_TYPE_GRAPH;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI child_graph(CUgraphNode node, CUgraph* output) {
    if (node != kChildNode || !output) return CUDA_ERROR_INVALID_VALUE;
    *output = kSource;
    return CUDA_SUCCESS;
}
void child_mutation_allocation_rejection() {
    CudaEventProjectionRegistry events;
    register_events(events);
    const auto next = reinterpret_cast<CUevent>(0x9000);
    require(events.register_event(kContext, next, true) == CUDA_SUCCESS,
            "register child update's new exact event binding");
    CudaNativeGraphEffects original, replacement;
    require(prepare_native_graph_effects(kParentSource,
                                         {child_nodes, child_type, child_graph, event, event},
                                         original) == CUDA_SUCCESS &&
                effects().replace_event(kRecord, next, CudaNativeGraphEventKind::kRecord,
                                        replacement) == CUDA_SUCCESS,
            "prepare original child hierarchy and complete replacement before SDK issue");
    std::size_t failures{};
    bool has_published{};
    for (std::size_t permitted = 0; permitted < 32; ++permitted) {
        CudaNativeGraphRegistry graphs;
        CudaNativeGraphReservation constructor, mutation;
        require(graphs.prepare(kContext, original, events, constructor) == CUDA_SUCCESS &&
                    graphs.publish(constructor, kExecutable, CUDA_SUCCESS) == CUDA_SUCCESS,
                "publish original child plan before allocation injection");
        CudaNativeGraphExecutable previous;
        require(graphs.lookup(kExecutable, previous) == CUDA_SUCCESS,
                "retain child snapshot before rejecting staging allocations");
        allocations_before_failure = permitted;
        const auto prepared =
            graphs.prepare_graph_update(kExecutable, replacement, kChildNode, events, mutation);
        allocations_before_failure.reset();
        if (prepared == CUDA_ERROR_OUT_OF_MEMORY) {
            ++failures;
            require(!mutation && graphs.is_current(previous) == CUDA_SUCCESS &&
                        previous.events()[0].effect.event == kEvent,
                    "every child staging allocation rejection preserves prior permission");
            continue;
        }
        require(prepared == CUDA_SUCCESS && mutation, "finite child staging succeeds");
        allocations_before_failure = 0;
        const auto published = graphs.publish_event_update(mutation, CUDA_SUCCESS);
        allocations_before_failure.reset();
        CudaNativeGraphExecutable current;
        require(published == CUDA_SUCCESS && graphs.lookup(kExecutable, current) == CUDA_SUCCESS &&
                    current.effects().node_count() == 3 &&
                    current.events()[0].effect.event == next &&
                    current.events()[0].effect.node == kRecord &&
                    current.events()[1].effect.event == kOtherEvent &&
                    graphs.is_current(previous) == CUDA_ERROR_INVALID_HANDLE &&
                    previous.events()[0].effect.event == kEvent,
                "child publication is allocation-free, exact and immutable");
        has_published = true;
        break;
    }
    require(has_published && failures != 0,
            "child hierarchy allocation rejection and successful publication are both exercised");
}

void whole_update_qualification() {
    CudaEventProjectionRegistry events;
    register_events(events);
    const auto absent = reinterpret_cast<CUevent>(0xa000);
    const auto foreign = reinterpret_cast<CUevent>(0xb000);
    require(events.register_event(kOtherContext, foreign, true) == CUDA_SUCCESS,
            "register actual foreign-context replacement event");
    const auto original = effects();
    for (const auto new_event : {absent, foreign}) {
        CudaNativeGraphRegistry graphs;
        CudaNativeGraphReservation constructor, mutation;
        require(graphs.prepare(kContext, original, events, constructor) == CUDA_SUCCESS &&
                    graphs.publish(constructor, kExecutable, CUDA_SUCCESS) == CUDA_SUCCESS,
                "publish supported origin before incomplete whole binding");
        CudaNativeGraphEffects replacement;
        require(original.replace_event(kRecord, new_event, CudaNativeGraphEventKind::kRecord,
                                       replacement) == CUDA_SUCCESS &&
                    graphs.prepare_graph_update(kExecutable, replacement, std::nullopt, events,
                                                mutation) == CUDA_SUCCESS &&
                    graphs.publish_event_update(mutation, CUDA_SUCCESS) == CUDA_SUCCESS,
                "unknown or foreign whole effects still preserve native validation");
        CudaNativeGraphExecutable unavailable;
        require(graphs.lookup(kExecutable, unavailable) == CUDA_ERROR_NOT_SUPPORTED,
                "complete SDK success alone does not prove event registration context");
        require(graphs.prepare_graph_update(kExecutable, original, std::nullopt, events,
                                            mutation) == CUDA_SUCCESS &&
                    graphs.publish_event_update(mutation, CUDA_ERROR_LAUNCH_FAILED) ==
                        CUDA_ERROR_LAUNCH_FAILED &&
                    graphs.lookup(kExecutable, unavailable) == CUDA_ERROR_NOT_SUPPORTED,
                "whole SDK error retains exact native status and grants no permission");
        require(
            graphs.prepare_graph_update(kExecutable, original, std::nullopt, events, mutation) ==
                    CUDA_SUCCESS &&
                graphs.publish_event_update(mutation, CUDA_SUCCESS) == CUDA_SUCCESS &&
                graphs.lookup(kExecutable, unavailable) == CUDA_SUCCESS,
            "known complete supported effects recover metadata, not physical failure ownership");
    }
    CudaNativeGraphRegistry unknown;
    CudaNativeGraphReservation constructor, mutation;
    require(unknown.prepare(kContext, {}, events, constructor) == CUDA_SUCCESS &&
                unknown.publish(constructor, kExecutable, CUDA_SUCCESS) == CUDA_SUCCESS &&
                unknown.prepare_graph_update(kExecutable, original, std::nullopt, events,
                                             mutation) == CUDA_SUCCESS &&
                unknown.publish_event_update(mutation, CUDA_SUCCESS) == CUDA_SUCCESS,
            "unknown constructor origin receives original whole-update validation");
    CudaNativeGraphExecutable unavailable;
    require(unknown.lookup(kExecutable, unavailable) == CUDA_ERROR_NOT_SUPPORTED,
            "whole update cannot upgrade unmodeled constructor flags or origin contracts");
}

CUresult CUDAAPI empty_nodes(CUgraph graph, CUgraphNode*, std::size_t* count) {
    if (graph != kSource || !count) return CUDA_ERROR_INVALID_VALUE;
    *count = 0;
    return CUDA_SUCCESS;
}
void whole_update_origin_proof() {
    CudaEventProjectionRegistry events;
    register_events(events);
    CudaNativeGraphEffects empty;
    require(prepare_native_graph_effects(kSource, {empty_nodes, type, nullptr, nullptr, nullptr},
                                         empty) == CUDA_SUCCESS,
            "known empty metadata is distinct from unknown constructor effects");
    for (const bool is_contextless : {true, false}) {
        CudaNativeGraphRegistry graphs;
        CudaNativeGraphReservation constructor, mutation;
        const auto original = is_contextless ? empty : effects();
        const auto prepared = is_contextless
                                  ? graphs.prepare_unbound(original, events, constructor)
                                  : graphs.prepare(kContext, original, events, constructor);
        require(prepared == CUDA_SUCCESS, "prepare supported effects before uncertain SDK origin");
        const auto native = is_contextless ? CUDA_SUCCESS : CUDA_ERROR_LAUNCH_FAILED;
        require(graphs.publish(constructor, kExecutable, native,
                               is_contextless ? std::optional<CUcontext>{nullptr} : std::nullopt) ==
                    native,
                "retain contextless success or failed constructor's actual output");
        require(graphs.prepare_graph_update(kExecutable, original, std::nullopt, events,
                                            mutation) == CUDA_SUCCESS &&
                    graphs.publish_event_update(mutation, CUDA_SUCCESS) == CUDA_SUCCESS,
                "uncertain construction still permits original SDK update validation");
        CudaNativeGraphExecutable unavailable;
        require(graphs.lookup(kExecutable, unavailable) == CUDA_ERROR_NOT_SUPPORTED,
                "whole update cannot invent constructor or context proof from metadata alone");
    }
}

CUresult CUDAAPI non_event_type(CUgraphNode node, CUgraphNodeType* output) {
    if (node != kRecord && node != kWait) return CUDA_ERROR_INVALID_HANDLE;
    *output = node == kRecord ? CU_GRAPH_NODE_TYPE_KERNEL : CU_GRAPH_NODE_TYPE_MEMCPY;
    return CUDA_SUCCESS;
}

void non_event_mutation_replacement() {
    CudaNativeGraphEffects non_events;
    require(
        prepare_native_graph_effects(kSource, {nodes, non_event_type, nullptr, nullptr, nullptr},
                                     non_events) == CUDA_SUCCESS,
        "retain actual non-event node identities and types before construction");
    CudaEventProjectionRegistry events;
    CudaNativeGraphRegistry graphs;
    CudaNativeGraphReservation constructor;
    require(graphs.prepare(kContext, non_events, events, constructor) == CUDA_SUCCESS &&
                graphs.publish(constructor, kExecutable, CUDA_SUCCESS) == CUDA_SUCCESS,
            "publish kernel/copy plan with no event dependencies");
    CudaNativeGraphExecutable before;
    require(graphs.lookup(kExecutable, before) == CUDA_SUCCESS,
            "retain prior plan before generic mutation");
    CudaNativeGraphReservation mutation;
    allocations_before_failure = 0;
    const auto rejected =
        graphs.prepare_node_update(kExecutable, kRecord, CU_GRAPH_NODE_TYPE_KERNEL, mutation);
    allocations_before_failure.reset();
    require(rejected == CUDA_ERROR_OUT_OF_MEMORY && !mutation &&
                graphs.is_current(before) == CUDA_SUCCESS,
            "non-event preparation allocation failure cannot invalidate the old plan");
    for (const auto expected :
         {std::optional{CU_GRAPH_NODE_TYPE_KERNEL}, std::optional<CUgraphNodeType>{}}) {
        require(
            graphs.prepare_node_update(kExecutable, kRecord, expected, mutation) == CUDA_SUCCESS,
            "generic kernel parameters and enablement prepare unchanged dependencies");
        allocations_before_failure = 0;
        const auto published = graphs.publish_event_update(mutation, CUDA_SUCCESS);
        allocations_before_failure.reset();
        CudaNativeGraphExecutable current;
        require(published == CUDA_SUCCESS && graphs.lookup(kExecutable, current) == CUDA_SUCCESS &&
                    current.effects().node_count() == 2 && current.events().empty() &&
                    graphs.is_current(before) == CUDA_ERROR_INVALID_HANDLE,
                "successful node modification replaces generation without post-issue allocation");
    }
    CudaNativeGraphEffects mismatched;
    require(
        non_events.preserve_node(kRecord, CU_GRAPH_NODE_TYPE_MEMCPY, mismatched) ==
                CUDA_ERROR_NOT_SUPPORTED &&
            !mismatched &&
            non_events.preserve_node(reinterpret_cast<CUgraphNode>(0xc000), std::nullopt,
                                     mismatched) == CUDA_ERROR_INVALID_HANDLE &&
            effects().preserve_node(kRecord, std::nullopt, mismatched) == CUDA_ERROR_NOT_SUPPORTED,
        "caller type, unknown node and event enablement cannot guess preserved dependencies");
    require(graphs.prepare_node_update(kExecutable, kRecord, CU_GRAPH_NODE_TYPE_MEMSET, mutation) ==
                    CUDA_SUCCESS &&
                graphs.publish_event_update(mutation, CUDA_SUCCESS) == CUDA_SUCCESS,
            "unproven type still retains original SDK validation");
    CudaNativeGraphExecutable unprepared;
    require(graphs.lookup(kExecutable, unprepared) == CUDA_ERROR_NOT_SUPPORTED,
            "SDK success with unmatched metadata cannot publish guessed dependencies");
}

void ownership_and_event_aba() {
    CudaEventProjectionRegistry events;
    register_events(events);
    CudaNativeGraphRegistry graphs;
    CudaNativeGraphReservation reservation;
    auto source = effects();
    require(graphs.prepare(kContext, source, events, reservation) == CUDA_SUCCESS && reservation,
            "prepare all CPU ownership before original construction");
    graphs.invalidate(nullptr);
    graphs.forget(nullptr);
    require(graphs.publish(reservation, kExecutable, CUDA_SUCCESS) == CUDA_SUCCESS,
            "publish actual executable with preallocated ownership");
    source = {};
    CudaNativeGraphExecutable owned;
    require(graphs.lookup(kExecutable, owned) == CUDA_SUCCESS && owned &&
                owned.context() == kContext && owned.effects().node_count() == 2 &&
                owned.events().size() == 2 && owned.events()[0].effect.node == kRecord &&
                owned.events()[1].effect.kind == CudaNativeGraphEventKind::kWait,
            "source retirement cannot erase immutable executable effects");
    require(graphs.lookup(kExecutable, owned) == CUDA_ERROR_INVALID_VALUE,
            "occupied output cannot be overwritten");
    require(events.invalidate_native_event(owned.events()[0].registration) == CUDA_SUCCESS,
            "a native epoch can advance without replacing registration");
    CudaEventProjection refreshed;
    require(events.lookup_owned(owned.events()[0].registration, refreshed) == CUDA_SUCCESS &&
                !refreshed.is_logical() && !refreshed.has_native_completion(),
            "refresh must obtain current epoch without trusting a raw handle");
    bool is_current = true;
    require(
        events.matches_native_epoch(owned.events()[0].registration, is_current) == CUDA_SUCCESS &&
            !is_current,
        "old epoch remains superseded");
    require(
        events.lookup_owned(owned.events()[0].registration, refreshed) == CUDA_ERROR_INVALID_VALUE,
        "occupied refreshed output rejects");
    events.forget(kEvent);
    require(events.register_event(kContext, kEvent, true) == CUDA_SUCCESS,
            "native event handle reuse");
    CudaEventProjection replacement;
    require(events.lookup_owned(owned.events()[0].registration, replacement) ==
                    CUDA_ERROR_INVALID_HANDLE &&
                !replacement,
            "graph must not bind a recycled new event registration");
    CudaEventProjectionRegistry foreign;
    register_events(foreign);
    require(foreign.lookup_owned(owned.events()[1].registration, replacement) ==
                CUDA_ERROR_INVALID_HANDLE,
            "foreign event registry identity rejects");
    graphs.invalidate(kExecutable);
    CudaNativeGraphExecutable after_mutation;
    require(graphs.lookup(kExecutable, after_mutation) == CUDA_ERROR_NOT_SUPPORTED &&
                !after_mutation && graphs.is_current(owned) == CUDA_ERROR_NOT_SUPPORTED &&
                owned.effects().node_count() == 2 && owned.events().size() == 2,
            "mutation invalidates future routing without erasing retained CPU snapshot");
    graphs.forget(kExecutable);
    require(graphs.is_current(owned) == CUDA_ERROR_INVALID_HANDLE,
            "original executable destruction retires its registration");
    CudaNativeGraphReservation reused;
    require(graphs.prepare(kContext, effects(), events, reused) == CUDA_SUCCESS &&
                graphs.publish(reused, kExecutable, CUDA_SUCCESS) == CUDA_SUCCESS &&
                graphs.is_current(owned) == CUDA_ERROR_INVALID_HANDLE,
            "recycled executable handle cannot revive an old owner");
    CudaNativeGraphExecutable current;
    require(graphs.lookup(kExecutable, current) == CUDA_SUCCESS &&
                graphs.is_current(current) == CUDA_SUCCESS,
            "fresh executable ownership");
    const auto copied = current;
    auto moved = std::move(current);
    // NOLINTNEXTLINE(bugprone-use-after-move): empty moved-from snapshot is a documented invariant.
    require(!current && copied.events().size() == 2 && moved.events().size() == 2,
            "snapshot move/copy owns CPU data only");
    graphs.forget_context(kOtherContext);
    require(graphs.is_current(moved) == CUDA_SUCCESS, "unrelated context is unchanged");
    graphs.forget_context(kContext);
    require(graphs.is_current(moved) == CUDA_ERROR_INVALID_HANDLE && moved.events().size() == 2,
            "checked context teardown retires lookup not CPU data");
}

void late_context_publication() {
    CudaEventProjectionRegistry events;
    register_events(events);
    CudaNativeGraphRegistry graphs;
    CudaNativeGraphReservation pending;
    require(graphs.prepare_unbound(effects(), events, pending) == CUDA_SUCCESS && pending,
            "SDK metadata is completely prepared before lazy context binding");
    require(graphs.prepare_unbound({}, events, pending) == CUDA_ERROR_INVALID_VALUE,
            "late-bound occupied reservation rejects");
    require(graphs.publish(pending, kExecutable, CUDA_SUCCESS, kContext) == CUDA_SUCCESS,
            "actual post-SDK context and native handle publish atomically");
    CudaNativeGraphExecutable owned;
    require(graphs.lookup(kExecutable, owned) == CUDA_SUCCESS && owned.context() == kContext &&
                owned.events().size() == 2,
            "late context matches every prepared original event registration");
    graphs.forget(kExecutable);
    require(graphs.is_current(owned) == CUDA_ERROR_INVALID_HANDLE,
            "late-published snapshots retain ordinary executable lifetime checks");
    CudaNativeGraphReservation cross_context;
    require(
        graphs.prepare_unbound(effects(), events, cross_context) == CUDA_SUCCESS &&
            graphs.publish(cross_context, kExecutable, CUDA_SUCCESS, kOtherContext) == CUDA_SUCCESS,
        "unsupported event/context combination preserves native constructor success");
    CudaNativeGraphExecutable unsupported;
    require(graphs.lookup(kExecutable, unsupported) == CUDA_ERROR_NOT_SUPPORTED,
            "cross-context native construction supplies no replay permission");
    graphs.forget(kExecutable);
    CudaNativeGraphReservation contextless;
    require(graphs.prepare_unbound({}, events, contextless) == CUDA_SUCCESS &&
                graphs.publish(contextless, kExecutable, CUDA_SUCCESS,
                               std::optional<CUcontext>{nullptr}) == CUDA_SUCCESS &&
                graphs.lookup(kExecutable, unsupported) == CUDA_ERROR_NOT_SUPPORTED,
            "known successful unbound observation keeps native handle but grants no replay");
    graphs.forget(kExecutable);
    CudaNativeGraphReservation unpublished;
    require(graphs.prepare_unbound({}, events, unpublished) == CUDA_SUCCESS &&
                graphs.publish(unpublished, nullptr, CUDA_ERROR_INVALID_VALUE) ==
                    CUDA_ERROR_INVALID_VALUE,
            "failed SDK without a native handle needs no context and cancels its CPU slot");
    for (const auto error : {CUDA_SUCCESS, CUDA_ERROR_LAUNCH_FAILED}) {
        CudaNativeGraphRegistry uncertain;
        CudaNativeGraphReservation missing;
        require(uncertain.prepare_unbound(effects(), events, missing) == CUDA_SUCCESS &&
                    uncertain.publish(missing, kExecutable, error) ==
                        (error == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : error),
                "unobserved post-SDK context retains published handle and original error");
        uncertain.forget(kExecutable);
        uncertain.forget_context(kContext);
        missing = {};
        CudaNativeGraphReservation next;
        require(uncertain.lookup(kExecutable, unsupported) == CUDA_ERROR_UNKNOWN &&
                    uncertain.prepare_unbound({}, events, next) == CUDA_ERROR_UNKNOWN,
                "unknown ownership cannot be cleared by scope cancellation or guessed cleanup");
    }
    CudaNativeGraphReservation bound;
    require(
        graphs.prepare(kContext, effects(), events, bound) == CUDA_SUCCESS &&
            graphs.publish(bound, kExecutable, CUDA_SUCCESS, kOtherContext) == CUDA_ERROR_UNKNOWN,
        "a known Driver context cannot be silently replaced during publication");
}

void bounded_cancellation() {
    CudaEventProjectionRegistry events;
    CudaNativeGraphRegistry graphs;
    std::array<CudaNativeGraphReservation, 256> reservations;
    for (auto& reservation : reservations) {
        require(graphs.prepare(kContext, {}, events, reservation) == CUDA_SUCCESS,
                "exact bounded slot count");
    }
    CudaNativeGraphReservation extra;
    require(graphs.prepare(kContext, {}, events, extra) == CUDA_ERROR_OUT_OF_MEMORY && !extra,
            "capacity must reject before original construction");
    require(graphs.prepare(kContext, {}, events, reservations[0]) == CUDA_ERROR_INVALID_VALUE,
            "occupied constructor reservation rejects");
    reservations[0] = {};
    require(graphs.prepare(kContext, {}, events, extra) == CUDA_SUCCESS,
            "CPU scope cancellation returns exactly one slot");
    CudaNativeGraphRegistry foreign;
    require(foreign.publish(extra, kExecutable, CUDA_SUCCESS) == CUDA_ERROR_INVALID_HANDLE,
            "foreign constructor ticket rejects");
    require(graphs.publish(extra, kExecutable, CUDA_SUCCESS) == CUDA_SUCCESS,
            "the original owner can still publish");
    CudaNativeGraphExecutable output;
    require(graphs.lookup(kExecutable, output) == CUDA_ERROR_NOT_SUPPORTED && !output,
            "unknown effects remain explicitly unprepared");
    require(graphs.publish(extra, kExecutable, CUDA_SUCCESS) == CUDA_ERROR_INVALID_HANDLE,
            "original construction publication cannot repeat");
    extra = {};
    require(graphs.prepare(kContext, {}, events, extra) == CUDA_ERROR_OUT_OF_MEMORY,
            "destroying a committed ticket cannot retire its native object");
    graphs.forget(kExecutable);
    require(graphs.prepare(kContext, {}, events, extra) == CUDA_SUCCESS,
            "checked original destruction reclaims slot");
}

void partial_native_construction() {
    CudaEventProjectionRegistry events;
    register_events(events);
    CudaNativeGraphRegistry graphs;
    {
        CudaNativeGraphReservation failed;
        require(graphs.prepare(kContext, effects(), events, failed) == CUDA_SUCCESS &&
                    graphs.publish(failed, nullptr, CUDA_ERROR_INVALID_CONTEXT) ==
                        CUDA_ERROR_INVALID_CONTEXT,
                "known unpublished native error is exact");
    }
    CudaNativeGraphReservation partial;
    require(graphs.prepare(kContext, effects(), events, partial) == CUDA_SUCCESS &&
                graphs.publish(partial, kExecutable, CUDA_ERROR_LAUNCH_FAILED) ==
                    CUDA_ERROR_LAUNCH_FAILED,
            "published-on-error native status is exact");
    CudaNativeGraphExecutable output;
    require(graphs.lookup(kExecutable, output) == CUDA_ERROR_NOT_SUPPORTED && !output,
            "published-on-error object retains no trusted effects");
    graphs.forget(kExecutable);
    CudaNativeGraphReservation valid;
    require(graphs.prepare(kContext, effects(), events, valid) == CUDA_SUCCESS &&
                graphs.publish(valid, kExecutable, CUDA_SUCCESS) == CUDA_SUCCESS,
            "checked cleanup permits native handle reuse");
    CudaNativeGraphReservation duplicate;
    require(graphs.prepare(kContext, effects(), events, duplicate) == CUDA_SUCCESS &&
                graphs.publish(duplicate, kExecutable, CUDA_SUCCESS) == CUDA_ERROR_UNKNOWN &&
                graphs.lookup(kExecutable, output) == CUDA_ERROR_UNKNOWN,
            "duplicate live native identity quarantines instead of guessing ownership");
    CudaNativeGraphRegistry null_graphs;
    CudaNativeGraphReservation null_success;
    require(null_graphs.prepare(kContext, effects(), events, null_success) == CUDA_SUCCESS &&
                null_graphs.publish(null_success, nullptr, CUDA_SUCCESS) == CUDA_ERROR_UNKNOWN &&
                null_graphs.lookup(kExecutable, output) == CUDA_ERROR_UNKNOWN,
            "successful unwritten/null constructor output fails closed");
}

void concurrent_metadata() {
    CudaEventProjectionRegistry events;
    register_events(events);
    CudaNativeGraphRegistry graphs;
    CudaNativeGraphReservation reservation;
    require(graphs.prepare(kContext, effects(), events, reservation) == CUDA_SUCCESS &&
                graphs.publish(reservation, kExecutable, CUDA_SUCCESS) == CUDA_SUCCESS,
            "prepare shared immutable CPU owner");
    std::barrier rendezvous(5);
    std::array<bool, 4> passed{};
    std::array<std::thread, 4> readers;
    for (std::size_t i = 0; i < readers.size(); ++i) {
        readers[i] = std::thread([&, i] {
            CudaNativeGraphExecutable output;
            passed[i] = graphs.lookup(kExecutable, output) == CUDA_SUCCESS;
            rendezvous.arrive_and_wait();
            rendezvous.arrive_and_wait();
            passed[i] = passed[i] && output.events().size() == 2 &&
                        graphs.is_current(output) == CUDA_ERROR_NOT_SUPPORTED;
        });
    }
    rendezvous.arrive_and_wait();
    graphs.invalidate(kExecutable);
    rendezvous.arrive_and_wait();
    for (auto& reader : readers) {
        reader.join();
    }
    for (bool result : passed) {
        require(result, "immutable snapshots survive serialized concurrent metadata invalidation");
    }
}
}  // namespace

int main() {
    try {
        event_mutation_replacement();
        event_mutation_identity_and_rejection();
        event_mutation_allocation_failure();
        non_event_mutation_replacement();
        whole_mutation_replacement();
        whole_update_qualification();
        child_mutation_allocation_rejection();
        whole_update_origin_proof();
        ownership_and_event_aba();
        late_context_publication();
        bounded_cancellation();
        partial_native_construction();
        concurrent_metadata();
        // Explicit validation cases avoid re-registering an already owned event.
        CudaEventProjectionRegistry events;
        register_events(events);
        CudaNativeGraphRegistry graphs;
        CudaNativeGraphReservation output;
        require(graphs.prepare(nullptr, {}, events, output) == CUDA_ERROR_INVALID_VALUE &&
                    graphs.prepare(kOtherContext, effects(), events, output) ==
                        CUDA_ERROR_NOT_SUPPORTED &&
                    !output,
                "invalid/context rejection");
        events.forget(kEvent);
        require(graphs.prepare(kContext, effects(), events, output) == CUDA_ERROR_INVALID_HANDLE &&
                    !output,
                "missing origin rejects transactional preparation");
        std::cout << "native graph executable registration passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
