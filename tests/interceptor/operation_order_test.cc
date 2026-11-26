#include "internal/operation_order.h"

#include <array>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace {
using glimmer::interceptor::OperationOrder;
using glimmer::interceptor::OrderedOperationState;
using glimmer::interceptor::OrderedOperationToken;
using glimmer::interceptor::OrderedStreamKind;
using glimmer::interceptor::PhysicalExecutionTicket;

void require(bool value, const char* message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}
PhysicalExecutionTicket start(OperationOrder& order, const OrderedOperationToken& operation) {
    const auto ticket = order.start(operation);
    if (!ticket) throw std::runtime_error("ready operation must start");
    return *ticket;
}
void finish(OperationOrder& order, const OrderedOperationToken& operation) {
    require(order.complete(start(order, operation), true),
            "matching logical completion must succeed");
}

void continuation_keeps_following_operations_blocked() {
    OperationOrder order;
    const auto stream = order.create_stream(OrderedStreamKind::kNonBlocking);
    const auto kernel = order.submit(stream, {}, true);
    const auto copy = order.submit(stream);
    const auto event_record = order.submit(stream);
    const auto free = order.submit(stream);
    require(kernel && copy && event_record && free, "operation submissions must succeed");
    auto execution = start(order, kernel);
    require(!order.fail_stopped(kernel), "running work cannot be failed as physically stopped");
    for (int iteration = 0; iteration < 16; ++iteration) {
        const auto previous = execution;
        require(order.suspend(execution), "cooperative physical stop must suspend");
        require(order.state(kernel) == OrderedOperationState::kSuspended,
                "suspension must not publish logical completion");
        require(order.pending_count() == 4 && !order.is_ready(copy) &&
                    !order.is_ready(event_record) && !order.is_ready(free),
                "copy, record, free, and pending ownership must survive every suspension");
        require(!order.complete(execution, true) && !order.suspend(execution),
                "stopped ticket cannot report completion or stop twice");
        require(order.resume(kernel), "stopped logical work must resume");
        execution = start(order, kernel);
        require(execution.generation > previous.generation && !order.complete(previous, true) &&
                    !order.suspend(previous),
                "stale physical generation cannot advance a continuation");
    }
    require(order.complete(execution, true) && !order.complete(execution, true),
            "final logical completion is exactly once");
    require(order.is_ready(copy) && !order.is_ready(event_record) && !order.is_ready(free),
            "completion only releases the immediate stream successor");
    finish(order, copy);
    finish(order, event_record);
    finish(order, free);
    require(order.pending_count() == 0 && order.state(kernel) == OrderedOperationState::kSucceeded,
            "terminal snapshots survive pending-record collection");
}

void event_rerecord_and_synchronization_use_immutable_snapshots() {
    OperationOrder order;
    const auto producer = order.create_stream(OrderedStreamKind::kNonBlocking);
    const auto consumer = order.create_stream(OrderedStreamKind::kNonBlocking);
    const auto kernel = order.submit(producer, {}, true);
    auto event_snapshot = order.submit(producer);
    const auto old_record = event_snapshot;
    const std::array first_wait_dependency{event_snapshot};
    const auto first_wait = order.submit(consumer, first_wait_dependency);
    const auto producer_sync = order.stream_snapshot(producer);
    const auto context_sync = order.context_snapshot();
    const auto later_kernel = order.submit(producer, {}, true);
    event_snapshot = order.submit(producer);
    const std::array second_wait_dependency{event_snapshot};
    const auto second_wait = order.submit(consumer, second_wait_dependency);
    require(producer_sync && *producer_sync == old_record && context_sync.size() == 2,
            "synchronization snapshots must bind to work submitted at the call boundary");
    finish(order, kernel);
    finish(order, old_record);
    require(order.is_ready(first_wait) && !order.is_ready(second_wait),
            "event re-recording must not retarget an already queued wait");
    finish(order, first_wait);
    for (const auto& token : context_sync) {
        require(order.state(token) == OrderedOperationState::kSucceeded,
                "context snapshot must not include later submissions");
    }
    const auto execution = start(order, later_kernel);
    require(
        order.suspend(execution) && !order.is_ready(event_snapshot) && !order.is_ready(second_wait),
        "later event/wait must remain incomplete across a continuation");
    require(order.resume(later_kernel), "later work resumes");
    finish(order, later_kernel);
    finish(order, event_snapshot);
    finish(order, second_wait);
    require(order.pending_count() == 0, "event dependency graph must drain");
}

void legacy_edges_include_ptds_but_exclude_nonblocking_streams() {
    OperationOrder order;
    const auto legacy = order.create_stream(OrderedStreamKind::kLegacyDefault);
    const auto blocking = order.create_stream(OrderedStreamKind::kBlocking);
    const auto thread_one = order.create_stream(OrderedStreamKind::kPerThreadDefault);
    const auto thread_two = order.create_stream(OrderedStreamKind::kPerThreadDefault);
    const auto nonblocking = order.create_stream(OrderedStreamKind::kNonBlocking);
    require(!order.create_stream(OrderedStreamKind::kLegacyDefault),
            "one context cannot register multiple legacy default streams");
    const auto a = order.submit(blocking);
    const auto t1 = order.submit(thread_one);
    const auto t2 = order.submit(thread_two);
    const auto n = order.submit(nonblocking);
    const auto barrier = order.submit(legacy);
    const auto after = order.submit(thread_one);
    const auto independent = order.submit(nonblocking);
    require(order.is_ready(a) && order.is_ready(t1) && order.is_ready(t2) && order.is_ready(n),
            "ordinary streams and distinct PTDS streams have no implicit mutual edge");
    require(!order.is_ready(barrier) && !order.is_ready(after),
            "legacy submission and later PTDS submission must retain implicit edges");
    finish(order, n);
    require(order.is_ready(independent), "nonblocking stream must not wait on the legacy barrier");
    finish(order, a);
    finish(order, t1);
    require(!order.is_ready(barrier), "legacy work must also wait on the other thread's PTDS tail");
    finish(order, t2);
    finish(order, barrier);
    require(order.is_ready(after), "PTDS is blocking with respect to legacy work");
    finish(order, after);
    finish(order, independent);
}

void closed_stream_work_and_old_tokens_survive_handle_reuse() {
    OperationOrder order({.pending_operations = 8, .streams = 2});
    const auto stream = order.create_stream(OrderedStreamKind::kBlocking);
    const auto legacy = order.create_stream(OrderedStreamKind::kLegacyDefault);
    const auto kernel = order.submit(stream, {}, true);
    const auto execution = start(order, kernel);
    require(order.close_stream(stream) && !order.close_stream(stream),
            "closing a stream is once-only and does not finish its work");
    require(!order.submit(stream) && !order.stream_snapshot(stream),
            "closed stream cannot accept new operations or handle queries");
    const auto barrier = order.submit(legacy);
    require(!order.is_ready(barrier) && order.context_snapshot().size() == 2 &&
                !order.create_stream(OrderedStreamKind::kBlocking),
            "closed pending stream retains context/default dependencies and bounded metadata");
    require(order.suspend(execution) && !order.is_ready(barrier),
            "destroy must not let a default-stream barrier overtake a continuation");
    require(order.resume(kernel), "closed-stream continuation remains resumable");
    finish(order, kernel);
    const auto replacement = order.create_stream(OrderedStreamKind::kBlocking);
    require(static_cast<bool>(replacement),
            "terminal closed stream must release metadata capacity");
    require(!order.submit(stream) && !order.complete(execution, true),
            "recycled handle must not revive old stream/ticket ownership");
    finish(order, barrier);
    finish(order, order.submit(replacement));
}

void rejection_is_transactional_and_foreign_tokens_cannot_advance_work() {
    OperationOrder order({.pending_operations = 2, .streams = 1});
    OperationOrder other;
    const auto stream = order.create_stream(OrderedStreamKind::kNonBlocking);
    const auto foreign_stream = other.create_stream(OrderedStreamKind::kNonBlocking);
    const auto foreign_operation = other.submit(foreign_stream);
    const auto first = order.submit(stream);
    const auto tail = order.submit(stream);
    require(first && tail && !order.submit(stream), "pending capacity rejects excess operations");
    require(order.stream_snapshot(stream) == tail && order.pending_count() == 2,
            "rejected submission must not replace the previous tail");
    require(!order.create_stream(OrderedStreamKind::kBlocking) && !order.submit(foreign_stream),
            "stream bound and owner identity must reject before publication");
    finish(order, first);
    const std::array foreign_dependency{foreign_operation};
    require(!order.submit(stream, foreign_dependency) && order.stream_snapshot(stream) == tail &&
                !order.state(foreign_operation) && !order.start(foreign_operation) &&
                !order.fail_stopped(foreign_operation) && !order.close_stream(foreign_stream),
            "foreign metadata cannot enter or advance this context");
    const auto execution = start(order, tail);
    require(!other.complete(execution, true) && !order.suspend(execution),
            "foreign tickets and noncooperative suspension must reject");
    auto forged = execution;
    forged.generation = 0;
    require(!order.complete(forged, true), "zero generation cannot report completion");
    require(order.complete(execution, true), "valid completion must still work after rejection");
    require(!order.submit({}) && !order.create_stream(static_cast<OrderedStreamKind>(99)),
            "empty/unknown stream identity must reject");
    require(order.state({}) == OrderedOperationState::kSucceeded,
            "an empty record snapshot represents no preceding work");
}

void failures_poison_dependents_without_running_independent_work() {
    OperationOrder order;
    const auto a = order.create_stream(OrderedStreamKind::kNonBlocking);
    const auto b = order.create_stream(OrderedStreamKind::kNonBlocking);
    const auto c = order.create_stream(OrderedStreamKind::kNonBlocking);
    const auto kernel = order.submit(a, {}, true);
    const auto record = order.submit(a);
    const std::array dependency{record};
    const auto wait = order.submit(b, dependency);
    const auto copy = order.submit(b);
    const auto independent = order.submit(c);
    const auto execution = start(order, kernel);
    require(order.suspend(execution) && order.fail_stopped(kernel),
            "physically stopped failure must become terminal without a new launch");
    for (const auto& failed : {kernel, record, wait, copy}) {
        require(order.state(failed) == OrderedOperationState::kFailed && !order.start(failed),
                "failed dependencies must never run or report success");
    }
    require(order.pending_count() == 1 && order.is_ready(independent),
            "unrelated nonblocking work must remain available after failure");
    require(
        !order.resume(kernel) && !order.fail_stopped(kernel) && !order.complete(execution, true),
        "terminal failure cannot be resumed or overwritten by a stale report");
    finish(order, independent);
    const auto poisoned_tail = order.submit(a);
    require(poisoned_tail && order.state(poisoned_tail) == OrderedOperationState::kFailed &&
                order.pending_count() == 0,
            "subsequent work on a failed stream must not hide the prior error");
}

void long_stream_history_reclaims_pending_metadata() {
    OperationOrder order({.pending_operations = 1, .streams = 1});
    const auto stream = order.create_stream(OrderedStreamKind::kNonBlocking);
    for (int iteration = 0; iteration < 10000; ++iteration) {
        const auto operation = order.submit(stream);
        require(static_cast<bool>(operation), "terminal history must not exhaust pending capacity");
        finish(order, operation);
    }
    require(order.pending_count() == 0 && order.close_stream(stream),
            "long history must drain without retaining predecessor chains");
}

void physical_failure_and_queued_failure_preserve_terminal_state() {
    OperationOrder order;
    const auto stream = order.create_stream(OrderedStreamKind::kNonBlocking);
    const auto operation = order.submit(stream);
    const auto successor = order.submit(stream);
    const auto execution = start(order, operation);
    require(order.complete(execution, false) && !order.complete(execution, true),
            "physical execution failure cannot later become successful");
    require(order.state(successor) == OrderedOperationState::kFailed && order.pending_count() == 0,
            "physical error must poison its queued successor");
    const auto other = order.create_stream(OrderedStreamKind::kNonBlocking);
    const auto queued = order.submit(other);
    const auto following = order.submit(other);
    require(order.fail_stopped(queued) &&
                order.state(following) == OrderedOperationState::kFailed && !order.start(queued) &&
                !order.start(following),
            "a prelaunch failure must not execute either queued operation");
    const std::array<OrderedOperationToken, 3> excessive_dependencies{};
    OperationOrder bounded({.pending_operations = 2, .streams = 1});
    const auto bounded_stream = bounded.create_stream(OrderedStreamKind::kNonBlocking);
    require(!bounded.submit(bounded_stream, excessive_dependencies) && bounded.pending_count() == 0,
            "oversized explicit dependencies reject before publishing work");
    require(static_cast<bool>(bounded.submit(bounded_stream)),
            "dependency rejection must leave subsequent valid submissions possible");
}

void closed_stream_error_remains_visible_to_context_and_legacy_boundaries() {
    OperationOrder order({.pending_operations = 4, .streams = 2});
    const auto stream = order.create_stream(OrderedStreamKind::kBlocking);
    const auto legacy = order.create_stream(OrderedStreamKind::kLegacyDefault);
    const auto operation = order.submit(stream);
    const auto execution = start(order, operation);
    require(order.close_stream(stream) && order.complete(execution, false),
            "closed stream must still accept its physical error report");
    const auto snapshot = order.context_snapshot();
    require(snapshot.size() == 1 && order.state(snapshot[0]) == OrderedOperationState::kFailed,
            "context query must not forget a destroyed stream's asynchronous error");
    const auto barrier = order.submit(legacy);
    require(barrier && order.state(barrier) == OrderedOperationState::kFailed &&
                !order.is_ready(barrier) && !order.create_stream(OrderedStreamKind::kBlocking),
            "closed error stays fail-closed at default-stream and metadata-capacity boundaries");
}
}  // namespace

int main() {
    try {
        continuation_keeps_following_operations_blocked();
        event_rerecord_and_synchronization_use_immutable_snapshots();
        legacy_edges_include_ptds_but_exclude_nonblocking_streams();
        closed_stream_work_and_old_tokens_survive_handle_reuse();
        rejection_is_transactional_and_foreign_tokens_cannot_advance_work();
        failures_poison_dependents_without_running_independent_work();
        long_stream_history_reclaims_pending_metadata();
        physical_failure_and_queued_failure_preserve_terminal_state();
        closed_stream_error_remains_visible_to_context_and_legacy_boundaries();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
