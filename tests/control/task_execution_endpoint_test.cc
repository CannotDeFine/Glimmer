#include "glimmer/control/native_epoch_protocol.h"
#include "glimmer/control/task_endpoint.h"
#include "glimmer/control/task_execution_endpoint.h"
#include "glimmer/control/task_execution_protocol.h"
#include "internal/task_execution_endpoint_test_hooks.h"
#include "../../src/core/dispatch_test_hooks.h"

#include <array>
#include <atomic>
#include <barrier>
#include <cstdlib>
#include <future>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <thread>

namespace {

using namespace glimmer::control;
constexpr TaskPeerIdentity kOwner{10, 1000, 20};
constexpr TaskPeerIdentity kForeign{10, 1000, 21};
constexpr NativeEpochExtent kExtent{4, 2, 2, 1000000};

struct Resources {
    std::atomic<unsigned> registered{0};
    std::atomic<unsigned> rolled_back{0};
    std::atomic<bool> hold_registration{false};
    std::atomic<bool> registration_entered{false};
    std::atomic<bool> registration_release{false};
    std::atomic<bool> reject_registration{false};
};

bool register_resource(void* context, glimmer::core::TaskId) noexcept {
    auto* resources = static_cast<Resources*>(context);
    resources->registered.fetch_add(1, std::memory_order_relaxed);
    if (resources->hold_registration.load()) {
        resources->registration_entered.store(true);
        resources->registration_entered.notify_all();
        resources->registration_release.wait(false);
    }
    return !resources->reject_registration.load();
}

void rollback_resource(void* context, glimmer::core::TaskId) noexcept {
    static_cast<Resources*>(context)->rolled_back.fetch_add(1, std::memory_order_relaxed);
}

void expect(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "Test failed: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

std::unique_ptr<TaskExecutionEndpoint> endpoint(Resources& resources, bool bound = true) {
    auto service = std::make_unique<TaskExecutionAdmissionService>(
        100, glimmer::core::SchedulerOptions{}, std::chrono::milliseconds::zero(), bound, nullptr,
        nullptr, nullptr, nullptr, kExtent);
    return std::make_unique<TaskExecutionEndpoint>(std::move(service), register_resource,
                                                   rollback_resource, &resources);
}

TaskExecutionProtocolResponse exec(TaskExecutionEndpoint& service, std::string_view line,
                                   TaskPeerIdentity peer = kOwner) {
    const auto wire = service.handle(line, peer);
    expect(wire.has_value(), "execution endpoint must reply");
    const auto parsed = parse_task_execution_protocol_response(wire.value_or(""));
    expect(parsed.parsed(), "execution reply must decode");
    return parsed.response.value_or(TaskExecutionProtocolResponse{});
}

TaskProtocolResponse legacy(TaskExecutionEndpoint& service, std::string_view line,
                            TaskPeerIdentity peer = kOwner) {
    const auto wire = service.handle(line, peer);
    expect(wire.has_value(), "legacy endpoint must reply");
    const auto parsed = parse_task_protocol_response(wire.value_or(""));
    expect(parsed.parsed(), "legacy reply must decode");
    return parsed.response.value_or(TaskProtocolResponse{});
}

NativeEpochResponse epoch(TaskExecutionEndpoint& service, const NativeEpochRequest& request,
                          TaskPeerIdentity peer = kOwner) {
    const auto encoded = format_native_epoch_request(request);
    expect(encoded.has_value(), "native request must encode");
    const auto wire = service.handle(encoded.value_or(""), peer);
    expect(wire.has_value(), "native endpoint must reply");
    const auto decoded = parse_native_epoch_response(wire.value_or(""));
    expect(decoded.has_value(), "native reply must decode");
    return decoded.value_or(NativeEpochResponse{});
}

std::string operation(std::string_view name, TaskExecutionIdentity identity) {
    return std::string(kTaskExecutionProtocolVersion) + " " + std::string(name) + " " +
           std::to_string(identity.task_id) + " " + std::to_string(identity.generation) + "\n";
}

std::uint64_t used(TaskExecutionEndpoint& service) {
    const auto stats = legacy(service, "GLIMMER_TASK_V1 STATS\n").stats;
    return stats.quota_reserved_bytes + stats.quota_allocated_bytes;
}

void test_legacy_golden() {
    Resources resources;
    glimmer::core::Scheduler scheduler(100);
    TaskAdmissionService old_service(scheduler);
    TaskControlEndpoint old_endpoint(old_service, register_resource, &resources);
    auto new_endpoint = endpoint(resources, false);
    for (const auto* line :
         {"GLIMMER_TASK_V1 STATS\n", "GLIMMER_TASK_V1 SUBMIT tenant 20 1 1\n",
          "GLIMMER_TASK_V1 QUERY 1\n", "GLIMMER_TASK_V1 CLAIM 1\n", "GLIMMER_TASK_V1 HEARTBEAT 1\n",
          "GLIMMER_TASK_V1 COMPLETE 1\n", "GLIMMER_TASK_V1 QUERY 1\n", "GLIMMER_TASK_V1 STATS\n",
          "GLIMMER_TASK_V1 SUBMIT tenant 20 1 1\n", "GLIMMER_TASK_V1 CANCEL 2\n",
          "GLIMMER_TASK_V1 UNKNOWN\n"}) {
        expect(old_endpoint.handle(line) == new_endpoint->handle(line),
               "new endpoint preserves legacy response bytes");
    }
}

void test_handoff_preserves_completion_on_registration_failure() {
    Resources resources;
    auto service = endpoint(resources);
    const auto running = exec(*service, "GLIMMER_TASK_EXEC_V1 EXEC_ACQUIRE tenant 20 1 1\n");
    expect(running.kind == TaskExecutionProtocolResponseKind::kExecutionLease,
           "handoff fixture acquires a real generation");
    const TaskExecutionIdentity execution{running.task_id, running.generation};
    const auto prefix = operation("EXEC_COMPLETE_ACQUIRE", execution);
    const auto request = prefix.substr(0, prefix.size() - 1) + " tenant 20 1 1\n";
    expect(exec(*service, request, kForeign).kind == TaskExecutionProtocolResponseKind::kError &&
               resources.registered.load() == 1 && used(*service) == 20,
           "foreign handoff cannot retire or admit work");
    resources.reject_registration.store(true);
    const auto rejected = exec(*service, request);
    expect(rejected.kind == TaskExecutionProtocolResponseKind::kExecutionHandoff &&
               rejected.completed_execution.task_id == execution.task_id &&
               rejected.completed_execution.generation == execution.generation &&
               rejected.task_id == 0 && rejected.generation == 0 &&
               resources.registered.load() == 2 && used(*service) == 0,
           "failed successor registration retains the completed generation acknowledgement");
    expect(exec(*service, request).kind == TaskExecutionProtocolResponseKind::kError &&
               resources.registered.load() == 2,
           "replayed handoff cannot register a successor");
}

void test_routing_and_ownership() {
    Resources resources;
    auto service = endpoint(resources);
    expect(legacy(*service, "GLIMMER_EPOCH_V2suffix ACQUIRE\n").error ==
               TaskProtocolErrorCode::kUnsupportedVersion,
           "version-prefix collision cannot enter native handler");
    expect(exec(*service, "GLIMMER_TASK_EXEC_V1 EXEC_CLAIM\n").kind ==
               TaskExecutionProtocolResponseKind::kError,
           "malformed known execution envelope gets its own error");
    expect(service->handle("GLIMMER_EPOCH_V2 CLAIM\n").has_value() &&
               parse_native_epoch_response(service->handle("GLIMMER_EPOCH_V2 CLAIM\n").value_or(""))
                   .has_value(),
           "malformed known native envelope gets native error");
    const auto first = exec(*service, "GLIMMER_TASK_EXEC_V1 EXEC_ACQUIRE tenant 20 1 1\n");
    expect(first.kind == TaskExecutionProtocolResponseKind::kExecutionLease && first.task_id != 0 &&
               first.generation != 0,
           "cooperative acquire returns one generation");
    const TaskExecutionIdentity execution{first.task_id, first.generation};
    expect(
        legacy(*service, "GLIMMER_TASK_V1 COMPLETE 1\n").kind == TaskProtocolResponseKind::kError &&
            exec(*service, "GLIMMER_TASK_EXEC_V1 EXEC_CLAIM 1\n").kind ==
                TaskExecutionProtocolResponseKind::kEmpty &&
            exec(*service, operation("EXEC_COMPLETE", execution), kForeign).kind ==
                TaskExecutionProtocolResponseKind::kError &&
            used(*service) == 20,
        "legacy, duplicate and foreign paths cannot complete or regrant work");
    const auto second = legacy(*service, "GLIMMER_TASK_V1 SUBMIT tenant 20 1 1\n");
    expect(
        second.kind == TaskProtocolResponseKind::kAccepted &&
            exec(*service, "GLIMMER_TASK_EXEC_V1 EXEC_CLAIM 2\n").kind ==
                TaskExecutionProtocolResponseKind::kEmpty &&
            legacy(*service, "GLIMMER_TASK_V1 CANCEL 2\n").state == TaskProtocolState::kCancelled,
        "legacy task cannot be silently upgraded");
    expect(exec(*service, operation("EXEC_COMPLETE", {execution.task_id, execution.generation + 1}))
                       .kind == TaskExecutionProtocolResponseKind::kError &&
               exec(*service, operation("EXEC_COMPLETE", execution)).state ==
                   TaskProtocolState::kCompleted &&
               exec(*service, operation("EXEC_COMPLETE", execution)).kind ==
                   TaskExecutionProtocolResponseKind::kError &&
               used(*service) == 0,
           "only current generation completes once");
}

void test_native_routing() {
    Resources resources;
    auto service = endpoint(resources);
    NativeEpochRequest request;
    request.request_sequence = 7;
    request.admission = {.tenant_id = "tenant", .memory_bytes = 20, .work_units = 4};
    request.extent = kExtent;
    const auto grant = epoch(*service, request);
    expect(grant.kind == NativeEpochResponseKind::kGranted && grant.request_sequence == 7,
           "native acquire echoes request sequence");
    request.operation = NativeEpochOperation::kClaim;
    request.execution = {grant.grant.execution.task_id, 0};
    request.request_sequence = 8;
    expect(epoch(*service, request).kind == NativeEpochResponseKind::kError && used(*service) == 20,
           "native repeated claim cannot reissue");
    request.operation = NativeEpochOperation::kComplete;
    request.execution = grant.grant.execution;
    request.issued_calls = 4;
    request.request_sequence = 9;
    expect(epoch(*service, request, kForeign).kind == NativeEpochResponseKind::kError &&
               epoch(*service, request).kind == NativeEpochResponseKind::kAcknowledged &&
               epoch(*service, request).kind == NativeEpochResponseKind::kError &&
               used(*service) == 0,
           "native terminal requires owner and is exactly once");
}

void test_unpublished_rollback() {
    Resources resources;
    auto service = endpoint(resources);
    testing::fail_next_endpoint_response_format();
    expect(!service->handle("GLIMMER_TASK_EXEC_V1 SUBMIT tenant 20 1 1\n", kOwner) &&
               used(*service) == 0 && resources.rolled_back.load() == 1,
           "unpublished cooperative pending task is rolled back");
    testing::fail_next_endpoint_response_format();
    expect(!service->handle("GLIMMER_TASK_EXEC_V1 EXEC_ACQUIRE tenant 20 1 1\n", kOwner) &&
               used(*service) == 0 && resources.rolled_back.load() == 2,
           "unpublished cooperative grant is retired and released");
    NativeEpochRequest native;
    native.request_sequence = 1;
    native.admission = {.tenant_id = "tenant", .memory_bytes = 20, .work_units = 4};
    native.extent = kExtent;
    const auto native_wire = format_native_epoch_request(native);
    testing::fail_next_endpoint_response_format();
    expect(!service->handle(native_wire.value_or(""), kOwner) && used(*service) == 0 &&
               resources.rolled_back.load() == 3,
           "unpublished native grant is retired and released");
    testing::throw_next_endpoint_response_format();
    expect(!service->handle(native_wire.value_or(""), kOwner) && used(*service) == 0 &&
               resources.rolled_back.load() == 4,
           "native formatter exceptions never become legacy replies and roll back");
    const auto pending = exec(*service, "GLIMMER_TASK_EXEC_V1 SUBMIT tenant 20 1 1\n");
    expect(pending.kind == TaskExecutionProtocolResponseKind::kAccepted,
           "existing cooperative pending task published");
    testing::fail_next_endpoint_response_format();
    expect(
        !service->handle(
            "GLIMMER_TASK_EXEC_V1 EXEC_CLAIM " + std::to_string(pending.task_id) + "\n", kOwner) &&
            used(*service) == 0 && resources.rolled_back.load() == 4 &&
            legacy(*service, "GLIMMER_TASK_V1 QUERY " + std::to_string(pending.task_id) + "\n")
                    .state == TaskProtocolState::kFailed,
        "unpublished grant retires accepted work without rolling back its published resource");
    const auto old = exec(*service, "GLIMMER_TASK_EXEC_V1 EXEC_ACQUIRE tenant 20 1 1\n");
    expect(old.kind == TaskExecutionProtocolResponseKind::kExecutionLease, "handoff fixture runs");
    const TaskExecutionIdentity execution{old.task_id, old.generation};
    testing::fail_next_endpoint_response_format();
    const auto handoff = operation("EXEC_COMPLETE_ACQUIRE", execution);
    const auto request = handoff.substr(0, handoff.size() - 1) + " tenant 20 1 1\n";
    expect(
        !service->handle(request, kOwner) && used(*service) == 0 &&
            resources.rolled_back.load() == 5 &&
            legacy(*service, "GLIMMER_TASK_V1 QUERY " + std::to_string(old.task_id) + "\n").state ==
                TaskProtocolState::kCompleted,
        "handoff failure keeps old completion and rolls back only new work");
}

void test_dispatch_failures_preserve_ownership() {
    namespace faults = glimmer::core::internal;
    for (const bool uncertain : {false, true}) {
        for (const auto* operation_name : {"acquire", "claim", "handoff"}) {
            Resources resources;
            auto service = endpoint(resources);
            std::string request = "GLIMMER_TASK_EXEC_V1 EXEC_ACQUIRE tenant 20 1 1\n";
            if (std::string_view(operation_name) == "claim") {
                const auto pending = exec(*service, "GLIMMER_TASK_EXEC_V1 SUBMIT tenant 20 1 1\n");
                request =
                    "GLIMMER_TASK_EXEC_V1 EXEC_CLAIM " + std::to_string(pending.task_id) + "\n";
            } else if (std::string_view(operation_name) == "handoff") {
                const auto old =
                    exec(*service, "GLIMMER_TASK_EXEC_V1 EXEC_ACQUIRE tenant 20 1 1\n");
                const auto handoff =
                    operation("EXEC_COMPLETE_ACQUIRE", {old.task_id, old.generation});
                request = handoff.substr(0, handoff.size() - 1) + " tenant 20 1 1\n";
            }
            faults::dispatch_test_faults =
                faults::kDispatchFailSnapshot | (uncertain ? faults::kDispatchFailCancel : 0U);
            const auto response = service->handle(request, kOwner);
            faults::dispatch_test_faults = 0;
            const auto status = service->lease_status();
            expect(status.has_value(), "fault fixture status available");
            const auto lease_status = status.value_or(TaskExecutionLeaseStatus{});
            if (uncertain) {
                expect(!response && lease_status.pending_cooperative == 1 &&
                           lease_status.reaping == 1 && resources.rolled_back.load() == 0,
                       "uncertain dispatch cannot be published as pending or released");
            } else {
                const bool was_published = std::string_view(operation_name) == "claim";
                expect(response && lease_status.pending_cooperative == 0 && used(*service) == 0 &&
                           resources.rolled_back.load() == (was_published ? 0U : 1U),
                       "clean failure rolls back only an unpublished resource");
                const auto decoded = parse_task_execution_protocol_response(response.value_or(""));
                expect(decoded.parsed(), "clean failure response parses");
                const auto value = decoded.response.value_or(TaskExecutionProtocolResponse{});
                if (std::string_view(operation_name) == "acquire") {
                    expect(value.kind == TaskExecutionProtocolResponseKind::kError,
                           "clean acquire failure is not accepted");
                } else if (std::string_view(operation_name) == "claim") {
                    expect(value.kind == TaskExecutionProtocolResponseKind::kState &&
                               value.state == TaskProtocolState::kFailed,
                           "clean claim failure reports terminal state");
                } else {
                    expect(value.kind == TaskExecutionProtocolResponseKind::kExecutionHandoff &&
                               value.task_id == 0,
                           "handoff retains old completion without accepting failed work");
                }
            }
        }
    }
    for (const bool uncertain : {false, true}) {
        Resources resources;
        auto service = endpoint(resources);
        const auto grant = exec(*service, "GLIMMER_TASK_EXEC_V1 EXEC_ACQUIRE tenant 20 1 1\n");
        const TaskExecutionIdentity execution{grant.task_id, grant.generation};
        expect(exec(*service, operation("SUSPEND", execution)).state == TaskProtocolState::kQueued,
               "continuation is stopped before reclaim");
        faults::dispatch_test_faults =
            faults::kDispatchFailSnapshot | (uncertain ? faults::kDispatchFailRelease : 0U);
        const auto reply = service->handle(
            "GLIMMER_TASK_EXEC_V1 EXEC_CLAIM " + std::to_string(grant.task_id) + "\n", kOwner);
        faults::dispatch_test_faults = 0;
        const auto status = service->lease_status();
        expect(status && resources.rolled_back.load() == 0,
               "published continuation resource never uses unpublished rollback");
        const auto lease_status = status.value_or(TaskExecutionLeaseStatus{});
        if (uncertain) {
            expect(!reply && lease_status.pending_cooperative == 1 && lease_status.reaping == 1,
                   "uncertain continuation retains ownership");
        } else {
            const auto decoded = reply ? parse_task_execution_protocol_response(*reply)
                                       : TaskExecutionProtocolResponseParseResult{};
            expect(decoded.parsed() && decoded.response &&
                       decoded.response->state == TaskProtocolState::kFailed && used(*service) == 0,
                   "clean continuation failure reports terminal without releasing owner resource");
        }
    }
}

void test_published_native_claim_format_failure() {
    for (const bool should_throw : {false, true}) {
        Resources resources;
        auto service = endpoint(resources);
        const auto holder = exec(*service, "GLIMMER_TASK_EXEC_V1 EXEC_ACQUIRE tenant 20 1 1\n");
        NativeEpochRequest request;
        request.request_sequence = 1;
        request.admission = {.tenant_id = "tenant", .memory_bytes = 20, .work_units = 4};
        request.extent = kExtent;
        const auto pending = epoch(*service, request);
        expect(pending.kind == NativeEpochResponseKind::kPending,
               "native resource is published as pending");
        expect(
            exec(*service, operation("EXEC_COMPLETE", {holder.task_id, holder.generation})).state ==
                TaskProtocolState::kCompleted,
            "release capacity for the published native claim");
        request.operation = NativeEpochOperation::kClaim;
        request.execution = pending.grant.execution;
        request.request_sequence = 2;
        const auto wire = format_native_epoch_request(request);
        expect(wire.has_value(), "format native pending claim");
        if (should_throw) {
            testing::throw_next_endpoint_response_format();
        } else {
            testing::fail_next_endpoint_response_format();
        }
        expect(!service->handle(wire.value_or(""), kOwner) && resources.rolled_back.load() == 0 &&
                   legacy(*service, "GLIMMER_TASK_V1 QUERY " +
                                        std::to_string(pending.grant.execution.task_id) + "\n")
                           .state == TaskProtocolState::kFailed &&
                   used(*service) == 0,
               "unpublished native grant does not roll back an already published resource");
    }
}

void test_existing_pending_not_rolled_back() {
    Resources resources;
    auto service = endpoint(resources);
    const auto running = exec(*service, "GLIMMER_TASK_EXEC_V1 EXEC_ACQUIRE tenant 20 1 1\n");
    const auto pending = exec(*service, "GLIMMER_TASK_EXEC_V1 SUBMIT tenant 20 1 1\n");
    expect(running.kind == TaskExecutionProtocolResponseKind::kExecutionLease &&
               pending.kind == TaskExecutionProtocolResponseKind::kAccepted,
           "pending fixture has one running task");
    testing::fail_next_endpoint_response_format();
    expect(
        !service->handle(
            "GLIMMER_TASK_EXEC_V1 EXEC_CLAIM " + std::to_string(pending.task_id) + "\n", kOwner) &&
            used(*service) == 40 && resources.rolled_back.load() == 0,
        "failed empty reply does not cancel already published pending work");
    expect(
        exec(*service, operation("EXEC_COMPLETE", {running.task_id, running.generation})).state ==
            TaskProtocolState::kCompleted,
        "running fixture completes");
    const auto claimed =
        exec(*service, "GLIMMER_TASK_EXEC_V1 EXEC_CLAIM " + std::to_string(pending.task_id) + "\n");
    expect(claimed.kind == TaskExecutionProtocolResponseKind::kExecutionLease &&
               exec(*service, operation("EXEC_COMPLETE", {claimed.task_id, claimed.generation}))
                       .state == TaskProtocolState::kCompleted &&
               used(*service) == 0,
           "previously pending work remains claimable");
}

void test_concurrent_claims() {
    Resources resources;
    auto service = endpoint(resources);
    const auto pending = exec(*service, "GLIMMER_TASK_EXEC_V1 SUBMIT tenant 20 1 1\n");
    const auto line = "GLIMMER_TASK_EXEC_V1 EXEC_CLAIM " + std::to_string(pending.task_id) + "\n";
    std::array<std::optional<std::string>, 2> replies;
    std::barrier start(3);
    std::thread first([&] {
        start.arrive_and_wait();
        replies[0] = service->handle(line, kOwner);
    });
    std::thread second([&] {
        start.arrive_and_wait();
        replies[1] = service->handle(line, kOwner);
    });
    start.arrive_and_wait();
    first.join();
    second.join();
    const auto left = parse_task_execution_protocol_response(replies[0].value_or(""));
    const auto right = parse_task_execution_protocol_response(replies[1].value_or(""));
    expect(
        left.parsed() && right.parsed() && left.response.has_value() && right.response.has_value(),
        "concurrent claims both reply");
    const auto left_response = left.response.value_or(TaskExecutionProtocolResponse{});
    const auto right_response = right.response.value_or(TaskExecutionProtocolResponse{});
    expect((left_response.kind == TaskExecutionProtocolResponseKind::kExecutionLease) !=
               (right_response.kind == TaskExecutionProtocolResponseKind::kExecutionLease),
           "only one endpoint response grants a dispatch");
    const auto& winner = left_response.kind == TaskExecutionProtocolResponseKind::kExecutionLease
                             ? left_response
                             : right_response;
    expect(exec(*service, operation("EXEC_COMPLETE", {winner.task_id, winner.generation})).state ==
                   TaskProtocolState::kCompleted &&
               used(*service) == 0,
           "winning dispatch retires");
}

void test_shutdown_and_status() {
    Resources resources;
    auto service = endpoint(resources);
    const auto active = exec(*service, "GLIMMER_TASK_EXEC_V1 EXEC_ACQUIRE tenant 20 1 1\n");
    const auto pending = exec(*service, "GLIMMER_TASK_EXEC_V1 SUBMIT tenant 20 1 1\n");
    const auto legacy_pending = legacy(*service, "GLIMMER_TASK_V1 SUBMIT tenant 20 1 1\n");
    NativeEpochRequest native;
    native.request_sequence = 1;
    native.admission = {.tenant_id = "tenant", .memory_bytes = 20, .work_units = 4};
    native.extent = kExtent;
    const auto native_pending = epoch(*service, native);
    const auto before = service->lease_status();
    expect(before && before->running_tasks == 1 && before->queued_tasks == 3 &&
               before->active_cooperative == 1 && before->pending_cooperative == 1 &&
               before->pending_legacy == 1 && before->pending_native == 1,
           "status distinguishes active and pending lease types");
    expect(service->begin_shutdown() && service->begin_shutdown(), "shutdown is idempotent");
    expect(exec(*service, "GLIMMER_TASK_EXEC_V1 SUBMIT tenant 1 1 1\n").kind ==
                   TaskExecutionProtocolResponseKind::kError &&
               exec(*service,
                    "GLIMMER_TASK_EXEC_V1 EXEC_CLAIM " + std::to_string(pending.task_id) + "\n")
                       .kind == TaskExecutionProtocolResponseKind::kError,
           "shutdown rejects admission and dispatch");
    const auto handoff = operation("EXEC_COMPLETE_ACQUIRE", {active.task_id, active.generation});
    expect(exec(*service, handoff.substr(0, handoff.size() - 1) + " tenant 1 1 1\n").kind ==
                   TaskExecutionProtocolResponseKind::kError &&
               service->lease_status().value_or(TaskExecutionLeaseStatus{}).active_cooperative == 1,
           "shutdown rejects handoff without completing the original task");
    native.operation = NativeEpochOperation::kClaim;
    native.execution = native_pending.grant.execution;
    native.request_sequence = 2;
    expect(epoch(*service, native).kind == NativeEpochResponseKind::kError,
           "shutdown refuses native grant");
    native.operation = NativeEpochOperation::kCancelPending;
    native.request_sequence = 3;
    expect(
        epoch(*service, native).kind == NativeEpochResponseKind::kAcknowledged &&
            exec(*service, "GLIMMER_TASK_EXEC_V1 CANCEL " + std::to_string(pending.task_id) + "\n")
                    .state == TaskProtocolState::kCancelled &&
            legacy(*service,
                   "GLIMMER_TASK_V1 CANCEL " + std::to_string(legacy_pending.task_id) + "\n")
                    .state == TaskProtocolState::kCancelled &&
            exec(*service, operation("EXEC_COMPLETE", {active.task_id, active.generation})).state ==
                TaskProtocolState::kCompleted,
        "owners can retire pending and physically completed work during drain");
    const auto after = service->lease_status();
    expect(after && after->running_tasks == 0 && after->queued_tasks == 0 && after->reaping == 0 &&
               after->active_cooperative == 0 && after->pending_cooperative == 0 &&
               after->pending_legacy == 0 && after->pending_native == 0,
           "internal lease state is empty after retirement");
}

void test_untracked_legacy_and_native_drain() {
    Resources resources;
    auto unbound = endpoint(resources, false);
    const auto legacy_pending = legacy(*unbound, "GLIMMER_TASK_V1 SUBMIT tenant 20 1 1\n");
    const auto untracked = unbound->lease_status();
    expect(untracked && untracked->queued_tasks == 1 && untracked->pending_legacy == 0,
           "untracked legacy tasks remain visible in scheduler counts");
    expect(unbound->begin_shutdown() &&
               legacy(*unbound,
                      "GLIMMER_TASK_V1 CANCEL " + std::to_string(legacy_pending.task_id) + "\n")
                       .state == TaskProtocolState::kCancelled,
           "untracked legacy pending task can be cancelled during drain");

    auto bound = endpoint(resources);
    NativeEpochRequest native;
    native.request_sequence = 1;
    native.admission = {.tenant_id = "tenant", .memory_bytes = 20, .work_units = 4};
    native.extent = kExtent;
    const auto grant = epoch(*bound, native);
    expect(grant.kind == NativeEpochResponseKind::kGranted && bound->begin_shutdown() &&
               bound->lease_status().value_or(TaskExecutionLeaseStatus{}).active_native == 1,
           "running native lease remains owned during shutdown");
    native.operation = NativeEpochOperation::kComplete;
    native.execution = grant.grant.execution;
    native.issued_calls = 4;
    native.request_sequence = 2;
    expect(epoch(*bound, native).kind == NativeEpochResponseKind::kAcknowledged &&
               bound->lease_status().value_or(TaskExecutionLeaseStatus{}).running_tasks == 0,
           "owner can report actual native completion during drain");
}

void test_shutdown_waits_for_registration() {
    using namespace std::chrono_literals;
    Resources resources;
    resources.hold_registration.store(true);
    auto service = endpoint(resources);
    auto submit = std::async(std::launch::async, [&] {
        return service->handle("GLIMMER_TASK_EXEC_V1 SUBMIT tenant 20 1 1\n", kOwner);
    });
    resources.registration_entered.wait(false);
    auto shutdown = std::async(std::launch::async, [&] { return service->begin_shutdown(); });
    expect(shutdown.wait_for(20ms) == std::future_status::timeout,
           "shutdown waits for an in-flight registration");
    std::atomic<bool> stop_readers{false};
    std::array<std::thread, 4> readers;
    for (auto& reader : readers) {
        reader = std::thread([&] {
            while (!stop_readers.load()) {
                static_cast<void>(service->handle("GLIMMER_TASK_V1 STATS\n", kOwner));
            }
        });
    }
    resources.registration_release.store(true);
    resources.registration_release.notify_all();
    expect(shutdown.wait_for(2s) == std::future_status::ready,
           "continuous readers cannot starve shutdown");
    stop_readers.store(true);
    for (auto& reader : readers) {
        reader.join();
    }
    const auto reply = parse_task_execution_protocol_response(submit.get().value_or(""));
    expect(reply.parsed() && shutdown.get(), "in-flight admission completes before shutdown");
    const auto pending = reply.response.value_or(TaskExecutionProtocolResponse{});
    expect(exec(*service, "GLIMMER_TASK_EXEC_V1 CANCEL " + std::to_string(pending.task_id) + "\n")
                       .state == TaskProtocolState::kCancelled &&
               used(*service) == 0,
           "admission that won the race remains owner-cancellable");
}

}  // namespace

int main() {
    test_legacy_golden();
    test_handoff_preserves_completion_on_registration_failure();
    test_routing_and_ownership();
    test_native_routing();
    test_unpublished_rollback();
    test_dispatch_failures_preserve_ownership();
    test_published_native_claim_format_failure();
    test_existing_pending_not_rolled_back();
    test_concurrent_claims();
    test_shutdown_and_status();
    test_untracked_legacy_and_native_drain();
    test_shutdown_waits_for_registration();
    return EXIT_SUCCESS;
}
