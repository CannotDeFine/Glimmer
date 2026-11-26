#include "glimmer/control/task_endpoint.h"
#include "glimmer/control/task_protocol.h"

#include <array>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <thread>

namespace {

using glimmer::control::TaskAdmissionService;
using glimmer::control::TaskControlEndpoint;
using glimmer::control::TaskPeerIdentity;
using glimmer::control::TaskProtocolErrorCode;
using glimmer::control::TaskProtocolOperation;
using glimmer::control::TaskProtocolRequest;
using glimmer::control::TaskProtocolResponse;
using glimmer::control::TaskProtocolResponseKind;
using glimmer::control::TaskProtocolState;
using glimmer::core::Scheduler;
using glimmer::core::TaskId;

constexpr TaskPeerIdentity kOwner{.pid = 10, .uid = 1000, .start_time_ticks = 20};
constexpr TaskPeerIdentity kOther{.pid = 11, .uid = 1000, .start_time_ticks = 21};
constexpr TaskPeerIdentity kRecycled{.pid = 10, .uid = 1000, .start_time_ticks = 99};

void expect(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "Test failed: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

bool register_resource(void*, TaskId) noexcept {
    return true;
}

TaskProtocolResponse request(TaskControlEndpoint& endpoint, std::string_view text,
                             TaskPeerIdentity peer = kOwner) {
    const auto encoded = endpoint.handle(text, peer);
    expect(encoded.has_value(), "endpoint should produce a response");
    const auto parsed = glimmer::control::parse_task_protocol_response(encoded.value_or(""));
    expect(parsed.parsed(), "execution response should parse");
    return parsed.response.value_or(TaskProtocolResponse{});
}

// Exercise distinct wire integer fields, including deliberately stale identities.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
TaskProtocolResponse execution(TaskControlEndpoint& endpoint, TaskProtocolOperation operation,
                               TaskId task, std::uint64_t generation,
                               TaskPeerIdentity peer = kOwner) {
    TaskProtocolRequest message;
    message.operation = operation;
    message.task_id = task;
    message.generation = generation;
    const auto encoded = glimmer::control::format_task_protocol_request(message);
    expect(encoded.has_value(), "execution request should format");
    return request(endpoint, encoded.value_or(""), peer);
}
// NOLINTEND(bugprone-easily-swappable-parameters)

bool invalid(const TaskProtocolResponse& response) {
    return response.kind == TaskProtocolResponseKind::kError &&
           response.error == TaskProtocolErrorCode::kInvalidRequest;
}

void test_protocol() {
    constexpr std::array operations{
        TaskProtocolOperation::kExecutionClaim,    TaskProtocolOperation::kRetreat,
        TaskProtocolOperation::kSuspend,           TaskProtocolOperation::kExecutionHeartbeat,
        TaskProtocolOperation::kExecutionComplete, TaskProtocolOperation::kExecutionFail,
        TaskProtocolOperation::kExecutionCancel};
    for (const auto operation : operations) {
        TaskProtocolRequest message;
        message.operation = operation;
        message.task_id = 42;
        message.generation = operation == TaskProtocolOperation::kExecutionClaim
                                 ? 0
                                 : std::numeric_limits<std::uint64_t>::max();
        const auto encoded = glimmer::control::format_task_protocol_request(message);
        expect(encoded.has_value(), "valid execution operation should format");
        const auto parsed = glimmer::control::parse_task_protocol_request(encoded.value_or(""));
        expect(parsed.parsed() && parsed.request->operation == operation &&
                   parsed.request->task_id == message.task_id &&
                   parsed.request->generation == message.generation,
               "execution identity should round trip without narrowing");
        message.task_id = 0;
        expect(!glimmer::control::format_task_protocol_request(message).has_value(),
               "execution operations require a task id");
        message.task_id = 42;
        message.generation = 0;
        if (operation != TaskProtocolOperation::kExecutionClaim) {
            expect(!glimmer::control::format_task_protocol_request(message).has_value(),
                   "generation-free execution report should not format");
        }
    }
    for (const std::string_view malformed :
         {"GLIMMER_TASK_V1 EXEC_CLAIM", "GLIMMER_TASK_V1 EXEC_CLAIM 0",
          "GLIMMER_TASK_V1 EXEC_CLAIM 1 1", "GLIMMER_TASK_V1 SUSPEND 1",
          "GLIMMER_TASK_V1 SUSPEND 1 0", "GLIMMER_TASK_V1 RETREAT 1 -1",
          "GLIMMER_TASK_V1 EXEC_COMPLETE 1 18446744073709551616",
          "GLIMMER_TASK_V1 EXEC_HEARTBEAT 1 1 trailing", "GLIMMER_TASK_V1 EXEC_CANCEL 0 1"}) {
        expect(!glimmer::control::parse_task_protocol_request(malformed).parsed(),
               "malformed execution report should be rejected");
    }
    TaskProtocolResponse lease;
    lease.kind = TaskProtocolResponseKind::kExecutionLease;
    lease.task_id = 42;
    lease.tenant_id = "tenant-a";
    lease.memory_bytes = 30;
    lease.weight = 2;
    lease.work_units = 3;
    lease.generation = std::numeric_limits<std::uint64_t>::max();
    const auto encoded = glimmer::control::format_task_protocol_response(lease);
    expect(encoded.has_value(), "execution lease should format");
    const auto decoded = glimmer::control::parse_task_protocol_response(encoded.value_or(""));
    expect(decoded.parsed() && decoded.response->kind == lease.kind &&
               decoded.response->generation == lease.generation &&
               decoded.response->tenant_id == lease.tenant_id &&
               decoded.response->memory_bytes == lease.memory_bytes &&
               decoded.response->weight == lease.weight &&
               decoded.response->work_units == lease.work_units,
           "execution lease metadata should round trip");
    for (const bool retreat : {false, true}) {
        lease.kind = TaskProtocolResponseKind::kRetreat;
        lease.should_retreat = retreat;
        const auto retreat_text = glimmer::control::format_task_protocol_response(lease);
        expect(retreat_text.has_value(), "retreat response should format");
        const auto parsed =
            glimmer::control::parse_task_protocol_response(retreat_text.value_or(""));
        expect(parsed.parsed() && parsed.response->kind == lease.kind &&
                   parsed.response->generation == lease.generation &&
                   parsed.response->should_retreat == retreat,
               "retreat response should retain the decision and generation");
    }
    for (const std::string_view malformed :
         {"GLIMMER_TASK_V1 RETREAT 1 1 2", "GLIMMER_TASK_V1 RETREAT 1 0 1",
          "GLIMMER_TASK_V1 EXEC_LEASE 1 tenant-a 30 1 1 0",
          "GLIMMER_TASK_V1 EXEC_LEASE 1 tenant-a 30 1 1 1 x"}) {
        expect(!glimmer::control::parse_task_protocol_response(malformed).parsed(),
               "malformed execution response should be rejected");
    }
}

void test_priority_resume_and_generations() {
    Scheduler scheduler(100, {.max_running_tasks = 1,
                              .max_queued_tasks = 1,
                              .scheduling_policy = glimmer::core::SchedulingPolicy::kPriority});
    TaskAdmissionService service(scheduler, std::chrono::milliseconds::zero(), true);
    TaskControlEndpoint endpoint(service, register_resource, nullptr);
    const auto low = request(endpoint, "GLIMMER_TASK_V1 SUBMIT low 60 1 1 1");
    auto lease = execution(endpoint, TaskProtocolOperation::kExecutionClaim, low.task_id, 0);
    expect(lease.kind == TaskProtocolResponseKind::kExecutionLease && lease.generation != 0,
           "first execution claim should publish a generation");
    expect(execution(endpoint, TaskProtocolOperation::kExecutionClaim, low.task_id, 0).generation ==
               lease.generation,
           "owner re-claim should be idempotent");
    const auto high = request(endpoint, "GLIMMER_TASK_V1 SUBMIT high 30 1 1 9", kOther);
    const auto retreat =
        execution(endpoint, TaskProtocolOperation::kRetreat, low.task_id, lease.generation);
    expect(retreat.kind == TaskProtocolResponseKind::kRetreat && retreat.should_retreat,
           "higher-priority arrival should request retreat");
    expect(
        execution(endpoint, TaskProtocolOperation::kExecutionClaim, high.task_id, 0, kOther).kind ==
                TaskProtocolResponseKind::kEmpty &&
            scheduler.stats().running_task_count == 1,
        "retreat request alone must not release running capacity");
    for (const auto stranger : {kOther, kRecycled, TaskPeerIdentity{}}) {
        expect(invalid(execution(endpoint, TaskProtocolOperation::kSuspend, low.task_id,
                                 lease.generation, stranger)),
               "different process, recycled PID, and invalid owner must not suspend");
        expect(invalid(execution(endpoint, TaskProtocolOperation::kExecutionComplete, low.task_id,
                                 lease.generation, stranger)),
               "only the authenticated owner may complete");
    }
    expect(invalid(execution(endpoint, TaskProtocolOperation::kSuspend, low.task_id,
                             lease.generation + 1)),
           "a fabricated generation must not suspend");
    expect(
        execution(endpoint, TaskProtocolOperation::kSuspend, low.task_id, lease.generation).state ==
            TaskProtocolState::kQueued,
        "physical stop report should queue the continuation even when submission queue is full");
    expect(scheduler.usage().allocated_bytes == 60 && scheduler.usage().reserved_bytes == 30,
           "suspension retains committed quota, not a new reservation");
    expect(execution(endpoint, TaskProtocolOperation::kExecutionHeartbeat, low.task_id,
                     lease.generation)
                   .state == TaskProtocolState::kQueued,
           "paused ownership can be renewed");
    const auto high_lease =
        execution(endpoint, TaskProtocolOperation::kExecutionClaim, high.task_id, 0, kOther);
    expect(high_lease.kind == TaskProtocolResponseKind::kExecutionLease,
           "high-priority owner should run after physical suspension");
    expect(execution(endpoint, TaskProtocolOperation::kExecutionComplete, high.task_id,
                     high_lease.generation, kOther)
                   .state == TaskProtocolState::kCompleted,
           "high-priority execution should complete");
    for (int cycle = 0; cycle != 16; ++cycle) {
        for (const std::string_view token :
             {"CLAIM", "WAIT", "CANCEL", "HEARTBEAT", "COMPLETE", "FAIL"}) {
            std::string text =
                "GLIMMER_TASK_V1 " + std::string(token) + " " + std::to_string(low.task_id);
            if (token == "WAIT") {
                text += " 1";
            }
            const auto legacy = request(endpoint, text);
            expect(legacy.kind == TaskProtocolResponseKind::kEmpty || invalid(legacy),
                   "generation-free requests cannot advance paused cooperative work");
        }
        const auto previous = lease.generation;
        lease = execution(endpoint, TaskProtocolOperation::kExecutionClaim, low.task_id, 0);
        expect(lease.kind == TaskProtocolResponseKind::kExecutionLease &&
                   lease.generation > previous && scheduler.usage().allocated_bytes == 60,
               "resumption should advance identity without committing quota again");
        for (const auto operation :
             {TaskProtocolOperation::kSuspend, TaskProtocolOperation::kExecutionHeartbeat,
              TaskProtocolOperation::kExecutionComplete, TaskProtocolOperation::kExecutionFail,
              TaskProtocolOperation::kExecutionCancel}) {
            expect(invalid(execution(endpoint, operation, low.task_id, previous)),
                   "delayed reports from a prior generation must not change resumed work");
        }
        expect(
            invalid(request(endpoint, "GLIMMER_TASK_V1 COMPLETE " + std::to_string(low.task_id))),
            "legacy terminal reports cannot complete an active continuation");
        if (cycle != 15) {
            expect(
                execution(endpoint, TaskProtocolOperation::kSuspend, low.task_id, lease.generation)
                        .state == TaskProtocolState::kQueued,
                "repeated physical suspension should retain ownership");
        }
    }
    expect(execution(endpoint, TaskProtocolOperation::kExecutionComplete, low.task_id,
                     lease.generation)
                   .state == TaskProtocolState::kCompleted,
           "final generation should complete logical execution");
    expect(invalid(execution(endpoint, TaskProtocolOperation::kExecutionComplete, low.task_id,
                             lease.generation)) &&
               scheduler.usage().used_bytes() == 0,
           "duplicate completion must not release quota twice");
}

void test_cancel_expiry_and_rejection() {
    Scheduler scheduler(100);
    TaskAdmissionService service(scheduler, std::chrono::milliseconds{10}, true);
    TaskControlEndpoint endpoint(service, register_resource, nullptr);
    const auto pending = request(endpoint, "GLIMMER_TASK_V1 SUBMIT initial 60 1 1");
    const auto cancellation = "GLIMMER_TASK_V1 CANCEL " + std::to_string(pending.task_id);
    for (const auto stranger : {kOther, kRecycled, TaskPeerIdentity{}}) {
        expect(invalid(request(endpoint, cancellation, stranger)),
               "another process cannot cancel a submission before its first execution claim");
    }
    expect(request(endpoint, cancellation).state == TaskProtocolState::kCancelled &&
               scheduler.usage().used_bytes() == 0,
           "the submitting owner can cancel initial queued work without an execution generation");
    const auto task = request(endpoint, "GLIMMER_TASK_V1 SUBMIT paused 60 1 1");
    const auto lease = execution(endpoint, TaskProtocolOperation::kExecutionClaim, task.task_id, 0);
    expect(service.suspend_execution({task.task_id, lease.generation}, kOwner),
           "owned execution should suspend");
    expect(!service.cancel_queued(task.task_id) && !service.claim(task.task_id, kOwner).has_value(),
           "legacy service entry points must not bypass continuation ownership");
    expect(invalid(execution(endpoint, TaskProtocolOperation::kExecutionCancel, task.task_id,
                             lease.generation, kOther)),
           "non-owner must not cancel stopped execution");
    expect(
        execution(endpoint, TaskProtocolOperation::kExecutionCancel, task.task_id, lease.generation)
                    .state == TaskProtocolState::kCancelled &&
            scheduler.usage().used_bytes() == 0,
        "authenticated cancellation releases stopped committed quota once");
    const auto expired = request(endpoint, "GLIMMER_TASK_V1 SUBMIT expired 60 1 1");
    const auto expired_lease =
        execution(endpoint, TaskProtocolOperation::kExecutionClaim, expired.task_id, 0);
    expect(service.suspend_execution({expired.task_id, expired_lease.generation}, kOwner),
           "expiry fixture should be physically stopped");
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{2};
    bool reaped = false;
    while (!reaped && std::chrono::steady_clock::now() < deadline) {
        reaped = service.reap_expired();
        std::this_thread::yield();
    }
    expect(reaped && scheduler.usage().used_bytes() == 0 &&
               service.find(expired.task_id)->state == glimmer::core::TaskState::kCancelled,
           "expired stopped continuation must release its retained allocation");
    expect(invalid(execution(endpoint, TaskProtocolOperation::kExecutionHeartbeat, expired.task_id,
                             expired_lease.generation)),
           "expired generation cannot resurrect ownership");

    const auto live = request(endpoint, "GLIMMER_TASK_V1 SUBMIT live 60 1 1");
    const auto live_lease =
        execution(endpoint, TaskProtocolOperation::kExecutionClaim, live.task_id, 0);
    const auto expiry = std::chrono::steady_clock::now() + std::chrono::milliseconds{20};
    while (std::chrono::steady_clock::now() < expiry) {
        expect(!service.reap_expired(),
               "an expired heartbeat cannot prove physical execution stopped");
        std::this_thread::yield();
    }
    expect(scheduler.stats().running_task_count == 1 && scheduler.usage().allocated_bytes == 60,
           "unacknowledged cooperative work retains capacity and quota after timeout");
    expect(execution(endpoint, TaskProtocolOperation::kExecutionComplete, live.task_id,
                     live_lease.generation)
                   .state == TaskProtocolState::kCompleted,
           "physical completion report can retire quarantined ownership");

    Scheduler legacy_scheduler(100);
    TaskAdmissionService unbound(legacy_scheduler);
    TaskControlEndpoint legacy_endpoint(unbound, register_resource, nullptr);
    const auto legacy_task = request(legacy_endpoint, "GLIMMER_TASK_V1 SUBMIT legacy 60 1 1");
    expect(invalid(execution(legacy_endpoint, TaskProtocolOperation::kExecutionClaim,
                             legacy_task.task_id, 0)) &&
               !unbound.claim_execution(legacy_task.task_id, kOwner).has_value(),
           "unbound service must explicitly reject cooperative execution");
    expect(request(legacy_endpoint, "GLIMMER_TASK_V1 CLAIM " + std::to_string(legacy_task.task_id))
                   .kind == TaskProtocolResponseKind::kLease,
           "non-cooperative compatibility must remain intact");
    expect(
        request(legacy_endpoint, "GLIMMER_TASK_V1 COMPLETE " + std::to_string(legacy_task.task_id))
                .state == TaskProtocolState::kCompleted,
        "non-cooperative completion must remain intact");
}

struct CompletionTiming {
    int calls = 0;
    std::uint64_t queue_wait_microseconds = 0;
    std::uint64_t service_time_microseconds = 0;
};

void capture_timing(void* context,
                    const glimmer::control::TaskCompletionObservation& observation) noexcept {
    auto& captured = *static_cast<CompletionTiming*>(context);
    ++captured.calls;
    captured.queue_wait_microseconds = observation.queue_wait_microseconds;
    captured.service_time_microseconds = observation.service_time_microseconds;
}

void test_accumulated_timing() {
    Scheduler scheduler(100);
    CompletionTiming timing;
    TaskAdmissionService service(scheduler, std::chrono::milliseconds::zero(), true, nullptr,
                                 nullptr, capture_timing, &timing);
    const auto submitted = service.submit({.tenant_id = "timed", .memory_bytes = 60},
                                          register_resource, nullptr, kOwner);
    const auto first = service.claim_execution(submitted.task_id, kOwner);
    expect(first.has_value(), "timing fixture should acquire execution");
    std::this_thread::sleep_for(std::chrono::milliseconds{10});
    expect(service.suspend_execution({submitted.task_id, first->dispatch_sequence}, kOwner) &&
               timing.calls == 0,
           "suspension is not a terminal observation");
    std::this_thread::sleep_for(std::chrono::milliseconds{10});
    const auto second = service.claim_execution(submitted.task_id, kOwner);
    expect(second.has_value() &&
               service.complete_execution({submitted.task_id, second->dispatch_sequence}, kOwner),
           "timing fixture should resume and complete");
    // Only lower bounds: scheduler delays can increase either duration. A
    // last-segment-only implementation loses the deliberately long first run.
    expect(timing.calls == 1 && timing.service_time_microseconds >= 9000 &&
               timing.queue_wait_microseconds >= 9000,
           "terminal feedback must include earlier running segments and paused queue wait");
}

}  // namespace

int main() {
    test_protocol();
    test_priority_resume_and_generations();
    test_cancel_expiry_and_rejection();
    test_accumulated_timing();
}
