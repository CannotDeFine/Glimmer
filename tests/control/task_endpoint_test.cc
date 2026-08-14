#include "glimmer/backend/task_backend.h"
#include "glimmer/control/task_endpoint.h"
#include "glimmer/control/task_protocol.h"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string_view>
#include <thread>

namespace {

using glimmer::backend::ExecutorStepStatus;
using glimmer::backend::SimulatedBackend;
using glimmer::backend::TaskExecutor;
using glimmer::control::parse_task_protocol_response;
using glimmer::control::TaskAdmissionService;
using glimmer::control::TaskControlEndpoint;
using glimmer::control::TaskPeerIdentity;
using glimmer::control::TaskProtocolErrorCode;
using glimmer::control::TaskProtocolResponseKind;
using glimmer::control::TaskProtocolState;
using glimmer::core::Scheduler;
using glimmer::core::TaskId;

struct RegistrarState {
    bool should_register = true;
    int calls = 0;
};

bool register_resource(void* context, TaskId) noexcept {
    if (context == nullptr) {
        return false;
    }
    auto* state = static_cast<RegistrarState*>(context);
    ++state->calls;
    return state->should_register;
}

void expect(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "Test failed: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

glimmer::control::TaskProtocolResponse response_from(TaskControlEndpoint& endpoint,
                                                     std::string_view request) {
    const auto encoded = endpoint.handle(request);
    expect(encoded.has_value(), "endpoint should return a response");
    const auto parsed = parse_task_protocol_response(encoded.value_or(""));
    expect(parsed.parsed(), "endpoint response should parse");
    expect(parsed.response.has_value(), "endpoint response should contain a value");
    return parsed.response.value_or(glimmer::control::TaskProtocolResponse{});
}

glimmer::control::TaskProtocolResponse response_from(TaskControlEndpoint& endpoint,
                                                     std::string_view request,
                                                     TaskPeerIdentity peer) {
    const auto encoded = endpoint.handle(request, peer);
    expect(encoded.has_value(), "endpoint should return a response");
    const auto parsed = parse_task_protocol_response(encoded.value_or(""));
    expect(parsed.parsed(), "endpoint response should parse");
    expect(parsed.response.has_value(), "endpoint response should contain a value");
    return parsed.response.value_or(glimmer::control::TaskProtocolResponse{});
}

void test_submit_query_cancel_lifecycle() {
    Scheduler scheduler(100);
    TaskAdmissionService service(scheduler);
    RegistrarState registrar_state;
    TaskControlEndpoint endpoint(service, register_resource, &registrar_state);

    const auto accepted = response_from(endpoint, "GLIMMER_TASK_V1 SUBMIT tenant-a 20 2 1\n");
    expect(accepted.kind == TaskProtocolResponseKind::kAccepted && accepted.task_id != 0,
           "submit should return a task id");
    expect(registrar_state.calls == 1, "submit should invoke the registrar");

    const std::string query = "GLIMMER_TASK_V1 QUERY " + std::to_string(accepted.task_id);
    const auto queued = response_from(endpoint, query);
    expect(queued.kind == TaskProtocolResponseKind::kState &&
               queued.state == TaskProtocolState::kQueued,
           "new task should be queued");

    const std::string cancel = "GLIMMER_TASK_V1 CANCEL " + std::to_string(accepted.task_id);
    const auto cancelled = response_from(endpoint, cancel);
    expect(cancelled.kind == TaskProtocolResponseKind::kState &&
               cancelled.state == TaskProtocolState::kCancelled,
           "queued task should be cancelled");
    expect(scheduler.usage().used_bytes() == 0, "cancel should release the reservation");
}

void test_rejection_and_unknown_task() {
    Scheduler scheduler(10);
    TaskAdmissionService service(scheduler);
    RegistrarState registrar_state;
    TaskControlEndpoint endpoint(service, register_resource, &registrar_state);

    const auto quota = response_from(endpoint, "GLIMMER_TASK_V1 SUBMIT tenant-a 11 1 1");
    expect(quota.kind == TaskProtocolResponseKind::kError &&
               quota.error == TaskProtocolErrorCode::kQuotaExceeded,
           "quota rejection should be visible in the protocol");
    const auto unsupported = response_from(endpoint, "GLIMMER_TASK_V2 QUERY 1");
    expect(unsupported.kind == TaskProtocolResponseKind::kError &&
               unsupported.error == TaskProtocolErrorCode::kUnsupportedVersion,
           "unsupported version should be visible in the protocol");
    const auto unknown = response_from(endpoint, "GLIMMER_TASK_V1 QUERY 999");
    expect(unknown.kind == TaskProtocolResponseKind::kError &&
               unknown.error == TaskProtocolErrorCode::kUnknownTask,
           "unknown task should be visible in the protocol");
}

void test_endpoint_reports_queue_backpressure() {
    Scheduler scheduler(100, glimmer::core::SchedulerOptions{.max_queued_tasks = 1});
    TaskAdmissionService service(scheduler);
    RegistrarState registrar_state;
    TaskControlEndpoint endpoint(service, register_resource, &registrar_state);

    const auto accepted = response_from(endpoint, "GLIMMER_TASK_V1 SUBMIT tenant-a 20 1 1");
    expect(accepted.kind == TaskProtocolResponseKind::kAccepted,
           "the first task should fit the waiting queue");
    const auto rejected = response_from(endpoint, "GLIMMER_TASK_V1 SUBMIT tenant-b 20 1 1");
    expect(rejected.kind == TaskProtocolResponseKind::kError &&
               rejected.error == TaskProtocolErrorCode::kQueueFull,
           "queue backpressure should be visible through the endpoint");
    expect(scheduler.usage().used_bytes() == 20,
           "queue rejection should not consume additional quota");
    expect(response_from(endpoint, "GLIMMER_TASK_V1 CANCEL " + std::to_string(accepted.task_id))
                   .state == TaskProtocolState::kCancelled,
           "queue backpressure test should release its accepted task");
}

void test_endpoint_reports_scheduler_stats() {
    Scheduler scheduler(
        100, glimmer::core::SchedulerOptions{.max_running_tasks = 2, .max_queued_tasks = 4});
    TaskAdmissionService service(scheduler);
    RegistrarState registrar_state;
    TaskControlEndpoint endpoint(service, register_resource, &registrar_state);

    const auto first = response_from(endpoint, "GLIMMER_TASK_V1 SUBMIT tenant-a 20 1 1");
    const auto second = response_from(endpoint, "GLIMMER_TASK_V1 SUBMIT tenant-b 20 1 1");
    expect(first.kind == TaskProtocolResponseKind::kAccepted &&
               second.kind == TaskProtocolResponseKind::kAccepted,
           "stats endpoint tasks should be admitted");
    const auto lease = response_from(endpoint, "GLIMMER_TASK_V1 CLAIM");
    expect(lease.kind == TaskProtocolResponseKind::kLease,
           "stats endpoint should expose a running task");
    expect(response_from(endpoint, "GLIMMER_TASK_V1 COMPLETE " + std::to_string(lease.task_id))
                   .state == TaskProtocolState::kCompleted,
           "stats endpoint task should complete");
    expect(
        response_from(endpoint, "GLIMMER_TASK_V1 CANCEL " + std::to_string(second.task_id)).state ==
            TaskProtocolState::kCancelled,
        "stats endpoint queued task should cancel");

    const auto stats = response_from(endpoint, "GLIMMER_TASK_V1 STATS");
    expect(stats.kind == TaskProtocolResponseKind::kStats && stats.stats.total_task_count == 2 &&
               stats.stats.queued_task_count == 0 && stats.stats.running_task_count == 0 &&
               stats.stats.completed_task_count == 1 && stats.stats.cancelled_task_count == 1 &&
               stats.stats.failed_task_count == 0 && stats.stats.quota_limit_bytes == 100 &&
               stats.stats.quota_reserved_bytes == 0 && stats.stats.quota_allocated_bytes == 0 &&
               stats.stats.max_running_tasks == 2 && stats.stats.max_queued_tasks == 4,
           "stats endpoint should expose scheduler state and quota usage");
}

void test_registration_failure_and_running_cancel() {
    Scheduler scheduler(100);
    TaskAdmissionService service(scheduler);
    RegistrarState registrar_state{.should_register = false};
    TaskControlEndpoint endpoint(service, register_resource, &registrar_state);
    const auto registration_failure =
        response_from(endpoint, "GLIMMER_TASK_V1 SUBMIT tenant-a 20 1 1");
    expect(registration_failure.kind == TaskProtocolResponseKind::kError &&
               registration_failure.error == TaskProtocolErrorCode::kInternalError,
           "registration failure should be internal error");
    expect(scheduler.usage().used_bytes() == 0, "registration failure should roll back quota");

    registrar_state.should_register = true;
    const auto accepted = response_from(endpoint, "GLIMMER_TASK_V1 SUBMIT tenant-a 20 1 1");
    expect(accepted.kind == TaskProtocolResponseKind::kAccepted, "second submit should succeed");
    expect(scheduler.dispatch_next().has_value(), "task should become running");
    const std::string cancel = "GLIMMER_TASK_V1 CANCEL " + std::to_string(accepted.task_id);
    const auto running_cancel = response_from(endpoint, cancel);
    expect(running_cancel.kind == TaskProtocolResponseKind::kError &&
               running_cancel.error == TaskProtocolErrorCode::kInvalidRequest,
           "running task cancellation should be rejected");
    expect(scheduler.fail(accepted.task_id), "test should clean up the running task");
}

void test_endpoint_observes_executor_lifecycle() {
    Scheduler scheduler(100);
    TaskAdmissionService service(scheduler);
    RegistrarState registrar_state;
    TaskControlEndpoint endpoint(service, register_resource, &registrar_state);
    SimulatedBackend backend(1);
    TaskExecutor executor(scheduler, backend);

    const auto accepted = response_from(endpoint, "GLIMMER_TASK_V1 SUBMIT tenant-a 20 1 1");
    expect(accepted.kind == TaskProtocolResponseKind::kAccepted, "task should be admitted");
    const auto submitted = executor.step();
    expect(submitted.status == ExecutorStepStatus::kSubmitted,
           "executor should submit the admitted task");
    const std::string query = "GLIMMER_TASK_V1 QUERY " + std::to_string(accepted.task_id);
    const auto running = response_from(endpoint, query);
    expect(running.kind == TaskProtocolResponseKind::kState &&
               running.state == TaskProtocolState::kRunning,
           "query should report a running task after dispatch");

    backend.advance();
    const auto completed = executor.step();
    expect(completed.status == ExecutorStepStatus::kCompleted,
           "executor should observe backend completion");
    const auto terminal = response_from(endpoint, query);
    expect(terminal.kind == TaskProtocolResponseKind::kState &&
               terminal.state == TaskProtocolState::kCompleted,
           "query should report a completed task");
    expect(scheduler.usage().used_bytes() == 0, "completion should release the reservation");
}

void test_endpoint_manages_remote_leases() {
    Scheduler scheduler(100);
    TaskAdmissionService service(scheduler);
    RegistrarState registrar_state;
    TaskControlEndpoint endpoint(service, register_resource, &registrar_state);

    const auto empty = response_from(endpoint, "GLIMMER_TASK_V1 CLAIM");
    expect(empty.kind == TaskProtocolResponseKind::kEmpty,
           "claim should report an empty queue without submitted tasks");

    const auto accepted = response_from(endpoint, "GLIMMER_TASK_V1 SUBMIT tenant-a 20 2 3");
    expect(accepted.kind == TaskProtocolResponseKind::kAccepted, "remote task should be admitted");

    const auto lease = response_from(endpoint, "GLIMMER_TASK_V1 CLAIM");
    expect(lease.kind == TaskProtocolResponseKind::kLease && lease.task_id == accepted.task_id &&
               lease.tenant_id == "tenant-a" && lease.memory_bytes == 20 && lease.weight == 2 &&
               lease.work_units == 3,
           "claim should return the task metadata and transition it to running");
    expect(scheduler.usage().used_bytes() == 20, "running lease should retain its reservation");

    const auto blocked_claim = response_from(endpoint, "GLIMMER_TASK_V1 CLAIM");
    expect(blocked_claim.kind == TaskProtocolResponseKind::kEmpty,
           "a second claim must not duplicate the running lease");

    const std::string heartbeat = "GLIMMER_TASK_V1 HEARTBEAT " + std::to_string(accepted.task_id);
    expect(response_from(endpoint, heartbeat).state == TaskProtocolState::kRunning,
           "an unconfigured lease timeout should preserve heartbeat compatibility");

    const std::string complete = "GLIMMER_TASK_V1 COMPLETE " + std::to_string(accepted.task_id);
    const auto completed = response_from(endpoint, complete);
    expect(completed.kind == TaskProtocolResponseKind::kState &&
               completed.state == TaskProtocolState::kCompleted,
           "complete should finish a running lease");
    expect(scheduler.usage().used_bytes() == 0, "complete should release the reservation");

    const auto second = response_from(endpoint, "GLIMMER_TASK_V1 SUBMIT tenant-a 20 2 1");
    expect(second.kind == TaskProtocolResponseKind::kAccepted, "second task should be admitted");
    const auto second_lease = response_from(endpoint, "GLIMMER_TASK_V1 CLAIM");
    expect(second_lease.kind == TaskProtocolResponseKind::kLease,
           "second task should be claimable");
    const std::string fail = "GLIMMER_TASK_V1 FAIL " + std::to_string(second.task_id);
    const auto failed = response_from(endpoint, fail);
    expect(failed.kind == TaskProtocolResponseKind::kState &&
               failed.state == TaskProtocolState::kFailed,
           "fail should finish a running lease as failed");
    expect(scheduler.usage().used_bytes() == 0, "fail should release the failed lease reservation");

    const auto invalid_complete = response_from(endpoint, "GLIMMER_TASK_V1 COMPLETE 999");
    expect(invalid_complete.kind == TaskProtocolResponseKind::kError &&
               invalid_complete.error == TaskProtocolErrorCode::kUnknownTask,
           "complete should reject an unknown task");
}

void test_endpoint_manages_concurrent_remote_leases() {
    Scheduler scheduler(100, glimmer::core::SchedulerOptions{.max_running_tasks = 2});
    TaskAdmissionService service(scheduler, std::chrono::milliseconds(100));
    RegistrarState registrar_state;
    TaskControlEndpoint endpoint(service, register_resource, &registrar_state);

    const auto first = response_from(endpoint, "GLIMMER_TASK_V1 SUBMIT tenant-a 20 1 1");
    const auto second = response_from(endpoint, "GLIMMER_TASK_V1 SUBMIT tenant-b 20 1 1");
    const auto third = response_from(endpoint, "GLIMMER_TASK_V1 SUBMIT tenant-c 20 1 1");
    expect(first.kind == TaskProtocolResponseKind::kAccepted &&
               second.kind == TaskProtocolResponseKind::kAccepted &&
               third.kind == TaskProtocolResponseKind::kAccepted,
           "all tasks should fit the configured quota");

    const auto first_lease = response_from(endpoint, "GLIMMER_TASK_V1 CLAIM");
    const auto second_lease = response_from(endpoint, "GLIMMER_TASK_V1 CLAIM");
    const auto blocked_claim = response_from(endpoint, "GLIMMER_TASK_V1 CLAIM");
    expect(first_lease.kind == TaskProtocolResponseKind::kLease &&
               second_lease.kind == TaskProtocolResponseKind::kLease &&
               blocked_claim.kind == TaskProtocolResponseKind::kEmpty,
           "endpoint should expose exactly two concurrent leases");

    const std::string first_heartbeat =
        "GLIMMER_TASK_V1 HEARTBEAT " + std::to_string(first_lease.task_id);
    const std::string second_heartbeat =
        "GLIMMER_TASK_V1 HEARTBEAT " + std::to_string(second_lease.task_id);
    expect(response_from(endpoint, first_heartbeat).state == TaskProtocolState::kRunning &&
               response_from(endpoint, second_heartbeat).state == TaskProtocolState::kRunning,
           "each active lease should be independently renewable");

    const std::string complete = "GLIMMER_TASK_V1 COMPLETE " + std::to_string(first_lease.task_id);
    expect(response_from(endpoint, complete).state == TaskProtocolState::kCompleted,
           "completing one lease should release one running slot");
    const auto third_lease = response_from(endpoint, "GLIMMER_TASK_V1 CLAIM");
    expect(third_lease.kind == TaskProtocolResponseKind::kLease &&
               third_lease.task_id == third.task_id,
           "the queued task should use the released running slot");

    const std::string fail = "GLIMMER_TASK_V1 FAIL " + std::to_string(second_lease.task_id);
    const std::string complete_third =
        "GLIMMER_TASK_V1 COMPLETE " + std::to_string(third_lease.task_id);
    expect(response_from(endpoint, fail).state == TaskProtocolState::kFailed,
           "one concurrent lease should be independently failed");
    expect(response_from(endpoint, complete_third).state == TaskProtocolState::kCompleted,
           "the queued lease should complete after dispatch");
    expect(scheduler.running_task_count() == 0 && scheduler.usage().used_bytes() == 0,
           "all concurrent endpoint leases should release their quota");
}

void test_endpoint_reaps_expired_lease() {
    Scheduler scheduler(100);
    TaskAdmissionService service(scheduler, std::chrono::milliseconds(10));
    RegistrarState registrar_state;
    TaskControlEndpoint endpoint(service, register_resource, &registrar_state);

    const auto accepted = response_from(endpoint, "GLIMMER_TASK_V1 SUBMIT tenant-a 20 1 1");
    expect(accepted.kind == TaskProtocolResponseKind::kAccepted,
           "expiring task should be admitted");
    const auto lease = response_from(endpoint, "GLIMMER_TASK_V1 CLAIM");
    expect(lease.kind == TaskProtocolResponseKind::kLease, "expiring task should be leased");
    const std::string heartbeat = "GLIMMER_TASK_V1 HEARTBEAT " + std::to_string(accepted.task_id);
    const auto alive = response_from(endpoint, heartbeat);
    expect(alive.kind == TaskProtocolResponseKind::kState &&
               alive.state == TaskProtocolState::kRunning,
           "heartbeat should renew a running lease");

    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    expect(service.reap_expired(), "expired lease should be reaped");
    const std::string query = "GLIMMER_TASK_V1 QUERY " + std::to_string(accepted.task_id);
    const auto expired = response_from(endpoint, query);
    expect(expired.kind == TaskProtocolResponseKind::kState &&
               expired.state == TaskProtocolState::kFailed,
           "expired lease should become failed");
    const auto late_heartbeat = response_from(endpoint, heartbeat);
    expect(late_heartbeat.kind == TaskProtocolResponseKind::kError &&
               late_heartbeat.error == TaskProtocolErrorCode::kInvalidRequest,
           "terminal task should not accept a heartbeat");
    expect(scheduler.usage().used_bytes() == 0,
           "reaping an expired lease should release its reservation");
}

void test_endpoint_reaps_only_the_expired_concurrent_lease() {
    Scheduler scheduler(100, glimmer::core::SchedulerOptions{.max_running_tasks = 2});
    TaskAdmissionService service(scheduler, std::chrono::milliseconds(10));
    RegistrarState registrar_state;
    TaskControlEndpoint endpoint(service, register_resource, &registrar_state);

    const auto first = response_from(endpoint, "GLIMMER_TASK_V1 SUBMIT tenant-a 20 1 1");
    const auto second = response_from(endpoint, "GLIMMER_TASK_V1 SUBMIT tenant-b 20 1 1");
    expect(first.kind == TaskProtocolResponseKind::kAccepted &&
               second.kind == TaskProtocolResponseKind::kAccepted,
           "concurrent expiry tasks should be admitted");
    const auto first_lease = response_from(endpoint, "GLIMMER_TASK_V1 CLAIM");
    const auto second_lease = response_from(endpoint, "GLIMMER_TASK_V1 CLAIM");
    expect(first_lease.kind == TaskProtocolResponseKind::kLease &&
               second_lease.kind == TaskProtocolResponseKind::kLease,
           "both tasks should hold independent leases");

    const std::string first_heartbeat =
        "GLIMMER_TASK_V1 HEARTBEAT " + std::to_string(first.task_id);
    expect(response_from(endpoint, first_heartbeat).state == TaskProtocolState::kRunning,
           "the first lease should be renewed");
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    expect(service.reap_expired(), "the unrenewed second lease should expire");

    const auto first_state =
        response_from(endpoint, "GLIMMER_TASK_V1 QUERY " + std::to_string(first.task_id));
    const auto second_state =
        response_from(endpoint, "GLIMMER_TASK_V1 QUERY " + std::to_string(second.task_id));
    expect(first_state.state == TaskProtocolState::kRunning &&
               second_state.state == TaskProtocolState::kFailed,
           "expiry should affect only the unrenewed lease");
    expect(scheduler.usage().used_bytes() == 20,
           "the renewed lease should retain its quota after another lease expires");
    expect(response_from(endpoint, "GLIMMER_TASK_V1 COMPLETE " + std::to_string(first.task_id))
                   .state == TaskProtocolState::kCompleted,
           "the renewed lease should complete normally");
    expect(scheduler.usage().used_bytes() == 0,
           "completion should release the remaining concurrent lease quota");
}

void test_endpoint_binds_lease_to_peer_process() {
    Scheduler scheduler(100);
    TaskAdmissionService service(scheduler, std::chrono::milliseconds::zero(), true);
    RegistrarState registrar_state;
    TaskControlEndpoint endpoint(service, register_resource, &registrar_state);
    const TaskPeerIdentity owner{.pid = 101, .uid = 1000, .start_time_ticks = 1};
    const TaskPeerIdentity other_process{.pid = 202, .uid = 1000, .start_time_ticks = 2};

    const auto accepted = response_from(endpoint, "GLIMMER_TASK_V1 SUBMIT tenant-a 20 1 1");
    expect(accepted.kind == TaskProtocolResponseKind::kAccepted,
           "process-bound lease task should be admitted");
    const auto missing_identity = response_from(endpoint, "GLIMMER_TASK_V1 CLAIM");
    expect(missing_identity.kind == TaskProtocolResponseKind::kEmpty,
           "a process-bound claim without peer identity must be rejected");
    const TaskPeerIdentity incomplete_identity{
        .pid = owner.pid, .uid = owner.uid, .start_time_ticks = 0};
    const auto incomplete_claim =
        response_from(endpoint, "GLIMMER_TASK_V1 CLAIM", incomplete_identity);
    expect(incomplete_claim.kind == TaskProtocolResponseKind::kEmpty,
           "a process-bound claim without start-time identity must be rejected");
    const auto lease = response_from(endpoint, "GLIMMER_TASK_V1 CLAIM", owner);
    expect(lease.kind == TaskProtocolResponseKind::kLease,
           "the owner process should claim the task");

    const std::string heartbeat = "GLIMMER_TASK_V1 HEARTBEAT " + std::to_string(lease.task_id);
    const auto missing_heartbeat = response_from(endpoint, heartbeat);
    expect(missing_heartbeat.kind == TaskProtocolResponseKind::kError &&
               missing_heartbeat.error == TaskProtocolErrorCode::kInvalidRequest,
           "a process-bound heartbeat without peer identity must be rejected");
    const auto rejected_heartbeat = response_from(endpoint, heartbeat, other_process);
    expect(rejected_heartbeat.kind == TaskProtocolResponseKind::kError &&
               rejected_heartbeat.error == TaskProtocolErrorCode::kInvalidRequest,
           "a different process must not renew the lease");
    const auto renewed = response_from(endpoint, heartbeat, owner);
    expect(renewed.kind == TaskProtocolResponseKind::kState &&
               renewed.state == TaskProtocolState::kRunning,
           "the owner process should renew the lease");

    const std::string complete = "GLIMMER_TASK_V1 COMPLETE " + std::to_string(lease.task_id);
    const auto rejected_complete = response_from(endpoint, complete, other_process);
    expect(rejected_complete.kind == TaskProtocolResponseKind::kError &&
               rejected_complete.error == TaskProtocolErrorCode::kInvalidRequest,
           "a different process must not complete the lease");
    const auto completed = response_from(endpoint, complete, owner);
    expect(completed.kind == TaskProtocolResponseKind::kState &&
               completed.state == TaskProtocolState::kCompleted,
           "the owner process should complete the lease");
}

}  // namespace

int main() {
    test_submit_query_cancel_lifecycle();
    test_rejection_and_unknown_task();
    test_endpoint_reports_queue_backpressure();
    test_endpoint_reports_scheduler_stats();
    test_registration_failure_and_running_cancel();
    test_endpoint_observes_executor_lifecycle();
    test_endpoint_manages_remote_leases();
    test_endpoint_manages_concurrent_remote_leases();
    test_endpoint_reaps_expired_lease();
    test_endpoint_reaps_only_the_expired_concurrent_lease();
    test_endpoint_binds_lease_to_peer_process();
    return EXIT_SUCCESS;
}
