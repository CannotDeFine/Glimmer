#include "internal/task_recovery_protocol.h"
#include "glimmer/control/task_execution_protocol.h"
#include "glimmer/control/task_protocol.h"

#include <array>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>

namespace {
using namespace glimmer::control;
using namespace glimmer::control::internal;

void expect(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "Test failed: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

void roundtrip(const TaskRecoveryRequest& request) {
    const auto encoded = format_task_recovery_request(request);
    expect(encoded.has_value(), "request must format");
    const auto wire = encoded.value_or(TaskRecoveryLine{});
    const auto decoded = parse_task_recovery_request(wire.view());
    expect(decoded.has_value(), "request must parse");
    const auto reformatted = format_task_recovery_request(decoded.value_or(TaskRecoveryRequest{}));
    expect(reformatted && reformatted->view() == encoded.value_or(TaskRecoveryLine{}).view(),
           "request encoding remains canonical");
}

void test_requests() {
    TaskRecoveryRequest request;
    request.nonce = {12, 34};
    const auto open = format_task_recovery_request(request);
    expect(open && open->view() == "GLIMMER_TASK_EXEC_V2 O 12 34\n", "open golden bytes");
    roundtrip(request);
    request.incarnation = {56, 78};
    expect(!format_task_recovery_request(request), "open has no expected incarnation");
    request.command = TaskRecoveryCommand::kAcknowledge;
    request.session_id = 5;
    request.acknowledged_through = 6;
    roundtrip(request);
    request.incarnation = {};
    expect(!format_task_recovery_request(request), "ack needs incarnation");
    request.incarnation = {56, 78};
    request.command = TaskRecoveryCommand::kClose;
    expect(!format_task_recovery_request(request), "close cannot implicitly acknowledge");
    request.acknowledged_through = 0;
    roundtrip(request);
    request.command = TaskRecoveryCommand::kExecute;
    request.operation_id = 7;
    request.acknowledged_through = 3;
    request.operation = TaskRecoveryOperation::kSubmit;
    request.tenant = "tenant-1";
    request.memory_bytes = 10;
    request.weight = 2;
    request.work_units = 3;
    request.priority = 4;
    roundtrip(request);
    const auto key = format_task_recovery_journal_key(request);
    request.acknowledged_through = 4;
    const auto advanced_key = format_task_recovery_journal_key(request);
    expect(key && advanced_key && key->view() == advanced_key->view(),
           "ACK advancement does not alter operation identity");
    request.incarnation = {56, 79};
    const auto new_incarnation_key = format_task_recovery_journal_key(request);
    expect(key && new_incarnation_key && key->view() != new_incarnation_key->view(),
           "operation identity includes incarnation");
    request.incarnation = {56, 78};
    request.acknowledged_through = request.operation_id;
    expect(!format_task_recovery_journal_key(request), "invalid ACK cannot create a journal key");
    request.acknowledged_through = 3;
    request.tenant = {};
    request.memory_bytes = 0;
    request.weight = 0;
    request.work_units = 0;
    request.priority = 0;
    constexpr std::array task_only{TaskRecoveryOperation::kClaim,
                                   TaskRecoveryOperation::kCancelPending,
                                   TaskRecoveryOperation::kForget};
    for (const auto operation : task_only) {
        request.operation = operation;
        request.execution = {42, 0};
        roundtrip(request);
        request.execution.generation = 1;
        expect(!format_task_recovery_request(request), "task-only operation rejects generation");
    }
    constexpr std::array executions{TaskRecoveryOperation::kHeartbeat,
                                    TaskRecoveryOperation::kRetreat,
                                    TaskRecoveryOperation::kSuspend,
                                    TaskRecoveryOperation::kComplete, TaskRecoveryOperation::kFail};
    for (const auto operation : executions) {
        request.operation = operation;
        request.execution = {42, 9};
        roundtrip(request);
        request.execution.generation = 0;
        expect(!format_task_recovery_request(request), "execution operation needs generation");
    }
    request.operation = TaskRecoveryOperation::kStatus;
    request.execution = {};
    roundtrip(request);
    request.operation = static_cast<TaskRecoveryOperation>(255);
    expect(!format_task_recovery_request(request), "unknown operation rejects");
}

void roundtrip(const TaskRecoveryResponse& response) {
    const auto encoded = format_task_recovery_response(response);
    expect(encoded && encoded->size <= kTaskRecoveryMaxLineBytes, "response must fit");
    const auto decoded = parse_task_recovery_response(encoded.value_or(TaskRecoveryLine{}).view());
    expect(decoded.has_value(), "response must parse");
    const auto again = format_task_recovery_response(decoded.value_or(TaskRecoveryResponse{}));
    expect(again && again->view() == encoded.value_or(TaskRecoveryLine{}).view(),
           "response encoding remains canonical");
}

void test_responses() {
    TaskRecoveryResponse response;
    response.kind = TaskRecoveryResultKind::kOpened;
    response.nonce = {12, 34};
    response.incarnation = {56, 78};
    response.session_id = 5;
    roundtrip(response);
    response.operation_id = 7;
    response.execution = {42, 0};
    response.kind = TaskRecoveryResultKind::kAccepted;
    roundtrip(response);
    response.kind = TaskRecoveryResultKind::kPending;
    roundtrip(response);
    response.kind = TaskRecoveryResultKind::kGranted;
    expect(!format_task_recovery_response(response), "grant needs generation");
    response.execution.generation = 9;
    roundtrip(response);
    response.kind = TaskRecoveryResultKind::kRetreat;
    response.should_retreat = true;
    roundtrip(response);
    response.should_retreat = false;
    response.execution = {};
    response.kind = TaskRecoveryResultKind::kAcknowledged;
    roundtrip(response);
    response.operation_id = 0;
    response.acknowledged_through = 7;
    roundtrip(response);
    response.acknowledged_through = 0;
    response.operation_id = 8;
    response.kind = TaskRecoveryResultKind::kStatus;
    response.pending_tasks = 3;
    response.active_tasks = 2;
    roundtrip(response);
    response.pending_tasks = 0;
    response.active_tasks = 0;
    response.kind = TaskRecoveryResultKind::kRejected;
    response.error = TaskRecoveryError::kUncertain;
    roundtrip(response);
    response.error = static_cast<TaskRecoveryError>(255);
    expect(!format_task_recovery_response(response), "unknown error code rejects");
    response.error = TaskRecoveryError::kNone;
    response.operation_id = 0;
    response.kind = TaskRecoveryResultKind::kClosed;
    roundtrip(response);
}

void test_malformed_and_limits() {
    for (const std::string_view invalid :
         {"", "GLIMMER_TASK_V1 STATS\n", "GLIMMER_TASK_EXEC_V1 STATS\n",
          "GLIMMER_TASK_EXEC_V2 O 01 2\n", "GLIMMER_TASK_EXEC_V2  O 1 2\n",
          "GLIMMER_TASK_EXEC_V2 O 0 0\n", "GLIMMER_TASK_EXEC_V2 O 1 2\nEXTRA\n",
          "GLIMMER_TASK_EXEC_V2 O 1 2", "GLIMMER_TASK_EXEC_V2 O 18446744073709551616 2\n",
          "GLIMMER_TASK_EXEC_V2 X 1 2 3 4 5 1 0 S tenant 1 4294967296 1 0\n"}) {
        expect(!parse_task_recovery_request(invalid), "malformed request rejects");
    }
    expect(!parse_task_recovery_response("GLIMMER_TASK_EXEC_V1 ERROR INVALID_REQUEST\n"),
           "older envelope never falls back");
    constexpr auto maximum = std::numeric_limits<std::uint64_t>::max();
    TaskRecoveryRequest request;
    request.command = TaskRecoveryCommand::kExecute;
    request.operation = TaskRecoveryOperation::kSubmit;
    request.nonce = {maximum, maximum};
    request.incarnation = {maximum, maximum};
    request.session_id = maximum;
    request.operation_id = maximum;
    request.acknowledged_through = maximum - 1;
    request.memory_bytes = maximum;
    request.weight = std::numeric_limits<std::uint32_t>::max();
    request.work_units = request.weight;
    request.priority = request.weight;
    std::string tenant(kTaskRecoveryMaxTenantIdBytes, 'x');
    request.tenant = tenant;
    const auto encoded = format_task_recovery_request(request);
    expect(encoded && encoded->size == kTaskRecoveryMaxLineBytes &&
               parse_task_recovery_request(encoded->view()).has_value(),
           "maximum tenant and integer fields fit and roundtrip");
    std::string oversized(encoded.value_or(TaskRecoveryLine{}).view());
    oversized.insert(oversized.end() - 1, 'x');
    expect(!parse_task_recovery_request(oversized), "overlong request rejects");
    tenant.push_back('x');
    request.tenant = tenant;
    expect(!format_task_recovery_request(request), "tenant length has a stable bound");
    TaskRecoveryResponse response;
    response.kind = TaskRecoveryResultKind::kStatus;
    response.nonce = {maximum, maximum};
    response.incarnation = {maximum, maximum};
    response.session_id = maximum;
    response.operation_id = maximum;
    response.pending_tasks = maximum;
    response.active_tasks = maximum;
    roundtrip(response);
}
}  // namespace

int main() {
    test_requests();
    test_responses();
    test_malformed_and_limits();
    return EXIT_SUCCESS;
}
