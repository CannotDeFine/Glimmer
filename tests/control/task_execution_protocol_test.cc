#include "glimmer/control/task_execution_protocol.h"

#include <array>
#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>

namespace {

using namespace glimmer::control;

void expect(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "Test failed: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

void test_execution_requests() {
    constexpr std::array operations{
        std::pair{TaskExecutionProtocolOperation::kExecutionClaim, "EXEC_CLAIM"},
        std::pair{TaskExecutionProtocolOperation::kRetreat, "RETREAT"},
        std::pair{TaskExecutionProtocolOperation::kSuspend, "SUSPEND"},
        std::pair{TaskExecutionProtocolOperation::kExecutionHeartbeat, "EXEC_HEARTBEAT"},
        std::pair{TaskExecutionProtocolOperation::kExecutionComplete, "EXEC_COMPLETE"},
        std::pair{TaskExecutionProtocolOperation::kExecutionFail, "EXEC_FAIL"},
        std::pair{TaskExecutionProtocolOperation::kExecutionCancel, "EXEC_CANCEL"},
    };
    for (const auto& [operation, token] : operations) {
        const bool claim = operation == TaskExecutionProtocolOperation::kExecutionClaim;
        TaskExecutionProtocolRequest request{
            .operation = operation, .admission = {}, .task_id = 42, .generation = claim ? 0U : 7U};
        const auto expected = std::string(kTaskExecutionProtocolVersion) + " " + token + " 42" +
                              (claim ? "\n" : " 7\n");
        const auto wire = format_task_execution_protocol_request(request);
        const auto parsed = parse_task_execution_protocol_request(expected);
        expect(wire == expected && parsed.parsed() && parsed.request.has_value() &&
                   parsed.request->operation == operation && parsed.request->task_id == 42 &&
                   parsed.request->generation == request.generation,
               "execution operation must round trip exactly");
        expect(!parse_task_protocol_request(expected).parsed(),
               "legacy codec rejects new envelope");
        request.generation = claim ? 7 : 0;
        expect(!format_task_execution_protocol_request(request), "invalid generation rejected");
    }

    TaskExecutionProtocolRequest acquire;
    acquire.operation = TaskExecutionProtocolOperation::kExecutionAcquire;
    acquire.admission = {
        .tenant_id = "tenant", .memory_bytes = 1, .weight = 2, .work_units = 3, .priority = 9};
    const auto acquire_wire = format_task_execution_protocol_request(acquire);
    expect(acquire_wire == "GLIMMER_TASK_EXEC_V1 EXEC_ACQUIRE tenant 1 2 3 9\n",
           "execution acquire preserves admission");
    const auto parsed_acquire = parse_task_execution_protocol_request(acquire_wire.value_or(""));
    expect(parsed_acquire.parsed() && parsed_acquire.request.has_value() &&
               parsed_acquire.request->admission.priority == 9,
           "execution acquire round trips");
    acquire.operation = TaskExecutionProtocolOperation::kExecutionCompleteAcquire;
    acquire.task_id = 42;
    acquire.generation = 7;
    const auto handoff = format_task_execution_protocol_request(acquire);
    expect(handoff == "GLIMMER_TASK_EXEC_V1 EXEC_COMPLETE_ACQUIRE 42 7 tenant 1 2 3 9\n",
           "handoff preserves both identities and admission");
    const auto parsed_handoff = parse_task_execution_protocol_request(handoff.value_or(""));
    expect(parsed_handoff.parsed() && parsed_handoff.request.has_value() &&
               parsed_handoff.request->task_id == 42 && parsed_handoff.request->generation == 7,
           "handoff identity round trips");
    acquire.operation = TaskExecutionProtocolOperation::kExecutionAcquire;
    expect(!format_task_execution_protocol_request(acquire),
           "acquire cannot silently drop a task identity or generation");
    acquire.operation = TaskExecutionProtocolOperation::kComplete;
    expect(!format_task_execution_protocol_request(acquire),
           "legacy-style completion cannot silently drop a generation");

    for (const auto* text :
         {"GLIMMER_TASK_V1 EXEC_CLAIM 42", "GLIMMER_TASK_EXEC_V1 EXEC_CLAIM",
          "GLIMMER_TASK_EXEC_V1 EXEC_CLAIM 42 7", "GLIMMER_TASK_EXEC_V1 RETREAT 42 0",
          "GLIMMER_TASK_EXEC_V1 RETREAT 42 7 extra",
          "GLIMMER_TASK_EXEC_V1 EXEC_COMPLETE_ACQUIRE 42 0 tenant 1 2 3",
          "GLIMMER_TASK_EXEC_V1 EXEC_COMPLETE_ACQUIRE 42 7 tenant 1 2 0",
          "GLIMMER_TASK_EXEC_V1 EXEC_COMPLETE_ACQUIRE 42 18446744073709551616 tenant 1 2 3",
          "GLIMMER_TASK_EXEC_V1 UNKNOWN 42"}) {
        expect(!parse_task_execution_protocol_request(text).parsed(), "invalid request rejected");
    }
    expect(!parse_task_execution_protocol_request(std::string(kTaskProtocolMaxLineBytes + 1, 'x'))
                .parsed(),
           "oversized request rejected");

    for (const auto* text :
         {"GLIMMER_TASK_EXEC_V1 SUBMIT tenant 1 2 3\n",
          "GLIMMER_TASK_EXEC_V1 ACQUIRE tenant 1 2 3\n", "GLIMMER_TASK_EXEC_V1 CANCEL 42\n",
          "GLIMMER_TASK_EXEC_V1 QUERY 42\n", "GLIMMER_TASK_EXEC_V1 CLAIM\n",
          "GLIMMER_TASK_EXEC_V1 CLAIM 42\n", "GLIMMER_TASK_EXEC_V1 WAIT 42 10\n",
          "GLIMMER_TASK_EXEC_V1 HEARTBEAT 42\n", "GLIMMER_TASK_EXEC_V1 COMPLETE 42\n",
          "GLIMMER_TASK_EXEC_V1 FAIL 42\n", "GLIMMER_TASK_EXEC_V1 STATS\n",
          "GLIMMER_TASK_EXEC_V1 METRICS\n"}) {
        const auto parsed = parse_task_execution_protocol_request(text);
        expect(parsed.parsed() && parsed.request.has_value() &&
                   format_task_execution_protocol_request(*parsed.request) == text &&
                   !parse_task_protocol_request(text).parsed(),
               "common operation round trips only in its own envelope");
    }
}

void test_execution_responses() {
    for (const auto* text :
         {"GLIMMER_TASK_EXEC_V1 EXEC_HANDOFF 42 7 0 0\n",
          "GLIMMER_TASK_EXEC_V1 EXEC_HANDOFF 42 7 43 0\n",
          "GLIMMER_TASK_EXEC_V1 EXEC_HANDOFF 42 7 43 8\n",
          "GLIMMER_TASK_EXEC_V1 EXEC_LEASE 42 tenant 1 2 3 7\n",
          "GLIMMER_TASK_EXEC_V1 RETREAT 42 7 1\n", "GLIMMER_TASK_EXEC_V1 ERROR INVALID_REQUEST\n",
          "GLIMMER_TASK_EXEC_V1 OK 42\n", "GLIMMER_TASK_EXEC_V1 STATE 42 RUNNING\n",
          "GLIMMER_TASK_EXEC_V1 LEASE 42 tenant 1 2 3\n", "GLIMMER_TASK_EXEC_V1 EMPTY\n",
          "GLIMMER_TASK_EXEC_V1 STATS 0 0 0 0 0 0 1 0 0 1 0\n",
          "GLIMMER_TASK_EXEC_V1 METRICS 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0\n"}) {
        const auto response = parse_task_execution_protocol_response(text);
        expect(response.parsed() && response.response.has_value() &&
                   format_task_execution_protocol_response(*response.response) == text,
               "execution response round trips exactly");
        expect(!parse_task_protocol_response(text).parsed(), "legacy codec rejects new response");
    }
    for (const auto* text : {"GLIMMER_TASK_V1 EMPTY", "GLIMMER_TASK_EXEC_V1 EXEC_HANDOFF 0 7 43 8",
                             "GLIMMER_TASK_EXEC_V1 EXEC_HANDOFF 42 0 43 8",
                             "GLIMMER_TASK_EXEC_V1 EXEC_HANDOFF 42 7 42 8",
                             "GLIMMER_TASK_EXEC_V1 EXEC_HANDOFF 42 7 0 8",
                             "GLIMMER_TASK_EXEC_V1 EXEC_LEASE 42 tenant 1 2 3 0",
                             "GLIMMER_TASK_EXEC_V1 RETREAT 42 7 2",
                             "GLIMMER_TASK_EXEC_V1 METRICS 0 0 0 0 0 0 1 0 0 1 0",
                             "GLIMMER_TASK_EXEC_V1 STATS 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0"}) {
        expect(!parse_task_execution_protocol_response(text).parsed(), "invalid response rejected");
    }
    TaskExecutionProtocolResponse invalid;
    invalid.kind = static_cast<TaskExecutionProtocolResponseKind>(255);
    expect(!format_task_execution_protocol_response(invalid), "unknown kind cannot format");
    invalid = {.kind = TaskExecutionProtocolResponseKind::kLease,
               .task_id = 42,
               .tenant_id = "tenant",
               .memory_bytes = 1,
               .weight = 2,
               .work_units = 3,
               .stats = {},
               .generation = 7};
    expect(!format_task_execution_protocol_response(invalid),
           "ordinary lease cannot silently drop a generation");
    invalid.generation = 0;
    invalid.completed_execution = {8, 9};
    expect(!format_task_execution_protocol_response(invalid),
           "ordinary lease cannot silently drop a completed identity");
    invalid = {};
    invalid.kind = TaskExecutionProtocolResponseKind::kEmpty;
    invalid.task_id = 42;
    expect(!format_task_execution_protocol_response(invalid),
           "empty reply cannot silently drop a task identity");
}

}  // namespace

int main() {
    test_execution_requests();
    test_execution_responses();
    return EXIT_SUCCESS;
}
