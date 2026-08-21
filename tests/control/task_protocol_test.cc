#include "glimmer/control/task_protocol.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>

namespace {

using glimmer::control::format_task_protocol_request;
using glimmer::control::format_task_protocol_response;
using glimmer::control::kTaskProtocolMaxLineBytes;
using glimmer::control::kTaskProtocolMaxTenantIdBytes;
using glimmer::control::parse_task_protocol_request;
using glimmer::control::parse_task_protocol_response;
using glimmer::control::TaskProtocolErrorCode;
using glimmer::control::TaskProtocolOperation;
using glimmer::control::TaskProtocolParseError;
using glimmer::control::TaskProtocolRequest;
using glimmer::control::TaskProtocolResponse;
using glimmer::control::TaskProtocolResponseKind;
using glimmer::control::TaskProtocolState;

void expect(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "Test failed: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

void test_request_round_trip() {
    const TaskProtocolRequest submit{
        .operation = TaskProtocolOperation::kSubmit,
        .admission = {
            .tenant_id = "tenant-a_1", .memory_bytes = 1048576, .weight = 2, .work_units = 3}};
    const auto encoded = format_task_protocol_request(submit);
    expect(encoded.has_value(), "valid submit should format");
    const std::string encoded_text = encoded.value_or("");
    expect(encoded_text == "GLIMMER_TASK_V1 SUBMIT tenant-a_1 1048576 2 3\n",
           "submit format should be canonical");
    const auto parsed = parse_task_protocol_request(encoded_text);
    expect(parsed.parsed(), "formatted submit should parse");
    if (parsed.request.has_value()) {
        expect(parsed.request->operation == TaskProtocolOperation::kSubmit,
               "submit operation should round trip");
        expect(parsed.request->admission.tenant_id == "tenant-a_1" &&
                   parsed.request->admission.memory_bytes == 1048576 &&
                   parsed.request->admission.weight == 2 &&
                   parsed.request->admission.work_units == 3,
               "submit fields should round trip");
    }

    const TaskProtocolRequest prioritized_submit{.operation = TaskProtocolOperation::kSubmit,
                                                 .admission = {.tenant_id = "tenant-a_1",
                                                               .memory_bytes = 1048576,
                                                               .weight = 2,
                                                               .work_units = 3,
                                                               .priority = 7}};
    const auto prioritized_text = format_task_protocol_request(prioritized_submit);
    expect(prioritized_text.has_value() &&
               prioritized_text.value_or("") == "GLIMMER_TASK_V1 SUBMIT tenant-a_1 1048576 2 3 7\n",
           "non-default priority should be included in submit format");
    const auto parsed_prioritized = parse_task_protocol_request(prioritized_text.value_or(""));
    expect(parsed_prioritized.parsed() && parsed_prioritized.request.has_value() &&
               parsed_prioritized.request->admission.priority == 7,
           "priority should round trip through submit protocol");
    const auto parsed_legacy =
        parse_task_protocol_request("GLIMMER_TASK_V1 SUBMIT tenant-a_1 1048576 2 3\n");
    expect(parsed_legacy.parsed() && parsed_legacy.request.has_value() &&
               parsed_legacy.request->admission.priority == 0,
           "legacy submit format should default priority to zero");
    const auto parsed_explicit_zero =
        parse_task_protocol_request("GLIMMER_TASK_V1 SUBMIT tenant-a_1 1048576 2 3 0\n");
    expect(parsed_explicit_zero.parsed() && parsed_explicit_zero.request.has_value() &&
               parsed_explicit_zero.request->admission.priority == 0,
           "explicit zero priority should remain valid");

    for (const TaskProtocolOperation operation :
         {TaskProtocolOperation::kCancel, TaskProtocolOperation::kQuery,
          TaskProtocolOperation::kHeartbeat, TaskProtocolOperation::kComplete,
          TaskProtocolOperation::kFail}) {
        const TaskProtocolRequest request{.operation = operation, .admission = {}, .task_id = 42};
        const auto request_text = format_task_protocol_request(request);
        expect(request_text.has_value(), "task operation should format");
        expect(parse_task_protocol_request(request_text.value_or("")).parsed(),
               "formatted task operation should parse");
    }

    const TaskProtocolRequest claim{
        .operation = TaskProtocolOperation::kClaim, .admission = {}, .task_id = 0};
    const auto claim_text = format_task_protocol_request(claim);
    expect(claim_text.has_value() && claim_text.value_or("") == "GLIMMER_TASK_V1 CLAIM\n",
           "claim should format canonically");
    const auto parsed_claim = parse_task_protocol_request(claim_text.value_or(""));
    expect(parsed_claim.parsed() && parsed_claim.request.has_value() &&
               parsed_claim.request->operation == TaskProtocolOperation::kClaim,
           "claim should round trip");

    const TaskProtocolRequest specific_claim{
        .operation = TaskProtocolOperation::kClaim, .admission = {}, .task_id = 42};
    const auto specific_claim_text = format_task_protocol_request(specific_claim);
    expect(specific_claim_text.has_value() &&
               specific_claim_text.value_or("") == "GLIMMER_TASK_V1 CLAIM 42\n",
           "specific claim should format canonically");
    const auto parsed_specific_claim =
        parse_task_protocol_request(specific_claim_text.value_or(""));
    expect(parsed_specific_claim.parsed() && parsed_specific_claim.request.has_value() &&
               parsed_specific_claim.request->task_id == 42,
           "specific claim should round trip");

    const TaskProtocolRequest stats{.operation = TaskProtocolOperation::kStats, .admission = {}};
    const auto stats_text = format_task_protocol_request(stats);
    expect(stats_text.has_value() && stats_text.value_or("") == "GLIMMER_TASK_V1 STATS\n",
           "stats should format canonically");
    const auto parsed_stats = parse_task_protocol_request(stats_text.value_or(""));
    expect(parsed_stats.parsed() && parsed_stats.request.has_value() &&
               parsed_stats.request->operation == TaskProtocolOperation::kStats,
           "stats should round trip");

    const TaskProtocolRequest metrics{.operation = TaskProtocolOperation::kMetrics,
                                      .admission = {}};
    const auto metrics_text = format_task_protocol_request(metrics);
    expect(metrics_text.has_value() && metrics_text.value_or("") == "GLIMMER_TASK_V1 METRICS\n",
           "metrics should format canonically");
    const auto parsed_metrics = parse_task_protocol_request(metrics_text.value_or(""));
    expect(parsed_metrics.parsed() && parsed_metrics.request.has_value() &&
               parsed_metrics.request->operation == TaskProtocolOperation::kMetrics,
           "metrics should round trip");
}

void test_request_rejects_invalid_input() {
    expect(parse_task_protocol_request("").error == TaskProtocolParseError::kEmpty,
           "empty request should be rejected");
    expect(parse_task_protocol_request("GLIMMER_TASK_V2 QUERY 1").error ==
               TaskProtocolParseError::kUnsupportedVersion,
           "unsupported version should be rejected");
    expect(parse_task_protocol_request("GLIMMER_TASK_V1 SUBMIT tenant/a 1 1 1").error ==
               TaskProtocolParseError::kInvalidTenant,
           "unsafe tenant should be rejected");
    expect(parse_task_protocol_request("GLIMMER_TASK_V1 SUBMIT tenant 0 1 1").error ==
               TaskProtocolParseError::kInvalidValue,
           "zero memory should be rejected");
    expect(parse_task_protocol_request("GLIMMER_TASK_V1 SUBMIT tenant 1 1 1 4294967296").error ==
               TaskProtocolParseError::kInvalidValue,
           "overflowing priority should be rejected");
    expect(parse_task_protocol_request("GLIMMER_TASK_V1 QUERY 18446744073709551616").error ==
               TaskProtocolParseError::kInvalidValue,
           "overflowing task id should be rejected");
    expect(parse_task_protocol_request("GLIMMER_TASK_V1 QUERY 1 trailing").error ==
               TaskProtocolParseError::kMalformed,
           "extra request fields should be rejected");
    expect(parse_task_protocol_request("GLIMMER_TASK_V1 CLAIM trailing").error ==
               TaskProtocolParseError::kMalformed,
           "claim fields should be rejected");
    expect(parse_task_protocol_request("GLIMMER_TASK_V1 STATS trailing").error ==
               TaskProtocolParseError::kMalformed,
           "stats fields should be rejected");
    expect(parse_task_protocol_request("GLIMMER_TASK_V1 HEARTBEAT 0").error ==
               TaskProtocolParseError::kInvalidValue,
           "zero heartbeat task id should be rejected");
    expect(parse_task_protocol_request(std::string(kTaskProtocolMaxLineBytes + 1, 'x')).error ==
               TaskProtocolParseError::kLineTooLong,
           "oversized request should be rejected");
    expect(
        parse_task_protocol_request("GLIMMER_TASK_V1 SUBMIT " +
                                    std::string(kTaskProtocolMaxTenantIdBytes + 1, 'x') + " 1 1 1")
                .error == TaskProtocolParseError::kInvalidTenant,
        "overlong tenant should be rejected");
    expect(!format_task_protocol_request(
                TaskProtocolRequest{.operation = TaskProtocolOperation::kSubmit,
                                    .admission = {.tenant_id = "tenant/a", .memory_bytes = 1}})
                .has_value(),
           "invalid request should not format");
}

void test_response_round_trip() {
    for (const TaskProtocolResponse& response :
         {TaskProtocolResponse{.kind = TaskProtocolResponseKind::kAccepted,
                               .task_id = 7,
                               .state = TaskProtocolState::kQueued,
                               .error = TaskProtocolErrorCode::kInternalError,
                               .tenant_id = {},
                               .memory_bytes = 0,
                               .weight = 0,
                               .work_units = 0,
                               .stats = {}},
          TaskProtocolResponse{.kind = TaskProtocolResponseKind::kState,
                               .task_id = 7,
                               .state = TaskProtocolState::kCompleted,
                               .tenant_id = {},
                               .memory_bytes = 0,
                               .weight = 0,
                               .work_units = 0,
                               .stats = {}},
          TaskProtocolResponse{.kind = TaskProtocolResponseKind::kError,
                               .error = TaskProtocolErrorCode::kQuotaExceeded,
                               .tenant_id = {},
                               .memory_bytes = 0,
                               .weight = 0,
                               .work_units = 0,
                               .stats = {}},
          TaskProtocolResponse{.kind = TaskProtocolResponseKind::kError,
                               .error = TaskProtocolErrorCode::kQueueFull,
                               .tenant_id = {},
                               .memory_bytes = 0,
                               .weight = 0,
                               .work_units = 0,
                               .stats = {}},
          TaskProtocolResponse{.kind = TaskProtocolResponseKind::kEmpty,
                               .task_id = 0,
                               .state = TaskProtocolState::kQueued,
                               .error = TaskProtocolErrorCode::kInternalError,
                               .tenant_id = {},
                               .memory_bytes = 0,
                               .weight = 0,
                               .work_units = 0,
                               .stats = {}},
          TaskProtocolResponse{.kind = TaskProtocolResponseKind::kLease,
                               .task_id = 7,
                               .state = TaskProtocolState::kQueued,
                               .error = TaskProtocolErrorCode::kInternalError,
                               .tenant_id = "tenant-a",
                               .memory_bytes = 1024,
                               .weight = 2,
                               .work_units = 3,
                               .stats = {}},
          TaskProtocolResponse{.kind = TaskProtocolResponseKind::kStats,
                               .task_id = 0,
                               .state = TaskProtocolState::kQueued,
                               .error = TaskProtocolErrorCode::kInternalError,
                               .tenant_id = {},
                               .memory_bytes = 0,
                               .weight = 0,
                               .work_units = 0,
                               .stats = {.total_task_count = 7,
                                         .queued_task_count = 1,
                                         .running_task_count = 2,
                                         .completed_task_count = 2,
                                         .cancelled_task_count = 1,
                                         .failed_task_count = 1,
                                         .quota_limit_bytes = 8192,
                                         .quota_reserved_bytes = 1024,
                                         .quota_allocated_bytes = 2048,
                                         .max_running_tasks = 2,
                                         .max_queued_tasks = 4}},
          TaskProtocolResponse{.kind = TaskProtocolResponseKind::kMetrics,
                               .task_id = 0,
                               .state = TaskProtocolState::kQueued,
                               .error = TaskProtocolErrorCode::kInternalError,
                               .tenant_id = {},
                               .memory_bytes = 0,
                               .weight = 0,
                               .work_units = 0,
                               .stats = {.total_task_count = 7,
                                         .queued_task_count = 1,
                                         .running_task_count = 2,
                                         .completed_task_count = 2,
                                         .cancelled_task_count = 1,
                                         .failed_task_count = 1,
                                         .quota_limit_bytes = 8192,
                                         .quota_reserved_bytes = 1024,
                                         .quota_allocated_bytes = 2048,
                                         .max_running_tasks = 2,
                                         .max_queued_tasks = 4,
                                         .total_queue_wait_microseconds = 11,
                                         .max_queue_wait_microseconds = 7,
                                         .total_service_time_microseconds = 19,
                                         .max_service_time_microseconds = 13}}}) {
        const auto encoded = format_task_protocol_response(response);
        expect(encoded.has_value(), "valid response should format");
        const auto parsed = parse_task_protocol_response(encoded.value_or(""));
        expect(parsed.parsed(), "formatted response should parse");
        if (parsed.response.has_value()) {
            expect(parsed.response->kind == response.kind, "response kind should round trip");
            expect(
                parsed.response->task_id == response.task_id &&
                    parsed.response->state == response.state &&
                    parsed.response->error == response.error &&
                    parsed.response->tenant_id == response.tenant_id &&
                    parsed.response->memory_bytes == response.memory_bytes &&
                    parsed.response->weight == response.weight &&
                    parsed.response->work_units == response.work_units &&
                    parsed.response->stats.total_task_count == response.stats.total_task_count &&
                    parsed.response->stats.queued_task_count == response.stats.queued_task_count &&
                    parsed.response->stats.running_task_count ==
                        response.stats.running_task_count &&
                    parsed.response->stats.completed_task_count ==
                        response.stats.completed_task_count &&
                    parsed.response->stats.cancelled_task_count ==
                        response.stats.cancelled_task_count &&
                    parsed.response->stats.failed_task_count == response.stats.failed_task_count &&
                    parsed.response->stats.quota_limit_bytes == response.stats.quota_limit_bytes &&
                    parsed.response->stats.quota_reserved_bytes ==
                        response.stats.quota_reserved_bytes &&
                    parsed.response->stats.quota_allocated_bytes ==
                        response.stats.quota_allocated_bytes &&
                    parsed.response->stats.max_running_tasks == response.stats.max_running_tasks &&
                    parsed.response->stats.max_queued_tasks == response.stats.max_queued_tasks &&
                    parsed.response->stats.total_queue_wait_microseconds ==
                        response.stats.total_queue_wait_microseconds &&
                    parsed.response->stats.max_queue_wait_microseconds ==
                        response.stats.max_queue_wait_microseconds &&
                    parsed.response->stats.total_service_time_microseconds ==
                        response.stats.total_service_time_microseconds &&
                    parsed.response->stats.max_service_time_microseconds ==
                        response.stats.max_service_time_microseconds,
                "response fields should round trip");
        }
    }
    expect(parse_task_protocol_response("GLIMMER_TASK_V1 ERROR UNKNOWN").error ==
               TaskProtocolParseError::kMalformed,
           "unknown response error should be rejected");
    expect(parse_task_protocol_response(
               "GLIMMER_TASK_V1 STATS 0 0 0 0 0 0 18446744073709551616 0 0 1 0")
                   .error == TaskProtocolParseError::kMalformed,
           "overflowing stats values should be rejected");
    expect(parse_task_protocol_response("GLIMMER_TASK_V1 STATS 0 0 0 0 0 0 1 0 0 1").error ==
               TaskProtocolParseError::kMalformed,
           "incomplete stats responses should be rejected");
    const auto queue_full = parse_task_protocol_response("GLIMMER_TASK_V1 ERROR QUEUE_FULL");
    expect(queue_full.parsed() && queue_full.response.has_value() &&
               queue_full.response->error == TaskProtocolErrorCode::kQueueFull,
           "queue-full response error should round trip");
    expect(!format_task_protocol_response(
                TaskProtocolResponse{.kind = TaskProtocolResponseKind::kAccepted,
                                     .tenant_id = {},
                                     .memory_bytes = 0,
                                     .weight = 0,
                                     .work_units = 0,
                                     .stats = {}})
                .has_value(),
           "response without task id should not format");
}

}  // namespace

int main() {
    test_request_round_trip();
    test_request_rejects_invalid_input();
    test_response_round_trip();
    return EXIT_SUCCESS;
}
