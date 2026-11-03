#include "glimmer/control/native_epoch_protocol.h"

#include <array>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>

namespace {

using namespace glimmer::control;
using glimmer::core::TaskId;

constexpr NativeEpochExtent kExtent{16, 4, 2, 1'000'000};

static_assert(static_cast<unsigned>(NativeEpochOperation::kAcquire) == 0);
static_assert(static_cast<unsigned>(NativeEpochOperation::kClaim) == 1);
static_assert(static_cast<unsigned>(NativeEpochOperation::kRetreat) == 2);
static_assert(static_cast<unsigned>(NativeEpochOperation::kHeartbeat) == 3);
static_assert(static_cast<unsigned>(NativeEpochOperation::kComplete) == 4);
static_assert(static_cast<unsigned>(NativeEpochOperation::kFail) == 5);
static_assert(static_cast<unsigned>(NativeEpochOperation::kCancelPending) == 6);
static_assert(static_cast<unsigned>(NativeEpochResponseKind::kPending) == 0);
static_assert(static_cast<unsigned>(NativeEpochResponseKind::kGranted) == 1);
static_assert(static_cast<unsigned>(NativeEpochResponseKind::kRetreat) == 2);
static_assert(static_cast<unsigned>(NativeEpochResponseKind::kAcknowledged) == 3);
static_assert(static_cast<unsigned>(NativeEpochResponseKind::kError) == 4);

void expect(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "Test failed: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

NativeEpochRequest acquire_request() {
    NativeEpochRequest result;
    result.request_sequence = 1;
    result.admission = {.tenant_id = "epoch", .memory_bytes = 1, .work_units = kExtent.max_calls};
    result.extent = kExtent;
    return result;
}

NativeEpochRequest report(NativeEpochOperation operation, const NativeEpochGrant& grant) {
    NativeEpochRequest result;
    result.request_sequence = 1;
    result.operation = operation;
    result.execution = grant.execution;
    if (operation == NativeEpochOperation::kClaim ||
        operation == NativeEpochOperation::kCancelPending) {
        result.execution.generation = 0;
    }
    return result;
}

void test_codec() {
    const auto encoded = format_native_epoch_request(acquire_request());
    expect(encoded == "GLIMMER_EPOCH_V2 ACQUIRE epoch 1 1 16 0 16 4 2 1000000 1\n",
           "independent exact version and bounds");
    expect(!parse_task_protocol_request(*encoded).parsed(),
           "unit codec cannot consume epoch request");
    expect(!parse_native_epoch_response("GLIMMER_TASK_V1 EXEC_LEASE 1 epoch 1 1 1 1\n"),
           "unit grant cannot authorize epoch");
    auto missing_sequence = acquire_request();
    missing_sequence.request_sequence = 0;
    expect(!format_native_epoch_request(missing_sequence), "zero request sequence cannot issue");
    expect(!parse_native_epoch_request("GLIMMER_EPOCH_V2 CLAIM 1 0") &&
               !parse_native_epoch_response("GLIMMER_EPOCH_V2 ACK COMPLETE 1 1 1 0") &&
               !parse_native_epoch_response("GLIMMER_EPOCH_V1 GRANTED 1 1 16 4 2 1000000 1"),
           "zero/legacy native correlation cannot authorize or acknowledge work");
    constexpr std::array bad_requests{"GLIMMER_EPOCH_V3 ACQUIRE epoch 1 1 16 0 16 4 2 1000000",
                                      "GLIMMER_EPOCH_V2 ACQUIRE epoch 1 1 1 0 16 4 2 1000000",
                                      "GLIMMER_EPOCH_V2 ACQUIRE epoch 1 1 16 0 16 17 2 1000000",
                                      "GLIMMER_EPOCH_V2 ACQUIRE epoch 1 1 16 0 16 4 17 1000000",
                                      "GLIMMER_EPOCH_V2 ACQUIRE epoch 1 1 16 0 16 4 2 1000001",
                                      "GLIMMER_EPOCH_V2 ACQUIRE bad/tenant 1 1 16 0 16 4 2 1000000",
                                      "GLIMMER_EPOCH_V2 CLAIM 0",
                                      "GLIMMER_EPOCH_V2 CLAIM 1 1",
                                      "GLIMMER_EPOCH_V2 COMPLETE 1 0 0",
                                      "GLIMMER_EPOCH_V2 COMPLETE 1 1 65537",
                                      "GLIMMER_EPOCH_V2 COMPLETE 1 1 -1",
                                      "GLIMMER_EPOCH_V2 RETREAT 1 1 extra",
                                      "GLIMMER_EPOCH_V2 CANCEL_PENDING 1 1",
                                      "GLIMMER_EPOCH_V2 SUSPEND 1 1"};
    for (const auto* bad : bad_requests) {
        expect(!parse_native_epoch_request(std::string(bad) + " 1"),
               "malformed extent/identity/cost rejected");
    }
    constexpr std::array bad_replies{"GLIMMER_EPOCH_V2 GRANTED 1 0 16 4 2 1000000 1",
                                     "GLIMMER_EPOCH_V2 GRANTED 1 1 16 4 2 1000000 0",
                                     "GLIMMER_EPOCH_V2 PENDING 1 0 4 2 1000000",
                                     "GLIMMER_EPOCH_V2 ACK COMPLETE 1 0 1",
                                     "GLIMMER_EPOCH_V2 ACK HEARTBEAT 1 1 1",
                                     "GLIMMER_EPOCH_V2 ACK CANCEL_PENDING 1 1 0",
                                     "GLIMMER_EPOCH_V2 ACK RETREAT 1 1 0",
                                     "GLIMMER_EPOCH_V2 RETREAT 1 1 2",
                                     "GLIMMER_EPOCH_V2 ACK FAIL 1 1 65537"};
    for (const auto* bad : bad_replies) {
        expect(!parse_native_epoch_response(std::string(bad) + " 1"),
               "malformed response rejected");
    }
    expect(!parse_native_epoch_request(std::string(257, 'x')) &&
               !parse_native_epoch_response(std::string(257, 'x')),
           "bounded lines");
    NativeEpochGrant grant{
        {std::numeric_limits<TaskId>::max(), std::numeric_limits<std::uint64_t>::max()},
        kExtent,
        std::numeric_limits<std::uint64_t>::max()};
    for (const auto op : {NativeEpochOperation::kClaim, NativeEpochOperation::kRetreat,
                          NativeEpochOperation::kHeartbeat, NativeEpochOperation::kComplete,
                          NativeEpochOperation::kFail, NativeEpochOperation::kCancelPending}) {
        auto request = report(op, grant);
        request.issued_calls =
            op == NativeEpochOperation::kComplete || op == NativeEpochOperation::kFail ? 16 : 0;
        const auto line = format_native_epoch_request(request);
        const auto parsed = line ? parse_native_epoch_request(*line) : std::nullopt;
        expect(parsed && parsed->operation == op &&
                   parsed->execution.task_id == request.execution.task_id &&
                   parsed->execution.generation == request.execution.generation &&
                   parsed->issued_calls == request.issued_calls,
               "all operation identities round trip without narrowing");
    }
    NativeEpochResponse granted;
    granted.request_sequence = 1;
    granted.kind = NativeEpochResponseKind::kGranted;
    granted.grant = grant;
    const auto wire = format_native_epoch_response(granted);
    const auto decoded = wire ? parse_native_epoch_response(*wire) : std::nullopt;
    expect(
        decoded && decoded->grant.extent == kExtent &&
            decoded->grant.acceptance_deadline_nanoseconds == grant.acceptance_deadline_nanoseconds,
        "maximum grant identity/deadline and full extent round trip");
    auto maximum_sequence = acquire_request();
    maximum_sequence.request_sequence = std::numeric_limits<std::uint64_t>::max();
    const auto sequence_wire = format_native_epoch_request(maximum_sequence);
    const auto sequence_reply =
        sequence_wire ? parse_native_epoch_request(*sequence_wire) : std::nullopt;
    expect(sequence_reply && sequence_reply->request_sequence == maximum_sequence.request_sequence,
           "maximum correlation sequence round trips without narrowing");

    expect(parse_native_epoch_request("GLIMMER_EPOCH_V2 CLAIM 7 1\r\n").has_value(),
           "CRLF is accepted");
    constexpr std::array replies{
        "GLIMMER_EPOCH_V2 PENDING 7 16 4 2 1000000 1\n",
        "GLIMMER_EPOCH_V2 GRANTED 7 8 16 4 2 1000000 9 1\n",
        "GLIMMER_EPOCH_V2 RETREAT 7 8 1 1\n",
        "GLIMMER_EPOCH_V2 ACK HEARTBEAT 7 8 0 1\n",
        "GLIMMER_EPOCH_V2 ACK COMPLETE 7 8 16 1\n",
        "GLIMMER_EPOCH_V2 ACK FAIL 7 8 16 1\n",
        "GLIMMER_EPOCH_V2 ACK CANCEL_PENDING 7 0 0 1\n",
        "GLIMMER_EPOCH_V2 ERROR INVALID_REQUEST 0\n",
    };
    for (const auto* reply : replies) {
        const auto parsed = parse_native_epoch_response(reply);
        expect(parsed && format_native_epoch_response(*parsed) == reply,
               "each response kind preserves the exact wire encoding");
    }
}

}  // namespace

int main() {
    test_codec();
    return 0;
}
