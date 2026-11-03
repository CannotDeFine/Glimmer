#include "internal/task_recovery_endpoint.h"
#include "dispatch_test_hooks.h"

#include <array>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <string_view>
#include <thread>

namespace {
using namespace glimmer::control;
using namespace glimmer::control::internal;

void expect(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "Test failed: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

constexpr TaskPeerIdentity kPeer{.pid = 10, .uid = 20, .start_time_ticks = 30};
constexpr TaskRecoveryToken kIncarnation{7, 8};

[[nodiscard]] std::optional<TaskRecoveryLine> invoke(TaskRecoveryEndpoint& endpoint,
                                                     const TaskRecoveryRequest& request,
                                                     TaskPeerIdentity peer = kPeer) {
    const auto wire = format_task_recovery_request(request);
    expect(wire.has_value(), "test request must format");
    const auto request_line = wire.value_or(TaskRecoveryLine{});
    return endpoint.handle(request_line.view(), peer);
}

[[nodiscard]] TaskRecoveryResponse response(const std::optional<TaskRecoveryLine>& line) {
    expect(line.has_value(), "response exists");
    const auto response_line = line.value_or(TaskRecoveryLine{});
    const auto parsed = parse_task_recovery_response(response_line.view());
    expect(parsed.has_value(), "response parses");
    return parsed.value_or(TaskRecoveryResponse{});
}

[[nodiscard]] bool same_reply(const std::optional<TaskRecoveryLine>& first,
                              const std::optional<TaskRecoveryLine>& second) {
    return first.has_value() && second.has_value() &&
           first.value_or(TaskRecoveryLine{}).view() == second.value_or(TaskRecoveryLine{}).view();
}

[[nodiscard]] TaskRecoveryRequest open(TaskRecoveryEndpoint& endpoint, TaskRecoveryToken nonce) {
    TaskRecoveryRequest request;
    request.nonce = nonce;
    const auto opened = response(invoke(endpoint, request));
    expect(opened.kind == TaskRecoveryResultKind::kOpened, "session opens");
    request.command = TaskRecoveryCommand::kExecute;
    request.session_id = opened.session_id;
    request.incarnation = kIncarnation;
    return request;
}

void test_replay_and_lifecycle() {
    TaskRecoveryEndpoint endpoint(kIncarnation, 1024);
    auto request = open(endpoint, {1, 2});
    request.operation_id = 1;
    request.operation = TaskRecoveryOperation::kSubmit;
    request.tenant = "worker";
    request.memory_bytes = 32;
    request.weight = 1;
    request.work_units = 1;
    const auto first = invoke(endpoint, request);
    const auto accepted = response(first);
    expect(accepted.kind == TaskRecoveryResultKind::kAccepted, "submit accepted");
    expect(same_reply(invoke(endpoint, request), first), "lost submit response replayed");
    request.operation_id = 2;
    request.acknowledged_through = 1;
    request.operation = TaskRecoveryOperation::kClaim;
    request.tenant = {};
    request.memory_bytes = 0;
    request.weight = 0;
    request.work_units = 0;
    request.execution.task_id = accepted.execution.task_id;
    const auto claim = invoke(endpoint, request);
    const auto granted = response(claim);
    expect(granted.kind == TaskRecoveryResultKind::kGranted, "claim granted");
    expect(same_reply(invoke(endpoint, request), claim), "lost grant response replayed");
    request.operation_id = 3;
    request.acknowledged_through = 2;
    expect(response(invoke(endpoint, request)).kind == TaskRecoveryResultKind::kRejected,
           "new operation cannot grant active work again");
    request.operation_id = 4;
    request.acknowledged_through = 3;
    request.operation = TaskRecoveryOperation::kSuspend;
    request.execution = granted.execution;
    const auto suspended = invoke(endpoint, request);
    expect(response(suspended).kind == TaskRecoveryResultKind::kAcknowledged, "suspend accepted");
    expect(same_reply(invoke(endpoint, request), suspended), "suspend replayed");
    request.operation_id = 5;
    request.acknowledged_through = 4;
    request.operation = TaskRecoveryOperation::kHeartbeat;
    expect(response(invoke(endpoint, request)).kind == TaskRecoveryResultKind::kAcknowledged,
           "suspended execution accepts its previous generation heartbeat");
    request.operation_id = 6;
    request.acknowledged_through = 5;
    request.operation = TaskRecoveryOperation::kClaim;
    request.execution.generation = 0;
    const auto next = response(invoke(endpoint, request));
    expect(next.kind == TaskRecoveryResultKind::kGranted &&
               next.execution.generation != granted.execution.generation,
           "resumed work gets a new generation");
    request.operation_id = 7;
    request.acknowledged_through = 6;
    request.operation = TaskRecoveryOperation::kComplete;
    request.execution = next.execution;
    const auto complete = invoke(endpoint, request);
    expect(response(complete).kind == TaskRecoveryResultKind::kAcknowledged, "complete accepted");
    expect(same_reply(invoke(endpoint, request), complete), "completion replayed");
    request.operation_id = 8;
    request.acknowledged_through = 7;
    request.operation = TaskRecoveryOperation::kForget;
    request.execution.generation = 0;
    const auto forgotten = invoke(endpoint, request);
    expect(response(forgotten).kind == TaskRecoveryResultKind::kAcknowledged, "terminal forgotten");
    expect(same_reply(invoke(endpoint, request), forgotten), "forget replayed until ACK");
    request.command = TaskRecoveryCommand::kClose;
    request.operation = TaskRecoveryOperation::kStatus;
    request.operation_id = 0;
    request.acknowledged_through = 0;
    request.execution = {};
    expect(response(invoke(endpoint, request)).kind == TaskRecoveryResultKind::kRejected,
           "close refuses unacknowledged forget");
    request.command = TaskRecoveryCommand::kAcknowledge;
    request.acknowledged_through = 8;
    expect(response(invoke(endpoint, request)).kind == TaskRecoveryResultKind::kAcknowledged,
           "forget acknowledged");
    request.command = TaskRecoveryCommand::kClose;
    request.acknowledged_through = 0;
    const auto closed = invoke(endpoint, request);
    expect(response(closed).kind == TaskRecoveryResultKind::kClosed, "drained session closes");
    expect(same_reply(invoke(endpoint, request), closed), "lost close response is recoverable");
    expect(endpoint.retire_after_backend_stopped(), "empty endpoint retires");
}

void test_identity_and_poison_cleanup() {
    TaskRecoveryEndpoint endpoint(kIncarnation, 1024);
    auto first = open(endpoint, {10, 11});
    auto other = open(endpoint, {12, 13});
    first.operation_id = 1;
    first.operation = TaskRecoveryOperation::kSubmit;
    first.tenant = "first";
    first.memory_bytes = 10;
    first.weight = 1;
    first.work_units = 1;
    const auto accepted = response(invoke(endpoint, first));
    other.operation_id = 1;
    other.operation = TaskRecoveryOperation::kClaim;
    other.execution.task_id = accepted.execution.task_id;
    expect(response(invoke(endpoint, other)).kind == TaskRecoveryResultKind::kRejected,
           "sessions cannot claim each other's task");
    first.incarnation = {7, 9};
    expect(!invoke(endpoint, first), "stale incarnation cannot replay or mutate");
    auto stale_control = first;
    stale_control.command = TaskRecoveryCommand::kAcknowledge;
    stale_control.operation = TaskRecoveryOperation::kStatus;
    stale_control.operation_id = 0;
    stale_control.acknowledged_through = 1;
    stale_control.tenant = {};
    stale_control.memory_bytes = 0;
    stale_control.weight = 0;
    stale_control.work_units = 0;
    expect(!invoke(endpoint, stale_control), "stale ACK cannot retire responses");
    stale_control.command = TaskRecoveryCommand::kClose;
    stale_control.acknowledged_through = 0;
    expect(!invoke(endpoint, stale_control), "stale close cannot remove session");
    first.incarnation = kIncarnation;
    expect(response(invoke(endpoint, first)).execution.task_id == accepted.execution.task_id,
           "stale messages leave the original result recoverable");
    expect(!invoke(endpoint, first, {.pid = 99, .uid = 20, .start_time_ticks = 30}),
           "different peer cannot access session");
    first.tenant = "different";
    expect(!invoke(endpoint, first), "different payload poisons duplicate");
    first.operation_id = 2;
    first.acknowledged_through = 1;
    first.operation = TaskRecoveryOperation::kStatus;
    first.tenant = {};
    first.memory_bytes = 0;
    first.weight = 0;
    first.work_units = 0;
    expect(!invoke(endpoint, first), "poisoned session rejects new operations");
    expect(endpoint.retire_after_backend_stopped(), "owner can retire poisoned pending task");
}

void test_active_poison_owner_cleanup() {
    TaskRecoveryEndpoint endpoint(kIncarnation, 1024);
    auto request = open(endpoint, {30, 31});
    request.operation_id = 1;
    request.operation = TaskRecoveryOperation::kSubmit;
    request.tenant = "active";
    request.memory_bytes = 10;
    request.weight = 1;
    request.work_units = 1;
    const auto accepted = response(invoke(endpoint, request));
    request.operation_id = 2;
    request.acknowledged_through = 1;
    request.operation = TaskRecoveryOperation::kClaim;
    request.tenant = {};
    request.memory_bytes = 0;
    request.weight = 0;
    request.work_units = 0;
    request.execution.task_id = accepted.execution.task_id;
    expect(response(invoke(endpoint, request)).kind == TaskRecoveryResultKind::kGranted,
           "active cleanup fixture grants work");
    request.execution.task_id += 1;
    expect(!invoke(endpoint, request), "conflicting claim poisons active session");
    expect(endpoint.retire_after_backend_stopped(),
           "owner retires active work only after backend stop");
}

void test_concurrent_duplicate() {
    TaskRecoveryEndpoint endpoint(kIncarnation, 1024);
    auto request = open(endpoint, {20, 21});
    request.operation_id = 1;
    request.operation = TaskRecoveryOperation::kSubmit;
    request.tenant = "parallel";
    request.memory_bytes = 10;
    request.weight = 1;
    request.work_units = 1;
    std::array<std::optional<TaskRecoveryLine>, 2> replies;
    std::thread first([&] { replies[0] = invoke(endpoint, request); });
    std::thread second([&] { replies[1] = invoke(endpoint, request); });
    first.join();
    second.join();
    expect(same_reply(replies[0], replies[1]), "concurrent duplicate has one stable result");
    expect(endpoint.retire_after_backend_stopped(), "owner retires pending task");
}

void test_uncertain_teardown() {
    namespace faults = glimmer::core::internal;
    TaskRecoveryEndpoint endpoint(kIncarnation, 1024);
    auto request = open(endpoint, {40, 41});
    request.operation_id = 1;
    request.operation = TaskRecoveryOperation::kSubmit;
    request.tenant = "uncertain";
    request.memory_bytes = 10;
    request.weight = 1;
    request.work_units = 1;
    const auto accepted = response(invoke(endpoint, request));
    request.operation_id = 2;
    request.acknowledged_through = 1;
    request.operation = TaskRecoveryOperation::kClaim;
    request.tenant = {};
    request.memory_bytes = 0;
    request.weight = 0;
    request.work_units = 0;
    request.execution.task_id = accepted.execution.task_id;
    faults::dispatch_test_faults = faults::kDispatchFailSnapshot | faults::kDispatchFailCancel;
    expect(!invoke(endpoint, request), "uncertain accounting never publishes a grant");
    faults::dispatch_test_faults = 0;
    expect(!invoke(endpoint, request), "uncertain endpoint cannot grant on retry");
    expect(!endpoint.retire_after_backend_stopped(),
           "uncertain local accounting is not presented as a clean retirement");
    // This fixture has no backend or external resources. Destruction discards
    // the entire private state, not a guessed per-task quota rollback.
}
}  // namespace

int main() {
    test_replay_and_lifecycle();
    test_identity_and_poison_cleanup();
    test_active_poison_owner_cleanup();
    test_concurrent_duplicate();
    test_uncertain_teardown();
    return EXIT_SUCCESS;
}
