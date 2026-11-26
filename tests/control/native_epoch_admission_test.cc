#include "glimmer/control/native_epoch_protocol.h"
#include "glimmer/control/task_endpoint.h"
#include "glimmer/control/unix_socket_client.h"
#include "glimmer/control/unix_socket_server.h"

#include <array>
#include <atomic>
#include <barrier>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <thread>
#include <unistd.h>

namespace {

using namespace glimmer::control;
using glimmer::core::Scheduler;
using glimmer::core::TaskId;

constexpr TaskPeerIdentity kOwner{10, 1000, 20};
constexpr std::array kForeign{TaskPeerIdentity{11, 1000, 20}, TaskPeerIdentity{10, 1001, 20},
                              TaskPeerIdentity{10, 1000, 21}, TaskPeerIdentity{0, 1000, 20},
                              TaskPeerIdentity{10, 1000, 0}};
constexpr NativeEpochExtent kExtent{16, 4, 2, 1'000'000};

template <typename T>
decltype(auto) checked(T&& value) {
    if (!value.has_value()) std::abort();
    return *value;
}

void expect(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "Test failed: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

bool register_resource(void*, TaskId) noexcept {
    return true;
}

bool reject_resource(void*, TaskId) noexcept {
    return false;
}

NativeEpochRequest acquire_request() {
    NativeEpochRequest result;
    result.request_sequence = 1;
    result.admission = {.tenant_id = "epoch", .memory_bytes = 1, .work_units = kExtent.max_calls};
    result.extent = kExtent;
    return result;
}

NativeEpochResponse exchange(TaskControlEndpoint& endpoint, const NativeEpochRequest& request,
                             std::optional<TaskPeerIdentity> peer = kOwner) {
    const auto line = format_native_epoch_request(request);
    expect(line.has_value(), "fixture request formats");
    const auto reply = endpoint.handle(checked(line), peer);
    expect(reply.has_value(), "endpoint replies");
    const auto decoded = parse_native_epoch_response(checked(reply));
    expect(decoded.has_value(), "endpoint returns only epoch response");
    return checked(decoded);
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
    expect(sequence_wire && parse_native_epoch_request(*sequence_wire)->request_sequence ==
                                maximum_sequence.request_sequence,
           "maximum correlation sequence round trips without narrowing");
}

void test_policy_and_owner() {
    Scheduler scheduler(100);
    TaskAdmissionService default_service(scheduler, {}, true);
    TaskControlEndpoint default_endpoint(default_service, register_resource, nullptr);
    expect(exchange(default_endpoint, acquire_request()).kind == NativeEpochResponseKind::kError &&
               scheduler.stats().total_task_count == 0,
           "default service does not approve epochs or mutate ledger");
    TaskAdmissionService unbound(scheduler, {}, false, nullptr, nullptr, nullptr, nullptr, kExtent);
    TaskControlEndpoint unbound_endpoint(unbound, register_resource, nullptr);
    expect(exchange(unbound_endpoint, acquire_request()).kind == NativeEpochResponseKind::kError,
           "policy cannot enable unauthenticated epochs");
    TaskAdmissionService service(scheduler, {}, true, nullptr, nullptr, nullptr, nullptr, kExtent);
    TaskControlEndpoint endpoint(service, register_resource, nullptr);
    expect(
        exchange(endpoint, acquire_request(), std::nullopt).kind == NativeEpochResponseKind::kError,
        "missing trusted identity rejects");
    for (const auto peer : kForeign) {
        if (peer.pid == 0 || peer.start_time_ticks == 0) {
            expect(
                exchange(endpoint, acquire_request(), peer).kind == NativeEpochResponseKind::kError,
                "invalid owner rejects submission");
        }
    }
    auto excessive = acquire_request();
    excessive.extent.max_calls = 17;
    excessive.admission.work_units = 17;
    expect(exchange(endpoint, excessive).kind == NativeEpochResponseKind::kError,
           "over-policy calls reject");
    excessive = acquire_request();
    excessive.extent.calls_per_prefix = 5;
    expect(exchange(endpoint, excessive).kind == NativeEpochResponseKind::kError,
           "over-policy prefix width rejects");
    excessive = acquire_request();
    excessive.extent.max_pending_prefixes = 3;
    expect(exchange(endpoint, excessive).kind == NativeEpochResponseKind::kError,
           "over-policy capacity rejects");
    expect(scheduler.stats().total_task_count == 0, "all rejection precedes admission");
    auto wrong_cost = acquire_request();
    wrong_cost.admission.work_units = 1;
    expect(
        !service
             .submit_native_epoch(wrong_cost.admission, kExtent, register_resource, nullptr, kOwner)
             .accepted(),
        "direct API cannot undercharge calls");
    const auto granted = exchange(endpoint, acquire_request());
    expect(granted.kind == NativeEpochResponseKind::kGranted && granted.grant.extent == kExtent &&
               granted.grant.execution.generation != 0,
           "separate authorized grant");
    const auto task = granted.grant.execution.task_id;
    expect(checked(service.find(task)).work_units == kExtent.max_calls,
           "epoch costs approved call cap");
    for (const auto peer : kForeign) {
        for (const auto op : {NativeEpochOperation::kClaim, NativeEpochOperation::kRetreat,
                              NativeEpochOperation::kHeartbeat, NativeEpochOperation::kComplete,
                              NativeEpochOperation::kFail, NativeEpochOperation::kCancelPending}) {
            expect(exchange(endpoint, report(op, granted.grant), peer).kind ==
                       NativeEpochResponseKind::kError,
                   "foreign/recycled/invalid owner cannot use grant");
        }
    }
    expect(!service.claim(task, kOwner) &&
               !service.wait_claim(task, std::chrono::milliseconds(1), kOwner) &&
               !service.claim_execution(task, kOwner),
           "old claims cannot consume epoch");
    const auto identity = granted.grant.execution;
    expect(!service.heartbeat(task, kOwner) && !service.complete(task, kOwner) &&
               !service.fail(task, kOwner) && !service.cancel_queued(task, kOwner) &&
               !service.cancel(task),
           "old and administrative terminals cannot free active epoch");
    expect(!service.retreat_requested(identity, kOwner) &&
               !service.suspend_execution(identity, kOwner) &&
               !service.heartbeat_execution(identity, kOwner) &&
               !service.complete_execution(identity, kOwner) &&
               !service.fail_execution(identity, kOwner) &&
               !service.cancel_execution(identity, kOwner),
           "cooperative reports cannot mutate epoch");
    expect(
        scheduler.stats().running_task_count == 1 && scheduler.stats().quota.allocated_bytes == 1,
        "rejected reports retain running slot/quota");
    const auto repeated = exchange(endpoint, report(NativeEpochOperation::kClaim, granted.grant));
    expect(repeated.kind == NativeEpochResponseKind::kGranted &&
               repeated.grant.acceptance_deadline_nanoseconds ==
                   granted.grant.acceptance_deadline_nanoseconds,
           "repeated claim cannot renew acceptance");
    auto terminal = report(NativeEpochOperation::kComplete, granted.grant);
    terminal.issued_calls = 17;
    expect(exchange(endpoint, terminal).kind == NativeEpochResponseKind::kError,
           "above-approved terminal count rejects");
    terminal.issued_calls = 16;
    ++terminal.execution.generation;
    expect(exchange(endpoint, terminal).kind == NativeEpochResponseKind::kError,
           "stale/foreign generation rejects");
    terminal.execution = identity;
    // Discard the successful reply to model transport loss after mutation.
    expect(endpoint.handle(checked(format_native_epoch_request(terminal)), kOwner).has_value(),
           "terminal executes once even if reply lost");
    expect(exchange(endpoint, terminal).kind == NativeEpochResponseKind::kError,
           "lost terminal cannot replay");
    expect(
        scheduler.stats().running_task_count == 0 && scheduler.stats().quota.allocated_bytes == 0,
        "exact terminal releases once");
}

void test_priority_pending_and_rollback() {
    Scheduler scheduler(100, {.scheduling_policy = glimmer::core::SchedulingPolicy::kPriority});
    TaskAdmissionService service(scheduler, {}, true, nullptr, nullptr, nullptr, nullptr, kExtent);
    TaskControlEndpoint endpoint(service, register_resource, nullptr);
    const auto low = exchange(endpoint, acquire_request());
    auto high_request = acquire_request();
    high_request.admission.tenant_id = "high";
    high_request.admission.priority = 10;
    const auto high = exchange(endpoint, high_request);
    expect(high.kind == NativeEpochResponseKind::kPending && high.grant.execution.generation == 0 &&
               high.grant.extent == kExtent,
           "pending approval is not dispatch authority");
    expect(!service.claim(high.grant.execution.task_id, kOwner) &&
               !service.claim_execution(high.grant.execution.task_id, kOwner) &&
               !service.cancel_queued(high.grant.execution.task_id, kOwner),
           "old pending operations excluded");
    const auto retreat = exchange(endpoint, report(NativeEpochOperation::kRetreat, low.grant));
    expect(retreat.kind == NativeEpochResponseKind::kRetreat && retreat.should_retreat &&
               scheduler.stats().running_task_count == 1,
           "strict-priority retreat preserves capacity until proof report");
    expect(exchange(endpoint, report(NativeEpochOperation::kCancelPending, low.grant)).kind ==
               NativeEpochResponseKind::kError,
           "running epoch cannot be cancelled as pending");
    const auto complete = exchange(endpoint, report(NativeEpochOperation::kComplete, low.grant));
    expect(complete.kind == NativeEpochResponseKind::kAcknowledged &&
               complete.operation == NativeEpochOperation::kComplete,
           "unused physically stopped epoch may complete with zero issued calls");
    const auto high_grant = exchange(endpoint, report(NativeEpochOperation::kClaim, high.grant));
    expect(high_grant.kind == NativeEpochResponseKind::kGranted,
           "distinct pending epoch dispatches after terminal");
    const auto more = exchange(endpoint, acquire_request());
    expect(more.kind == NativeEpochResponseKind::kPending, "new lower priority pending");
    const auto cancelled =
        exchange(endpoint, report(NativeEpochOperation::kCancelPending, more.grant));
    expect(cancelled.kind == NativeEpochResponseKind::kAcknowledged &&
               cancelled.grant.execution.generation == 0,
           "never-dispatched cancellation acknowledges exact task");
    expect(exchange(endpoint, report(NativeEpochOperation::kCancelPending, more.grant)).kind ==
               NativeEpochResponseKind::kError,
           "cancellation cannot replay");
    expect(exchange(endpoint, report(NativeEpochOperation::kFail, high_grant.grant)).kind ==
               NativeEpochResponseKind::kAcknowledged,
           "physically stopped failure terminal");
    TaskControlEndpoint rejected(service, reject_resource, nullptr);
    const auto before = scheduler.stats();
    expect(exchange(rejected, acquire_request()).kind == NativeEpochResponseKind::kError,
           "registration failure rejects without grant");
    expect(scheduler.stats().quota.allocated_bytes == 0 &&
               scheduler.stats().quota.reserved_bytes == 0 &&
               scheduler.stats().cancelled_task_count == before.cancelled_task_count + 1,
           "registration rollback conserves quota");
    const auto unit = service.submit({.tenant_id = "unit", .memory_bytes = 1}, register_resource,
                                     nullptr, kOwner);
    expect(unit.accepted(), "ordinary submission unchanged");
    expect(!service.claim_native_epoch(unit.task_id, kOwner) &&
               !service.cancel_pending_native_epoch(unit.task_id, kOwner),
           "epoch API cannot claim/cancel ordinary task");
    const auto original = service.claim_execution(unit.task_id, kOwner);
    expect(original.has_value(), "ordinary generation claim unchanged");
    const TaskExecutionIdentity id{checked(original).task_id, checked(original).dispatch_sequence};
    expect(!service.native_epoch_retreat_requested(id, kOwner) &&
               !service.heartbeat_native_epoch(id, kOwner) &&
               !service.finish_native_epoch(id, 0, false, kOwner),
           "epoch reports cannot consume ordinary execution");
    expect(service.complete_execution(id, kOwner), "old completion remains valid");
}

void test_deadline_loss_and_races() {
    auto extent = kExtent;
    extent.acceptance_microseconds = 1;
    Scheduler scheduler(100);
    TaskAdmissionService service(scheduler, std::chrono::milliseconds(1), true, nullptr, nullptr,
                                 nullptr, nullptr, extent);
    TaskControlEndpoint endpoint(service, register_resource, nullptr);
    auto request = acquire_request();
    request.extent = extent;
    // Discard acquire reply: running grant must remain owned, not auto-cancel.
    expect(endpoint.handle(checked(format_native_epoch_request(request)), kOwner).has_value(),
           "lost acquisition executes once");
    const auto pending = scheduler.stats();
    expect(pending.running_task_count == 1 && pending.quota.allocated_bytes == 1,
           "lost acquire retains real service ownership");
    const auto grant = service.claim_native_epoch(1, kOwner);
    expect(grant.has_value(), "fixture independently locates original task, never resubmits");
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    expect(service.heartbeat_native_epoch(checked(grant).execution, kOwner),
           "heartbeat checks owner after expiry");
    expect(checked(service.claim_native_epoch(1, kOwner)).acceptance_deadline_nanoseconds ==
               checked(grant).acceptance_deadline_nanoseconds,
           "heartbeat/reclaim cannot extend deadline");
    expect(service.native_epoch_retreat_requested(checked(grant).execution, kOwner) == true,
           "acceptance expiry requests stop");
    expect(!service.reap_expired() && scheduler.stats().running_task_count == 1,
           "timeout cannot prove native stop or release");
    expect(service.finish_native_epoch(checked(grant).execution, 0, true, kOwner),
           "explicit stopped failure releases retained grant");
    for (int iteration = 0; iteration < 100; ++iteration) {
        Scheduler concurrent_scheduler(100);
        TaskAdmissionService concurrent(concurrent_scheduler, {}, true, nullptr, nullptr, nullptr,
                                        nullptr, kExtent);
        const auto accepted = concurrent.submit_native_epoch(acquire_request().admission, kExtent,
                                                             register_resource, nullptr, kOwner);
        const auto active = concurrent.claim_native_epoch(accepted.task_id, kOwner);
        expect(active.has_value(), "race fixture grants");
        std::barrier start(3);
        std::atomic<unsigned int> terminals{0};
        std::thread first([&] {
            start.arrive_and_wait();
            if (concurrent.finish_native_epoch(active->execution, 1, false, kOwner)) {
                ++terminals;
            }
        });
        std::thread second([&] {
            start.arrive_and_wait();
            static_cast<void>(concurrent.native_epoch_retreat_requested(active->execution, kOwner));
            if (concurrent.finish_native_epoch(active->execution, 1, true, kOwner)) {
                ++terminals;
            }
        });
        start.arrive_and_wait();
        first.join();
        second.join();
        expect(terminals == 1 && concurrent_scheduler.stats().running_task_count == 0 &&
                   concurrent_scheduler.stats().quota.allocated_bytes == 0,
               "concurrent terminal/retreat releases exactly once");
    }
}

int test_socket() {
    std::array<char, 40> pattern{};
    constexpr std::string_view prefix = "/tmp/glimmer-epoch-XXXXXX";
    std::copy(prefix.begin(), prefix.end(), pattern.begin());
    const char* directory = ::mkdtemp(pattern.data());
    expect(directory != nullptr, "temporary socket directory");
    const std::string path = std::string(directory) + "/control.sock";
    Scheduler scheduler(100);
    TaskAdmissionService service(scheduler, {}, true, nullptr, nullptr, nullptr, nullptr, kExtent);
    TaskControlEndpoint endpoint(service, register_resource, nullptr);
    UnixSocketControlServer server(endpoint, {.socket_path = path, .allowed_uid = std::nullopt});
    if (!server.start()) {
        expect(::rmdir(directory) == 0, "unstarted socket directory removed");
        return 77;
    }
    std::thread accept([&] { expect(server.serve_one(), "real peer connection accepted"); });
    UnixSocketControlClient client(path);
    const auto encoded = format_native_epoch_request(acquire_request());
    const auto reply = client.request(checked(encoded));
    const auto decoded = reply ? parse_native_epoch_response(*reply) : std::nullopt;
    expect(decoded && decoded->kind == NativeEpochResponseKind::kGranted,
           "actual SO_PEERCRED identity authorizes distinct epoch");
    auto terminal = report(NativeEpochOperation::kComplete, checked(decoded).grant);
    terminal.issued_calls = 2;
    const auto terminal_reply = client.request(checked(format_native_epoch_request(terminal)));
    const auto ack = terminal_reply ? parse_native_epoch_response(*terminal_reply) : std::nullopt;
    expect(ack && ack->kind == NativeEpochResponseKind::kAcknowledged && ack->issued_calls == 2 &&
               ack->grant.execution.task_id == checked(decoded).grant.execution.task_id &&
               ack->grant.execution.generation == checked(decoded).grant.execution.generation,
           "real transport exact count/task/generation acknowledgement");
    const auto replay = client.request(checked(format_native_epoch_request(terminal)));
    expect(replay && checked(parse_native_epoch_response(*replay)).kind ==
                         NativeEpochResponseKind::kError,
           "actual connection cannot replay terminal");
    accept.join();
    server.stop();
    expect(server.served_request_count() == 3 && scheduler.stats().running_task_count == 0 &&
               scheduler.stats().quota.allocated_bytes == 0,
           "actual service drained exactly three requests");
    expect(::rmdir(directory) == 0, "only owned socket directory removed");
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--socket") {
        return test_socket();
    }
    expect(argc == 1, "known test mode");
    test_codec();
    test_policy_and_owner();
    test_priority_pending_and_rollback();
    test_deadline_loss_and_races();
    std::cout << "Native epoch admission tests passed\n";
    return 0;
}
