#include "internal/execution_submission_queue.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

namespace {
using namespace glimmer::backend;

void expect(bool condition, std::string_view message) {
    if (!condition) throw std::runtime_error(std::string(message));
}

ExecutionRequest request(std::span<const std::byte> bytes,
                         std::shared_ptr<const void> lifetime = std::make_shared<int>(0)) {
    return {.operation = 1,
            .context = 7,
            .stream = 0,
            .parameters = bytes,
            .lifetime = std::move(lifetime)};
}

ExecutionIssue take(ExecutionSubmissionQueue& queue) {
    auto issue = queue.take_next();
    expect(issue.has_value(), "a queued command must become issuable");
    return *issue;
}

void settle(ExecutionSubmissionQueue& queue, const ExecutionIssue& issue, std::int32_t status = 0) {
    expect(queue.publish_host_result(issue.token, status) &&
               queue.complete_after_physical_stop(issue.token) &&
               queue.acknowledge_result(issue.receipt),
           "the owner must settle host, physical and receipt ownership separately");
}

void charged_host_storage() {
    ExecutionSubmissionQueue queue({1, 2}, {.max_retained_host_bytes = 128});
    expect(!queue.allocate_host_storage(64).storage && queue.bind_executor(),
           "host storage requires the bound executor");
    expect(queue.allocate_host_storage(129).status == ExecutionAcceptStatus::kOutOfMemory &&
               queue.allocate_host_storage(SIZE_MAX).status == ExecutionAcceptStatus::kOutOfMemory,
           "whole-budget oversize is terminal without arithmetic wrap");
    auto first = queue.allocate_host_storage(128);
    expect(first.status == ExecutionAcceptStatus::kAccepted && first.storage->bytes() == 128 &&
               queue.snapshot().retained_host_bytes == 128 &&
               queue.snapshot().retained_parameter_bytes == 0 &&
               queue.allocate_host_storage(1).status == ExecutionAcceptStatus::kFull,
           "real backing storage consumes its exact independent budget");
    {
        std::pmr::vector<std::byte> values{first.storage->resource()};
        values.resize(128);
        bool exhausted{};
        try {
            values.resize(129);
        } catch (const std::bad_alloc&) {
            exhausted = true;
        }
        expect(exhausted && values.size() == 128 && queue.snapshot().retained_host_bytes == 128,
               "exact-capacity allocation succeeds and null upstream forbids extra storage");
    }
    const auto accepted = queue.accept(1, request({}, first.storage));
    first.storage.reset();
    const auto issued = take(queue);
    expect(accepted.accepted() && queue.publish_host_result(issued.token, 0) &&
               queue.acknowledge_result(accepted.receipt) &&
               queue.snapshot().retained_host_bytes == 128,
           "host ACK retains the real array backing until physical retirement");
    expect(queue.pause_after_physical_stop(issued.token) &&
               queue.snapshot().retained_host_bytes == 128,
           "incomplete physical retreat cannot refund host storage");
    const auto resumed = queue.resume_after_physical_stop(issued.token);
    expect(resumed && queue.finish_continuation_issue(resumed->token) &&
               queue.complete_after_physical_stop(resumed->token) &&
               queue.snapshot().retained_host_bytes == 0,
           "exact final stop refunds the backing allocation once");
    auto second = queue.allocate_host_storage(128);
    expect(second.status == ExecutionAcceptStatus::kAccepted, "released capacity is reusable");
    const auto rejected = queue.accept(2, request({}, second.storage));
    second.storage.reset();
    expect(rejected.accepted() && queue.reject_unissued(rejected.receipt, 7) &&
               queue.snapshot().retained_host_bytes == 0 &&
               queue.acknowledge_result(rejected.receipt),
           "never-issued cancellation releases storage without physical work");
    auto other = std::async(std::launch::async, [&] { return queue.allocate_host_storage(1); });
    expect(!other.get().storage, "another thread cannot allocate executor storage");
    const auto child = fork();
    expect(child >= 0, "fork host accounting control");
    if (!child)
        _exit(queue.allocate_host_storage(1).status == ExecutionAcceptStatus::kWrongProcess ? 0
                                                                                            : 1);
    int status{};
    expect(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "fork rejects before touching inherited locks");
    queue.close();
    expect(
        queue.allocate_host_storage(1).status == ExecutionAcceptStatus::kClosed && queue.shutdown(),
        "closed queues grant no new host storage");

    // Storage may outlive an otherwise empty facade. Its counter and resource
    // have no dangling queue callback, even when the final release is remote.
    std::shared_ptr<ExecutionHostStorage> survivor;
    {
        ExecutionSubmissionQueue owner({3, 4});
        expect(owner.bind_executor(), "bind detached storage owner");
        survivor = owner.allocate_host_storage(64).storage;
        std::pmr::vector<std::byte> values{survivor->resource()};
        values.resize(64);
    }
    auto cleanup = std::async(std::launch::async, [storage = std::move(survivor)] {
        expect(storage->bytes() == 64, "detached backing remains alive through remote release");
    });
    cleanup.get();
}

void owned_arguments_and_pipelined_issue() {
    ExecutionSubmissionQueue queue({1, 2});
    expect(queue.bind_executor() && queue.bind_executor(),
           "one live executor may bind idempotently");
    std::vector bytes{std::byte{1}, std::byte{2}, std::byte{3}};
    auto lease = std::make_shared<int>(9);
    std::weak_ptr<const void> retained = lease;
    const auto first = queue.accept(1, request(bytes, lease));
    expect(first.accepted() && first.receipt.sequence() == 1, "accept exact first owned command");
    lease.reset();
    bytes.assign(bytes.size(), std::byte{99});
    expect(!retained.expired() && !queue.host_result(first.receipt) &&
               !queue.acknowledge_result(first.receipt),
           "acceptance is neither an SDK reply nor resource retirement");
    const auto one = take(queue);
    expect(one.token && one.token.sequence() == 1 && one.context == 7 && one.stream == 0 &&
               one.parameters.size() == 3 && one.parameters[0] == std::byte{1},
           "issue must use a copied ABI and preserve logical context/default stream");
    const auto second = queue.accept(2, request(std::span(bytes).first(2)));
    expect(
        second.accepted() && !queue.take_next() && !queue.complete_after_physical_stop(one.token),
        "a second SDK call cannot overtake a still-issuing host call");
    expect(queue.publish_host_result(one.token, 37), "publish the real original SDK error");
    const auto reply = queue.host_result(first.receipt);
    expect(reply && reply->status == 37 && reply->origin == ExecutionReplyOrigin::kNativeSdk &&
               !queue.publish_host_result(one.token, 0),
           "original signed SDK status must publish once without being rewritten");
    expect(queue.acknowledge_result(first.receipt) && !retained.expired(),
           "client acknowledgment must not free GPU inputs, even after SDK error");
    const auto two = take(queue);
    expect(two.receipt.sequence() == 2 && queue.publish_host_result(two.token, 0),
           "a second SDK issue must progress before first GPU completion");
    expect(queue.snapshot().awaiting_physical_stop == 2 && queue.snapshot().outstanding == 2 &&
               queue.snapshot().retained_parameter_bytes == 5,
           "multiple host-issued commands retain independent physical ownership");
    expect(queue.complete_after_physical_stop(one.token) && retained.expired() &&
               !queue.complete_after_physical_stop(one.token) &&
               !queue.acknowledge_result(first.receipt),
           "exact physical proof must retire the acknowledged first command exactly once");
    expect(queue.complete_after_physical_stop(two.token) &&
               queue.snapshot().retained_parameter_bytes == 0 &&
               queue.snapshot().outstanding == 1 && queue.host_result(second.receipt),
           "physical completion releases payload but retains an unacknowledged host receipt");
    expect(queue.acknowledge_result(second.receipt) && queue.shutdown(),
           "all command and receipt ownership must drain");
}

void original_abi_prefix_keeps_full_payload_charge() {
    ExecutionSubmissionQueue queue({1, 2}, {.max_commands = 2, .max_parameter_bytes = 8});
    ExecutionSubmissionQueue foreign({3, 4});
    expect(queue.bind_executor() && foreign.bind_executor(), "bind exact prefix owners");
    const std::array input{std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4},
                           std::byte{5}, std::byte{6}, std::byte{7}, std::byte{8}};
    const auto accepted = queue.accept(1, request(input));
    const auto other = foreign.accept(1, request(input));
    expect(accepted.accepted() && other.accepted() &&
               !queue.limit_unissued_parameters(other.receipt, 4) &&
               !queue.limit_unissued_parameters(accepted.receipt, 9),
           "prefix selection requires the exact owned receipt and never grows storage");
    auto wrong_thread = std::async(
        std::launch::async, [&] { return queue.limit_unissued_parameters(accepted.receipt, 4); });
    expect(!wrong_thread.get(), "another executor cannot choose a prefix");
    const auto child = fork();
    expect(child >= 0, "fork prefix owner control");
    if (!child) _exit(queue.limit_unissued_parameters(accepted.receipt, 4) ? 1 : 0);
    int child_status{};
    expect(waitpid(child, &child_status, 0) == child && WIFEXITED(child_status) &&
               WEXITSTATUS(child_status) == 0,
           "inherited prefix selection rejects before locking");
    expect(queue.limit_unissued_parameters(accepted.receipt, 4) &&
               !queue.limit_unissued_parameters(accepted.receipt, 8) &&
               queue.snapshot().retained_parameter_bytes == 8 &&
               queue.accept(2, request(std::span{input}.first(1))).status ==
                   ExecutionAcceptStatus::kFull,
           "a shorter issue ABI cannot refund the still-allocated private argument payload");
    const auto issue = take(queue);
    expect(
        issue.parameters.size() == 4 &&
            std::ranges::equal(issue.parameters, std::span{input}.first(4)) &&
            !queue.limit_unissued_parameters(accepted.receipt, 0) &&
            !queue.patch_issue_parameters(issue.token, 4, std::span{input}.first(1)),
        "native issue sees exactly the original prefix; hidden trailing bytes cannot be patched");
    expect(queue.publish_host_result(issue.token, 37) && queue.acknowledge_result(issue.receipt) &&
               queue.snapshot().retained_parameter_bytes == 8 &&
               queue.complete_after_physical_stop(issue.token) &&
               queue.snapshot().retained_parameter_bytes == 0,
           "only actual physical retirement refunds the complete charged payload");
    const auto zero = queue.accept(2, request(input));
    queue.close();
    expect(zero.accepted() && queue.limit_unissued_parameters(zero.receipt, 0),
           "already accepted zero-argument original work can drain after close");
    const auto empty = take(queue);
    expect(empty.parameters.empty() && queue.snapshot().retained_parameter_bytes == 8,
           "zero ABI length is not zero retained storage");
    settle(queue, empty);
    settle(foreign, take(foreign));
    expect(queue.shutdown() && foreign.shutdown(), "exact prefix owners clean once");
}

void private_parameter_patches_require_live_issue_authority() {
    ExecutionSubmissionQueue queue({1, 2});
    ExecutionSubmissionQueue foreign({3, 4});
    expect(queue.bind_executor() && foreign.bind_executor(), "bind patch test executors");
    const std::array input{std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}};
    const std::array update{std::byte{9}, std::byte{8}};
    const auto accepted = queue.accept(1, request(input));
    expect(accepted.accepted() && foreign.accept(1, request(input)).accepted(),
           "accept independently owned patch targets");
    const auto issue = take(queue), other = take(foreign);
    expect(!queue.patch_issue_parameters(other.token, 0, update) &&
               !queue.patch_issue_parameters(issue.token, SIZE_MAX, update) &&
               !queue.patch_issue_parameters(issue.token, 3, update) &&
               !queue.patch_issue_parameters(issue.token, 0, {}) &&
               !std::async(std::launch::async,
                           [&] { return queue.patch_issue_parameters(issue.token, 0, update); })
                    .get(),
           "wrong queue, thread and invalid extents cannot mutate payloads");
    const auto child = fork();
    expect(child >= 0, "fork patch rejection control");
    if (child == 0) _exit(queue.patch_issue_parameters(issue.token, 0, update) ? 1 : 0);
    int status{};
    expect(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "fork rejects patch before inherited exclusion");
    expect(queue.patch_issue_parameters(issue.token, 1, update) &&
               issue.parameters[0] == input[0] && issue.parameters[1] == update[0] &&
               issue.parameters[2] == update[1] && issue.parameters[3] == input[3] &&
               input[1] == std::byte{2} && queue.snapshot().retained_parameter_bytes == 4,
           "patch only charged owned bytes; preserve caller input and payload accounting");
    expect(queue.publish_host_result(issue.token, 0) &&
               !queue.patch_issue_parameters(issue.token, 0, update) &&
               queue.pause_after_physical_stop(issue.token) &&
               !queue.patch_issue_parameters(issue.token, 0, update),
           "physical in-flight and paused commands are not issuing authority");
    queue.close();
    const auto resumed = queue.resume_after_physical_stop(issue.token);
    expect(resumed && !queue.patch_issue_parameters(issue.token, 0, update) &&
               queue.patch_issue_parameters(resumed->token, 0, update),
           "closed drain permits only the fresh continuation generation");
    queue.quarantine();
    expect(!queue.patch_issue_parameters(resumed->token, 0, update) &&
               queue.finish_continuation_issue(resumed->token) &&
               queue.complete_after_physical_stop(resumed->token) &&
               !queue.patch_issue_parameters(resumed->token, 0, update) &&
               queue.acknowledge_result(accepted.receipt) && queue.shutdown(),
           "quarantine and retirement forbid mutation, retaining real cleanup authority");
    settle(foreign, other);
    expect(foreign.shutdown(), "foreign patch target retires independently");
}

void capacity_and_sequence_backpressure() {
    ExecutionSubmissionQueue queue({1, 2}, {.max_commands = 2, .max_parameter_bytes = 6});
    expect(queue.bind_executor(), "bind bounded queue executor");
    const std::array<std::byte, 7> oversized{};
    expect(queue.accept(1, request(oversized)).status == ExecutionAcceptStatus::kOutOfMemory &&
               queue.snapshot().outstanding == 0,
           "a single reservation exceeding the whole byte budget cannot become recoverable Full");
    const std::array bytes{std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}, std::byte{5}};
    const auto first = queue.accept(1, request(bytes));
    expect(first.accepted() &&
               queue.accept(1, request({})).status == ExecutionAcceptStatus::kWrongSequence &&
               queue.accept(3, request({})).status == ExecutionAcceptStatus::kWrongSequence &&
               queue.accept(2, request(std::span(bytes).first(2))).status ==
                   ExecutionAcceptStatus::kFull,
           "duplicate/future IDs and payload overflow must reject before sequence publication");
    const auto second = queue.accept(2, request(std::span(bytes).first(1)));
    expect(second.accepted(), "full rejection must not consume sequence or byte capacity");
    const auto one = take(queue);
    const auto first_result = queue.publish_host_result(one.token, 0);
    expect(first_result && queue.complete_after_physical_stop(one.token),
           "retire first payload only");
    const auto two = take(queue);
    expect(queue.publish_host_result(two.token, 0) && queue.complete_after_physical_stop(two.token),
           "retire second payload only");
    expect(queue.snapshot().retained_parameter_bytes == 0 &&
               queue.snapshot().unacknowledged_replies == 2 &&
               queue.accept(3, request({})).status == ExecutionAcceptStatus::kFull,
           "unacknowledged replies must remain bounded even after every GPU stop");
    expect(queue.acknowledge_result(first.receipt), "release one bounded receipt");
    const auto third = queue.accept(3, request(bytes));
    expect(third.accepted(), "released receipt must permit exactly the next sequence");
    settle(queue, take(queue));
    expect(queue.acknowledge_result(second.receipt) && queue.shutdown(), "drain bounded receipts");
}

void incomplete_epochs_retain_one_logical_command() {
    ExecutionSubmissionQueue queue({1, 2}, {.max_commands = 2, .max_parameter_bytes = 4});
    expect(queue.bind_executor(), "bind persistent logical command owner");
    auto lease = std::make_shared<int>(17);
    std::weak_ptr<const void> retained = lease;
    const std::array bytes{std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}};
    const auto accepted = queue.accept(1, request(bytes, lease));
    lease.reset();
    auto current = take(queue);
    expect(current.token.generation() == 1 && !queue.pause_after_physical_stop(current.token) &&
               !queue.resume_after_physical_stop(current.token) &&
               queue.publish_host_result(current.token, 0),
           "only an actual host-returned initial epoch may acquire trusted stop proof");
    for (std::uint64_t generation = 2; generation <= 4; ++generation) {
        const auto previous = current.token;
        expect(queue.pause_after_physical_stop(previous) && !retained.expired() &&
                   queue.snapshot().paused == 1 && queue.snapshot().awaiting_physical_stop == 0 &&
                   queue.snapshot().retained_parameter_bytes == bytes.size() &&
                   !queue.complete_after_physical_stop(previous) &&
                   !queue.pause_after_physical_stop(previous),
               "incomplete physical stop is neither logical completion nor payload retirement");
        expect(queue.accept(2, request(std::array{std::byte{9}})).status ==
                   ExecutionAcceptStatus::kFull,
               "paused private payloads remain charged against backpressure");
        auto resumed = queue.resume_after_physical_stop(previous);
        expect(resumed && resumed->token.generation() == generation &&
                   resumed->receipt.sequence() == accepted.receipt.sequence() &&
                   resumed->parameters.data() == current.parameters.data() &&
                   resumed->parameters[3] == bytes[3],
               "continuation retains one exact command and immutable owned ABI");
        current = *resumed;
        expect(queue.snapshot().issuing == 1 && queue.snapshot().paused == 0 &&
                   !queue.publish_host_result(current.token, 71) &&
                   !queue.finish_continuation_issue(previous) &&
                   !queue.complete_after_physical_stop(previous) &&
                   !queue.pause_after_physical_stop(previous) &&
                   !queue.resume_after_physical_stop(previous) &&
                   queue.finish_continuation_issue(current.token) &&
                   !queue.finish_continuation_issue(current.token),
               "new generation cannot republish the reply or accept stale physical proof");
        const auto reply = queue.host_result(accepted.receipt);
        expect(reply && reply->status == 0 && reply->origin == ExecutionReplyOrigin::kNativeSdk,
               "the sole original SDK result survives every continuation unchanged");
    }
    expect(queue.acknowledge_result(accepted.receipt) && !retained.expired(),
           "ACK after several partial epochs still cannot release GPU ownership");
    queue.close();
    expect(queue.pause_after_physical_stop(current.token), "pause accepted work after close");
    const auto last = queue.resume_after_physical_stop(current.token);
    expect(last && queue.finish_continuation_issue(last->token) &&
               queue.complete_after_physical_stop(last->token) && retained.expired() &&
               queue.snapshot().outstanding == 0 && queue.shutdown(),
           "close must permit exact accepted continuation drain and one final release");
}

void private_preparation_failure_retains_physical_ownership() {
    ExecutionSubmissionQueue queue({1, 2});
    expect(queue.bind_executor(), "bind private-prefix issue owner");
    auto lease = std::make_shared<int>(0);
    std::weak_ptr<const void> retained = lease;
    const auto accepted = queue.accept(1, request(std::array{std::byte{1}}, lease));
    lease.reset();
    const auto issue = take(queue);
    expect(!queue.publish_preparation_failure(issue.token, 0) &&
               queue.publish_preparation_failure(issue.token, -17),
           "private asynchronous failure is not an original SDK launch result");
    const auto reply = queue.host_result(accepted.receipt);
    expect(reply && reply->origin == ExecutionReplyOrigin::kRejectedBeforeIssue &&
               reply->status == -17 && !queue.publish_preparation_failure(issue.token, -18) &&
               !queue.publish_host_result(issue.token, 0) &&
               queue.acknowledge_result(accepted.receipt) && !retained.expired() &&
               queue.snapshot().awaiting_physical_stop == 1 &&
               !queue.pause_after_physical_stop(issue.token),
           "rejected target and acknowledged reply still retain possibly active private inputs");
    expect(
        queue.complete_after_physical_stop(issue.token) && retained.expired() && queue.shutdown(),
        "only real prefix stop retires resources after preparation rejection");
}

void continuation_capabilities_and_uncertainty() {
    ExecutionSubmissionQueue queue({1, 2}), foreign({1, 2});
    expect(queue.bind_executor() && foreign.bind_executor(), "bind exact continuation domains");
    const auto accepted = queue.accept(1, request({}));
    const auto second = queue.accept(2, request({}));
    auto first = take(queue);
    expect(accepted.accepted() && second.accepted() && queue.publish_host_result(first.token, 0) &&
               queue.pause_after_physical_stop(first.token),
           "prepare a stopped incomplete command");
    auto other_thread = std::async(std::launch::async, [&] {
        return !queue.resume_after_physical_stop(first.token) &&
               !queue.pause_after_physical_stop(first.token) &&
               !queue.finish_continuation_issue(first.token);
    });
    expect(other_thread.get() && !foreign.resume_after_physical_stop(first.token) &&
               !foreign.pause_after_physical_stop(first.token) &&
               !foreign.finish_continuation_issue(first.token) &&
               !queue.resume_after_physical_stop({}) && !queue.pause_after_physical_stop({}) &&
               !queue.finish_continuation_issue({}),
           "wrong/default/foreign continuation authority rejects without state mutation");
    const auto issuing = take(queue);
    expect(!queue.resume_after_physical_stop(first.token),
           "continuation cannot overtake another host SDK issue");
    settle(queue, issuing);
    const auto resumed = queue.resume_after_physical_stop(first.token);
    expect(resumed && !queue.take_next() && !queue.complete_after_physical_stop(resumed->token) &&
               queue.fail_uncertain_issue(resumed->token, -19),
           "uncertain continuation preserves owned inputs and quarantines issue");
    const auto original = queue.host_result(accepted.receipt);
    expect(original && original->status == 0 &&
               original->origin == ExecutionReplyOrigin::kNativeSdk &&
               queue.snapshot().is_quarantined && queue.snapshot().awaiting_physical_stop == 1 &&
               !queue.resume_after_physical_stop(resumed->token) &&
               !queue.finish_continuation_issue(resumed->token),
           "uncertainty never rewrites the already published application result");
    expect(queue.complete_after_physical_stop(resumed->token) &&
               queue.acknowledge_result(accepted.receipt) && queue.shutdown() && foreign.shutdown(),
           "only independent real stop can settle an uncertain continuation");
}

void exact_executor_selection_preserves_capabilities() {
    ExecutionSubmissionQueue queue({1, 2}), foreign({1, 2});
    expect(queue.bind_executor() && foreign.bind_executor(), "bind selected issue owner");
    const auto first = queue.accept(1, request({})), second = queue.accept(2, request({}));
    const auto other = foreign.accept(1, request({}));
    expect(first.accepted() && second.accepted() && other.accepted() &&
               !queue.take_next(other.receipt) && !queue.take_next(ExecutionReceipt{}),
           "executor selection still requires the exact live queue receipt");
    const auto selected = queue.take_next(second.receipt);
    expect(selected && selected->token.sequence() == 2 && !queue.take_next(first.receipt) &&
               !queue.take_next(second.receipt),
           "trusted ready selection may pass blocked work, never a still-issuing host call");
    settle(queue, *selected);
    expect(!queue.take_next(second.receipt), "a retired selection cannot be revived");
    const auto remaining = queue.take_next(first.receipt);
    expect(remaining && remaining->token.sequence() == 1, "preserve the skipped exact command");
    settle(queue, *remaining);
    expect(foreign.reject_unissued(other.receipt, 1) && foreign.acknowledge_result(other.receipt) &&
               queue.shutdown() && foreign.shutdown(),
           "selected and skipped commands retain independent once-only lifetime ownership");
}

void reentrant_cleanup_keeps_bytes_charged() {
    ExecutionSubmissionQueue queue({1, 2}, {.max_commands = 2, .max_parameter_bytes = 4});
    expect(queue.bind_executor(), "bind reentrant-cleanup owner");
    ExecutionReceipt receipt;
    bool callback_ran = false;
    bool bound_preserved = false;
    bool acknowledgment_ran = false;
    auto lifetime = std::shared_ptr<const void>(new int(0), [&](const void* value) {
        callback_ran = true;
        bound_preserved = queue.snapshot().retained_parameter_bytes == 4 &&
                          queue.accept(2, request(std::array{std::byte{1}})).status ==
                              ExecutionAcceptStatus::kFull;
        acknowledgment_ran = queue.acknowledge_result(receipt);
        delete static_cast<const int*>(value);
    });
    const std::array bytes{std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}};
    const auto accepted = queue.accept(1, request(bytes, lifetime));
    expect(accepted.accepted(), "accept payload with a reentrant CPU lifetime callback");
    receipt = accepted.receipt;
    lifetime.reset();
    const auto issue = take(queue);
    expect(queue.publish_host_result(issue.token, 0) &&
               queue.complete_after_physical_stop(issue.token),
           "physical retirement must invoke the lifetime callback outside metadata locks");
    expect(callback_ran && bound_preserved && acknowledgment_ran &&
               queue.snapshot().outstanding == 0 && queue.snapshot().retained_parameter_bytes == 0,
           "callback may reenter metadata/acknowledgment without deadlock or overcommitting bytes");
    const auto next = queue.accept(2, request(bytes));
    expect(next.accepted(), "cleanup must finally release the original byte budget");
    settle(queue, take(queue));
    expect(queue.shutdown(), "reentrant cleanup leaves no retained owner");
}

void capabilities_owner_and_close() {
    ExecutionSubmissionQueue first({1, 2});
    ExecutionSubmissionQueue other({1, 2});
    expect(first.bind_executor() && other.bind_executor(), "bind two independent queues");
    const auto a = first.accept(1, request({}));
    const auto b = first.accept(2, request({}));
    const auto foreign = other.accept(1, request({}));
    expect(a.accepted() && b.accepted() && foreign.accepted(), "accept independent domains");
    const auto issue = take(first);
    auto wrong_owner = std::async(std::launch::async, [&] {
        return !first.bind_executor() && !first.take_next() &&
               !first.publish_host_result(issue.token, 0) &&
               !first.complete_after_physical_stop(issue.token) &&
               !first.reject_unissued(b.receipt, 7);
    });
    expect(wrong_owner.get(), "another executor lifetime must not mutate original issue ownership");
    expect(!other.publish_host_result(issue.token, 0) &&
               !other.complete_after_physical_stop(issue.token) &&
               !other.acknowledge_result(a.receipt) && !other.host_result(a.receipt) &&
               !first.publish_host_result({}, 0) && !first.complete_after_physical_stop({}) &&
               !first.acknowledge_result({}),
           "matching numeric sequences do not authorize foreign/default capabilities");
    first.close();
    expect(
        first.accept(3, request({})).status == ExecutionAcceptStatus::kClosed && !first.shutdown(),
        "close must reject new commands without dropping accepted work");
    settle(first, issue);
    settle(first, take(first));
    expect(first.shutdown() && !first.take_next(), "accepted work drains normally after close");
    expect(other.reject_unissued(foreign.receipt, 9), "reject only known unissued work");
    const auto rejected = other.host_result(foreign.receipt);
    expect(rejected && rejected->origin == ExecutionReplyOrigin::kRejectedBeforeIssue &&
               rejected->status == 9 && other.acknowledge_result(foreign.receipt) &&
               other.shutdown(),
           "unissued rejection must retain its explicit origin and receipt");
}

void uncertain_issue_is_not_completion() {
    ExecutionSubmissionQueue queue({1, 2});
    expect(queue.bind_executor(), "bind uncertain-issue owner");
    auto lease = std::make_shared<int>(0);
    std::weak_ptr<const void> retained = lease;
    const auto first = queue.accept(1, request(std::array{std::byte{1}}, lease));
    const auto second = queue.accept(2, request({}));
    expect(first.accepted() && second.accepted(), "prepare uncertain issue and queued successor");
    lease.reset();
    const auto issue = take(queue);
    expect(!queue.fail_uncertain_issue(issue.token, 0) && !queue.snapshot().is_quarantined,
           "uncertainty cannot fabricate a successful original result");
    expect(queue.fail_uncertain_issue(issue.token, -99), "record one uncertain original issue");
    const auto result = queue.host_result(first.receipt);
    expect(result && result->origin == ExecutionReplyOrigin::kUncertainIssue &&
               result->status == -99 && !queue.publish_host_result(issue.token, 0) &&
               queue.acknowledge_result(first.receipt) && !retained.expired() &&
               queue.snapshot().is_quarantined && !queue.take_next() &&
               queue.accept(3, request({})).status == ExecutionAcceptStatus::kQuarantined &&
               !queue.shutdown(),
           "uncertain work retains resources, forbids successor issue and cannot be replayed");
    expect(!queue.reject_unissued(second.receipt, 0) && queue.reject_unissued(second.receipt, 7) &&
               queue.acknowledge_result(second.receipt),
           "known unissued successors may be rejected during quarantine, never issued as fallback");
    expect(
        queue.complete_after_physical_stop(issue.token) && retained.expired() && queue.shutdown(),
        "only independently proven stop may retire the acknowledged uncertain original");
}

void taken_but_never_issued_rejection() {
    ExecutionSubmissionQueue queue({1, 2});
    expect(!queue.owns_executor() && queue.bind_executor() && queue.owns_executor(),
           "only explicit binding supplies executor authority");
    const auto accepted = queue.accept(1, request({}));
    expect(accepted.accepted(), "accept pre-issue rejection case");
    const auto issue = take(queue);
    auto other = std::async(std::launch::async, [&] {
        return !queue.owns_executor() && !queue.reject_before_native_issue(issue.token, 17);
    });
    expect(other.get() && !queue.reject_before_native_issue({}, 17) &&
               !queue.reject_before_native_issue(issue.token, 0),
           "pre-native rejection must require exact owner/token and a real failure");
    expect(queue.reject_before_native_issue(issue.token, 17),
           "trusted pre-native failure must reject an already taken but unissued command");
    const auto reply = queue.host_result(accepted.receipt);
    expect(reply && reply->origin == ExecutionReplyOrigin::kRejectedBeforeIssue &&
               reply->status == 17 && queue.snapshot().awaiting_physical_stop == 0 &&
               !queue.publish_host_result(issue.token, 0) &&
               !queue.reject_before_native_issue(issue.token, 17) &&
               !queue.complete_after_physical_stop(issue.token) &&
               queue.acknowledge_result(accepted.receipt) && queue.shutdown() &&
               queue.owns_executor(),
           "known pre-issue rejection retires once, preserving authority for post-close cleanup");
}

void configuration_and_exhaustion() {
    for (const auto identity : {ExecutionSessionIdentity{0, 2}, ExecutionSessionIdentity{1, 0},
                                ExecutionSessionIdentity{1, 2, 0}}) {
        bool rejected = false;
        try {
            ExecutionSubmissionQueue invalid(identity);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        expect(rejected, "invalid immutable domain identity must reject before publication");
    }
    for (const auto limits : {ExecutionSubmissionLimits{0, 1}, ExecutionSubmissionLimits{1, 0}}) {
        bool rejected = false;
        try {
            ExecutionSubmissionQueue invalid({1, 2}, limits);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        expect(rejected, "zero resource/receipt bounds must reject");
    }
    constexpr auto maximum = std::numeric_limits<std::uint64_t>::max();
    ExecutionSubmissionQueue queue({1, 2, maximum - 1});
    expect(queue.bind_executor(), "bind a domain near sequence exhaustion");
    auto invalid = request({});
    invalid.context = 0;
    expect(queue.accept(maximum - 1, invalid).status == ExecutionAcceptStatus::kInvalid,
           "unvalidated context cannot become a request");
    invalid = request({});
    invalid.operation = 0;
    expect(queue.accept(maximum - 1, invalid).status == ExecutionAcceptStatus::kInvalid,
           "missing operation cannot become a request");
    invalid = request({});
    invalid.lifetime.reset();
    expect(queue.accept(maximum - 1, invalid).status == ExecutionAcceptStatus::kInvalid,
           "accepted work must have an explicit backend lifetime");
    expect(queue.accept(maximum - 1, request({})).accepted(), "accept penultimate sequence");
    settle(queue, take(queue));
    expect(queue.accept(maximum, request({})).accepted(),
           "accept maximum sequence without wrapping");
    settle(queue, take(queue));
    expect(queue.identity().first_sequence == maximum - 1 &&
               queue.accept(1, request({})).status == ExecutionAcceptStatus::kWrongSequence &&
               queue.accept(maximum, request({})).status == ExecutionAcceptStatus::kWrongSequence &&
               queue.accept(0, request({})).status == ExecutionAcceptStatus::kInvalid &&
               queue.shutdown(),
           "exhaustion must never recycle a stale or wrapped command identity");
}

void executor_lifetime_cannot_be_rebound() {
    ExecutionSubmissionQueue queue({1, 2});
    const auto accepted = queue.accept(1, request({}));
    expect(accepted.accepted(), "accept prior to cold executor binding");
    auto executor = std::async(std::launch::async, [&] {
        expect(queue.bind_executor(), "bind a short-lived actual executor");
        const auto issue = take(queue);
        expect(queue.publish_host_result(issue.token, 0) &&
                   queue.complete_after_physical_stop(issue.token),
               "finish physical work before the executor lifetime ends");
    });
    executor.get();
    expect(!queue.snapshot().has_executor,
           "executor capability must expire with its actual thread");
    for (unsigned int iteration = 0; iteration < 4; ++iteration) {
        auto successor = std::async(std::launch::async, [&] { return queue.bind_executor(); });
        expect(!successor.get(),
               "a fresh or recycled thread ID cannot inherit old executor authority");
    }
    expect(queue.accept(2, request({})).status == ExecutionAcceptStatus::kQuarantined &&
               queue.acknowledge_result(accepted.receipt) && queue.shutdown(),
           "owner loss rejects new work but preserves already proven host receipts");
}

void fork_cannot_touch_parent_ownership() {
    auto queue = std::make_unique<ExecutionSubmissionQueue>(ExecutionSessionIdentity{1, 2});
    expect(queue->bind_executor(), "bind parent executor");
    unsigned int released{};
    auto lifetime = std::shared_ptr<const void>(new int(0), [&](const void* value) {
        ++released;
        delete static_cast<const int*>(value);
    });
    const auto accepted = queue->accept(1, request({}, lifetime));
    expect(accepted.accepted(), "prepare inherited live parent issue");
    lifetime.reset();
    const auto issue = take(*queue);
    const auto child = fork();
    expect(child >= 0, "create isolated fork ownership fixture");
    if (child == 0) {
        const bool guarded =
            !queue->bind_executor() && !queue->take_next() &&
            queue->accept(2, request({})).status == ExecutionAcceptStatus::kWrongProcess &&
            !queue->publish_host_result(issue.token, 0) &&
            !queue->fail_uncertain_issue(issue.token, 9) &&
            !queue->reject_unissued(accepted.receipt, 9) &&
            !queue->complete_after_physical_stop(issue.token) &&
            !queue->pause_after_physical_stop(issue.token) &&
            !queue->resume_after_physical_stop(issue.token) &&
            !queue->finish_continuation_issue(issue.token) && !queue->take_next(accepted.receipt) &&
            !queue->acknowledge_result(accepted.receipt) && !queue->host_result(accepted.receipt) &&
            !queue->shutdown() && queue->snapshot().is_quarantined &&
            queue->identity().client_id == 0;
        queue->close();
        queue->quarantine();
        queue.reset();
        _exit(guarded && released == 0 ? 0 : 1);
    }
    int status{};
    expect(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0 &&
               released == 0,
           "child must neither lock inherited state nor release parent-owned work");
    settle(*queue, issue);
    expect(released == 1 && queue->shutdown(),
           "parent retains exact independent cleanup authority");
}

void unfinished_destruction_retains_ownership() {
    // Intentionally retained unsafe work belongs only to this isolated child.
    // Normal parent teardown remains subject to fatal leak checking.
    const auto child = fork();
    expect(child >= 0, "create isolated unsafe-destruction fixture");
    if (child == 0) {
        unsigned int released{};
        auto queue = std::make_unique<ExecutionSubmissionQueue>(ExecutionSessionIdentity{1, 2});
        auto lifetime = std::shared_ptr<const void>(new int(0), [&](const void* value) {
            ++released;
            delete static_cast<const int*>(value);
        });
        bool valid = queue->bind_executor();
        const auto accepted = queue->accept(1, request({}, lifetime));
        lifetime.reset();
        const auto issue = queue->take_next();
        valid = valid && accepted.accepted() && issue.has_value();
        if (issue)
            valid = valid && queue->publish_host_result(issue->token, 0) &&
                    queue->acknowledge_result(accepted.receipt) && !queue->shutdown();
        queue.reset();
        _exit(valid && released == 0 ? 0 : 1);
    }
    int status{};
    expect(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "destruction without physical proof must retain payload even after successful reply");
}

void concurrent_owned_pipeline() {
    constexpr unsigned int count = 256;
    ExecutionSubmissionQueue queue({1, 2}, {.max_commands = 32, .max_parameter_bytes = 32});
    std::atomic<bool> producer_done{};
    std::atomic<unsigned int> destroyed{};
    auto executor = std::async(std::launch::async, [&] {
        expect(queue.bind_executor(), "bind concurrent executor");
        std::vector<ExecutionIssueToken> physical;
        unsigned int issued{};
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{10};
        for (;;) {
            if (const auto issue = queue.take_next()) {
                expect(issue->parameters.size() == 1 &&
                           issue->parameters[0] == std::byte{static_cast<unsigned char>(issued)},
                       "concurrent original issue must preserve copied payload and FIFO order");
                ++issued;
                expect(queue.publish_host_result(issue->token, 0),
                       "publish actual concurrent SDK return");
                physical.push_back(issue->token);
            }
            if (physical.size() == 16 || (producer_done && queue.snapshot().queued == 0)) {
                for (const auto& token : physical)
                    expect(queue.complete_after_physical_stop(token),
                           "complete each fake physical issue once");
                physical.clear();
            }
            if (producer_done && queue.snapshot().outstanding == 0) break;
            expect(std::chrono::steady_clock::now() < deadline,
                   "concurrent pipeline must finish within bound");
            std::this_thread::yield();
        }
        return issued;
    });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{10};
    for (unsigned int index = 0; index < count; ++index) {
        const auto bytes = std::array{std::byte{static_cast<unsigned char>(index)}};
        auto lifetime = std::shared_ptr<const void>(new int(0), [&](const void* value) {
            ++destroyed;
            delete static_cast<const int*>(value);
        });
        ExecutionAcceptance accepted;
        do {
            accepted = queue.accept(index + 1, request(bytes, lifetime));
            expect(accepted.accepted() || accepted.status == ExecutionAcceptStatus::kFull,
                   "concurrent acceptance may backpressure but not corrupt sequence");
            expect(std::chrono::steady_clock::now() < deadline, "bounded producer acceptance");
            std::this_thread::yield();
        } while (!accepted.accepted());
        lifetime.reset();
        while (!queue.host_result(accepted.receipt)) {
            expect(std::chrono::steady_clock::now() < deadline,
                   "SDK reply must precede client acknowledgment");
            std::this_thread::yield();
        }
        expect(queue.acknowledge_result(accepted.receipt),
               "producer acknowledges each real host reply once");
    }
    producer_done = true;
    expect(executor.get() == count && destroyed == count && queue.shutdown(),
           "all accepted payloads, physical work and host receipts must drain without leaks");
}

}  // namespace

int main() {
    try {
        owned_arguments_and_pipelined_issue();
        charged_host_storage();
        private_parameter_patches_require_live_issue_authority();
        original_abi_prefix_keeps_full_payload_charge();
        capacity_and_sequence_backpressure();
        incomplete_epochs_retain_one_logical_command();
        private_preparation_failure_retains_physical_ownership();
        continuation_capabilities_and_uncertainty();
        exact_executor_selection_preserves_capabilities();
        reentrant_cleanup_keeps_bytes_charged();
        capabilities_owner_and_close();
        uncertain_issue_is_not_completion();
        taken_but_never_issued_rejection();
        configuration_and_exhaustion();
        executor_lifetime_cannot_be_rebound();
        fork_cannot_touch_parent_ownership();
        unfinished_destruction_retains_ownership();
        concurrent_owned_pipeline();
        std::cout << "Owned execution submission lifecycle checks passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
