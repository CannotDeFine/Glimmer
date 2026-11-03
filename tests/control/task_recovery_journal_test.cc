#include "internal/task_recovery_journal.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <span>
#include <string_view>
#include <thread>

namespace {
using namespace glimmer::control;
using glimmer::control::internal::TaskRecoveryJournal;
using Decision = TaskRecoveryJournal::Decision;
constexpr TaskRecoveryJournal::Token kIncarnation{11, 12};
constexpr TaskRecoveryJournal::Token kNonce{21, 22};
constexpr TaskPeerIdentity kPeer{123, 456, 789};

void expect(bool value, std::string_view message) {
    if (!value) {
        std::cerr << "Test failed: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

std::span<const std::byte> bytes(std::string_view value) {
    return std::as_bytes(std::span{value.data(), value.size()});
}

bool result_is(const TaskRecoveryJournal::BeginResult& result, std::string_view expected) {
    return result.result_size == expected.size() &&
           std::equal(bytes(expected).begin(), bytes(expected).end(), result.result.begin());
}

void test_recovery_and_identity() {
    TaskRecoveryJournal journal(kIncarnation, {.sessions = 2, .entries = 2});
    const auto opened = journal.open(kNonce, kPeer);
    expect(opened && journal.open(kNonce, kPeer) == opened, "OPEN retries retain session id");
    const auto session = opened.value_or(TaskRecoveryJournal::Session{});
    expect(!journal.open(kNonce, {kPeer.pid, kPeer.uid, kPeer.start_time_ticks + 1}),
           "another process cannot reuse a live nonce");
    auto wrong = session;
    wrong.incarnation[0]++;
    expect(journal.begin(wrong, 1, 0, bytes("submit")).decision == Decision::kRejected,
           "different incarnation rejects");
    wrong = session;
    wrong.peer.uid++;
    expect(journal.begin(wrong, 1, 0, bytes("submit")).decision == Decision::kRejected,
           "different uid rejects");
    wrong = session;
    wrong.peer.pid++;
    expect(journal.begin(wrong, 1, 0, bytes("submit")).decision == Decision::kRejected,
           "different pid rejects");
    wrong = session;
    wrong.peer.start_time_ticks++;
    expect(journal.begin(wrong, 1, 0, bytes("submit")).decision == Decision::kRejected,
           "pid reuse rejects");
    wrong = session;
    wrong.nonce[1]++;
    expect(journal.begin(wrong, 1, 0, bytes("submit")).decision == Decision::kRejected,
           "different nonce rejects");
    expect(journal.begin(session, 2, 0, bytes("skip")).decision == Decision::kRejected &&
               journal.begin(session, 1, 0, bytes("submit")).decision == Decision::kExecute &&
               journal.begin(session, 1, 0, bytes("submit")).decision == Decision::kInProgress &&
               !journal.acknowledge(session, 1) && !journal.close(session),
           "sequence, in-progress and premature ACK fail closed");
    expect(journal.commit(session, 1, bytes("accepted 42")) &&
               !journal.commit(session, 1, bytes("accepted 42")),
           "result commits exactly once");
    const auto recovered = journal.begin(session, 1, 0, bytes("submit"));
    expect(recovered.decision == Decision::kReplay && result_is(recovered, "accepted 42"),
           "lost admission response recovers the same task id");
    expect(journal.begin(session, 2, 1, bytes("claim 42")).decision == Decision::kExecute &&
               journal.commit(session, 2, bytes("grant 42 7")),
           "ACK reclaims the old entry and reserves a claim");
    const auto grant = journal.begin(session, 2, 1, bytes("claim 42"));
    expect(grant.decision == Decision::kReplay && result_is(grant, "grant 42 7") &&
               !journal.close(session) && journal.acknowledge(session, 2) && journal.close(session),
           "claim generation survives lost reply until acknowledged");
    const auto reopened = journal.open(kNonce, kPeer);
    expect(reopened && reopened->id > session.id &&
               journal.begin(session, 3, 2, bytes("old")).decision == Decision::kRejected,
           "reused session slot cannot accept a stale session id");
    expect(journal.close(reopened.value_or(TaskRecoveryJournal::Session{})),
           "unused reopened session closes");
}

void test_mismatch_and_capacity() {
    TaskRecoveryJournal journal(kIncarnation, {.sessions = 1, .entries = 1});
    const auto session = journal.open(kNonce, kPeer).value_or(TaskRecoveryJournal::Session{});
    expect(session.id != 0 && !journal.open({23, 24}, kPeer), "session cap is bounded");
    expect(journal.begin(session, 1, 0, bytes("submit")).decision == Decision::kExecute &&
               journal.begin(session, 1, 0, bytes("other")).decision == Decision::kRejected &&
               journal.commit(session, 1, bytes("accepted 9")),
           "mismatched duplicate poisons new operations but preserves the committed result");
    const auto recovery = journal.begin(session, 1, 0, bytes("submit"));
    expect(recovery.decision == Decision::kReplay && result_is(recovery, "accepted 9") &&
               journal.begin(session, 2, 0, bytes("new")).decision == Decision::kRejected &&
               journal.acknowledge(session, 1) && journal.close(session),
           "poisoned session can recover and acknowledge its already reserved result");
    const auto fresh = journal.open({23, 24}, kPeer).value_or(TaskRecoveryJournal::Session{});
    expect(journal.begin(fresh, 1, 0, bytes("one")).decision == Decision::kExecute &&
               journal.commit(fresh, 1, bytes("done")) &&
               journal.begin(fresh, 2, 0, bytes("two")).decision == Decision::kFull &&
               journal.begin(fresh, 2, 1, bytes("two")).decision == Decision::kExecute &&
               journal.commit(fresh, 2, bytes("done two")) && journal.acknowledge(fresh, 2) &&
               journal.close(fresh),
           "full journal retains unacknowledged results and frees only acknowledged entries");
}

void test_concurrent_duplicate() {
    TaskRecoveryJournal journal(kIncarnation, {.sessions = 1, .entries = 1});
    const auto session = journal.open(kNonce, kPeer).value_or(TaskRecoveryJournal::Session{});
    std::barrier start(9);
    std::array<std::thread, 8> workers;
    std::atomic<unsigned> executions{0};
    std::atomic<unsigned> in_progress{0};
    for (auto& worker : workers) {
        worker = std::thread([&] {
            start.arrive_and_wait();
            const auto result = journal.begin(session, 1, 0, bytes("claim"));
            if (result.decision == Decision::kExecute) ++executions;
            if (result.decision == Decision::kInProgress) ++in_progress;
        });
    }
    start.arrive_and_wait();
    for (auto& worker : workers) worker.join();
    expect(executions == 1 && in_progress == 7 && journal.commit(session, 1, bytes("grant 9")),
           "concurrent duplicate operations execute once");
    const auto replay = journal.begin(session, 1, 0, bytes("claim"));
    expect(replay.decision == Decision::kReplay && result_is(replay, "grant 9") &&
               journal.acknowledge(session, 1) && journal.close(session),
           "concurrent claim recovers one grant");
}

void test_concurrent_close_and_commit_bounds() {
    TaskRecoveryJournal journal(kIncarnation, {.sessions = 1, .entries = 1});
    const auto session = journal.open(kNonce, kPeer).value_or(TaskRecoveryJournal::Session{});
    std::array<std::byte, TaskRecoveryJournal::k_max_message_bytes + 1> oversized{};
    expect(journal.begin(session, 1, 0, bytes("claim")).decision == Decision::kExecute &&
               !journal.commit(session, 1, oversized) && !journal.commit(session, 1, {}) &&
               journal.commit(session, 1, bytes("grant")),
           "invalid result size cannot consume a reservation");
    std::barrier start(4);
    bool acknowledged = false;
    bool closed = false;
    TaskRecoveryJournal::BeginResult replay;
    std::thread ack([&] {
        start.arrive_and_wait();
        acknowledged = journal.acknowledge(session, 1);
    });
    std::thread close([&] {
        start.arrive_and_wait();
        closed = journal.close(session);
    });
    std::thread read([&] {
        start.arrive_and_wait();
        replay = journal.begin(session, 1, 0, bytes("claim"));
    });
    start.arrive_and_wait();
    ack.join();
    close.join();
    read.join();
    expect(acknowledged && (replay.decision == Decision::kRejected ||
                            (replay.decision == Decision::kReplay && result_is(replay, "grant"))),
           "close, ACK and replay produce only complete copied results");
    expect(closed || journal.close(session), "acknowledged session can close");
    expect(!journal.commit(session, 1, bytes("late")) &&
               journal.begin(session, 1, 0, bytes("claim")).decision == Decision::kRejected,
           "stale concurrent operations cannot affect a closed session");
}

void test_concurrent_mismatch_and_commit() {
    for (unsigned iteration = 0; iteration < 32; ++iteration) {
        TaskRecoveryJournal journal(kIncarnation, {.sessions = 1, .entries = 1});
        const auto session = journal.open(kNonce, kPeer).value_or(TaskRecoveryJournal::Session{});
        expect(journal.begin(session, 1, 0, bytes("original")).decision == Decision::kExecute,
               "reserve original operation");
        std::barrier start(3);
        bool committed = false;
        Decision mismatch = Decision::kExecute;
        std::thread writer([&] {
            start.arrive_and_wait();
            committed = journal.commit(session, 1, bytes("result"));
        });
        std::thread conflicting([&] {
            start.arrive_and_wait();
            mismatch = journal.begin(session, 1, 0, bytes("different")).decision;
        });
        start.arrive_and_wait();
        writer.join();
        conflicting.join();
        expect(committed && mismatch == Decision::kRejected &&
                   result_is(journal.begin(session, 1, 0, bytes("original")), "result") &&
                   journal.begin(session, 2, 0, bytes("new")).decision == Decision::kRejected &&
                   journal.acknowledge(session, 1) && journal.close(session),
               "concurrent mismatch cannot erase the original recoverable result");
    }
}

void test_cross_session_capacity_and_ack() {
    TaskRecoveryJournal journal(kIncarnation, {.sessions = 2, .entries = 1});
    const auto first = journal.open(kNonce, kPeer).value_or(TaskRecoveryJournal::Session{});
    const auto second = journal.open({23, 24}, kPeer).value_or(TaskRecoveryJournal::Session{});
    expect(first.id != second.id &&
               journal.begin(first, 1, 0, bytes("one")).decision == Decision::kExecute &&
               journal.commit(first, 1, bytes("first result")) &&
               journal.begin(second, 1, 0, bytes("other")).decision == Decision::kFull &&
               !journal.acknowledge(first, 2) &&
               result_is(journal.begin(first, 1, 0, bytes("one")), "first result") &&
               journal.acknowledge(first, 1) &&
               journal.begin(second, 1, 0, bytes("other")).decision == Decision::kExecute &&
               journal.commit(second, 1, bytes("second result")) &&
               journal.acknowledge(second, 1) && journal.close(first) && journal.close(second),
           "sessions cannot evict each other's unacknowledged entries");

    TaskRecoveryJournal two_entries(kIncarnation, {.sessions = 1, .entries = 2});
    const auto session = two_entries.open(kNonce, kPeer).value_or(TaskRecoveryJournal::Session{});
    expect(two_entries.begin(session, 1, 0, bytes("one")).decision == Decision::kExecute &&
               two_entries.commit(session, 1, bytes("result one")) &&
               two_entries.begin(session, 2, 0, bytes("two")).decision == Decision::kExecute &&
               !two_entries.acknowledge(session, 2) &&
               result_is(two_entries.begin(session, 1, 0, bytes("one")), "result one") &&
               two_entries.commit(session, 2, bytes("result two")) &&
               two_entries.acknowledge(session, 2) && two_entries.close(session),
           "failed ACK cannot partially reclaim a committed prefix");
}

void test_message_boundaries() {
    TaskRecoveryJournal journal(kIncarnation, {.sessions = 1, .entries = 1});
    const auto session = journal.open(kNonce, kPeer).value_or(TaskRecoveryJournal::Session{});
    std::array<std::byte, TaskRecoveryJournal::k_max_message_bytes> full{};
    std::array<std::byte, TaskRecoveryJournal::k_max_message_bytes + 1> oversized{};
    full.front() = std::byte{1};
    full.back() = std::byte{2};
    expect(journal.begin(session, 1, 0, {}).decision == Decision::kRejected &&
               journal.begin(session, 1, 0, oversized).decision == Decision::kRejected &&
               journal.begin(session, 1, 0, full).decision == Decision::kExecute &&
               journal.commit(session, 1, full),
           "empty and oversized messages reject without consuming the sequence");
    const auto replay = journal.begin(session, 1, 0, full);
    expect(replay.decision == Decision::kReplay && replay.result_size == full.size() &&
               std::equal(full.begin(), full.end(), replay.result.begin()) &&
               journal.acknowledge(session, 1) && journal.close(session),
           "maximum-size request and result replay without truncation");
}

void test_exhaustion() {
    constexpr auto maximum = std::numeric_limits<std::uint64_t>::max();
    TaskRecoveryJournal journal(
        kIncarnation,
        {.sessions = 1, .entries = 1, .first_session_id = maximum, .first_operation_id = maximum});
    const auto last = journal.open(kNonce, kPeer).value_or(TaskRecoveryJournal::Session{});
    expect(last.id == maximum &&
               journal.begin(last, maximum, maximum - 1, bytes("last")).decision ==
                   Decision::kExecute &&
               journal.commit(last, maximum, bytes("done")) &&
               journal.begin(last, maximum, maximum - 1, bytes("last")).decision ==
                   Decision::kReplay &&
               journal.acknowledge(last, maximum) && journal.close(last) &&
               !journal.open({23, 24}, kPeer),
           "counter exhaustion never reuses an id");
}
}  // namespace

int main() {
    test_recovery_and_identity();
    test_mismatch_and_capacity();
    test_concurrent_duplicate();
    test_concurrent_close_and_commit_bounds();
    test_concurrent_mismatch_and_commit();
    test_cross_session_capacity_and_ack();
    test_message_boundaries();
    test_exhaustion();
    return EXIT_SUCCESS;
}
