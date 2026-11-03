#include "internal/native_execution_epoch.h"

#include <array>
#include <barrier>
#include <iostream>
#include <latch>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
using namespace glimmer::interceptor;
using namespace std::chrono_literals;
constexpr NativeEpochScope kScope{1, 2, 3};
const NativeExecutionEpoch::TimePoint kStart{1us};

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
NativeEpochAuthorization authorization() {
    return {{100, 200}, 8, 2, 2, 10us};
}
NativeEpochIssue enter(NativeExecutionEpoch& epoch) {
    auto issue = epoch.reserve(kScope, kStart);
    require(issue.has_value() && epoch.enter_original(*issue, kStart), "synthetic original entry");
    return issue.value_or(NativeEpochIssue{});
}
void call(NativeExecutionEpoch& epoch) {
    const auto issue = enter(epoch);
    require(epoch.original_returned(issue, NativeEpochCallOutcome::kSuccess),
            "synthetic original return");
}
NativeEpochFence record(NativeExecutionEpoch& epoch) {
    auto fence = epoch.prepare_fence();
    require(fence.has_value() && epoch.publish_fence_result(*fence, true),
            "synthetic fence record, NOT physical proof");
    return fence.value_or(NativeEpochFence{});
}
void configuration() {
    for (unsigned int index = 0; index < 11; ++index) {
        auto bounds = authorization();
        auto scope = kScope;
        switch (index) {
            case 0:
                bounds.execution.task_id = 0;
                break;
            case 1:
                bounds.execution.generation = 0;
                break;
            case 2:
                bounds.max_calls = 0;
                break;
            case 3:
                bounds.max_calls = 65537;
                break;
            case 4:
                bounds.calls_per_prefix = 0;
                break;
            case 5:
                bounds.calls_per_prefix = 9;
                break;
            case 6:
                bounds.max_pending_prefixes = 0;
                break;
            case 7:
                bounds.max_pending_prefixes = 17;
                break;
            case 8:
                bounds.acceptance_duration = 0us;
                break;
            case 9:
                bounds.acceptance_duration = 2s;
                break;
            case 10:
                scope.caller_registration = 0;
                break;
            default:
                break;
        }
        bool rejected = false;
        try {
            NativeExecutionEpoch invalid(scope, bounds, kStart);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected, "invalid explicit authorization rejects before use");
    }
    for (const auto scope : {NativeEpochScope{0, 2, 3}, NativeEpochScope{1, 0, 3}}) {
        bool rejected = false;
        try {
            NativeExecutionEpoch invalid(scope, authorization(), kStart);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected, "invalid registration scope rejects");
    }
}
void issue_identity_and_entry() {
    NativeExecutionEpoch epoch(kScope, authorization(), kStart);
    NativeExecutionEpoch foreign(kScope, authorization(), kStart);
    const auto initial = epoch.snapshot();
    for (const auto scope :
         {NativeEpochScope{4, 2, 3}, NativeEpochScope{1, 4, 3}, NativeEpochScope{1, 2, 4}}) {
        require(!epoch.reserve(scope, kStart) && epoch.snapshot() == initial,
                "foreign scope cannot mutate epoch");
    }
    const auto issue = epoch.reserve(kScope, kStart);
    const auto other = foreign.reserve(kScope, kStart);
    require(issue && other && epoch.snapshot().has_reserved_issue && !epoch.reserve(kScope, kStart),
            "one borrowed original acceptance at a time");
    if (!issue || !other) throw std::runtime_error("missing issue after acceptance");
    const auto reserved = epoch.snapshot();
    require(!epoch.enter_original(*other, kStart) && !epoch.discard_unissued(*other) &&
                !epoch.original_returned(*issue, NativeEpochCallOutcome::kSuccess) &&
                !epoch.enter_original({}, kStart) && epoch.snapshot() == reserved,
            "foreign/default/unentered issue changes nothing");
    require(epoch.discard_unissued(*issue) && !epoch.discard_unissued(*issue) &&
                !epoch.enter_original(*issue, kStart) && epoch.snapshot().issued_calls == 0,
            "pre-entry cancellation consumes only its ticket");
    const auto accepted = enter(epoch);
    const auto entered = epoch.snapshot();
    require(entered.has_entered_original && !entered.has_reserved_issue &&
                entered.issued_calls == 1 && !epoch.discard_unissued(accepted) &&
                !epoch.enter_original(accepted, kStart) && !epoch.prepare_fence() &&
                !epoch.prove_scope_stopped(kScope) && !epoch.prepare_terminal() &&
                epoch.snapshot() == entered,
            "entry never authorizes replay, cancellation, stop or terminal proof");
    require(!epoch.original_returned(accepted, static_cast<NativeEpochCallOutcome>(255)) &&
                epoch.snapshot() == entered,
            "invalid outcome retains exact active original");
    require(epoch.request_stop() &&
                epoch.original_returned(accepted, NativeEpochCallOutcome::kSuccess) &&
                !epoch.original_returned(accepted, NativeEpochCallOutcome::kSuccess) &&
                !epoch.reserve(kScope, kStart) && !epoch.prepare_terminal(),
            "stop settles accepted work but cannot complete unfenced GPU work");
    const auto fence = record(epoch);
    require(epoch.snapshot().completed_calls == 0 && !epoch.prepare_terminal(),
            "record is not physical proof");
    require(epoch.prove_fence(fence) && !epoch.prove_fence(fence),
            "synthetic physical proof exactly once");
    const auto terminal = epoch.prepare_terminal();
    require(terminal && terminal->execution().task_id == 100 &&
                terminal->execution().generation == 200 && terminal->issued_calls() == 1 &&
                terminal->disposition() == NativeEpochDisposition::kComplete &&
                epoch.snapshot().has_ownership && !epoch.prepare_terminal(),
            "one exact pending terminal report");
    if (!terminal) throw std::runtime_error("missing terminal after physical proof");
    require(epoch.publish_terminal_result(*terminal, true) && !epoch.snapshot().has_ownership &&
                !epoch.publish_terminal_result(*terminal, true) && !epoch.request_stop() &&
                !epoch.prove_scope_stopped(kScope),
            "only once-only terminal acknowledgment releases ownership");
}
void deadlines() {
    NativeExecutionEpoch epoch(kScope, authorization(), kStart);
    auto issue = epoch.reserve(kScope, kStart + 9us);
    require(issue && !epoch.enter_original(*issue, kStart + 10us) &&
                epoch.snapshot().has_ownership && !epoch.prepare_terminal() &&
                epoch.discard_unissued(*issue),
            "expiry retains accepted CPU ticket until discarded");
    const auto empty = epoch.prepare_terminal();
    require(empty && empty->issued_calls() == 0 &&
                empty->disposition() == NativeEpochDisposition::kComplete,
            "unused expired extent may finish without invented GPU work");
    NativeExecutionEpoch backwards(kScope, authorization(), kStart);
    issue = backwards.reserve(kScope, kStart + 1us);
    require(issue && !backwards.enter_original(*issue, kStart) && backwards.snapshot().has_failed &&
                backwards.discard_unissued(*issue),
            "backward clock stops instead of wrapping duration");
    const auto failed = backwards.prepare_terminal();
    require(failed && failed->disposition() == NativeEpochDisposition::kFail,
            "backward-clock unused grant requires failure report");
    NativeExecutionEpoch saturated(kScope, authorization(), NativeExecutionEpoch::TimePoint::max());
    require(!saturated.reserve(kScope, NativeExecutionEpoch::TimePoint::max()) &&
                saturated.prepare_terminal(),
            "saturated deadline cannot wrap into another acceptance window");
    NativeExecutionEpoch minimum(kScope, authorization(), NativeExecutionEpoch::TimePoint::min());
    issue = minimum.reserve(kScope, NativeExecutionEpoch::TimePoint::min());
    require(issue && minimum.discard_unissued(*issue),
            "minimum clock boundary accepts without overflowing subtraction");
}
void prefix_capacity_and_covering_proof() {
    NativeExecutionEpoch epoch(kScope, authorization(), kStart);
    call(epoch);
    call(epoch);
    require(!epoch.reserve(kScope, kStart), "per-prefix call bound backpressures acceptance");
    const auto prepared = epoch.prepare_fence();
    require(prepared && prepared->covered_calls() == 2 && !epoch.prove_fence(*prepared) &&
                !epoch.prepare_fence() && !epoch.reserve(kScope, kStart) &&
                !epoch.prove_scope_stopped(kScope),
            "in-flight record is neither completion nor permission to enter another original");
    if (!prepared) throw std::runtime_error("missing prepared fence");
    require(
        epoch.publish_fence_result(*prepared, true) && !epoch.publish_fence_result(*prepared, true),
        "record publication once");
    call(epoch);
    call(epoch);
    const auto later = record(epoch);
    require(later.covered_calls() == 4 && !epoch.reserve(kScope, kStart) &&
                epoch.snapshot().pending_prefixes == 2,
            "pending physical-prefix capacity remains bounded");
    require(
        epoch.prove_fence(later) && epoch.snapshot().completed_calls == 4 &&
            epoch.snapshot().pending_prefixes == 0 && !epoch.prove_fence(*prepared),
        "later same-stream proof covers earlier recorded prefixes without duplicate completion");
    call(epoch);
    const auto reused = record(epoch);
    require(!epoch.prove_fence(later) && !epoch.publish_fence_result(later, false) &&
                epoch.prove_fence(reused),
            "reused slot does not reuse old fence capability");
    const auto issue = epoch.reserve(kScope, kStart);
    require(issue && epoch.discard_unissued(*issue),
            "capacity resumes after actual declared proof");
}
void failure_and_whole_scope_proof() {
    for (const auto outcome :
         {NativeEpochCallOutcome::kRejected, NativeEpochCallOutcome::kUncertain}) {
        NativeExecutionEpoch epoch(kScope, authorization(), kStart);
        const auto issue = enter(epoch);
        require(epoch.original_returned(issue, outcome) && !epoch.reserve(kScope, kStart) &&
                    !epoch.prepare_terminal() && epoch.snapshot().issued_calls == 1,
                "native failure/uncertainty still retains entered original prefix");
        const auto fence = epoch.prepare_fence();
        require(fence && epoch.publish_fence_result(*fence, false) && !epoch.prove_fence(*fence) &&
                    !epoch.publish_fence_result(*fence, true) && !epoch.prepare_terminal() &&
                    !epoch.prepare_fence(),
                "failed recording cannot replay or release ownership");
        const auto retained = epoch.snapshot();
        require(!epoch.prove_scope_stopped({1, 2, 4}) && epoch.snapshot() == retained,
                "foreign completion cannot settle a failed prefix");
        require(epoch.prove_scope_stopped(kScope) && !epoch.prove_scope_stopped(kScope),
                "independent synthetic scope proof settles exact failed metadata once");
        const auto terminal = epoch.prepare_terminal();
        require(terminal && terminal->disposition() == NativeEpochDisposition::kFail &&
                    epoch.publish_terminal_result(*terminal, false) &&
                    epoch.snapshot().has_ownership &&
                    epoch.snapshot().phase == NativeEpochPhase::kRetained &&
                    !epoch.publish_terminal_result(*terminal, true) && !epoch.prepare_terminal() &&
                    !epoch.prove_scope_stopped(kScope),
                "lost terminal response permanently retains ownership without retry");
    }
}
void completion_and_control_failures() {
    NativeExecutionEpoch epoch(kScope, authorization(), kStart);
    call(epoch);
    const auto fence = record(epoch);
    const auto before = epoch.snapshot();
    require(!epoch.fail_fence_completion({}) &&
                !epoch.request_stop(static_cast<NativeEpochStopReason>(255)) &&
                epoch.snapshot() == before,
            "invalid failure publication cannot mutate live metadata");
    const auto entered = enter(epoch);
    require(epoch.fail_fence_completion(fence) && !epoch.fail_fence_completion(fence) &&
                !epoch.prove_fence(fence) && !epoch.prove_scope_stopped(kScope) &&
                !epoch.prepare_terminal(),
            "completion-query failure retains prior prefix and entered original");
    require(epoch.original_returned(entered, NativeEpochCallOutcome::kSuccess) &&
                !epoch.reserve(kScope, kStart) && !epoch.prepare_terminal(),
            "returned original cannot bypass failed preceding completion proof");
    const auto tail = record(epoch);
    require(epoch.prove_fence(tail) && epoch.snapshot().completed_calls == 2 &&
                epoch.snapshot().pending_prefixes == 1 && !epoch.prepare_terminal(),
            "covering proof does not silently retire the failed event owner's metadata");
    require(epoch.prove_scope_stopped(kScope),
            "independent scope proof covers failed query without retry");
    auto terminal = epoch.prepare_terminal();
    require(terminal && terminal->disposition() == NativeEpochDisposition::kFail &&
                epoch.publish_terminal_result(*terminal, true),
            "completion failure reports failure despite later physical stop");
    NativeExecutionEpoch control_failure(kScope, authorization(), kStart);
    const auto ticket = control_failure.reserve(kScope, kStart);
    require(ticket && control_failure.request_stop(NativeEpochStopReason::kFailure) &&
                !control_failure.enter_original(*ticket, kStart) &&
                control_failure.discard_unissued(*ticket),
            "control failure stops without issuing borrowed original");
    terminal = control_failure.prepare_terminal();
    require(terminal && terminal->disposition() == NativeEpochDisposition::kFail &&
                control_failure.publish_terminal_result(*terminal, false) &&
                control_failure.snapshot().has_ownership,
            "unknown control report cannot release even an unused grant");
}
void foreign_fences_and_terminal() {
    NativeExecutionEpoch epoch(kScope, authorization(), kStart);
    NativeExecutionEpoch foreign(kScope, authorization(), kStart);
    call(epoch);
    call(foreign);
    const auto own = record(epoch);
    const auto other = record(foreign);
    const auto before = epoch.snapshot();
    require(!epoch.prove_fence(other) && !epoch.fail_fence_completion(other) &&
                !epoch.publish_fence_result(other, false) && !epoch.prove_fence({}) &&
                epoch.snapshot() == before,
            "same counts/grant do not transfer fence ownership");
    require(epoch.prove_fence(own) && foreign.prove_fence(other) && epoch.request_stop() &&
                foreign.request_stop(),
            "each owner completes only its own recorded prefix");
    const auto own_terminal = epoch.prepare_terminal();
    const auto foreign_terminal = foreign.prepare_terminal();
    const auto reporting = epoch.snapshot();
    require(own_terminal && foreign_terminal &&
                !epoch.publish_terminal_result(*foreign_terminal, true) &&
                !epoch.publish_terminal_result({}, true) && epoch.snapshot() == reporting &&
                epoch.publish_terminal_result(*own_terminal, true),
            "terminal ownership cannot transfer even with equal identities");
    std::optional<NativeExecutionEpoch> replacement;
    replacement.emplace(kScope, authorization(), kStart);
    const auto stale_issue = replacement->reserve(kScope, kStart);
    replacement.reset();
    replacement.emplace(kScope, authorization(), kStart);
    const auto live_issue = replacement->reserve(kScope, kStart);
    require(stale_issue && live_issue && !replacement->enter_original(*stale_issue, kStart) &&
                replacement->discard_unissued(*live_issue),
            "recycled owner storage cannot revive retained CPU capability");
}
void exhaustive_bounded_prefixes() {
    for (std::uint32_t calls = 1; calls <= 16; ++calls) {
        for (std::uint32_t width = 1; width <= calls; ++width) {
            for (std::uint32_t capacity = 1; capacity <= 4; ++capacity) {
                auto bounds = authorization();
                bounds.max_calls = calls;
                bounds.calls_per_prefix = width;
                bounds.max_pending_prefixes = capacity;
                NativeExecutionEpoch epoch(kScope, bounds, kStart);
                std::vector<NativeEpochFence> recorded;
                for (std::uint32_t issued = 0; issued < calls; ++issued) {
                    call(epoch);
                    const auto state = epoch.snapshot();
                    require(
                        state.issued_calls == issued + 1 && state.completed_calls <= issued &&
                            state.unfenced_calls <= width && state.pending_prefixes <= capacity,
                        "bounded matrix conserves entered calls independently of physical proof");
                    if (state.unfenced_calls == width || issued + 1 == calls) {
                        recorded.push_back(record(epoch));
                        if (recorded.size() == capacity || issued + 1 == calls) {
                            require(epoch.prove_fence(recorded.back()),
                                    "synthetic covering completion in bounded matrix");
                            recorded.clear();
                        }
                    }
                }
                require(!epoch.reserve(kScope, kStart) && epoch.snapshot().issued_calls == calls &&
                            epoch.snapshot().completed_calls == calls,
                        "call cap closes exact authorized extent");
                const auto terminal = epoch.prepare_terminal();
                require(terminal && terminal->issued_calls() == calls &&
                            epoch.publish_terminal_result(*terminal, true),
                        "all 544 bounded matrices acknowledge only completed extent");
            }
        }
    }
}
void concurrent_reservation_and_stop() {
    for (unsigned int repeat = 0; repeat < 100; ++repeat) {
        NativeExecutionEpoch epoch(kScope, authorization(), kStart);
        std::array<std::optional<NativeEpochIssue>, 2> tickets;
        std::barrier ready(3);
        std::thread first([&] {
            ready.arrive_and_wait();
            tickets[0] = epoch.reserve(kScope, kStart);
        });
        std::thread second([&] {
            ready.arrive_and_wait();
            tickets[1] = epoch.reserve(kScope, kStart);
        });
        ready.arrive_and_wait();
        first.join();
        second.join();
        require(tickets[0].has_value() != tickets[1].has_value(),
                "concurrent acceptance cannot borrow two originals");
        const auto& accepted = tickets[tickets[0] ? 0 : 1];
        if (!accepted) throw std::runtime_error("missing accepted ticket");
        std::latch stopped(1);
        std::thread requester([&] {
            (void)epoch.request_stop();
            stopped.count_down();
        });
        stopped.wait();
        require(!epoch.enter_original(*accepted, kStart) && epoch.discard_unissued(*accepted) &&
                    epoch.snapshot().issued_calls == 0,
                "stop before entry prevents original issue without affecting another callback");
        requester.join();
        const auto terminal = epoch.prepare_terminal();
        require(terminal && epoch.publish_terminal_result(*terminal, true),
                "concurrent unused extent settles without physical fiction");
    }
}

void concurrent_terminal_and_fence_results() {
    for (unsigned int repeat = 0; repeat < 100; ++repeat) {
        NativeExecutionEpoch epoch(kScope, authorization(), kStart);
        call(epoch);
        const auto fence = record(epoch);
        std::array<bool, 2> won{};
        std::barrier ready(3);
        std::thread proof([&] {
            ready.arrive_and_wait();
            won[0] = epoch.prove_fence(fence);
        });
        std::thread failure([&] {
            ready.arrive_and_wait();
            won[1] = epoch.fail_fence_completion(fence);
        });
        ready.arrive_and_wait();
        proof.join();
        failure.join();
        require(won[0] != won[1], "exactly one fence outcome is accepted");
        if (won[1]) {
            require(epoch.prove_scope_stopped(kScope),
                    "failed fence requires independent whole-scope proof");
        } else {
            require(epoch.request_stop(), "proved fence may stop without further work");
        }
        const auto pending = epoch.prepare_terminal();
        require(pending.has_value(), "settled epoch prepares terminal result");
        const auto terminal = pending.value_or(NativeEpochTerminal{});
        std::array<bool, 2> published{};
        std::barrier publish_ready(3);
        std::thread acknowledge([&] {
            publish_ready.arrive_and_wait();
            published[0] = epoch.publish_terminal_result(terminal, true);
        });
        std::thread lose_reply([&] {
            publish_ready.arrive_and_wait();
            published[1] = epoch.publish_terminal_result(terminal, false);
        });
        publish_ready.arrive_and_wait();
        acknowledge.join();
        lose_reply.join();
        require(published[0] != published[1] && epoch.snapshot().has_ownership == published[1],
                "one terminal result wins and lost acknowledgment retains ownership");
    }
}
}  // namespace

int main() {
    try {
        configuration();
        issue_identity_and_entry();
        deadlines();
        prefix_capacity_and_covering_proof();
        failure_and_whole_scope_proof();
        completion_and_control_failures();
        foreign_fences_and_terminal();
        exhaustive_bounded_prefixes();
        concurrent_reservation_and_stop();
        concurrent_terminal_and_fence_results();
        std::cout << "Native epoch CPU ownership checks passed; all physical proof is synthetic.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
