#include "glimmer/core/execution_priority.h"

#include <array>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
using namespace glimmer::core;
void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void priority_idle_window() {
    for (const auto mode :
         {ExecutionSchedulingMode::kPriority, ExecutionSchedulingMode::kWorkloadAware}) {
        ExecutionPriorityIdleWindow idle{{100, 100000}};
        std::array peers{ExecutionPriorityPeer{1, 0, false, false, true, false},
                         ExecutionPriorityPeer{2, 100, false, false, false, false}};
        peers[0].command = ExecutionPriorityCommand{.sequence = 7,
                                                    .acceptance_order = 2,
                                                    .physical_generation = 1,
                                                    .geometry_epoch = 1,
                                                    .command_epoch = 1,
                                                    .control_epoch = 1,
                                                    .is_resume = true};
        const auto plan_at = [&](std::uint64_t now) {
            return idle.apply(peers, mode, now, plan_execution_priority(peers, mode));
        };
        expect(plan_at(0).decisions[0].action == ExecutionPriorityAction::kExecute,
               "an idle configured high role without actual work creates no artificial window");
        peers[1].has_physical_work = true;
        peers[1].has_ready_work = true;
        expect(plan_at(1).decisions[1].action == ExecutionPriorityAction::kExecute,
               "high native pipeline remains open while prior high GPU work is outstanding");
        peers[1].has_ready_work = false;
        expect(plan_at(900000).decisions[0].action == ExecutionPriorityAction::kWait &&
                   !idle.has_idle_window(),
               "host elapsed time cannot start the interval before physical completion");
        peers[1].has_physical_work = false;
        auto plan = plan_at(900001);
        expect(plan.is_valid && plan.decisions[0].action == ExecutionPriorityAction::kWait &&
                   !plan.decisions[0].command && idle.remaining_ns(900001) == 100000,
               "healthy physical drain withholds exact lower continuation without a new grant");
        expect(plan_at(1000000).decisions[0].action == ExecutionPriorityAction::kWait &&
                   idle.remaining_ns(1000000) == 1,
               "repeated plans do not extend the deadline, and the last nanosecond still waits");
        plan = plan_at(1000001);
        expect(plan.decisions[0].action == ExecutionPriorityAction::kExecute &&
                   plan.decisions[0].command == peers[0].command && !idle.has_idle_window(),
               "expiration restores only the original exact retained command grant");
        idle.note_issue(100);  // A short command can retire between two snapshots.
        expect(plan_at(1000002).decisions[0].action == ExecutionPriorityAction::kWait,
               "successful short issue arms even when no active snapshot was sampled");
        peers[1].has_ready_work = true;
        expect(plan_at(1000003).decisions[1].action == ExecutionPriorityAction::kExecute &&
                   !idle.has_idle_window(),
               "a new high arrival immediately cancels the idle interval");
        peers[1].has_physical_work = true;
        (void)plan_at(1000004);
        peers[1].has_ready_work = peers[1].has_physical_work = false;
        expect(plan_at(2000000).decisions[0].action == ExecutionPriorityAction::kWait &&
                   idle.remaining_ns(2000000) == 100000,
               "the next complete high drain starts a new full interval");
        peers[1].has_unknown_state = true;
        expect(plan_at(2200000).decisions[0].action == ExecutionPriorityAction::kQuarantined &&
                   !idle.has_idle_window(),
               "elapsed idle time cannot turn unknown ownership into a low grant");
        peers[1].has_unknown_state = false;
        expect(plan_at(2200001).decisions[0].action == ExecutionPriorityAction::kWait,
               "only a later healthy snapshot restarts the interrupted interval");
        expect(!plan_at(2200000).is_valid, "a regressing owner clock cannot release low work");
        expect(plan_at(2300001).decisions[0].action == ExecutionPriorityAction::kExecute,
               "a valid later clock can recover without replay or stale authority");
        idle.note_issue(0);
        expect(plan_at(2300002).decisions[0].action == ExecutionPriorityAction::kExecute,
               "lower issue cannot renew a highest-priority hold");
        idle.note_issue(100);
        const auto last = std::numeric_limits<std::uint64_t>::max();
        expect(plan_at(last - 2).decisions[0].action == ExecutionPriorityAction::kWait &&
                   plan_at(last).decisions[0].action == ExecutionPriorityAction::kWait &&
                   idle.remaining_ns(last) == 99998,
               "elapsed subtraction does not wrap an absolute deadline near UINT64_MAX");
        peers[1].session = peers[0].session;
        expect(!plan_at(last).is_valid, "invalid peer identity is never repaired by idle policy");
    }
    std::array peers{ExecutionPriorityPeer{1, 0, true, false, false, false},
                     ExecutionPriorityPeer{2, 100, false, false, false, false}};
    for (const auto mode : {ExecutionSchedulingMode::kNaive, ExecutionSchedulingMode::kProfile,
                            ExecutionSchedulingMode::kWorkloadAgnostic}) {
        ExecutionPriorityIdleWindow idle{{100, 100000}};
        idle.note_issue(100);
        expect(
            idle.apply(peers, mode, 1, plan_execution_priority(peers, mode)).decisions[0].action ==
                ExecutionPriorityAction::kExecute,
            "nonpriority modes never acquire an idle hold");
    }
    ExecutionPriorityIdleWindow disabled{{100, 0}};
    disabled.note_issue(100);
    expect(!disabled.enabled() && disabled.apply(peers, ExecutionSchedulingMode::kPriority, 0,
                                                 plan_execution_priority(peers))
                                          .decisions[0]
                                          .action == ExecutionPriorityAction::kExecute,
           "explicit zero retains the prior policy");
    ExecutionPriorityIdleWindow idle{{100, 100000}};
    idle.note_issue(100);
    peers[1].priority = 0;
    peers[1].has_physical_work = true;
    peers[1].reservation_generation = 9;
    peers[0].progress = ExecutionPriorityProgress{2, 9, 17};
    const auto progress =
        idle.apply(peers, ExecutionSchedulingMode::kPriority, 0, plan_execution_priority(peers));
    expect(progress.decisions[0].action == ExecutionPriorityAction::kExecuteProgress &&
               progress.decisions[0].progress == peers[0].progress &&
               !progress.decisions[0].command,
           "the hold preserves exact dependency progress, never broad ordinary issue");
}
void exact_priority_transitions() {
    std::array peers{ExecutionPriorityPeer{1, 0, false, true, false, false},
                     ExecutionPriorityPeer{2, 100, true, false, false, false}};
    auto plan = plan_execution_priority(peers);
    expect(plan.is_valid && plan.selected_priority == 100 &&
               plan.decisions[0].action == ExecutionPriorityAction::kRequestRetreat &&
               plan.decisions[1].action == ExecutionPriorityAction::kWait,
           "high demand must request retreat and wait for actual lower physical stop");
    expect(plan_execution_priority(peers).decisions[1].action == ExecutionPriorityAction::kWait,
           "a request/timeout cannot replace updated physical observations");
    peers[0].has_physical_work = false;
    peers[0].has_paused_work = true;
    plan = plan_execution_priority(peers);
    expect(plan.decisions[0].action == ExecutionPriorityAction::kWait &&
               plan.decisions[1].action == ExecutionPriorityAction::kExecute,
           "real stopped lower work remains incomplete while high can execute");
    peers[1].has_physical_work = true;
    expect(plan_execution_priority(peers).decisions[1].action == ExecutionPriorityAction::kExecute,
           "ready high work pipelines before earlier high completion");
    peers[1].has_ready_work = false;
    plan = plan_execution_priority(peers);
    expect(
        plan.selected_priority == 100 && plan.decisions[0].action == ExecutionPriorityAction::kWait,
        "SDK reply or disconnect without physical completion cannot resume low");
    peers[1].has_physical_work = false;
    plan = plan_execution_priority(peers);
    expect(plan.selected_priority == 0 &&
               plan.decisions[0].action == ExecutionPriorityAction::kExecute,
           "only drained high work permits retained low resumption");
}

void exact_commands_are_not_connection_wide_grants() {
    const ExecutionPriorityCommand initial{.sequence = 17,
                                           .acceptance_order = 23,
                                           .geometry_epoch = 1,
                                           .command_epoch = 1,
                                           .control_epoch = 3};
    std::array peers{ExecutionPriorityPeer{1, 0, true, false, false, false},
                     ExecutionPriorityPeer{2, 100, true, false, false, false}};
    peers[0].command = initial;
    auto high = initial;
    high.sequence = 31;
    high.member_generation = 7;
    high.configuration_epoch = 11;
    high.physical_configuration_epoch = 19;
    peers[1].command = high;
    for (const auto mode :
         {ExecutionSchedulingMode::kPriority, ExecutionSchedulingMode::kNaive,
          ExecutionSchedulingMode::kProfile, ExecutionSchedulingMode::kWorkloadAgnostic,
          ExecutionSchedulingMode::kWorkloadAware, ExecutionSchedulingMode::kPrioritySpaceShare}) {
        const auto plan = plan_execution_priority(peers, mode);
        expect(plan.is_valid, "complete exact command descriptions preserve every old mode");
        for (std::size_t index = 0; index < peers.size(); ++index)
            expect(plan.decisions[index].command ==
                       (plan.decisions[index].action == ExecutionPriorityAction::kExecute
                            ? peers[index].command
                            : std::nullopt),
                   "only Execute echoes the exact observed command, never Wait/retreat");
    }
    peers[0].has_physical_work = true;
    auto plan = plan_execution_priority(peers);
    expect(plan.decisions[0].action == ExecutionPriorityAction::kRequestRetreat &&
               !plan.decisions[0].command && !plan.decisions[1].command,
           "high arrival waits for physical stop without borrowing an old ordinary command");
    peers[0].has_unknown_state = true;
    plan = plan_execution_priority(peers);
    expect(!plan.decisions[0].command && !plan.decisions[1].command,
           "quarantine never echoes ordinary command permission");
    peers[0].has_unknown_state = peers[0].has_physical_work = false;
    peers[1].has_ready_work = false;
    peers[1].command.reset();
    peers[0].has_paused_work = true;
    peers[0].command->is_resume = true;
    peers[0].command->physical_generation = 5;
    const auto old = peers[0].command;
    expect(plan_execution_priority(peers).decisions[0].command == old,
           "a known stopped continuation retains its exact previous physical generation");
    ++peers[0].command->geometry_epoch;
    ++peers[0].command->control_epoch;
    expect(plan_execution_priority(peers).decisions[0].command != old,
           "a fresh plan cannot return a command from a previous geometry/control revision");

    peers[0].command = initial;
    for (auto field :
         {&ExecutionPriorityCommand::sequence, &ExecutionPriorityCommand::acceptance_order,
          &ExecutionPriorityCommand::geometry_epoch, &ExecutionPriorityCommand::command_epoch,
          &ExecutionPriorityCommand::control_epoch}) {
        auto invalid = initial;
        invalid.*field = 0;
        peers[0].command = invalid;
        expect(!plan_execution_priority(peers).is_valid, "missing exact identity rejects");
    }
    for (auto invalid : {initial, high}) {
        invalid.physical_generation = 1;
        peers[0].command = invalid;
        expect(!plan_execution_priority(peers).is_valid,
               "an initial command cannot borrow a continuation generation");
        invalid.is_resume = true;
        invalid.physical_generation = 0;
        peers[0].command = invalid;
        expect(!plan_execution_priority(peers).is_valid,
               "a continuation needs an actual prior physical generation");
    }
    for (auto field : {&ExecutionPriorityCommand::member_generation,
                       &ExecutionPriorityCommand::configuration_epoch,
                       &ExecutionPriorityCommand::physical_configuration_epoch}) {
        auto invalid = high;
        invalid.*field = 0;
        peers[0].command = invalid;
        expect(!plan_execution_priority(peers).is_valid,
               "partial shared-domain configuration identity rejects");
    }
    peers[0].command = initial;
    peers[0].has_ready_work = peers[0].has_paused_work = false;
    expect(!plan_execution_priority(peers).is_valid, "idle state cannot carry ordinary work");

    peers[0].has_physical_work = true;
    peers[0].reservation_generation = 71;
    peers[0].command.reset();
    peers[1].priority = 0;
    peers[1].has_ready_work = true;
    peers[1].command = high;
    peers[1].progress = ExecutionPriorityProgress{1, 71, 12};
    peers[0].priority = 100;
    plan = plan_execution_priority(peers);
    expect(plan.is_valid && plan.decisions[1].action == ExecutionPriorityAction::kExecuteProgress &&
               !plan.decisions[1].command && plan.decisions[1].progress == peers[1].progress,
           "an exact native dependency-cut grant is never general ordinary command authority");
}
void bounded_identity_and_unknown_state() {
    expect(plan_execution_priority({}).is_valid, "empty domain has no work");
    std::array peers{ExecutionPriorityPeer{1, 5, true, false, false, false},
                     ExecutionPriorityPeer{2, 5, true, true, false, false}};
    auto plan = plan_execution_priority(peers);
    expect(plan.decisions[0].action == ExecutionPriorityAction::kExecute &&
               plan.decisions[1].action == ExecutionPriorityAction::kExecute,
           "equal highest peers may pipeline without a one-task high cap");
    peers[0].has_unknown_state = true;
    plan = plan_execution_priority(peers);
    expect(plan.decisions[0].action == ExecutionPriorityAction::kQuarantined &&
               plan.decisions[1].action == ExecutionPriorityAction::kQuarantined,
           "unknown ownership never grants issue based on priority alone");
    peers[1].session = 1;
    expect(!plan_execution_priority(peers).is_valid, "duplicate numeric domains reject");
    peers[1].session = 0;
    expect(!plan_execution_priority(peers).is_valid, "zero authority rejects");
    std::vector<ExecutionPriorityPeer> oversized(kMaxExecutionPriorityPeers + 1);
    expect(!plan_execution_priority(oversized).is_valid,
           "bounded snapshots cannot overflow the plan");
}
void naive_preserves_authority_without_priority_retreat() {
    constexpr auto mode = ExecutionSchedulingMode::kNaive;
    std::array peers{ExecutionPriorityPeer{1, 0, true, true, false, false},
                     ExecutionPriorityPeer{2, 100, true, false, false, false},
                     ExecutionPriorityPeer{3, 7, false, false, true, false},
                     ExecutionPriorityPeer{4, 1, false, true, false, false},
                     ExecutionPriorityPeer{5, 0, false, false, false, false}};
    auto plan = plan_execution_priority(peers, mode);
    expect(
        plan.is_valid && plan.size == peers.size() && !plan.selected_priority &&
            plan.decisions[0].action == ExecutionPriorityAction::kExecute &&
            plan.decisions[1].action == ExecutionPriorityAction::kExecute &&
            plan.decisions[2].action == ExecutionPriorityAction::kExecute &&
            plan.decisions[3].action == ExecutionPriorityAction::kWait &&
            plan.decisions[4].action == ExecutionPriorityAction::kIdle,
        "naive issues ready/paused peers without a false priority retreat or physical-stop claim");
    peers[4].has_unknown_state = true;
    plan = plan_execution_priority(peers, mode);
    for (std::size_t index = 0; index < peers.size(); ++index)
        expect(plan.decisions[index].action == ExecutionPriorityAction::kQuarantined,
               "naive admission never bypasses unknown resource ownership, even on an idle peer");
    peers[4].has_unknown_state = false;
    peers[4].session = 1;
    expect(!plan_execution_priority(peers, mode).is_valid, "naive duplicate identities reject");
    peers[4].session = 0;
    expect(!plan_execution_priority(peers, mode).is_valid, "naive zero identity rejects");
    std::vector<ExecutionPriorityPeer> oversized(kMaxExecutionPriorityPeers + 1);
    expect(!plan_execution_priority(oversized, mode).is_valid &&
               !plan_execution_priority({}, static_cast<ExecutionSchedulingMode>(255)).is_valid,
           "unknown scheduler modes and excessive peers reject rather than falling back");
    expect(plan_execution_priority({}, mode).is_valid, "empty naive domain has no artificial work");
}
void profile_isolates_physical_work_and_retains_continuation() {
    constexpr auto mode = ExecutionSchedulingMode::kProfile;
    std::array peers{ExecutionPriorityPeer{1, 0, true, false, false, false},
                     ExecutionPriorityPeer{2, 100, true, false, false, false}};
    auto plan = plan_execution_priority(peers, mode);
    expect(plan.is_valid && !plan.selected_priority &&
               plan.decisions[0].action == ExecutionPriorityAction::kExecute &&
               plan.decisions[1].action == ExecutionPriorityAction::kWait,
           "profile chooses only one ready peer, independently of priority labels");
    peers[0].has_physical_work = true;
    plan = plan_execution_priority(peers, mode);
    expect(plan.decisions[0].action == ExecutionPriorityAction::kWait &&
               plan.decisions[1].action == ExecutionPriorityAction::kWait,
           "even the current peer cannot start new work during a physical profile interval");
    peers[0].has_physical_work = false;
    peers[0].has_paused_work = true;
    std::swap(peers[0], peers[1]);
    plan = plan_execution_priority(peers, mode);
    expect(plan.decisions[0].action == ExecutionPriorityAction::kWait &&
               plan.decisions[1].action == ExecutionPriorityAction::kExecute,
           "retain sliced logical interval before selecting another participant");
    peers[1].has_unknown_state = true;
    plan = plan_execution_priority(peers, mode);
    expect(plan.decisions[0].action == ExecutionPriorityAction::kQuarantined &&
               plan.decisions[1].action == ExecutionPriorityAction::kQuarantined,
           "profile never bypasses global unknown ownership");
}
void workload_agnostic_preserves_physical_authority() {
    static_assert(static_cast<unsigned int>(ExecutionSchedulingMode::kPriority) == 0);
    static_assert(static_cast<unsigned int>(ExecutionSchedulingMode::kNaive) == 1);
    static_assert(static_cast<unsigned int>(ExecutionSchedulingMode::kProfile) == 2);
    static_assert(static_cast<unsigned int>(ExecutionSchedulingMode::kWorkloadAgnostic) == 3);
    constexpr auto mode = ExecutionSchedulingMode::kWorkloadAgnostic;
    std::array peers{ExecutionPriorityPeer{1, 0, true, true, false, false},
                     ExecutionPriorityPeer{2, 100, true, false, false, false},
                     ExecutionPriorityPeer{3, 7, false, false, true, false},
                     ExecutionPriorityPeer{4, 1, false, true, false, false},
                     ExecutionPriorityPeer{5, 0, false, false, false, false}};
    auto plan = plan_execution_priority(peers, mode);
    expect(plan.is_valid && plan.size == peers.size() && !plan.selected_priority &&
               plan.decisions[0].action == ExecutionPriorityAction::kExecute &&
               plan.decisions[1].action == ExecutionPriorityAction::kExecute &&
               plan.decisions[2].action == ExecutionPriorityAction::kExecute &&
               plan.decisions[3].action == ExecutionPriorityAction::kWait &&
               plan.decisions[4].action == ExecutionPriorityAction::kIdle,
           "sharing permits concurrent ready/paused peers without inventing physical stop");
    expect(plan_execution_priority(peers).decisions[0].action ==
               ExecutionPriorityAction::kRequestRetreat,
           "the existing default remains strict priority, not sharing");
    for (std::size_t unknown = 0; unknown < peers.size(); ++unknown) {
        peers[unknown].has_unknown_state = true;
        plan = plan_execution_priority(peers, mode);
        for (std::size_t index = 0; index < peers.size(); ++index)
            expect(plan.decisions[index].session == peers[index].session &&
                       plan.decisions[index].action == ExecutionPriorityAction::kQuarantined,
                   "any sharing peer's unknown ownership prevents all issue and continuation");
        peers[unknown].has_unknown_state = false;
    }
    peers[4].session = 1;
    expect(!plan_execution_priority(peers, mode).is_valid, "sharing rejects duplicate domains");
    peers[4].session = 0;
    expect(!plan_execution_priority(peers, mode).is_valid, "sharing rejects zero authority");
    std::vector<ExecutionPriorityPeer> oversized(kMaxExecutionPriorityPeers + 1);
    expect(!plan_execution_priority(oversized, mode).is_valid &&
               plan_execution_priority({}, mode).is_valid,
           "sharing retains bounded snapshots and an empty valid domain");
}

void priority_space_share_keeps_native_work_and_exact_authority() {
    static_assert(static_cast<unsigned int>(ExecutionSchedulingMode::kPrioritySpaceShare) == 5);
    static_assert(!is_priority_execution_mode(ExecutionSchedulingMode::kPrioritySpaceShare));
    constexpr auto mode = ExecutionSchedulingMode::kPrioritySpaceShare;
    std::array peers{ExecutionPriorityPeer{1, 0, true, true, false, false},
                     ExecutionPriorityPeer{2, 100, true, true, false, false},
                     ExecutionPriorityPeer{3, 17, false, true, false, false},
                     ExecutionPriorityPeer{4, 200, false, false, false, false}};
    peers[0].command = ExecutionPriorityCommand{.sequence = 19,
                                                .acceptance_order = 23,
                                                .geometry_epoch = 1,
                                                .command_epoch = 1,
                                                .control_epoch = 7};
    auto plan = plan_execution_priority(peers, mode);
    expect(plan.is_valid && plan.size == peers.size() && plan.selected_priority == 200 &&
               plan.decisions[0].action == ExecutionPriorityAction::kExecute &&
               plan.decisions[0].command == peers[0].command &&
               plan.decisions[1].action == ExecutionPriorityAction::kExecute &&
               plan.decisions[2].action == ExecutionPriorityAction::kWait &&
               plan.decisions[3].action == ExecutionPriorityAction::kIdle,
           "space sharing keeps configured high identity, native pipelining and no retreat");
    expect(plan_execution_priority(peers).decisions[0].action ==
               ExecutionPriorityAction::kRequestRetreat,
           "default preemptive priority does not silently become original space sharing");
    ExecutionPriorityIdleWindow idle{{200, 100000}};
    idle.note_issue(200);
    const auto unchanged = idle.apply(peers, mode, 1, plan);
    expect(unchanged.is_valid && unchanged.decisions[0].command == peers[0].command &&
               unchanged.decisions[0].action == ExecutionPriorityAction::kExecute &&
               !idle.has_idle_window(),
           "preemptive idle holding cannot suppress an original space-sharing grant");
    for (auto& peer : peers) {
        peer.has_unknown_state = true;
        plan = plan_execution_priority(peers, mode);
        for (const auto& decision : std::span(plan.decisions).first(plan.size))
            expect(decision.action == ExecutionPriorityAction::kQuarantined && !decision.command,
                   "one uncertain member quarantines every original issue");
        peer.has_unknown_state = false;
    }
    auto& command = peers[0].command;
    if (!command) throw std::runtime_error("missing command fixture");
    command->control_epoch = 0;
    expect(!plan_execution_priority(peers, mode).is_valid,
           "space sharing cannot bypass exact ordinary command validation");
    peers[0].command.reset();
    peers[3].session = peers[0].session;
    expect(!plan_execution_priority(peers, mode).is_valid,
           "duplicate member identity cannot gain space-sharing permission");
    peers[3].session = 0;
    expect(!plan_execution_priority(peers, mode).is_valid,
           "zero member identity cannot gain space-sharing permission");
    expect(plan_execution_priority({}, mode).is_valid,
           "an empty space-sharing plan grants no work");
    const std::vector<ExecutionPriorityPeer> oversized(kMaxExecutionPriorityPeers + 1);
    expect(!plan_execution_priority(oversized, mode).is_valid,
           "space sharing retains the bounded complete snapshot");
}

void retained_dependency_progress_is_not_general_issue() {
    std::array peers{ExecutionPriorityPeer{1, 0, false, true, false, false},
                     ExecutionPriorityPeer{2, 0, true, false, false, false},
                     ExecutionPriorityPeer{3, 100, true, false, false, false},
                     ExecutionPriorityPeer{4, 0, true, false, false, false}};
    peers[0].reservation_generation = 71;
    const ExecutionPriorityProgress dependency{1, 71, 19};
    peers[1].progress = dependency;
    auto plan = plan_execution_priority(peers);
    expect(plan.is_valid && plan.selected_priority == 100 &&
               plan.decisions[0].action == ExecutionPriorityAction::kRequestRetreat &&
               plan.decisions[1].action == ExecutionPriorityAction::kExecuteProgress &&
               plan.decisions[1].progress == dependency &&
               plan.decisions[2].action == ExecutionPriorityAction::kWait &&
               plan.decisions[3].action == ExecutionPriorityAction::kWait &&
               !plan.decisions[0].progress && !plan.decisions[2].progress &&
               !plan.decisions[3].progress,
           "only the exact needed lower prefix progresses; high still waits and unrelated low "
           "work receives no bypass");
    peers[1].has_ready_work = false;
    peers[1].has_physical_work = true;
    plan = plan_execution_priority(peers);
    expect(plan.decisions[1].action == ExecutionPriorityAction::kExecuteProgress &&
               plan.decisions[1].progress == dependency,
           "needed ongoing source work is not falsely retreated or declared stopped");
    peers[1].has_physical_work = false;
    peers[1].has_paused_work = true;
    plan = plan_execution_priority(peers);
    expect(plan.decisions[1].action == ExecutionPriorityAction::kExecuteProgress,
           "a paused required source prefix may resume only under its existing native grant");

    peers[0].has_ready_work = true;
    peers[0].progress = ExecutionPriorityProgress{1, 71, 23};
    plan = plan_execution_priority(peers);
    expect(plan.decisions[0].action == ExecutionPriorityAction::kExecuteProgress &&
               plan.decisions[0].progress == peers[0].progress,
           "the reserved batch can perform its own exact End/query progress without a new grant");
    peers[0].progress.reset();
    peers[0].has_ready_work = false;
    peers[0].has_physical_work = false;
    peers[0].reservation_generation = 0;
    peers[1].progress.reset();
    plan = plan_execution_priority(peers);
    expect(plan.decisions[2].action == ExecutionPriorityAction::kExecute &&
               plan.decisions[1].action == ExecutionPriorityAction::kWait,
           "only real reservation retirement restores normal priority, not a stale progress cut");
}

void progress_authority_validation_and_unknown_state() {
    std::array peers{ExecutionPriorityPeer{1, 0, false, true, false, false},
                     ExecutionPriorityPeer{2, 0, true, false, false, false},
                     ExecutionPriorityPeer{3, 100, true, false, false, false}};
    peers[0].reservation_generation = 17;
    const ExecutionPriorityProgress valid{1, 17, 9};
    constexpr std::array invalid{
        ExecutionPriorityProgress{0, 17, 9}, ExecutionPriorityProgress{4, 17, 9},
        ExecutionPriorityProgress{1, 0, 9},  ExecutionPriorityProgress{1, 18, 9},
        ExecutionPriorityProgress{1, 17, 0}, ExecutionPriorityProgress{3, 17, 9}};
    for (const auto mode :
         {ExecutionSchedulingMode::kPriority, ExecutionSchedulingMode::kNaive,
          ExecutionSchedulingMode::kProfile, ExecutionSchedulingMode::kWorkloadAgnostic,
          ExecutionSchedulingMode::kWorkloadAware, ExecutionSchedulingMode::kPrioritySpaceShare}) {
        for (const auto& bad : invalid) {
            peers[1].progress = bad;
            expect(!plan_execution_priority(peers, mode).is_valid,
                   "every mode rejects missing/stale/zero/idle reservation authority");
        }
        peers[1].progress = valid;
        auto plan = plan_execution_priority(peers, mode);
        expect(plan.is_valid, "a valid dependency has bounded authority in every mode");
        if (mode == ExecutionSchedulingMode::kNaive ||
            mode == ExecutionSchedulingMode::kWorkloadAgnostic ||
            mode == ExecutionSchedulingMode::kPrioritySpaceShare) {
            expect(plan.decisions[1].action == ExecutionPriorityAction::kExecute &&
                       !plan.decisions[1].progress,
                   "naturally allowed ordinary issue does not invent a restricted reservation");
        } else {
            expect(plan.decisions[1].action == ExecutionPriorityAction::kExecuteProgress &&
                       plan.decisions[1].progress == valid,
                   "priority/profile retain the exact dependency instead of granting general work");
        }
        for (auto& peer : peers) {
            peer.has_unknown_state = true;
            plan = plan_execution_priority(peers, mode);
            expect(plan.is_valid, "well-formed unknown ownership is representable");
            for (std::size_t index = 0; index != peers.size(); ++index)
                expect(plan.decisions[index].action == ExecutionPriorityAction::kQuarantined &&
                           !plan.decisions[index].progress,
                       "unknown ownership dominates native progress and every other grant");
            peer.has_unknown_state = false;
        }
        peers[1].has_ready_work = false;
        expect(!plan_execution_priority(peers, mode).is_valid,
               "an idle requester cannot retain an actionable dependency grant");
        peers[1].has_ready_work = true;
        peers[0].has_physical_work = false;
        expect(!plan_execution_priority(peers, mode).is_valid,
               "a completed reservation cannot authorize later dependency work");
        peers[0].has_unknown_state = true;
        plan = plan_execution_priority(peers, mode);
        expect(plan.is_valid && plan.decisions[1].action == ExecutionPriorityAction::kQuarantined &&
                   !plan.decisions[1].progress,
               "lost physical observation retains quarantine, never a new progress permission");
        peers[0].has_unknown_state = false;
        peers[0].has_physical_work = true;
    }
}
void arrival_checkpoint_is_stop_only() {
    const ExecutionPriorityCommand initial{.sequence = 17,
                                           .acceptance_order = 1,
                                           .geometry_epoch = 1,
                                           .command_epoch = 1,
                                           .control_epoch = 1};
    std::array peers{ExecutionPriorityPeer{1, 0, false, true},
                     ExecutionPriorityPeer{2, 0, false, true}, ExecutionPriorityPeer{3, 0, true},
                     ExecutionPriorityPeer{4, 0}};
    peers[0].physical = ExecutionPriorityPhysical{7, 3};
    peers[1].physical = ExecutionPriorityPhysical{9, 4};
    peers[2].command = initial;
    const ExecutionRebalanceProposal checkpoint{
        ExecutionRebalanceTrigger{3, initial},
        {ExecutionRebalanceLeg{1, {7, 3}}, ExecutionRebalanceLeg{2, {9, 4}}},
        2};
    auto plan = plan_execution_rebalance(peers, checkpoint);
    expect(plan.is_valid && plan.decisions[0].action == ExecutionPriorityAction::kRequestRetreat &&
               plan.decisions[1].action == ExecutionPriorityAction::kRequestRetreat &&
               plan.decisions[2].action == ExecutionPriorityAction::kWait,
           "same-priority arrival requests both exact physical retreats, not new issue");
    for (const auto& decision : std::span{plan.decisions}.first(plan.size))
        expect(!decision.command && decision.action != ExecutionPriorityAction::kExecute,
               "stop plan cannot echo a general command capability");
    for (unsigned int scenario = 0; scenario < 10; ++scenario) {
        auto invalid = checkpoint;
        if (scenario == 0) invalid.size = 0;
        if (scenario == 1) invalid.size = 3;
        if (scenario == 2) ++invalid.legs[0].physical.generation;
        if (scenario == 3) ++invalid.legs[1].physical.sequence;
        if (scenario == 4) invalid.legs[1] = invalid.legs[0];
        if (scenario == 5) ++invalid.trigger->command.geometry_epoch;
        if (scenario == 6) invalid.trigger->session = 1;
        if (scenario == 7) invalid.trigger->command.is_resume = true;
        if (scenario == 8) invalid.trigger->command.physical_generation = 1;
        if (scenario == 9) invalid.trigger->command.command_epoch = 2;
        expect(!plan_execution_rebalance(peers, invalid).is_valid,
               "stale/duplicate/zero/missing checkpoint inputs never grant retreat or issue");
    }
    auto retained = checkpoint;
    retained.trigger.reset();
    expect(plan_execution_rebalance(peers, retained).decisions[0].action ==
               ExecutionPriorityAction::kRequestRetreat,
           "an owner-retained checkpoint may drain its exact epoch after trigger cancellation");
    peers[2].has_ready_work = false;
    peers[2].command.reset();
    peers[0].has_physical_work = false;
    peers[0].has_paused_work = true;
    peers[0].physical.reset();
    retained.legs[0] = retained.legs[1];
    retained.size = 1;
    plan = plan_execution_rebalance(peers, retained);
    expect(plan.is_valid && plan.decisions[0].action == ExecutionPriorityAction::kWait &&
               plan.decisions[1].action == ExecutionPriorityAction::kRequestRetreat,
           "cancelled trigger and one real stop still hold paused work until its partner stops");
    peers[0].has_physical_work = true;
    peers[0].has_paused_work = false;
    peers[0].physical = ExecutionPriorityPhysical{7, 3};
    peers[2].has_ready_work = true;
    peers[2].command = initial;
    peers[3].priority = 100;
    peers[3].has_ready_work = true;
    plan = plan_execution_rebalance(peers, checkpoint);
    expect(plan.is_valid && plan.selected_priority == 100 &&
               plan.decisions[3].action == ExecutionPriorityAction::kWait,
           "higher demand retains its ordinary actual-stop requirement");
    peers[3].priority = 0;
    for (auto& peer : peers) {
        peer.has_unknown_state = true;
        plan = plan_execution_rebalance(peers, checkpoint);
        expect(plan.is_valid, "well-formed unknown state remains representable");
        for (const auto& decision : std::span{plan.decisions}.first(plan.size))
            expect(decision.action == ExecutionPriorityAction::kQuarantined && !decision.command,
                   "unknown ownership dominates arrival checkpoints");
        peer.has_unknown_state = false;
    }
    peers[3].has_physical_work = true;
    peers[3].reservation_generation = 19;
    peers[3].progress = ExecutionPriorityProgress{4, 19, 23};
    plan = plan_execution_rebalance(peers, checkpoint);
    expect(plan.is_valid && plan.decisions[3].action == ExecutionPriorityAction::kExecuteProgress &&
               plan.decisions[3].progress == peers[3].progress && !plan.decisions[3].command,
           "closing ordinary admission preserves exact existing native dependency progress");
    auto& physical = peers[0].physical;
    if (!physical) throw std::runtime_error("missing physical fixture");
    physical->generation = 0;
    expect(!plan_execution_priority(peers).is_valid,
           "ordinary physical observations need positive generations");
}

void workload_aware_preserves_priority_before_pair_selection() {
    static_assert(static_cast<unsigned int>(ExecutionSchedulingMode::kWorkloadAware) == 4);
    static_assert(is_priority_execution_mode(ExecutionSchedulingMode::kWorkloadAware));
    static_assert(!is_priority_execution_mode(ExecutionSchedulingMode::kNaive));
    std::array peers{ExecutionPriorityPeer{1, 0, false, true}, ExecutionPriorityPeer{2, 100, true},
                     ExecutionPriorityPeer{3, 0, true}};
    auto plan = plan_execution_priority(peers, ExecutionSchedulingMode::kWorkloadAware);
    expect(plan.is_valid && plan.selected_priority == 100 &&
               plan.decisions[0].action == ExecutionPriorityAction::kRequestRetreat &&
               plan.decisions[1].action == ExecutionPriorityAction::kWait &&
               plan.decisions[2].action == ExecutionPriorityAction::kWait,
           "workload-aware waits for actual lower stop, never naive/all-ready admission");
    peers[0].has_physical_work = false;
    peers[0].has_paused_work = true;
    plan = plan_execution_priority(peers, ExecutionSchedulingMode::kWorkloadAware);
    expect(plan.decisions[1].action == ExecutionPriorityAction::kExecute &&
               plan.decisions[0].action == ExecutionPriorityAction::kWait,
           "only actually drained lower work permits high before retained continuation");
}

void exact_pair_grants_never_escape_the_selected_commands() {
    const ExecutionPriorityCommand first{.sequence = 17,
                                         .acceptance_order = 23,
                                         .geometry_epoch = 2,
                                         .command_epoch = 1,
                                         .control_epoch = 3,
                                         .member_generation = 5,
                                         .configuration_epoch = 1,
                                         .physical_configuration_epoch = 9};
    auto second = first;
    second.sequence = 19;
    second.member_generation = 7;
    std::array peers{ExecutionPriorityPeer{1, 0, true, false, false, false},
                     ExecutionPriorityPeer{2, 0, true, false, false, false},
                     ExecutionPriorityPeer{3, 0, true, false, false, false}};
    peers[0].command = first;
    peers[1].command = second;
    peers[0].pair_context = ExecutionPairContext{99, 11, 80, true};
    peers[1].pair_context = ExecutionPairContext{99, 13, 160, true};
    ExecutionPairProposal proposal{
        {ExecutionPairLeg{1, first, 11, 80}, ExecutionPairLeg{2, second, 13, 160}}, true, 0};
    for (const auto mode :
         {ExecutionSchedulingMode::kPriority, ExecutionSchedulingMode::kNaive,
          ExecutionSchedulingMode::kWorkloadAgnostic, ExecutionSchedulingMode::kWorkloadAware}) {
        const auto plan = plan_execution_pair(peers, proposal, mode);
        expect(plan.is_valid && plan.is_shared && plan.grant_count == 2 &&
                   plan.grants == proposal.legs && plan.priority.decisions[0].command == first &&
                   plan.priority.decisions[1].command == second &&
                   plan.priority.decisions[2].action == ExecutionPriorityAction::kWait &&
                   !plan.priority.decisions[2].command,
               "a measured proposal grants only its exact two commands, never an unrelated ready "
               "peer");
    }
    peers[0].pair_context->is_configuration_materialized = false;
    auto plan = plan_execution_pair(peers, proposal);
    expect(plan.is_valid && !plan.grant_count && !plan.priority.decisions[0].command &&
               !plan.priority.decisions[1].command && !plan.priority.decisions[2].command,
           "an unavailable materialization yields no pair or leaked broad Execute grant");
    peers[0].pair_context->is_configuration_materialized = true;
    for (std::size_t index = 0; index < peers.size(); ++index) {
        peers[index].has_physical_work = true;
        plan = plan_execution_pair(peers, proposal);
        expect(plan.is_valid && !plan.grant_count,
               "any existing actual physical peer prevents admission of another pair");
        peers[index].has_physical_work = false;
        peers[index].has_unknown_state = true;
        plan = plan_execution_pair(peers, proposal);
        expect(plan.is_valid && !plan.grant_count &&
                   plan.priority.decisions[0].action == ExecutionPriorityAction::kQuarantined &&
                   !plan.priority.decisions[0].command,
               "unknown state anywhere retains quarantine rather than paired computation");
        peers[index].has_unknown_state = false;
    }
    peers[2].priority = 100;
    expect(!plan_execution_pair(peers, proposal).grant_count,
           "new higher demand cannot inherit an earlier low pair proposal");
    peers[0].has_physical_work = true;
    plan = plan_execution_pair(peers, proposal);
    expect(plan.priority.decisions[0].action == ExecutionPriorityAction::kRequestRetreat &&
               !plan.grant_count,
           "ordinary physical priority retreat survives the paired adapter");
    peers[0].has_physical_work = false;
    peers[2].has_physical_work = true;
    peers[2].reservation_generation = 71;
    peers[0].progress = ExecutionPriorityProgress{3, 71, 12};
    plan = plan_execution_pair(peers, proposal);
    expect(
        plan.is_valid && !plan.grant_count &&
            plan.priority.decisions[0].action == ExecutionPriorityAction::kExecuteProgress &&
            plan.priority.decisions[0].progress == peers[0].progress &&
            !plan.priority.decisions[0].command,
        "retained native dependency progress stays exact and separate from paired ordinary issue");
    peers[0].progress.reset();
    peers[2].has_physical_work = false;
    peers[2].priority = 0;

    auto stale = proposal;
    ++stale.legs[0].command.control_epoch;
    expect(!plan_execution_pair(peers, stale).is_valid, "changed exact command rejects old pair");
    stale = proposal;
    ++stale.legs[0].catalog_generation;
    expect(!plan_execution_pair(peers, stale).is_valid, "stale native resource catalog rejects");
    stale = proposal;
    stale.legs[0].workers = 160;
    expect(!plan_execution_pair(peers, stale).is_valid,
           "workers must match actual prepared geometry");
    stale = proposal;
    stale.legs[1].session = 1;
    expect(!plan_execution_pair(peers, stale).is_valid, "one owner cannot masquerade as two legs");
    ++peers[1].pair_context->execution_domain;
    expect(!plan_execution_pair(peers, proposal).is_valid,
           "distinct contexts are not spatial sharing");
    --peers[1].pair_context->execution_domain;
    peers[1].command->member_generation = first.member_generation;
    stale = proposal;
    stale.legs[1].command = *peers[1].command;
    expect(!plan_execution_pair(peers, stale).is_valid,
           "shared legs need distinct logical members");
    peers[1].command = second;
    expect(!plan_execution_pair(peers, proposal, ExecutionSchedulingMode::kProfile).is_valid,
           "single-command profiling is not isolated paired profiling");
    expect(!plan_execution_pair(peers, proposal, ExecutionSchedulingMode::kPrioritySpaceShare)
                .is_valid,
           "original priority space sharing cannot gain measured persistent-pair authority");

    proposal.is_shared = false;
    proposal.first_serial = 1;
    proposal.legs[1].workers = peers[1].pair_context->workers = 0;
    plan = plan_execution_pair(peers, proposal);
    expect(plan.is_valid && !plan.is_shared && plan.grant_count == 1 &&
               plan.grants[0] == proposal.legs[1] &&
               plan.priority.decisions[0].action == ExecutionPriorityAction::kWait &&
               plan.priority.decisions[1].command == second &&
               plan.priority.decisions[2].action == ExecutionPriorityAction::kWait,
           "unissued serial original fallback grants exactly the chosen command only");
    peers[1].has_ready_work = false;
    peers[1].has_paused_work = true;
    peers[1].command->is_resume = true;
    peers[1].command->physical_generation = 1;
    proposal.legs[1].command = *peers[1].command;
    expect(!plan_execution_pair(peers, proposal).is_valid,
           "a partial command can never acquire original fallback or replay authority");
}
}  // namespace
int main() {
    try {
        priority_idle_window();
        exact_priority_transitions();
        workload_aware_preserves_priority_before_pair_selection();
        exact_commands_are_not_connection_wide_grants();
        exact_pair_grants_never_escape_the_selected_commands();
        arrival_checkpoint_is_stop_only();
        bounded_identity_and_unknown_state();
        naive_preserves_authority_without_priority_retreat();
        profile_isolates_physical_work_and_retains_continuation();
        workload_agnostic_preserves_physical_authority();
        priority_space_share_keeps_native_work_and_exact_authority();
        retained_dependency_progress_is_not_general_issue();
        progress_authority_validation_and_unknown_state();
        std::cout << "Execution priority physical-boundary policy checks passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
