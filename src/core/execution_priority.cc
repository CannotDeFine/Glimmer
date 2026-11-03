#include "glimmer/core/execution_priority.h"

#include <algorithm>

namespace glimmer::core {
ExecutionPriorityIdleWindow::ExecutionPriorityIdleWindow(
    ExecutionPriorityIdleConfig config) noexcept
    : config_(config) {}

bool ExecutionPriorityIdleWindow::enabled() const noexcept {
    return config_.priority != 0 && config_.duration_ns != 0;
}

void ExecutionPriorityIdleWindow::note_issue(std::uint32_t priority) noexcept {
    if (!enabled() || priority < config_.priority) return;
    has_activity_ = true;
    idle_since_ns_.reset();
}

std::uint64_t ExecutionPriorityIdleWindow::remaining_ns(std::uint64_t now_ns) const noexcept {
    if (!idle_since_ns_) return 0;
    if (now_ns < *idle_since_ns_) return config_.duration_ns;
    const auto elapsed = now_ns - *idle_since_ns_;
    return elapsed < config_.duration_ns ? config_.duration_ns - elapsed : 0;
}

bool ExecutionPriorityIdleWindow::has_idle_window() const noexcept {
    return idle_since_ns_.has_value();
}

ExecutionPriorityPlan ExecutionPriorityIdleWindow::apply(
    std::span<const ExecutionPriorityPeer> peers, ExecutionSchedulingMode mode,
    std::uint64_t now_ns, ExecutionPriorityPlan plan) noexcept {
    if (!enabled() || !is_priority_execution_mode(mode)) return plan;
    if (!plan.is_valid) return plan;
    if (plan.size != peers.size() || peers.size() > kMaxExecutionPriorityPeers ||
        (last_now_ns_ && now_ns < *last_now_ns_))
        return {};
    for (std::size_t index = 0; index < peers.size(); ++index)
        if (plan.decisions[index].session != peers[index].session) return {};
    last_now_ns_ = now_ns;
    if (std::ranges::any_of(peers, [](const auto& peer) { return peer.has_unknown_state; })) {
        idle_since_ns_.reset();
        return plan;
    }
    bool has_high_work{};
    for (const auto& peer : peers) {
        if (peer.priority < config_.priority) continue;
        if (peer.has_physical_work) note_issue(peer.priority);
        has_high_work =
            has_high_work || peer.has_ready_work || peer.has_physical_work || peer.has_paused_work;
    }
    if (has_high_work) {
        idle_since_ns_.reset();
        return plan;
    }
    if (!has_activity_) return plan;
    if (!idle_since_ns_) idle_since_ns_ = now_ns;
    if (!remaining_ns(now_ns)) {
        has_activity_ = false;
        idle_since_ns_.reset();
        return plan;
    }
    for (std::size_t index = 0; index < peers.size(); ++index) {
        auto& decision = plan.decisions[index];
        if (peers[index].priority >= config_.priority ||
            decision.action != ExecutionPriorityAction::kExecute)
            continue;
        decision.command.reset();
        decision.action = peers[index].progress ? ExecutionPriorityAction::kExecuteProgress
                                                : ExecutionPriorityAction::kWait;
        decision.progress = peers[index].progress;
    }
    return plan;
}

ExecutionPriorityPlan plan_execution_rebalance(
    std::span<const ExecutionPriorityPeer> peers,
    const ExecutionRebalanceProposal& proposal) noexcept {
    if (!proposal.size || proposal.size > proposal.legs.size()) return {};
    auto plan = plan_execution_priority(peers);
    if (!plan.is_valid) return {};
    std::array<std::size_t, 2> indices{};
    std::optional<std::uint32_t> priority;
    for (std::size_t leg = 0; leg < proposal.size; ++leg) {
        const auto& target = proposal.legs[leg];
        const auto found =
            std::ranges::find(peers, target.session, &ExecutionPriorityPeer::session);
        if (!target.session || found == peers.end() || !found->has_physical_work ||
            found->physical != target.physical || !target.physical.sequence ||
            !target.physical.generation || found->reservation_generation || found->progress ||
            (priority && found->priority != *priority) ||
            (leg && target.session == proposal.legs[0].session))
            return {};
        indices[leg] = static_cast<std::size_t>(found - peers.begin());
        priority = found->priority;
    }
    if (proposal.trigger) {
        const auto& trigger = *proposal.trigger;
        const auto found =
            std::ranges::find(peers, trigger.session, &ExecutionPriorityPeer::session);
        if (!trigger.session || found == peers.end() || !found->has_ready_work ||
            found->has_physical_work || found->has_paused_work || found->priority != priority ||
            found->command != trigger.command || trigger.command.is_resume ||
            trigger.command.physical_generation || trigger.command.command_epoch != 1 ||
            std::ranges::any_of(std::span{proposal.legs}.first(proposal.size),
                                [&](const auto& leg) { return leg.session == trigger.session; }))
            return {};
    }
    if (plan.selected_priority != priority ||
        std::ranges::any_of(peers, [](const auto& peer) { return peer.has_unknown_state; }))
        return plan;
    for (std::size_t index = 0; index < plan.size; ++index) {
        auto& decision = plan.decisions[index];
        if (decision.action != ExecutionPriorityAction::kExecute) continue;
        decision.command.reset();
        if (peers[index].progress) {
            decision.action = ExecutionPriorityAction::kExecuteProgress;
            decision.progress = peers[index].progress;
        } else {
            decision.action = ExecutionPriorityAction::kWait;
        }
    }
    for (std::size_t leg = 0; leg < proposal.size; ++leg) {
        auto& decision = plan.decisions[indices[leg]];
        decision.action = ExecutionPriorityAction::kRequestRetreat;
        decision.command.reset();
    }
    return plan;
}

ExecutionPairPlan plan_execution_pair(std::span<const ExecutionPriorityPeer> peers,
                                      const ExecutionPairProposal& proposal,
                                      ExecutionSchedulingMode mode) noexcept {
    if (mode == ExecutionSchedulingMode::kProfile ||
        mode == ExecutionSchedulingMode::kPrioritySpaceShare || proposal.first_serial > 1 ||
        (proposal.is_shared && proposal.first_serial) || !proposal.legs[0].session ||
        proposal.legs[0].session == proposal.legs[1].session)
        return {};
    const auto ordinary = plan_execution_priority(peers, mode);
    if (!ordinary.is_valid) return {};
    std::array<std::size_t, 2> indices{};
    for (std::size_t leg = 0; leg < proposal.legs.size(); ++leg) {
        const auto& selected = proposal.legs[leg];
        const auto found =
            std::ranges::find(peers, selected.session, &ExecutionPriorityPeer::session);
        if (found == peers.end() || found->command != selected.command) return {};
        const auto context = found->pair_context;
        if (!context) return {};
        if (!selected.catalog_generation ||
            selected.catalog_generation != context->catalog_generation ||
            selected.workers != context->workers ||
            selected.workers > static_cast<unsigned int>(INT32_MAX) ||
            (selected.command.is_resume && !selected.workers) ||
            (proposal.is_shared && !selected.workers))
            return {};
        indices[leg] = static_cast<std::size_t>(found - peers.begin());
    }
    if (proposal.is_shared) {
        const auto& first = peers[indices[0]];
        const auto& second = peers[indices[1]];
        if (!first.pair_context->execution_domain ||
            first.pair_context->execution_domain != second.pair_context->execution_domain ||
            !first.command->member_generation || !second.command->member_generation ||
            first.command->member_generation == second.command->member_generation ||
            first.command->physical_configuration_epoch !=
                second.command->physical_configuration_epoch)
            return {};
    }
    ExecutionPairPlan result;
    result.priority = ordinary;
    result.is_valid = true;
    // A failed/missing pair must not leak the ordinary planner's broad Execute
    // permissions to unrelated work. Only exact native progress remains usable.
    for (auto& decision : std::span{result.priority.decisions}.first(result.priority.size)) {
        if (decision.action == ExecutionPriorityAction::kExecute)
            decision.action = ExecutionPriorityAction::kWait;
        decision.command.reset();
    }
    if (std::ranges::any_of(peers, [](const auto& peer) {
            return peer.has_physical_work || peer.has_unknown_state;
        }))
        return result;
    const auto can_grant = [&](std::size_t leg) {
        return ordinary.decisions[indices[leg]].action == ExecutionPriorityAction::kExecute &&
               peers[indices[leg]].pair_context->is_configuration_materialized;
    };
    if (proposal.is_shared ? !can_grant(0) || !can_grant(1) : !can_grant(proposal.first_serial))
        return result;
    result.is_shared = proposal.is_shared;
    result.grant_count = proposal.is_shared ? 2 : 1;
    for (std::size_t index = 0; index < result.grant_count; ++index) {
        const auto leg = proposal.is_shared ? index : proposal.first_serial;
        result.grants[index] = proposal.legs[leg];
        auto& decision = result.priority.decisions[indices[leg]];
        decision.action = ExecutionPriorityAction::kExecute;
        decision.command = proposal.legs[leg].command;
    }
    return result;
}

ExecutionPriorityPlan plan_execution_priority(
    std::span<const ExecutionPriorityPeer> peers) noexcept {
    return plan_execution_priority(peers, ExecutionSchedulingMode::kPriority);
}
ExecutionPriorityPlan plan_execution_priority(std::span<const ExecutionPriorityPeer> peers,
                                              ExecutionSchedulingMode mode) noexcept {
    ExecutionPriorityPlan result;
    if (peers.size() > kMaxExecutionPriorityPeers ||
        (!is_priority_execution_mode(mode) && mode != ExecutionSchedulingMode::kNaive &&
         mode != ExecutionSchedulingMode::kProfile &&
         mode != ExecutionSchedulingMode::kWorkloadAgnostic &&
         mode != ExecutionSchedulingMode::kPrioritySpaceShare))
        return result;
    bool has_unknown = false;
    for (std::size_t index = 0; index < peers.size(); ++index) {
        const auto& peer = peers[index];
        if (!peer.session || std::ranges::any_of(peers.first(index), [&](const auto& earlier) {
                return earlier.session == peer.session;
            }))
            return {};
        has_unknown = has_unknown || peer.has_unknown_state;
        if (peer.physical &&
            ((!peer.has_physical_work && !peer.has_unknown_state) || !peer.physical->sequence ||
             !peer.physical->generation || peer.reservation_generation))
            return {};
        if (peer.command) {
            const auto& command = *peer.command;
            const bool has_shared = command.member_generation || command.configuration_epoch ||
                                    command.physical_configuration_epoch;
            if (!command.sequence || !command.acceptance_order || !command.geometry_epoch ||
                !command.command_epoch || !command.control_epoch ||
                (command.is_resume ? !command.physical_generation || !peer.has_paused_work
                                   : command.physical_generation || !peer.has_ready_work) ||
                (has_shared && (!command.member_generation || !command.configuration_epoch ||
                                !command.physical_configuration_epoch)))
                return {};
        }
        if (mode == ExecutionSchedulingMode::kPrioritySpaceShare ||
            (is_priority_execution_mode(mode) &&
             (peer.has_ready_work || peer.has_physical_work || peer.has_paused_work)))
            result.selected_priority = result.selected_priority
                                           ? std::max(*result.selected_priority, peer.priority)
                                           : peer.priority;
    }
    for (const auto& peer : peers) {
        if (!peer.progress) continue;
        const auto& progress = *peer.progress;
        if (!progress.reservation || !progress.generation || !progress.cutoff ||
            (!peer.has_ready_work && !peer.has_paused_work && !peer.has_physical_work))
            return {};
        const auto owner =
            std::ranges::find(peers, progress.reservation, &ExecutionPriorityPeer::session);
        if (owner == peers.end() || owner->reservation_generation != progress.generation ||
            (!owner->has_physical_work && !owner->has_unknown_state))
            return {};
    }
    const bool has_lower_physical =
        result.selected_priority && std::ranges::any_of(peers, [&](const auto& peer) {
            return peer.priority < *result.selected_priority && peer.has_physical_work;
        });
    std::optional<std::uint64_t> profile_peer;
    if (mode == ExecutionSchedulingMode::kProfile &&
        !std::ranges::any_of(peers, [](const auto& peer) { return peer.has_physical_work; })) {
        auto found =
            std::ranges::find_if(peers, [](const auto& peer) { return peer.has_paused_work; });
        if (found == peers.end())
            found =
                std::ranges::find_if(peers, [](const auto& peer) { return peer.has_ready_work; });
        if (found != peers.end()) profile_peer = found->session;
    }
    result.size = peers.size();
    result.is_valid = true;
    for (std::size_t index = 0; index < peers.size(); ++index) {
        const auto& peer = peers[index];
        auto& decision = result.decisions[index];
        decision.session = peer.session;
        if (has_unknown) {
            decision.action = ExecutionPriorityAction::kQuarantined;
        } else if (mode == ExecutionSchedulingMode::kProfile) {
            if (profile_peer == peer.session)
                decision.action = ExecutionPriorityAction::kExecute;
            else if (peer.has_ready_work || peer.has_physical_work || peer.has_paused_work)
                decision.action = ExecutionPriorityAction::kWait;
        } else if (mode == ExecutionSchedulingMode::kNaive ||
                   mode == ExecutionSchedulingMode::kWorkloadAgnostic ||
                   mode == ExecutionSchedulingMode::kPrioritySpaceShare) {
            if (peer.has_ready_work || peer.has_paused_work)
                decision.action = ExecutionPriorityAction::kExecute;
            else if (peer.has_physical_work)
                decision.action = ExecutionPriorityAction::kWait;
        } else if (result.selected_priority) {
            if (peer.priority < *result.selected_priority && peer.has_physical_work)
                decision.action = ExecutionPriorityAction::kRequestRetreat;
            else if (peer.priority == *result.selected_priority &&
                     (peer.has_ready_work || peer.has_paused_work) && !has_lower_physical)
                decision.action = ExecutionPriorityAction::kExecute;
            else if (peer.has_ready_work || peer.has_physical_work || peer.has_paused_work)
                decision.action = ExecutionPriorityAction::kWait;
        }
        if (!has_unknown && peer.progress && decision.action != ExecutionPriorityAction::kExecute) {
            decision.action = ExecutionPriorityAction::kExecuteProgress;
            decision.progress = peer.progress;
        }
        if (decision.action == ExecutionPriorityAction::kExecute) decision.command = peer.command;
    }
    return result;
}
}  // namespace glimmer::core
