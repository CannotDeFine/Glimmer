#include "task_recovery_journal.h"

#include <algorithm>
#include <limits>

namespace glimmer::control::internal {
namespace {

[[nodiscard]] bool nonzero(TaskRecoveryJournal::Token token) noexcept {
    return token[0] != 0 || token[1] != 0;
}

}  // namespace

TaskRecoveryJournal::TaskRecoveryJournal(Token incarnation, Limits limits) noexcept
    : incarnation(incarnation),
      limits(limits),
      valid(nonzero(incarnation) && limits.sessions > 0 && limits.sessions <= k_max_sessions &&
            limits.entries > 0 && limits.entries <= k_max_entries && limits.first_session_id > 0 &&
            limits.first_operation_id > 0),
      next_session_id_(limits.first_session_id) {}

std::optional<TaskRecoveryJournal::Session> TaskRecoveryJournal::open(
    Token nonce, TaskPeerIdentity peer) noexcept {
    if (!valid || !nonzero(nonce) || peer.pid == 0 || peer.start_time_ticks == 0) {
        return std::nullopt;
    }
    try {
        std::scoped_lock lock(mutex_);
        SessionSlot* free = nullptr;
        for (std::size_t i = 0; i < limits.sessions; ++i) {
            auto& slot = sessions_[i];
            if (!slot.used) {
                if (free == nullptr) free = &slot;
                continue;
            }
            if (slot.session.nonce == nonce) {
                return slot.session.peer == peer ? std::optional{slot.session} : std::nullopt;
            }
        }
        if (free == nullptr || sessions_exhausted_) {
            return std::nullopt;
        }
        free->used = true;
        free->session = {
            .incarnation = incarnation, .nonce = nonce, .peer = peer, .id = next_session_id_};
        free->next_operation = limits.first_operation_id;
        free->completed_through = limits.first_operation_id - 1;
        free->acknowledged_through = limits.first_operation_id - 1;
        if (next_session_id_ == std::numeric_limits<std::uint64_t>::max()) {
            sessions_exhausted_ = true;
        } else {
            ++next_session_id_;
        }
        return free->session;
    } catch (...) {
        return std::nullopt;
    }
}

TaskRecoveryJournal::SessionSlot* TaskRecoveryJournal::find_session(
    const Session& session) noexcept {
    for (std::size_t i = 0; i < limits.sessions; ++i) {
        auto& candidate = sessions_[i];
        if (candidate.used && candidate.session == session) {
            return &candidate;
        }
    }
    return nullptr;
}

bool TaskRecoveryJournal::acknowledge_locked(SessionSlot& session, std::uint64_t through) noexcept {
    if (through < session.acknowledged_through || through > session.completed_through) {
        return false;
    }
    for (std::size_t i = 0; i < limits.entries; ++i) {
        auto& entry = entries_[i];
        if (entry.used && entry.session_id == session.session.id && entry.operation <= through) {
            if (!entry.committed) return false;
        }
    }
    for (std::size_t i = 0; i < limits.entries; ++i) {
        auto& entry = entries_[i];
        if (entry.used && entry.session_id == session.session.id && entry.operation <= through) {
            entry = {};
        }
    }
    session.acknowledged_through = through;
    return true;
}

TaskRecoveryJournal::BeginResult TaskRecoveryJournal::begin(
    const Session& session, std::uint64_t operation, std::uint64_t acknowledge_through,
    std::span<const std::byte> request) noexcept {
    if (!valid || operation == 0 || request.empty() || request.size() > k_max_message_bytes) {
        return {};
    }
    try {
        std::scoped_lock lock(mutex_);
        auto* state = find_session(session);
        if (state == nullptr || operation <= state->acknowledged_through ||
            acknowledge_through >= operation || acknowledge_through < state->acknowledged_through ||
            acknowledge_through > state->completed_through) {
            return {};
        }
        for (std::size_t i = 0; i < limits.entries; ++i) {
            auto& entry = entries_[i];
            if (!entry.used || entry.session_id != session.id || entry.operation != operation) {
                continue;
            }
            if (entry.request_size != request.size() ||
                !std::equal(request.begin(), request.end(), entry.request.begin())) {
                state->poisoned = true;
                return {};
            }
            if (!acknowledge_locked(*state, acknowledge_through)) return {};
            if (!entry.committed) return {.decision = Decision::kInProgress};
            return {.decision = Decision::kReplay,
                    .result = entry.result,
                    .result_size = entry.result_size};
        }
        if (state->poisoned || state->in_progress || state->operations_exhausted ||
            operation != state->next_operation ||
            !acknowledge_locked(*state, acknowledge_through)) {
            return {};
        }
        for (std::size_t i = 0; i < limits.entries; ++i) {
            auto& entry = entries_[i];
            if (entry.used) continue;
            entry.used = true;
            entry.session_id = session.id;
            entry.operation = operation;
            entry.request_size = request.size();
            std::copy(request.begin(), request.end(), entry.request.begin());
            state->in_progress = true;
            if (operation == std::numeric_limits<std::uint64_t>::max()) {
                state->operations_exhausted = true;
            } else {
                ++state->next_operation;
            }
            return {.decision = Decision::kExecute};
        }
        return {.decision = Decision::kFull};
    } catch (...) {
        return {};
    }
}

bool TaskRecoveryJournal::commit(const Session& session, std::uint64_t operation,
                                 std::span<const std::byte> result) noexcept {
    if (!valid || result.empty() || result.size() > k_max_message_bytes) return false;
    try {
        std::scoped_lock lock(mutex_);
        auto* state = find_session(session);
        if (state == nullptr || !state->in_progress) return false;
        for (std::size_t i = 0; i < limits.entries; ++i) {
            auto& entry = entries_[i];
            if (entry.used && entry.session_id == session.id && entry.operation == operation &&
                !entry.committed) {
                std::copy(result.begin(), result.end(), entry.result.begin());
                entry.result_size = result.size();
                entry.committed = true;
                state->completed_through = operation;
                state->in_progress = false;
                return true;
            }
        }
    } catch (...) {
        return false;
    }
    return false;
}

bool TaskRecoveryJournal::acknowledge(const Session& session, std::uint64_t through) noexcept {
    if (!valid) return false;
    try {
        std::scoped_lock lock(mutex_);
        auto* state = find_session(session);
        return state != nullptr && acknowledge_locked(*state, through);
    } catch (...) {
        return false;
    }
}

bool TaskRecoveryJournal::close(const Session& session) noexcept {
    if (!valid) return false;
    try {
        std::scoped_lock lock(mutex_);
        auto* state = find_session(session);
        if (state == nullptr || state->in_progress ||
            state->acknowledged_through != state->completed_through) {
            return false;
        }
        *state = {};
        return true;
    } catch (...) {
        return false;
    }
}

}  // namespace glimmer::control::internal
