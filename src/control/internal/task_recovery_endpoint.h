#pragma once

#include "task_recovery_journal.h"
#include "task_recovery_protocol.h"

#include "glimmer/control/task_execution_admission.h"

#include <array>
#include <mutex>
#include <optional>
#include <string_view>

namespace glimmer::control::internal {

// Host-only, process-local recovery endpoint. The caller supplies authenticated
// peer identities and a fresh random incarnation. It must serialize delivery
// of an individual operation to its application: a replay is not permission
// to issue physical work twice. No transport or backend is owned here.
// Before destruction the owner must stop and join callers and backend work,
// prove physical execution stopped, and retire all external resources. A failed
// retire_after_backend_stopped() leaves local accounting uncertain and this
// endpoint unusable; after those external preconditions the owner may destroy
// the entire endpoint and its private accounting, but cannot reuse its state.
class TaskRecoveryEndpoint final {
   public:
    explicit TaskRecoveryEndpoint(TaskRecoveryToken incarnation, core::MemoryBytes memory_limit,
                                  core::SchedulerOptions options = {});
    TaskRecoveryEndpoint(const TaskRecoveryEndpoint&) = delete;
    TaskRecoveryEndpoint& operator=(const TaskRecoveryEndpoint&) = delete;

    [[nodiscard]] std::optional<TaskRecoveryLine> handle(std::string_view line,
                                                         TaskPeerIdentity peer) noexcept;
    // Owner-only, after backend stop and join. This disables all wire requests.
    // No timeouts, disconnects, or journal ACKs establish its precondition.
    [[nodiscard]] bool retire_after_backend_stopped() noexcept;

   private:
    enum class BindingState : std::uint8_t {
        kFree,
        kReserved,
        kSubmitting,
        kPending,
        kActive,
        kTerminal,
        kForgotten
    };
    struct SessionSlot {
        bool used{};
        TaskRecoveryJournal::Session session{};
    };
    struct Binding {
        BindingState state{BindingState::kFree};
        std::uint64_t session_id{};
        TaskExecutionIdentity execution{};
        std::uint64_t forget_operation{};
    };

    [[nodiscard]] SessionSlot* find_session(const TaskRecoveryRequest& request,
                                            TaskPeerIdentity peer) noexcept;
    [[nodiscard]] Binding* find_binding(std::uint64_t session_id, core::TaskId task_id) noexcept;
    void reclaim_forgotten(std::uint64_t session_id, std::uint64_t through) noexcept;
    [[nodiscard]] TaskRecoveryResponse execute(const TaskRecoveryRequest& request,
                                               TaskPeerIdentity peer,
                                               const TaskAdmissionRequest* prepared_submission);

    const TaskRecoveryToken incarnation;
    TaskRecoveryJournal journal_;
    TaskExecutionAdmissionService admission_;
    std::mutex mutex_;
    bool stopped_{};
    bool uncertain_{};
    std::uint64_t last_session_id_{};
    std::array<SessionSlot, TaskRecoveryJournal::k_max_sessions> sessions_{};
    std::array<Binding, TaskRecoveryJournal::k_max_entries> bindings_{};
};

}  // namespace glimmer::control::internal
