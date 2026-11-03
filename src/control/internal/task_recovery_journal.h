#pragma once

#include "glimmer/control/task_admission.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <span>

namespace glimmer::control::internal {

// Bounded, process-local operation deduplication. The caller must reserve an
// entry before any side effect and commit its complete result without allocating
// afterward. This journal neither dispatches nor owns tasks, proves physical
// stop, persists across restart, nor authorizes replay after delivery to a caller.
// One journal owns an incarnation. The caller supplies trusted, authenticated
// peer identities and cryptographically random nonces; nonzero is not evidence
// of randomness. A restarted service requires a fresh incarnation and a separate
// proof that old physical work has stopped before granting overlapping work.
class TaskRecoveryJournal final {
   public:
    static constexpr std::size_t k_max_sessions = 32;
    static constexpr std::size_t k_max_entries = 128;
    static constexpr std::size_t k_max_message_bytes = 256;
    using Token = std::array<std::uint64_t, 2>;

    struct Session {
        Token incarnation{};
        Token nonce{};
        TaskPeerIdentity peer{};
        std::uint64_t id{};
        friend bool operator==(const Session&, const Session&) = default;
    };
    struct Limits {
        std::size_t sessions = k_max_sessions;
        std::size_t entries = k_max_entries;
        // Trusted test seams. Production uses 1 for both counters.
        std::uint64_t first_session_id = 1;
        std::uint64_t first_operation_id = 1;
    };
    enum class Decision : std::uint8_t {
        kExecute,
        kReplay,
        kInProgress,
        kRejected,
        kFull,
    };
    struct BeginResult {
        Decision decision{Decision::kRejected};
        std::array<std::byte, k_max_message_bytes> result{};
        std::size_t result_size{};
    };

    explicit TaskRecoveryJournal(Token incarnation, Limits limits) noexcept;
    TaskRecoveryJournal(const TaskRecoveryJournal&) = delete;
    TaskRecoveryJournal& operator=(const TaskRecoveryJournal&) = delete;

    // Retrying OPEN with the same nonce and peer returns the same live session.
    // IDs are monotonic and never reused within this journal's incarnation.
    [[nodiscard]] std::optional<Session> open(Token nonce, TaskPeerIdentity peer) noexcept;
    // The request must be canonical. A duplicate with different bytes poisons
    // the session against new operations; an already reserved operation may
    // still commit and recover. ACK advances only over a committed prefix.
    [[nodiscard]] BeginResult begin(const Session& session, std::uint64_t operation,
                                    std::uint64_t acknowledge_through,
                                    std::span<const std::byte> request) noexcept;
    // No allocation: failure leaves the reservation in progress. The caller
    // must never call commit with an incomplete or guessed operation result.
    [[nodiscard]] bool commit(const Session& session, std::uint64_t operation,
                              std::span<const std::byte> result) noexcept;
    // Only the owner can prove all tasks and external resources are retired.
    // It must establish that before close. Unacknowledged operations reject.
    [[nodiscard]] bool acknowledge(const Session& session, std::uint64_t through) noexcept;
    [[nodiscard]] bool close(const Session& session) noexcept;

   private:
    struct SessionSlot {
        bool used{};
        bool poisoned{};
        bool in_progress{};
        bool operations_exhausted{};
        Session session{};
        std::uint64_t next_operation{};
        std::uint64_t completed_through{};
        std::uint64_t acknowledged_through{};
    };
    struct Entry {
        bool used{};
        bool committed{};
        std::uint64_t session_id{};
        std::uint64_t operation{};
        std::array<std::byte, k_max_message_bytes> request{};
        std::array<std::byte, k_max_message_bytes> result{};
        std::size_t request_size{};
        std::size_t result_size{};
    };

    [[nodiscard]] SessionSlot* find_session(const Session& session) noexcept;
    [[nodiscard]] bool acknowledge_locked(SessionSlot& session, std::uint64_t through) noexcept;

    const Token incarnation;
    const Limits limits;
    const bool valid;
    std::mutex mutex_;
    std::uint64_t next_session_id_;
    bool sessions_exhausted_{};
    std::array<SessionSlot, k_max_sessions> sessions_{};
    std::array<Entry, k_max_entries> entries_{};
};

}  // namespace glimmer::control::internal
