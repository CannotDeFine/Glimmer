#pragma once

#include "glimmer/control/task_admission.h"

#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

namespace glimmer::interceptor {

struct NativeEpochScope {
    std::uint64_t context_registration{};
    std::uint64_t stream_registration{};
    std::uint64_t caller_registration{};
    friend bool operator==(const NativeEpochScope&, const NativeEpochScope&) = default;
};

// Explicit server-approved extent, not an ordinary single-launch grant.
// This private CPU model validates bounds, not external authentication.
struct NativeEpochAuthorization {
    control::TaskExecutionIdentity execution;
    std::uint32_t max_calls{};
    std::uint32_t calls_per_prefix{};
    std::uint32_t max_pending_prefixes{};
    std::chrono::microseconds acceptance_duration{};
};

class NativeEpochIssue final {
   public:
    [[nodiscard]] explicit operator bool() const noexcept {
        return owner_ && serial_ != 0;
    }

   private:
    friend class NativeExecutionEpoch;
    std::shared_ptr<const void> owner_;
    std::uint64_t serial_{};
};

class NativeEpochFence final {
   public:
    [[nodiscard]] explicit operator bool() const noexcept {
        return owner_ && serial_ != 0;
    }
    [[nodiscard]] std::uint32_t covered_calls() const noexcept {
        return through_calls_;
    }

   private:
    friend class NativeExecutionEpoch;
    std::shared_ptr<const void> owner_;
    std::uint64_t serial_{};
    std::uint32_t through_calls_{};
};

// Preserve the existing enum representation used by the execution components.
// NOLINTBEGIN(performance-enum-size)
enum class NativeEpochCallOutcome : std::uint8_t { kSuccess, kRejected, kUncertain };
enum class NativeEpochStopReason : std::uint8_t { kRequested, kFailure };
enum class NativeEpochDisposition : std::uint8_t { kComplete, kFail };
enum class NativeEpochPhase : std::uint8_t { kOpen, kStopping, kReporting, kRetained, kComplete };
// NOLINTEND(performance-enum-size)

class NativeEpochTerminal final {
   public:
    [[nodiscard]] explicit operator bool() const noexcept {
        return owner_ != nullptr;
    }
    [[nodiscard]] control::TaskExecutionIdentity execution() const noexcept {
        return execution_;
    }
    [[nodiscard]] NativeEpochDisposition disposition() const noexcept {
        return disposition_;
    }
    [[nodiscard]] std::uint32_t issued_calls() const noexcept {
        return issued_calls_;
    }

   private:
    friend class NativeExecutionEpoch;
    std::shared_ptr<const void> owner_;
    control::TaskExecutionIdentity execution_;
    NativeEpochDisposition disposition_{NativeEpochDisposition::kFail};
    std::uint32_t issued_calls_{};
};

struct NativeEpochSnapshot {
    NativeEpochPhase phase{};
    std::uint32_t issued_calls{};
    std::uint32_t completed_calls{};
    std::uint32_t unfenced_calls{};
    std::uint32_t pending_prefixes{};
    bool has_reserved_issue{};
    bool has_entered_original{};
    bool has_failed{};
    bool has_ownership{};
    friend bool operator==(const NativeEpochSnapshot&, const NativeEpochSnapshot&) = default;
};

// Thread-safe CPU metadata only; destruction is externally serialized.
// No method calls CUDA/control, invokes a callback, or proves hardware state.
// Trusted adapters must authenticate authorization and independently establish
// physical proof before calling prove_*. Destruction never releases execution.
// One instance owns one epoch. Tokens retain CPU identity, not this owner or GPU.
class NativeExecutionEpoch final {
   public:
    using TimePoint = std::chrono::steady_clock::time_point;
    NativeExecutionEpoch(NativeEpochScope scope, NativeEpochAuthorization authorization,
                         TimePoint started);
    NativeExecutionEpoch(const NativeExecutionEpoch&) = delete;
    NativeExecutionEpoch& operator=(const NativeExecutionEpoch&) = delete;

    [[nodiscard]] std::optional<NativeEpochIssue> reserve(NativeEpochScope scope, TimePoint now);
    [[nodiscard]] bool enter_original(const NativeEpochIssue& issue, TimePoint now);
    [[nodiscard]] bool discard_unissued(const NativeEpochIssue& issue);
    [[nodiscard]] bool original_returned(const NativeEpochIssue& issue,
                                         NativeEpochCallOutcome outcome);
    [[nodiscard]] bool request_stop(
        NativeEpochStopReason reason = NativeEpochStopReason::kRequested);
    [[nodiscard]] std::optional<NativeEpochFence> prepare_fence();
    // Publish only after the actual original record call has returned. False
    // recording retains a failed slot; it is never retried by this model.
    // Result publication returns metadata acceptance, not CUDA/control success.
    [[nodiscard]] bool publish_fence_result(const NativeEpochFence& fence, bool has_succeeded);
    // Requires actual matching same-stream prefix completion. Later proof covers
    // earlier recorded prefixes too; adapters still own all native resources.
    [[nodiscard]] bool prove_fence(const NativeEpochFence& fence);
    // A failed/unknown actual completion query is not NOT_READY. Retain the
    // recorded slot as failed; only independent whole-scope proof can settle it.
    [[nodiscard]] bool fail_fence_completion(const NativeEpochFence& fence);
    // Requires independent whole-scope completion/destruction and no active
    // original or record call. Covers failed fences without querying old handles.
    [[nodiscard]] bool prove_scope_stopped(NativeEpochScope scope);
    [[nodiscard]] std::optional<NativeEpochTerminal> prepare_terminal();
    // An accepted false result enters retained ownership, not completion.
    [[nodiscard]] bool publish_terminal_result(const NativeEpochTerminal& terminal,
                                               bool has_succeeded);
    [[nodiscard]] NativeEpochSnapshot snapshot() const;

   private:
    // Preserve the existing representation of the epoch state.
    // NOLINTNEXTLINE(performance-enum-size)
    enum class FenceState : std::uint8_t { kFree, kPrepared, kRecorded, kFailed };
    struct Prefix {
        NativeEpochFence fence;
        FenceState state{FenceState::kFree};
    };
    [[nodiscard]] bool can_settle() const noexcept;
    [[nodiscard]] bool observe_time(TimePoint now) noexcept;
    void stop(bool has_failed) noexcept;
    [[nodiscard]] bool matches(const NativeEpochIssue& issue) const noexcept;
    [[nodiscard]] Prefix* find(const NativeEpochFence& fence) noexcept;
    [[nodiscard]] bool has_prepared_fence() const noexcept;
    [[nodiscard]] std::uint32_t pending_prefixes() const noexcept;

    const NativeEpochScope epoch_scope;
    const NativeEpochAuthorization extent;
    const std::shared_ptr<const void> epoch_identity;
    const TimePoint acceptance_deadline;
    TimePoint last_time_;
    std::vector<Prefix> prefixes_;
    mutable std::mutex mutex_;
    NativeEpochPhase phase_{NativeEpochPhase::kOpen};
    std::uint64_t issue_serial_{};
    std::uint64_t current_issue_{};
    std::uint64_t fence_serial_{};
    std::uint32_t issued_calls_{};
    std::uint32_t completed_calls_{};
    std::uint32_t unfenced_calls_{};
    bool has_entered_original_{};
    bool has_failed_{};
    bool has_scope_proof_{};
};

}  // namespace glimmer::interceptor
