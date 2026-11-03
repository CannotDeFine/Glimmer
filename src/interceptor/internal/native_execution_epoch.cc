#include "native_execution_epoch.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace glimmer::interceptor {
namespace {
NativeEpochAuthorization checked_authorization(NativeEpochScope scope,
                                               NativeEpochAuthorization authorization) {
    if (!scope.context_registration || !scope.stream_registration || !scope.caller_registration ||
        !authorization.execution.task_id || !authorization.execution.generation ||
        !authorization.max_calls || authorization.max_calls > 65536 ||
        !authorization.calls_per_prefix ||
        authorization.calls_per_prefix > authorization.max_calls ||
        !authorization.max_pending_prefixes || authorization.max_pending_prefixes > 16 ||
        authorization.acceptance_duration <= std::chrono::microseconds::zero() ||
        authorization.acceptance_duration > std::chrono::seconds{1}) {
        throw std::invalid_argument("invalid native execution epoch authorization");
    }
    return authorization;
}
NativeExecutionEpoch::TimePoint bounded_deadline(NativeExecutionEpoch::TimePoint started,
                                                 std::chrono::microseconds duration) {
    const auto interval = std::chrono::duration_cast<std::chrono::steady_clock::duration>(duration);
    const auto maximum = NativeExecutionEpoch::TimePoint::max();
    return started > maximum - interval ? maximum : started + interval;
}
}  // namespace

NativeExecutionEpoch::NativeExecutionEpoch(NativeEpochScope scope,
                                           NativeEpochAuthorization authorization,
                                           TimePoint started)
    : epoch_scope(scope),
      extent(checked_authorization(scope, authorization)),
      epoch_identity(std::make_shared<int>(0)),
      acceptance_deadline(bounded_deadline(started, extent.acceptance_duration)),
      last_time_(started),
      prefixes_(extent.max_pending_prefixes) {}

bool NativeExecutionEpoch::can_settle() const noexcept {
    return phase_ == NativeEpochPhase::kOpen || phase_ == NativeEpochPhase::kStopping;
}
void NativeExecutionEpoch::stop(bool has_failed) noexcept {
    has_failed_ = has_failed_ || has_failed;
    if (phase_ == NativeEpochPhase::kOpen) phase_ = NativeEpochPhase::kStopping;
}
bool NativeExecutionEpoch::observe_time(TimePoint now) noexcept {
    if (now < last_time_) {
        stop(true);
        return false;
    }
    last_time_ = now;
    if (now >= acceptance_deadline) stop(false);
    return phase_ == NativeEpochPhase::kOpen;
}
bool NativeExecutionEpoch::matches(const NativeEpochIssue& issue) const noexcept {
    return issue.owner_ == epoch_identity && issue.serial_ != 0 && issue.serial_ == current_issue_;
}
bool NativeExecutionEpoch::has_prepared_fence() const noexcept {
    return std::ranges::any_of(
        prefixes_, [](const Prefix& prefix) { return prefix.state == FenceState::kPrepared; });
}
std::uint32_t NativeExecutionEpoch::pending_prefixes() const noexcept {
    return static_cast<std::uint32_t>(std::ranges::count_if(
        prefixes_, [](const Prefix& prefix) { return prefix.state != FenceState::kFree; }));
}
NativeExecutionEpoch::Prefix* NativeExecutionEpoch::find(const NativeEpochFence& fence) noexcept {
    if (fence.owner_ != epoch_identity || !fence.serial_) return nullptr;
    const auto found = std::ranges::find_if(prefixes_, [&](const Prefix& prefix) {
        return prefix.state != FenceState::kFree && prefix.fence.serial_ == fence.serial_ &&
               prefix.fence.through_calls_ == fence.through_calls_;
    });
    return found == prefixes_.end() ? nullptr : &*found;
}
std::optional<NativeEpochIssue> NativeExecutionEpoch::reserve(NativeEpochScope scope,
                                                              TimePoint now) {
    const std::lock_guard lock(mutex_);
    if (scope != epoch_scope || phase_ != NativeEpochPhase::kOpen || !observe_time(now) ||
        current_issue_ || has_prepared_fence() || unfenced_calls_ >= extent.calls_per_prefix ||
        pending_prefixes() == extent.max_pending_prefixes)
        return std::nullopt;
    if (issue_serial_ == std::numeric_limits<std::uint64_t>::max()) {
        stop(true);
        return std::nullopt;
    }
    NativeEpochIssue issue;
    issue.owner_ = epoch_identity;
    issue.serial_ = ++issue_serial_;
    current_issue_ = issue.serial_;
    return issue;
}
bool NativeExecutionEpoch::enter_original(const NativeEpochIssue& issue, TimePoint now) {
    const std::lock_guard lock(mutex_);
    if (!matches(issue) || has_entered_original_ || phase_ != NativeEpochPhase::kOpen ||
        !observe_time(now))
        return false;
    has_entered_original_ = true;
    ++issued_calls_;
    if (issued_calls_ == extent.max_calls) stop(false);
    return true;
}
bool NativeExecutionEpoch::discard_unissued(const NativeEpochIssue& issue) {
    const std::lock_guard lock(mutex_);
    if (!matches(issue) || has_entered_original_) return false;
    current_issue_ = 0;
    return true;
}
bool NativeExecutionEpoch::original_returned(const NativeEpochIssue& issue,
                                             NativeEpochCallOutcome outcome) {
    const std::lock_guard lock(mutex_);
    if (!matches(issue) || !has_entered_original_ || !can_settle() ||
        (outcome != NativeEpochCallOutcome::kSuccess &&
         outcome != NativeEpochCallOutcome::kRejected &&
         outcome != NativeEpochCallOutcome::kUncertain))
        return false;
    ++unfenced_calls_;
    current_issue_ = 0;
    has_entered_original_ = false;
    if (outcome != NativeEpochCallOutcome::kSuccess) stop(true);
    return true;
}
bool NativeExecutionEpoch::request_stop(NativeEpochStopReason reason) {
    const std::lock_guard lock(mutex_);
    if (!can_settle() ||
        (reason != NativeEpochStopReason::kRequested && reason != NativeEpochStopReason::kFailure))
        return false;
    stop(reason == NativeEpochStopReason::kFailure);
    return true;
}
std::optional<NativeEpochFence> NativeExecutionEpoch::prepare_fence() {
    const std::lock_guard lock(mutex_);
    if (!can_settle() || current_issue_ || !unfenced_calls_ || has_prepared_fence())
        return std::nullopt;
    const auto free = std::ranges::find(prefixes_, FenceState::kFree, &Prefix::state);
    if (free == prefixes_.end()) return std::nullopt;
    if (fence_serial_ == std::numeric_limits<std::uint64_t>::max()) {
        stop(true);
        return std::nullopt;
    }
    free->fence.owner_ = epoch_identity;
    free->fence.serial_ = ++fence_serial_;
    free->fence.through_calls_ = issued_calls_;
    free->state = FenceState::kPrepared;
    unfenced_calls_ = 0;
    return free->fence;
}
bool NativeExecutionEpoch::publish_fence_result(const NativeEpochFence& fence, bool has_succeeded) {
    const std::lock_guard lock(mutex_);
    auto* prefix = find(fence);
    if (!can_settle() || !prefix || prefix->state != FenceState::kPrepared) return false;
    prefix->state = has_succeeded ? FenceState::kRecorded : FenceState::kFailed;
    if (!has_succeeded) stop(true);
    return true;
}
bool NativeExecutionEpoch::prove_fence(const NativeEpochFence& fence) {
    const std::lock_guard lock(mutex_);
    const auto* prefix = find(fence);
    if (!can_settle() || !prefix || prefix->state != FenceState::kRecorded ||
        fence.through_calls_ <= completed_calls_)
        return false;
    completed_calls_ = fence.through_calls_;
    for (auto& covered : prefixes_) {
        if (covered.state == FenceState::kRecorded &&
            covered.fence.through_calls_ <= completed_calls_) {
            covered = {};
        }
    }
    return true;
}
bool NativeExecutionEpoch::fail_fence_completion(const NativeEpochFence& fence) {
    const std::lock_guard lock(mutex_);
    auto* prefix = find(fence);
    if (!can_settle() || !prefix || prefix->state != FenceState::kRecorded) return false;
    prefix->state = FenceState::kFailed;
    stop(true);
    return true;
}
bool NativeExecutionEpoch::prove_scope_stopped(NativeEpochScope scope) {
    const std::lock_guard lock(mutex_);
    if (!can_settle() || scope != epoch_scope || current_issue_ || has_prepared_fence() ||
        has_scope_proof_)
        return false;
    has_scope_proof_ = true;
    stop(false);
    completed_calls_ = issued_calls_;
    unfenced_calls_ = 0;
    std::ranges::fill(prefixes_, Prefix{});
    return true;
}
std::optional<NativeEpochTerminal> NativeExecutionEpoch::prepare_terminal() {
    const std::lock_guard lock(mutex_);
    if (phase_ != NativeEpochPhase::kStopping || current_issue_ || unfenced_calls_ ||
        pending_prefixes() || completed_calls_ != issued_calls_)
        return std::nullopt;
    NativeEpochTerminal terminal;
    terminal.owner_ = epoch_identity;
    terminal.execution_ = extent.execution;
    terminal.disposition_ =
        has_failed_ ? NativeEpochDisposition::kFail : NativeEpochDisposition::kComplete;
    terminal.issued_calls_ = issued_calls_;
    phase_ = NativeEpochPhase::kReporting;
    return terminal;
}
bool NativeExecutionEpoch::publish_terminal_result(const NativeEpochTerminal& terminal,
                                                   bool has_succeeded) {
    const std::lock_guard lock(mutex_);
    if (phase_ != NativeEpochPhase::kReporting || terminal.owner_ != epoch_identity) return false;
    phase_ = has_succeeded ? NativeEpochPhase::kComplete : NativeEpochPhase::kRetained;
    return true;
}
NativeEpochSnapshot NativeExecutionEpoch::snapshot() const {
    const std::lock_guard lock(mutex_);
    return {phase_,
            issued_calls_,
            completed_calls_,
            unfenced_calls_,
            pending_prefixes(),
            current_issue_ != 0 && !has_entered_original_,
            has_entered_original_,
            has_failed_,
            phase_ != NativeEpochPhase::kComplete};
}

}  // namespace glimmer::interceptor
