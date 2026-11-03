#include "prepared_control_future.h"

#include <chrono>
#include <type_traits>

namespace glimmer::control::internal {

std::uint64_t PreparedControlFuture::timestamp_now() noexcept {
    const auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
        RemoteControlOperation::Clock::now().time_since_epoch());
    return duration.count() > 0 ? static_cast<std::uint64_t>(duration.count()) : 0;
}

bool PreparedControlFuture::can_accept() const noexcept {
    return process_id == ::getpid() && has_future_ && !is_published_ && operation_ &&
           operation_->is_prepared();
}

void PreparedControlFuture::started() noexcept {
    if (should_measure_timing_) started_nanoseconds_ = timestamp_now();
}

bool PreparedControlFuture::publish(const RemoteControlResult& result) noexcept {
    if (process_id != ::getpid() || !has_future_ || is_published_) return false;
    try {
        const auto timing = should_measure_timing_
                                ? CooperativeRequestTiming{enqueued_nanoseconds_,
                                                           started_nanoseconds_, timestamp_now()}
                                : CooperativeRequestTiming{};
        const bool published = std::visit(
            [&](const auto& value) {
                using Result = std::decay_t<decltype(value)>;
                static_assert(std::is_nothrow_copy_constructible_v<Result>);
                auto* promise = std::get_if<std::promise<Result>>(&promise_);
                if (promise == nullptr) return false;
                auto original = value;
                if constexpr (std::is_same_v<Result, CooperativeClaim>) {
                    original.timing = timing;
                } else if constexpr (std::is_same_v<Result, CooperativeSubmission>) {
                    original.claim.timing = timing;
                } else if constexpr (std::is_same_v<Result, CooperativeHandoff>) {
                    original.submission.claim.timing = timing;
                }
                promise->set_value(original);
                return true;
            },
            result);
        is_published_ = published;
        return published;
    } catch (...) {
        // The mailbox retains/poisons its exact head; no successor or retry.
        return false;
    }
}

void PreparedControlFuture::notify() noexcept {
    if (process_id == ::getpid() && is_published_ && signal_) signal_->notify();
}

bool PreparedControlFuture::requires_terminal_ack(
    const RemoteControlOperation& operation) noexcept {
    if (operation.is_native_) {
        const auto kind = operation.epoch_request_.operation;
        return kind == NativeEpochOperation::kComplete || kind == NativeEpochOperation::kFail ||
               kind == NativeEpochOperation::kCancelPending;
    }
    switch (operation.task_request_.operation) {
        case TaskExecutionProtocolOperation::kSuspend:
        case TaskExecutionProtocolOperation::kExecutionComplete:
        case TaskExecutionProtocolOperation::kExecutionCompleteAcquire:
        case TaskExecutionProtocolOperation::kExecutionFail:
        case TaskExecutionProtocolOperation::kExecutionCancel:
        case TaskExecutionProtocolOperation::kCancel:
            return true;
        default:
            return false;
    }
}

bool PreparedControlFuture::terminal_acknowledged(const RemoteControlResult& result) noexcept {
    if (const auto* acknowledgement = std::get_if<bool>(&result)) return *acknowledgement;
    const auto* handoff = std::get_if<CooperativeHandoff>(&result);
    return handoff != nullptr && handoff->is_completed;
}

}  // namespace glimmer::control::internal
