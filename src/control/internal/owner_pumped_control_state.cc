#include "owner_pumped_control_state.h"

namespace glimmer::control::internal {

OwnerPumpedControlState::OwnerPumpedControlState(LaunchGateOptions settings,
                                                 Gate::OwnerPumpOptions bounds,
                                                 std::optional<NativeEpochExtent> policy)
    : mailbox({.max_outstanding_requests = bounds.max_queued_requests + 2,
               .max_owners = bounds.max_cpu_owners + 1,
               .should_combine_adjacent_requests = true,
               .max_queued_requests = bounds.max_queued_requests}),
      options(std::move(settings)),
      epoch_policy(policy),
      subscriber_limit(bounds.max_queued_requests) {
    subscribers_.reserve(subscriber_limit);
}
std::shared_ptr<OwnerPumpedControlState> OwnerPumpedControlState::create(
    LaunchGateOptions options, Gate::OwnerPumpOptions bounds,
    std::optional<NativeEpochExtent> policy) noexcept {
    if (options.control_socket.empty() || bounds.max_queued_requests == 0 ||
        bounds.max_queued_requests > 65536 || bounds.max_cpu_owners == 0 ||
        bounds.max_cpu_owners > 65535 || bounds.transport_timeout.count() <= 0 ||
        bounds.transport_timeout > std::chrono::seconds{60} ||
        (policy && !valid_native_epoch_extent(*policy)))
        return {};
    try {
        auto state = std::shared_ptr<OwnerPumpedControlState>(
            new OwnerPumpedControlState(std::move(options), bounds, policy));
        if (state->mailbox.initialize({state->options.control_socket, bounds.transport_timeout}) !=
            OwnedExchangeColdStatus::kReady)
            return {};
        state->producer_ = state->mailbox.register_producer();
        if (!state->producer_) return {};
        return state;
    } catch (...) {
        return {};
    }
}
OwnerPumpedControlState::~OwnerPumpedControlState() {
    close();
}
bool OwnerPumpedControlState::same_process() const noexcept {
    return process_id == ::getpid();
}
std::shared_ptr<OwnerPumpedCpuRegistration> OwnerPumpedControlState::register_cpu() noexcept {
    if (!same_process() || is_closed_.load(std::memory_order_acquire)) return {};
    try {
        auto registration = std::make_shared<OwnerPumpedCpuRegistration>();
        registration->state = shared_from_this();
        registration->token = mailbox.register_owner();
        if (!registration->token) return {};
        return registration;
    } catch (...) {
        return {};
    }
}
std::shared_ptr<OwnerPumpedControlState::Gate::ProgressSignal>
OwnerPumpedControlState::subscribe() noexcept {
    if (!same_process() || is_closed_.load(std::memory_order_acquire)) return {};
    try {
        auto signal = std::make_shared<Gate::ProgressSignal>();
        signal->owner_process_ = process_id;
        signal->activity_ = shared_from_this();
        const std::lock_guard lock(subscribers_mutex_);
        if (is_closed_.load(std::memory_order_acquire)) return {};
        for (auto& subscriber : subscribers_) {
            if (subscriber.expired()) {
                subscriber = signal;
                return signal;
            }
        }
        if (subscribers_.size() >= subscriber_limit) return {};
        subscribers_.emplace_back(signal);
        return signal;
    } catch (...) {
        return {};
    }
}
void OwnerPumpedControlState::broadcast() noexcept {
    if (!same_process()) return;
    try {
        std::size_t count = 0;
        {
            const std::lock_guard lock(subscribers_mutex_);
            count = subscribers_.size();
        }
        for (std::size_t index = 0; index < count; ++index) {
            std::shared_ptr<Gate::ProgressSignal> signal;
            {
                const std::lock_guard lock(subscribers_mutex_);
                signal = subscribers_[index].lock();
            }
            if (signal) signal->notify_result();
        }
    } catch (...) {
        contain();
    }
}
void OwnerPumpedControlState::close() noexcept {
    if (!same_process()) return;
    is_closed_.store(true, std::memory_order_release);
    mailbox.close();
}
void OwnerPumpedControlState::contain() noexcept {
    if (!same_process()) return;
    is_closed_.store(true, std::memory_order_release);
    mailbox.quarantine();
}
void OwnerPumpedControlState::notify_activity() noexcept {
    if (same_process() && !mailbox.notify_activity()) contain();
}
OwnerPumpedControlState::Gate::OwnerPumpSnapshot OwnerPumpedControlState::snapshot()
    const noexcept {
    const auto status = mailbox.snapshot();
    return {true,
            status.unsettled_requests,
            status.executor_owners,
            status.has_executor,
            status.is_closed,
            status.is_quarantined,
            status.accepting_executor_owners};
}
OwnerPumpedCpuRegistration::~OwnerPumpedCpuRegistration() {
    if (state && token && state->same_process() && !is_detached.load(std::memory_order_acquire) &&
        !detach())
        state->contain();
}
bool OwnerPumpedCpuRegistration::detach() noexcept {
    if (!state || !token || !state->same_process() || is_detached.load(std::memory_order_acquire))
        return false;
    if (!state->mailbox.detach_owner(token)) return false;
    is_detached.store(true, std::memory_order_release);
    return true;
}
}  // namespace glimmer::control::internal
