#include "process_control_mailbox.h"

#include <atomic>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <type_traits>
#include <unistd.h>
#include <utility>
#include <vector>

namespace glimmer::control::internal {

struct ControlCompletionState {
    static_assert(std::atomic<bool>::is_always_lock_free);
    const pid_t process_id = ::getpid();
    ControlRequestOutcome outcome;
    std::atomic<bool> is_ready{false};
    std::atomic<bool> is_consumed{false};

    void publish(ControlRequestOutcome value) noexcept {
        static_assert(std::is_nothrow_copy_assignable_v<ControlRequestOutcome>);
        outcome = value;
        is_ready.store(true, std::memory_order_release);
    }
};

struct ProcessControlState {
    struct Owner {
        std::uint64_t generation = 0;
        bool is_registered = false;
        bool can_execute = true;
        bool is_retiring = false;
    };
    struct Request {
        ControlOwnerToken producer;
        std::unique_ptr<RemoteControlOperation> operation;
        std::shared_ptr<ControlCompletionState> completion;
        std::unique_ptr<PreparedControlFuture> original;
        std::shared_ptr<ControlCompletionState> combination_predecessor;
        std::unique_ptr<RemoteControlOperation> combined_operation;
        std::unique_ptr<Request> retained_successor;
        bool is_started = false;
        bool has_successor = false;
    };

    const pid_t process_id = ::getpid();
    const std::shared_ptr<const void> identity = std::make_shared<const std::uint8_t>(0);
    const bool should_combine_adjacent_requests;
    const std::size_t max_queued_requests;
    mutable std::mutex mutex;
    LocalControlWake wake;
    ServiceIncarnationBinding binding;
    OwnedControlExchange endpoint{binding};
    std::vector<std::unique_ptr<Request>> requests;
    std::vector<Owner> owners;
    std::unique_ptr<Request> uncertain;
    std::size_t head = 0;
    std::size_t count = 0;
    std::size_t owner_count = 0;
    std::size_t executor_owner_count = 0;
    std::size_t accepting_executor_owner_count = 0;
    std::uint64_t owner_generation = 0;
    std::uint64_t lease_generation = 0;
    std::uint64_t active_lease = 0;
    ControlOwnerToken executor;
    bool has_startup = false;
    bool is_initialized = false;
    bool is_closed = false;
    bool is_containing = false;
    bool is_advancing = false;
    std::atomic<bool> is_poisoned{false};

    explicit ProcessControlState(ProcessControlMailboxBounds bounds)
        : should_combine_adjacent_requests(bounds.should_combine_adjacent_requests),
          max_queued_requests(bounds.max_queued_requests),
          requests(bounds.max_outstanding_requests),
          owners(bounds.max_owners) {}

    bool same_process() const noexcept {
        return process_id == ::getpid();
    }
    bool can_operate() const noexcept {
        return same_process() && !is_poisoned.load(std::memory_order_acquire);
    }
    void poison() noexcept {
        if (same_process()) {
            is_poisoned.store(true, std::memory_order_release);
            static_cast<void>(wake.notify());
        }
    }
    // Caller holds mutex and has checked creating PID before locking.
    bool owns(const ControlOwnerToken& owner) const noexcept {
        return !is_poisoned.load(std::memory_order_acquire) && owner.identity_ == identity &&
               owner.slot_ < owners.size() && owner.generation_ != 0 &&
               owners[owner.slot_].is_registered &&
               owners[owner.slot_].generation == owner.generation_;
    }
    bool owns_lease(const ControlOwnerToken& owner, std::uint64_t generation) const noexcept {
        return owns(owner) && generation != 0 && active_lease == generation &&
               executor.slot_ == owner.slot_ && executor.generation_ == owner.generation_;
    }

    OwnedExchangeColdStatus initialize(OwnedExchangeConfig config) {
        if (!can_operate()) return OwnedExchangeColdStatus::kRejected;
        {
            const std::lock_guard lock(mutex);
            if (!can_operate() || has_startup || is_closed)
                return OwnedExchangeColdStatus::kRejected;
            has_startup = true;
        }
        const auto result = endpoint.initialize(config);
        {
            const std::lock_guard lock(mutex);
            is_initialized = result == OwnedExchangeColdStatus::kReady;
            if (!is_initialized) is_closed = true;
        }
        return result;
    }

    ControlOwnerToken register_owner(bool can_execute = true) {
        if (!can_operate()) return {};
        const std::lock_guard lock(mutex);
        if (!can_operate() || !is_initialized || is_closed || uncertain ||
            owner_generation == std::numeric_limits<std::uint64_t>::max())
            return {};
        for (std::size_t slot = 0; slot < owners.size(); ++slot) {
            if (owners[slot].is_registered) continue;
            owners[slot] = {++owner_generation, true, can_execute};
            ++owner_count;
            if (can_execute) {
                ++executor_owner_count;
                ++accepting_executor_owner_count;
            }
            ControlOwnerToken owner;
            owner.identity_ = identity;
            owner.slot_ = slot;
            owner.generation_ = owner_generation;
            return owner;
        }
        return {};
    }

    bool retire(const ControlOwnerToken& owner) {
        if (!can_operate()) return false;
        {
            const std::lock_guard lock(mutex);
            if (!owns(owner) || !owners[owner.slot_].can_execute) return false;
            if (!owners[owner.slot_].is_retiring) {
                owners[owner.slot_].is_retiring = true;
                --accepting_executor_owner_count;
            }
        }
        if (!wake.notify()) {
            poison();
            return false;
        }
        return true;
    }
    bool detach(const ControlOwnerToken& owner) {
        if (!can_operate()) return false;
        const std::lock_guard lock(mutex);
        if (!owns(owner) || (active_lease != 0 && executor.slot_ == owner.slot_) ||
            (owner_count == 1 && count != 0) ||
            (owners[owner.slot_].can_execute && executor_owner_count == 1 && count != 0))
            return false;
        owners[owner.slot_].is_registered = false;
        --owner_count;
        if (owners[owner.slot_].can_execute) {
            --executor_owner_count;
            if (!owners[owner.slot_].is_retiring) --accepting_executor_owner_count;
        }
        return true;
    }

    std::optional<ControlExecutorLease> acquire(const std::shared_ptr<ProcessControlState>& self,
                                                const ControlOwnerToken& owner) {
        if (!can_operate()) return std::nullopt;
        const std::lock_guard lock(mutex);
        if (!owns(owner) || !owners[owner.slot_].can_execute || active_lease != 0 ||
            lease_generation == std::numeric_limits<std::uint64_t>::max())
            return std::nullopt;
        active_lease = ++lease_generation;
        executor = owner;
        return ControlExecutorLease{self, owner, active_lease};
    }

    void release(const ControlOwnerToken& owner, std::uint64_t generation) {
        if (!can_operate()) return;
        const std::lock_guard lock(mutex);
        if (owns_lease(owner, generation) && !is_advancing) {
            active_lease = 0;
            executor = {};
        }
    }

    static bool is_unstarted_completion(const Request& request) noexcept {
        return !request.is_started && request.original &&
               std::holds_alternative<std::promise<bool>>(request.original->promise_) &&
               !request.operation->is_native_ &&
               request.operation->task_request_.operation ==
                   TaskExecutionProtocolOperation::kExecutionComplete;
    }

    void prepare_combination(Request& request) {
        if (!should_combine_adjacent_requests || !request.original ||
            !std::holds_alternative<std::promise<CooperativeSubmission>>(
                request.original->promise_) ||
            request.operation->is_native_ ||
            request.operation->task_request_.operation !=
                TaskExecutionProtocolOperation::kExecutionAcquire)
            return;
        TaskExecutionIdentity completed{};
        {
            const std::lock_guard lock(mutex);
            if (!can_operate() || is_closed || uncertain || count == 0) return;
            const auto& previous = *requests[(head + count - 1) % requests.size()];
            if (!is_unstarted_completion(previous)) return;
            request.combination_predecessor = previous.completion;
            completed = {previous.operation->task_request_.task_id,
                         previous.operation->task_request_.generation};
        }
        // All fallible wire preparation is before acceptance and outside mutex.
        request.combined_operation =
            RemoteControlOperation::prepare_handoff(completed, *request.operation);
    }

    ControlCompletionTicket submit(const ControlOwnerToken& owner,
                                   std::unique_ptr<RemoteControlOperation> operation,
                                   std::unique_ptr<PreparedControlFuture> original = nullptr) {
        if (!can_operate() || !operation || !operation->is_prepared()) return {};
        auto request = std::make_unique<Request>();
        request->producer = owner;
        request->operation = std::move(operation);
        request->original = std::move(original);
        request->completion = std::make_shared<ControlCompletionState>();
        prepare_combination(*request);
        ControlCompletionTicket ticket;
        ticket.state_ = request->completion;
        {
            const std::lock_guard lock(mutex);
            const auto active = count != 0 && requests[head]->is_started
                                    ? (requests[head]->has_successor ? 2U : 1U)
                                    : 0U;
            if (!owns(owner) || accepting_executor_owner_count == 0 || !is_initialized ||
                is_closed || uncertain ||
                (max_queued_requests != 0 && count - active >= max_queued_requests) ||
                count >= requests.size())
                return {};
            requests[(head + count) % requests.size()] = std::move(request);
            ++count;
        }
        if (!wake.notify()) poison();
        return ticket;
    }

    ControlCompletionTicket submit(const ControlOwnerToken& owner,
                                   std::unique_ptr<PreparedControlFuture> original) {
        if (!can_operate() || !original || !original->can_accept()) return {};
        auto operation = std::move(original->operation_);
        return submit(owner, std::move(operation), std::move(original));
    }

    bool publish_result(Request& request, ControlRequestOutcome outcome) noexcept {
        if (request.original && !request.original->publish(outcome.result)) return false;
        request.completion->publish(outcome);
        return true;
    }

    ControlMailboxProgress progress(const ControlOwnerToken& owner, std::uint64_t generation,
                                    RemoteControlOperation::Clock::time_point now) {
        if (!same_process()) return ControlMailboxProgress::kRejected;
        if (is_poisoned.load(std::memory_order_acquire))
            return ControlMailboxProgress::kQuarantined;
        Request* request = nullptr;
        Request* successor = nullptr;
        bool should_reject = false;
        bool should_begin = false;
        {
            const std::lock_guard lock(mutex);
            if (!owns_lease(owner, generation) || is_advancing)
                return ControlMailboxProgress::kRejected;
            if (count == 0)
                return uncertain ? ControlMailboxProgress::kQuarantined
                                 : ControlMailboxProgress::kIdle;
            is_advancing = true;
            request = requests[head].get();
            should_reject = uncertain != nullptr;
            should_begin = !request->is_started;
            if (!should_reject && should_begin && is_unstarted_completion(*request) && count > 1) {
                auto& next = *requests[(head + 1) % requests.size()];
                if (!next.is_started && next.original && next.combined_operation &&
                    next.combination_predecessor == request->completion) {
                    request->operation.swap(next.combined_operation);
                    request->has_successor = true;
                    next.is_started = true;
                }
            }
            if (request->has_successor) successor = requests[(head + 1) % requests.size()].get();
            // Producers inspect these flags only under this same mutex.
            request->is_started = true;
        }
        RemoteControlStep step;
        if (should_begin) {
            if (request->original) request->original->started();
            if (successor) successor->original->started();
        }
        if (should_reject) {
            step = request->operation->reject_unpublished() ? RemoteControlStep::kRejected
                                                            : RemoteControlStep::kUncertain;
        } else if (should_begin) {
            step = request->operation->begin(endpoint, now);
        } else {
            step = request->operation->poll(now);
        }
        if (step == RemoteControlStep::kPending) {
            const std::lock_guard lock(mutex);
            is_advancing = false;
            return ControlMailboxProgress::kPending;
        }
        const auto result = request->operation->take_result();
        const bool has_failed_terminal = !should_reject && result && request->original &&
                                         request->original->needs_terminal_ack_ &&
                                         !PreparedControlFuture::terminal_acknowledged(*result);
        const bool has_uncertainty =
            step == RemoteControlStep::kUncertain || !result || has_failed_terminal;
        if (has_uncertainty) {
            // Publish containment before any result or wake makes this outcome
            // observable to a producer. Keep the exact head until retirement.
            const std::lock_guard lock(mutex);
            is_closed = true;
            is_containing = true;
        }
        if (result) {
            const bool was_published = request->operation->was_published();
            if (successor) {
                const auto* handoff = std::get_if<CooperativeHandoff>(&*result);
                if (!handoff ||
                    !request->original->publish(RemoteControlResult{handoff->is_completed}) ||
                    !successor->original->publish(RemoteControlResult{handoff->submission})) {
                    poison();
                    return ControlMailboxProgress::kQuarantined;
                }
                request->completion->publish({step, handoff->is_completed, was_published});
                successor->completion->publish({step, handoff->submission, was_published});
            } else if (!publish_result(*request, {step, *result, was_published})) {
                poison();
                return ControlMailboxProgress::kQuarantined;
            }
        }
        if (request->original) request->original->notify();
        if (successor) successor->original->notify();
        // Head retirement is AFTER actual outcome publication. No new lease
        // or independent successor can observe a retired-but-unpublished head.
        std::unique_ptr<Request> retired;
        {
            const std::lock_guard lock(mutex);
            retired = std::move(requests[head]);
            head = (head + 1) % requests.size();
            --count;
            if (successor) {
                retired->retained_successor = std::move(requests[head]);
                head = (head + 1) % requests.size();
                --count;
            }
            if (has_uncertainty) {
                is_closed = true;
                if (!uncertain) uncertain = std::move(retired);
            }
            is_advancing = false;
        }
        return has_uncertainty ? ControlMailboxProgress::kQuarantined
                               : ControlMailboxProgress::kCompleted;
    }

    std::optional<OwnedExchangeWait> wait_state(const ControlOwnerToken& owner,
                                                std::uint64_t generation) const {
        if (!can_operate()) return std::nullopt;
        Request* request = nullptr;
        {
            const std::lock_guard lock(mutex);
            if (!owns_lease(owner, generation) || is_advancing || count == 0 || uncertain)
                return std::nullopt;
            request = requests[head].get();
        }
        return request->operation->wait_state();
    }

    void close() {
        if (!can_operate()) return;
        {
            const std::lock_guard lock(mutex);
            is_closed = true;
        }
        if (!wake.notify()) poison();
    }
    bool notify_activity() noexcept {
        if (!can_operate()) return false;
        if (wake.notify()) return true;
        poison();
        return false;
    }
    ControlWaitOutcome wait_until(const ControlOwnerToken& owner, std::uint64_t generation,
                                  RemoteControlOperation::Clock::time_point deadline) {
        if (!can_operate()) return {ControlWaitStatus::kRejected};
        {
            const std::lock_guard lock(mutex);
            if (!owns_lease(owner, generation) || is_advancing)
                return {ControlWaitStatus::kRejected};
            if (uncertain || (is_closed && count == 0) ||
                (count != 0 && !requests[head]->is_started))
                return {ControlWaitStatus::kReady, true};
        }
        const auto outcome = wake.wait_until({deadline, wait_state(owner, generation)});
        if (outcome.status == ControlWaitStatus::kError) poison();
        return outcome;
    }
    ControlMailboxSnapshot snapshot() const {
        if (!can_operate()) return {.is_closed = true, .is_quarantined = true};
        const std::lock_guard lock(mutex);
        return {count,
                owner_count,
                active_lease != 0,
                is_advancing,
                is_closed,
                is_containing || uncertain != nullptr,
                executor_owner_count,
                accepting_executor_owner_count};
    }
};

ControlCompletionTicket::operator bool() const noexcept {
    return state_ != nullptr;
}
bool ControlCompletionTicket::is_ready() const noexcept {
    return state_ && state_->process_id == ::getpid() &&
           state_->is_ready.load(std::memory_order_acquire);
}
std::optional<ControlRequestOutcome> ControlCompletionTicket::take_result() noexcept {
    static_assert(std::is_nothrow_copy_constructible_v<ControlRequestOutcome>);
    if (!is_ready()) return std::nullopt;
    bool expected = false;
    if (!state_->is_consumed.compare_exchange_strong(expected, true, std::memory_order_relaxed))
        return std::nullopt;
    return state_->outcome;
}
ControlOwnerToken::operator bool() const noexcept {
    return identity_ && generation_ != 0;
}

ControlExecutorLease::ControlExecutorLease(std::shared_ptr<ProcessControlState> state,
                                           ControlOwnerToken owner,
                                           std::uint64_t generation) noexcept
    : state_(std::move(state)),
      owner_(std::move(owner)),
      generation_(generation),
      thread_(std::this_thread::get_id()) {}
ControlExecutorLease::ControlExecutorLease(ControlExecutorLease&& other) noexcept
    : state_(std::move(other.state_)),
      owner_(std::move(other.owner_)),
      generation_(std::exchange(other.generation_, 0)),
      thread_(other.thread_) {}
ControlExecutorLease::~ControlExecutorLease() {
    try {
        if (state_) state_->release(owner_, generation_);
    } catch (...) {
        state_->poison();
    }
}
ControlMailboxProgress ControlExecutorLease::progress(
    RemoteControlOperation::Clock::time_point now) noexcept {
    if (!state_ || thread_ != std::this_thread::get_id()) return ControlMailboxProgress::kRejected;
    try {
        return state_->progress(owner_, generation_, now);
    } catch (...) {
        state_->poison();
        return ControlMailboxProgress::kQuarantined;
    }
}
std::optional<OwnedExchangeWait> ControlExecutorLease::wait_state() const noexcept {
    if (!state_ || thread_ != std::this_thread::get_id()) return std::nullopt;
    try {
        return state_->wait_state(owner_, generation_);
    } catch (...) {
        state_->poison();
        return std::nullopt;
    }
}

ControlWaitOutcome ControlExecutorLease::wait_until(
    RemoteControlOperation::Clock::time_point deadline) noexcept {
    if (!state_ || thread_ != std::this_thread::get_id()) return {ControlWaitStatus::kRejected};
    try {
        return state_->wait_until(owner_, generation_, deadline);
    } catch (...) {
        state_->poison();
        return {ControlWaitStatus::kError};
    }
}

ProcessControlMailbox::ProcessControlMailbox(ProcessControlMailboxBounds bounds) {
    const bool has_queued_bound = bounds.max_queued_requests != 0;
    if (bounds.max_outstanding_requests == 0 ||
        bounds.max_outstanding_requests > (has_queued_bound ? 65538U : 65536U) ||
        (has_queued_bound && (bounds.max_queued_requests > 65536 ||
                              bounds.max_outstanding_requests != bounds.max_queued_requests + 2)) ||
        bounds.max_owners == 0 || bounds.max_owners > 65536)
        throw std::invalid_argument("invalid process control mailbox bounds");
    state_ = std::make_shared<ProcessControlState>(bounds);
}
ProcessControlMailbox::~ProcessControlMailbox() {
    close();
}
OwnedExchangeColdStatus ProcessControlMailbox::initialize(OwnedExchangeConfig config) noexcept {
    try {
        return state_->initialize(config);
    } catch (...) {
        state_->poison();
        return OwnedExchangeColdStatus::kRejected;
    }
}
ControlOwnerToken ProcessControlMailbox::register_owner() noexcept {
    try {
        return state_->register_owner();
    } catch (...) {
        state_->poison();
        return {};
    }
}
ControlOwnerToken ProcessControlMailbox::register_producer() noexcept {
    try {
        return state_->register_owner(false);
    } catch (...) {
        state_->poison();
        return {};
    }
}
void ProcessControlMailbox::quarantine() noexcept {
    state_->poison();
}
bool ProcessControlMailbox::retire_owner(const ControlOwnerToken& owner) noexcept {
    try {
        return state_->retire(owner);
    } catch (...) {
        state_->poison();
        return false;
    }
}
bool ProcessControlMailbox::detach_owner(const ControlOwnerToken& owner) noexcept {
    try {
        return state_->detach(owner);
    } catch (...) {
        state_->poison();
        return false;
    }
}
std::optional<ControlExecutorLease> ProcessControlMailbox::try_acquire(
    const ControlOwnerToken& owner) noexcept {
    try {
        return state_->acquire(state_, owner);
    } catch (...) {
        state_->poison();
        return std::nullopt;
    }
}
ControlCompletionTicket ProcessControlMailbox::submit(
    const ControlOwnerToken& owner, std::unique_ptr<RemoteControlOperation> operation) noexcept {
    try {
        return state_->submit(owner, std::move(operation));
    } catch (...) {
        return {};
    }
}
ControlCompletionTicket ProcessControlMailbox::submit(
    const ControlOwnerToken& owner, std::unique_ptr<PreparedControlFuture> request) noexcept {
    try {
        return state_->submit(owner, std::move(request));
    } catch (...) {
        return {};
    }
}
void ProcessControlMailbox::close() noexcept {
    try {
        state_->close();
    } catch (...) {
        state_->poison();
    }
}
bool ProcessControlMailbox::notify_activity() noexcept {
    return state_->notify_activity();
}
ControlMailboxSnapshot ProcessControlMailbox::snapshot() const noexcept {
    try {
        return state_->snapshot();
    } catch (...) {
        state_->poison();
        return {.is_closed = true, .is_quarantined = true};
    }
}

}  // namespace glimmer::control::internal
