#include "glimmer/control/async_cooperative_launch_gate.h"

#include "internal/owner_pumped_control_state.h"

#include <algorithm>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace glimmer::control {
namespace {
std::uint64_t timestamp_now() noexcept {
    const auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch());
    return duration.count() > 0 ? static_cast<std::uint64_t>(duration.count()) : 0;
}
}  // namespace

void AsyncCooperativeLaunchGate::ProgressSignal::notify() noexcept {
    if (owner_process_ != 0 && owner_process_ != ::getpid()) return;
    notify_result();
    if (const auto state = activity_.lock()) state->notify_activity();
}
void AsyncCooperativeLaunchGate::ProgressSignal::notify_result() noexcept {
    {
        const std::lock_guard lock(mutex_);
        is_pending_ = true;
    }
    condition_.notify_one();
}
bool AsyncCooperativeLaunchGate::ProgressSignal::wait_for(std::chrono::microseconds timeout) {
    if (owner_process_ != 0 && owner_process_ != ::getpid()) return false;
    std::unique_lock lock(mutex_);
    if (!condition_.wait_for(lock, timeout, [&] { return is_pending_; })) {
        return false;
    }
    is_pending_ = false;
    return true;
}

struct AsyncCooperativeLaunchGate::Implementation {
    struct Completion {
        TaskExecutionIdentity execution;
        std::promise<bool> result;
    };
    struct Submission {
        CooperativeLaunchRequest settings;
        std::uint64_t enqueued_nanoseconds{};
        std::promise<CooperativeSubmission> result;
    };
    using Request = std::variant<std::function<void()>, std::unique_ptr<Completion>,
                                 std::unique_ptr<Submission>>;
    CooperativeLaunchGate gate;
    const std::size_t limit;
    std::mutex mutex;
    std::condition_variable condition;
    std::deque<Request> requests;
    std::vector<std::weak_ptr<ProgressSignal>> subscribers;
    bool is_closing{};
    std::thread thread;
    std::shared_ptr<internal::OwnerPumpedControlState> pumped;

    Implementation(LaunchGateOptions options, std::size_t max_queued_requests,
                   std::optional<NativeEpochExtent> policy)
        : gate(std::move(options), policy), limit(max_queued_requests) {
        if (limit == 0 || limit > 65536) {
            throw std::invalid_argument("invalid cooperative control mailbox capacity");
        }
        subscribers.reserve(limit);
        thread = std::thread([this] { run(); });
    }
    explicit Implementation(std::shared_ptr<internal::OwnerPumpedControlState> state)
        : gate(), limit(0), pumped(std::move(state)) {}
    void run() noexcept {
        for (;;) {
            Request request;
            std::unique_ptr<Submission> successor;
            {
                std::unique_lock lock(mutex);
                condition.wait(lock, [&] { return is_closing || !requests.empty(); });
                if (requests.empty()) {
                    return;
                }
                request = std::move(requests.front());
                requests.pop_front();
                // Only already accepted adjacent requests can combine. Never
                // wait, cross another operation, or revisit an issued report.
                if (std::holds_alternative<std::unique_ptr<Completion>>(request) &&
                    !requests.empty()) {
                    if (auto* next = std::get_if<std::unique_ptr<Submission>>(&requests.front())) {
                        successor = std::move(*next);
                        requests.pop_front();
                    }
                }
            }
            execute(request, std::move(successor));
            // Result publication precedes the hint. Signal notification cannot
            // call an owner or CUDA; mailbox -> signal is the only lock order.
            const std::lock_guard lock(mutex);
            std::erase_if(subscribers, [](const auto& subscriber) { return subscriber.expired(); });
            for (const auto& subscriber : subscribers) {
                if (const auto signal = subscriber.lock()) {
                    signal->notify();
                }
            }
        }
    }
    void execute(Request& request, std::unique_ptr<Submission> successor) {
        if (auto* completion = std::get_if<std::unique_ptr<Completion>>(&request)) {
            if (successor) {
                const auto started =
                    successor->settings.should_measure_timing ? timestamp_now() : 0;
                auto result = gate.complete_and_submit_and_claim((*completion)->execution,
                                                                 successor->settings);
                if (successor->settings.should_measure_timing) {
                    result.submission.claim.timing = {successor->enqueued_nanoseconds, started,
                                                      timestamp_now()};
                }
                (*completion)->result.set_value(result.is_completed);
                successor->result.set_value(result.submission);
            } else {
                (*completion)->result.set_value(gate.complete((*completion)->execution));
            }
        } else if (auto* submission = std::get_if<std::unique_ptr<Submission>>(&request)) {
            auto& value = **submission;
            const auto started = value.settings.should_measure_timing ? timestamp_now() : 0;
            auto result = gate.submit_and_claim(value.settings);
            if (value.settings.should_measure_timing) {
                result.claim.timing = {value.enqueued_nanoseconds, started, timestamp_now()};
            }
            value.result.set_value(result);
        } else {
            // All generic requests are valid exactly-once packaged tasks.
            std::get<std::function<void()>>(request)();
        }
    }
    template <typename Value>
    auto enqueue_typed(std::unique_ptr<Value> request) noexcept
        -> decltype(request->result.get_future()) {
        try {
            auto future = request->result.get_future();
            {
                const std::lock_guard lock(mutex);
                if (is_closing || requests.size() >= limit) {
                    return {};
                }
                requests.emplace_back(std::move(request));
            }
            condition.notify_one();
            return future;
        } catch (...) {
            return {};
        }
    }
    template <typename Result, typename Function>
    std::future<Result> enqueue(Function operation) noexcept {
        try {
            auto request = std::make_shared<std::packaged_task<Result()>>(std::move(operation));
            auto future = request->get_future();
            std::function<void()> invoke = [request] { (*request)(); };
            {
                const std::lock_guard lock(mutex);
                if (is_closing || requests.size() >= limit) {
                    return {};
                }
                requests.push_back(std::move(invoke));
            }
            condition.notify_one();
            return future;
        } catch (...) {
            return {};
        }
    }
    template <typename Result, typename Function>
    std::future<Result> enqueue_measured(Function operation, bool should_measure_timing) noexcept {
        if (!should_measure_timing) {
            return enqueue<Result>(std::move(operation));
        }
        const auto enqueued = timestamp_now();
        return enqueue<Result>([operation = std::move(operation), enqueued] {
            const auto started = timestamp_now();
            auto result = operation();
            const CooperativeRequestTiming timing{enqueued, started, timestamp_now()};
            if constexpr (std::is_same_v<Result, CooperativeSubmission>) {
                result.claim.timing = timing;
            } else {
                result.timing = timing;
            }
            return result;
        });
    }
};

AsyncCooperativeLaunchGate::AsyncCooperativeLaunchGate(
    LaunchGateOptions options, std::size_t max_queued_requests,
    std::optional<NativeEpochExtent> native_epoch_policy)
    : implementation_(std::make_unique<Implementation>(std::move(options), max_queued_requests,
                                                       native_epoch_policy)) {}
AsyncCooperativeLaunchGate::~AsyncCooperativeLaunchGate() {
    shutdown();
}

AsyncCooperativeLaunchGate::AsyncCooperativeLaunchGate(
    OwnerPumpTag, std::shared_ptr<internal::OwnerPumpedControlState> state)
    : implementation_(std::make_unique<Implementation>(std::move(state))) {}
std::unique_ptr<AsyncCooperativeLaunchGate> AsyncCooperativeLaunchGate::create_owner_pumped(
    LaunchGateOptions options, OwnerPumpOptions bounds,
    std::optional<NativeEpochExtent> policy) noexcept {
    try {
        auto state = internal::OwnerPumpedControlState::create(std::move(options), bounds, policy);
        if (!state) return {};
        return std::unique_ptr<AsyncCooperativeLaunchGate>(
            new AsyncCooperativeLaunchGate(OwnerPumpTag{}, std::move(state)));
    } catch (...) {
        return {};
    }
}
AsyncCooperativeLaunchGate::CpuOwner::~CpuOwner() = default;
AsyncCooperativeLaunchGate::CpuOwner::CpuOwner(CpuOwner&&) noexcept = default;
AsyncCooperativeLaunchGate::CpuOwner::operator bool() const noexcept {
    return registration_ && registration_->state->same_process() &&
           !registration_->is_detached.load(std::memory_order_acquire);
}
bool AsyncCooperativeLaunchGate::CpuOwner::detach() noexcept {
    return registration_ && registration_->detach();
}
bool AsyncCooperativeLaunchGate::CpuOwner::begin_retirement() noexcept {
    return static_cast<bool>(*this) &&
           registration_->state->mailbox.retire_owner(registration_->token);
}
AsyncCooperativeLaunchGate::CpuExecutor::CpuExecutor(
    std::shared_ptr<internal::OwnerPumpedCpuRegistration> registration) noexcept
    : registration_(std::move(registration)) {}
AsyncCooperativeLaunchGate::CpuExecutor::CpuExecutor(CpuExecutor&&) noexcept = default;
AsyncCooperativeLaunchGate::CpuExecutor::~CpuExecutor() {
    if (registration_) registration_->lease.reset();
}
AsyncCooperativeLaunchGate::CpuProgress
AsyncCooperativeLaunchGate::CpuExecutor::progress() noexcept {
    if (!registration_) return CpuProgress::kRejected;
    auto& lease = registration_->lease;
    if (!lease) return CpuProgress::kRejected;
    const auto result = lease->progress();
    if (result == internal::ControlMailboxProgress::kCompleted ||
        result == internal::ControlMailboxProgress::kQuarantined)
        registration_->state->broadcast();
    return static_cast<CpuProgress>(result);
}
AsyncCooperativeLaunchGate::CpuWaitOutcome AsyncCooperativeLaunchGate::CpuExecutor::wait_until(
    std::chrono::steady_clock::time_point deadline) noexcept {
    if (!registration_) return {};
    auto& lease = registration_->lease;
    if (!lease) return {};
    const auto result = lease->wait_until(deadline);
    return {static_cast<CpuWaitStatus>(result.status), result.has_activity, result.has_reply,
            result.has_peer_closure};
}
AsyncCooperativeLaunchGate::CpuOwner AsyncCooperativeLaunchGate::register_cpu_owner() noexcept {
    CpuOwner owner;
    if (implementation_->pumped) owner.registration_ = implementation_->pumped->register_cpu();
    return owner;
}
std::optional<AsyncCooperativeLaunchGate::CpuExecutor> AsyncCooperativeLaunchGate::try_acquire_cpu(
    const CpuOwner& owner) noexcept {
    if (!implementation_->pumped || !owner ||
        owner.registration_->state != implementation_->pumped || owner.registration_->lease)
        return std::nullopt;
    auto lease = implementation_->pumped->mailbox.try_acquire(owner.registration_->token);
    if (!lease) return std::nullopt;
    owner.registration_->lease.emplace(std::move(*lease));
    return CpuExecutor{owner.registration_};
}
AsyncCooperativeLaunchGate::OwnerPumpSnapshot AsyncCooperativeLaunchGate::owner_pump_snapshot()
    const noexcept {
    return implementation_->pumped ? implementation_->pumped->snapshot() : OwnerPumpSnapshot{};
}
bool AsyncCooperativeLaunchGate::remote_outcome_uncertain() const noexcept {
    return implementation_->pumped ? implementation_->pumped->snapshot().is_quarantined
                                   : implementation_->gate.remote_outcome_uncertain();
}
void AsyncCooperativeLaunchGate::quarantine_owner_pump() noexcept {
    if (implementation_->pumped) {
        implementation_->pumped->contain();
        implementation_->pumped->broadcast();
    }
}

std::future<NativeEpochClaim> AsyncCooperativeLaunchGate::acquire_native_epoch(
    NativeEpochLaunchRequest request) noexcept {
    auto& state = *implementation_;
    if (state.pumped)
        return state.pumped->enqueue<NativeEpochClaim>(internal::RemoteNativeCommand{
            .operation = NativeEpochOperation::kAcquire, .settings = request});
    return state.enqueue<NativeEpochClaim>(
        [&state, request] { return state.gate.acquire_native_epoch(request); });
}
std::future<NativeEpochClaim> AsyncCooperativeLaunchGate::claim_native_epoch(
    NativeEpochGrant pending) noexcept {
    auto& state = *implementation_;
    if (state.pumped)
        return state.pumped->enqueue<NativeEpochClaim>(internal::RemoteNativeCommand{
            .operation = NativeEpochOperation::kClaim, .grant = pending});
    return state.enqueue<NativeEpochClaim>(
        [&state, pending] { return state.gate.claim_native_epoch(pending); });
}
std::future<std::optional<bool>> AsyncCooperativeLaunchGate::retreat_native_epoch(
    NativeEpochGrant grant) noexcept {
    auto& state = *implementation_;
    if (state.pumped)
        return state.pumped->enqueue<std::optional<bool>>(internal::RemoteNativeCommand{
            .operation = NativeEpochOperation::kRetreat, .grant = grant});
    return state.enqueue<std::optional<bool>>(
        [&state, grant] { return state.gate.retreat_native_epoch(grant); });
}
std::future<bool> AsyncCooperativeLaunchGate::heartbeat_native_epoch(
    NativeEpochGrant grant) noexcept {
    auto& state = *implementation_;
    if (state.pumped)
        return state.pumped->enqueue<bool>(internal::RemoteNativeCommand{
            .operation = NativeEpochOperation::kHeartbeat, .grant = grant});
    return state.enqueue<bool>(
        [&state, grant] { return state.gate.heartbeat_native_epoch(grant); });
}
std::future<bool> AsyncCooperativeLaunchGate::finish_native_epoch(NativeEpochGrant grant,
                                                                  std::uint32_t issued_calls,
                                                                  bool has_failed) noexcept {
    auto& state = *implementation_;
    if (state.pumped)
        return state.pumped->enqueue<bool>(internal::RemoteNativeCommand{
            .operation = has_failed ? NativeEpochOperation::kFail : NativeEpochOperation::kComplete,
            .grant = grant,
            .issued_calls = issued_calls});
    return state.enqueue<bool>([&state, grant, issued_calls, has_failed] {
        return state.gate.finish_native_epoch(grant, issued_calls, has_failed);
    });
}
std::future<bool> AsyncCooperativeLaunchGate::cancel_native_epoch(
    NativeEpochGrant pending) noexcept {
    auto& state = *implementation_;
    if (state.pumped)
        return state.pumped->enqueue<bool>(internal::RemoteNativeCommand{
            .operation = NativeEpochOperation::kCancelPending, .grant = pending});
    return state.enqueue<bool>(
        [&state, pending] { return state.gate.cancel_native_epoch(pending); });
}

std::shared_ptr<AsyncCooperativeLaunchGate::ProgressSignal>
AsyncCooperativeLaunchGate::subscribe_progress() noexcept {
    auto& state = *implementation_;
    if (state.pumped) return state.pumped->subscribe();
    try {
        const std::lock_guard lock(state.mutex);
        std::erase_if(state.subscribers,
                      [](const auto& subscriber) { return subscriber.expired(); });
        if (state.is_closing || state.subscribers.size() >= state.limit) {
            return {};
        }
        auto signal = std::make_shared<ProgressSignal>();
        state.subscribers.push_back(signal);
        return signal;
    } catch (...) {
        return {};
    }
}

std::future<std::optional<core::TaskId>> AsyncCooperativeLaunchGate::submit(
    CooperativeLaunchRequest request) noexcept {
    auto& state = *implementation_;
    if (state.pumped)
        return state.pumped->enqueue<std::optional<core::TaskId>>(
            internal::RemoteCooperativeCommand{
                TaskExecutionProtocolOperation::kSubmit, {}, request});
    return state.enqueue<std::optional<core::TaskId>>(
        [&state, request] { return state.gate.submit(request); });
}
std::future<CooperativeClaim> AsyncCooperativeLaunchGate::claim(
    core::TaskId task, bool should_measure_timing) noexcept {
    auto& state = *implementation_;
    if (state.pumped)
        return state.pumped->enqueue<CooperativeClaim>(
            internal::RemoteCooperativeCommand{TaskExecutionProtocolOperation::kExecutionClaim,
                                               {task, 0}},
            should_measure_timing);
    return state.enqueue_measured<CooperativeClaim>(
        [&state, task] { return state.gate.claim(task); }, should_measure_timing);
}
std::future<CooperativeSubmission> AsyncCooperativeLaunchGate::submit_and_claim(
    CooperativeLaunchRequest request) noexcept {
    if (implementation_->pumped)
        return implementation_->pumped->enqueue<CooperativeSubmission>(
            internal::RemoteCooperativeCommand{
                TaskExecutionProtocolOperation::kExecutionAcquire, {}, request});
    try {
        const auto enqueued = request.should_measure_timing ? timestamp_now() : 0;
        auto value = std::make_unique<Implementation::Submission>();
        value->settings = request;
        value->enqueued_nanoseconds = enqueued;
        return implementation_->enqueue_typed(std::move(value));
    } catch (...) {
        return {};
    }
}
std::future<std::optional<bool>> AsyncCooperativeLaunchGate::retreat_requested(
    TaskExecutionIdentity execution) noexcept {
    auto& state = *implementation_;
    if (state.pumped)
        return state.pumped->enqueue<std::optional<bool>>(internal::RemoteCooperativeCommand{
            TaskExecutionProtocolOperation::kRetreat, execution});
    return state.enqueue<std::optional<bool>>(
        [&state, execution] { return state.gate.retreat_requested(execution); });
}
std::future<bool> AsyncCooperativeLaunchGate::heartbeat(TaskExecutionIdentity execution) noexcept {
    auto& state = *implementation_;
    if (state.pumped)
        return state.pumped->enqueue<bool>(internal::RemoteCooperativeCommand{
            TaskExecutionProtocolOperation::kExecutionHeartbeat, execution});
    return state.enqueue<bool>([&state, execution] { return state.gate.heartbeat(execution); });
}
std::future<bool> AsyncCooperativeLaunchGate::suspend(TaskExecutionIdentity execution) noexcept {
    auto& state = *implementation_;
    if (state.pumped)
        return state.pumped->enqueue<bool>(internal::RemoteCooperativeCommand{
            TaskExecutionProtocolOperation::kSuspend, execution});
    return state.enqueue<bool>([&state, execution] { return state.gate.suspend(execution); });
}
std::future<bool> AsyncCooperativeLaunchGate::complete(TaskExecutionIdentity execution) noexcept {
    if (implementation_->pumped)
        return implementation_->pumped->enqueue<bool>(internal::RemoteCooperativeCommand{
            TaskExecutionProtocolOperation::kExecutionComplete, execution});
    try {
        auto value = std::make_unique<Implementation::Completion>();
        value->execution = execution;
        return implementation_->enqueue_typed(std::move(value));
    } catch (...) {
        return {};
    }
}
std::future<bool> AsyncCooperativeLaunchGate::fail(TaskExecutionIdentity execution) noexcept {
    auto& state = *implementation_;
    if (state.pumped)
        return state.pumped->enqueue<bool>(internal::RemoteCooperativeCommand{
            TaskExecutionProtocolOperation::kExecutionFail, execution});
    return state.enqueue<bool>([&state, execution] { return state.gate.fail(execution); });
}
std::future<bool> AsyncCooperativeLaunchGate::cancel_stopped(
    TaskExecutionIdentity execution) noexcept {
    auto& state = *implementation_;
    if (state.pumped)
        return state.pumped->enqueue<bool>(internal::RemoteCooperativeCommand{
            TaskExecutionProtocolOperation::kExecutionCancel, execution});
    return state.enqueue<bool>(
        [&state, execution] { return state.gate.cancel_stopped(execution); });
}
std::future<bool> AsyncCooperativeLaunchGate::cancel_pending(core::TaskId task) noexcept {
    auto& state = *implementation_;
    if (state.pumped)
        return state.pumped->enqueue<bool>(
            internal::RemoteCooperativeCommand{TaskExecutionProtocolOperation::kCancel, {task, 0}});
    return state.enqueue<bool>([&state, task] { return state.gate.cancel_pending(task); });
}
void AsyncCooperativeLaunchGate::shutdown() noexcept {
    auto& state = *implementation_;
    if (state.pumped) {
        state.pumped->close();
        return;
    }
    {
        const std::lock_guard lock(state.mutex);
        state.is_closing = true;
    }
    state.condition.notify_one();
    if (state.thread.joinable()) {
        state.thread.join();
    }
}

}  // namespace glimmer::control
