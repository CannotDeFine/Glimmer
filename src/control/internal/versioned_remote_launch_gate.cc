#include "versioned_remote_launch_gate.h"

#include "glimmer/control/task_protocol.h"

#include <algorithm>
#include <condition_variable>
#include <exception>
#include <limits>
#include <list>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unistd.h>
#include <utility>

namespace glimmer::control::internal {
namespace {
using Clock = std::chrono::steady_clock;
constexpr std::size_t kMaxOwnedRequests = 65536;
constexpr unsigned int kUrgentBurst = 8;

class Timing final {
   public:
    explicit Timing(LaunchGateTiming* timing) : timing_(timing), started_(Clock::now()) {
        if (timing_) {
            *timing_ = {};
            timing_->remote = true;
        }
    }
    ~Timing() {
        if (timing_)
            timing_->elapsed_nanoseconds = static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - started_)
                    .count());
    }
    Timing(const Timing&) = delete;
    Timing& operator=(const Timing&) = delete;

   private:
    LaunchGateTiming* timing_;
    Clock::time_point started_;
};
void account(LaunchGateTiming* timing, const VersionedControlTiming& value, bool claim) {
    if (!timing) return;
    const auto max = std::numeric_limits<std::uint64_t>::max();
    timing->transport_nanoseconds = value.elapsed_nanoseconds > max - timing->transport_nanoseconds
                                        ? max
                                        : timing->transport_nanoseconds + value.elapsed_nanoseconds;
    if (timing->request_count != std::numeric_limits<std::uint32_t>::max()) ++timing->request_count;
    if (claim && timing->claim_poll_count != std::numeric_limits<std::uint32_t>::max())
        ++timing->claim_poll_count;
}
}  // namespace

struct VersionedRemoteLaunchGate::State {
    enum class Phase : std::uint8_t { kReserved, kPending, kRunning, kUncertain };
    struct Slot {
        core::TaskId task{};
        Phase phase{Phase::kReserved};
    };
    explicit State(LaunchGateOptions settings, VersionedControlTransport transport)
        : options(std::move(settings)),
          client(options.control_socket, transport),
          owner(::getpid()) {
        if (options.control_socket.empty() ||
            options.remote_acquire_timeout != std::chrono::milliseconds::zero())
            throw std::invalid_argument(
                "versioned interceptor requires indefinite remote admission");
        if (options.tenant_id.empty()) options.tenant_id = "default";
        if (options.tenant_weight == 0) options.tenant_weight = 1;
        if (options.remote_poll_interval <= std::chrono::milliseconds::zero())
            options.remote_poll_interval = std::chrono::milliseconds{1};
    }

    // FIFO within each lane, with terminal and heartbeat messages ahead of
    // queued admissions. The mutex is not held during transport I/O.
    struct Permit {
        Permit(State& value, bool important)
            : state(value),
              urgent(important),
              lock(state.mutex),
              uncaught(std::uncaught_exceptions()) {
            if (state.poisoned) return;
            auto& next = urgent ? state.urgent_next : state.normal_next;
            if (next == std::numeric_limits<std::uint64_t>::max()) {
                state.poisoned = true;
                state.changed.notify_all();
                return;
            }
            const auto ticket = next++;
            try {
                state.changed.wait(lock, [&] {
                    const bool normals_waiting = state.normal_next != state.normal_turn;
                    const bool urgents_waiting = state.urgent_next != state.urgent_turn;
                    return state.poisoned ||
                           (!state.busy &&
                            (urgent
                                 ? ticket == state.urgent_turn &&
                                       (!normals_waiting || state.urgent_streak < kUrgentBurst)
                                 : ticket == state.normal_turn &&
                                       (!urgents_waiting || state.urgent_streak >= kUrgentBurst)));
                });
            } catch (...) {
                state.poisoned = true;
                state.changed.notify_all();
                throw;
            }
            if (state.poisoned) return;
            state.busy = true;
            acquired = true;
        }
        ~Permit() {
            if (!acquired) return;
            if (!lock.owns_lock()) lock.lock();
            if (std::uncaught_exceptions() > uncaught) state.poisoned = true;
            state.busy = false;
            ++(urgent ? state.urgent_turn : state.normal_turn);
            state.urgent_streak = urgent ? std::min(kUrgentBurst, state.urgent_streak + 1) : 0;
            lock.unlock();
            state.changed.notify_all();
        }
        Permit(const Permit&) = delete;
        Permit& operator=(const Permit&) = delete;
        void unlock() {
            lock.unlock();
        }
        void relock() {
            lock.lock();
        }
        void poison() {
            state.poisoned = true;
            state.changed.notify_all();
        }
        State& state;
        bool urgent{};
        bool acquired{};
        std::unique_lock<std::mutex> lock;
        const int uncaught;
    };

    [[nodiscard]] std::list<Slot>::iterator find(core::TaskId id) {
        return std::find_if(slots.begin(), slots.end(),
                            [id](const Slot& slot) { return slot.task == id; });
    }
    [[nodiscard]] bool report(core::TaskId id, TaskProtocolOperation operation,
                              TaskProtocolState expected, LaunchGateTiming* timing) noexcept {
        if (!id) return false;
        try {
            const auto line = format_task_protocol_request(
                {.operation = operation, .admission = {}, .task_id = id});
            if (!line) return false;
            Permit permit(*this, true);
            if (!permit.acquired) return false;
            const auto slot = find(id);
            if (slot == slots.end() || slot->phase != Phase::kRunning) return false;
            permit.unlock();
            VersionedControlTiming measured;
            auto exchange = client.request_detailed(*line, timing ? &measured : nullptr);
            account(timing, measured, false);
            permit.relock();
            if (exchange.status != VersionedExchangeStatus::kResponse || !exchange.response) {
                slot->phase = Phase::kUncertain;
                permit.poison();
                return false;
            }
            const auto parsed = parse_task_protocol_response(*exchange.response);
            if (!parsed.parsed() || !parsed.response ||
                parsed.response->kind != TaskProtocolResponseKind::kState ||
                parsed.response->task_id != id || parsed.response->state != expected) {
                slot->phase = Phase::kUncertain;
                permit.poison();
                return false;
            }
            if (operation != TaskProtocolOperation::kHeartbeat) slots.erase(slot);
            return true;
        } catch (...) {
            std::scoped_lock lock(mutex);
            poisoned = true;
            changed.notify_all();
            return false;
        }
    }
    LaunchGateOptions options;
    VersionedTaskControlClient client;
    const pid_t owner;
    mutable std::mutex mutex;
    std::condition_variable changed;
    std::list<Slot> slots;
    std::uint64_t normal_next{}, normal_turn{}, urgent_next{}, urgent_turn{};
    unsigned int urgent_streak{};
    bool busy{}, poisoned{};
};

VersionedRemoteLaunchGate::VersionedRemoteLaunchGate(LaunchGateOptions options,
                                                     VersionedControlTransport transport)
    : state_(std::make_unique<State>(std::move(options), transport)) {}
VersionedRemoteLaunchGate::~VersionedRemoteLaunchGate() {
    // In a fork child a vanished parent thread may own either session mutex.
    if (state_ && ::getpid() != state_->owner) {
        [[maybe_unused]] const auto* inherited = state_.release();
    }
}

std::optional<core::TaskId> VersionedRemoteLaunchGate::acquire(LaunchGateTiming* timing) noexcept {
    const Timing timer(timing);
    if (::getpid() != state_->owner) return std::nullopt;
    State::Slot* owned = nullptr;
    try {
        {
            std::scoped_lock lock(state_->mutex);
            if (state_->poisoned || state_->slots.size() >= kMaxOwnedRequests) return std::nullopt;
            state_->slots.emplace_back();
            owned = &state_->slots.back();
        }
        const auto discard = [&] {
            state_->slots.remove_if([owned](const State::Slot& slot) { return &slot == owned; });
        };
        TaskProtocolRequest request{.operation = TaskProtocolOperation::kAcquire,
                                    .admission = {.tenant_id = state_->options.tenant_id,
                                                  .memory_bytes = 1,
                                                  .weight = state_->options.tenant_weight,
                                                  .work_units = 1,
                                                  .priority = state_->options.task_priority},
                                    .task_id = 0};
        bool is_initial = true;
        for (;;) {
            const auto line = format_task_protocol_request(request);
            if (!line) {
                std::scoped_lock lock(state_->mutex);
                if (is_initial)
                    discard();
                else
                    state_->poisoned = true;
                state_->changed.notify_all();
                return std::nullopt;
            }
            {
                State::Permit permit(*state_, false);
                if (!permit.acquired) return std::nullopt;
                permit.unlock();
                VersionedControlTiming measured;
                auto exchange =
                    state_->client.request_detailed(*line, timing ? &measured : nullptr);
                account(timing, measured, request.operation == TaskProtocolOperation::kClaim);
                permit.relock();
                if (exchange.status != VersionedExchangeStatus::kResponse || !exchange.response) {
                    if (is_initial && exchange.status == VersionedExchangeStatus::kNotPublished)
                        discard();
                    else {
                        owned->phase = State::Phase::kUncertain;
                        permit.poison();
                    }
                    return std::nullopt;
                }
                const auto parsed = parse_task_protocol_response(*exchange.response);
                if (!parsed.parsed() || !parsed.response) {
                    owned->phase = State::Phase::kUncertain;
                    permit.poison();
                    return std::nullopt;
                }
                const auto& response = *parsed.response;
                if (is_initial && request.operation == TaskProtocolOperation::kAcquire &&
                    response.kind == TaskProtocolResponseKind::kError &&
                    (response.error == TaskProtocolErrorCode::kInvalidRequest ||
                     response.error == TaskProtocolErrorCode::kUnsupportedVersion)) {
                    request.operation = TaskProtocolOperation::kSubmit;
                    continue;
                }
                if (response.kind == TaskProtocolResponseKind::kError && is_initial) {
                    if (response.error == TaskProtocolErrorCode::kQuotaExceeded ||
                        response.error == TaskProtocolErrorCode::kQueueFull ||
                        response.error == TaskProtocolErrorCode::kInvalidRequest ||
                        response.error == TaskProtocolErrorCode::kUnsupportedVersion)
                        discard();
                    else {
                        owned->phase = State::Phase::kUncertain;
                        permit.poison();
                    }
                    return std::nullopt;
                }
                const auto existing = state_->find(response.task_id);
                if (response.kind == TaskProtocolResponseKind::kLease && response.task_id != 0 &&
                    (is_initial || response.task_id == owned->task) &&
                    response.tenant_id == state_->options.tenant_id && response.memory_bytes == 1 &&
                    response.weight == state_->options.tenant_weight && response.work_units == 1 &&
                    (is_initial ? existing == state_->slots.end()
                                : existing != state_->slots.end() && &*existing == owned)) {
                    owned->task = response.task_id;
                    owned->phase = State::Phase::kRunning;
                    return owned->task;
                }
                if (is_initial && response.kind == TaskProtocolResponseKind::kAccepted &&
                    response.task_id != 0 &&
                    state_->find(response.task_id) == state_->slots.end()) {
                    owned->task = response.task_id;
                    owned->phase = State::Phase::kPending;
                    request = {.operation = TaskProtocolOperation::kClaim,
                               .admission = {},
                               .task_id = owned->task};
                    is_initial = false;
                } else if (is_initial || response.kind != TaskProtocolResponseKind::kEmpty) {
                    owned->phase = State::Phase::kUncertain;
                    permit.poison();
                    return std::nullopt;
                }
            }
            std::this_thread::sleep_for(state_->options.remote_poll_interval);
        }
    } catch (...) {
        std::scoped_lock lock(state_->mutex);
        state_->poisoned = true;
        state_->changed.notify_all();
        return std::nullopt;
    }
}

bool VersionedRemoteLaunchGate::heartbeat(core::TaskId task) noexcept {
    if (::getpid() != state_->owner) return false;
    return state_->report(task, TaskProtocolOperation::kHeartbeat, TaskProtocolState::kRunning,
                          nullptr);
}
bool VersionedRemoteLaunchGate::complete(core::TaskId task, LaunchGateTiming* timing) noexcept {
    const Timing timer(timing);
    if (::getpid() != state_->owner) return false;
    return state_->report(task, TaskProtocolOperation::kComplete, TaskProtocolState::kCompleted,
                          timing);
}
bool VersionedRemoteLaunchGate::fail(core::TaskId task, LaunchGateTiming* timing) noexcept {
    const Timing timer(timing);
    if (::getpid() != state_->owner) return false;
    return state_->report(task, TaskProtocolOperation::kFail, TaskProtocolState::kFailed, timing);
}
bool VersionedRemoteLaunchGate::quarantined() const noexcept {
    if (::getpid() != state_->owner) return true;
    try {
        std::scoped_lock lock(state_->mutex);
        return state_->poisoned;
    } catch (...) {
        return true;
    }
}
std::optional<std::vector<VersionedRemoteLaunchGate::RetainedTask>>
VersionedRemoteLaunchGate::retained() const noexcept {
    if (::getpid() != state_->owner) return std::nullopt;
    try {
        std::scoped_lock lock(state_->mutex);
        if (!state_->poisoned) return std::nullopt;
        std::vector<RetainedTask> result;
        result.reserve(state_->slots.size());
        for (const auto& slot : state_->slots) {
            const auto phase = slot.task == 0 ? RetainedPhase::kUnknownIdentity
                               : slot.phase == State::Phase::kPending ? RetainedPhase::kPending
                               : slot.phase == State::Phase::kRunning ? RetainedPhase::kRunning
                                                                      : RetainedPhase::kUncertain;
            result.push_back({slot.task, phase});
        }
        return result;
    } catch (...) {
        return std::nullopt;
    }
}
#if defined(GLIMMER_ENABLE_TEST_HOOKS)
std::optional<VersionedRemoteLaunchGate::TestLanes> VersionedRemoteLaunchGate::test_lanes()
    const noexcept {
    if (::getpid() != state_->owner) return std::nullopt;
    try {
        std::scoped_lock lock(state_->mutex);
        return TestLanes{state_->normal_next, state_->normal_turn, state_->urgent_next,
                         state_->urgent_turn};
    } catch (...) {
        return std::nullopt;
    }
}
#endif

}  // namespace glimmer::control::internal
