#include "internal/cuda_native_execution_lanes.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cerrno>
#include <condition_variable>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>
#include <pthread.h>
#include <unistd.h>

namespace glimmer::backend::cuda {
namespace {
thread_local std::shared_ptr<const void> lane_owner_identity;

using Phase = CudaNativeExecutionCallPhase;

// One publication carries all three fields, including the distinction between
// entering context preparation and actually invoking the trusted callable.
[[nodiscard]] std::uint64_t encode(CudaNativeExecutionCallObservation value) noexcept {
    return std::bit_cast<std::uint32_t>(static_cast<std::int32_t>(value.status)) |
           (static_cast<std::uint64_t>(value.phase) << 32U) |
           (static_cast<std::uint64_t>(value.was_invoked) << 40U);
}

[[nodiscard]] CudaNativeExecutionCallObservation decode(std::uint64_t value) noexcept {
    return {static_cast<Phase>((value >> 32U) & 0xffU),
            static_cast<CUresult>(std::bit_cast<std::int32_t>(static_cast<std::uint32_t>(value))),
            ((value >> 40U) & 1U) != 0};
}

[[nodiscard]] bool valid_call(const CudaNativeExecutionCall& call) noexcept {
    return call.sequence != 0 && call.device_ordinal >= 0 && call.context.id != 0 &&
           call.context.native && call.context.lifetime && call.arguments_lifetime && call.invoke;
}
}  // namespace

struct CudaNativeExecutionCallState {
    const pid_t process{::getpid()};
    const std::weak_ptr<const void> owner;
    CudaNativeExecutionCall call;
    std::atomic<std::uint64_t> observation{encode({})};

    CudaNativeExecutionCallState(std::weak_ptr<const void> owner_value,
                                 CudaNativeExecutionCall call_value)
        : owner(std::move(owner_value)), call(std::move(call_value)) {}

    void publish(Phase phase, CUresult status, bool was_invoked = false) noexcept {
        observation.store(encode({phase, status, was_invoked}), std::memory_order_release);
    }

    [[nodiscard]] CudaNativeExecutionCallObservation observe() const noexcept {
        return decode(observation.load(std::memory_order_acquire));
    }
};

CudaNativeExecutionCallObservation CudaNativeExecutionTicket::observe() const noexcept {
    if (!state_) return {Phase::kCancelled, CUDA_ERROR_INVALID_VALUE, false};
    // No inherited mutex or native state is accessed after fork.
    if (state_->process != ::getpid())
        return {Phase::kUncertain, CUDA_ERROR_INVALID_CONTEXT, false};
    return state_->observe();
}

struct CudaNativeExecutionLanes::Implementation {
    struct Lane {
        Implementation* hub{};
        CudaNativeExecutionLaneKey key;
        std::uint64_t next_sequence{1};
        bool has_sequence_space{true};
        std::vector<std::shared_ptr<CudaNativeExecutionCallState>> jobs;
        pthread_t worker{};
        bool is_joined{};
        bool has_join_failure{};
    };

    struct Device {
        int ordinal;
        bool is_executing{};
        CUresult failure{CUDA_SUCCESS};
    };

    const pid_t process{::getpid()};
    const std::weak_ptr<const void> owner_thread;
    const std::shared_ptr<const void> identity{std::make_shared<int>(0)};
    const CudaNativeExecutionLaneDispatch dispatch;
    const CudaNativeExecutionLaneLimits limits;
    mutable std::mutex mutex;
    std::condition_variable changed;
    std::vector<std::unique_ptr<Lane>> lanes;
    // Ordinals are retained for this hub's lifetime, including quarantines. A
    // separate quota bounds sequential requests using previously unseen devices.
    std::vector<Device> devices;
    std::size_t retained{};
    bool is_stopping{};
    std::atomic<bool> has_internal_failure{};

    Implementation(CudaNativeExecutionLaneDispatch dispatch_value,
                   CudaNativeExecutionLaneLimits limits_value)
        : owner_thread(lane_owner_identity), dispatch(dispatch_value), limits(limits_value) {
        lanes.reserve(limits.maximum_lanes);
        devices.reserve(limits.maximum_devices);
    }

    [[nodiscard]] bool owns_thread() const noexcept {
        return process == ::getpid() && lane_owner_identity &&
               owner_thread.lock() == lane_owner_identity;
    }

    [[nodiscard]] Lane* find_lane(CudaNativeExecutionLaneKey key) const noexcept {
        const auto found =
            std::ranges::find_if(lanes, [&](const auto& lane) { return lane->key == key; });
        return found == lanes.end() ? nullptr : found->get();
    }

    [[nodiscard]] Device* find_device(int ordinal) noexcept {
        const auto found = std::ranges::find_if(
            devices, [&](const auto& device) { return device.ordinal == ordinal; });
        return found == devices.end() ? nullptr : &*found;
    }

    [[nodiscard]] static std::shared_ptr<CudaNativeExecutionCallState> next_job(
        const Lane& lane) noexcept {
        const auto found = std::ranges::find_if(
            lane.jobs, [](const auto& job) { return job->observe().phase == Phase::kPending; });
        return found == lane.jobs.end() ? nullptr : *found;
    }

    // Submitted calls have a nonempty, sorted unique set. These observations
    // and all gate transitions occur under the one shared mailbox lock.
    [[nodiscard]] CUresult device_failure(const CudaNativeExecutionCall& call) noexcept {
        for (const auto ordinal : call.device_ordinals) {
            const auto* device = find_device(ordinal);
            if (!device) return CUDA_ERROR_UNKNOWN;
            if (device->failure != CUDA_SUCCESS) return device->failure;
        }
        return CUDA_SUCCESS;
    }

    [[nodiscard]] bool devices_available(const CudaNativeExecutionCall& call) noexcept {
        return std::ranges::all_of(call.device_ordinals, [&](int ordinal) {
            const auto* device = find_device(ordinal);
            return device && !device->is_executing;
        });
    }

    // Called only under the mailbox lock. A cancellation proves no callable or
    // context callback was entered; it does not prove GPU completion elsewhere.
    void cancel_pending(CUresult status) noexcept {
        for (const auto& lane : lanes) {
            for (const auto& job : lane->jobs) {
                if (job->observe().phase == Phase::kPending)
                    job->publish(Phase::kCancelled, status);
            }
        }
    }

    void cancel_failed_pending() noexcept {
        for (const auto& lane : lanes) {
            for (const auto& job : lane->jobs) {
                if (job->observe().phase != Phase::kPending) continue;
                if (const auto status = device_failure(job->call); status != CUDA_SUCCESS)
                    job->publish(Phase::kCancelled, status);
                // A never-entered superset has no native uncertainty on its
                // other devices. Cancellation cannot propagate quarantine.
            }
        }
    }

    [[nodiscard]] CUresult check_context(CUcontext expected) const noexcept {
        try {
            CUcontext current{};
            const auto status = dispatch.context_get_current(&current);
            if (status != CUDA_SUCCESS) return status;
            return current == expected ? CUDA_SUCCESS : CUDA_ERROR_INVALID_CONTEXT;
        } catch (...) {
            return CUDA_ERROR_UNKNOWN;
        }
    }

    [[nodiscard]] CUresult run_call(CudaNativeExecutionCallState& job) const noexcept {
        CUresult status = CUDA_ERROR_UNKNOWN;
        try {
            status = dispatch.context_set_current(job.call.context.native);
        } catch (...) {
            return CUDA_ERROR_UNKNOWN;
        }
        if (status != CUDA_SUCCESS) return status;
        status = check_context(job.call.context.native);
        if (status != CUDA_SUCCESS) return status;

        job.publish(Phase::kEntered, CUDA_ERROR_NOT_READY, true);
        try {
            status = job.call.invoke();
        } catch (...) {
            status = CUDA_ERROR_UNKNOWN;
        }
        // Always observe the post-call context, including after error/throw.
        // The first carrier failure remains authoritative; vendor status lives
        // independently in the caller-owned packet and is never rewritten here.
        const auto proof = check_context(job.call.context.native);
        return status == CUDA_SUCCESS ? proof : status;
    }

    void run(Lane& lane) {
        for (;;) {
            std::shared_ptr<CudaNativeExecutionCallState> job;
            {
                std::unique_lock lock(mutex);
                changed.wait(lock, [&] {
                    if (is_stopping || has_internal_failure.load(std::memory_order_acquire))
                        return true;
                    const auto pending = next_job(lane);
                    if (!pending) return false;
                    return device_failure(pending->call) != CUDA_SUCCESS ||
                           devices_available(pending->call);
                });
                if (is_stopping || has_internal_failure.load(std::memory_order_acquire)) return;
                job = next_job(lane);
                if (const auto failure = device_failure(job->call); failure != CUDA_SUCCESS) {
                    job->publish(Phase::kCancelled, failure);
                    continue;
                }
                // The wait predicate proved every gate free while retaining
                // this lock. Acquire the whole set, never a waiting subset.
                for (const auto ordinal : job->call.device_ordinals)
                    find_device(ordinal)->is_executing = true;
                job->publish(Phase::kEntered, CUDA_ERROR_NOT_READY);
            }

            // Neither the shared mailbox nor a per-device mutex is held across
            // SDK/user calls. Other devices and the owner remain able to progress.
            const auto status = run_call(*job);
            {
                std::lock_guard lock(mutex);
                for (const auto ordinal : job->call.device_ordinals) {
                    auto* device = find_device(ordinal);
                    device->is_executing = false;
                    if (status != CUDA_SUCCESS && device->failure == CUDA_SUCCESS)
                        device->failure = status;
                }
                const auto was_invoked = job->observe().was_invoked;
                if (status == CUDA_SUCCESS) {
                    job->publish(Phase::kReturned, CUDA_SUCCESS, was_invoked);
                } else {
                    job->publish(Phase::kUncertain, status, was_invoked);
                    cancel_failed_pending();
                }
            }
            changed.notify_all();
        }
    }

    static void* worker_entry(void* argument) noexcept {
        auto& lane = *static_cast<Lane*>(argument);
        try {
            lane.hub->run(lane);
        } catch (...) {
            // An internal synchronization failure is not evidence that any
            // entered packet can be released. Keep the entire hub fail-closed.
            lane.hub->has_internal_failure.store(true, std::memory_order_release);
            lane.hub->changed.notify_all();
        }
        return nullptr;
    }

    [[nodiscard]] CUresult admit(CudaNativeExecutionLaneKey key,
                                 const std::shared_ptr<CudaNativeExecutionCallState>& job) {
        // These owners precede the lock so even failed admission destroys no
        // captured callable/resource deleter while the mailbox is locked.
        std::unique_ptr<Lane> new_lane;
        std::lock_guard lock(mutex);
        if (is_stopping) return CUDA_ERROR_DEINITIALIZED;
        if (has_internal_failure.load(std::memory_order_acquire)) return CUDA_ERROR_UNKNOWN;
        auto* lane = find_lane(key);
        if ((lane && (!lane->has_sequence_space || job->call.sequence != lane->next_sequence)) ||
            (!lane && job->call.sequence != 1))
            return CUDA_ERROR_INVALID_VALUE;
        const auto new_devices = static_cast<std::size_t>(std::ranges::count_if(
            job->call.device_ordinals, [&](int ordinal) { return !find_device(ordinal); }));
        if (retained >= limits.maximum_jobs ||
            (lane && lane->jobs.size() >= limits.maximum_jobs_per_lane) ||
            (!lane && lanes.size() >= limits.maximum_lanes) ||
            devices.size() > limits.maximum_devices ||
            new_devices > limits.maximum_devices - devices.size())
            return CUDA_ERROR_OUT_OF_MEMORY;
        if (!lane) {
            new_lane = std::make_unique<Lane>();
            new_lane->hub = this;
            new_lane->key = key;
            new_lane->jobs.reserve(std::min(limits.maximum_jobs, limits.maximum_jobs_per_lane));
            lane = new_lane.get();
        }

        const auto previous_device_count = devices.size();
        for (const auto ordinal : job->call.device_ordinals)
            if (!find_device(ordinal)) devices.push_back({ordinal});
        lane->jobs.push_back(job);
        if (new_lane) {
            // The new worker can only acquire the mailbox after this atomic
            // publication has finished. All remaining container inserts have
            // reserved capacity and do not allocate.
            const auto result = ::pthread_create(&lane->worker, nullptr, worker_entry, lane);
            if (result != 0) {
                lane->jobs.pop_back();
                devices.resize(previous_device_count);
                return result == EAGAIN ? CUDA_ERROR_OUT_OF_MEMORY : CUDA_ERROR_UNKNOWN;
            }
            lanes.push_back(std::move(new_lane));
        }
        if (const auto failure = device_failure(job->call); failure != CUDA_SUCCESS)
            job->publish(Phase::kCancelled, failure);
        if (lane->next_sequence == std::numeric_limits<std::uint64_t>::max())
            lane->has_sequence_space = false;
        else
            ++lane->next_sequence;
        ++retained;
        changed.notify_all();
        return CUDA_SUCCESS;
    }
};

CudaNativeExecutionLanes::CudaNativeExecutionLanes(CudaNativeExecutionLaneDispatch dispatch,
                                                   CudaNativeExecutionLaneLimits limits) {
    if (!dispatch.context_set_current || !dispatch.context_get_current || !limits.maximum_lanes ||
        !limits.maximum_jobs || !limits.maximum_jobs_per_lane || !limits.maximum_devices)
        throw std::invalid_argument("invalid CUDA native execution lane configuration");
    if (!lane_owner_identity) lane_owner_identity = std::make_shared<int>(0);
    implementation_ = std::make_unique<Implementation>(dispatch, limits);
}

CudaNativeExecutionLanes::~CudaNativeExecutionLanes() {
    if (!implementation_->owns_thread()) {
        (void)implementation_.release();
        return;
    }
    const auto stop = request_stop();
    const auto joined = shutdown();
    if (stop != CUDA_SUCCESS || joined != CUDA_SUCCESS || retained_jobs() != 0 ||
        implementation_->has_internal_failure.load(std::memory_order_acquire)) {
        // A blocked callback/TLS destructor, unknown proof, or unreleased packet
        // retains the stable workers and their complete CPU/native ownership.
        // No destructor performs SDK cleanup, cancellation, or a blocking join.
        (void)implementation_.release();
    }
}

CudaNativeExecutionAcceptance CudaNativeExecutionLanes::submit(
    CudaNativeExecutionLaneKey key, CudaNativeExecutionCall call) noexcept {
    auto& state = *implementation_;
    if (!state.owns_thread()) return {CUDA_ERROR_INVALID_CONTEXT, {}};
    if (!key.peer || (key.thread[0] == 0 && key.thread[1] == 0) || !valid_call(call))
        return {CUDA_ERROR_INVALID_VALUE, {}};
    if (call.device_ordinals.size() > state.limits.maximum_devices)
        return {CUDA_ERROR_INVALID_VALUE, {}};
    try {
        if (call.device_ordinals.empty()) {
            call.device_ordinals.push_back(call.device_ordinal);
        } else {
            std::ranges::sort(call.device_ordinals);
            if (call.device_ordinals.front() < 0 ||
                std::ranges::adjacent_find(call.device_ordinals) != call.device_ordinals.end() ||
                !std::ranges::binary_search(call.device_ordinals, call.device_ordinal))
                return {CUDA_ERROR_INVALID_VALUE, {}};
        }
        auto job = std::make_shared<CudaNativeExecutionCallState>(state.identity, std::move(call));
        const auto status = state.admit(key, job);
        if (status != CUDA_SUCCESS) return {status, {}};
        CudaNativeExecutionTicket ticket;
        ticket.state_ = std::move(job);
        return {CUDA_SUCCESS, std::move(ticket)};
    } catch (const std::bad_alloc&) {
        return {CUDA_ERROR_OUT_OF_MEMORY, {}};
    } catch (...) {
        return {CUDA_ERROR_UNKNOWN, {}};
    }
}

CUresult CudaNativeExecutionLanes::release(const CudaNativeExecutionTicket& ticket) noexcept {
    auto& state = *implementation_;
    if (!state.owns_thread()) return CUDA_ERROR_INVALID_CONTEXT;
    if (!ticket.state_ || ticket.state_->process != state.process ||
        ticket.state_->owner.lock() != state.identity)
        return CUDA_ERROR_INVALID_VALUE;
    std::shared_ptr<CudaNativeExecutionCallState> retired_job;
    try {
        {
            std::lock_guard lock(state.mutex);
            for (const auto& lane : state.lanes) {
                const auto found = std::ranges::find(lane->jobs, ticket.state_);
                if (found == lane->jobs.end()) continue;
                if (state.has_internal_failure.load(std::memory_order_acquire))
                    return CUDA_ERROR_NOT_READY;
                const auto observed = ticket.state_->observe();
                if (observed.phase != Phase::kCancelled &&
                    !(observed.phase == Phase::kReturned && observed.status == CUDA_SUCCESS))
                    return CUDA_ERROR_NOT_READY;
                retired_job = std::move(*found);
                lane->jobs.erase(found);
                --state.retained;
                break;
            }
        }
        if (!retired_job) return CUDA_ERROR_INVALID_VALUE;
        // Clear all captured owners outside the lock, even while callers retain
        // copies of the final observation ticket. This is HOST release only.
        retired_job->call = {};
        return CUDA_SUCCESS;
    } catch (...) {
        state.has_internal_failure.store(true, std::memory_order_release);
        state.changed.notify_all();
        return CUDA_ERROR_UNKNOWN;
    }
}

CUresult CudaNativeExecutionLanes::request_stop() noexcept {
    auto& state = *implementation_;
    if (!state.owns_thread()) return CUDA_ERROR_INVALID_CONTEXT;
    try {
        {
            std::lock_guard lock(state.mutex);
            state.is_stopping = true;
            state.cancel_pending(CUDA_ERROR_DEINITIALIZED);
        }
        state.changed.notify_all();
        return CUDA_SUCCESS;
    } catch (...) {
        state.has_internal_failure.store(true, std::memory_order_release);
        state.changed.notify_all();
        return CUDA_ERROR_UNKNOWN;
    }
}

CUresult CudaNativeExecutionLanes::shutdown() noexcept {
    auto& state = *implementation_;
    if (!state.owns_thread()) return CUDA_ERROR_INVALID_CONTEXT;
    try {
        {
            std::lock_guard lock(state.mutex);
            if (!state.is_stopping) return CUDA_ERROR_NOT_READY;
        }
        // Linux tryjoin is the actual exit observation, including vendor TLS
        // destructors. A flag set at worker-body return would not establish exit
        // and a subsequent ordinary join could block the service owner forever.
        // Only the original owner accesses these handles; no lock covers tryjoin.
        bool has_live_worker = false;
        bool has_join_failure = false;
        for (const auto& lane : state.lanes) {
            if (lane->is_joined) continue;
            if (lane->has_join_failure) {
                has_join_failure = true;
                continue;
            }
            const auto result = ::pthread_tryjoin_np(lane->worker, nullptr);
            if (result == 0) {
                lane->is_joined = true;
            } else if (result == EBUSY) {
                has_live_worker = true;
            } else {
                lane->has_join_failure = true;
                has_join_failure = true;
            }
        }
        if (has_join_failure) return CUDA_ERROR_UNKNOWN;
        return has_live_worker ? CUDA_ERROR_NOT_READY : CUDA_SUCCESS;
    } catch (...) {
        state.has_internal_failure.store(true, std::memory_order_release);
        state.changed.notify_all();
        return CUDA_ERROR_UNKNOWN;
    }
}

std::size_t CudaNativeExecutionLanes::retained_jobs() const noexcept {
    auto& state = *implementation_;
    if (!state.owns_thread()) return 0;
    try {
        std::lock_guard lock(state.mutex);
        return state.retained;
    } catch (...) {
        state.has_internal_failure.store(true, std::memory_order_release);
        state.changed.notify_all();
        return 0;
    }
}

}  // namespace glimmer::backend::cuda
