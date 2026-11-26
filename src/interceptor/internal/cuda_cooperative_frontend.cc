#include "cuda_cooperative_frontend.h"

#include "diagnostics.h"
#include "native_execution_heartbeat.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <limits>
#include <mutex>
#include <new>
#include <thread>
#include <utility>
#include <vector>

namespace glimmer::interceptor {
namespace {
constexpr std::size_t kNativeEventWaitLimit = 4096;
// Private registration authority, never a CUDA handle or scheduler identity.
// Process-lifetime tags do not wrap, including after stream/caller retirement.
std::uint64_t next_epoch_registration() noexcept {
    static std::atomic<std::uint64_t> sequence{0};
    auto current = sequence.load(std::memory_order_relaxed);
    while (current != std::numeric_limits<std::uint64_t>::max()) {
        if (sequence.compare_exchange_weak(current, current + 1, std::memory_order_relaxed))
            return current + 1;
    }
    return 0;
}
// Caller-local diagnostics only; disabled timing performs no clock reads.
class NativeTiming final {
   public:
    explicit NativeTiming(bool is_timing_enabled)
        : is_enabled(is_timing_enabled),
          previous_(is_timing_enabled ? std::chrono::steady_clock::now()
                                      : std::chrono::steady_clock::time_point{}) {
        if (is_enabled) {
            observation_.started_nanoseconds = timestamp(previous_);
        }
    }

    void finish_phase(NativeTimingPhase phase) noexcept {
        if (!is_enabled) {
            return;
        }
        const auto now = std::chrono::steady_clock::now();
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(now - previous_);
        observation_.phase_nanoseconds[static_cast<std::size_t>(phase)] =
            elapsed.count() > 0 ? static_cast<std::uint64_t>(elapsed.count()) : 0;
        previous_ = now;
        observation_.ended_nanoseconds = timestamp(now);
    }

    void retry(bool is_admission) noexcept {
        if (!is_enabled) {
            return;
        }
        auto& count = is_admission ? observation_.admission_retries : observation_.physical_retries;
        if (count != std::numeric_limits<std::uint64_t>::max()) {
            ++count;
        }
    }

    template <typename Function>
    auto measure_admission(NativeAdmissionPart part, Function operation) {
        if (!is_enabled) {
            return operation();
        }
        const auto started = std::chrono::steady_clock::now();
        const auto result = operation();
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started);
        const auto duration = elapsed.count() > 0 ? static_cast<std::uint64_t>(elapsed.count()) : 0;
        auto& total = observation_.admission_nanoseconds[static_cast<std::size_t>(part)];
        total += std::min(duration, std::numeric_limits<std::uint64_t>::max() - total);
        return result;
    }

    void report(bool is_execution, const std::optional<control::TaskExecutionIdentity>& execution,
                CUresult original_status) noexcept {
        if (!is_enabled) {
            return;
        }
        if (execution) {
            observation_.task_id = execution->task_id;
            observation_.generation = execution->generation;
        }
        observation_.is_execution = is_execution;
        observation_.original_status = static_cast<std::uint32_t>(original_status);
        report_native_timing_diagnostic(observation_);
    }

   private:
    static std::uint64_t timestamp(std::chrono::steady_clock::time_point point) noexcept {
        const auto duration =
            std::chrono::duration_cast<std::chrono::nanoseconds>(point.time_since_epoch());
        return duration.count() > 0 ? static_cast<std::uint64_t>(duration.count()) : 0;
    }
    const bool is_enabled;
    std::chrono::steady_clock::time_point previous_;
    NativeTimingObservation observation_;
};

struct CallerThreadIdentity {
    std::atomic<bool> is_active{true};
    const std::uint64_t registration{next_epoch_registration()};
};
// Thread exit does not release queued arguments. It only enables reclamation
// after the owner separately proves that the proxy's logical work completed.
std::shared_ptr<CallerThreadIdentity> caller_thread_lifetime() {
    struct ThreadOwner {
        const std::shared_ptr<CallerThreadIdentity> identity{
            std::make_shared<CallerThreadIdentity>()};
        ~ThreadOwner() {
            identity->is_active.store(false, std::memory_order_release);
        }
    };
    thread_local const ThreadOwner owner;
    return owner.identity;
}

std::uint32_t relative_priority(int priority) noexcept {
    if (priority >= 0) {
        return 0;
    }
    return static_cast<std::uint32_t>(-static_cast<std::int64_t>(priority));
}
}  // namespace

struct CudaCooperativeFrontend::Implementation {
    struct ImportArena {
        std::mutex mutex;
        std::vector<CUevent> events;
        std::vector<bool> occupied;
    };
    struct ImportLease {
        std::shared_ptr<ImportArena> arena;
        std::size_t index{};
        bool was_recorded{};
        bool has_queue_owner{};
        ~ImportLease() {
            // Queue owners release only after prefix proof. A recorded but
            // unimported event stays quarantined until checked native teardown.
            if (!was_recorded || has_queue_owner) {
                const std::lock_guard lock(arena->mutex);
                arena->occupied[index] = false;
            }
        }
        [[nodiscard]] CUevent event() const noexcept {
            return arena->events[index];
        }
    };
    struct StreamBinding {
        CUstream original{};
        std::shared_ptr<CallerThreadIdentity> thread;
        OrderedStreamToken logical;
        std::uint32_t priority{};
        std::optional<std::size_t> proxy;
        std::uint64_t registration{};
        OrderedStreamKind kind{OrderedStreamKind::kNonBlocking};
    };
    struct EpochEntry {
        StreamBinding binding;
        std::shared_ptr<CallerThreadIdentity> caller;
        CudaNativeBoundaryToken boundary;
        // Published under mutex; immutable thereafter. No caller destroys it.
        std::shared_ptr<CudaNativeExecutionEpoch> owner;
        std::atomic<bool> is_close_requested{false};
        std::atomic<bool> is_done{false};
        std::atomic<bool> is_owner_joining{false};
        std::atomic<unsigned int> active_owner_calls{};
        bool can_recover{};           // frontend mutex; exclusive epoch failure only
        bool has_recovery_attempt{};  // frontend mutex; never retry effects
    };
    struct EpochCallerUse {
        EpochEntry& entry;
        bool has_contribution{};
        explicit EpochCallerUse(EpochEntry& value) : entry(value) {
            if (entry.is_owner_joining.load()) return;
            ++entry.active_owner_calls;
            if (entry.is_owner_joining.load())
                --entry.active_owner_calls;
            else
                has_contribution = true;
        }
        ~EpochCallerUse() {
            if (has_contribution) --entry.active_owner_calls;
        }
        EpochCallerUse(const EpochCallerUse&) = delete;
        EpochCallerUse& operator=(const EpochCallerUse&) = delete;
    };
    struct EventCallGate {
        std::mutex mutex;
        std::condition_variable condition;
        std::thread::id caller;
        bool is_busy{};
    };
    struct EventCallPermit {
        const std::shared_ptr<EventCallGate> gate;
        CUresult result{CUDA_ERROR_INVALID_VALUE};
        explicit EventCallPermit(std::shared_ptr<EventCallGate> value) : gate(std::move(value)) {
            if (!gate) {
                return;
            }
            std::unique_lock lock(gate->mutex);
            if (gate->is_busy && gate->caller == std::this_thread::get_id()) {
                return;
            }
            gate->condition.wait(lock, [&] { return !gate->is_busy; });
            gate->is_busy = true;
            gate->caller = std::this_thread::get_id();
            result = CUDA_SUCCESS;
        }
        ~EventCallPermit() {
            if (result == CUDA_SUCCESS) {
                {
                    const std::lock_guard lock(gate->mutex);
                    gate->is_busy = false;
                    gate->caller = {};
                }
                gate->condition.notify_all();
            }
        }
        EventCallPermit(const EventCallPermit&) = delete;
        EventCallPermit& operator=(const EventCallPermit&) = delete;
    };
    struct EventBinding {
        CUevent original{};
        CudaQueueEventToken logical;
        std::shared_ptr<EventCallGate> calls;
    };
    struct Prefix {
        Implementation& owner;
        std::shared_ptr<ImportLease> lease;
        CUstream physical{};
        OrderedStreamKind kind{};
        std::uint32_t priority{};
        std::optional<std::size_t> proxy;
        bool has_queue_owner{};
        explicit Prefix(Implementation& value) : owner(value) {}
        ~Prefix() {
            if (proxy && !has_queue_owner) {
                const std::lock_guard lock(owner.mutex);
                owner.occupied_proxies[*proxy] = false;
            }
        }
        Prefix(const Prefix&) = delete;
        Prefix& operator=(const Prefix&) = delete;
    };

    CUcontext context;
    CudaFrontendDispatch dispatch;
    std::shared_ptr<control::AsyncCooperativeLaunchGate> gate;
    CudaFrontendOptions options;
    std::shared_ptr<CudaCooperativePool> pool;
    std::shared_ptr<CudaAdmissionCoordinator> coordinator;
    std::unique_ptr<CudaQueueWorker> worker;
    // Lazy CPU-only owner, accessed inside the exclusive native transaction.
    std::unique_ptr<NativeExecutionHeartbeat> native_heartbeat;
    const std::shared_ptr<void> resource_lifetime{std::make_shared<int>(0)};
    const std::shared_ptr<ImportArena> imports{std::make_shared<ImportArena>()};
    std::vector<CUstream> thread_streams;
    CUevent pause_fence{};
    std::vector<bool> occupied_proxies;
    // Owner-thread metadata. Access only inside an accepted worker command.
    std::vector<StreamBinding> streams;
    std::vector<EventBinding> events;
    // This lock never spans native CUDA, worker commands or control transport.
    std::mutex mutex;
    std::condition_variable condition;
    std::size_t active_calls{};
    CUresult status{CUDA_ERROR_NOT_INITIALIZED};
    bool is_transaction{};
    bool is_pause_finishing{};
    bool is_closing{};
    bool has_ambiguous_cleanup{};
    CudaWorkerCheckpoint checkpoint;
    const std::uint64_t epoch_context_registration;
    std::shared_ptr<EpochEntry> epoch;
    std::thread epoch_watcher;
    bool is_epoch_watcher_closing{};

    Implementation(CUcontext context_value, CudaFrontendDispatch dispatch_value,
                   std::shared_ptr<control::AsyncCooperativeLaunchGate> gate_value,
                   CudaFrontendOptions options_value)
        : context(context_value),
          dispatch(std::move(dispatch_value)),
          gate(std::move(gate_value)),
          options(std::move(options_value)),
          epoch_context_registration(options.native_epoch_policy ? next_epoch_registration() : 0) {}

    // Closing/retrieval never holds the metadata mutex across owner methods.
    CUresult close_epoch(bool should_wait) {
        if (!options.native_epoch_policy) return CUDA_SUCCESS;
        for (;;) {
            std::shared_ptr<EpochEntry> entry;
            std::shared_ptr<CudaNativeExecutionEpoch> execution;
            {
                const std::lock_guard lock(mutex);
                if (status != CUDA_SUCCESS) return status;
                entry = epoch;
                if (!entry || entry->is_done.load()) return CUDA_SUCCESS;
                entry->is_close_requested.store(true);
                execution = entry->owner;
            }
            if (execution) execution->close();
            condition.notify_all();
            if (!should_wait) return CUDA_ERROR_NOT_READY;
            std::unique_lock lock(mutex);
            condition.wait_for(lock, options.worker.poll_interval,
                               [&] { return entry->is_done.load() || status != CUDA_SUCCESS; });
        }
    }
    void watch_epochs() noexcept {
        try {
            for (;;) {
                std::shared_ptr<EpochEntry> entry;
                std::shared_ptr<CudaNativeExecutionEpoch> execution;
                {
                    std::unique_lock lock(mutex);
                    condition.wait(lock, [&] {
                        return is_epoch_watcher_closing ||
                               (epoch && !epoch->is_done.load() && epoch->owner);
                    });
                    if (is_epoch_watcher_closing &&
                        (!epoch || epoch->is_done.load() || !epoch->owner))
                        return;
                    entry = epoch;
                    execution = entry->owner;
                }
                CudaNativeEpochSnapshot stopped;
                for (;;) {
                    bool is_stopping = false;
                    {
                        const std::lock_guard lock(mutex);
                        is_stopping = is_epoch_watcher_closing;
                    }
                    if (is_stopping)
                        join_epoch_owner(*entry, *execution);
                    else if (entry->is_close_requested.load() || !entry->caller->is_active.load())
                        execution->close();
                    stopped = execution->snapshot();
                    if (stopped.phase == CudaNativeEpochPhase::kComplete ||
                        stopped.phase == CudaNativeEpochPhase::kRejected ||
                        stopped.phase == CudaNativeEpochPhase::kRetained)
                        break;
                    std::this_thread::sleep_for(options.worker.poll_interval);
                }
                join_epoch_owner(*entry, *execution);  // joins event cleanup before any thaw
                stopped = execution->snapshot();
                const bool unused_failure = stopped.phase == CudaNativeEpochPhase::kComplete &&
                                            !stopped.execution &&
                                            stopped.status == CUDA_ERROR_OUT_OF_MEMORY &&
                                            stopped.created_events == stopped.destroyed_events;
                CUresult result =
                    stopped.phase == CudaNativeEpochPhase::kRetained
                        ? (stopped.status == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : stopped.status)
                    : unused_failure ? CUDA_SUCCESS
                                     : stopped.status;
                {
                    const std::lock_guard lock(mutex);
                    if (status != CUDA_SUCCESS) result = status;
                }
                if (result == CUDA_SUCCESS) {
                    // All originals and private proof resources already stopped.
                    // Publish logical order only; this marker supplies no grant.
                    result = worker->finish_native(entry->boundary, [&](CudaOperationQueue& queue) {
                        if (queue.pending_count() != 0) return CUDA_ERROR_UNKNOWN;
                        if (!stopped.execution || stopped.execution->issued_calls == 0)
                            return CUDA_SUCCESS;
                        return queue.submit(entry->binding.logical, CudaQueuedMarker{})
                                   ? CUDA_SUCCESS
                                   : CUDA_ERROR_OUT_OF_MEMORY;
                    });
                }
                {
                    const std::lock_guard lock(mutex);
                    if (result != CUDA_SUCCESS) {
                        entry->can_recover =
                            status == CUDA_SUCCESS &&
                            stopped.phase == CudaNativeEpochPhase::kRetained && stopped.execution &&
                            stopped.execution->phase == NativeEpochPhase::kStopping &&
                            stopped.execution->has_ownership;
                        status = result;
                    }
                    entry->is_done.store(true);
                    is_transaction = false;
                }
                condition.notify_all();
            }
        } catch (...) {
            fail_closed(CUDA_ERROR_UNKNOWN);
        }
    }
    void join_epoch_owner(EpochEntry& entry, CudaNativeExecutionEpoch& execution) {
        // A physical/control failure can become terminal during another
        // original callback. Exclude new caller methods, then wait for the
        // admitted caller to publish return before externally serialized join.
        // No metadata lock spans original effects or this CPU contribution.
        entry.is_owner_joining.store(true);
        while (entry.active_owner_calls.load() != 0)
            std::this_thread::sleep_for(options.worker.poll_interval);
        execution.shutdown();
    }
    void stop_epoch_watcher() noexcept {
        if (!epoch_watcher.joinable()) return;
        {
            const std::lock_guard lock(mutex);
            is_epoch_watcher_closing = true;
        }
        condition.notify_all();
        epoch_watcher.join();
    }
    // Exact drained-pause reads only. Never wait for physical epoch completion
    // or retry an executed effect. Completion/next-pause races are bounded reads.
    CUresult query_command(const CudaQueueWorker::Command& operation) {
        CUcontext actual{};
        const auto binding = dispatch.worker.queue.context_get_current(&actual);
        if (binding != CUDA_SUCCESS || actual != context)
            return binding == CUDA_SUCCESS ? CUDA_ERROR_INVALID_CONTEXT : binding;
        for (unsigned int attempt = 0; attempt < 4; ++attempt) {
            std::shared_ptr<EpochEntry> entry;
            {
                const std::lock_guard lock(mutex);
                if (status != CUDA_SUCCESS || is_closing)
                    return is_closing ? CUDA_ERROR_DEINITIALIZED : status;
                entry = epoch;
            }
            bool has_run = false;
            const auto read = [&](CudaOperationQueue& queue) {
                has_run = true;
                return operation(queue);
            };
            const auto result = entry && !entry->is_done.load()
                                    ? worker->invoke_drained_paused(entry->boundary, read)
                                    : worker->invoke(read);
            if (has_run) return result;
            if (result != CUDA_ERROR_INVALID_VALUE && result != CUDA_ERROR_NOT_READY) return result;
        }
        return CUDA_ERROR_NOT_READY;
    }

    struct Permit {
        Implementation& owner;
        CUresult result;
        inline static thread_local Permit* current{};
        Permit* previous{};
        bool has_contribution{};
        explicit Permit(Implementation& value) : owner(value), previous(current) {
            const auto closed = owner.close_epoch(true);
            if (closed != CUDA_SUCCESS) {
                result = closed;
                return;
            }
            for (auto* parent = previous; parent; parent = parent->previous) {
                if (&parent->owner == &owner && parent->result == CUDA_SUCCESS) {
                    // An already admitted caller must finish its queue
                    // publication even after a native pause starts exclusion.
                    const std::lock_guard lock(owner.mutex);
                    result = owner.is_closing ? CUDA_ERROR_DEINITIALIZED : owner.status;
                    if (result == CUDA_SUCCESS) {
                        current = this;
                    }
                    return;
                }
            }
            std::unique_lock lock(owner.mutex);
            owner.condition.wait(lock, [&] { return !owner.is_transaction; });
            result = owner.is_closing ? CUDA_ERROR_DEINITIALIZED : owner.status;
            if (result == CUDA_SUCCESS) {
                ++owner.active_calls;
                has_contribution = true;
                current = this;
            }
        }
        ~Permit() {
            if (result == CUDA_SUCCESS) {
                current = previous;
            }
            if (has_contribution) {
                {
                    const std::lock_guard lock(owner.mutex);
                    --owner.active_calls;
                }
                owner.condition.notify_all();
            }
        }
        Permit(const Permit&) = delete;
        Permit& operator=(const Permit&) = delete;
        static bool is_admitted(const Implementation& owner) noexcept {
            for (auto* parent = current; parent; parent = parent->previous) {
                if (&parent->owner == &owner && parent->result == CUDA_SUCCESS) {
                    return true;
                }
            }
            return false;
        }
    };

    CUresult try_pause(const NativeOperation& operation) {
        const auto closed = close_epoch(false);
        if (closed != CUDA_SUCCESS) return closed;
        {
            std::unique_lock lock(mutex);
            if (status != CUDA_SUCCESS || is_closing) {
                return is_closing ? CUDA_ERROR_DEINITIALIZED : status;
            }
            if (is_transaction) {
                return CUDA_ERROR_NOT_READY;
            }
            // Do not acknowledge a worker pause while an admitted caller still
            // needs that worker to publish an already validated SDK effect.
            is_transaction = true;
            condition.wait(lock, [&] { return active_calls == 0 || status != CUDA_SUCCESS; });
            if (status != CUDA_SUCCESS) {
                is_transaction = false;
                return status;
            }
        }
        const auto result = operation();
        if (result != CUDA_SUCCESS) {
            {
                const std::lock_guard lock(mutex);
                is_transaction = false;
            }
            condition.notify_all();
        }
        return result;
    }

    CUresult command(CudaQueueWorker::Command operation) {
        const auto closed = close_epoch(true);
        if (closed != CUDA_SUCCESS) return closed;
        for (;;) {
            bool has_run = false;
            CUresult result = CUDA_ERROR_UNKNOWN;
            {
                Permit permit(*this);
                if (permit.result != CUDA_SUCCESS) {
                    return permit.result;
                }
                CUcontext caller = nullptr;
                result = dispatch.worker.queue.context_get_current(&caller);
                if (result != CUDA_SUCCESS || caller != context) {
                    return result == CUDA_SUCCESS ? CUDA_ERROR_INVALID_CONTEXT : result;
                }
                result = worker->invoke([&](CudaOperationQueue& queue) {
                    has_run = true;
                    return operation(queue);
                });
            }
            if (has_run || result != CUDA_ERROR_NOT_READY || worker->status() != CUDA_SUCCESS) {
                return result;
            }
            // A rejected command has not consumed its captures. Leave the
            // permit before waiting so the native transaction can proceed.
            std::this_thread::sleep_for(std::chrono::milliseconds{1});
        }
    }

    std::shared_ptr<ImportLease> take_event() {
        std::size_t index = imports->events.size();
        {
            const std::lock_guard lock(imports->mutex);
            for (std::size_t candidate = 0; candidate < imports->occupied.size(); ++candidate) {
                if (!imports->occupied[candidate]) {
                    index = candidate;
                    imports->occupied[index] = true;
                    break;
                }
            }
        }
        if (index == imports->events.size()) {
            return {};
        }
        try {
            auto lease = std::make_shared<ImportLease>();
            lease->arena = imports;
            lease->index = index;
            return lease;
        } catch (...) {
            const std::lock_guard lock(imports->mutex);
            imports->occupied[index] = false;
            throw;
        }
    }

    CUresult stream_binding(CUstream stream, bool per_thread, StreamBinding& output) {
        const bool is_thread_stream =
            stream == CU_STREAM_PER_THREAD || (stream == nullptr && per_thread);
        const bool is_legacy_stream =
            stream == CU_STREAM_LEGACY || (stream == nullptr && !per_thread);
        const CUstream original = is_thread_stream   ? CU_STREAM_PER_THREAD
                                  : is_legacy_stream ? CU_STREAM_LEGACY
                                                     : stream;
        const auto thread = is_thread_stream ? caller_thread_lifetime() : nullptr;
        bool exists = false;
        auto result = command([&](CudaOperationQueue& queue) {
            std::erase_if(streams, [&](const auto& binding) {
                if (!binding.thread || binding.thread->is_active.load(std::memory_order_acquire)) {
                    return false;
                }
                const auto tail = queue.stream_snapshot(binding.logical);
                if (!tail || (*tail && queue.query(*tail) != CUDA_SUCCESS) ||
                    !queue.close_stream(binding.logical)) {
                    return false;
                }
                const std::lock_guard lock(mutex);
                occupied_proxies[*binding.proxy] = false;
                return true;
            });
            for (const auto& binding : streams) {
                if (binding.original == original && binding.thread == thread) {
                    output = binding;
                    exists = true;
                    break;
                }
            }
            return CUDA_SUCCESS;
        });
        if (result != CUDA_SUCCESS || exists) {
            return result;
        }
        Prefix prefix(*this);
        // Preserve a successfully recorded prefix through pause rejection. It
        // is never rerecorded or returned to the arena before physical drain.
        bool has_prepared = false;
        for (;;) {
            bool has_run = false;
            {
                Permit permit(*this);
                if (permit.result != CUDA_SUCCESS) {
                    return permit.result;
                }
                if (!has_prepared) {
                    prefix.physical = original;
                    prefix.kind = is_legacy_stream   ? OrderedStreamKind::kLegacyDefault
                                  : is_thread_stream ? OrderedStreamKind::kPerThreadDefault
                                                     : OrderedStreamKind::kBlocking;
                    if (!is_thread_stream && !is_legacy_stream) {
                        unsigned int flags = 0;
                        int priority = 0;
                        result = dispatch.stream_get_flags(original, &flags);
                        if (result != CUDA_SUCCESS) {
                            return result;
                        }
                        result = dispatch.stream_get_priority(original, &priority);
                        if (result != CUDA_SUCCESS) {
                            return result;
                        }
                        prefix.kind = (flags & CU_STREAM_NON_BLOCKING) != 0
                                          ? OrderedStreamKind::kNonBlocking
                                          : OrderedStreamKind::kBlocking;
                        prefix.priority = relative_priority(priority);
                    }
                    if (is_thread_stream) {
                        const std::lock_guard lock(mutex);
                        for (std::size_t index = 0; index < occupied_proxies.size(); ++index) {
                            if (!occupied_proxies[index]) {
                                occupied_proxies[index] = true;
                                prefix.proxy = index;
                                break;
                            }
                        }
                        if (!prefix.proxy) {
                            return CUDA_ERROR_OUT_OF_MEMORY;
                        }
                        prefix.physical = thread_streams[*prefix.proxy];
                    }
                    prefix.lease = take_event();
                    if (!prefix.lease) {
                        return CUDA_ERROR_OUT_OF_MEMORY;
                    }
                    prefix.lease->was_recorded = true;
                    result = dispatch.worker.queue.event_record(prefix.lease->event(), original);
                    if (result != CUDA_SUCCESS) {
                        // Record may have issued before returning an error.
                        // Retain the event and reject future frontend entry.
                        fail_closed(result);
                        return result;
                    }
                    has_prepared = true;
                }
                result = worker->invoke([&](CudaOperationQueue& queue) {
                    has_run = true;
                    for (const auto& binding : streams) {
                        if (binding.original == original && binding.thread == thread) {
                            output = binding;
                            return CUDA_SUCCESS;
                        }
                    }
                    streams.reserve(streams.size() + 1);
                    const auto logical =
                        queue.bind_stream(prefix.physical, prefix.kind, resource_lifetime);
                    if (!logical) {
                        return CUDA_ERROR_OUT_OF_MEMORY;
                    }
                    const auto imported = queue.submit(
                        logical, CudaQueuedNativeFence{prefix.lease->event(), prefix.lease});
                    if (!imported) {
                        if (!queue.close_stream(logical)) {
                            prefix.has_queue_owner = true;
                            return CUDA_ERROR_UNKNOWN;
                        }
                        return CUDA_ERROR_OUT_OF_MEMORY;
                    }
                    prefix.lease->has_queue_owner = true;
                    prefix.has_queue_owner = true;
                    output = {
                        original,     thread,
                        logical,      prefix.priority,
                        prefix.proxy, options.native_epoch_policy ? next_epoch_registration() : 0};
                    output.kind = prefix.kind;
                    streams.push_back(output);
                    return CUDA_SUCCESS;
                });
            }
            if (has_run || result != CUDA_ERROR_NOT_READY || worker->status() != CUDA_SUCCESS) {
                return result;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds{1});
        }
    }

    void fail_closed(CUresult result) noexcept {
        {
            const std::lock_guard lock(mutex);
            if (epoch) epoch->can_recover = false;
            status = result == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : result;
            // An uncertain native boundary stays paused. Wake callers to
            // report the failure rather than waiting forever on metadata.
            is_transaction = false;
        }
        condition.notify_all();
    }

    CUresult await(std::span<const CudaOperationToken> snapshot) {
        for (;;) {
            const auto result = command([&](CudaOperationQueue& queue) {
                CUresult status_value = CUDA_SUCCESS;
                for (const auto& token : snapshot) {
                    const auto value = queue.query(token);
                    if (value != CUDA_SUCCESS && value != CUDA_ERROR_NOT_READY) {
                        return value;
                    }
                    if (value == CUDA_ERROR_NOT_READY) {
                        status_value = value;
                    }
                }
                return status_value;
            });
            if (result != CUDA_ERROR_NOT_READY) {
                return result;
            }
            std::this_thread::sleep_for(options.worker.poll_interval);
        }
    }

    CUresult drain_stream(const StreamBinding& binding, bool should_wait) {
        std::optional<CudaOperationToken> snapshot;
        const auto result = command([&](CudaOperationQueue& queue) {
            snapshot = queue.stream_snapshot(binding.logical);
            if (!snapshot) {
                return CUDA_ERROR_INVALID_HANDLE;
            }
            return should_wait || !*snapshot ? CUDA_SUCCESS : queue.query(*snapshot);
        });
        if (result != CUDA_SUCCESS || !should_wait || !snapshot || !*snapshot) {
            return result;
        }
        return await(std::span{&*snapshot, 1});
    }

    CUresult cleanup() noexcept {
        if (has_ambiguous_cleanup) {
            return CUDA_ERROR_UNKNOWN;
        }
        // No automatic native cleanup retry after an error: an API error may
        // follow successful release. Keep remaining owners for trusted recovery.
        for (auto& event : imports->events) {
            if (!event) {
                continue;
            }
            const auto stopped = dispatch.worker.queue.event_synchronize(event);
            if (stopped != CUDA_SUCCESS) {
                return stopped;
            }
            const auto result = dispatch.worker.queue.event_destroy(event);
            if (result != CUDA_SUCCESS) {
                has_ambiguous_cleanup = true;
                return result;
            }
            event = nullptr;
        }
        for (auto& stream : thread_streams) {
            if (!stream) {
                continue;
            }
            const auto result = dispatch.stream_destroy(stream);
            if (result != CUDA_SUCCESS) {
                has_ambiguous_cleanup = true;
                return result;
            }
            stream = nullptr;
        }
        if (pause_fence) {
            const auto stopped = dispatch.worker.queue.event_synchronize(pause_fence);
            if (stopped != CUDA_SUCCESS) {
                return stopped;
            }
            const auto destroyed = dispatch.worker.queue.event_destroy(pause_fence);
            if (destroyed != CUDA_SUCCESS) {
                has_ambiguous_cleanup = true;
                return destroyed;
            }
            pause_fence = nullptr;
        }
        return pool ? pool->release() : CUDA_SUCCESS;
    }
};

CudaCooperativeFrontend::CudaCooperativeFrontend(
    CUcontext context, CudaFrontendDispatch dispatch,
    std::shared_ptr<control::AsyncCooperativeLaunchGate> gate, CudaFrontendOptions options)
    : implementation_(std::make_unique<Implementation>(context, std::move(dispatch),
                                                       std::move(gate), std::move(options))) {}

CudaCooperativeFrontend::~CudaCooperativeFrontend() {
    auto& state = *implementation_;
    state.stop_epoch_watcher();
    // No session survives its caller scope. Stop the idle CPU owner even when
    // unfinished CUDA ownership must be quarantined below; this proves no stop.
    state.native_heartbeat.reset();
    const bool has_events = std::any_of(state.imports->events.begin(), state.imports->events.end(),
                                        [](CUevent event) { return event != nullptr; });
    const bool has_streams = std::any_of(state.thread_streams.begin(), state.thread_streams.end(),
                                         [](CUstream stream) { return stream != nullptr; });
    if (state.worker || has_events || has_streams || state.pause_fence ||
        state.has_ambiguous_cleanup) {
        // Destruction is not physical-stop proof. Preserve an unfinished owner,
        // its worker and all native handles instead of releasing live resources.
        report_diagnostic("[glimmer] unfinished cooperative frontend retained\n");
        (void)implementation_.release();
    }
}

CUresult CudaCooperativeFrontend::initialize() noexcept {
    auto& state = *implementation_;
    try {
        if (state.status != CUDA_ERROR_NOT_INITIALIZED) {
            return state.status == CUDA_SUCCESS ? CUDA_ERROR_INVALID_VALUE : state.status;
        }
        const auto& queue = state.dispatch.worker.queue;
        if (!state.context || !state.gate || !state.dispatch.stream_create ||
            !state.dispatch.stream_destroy || !state.dispatch.stream_get_flags ||
            !state.dispatch.stream_get_priority || !queue.context_get_current ||
            !queue.event_create || !queue.event_record || !queue.event_destroy ||
            !queue.event_synchronize || state.options.import_events == 0 ||
            state.options.import_events > 65536 || state.options.per_thread_streams > 256 ||
            state.options.worker.poll_interval.count() <= 0) {
            state.status = CUDA_ERROR_INVALID_VALUE;
            return state.status;
        }
        if (state.options.native_epoch_policy &&
            (!control::valid_native_epoch_extent(*state.options.native_epoch_policy) ||
             !state.epoch_context_registration ||
             !state.options.native_epoch_dispatch.context_get_current ||
             !state.options.native_epoch_dispatch.context_push ||
             !state.options.native_epoch_dispatch.context_pop ||
             !state.options.native_epoch_dispatch.stream_get_context ||
             !state.options.native_epoch_dispatch.stream_is_capturing ||
             !state.options.native_epoch_dispatch.event_create ||
             !state.options.native_epoch_dispatch.event_record ||
             !state.options.native_epoch_dispatch.event_query ||
             !state.options.native_epoch_dispatch.event_destroy ||
             static_cast<bool>(state.options.native_epoch_dispatch.enter_native_scope) !=
                 static_cast<bool>(state.options.native_epoch_dispatch.leave_native_scope) ||
             state.options.native_epoch_monitor.poll_interval.count() <= 0 ||
             state.options.native_epoch_monitor.poll_interval > std::chrono::seconds{1} ||
             state.options.native_epoch_monitor.control_interval.count() <= 0 ||
             state.options.native_epoch_monitor.control_interval > std::chrono::seconds{1} ||
             state.options.native_epoch_monitor.shutdown_grace.count() <= 0 ||
             state.options.native_epoch_monitor.shutdown_grace > std::chrono::seconds{10})) {
            state.status = CUDA_ERROR_INVALID_VALUE;
            return state.status;
        }
        CUcontext context = nullptr;
        state.status = queue.context_get_current(&context);
        if (state.status != CUDA_SUCCESS || context != state.context) {
            state.status = state.status == CUDA_SUCCESS ? CUDA_ERROR_INVALID_CONTEXT : state.status;
            return state.status;
        }
        state.options.worker.progress_signal = state.gate->subscribe_progress();
        state.options.worker.control_gate = state.gate;
        if (!state.options.worker.progress_signal) {
            state.status = CUDA_ERROR_OUT_OF_MEMORY;
            return state.status;
        }
        state.imports->events.resize(state.options.import_events);
        state.imports->occupied.resize(state.options.import_events);
        state.thread_streams.resize(state.options.per_thread_streams);
        state.occupied_proxies.resize(state.options.per_thread_streams);
        state.pool = std::make_shared<CudaCooperativePool>(state.context, state.dispatch.pool,
                                                           state.options.progress_slots,
                                                           state.options.audit_block_capacity);
        state.status = state.pool->initialize();
        if (state.status != CUDA_SUCCESS) {
            return state.status;
        }
        for (auto& event : state.imports->events) {
            state.status = queue.event_create(&event, CU_EVENT_DISABLE_TIMING);
            if (state.status != CUDA_SUCCESS || !event) {
                state.has_ambiguous_cleanup = event != nullptr || state.status == CUDA_SUCCESS;
                state.status = state.status == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : state.status;
                return state.status;
            }
        }
        for (auto& stream : state.thread_streams) {
            state.status = state.dispatch.stream_create(&stream, CU_STREAM_NON_BLOCKING);
            if (state.status != CUDA_SUCCESS || !stream) {
                state.has_ambiguous_cleanup = stream != nullptr || state.status == CUDA_SUCCESS;
                state.status = state.status == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : state.status;
                return state.status;
            }
        }
        state.status = queue.event_create(&state.pause_fence, CU_EVENT_DISABLE_TIMING);
        if (state.status != CUDA_SUCCESS || !state.pause_fence) {
            state.has_ambiguous_cleanup =
                state.pause_fence != nullptr || state.status == CUDA_SUCCESS;
            state.status = state.status == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : state.status;
            return state.status;
        }
        state.coordinator = std::make_shared<CudaAdmissionCoordinator>(state.gate);
        state.worker = std::make_unique<CudaQueueWorker>(state.context, state.dispatch.worker,
                                                         state.coordinator, state.options.worker);
        state.status = state.worker->status();
        if (state.status == CUDA_SUCCESS && state.options.native_epoch_policy)
            state.epoch_watcher = std::thread([&state] { state.watch_epochs(); });
        return state.status;
    } catch (...) {
        state.status = CUDA_ERROR_UNKNOWN;
        return state.status;
    }
}

CUresult CudaCooperativeFrontend::launch(const CooperativeKernelVariant& variant, CUstream stream,
                                         bool per_thread_default_stream,
                                         std::array<unsigned int, 3> grid,
                                         std::array<unsigned int, 3> block, void* const* arguments,
                                         std::shared_ptr<void> kernel_resources) noexcept {
    try {
        auto& state = *implementation_;
        if (variant.context != state.context || !kernel_resources) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        Implementation::StreamBinding binding;
        auto result = state.stream_binding(stream, per_thread_default_stream, binding);
        if (result != CUDA_SUCCESS) {
            return result;
        }
        return state.command([&](CudaOperationQueue& queue) {
            // Pool preparation and queue mutation share the one owner. No CUDA
            // calls or borrowed stack storage survive this command.
            auto prepared = state.pool->prepare(variant, grid, block, arguments, kernel_resources);
            if (!prepared) {
                return CUDA_ERROR_OUT_OF_MEMORY;
            }
            const auto initialization =
                queue.submit(binding.logical, std::move(prepared->initialization));
            if (!initialization) {
                return CUDA_ERROR_OUT_OF_MEMORY;
            }
            const auto kernel = queue.submit(binding.logical, std::move(prepared->kernel));
            if (!kernel) {
                return queue.cancel_stopped(initialization) ? CUDA_ERROR_OUT_OF_MEMORY
                                                            : CUDA_ERROR_UNKNOWN;
            }
            if (!state.coordinator->track(
                    queue, kernel,
                    {binding.priority, variant.shape.logical_blocks, variant.audit.has_value()})) {
                const bool cancelled_kernel = queue.cancel_stopped(kernel);
                const bool cancelled_initialization = queue.cancel_stopped(initialization);
                return cancelled_kernel && cancelled_initialization ? CUDA_ERROR_OUT_OF_MEMORY
                                                                    : CUDA_ERROR_UNKNOWN;
            }
            return CUDA_SUCCESS;
        });
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}

CUresult CudaCooperativeFrontend::launch_sliced(CUcontext context, CUstream stream,
                                                bool per_thread_default_stream,
                                                CudaQueuedKernel kernel) noexcept {
    try {
        auto& state = *implementation_;
        if (context != state.context || !kernel.function || !kernel.resources ||
            !kernel.continuation)
            return CUDA_ERROR_INVALID_VALUE;
        std::uint32_t logical_blocks = 1;
        for (const auto extent : kernel.grid) {
            if (!extent || extent > std::numeric_limits<std::uint32_t>::max() / logical_blocks)
                return CUDA_ERROR_INVALID_VALUE;
            logical_blocks *= extent;
        }
        Implementation::StreamBinding binding;
        const auto result = state.stream_binding(stream, per_thread_default_stream, binding);
        if (result != CUDA_SUCCESS) return result;
        return state.command([&](CudaOperationQueue& queue) {
            const auto token = queue.submit(binding.logical, std::move(kernel));
            if (!token) return CUDA_ERROR_OUT_OF_MEMORY;
            if (!state.coordinator->track(queue, token, {binding.priority, logical_blocks}))
                return queue.cancel_stopped(token) ? CUDA_ERROR_OUT_OF_MEMORY : CUDA_ERROR_UNKNOWN;
            return CUDA_SUCCESS;
        });
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}

CUresult CudaCooperativeFrontend::copy(CUstream stream, bool per_thread_default_stream,
                                       CudaQueuedCopy operation) noexcept {
    try {
        auto& state = *implementation_;
        Implementation::StreamBinding binding;
        const auto result = state.stream_binding(stream, per_thread_default_stream, binding);
        if (result != CUDA_SUCCESS) {
            return result;
        }
        return state.command([&](CudaOperationQueue& queue) {
            return queue.submit(binding.logical, std::move(operation)) ? CUDA_SUCCESS
                                                                       : CUDA_ERROR_OUT_OF_MEMORY;
        });
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}

CudaQueuedReleaseAdmission CudaCooperativeFrontend::release(CUstream stream,
                                                            bool per_thread_default_stream,
                                                            CudaQueuedFree operation) noexcept {
    enum class Acceptance { kPending, kSubmitting, kAccepted, kRejected, kCancelled };
    struct Admission {
        CudaQueuedFree operation;
        std::atomic<Acceptance> state{Acceptance::kPending};
        explicit Admission(CudaQueuedFree value) : operation(std::move(value)) {}
    };
    std::shared_ptr<Admission> admission;
    CUresult result = CUDA_ERROR_UNKNOWN;
    try {
        if (!operation.pointer || !operation.resources || !operation.release_observer) {
            return {CUDA_ERROR_INVALID_VALUE, false};
        }
        admission = std::make_shared<Admission>(std::move(operation));
        auto& state = *implementation_;
        Implementation::StreamBinding binding;
        result = state.stream_binding(stream, per_thread_default_stream, binding);
        if (result == CUDA_SUCCESS) {
            for (;;) {
                {
                    Implementation::Permit permit(state);
                    if (permit.result != CUDA_SUCCESS) {
                        result = permit.result;
                        break;
                    }
                    CUcontext caller{};
                    result = state.dispatch.worker.queue.context_get_current(&caller);
                    if (result != CUDA_SUCCESS || caller != state.context) {
                        result = result == CUDA_SUCCESS ? CUDA_ERROR_INVALID_CONTEXT : result;
                        break;
                    }
                    // Owned captures survive even an uncertain mailbox wait.
                    // Cancelling PENDING races atomically with worker submission.
                    result = state.worker->invoke([admission, logical = binding.logical](
                                                      CudaOperationQueue& queue) {
                        auto expected = Acceptance::kPending;
                        if (!admission->state.compare_exchange_strong(expected,
                                                                      Acceptance::kSubmitting)) {
                            return CUDA_ERROR_OUT_OF_MEMORY;
                        }
                        const auto token = queue.submit(logical, std::move(admission->operation));
                        admission->state.store(token ? Acceptance::kAccepted
                                                     : Acceptance::kRejected);
                        return token ? CUDA_SUCCESS : CUDA_ERROR_OUT_OF_MEMORY;
                    });
                }
                if (result != CUDA_ERROR_NOT_READY || state.worker->status() != CUDA_SUCCESS ||
                    admission->state.load() != Acceptance::kPending) {
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds{1});
            }
        }
    } catch (...) {
        result = CUDA_ERROR_UNKNOWN;
    }
    if (!admission) {
        return {result, false};
    }
    auto expected = Acceptance::kPending;
    (void)admission->state.compare_exchange_strong(expected, Acceptance::kCancelled);
    const auto acceptance = admission->state.load();
    const bool has_owned_release =
        acceptance == Acceptance::kSubmitting || acceptance == Acceptance::kAccepted;
    return {result == CUDA_SUCCESS && !has_owned_release ? CUDA_ERROR_UNKNOWN : result,
            has_owned_release};
}

CUresult CudaCooperativeFrontend::bind_event(CUevent event,
                                             std::shared_ptr<void> resources) noexcept {
    try {
        auto& state = *implementation_;
        if (!event || !resources) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        return state.command([&](CudaOperationQueue& queue) {
            const auto duplicate =
                std::find_if(state.events.begin(), state.events.end(),
                             [&](const auto& value) { return value.original == event; });
            if (duplicate != state.events.end()) {
                return CUDA_ERROR_INVALID_HANDLE;
            }
            state.events.reserve(state.events.size() + 1);
            auto calls = std::make_shared<Implementation::EventCallGate>();
            const auto token = queue.bind_event(event, resources);
            if (!token) {
                return CUDA_ERROR_OUT_OF_MEMORY;
            }
            state.events.push_back({event, token, std::move(calls)});
            return CUDA_SUCCESS;
        });
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}

CUresult CudaCooperativeFrontend::record_event(
    CUevent event, CUstream stream, bool per_thread_default_stream,
    std::shared_ptr<CudaQueueEventRecordGuard> guard) noexcept {
    try {
        auto& state = *implementation_;
        Implementation::StreamBinding binding;
        const auto result = state.stream_binding(stream, per_thread_default_stream, binding);
        if (result != CUDA_SUCCESS) {
            return result;
        }
        return state.command([&](CudaOperationQueue& queue) {
            for (const auto& value : state.events) {
                if (value.original == event) {
                    CudaEventRecordPreparation preparation;
                    if (guard) {
                        const auto status = guard->prepare_record(preparation);
                        if (status != CUDA_SUCCESS) {
                            return status;
                        }
                    }
                    const std::span<const CudaOperationToken> dependencies =
                        preparation.native_prefix ? std::span{&preparation.native_prefix, 1}
                                                  : std::span<const CudaOperationToken>{};
                    if (!queue.submit(binding.logical,
                                      CudaQueuedEventRecord{value.logical, guard,
                                                            preparation.should_order_previous},
                                      dependencies)) {
                        return CUDA_ERROR_OUT_OF_MEMORY;
                    }
                    // The actor and this command's caller contribution still
                    // exclude native pause/issue across acceptance/publication.
                    return guard ? guard->record_accepted() : CUDA_SUCCESS;
                }
            }
            return CUDA_ERROR_INVALID_HANDLE;
        });
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}

CUresult CudaCooperativeFrontend::admit_event_metadata(CUevent event,
                                                       const NativeOperation& operation) noexcept {
    try {
        if (!operation) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        auto& state = *implementation_;
        Implementation::Permit permit(state);
        if (permit.result != CUDA_SUCCESS) {
            return permit.result;
        }
        std::shared_ptr<Implementation::EventCallGate> calls;
        const auto found = state.worker->invoke([&](CudaOperationQueue& queue) {
            for (const auto& value : state.events) {
                if (value.original == event && queue.event_snapshot(value.logical)) {
                    calls = value.calls;
                    return CUDA_SUCCESS;
                }
            }
            return CUDA_ERROR_INVALID_HANDLE;
        });
        if (found != CUDA_SUCCESS) {
            return found;
        }
        const Implementation::EventCallPermit call(std::move(calls));
        if (call.result != CUDA_SUCCESS) {
            return call.result;
        }
        // A prior serialized caller may have failed while this caller waited.
        const Implementation::Permit resumed(state);
        if (resumed.result != CUDA_SUCCESS) {
            return resumed.result;
        }
        try {
            return operation();
        } catch (...) {
            state.fail_closed(CUDA_ERROR_UNKNOWN);
            return CUDA_ERROR_UNKNOWN;
        }
    } catch (...) {
        implementation_->fail_closed(CUDA_ERROR_UNKNOWN);
        return CUDA_ERROR_UNKNOWN;
    }
}

CUresult CudaCooperativeFrontend::prepare_event_wait(CUevent event,
                                                     CudaFrontendEventWait& output) noexcept {
    try {
        if (output) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        auto& state = *implementation_;
        return state.command([&](CudaOperationQueue& queue) {
            for (const auto& value : state.events) {
                if (value.original == event) {
                    const auto snapshot = queue.event_snapshot(value.logical);
                    if (!snapshot) {
                        return CUDA_ERROR_INVALID_HANDLE;
                    }
                    output.owner_ = state.resource_lifetime;
                    output.event_ = value.logical;
                    output.prefix_ = *snapshot;
                    return CUDA_SUCCESS;
                }
            }
            return CUDA_ERROR_INVALID_HANDLE;
        });
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}

CUresult CudaCooperativeFrontend::wait_event_snapshot(
    CUstream stream, bool per_thread_default_stream,
    const CudaFrontendEventWait& snapshot) noexcept {
    try {
        auto& state = *implementation_;
        if (!snapshot || snapshot.owner_ != state.resource_lifetime) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        Implementation::StreamBinding binding;
        const auto result = state.stream_binding(stream, per_thread_default_stream, binding);
        if (result != CUDA_SUCCESS) {
            return result;
        }
        return state.command([&](CudaOperationQueue& queue) {
            if (!queue.event_snapshot(snapshot.event_)) {
                return CUDA_ERROR_INVALID_HANDLE;
            }
            const std::span<const CudaOperationToken> dependencies =
                snapshot.prefix_ ? std::span{&snapshot.prefix_, 1}
                                 : std::span<const CudaOperationToken>{};
            return queue.submit(binding.logical, CudaQueuedMarker{}, dependencies)
                       ? CUDA_SUCCESS
                       : CUDA_ERROR_OUT_OF_MEMORY;
        });
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}

CUresult CudaCooperativeFrontend::wait_event(CUstream stream, bool per_thread_default_stream,
                                             CUevent event) noexcept {
    try {
        auto& state = *implementation_;
        Implementation::StreamBinding binding;
        const auto result = state.stream_binding(stream, per_thread_default_stream, binding);
        if (result != CUDA_SUCCESS) {
            return result;
        }
        return state.command([&](CudaOperationQueue& queue) {
            for (const auto& value : state.events) {
                if (value.original == event) {
                    return queue.wait_event(binding.logical, value.logical)
                               ? CUDA_SUCCESS
                               : CUDA_ERROR_OUT_OF_MEMORY;
                }
            }
            return CUDA_ERROR_INVALID_HANDLE;
        });
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}

CUresult CudaCooperativeFrontend::query_event(CUevent event) noexcept {
    try {
        auto& state = *implementation_;
        const CudaQueueWorker::Command read = [&](CudaOperationQueue& queue) {
            for (const auto& value : state.events) {
                if (value.original == event) {
                    return queue.query_event(value.logical);
                }
            }
            return CUDA_ERROR_INVALID_HANDLE;
        };
        return state.options.native_epoch_policy ? state.query_command(read) : state.command(read);
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}

CUresult CudaCooperativeFrontend::synchronize_event(CUevent event) noexcept {
    try {
        auto& state = *implementation_;
        std::optional<CudaOperationToken> snapshot;
        const auto result = state.command([&](CudaOperationQueue& queue) {
            for (const auto& value : state.events) {
                if (value.original == event) {
                    snapshot = queue.event_snapshot(value.logical);
                    return snapshot ? CUDA_SUCCESS : CUDA_ERROR_INVALID_HANDLE;
                }
            }
            return CUDA_ERROR_INVALID_HANDLE;
        });
        if (result != CUDA_SUCCESS || !snapshot || !*snapshot) {
            return result;
        }
        return state.await(std::span{&*snapshot, 1});
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}

CUresult CudaCooperativeFrontend::query_stream(CUstream stream,
                                               bool per_thread_default_stream) noexcept {
    try {
        auto& state = *implementation_;
        if (state.options.native_epoch_policy) {
            bool has_prefix{};
            return complete_stream_snapshot(stream, per_thread_default_stream, false, has_prefix);
        }
        Implementation::StreamBinding binding;
        const auto result = state.stream_binding(stream, per_thread_default_stream, binding);
        return result == CUDA_SUCCESS ? state.drain_stream(binding, false) : result;
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}

CUresult CudaCooperativeFrontend::complete_stream_snapshot(CUstream stream,
                                                           bool per_thread_default_stream,
                                                           bool should_wait,
                                                           bool& has_logical_prefix) noexcept {
    has_logical_prefix = false;
    try {
        auto& state = *implementation_;
        const bool is_thread_stream =
            stream == CU_STREAM_PER_THREAD || (!stream && per_thread_default_stream);
        const bool is_legacy_stream =
            stream == CU_STREAM_LEGACY || (!stream && !per_thread_default_stream);
        const CUstream original = is_thread_stream   ? CU_STREAM_PER_THREAD
                                  : is_legacy_stream ? CU_STREAM_LEGACY
                                                     : stream;
        const auto thread = is_thread_stream ? caller_thread_lifetime() : nullptr;
        if (!should_wait && state.options.native_epoch_policy) {
            CUcontext actual{};
            const auto result = state.dispatch.worker.queue.context_get_current(&actual);
            if (result != CUDA_SUCCESS || actual != state.context)
                return result == CUDA_SUCCESS ? CUDA_ERROR_INVALID_CONTEXT : result;
            std::shared_ptr<Implementation::EpochEntry> entry;
            std::shared_ptr<CudaNativeExecutionEpoch> execution;
            {
                const std::lock_guard lock(state.mutex);
                if (state.status != CUDA_SUCCESS) return state.status;
                entry = state.epoch;
                if (entry && !entry->is_done.load()) execution = entry->owner;
            }
            bool is_affected =
                entry && !entry->is_done.load() &&
                ((entry->binding.original == original && entry->binding.thread == thread) ||
                 (is_legacy_stream && entry->binding.kind != OrderedStreamKind::kNonBlocking));
            if (entry && !entry->is_done.load() &&
                entry->binding.kind == OrderedStreamKind::kLegacyDefault && !is_affected) {
                bool is_blocking = is_thread_stream || is_legacy_stream;
                if (!is_blocking) {
                    unsigned int flags{};
                    const auto status = state.dispatch.stream_get_flags(original, &flags);
                    if (status != CUDA_SUCCESS) return status;
                    is_blocking = (flags & CU_STREAM_NON_BLOCKING) == 0;
                }
                is_affected = is_blocking;
            }
            if (is_affected) {
                entry->is_close_requested.store(true);
                if (execution) execution->close();
                has_logical_prefix = true;
                return CUDA_ERROR_NOT_READY;
            }
        }
        std::vector<CudaOperationToken> snapshot;
        const CudaQueueWorker::Command read = [&](CudaOperationQueue& queue) {
            for (const auto& value : state.streams) {
                if (value.original == original && value.thread == thread) {
                    const auto tail = queue.completion_snapshot(value.logical);
                    if (!tail) {
                        return CUDA_ERROR_INVALID_HANDLE;
                    }
                    has_logical_prefix = true;
                    snapshot = *tail;
                    return should_wait ? CUDA_SUCCESS : queue.query_native_boundary(snapshot);
                }
            }
            bool is_blocking = true;
            if (!is_thread_stream && !is_legacy_stream) {
                unsigned int flags{};
                const auto status = state.dispatch.stream_get_flags(original, &flags);
                if (status != CUDA_SUCCESS) {
                    return status;
                }
                is_blocking = (flags & CU_STREAM_NON_BLOCKING) == 0;
            }
            if (is_blocking) {
                snapshot = queue.implicit_completion_snapshot(is_legacy_stream);
                has_logical_prefix = !snapshot.empty();
            }
            return should_wait ? CUDA_SUCCESS : queue.query_native_boundary(snapshot);
        };
        const auto result = !should_wait && state.options.native_epoch_policy
                                ? state.query_command(read)
                                : state.command(read);
        return result == CUDA_SUCCESS && should_wait && !snapshot.empty() ? state.await(snapshot)
                                                                          : result;
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}

CUresult CudaCooperativeFrontend::synchronize_stream(CUstream stream,
                                                     bool per_thread_default_stream) noexcept {
    try {
        auto& state = *implementation_;
        Implementation::StreamBinding binding;
        const auto result = state.stream_binding(stream, per_thread_default_stream, binding);
        return result == CUDA_SUCCESS ? state.drain_stream(binding, true) : result;
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}

CUresult CudaCooperativeFrontend::synchronize_context() noexcept {
    try {
        auto& state = *implementation_;
        std::vector<CudaOperationToken> snapshot;
        const auto result = state.command([&](CudaOperationQueue& queue) {
            snapshot = queue.context_snapshot();
            return CUDA_SUCCESS;
        });
        return result == CUDA_SUCCESS ? state.await(snapshot) : result;
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}

CUresult CudaCooperativeFrontend::native_epoch(CUstream stream, const NativeOperation& original,
                                               bool per_thread_default_stream) noexcept {
    bool has_paused = false;
    try {
        auto& state = *implementation_;
        if (!state.options.native_epoch_policy || !original ||
            Implementation::Permit::is_admitted(state))
            return CUDA_ERROR_INVALID_VALUE;
        {
            const std::lock_guard lock(state.mutex);
            if (state.status != CUDA_SUCCESS || state.is_closing)
                return state.is_closing ? CUDA_ERROR_DEINITIALIZED : state.status;
        }
        const bool is_thread_stream =
            stream == CU_STREAM_PER_THREAD || (!stream && per_thread_default_stream);
        const CUstream canonical = is_thread_stream                          ? CU_STREAM_PER_THREAD
                                   : (!stream || stream == CU_STREAM_LEGACY) ? CU_STREAM_LEGACY
                                                                             : stream;
        CUcontext actual{};
        const auto context_status = state.dispatch.worker.queue.context_get_current(&actual);
        if (context_status != CUDA_SUCCESS || actual != state.context)
            return context_status == CUDA_SUCCESS ? CUDA_ERROR_INVALID_CONTEXT : context_status;
        CUstreamCaptureStatus capture{};
        const auto capture_status =
            state.options.native_epoch_dispatch.stream_is_capturing(canonical, &capture);
        if (capture_status != CUDA_SUCCESS) return capture_status;
        if (capture != CU_STREAM_CAPTURE_STATUS_NONE) return CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED;
        const auto caller = caller_thread_lifetime();
        if (!caller->registration) return CUDA_ERROR_UNKNOWN;
        // Retain this call's attempted owner even after asynchronous thaw. A
        // known rejection or construction failure must not become a new acquire.
        std::shared_ptr<Implementation::EpochEntry> attempted;
        for (;;) {
            if (attempted && attempted->owner) {
                const auto snapshot = attempted->owner->snapshot();
                if (snapshot.phase == CudaNativeEpochPhase::kRejected)
                    return snapshot.status == CUDA_SUCCESS ? CUDA_ERROR_NOT_PERMITTED
                                                           : snapshot.status;
                if (snapshot.phase == CudaNativeEpochPhase::kRetained)
                    return snapshot.status == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : snapshot.status;
                if (snapshot.phase == CudaNativeEpochPhase::kComplete &&
                    snapshot.status != CUDA_SUCCESS)
                    return snapshot.status;
            }
            std::shared_ptr<Implementation::EpochEntry> entry;
            std::shared_ptr<CudaNativeExecutionEpoch> execution;
            {
                const std::lock_guard lock(state.mutex);
                if (state.status != CUDA_SUCCESS || state.is_closing)
                    return state.is_closing ? CUDA_ERROR_DEINITIALIZED : state.status;
                entry = state.epoch;
                if (entry && !entry->is_done.load()) execution = entry->owner;
            }
            if (entry && !entry->is_done.load()) {
                if (entry->binding.original != canonical || entry->caller != caller) {
                    const auto closed = state.close_epoch(true);
                    if (closed != CUDA_SUCCESS) return closed;
                    continue;
                }
                if (execution) {
                    attempted = entry;
                    Implementation::EpochCallerUse use(*entry);
                    const auto snapshot = execution->snapshot();
                    if (snapshot.phase == CudaNativeEpochPhase::kRejected)
                        return CUDA_ERROR_NOT_PERMITTED;
                    if (snapshot.phase == CudaNativeEpochPhase::kRetained)
                        return snapshot.status == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN
                                                               : snapshot.status;
                    if (snapshot.phase == CudaNativeEpochPhase::kComplete &&
                        snapshot.status != CUDA_SUCCESS)
                        return snapshot.status;
                    auto issue = !use.has_contribution || entry->is_close_requested.load()
                                     ? std::nullopt
                                     : execution->begin_call();
                    if (issue) {
                        CUresult original_status = CUDA_ERROR_UNKNOWN;
                        try {
                            original_status = original();
                        } catch (...) {
                            static_cast<void>(execution->original_returned(
                                *issue, NativeEpochCallOutcome::kUncertain));
                            state.fail_closed(CUDA_ERROR_UNKNOWN);
                            return CUDA_ERROR_UNKNOWN;
                        }
                        if (!execution->original_returned(
                                *issue, original_status == CUDA_SUCCESS
                                            ? NativeEpochCallOutcome::kSuccess
                                            : NativeEpochCallOutcome::kRejected)) {
                            state.fail_closed(CUDA_ERROR_UNKNOWN);
                            return original_status == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN
                                                                   : original_status;
                        }
                        return original_status;
                    }
                }
                std::this_thread::sleep_for(state.options.worker.poll_interval);
                continue;
            }
            // Binding may import original predecessor work, but never a new
            // authorization. Drain it before the epoch's independent request.
            Implementation::StreamBinding binding;
            auto result = state.stream_binding(stream, per_thread_default_stream, binding);
            if (result != CUDA_SUCCESS) return result;
            if (!binding.registration) return CUDA_ERROR_UNKNOWN;
            auto prepared = std::make_shared<Implementation::EpochEntry>();
            prepared->binding = binding;
            prepared->caller = caller;
            for (;;) {
                result = state.try_pause(
                    [&] { return state.worker->try_pause_native(prepared->boundary); });
                if (result == CUDA_SUCCESS) break;
                if (result != CUDA_ERROR_NOT_READY) return result;
                std::this_thread::sleep_for(state.options.worker.poll_interval);
            }
            has_paused = true;
            // A concurrent logical close can retire the binding between the
            // lookup and pause. Revalidate its registration on the exact empty
            // actor boundary before creating any physical epoch owner.
            result =
                state.worker->invoke_drained_paused(prepared->boundary, [&](CudaOperationQueue&) {
                    const auto found = std::find_if(
                        state.streams.begin(), state.streams.end(), [&](const auto& current) {
                            return current.original == canonical &&
                                   current.thread == binding.thread &&
                                   current.logical == binding.logical &&
                                   current.registration == binding.registration;
                        });
                    return found != state.streams.end() ? CUDA_SUCCESS : CUDA_ERROR_INVALID_HANDLE;
                });
            if (result != CUDA_SUCCESS) {
                const auto released =
                    state.worker->finish_native(prepared->boundary, [](CudaOperationQueue& queue) {
                        return queue.pending_count() == 0 ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
                    });
                {
                    const std::lock_guard lock(state.mutex);
                    if (released != CUDA_SUCCESS) state.status = released;
                    state.is_transaction = false;
                }
                state.condition.notify_all();
                has_paused = false;
                return released == CUDA_SUCCESS ? result : released;
            }
            auto extent = *state.options.native_epoch_policy;
            if (is_thread_stream) extent.calls_per_prefix = 1;
            prepared->owner = std::make_shared<CudaNativeExecutionEpoch>(
                state.context, canonical,
                NativeEpochScope{state.epoch_context_registration, binding.registration,
                                 caller->registration},
                control::NativeEpochLaunchRequest{extent, binding.priority}, state.gate,
                state.options.native_epoch_dispatch, state.options.native_epoch_monitor);
            std::shared_ptr<Implementation::EpochEntry> retired;
            {
                const std::lock_guard lock(state.mutex);
                retired = std::exchange(state.epoch, prepared);
            }
            attempted = prepared;
            has_paused = false;  // published ownership now belongs to the watcher
            state.condition.notify_all();
            // Retired monitor was already joined; all destruction is outside
            // the metadata lock. New original entry occurs only in the loop.
        }
    } catch (const std::bad_alloc&) {
        if (has_paused) implementation_->fail_closed(CUDA_ERROR_OUT_OF_MEMORY);
        return CUDA_ERROR_OUT_OF_MEMORY;
    } catch (...) {
        if (has_paused) implementation_->fail_closed(CUDA_ERROR_UNKNOWN);
        return CUDA_ERROR_UNKNOWN;
    }
}

std::optional<CudaNativeEpochSnapshot> CudaCooperativeFrontend::native_epoch_snapshot()
    const noexcept {
    try {
        auto& state = *implementation_;
        std::shared_ptr<CudaNativeExecutionEpoch> execution;
        {
            const std::lock_guard lock(state.mutex);
            if (state.epoch) execution = state.epoch->owner;
        }
        return execution ? std::optional{execution->snapshot()} : std::nullopt;
    } catch (...) {
        return std::nullopt;
    }
}

CUresult CudaCooperativeFrontend::close_native_epoch(bool should_wait) noexcept {
    try {
        return implementation_->close_epoch(should_wait);
    } catch (...) {
        implementation_->fail_closed(CUDA_ERROR_UNKNOWN);
        return CUDA_ERROR_UNKNOWN;
    }
}

CUresult CudaCooperativeFrontend::recover_native_epoch(
    const control::NativeEpochGrant& expected, const NativeOperation& context_completion) noexcept {
    auto& state = *implementation_;
    try {
        if (!state.options.native_epoch_policy) return CUDA_ERROR_NOT_SUPPORTED;
        if (Implementation::Permit::is_admitted(state)) return CUDA_ERROR_NOT_PERMITTED;
        std::shared_ptr<Implementation::EpochEntry> entry;
        {
            const std::lock_guard lock(state.mutex);
            if (state.is_closing || state.is_epoch_watcher_closing) return CUDA_ERROR_DEINITIALIZED;
            entry = state.epoch;
            if (!entry || !entry->owner) return CUDA_ERROR_INVALID_VALUE;
            if (!entry->is_done.load() || state.is_transaction || state.active_calls != 0 ||
                entry->active_owner_calls.load() != 0)
                return CUDA_ERROR_NOT_READY;
            if (state.status == CUDA_SUCCESS || !entry->can_recover ||
                entry->has_recovery_attempt || state.has_ambiguous_cleanup || state.checkpoint)
                return CUDA_ERROR_UNKNOWN;
        }
        const auto before = entry->owner->snapshot();
        if (expected.execution.task_id == 0 ||
            expected.execution.task_id != before.grant.execution.task_id ||
            expected.execution.generation != before.grant.execution.generation ||
            expected.extent != before.grant.extent ||
            expected.acceptance_deadline_nanoseconds !=
                before.grant.acceptance_deadline_nanoseconds)
            return CUDA_ERROR_INVALID_VALUE;
        if (!before.execution || before.phase != CudaNativeEpochPhase::kRetained ||
            before.execution->phase != NativeEpochPhase::kStopping ||
            !before.execution->has_ownership)
            return CUDA_ERROR_UNKNOWN;
        {
            const std::lock_guard lock(state.mutex);
            if (state.epoch != entry || !entry->can_recover || entry->has_recovery_attempt)
                return CUDA_ERROR_UNKNOWN;
            state.is_transaction = true;
            entry->has_recovery_attempt = true;
        }
        auto result =
            state.worker->invoke_drained_paused(entry->boundary, [&](CudaOperationQueue&) {
                const auto& binding = entry->binding;
                const auto found = std::find_if(
                    state.streams.begin(), state.streams.end(), [&](const auto& current) {
                        return current.original == binding.original &&
                               current.thread == binding.thread &&
                               current.logical == binding.logical &&
                               current.registration == binding.registration;
                    });
                return found == state.streams.end() ? CUDA_ERROR_INVALID_HANDLE : CUDA_SUCCESS;
            });
        if (result == CUDA_SUCCESS) {
            result = entry->owner->recover_context_completion(
                {state.epoch_context_registration, entry->binding.registration,
                 entry->caller->registration},
                context_completion);
        }
        if (result == CUDA_SUCCESS) {
            const auto stopped = entry->owner->snapshot();
            if (!stopped.execution || stopped.execution->has_ownership ||
                stopped.created_events != stopped.destroyed_events ||
                stopped.execution->issued_calls != before.execution->issued_calls)
                result = CUDA_ERROR_UNKNOWN;
            else {
                result =
                    state.worker->finish_native(entry->boundary, [&](CudaOperationQueue& queue) {
                        if (queue.pending_count() != 0) return CUDA_ERROR_UNKNOWN;
                        return queue.submit(entry->binding.logical, CudaQueuedMarker{})
                                   ? CUDA_SUCCESS
                                   : CUDA_ERROR_OUT_OF_MEMORY;
                    });
            }
        }
        {
            const std::lock_guard lock(state.mutex);
            if (result == CUDA_SUCCESS && entry->can_recover && state.epoch == entry)
                state.status = CUDA_SUCCESS;
            else if (result == CUDA_SUCCESS)
                result = CUDA_ERROR_UNKNOWN;
            entry->can_recover = false;
            state.is_transaction = false;
        }
        state.condition.notify_all();
        return result;
    } catch (...) {
        state.fail_closed(CUDA_ERROR_UNKNOWN);
        return CUDA_ERROR_UNKNOWN;
    }
}

std::optional<CUresult> CudaCooperativeFrontend::recover_native_epoch(
    const NativeOperation& context_completion) noexcept {
    auto& state = *implementation_;
    try {
        if (!context_completion) return CUDA_ERROR_INVALID_VALUE;
        std::shared_ptr<Implementation::EpochEntry> entry;
        {
            const std::lock_guard lock(state.mutex);
            if (state.status == CUDA_SUCCESS) return std::nullopt;
            entry = state.epoch;
            if (!entry || !entry->owner || !entry->can_recover) return CUDA_ERROR_UNKNOWN;
        }
        return recover_native_epoch(entry->owner->snapshot().grant, context_completion);
    } catch (...) {
        state.fail_closed(CUDA_ERROR_UNKNOWN);
        return CUDA_ERROR_UNKNOWN;
    }
}

CUresult CudaCooperativeFrontend::native(
    CUstream stream, bool per_thread_default_stream, const NativeOperation& operation,
    bool is_execution, std::shared_ptr<CudaQueueNativeFenceObserver> completion_observer,
    std::span<const CUevent> waited_events,
    std::shared_ptr<CudaFrontendNativeWaitGuard> wait_guard) noexcept {
    try {
        auto& state = *implementation_;
        NativeTiming timing(state.options.should_trace_native_timings);
        if (Implementation::Permit::is_admitted(state)) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        if (!operation || (!is_execution && !waited_events.empty()) ||
            waited_events.size() > kNativeEventWaitLimit) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        if (std::ranges::find(waited_events, nullptr) != waited_events.end()) {
            return CUDA_ERROR_INVALID_HANDLE;
        }
        std::vector<CudaOperationToken> dependencies;
        try {
            dependencies.reserve(waited_events.size());
        } catch (const std::bad_alloc&) {
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
        Implementation::StreamBinding binding;
        auto result = state.stream_binding(stream, per_thread_default_stream, binding);
        if (result != CUDA_SUCCESS) {
            return result;
        }
        const auto lease = state.take_event();
        if (!lease) {
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
        CudaOperationToken reservation;
        std::optional<control::TaskExecutionIdentity> execution;
        timing.finish_phase(NativeTimingPhase::kPrepare);
        if (is_execution) {
            result = state.command([&](CudaOperationQueue& queue) {
                for (std::size_t index = 0; index < waited_events.size(); ++index) {
                    const auto event = waited_events[index];
                    const auto preceding = waited_events.first(index);
                    if (std::ranges::find(preceding, event) != preceding.end()) {
                        continue;
                    }
                    bool should_wait = true;
                    if (wait_guard) {
                        const auto status = wait_guard->should_wait(event, should_wait);
                        if (status != CUDA_SUCCESS) {
                            return status;
                        }
                    }
                    if (!should_wait) {
                        continue;
                    }
                    const auto found = std::ranges::find(state.events, event,
                                                         &Implementation::EventBinding::original);
                    if (found == state.events.end()) {
                        return CUDA_ERROR_INVALID_HANDLE;
                    }
                    const auto snapshot = queue.event_snapshot(found->logical);
                    if (!snapshot) {
                        return CUDA_ERROR_INVALID_HANDLE;
                    }
                    if (*snapshot) {
                        dependencies.push_back(*snapshot);
                    }
                }
                reservation = queue.submit(binding.logical,
                                           CudaQueuedNativeReservation{state.resource_lifetime},
                                           dependencies);
                if (!reservation) {
                    return CUDA_ERROR_OUT_OF_MEMORY;
                }
                if (completion_observer) {
                    const auto status =
                        completion_observer->native_reservation_accepted(reservation);
                    if (status != CUDA_SUCCESS) {
                        return queue.cancel_stopped(reservation) ? status : CUDA_ERROR_UNKNOWN;
                    }
                }
                if (!state.coordinator->track(
                        queue, reservation,
                        {binding.priority, 1, state.options.should_trace_native_timings})) {
                    return queue.cancel_stopped(reservation) ? CUDA_ERROR_OUT_OF_MEMORY
                                                             : CUDA_ERROR_UNKNOWN;
                }
                return CUDA_SUCCESS;
            });
            if (result != CUDA_SUCCESS) {
                return result;
            }
        }
        timing.finish_phase(NativeTimingPhase::kReservation);
        if (!is_execution) {
            timing.finish_phase(NativeTimingPhase::kAdmission);
        }
        CudaNativeBoundaryToken boundary;
        for (;;) {
            if (is_execution && !execution) {
                result = timing.measure_admission(NativeAdmissionPart::kOwnerProbe, [&] {
                    const auto closed = state.close_epoch(false);
                    if (closed != CUDA_SUCCESS) return closed;
                    bool has_transaction = false;
                    const auto paused = [&] {
                        CUcontext caller{};
                        const auto current =
                            state.dispatch.worker.queue.context_get_current(&caller);
                        if (current != CUDA_SUCCESS || caller != state.context) {
                            return current == CUDA_SUCCESS ? CUDA_ERROR_INVALID_CONTEXT : current;
                        }
                        // The owner advances control and physical proof itself;
                        // the producer submits once, rather than polling it.
                        return state.worker->await_native_reservation(
                            reservation, boundary,
                            [&](CudaOperationQueue& queue) {
                                const auto progressed = state.coordinator->progress(queue);
                                if (progressed != CUDA_SUCCESS) return progressed;
                                const auto status = queue.query(reservation);
                                if (status != CUDA_SUCCESS && status != CUDA_ERROR_NOT_READY) {
                                    return status;
                                }
                                execution =
                                    state.coordinator->native_reservation_grant(queue, reservation);
                                const auto control_status = state.coordinator->control_status();
                                if (control_status != CUDA_SUCCESS) return control_status;
                                return execution ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
                            },
                            [&](CudaOperationQueue&) {
                                // Do not hold frontend exclusion while awaiting
                                // authorization or dependencies. An admitted SDK
                                // caller may still need this owner to publish.
                                const std::lock_guard lock(state.mutex);
                                if (state.status != CUDA_SUCCESS || state.is_closing)
                                    return state.is_closing ? CUDA_ERROR_DEINITIALIZED
                                                            : state.status;
                                if (state.is_transaction || state.active_calls != 0)
                                    return CUDA_ERROR_NOT_READY;
                                state.is_transaction = true;
                                has_transaction = true;
                                return CUDA_SUCCESS;
                            });
                    }();
                    if (paused != CUDA_SUCCESS && has_transaction) {
                        {
                            const std::lock_guard lock(state.mutex);
                            state.is_transaction = false;
                        }
                        state.condition.notify_all();
                    }
                    return paused;
                });
                if (execution) {
                    timing.finish_phase(NativeTimingPhase::kAdmission);
                }
            } else {
                result = state.try_pause([&] {
                    return is_execution
                               ? state.worker->try_pause_native_reservation(reservation, boundary)
                               : state.worker->try_pause_native(boundary);
                });
            }
            if (result == CUDA_SUCCESS) {
                break;
            }
            if (result != CUDA_ERROR_NOT_READY) {
                return result;
            }
            if (is_execution && !execution) {
                timing.retry(true);
                (void)timing.measure_admission(NativeAdmissionPart::kSignalWait, [&] {
                    // Only competing frontend ownership can reject the wait
                    // before acceptance. An accepted owner wait never polls here.
                    std::this_thread::sleep_for(state.options.worker.poll_interval);
                    return false;
                });
            } else {
                timing.retry(false);
                std::this_thread::sleep_for(state.options.worker.poll_interval);
            }
        }
        timing.finish_phase(NativeTimingPhase::kPause);
        std::optional<NativeExecutionHeartbeat::Session> heartbeat;
        if (execution) {
            if (!state.native_heartbeat) {
                state.native_heartbeat = std::make_unique<NativeExecutionHeartbeat>(
                    [gate = state.gate](control::TaskExecutionIdentity identity) {
                        auto result = gate->heartbeat(identity);
                        if (gate->owner_pump_snapshot().is_enabled) {
                            while (result.valid()) {
                                if (gate->owner_pump_snapshot().is_quarantined) return false;
                                if (result.wait_for(std::chrono::milliseconds{1}) ==
                                    std::future_status::ready)
                                    break;
                            }
                        }
                        return result.valid() && result.get();
                    });
            }
            heartbeat.emplace(*state.native_heartbeat, *execution);
            if (!heartbeat->valid()) {
                state.fail_closed(CUDA_ERROR_UNKNOWN);
                return CUDA_ERROR_UNKNOWN;
            }
        }
        timing.finish_phase(NativeTimingPhase::kHeartbeatStart);
        CUresult native_status = CUDA_ERROR_UNKNOWN;
        try {
            native_status = operation();
        } catch (...) {
            state.fail_closed(CUDA_ERROR_UNKNOWN);
            return CUDA_ERROR_UNKNOWN;
        }
        timing.finish_phase(NativeTimingPhase::kOriginal);
        lease->was_recorded = true;
        result = state.dispatch.worker.queue.event_record(lease->event(), binding.original);
        if (result != CUDA_SUCCESS) {
            state.fail_closed(result);
            return result;
        }
        timing.finish_phase(NativeTimingPhase::kFence);
        if (heartbeat && !heartbeat->stop()) {
            state.fail_closed(CUDA_ERROR_UNKNOWN);
            return CUDA_ERROR_UNKNOWN;
        }
        timing.finish_phase(NativeTimingPhase::kHeartbeatStop);
        result = state.worker->finish_native(boundary, [&](CudaOperationQueue& queue) {
            const CudaQueuedNativeFence fence{lease->event(), lease, completion_observer};
            if (is_execution) {
                if (!state.coordinator->commit_native_reservation(queue, reservation, *execution,
                                                                  fence, native_status)) {
                    return CUDA_ERROR_UNKNOWN;
                }
            } else if (!queue.submit(binding.logical, fence)) {
                return CUDA_ERROR_OUT_OF_MEMORY;
            }
            lease->has_queue_owner = true;
            return CUDA_SUCCESS;
        });
        if (result != CUDA_SUCCESS) {
            state.fail_closed(result);
            return result;
        }
        {
            const std::lock_guard lock(state.mutex);
            state.is_transaction = false;
        }
        state.condition.notify_all();
        timing.finish_phase(NativeTimingPhase::kCommit);
        timing.report(is_execution, execution, native_status);
        return native_status;
    } catch (...) {
        // In particular, allocation failures after pause cannot authorize
        // automatic thaw or replay. The owner remains retained for recovery.
        implementation_->fail_closed(CUDA_ERROR_UNKNOWN);
        return CUDA_ERROR_UNKNOWN;
    }
}

CUresult CudaCooperativeFrontend::pause(CudaNativeBoundaryToken& boundary) noexcept {
    try {
        auto& state = *implementation_;
        if (Implementation::Permit::is_admitted(state)) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        Implementation::StreamBinding legacy;
        auto result = state.stream_binding(CU_STREAM_LEGACY, false, legacy);
        if (result != CUDA_SUCCESS) {
            return result;
        }
        for (;;) {
            result = state.try_pause([&] { return state.worker->try_pause_native(boundary); });
            if (result == CUDA_SUCCESS) {
                const std::lock_guard lock(state.mutex);
                state.is_pause_finishing = false;
                return CUDA_SUCCESS;
            }
            if (result != CUDA_ERROR_NOT_READY || state.worker->status() != CUDA_SUCCESS) {
                return result;
            }
            std::this_thread::sleep_for(state.options.worker.poll_interval);
        }
    } catch (...) {
        implementation_->fail_closed(CUDA_ERROR_UNKNOWN);
        return CUDA_ERROR_UNKNOWN;
    }
}

bool CudaCooperativeFrontend::native_boundary_is_paused(
    const CudaNativeBoundaryToken& boundary) const noexcept {
    const auto& state = *implementation_;
    return state.worker && state.worker->native_boundary_is_paused(boundary);
}

CUresult CudaCooperativeFrontend::finish_pause(
    const CudaNativeBoundaryToken& boundary, const NativeOperation& prove_native_context) noexcept {
    try {
        auto& state = *implementation_;
        if (!prove_native_context || !state.worker ||
            !state.worker->native_boundary_is_paused(boundary)) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        {
            const std::lock_guard lock(state.mutex);
            if (!state.is_transaction || state.is_pause_finishing || state.status != CUDA_SUCCESS) {
                return CUDA_ERROR_INVALID_VALUE;
            }
            state.is_pause_finishing = true;
        }
        CUcontext caller = nullptr;
        auto result = state.dispatch.worker.queue.context_get_current(&caller);
        if (result != CUDA_SUCCESS || caller != state.context) {
            const std::lock_guard lock(state.mutex);
            state.is_pause_finishing = false;
            return result == CUDA_SUCCESS ? CUDA_ERROR_INVALID_CONTEXT : result;
        }
        result = prove_native_context();
        if (result != CUDA_SUCCESS) {
            state.fail_closed(result);
            return result;
        }
        result = state.dispatch.worker.queue.event_record(state.pause_fence, CU_STREAM_LEGACY);
        if (result != CUDA_SUCCESS) {
            state.fail_closed(result);
            return result;
        }
        result = state.worker->finish_native(boundary, [&](CudaOperationQueue& queue) {
            const auto legacy = std::find_if(
                state.streams.begin(), state.streams.end(),
                [](const auto& stream) { return stream.original == CU_STREAM_LEGACY; });
            return legacy != state.streams.end() &&
                           queue.submit(
                               legacy->logical,
                               CudaQueuedNativeFence{state.pause_fence, state.resource_lifetime})
                       ? CUDA_SUCCESS
                       : CUDA_ERROR_UNKNOWN;
        });
        if (result != CUDA_SUCCESS) {
            state.fail_closed(result);
            return result;
        }
        {
            const std::lock_guard lock(state.mutex);
            state.is_transaction = false;
            state.is_pause_finishing = false;
        }
        state.condition.notify_all();
        return CUDA_SUCCESS;
    } catch (...) {
        implementation_->fail_closed(CUDA_ERROR_UNKNOWN);
        return CUDA_ERROR_UNKNOWN;
    }
}

void CudaCooperativeFrontend::fail_native_boundary(CUresult error) noexcept {
    auto& state = *implementation_;
    state.fail_closed(error == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : error);
}

CUresult CudaCooperativeFrontend::retire_paused_event(const CudaNativeBoundaryToken& boundary,
                                                      CUevent event, bool should_retire) noexcept {
    try {
        auto& state = *implementation_;
        if (!state.worker || !event) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        {
            const std::lock_guard lock(state.mutex);
            if (!state.is_transaction || state.is_pause_finishing || state.status != CUDA_SUCCESS) {
                return CUDA_ERROR_INVALID_VALUE;
            }
        }
        return state.worker->invoke_drained_paused(boundary, [&](CudaOperationQueue& queue) {
            const auto found =
                std::find_if(state.events.begin(), state.events.end(),
                             [&](const auto& value) { return value.original == event; });
            if (found == state.events.end()) {
                return CUDA_ERROR_INVALID_HANDLE;
            }
            if (should_retire) {
                if (!queue.close_event(found->logical)) {
                    return CUDA_ERROR_INVALID_HANDLE;
                }
                state.events.erase(found);
            }
            return CUDA_SUCCESS;
        });
    } catch (...) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
}

CUresult CudaCooperativeFrontend::close_event(CUevent event) noexcept {
    try {
        auto& state = *implementation_;
        const auto result = synchronize_context();
        if (result != CUDA_SUCCESS) {
            return result;
        }
        return state.command([&](CudaOperationQueue& queue) {
            const auto found =
                std::find_if(state.events.begin(), state.events.end(),
                             [&](const auto& value) { return value.original == event; });
            if (found == state.events.end() || !queue.close_event(found->logical)) {
                return CUDA_ERROR_INVALID_HANDLE;
            }
            state.events.erase(found);
            return CUDA_SUCCESS;
        });
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}

CUresult CudaCooperativeFrontend::retire_paused_stream(const CudaNativeBoundaryToken& boundary,
                                                       CUstream stream,
                                                       bool should_retire) noexcept {
    try {
        auto& state = *implementation_;
        if (!state.worker || !stream || stream == CU_STREAM_LEGACY ||
            stream == CU_STREAM_PER_THREAD) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        {
            const std::lock_guard lock(state.mutex);
            if (!state.is_transaction || state.is_pause_finishing || state.status != CUDA_SUCCESS) {
                return CUDA_ERROR_INVALID_VALUE;
            }
        }
        return state.worker->invoke_drained_paused(boundary, [&](CudaOperationQueue& queue) {
            const auto found =
                std::find_if(state.streams.begin(), state.streams.end(),
                             [&](const auto& value) { return value.original == stream; });
            if (found == state.streams.end()) {
                return CUDA_ERROR_INVALID_HANDLE;
            }
            if (should_retire) {
                if (!queue.close_stream(found->logical)) {
                    return CUDA_ERROR_INVALID_HANDLE;
                }
                state.streams.erase(found);
            }
            return CUDA_SUCCESS;
        });
    } catch (...) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
}

CUresult CudaCooperativeFrontend::close_stream(CUstream stream) noexcept {
    try {
        auto& state = *implementation_;
        const auto result = synchronize_context();
        if (result != CUDA_SUCCESS) {
            return result;
        }
        return state.command([&](CudaOperationQueue& queue) {
            const auto found =
                std::find_if(state.streams.begin(), state.streams.end(),
                             [&](const auto& value) { return value.original == stream; });
            if (found == state.streams.end() || !queue.close_stream(found->logical)) {
                return CUDA_ERROR_INVALID_HANDLE;
            }
            state.streams.erase(found);
            return CUDA_SUCCESS;
        });
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}

CUresult CudaCooperativeFrontend::shutdown() noexcept {
    auto& state = *implementation_;
    try {
        {
            const std::lock_guard lock(state.mutex);
            if (state.checkpoint) {
                return CUDA_ERROR_NOT_READY;
            }
        }
        if (state.worker) {
            for (;;) {
                const auto drained = state.command([](CudaOperationQueue& queue) {
                    return queue.pending_count() == 0 ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
                });
                if (drained == CUDA_SUCCESS) {
                    break;
                }
                if (drained != CUDA_ERROR_NOT_READY) {
                    return drained;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds{1});
            }
            {
                const std::lock_guard lock(state.mutex);
                if (state.active_calls != 0 || state.is_transaction) {
                    return CUDA_ERROR_NOT_READY;
                }
                state.is_closing = true;
            }
            const auto stopped = state.worker->shutdown();
            if (stopped != CUDA_SUCCESS) {
                state.fail_closed(stopped);
                return stopped;
            }
            state.worker.reset();
        }
        state.stop_epoch_watcher();
        state.streams.clear();
        state.events.clear();
        state.status = CUDA_ERROR_DEINITIALIZED;
        return state.cleanup();
    } catch (...) {
        state.fail_closed(CUDA_ERROR_UNKNOWN);
        return CUDA_ERROR_UNKNOWN;
    }
}

CUresult CudaCooperativeFrontend::checkpoint(CudaWorkerCheckpoint& output) noexcept {
    auto& state = *implementation_;
    if (output || Implementation::Permit::is_admitted(state)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    try {
        const auto result = state.try_pause([&] { return state.worker->checkpoint(output); });
        if (result == CUDA_SUCCESS) {
            const std::lock_guard lock(state.mutex);
            state.checkpoint = output;
        } else if (result != CUDA_ERROR_NOT_READY && state.worker &&
                   state.worker->status() != CUDA_SUCCESS) {
            state.fail_closed(result);
        }
        return result;
    } catch (...) {
        state.fail_closed(CUDA_ERROR_UNKNOWN);
        return CUDA_ERROR_UNKNOWN;
    }
}

CUresult CudaCooperativeFrontend::resume_checkpoint(
    const CudaWorkerCheckpoint& checkpoint) noexcept {
    auto& state = *implementation_;
    {
        const std::lock_guard lock(state.mutex);
        if (!state.checkpoint || !state.is_transaction || state.status != CUDA_SUCCESS ||
            !state.pool) {
            return CUDA_ERROR_INVALID_VALUE;
        }
    }
    const auto result = state.worker->resume_checkpoint(checkpoint);
    if (result != CUDA_SUCCESS) {
        if (!state.worker->owns_checkpoint(state.checkpoint)) {
            state.fail_closed(result);
        }
        return result;
    }
    {
        const std::lock_guard lock(state.mutex);
        state.checkpoint = {};
        state.is_transaction = false;
    }
    state.condition.notify_all();
    return CUDA_SUCCESS;
}

CUresult CudaCooperativeFrontend::release_checkpoint_progress(
    const CudaWorkerCheckpoint& checkpoint) noexcept {
    auto& state = *implementation_;
    {
        const std::lock_guard lock(state.mutex);
        if (!state.checkpoint || !state.is_transaction || state.status != CUDA_SUCCESS ||
            !state.pool || !state.worker->owns_checkpoint(checkpoint)) {
            return CUDA_ERROR_INVALID_VALUE;
        }
    }
    CUcontext current{};
    const auto binding = state.dispatch.worker.queue.context_get_current(&current);
    if (binding != CUDA_SUCCESS || current != state.context) {
        return binding == CUDA_SUCCESS ? CUDA_ERROR_INVALID_CONTEXT : binding;
    }
    const auto result = state.pool->release();
    if (result != CUDA_SUCCESS) {
        // Even a free error may follow physical release. Do not reissue it.
        if (result != CUDA_ERROR_NOT_READY) {
            state.fail_closed(result);
        }
        return result;
    }
    state.pool.reset();
    return CUDA_SUCCESS;
}

CUresult CudaCooperativeFrontend::restore_checkpoint_progress(
    const CudaWorkerCheckpoint& checkpoint) noexcept {
    auto& state = *implementation_;
    {
        const std::lock_guard lock(state.mutex);
        if (!state.checkpoint || !state.is_transaction || state.status != CUDA_SUCCESS ||
            state.pool || !state.worker->owns_checkpoint(checkpoint)) {
            return CUDA_ERROR_INVALID_VALUE;
        }
    }
    // Reject a wrong binding before acquiring any new pool ownership. A partial
    // native initialization is retained and cannot be automatically retried.
    CUcontext current{};
    const auto binding = state.dispatch.worker.queue.context_get_current(&current);
    if (binding != CUDA_SUCCESS || current != state.context) {
        return binding == CUDA_SUCCESS ? CUDA_ERROR_INVALID_CONTEXT : binding;
    }
    try {
        state.pool = std::make_shared<CudaCooperativePool>(state.context, state.dispatch.pool,
                                                           state.options.progress_slots,
                                                           state.options.audit_block_capacity);
        const auto result = state.pool->initialize();
        if (result != CUDA_SUCCESS) {
            state.fail_closed(result);
        }
        return result;
    } catch (...) {
        state.fail_closed(CUDA_ERROR_OUT_OF_MEMORY);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
}

CUresult CudaCooperativeFrontend::retire_checkpoint_after_context_destroy(
    const CudaWorkerCheckpoint& checkpoint) noexcept {
    auto& state = *implementation_;
    {
        const std::lock_guard lock(state.mutex);
        if (!state.checkpoint || !state.is_transaction || state.status != CUDA_SUCCESS ||
            state.pool || !state.worker->owns_checkpoint(checkpoint)) {
            return CUDA_ERROR_INVALID_VALUE;
        }
    }
    const auto result = state.worker->retire_checkpoint_after_context_destroy(checkpoint);
    if (result != CUDA_SUCCESS) {
        return result;
    }
    state.stop_epoch_watcher();
    state.worker.reset();
    state.streams.clear();
    state.events.clear();
    state.imports->events.clear();
    state.imports->occupied.clear();
    state.thread_streams.clear();
    state.occupied_proxies.clear();
    state.pause_fence = nullptr;
    {
        const std::lock_guard lock(state.mutex);
        state.checkpoint = {};
        state.is_closing = true;
        state.status = CUDA_ERROR_DEINITIALIZED;
        state.is_transaction = false;
    }
    state.condition.notify_all();
    return CUDA_SUCCESS;
}

CudaCoordinationStats CudaCooperativeFrontend::stats() noexcept {
    CudaCoordinationStats output;
    try {
        auto& state = *implementation_;
        if (state.worker) {
            const auto result = state.command([&](CudaOperationQueue&) {
                output = state.coordinator->stats();
                return CUDA_SUCCESS;
            });
            if (result != CUDA_SUCCESS) {
                return {};
            }
        } else if (state.coordinator) {
            output = state.coordinator->stats();
        }
    } catch (...) {
        return {};
    }
    return output;
}

}  // namespace glimmer::interceptor
