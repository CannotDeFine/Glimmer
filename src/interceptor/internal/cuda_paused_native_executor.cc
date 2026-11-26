#include "cuda_paused_native_executor.h"
#include "owner_control_pump.h"

#include "diagnostics.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace glimmer::interceptor {
namespace {
template <typename Value>
bool ready(const std::future<Value>& future) {
    return future.valid() &&
           future.wait_for(std::chrono::milliseconds::zero()) == std::future_status::ready;
}
const std::shared_ptr<const void>& native_caller() {
    static thread_local const std::shared_ptr<const void> identity = std::make_shared<int>(0);
    return identity;
}
class NativeScope final {
   public:
    explicit NativeScope(const CudaPausedNativeDispatch& dispatch)
        : dispatch_(dispatch),
          was_entered_(dispatch.enter_native_scope && dispatch.enter_native_scope()) {}
    ~NativeScope() {
        if (dispatch_.leave_native_scope) {
            dispatch_.leave_native_scope(was_entered_);
        }
    }
    NativeScope(const NativeScope&) = delete;
    NativeScope& operator=(const NativeScope&) = delete;

   private:
    const CudaPausedNativeDispatch& dispatch_;
    bool was_entered_;
};
}  // namespace
struct CudaPausedNativeExecutor::Implementation {
    enum class Phase {
        kPreparing,
        kSubmitting,
        kPending,
        kClaiming,
        kReady,
        kIssuing,
        kRunning,
        kReporting,
        kNeedsReport,
        kTerminal,
        kRejected,
        kUncertain
    };
    struct Entry {
        CUcontext context{};
        CUstream stream{};
        bool is_ptds{};
        std::weak_ptr<const void> caller;
        std::shared_ptr<void> resources;
        Phase phase{Phase::kPreparing};
        core::TaskId task{};
        control::TaskExecutionIdentity execution{};
        CUevent event{};
        CUresult original{CUDA_SUCCESS};
        CUresult error{CUDA_SUCCESS};
        std::future<std::optional<core::TaskId>> submission;
        std::future<control::CooperativeClaim> claim;
        std::future<bool> report;
        std::future<bool> heartbeat;
        std::chrono::steady_clock::time_point last_heartbeat{};
        std::chrono::steady_clock::time_point next_claim{};
        std::chrono::steady_clock::time_point next_query{};
    };
    std::shared_ptr<control::AsyncCooperativeLaunchGate> gate;
    std::shared_ptr<control::AsyncCooperativeLaunchGate::ProgressSignal> progress_signal;
    const CudaPausedNativeDispatch dispatch;
    const ContextOperation in_context;
    const std::size_t limit;
    const std::shared_ptr<const void> identity{std::make_shared<int>(0)};
    std::mutex mutex;
    std::condition_variable condition;
    std::vector<std::shared_ptr<Entry>> entries;
    std::vector<std::shared_ptr<Entry>> snapshot;
    std::atomic<CUresult> failure{CUDA_SUCCESS};
    CudaPausedNativeStats stats;
    bool is_closing{};
    std::thread thread;
    OwnerControlPump control;

    Implementation(std::shared_ptr<control::AsyncCooperativeLaunchGate> control_gate,
                   CudaPausedNativeDispatch native, ContextOperation activate, std::size_t capacity)
        : gate(std::move(control_gate)),
          dispatch(native),
          in_context(std::move(activate)),
          limit(capacity),
          control(gate) {
        if (!gate || !in_context || !dispatch.event_create || !dispatch.event_record ||
            !dispatch.event_query || !dispatch.event_destroy || !limit || limit > 65536 ||
            (bool(dispatch.enter_native_scope) != bool(dispatch.leave_native_scope))) {
            throw std::invalid_argument("invalid paused native executor configuration");
        }
        entries.reserve(limit);
        snapshot.reserve(limit);
        progress_signal = gate->subscribe_progress();
        if (!progress_signal) {
            throw std::runtime_error("paused native progress subscription unavailable");
        }
        thread = std::thread([this] { run(); });
    }
    void notify() noexcept {
        condition.notify_all();
        progress_signal->notify();
    }
    void uncertain(Entry& entry, CUresult error) noexcept {
        if (error == CUDA_SUCCESS || error == CUDA_ERROR_NOT_READY) {
            error = CUDA_ERROR_UNKNOWN;
        }
        CUresult expected = CUDA_SUCCESS;
        failure.compare_exchange_strong(expected, error);
        {
            const std::lock_guard lock(mutex);
            entry.error = error;
            entry.phase = Phase::kUncertain;
        }
        notify();
    }
    void phase(Entry& entry, Phase value) {
        {
            const std::lock_guard lock(mutex);
            entry.phase = value;
        }
        notify();
    }
    CUresult private_event_operation(const NativeOperation& operation) {
        if (!dispatch.thread_exchange_capture_mode) {
            return operation();
        }
        CUstreamCaptureMode mode = CU_STREAM_CAPTURE_MODE_RELAXED;
        const auto exchanged = dispatch.thread_exchange_capture_mode(&mode);
        if (exchanged != CUDA_SUCCESS) {
            return exchanged;
        }
        CUresult result{CUDA_ERROR_UNKNOWN};
        try {
            result = operation();
        } catch (...) {
            // Restore caller mode even when the private operation throws.
        }
        const auto restored = dispatch.thread_exchange_capture_mode(&mode);
        return restored == CUDA_SUCCESS ? result : restored;
    }
    bool heartbeat(Entry& entry) {
        if (entry.heartbeat.valid()) {
            if (!ready(entry.heartbeat)) {
                return true;
            }
            if (!entry.heartbeat.get()) {
                return false;
            }
            entry.last_heartbeat = std::chrono::steady_clock::now();
            const std::lock_guard lock(mutex);
            ++stats.heartbeats;
        }
        if (std::chrono::steady_clock::now() - entry.last_heartbeat >=
            std::chrono::milliseconds{25}) {
            entry.heartbeat = gate->heartbeat(entry.execution);
            return entry.heartbeat.valid();
        }
        return true;
    }
    void progress(Entry& entry) {
        Phase current;
        {
            const std::lock_guard lock(mutex);
            current = entry.phase;
        }
        switch (current) {
            case Phase::kSubmitting:
                if (ready(entry.submission)) {
                    const auto task = entry.submission.get();
                    if (!task) {
                        uncertain(entry, CUDA_ERROR_UNKNOWN);
                        return;
                    }
                    entry.task = *task;
                    phase(entry, Phase::kPending);
                }
                break;
            case Phase::kPending:
                if (std::chrono::steady_clock::now() < entry.next_claim) {
                    return;
                }
                entry.claim = gate->claim(entry.task);
                if (!entry.claim.valid()) {
                    uncertain(entry, CUDA_ERROR_UNKNOWN);
                    return;
                }
                phase(entry, Phase::kClaiming);
                break;
            case Phase::kClaiming:
                if (ready(entry.claim)) {
                    const auto claim = entry.claim.get();
                    if (claim.status == control::CooperativeClaimStatus::kPending) {
                        entry.next_claim =
                            std::chrono::steady_clock::now() + std::chrono::milliseconds{1};
                        phase(entry, Phase::kPending);
                    } else if (claim.status != control::CooperativeClaimStatus::kGranted ||
                               claim.execution.task_id != entry.task ||
                               !claim.execution.generation) {
                        uncertain(entry, CUDA_ERROR_UNKNOWN);
                    } else {
                        entry.execution = claim.execution;
                        entry.last_heartbeat = std::chrono::steady_clock::now();
                        {
                            const std::lock_guard lock(mutex);
                            ++stats.grants;
                        }
                        phase(entry, Phase::kReady);
                    }
                }
                break;
            case Phase::kReady:
            case Phase::kIssuing:
            case Phase::kRunning:
                if (!heartbeat(entry)) {
                    uncertain(entry, CUDA_ERROR_UNKNOWN);
                    return;
                }
                if (current == Phase::kRunning) {
                    if (std::chrono::steady_clock::now() < entry.next_query) {
                        return;
                    }
                    const NativeScope scope(dispatch);
                    const auto query = private_event_operation([&] {
                        return in_context(entry.context,
                                          [&] { return dispatch.event_query(entry.event); });
                    });
                    if (query == CUDA_ERROR_NOT_READY) {
                        entry.next_query =
                            std::chrono::steady_clock::now() + std::chrono::milliseconds{1};
                        return;
                    }
                    if (query != CUDA_SUCCESS) {
                        uncertain(entry, query);
                        return;
                    }
                    const auto destroyed = private_event_operation([&] {
                        return in_context(entry.context,
                                          [&] { return dispatch.event_destroy(entry.event); });
                    });
                    if (destroyed != CUDA_SUCCESS) {
                        uncertain(entry, destroyed);
                        return;
                    }
                    entry.event = nullptr;
                    phase(entry, Phase::kNeedsReport);
                }
                break;
            case Phase::kNeedsReport:
                // A heartbeat already queued before proof must settle before
                // terminal reporting. Never let a late heartbeat hit a retired task.
                if (entry.heartbeat.valid()) {
                    if (!ready(entry.heartbeat)) {
                        return;
                    }
                    if (!entry.heartbeat.get()) {
                        uncertain(entry, CUDA_ERROR_UNKNOWN);
                        return;
                    }
                    {
                        const std::lock_guard lock(mutex);
                        ++stats.heartbeats;
                    }
                }
                entry.report = entry.original == CUDA_SUCCESS ? gate->complete(entry.execution)
                                                              : gate->fail(entry.execution);
                if (!entry.report.valid()) {
                    uncertain(entry, CUDA_ERROR_UNKNOWN);
                    return;
                }
                phase(entry, Phase::kReporting);
                break;
            case Phase::kReporting:
                if (ready(entry.report)) {
                    if (!entry.report.get()) {
                        uncertain(entry, CUDA_ERROR_UNKNOWN);
                        return;
                    }
                    // Release supplied CPU-only owners outside this mutex;
                    // a retained capability must not pin completed resources.
                    entry.resources.reset();
                    {
                        const std::lock_guard lock(mutex);
                        if (entry.original == CUDA_SUCCESS) {
                            ++stats.completed;
                        } else {
                            ++stats.failed;
                        }
                    }
                    phase(entry, Phase::kTerminal);
                }
                break;
            default:
                break;
        }
    }
    void run() noexcept {
        for (;;) {
            const auto cpu_result = control.progress();
            {
                std::unique_lock lock(mutex);
                if (!control.enabled())
                    condition.wait(lock, [&] { return is_closing || !entries.empty(); });
                if (is_closing) {
                    break;
                }
                snapshot.assign(entries.begin(), entries.end());
            }
            for (const auto& entry : snapshot) {
                try {
                    if (cpu_result == CUDA_SUCCESS)
                        progress(*entry);
                    else
                        uncertain(*entry, cpu_result);
                } catch (...) {
                    uncertain(*entry, CUDA_ERROR_UNKNOWN);
                }
            }
            if (cpu_result != CUDA_SUCCESS) {
                auto expected = CUDA_SUCCESS;
                failure.compare_exchange_strong(expected, cpu_result);
                notify();
                break;
            }
            {
                std::unique_lock lock(mutex);
                std::erase_if(entries, [](const auto& entry) {
                    return entry->phase == Phase::kTerminal || entry->phase == Phase::kRejected;
                });
                lock.unlock();
                snapshot.clear();
                condition.notify_all();
            }
            // Future publication and caller handoff wake ready phases. Pending
            // claim/query deadlines still bound retries under unrelated wakeups.
            if (control.enabled()) {
                bool is_empty = false;
                {
                    const std::lock_guard lock(mutex);
                    is_empty = entries.empty();
                }
                const auto delay = std::chrono::milliseconds{is_empty ? 25 : 1};
                if (const auto waited = control.wait_until(std::chrono::steady_clock::now() + delay,
                                                           progress_signal);
                    waited != CUDA_SUCCESS) {
                    auto expected = CUDA_SUCCESS;
                    failure.compare_exchange_strong(expected, waited);
                    notify();
                }
            } else
                (void)progress_signal->wait_for(std::chrono::milliseconds{1});
        }
        if (const auto retired = control.retire(); retired != CUDA_SUCCESS) {
            auto expected = CUDA_SUCCESS;
            failure.compare_exchange_strong(expected, retired);
            notify();
        }
    }
};
CudaPausedNativeExecutor::CudaPausedNativeExecutor(
    std::shared_ptr<control::AsyncCooperativeLaunchGate> gate, CudaPausedNativeDispatch dispatch,
    ContextOperation in_context, std::size_t max_entries)
    : implementation_(std::make_unique<Implementation>(std::move(gate), dispatch,
                                                       std::move(in_context), max_entries)) {}
CudaPausedNativeExecutor::~CudaPausedNativeExecutor() {
    auto& state = *implementation_;
    {
        const std::lock_guard lock(state.mutex);
        state.is_closing = true;
    }
    state.notify();
    state.thread.join();
    if (!state.entries.empty() || state.failure.load() != CUDA_SUCCESS) {
        report_diagnostic("[glimmer] unfinished paused native execution retained\n");
        (void)implementation_.release();
    }
}
CUresult CudaPausedNativeExecutor::acquire(CUcontext context, CUstream actual_stream, bool is_ptds,
                                           std::shared_ptr<void> resources,
                                           control::CooperativeLaunchRequest request,
                                           CudaPausedNativePermit& output) noexcept {
    if (!context || !resources || output) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    auto& state = *implementation_;
    if (const auto cpu = state.control.status(); cpu != CUDA_SUCCESS) return cpu;
    if (is_ptds && !state.dispatch.event_record_per_thread) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    std::shared_ptr<Implementation::Entry> entry;
    bool was_published{};
    try {
        entry = std::make_shared<Implementation::Entry>();
        entry->context = context;
        entry->stream = actual_stream;
        entry->is_ptds = is_ptds;
        entry->caller = native_caller();
        entry->resources = std::move(resources);
        {
            const std::lock_guard lock(state.mutex);
            if (state.is_closing) {
                return CUDA_ERROR_DEINITIALIZED;
            }
            if (state.failure.load() != CUDA_SUCCESS) {
                return state.failure.load();
            }
            if (state.entries.size() >= state.limit) {
                return CUDA_ERROR_OUT_OF_MEMORY;
            }
            state.entries.push_back(entry);
            was_published = true;
        }
        auto submission = state.gate->submit(request);
        {
            const std::lock_guard lock(state.mutex);
            entry->submission = std::move(submission);
            if (!entry->submission.valid()) {
                entry->phase = Implementation::Phase::kRejected;
                entry->error = CUDA_ERROR_OUT_OF_MEMORY;
            } else {
                entry->phase = Implementation::Phase::kSubmitting;
            }
        }
        state.notify();
        std::unique_lock lock(state.mutex);
        state.condition.wait(lock, [&] {
            return entry->phase == Implementation::Phase::kReady ||
                   entry->phase == Implementation::Phase::kRejected ||
                   entry->phase == Implementation::Phase::kUncertain || state.is_closing ||
                   state.failure.load() != CUDA_SUCCESS;
        });
        if (state.failure.load() != CUDA_SUCCESS) {
            return state.failure.load();
        }
        if (entry->phase != Implementation::Phase::kReady) {
            return state.is_closing ? CUDA_ERROR_DEINITIALIZED : entry->error;
        }
        output.owner_ = state.identity;
        output.entry_ = std::move(entry);
        return CUDA_SUCCESS;
    } catch (...) {
        if (was_published) {
            state.uncertain(*entry, CUDA_ERROR_UNKNOWN);
        }
        return CUDA_ERROR_UNKNOWN;
    }
}
CudaPausedNativeResult CudaPausedNativeExecutor::issue(const CudaPausedNativePermit& permit,
                                                       const NativeOperation& operation) noexcept {
    auto& state = *implementation_;
    if (const auto cpu = state.control.status(); cpu != CUDA_SUCCESS) return {cpu, cpu, false};
    if (!permit || permit.owner_ != state.identity || !operation) {
        return {CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_VALUE, false};
    }
    auto& entry = *static_cast<Implementation::Entry*>(permit.entry_.get());
    bool was_invoked{};
    CUresult original = CUDA_ERROR_UNKNOWN;
    try {
        {
            const std::lock_guard lock(state.mutex);
            const auto failure = state.failure.load();
            if (failure != CUDA_SUCCESS) {
                return {failure, failure, false};
            }
            if (entry.phase != Implementation::Phase::kReady ||
                entry.caller.lock() != native_caller()) {
                return {CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_VALUE, false};
            }
            entry.phase = Implementation::Phase::kIssuing;
        }
        state.notify();
        const NativeScope scope(state.dispatch);
        const auto tracked = state.in_context(entry.context, [&] {
            const auto created = state.private_event_operation(
                [&] { return state.dispatch.event_create(&entry.event, CU_EVENT_DISABLE_TIMING); });
            if (created != CUDA_SUCCESS || !entry.event) {
                return created == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : created;
            }
            was_invoked = true;
            original = operation();
            return state.private_event_operation([&] {
                return (entry.is_ptds ? state.dispatch.event_record_per_thread
                                      : state.dispatch.event_record)(entry.event, entry.stream);
            });
        });
        if (tracked != CUDA_SUCCESS) {
            state.uncertain(entry, tracked);
            return {was_invoked && original != CUDA_SUCCESS ? original : tracked, tracked,
                    was_invoked};
        }
        {
            const std::lock_guard lock(state.mutex);
            entry.original = original;
            if (entry.phase != Implementation::Phase::kIssuing) {
                return {original == CUDA_SUCCESS ? entry.error : original, entry.error,
                        was_invoked};
            }
            entry.phase = Implementation::Phase::kRunning;
        }
        state.notify();
        return {original, CUDA_SUCCESS, was_invoked};
    } catch (...) {
        state.uncertain(entry, CUDA_ERROR_UNKNOWN);
        return {CUDA_ERROR_UNKNOWN, CUDA_ERROR_UNKNOWN, was_invoked};
    }
}
CUresult CudaPausedNativeExecutor::retire_unissued(const CudaPausedNativePermit& permit) noexcept {
    auto& state = *implementation_;
    if (!permit || permit.owner_ != state.identity) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    auto& entry = *static_cast<Implementation::Entry*>(permit.entry_.get());
    try {
        std::unique_lock lock(state.mutex);
        if (entry.phase != Implementation::Phase::kReady) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        entry.original = CUDA_ERROR_NOT_READY;
        entry.phase = Implementation::Phase::kNeedsReport;
        state.notify();
        state.condition.wait(lock, [&] {
            return entry.phase == Implementation::Phase::kTerminal ||
                   entry.phase == Implementation::Phase::kUncertain || state.is_closing;
        });
        return entry.phase == Implementation::Phase::kTerminal ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
    } catch (...) {
        state.uncertain(entry, CUDA_ERROR_UNKNOWN);
        return CUDA_ERROR_UNKNOWN;
    }
}
void CudaPausedNativeExecutor::fail_closed(CUresult error) noexcept {
    if (error == CUDA_SUCCESS || error == CUDA_ERROR_NOT_READY) {
        error = CUDA_ERROR_UNKNOWN;
    }
    {
        // Serialize with each wait's predicate-to-sleep interval. Atomic status
        // alone would allow a notification to be lost before the caller sleeps.
        const std::lock_guard lock(implementation_->mutex);
        CUresult expected{CUDA_SUCCESS};
        implementation_->failure.compare_exchange_strong(expected, error);
    }
    implementation_->notify();
}

CUresult CudaPausedNativeExecutor::drain() noexcept {
    auto& state = *implementation_;
    try {
        std::unique_lock lock(state.mutex);
        state.condition.wait(lock, [&] {
            return state.entries.empty() || state.failure.load() != CUDA_SUCCESS ||
                   state.is_closing;
        });
        return state.is_closing ? CUDA_ERROR_DEINITIALIZED : state.failure.load();
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}
CUresult CudaPausedNativeExecutor::status() const noexcept {
    const auto failure = implementation_->failure.load();
    return failure != CUDA_SUCCESS ? failure : implementation_->control.status();
}
CudaPausedNativeStats CudaPausedNativeExecutor::stats() const noexcept {
    auto& state = *implementation_;
    const std::lock_guard lock(state.mutex);
    auto stats = state.stats;
    stats.retained = state.entries.size();
    return stats;
}
}  // namespace glimmer::interceptor
