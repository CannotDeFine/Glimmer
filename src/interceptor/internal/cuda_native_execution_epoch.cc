#include "cuda_native_execution_epoch.h"
#include "owner_control_pump.h"

#include <algorithm>
#include <atomic>
#include <deque>
#include <future>
#include <limits>
#include <stdexcept>
#include <utility>

namespace glimmer::interceptor {
namespace {
// A std::thread::id can be recycled after caller exit. This TLS lifetime identity
// is initialized only by construction, never on a foreign caller's hot path.
// Owners retain the old allocation, so its identity cannot be reused while an
// epoch survives the constructing thread. No teardown callback invokes CUDA.
thread_local std::shared_ptr<const void> caller_lifetime;
std::shared_ptr<const void> current_caller_lifetime() {
    if (!caller_lifetime) caller_lifetime = std::make_shared<int>(0);
    return caller_lifetime;
}
template <typename Value>
bool ready(std::future<Value>& future) {
    return future.valid() &&
           future.wait_for(std::chrono::seconds::zero()) == std::future_status::ready;
}
bool terminal_phase(CudaNativeEpochPhase phase) noexcept {
    return phase == CudaNativeEpochPhase::kComplete || phase == CudaNativeEpochPhase::kRejected ||
           phase == CudaNativeEpochPhase::kRetained;
}
class NativeEpochCallScope final {
   public:
    explicit NativeEpochCallScope(const CudaNativeEpochDispatch& value) noexcept
        : dispatch(value),
          previous(dispatch.enter_native_scope ? dispatch.enter_native_scope() : false) {}
    ~NativeEpochCallScope() {
        if (dispatch.leave_native_scope) dispatch.leave_native_scope(previous);
    }
    NativeEpochCallScope(const NativeEpochCallScope&) = delete;
    NativeEpochCallScope& operator=(const NativeEpochCallScope&) = delete;

   private:
    const CudaNativeEpochDispatch& dispatch;
    const bool previous;
};
}  // namespace

struct CudaNativeExecutionEpoch::Implementation {
    enum class CallerRecordState { kFree, kReserved, kRecording, kRecorded, kFailed };
    struct Event {
        CUevent handle{};
        std::optional<NativeEpochFence> fence{};
        bool is_recorded{};
        bool is_proven{};
        std::atomic<CallerRecordState> caller_record{CallerRecordState::kFree};
    };
    const CUcontext context;
    const CUstream stream;
    const bool is_caller_fenced;
    const NativeEpochScope scope;
    const control::NativeEpochLaunchRequest request;
    const std::shared_ptr<control::AsyncCooperativeLaunchGate> gate;
    const CudaNativeEpochDispatch dispatch;
    const CudaNativeEpochOptions options;
    const std::thread::id caller;
    const std::shared_ptr<const void> caller_identity;
    std::atomic<CudaNativeEpochPhase> phase{CudaNativeEpochPhase::kPreparing};
    std::atomic<CUresult> status{CUDA_SUCCESS};
    std::atomic<bool> is_closing{false};
    std::atomic<bool> is_shutting_down{false};
    std::atomic<bool> has_unverified_return_context{false};
    std::atomic<NativeExecutionEpoch*> published_model{nullptr};
    std::atomic<std::uint32_t> recorded_prefixes{0};
    std::atomic<std::uint32_t> proven_prefixes{0};
    std::atomic<std::uint32_t> created_events{0};
    std::atomic<std::uint32_t> destroyed_events{0};
    mutable std::mutex metadata_mutex;
    control::NativeEpochGrant grant{};
    std::unique_ptr<NativeExecutionEpoch> model;
    std::deque<Event> events;
    Event* caller_event{};  // actual caller only, from begin through return
    bool has_caller_initialization{};
    bool has_joined_cleanup{};    // externally serialized shutdown only
    bool has_recovery_attempt{};  // externally serialized recovery only
    std::future<control::NativeEpochClaim> acquisition;
    std::future<std::optional<bool>> retreat;
    std::future<bool> heartbeat;
    std::future<bool> terminal;
    std::optional<NativeEpochTerminal> terminal_capability;
    NativeExecutionEpoch::TimePoint acceptance_deadline{};
    NativeExecutionEpoch::TimePoint next_control{};
    NativeExecutionEpoch::TimePoint next_heartbeat{};
    std::thread monitor;
    OwnerControlPump control;
    std::shared_ptr<control::AsyncCooperativeLaunchGate::ProgressSignal> progress_signal;

    Implementation(CUcontext execution_context, CUstream execution_stream,
                   NativeEpochScope execution_scope, control::NativeEpochLaunchRequest settings,
                   std::shared_ptr<control::AsyncCooperativeLaunchGate> control_gate,
                   CudaNativeEpochDispatch native_dispatch, CudaNativeEpochOptions monitor_options)
        : context(execution_context),
          stream(execution_stream ? execution_stream : CU_STREAM_LEGACY),
          is_caller_fenced(stream == CU_STREAM_PER_THREAD),
          scope(execution_scope),
          request(settings),
          gate(std::move(control_gate)),
          dispatch(native_dispatch),
          options(monitor_options),
          caller(std::this_thread::get_id()),
          caller_identity(current_caller_lifetime()),
          control(gate) {
        if (!context || (is_caller_fenced && request.extent.calls_per_prefix != 1) ||
            !scope.context_registration || !scope.stream_registration ||
            !scope.caller_registration || !gate ||
            !control::valid_native_epoch_extent(request.extent) || !dispatch.context_get_current ||
            !dispatch.context_push || !dispatch.context_pop || !dispatch.stream_get_context ||
            !dispatch.stream_is_capturing || !dispatch.event_create || !dispatch.event_record ||
            !dispatch.event_query || !dispatch.event_destroy ||
            static_cast<bool>(dispatch.enter_native_scope) !=
                static_cast<bool>(dispatch.leave_native_scope) ||
            options.poll_interval <= std::chrono::microseconds::zero() ||
            options.poll_interval > std::chrono::seconds{1} ||
            options.control_interval <= std::chrono::microseconds::zero() ||
            options.control_interval > std::chrono::seconds{1} ||
            options.shutdown_grace <= std::chrono::milliseconds::zero() ||
            options.shutdown_grace > std::chrono::seconds{10}) {
            throw std::invalid_argument("invalid physical native epoch owner");
        }
        if (control.enabled()) {
            progress_signal = gate->subscribe_progress();
            if (!progress_signal)
                throw std::runtime_error("native epoch CPU signal subscription rejected");
        }
        events.resize(static_cast<std::size_t>(request.extent.max_pending_prefixes) + 1);
        if (is_caller_fenced) {
            // PTDS belongs to this constructing caller, never the monitor.
            const NativeEpochCallScope native_scope(dispatch);
            try {
                has_caller_initialization = initialize();
            } catch (...) {
                retain(CUDA_ERROR_UNKNOWN);
            }
            if (!has_caller_initialization) {
                static_cast<void>(control.retire());
                return;
            }
        }
        try {
            monitor = std::thread([this] { run(); });
        } catch (...) {
            if (!is_caller_fenced) throw;         // no native effect yet
            retain(CUDA_ERROR_OUT_OF_MEMORY);     // retain caller-created native handles
            static_cast<void>(control.retire());  // no monitor will advance this CPU role
        }
    }
    void fail(CUresult result) noexcept {
        auto expected = CUDA_SUCCESS;
        status.compare_exchange_strong(expected,
                                       result == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : result);
        is_closing.store(true);
        if (auto* execution = published_model.load()) {
            static_cast<void>(execution->request_stop(NativeEpochStopReason::kFailure));
        }
    }
    void retain(CUresult result) noexcept {
        fail(result);
        phase.store(CudaNativeEpochPhase::kRetained);
    }
    bool advance_running_phase(CudaNativeEpochPhase next) noexcept {
        auto current = phase.load();
        while (!terminal_phase(current)) {
            if (phase.compare_exchange_weak(current, next)) return true;
        }
        return false;
    }
    void close() noexcept {
        is_closing.store(true);
        if (auto* execution = published_model.load()) {
            static_cast<void>(execution->request_stop());
        }
        if (progress_signal) progress_signal->notify();
    }
    bool initialize() {
        CUcontext actual{};
        auto result = dispatch.context_get_current(&actual);
        if (result == CUDA_SUCCESS && actual != context) result = CUDA_ERROR_INVALID_CONTEXT;
        if (result == CUDA_SUCCESS) result = dispatch.stream_get_context(stream, &actual);
        if (result == CUDA_SUCCESS && actual != context) result = CUDA_ERROR_INVALID_CONTEXT;
        CUstreamCaptureStatus capture = CU_STREAM_CAPTURE_STATUS_NONE;
        if (result == CUDA_SUCCESS) result = dispatch.stream_is_capturing(stream, &capture);
        if (result == CUDA_SUCCESS && capture != CU_STREAM_CAPTURE_STATUS_NONE)
            result = CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED;
        if (result != CUDA_SUCCESS) {
            retain(result);
            return false;
        }
        for (auto& event : events) {
            CUevent created{};
            result = dispatch.event_create(&created, CU_EVENT_DISABLE_TIMING);
            if (result != CUDA_SUCCESS || !created) {
                if (created) {
                    event.handle = created;
                    event.is_recorded = true;  // ambiguous creation; never destroy/retry
                }
                retain(result == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : result);
                return false;
            }
            event.handle = created;
            ++created_events;
        }
        // Set uncertain ownership BEFORE the real record; failure cannot be
        // converted into an unused event or retried.
        events.front().is_recorded = true;
        result = dispatch.event_record(events.front().handle, stream);
        if (result != CUDA_SUCCESS) {
            retain(result);
            return false;
        }
        return true;
    }
    void prove_predecessor() {
        const auto result = dispatch.event_query(events.front().handle);
        if (result == CUDA_ERROR_NOT_READY) return;
        if (result != CUDA_SUCCESS) {
            retain(result);
            return;
        }
        events.front().is_proven = true;
        if (is_closing.load()) {
            phase.store(CudaNativeEpochPhase::kRejected);
            return;
        }
        acquisition = gate->acquire_native_epoch(request);
        phase.store(acquisition.valid() ? CudaNativeEpochPhase::kPending
                                        : CudaNativeEpochPhase::kRejected);
    }
    void accept_grant(control::NativeEpochGrant accepted) {
        {
            const std::lock_guard lock(metadata_mutex);
            grant = accepted;
        }
        try {
            // Linux's steady_clock and the service use the same monotonic
            // nanosecond domain. Do NOT restart the duration on receipt.
            static_assert(std::chrono::steady_clock::period::num == 1 &&
                          std::chrono::steady_clock::period::den == 1'000'000'000);
            if (accepted.acceptance_deadline_nanoseconds >
                static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
                throw std::invalid_argument("unrepresentable native epoch deadline");
            }
            acceptance_deadline = NativeExecutionEpoch::TimePoint{std::chrono::nanoseconds{
                static_cast<std::int64_t>(accepted.acceptance_deadline_nanoseconds)}};
            const auto duration =
                std::chrono::microseconds{accepted.extent.acceptance_microseconds};
            const NativeEpochAuthorization authorization{
                accepted.execution, accepted.extent.max_calls, accepted.extent.calls_per_prefix,
                accepted.extent.max_pending_prefixes, duration};
            model =
                options.create_execution
                    ? options.create_execution(scope, authorization, acceptance_deadline - duration)
                    : std::make_unique<NativeExecutionEpoch>(scope, authorization,
                                                             acceptance_deadline - duration);
            if (!model) throw std::bad_alloc{};
            published_model.store(model.get());
            if (is_closing.load() || std::chrono::steady_clock::now() >= acceptance_deadline) {
                static_cast<void>(model->request_stop());
                phase.store(CudaNativeEpochPhase::kStopping);
            } else {
                phase.store(CudaNativeEpochPhase::kReady);
            }
        } catch (...) {
            // No original has been accepted; the predecessor was proven before
            // acquisition. Keep the identity through allocation failure.
            fail(CUDA_ERROR_OUT_OF_MEMORY);
            terminal = gate->finish_native_epoch(accepted, 0, true);
            phase.store(CudaNativeEpochPhase::kReporting);
            if (!terminal.valid()) retain(CUDA_ERROR_UNKNOWN);
        }
    }
    void poll_acquisition(NativeExecutionEpoch::TimePoint now) {
        if (ready(acquisition)) {
            const auto result = acquisition.get();
            if (result.status == control::NativeEpochClaimStatus::kGranted) {
                accept_grant(result.grant);
                return;
            }
            if (result.status == control::NativeEpochClaimStatus::kRejected) {
                phase.store(CudaNativeEpochPhase::kRejected);
                return;
            }
            {
                const std::lock_guard lock(metadata_mutex);
                grant = result.grant;
            }
            if (result.status == control::NativeEpochClaimStatus::kUncertain) {
                retain(CUDA_ERROR_UNKNOWN);
                return;
            }
            next_control = now + options.control_interval;
        }
        // A pending response carries a known, never-issued identity. A lost
        // claim is retained instead of retried or canceled as if still pending.
        if (!acquisition.valid() && grant.execution.task_id != 0) {
            if (is_closing.load()) {
                terminal = gate->cancel_native_epoch(grant);
                phase.store(CudaNativeEpochPhase::kReporting);
                if (!terminal.valid()) retain(CUDA_ERROR_UNKNOWN);
            } else if (now >= next_control) {
                acquisition = gate->claim_native_epoch(grant);
                if (!acquisition.valid()) retain(CUDA_ERROR_UNKNOWN);
            }
        }
    }
    void poll_control(NativeExecutionEpoch::TimePoint now) {
        if (ready(retreat)) {
            const auto result = retreat.get();
            if (!result)
                fail(CUDA_ERROR_UNKNOWN);
            else if (*result)
                close();
        }
        if (ready(heartbeat) && !heartbeat.get()) fail(CUDA_ERROR_UNKNOWN);
        if (is_closing.load() || model->snapshot().phase != NativeEpochPhase::kOpen) return;
        if (!retreat.valid() && now >= next_control) {
            retreat = gate->retreat_native_epoch(grant);
            next_control = now + options.control_interval;
            if (!retreat.valid()) fail(CUDA_ERROR_UNKNOWN);
        }
        if (!heartbeat.valid() && now >= next_heartbeat) {
            heartbeat = gate->heartbeat_native_epoch(grant);
            next_heartbeat = now + std::chrono::milliseconds{25};
            if (!heartbeat.valid()) fail(CUDA_ERROR_UNKNOWN);
        }
    }
    void prove_prefixes() {
        for (auto& event : events) {
            if (is_caller_fenced &&
                event.caller_record.load(std::memory_order_acquire) != CallerRecordState::kRecorded)
                continue;
            if (!event.fence) continue;
            const auto result = dispatch.event_query(event.handle);
            if (result == CUDA_ERROR_NOT_READY) continue;
            if (result != CUDA_SUCCESS) {
                static_cast<void>(model->fail_fence_completion(*event.fence));
                retain(result);
                return;
            }
            if (!model->prove_fence(*event.fence)) {
                retain(CUDA_ERROR_UNKNOWN);
                return;
            }
            const auto completed = model->snapshot().completed_calls;
            for (auto& covered : events) {
                if (is_caller_fenced && covered.caller_record.load(std::memory_order_acquire) !=
                                            CallerRecordState::kRecorded)
                    continue;
                if (covered.fence && covered.fence->covered_calls() <= completed) {
                    covered.fence.reset();
                    covered.is_proven = true;
                    ++proven_prefixes;
                    if (is_caller_fenced)
                        covered.caller_record.store(CallerRecordState::kFree,
                                                    std::memory_order_release);
                }
            }
        }
    }
    void record_prefix() {
        const auto state = model->snapshot();
        if (!state.unfenced_calls || (state.phase == NativeEpochPhase::kOpen &&
                                      state.unfenced_calls < request.extent.calls_per_prefix))
            return;
        auto event = std::ranges::find_if(events.begin() + 1, events.end(),
                                          [](const Event& candidate) { return !candidate.fence; });
        if (event == events.end()) return;
        auto fence = model->prepare_fence();
        if (!fence) return;
        event->fence = *fence;
        event->is_recorded = true;
        event->is_proven = false;
        const auto result = dispatch.event_record(event->handle, stream);
        const bool published = model->publish_fence_result(*fence, result == CUDA_SUCCESS);
        if (result != CUDA_SUCCESS || !published) {
            retain(result == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : result);
            return;
        }
        ++recorded_prefixes;
    }
    void progress_execution(NativeExecutionEpoch::TimePoint now) {
        if (now >= acceptance_deadline) close();
        poll_control(now);
        prove_prefixes();
        if (terminal_phase(phase.load())) return;
        if (!is_caller_fenced) record_prefix();
        if (terminal_phase(phase.load())) return;
        // Caller-side PTDS failure can publish Retained between monitor reads.
        // A later monitor transition must never reopen that terminal owner.
        if (model->snapshot().phase != NativeEpochPhase::kOpen &&
            !advance_running_phase(CudaNativeEpochPhase::kStopping))
            return;
        auto report = model->prepare_terminal();
        if (!report) return;
        terminal_capability = *report;
        if (!advance_running_phase(CudaNativeEpochPhase::kReporting)) {
            static_cast<void>(model->publish_terminal_result(*report, false));
            return;
        }
        terminal = gate->finish_native_epoch(
            grant, report->issued_calls(), report->disposition() == NativeEpochDisposition::kFail);
        if (!terminal.valid()) {
            static_cast<void>(model->publish_terminal_result(*report, false));
            retain(CUDA_ERROR_UNKNOWN);
        }
    }
    void poll_terminal() {
        if (!ready(terminal)) return;
        const bool acknowledged = terminal.get();
        if (terminal_capability &&
            !model->publish_terminal_result(*terminal_capability, acknowledged)) {
            retain(CUDA_ERROR_UNKNOWN);
            return;
        }
        if (acknowledged)
            phase.store(CudaNativeEpochPhase::kComplete);
        else
            retain(CUDA_ERROR_UNKNOWN);
    }
    void cleanup_events() noexcept {
        for (auto& event : events) {
            // Failed/unknown records or queries retain native resources until
            // independently proven whole-context stop. Never query them again.
            if (!event.handle || (event.is_recorded && !event.is_proven)) continue;
            const auto result = dispatch.event_destroy(event.handle);
            event.handle = nullptr;  // unknown destruction is never retried
            if (result == CUDA_SUCCESS)
                ++destroyed_events;
            else
                retain(result);
        }
    }
    void run() noexcept {
        const bool native_scope =
            dispatch.enter_native_scope ? dispatch.enter_native_scope() : false;
        const auto pushed = dispatch.context_push(context);
        if (pushed != CUDA_SUCCESS) {
            retain(pushed);
        } else {
            try {
                if (is_caller_fenced ? has_caller_initialization : initialize()) {
                    std::optional<NativeExecutionEpoch::TimePoint> shutdown_deadline;
                    while (!terminal_phase(phase.load())) {
                        if (const auto cpu = control.progress(); cpu != CUDA_SUCCESS) {
                            retain(cpu);
                            break;
                        }
                        const auto now = std::chrono::steady_clock::now();
                        if (is_shutting_down.load()) {
                            close();
                            if (!shutdown_deadline)
                                shutdown_deadline = now + options.shutdown_grace;
                            if (now >= *shutdown_deadline) {
                                retain(CUDA_ERROR_NOT_READY);
                                break;
                            }
                        }
                        switch (phase.load()) {
                            case CudaNativeEpochPhase::kPreparing:
                                prove_predecessor();
                                break;
                            case CudaNativeEpochPhase::kPending:
                                poll_acquisition(now);
                                break;
                            case CudaNativeEpochPhase::kReady:
                            case CudaNativeEpochPhase::kStopping:
                                progress_execution(now);
                                break;
                            case CudaNativeEpochPhase::kReporting:
                                poll_terminal();
                                break;
                            default:
                                break;
                        }
                        if (!terminal_phase(phase.load())) {
                            if (control.enabled()) {
                                if (const auto waited = control.wait_until(
                                        std::chrono::steady_clock::now() + options.poll_interval,
                                        progress_signal);
                                    waited != CUDA_SUCCESS)
                                    retain(waited);
                            } else
                                std::this_thread::sleep_for(options.poll_interval);
                        }
                    }
                }
            } catch (...) {
                retain(CUDA_ERROR_UNKNOWN);
            }
            // Caller records can still be returning after a monitor failure.
            // PTDS cleanup belongs to serialized shutdown after actual return.
            if (!is_caller_fenced) cleanup_events();
            CUcontext popped{};
            const auto result = dispatch.context_pop(&popped);
            if (result != CUDA_SUCCESS || popped != context)
                retain(result == CUDA_SUCCESS ? CUDA_ERROR_INVALID_CONTEXT : result);
        }
        if (dispatch.leave_native_scope) dispatch.leave_native_scope(native_scope);
        if (const auto retired = control.retire(); retired != CUDA_SUCCESS) retain(retired);
    }
    void joined_caller_cleanup() noexcept {
        if (!is_caller_fenced || has_joined_cleanup) return;
        has_joined_cleanup = true;
        if (std::ranges::none_of(events,
                                 [](const Event& event) { return event.handle != nullptr; }))
            return;
        const bool native_scope =
            dispatch.enter_native_scope ? dispatch.enter_native_scope() : false;
        const auto pushed = dispatch.context_push(context);
        if (pushed != CUDA_SUCCESS)
            retain(pushed);
        else {
            cleanup_events();
            CUcontext popped{};
            const auto result = dispatch.context_pop(&popped);
            if (result != CUDA_SUCCESS || popped != context)
                retain(result == CUDA_SUCCESS ? CUDA_ERROR_INVALID_CONTEXT : result);
        }
        if (dispatch.leave_native_scope) dispatch.leave_native_scope(native_scope);
    }
    bool record_caller_return(const NativeEpochIssue& issue,
                              NativeEpochCallOutcome outcome) noexcept {
        try {
            if (!caller_event || !model->original_returned(issue, outcome)) return false;
            auto* event = std::exchange(caller_event, nullptr);
            const auto fence = model->prepare_fence();
            if (!fence) {
                event->caller_record.store(CallerRecordState::kFailed);
                retain(CUDA_ERROR_UNKNOWN);
                return true;
            }
            event->fence = *fence;
            event->is_recorded = true;
            event->is_proven = false;
            event->caller_record.store(CallerRecordState::kRecording, std::memory_order_release);
            CUcontext actual{};
            CUresult result{};
            {
                const NativeEpochCallScope native_scope(dispatch);
                result = dispatch.context_get_current(&actual);
                if (result != CUDA_SUCCESS || actual != context) {
                    // An old-context stop proof cannot cover an original whose
                    // actual return scope was different or unobservable.
                    has_unverified_return_context.store(true);
                    if (result == CUDA_SUCCESS) result = CUDA_ERROR_INVALID_CONTEXT;
                }
                if (result == CUDA_SUCCESS) result = dispatch.event_record(event->handle, stream);
            }
            const bool published = model->publish_fence_result(*fence, result == CUDA_SUCCESS);
            if (result != CUDA_SUCCESS || !published) {
                event->caller_record.store(CallerRecordState::kFailed, std::memory_order_release);
                retain(result == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : result);
            } else {
                ++recorded_prefixes;
                event->caller_record.store(CallerRecordState::kRecorded, std::memory_order_release);
            }
            // CPU acceptance, not native proof success. Preserve original status.
            return true;
        } catch (...) {
            retain(CUDA_ERROR_UNKNOWN);
            return false;
        }
    }

    CUresult recover_context_completion(NativeEpochScope exact_scope,
                                        const std::function<CUresult()>& context_completion) {
        if (exact_scope != scope) return CUDA_ERROR_INVALID_VALUE;
        if (has_recovery_attempt) return CUDA_ERROR_UNKNOWN;
        auto* execution = published_model.load();
        if (!execution || phase.load() != CudaNativeEpochPhase::kRetained)
            return CUDA_ERROR_INVALID_VALUE;
        const auto before = execution->snapshot();
        if (before.has_entered_original || before.has_reserved_issue ||
            std::ranges::any_of(events, [](const Event& event) {
                const auto record = event.caller_record.load(std::memory_order_acquire);
                return record == CallerRecordState::kReserved ||
                       record == CallerRecordState::kRecording;
            }))
            return CUDA_ERROR_NOT_READY;
        if (has_unverified_return_context.load()) return CUDA_ERROR_INVALID_CONTEXT;
        // Reporting uncertainty cannot be reopened by another physical proof.
        if (before.phase != NativeEpochPhase::kStopping || !before.has_ownership)
            return CUDA_ERROR_UNKNOWN;
        if (!context_completion && !dispatch.context_synchronize) return CUDA_ERROR_NOT_SUPPORTED;
        is_shutting_down.store(true);
        if (monitor.joinable()) monitor.join();
        const auto live = std::ranges::count_if(
            events, [](const Event& event) { return event.handle != nullptr; });
        if (static_cast<std::uint32_t>(live) + destroyed_events.load() != created_events.load())
            return CUDA_ERROR_UNKNOWN;  // ambiguous creation/destruction is not replayable
        has_recovery_attempt = true;
        CUresult result = CUDA_ERROR_UNKNOWN;
        const auto retire_events = [&] {
            // Independent whole-context proof, not another query of a failed
            // private event. All originals and caller records already returned.
            for (auto& event : events) event.is_proven = true;
            cleanup_events();
            return created_events.load() == destroyed_events.load() ? CUDA_SUCCESS
                                                                    : CUDA_ERROR_UNKNOWN;
        };
        if (context_completion) {
            const NativeEpochCallScope native_scope(dispatch);
            CUcontext actual{};
            result = dispatch.context_get_current(&actual);
            if (result != CUDA_SUCCESS || actual != context)
                return result == CUDA_SUCCESS ? CUDA_ERROR_INVALID_CONTEXT : result;
            try {
                result = context_completion();  // actual application caller; never push/pop
            } catch (...) {
                result = CUDA_ERROR_UNKNOWN;
            }
            const auto observed = dispatch.context_get_current(&actual);
            if (observed != CUDA_SUCCESS || actual != context)
                result = observed == CUDA_SUCCESS ? CUDA_ERROR_INVALID_CONTEXT : observed;
            if (result == CUDA_SUCCESS) result = retire_events();
        } else {
            const NativeEpochCallScope native_scope(dispatch);
            CUcontext caller_context{}, actual{};
            result = dispatch.context_get_current(&caller_context);
            if (result != CUDA_SUCCESS) return result;
            result = dispatch.context_push(context);
            if (result != CUDA_SUCCESS) return result;
            try {
                result = dispatch.context_get_current(&actual);
                if (result == CUDA_SUCCESS && actual != context)
                    result = CUDA_ERROR_INVALID_CONTEXT;
                if (result == CUDA_SUCCESS) result = dispatch.context_synchronize();
                if (result == CUDA_SUCCESS) result = retire_events();
            } catch (...) {
                result = CUDA_ERROR_UNKNOWN;
            }
            CUcontext popped{};
            const auto restored = dispatch.context_pop(&popped);
            if (restored != CUDA_SUCCESS || popped != context)
                result = restored == CUDA_SUCCESS ? CUDA_ERROR_INVALID_CONTEXT : restored;
            else {
                const auto observed = dispatch.context_get_current(&actual);
                if (observed != CUDA_SUCCESS || actual != caller_context)
                    result = observed == CUDA_SUCCESS ? CUDA_ERROR_INVALID_CONTEXT : observed;
            }
        }
        if (result != CUDA_SUCCESS) return result;
        if (!execution->prove_scope_stopped(scope)) return CUDA_ERROR_UNKNOWN;
        const auto report = execution->prepare_terminal();
        if (!report || report->disposition() != NativeEpochDisposition::kFail)
            return CUDA_ERROR_UNKNOWN;
        terminal_capability = *report;
        terminal = gate->finish_native_epoch(grant, report->issued_calls(), true);
        const bool acknowledged =
            terminal.valid() &&
            terminal.wait_for(options.shutdown_grace) == std::future_status::ready &&
            terminal.get();
        if (!execution->publish_terminal_result(*report, acknowledged) || !acknowledged)
            return CUDA_ERROR_UNKNOWN;
        phase.store(CudaNativeEpochPhase::kComplete);
        return CUDA_SUCCESS;
    }
};

CudaNativeExecutionEpoch::CudaNativeExecutionEpoch(
    CUcontext context, CUstream stream, NativeEpochScope scope,
    control::NativeEpochLaunchRequest request,
    std::shared_ptr<control::AsyncCooperativeLaunchGate> gate, CudaNativeEpochDispatch dispatch,
    CudaNativeEpochOptions options)
    : implementation_(std::make_unique<Implementation>(context, stream, scope, request,
                                                       std::move(gate), dispatch, options)) {}
CudaNativeExecutionEpoch::~CudaNativeExecutionEpoch() {
    shutdown();
}

std::optional<NativeEpochIssue> CudaNativeExecutionEpoch::begin_call() noexcept {
    try {
        auto& state = *implementation_;
        if (state.control.status() != CUDA_SUCCESS) return std::nullopt;
        if (std::this_thread::get_id() != state.caller ||
            caller_lifetime != state.caller_identity || state.is_closing.load() ||
            state.phase.load() != CudaNativeEpochPhase::kReady)
            return std::nullopt;
        CUcontext actual{};
        const auto result = state.dispatch.context_get_current(&actual);
        if (result != CUDA_SUCCESS || actual != state.context) return std::nullopt;
        auto* model = state.published_model.load();
        if (!model) return std::nullopt;
        if (state.is_caller_fenced) {
            if (state.caller_event) return std::nullopt;
            for (auto iterator = state.events.begin() + 1; iterator != state.events.end();
                 ++iterator) {
                auto& event = *iterator;
                auto expected = Implementation::CallerRecordState::kFree;
                if (event.caller_record.compare_exchange_strong(
                        expected, Implementation::CallerRecordState::kReserved)) {
                    state.caller_event = &event;
                    break;
                }
            }
            if (!state.caller_event) return std::nullopt;
        }
        auto issue = model->reserve(state.scope, std::chrono::steady_clock::now());
        if (!issue) {
            if (state.caller_event)
                std::exchange(state.caller_event, nullptr)
                    ->caller_record.store(Implementation::CallerRecordState::kFree);
            return std::nullopt;
        }
        if (!model->enter_original(*issue, std::chrono::steady_clock::now())) {
            static_cast<void>(model->discard_unissued(*issue));
            if (state.caller_event)
                std::exchange(state.caller_event, nullptr)
                    ->caller_record.store(Implementation::CallerRecordState::kFree);
            return std::nullopt;
        }
        return issue;
    } catch (...) {
        implementation_->fail(CUDA_ERROR_UNKNOWN);
        return std::nullopt;
    }
}
bool CudaNativeExecutionEpoch::original_returned(const NativeEpochIssue& issue,
                                                 NativeEpochCallOutcome outcome) noexcept {
    auto& state = *implementation_;
    auto* model = state.published_model.load();
    if (std::this_thread::get_id() != state.caller || caller_lifetime != state.caller_identity ||
        !model)
        return false;
    return state.is_caller_fenced ? state.record_caller_return(issue, outcome)
                                  : model->original_returned(issue, outcome);
}
void CudaNativeExecutionEpoch::close() noexcept {
    implementation_->close();
}
void CudaNativeExecutionEpoch::shutdown() noexcept {
    auto& state = *implementation_;
    state.close();
    state.is_shutting_down.store(true);
    if (state.monitor.joinable()) state.monitor.join();
    state.joined_caller_cleanup();
}
CUresult CudaNativeExecutionEpoch::recover_context_completion(
    NativeEpochScope scope, const std::function<CUresult()>& context_completion) noexcept {
    try {
        return implementation_->recover_context_completion(scope, context_completion);
    } catch (...) {
        implementation_->retain(CUDA_ERROR_UNKNOWN);
        return CUDA_ERROR_UNKNOWN;
    }
}
CudaNativeEpochSnapshot CudaNativeExecutionEpoch::snapshot() const noexcept {
    auto& state = *implementation_;
    CudaNativeEpochSnapshot result;
    result.phase = state.phase.load();
    result.status = state.status.load();
    {
        const std::lock_guard lock(state.metadata_mutex);
        result.grant = state.grant;
    }
    if (const auto* model = state.published_model.load()) result.execution = model->snapshot();
    result.recorded_prefixes = state.recorded_prefixes.load();
    result.proven_prefixes = state.proven_prefixes.load();
    result.created_events = state.created_events.load();
    result.destroyed_events = state.destroyed_events.load();
    return result;
}

}  // namespace glimmer::interceptor
