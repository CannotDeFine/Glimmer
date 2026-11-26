#include "driver_cooperative_router.h"

#include "diagnostics.h"
#include "driver_api_interceptor.h"
#include "cuda_deferred_event_binding.h"
#include "cuda_capture_origin_registry.h"

#include <atomic>
#include <algorithm>
#include <condition_variable>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>

namespace glimmer::interceptor {
namespace {
template <typename Function>
Function resolve_native(const DriverDispatch& driver, const char* name, int cuda_version = 12600) {
    void* address = nullptr;
    if (driver.get_proc_address(name, &address, cuda_version, CU_GET_PROC_ADDRESS_LEGACY_STREAM) !=
            CUDA_SUCCESS ||
        !address || is_interceptor_driver_function(address)) {
        throw std::runtime_error("required cooperative native Driver symbol is unavailable");
    }
    return reinterpret_cast<Function>(address);
}
void report_context_stats(const CudaCoordinationStats& stats) noexcept {
    char message[256];
    const auto written =
        std::snprintf(message, sizeof(message),
                      "[glimmer] cooperative_context grants=%llu retreat=%llu "
                      "suspended=%llu resumed=%llu completed=%llu failed=%llu\n",
                      static_cast<unsigned long long>(stats.granted_executions),
                      static_cast<unsigned long long>(stats.retreat_requests),
                      static_cast<unsigned long long>(stats.acknowledged_suspensions),
                      static_cast<unsigned long long>(stats.logical_resumes),
                      static_cast<unsigned long long>(stats.acknowledged_completions),
                      static_cast<unsigned long long>(stats.acknowledged_failures));
    if (written > 0 && static_cast<std::size_t>(written) < sizeof(message)) {
        report_diagnostic(message);
    }
}
void report_source_stats(const CudaPausedNativeStats& current,
                         const CudaPausedNativeStats& baseline) noexcept {
    if (current.grants < baseline.grants || current.completed < baseline.completed ||
        current.failed < baseline.failed)
        return;
    char message[192];
    const auto written = std::snprintf(
        message, sizeof(message),
        "[glimmer] cooperative_native_session grants=%llu completed=%llu failed=%llu "
        "retained=%zu\n",
        static_cast<unsigned long long>(current.grants - baseline.grants),
        static_cast<unsigned long long>(current.completed - baseline.completed),
        static_cast<unsigned long long>(current.failed - baseline.failed), current.retained);
    if (written > 0 && static_cast<std::size_t>(written) < sizeof(message))
        report_diagnostic(message);
}
class EventRecordGuard final : public CudaQueueEventRecordGuard {
   public:
    EventRecordGuard(std::shared_ptr<CudaEventProjectionRegistry> registry_value,
                     CudaEventProjection accepted_value)
        : registry(std::move(registry_value)), accepted(std::move(accepted_value)) {}
    CUresult should_write(bool& output) noexcept override {
        return registry->matches_native_epoch(accepted, output);
    }
    CUresult record_accepted() noexcept override {
        if (!has_prepared || has_accepted) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        has_accepted = true;
        return registry->publish_logical(accepted, prefix ? &prefix : nullptr);
    }
    CUresult prepare_record(CudaEventRecordPreparation& output) noexcept override {
        output = {};
        if (has_prepared) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        // Reuse must not mutate a snapshot retained by an older deferred write.
        has_prepared = true;
        CudaEventProjection current;
        const auto result = registry->lookup_owned(accepted, current);
        if (result != CUDA_SUCCESS) {
            return result;
        }
        if (!current.is_logical()) {
            const auto& issued = current.issued_native_prefix();
            if (!current.has_native_completion() && !issued) {
                return CUDA_ERROR_NOT_READY;
            }
            output.should_order_previous = false;
            output.native_prefix = issued ? *issued : CudaOperationToken{};
        }
        accepted = std::move(current);
        prefix = output.native_prefix;
        return CUDA_SUCCESS;
    }

   private:
    const std::shared_ptr<CudaEventProjectionRegistry> registry;
    CudaEventProjection accepted;
    CudaOperationToken prefix;
    bool has_prepared{};
    bool has_accepted{};
};
class NativeGraphExecution final : public CudaFrontendNativeWaitGuard,
                                   public CudaQueueNativeFenceObserver {
   public:
    NativeGraphExecution(std::shared_ptr<CudaEventProjectionRegistry> registry_value,
                         CudaNativeGraphExecutable executable_value)
        : registry(std::move(registry_value)), executable(std::move(executable_value)) {}

    CUresult prepare() {
        records.reserve(executable.events().size());
        waits.reserve(executable.events().size());
        record_handles.reserve(executable.events().size());
        for (const auto& bound : executable.events()) {
            if (bound.effect.kind == CudaNativeGraphEventKind::kWait) {
                if (std::ranges::find(waits, bound.effect.event) == waits.end()) {
                    waits.push_back(bound.effect.event);
                }
            } else {
                if (bound.registration.context() != executable.context()) {
                    return CUDA_ERROR_NOT_SUPPORTED;
                }
                if (std::ranges::find(record_handles, bound.effect.event) == record_handles.end()) {
                    records.push_back(bound.registration);
                    record_handles.push_back(bound.effect.event);
                }
            }
        }
        for (const auto event : waits) {
            if (std::ranges::find(record_handles, event) != record_handles.end()) {
                // Source node enumeration does not prove record/wait topology.
                return CUDA_ERROR_NOT_SUPPORTED;
            }
        }
        return CUDA_SUCCESS;
    }

    CUresult should_wait(CUevent event, bool& output) noexcept override {
        output = false;
        for (const auto& bound : executable.events()) {
            if (bound.effect.kind == CudaNativeGraphEventKind::kWait &&
                bound.effect.event == event) {
                CudaEventProjection current;
                const auto status = registry->lookup_owned(bound.registration, current);
                if (status != CUDA_SUCCESS) {
                    return status;
                }
                if (current.is_logical() && current.context() != executable.context()) {
                    return CUDA_ERROR_NOT_SUPPORTED;
                }
                output = current.is_logical();
                return CUDA_SUCCESS;
            }
        }
        return CUDA_ERROR_INVALID_HANDLE;
    }
    CUresult native_reservation_accepted(const CudaOperationToken& token) noexcept override {
        if (*prefix || !token) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        *prefix = token;
        return CUDA_SUCCESS;
    }
    CUresult refresh_records() noexcept {
        for (auto& record : records) {
            CudaEventProjection current;
            const auto status = registry->lookup_owned(record, current);
            if (status != CUDA_SUCCESS) {
                return status;
            }
            record = std::move(current);
        }
        return CUDA_SUCCESS;
    }
    CUresult publish_issued_records() noexcept {
        const auto status = registry->invalidate_native_events(records, prefix);
        has_published_records = status == CUDA_SUCCESS;
        return status;
    }
    CUresult confirmed_native_completion() noexcept override {
        // A grant can be fenced/failed after pre-issue version or registration
        // rejection. Those logical snapshots are not native record proof.
        return has_published_records ? registry->complete_native_events(records) : CUDA_SUCCESS;
    }
    [[nodiscard]] std::span<const CUevent> waited_events() const noexcept {
        return waits;
    }

   private:
    const std::shared_ptr<CudaEventProjectionRegistry> registry;
    const CudaNativeGraphExecutable executable;
    // Filled once on reservation acceptance. Its command completes before the
    // caller's grant/pause/issue, and physical completion follows publication.
    const std::shared_ptr<CudaOperationToken> prefix{std::make_shared<CudaOperationToken>()};
    std::vector<CudaEventProjection> records;
    std::vector<CUevent> waits;
    std::vector<CUevent> record_handles;
    bool has_published_records{};
};
struct UncertainNativeGraphIssue {};
class SourceAdmission final {
   public:
    explicit SourceAdmission(std::atomic<std::size_t>& counter) : counter_(counter) {
        auto previous = counter_.load(std::memory_order_acquire);
        while (previous < 256) {
            if (counter_.compare_exchange_weak(previous, previous + 1, std::memory_order_acq_rel)) {
                has_acquired_ = true;
                break;
            }
        }
    }
    ~SourceAdmission() {
        if (has_acquired_) {
            counter_.fetch_sub(1, std::memory_order_acq_rel);
        }
    }
    SourceAdmission(const SourceAdmission&) = delete;
    SourceAdmission& operator=(const SourceAdmission&) = delete;
    [[nodiscard]] bool has_acquired() const noexcept {
        return has_acquired_;
    }

   private:
    std::atomic<std::size_t>& counter_;
    bool has_acquired_{};
};
}  // namespace

struct DriverCooperativeRouter::Implementation {
    static std::shared_ptr<control::AsyncCooperativeLaunchGate> create_gate(
        control::LaunchGateOptions options, std::optional<control::NativeEpochExtent> policy,
        bool should_owner_pump) {
        if (!should_owner_pump)
            return std::make_shared<control::AsyncCooperativeLaunchGate>(std::move(options), 1024,
                                                                         policy);
        auto selected = control::AsyncCooperativeLaunchGate::create_owner_pumped(
            std::move(options), {.max_queued_requests = 1024, .max_cpu_owners = 65535}, policy);
        if (!selected) throw std::runtime_error("owner-pumped control cold startup rejected");
        return std::shared_ptr<control::AsyncCooperativeLaunchGate>(std::move(selected));
    }
    struct Context {
        std::once_flag initialization;
        std::shared_ptr<CudaCooperativeFrontend> frontend;
        std::atomic<CUresult> status{CUDA_ERROR_NOT_INITIALIZED};
    };
    struct EventBinding {
        const CUcontext context;
        const CUevent event;
        CudaDeferredEventBinding binding;
        EventBinding(CUcontext origin, CUevent handle, bool is_bound)
            : context(origin), event(handle), binding(is_bound) {}
    };
    const DriverDispatch& driver;
    CudaFrontendDispatch dispatch;
    decltype(&cuPointerGetAttribute) pointer_attribute{};
    CudaCopyRangeDispatch copy_range;
    const std::optional<control::NativeEpochExtent> native_epoch_policy;
    CudaNativeEpochDispatch epoch_dispatch;
    std::shared_ptr<control::AsyncCooperativeLaunchGate> gate;
    const std::shared_ptr<void> lifetime{std::make_shared<int>(0)};
    const bool should_trace;
    const bool should_trace_native_timings;
    const std::size_t audit_block_capacity;
    std::mutex mutex;
    std::condition_variable condition;
    bool owns_native_transaction{};
    std::mutex epoch_mutex;
    std::condition_variable epoch_condition;
    bool owns_epoch_routing{};
    std::unordered_map<CUcontext, std::shared_ptr<Context>> contexts;
    const std::shared_ptr<CudaEventProjectionRegistry> events{
        std::make_shared<CudaEventProjectionRegistry>()};
    std::unordered_map<CUevent, std::shared_ptr<EventBinding>> event_bindings;
    CudaNativeGraphRegistry graphs;
    std::once_flag graph_dispatch_once;
    CudaNativeGraphMetadataDispatch graph_dispatch;
    std::shared_ptr<CudaProcessNativeSession> native_session;
    std::unique_ptr<CudaPausedNativeExecutor> native_executor;
    decltype(&cuThreadExchangeStreamCaptureMode) thread_exchange_capture_mode{};
    std::atomic<std::size_t> source_admissions{};
    // Mutations require the bound process transition; reads require its Call.
    std::atomic<ProcessCallGate*> source_gate{};
    CudaCaptureOriginRegistry source_origins;
    CudaProcessNativeToken source_token;
    CudaPausedNativeStats source_stats_baseline;
    CUresult source_status{CUDA_SUCCESS};
    CUresult lifecycle_status{CUDA_SUCCESS};

    // Source transitions serialize this metadata. Executor notification is
    // CPU-only and also invalidates cached permits/admission wait predicates.
    void fail_source(CUresult error) noexcept {
        if (source_status == CUDA_SUCCESS) {
            source_status =
                error == CUDA_SUCCESS || error == CUDA_ERROR_NOT_READY ? CUDA_ERROR_UNKNOWN : error;
        }
        native_executor->fail_closed(source_status);
    }

    // Caller holds the registry mutex. Read one atomic snapshot so a successful
    // finish cannot become a false SUCCESS with no frontend output published.
    CUresult retained_session_status() const noexcept {
        const auto execution_status = native_executor->status();
        if (execution_status != CUDA_SUCCESS) {
            return execution_status;
        }
        const auto status = native_session->status();
        return status == CUDA_SUCCESS ? CUDA_ERROR_NOT_READY : status;
    }

    // Serializes native transactions and context creation, not priority arrivals
    // into existing frontends. The mutex is released before any native/wait work.
    struct NativeTicket {
        Implementation& owner;
        bool has_acquired{};
        explicit NativeTicket(Implementation& value, bool should_wait = true) : owner(value) {
            std::unique_lock lock(owner.mutex);
            if (should_wait) {
                owner.condition.wait(lock, [&] { return !owner.owns_native_transaction; });
            } else if (owner.owns_native_transaction) {
                return;
            }
            owner.owns_native_transaction = true;
            has_acquired = true;
        }
        ~NativeTicket() {
            if (!has_acquired) {
                return;
            }
            {
                const std::lock_guard lock(owner.mutex);
                owner.owns_native_transaction = false;
            }
            owner.condition.notify_all();
        }
        NativeTicket(const NativeTicket&) = delete;
        NativeTicket& operator=(const NativeTicket&) = delete;
    };

    struct EpochRoutingTicket {
        Implementation& owner;
        bool has_acquired{};
        inline static thread_local EpochRoutingTicket* current{};
        explicit EpochRoutingTicket(Implementation& value, bool should_wait = true) : owner(value) {
            if (current) return;  // Never wait on a nested original's own ticket.
            std::unique_lock lock(owner.epoch_mutex);
            if (should_wait)
                owner.epoch_condition.wait(lock, [&] { return !owner.owns_epoch_routing; });
            else if (owner.owns_epoch_routing)
                return;
            owner.owns_epoch_routing = true;
            has_acquired = true;
            current = this;
        }
        ~EpochRoutingTicket() {
            if (!has_acquired) return;
            current = nullptr;
            {
                const std::lock_guard lock(owner.epoch_mutex);
                owner.owns_epoch_routing = false;
            }
            owner.epoch_condition.notify_all();
        }
        EpochRoutingTicket(const EpochRoutingTicket&) = delete;
        EpochRoutingTicket& operator=(const EpochRoutingTicket&) = delete;
    };

    Implementation(const DriverDispatch& driver_value, control::LaunchGateOptions options,
                   bool trace, bool trace_native_timings,
                   std::optional<control::NativeEpochExtent> epoch_policy,
                   CudaNativeEpochProjectionOverrides projections, std::size_t audit_capacity,
                   bool should_owner_pump_control)
        : driver(driver_value),
          native_epoch_policy(epoch_policy),
          gate(create_gate(std::move(options), epoch_policy, should_owner_pump_control)),
          should_trace(trace),
          should_trace_native_timings(trace_native_timings),
          audit_block_capacity(audit_capacity) {
        if (native_epoch_policy && !control::valid_native_epoch_extent(*native_epoch_policy))
            throw std::invalid_argument("invalid process native epoch policy");
        if (!native_epoch_policy && (projections.event_query || projections.event_destroy ||
                                     projections.context_synchronize))
            throw std::invalid_argument("native epoch projection requires policy");
        auto& queue = dispatch.worker.queue;
        queue.context_get_current =
            resolve_native<decltype(queue.context_get_current)>(driver, "cuCtxGetCurrent");
        queue.launch_kernel =
            resolve_native<decltype(queue.launch_kernel)>(driver, "cuLaunchKernel");
        queue.copy_host_to_device =
            resolve_native<decltype(queue.copy_host_to_device)>(driver, "cuMemcpyHtoDAsync");
        queue.copy_device_to_host =
            resolve_native<decltype(queue.copy_device_to_host)>(driver, "cuMemcpyDtoHAsync");
        queue.copy_device_to_device =
            resolve_native<decltype(queue.copy_device_to_device)>(driver, "cuMemcpyDtoDAsync");
        queue.free_async = resolve_native<decltype(queue.free_async)>(driver, "cuMemFreeAsync");
        queue.event_create = resolve_native<decltype(queue.event_create)>(driver, "cuEventCreate");
        queue.event_record = resolve_native<decltype(queue.event_record)>(driver, "cuEventRecord");
        queue.event_query = resolve_native<decltype(queue.event_query)>(driver, "cuEventQuery");
        queue.event_destroy =
            resolve_native<decltype(queue.event_destroy)>(driver, "cuEventDestroy");
        queue.stream_synchronize =
            resolve_native<decltype(queue.stream_synchronize)>(driver, "cuStreamSynchronize");
        queue.stream_wait_event =
            resolve_native<decltype(queue.stream_wait_event)>(driver, "cuStreamWaitEvent");
        queue.event_synchronize =
            resolve_native<decltype(queue.event_synchronize)>(driver, "cuEventSynchronize");
        dispatch.worker.context_push =
            resolve_native<decltype(dispatch.worker.context_push)>(driver, "cuCtxPushCurrent");
        dispatch.worker.context_pop =
            resolve_native<decltype(dispatch.worker.context_pop)>(driver, "cuCtxPopCurrent");
        dispatch.worker.enter_native_scope = enter_native_driver_scope;
        dispatch.worker.leave_native_scope = leave_native_driver_scope;
        if (native_epoch_policy) {
            epoch_dispatch = {queue.context_get_current, dispatch.worker.context_push,
                              dispatch.worker.context_pop,
                              // Pin the two-argument ABI; 12.6 lookup selects green-context v2.
                              resolve_native<decltype(epoch_dispatch.stream_get_context)>(
                                  driver, "cuStreamGetCtx", 12000),
                              resolve_native<decltype(epoch_dispatch.stream_is_capturing)>(
                                  driver, "cuStreamIsCapturing"),
                              queue.event_create, queue.event_record, queue.event_query,
                              queue.event_destroy, enter_native_driver_scope,
                              leave_native_driver_scope,
                              resolve_native<decltype(epoch_dispatch.context_synchronize)>(
                                  driver, "cuCtxSynchronize")};
            if (projections.event_query) epoch_dispatch.event_query = projections.event_query;
            if (projections.event_destroy) epoch_dispatch.event_destroy = projections.event_destroy;
            if (projections.context_synchronize)
                epoch_dispatch.context_synchronize = projections.context_synchronize;
        }
        dispatch.stream_create =
            resolve_native<decltype(dispatch.stream_create)>(driver, "cuStreamCreate");
        dispatch.stream_destroy =
            resolve_native<decltype(dispatch.stream_destroy)>(driver, "cuStreamDestroy");
        dispatch.stream_get_flags =
            resolve_native<decltype(dispatch.stream_get_flags)>(driver, "cuStreamGetFlags");
        dispatch.stream_get_priority =
            resolve_native<decltype(dispatch.stream_get_priority)>(driver, "cuStreamGetPriority");
        auto& pool = dispatch.pool;
        pool.context_get_current = queue.context_get_current;
        pool.allocate_device = allocate_cooperative_progress;
        pool.free_device = release_cooperative_progress;
        pool.allocate_host = resolve_native<decltype(pool.allocate_host)>(driver, "cuMemHostAlloc");
        const auto free_host = resolve_native<decltype(&cuMemFreeHost)>(driver, "cuMemFreeHost");
        pool.free_host = [free_host](void* pointer) {
            const NativeDriverCallScope scope;
            const auto result = free_host(pointer);
            return CudaCooperativeReleaseResult{result, result == CUDA_SUCCESS
                                                            ? CudaNativeReleaseState::kReleased
                                                            : CudaNativeReleaseState::kUncertain};
        };
        pool.map_host =
            resolve_native<decltype(pool.map_host)>(driver, "cuMemHostGetDevicePointer");
        pool.copy_device_to_host = queue.copy_device_to_host;
        pool.accounting_owner = lifetime;
        pointer_attribute =
            resolve_native<decltype(pointer_attribute)>(driver, "cuPointerGetAttribute");
        copy_range.pointer_attribute = pointer_attribute;
        copy_range.address_range =
            resolve_native<decltype(copy_range.address_range)>(driver, "cuMemGetAddressRange");
        copy_range.host_flags =
            resolve_native<decltype(copy_range.host_flags)>(driver, "cuMemHostGetFlags");
        void* per_thread_record{};
        if (driver.get_proc_address("cuEventRecord", &per_thread_record, 12600,
                                    CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM) !=
                CUDA_SUCCESS ||
            is_interceptor_driver_function(per_thread_record)) {
            per_thread_record = nullptr;
        }
        void* exchange_mode{};
        if (driver.get_proc_address("cuThreadExchangeStreamCaptureMode", &exchange_mode, 12600,
                                    CU_GET_PROC_ADDRESS_DEFAULT) == CUDA_SUCCESS &&
            !is_interceptor_driver_function(exchange_mode)) {
            thread_exchange_capture_mode =
                reinterpret_cast<decltype(thread_exchange_capture_mode)>(exchange_mode);
        }
        const CudaPausedNativeDispatch native_dispatch{
            queue.event_create,
            queue.event_record,
            reinterpret_cast<decltype(queue.event_record)>(per_thread_record),
            queue.event_query,
            queue.event_destroy,
            enter_native_driver_scope,
            leave_native_driver_scope,
            thread_exchange_capture_mode};
        native_executor = std::make_unique<CudaPausedNativeExecutor>(
            gate, native_dispatch, [this](CUcontext context, const NativeOperation& operation) {
                return in_context(context, operation);
            });
    }

    CUresult in_context(CUcontext target, const NativeOperation& operation) {
        CUcontext current = nullptr;
        auto result = driver.context_get_current(&current);
        if (result != CUDA_SUCCESS) {
            return result;
        }
        bool has_pushed = false;
        if (current != target) {
            const NativeDriverCallScope scope;
            result = dispatch.worker.context_push(target);
            if (result != CUDA_SUCCESS) {
                return result;
            }
            has_pushed = true;
        }
        CUresult native_result = CUDA_ERROR_UNKNOWN;
        try {
            native_result = operation();
        } catch (...) {
            native_result = CUDA_ERROR_UNKNOWN;
        }
        if (has_pushed) {
            CUcontext popped = nullptr;
            const NativeDriverCallScope scope;
            result = dispatch.worker.context_pop(&popped);
            if (result != CUDA_SUCCESS || popped != target) {
                return result == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : result;
            }
        }
        return native_result;
    }
};

DriverCooperativeRouter::DriverCooperativeRouter(
    const DriverDispatch& driver, control::LaunchGateOptions options, bool should_trace,
    bool should_trace_native_timings, std::optional<control::NativeEpochExtent> native_epoch_policy,
    CudaNativeEpochProjectionOverrides projections, std::size_t audit_block_capacity,
    bool should_owner_pump_control)
    : implementation_(std::make_unique<Implementation>(
          driver, std::move(options), should_trace, should_trace_native_timings,
          native_epoch_policy, projections, audit_block_capacity, should_owner_pump_control)) {}
DriverCooperativeRouter::~DriverCooperativeRouter() {
    if (!implementation_->contexts.empty() || implementation_->native_session) {
        report_diagnostic("[glimmer] unfinished cooperative process owner retained\n");
        (void)implementation_.release();
    }
}
bool DriverCooperativeRouter::has_native_epoch_policy() const noexcept {
    return implementation_->native_epoch_policy.has_value();
}

CUresult DriverCooperativeRouter::frontend(
    CUcontext context, std::shared_ptr<CudaCooperativeFrontend>& output) noexcept {
    try {
        if (!context) {
            return CUDA_ERROR_INVALID_CONTEXT;
        }
        auto& state = *implementation_;
        std::shared_ptr<Implementation::Context> record;
        {
            const std::lock_guard lock(state.mutex);
            if (state.lifecycle_status != CUDA_SUCCESS) {
                return CUDA_ERROR_UNKNOWN;
            }
            if (state.native_session) {
                return state.retained_session_status();
            }
            const auto found = state.contexts.find(context);
            if (found == state.contexts.end()) {
                if (state.contexts.size() >= 256) {
                    return CUDA_ERROR_OUT_OF_MEMORY;
                }
                record = std::make_shared<Implementation::Context>();
                state.contexts.emplace(context, record);
            } else {
                record = found->second;
            }
        }
        std::call_once(record->initialization, [&] {
            const Implementation::NativeTicket ticket(state);
            // No context-map lock spans allocation, CUDA or worker startup.
            CudaFrontendOptions frontend_options;
            frontend_options.audit_block_capacity = state.audit_block_capacity;
            frontend_options.should_trace_native_timings = state.should_trace_native_timings;
            frontend_options.native_epoch_policy = state.native_epoch_policy;
            frontend_options.native_epoch_dispatch = state.epoch_dispatch;
            record->frontend = std::make_shared<CudaCooperativeFrontend>(
                context, state.dispatch, state.gate, frontend_options);
            const NativeDriverCallScope scope;
            record->status.store(
                state.in_context(context, [&] { return record->frontend->initialize(); }),
                std::memory_order_release);
        });
        const auto status = record->status.load(std::memory_order_acquire);
        if (status == CUDA_SUCCESS) {
            output = record->frontend;
        }
        return status;
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}

CUresult DriverCooperativeRouter::existing_frontend(
    CUcontext context, std::shared_ptr<CudaCooperativeFrontend>& output) noexcept {
    if (!context || output) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    auto& state = *implementation_;
    const std::lock_guard lock(state.mutex);
    if (state.lifecycle_status != CUDA_SUCCESS) {
        return CUDA_ERROR_UNKNOWN;
    }
    if (state.native_session) {
        return state.retained_session_status();
    }
    const auto found = state.contexts.find(context);
    if (found == state.contexts.end()) {
        return CUDA_SUCCESS;
    }
    const auto status = found->second->status.load(std::memory_order_acquire);
    if (status == CUDA_ERROR_NOT_INITIALIZED) {
        return CUDA_SUCCESS;
    }
    if (status == CUDA_SUCCESS) {
        output = found->second->frontend;
    }
    return status;
}

CUresult DriverCooperativeRouter::native_execution(
    ProcessCallGate& gate, const std::shared_ptr<CudaCooperativeFrontend>& frontend,
    CUstream stream, bool is_ptds, const NativeOperation& original) noexcept {
    try {
        if (!frontend || !original) return CUDA_ERROR_INVALID_VALUE;
        auto& state = *implementation_;
        if (!state.native_epoch_policy) return frontend->native(stream, is_ptds, original, true);
        if (!gate.is_entered_by_current_thread() || gate.is_native_only_on_current_thread())
            return CUDA_ERROR_NOT_PERMITTED;
        auto* expected = static_cast<ProcessCallGate*>(nullptr);
        if (!state.source_gate.compare_exchange_strong(expected, &gate) && expected != &gate)
            return CUDA_ERROR_INVALID_VALUE;
        const Implementation::EpochRoutingTicket ticket(state);
        if (!ticket.has_acquired) return CUDA_ERROR_NOT_PERMITTED;
        const auto owner_status = normal_owner_status();
        if (owner_status != CUDA_SUCCESS) return owner_status;
        CUcontext actual{};
        auto status = state.driver.context_get_current(&actual);
        if (status != CUDA_SUCCESS || !actual)
            return status == CUDA_SUCCESS ? CUDA_ERROR_INVALID_CONTEXT : status;
        const auto canonical = stream == CU_STREAM_PER_THREAD || (!stream && is_ptds)
                                   ? CU_STREAM_PER_THREAD
                               : (!stream || stream == CU_STREAM_LEGACY) ? CU_STREAM_LEGACY
                                                                         : stream;
        CUstreamCaptureStatus capture{};
        status = state.epoch_dispatch.stream_is_capturing(canonical, &capture);
        if (status != CUDA_SUCCESS) return status;
        if (capture != CU_STREAM_CAPTURE_STATUS_NONE) return CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED;
        std::vector<std::shared_ptr<CudaCooperativeFrontend>> other_contexts;
        {
            const std::lock_guard lock(state.mutex);
            if (state.lifecycle_status != CUDA_SUCCESS) return CUDA_ERROR_UNKNOWN;
            if (state.native_session) return state.retained_session_status();
            const auto found = state.contexts.find(actual);
            if (found == state.contexts.end()) return CUDA_ERROR_INVALID_CONTEXT;
            status = found->second->status.load(std::memory_order_acquire);
            if (status != CUDA_SUCCESS) return status;
            // Initialization publishes the shared pointer before its release
            // status store; do not inspect a cold record before this acquire.
            if (found->second->frontend != frontend) return CUDA_ERROR_INVALID_CONTEXT;
            other_contexts.reserve(state.contexts.size());
            for (const auto& [context, record] : state.contexts) {
                if (context == actual) continue;
                const auto ready = record->status.load(std::memory_order_acquire);
                if (ready == CUDA_SUCCESS)
                    other_contexts.push_back(record->frontend);
                else if (ready != CUDA_ERROR_NOT_INITIALIZED)
                    return ready;
            }
        }
        for (const auto& owner : other_contexts) {
            status = owner->close_native_epoch();
            if (status != CUDA_SUCCESS) return status;
        }
        // Existing process transitions exclude this Call. Original effects and
        // monitor/control progress run outside both routing metadata mutexes.
        return frontend->native_epoch(
            stream,
            [&] {
                const NativeDriverCallScope native;
                return original();
            },
            is_ptds);
    } catch (const std::bad_alloc&) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}

CUresult DriverCooperativeRouter::native_boundary(
    const NativeOperation& operation, const NativeOperation& context_completion) noexcept {
    return native_boundary_impl(operation, context_completion, nullptr);
}

CUresult DriverCooperativeRouter::recover_native_epoch(
    ProcessCallGate& gate, CUcontext context, const control::NativeEpochGrant& expected,
    const NativeOperation& context_completion) noexcept {
    try {
        auto& state = *implementation_;
        if (!state.native_epoch_policy) return CUDA_ERROR_NOT_SUPPORTED;
        if (!context || !gate.is_entered_by_current_thread() ||
            gate.is_native_only_on_current_thread() || state.source_gate.load() != &gate ||
            Implementation::EpochRoutingTicket::current)
            return CUDA_ERROR_NOT_PERMITTED;
        const ProcessCallGate::Transition transition(gate);
        const Implementation::NativeTicket native_ticket(state, false);
        if (!native_ticket.has_acquired) return CUDA_ERROR_NOT_READY;
        const Implementation::EpochRoutingTicket epoch_ticket(state, false);
        if (!epoch_ticket.has_acquired) return CUDA_ERROR_NOT_READY;
        const auto status = normal_owner_status();
        if (status != CUDA_SUCCESS) return status;
        {
            const std::lock_guard lock(state.mutex);
            if (state.source_status != CUDA_SUCCESS || state.source_token ||
                state.source_admissions.load() != 0)
                return CUDA_ERROR_UNKNOWN;
        }
        std::shared_ptr<CudaCooperativeFrontend> owner;
        const auto found = existing_frontend(context, owner);
        if (found != CUDA_SUCCESS) return found;
        if (!owner) return CUDA_ERROR_INVALID_CONTEXT;
        // The transition excludes normal issue, source mutation and teardown.
        // Recover only the target's exact scoped failure, never global status.
        return owner->recover_native_epoch(expected, context_completion);
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}

std::optional<CUresult> DriverCooperativeRouter::recover_current_native_epoch(
    ProcessCallGate& gate, const NativeOperation& context_completion) noexcept {
    try {
        auto& state = *implementation_;
        if (!state.native_epoch_policy) return std::nullopt;
        if (!context_completion || !gate.is_entered_by_current_thread() ||
            gate.is_native_only_on_current_thread() || Implementation::EpochRoutingTicket::current)
            return CUDA_ERROR_NOT_PERMITTED;
        const auto* bound_gate = state.source_gate.load();
        if (bound_gate && bound_gate != &gate) return CUDA_ERROR_NOT_PERMITTED;
        if (!bound_gate) {
            const auto status = normal_owner_status();
            if (status != CUDA_SUCCESS) return status;
            const std::lock_guard lock(state.mutex);
            return state.source_status == CUDA_SUCCESS && !state.source_token &&
                           state.source_admissions.load() == 0
                       ? std::nullopt
                       : std::optional{CUDA_ERROR_UNKNOWN};
        }
        const ProcessCallGate::Transition transition(gate);
        const Implementation::NativeTicket native_ticket(state, false);
        if (!native_ticket.has_acquired) return CUDA_ERROR_NOT_READY;
        const Implementation::EpochRoutingTicket epoch_ticket(state, false);
        if (!epoch_ticket.has_acquired) return CUDA_ERROR_NOT_READY;
        const auto status = normal_owner_status();
        if (status != CUDA_SUCCESS) return status;
        {
            const std::lock_guard lock(state.mutex);
            if (state.source_status != CUDA_SUCCESS || state.source_token ||
                state.source_admissions.load() != 0)
                return CUDA_ERROR_UNKNOWN;
        }
        CUcontext context{};
        {
            const NativeDriverCallScope native;
            const auto observed = state.driver.context_get_current(&context);
            if (observed != CUDA_SUCCESS) return observed;
        }
        if (!context) return std::nullopt;  // original Runtime may initialize a cold caller
        std::shared_ptr<CudaCooperativeFrontend> owner;
        const auto found = existing_frontend(context, owner);
        if (found != CUDA_SUCCESS) return found;
        if (!owner) return std::nullopt;  // no lazy actor creation or SDK issue
        // SDK return and watcher publication are distinct. Join the old epoch
        // before deciding whether healthy ownership can decline this root.
        const auto closed = owner->close_native_epoch();
        const auto recovered = owner->recover_native_epoch(context_completion);
        if (!recovered && closed != CUDA_SUCCESS) return closed;
        if (!recovered || *recovered != CUDA_SUCCESS) return recovered;
        // Complete the new logical marker inside the same exclusion domain.
        // A later application's queue operation is outside this SDK proof.
        return owner->synchronize_context();
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}

struct DriverCooperativeRouter::NativeHandleRetirement {
    CUevent event{};
    CUstream stream{};
    bool is_stream{};
    const bool* has_native_effect{};
    const std::function<CUresult(CUcontext)>* retire_releases{};
};

CUresult DriverCooperativeRouter::native_event_destroy(CUevent event,
                                                       const NativeOperation& original,
                                                       const NativeOperation& context_completion,
                                                       const bool* has_native_effect) noexcept {
    if (!has_native_effect) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const NativeHandleRetirement retirement{.event = event, .has_native_effect = has_native_effect};
    return native_boundary_impl(original, context_completion, &retirement);
}

CUresult DriverCooperativeRouter::native_stream_destroy(
    CUstream stream, const NativeOperation& original,
    const std::function<CUresult(CUcontext)>& retire_releases,
    const NativeOperation& context_completion, const bool* has_native_effect) noexcept {
    if (!has_native_effect || !retire_releases) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const NativeHandleRetirement retirement{.stream = stream,
                                            .is_stream = true,
                                            .has_native_effect = has_native_effect,
                                            .retire_releases = &retire_releases};
    return native_boundary_impl(original, context_completion, &retirement);
}

CUresult DriverCooperativeRouter::native_boundary_impl(
    const NativeOperation& operation, const NativeOperation& context_completion,
    const NativeHandleRetirement* retirement) noexcept {
    try {
        if (!operation || !context_completion) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        auto& state = *implementation_;
        {
            const std::lock_guard lock(state.mutex);
            if (state.lifecycle_status != CUDA_SUCCESS) {
                return CUDA_ERROR_UNKNOWN;
            }
            if (state.native_session) {
                return state.retained_session_status();
            }
        }
        const Implementation::NativeTicket ticket(state);
        struct Paused {
            CUcontext context;
            std::shared_ptr<CudaCooperativeFrontend> frontend;
            CudaNativeBoundaryToken boundary;
            std::shared_ptr<Implementation::Context> record;
        };
        std::vector<Paused> paused;
        {
            const std::lock_guard lock(state.mutex);
            paused.reserve(state.contexts.size());
            for (const auto& [context, record] : state.contexts) {
                const auto status = record->status.load(std::memory_order_acquire);
                if (status == CUDA_SUCCESS) {
                    paused.push_back({context, record->frontend, {}, record});
                } else if (record->frontend || status != CUDA_ERROR_NOT_INITIALIZED) {
                    return status == CUDA_ERROR_NOT_INITIALIZED ? CUDA_ERROR_UNKNOWN : status;
                }
            }
        }
        struct FailureGuard {
            std::vector<Paused>& paused;
            bool is_finished{};
            ~FailureGuard() {
                if (is_finished) {
                    return;
                }
                for (auto& item : paused) {
                    item.record->status.store(CUDA_ERROR_UNKNOWN, std::memory_order_release);
                    item.frontend->fail_native_boundary(CUDA_ERROR_UNKNOWN);
                }
            }
        } failure{paused};
        for (auto& item : paused) {
            const auto result =
                state.in_context(item.context, [&] { return item.frontend->pause(item.boundary); });
            if (result != CUDA_SUCCESS) {
                return result;
            }
        }
        CudaEventProjection owned;
        std::shared_ptr<Implementation::EventBinding> binding;
        Paused* actor{};
        CUcontext handle_context{};
        CUresult handle_status{CUDA_SUCCESS};
        bool has_actor_association{};
        if (retirement) {
            if (retirement->is_stream) {
                const NativeDriverCallScope native;
                handle_status =
                    state.driver.stream_get_context(retirement->stream, &handle_context);
                if (handle_status == CUDA_SUCCESS && (!handle_context || !retirement->stream ||
                                                      retirement->stream == CU_STREAM_LEGACY ||
                                                      retirement->stream == CU_STREAM_PER_THREAD)) {
                    handle_status = CUDA_ERROR_INVALID_HANDLE;
                }
            } else {
                const auto found = state.events->lookup(retirement->event, owned);
                if (found != CUDA_SUCCESS && found != CUDA_ERROR_INVALID_HANDLE) {
                    return found;
                }
                handle_context = owned.context();
                {
                    const std::lock_guard lock(state.mutex);
                    const auto registered = state.event_bindings.find(retirement->event);
                    if (registered != state.event_bindings.end()) {
                        binding = registered->second;
                    }
                }
                if (binding && (!owned || binding->context != handle_context ||
                                !binding->binding.has_finished() ||
                                binding->binding.bind({}) != CUDA_SUCCESS)) {
                    return CUDA_ERROR_INVALID_HANDLE;
                }
            }
            const auto found = std::find_if(paused.begin(), paused.end(), [&](const auto& item) {
                return item.context == handle_context;
            });
            if (found != paused.end()) {
                actor = &*found;
            }
            if (actor && ((retirement->is_stream && handle_status == CUDA_SUCCESS) || binding)) {
                const auto prepared = retirement->is_stream
                                          ? actor->frontend->retire_paused_stream(
                                                actor->boundary, retirement->stream, false)
                                          : actor->frontend->retire_paused_event(
                                                actor->boundary, retirement->event, false);
                if (prepared != CUDA_SUCCESS &&
                    (binding || prepared != CUDA_ERROR_INVALID_HANDLE)) {
                    return prepared;
                }
                has_actor_association = prepared == CUDA_SUCCESS;
            } else if (binding) {
                return CUDA_ERROR_INVALID_HANDLE;
            }
        }
        const auto invalidated = state.events->invalidate_native();
        if (invalidated != CUDA_SUCCESS) {
            return invalidated;
        }
        CUresult original = CUDA_ERROR_UNKNOWN;
        try {
            original = operation();
        } catch (...) {
            original = CUDA_ERROR_UNKNOWN;
        }
        if (retirement && *retirement->has_native_effect) {
            if (original != CUDA_SUCCESS || handle_status != CUDA_SUCCESS) {
                return handle_status == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : handle_status;
            }
            if (has_actor_association) {
                const auto retired = retirement->is_stream
                                         ? actor->frontend->retire_paused_stream(
                                               actor->boundary, retirement->stream, true)
                                         : actor->frontend->retire_paused_event(
                                               actor->boundary, retirement->event, true);
                if (retired != CUDA_SUCCESS) {
                    return retired;
                }
            }
            if (binding) {
                const std::lock_guard lock(state.mutex);
                const auto registered = state.event_bindings.find(retirement->event);
                if (registered == state.event_bindings.end() || registered->second != binding) {
                    return CUDA_ERROR_INVALID_HANDLE;
                }
                state.event_bindings.erase(registered);
            }
            if (owned) {
                const auto forgotten = state.events->forget_owned(owned);
                if (forgotten != CUDA_SUCCESS) {
                    return forgotten;
                }
            }
            if (retirement->is_stream) {
                const auto detached = (*retirement->retire_releases)(handle_context);
                if (detached != CUDA_SUCCESS) {
                    return detached;
                }
            }
        } else if (retirement && original == CUDA_SUCCESS) {
            // A successful owner result cannot manufacture missing SDK effect.
            return CUDA_ERROR_UNKNOWN;
        }
        // The caller's actual native context work, including asynchronous frees,
        // must be proved before releasing the logical/native transaction gate.
        CUcontext current = nullptr;
        const auto current_status = state.driver.context_get_current(&current);
        if (current_status != CUDA_SUCCESS) {
            return current_status;
        }
        if (current) {
            const auto completed = context_completion();
            if (completed != CUDA_SUCCESS) {
                return completed;
            }
        }
        for (auto& item : paused) {
            const auto result = state.in_context(item.context, [&] {
                return item.frontend->finish_pause(item.boundary, context_completion);
            });
            if (result != CUDA_SUCCESS) {
                return result;
            }
        }
        const auto proven = state.events->complete_native();
        if (proven != CUDA_SUCCESS) {
            return proven;
        }
        failure.is_finished = true;
        return original;
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}

CUresult DriverCooperativeRouter::begin_native_session(CudaProcessNativeToken& token) noexcept {
    try {
        if (token) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        auto& state = *implementation_;
        {
            const std::lock_guard lock(state.mutex);
            if (state.native_session) {
                return state.retained_session_status();
            }
        }
        auto reservation = std::make_shared<Implementation::NativeTicket>(state, false);
        if (!reservation->has_acquired) {
            return CUDA_ERROR_NOT_READY;
        }
        std::vector<CudaProcessNativeContext> contexts;
        {
            const std::lock_guard lock(state.mutex);
            if (state.native_session) {
                return state.retained_session_status();
            }
            contexts.reserve(state.contexts.size());
            for (const auto& [context, record] : state.contexts) {
                const auto status = record->status.load(std::memory_order_acquire);
                if (status == CUDA_SUCCESS) {
                    contexts.push_back({context, record->frontend, record});
                } else {
                    // Transition exclusion must finish initializers before snapshot.
                    return status == CUDA_ERROR_NOT_INITIALIZED ? CUDA_ERROR_UNKNOWN : status;
                }
            }
        }
        auto session = std::make_shared<CudaProcessNativeSession>(
            std::move(contexts),
            [&state](CUcontext context, const NativeOperation& operation) {
                return state.in_context(context, operation);
            },
            std::move(reservation));
        const auto invalidated = state.events->invalidate_native();
        if (invalidated != CUDA_SUCCESS) {
            return invalidated;
        }
        {
            const std::lock_guard lock(state.mutex);
            state.native_session = session;
        }
        // Store ownership before native pause. Any partial failure stays retained.
        return session->begin(token);
    } catch (const std::bad_alloc&) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}

CUresult DriverCooperativeRouter::observe_native_session_context(CUcontext context) noexcept {
    try {
        auto& state = *implementation_;
        std::shared_ptr<CudaProcessNativeSession> session;
        {
            const std::lock_guard lock(state.mutex);
            session = state.native_session;
        }
        return session ? session->observe_native_context(context) : CUDA_ERROR_INVALID_VALUE;
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}

// Named verifier/completion roles retain the established private API order.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
CUresult DriverCooperativeRouter::finish_native_session(const CudaProcessNativeToken& token,
                                                        const NativeOperation& verify_inactive,
                                                        const NativeOperation& context_completion,
                                                        bool should_wait_for_executions) noexcept {
    try {
        auto& state = *implementation_;
        std::shared_ptr<CudaProcessNativeSession> session;
        {
            const std::lock_guard lock(state.mutex);
            session = state.native_session;
        }
        if (!session) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        if (!verify_inactive) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        const auto result = session->finish(
            token,
            [&] {
                const auto inactive = verify_inactive();
                if (inactive != CUDA_SUCCESS) {
                    return inactive;
                }
                // A detached caller cannot resume/issue while this transition
                // is owned, regardless of whether issued fences may be waited.
                if (state.source_admissions.load(std::memory_order_acquire) != 0 ||
                    (!should_wait_for_executions && state.native_executor->stats().retained != 0)) {
                    return CUDA_ERROR_NOT_READY;
                }
                return state.native_executor->drain();
            },
            context_completion);
        if (result == CUDA_SUCCESS) {
            // The process transition still excludes all source issue/acceptance.
            // Completion keeps native routing; a fresh record can now restore it
            // without draining unrelated logical work accepted after this proof.
            const auto proven = state.events->complete_native();
            if (proven != CUDA_SUCCESS) {
                return proven;
            }
            const std::lock_guard lock(state.mutex);
            if (state.native_session != session) {
                return CUDA_ERROR_UNKNOWN;
            }
            state.native_session.reset();
        }
        return result;
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}

// NOLINTEND(bugprone-easily-swappable-parameters)
CUresult DriverCooperativeRouter::native_session_execution(
    const CudaProcessNativeToken& token, CUcontext context, CUstream stream, bool is_ptds,
    const NativeOperation& operation, const NativeIssueScope& source_scope,
    ProcessCallGate* process_gate, const CudaSourceEventOperation* event_operation,
    const CudaNativeGraphExecutable* graph_operation) noexcept {
    try {
        if (!context || !operation || !source_scope) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        if (process_gate && !process_gate->is_entered_by_current_thread()) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        auto& state = *implementation_;
        const auto* bound = state.source_gate.load(std::memory_order_acquire);
        if (bound && bound != process_gate) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        if (graph_operation && (!*graph_operation || event_operation || !bound || !process_gate ||
                                !process_gate->is_native_only_on_current_thread())) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        if (event_operation &&
            (!bound || !process_gate || !process_gate->is_native_only_on_current_thread() ||
             (event_operation->kind != CudaSourceEventOperation::Kind::kRecord &&
              event_operation->kind != CudaSourceEventOperation::Kind::kWait))) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        if (bound && state.source_status != CUDA_SUCCESS) {
            return state.source_status;
        }
        std::shared_ptr<CudaProcessNativeSession> session;
        {
            const std::lock_guard lock(state.mutex);
            session = state.native_session;
        }
        if (!session || !session->owns_paused_token(token)) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        auto result = session->observe_native_context(context);
        if (result != CUDA_SUCCESS) {
            return result;
        }
        CudaEventProjection recorded_event;
        CudaEventRegistrationReservation record_reservation;
        bool is_record_prepared{};
        bool is_captured{};
        bool is_joining_wait{};
        const std::shared_ptr<const void> capture_owner = session;
        const auto invoke_native = [&] {
            const auto invalidated = state.events->invalidate_native(context);
            if (invalidated != CUDA_SUCCESS) {
                return invalidated;
            }
            const auto original = operation();
            if (original != CUDA_SUCCESS) {
                return original;
            }
            if (graph_operation) {
                for (const auto& bound_event : graph_operation->events()) {
                    if (bound_event.effect.kind != CudaNativeGraphEventKind::kRecord) continue;
                    // CUDA 12.6 rejects captured record graphs. Do not guess
                    // capture provenance if a future SDK permits that effect.
                    CUresult committed = is_captured ? CUDA_ERROR_NOT_SUPPORTED : CUDA_SUCCESS;
                    CudaEventProjection current;
                    if (committed == CUDA_SUCCESS)
                        committed = state.events->lookup_owned(bound_event.registration, current);
                    if (committed == CUDA_SUCCESS)
                        committed = state.events->publish_capture_record(current, nullptr);
                    if (committed != CUDA_SUCCESS) {
                        state.fail_source(committed);
                        return state.source_status;
                    }
                }
                return original;
            }
            if (!event_operation) return original;
            CUresult committed{CUDA_SUCCESS};
            if (event_operation->kind == CudaSourceEventOperation::Kind::kRecord) {
                if (record_reservation) {
                    committed = state.events->publish_native_event(
                        record_reservation, event_operation->event, CUDA_SUCCESS);
                    if (committed == CUDA_SUCCESS) {
                        committed = state.events->lookup(event_operation->event, recorded_event);
                    }
                }
                CudaEventProjection current;
                if (committed == CUDA_SUCCESS) {
                    committed = state.events->lookup_owned(recorded_event, current);
                }
                if (committed == CUDA_SUCCESS && current.context() != context) {
                    committed = CUDA_ERROR_INVALID_CONTEXT;
                }
                if (committed == CUDA_SUCCESS) {
                    committed = state.events->publish_capture_record(
                        current,
                        is_captured && event_operation->flags == 0 ? capture_owner : nullptr);
                }
            } else if (is_joining_wait) {
                auto after = static_cast<CUstreamCaptureStatus>(3);
                committed = state.driver.stream_capture_status(stream, &after, is_ptds);
                if (committed == CUDA_SUCCESS && after != CU_STREAM_CAPTURE_STATUS_ACTIVE) {
                    committed = CUDA_ERROR_UNKNOWN;
                }
            }
            if (committed != CUDA_SUCCESS) {
                state.fail_source(committed);
                return state.source_status;
            }
            return original;
        };
        const auto classify = [&] {
            if (bound && state.source_status != CUDA_SUCCESS) {
                return state.source_status;
            }
            const auto tracked = state.native_executor->status();
            if (tracked != CUDA_SUCCESS) {
                return tracked;
            }
            if (!session->owns_paused_token(token)) {
                return CUDA_ERROR_UNKNOWN;
            }
            auto status = static_cast<CUstreamCaptureStatus>(3);
            const auto query = state.driver.stream_capture_status(stream, &status, is_ptds);
            if (query != CUDA_SUCCESS) {
                return query;
            }
            if (status != CU_STREAM_CAPTURE_STATUS_NONE &&
                status != CU_STREAM_CAPTURE_STATUS_ACTIVE &&
                status != CU_STREAM_CAPTURE_STATUS_INVALIDATED) {
                return CUDA_ERROR_UNKNOWN;
            }
            is_captured = status != CU_STREAM_CAPTURE_STATUS_NONE;
            if (graph_operation) {
                if (graph_operation->context() != context) return CUDA_ERROR_INVALID_CONTEXT;
                auto valid = state.graphs.is_current(*graph_operation);
                if (valid != CUDA_SUCCESS) return valid;
                for (const auto& bound_event : graph_operation->events()) {
                    CudaEventProjection current;
                    valid = state.events->lookup_owned(bound_event.registration, current);
                    if (valid != CUDA_SUCCESS) return valid;
                    if (current.context() != context) return CUDA_ERROR_INVALID_CONTEXT;
                }
            }
            is_joining_wait = false;
            if (event_operation) {
                if (event_operation->kind == CudaSourceEventOperation::Kind::kRecord &&
                    !is_record_prepared) {
                    auto prepared = state.events->lookup(event_operation->event, recorded_event);
                    if (prepared == CUDA_ERROR_INVALID_HANDLE) {
                        prepared = state.events->prepare_record_event(context, record_reservation);
                    }
                    if (prepared != CUDA_SUCCESS) {
                        return prepared;
                    }
                    is_record_prepared = true;
                } else if (event_operation->kind == CudaSourceEventOperation::Kind::kWait &&
                           event_operation->flags == 0) {
                    CudaEventProjection input;
                    const auto found = state.events->lookup(event_operation->event, input);
                    if (found != CUDA_SUCCESS && found != CUDA_ERROR_INVALID_HANDLE) {
                        return found;
                    }
                    is_joining_wait = input.is_captured_by(capture_owner);
                    is_captured = is_captured || is_joining_wait;
                }
            }
            if (bound && is_captured) {
                CudaCaptureOriginToken origin;
                return state.source_origins.observe_origin(context, stream, is_ptds, origin);
            }
            return CUDA_SUCCESS;
        };
        int priority{};
        bool scope_was_used{};
        result = source_scope([&] {
            if (scope_was_used) {
                return CUDA_ERROR_INVALID_VALUE;
            }
            scope_was_used = true;
            const NativeDriverCallScope scope;
            return state.in_context(context, [&] {
                auto classified = classify();
                if (classified != CUDA_SUCCESS) {
                    return classified;
                }
                if (is_captured) {
                    return invoke_native();
                }
                return state.dispatch.stream_get_priority(
                    stream == nullptr && is_ptds ? CU_STREAM_PER_THREAD : stream, &priority);
            });
        });
        if (!scope_was_used) {
            return CUDA_ERROR_UNKNOWN;
        }
        if (result != CUDA_SUCCESS || is_captured) {
            return result;
        }
        CudaPausedNativePermit permit;
        SourceAdmission admission(state.source_admissions);
        if (!admission.has_acquired()) {
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
        struct UnissuedGuard {
            CudaPausedNativeExecutor& executor;
            const CudaPausedNativePermit& permit;
            ~UnissuedGuard() {
                const auto result = executor.retire_unissued(permit);
                if (result != CUDA_SUCCESS && result != CUDA_ERROR_INVALID_VALUE) {
                    report_diagnostic("[glimmer] unissued native execution retained\n");
                }
            }
        } unissued{*state.native_executor, permit};
        const auto relative =
            priority < 0 ? static_cast<std::uint32_t>(-static_cast<std::int64_t>(priority)) : 0;
        {
            std::optional<ProcessCallGate::AdmissionWait> waiting;
            if (process_gate) {
                waiting.emplace(*process_gate);
            }
            result = state.native_executor->acquire(context, stream, is_ptds, session,
                                                    {relative, 1}, permit);
            if (waiting) {
                waiting->resume();
            }
        }
        if (result != CUDA_SUCCESS) {
            return result;
        }
        scope_was_used = false;
        result = source_scope([&] {
            if (scope_was_used) {
                return CUDA_ERROR_INVALID_VALUE;
            }
            scope_was_used = true;
            const NativeDriverCallScope scope;
            return state.in_context(context, [&] {
                const auto classified = classify();
                if (classified != CUDA_SUCCESS || is_captured) {
                    const auto retired = state.native_executor->retire_unissued(permit);
                    if (retired != CUDA_SUCCESS) {
                        return retired;
                    }
                    return classified != CUDA_SUCCESS ? classified : invoke_native();
                }
                return state.native_executor->issue(permit, invoke_native).status;
            });
        });
        if (!scope_was_used) {
            const auto retired = state.native_executor->retire_unissued(permit);
            return retired == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : retired;
        }
        return result;
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}
CudaPausedNativeStats DriverCooperativeRouter::native_execution_stats() const noexcept {
    return implementation_->native_executor->stats();
}

CUresult DriverCooperativeRouter::finish_source_capture(ProcessCallGate::Transition& transition,
                                                        bool should_wait_for_executions) noexcept {
    auto& state = *implementation_;
    if (state.source_status != CUDA_SUCCESS) {
        return state.source_status;
    }
    if (!state.source_token) {
        return CUDA_SUCCESS;
    }
    const auto finished = finish_native_session(
        state.source_token,
        [&] {
            return state.source_origins.verify_none([&](CUcontext context, CUstream stream,
                                                        bool is_ptds,
                                                        CUstreamCaptureStatus* output) {
                return state.in_context(context, [&] {
                    return state.driver.stream_capture_status(stream, output, is_ptds);
                });
            });
        },
        [&] { return state.driver.context_synchronize(); }, should_wait_for_executions);
    if (finished == CUDA_SUCCESS) {
        if (state.should_trace)
            report_source_stats(state.native_executor->stats(), state.source_stats_baseline);
        state.source_token = {};
        transition.set_native_only(false);
    } else if (finished != CUDA_ERROR_NOT_READY) {
        state.fail_source(finished);
    }
    return finished;
}

DriverCooperativeRouter::SourceCaptureDispatch::SourceCaptureDispatch(
    DriverCooperativeRouter& router, ProcessCallGate& gate, CudaSourceCaptureKind kind) noexcept
    : router_(router), gate_(gate), root_kind(kind), caller(std::this_thread::get_id()) {}

CUresult DriverCooperativeRouter::SourceCaptureDispatch::invoke(
    CUcontext context, CUstream stream, bool is_ptds, CudaSourceCaptureKind kind,
    const NativeOperation& original) noexcept {
    // Foreign threads must neither issue nor mutate the caller-owned capability.
    if (std::this_thread::get_id() != caller || !gate_.is_entered_by_current_thread()) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (has_dispatched_ || has_rejected_ || kind != root_kind || !context || !original) {
        has_rejected_ = true;
        return CUDA_ERROR_INVALID_VALUE;
    }
    has_dispatched_ = true;
    native_status_ = router_.dispatch_source_capture(context, stream, is_ptds, original);
    return native_status_;
}

CUresult DriverCooperativeRouter::dispatch_source_capture(
    CUcontext context, CUstream stream, bool is_ptds, const NativeOperation& original) noexcept {
    auto& state = *implementation_;
    try {
        CudaCaptureOriginToken origin;
        auto result = state.source_origins.observe_origin(context, stream, is_ptds, origin);
        if (result == CUDA_SUCCESS) {
            result = observe_native_session_context(context);
        }
        if (result != CUDA_SUCCESS) {
            state.fail_source(result);
            return result;
        }
        const NativeDriverCallScope scope;
        return original();
    } catch (...) {
        // The known actual origin still requires checked reconciliation. Never
        // replay an original that may have succeeded before throwing.
        return CUDA_ERROR_UNKNOWN;
    }
}

CUresult DriverCooperativeRouter::SourceCaptureDispatch::lookup_origin_context(
    CUstream stream, bool is_ptds, CUcontext& output) noexcept {
    if (std::this_thread::get_id() != caller || !gate_.is_entered_by_current_thread()) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (root_kind != CudaSourceCaptureKind::kEnd || has_dispatched_ || has_origin_lookup_ ||
        has_rejected_ || output) {
        has_rejected_ = true;
        return CUDA_ERROR_INVALID_VALUE;
    }
    has_origin_lookup_ = true;
    auto& state = *router_.implementation_;
    if (state.source_status != CUDA_SUCCESS) {
        return state.source_status;
    }
    CUcontext current{};
    if (!stream || stream == CU_STREAM_LEGACY || stream == CU_STREAM_PER_THREAD) {
        const auto queried = state.driver.context_get_current(&current);
        if (queried != CUDA_SUCCESS) {
            return queried;
        }
    }
    return state.source_origins.lookup_context(stream, is_ptds, current, output);
}

CUresult DriverCooperativeRouter::source_capture_root(
    ProcessCallGate& gate, CudaSourceCaptureKind kind,
    const RootCaptureOperation& original) noexcept {
    try {
        if (!gate.is_entered_by_current_thread() || !original ||
            (kind != CudaSourceCaptureKind::kBegin && kind != CudaSourceCaptureKind::kEnd)) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        auto& state = *implementation_;
        if (!state.thread_exchange_capture_mode) {
            return CUDA_ERROR_NOT_SUPPORTED;
        }
        ProcessCallGate* expected{};
        if (!state.source_gate.compare_exchange_strong(expected, &gate) && expected != &gate) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        ProcessCallGate::Transition transition(gate);
        if (state.source_status != CUDA_SUCCESS) {
            return state.source_status;
        }
        // New calls cannot observe this phase until the transition exits.
        // Partial physical pause must nevertheless remain native-only/failed.
        transition.set_native_only(true);
        if (!state.source_token) {
            state.source_stats_baseline = state.native_executor->stats();
            const auto result = begin_native_session(state.source_token);
            if (result != CUDA_SUCCESS) {
                state.fail_source(result);
                return result;
            }
        }
        SourceCaptureDispatch dispatch(*this, gate, kind);
        CUresult native_status{CUDA_ERROR_UNKNOWN};
        try {
            // Root Runtime callbacks must remain visible to the typed Driver
            // bridge. Only dispatch.invoke guards the actual native Driver call.
            native_status = original(dispatch);
        } catch (...) {
            // Query/physical proof may resolve ownership, never replay issue.
            native_status = CUDA_ERROR_UNKNOWN;
        }
        if (!dispatch.has_dispatched_ || dispatch.has_rejected_ ||
            (native_status == CUDA_SUCCESS && dispatch.native_status_ != CUDA_SUCCESS)) {
            // Unknown root effects cannot be proved inactive by an empty origin
            // registry. Retain every pause token, even when root returned error.
            state.fail_source(CUDA_ERROR_UNKNOWN);
            return native_status != CUDA_SUCCESS ? native_status : state.source_status;
        }
        const auto finished = finish_source_capture(transition);
        return native_status != CUDA_SUCCESS      ? native_status
               : finished == CUDA_ERROR_NOT_READY ? CUDA_SUCCESS
                                                  : finished;
    } catch (const std::bad_alloc&) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}

CUresult DriverCooperativeRouter::source_capture_transition(
    ProcessCallGate& gate, CUcontext context, CUstream stream, bool is_ptds,
    CudaSourceCaptureKind kind, const NativeOperation& original) noexcept {
    if (!context || !original) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    try {
        return source_capture_root(gate, kind, [&](SourceCaptureDispatch& dispatch) {
            return dispatch.invoke(context, stream, is_ptds, kind, original);
        });
    } catch (const std::bad_alloc&) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}

CUresult DriverCooperativeRouter::try_finish_source_capture(
    ProcessCallGate& gate, bool should_wait_for_executions) noexcept {
    try {
        if (!gate.is_entered_by_current_thread() || implementation_->source_gate.load() != &gate) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        ProcessCallGate::Transition transition(gate);
        return finish_source_capture(transition, should_wait_for_executions);
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}

CUresult DriverCooperativeRouter::source_native_token(ProcessCallGate& gate,
                                                      CudaProcessNativeToken& output) noexcept {
    if (output || !gate.is_native_only_on_current_thread() ||
        implementation_->source_gate.load() != &gate) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const auto& state = *implementation_;
    if (state.source_status != CUDA_SUCCESS) {
        return state.source_status;
    }
    if (!state.source_token) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    output = state.source_token;
    return CUDA_SUCCESS;
}

CUresult DriverCooperativeRouter::quarantine_source_call(ProcessCallGate& gate,
                                                         CUresult error) noexcept {
    try {
        auto& state = *implementation_;
        if (error == CUDA_SUCCESS || !gate.is_entered_by_current_thread() ||
            !gate.is_native_only_on_current_thread() || state.source_gate.load() != &gate) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        const ProcessCallGate::Transition transition(gate);
        state.fail_source(error);
        return state.source_status;
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}

CUresult DriverCooperativeRouter::source_memory_metadata(ProcessCallGate& gate,
                                                         const NativeOperation& original,
                                                         CudaSourceMemoryOrigin origin,
                                                         const bool* has_native_issue) noexcept {
    if ((origin != CudaSourceMemoryOrigin::kDriver && origin != CudaSourceMemoryOrigin::kRuntime) ||
        (origin == CudaSourceMemoryOrigin::kRuntime && !has_native_issue)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return source_metadata_original(gate, original, origin, has_native_issue);
}

CUresult DriverCooperativeRouter::source_module_metadata(ProcessCallGate& gate,
                                                         const NativeOperation& original) noexcept {
    return source_metadata_original(gate, original, CudaSourceMemoryOrigin::kDriver, nullptr);
}

CUresult DriverCooperativeRouter::source_event_destroy(ProcessCallGate& gate, CUevent event,
                                                       const NativeOperation& original,
                                                       CudaSourceMemoryOrigin origin,
                                                       const bool* has_native_issue) noexcept {
    try {
        if (!original ||
            (origin != CudaSourceMemoryOrigin::kDriver &&
             origin != CudaSourceMemoryOrigin::kRuntime) ||
            (origin == CudaSourceMemoryOrigin::kRuntime && !has_native_issue)) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        auto& state = *implementation_;
        // Construct the source callback before issue. A mailbox failure after
        // successful destruction quarantines ownership rather than retrying.
        return source_metadata_original(
            gate,
            [&] {
                CudaProcessNativeToken token;
                auto result = source_native_token(gate, token);
                if (result != CUDA_SUCCESS) {
                    return result;
                }
                CudaEventProjection owned;
                result = state.events->lookup(event, owned);
                if (result != CUDA_SUCCESS && result != CUDA_ERROR_INVALID_HANDLE) {
                    return result;
                }
                std::shared_ptr<Implementation::EventBinding> binding;
                std::shared_ptr<CudaProcessNativeSession> session;
                {
                    const std::lock_guard lock(state.mutex);
                    session = state.native_session;
                    const auto found = state.event_bindings.find(event);
                    if (found != state.event_bindings.end()) {
                        binding = found->second;
                    }
                }
                if (binding) {
                    if (!owned || binding->context != owned.context() || !session ||
                        !binding->binding.has_finished()) {
                        return CUDA_ERROR_INVALID_HANDLE;
                    }
                    result = binding->binding.bind({});
                    if (result == CUDA_SUCCESS) {
                        result = session->retire_event(token, owned.context(), event, false);
                    }
                    if (result != CUDA_SUCCESS) {
                        return result;
                    }
                }
                const auto destroyed = original();
                if (destroyed != CUDA_SUCCESS) {
                    return destroyed;
                }
                if (binding) {
                    result = session->retire_event(token, owned.context(), event, true);
                    if (result != CUDA_SUCCESS) {
                        state.fail_source(result);
                        return state.source_status;
                    }
                    bool has_exact_binding{};
                    {
                        const std::lock_guard lock(state.mutex);
                        const auto found = state.event_bindings.find(event);
                        has_exact_binding =
                            found != state.event_bindings.end() && found->second == binding;
                        if (has_exact_binding) {
                            state.event_bindings.erase(found);
                        }
                    }
                    if (!has_exact_binding) {
                        state.fail_source(CUDA_ERROR_INVALID_HANDLE);
                        return state.source_status;
                    }
                }
                if (owned) {
                    result = state.events->forget_owned(owned);
                    if (result != CUDA_SUCCESS) {
                        state.fail_source(result);
                        return state.source_status;
                    }
                }
                return destroyed;
            },
            origin, has_native_issue);
    } catch (const std::bad_alloc&) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}

CUresult DriverCooperativeRouter::source_stream_destroy(
    ProcessCallGate& gate, CUstream stream, const NativeOperation& original,
    const std::function<CUresult(CUcontext)>& retire_releases, CudaSourceMemoryOrigin origin,
    const bool* has_native_issue) noexcept {
    if (!original || !retire_releases ||
        (origin != CudaSourceMemoryOrigin::kDriver && origin != CudaSourceMemoryOrigin::kRuntime) ||
        (origin == CudaSourceMemoryOrigin::kRuntime && !has_native_issue)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    try {
        auto& state = *implementation_;
        return source_metadata_original(
            gate,
            [&] {
                CUcontext context{};
                CUresult metadata{};
                {
                    const NativeDriverCallScope native;
                    metadata = state.driver.stream_get_context(stream, &context);
                }
                const bool is_explicit =
                    stream && stream != CU_STREAM_LEGACY && stream != CU_STREAM_PER_THREAD;
                if (metadata != CUDA_SUCCESS || !context || !is_explicit) {
                    // Preserve SDK input validation, including lazy Runtime
                    // binding. Never query a successfully destroyed handle.
                    const auto result = original();
                    if (result == CUDA_SUCCESS) {
                        state.fail_source(metadata == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : metadata);
                        return state.source_status;
                    }
                    return result;
                }
                auto result = observe_native_session_context(context);
                CudaCaptureOriginToken captured;
                if (result == CUDA_SUCCESS) {
                    result = state.source_origins.observe_origin(context, stream, false, captured);
                }
                CudaProcessNativeToken token;
                if (result == CUDA_SUCCESS) {
                    result = source_native_token(gate, token);
                }
                if (result != CUDA_SUCCESS) {
                    return result;
                }
                const auto session = state.native_session;
                if (!session) {
                    return CUDA_ERROR_INVALID_VALUE;
                }
                const auto prepared = session->retire_stream(token, context, stream, false);
                if (prepared != CUDA_SUCCESS && prepared != CUDA_ERROR_INVALID_HANDLE) {
                    return prepared;
                }
                const auto destroyed = original();
                if (destroyed != CUDA_SUCCESS) {
                    return destroyed;
                }
                if (prepared == CUDA_SUCCESS) {
                    result = session->retire_stream(token, context, stream, true);
                }
                if (result == CUDA_SUCCESS) {
                    result = state.source_origins.retire_destroyed(captured);
                }
                if (result == CUDA_SUCCESS) {
                    result = retire_releases(context);
                }
                if (result != CUDA_SUCCESS) {
                    state.fail_source(result);
                    return state.source_status;
                }
                return destroyed;
            },
            origin, has_native_issue);
    } catch (const std::bad_alloc&) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}

CUresult DriverCooperativeRouter::normal_runtime_event_create(
    CUevent* event, const NativeOperation& original) noexcept {
    if (!original) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    auto& state = *implementation_;
    CudaEventRegistrationReservation reservation;
    bool has_started{};
    try {
        auto result = normal_owner_status();
        if (result != CUDA_SUCCESS) {
            return result;
        }
        result = state.events->prepare_unrecorded_runtime_event(reservation);
        if (result != CUDA_SUCCESS) {
            return result;
        }
        has_started = true;
        const auto native = original();
        if (native != CUDA_SUCCESS) {
            return native;  // Failed SDK output is neither read nor rewritten.
        }
        CUcontext context{};
        {
            const NativeDriverCallScope scope;
            result = state.driver.context_get_current(&context);
        }
        const auto published = state.events->publish_runtime_event(
            reservation, event ? *event : nullptr, result == CUDA_SUCCESS ? context : nullptr);
        return result == CUDA_SUCCESS ? published : result;
    } catch (...) {
        if (has_started) {
            // Original may have issued. Retain an uncertain CPU owner without
            // inspecting the output, querying a handle or retrying the SDK.
            const auto quarantined =
                state.events->publish_runtime_event(reservation, nullptr, nullptr);
            return quarantined == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : quarantined;
        }
        return CUDA_ERROR_UNKNOWN;
    }
}

CUresult DriverCooperativeRouter::source_runtime_event_create(
    ProcessCallGate& gate, CUevent* event, const NativeOperation& original,
    const bool* has_native_issue) noexcept {
    if (!original || !has_native_issue) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    try {
        auto& state = *implementation_;
        return source_metadata_original(
            gate,
            [&] {
                CudaEventRegistrationReservation reservation;
                auto result = state.events->prepare_runtime_event(reservation);
                if (result != CUDA_SUCCESS) {
                    return result;
                }
                const auto native_status = original();
                if (native_status != CUDA_SUCCESS) {
                    // SDK failure: caller output may be uninitialized or unchanged.
                    return native_status;
                }
                const auto created = event ? *event : nullptr;
                CUcontext context{};
                result = state.driver.context_get_current(&context);
                const auto published = state.events->publish_runtime_event(
                    reservation, created, result == CUDA_SUCCESS ? context : nullptr);
                if (result == CUDA_SUCCESS) {
                    result = published;
                }
                if (result == CUDA_SUCCESS) {
                    result = observe_native_session_context(context);
                }
                if (result != CUDA_SUCCESS) {
                    state.fail_source(result);
                    return state.source_status;
                }
                return CUDA_SUCCESS;
            },
            CudaSourceMemoryOrigin::kRuntime, has_native_issue);
    } catch (const std::bad_alloc&) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}

CUresult DriverCooperativeRouter::source_runtime_event_record(
    ProcessCallGate& gate, CUevent event, CUstream stream, bool is_ptds, unsigned int flags,
    const NativeOperation& original, const bool* has_native_issue) noexcept {
    if (!original || !has_native_issue) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    try {
        auto& state = *implementation_;
        return source_metadata_original(
            gate,
            [&] {
                CudaEventProjection owned;
                CudaEventRegistrationReservation reservation;
                auto result = state.events->lookup(event, owned);
                if (result == CUDA_ERROR_INVALID_HANDLE) {
                    result = state.events->prepare_runtime_record_event(reservation);
                }
                if (result != CUDA_SUCCESS) {
                    return result;
                }
                const auto native_status = original();
                if (native_status != CUDA_SUCCESS) {
                    return native_status;
                }
                CUcontext context{};
                {
                    const NativeDriverCallScope native;
                    result = is_ptds ? state.driver.stream_get_context_ptsz(stream, &context)
                                     : state.driver.stream_get_context(stream, &context);
                }
                if (reservation) {
                    const auto published = state.events->publish_runtime_event(
                        reservation, event, result == CUDA_SUCCESS ? context : nullptr);
                    if (result == CUDA_SUCCESS) {
                        result = published;
                    }
                    if (result == CUDA_SUCCESS) {
                        result = state.events->lookup(event, owned);
                    }
                }
                if (result == CUDA_SUCCESS && (!context || owned.context() != context)) {
                    result = CUDA_ERROR_INVALID_CONTEXT;
                }
                if (result == CUDA_SUCCESS) {
                    result = observe_native_session_context(context);
                }
                auto status = static_cast<CUstreamCaptureStatus>(3);
                if (result == CUDA_SUCCESS) {
                    result = state.in_context(context, [&] {
                        return state.driver.stream_capture_status(stream, &status, is_ptds);
                    });
                }
                if (result == CUDA_SUCCESS && status != CU_STREAM_CAPTURE_STATUS_NONE &&
                    status != CU_STREAM_CAPTURE_STATUS_ACTIVE &&
                    status != CU_STREAM_CAPTURE_STATUS_INVALIDATED) {
                    result = CUDA_ERROR_UNKNOWN;
                }
                const bool is_captured = status != CU_STREAM_CAPTURE_STATUS_NONE;
                if (result == CUDA_SUCCESS && is_captured) {
                    CudaCaptureOriginToken observed;
                    result =
                        state.source_origins.observe_origin(context, stream, is_ptds, observed);
                }
                if (result == CUDA_SUCCESS) {
                    result = state.events->invalidate_native_event(owned);
                }
                CudaEventProjection current;
                if (result == CUDA_SUCCESS) {
                    result = state.events->lookup_owned(owned, current);
                }
                if (result == CUDA_SUCCESS) {
                    const std::shared_ptr<const void> capture_owner =
                        is_captured && flags == 0 ? state.native_session : nullptr;
                    result = state.events->publish_capture_record(current, capture_owner);
                }
                if (result != CUDA_SUCCESS) {
                    state.fail_source(result);
                    return state.source_status;
                }
                return native_status;
            },
            CudaSourceMemoryOrigin::kRuntime, has_native_issue);
    } catch (const std::bad_alloc&) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}

CUresult DriverCooperativeRouter::source_metadata_original(ProcessCallGate& gate,
                                                           const NativeOperation& original,
                                                           CudaSourceMemoryOrigin origin,
                                                           const bool* has_native_issue) noexcept {
    try {
        auto& state = *implementation_;
        if (!original || !gate.is_entered_by_current_thread() ||
            !gate.is_native_only_on_current_thread() || state.source_gate.load() != &gate) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        const ProcessCallGate::Transition transition(gate);
        CudaProcessNativeToken token;
        auto result = source_native_token(gate, token);
        if (result != CUDA_SUCCESS) {
            return result;
        }
        result = state.native_executor->status();
        if (result != CUDA_SUCCESS) {
            state.fail_source(result);
            return state.source_status;
        }
        CUcontext context{};
        result = state.driver.context_get_current(&context);
        if (result != CUDA_SUCCESS) {
            return result;
        }
        if (context) {
            result = observe_native_session_context(context);
            if (result != CUDA_SUCCESS) {
                state.fail_source(result);
                return state.source_status;
            }
        }
        CUresult native_status{};
        try {
            // Do not guard the root Runtime scope or change SDK cached lookup.
            // Each actual Driver original already has its native leaf guard.
            native_status = original();
        } catch (...) {
            state.fail_source(CUDA_ERROR_UNKNOWN);
            return state.source_status;
        }
        CUcontext after{};
        result = state.driver.context_get_current(&after);
        if (result == CUDA_SUCCESS && origin == CudaSourceMemoryOrigin::kRuntime && after) {
            result = observe_native_session_context(after);
        }
        const bool is_runtime = origin == CudaSourceMemoryOrigin::kRuntime;
        if (result != CUDA_SUCCESS || (!is_runtime && after != context) ||
            (is_runtime ? (*has_native_issue && !after)
                        : (native_status == CUDA_SUCCESS && !context))) {
            state.fail_source(result == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : result);
            return native_status == CUDA_SUCCESS ? state.source_status : native_status;
        }
        return native_status;
    } catch (const std::bad_alloc&) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}

CUresult DriverCooperativeRouter::close_context(CUcontext context,
                                                const NativeOperation& destroy_native) noexcept {
    try {
        auto& state = *implementation_;
        {
            const std::lock_guard lock(state.mutex);
            if (state.lifecycle_status != CUDA_SUCCESS) {
                return CUDA_ERROR_UNKNOWN;
            }
            if (state.native_session) {
                return state.retained_session_status();
            }
        }
        const Implementation::NativeTicket ticket(state);
        std::shared_ptr<Implementation::Context> record;
        {
            const std::lock_guard lock(state.mutex);
            const auto found = state.contexts.find(context);
            if (found != state.contexts.end()) {
                record = found->second;
            }
        }
        if (record && record->frontend) {
            const auto released =
                state.in_context(context, [&] { return record->frontend->shutdown(); });
            if (released != CUDA_SUCCESS) {
                return released;
            }
            record->status.store(CUDA_ERROR_DEINITIALIZED, std::memory_order_release);
            if (state.should_trace) {
                report_context_stats(record->frontend->stats());
            }
        }
        const auto result = destroy_native();
        if (result == CUDA_SUCCESS) {
            {
                const std::lock_guard lock(state.mutex);
                state.contexts.erase(context);
                std::erase_if(state.event_bindings,
                              [&](const auto& entry) { return entry.second->context == context; });
            }
            state.events->forget_context(context);
            state.graphs.forget_context(context);
        }
        return result;
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}

CUresult DriverCooperativeRouter::try_rejected_primary_destroy(
    ProcessCallGate& gate, CUcontext context, const NativeOperation& original,
    CudaPrimaryDestroyDisposition& disposition) noexcept {
    return destroy_context_transaction(gate, context, original, {},
                                       ContextDestroyMode::kPrimaryOnly, disposition);
}

CUresult DriverCooperativeRouter::destroy_context(
    ProcessCallGate& gate, CUcontext context, const NativeOperation& original,
    const std::function<CUresult(CUcontext)>& retire_destroyed, bool is_modern,
    CudaContextDestroyDisposition& disposition) noexcept {
    return destroy_context_transaction(
        gate, context, original, retire_destroyed,
        is_modern ? ContextDestroyMode::kModern : ContextDestroyMode::kLegacy, disposition);
}

CUresult DriverCooperativeRouter::destroy_context_transaction(
    ProcessCallGate& gate, CUcontext context, const NativeOperation& original,
    const std::function<CUresult(CUcontext)>& retire_destroyed, ContextDestroyMode mode,
    CudaContextDestroyDisposition& disposition) noexcept {
    auto& was_handled = disposition.was_handled;
    auto& has_uncertain_lifetime = disposition.has_uncertain_lifetime;
    was_handled = true;
    has_uncertain_lifetime = false;
    if (!original || (mode != ContextDestroyMode::kPrimaryOnly && !retire_destroyed) ||
        !gate.is_entered_by_current_thread()) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (gate.is_native_only_on_current_thread()) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    auto& state = *implementation_;
    std::shared_ptr<Implementation::Context> record;
    const auto fail = [&](CUresult error) {
        has_uncertain_lifetime = true;
        const auto status =
            error == CUDA_SUCCESS || error == CUDA_ERROR_NOT_READY ? CUDA_ERROR_UNKNOWN : error;
        if (record) {
            record->status.store(status, std::memory_order_release);
            if (record->frontend) {
                record->frontend->fail_native_boundary(status);
            }
        }
        const std::lock_guard lock(state.mutex);
        if (state.lifecycle_status == CUDA_SUCCESS) {
            state.lifecycle_status = status;
        }
        return status;
    };
    try {
        const ProcessCallGate::Transition transition(gate);
        const Implementation::NativeTicket ticket(state);
        const auto owner_status = normal_owner_status();
        if (owner_status != CUDA_SUCCESS) {
            return owner_status;
        }
        try {
            {
                const std::lock_guard lock(state.mutex);
                const auto found = state.contexts.find(context);
                if (found != state.contexts.end()) {
                    record = found->second;
                }
            }
            if (!record || !record->frontend) {
                if (mode == ContextDestroyMode::kPrimaryOnly) {
                    was_handled = false;
                    return CUDA_SUCCESS;
                }
                // SDK validation without an actor never imports or binds a target.
                const auto result = original();
                if (result != CUDA_SUCCESS) {
                    return result;
                }
                if (!context) {
                    return fail(CUDA_ERROR_UNKNOWN);
                }
                const auto retired = retire_destroyed(context);
                if (retired != CUDA_SUCCESS) {
                    return fail(retired);
                }
                state.events->forget_context(context);
                state.graphs.forget_context(context);
                const std::lock_guard lock(state.mutex);
                state.contexts.erase(context);
                std::erase_if(state.event_bindings,
                              [&](const auto& entry) { return entry.second->context == context; });
                return CUDA_SUCCESS;
            }
            if (record->status.load(std::memory_order_acquire) != CUDA_SUCCESS) {
                return CUDA_ERROR_UNKNOWN;
            }
            CudaWorkerCheckpoint checkpoint;
            CUresult result{};
            do {
                result = record->frontend->checkpoint(checkpoint);
                if (result == CUDA_ERROR_NOT_READY) {
                    std::this_thread::sleep_for(std::chrono::milliseconds{1});
                }
            } while (result == CUDA_ERROR_NOT_READY);
            if (result != CUDA_SUCCESS) {
                return fail(result);
            }
            CUdevice device{};
            bool is_primary{};
            result = state.in_context(context, [&] {
                auto status = state.driver.context_get_device(&device);
                if (status == CUDA_SUCCESS) {
                    status = state.driver.current_context_is_active_primary(context, is_primary);
                }
                return status;
            });
            if (result != CUDA_SUCCESS) {
                return fail(result);
            }
            if (!is_primary && mode == ContextDestroyMode::kPrimaryOnly) {
                const auto resumed = state.in_context(
                    context, [&] { return record->frontend->resume_checkpoint(checkpoint); });
                if (resumed != CUDA_SUCCESS) {
                    return fail(resumed);
                }
                was_handled = false;
                return CUDA_SUCCESS;
            }
            unsigned int context_api_version{};
            bool has_qualified_legacy_context{};
            if (!is_primary) {
                if (mode == ContextDestroyMode::kLegacy) {
                    const auto observed =
                        state.driver.context_get_api_version(context, &context_api_version);
                    if (observed != CUDA_SUCCESS && observed != CUDA_ERROR_NOT_SUPPORTED) {
                        return fail(observed);
                    }
                    has_qualified_legacy_context =
                        observed == CUDA_SUCCESS && context_api_version == 3020;
                }
                result = state.in_context(context, [&] {
                    return record->frontend->release_checkpoint_progress(checkpoint);
                });
                if (result != CUDA_SUCCESS) {
                    return fail(result);
                }
            }
            // The worker is joined, but its exact queue/resources remain owned.
            // All private binding has ended before the original caller issue.
            result = original();
            if (!is_primary && result == CUDA_SUCCESS) {
                auto retired =
                    record->frontend->retire_checkpoint_after_context_destroy(checkpoint);
                if (retired == CUDA_SUCCESS) {
                    record->status.store(CUDA_ERROR_DEINITIALIZED, std::memory_order_release);
                    retired = retire_destroyed(context);
                }
                if (retired != CUDA_SUCCESS) {
                    return fail(retired);
                }
                // The retired frontend has no worker: stats are CPU-only and
                // preserve the existing opt-in public stop/resume evidence.
                if (state.should_trace) {
                    report_context_stats(record->frontend->stats());
                }
                state.events->forget_context(context);
                state.graphs.forget_context(context);
                const std::lock_guard lock(state.mutex);
                state.contexts.erase(context);
                std::erase_if(state.event_bindings,
                              [&](const auto& entry) { return entry.second->context == context; });
                return CUDA_SUCCESS;
            }
            if (result != CUDA_ERROR_INVALID_CONTEXT) {
                const auto failed = fail(result);
                return result == CUDA_SUCCESS ? failed : result;
            }
            if (is_primary) {
                bool is_active{};
                const auto observed = state.driver.primary_context_is_active(device, is_active);
                if (observed != CUDA_SUCCESS || !is_active) {
                    (void)fail(observed == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : observed);
                    return result;
                }
            } else if (mode != ContextDestroyMode::kLegacy || !has_qualified_legacy_context) {
                (void)fail(CUDA_ERROR_UNKNOWN);
                return result;
            }
            const auto resumed = state.in_context(context, [&] {
                auto restored = is_primary
                                    ? CUDA_SUCCESS
                                    : record->frontend->restore_checkpoint_progress(checkpoint);
                if (restored == CUDA_SUCCESS) {
                    restored = record->frontend->resume_checkpoint(checkpoint);
                }
                return restored;
            });
            if (resumed != CUDA_SUCCESS) {
                (void)fail(resumed);
            }
            return result;  // Preserve the actual SDK non-success, even on quarantine.
        } catch (...) {
            return fail(CUDA_ERROR_UNKNOWN);
        }
    } catch (...) {
        return fail(CUDA_ERROR_UNKNOWN);
    }
}

CUresult DriverCooperativeRouter::release_primary_context(
    ProcessCallGate& gate, CUdevice device, const NativeOperation& original,
    const std::function<CUresult(CUcontext)>& retire_destroyed,
    bool& has_uncertain_lifetime) noexcept {
    has_uncertain_lifetime = false;
    if (!original || !retire_destroyed || !gate.is_entered_by_current_thread()) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    // A transition itself makes nested public calls native-only. Reject the
    // caller's retained source phase before acquiring our own transition;
    // normal_owner_status rechecks actual retained ownership after exclusion.
    if (gate.is_native_only_on_current_thread()) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    auto& state = *implementation_;
    std::shared_ptr<Implementation::Context> record;
    const auto fail = [&](CUresult error) {
        has_uncertain_lifetime = true;
        const auto status =
            error == CUDA_SUCCESS || error == CUDA_ERROR_NOT_READY ? CUDA_ERROR_UNKNOWN : error;
        if (record) {
            record->status.store(status, std::memory_order_release);
            if (record->frontend) {
                record->frontend->fail_native_boundary(status);
            }
        }
        const std::lock_guard lock(state.mutex);
        if (state.lifecycle_status == CUDA_SUCCESS) {
            state.lifecycle_status = status;
        }
        return status;
    };
    try {
        const ProcessCallGate::Transition transition(gate);
        const Implementation::NativeTicket ticket(state);
        const auto owner_status = normal_owner_status();
        if (owner_status != CUDA_SUCCESS) {
            return owner_status;
        }
        try {
            CUcontext target{};
            bool is_deinitialized_before_identity{};
            auto result = state.driver.active_primary_context(device, target,
                                                              &is_deinitialized_before_identity);
            if (result == CUDA_ERROR_DEINITIALIZED && is_deinitialized_before_identity) {
                // SDK exit may precede a vendor's release callback. Only the
                // first non-activating query can prove no identity effects;
                // a failed balancing release must retain the old quarantine.
                const auto native_result = original();
                return native_result == CUDA_ERROR_DEINITIALIZED ? native_result
                                                                 : fail(native_result);
            }
            if (result == CUDA_ERROR_INVALID_DEVICE || result == CUDA_ERROR_NOT_INITIALIZED) {
                return original();  // Native validation cannot retire a modeled owner.
            }
            if (result != CUDA_SUCCESS) {
                return fail(result);
            }
            if (!target) {
                return original();  // No activation, owner import or destruction proof.
            }
            {
                const std::lock_guard lock(state.mutex);
                const auto found = state.contexts.find(target);
                if (found != state.contexts.end()) {
                    record = found->second;
                }
            }
            CudaWorkerCheckpoint checkpoint;
            if (record && record->frontend) {
                do {
                    result = record->frontend->checkpoint(checkpoint);
                    if (result == CUDA_ERROR_NOT_READY) {
                        std::this_thread::sleep_for(std::chrono::milliseconds{1});
                    }
                } while (result == CUDA_ERROR_NOT_READY);
                if (result == CUDA_SUCCESS) {
                    result = state.in_context(target, [&] {
                        return record->frontend->release_checkpoint_progress(checkpoint);
                    });
                }
                if (result != CUDA_SUCCESS) {
                    return fail(result);
                }
            }
            // All private target binding has ended. The SDK keeps its actual caller
            // and exact post-call context, including its destroyed bound identity.
            result = original();
            if (result != CUDA_SUCCESS) {
                return fail(result);
            }
            bool is_active{};
            result = state.driver.primary_context_is_active(device, is_active);
            if (result != CUDA_SUCCESS) {
                return fail(result);
            }
            if (is_active) {
                if (record && record->frontend) {
                    result = state.in_context(target, [&] {
                        auto restored = record->frontend->restore_checkpoint_progress(checkpoint);
                        if (restored == CUDA_SUCCESS) {
                            restored = record->frontend->resume_checkpoint(checkpoint);
                        }
                        return restored;
                    });
                }
                return result == CUDA_SUCCESS ? CUDA_SUCCESS : fail(result);
            }
            if (record && record->frontend) {
                result = record->frontend->retire_checkpoint_after_context_destroy(checkpoint);
                if (result != CUDA_SUCCESS) {
                    return fail(result);
                }
                record->status.store(CUDA_ERROR_DEINITIALIZED, std::memory_order_release);
            }
            result = retire_destroyed(target);
            if (result != CUDA_SUCCESS) {
                return fail(result);
            }
            state.events->forget_context(target);
            state.graphs.forget_context(target);
            {
                const std::lock_guard lock(state.mutex);
                state.contexts.erase(target);
                std::erase_if(state.event_bindings,
                              [&](const auto& entry) { return entry.second->context == target; });
            }
            return CUDA_SUCCESS;
        } catch (...) {
            // Publish failure before either exclusion owner unwinds, including
            // an actor-free allocation exception after original destruction.
            return fail(CUDA_ERROR_UNKNOWN);
        }
    } catch (...) {
        return fail(CUDA_ERROR_UNKNOWN);
    }
}

CUresult DriverCooperativeRouter::register_event(CUcontext context, CUevent event) noexcept {
    try {
        auto& state = *implementation_;
        std::shared_ptr<CudaCooperativeFrontend> owner;
        auto result = frontend(context, owner);
        if (result != CUDA_SUCCESS) {
            return result;
        }
        result = owner->bind_event(event, state.lifetime);
        if (result != CUDA_SUCCESS) {
            return result;
        }
        auto binding = std::make_shared<Implementation::EventBinding>(context, event, true);
        const std::lock_guard lock(state.mutex);
        if (!state.event_bindings.emplace(event, binding).second) {
            return CUDA_ERROR_INVALID_HANDLE;
        }
        result = state.events->register_event(context, event, true);
        if (result != CUDA_SUCCESS) {
            state.event_bindings.erase(event);
        }
        return result;
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}

CUresult DriverCooperativeRouter::register_native_event(const CudaProcessNativeToken& token,
                                                        CUcontext context, CUevent event) noexcept {
    try {
        if (!context || !event) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        auto& state = *implementation_;
        std::shared_ptr<CudaProcessNativeSession> session;
        {
            const std::lock_guard lock(state.mutex);
            session = state.native_session;
        }
        if (!session || !session->owns_paused_token(token)) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        const auto observed = session->observe_native_context(context);
        if (observed != CUDA_SUCCESS) {
            return observed;
        }
        return state.events->register_event(context, event, false);
    } catch (const std::bad_alloc&) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}

CUresult DriverCooperativeRouter::prepare_native_event(
    const CudaProcessNativeToken& token, CUcontext context,
    CudaEventRegistrationReservation& output) noexcept {
    if (!context || output) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    auto& state = *implementation_;
    std::shared_ptr<CudaProcessNativeSession> session;
    {
        const std::lock_guard lock(state.mutex);
        session = state.native_session;
    }
    if (!session || !session->owns_paused_token(token)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const auto observed = session->observe_native_context(context);
    return observed == CUDA_SUCCESS ? state.events->prepare_native_event(context, output)
                                    : observed;
}

CUresult DriverCooperativeRouter::publish_native_event(
    CudaEventRegistrationReservation& reservation, CUevent event, CUresult native_status) noexcept {
    return implementation_->events->publish_native_event(reservation, event, native_status);
}

CUresult DriverCooperativeRouter::retire_metadata_event(const CudaEventProjection& owned) noexcept {
    if (!owned || owned.allows_logical_binding()) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return implementation_->events->forget_owned(owned);
}

CUresult DriverCooperativeRouter::event_frontend(
    CUevent event, CUcontext& context, std::shared_ptr<CudaCooperativeFrontend>& output) noexcept {
    try {
        auto& state = *implementation_;
        CudaEventProjection projection;
        const auto found = state.events->lookup(event, projection);
        if (found != CUDA_SUCCESS) {
            return found;
        }
        if (!projection.allows_logical_binding()) {
            return CUDA_ERROR_INVALID_HANDLE;
        }
        context = projection.context();
        std::shared_ptr<Implementation::EventBinding> binding;
        {
            const std::lock_guard lock(state.mutex);
            if (state.native_session) {
                return state.retained_session_status();
            }
            auto registered = state.event_bindings.find(event);
            if (registered == state.event_bindings.end()) {
                if (projection.is_logical() ||
                    (!projection.has_native_completion() && !projection.issued_native_prefix())) {
                    return CUDA_ERROR_NOT_READY;
                }
                // Native-only creation registered no private binding. Allocate
                // this CPU once-owner only after proof, before any CUDA import.
                registered = state.event_bindings
                                 .emplace(event, std::make_shared<Implementation::EventBinding>(
                                                     context, event, false))
                                 .first;
            }
            if (registered->second->context != context) {
                return CUDA_ERROR_INVALID_HANDLE;
            }
            binding = registered->second;
        }
        CUresult status{};
        if (binding->binding.has_finished()) {
            // Avoid constructing/allocating a callback on ordinary event lookup.
            status = binding->binding.bind({});
        } else {
            if (!projection.has_native_completion() && !projection.issued_native_prefix()) {
                return CUDA_ERROR_NOT_READY;
            }
            status = binding->binding.bind([&] {
                std::shared_ptr<CudaCooperativeFrontend> owner;
                auto result = frontend(context, owner);
                if (result == CUDA_SUCCESS) {
                    const NativeDriverCallScope scope;
                    result = state.in_context(
                        context, [&] { return owner->bind_event(binding->event, state.lifetime); });
                }
                return result;
            });
        }
        if (status != CUDA_SUCCESS) {
            return status;
        }
        return frontend(context, output);
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}
void DriverCooperativeRouter::forget_event(CUevent event) noexcept {
    auto& state = *implementation_;
    {
        const std::lock_guard lock(state.mutex);
        state.event_bindings.erase(event);
    }
    state.events->forget(event);
}
CUresult DriverCooperativeRouter::event_projection(CUevent event,
                                                   CudaEventProjection& output) noexcept {
    return implementation_->events->lookup(event, output);
}
CUresult DriverCooperativeRouter::admit_event_metadata(std::span<const CudaEventProjection> owned,
                                                       const NativeOperation& operation) noexcept {
    return implementation_->events->admit_metadata(owned, operation);
}
CUresult DriverCooperativeRouter::refresh_event_projection(const CudaEventProjection& owned,
                                                           CudaEventProjection& output) noexcept {
    return implementation_->events->lookup_owned(owned, output);
}
CUresult DriverCooperativeRouter::matches_event_epoch(const CudaEventProjection& owned,
                                                      bool& output) noexcept {
    return implementation_->events->matches_native_epoch(owned, output);
}
CUresult DriverCooperativeRouter::logical_event_completion(CUevent event, bool should_wait,
                                                           bool& needs_native_boundary) noexcept {
    needs_native_boundary = false;
    try {
        CudaEventProjection projection;
        auto status = event_projection(event, projection);
        if (status == CUDA_ERROR_INVALID_HANDLE) {
            needs_native_boundary = true;
            return CUDA_SUCCESS;
        }
        if (status != CUDA_SUCCESS) {
            return status;
        }
        if (!projection.is_logical()) {
            needs_native_boundary =
                !projection.has_native_completion() && !projection.issued_native_prefix();
            return CUDA_SUCCESS;
        }
        CUcontext context{};
        std::shared_ptr<CudaCooperativeFrontend> owner;
        status = event_frontend(event, context, owner);
        if (status != CUDA_SUCCESS) {
            return status;
        }
        const auto logical = with_context(context, [&] {
            return should_wait ? owner->synchronize_event(event) : owner->query_event(event);
        });
        // Original graph issue/publish may have occurred during actor exclusion.
        // Refresh the exact registration, not a possibly reused event handle.
        CudaEventProjection current;
        status = refresh_event_projection(projection, current);
        if (status != CUDA_SUCCESS) {
            return status;
        }
        if (!current.is_logical()) {
            needs_native_boundary =
                !current.has_native_completion() && !current.issued_native_prefix();
            return CUDA_SUCCESS;
        }
        return logical;
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}
CUresult DriverCooperativeRouter::normal_owner_status() noexcept {
    auto& state = *implementation_;
    const auto registry_status = state.events->status();
    if (registry_status != CUDA_SUCCESS) {
        return registry_status;
    }
    const std::lock_guard lock(state.mutex);
    if (state.lifecycle_status != CUDA_SUCCESS) {
        return CUDA_ERROR_UNKNOWN;
    }
    if (state.native_session) {
        return state.retained_session_status();
    }
    for (const auto& [context, record] : state.contexts) {
        (void)context;
        const auto status = record->status.load(std::memory_order_acquire);
        // An unrelated cold actor may still be initializing; it has not issued
        // an uncertain normal effect. Its eventual caller joins call_once.
        if (status != CUDA_SUCCESS && status != CUDA_ERROR_NOT_INITIALIZED) {
            return CUDA_ERROR_UNKNOWN;
        }
    }
    return CUDA_SUCCESS;
}

void DriverCooperativeRouter::quarantine_normal_launch(CUcontext context, CUresult error) noexcept {
    std::shared_ptr<CudaCooperativeFrontend> owner;
    auto& state = *implementation_;
    const auto failed = error == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : error;
    {
        const std::lock_guard lock(state.mutex);
        if (state.lifecycle_status == CUDA_SUCCESS) state.lifecycle_status = failed;
        const auto found = state.contexts.find(context);
        if (found != state.contexts.end()) {
            // A cold initializer publishes its pointer outside this mutex.
            if (found->second->status.load(std::memory_order_acquire) == CUDA_SUCCESS)
                owner = found->second->frontend;
            found->second->status.store(failed, std::memory_order_release);
        }
    }
    if (owner) owner->fail_native_boundary(failed);
}

void DriverCooperativeRouter::fail_context_owner(CUcontext context, CUresult error) noexcept {
    std::shared_ptr<CudaCooperativeFrontend> owner;
    {
        auto& state = *implementation_;
        const std::lock_guard lock(state.mutex);
        const auto found = state.contexts.find(context);
        if (found != state.contexts.end()) {
            found->second->status.store(CUDA_ERROR_UNKNOWN, std::memory_order_release);
            owner = found->second->frontend;
        }
    }
    if (owner) {
        owner->fail_native_boundary(error);
    }
}

CUresult DriverCooperativeRouter::publish_logical_event(
    const CudaEventProjection& accepted) noexcept {
    return implementation_->events->publish_logical(accepted);
}
CUresult DriverCooperativeRouter::prepare_event_record_guard(
    const CudaEventProjection& accepted,
    std::shared_ptr<CudaQueueEventRecordGuard>& output) noexcept {
    if (output) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    try {
        bool is_current = false;
        const auto result = implementation_->events->matches_native_epoch(accepted, is_current);
        if (result != CUDA_SUCCESS || !is_current) {
            return result == CUDA_SUCCESS ? CUDA_ERROR_NOT_READY : result;
        }
        output = std::make_shared<EventRecordGuard>(implementation_->events, accepted);
        return CUDA_SUCCESS;
    } catch (const std::bad_alloc&) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}
CUresult DriverCooperativeRouter::invalidate_native_events(CUcontext context) noexcept {
    return implementation_->events->invalidate_native(context);
}

CUresult DriverCooperativeRouter::native_graph_execution(CUgraphExec executable, CUstream stream,
                                                         bool is_ptds,
                                                         const NativeOperation& original) noexcept {
    try {
        if (!original) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        auto& state = *implementation_;
        CudaNativeGraphExecutable owned;
        auto status = state.graphs.lookup(executable, owned);
        if (status != CUDA_SUCCESS) {
            return status;
        }
        const NativeDriverCallScope native_scope;
        CUcontext caller{};
        status = state.dispatch.worker.queue.context_get_current(&caller);
        if (status != CUDA_SUCCESS || caller != owned.context()) {
            return status == CUDA_SUCCESS ? CUDA_ERROR_INVALID_CONTEXT : status;
        }
        auto plan = std::make_shared<NativeGraphExecution>(state.events, owned);
        status = plan->prepare();
        if (status != CUDA_SUCCESS) {
            return status;
        }
        std::shared_ptr<CudaCooperativeFrontend> owner;
        status = frontend(owned.context(), owner);
        if (status != CUDA_SUCCESS) {
            return status;
        }
        bool was_issued = false;
        CUresult issue_status = CUDA_ERROR_UNKNOWN;
        CUresult publication_status = CUDA_SUCCESS;
        status = owner->native(
            stream, is_ptds,
            [&] {
                const auto current = state.graphs.is_current(owned);
                if (current != CUDA_SUCCESS) {
                    return current;
                }
                const auto refreshed = plan->refresh_records();
                if (refreshed != CUDA_SUCCESS) {
                    return refreshed;
                }
                was_issued = true;
                issue_status = original();
                if (issue_status != CUDA_SUCCESS) {
                    throw UncertainNativeGraphIssue{};
                }
                publication_status = plan->publish_issued_records();
                if (publication_status != CUDA_SUCCESS) {
                    throw UncertainNativeGraphIssue{};
                }
                return CUDA_SUCCESS;
            },
            true, plan, plan->waited_events(), plan);
        if (was_issued && (issue_status != CUDA_SUCCESS || publication_status != CUDA_SUCCESS)) {
            const auto error = issue_status != CUDA_SUCCESS ? issue_status : publication_status;
            owner->fail_native_boundary(error);
            return error;
        }
        return status;
    } catch (const std::bad_alloc&) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}

CUresult DriverCooperativeRouter::prepare_graph(CUcontext context, CUgraph graph,
                                                CudaNativeGraphReservation& output,
                                                bool should_prepare_effects) noexcept {
    try {
        auto& state = *implementation_;
        CudaNativeGraphEffects effects;
        if (should_prepare_effects && graph) {
            // Missing/unsupported metadata retains an unprepared registration.
            // It must not invent a new native constructor validation result.
            const auto discovered = discover_graph(graph, effects);
            if (discovered == CUDA_ERROR_OUT_OF_MEMORY) {
                return discovered;
            }
        }
        const auto prepare = [&](const CudaNativeGraphEffects& value) {
            return context ? state.graphs.prepare(context, value, *state.events, output)
                           : state.graphs.prepare_unbound(value, *state.events, output);
        };
        auto prepared = prepare(effects);
        if (!output &&
            (prepared == CUDA_ERROR_INVALID_HANDLE || prepared == CUDA_ERROR_NOT_SUPPORTED)) {
            prepared = prepare({});
        }
        return prepared;
    } catch (const std::bad_alloc&) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}
CUresult DriverCooperativeRouter::discover_graph(CUgraph graph,
                                                 CudaNativeGraphEffects& output) noexcept {
    try {
        auto& state = *implementation_;
        const auto optional = [&]<typename Function>(const char* name) -> Function {
            try {
                return resolve_native<Function>(state.driver, name);
            } catch (...) {
                return nullptr;
            }
        };
        std::call_once(state.graph_dispatch_once, [&] {
            state.graph_dispatch = {
                optional.operator()<decltype(&cuGraphGetNodes)>("cuGraphGetNodes"),
                optional.operator()<decltype(&cuGraphNodeGetType)>("cuGraphNodeGetType"),
                optional.operator()<decltype(&cuGraphChildGraphNodeGetGraph)>(
                    "cuGraphChildGraphNodeGetGraph"),
                optional.operator()<decltype(&cuGraphEventRecordNodeGetEvent)>(
                    "cuGraphEventRecordNodeGetEvent"),
                optional.operator()<decltype(&cuGraphEventWaitNodeGetEvent)>(
                    "cuGraphEventWaitNodeGetEvent")};
        });
        const NativeDriverCallScope native;
        return prepare_native_graph_effects(graph, state.graph_dispatch, output);
    } catch (const std::bad_alloc&) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}

CUresult DriverCooperativeRouter::prepare_graph_update(
    CUgraphExec executable, CUgraph graph, std::optional<CUgraphNode> child_node,
    CudaNativeGraphReservation& output) noexcept {
    CudaNativeGraphEffects effects;
    if (graph) {
        const auto discovered = discover_graph(graph, effects);
        if (discovered == CUDA_ERROR_OUT_OF_MEMORY) return discovered;
    }
    return implementation_->graphs.prepare_graph_update(executable, effects, child_node,
                                                        *implementation_->events, output);
}

CudaNativeGraphRegistry& DriverCooperativeRouter::graphs() noexcept {
    return implementation_->graphs;
}
CUresult DriverCooperativeRouter::prepare_graph_event_update(
    CUgraphExec executable, CUgraphNode node, CUevent event, bool is_record,
    CudaNativeGraphReservation& output) noexcept {
    return implementation_->graphs.prepare_event_update(
        executable, node, event,
        is_record ? CudaNativeGraphEventKind::kRecord : CudaNativeGraphEventKind::kWait,
        *implementation_->events, output);
}
CUresult DriverCooperativeRouter::with_context(CUcontext context,
                                               const NativeOperation& operation) noexcept {
    try {
        return implementation_->in_context(context, operation);
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}
std::shared_ptr<void> DriverCooperativeRouter::resources() const noexcept {
    return implementation_->lifetime;
}
decltype(&cuPointerGetAttribute) DriverCooperativeRouter::pointer_attribute() const noexcept {
    return implementation_->pointer_attribute;
}
const CudaCopyRangeDispatch& DriverCooperativeRouter::copy_range_dispatch() const noexcept {
    return implementation_->copy_range;
}
}  // namespace glimmer::interceptor
