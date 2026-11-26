#include "internal/cuda_external_execution_bridge.h"

#include "internal/cuda_execution_session.h"

#include <atomic>
#include <stdexcept>
#include <utility>
#include <unistd.h>

namespace glimmer::backend::cuda {
namespace {
thread_local std::shared_ptr<const void> bridge_thread_identity;

enum class BridgePhase : unsigned char {
    kCreated,
    kPrepared,
    kBegun,
    kEnqueued,
    kJoined,
    kSourceCleaned,
    kComplete,
};

class BridgeOperation final {
   public:
    explicit BridgeOperation(std::atomic_flag& active) noexcept
        : active_(active), has_lock_(!active.test_and_set(std::memory_order_acquire)) {}
    ~BridgeOperation() {
        if (has_lock_) active_.clear(std::memory_order_release);
    }
    BridgeOperation(const BridgeOperation&) = delete;
    BridgeOperation& operator=(const BridgeOperation&) = delete;
    [[nodiscard]] bool acquired() const noexcept {
        return has_lock_;
    }

   private:
    std::atomic_flag& active_;
    bool has_lock_;
};
}  // namespace

// The worker can retain this state, but never the source frontier or a mutable
// session/dispatcher. All event/worker-identity access is operation-excluded;
// atomic phase/failure observations convey no native completion authority.
struct CudaExternalExecutionBridgeState {
    const pid_t process{::getpid()};
    const CudaSessionExternalCommand source;
    const CudaExternalExecutionDomain execution;
    const CudaExternalExecutionBridgeDispatch dispatch;
    const std::weak_ptr<const void> source_thread;
    std::atomic_flag active = ATOMIC_FLAG_INIT;
    std::atomic<BridgePhase> phase{BridgePhase::kCreated};
    std::atomic<CUresult> failure{CUDA_SUCCESS};
    std::atomic<bool> has_native_attempt{};
    std::weak_ptr<const void> worker_thread;
    bool has_worker{};
    CUevent source_event{};
    CUevent execution_event{};

    CudaExternalExecutionBridgeState(CudaSessionExternalCommand source_value,
                                     CudaExternalExecutionDomain execution_value,
                                     CudaExternalExecutionBridgeDispatch dispatch_value,
                                     std::weak_ptr<const void> source_thread_value)
        : source(std::move(source_value)),
          execution(std::move(execution_value)),
          dispatch(dispatch_value),
          source_thread(std::move(source_thread_value)) {}

    [[nodiscard]] CUresult fail(CUresult status) noexcept {
        const auto prior = failure.load(std::memory_order_acquire);
        if (prior != CUDA_SUCCESS) return prior;
        failure.store(status, std::memory_order_release);
        return status;
    }

    [[nodiscard]] CUresult check_context(CUcontext expected) noexcept {
        has_native_attempt.store(true, std::memory_order_release);
        CUcontext current{};
        const auto status = dispatch.context_get_current(&current);
        if (status != CUDA_SUCCESS) return fail(status);
        return current == expected ? CUDA_SUCCESS : fail(CUDA_ERROR_INVALID_CONTEXT);
    }

    template <typename Callback>
    [[nodiscard]] CUresult invoke(CUcontext context, Callback&& callback) noexcept {
        if (const auto status = check_context(context); status != CUDA_SUCCESS) return status;
        const auto status = std::forward<Callback>(callback)();
        // A failed callback may still have changed native current context or
        // written an event output. Never attempt cleanup after a lost proof.
        if (const auto proof = check_context(context); proof != CUDA_SUCCESS) return proof;
        return status == CUDA_SUCCESS ? CUDA_SUCCESS : fail(status);
    }

    [[nodiscard]] bool owns_worker() const noexcept {
        return has_worker && bridge_thread_identity &&
               worker_thread.lock() == bridge_thread_identity;
    }
};

struct CudaExternalExecutionBridge::Implementation {
    std::shared_ptr<CudaExternalExecutionBridgeState> state;
    CudaExecutionFrontier tail;
    std::weak_ptr<const void> source_thread;

    [[nodiscard]] bool owns_source() const noexcept {
        return state->process == ::getpid() && bridge_thread_identity &&
               source_thread.lock() == bridge_thread_identity;
    }
};

CudaExternalExecutionBridge::CudaExternalExecutionBridge(
    CudaSessionExternalCommand source, CudaExecutionFrontier tail,
    CudaExternalExecutionDomain execution, CudaExternalExecutionBridgeDispatch dispatch) {
    const auto source_context = source.context();
    if (!source_context.native() || !source_context.id() || source_context.device_ordinal() < 0 ||
        source.stream() == CU_STREAM_LEGACY || source.stream() == CU_STREAM_PER_THREAD ||
        (source.stream_id() != 0 && !source.stream()) || !execution.context.id ||
        !execution.context.native || !execution.context.lifetime || !execution.stream ||
        execution.stream == CU_STREAM_LEGACY || execution.stream == CU_STREAM_PER_THREAD ||
        !execution.stream_lifetime || execution.device_ordinal != source_context.device_ordinal() ||
        !dispatch.context_get_current || !dispatch.event_create || !dispatch.event_record ||
        !dispatch.event_destroy || !dispatch.stream_wait_event ||
        tail.observe() != CUDA_ERROR_NOT_READY)
        throw std::invalid_argument("invalid CUDA external execution bridge");
    if (!bridge_thread_identity) bridge_thread_identity = std::make_shared<int>(0);
    auto owned = std::make_unique<Implementation>();
    owned->state = std::make_shared<CudaExternalExecutionBridgeState>(
        std::move(source), std::move(execution), dispatch, bridge_thread_identity);
    owned->tail = std::move(tail);
    owned->source_thread = bridge_thread_identity;
    implementation_ = std::move(owned);
}

CudaExternalExecutionBridge::~CudaExternalExecutionBridge() {
    const auto& state = *implementation_->state;
    // A fully cleaned core has no remaining native objects and may be released
    // on another thread. Forked or unsettled owners are never cleaned here.
    if (state.process == ::getpid()) {
        if (state.phase.load(std::memory_order_acquire) == BridgePhase::kComplete) return;
        if (implementation_->owns_source() &&
            !state.has_native_attempt.load(std::memory_order_acquire) &&
            state.failure.load(std::memory_order_acquire) == CUDA_SUCCESS)
            return;
    }
    // Keep the frontier on the original source side, including when the last
    // worker endpoint is dropped. No destructor calls SDK or asserts GPU stop.
    (void)implementation_.release();
}

CudaExternalExecutionWorker CudaExternalExecutionBridge::worker() const noexcept {
    CudaExternalExecutionWorker worker;
    if (implementation_->owns_source()) worker.state_ = implementation_->state;
    return worker;
}

bool CudaExternalExecutionBridge::matches_source(CUcontext context,
                                                 std::uint64_t id) const noexcept {
    if (!implementation_->owns_source()) return false;
    const auto source = implementation_->state->source.context();
    return source.native() == context && source.id() == id;
}

bool CudaExternalExecutionBridge::completed() const noexcept {
    if (!implementation_->owns_source()) return false;
    const auto& state = *implementation_->state;
    return !state.active.test(std::memory_order_acquire) &&
           state.phase.load(std::memory_order_acquire) == BridgePhase::kComplete;
}

CUresult CudaExternalExecutionBridge::failure() const noexcept {
    if (!implementation_->owns_source()) return CUDA_ERROR_INVALID_CONTEXT;
    auto& state = *implementation_->state;
    const BridgeOperation operation(state.active);
    if (!operation.acquired()) return CUDA_ERROR_NOT_READY;
    return state.failure.load(std::memory_order_acquire);
}

CUresult CudaExternalExecutionBridge::prepare_source() noexcept {
    if (!implementation_->owns_source()) return CUDA_ERROR_INVALID_CONTEXT;
    auto& state = *implementation_->state;
    const BridgeOperation operation(state.active);
    if (!operation.acquired()) return CUDA_ERROR_NOT_READY;
    if (const auto status = state.failure.load(std::memory_order_acquire); status != CUDA_SUCCESS)
        return status;
    if (state.phase.load(std::memory_order_acquire) != BridgePhase::kCreated)
        return CUDA_ERROR_INVALID_VALUE;
    const auto context = state.source.context().native();
    auto status = state.invoke(context, [&] {
        return state.dispatch.event_create(&state.source_event, CU_EVENT_DISABLE_TIMING);
    });
    if (status != CUDA_SUCCESS) return status;
    if (!state.source_event) return state.fail(CUDA_ERROR_UNKNOWN);
    status = state.invoke(context, [&] {
        return state.dispatch.event_record(state.source_event, state.source.stream());
    });
    if (status == CUDA_SUCCESS)
        state.phase.store(BridgePhase::kPrepared, std::memory_order_release);
    return status;
}

CUresult CudaExternalExecutionWorker::begin() noexcept {
    if (!state_) return CUDA_ERROR_INVALID_HANDLE;
    if (state_->process != ::getpid()) return CUDA_ERROR_INVALID_CONTEXT;
    const auto state = state_;
    const BridgeOperation operation(state->active);
    if (!operation.acquired()) return CUDA_ERROR_NOT_READY;
    // Worker callbacks do not hold the mutable source session's native scope.
    // They must therefore run on an independent lane, never its owner thread.
    if (bridge_thread_identity && state->source_thread.lock() == bridge_thread_identity)
        return CUDA_ERROR_INVALID_CONTEXT;
    if (state->has_worker && !state->owns_worker()) return CUDA_ERROR_INVALID_CONTEXT;
    if (const auto status = state->failure.load(std::memory_order_acquire); status != CUDA_SUCCESS)
        return status;
    const auto phase = state->phase.load(std::memory_order_acquire);
    if (phase == BridgePhase::kCreated) return CUDA_ERROR_NOT_READY;
    if (phase != BridgePhase::kPrepared) return CUDA_ERROR_INVALID_VALUE;
    try {
        if (!bridge_thread_identity) bridge_thread_identity = std::make_shared<int>(0);
    } catch (...) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    state->worker_thread = bridge_thread_identity;
    state->has_worker = true;
    const auto context = state->execution.context.native;
    auto status = state->invoke(context, [&] {
        return state->dispatch.event_create(&state->execution_event, CU_EVENT_DISABLE_TIMING);
    });
    if (status != CUDA_SUCCESS) return status;
    if (!state->execution_event || state->execution_event == state->source_event)
        return state->fail(CUDA_ERROR_UNKNOWN);
    status = state->invoke(context, [&] {
        return state->dispatch.stream_wait_event(state->execution.stream, state->source_event, 0);
    });
    if (status == CUDA_SUCCESS) state->phase.store(BridgePhase::kBegun, std::memory_order_release);
    return status;
}

CUresult CudaExternalExecutionWorker::record_enqueue() noexcept {
    if (!state_) return CUDA_ERROR_INVALID_HANDLE;
    if (state_->process != ::getpid()) return CUDA_ERROR_INVALID_CONTEXT;
    const auto state = state_;
    const BridgeOperation operation(state->active);
    if (!operation.acquired()) return CUDA_ERROR_NOT_READY;
    if (state->has_worker && !state->owns_worker()) return CUDA_ERROR_INVALID_CONTEXT;
    if (const auto status = state->failure.load(std::memory_order_acquire); status != CUDA_SUCCESS)
        return status;
    if (!state->has_worker) return CUDA_ERROR_NOT_READY;
    if (state->phase.load(std::memory_order_acquire) != BridgePhase::kBegun)
        return CUDA_ERROR_INVALID_VALUE;
    const auto status = state->invoke(state->execution.context.native, [&] {
        return state->dispatch.event_record(state->execution_event, state->execution.stream);
    });
    if (status == CUDA_SUCCESS)
        state->phase.store(BridgePhase::kEnqueued, std::memory_order_release);
    return status;
}

CUresult CudaExternalExecutionBridge::join_source() noexcept {
    if (!implementation_->owns_source()) return CUDA_ERROR_INVALID_CONTEXT;
    auto& state = *implementation_->state;
    const BridgeOperation operation(state.active);
    if (!operation.acquired()) return CUDA_ERROR_NOT_READY;
    if (const auto status = state.failure.load(std::memory_order_acquire); status != CUDA_SUCCESS)
        return status;
    const auto phase = state.phase.load(std::memory_order_acquire);
    if (phase < BridgePhase::kEnqueued) return CUDA_ERROR_NOT_READY;
    if (phase != BridgePhase::kEnqueued) return CUDA_ERROR_INVALID_VALUE;
    const auto status = state.invoke(state.source.context().native(), [&] {
        return state.dispatch.stream_wait_event(state.source.stream(), state.execution_event, 0);
    });
    if (status == CUDA_SUCCESS) state.phase.store(BridgePhase::kJoined, std::memory_order_release);
    return status;
}

CUresult CudaExternalExecutionBridge::cleanup_source() noexcept {
    if (!implementation_->owns_source()) return CUDA_ERROR_INVALID_CONTEXT;
    auto& state = *implementation_->state;
    const BridgeOperation operation(state.active);
    if (!operation.acquired()) return CUDA_ERROR_NOT_READY;
    if (const auto status = state.failure.load(std::memory_order_acquire); status != CUDA_SUCCESS)
        return status;
    const auto phase = state.phase.load(std::memory_order_acquire);
    if (phase < BridgePhase::kJoined) return CUDA_ERROR_NOT_READY;
    if (phase != BridgePhase::kJoined) return CUDA_ERROR_INVALID_VALUE;
    const auto stopped = implementation_->tail.observe();
    if (stopped == CUDA_ERROR_NOT_READY) return stopped;
    if (stopped != CUDA_SUCCESS) return state.fail(stopped);
    const auto status = state.invoke(state.source.context().native(), [&] {
        return state.dispatch.event_destroy(state.source_event);
    });
    if (status == CUDA_SUCCESS) {
        state.source_event = nullptr;
        state.phase.store(BridgePhase::kSourceCleaned, std::memory_order_release);
    }
    return status;
}

CUresult CudaExternalExecutionWorker::cleanup() noexcept {
    if (!state_) return CUDA_ERROR_INVALID_HANDLE;
    if (state_->process != ::getpid()) return CUDA_ERROR_INVALID_CONTEXT;
    const auto state = state_;
    const BridgeOperation operation(state->active);
    if (!operation.acquired()) return CUDA_ERROR_NOT_READY;
    if (state->has_worker && !state->owns_worker()) return CUDA_ERROR_INVALID_CONTEXT;
    if (const auto status = state->failure.load(std::memory_order_acquire); status != CUDA_SUCCESS)
        return status;
    if (!state->has_worker) return CUDA_ERROR_NOT_READY;
    const auto phase = state->phase.load(std::memory_order_acquire);
    if (phase < BridgePhase::kSourceCleaned) return CUDA_ERROR_NOT_READY;
    if (phase != BridgePhase::kSourceCleaned) return CUDA_ERROR_INVALID_VALUE;
    const auto status = state->invoke(state->execution.context.native, [&] {
        return state->dispatch.event_destroy(state->execution_event);
    });
    if (status == CUDA_SUCCESS) {
        state->execution_event = nullptr;
        state->phase.store(BridgePhase::kComplete, std::memory_order_release);
    }
    return status;
}

}  // namespace glimmer::backend::cuda
