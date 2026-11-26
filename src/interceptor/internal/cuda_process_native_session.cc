#include "cuda_process_native_session.h"

#include "diagnostics.h"

#include <algorithm>
#include <atomic>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace glimmer::interceptor {
struct CudaProcessNativeSession::Implementation {
    enum class Phase { kPrepared, kPausing, kPaused, kCompleting, kFinished, kFailed };
    struct Context {
        CudaProcessNativeContext owner;
        CudaNativeBoundaryToken boundary;
    };
    std::vector<Context> contexts;
    ContextOperation in_context;
    std::shared_ptr<void> native_reservation;
    const std::shared_ptr<const void> identity{std::make_shared<int>(0)};
    const std::size_t max_contexts;
    Phase phase{Phase::kPrepared};
    std::atomic<CUresult> result{CUDA_ERROR_NOT_READY};
    std::mutex observation_mutex;

    Implementation(std::vector<CudaProcessNativeContext> owners, ContextOperation activate,
                   std::shared_ptr<void> reservation, std::size_t limit)
        : in_context(std::move(activate)),
          native_reservation(std::move(reservation)),
          max_contexts(limit) {
        if (!in_context || !native_reservation || !limit || limit > 256 || owners.size() > limit) {
            throw std::invalid_argument("invalid process-native session ownership");
        }
        contexts.reserve(limit);
        for (auto& owner : owners) {
            if (!owner.context ||
                std::any_of(contexts.begin(), contexts.end(), [&](const auto& item) {
                    return item.owner.context == owner.context;
                })) {
                throw std::invalid_argument("invalid process-native context snapshot");
            }
            contexts.push_back({std::move(owner), {}});
        }
    }
    CUresult mark_failed(CUresult error) noexcept {
        if (error == CUDA_SUCCESS || error == CUDA_ERROR_NOT_READY) {
            error = CUDA_ERROR_UNKNOWN;
        }
        phase = Phase::kFailed;
        result.store(error);
        return error;
    }
    void notify_failed(CUresult error) noexcept {
        for (const auto& item : contexts) {
            if (item.owner.frontend) {
                item.owner.frontend->fail_native_boundary(error);
            }
        }
    }
    CUresult fail(CUresult error) noexcept {
        error = mark_failed(error);
        notify_failed(error);
        return error;
    }
};

CudaProcessNativeSession::CudaProcessNativeSession(std::vector<CudaProcessNativeContext> contexts,
                                                   ContextOperation in_context,
                                                   std::shared_ptr<void> native_reservation,
                                                   std::size_t max_contexts)
    : implementation_(std::make_unique<Implementation>(std::move(contexts), std::move(in_context),
                                                       std::move(native_reservation),
                                                       max_contexts)) {}
CudaProcessNativeSession::~CudaProcessNativeSession() {
    if (implementation_->phase != Implementation::Phase::kPrepared &&
        implementation_->phase != Implementation::Phase::kFinished) {
        report_diagnostic("[glimmer] unfinished process-native session retained\n");
        (void)implementation_.release();
    }
}

CUresult CudaProcessNativeSession::begin(CudaProcessNativeToken& token) noexcept {
    auto& state = *implementation_;
    if (token || state.phase != Implementation::Phase::kPrepared) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    state.phase = Implementation::Phase::kPausing;
    try {
        for (auto& item : state.contexts) {
            if (!item.owner.frontend) {
                continue;
            }
            const auto result = state.in_context(
                item.owner.context, [&] { return item.owner.frontend->pause(item.boundary); });
            if (result != CUDA_SUCCESS) {
                return state.fail(result);
            }
        }
        state.phase = Implementation::Phase::kPaused;
        token.owner_ = state.identity;
        return CUDA_SUCCESS;
    } catch (...) {
        return state.fail(CUDA_ERROR_UNKNOWN);
    }
}

CUresult CudaProcessNativeSession::observe_native_context(CUcontext context) noexcept {
    auto& state = *implementation_;
    try {
        std::unique_lock lock(state.observation_mutex);
        if (!context || state.phase != Implementation::Phase::kPaused ||
            state.result.load() != CUDA_ERROR_NOT_READY) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        if (std::any_of(state.contexts.begin(), state.contexts.end(),
                        [&](const auto& item) { return item.owner.context == context; })) {
            return CUDA_SUCCESS;
        }
        if (state.contexts.size() >= state.max_contexts) {
            // No observer can append after FAILED. Notify only after unlocking;
            // frontend callbacks never execute under this metadata mutex.
            const auto error = state.mark_failed(CUDA_ERROR_OUT_OF_MEMORY);
            lock.unlock();
            state.notify_failed(error);
            return error;
        }
        // Capacity is reserved before pause and default shared_ptr/token copies
        // cannot throw. No allocation, native call or callback is needed here.
        state.contexts.push_back({{context, {}, {}}, {}});
        return CUDA_SUCCESS;
    } catch (...) {
        // A failed metadata lock cannot authorize iteration of a concurrently
        // observed snapshot. Retain it and make all later finish checks reject.
        state.result.store(CUDA_ERROR_UNKNOWN);
        return CUDA_ERROR_UNKNOWN;
    }
}

CUresult CudaProcessNativeSession::finish(const CudaProcessNativeToken& token,
                                          const NativeOperation& verify_inactive,
                                          const NativeOperation& complete_native_context) noexcept {
    auto& state = *implementation_;
    if (!token || token.owner_ != state.identity || state.phase != Implementation::Phase::kPaused ||
        state.result.load() != CUDA_ERROR_NOT_READY || !verify_inactive ||
        !complete_native_context) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    try {
        if (!is_paused()) {
            return state.fail(CUDA_ERROR_UNKNOWN);
        }
        state.phase = Implementation::Phase::kCompleting;
        const auto inactive = verify_inactive();
        if (inactive == CUDA_ERROR_NOT_READY) {
            state.phase = Implementation::Phase::kPaused;
            return inactive;
        }
        if (inactive != CUDA_SUCCESS) {
            return state.fail(inactive);
        }
        // All context completion must precede any private-worker thaw.
        for (const auto& item : state.contexts) {
            const auto result = state.in_context(item.owner.context, complete_native_context);
            if (result != CUDA_SUCCESS) {
                return state.fail(result);
            }
        }
        for (auto& item : state.contexts) {
            if (!item.owner.frontend) {
                continue;
            }
            const auto result = state.in_context(item.owner.context, [&] {
                return item.owner.frontend->finish_pause(item.boundary, complete_native_context);
            });
            if (result != CUDA_SUCCESS) {
                return state.fail(result);
            }
        }
        state.phase = Implementation::Phase::kFinished;
        state.result.store(CUDA_SUCCESS);
        state.native_reservation.reset();
        state.contexts.clear();
        return CUDA_SUCCESS;
    } catch (...) {
        return state.fail(CUDA_ERROR_UNKNOWN);
    }
}

CUresult CudaProcessNativeSession::status() const noexcept {
    return implementation_->result.load();
}
CUresult CudaProcessNativeSession::retire_event(const CudaProcessNativeToken& token,
                                                CUcontext context, CUevent event,
                                                bool should_retire) noexcept {
    auto& state = *implementation_;
    if (!owns_paused_token(token) || state.phase != Implementation::Phase::kPaused || !context ||
        !event) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const auto found =
        std::find_if(state.contexts.begin(), state.contexts.end(),
                     [&](const auto& item) { return item.owner.context == context; });
    if (found == state.contexts.end() || !found->owner.frontend) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    return found->owner.frontend->retire_paused_event(found->boundary, event, should_retire);
}
CUresult CudaProcessNativeSession::retire_stream(const CudaProcessNativeToken& token,
                                                 CUcontext context, CUstream stream,
                                                 bool should_retire) noexcept {
    auto& state = *implementation_;
    if (!owns_paused_token(token) || state.phase != Implementation::Phase::kPaused || !context ||
        !stream || stream == CU_STREAM_LEGACY || stream == CU_STREAM_PER_THREAD) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const auto found =
        std::find_if(state.contexts.begin(), state.contexts.end(),
                     [&](const auto& item) { return item.owner.context == context; });
    if (found == state.contexts.end()) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    return found->owner.frontend
               ? found->owner.frontend->retire_paused_stream(found->boundary, stream, should_retire)
               : CUDA_ERROR_INVALID_HANDLE;
}

bool CudaProcessNativeSession::owns_paused_token(
    const CudaProcessNativeToken& token) const noexcept {
    return token && token.owner_ == implementation_->identity &&
           implementation_->result.load() == CUDA_ERROR_NOT_READY;
}
bool CudaProcessNativeSession::is_paused() const noexcept {
    const auto& state = *implementation_;
    return (state.phase == Implementation::Phase::kPaused ||
            state.phase == Implementation::Phase::kCompleting) &&
           std::all_of(state.contexts.begin(), state.contexts.end(), [](const auto& item) {
               return !item.owner.frontend ||
                      item.owner.frontend->native_boundary_is_paused(item.boundary);
           });
}
}  // namespace glimmer::interceptor
