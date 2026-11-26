#include "internal/cuda_nccl_primary_domain.h"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>
#include <unistd.h>

namespace glimmer::backend::cuda {
namespace {
thread_local std::shared_ptr<const void> primary_thread_identity;

struct ContextOwner {
    CUcontext native{};
};
struct StreamOwner {
    std::uint64_t id{};
    CUstream native{};
    std::shared_ptr<ContextOwner> context;
};
struct ContextTransition {
    CUcontext before{}, after{};
};
}  // namespace

struct CudaNcclPrimaryDomain::Implementation {
    const pid_t process{::getpid()};
    const std::weak_ptr<const void> owner;
    const CudaNcclPrimaryDomainDispatch dispatch;
    const CudaNcclPrimaryDomainConfig config;
    std::shared_ptr<ContextOwner> context = std::make_shared<ContextOwner>();
    std::vector<std::shared_ptr<StreamOwner>> streams;
    CUdevice device{};
    CUresult failure{CUDA_SUCCESS};
    std::uint64_t next_stream{1};
    bool active{}, is_initialized{}, is_closed{}, is_shutdown{}, has_retain_attempt{};
    bool is_context_proven{};

    Implementation(CudaNcclPrimaryDomainDispatch callbacks, CudaNcclPrimaryDomainConfig settings)
        : owner(primary_thread_identity), dispatch(callbacks), config(settings) {
        streams.reserve(config.maximum_streams);
    }
    CUresult gate() const noexcept {
        if (::getpid() != process || !primary_thread_identity ||
            owner.lock() != primary_thread_identity)
            return CUDA_ERROR_INVALID_CONTEXT;
        if (active) return CUDA_ERROR_NOT_READY;
        return failure;
    }
    struct Scope {
        explicit Scope(Implementation& state) : state_(state) {
            state_.active = true;
        }
        ~Scope() {
            state_.active = false;
        }
        Scope(const Scope&) = delete;
        Scope& operator=(const Scope&) = delete;
        Implementation& state_;
    };
    CUresult fail(CUresult status) noexcept {
        if (failure == CUDA_SUCCESS) failure = status;
        return failure;
    }
    CUresult current(CUcontext& output) noexcept {
        is_context_proven = false;
        const auto status = dispatch.context_get_current(&output);
        if (status != CUDA_SUCCESS) return fail(status);
        is_context_proven = true;
        return CUDA_SUCCESS;
    }
    CUresult prove(CUcontext expected) noexcept {
        CUcontext observed{};
        const auto status = current(observed);
        if (status != CUDA_SUCCESS) return status;
        if (observed != expected) {
            is_context_proven = false;
            return fail(CUDA_ERROR_INVALID_CONTEXT);
        }
        return CUDA_SUCCESS;
    }
    template <typename Callback>
    CUresult invoke(CUcontext expected, Callback&& callback, bool can_be_pending = false) noexcept {
        if (const auto status = prove(expected); status != CUDA_SUCCESS) return status;
        const auto status = std::forward<Callback>(callback)();
        // Even a failed callback can change current context or partially create
        // an owner. Proof failure forbids restoration or further SDK calls.
        const auto proof = prove(expected);
        if (proof != CUDA_SUCCESS) return proof;
        if (status != CUDA_SUCCESS && !(can_be_pending && status == CUDA_ERROR_NOT_READY))
            return fail(status);
        return status;
    }
    CUresult select(ContextTransition transition) noexcept {
        const auto [before, after] = transition;
        if (before == after) return prove(before);
        if (const auto status = prove(before); status != CUDA_SUCCESS) return status;
        const auto status = dispatch.context_set_current(after);
        const auto proof = prove(status == CUDA_SUCCESS ? after : before);
        if (status != CUDA_SUCCESS) {
            is_context_proven = false;
            return fail(status);
        }
        return proof;
    }
    CUresult restore(CUcontext saved, CUresult result) noexcept {
        if (!is_context_proven) return failure;
        const auto restored = select({context->native, saved});
        if (restored != CUDA_SUCCESS) return restored;
        return result;
    }
    CudaExecutionContext exported_context() const noexcept {
        return {config.identity, context->native, context};
    }
    bool has_external_owners() const noexcept {
        // Each internal stream retains exactly one context reference. Exported
        // stream.domain.context copies and independent communicator capsules add
        // references beyond this count; stream-only aliases are checked too.
        if (context.use_count() != static_cast<long>(1 + streams.size())) return true;
        return std::ranges::any_of(streams, [](const auto& item) { return item.use_count() != 1; });
    }
    CUresult retire(const std::shared_ptr<StreamOwner>& stream, CUcontext saved) noexcept {
        auto status = select({saved, context->native});
        if (status != CUDA_SUCCESS) return status;
        status =
            invoke(context->native, [&] { return dispatch.stream_query(stream->native); }, true);
        if (status != CUDA_SUCCESS) return restore(saved, status);
        status = invoke(context->native, [&] { return dispatch.stream_destroy(stream->native); });
        status = restore(saved, status);
        if (status == CUDA_SUCCESS) std::erase(streams, stream);
        return status;
    }
};

CudaNcclPrimaryDomain::CudaNcclPrimaryDomain(CudaNcclPrimaryDomainDispatch dispatch,
                                             CudaNcclPrimaryDomainConfig config) {
    if (!config.identity || config.device_ordinal < 0 || !config.maximum_streams ||
        !dispatch.device_get || !dispatch.primary_retain || !dispatch.primary_release ||
        !dispatch.context_get_current || !dispatch.context_set_current || !dispatch.stream_create ||
        !dispatch.stream_query || !dispatch.stream_destroy)
        throw std::invalid_argument("invalid NCCL primary execution domain");
    if (!primary_thread_identity) primary_thread_identity = std::make_shared<int>(0);
    implementation_ = std::make_unique<Implementation>(dispatch, config);
}
CudaNcclPrimaryDomain::~CudaNcclPrimaryDomain() {
    if (::getpid() != implementation_->process ||
        (implementation_->has_retain_attempt && !implementation_->is_shutdown))
        (void)implementation_.release();
}
CUresult CudaNcclPrimaryDomain::initialize() noexcept {
    auto& state = *implementation_;
    const auto gate = state.gate();
    if (gate != CUDA_SUCCESS) return gate;
    const Implementation::Scope scope(state);
    if (state.is_closed) return CUDA_ERROR_DEINITIALIZED;
    if (state.is_initialized) return CUDA_ERROR_INVALID_VALUE;
    CUcontext saved{};
    if (const auto status = state.current(saved); status != CUDA_SUCCESS) return status;
    auto status = state.invoke(saved, [&] {
        return state.dispatch.device_get(&state.device, state.config.device_ordinal);
    });
    if (status != CUDA_SUCCESS) return status;
    if (state.device < 0) return state.fail(CUDA_ERROR_UNKNOWN);
    state.has_retain_attempt = true;
    status = state.invoke(
        saved, [&] { return state.dispatch.primary_retain(&state.context->native, state.device); });
    if (status != CUDA_SUCCESS) return status;
    if (!state.context->native) return state.fail(CUDA_ERROR_UNKNOWN);
    state.is_initialized = true;
    return CUDA_SUCCESS;
}
CudaNcclPrimaryContextCapture CudaNcclPrimaryDomain::capture_context() noexcept {
    auto& state = *implementation_;
    const auto gate = state.gate();
    if (gate != CUDA_SUCCESS) return {.status = gate};
    if (state.is_closed) return {.status = CUDA_ERROR_DEINITIALIZED};
    if (!state.is_initialized) return {.status = CUDA_ERROR_NOT_INITIALIZED};
    return {CUDA_SUCCESS, state.exported_context()};
}
CudaNcclPrimaryStreamCapture CudaNcclPrimaryDomain::create_stream() noexcept {
    auto& state = *implementation_;
    const auto gate = state.gate();
    if (gate != CUDA_SUCCESS) return {.status = gate};
    const Implementation::Scope scope(state);
    if (state.is_closed) return {.status = CUDA_ERROR_DEINITIALIZED};
    if (!state.is_initialized) return {.status = CUDA_ERROR_NOT_INITIALIZED};
    if (!state.next_stream || state.streams.size() == state.config.maximum_streams)
        return {.status = CUDA_ERROR_OUT_OF_MEMORY};
    std::shared_ptr<StreamOwner> stream;
    try {
        stream = std::make_shared<StreamOwner>();
    } catch (...) {
        return {.status = CUDA_ERROR_OUT_OF_MEMORY};
    }
    stream->id = state.next_stream;
    stream->context = state.context;
    CUcontext saved{};
    if (const auto status = state.current(saved); status != CUDA_SUCCESS) return {.status = status};
    auto status = state.select({saved, state.context->native});
    if (status != CUDA_SUCCESS) return {.status = status};
    // Keep partial outputs before any native attempt, without allocation after it.
    state.streams.push_back(stream);
    ++state.next_stream;
    status = state.invoke(state.context->native, [&] {
        return state.dispatch.stream_create(&stream->native, CU_STREAM_NON_BLOCKING);
    });
    if (status == CUDA_SUCCESS && (!stream->native || stream->native == CU_STREAM_LEGACY ||
                                   stream->native == CU_STREAM_PER_THREAD ||
                                   std::ranges::any_of(state.streams, [&](const auto& other) {
                                       return other != stream && other->native == stream->native;
                                   })))
        status = state.fail(CUDA_ERROR_UNKNOWN);
    status = state.restore(saved, status);
    if (status != CUDA_SUCCESS) return {.status = status};
    return {CUDA_SUCCESS,
            stream->id,
            {state.exported_context(), state.config.device_ordinal, stream->native, stream}};
}
CUresult CudaNcclPrimaryDomain::release_stream(std::uint64_t stream_id) noexcept {
    auto& state = *implementation_;
    const auto gate = state.gate();
    if (gate != CUDA_SUCCESS) return gate;
    const Implementation::Scope scope(state);
    const auto found = std::ranges::find_if(
        state.streams, [stream_id](const auto& stream) { return stream->id == stream_id; });
    if (found == state.streams.end()) return CUDA_ERROR_INVALID_HANDLE;
    if (found->use_count() != 1) return CUDA_ERROR_NOT_READY;
    CUcontext saved{};
    if (const auto status = state.current(saved); status != CUDA_SUCCESS) return status;
    const auto stream = *found;
    return state.retire(stream, saved);
}
CUresult CudaNcclPrimaryDomain::close() noexcept {
    auto& state = *implementation_;
    const auto gate = state.gate();
    if (gate != CUDA_SUCCESS) return gate;
    state.is_closed = true;
    return CUDA_SUCCESS;
}
CUresult CudaNcclPrimaryDomain::shutdown() noexcept {
    auto& state = *implementation_;
    const auto gate = state.gate();
    if (gate != CUDA_SUCCESS) return gate;
    const Implementation::Scope scope(state);
    if (state.is_shutdown) return CUDA_SUCCESS;
    if (!state.is_closed || state.has_external_owners()) return CUDA_ERROR_NOT_READY;
    if (!state.is_initialized) {
        state.is_shutdown = true;
        return CUDA_SUCCESS;
    }
    CUcontext saved{};
    if (const auto status = state.current(saved); status != CUDA_SUCCESS) return status;
    if (saved == state.context->native) return CUDA_ERROR_NOT_READY;
    while (!state.streams.empty()) {
        const auto stream = state.streams.back();
        const auto status = state.retire(stream, saved);
        if (status != CUDA_SUCCESS) return status;
    }
    const auto status =
        state.invoke(saved, [&] { return state.dispatch.primary_release(state.device); });
    if (status != CUDA_SUCCESS) return status;
    state.is_shutdown = true;
    return CUDA_SUCCESS;
}

}  // namespace glimmer::backend::cuda
