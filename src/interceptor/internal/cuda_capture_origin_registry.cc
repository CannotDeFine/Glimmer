#include "cuda_capture_origin_registry.h"

#include <algorithm>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <vector>

namespace glimmer::interceptor {
namespace {
const std::shared_ptr<const void>& current_caller() {
    // Weak origin ownership expires on thread exit. Reused OS thread IDs never
    // authorize querying another caller's implicit per-thread default stream.
    static thread_local const std::shared_ptr<const void> caller = std::make_shared<int>(0);
    return caller;
}
CUstream canonical_stream(CUstream stream, bool per_thread_default_stream) noexcept {
    return stream ? stream : (per_thread_default_stream ? CU_STREAM_PER_THREAD : CU_STREAM_LEGACY);
}
class VerificationScope final {
   public:
    explicit VerificationScope(bool& is_verifying) : is_verifying_(is_verifying) {
        is_verifying_ = true;
    }
    ~VerificationScope() {
        is_verifying_ = false;
    }
    VerificationScope(const VerificationScope&) = delete;
    VerificationScope& operator=(const VerificationScope&) = delete;

   private:
    bool& is_verifying_;
};
}  // namespace

struct CudaCaptureOriginRegistry::Implementation {
    struct Origin {
        CUcontext context{};
        CUstream actual_stream{};
        CUstream canonical_stream{};
        bool per_thread_default_stream{};
        std::weak_ptr<const void> caller;
        std::uint64_t generation{};
    };
    const std::shared_ptr<const void> identity{std::make_shared<int>(0)};
    const std::size_t max_origins;
    std::vector<Origin> origins;
    std::uint64_t next_generation{1};
    bool is_verifying{};
    bool has_failed{};
    CUresult failure{CUDA_ERROR_UNKNOWN};

    explicit Implementation(std::size_t limit) : max_origins(limit) {
        if (!limit || limit > 256) {
            throw std::invalid_argument("invalid native capture origin capacity");
        }
        origins.reserve(limit);
    }
    CUresult fail(CUresult error) noexcept {
        has_failed = true;
        failure = error == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : error;
        return failure;
    }
    CUresult query(const Origin& origin, const Query& original_query) {
        if (origin.canonical_stream == CU_STREAM_PER_THREAD) {
            const auto owner = origin.caller.lock();
            if (!owner) {
                return fail(CUDA_ERROR_INVALID_HANDLE);
            }
            if (owner != current_caller()) {
                return CUDA_ERROR_NOT_READY;
            }
        }
        // The enum's representable range includes 3 (highest defined value is
        // 2). An undefined but representable sentinel detects unwritten output
        // without introducing an out-of-range enum conversion under UBSan.
        auto status = static_cast<CUstreamCaptureStatus>(3);
        const auto result = original_query(origin.context, origin.actual_stream,
                                           origin.per_thread_default_stream, &status);
        if (result != CUDA_SUCCESS) {
            return fail(result);
        }
        switch (status) {
            case CU_STREAM_CAPTURE_STATUS_NONE:
                return CUDA_SUCCESS;
            case CU_STREAM_CAPTURE_STATUS_ACTIVE:
            case CU_STREAM_CAPTURE_STATUS_INVALIDATED:
                return CUDA_ERROR_NOT_READY;
            default:
                return fail(CUDA_ERROR_UNKNOWN);
        }
    }
};

CudaCaptureOriginRegistry::CudaCaptureOriginRegistry(std::size_t max_origins)
    : implementation_(std::make_unique<Implementation>(max_origins)) {}
CudaCaptureOriginRegistry::~CudaCaptureOriginRegistry() = default;

CUresult CudaCaptureOriginRegistry::observe_origin(CUcontext context, CUstream actual_stream,
                                                   bool per_thread_default_stream,
                                                   CudaCaptureOriginToken& output) noexcept {
    auto& state = *implementation_;
    if (!context || output || state.is_verifying) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (state.has_failed) {
        return state.failure;
    }
    try {
        const auto canonical = canonical_stream(actual_stream, per_thread_default_stream);
        const auto& caller = current_caller();
        auto found =
            std::find_if(state.origins.begin(), state.origins.end(), [&](const auto& item) {
                return item.context == context && item.canonical_stream == canonical &&
                       (canonical != CU_STREAM_PER_THREAD || item.caller.lock() == caller);
            });
        if (found == state.origins.end()) {
            if (state.origins.size() >= state.max_origins || !state.next_generation) {
                return CUDA_ERROR_OUT_OF_MEMORY;
            }
            state.origins.push_back({context, actual_stream, canonical, per_thread_default_stream,
                                     caller, state.next_generation});
            state.next_generation =
                state.next_generation == std::numeric_limits<std::uint64_t>::max()
                    ? 0
                    : state.next_generation + 1;
            found = std::prev(state.origins.end());
        }
        output.owner_ = state.identity;
        output.generation_ = found->generation;
        return CUDA_SUCCESS;
    } catch (const std::bad_alloc&) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}

CUresult CudaCaptureOriginRegistry::reconcile(const CudaCaptureOriginToken& token,
                                              const Query& original_query) noexcept {
    auto& state = *implementation_;
    if (!token || token.owner_ != state.identity || !original_query || state.is_verifying) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const auto found =
        std::find_if(state.origins.begin(), state.origins.end(),
                     [&](const auto& item) { return item.generation == token.generation_; });
    if (found == state.origins.end()) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (state.has_failed) {
        return state.failure;
    }
    const VerificationScope verification(state.is_verifying);
    try {
        const auto result = state.query(*found, original_query);
        if (result == CUDA_SUCCESS) {
            state.origins.erase(found);
        }
        return result;
    } catch (...) {
        return state.fail(CUDA_ERROR_UNKNOWN);
    }
}

CUresult CudaCaptureOriginRegistry::retire_destroyed(const CudaCaptureOriginToken& token) noexcept {
    auto& state = *implementation_;
    if (!token || token.owner_ != state.identity || state.is_verifying) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (state.has_failed) {
        return state.failure;
    }
    const auto found =
        std::find_if(state.origins.begin(), state.origins.end(),
                     [&](const auto& item) { return item.generation == token.generation_; });
    if (found == state.origins.end() || found->canonical_stream == CU_STREAM_LEGACY ||
        found->canonical_stream == CU_STREAM_PER_THREAD) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    state.origins.erase(found);
    return CUDA_SUCCESS;
}

CUresult CudaCaptureOriginRegistry::verify_none(const Query& original_query) noexcept {
    auto& state = *implementation_;
    if (!original_query || state.is_verifying) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (state.has_failed) {
        return state.failure;
    }
    const VerificationScope verification(state.is_verifying);
    try {
        for (auto item = state.origins.begin(); item != state.origins.end();) {
            const auto result = state.query(*item, original_query);
            if (state.has_failed) {
                return result;
            }
            if (result == CUDA_SUCCESS) {
                item = state.origins.erase(item);
            } else {
                ++item;
            }
        }
        return state.origins.empty() ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
    } catch (...) {
        return state.fail(CUDA_ERROR_UNKNOWN);
    }
}
std::size_t CudaCaptureOriginRegistry::retained_origins() const noexcept {
    return implementation_->origins.size();
}

CUresult CudaCaptureOriginRegistry::lookup_context(CUstream stream, bool is_ptds,
                                                   CUcontext default_context,
                                                   CUcontext& output) noexcept {
    auto& state = *implementation_;
    if (output || state.is_verifying) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (state.has_failed) {
        return state.failure;
    }
    try {
        const auto canonical = canonical_stream(stream, is_ptds);
        const bool is_default = canonical == CU_STREAM_LEGACY || canonical == CU_STREAM_PER_THREAD;
        if (is_default && !default_context) {
            return CUDA_ERROR_INVALID_CONTEXT;
        }
        CUcontext found{};
        bool has_foreign_caller{};
        for (const auto& origin : state.origins) {
            if (origin.canonical_stream != canonical ||
                (is_default && origin.context != default_context)) {
                continue;
            }
            if (canonical == CU_STREAM_PER_THREAD) {
                const auto owner = origin.caller.lock();
                if (!owner) {
                    return CUDA_ERROR_INVALID_HANDLE;
                }
                if (owner != current_caller()) {
                    has_foreign_caller = true;
                    continue;
                }
            }
            if (found && found != origin.context) {
                return CUDA_ERROR_INVALID_HANDLE;
            }
            found = origin.context;
        }
        if (!found) {
            return has_foreign_caller ? CUDA_ERROR_NOT_READY : CUDA_ERROR_INVALID_HANDLE;
        }
        output = found;
        return CUDA_SUCCESS;
    } catch (const std::bad_alloc&) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}
}  // namespace glimmer::interceptor
