#include "cuda_deferred_event_binding.h"

#include <new>

namespace glimmer::interceptor {

CudaDeferredEventBinding::CudaDeferredEventBinding(bool is_bound) noexcept
    : result_(is_bound ? CUDA_SUCCESS : CUDA_ERROR_NOT_INITIALIZED), has_finished_(is_bound) {}

bool CudaDeferredEventBinding::has_finished() const noexcept {
    return has_finished_.load(std::memory_order_acquire);
}

CUresult CudaDeferredEventBinding::bind(const Operation& operation) noexcept {
    if (has_finished()) {
        return result_.load(std::memory_order_relaxed);
    }
    if (!operation) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    try {
        std::call_once(once_, [&] {
            CUresult result = CUDA_ERROR_UNKNOWN;
            try {
                result = operation();
            } catch (const std::bad_alloc&) {
                result = CUDA_ERROR_OUT_OF_MEMORY;
            } catch (...) {
                result = CUDA_ERROR_UNKNOWN;
            }
            result_.store(result, std::memory_order_relaxed);
            has_finished_.store(true, std::memory_order_release);
        });
        return result_.load(std::memory_order_relaxed);
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}

}  // namespace glimmer::interceptor
