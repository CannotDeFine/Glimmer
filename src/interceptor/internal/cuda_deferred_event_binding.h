#pragma once

#include <cuda.h>

#include <atomic>
#include <functional>
#include <mutex>

namespace glimmer::interceptor {

// CPU-only per-registration once owner. The synchronous callback runs on one
// original caller outside metadata locks and must not recursively enter this
// same owner. Completion includes failure; even NOT_INITIALIZED is sticky.
// Native context, physical proof and callback resource ownership belong to the
// router. No callback or CUDA work is retained for destruction.
class CudaDeferredEventBinding final {
   public:
    using Operation = std::function<CUresult()>;
    explicit CudaDeferredEventBinding(bool is_bound = false) noexcept;
    [[nodiscard]] bool has_finished() const noexcept;
    [[nodiscard]] CUresult bind(const Operation& operation) noexcept;

   private:
    std::once_flag once_;
    std::atomic<CUresult> result_;
    std::atomic<bool> has_finished_;
};

}  // namespace glimmer::interceptor
