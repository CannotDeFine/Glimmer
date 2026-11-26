#pragma once

#include <cuda.h>

#include <cstdint>
#include <functional>
#include <memory>

namespace glimmer::interceptor {
class CudaCaptureOriginToken {
   public:
    [[nodiscard]] explicit operator bool() const noexcept {
        return owner_ != nullptr;
    }

   private:
    friend class CudaCaptureOriginRegistry;
    std::shared_ptr<const void> owner_;
    std::uint64_t generation_{};
};

// Metadata-only origin owner. All calls/destruction require an exclusive
// process transition. Query callbacks activate/restore the actual context and
// use original Driver ABI, outside locks; they must not mutate this registry.
// Register before native begin/end issue, not after an uncertain API return.
// This class supplies classification, never physical completion or thaw.
class CudaCaptureOriginRegistry final {
   public:
    using Query = std::function<CUresult(CUcontext, CUstream, bool, CUstreamCaptureStatus*)>;
    explicit CudaCaptureOriginRegistry(std::size_t max_origins = 256);
    ~CudaCaptureOriginRegistry();
    CudaCaptureOriginRegistry(const CudaCaptureOriginRegistry&) = delete;
    CudaCaptureOriginRegistry& operator=(const CudaCaptureOriginRegistry&) = delete;
    [[nodiscard]] CUresult observe_origin(CUcontext context, CUstream actual_stream,
                                          bool per_thread_default_stream,
                                          CudaCaptureOriginToken& output) noexcept;
    // SUCCESS retires only this exact origin, not other process captures.
    [[nodiscard]] CUresult reconcile(const CudaCaptureOriginToken& token,
                                     const Query& original_query) noexcept;
    // Trusted original destroy success, externally serialized. Retire only
    // this explicit stream generation; never query a destroyed handle or prove
    // other origins NONE. No native call, allocation or physical completion.
    [[nodiscard]] CUresult retire_destroyed(const CudaCaptureOriginToken& token) noexcept;
    // Only SUCCESS means all known origins have been proved NONE. ACTIVE and
    // INVALIDATED retain ownership. A foreign PTDS is never queried.
    [[nodiscard]] CUresult verify_none(const Query& original_query) noexcept;
    // Recover only a retained actual origin; implicit streams need the actual
    // current context, and PTDS additionally needs its live initiating caller.
    [[nodiscard]] CUresult lookup_context(CUstream stream, bool is_ptds, CUcontext default_context,
                                          CUcontext& output) noexcept;
    [[nodiscard]] std::size_t retained_origins() const noexcept;

   private:
    struct Implementation;
    std::unique_ptr<Implementation> implementation_;
};
}  // namespace glimmer::interceptor
