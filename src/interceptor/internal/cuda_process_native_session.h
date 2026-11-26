#pragma once

#include "cuda_cooperative_frontend.h"

#include <vector>

namespace glimmer::interceptor {

class CudaProcessNativeToken {
   public:
    [[nodiscard]] explicit operator bool() const noexcept {
        return owner_ != nullptr;
    }

   private:
    friend class CudaProcessNativeSession;
    std::shared_ptr<const void> owner_;
};

struct CudaProcessNativeContext {
    CUcontext context{};
    // Empty for native-only contexts observed while private creation is frozen.
    std::shared_ptr<CudaCooperativeFrontend> frontend;
    std::shared_ptr<void> record_lifetime;
};

// Trusted physical owner, not capture classification or public API activation.
// begin/finish/destruction exclude observations through the process transition
// gate; ordinary observations serialize internally. Status reads are atomic. Callbacks run
// outside metadata locks and on the original caller. Reservation and record
// destructors must be CPU-only. The activation callback must restore context.
class CudaProcessNativeSession final {
   public:
    using NativeOperation = CudaCooperativeFrontend::NativeOperation;
    using ContextOperation = std::function<CUresult(CUcontext, const NativeOperation&)>;
    CudaProcessNativeSession(std::vector<CudaProcessNativeContext> contexts,
                             ContextOperation in_context, std::shared_ptr<void> native_reservation,
                             std::size_t max_contexts = 256);
    ~CudaProcessNativeSession();
    CudaProcessNativeSession(const CudaProcessNativeSession&) = delete;
    CudaProcessNativeSession& operator=(const CudaProcessNativeSession&) = delete;
    [[nodiscard]] CUresult begin(CudaProcessNativeToken& token) noexcept;
    [[nodiscard]] CUresult observe_native_context(CUcontext context) noexcept;
    // NOT_READY from verify_inactive retains PAUSED ownership for later end.
    // Other failed/unknown proof retains failed ownership and never replays.
    [[nodiscard]] CUresult finish(const CudaProcessNativeToken& token,
                                  const NativeOperation& verify_inactive,
                                  const NativeOperation& complete_native_context) noexcept;
    [[nodiscard]] CUresult status() const noexcept;
    // CPU-only capability check. Process calls exclude begin/finish/destruction;
    // concurrent observations may fail status but never alter token identity.
    [[nodiscard]] bool owns_paused_token(const CudaProcessNativeToken& token) const noexcept;
    // Externally serialized, CPU-only verification of every exact worker token.
    [[nodiscard]] bool is_paused() const noexcept;
    // Source transition externally excludes observation/finish. CPU-only actor
    // retirement in an existing frontend; never activates a native context.
    [[nodiscard]] CUresult retire_event(const CudaProcessNativeToken& token, CUcontext context,
                                        CUevent event, bool should_retire) noexcept;
    // INVALID_HANDLE means no logical binding, including native-only contexts.
    // No context activation, original stream query, synchronization or thaw.
    [[nodiscard]] CUresult retire_stream(const CudaProcessNativeToken& token, CUcontext context,
                                         CUstream stream, bool should_retire) noexcept;

   private:
    struct Implementation;
    std::unique_ptr<Implementation> implementation_;
};
}  // namespace glimmer::interceptor
