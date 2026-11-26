#pragma once

#include "cuda_event_projection_registry.h"
#include "cuda_native_graph_effects.h"

#include <memory>
#include <optional>
#include <span>

namespace glimmer::interceptor {

struct CudaNativeGraphRegistration;
struct CudaNativeGraphRegistryState;
struct CudaNativeGraphBoundEvent {
    CudaNativeGraphEventEffect effect;
    CudaEventProjection registration;
};

// CPU ownership only, not permission to use retired CUDA objects or issue work.
class CudaNativeGraphExecutable final {
   public:
    [[nodiscard]] explicit operator bool() const noexcept {
        return registration_ != nullptr;
    }
    [[nodiscard]] CUcontext context() const noexcept;
    [[nodiscard]] const CudaNativeGraphEffects& effects() const noexcept;
    [[nodiscard]] std::span<const CudaNativeGraphBoundEvent> events() const noexcept;

   private:
    friend class CudaNativeGraphRegistry;
    std::shared_ptr<const CudaNativeGraphRegistration> registration_;
};

// One prepared constructor slot. Destruction cancels only uncommitted CPU state.
class CudaNativeGraphReservation final {
   public:
    CudaNativeGraphReservation() = default;
    ~CudaNativeGraphReservation();
    CudaNativeGraphReservation(CudaNativeGraphReservation&&) noexcept = default;
    CudaNativeGraphReservation& operator=(CudaNativeGraphReservation&&) noexcept;
    CudaNativeGraphReservation(const CudaNativeGraphReservation&) = delete;
    CudaNativeGraphReservation& operator=(const CudaNativeGraphReservation&) = delete;
    [[nodiscard]] explicit operator bool() const noexcept {
        return registration_ != nullptr;
    }

   private:
    friend class CudaNativeGraphRegistry;
    void cancel() noexcept;
    std::shared_ptr<CudaNativeGraphRegistryState> owner_;
    std::shared_ptr<CudaNativeGraphRegistration> registration_;
    std::shared_ptr<CudaNativeGraphRegistration> previous_;
};

// Thread-safe bounded CPU metadata. CUDA graph lifetime/mutation and check-to-
// issue exclusion remain caller obligations. No callbacks or CUDA under locks.
class CudaNativeGraphRegistry final {
   public:
    CudaNativeGraphRegistry();
    CudaNativeGraphRegistry(const CudaNativeGraphRegistry&) = delete;
    CudaNativeGraphRegistry& operator=(const CudaNativeGraphRegistry&) = delete;
    // Empty effects explicitly reserve an unprepared native executable.
    [[nodiscard]] CUresult prepare(CUcontext context, const CudaNativeGraphEffects& effects,
                                   const CudaEventProjectionRegistry& events,
                                   CudaNativeGraphReservation& output) noexcept;
    // SDK constructor may bind its context only during original invocation.
    // Preallocate ownership now; publication must supply its actual observation.
    [[nodiscard]] CUresult prepare_unbound(const CudaNativeGraphEffects& effects,
                                           const CudaEventProjectionRegistry& events,
                                           CudaNativeGraphReservation& output) noexcept;
    // No allocation after original construction. Failure with a published
    // handle retains an unprepared row; no handle permits safe CPU cancellation.
    [[nodiscard]] CUresult publish(CudaNativeGraphReservation& reservation, CUgraphExec executable,
                                   CUresult native_status,
                                   std::optional<CUcontext> actual_context = std::nullopt) noexcept;
    [[nodiscard]] CUresult lookup(CUgraphExec executable,
                                  CudaNativeGraphExecutable& output) const noexcept;
    [[nodiscard]] CUresult is_current(const CudaNativeGraphExecutable& owned) const noexcept;
    // Preallocate replacement in the existing slot, then invalidate the prior
    // association before SDK issue. No CUDA or additional constructor slot.
    [[nodiscard]] CUresult prepare_event_update(CUgraphExec executable, CUgraphNode node,
                                                CUevent event, CudaNativeGraphEventKind kind,
                                                const CudaEventProjectionRegistry& events,
                                                CudaNativeGraphReservation& output) noexcept;
    [[nodiscard]] CUresult prepare_node_update(CUgraphExec executable, CUgraphNode node,
                                               std::optional<CUgraphNodeType> expected_type,
                                               CudaNativeGraphReservation& output) noexcept;
    // Prepare complete whole-graph effects or splice one known child scope.
    // Only a whole update can reconstruct a known invalidated metadata plan.
    [[nodiscard]] CUresult prepare_graph_update(CUgraphExec executable,
                                                const CudaNativeGraphEffects& effects,
                                                std::optional<CUgraphNode> child_node,
                                                const CudaEventProjectionRegistry& events,
                                                CudaNativeGraphReservation& output) noexcept;
    // Success replaces only the exact prepared association; failure retains
    // invalidation. All post-SDK publication is allocation-free, once-only.
    [[nodiscard]] CUresult publish_event_update(CudaNativeGraphReservation& reservation,
                                                CUresult native_status) noexcept;
    // Invalidate before original mutation; uncertainty cannot restore the plan.
    void invalidate(CUgraphExec executable) noexcept;
    // Owner calls only after original successful destroy/context teardown.
    void forget(CUgraphExec executable) noexcept;
    void forget_context(CUcontext context) noexcept;

   private:
    [[nodiscard]] CUresult prepare_update(
        CUgraphExec executable, CUgraphNode node, CUevent event,
        std::optional<CudaNativeGraphEventKind> event_kind,
        std::optional<CUgraphNodeType> expected_type, const CudaEventProjectionRegistry* events,
        CudaNativeGraphReservation& output, const CudaNativeGraphEffects* replacement = nullptr,
        std::optional<CUgraphNode> child_node = std::nullopt) noexcept;
    [[nodiscard]] CUresult prepare_impl(CUcontext context, const CudaNativeGraphEffects& effects,
                                        const CudaEventProjectionRegistry& events,
                                        CudaNativeGraphReservation& output,
                                        bool needs_context_publication) noexcept;
    const std::shared_ptr<CudaNativeGraphRegistryState> state;
};

}  // namespace glimmer::interceptor
