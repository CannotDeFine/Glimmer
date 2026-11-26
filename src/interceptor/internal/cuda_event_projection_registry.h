#pragma once

#include <cuda.h>

#include <cstdint>
#include <memory>
#include <span>
#include <functional>

namespace glimmer::interceptor {

struct CudaEventPreparationState;
struct CudaEventPreparedRegistration;
class CudaOperationToken;

// CPU-only preparation; cancellation never destroys an original CUDA event.
class CudaEventRegistrationReservation final {
   public:
    CudaEventRegistrationReservation() = default;
    ~CudaEventRegistrationReservation();
    CudaEventRegistrationReservation(CudaEventRegistrationReservation&&) noexcept = default;
    CudaEventRegistrationReservation& operator=(CudaEventRegistrationReservation&&) noexcept;
    CudaEventRegistrationReservation(const CudaEventRegistrationReservation&) = delete;
    CudaEventRegistrationReservation& operator=(const CudaEventRegistrationReservation&) = delete;
    [[nodiscard]] explicit operator bool() const noexcept {
        return registration_ != nullptr;
    }

   private:
    friend class CudaEventProjectionRegistry;
    void cancel() noexcept;
    std::shared_ptr<CudaEventPreparationState> owner_;
    std::shared_ptr<CudaEventPreparedRegistration> registration_;
};

class CudaEventProjection final {
   public:
    [[nodiscard]] explicit operator bool() const noexcept {
        return identity_ != nullptr;
    }
    [[nodiscard]] CUcontext context() const noexcept {
        return context_;
    }
    [[nodiscard]] bool is_logical() const noexcept {
        return is_logical_;
    }
    [[nodiscard]] bool allows_logical_binding() const noexcept {
        return allows_logical_binding_;
    }
    // Meaningful for native projections only, not completion of a queue record.
    [[nodiscard]] bool has_native_completion() const noexcept {
        return has_native_completion_;
    }
    [[nodiscard]] bool is_captured_by(const std::shared_ptr<const void>& owner) const noexcept {
        const auto captured = capture_owner_.lock();
        return captured && captured == owner && !captured.owner_before(owner) &&
               !owner.owner_before(captured);
    }
    // Non-null only after checked original record issue. Not completion proof.
    // The token and its shared box own CPU metadata, never a CUDA resource.
    [[nodiscard]] const std::shared_ptr<const CudaOperationToken>& issued_native_prefix()
        const noexcept {
        return native_prefix_;
    }

   private:
    friend class CudaEventProjectionRegistry;
    std::shared_ptr<const void> identity_;
    std::shared_ptr<const void> registry_identity_;
    CUevent event_{};
    CUcontext context_{};
    std::uint64_t native_epoch_{};
    bool is_logical_{};
    bool has_native_completion_{};
    bool allows_logical_binding_{};
    std::shared_ptr<const CudaOperationToken> native_prefix_;
    std::weak_ptr<const void> capture_owner_;
};

// Thread-safe CPU metadata only. Registration is supplied by a checked actual
// Driver owner. Invalidation/accepted publication require external source call
// exclusion against native issue. This registry never changes queue snapshots
// or supplies physical completion. Destruction performs no CUDA calls.
class CudaEventProjectionRegistry final {
   public:
    CudaEventProjectionRegistry();
    CudaEventProjectionRegistry(const CudaEventProjectionRegistry&) = delete;
    CudaEventProjectionRegistry& operator=(const CudaEventProjectionRegistry&) = delete;
    [[nodiscard]] CUresult prepare_native_event(CUcontext context,
                                                CudaEventRegistrationReservation& output) noexcept;
    // Runtime may bind its actual context lazily in the original SDK call.
    [[nodiscard]] CUresult prepare_runtime_event(CudaEventRegistrationReservation& output) noexcept;
    // Trusted normal SDK creation only: successful publication proves an empty
    // unrecorded event, not completion of any submitted stream/context work.
    [[nodiscard]] CUresult prepare_unrecorded_runtime_event(
        CudaEventRegistrationReservation& output) noexcept;
    [[nodiscard]] CUresult status() const noexcept;
    // CPU-only exclusive permits for up to two exact registrations. Identity
    // order prevents reverse-pair deadlock. No lock spans the callback; it must
    // not wait for another caller that needs these same permissions.
    [[nodiscard]] CUresult admit_metadata(std::span<const CudaEventProjection> owned,
                                          const std::function<CUresult()>& operation) noexcept;
    // Record proves context, not non-IPC flags. Never a logical binding.
    [[nodiscard]] CUresult prepare_record_event(CUcontext context,
                                                CudaEventRegistrationReservation& output) noexcept;
    [[nodiscard]] CUresult prepare_runtime_record_event(
        CudaEventRegistrationReservation& output) noexcept;
    // Only for an unbound Runtime ticket after checked SDK success.
    [[nodiscard]] CUresult publish_runtime_event(CudaEventRegistrationReservation& reservation,
                                                 CUevent event, CUcontext context) noexcept;
    // No allocation or native rollback after the original creation call.
    [[nodiscard]] CUresult publish_native_event(CudaEventRegistrationReservation& reservation,
                                                CUevent event, CUresult native_status) noexcept;
    [[nodiscard]] CUresult register_event(CUcontext context, CUevent event,
                                          bool is_logical) noexcept;
    [[nodiscard]] CUresult lookup(CUevent event, CudaEventProjection& output) const noexcept;
    // Refresh epoch/state only if this exact registration still owns the handle.
    [[nodiscard]] CUresult lookup_owned(const CudaEventProjection& owned,
                                        CudaEventProjection& output) const noexcept;
    // Exact successful record under external source issue exclusion. Weak CPU
    // session identity is capture classification, never completion or a fence.
    [[nodiscard]] CUresult publish_capture_record(
        const CudaEventProjection& owned,
        const std::shared_ptr<const void>& capture_owner) noexcept;
    // A supplied exact prefix must already be a dependency of the accepted
    // queue record. Only the trusted actor guard may supply this alternative
    // to physical completion; this registry cannot validate queue acceptance.
    [[nodiscard]] CUresult publish_logical(
        const CudaEventProjection& accepted,
        const CudaOperationToken* accepted_native_prefix = nullptr) noexcept;
    // Epoch mismatch is successful supersession; absent/foreign owners reject.
    // Caller must physically exclude native issue across check and GPU write.
    [[nodiscard]] CUresult matches_native_epoch(const CudaEventProjection& accepted,
                                                bool& output) const noexcept;
    [[nodiscard]] CUresult invalidate_native_event(const CudaEventProjection& owned) noexcept;
    // Preallocated distinct snapshots, atomically refreshed on success only.
    // Source exclusion and acknowledged actor pause must precede original issue.
    // An optional issued prefix is published only AFTER original success under
    // acknowledged actor pause, never before original record issue.
    [[nodiscard]] CUresult invalidate_native_events(
        std::span<CudaEventProjection> owned,
        const std::shared_ptr<const CudaOperationToken>& issued_prefix = {}) noexcept;
    // Exact imported execution fence proof, not a context-wide completion.
    // Retired/reused/superseded epochs are ignored; foreign registries reject.
    [[nodiscard]] CUresult complete_native_events(
        std::span<const CudaEventProjection> proven) noexcept;
    // Null selects every context. Invalidates without allocating or queue work.
    [[nodiscard]] CUresult invalidate_native(CUcontext context = nullptr) noexcept;
    // Owner must exclude native issue and prove actual context completion first.
    // This publishes proof, not logical ownership or a queue record completion.
    [[nodiscard]] CUresult complete_native(CUcontext context = nullptr) noexcept;
    void forget(CUevent event) noexcept;
    // Retire only this exact registration, without changing historic snapshots.
    [[nodiscard]] CUresult forget_owned(const CudaEventProjection& owned) noexcept;
    void forget_context(CUcontext context) noexcept;

   private:
    [[nodiscard]] CUresult prepare_event(CUcontext context,
                                         CudaEventRegistrationReservation& output,
                                         bool allows_logical_binding = true,
                                         bool is_unrecorded = false) noexcept;
    [[nodiscard]] CUresult publish_prepared_event(CudaEventRegistrationReservation& reservation,
                                                  CUevent event, CUresult native_status,
                                                  CUcontext runtime_context,
                                                  bool is_runtime) noexcept;
    const std::shared_ptr<CudaEventPreparationState> state;
    const std::shared_ptr<const void> identity{std::make_shared<const int>(0)};
};

}  // namespace glimmer::interceptor
