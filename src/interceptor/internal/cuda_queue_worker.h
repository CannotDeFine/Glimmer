#pragma once

#include "cuda_operation_queue.h"
#include "glimmer/control/async_cooperative_launch_gate.h"

#include <chrono>
#include <functional>
#include <memory>

namespace glimmer::interceptor {

class CudaNativeBoundaryToken {
   public:
    [[nodiscard]] explicit operator bool() const noexcept {
        return owner_ != nullptr && generation_ != 0;
    }

   private:
    friend class CudaQueueWorker;
    std::shared_ptr<const void> owner_;
    std::uint64_t generation_{};
};

struct CudaQueueWorkerDispatch {
    CudaQueueDispatch queue;
    decltype(&cuCtxPushCurrent) context_push{};
    decltype(&cuCtxPopCurrent) context_pop{};
    bool (*enter_native_scope)() noexcept {};
    void (*leave_native_scope)(bool) noexcept {};
};

class CudaQueueCoordinator {
   public:
    virtual ~CudaQueueCoordinator() = default;
    // Nonblocking and serialized on the queue's owner thread, outside locks.
    // Admission/transport is delegated here, never implemented by the worker.
    virtual CUresult progress(CudaOperationQueue& queue) noexcept = 0;
    // A failed result is NOT proof of physical stop or permission to release
    // execution capacity. Invoked after the queue's actual shutdown attempt.
    virtual void stopped(CudaOperationQueue& queue, CUresult result) noexcept = 0;
    // Resumable CPU-only drain after stopped(): NOT_READY retains pending
    // originals; success requires actual terminal ACK. No CUDA or unready get.
    [[nodiscard]] virtual CUresult stopped_progress() noexcept {
        return CUDA_SUCCESS;
    }
    // An empty queue is not necessarily an acknowledged terminal control state.
    // Native empty boundaries must not strand another context behind this lease.
    // CPU-only and nonblocking; also queried while a drained pause is retained.
    [[nodiscard]] virtual CUresult native_boundary_status() const noexcept {
        return CUDA_SUCCESS;
    }
};

struct CudaQueueWorkerOptions {
    std::size_t max_queued_commands{64};
    std::chrono::microseconds poll_interval{1000};
    OperationOrderLimits queue_limits{};
    // Optional, independently subscribed CPU-only control/command wakeup.
    // Only the worker consumes it; a hint is never physical-stop proof.
    std::shared_ptr<control::AsyncCooperativeLaunchGate::ProgressSignal> progress_signal{};
    // Explicit shared process gate; null keeps ordinary worker behavior.
    std::shared_ptr<control::AsyncCooperativeLaunchGate> control_gate{};
};

class CudaWorkerCheckpoint final {
   public:
    [[nodiscard]] explicit operator bool() const noexcept {
        return owner_ && generation_ != 0;
    }

   private:
    friend class CudaQueueWorker;
    std::shared_ptr<const void> owner_;
    std::uint64_t generation_{};
};

// Invoke is thread-safe; teardown is externally serialized. No method may race
// destruction. Only trusted metadata commands are allowed: no GPU waits, public
// CUDA calls, or reentry. All captures must outlive command completion.
class CudaQueueWorker final {
   public:
    using Command = std::function<CUresult(CudaOperationQueue&)>;
    CudaQueueWorker(CUcontext context, CudaQueueWorkerDispatch dispatch,
                    std::shared_ptr<CudaQueueCoordinator> coordinator,
                    CudaQueueWorkerOptions options = {});
    ~CudaQueueWorker();
    CudaQueueWorker(const CudaQueueWorker&) = delete;
    CudaQueueWorker& operator=(const CudaQueueWorker&) = delete;
    CudaQueueWorker(CudaQueueWorker&&) = delete;
    CudaQueueWorker& operator=(CudaQueueWorker&&) = delete;

    [[nodiscard]] CUresult status() const noexcept;
    [[nodiscard]] CUresult invoke(Command command) noexcept;
    // Nonblocking GPU prerequisite; NOT_READY leaves normal priority issue on.
    // A successful pause stops CUDA/coordinator progress and rejects ordinary
    // commands. A selected CPU control actor remains eligible while paused. The
    // caller must hold this worker alive, perform native work outside locks and
    // import its real trailing native fence before finishing the transaction.
    [[nodiscard]] CUresult try_pause_native(CudaNativeBoundaryToken& token) noexcept;
    // Trusted caller already owns the reservation's exact control grant. This
    // pause retains suspended work and is NOT an empty logical/capture boundary.
    [[nodiscard]] CUresult try_pause_native_reservation(const CudaOperationToken& reservation,
                                                        CudaNativeBoundaryToken& token) noexcept;
    // Combine exact grant acquisition and physical-pause proof on one owner
    // command. The trusted prerequisite must be nonblocking and must not issue
    // application work. Rejection leaves normal progress enabled.
    [[nodiscard]] CUresult try_pause_native_reservation(const CudaOperationToken& reservation,
                                                        CudaNativeBoundaryToken& token,
                                                        Command acquire_grant) noexcept;
    // Submit once and wait on the caller, never on the CUDA owner. The owner
    // independently advances the exact grant and physical boundary; NOT_READY
    // retains this bounded request instead of sending it back to the producer.
    // Independently ready waiters may pass a pending waiter; only control grants
    // determine eligibility. Duplicate waits for the same reservation reject.
    // Trusted callbacks must be nonblocking, repeatable and metadata-only.
    // prepare_pause runs only after grant and physical proof, before publication.
    // NOT_READY retains the wait without publishing frontend ownership.
    // Its captures/token remain live until return. No original CUDA work here.
    [[nodiscard]] CUresult await_native_reservation(const CudaOperationToken& reservation,
                                                    CudaNativeBoundaryToken& token,
                                                    Command acquire_grant,
                                                    Command prepare_pause = {}) noexcept;
    [[nodiscard]] CUresult finish_native(const CudaNativeBoundaryToken& token,
                                         Command import_native_fence) noexcept;
    [[nodiscard]] bool native_boundary_is_paused(
        const CudaNativeBoundaryToken& token) const noexcept;
    [[nodiscard]] CUresult shutdown() noexcept;
    // Externally serialize checkpoint/resume/shutdown. NOT_READY accepts no
    // checkpoint. Success joins the worker without retiring queue bindings.
    [[nodiscard]] CUresult checkpoint(CudaWorkerCheckpoint& output) noexcept;
    // TRUSTED: independently prove the original context/resources survived and
    // bind that context on this caller. No automatic retry after native failure.
    [[nodiscard]] CUresult resume_checkpoint(const CudaWorkerCheckpoint& checkpoint) noexcept;
    [[nodiscard]] bool owns_checkpoint(const CudaWorkerCheckpoint& checkpoint) const noexcept;
    // TRUSTED: actual original context destruction is independently proven.
    // Exact joined checkpoint only; consumes CPU ownership without CUDA work.
    [[nodiscard]] CUresult retire_checkpoint_after_context_destroy(
        const CudaWorkerCheckpoint& checkpoint) noexcept;

   private:
    friend class CudaCooperativeFrontend;
    struct Implementation;
    // Only frontend-owned CPU metadata commands. Never CUDA, progress, thaw or
    // resource cleanup. Reservation pauses cannot authorize this path.
    [[nodiscard]] CUresult invoke_drained_paused(const CudaNativeBoundaryToken& token,
                                                 Command command) noexcept;
    [[nodiscard]] CUresult enqueue(Command command, bool can_run_paused,
                                   const CudaOperationToken& reservation = {}) noexcept;
    [[nodiscard]] CUresult try_pause_native_boundary(
        Command prerequisite, CudaNativeBoundaryToken& token, bool is_drained,
        bool should_wait = false, const CudaOperationToken& reservation = {}) noexcept;
    std::unique_ptr<Implementation> implementation_;
};

}  // namespace glimmer::interceptor
