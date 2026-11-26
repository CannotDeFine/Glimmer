#pragma once

#include "cuda_queue_worker.h"
#include "glimmer/control/async_cooperative_launch_gate.h"

namespace glimmer::interceptor {

struct CudaCoordinationStats {
    std::uint64_t granted_executions{};
    std::uint64_t retreat_requests{};
    std::uint64_t acknowledged_suspensions{};
    std::uint64_t logical_resumes{};
    std::uint64_t acknowledged_completions{};
    std::uint64_t acknowledged_heartbeats{};
    std::uint64_t acknowledged_failures{};
};
struct CudaAdmissionOptions {
    std::size_t pending_limit{1024};
    std::chrono::milliseconds heartbeat_interval{25};
    std::chrono::milliseconds request_retry_interval{1};
    std::chrono::microseconds native_request_retry_interval{25};
    std::chrono::steady_clock::time_point (*clock_now)() noexcept {&std::chrono::steady_clock::now};
};

// All methods are serialized on the CUDA owner (or read after worker join).
// Only trusted kernel tokens may be registered. Backend submission must cancel
// an unissued operation if registration fails. Policy and transport stay in
// control; progress never waits on a future or releases unproven capacity.
class CudaAdmissionCoordinator final : public CudaQueueCoordinator {
   public:
    explicit CudaAdmissionCoordinator(std::shared_ptr<control::AsyncCooperativeLaunchGate> gate,
                                      CudaAdmissionOptions options = {});
    ~CudaAdmissionCoordinator() override;
    [[nodiscard]] bool track(CudaOperationQueue& queue, const CudaOperationToken& kernel,
                             control::CooperativeLaunchRequest request = {}) noexcept;
    // Caller already owns this exact cooperative grant and has issued native
    // CUDA under a paused, empty context boundary. Import must be recorded and
    // dependency-ready. Failure is NOT permission to drop/release that grant.
    [[nodiscard]] bool adopt_native_execution(CudaOperationQueue& queue,
                                              const CudaOperationToken& fence,
                                              control::TaskExecutionIdentity execution,
                                              CUresult native_status = CUDA_SUCCESS) noexcept;
    // Poll on the owner only. Empty means not granted; it never waits on control.
    [[nodiscard]] std::optional<control::TaskExecutionIdentity> native_reservation_grant(
        CudaOperationQueue& queue, const CudaOperationToken& reservation) const noexcept;
    // Called under a reservation pause AFTER native issue and trailing record.
    // Failure retains the placeholder, grant and paused boundary; never replay.
    [[nodiscard]] bool commit_native_reservation(CudaOperationQueue& queue,
                                                 const CudaOperationToken& reservation,
                                                 control::TaskExecutionIdentity execution,
                                                 CudaQueuedNativeFence fence,
                                                 CUresult native_status = CUDA_SUCCESS) noexcept;
    CUresult progress(CudaOperationQueue& queue) noexcept override;
    void stopped(CudaOperationQueue& queue, CUresult result) noexcept override;
    [[nodiscard]] CUresult stopped_progress() noexcept override;
    [[nodiscard]] CUresult native_boundary_status() const noexcept override;
    [[nodiscard]] CudaCoordinationStats stats() const noexcept;
    [[nodiscard]] CUresult control_status() const noexcept;

   private:
    struct Implementation;
    std::unique_ptr<Implementation> implementation_;
};

}  // namespace glimmer::interceptor
