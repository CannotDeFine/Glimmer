#pragma once

#include "glimmer/control/async_cooperative_launch_gate.h"

#include <cuda.h>

#include <functional>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace glimmer::interceptor {
class CudaPausedNativePermit {
   public:
    [[nodiscard]] explicit operator bool() const noexcept {
        return owner_ && entry_;
    }

   private:
    friend class CudaPausedNativeExecutor;
    std::shared_ptr<const void> owner_;
    std::shared_ptr<void> entry_;
};
struct CudaPausedNativeDispatch {
    decltype(&cuEventCreate) event_create{};
    decltype(&cuEventRecord) event_record{};
    decltype(&cuEventRecord) event_record_per_thread{};
    decltype(&cuEventQuery) event_query{};
    decltype(&cuEventDestroy) event_destroy{};
    bool (*enter_native_scope)() noexcept {};
    void (*leave_native_scope)(bool) noexcept {};
    // Optional for existing owners; capture-owning source routing requires it.
    // Private event operations restore the caller mode before application issue.
    decltype(&cuThreadExchangeStreamCaptureMode) thread_exchange_capture_mode{};
};
struct CudaPausedNativeResult {
    CUresult status{CUDA_ERROR_UNKNOWN};
    CUresult tracking_status{CUDA_ERROR_UNKNOWN};
    bool was_invoked{};
};
struct CudaPausedNativeStats {
    std::uint64_t grants{};
    std::uint64_t completed{};
    std::uint64_t failed{};
    std::uint64_t heartbeats{};
    std::size_t retained{};
};

// Thread-safe native fence owner, distinct from paused cooperative workers.
// acquire waits for control only; no CUDA or transition lock is held. issue
// runs on that same caller and requires externally serialized NONE proof plus
// capture-changing-call exclusion until its record completes. The original
// operation issues only on its permit's actual context/stream. Monitor callbacks
// use original dispatch and restore context, outside metadata locks. Native
// context destruction must drain this owner first. Destruction is CPU-only and
// externally excludes source calls; unfinished ownership is quarantined.
class CudaPausedNativeExecutor final {
   public:
    using NativeOperation = std::function<CUresult()>;
    using ContextOperation = std::function<CUresult(CUcontext, const NativeOperation&)>;
    CudaPausedNativeExecutor(std::shared_ptr<control::AsyncCooperativeLaunchGate> gate,
                             CudaPausedNativeDispatch dispatch, ContextOperation in_context,
                             std::size_t max_entries = 256);
    ~CudaPausedNativeExecutor();
    CudaPausedNativeExecutor(const CudaPausedNativeExecutor&) = delete;
    CudaPausedNativeExecutor& operator=(const CudaPausedNativeExecutor&) = delete;
    [[nodiscard]] CUresult acquire(CUcontext context, CUstream actual_stream, bool is_ptds,
                                   std::shared_ptr<void> resources,
                                   control::CooperativeLaunchRequest request,
                                   CudaPausedNativePermit& output) noexcept;
    [[nodiscard]] CudaPausedNativeResult issue(const CudaPausedNativePermit& permit,
                                               const NativeOperation& operation) noexcept;
    // No native issue has happened. Report failure once and wait for exact ack;
    // only then may a reclassified captured call forward with no running lease.
    [[nodiscard]] CUresult retire_unissued(const CudaPausedNativePermit& permit) noexcept;
    // Waits for actual recorded fence completion and terminal acknowledgment.
    // Error/timeout is never physical stop proof. No native work is replayed.
    [[nodiscard]] CUresult drain() noexcept;
    // CPU-only source quarantine. Wake pending admission and reject new issue;
    // this notification never releases an entry, resource or generation.
    // First failure wins. SUCCESS/NOT_READY cannot express a terminal failure.
    void fail_closed(CUresult error) noexcept;
    [[nodiscard]] CUresult status() const noexcept;
    [[nodiscard]] CudaPausedNativeStats stats() const noexcept;

   private:
    struct Implementation;
    std::unique_ptr<Implementation> implementation_;
};
}  // namespace glimmer::interceptor
