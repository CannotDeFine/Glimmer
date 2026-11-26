#pragma once

#include "operation_order.h"
#include "owned_kernel_arguments.h"

#include <cuda.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <variant>
#include <vector>

namespace glimmer::interceptor {

// Native dispatch only. No public interceptor wrappers may populate this table.
struct CudaQueueDispatch {
    decltype(&cuCtxGetCurrent) context_get_current{};
    decltype(&cuLaunchKernel) launch_kernel{};
    decltype(&cuMemcpyHtoDAsync) copy_host_to_device{};
    decltype(&cuMemcpyDtoHAsync) copy_device_to_host{};
    decltype(&cuMemcpyDtoDAsync) copy_device_to_device{};
    decltype(&cuMemFreeAsync) free_async{};
    decltype(&cuEventCreate) event_create{};
    decltype(&cuEventRecord) event_record{};
    decltype(&cuEventQuery) event_query{};
    decltype(&cuEventDestroy) event_destroy{};
    decltype(&cuStreamSynchronize) stream_synchronize{};
    decltype(&cuStreamWaitEvent) stream_wait_event{};
    decltype(&cuEventSynchronize) event_synchronize{};
};

// Backend-owned progress resources must survive every continuation. Callbacks
// are serialized, nonblocking, noexcept, and never called under a metadata lock.
struct CudaQueueExecutionIdentity {
    std::uint64_t task{}, generation{};
};

class CudaQueueContinuation {
   public:
    virtual ~CudaQueueContinuation() = default;
    // Argument storage belongs to this queue entry. A continuation may update
    // bounded private bytes here, but must not retain its addresses/reference.
    virtual CUresult prepare_launch(std::array<unsigned int, 3>& physical_grid,
                                    OwnedKernelArguments& arguments) noexcept = 0;
    virtual CUresult request_retreat() noexcept = 0;
    // If needed, enqueue an owned pinned progress readback on this SAME stream. The queue
    // records its physical-stop event after this readback. No synchronous copy
    // or wait is allowed in these callbacks.
    virtual CUresult stage_completion(CUstream stream) noexcept = 0;
    // Called only after a physical stop proof, never just a retreat request.
    virtual CUresult inspect_stopped(bool& is_logically_complete) noexcept = 0;
    virtual void execution_granted(CudaQueueExecutionIdentity) noexcept {}
    // Observation only, after a real trailing event returned NOT_READY.
    virtual void observe_pending() noexcept {}
    virtual void observe_priority_retreat() noexcept {}
};

struct CudaQueuedKernel {
    CUfunction function{};
    std::array<unsigned int, 3> grid{};
    std::array<unsigned int, 3> block{};
    unsigned int shared_memory_bytes{};
    OwnedKernelArguments arguments;
    std::shared_ptr<void> resources;
    std::shared_ptr<CudaQueueContinuation> continuation;
};

enum class CudaQueuedCopyKind : std::uint8_t { kHostToDevice, kDeviceToHost, kDeviceToDevice };
struct CudaQueuedCopy {
    CudaQueuedCopyKind kind{};
    CUdeviceptr destination{};
    CUdeviceptr source{};
    void* host_pointer{};
    std::size_t memory_bytes{};
    // Must own all referenced allocations and PINNED host memory. No implicit
    // extension of an application's pointer lifetime is inferred by this type.
    std::shared_ptr<void> resources;
};
// Keep the backend allocation ledger/RAII owner consistent even if recording
// an event fails AFTER a successful native free enqueue. These metadata-only
// callbacks are noexcept and may not call CUDA or wait.
class CudaQueueReleaseObserver {
   public:
    virtual ~CudaQueueReleaseObserver() = default;
    virtual void confirmed_release(CUdeviceptr pointer) noexcept = 0;
    // A context fence is a distinct ownership boundary, not proof that a
    // context-independent pool allocation's quota can be released.
    virtual void context_destroyed(CUdeviceptr pointer) noexcept = 0;
};
struct CudaQueuedFree {
    CUdeviceptr pointer{};
    std::shared_ptr<void> resources;
    std::shared_ptr<CudaQueueReleaseObserver> release_observer;
};
struct CudaQueuedMarker {};
class CudaOperationToken;
// Metadata only, invoked once after actual native prefix completion proof.
// May briefly lock CPU metadata, but cannot call CUDA/control, wait or reenter.
// Errors fail closed without repeating the callback. Destruction has no CUDA.
class CudaQueueNativeFenceObserver {
   public:
    virtual ~CudaQueueNativeFenceObserver() = default;
    // Optional CPU-only preparation on frontend reservation acceptance, before
    // admission. This token is not a grant, native issue or completion proof.
    virtual CUresult native_reservation_accepted(const CudaOperationToken&) noexcept {
        return CUDA_SUCCESS;
    }
    virtual CUresult confirmed_native_completion() noexcept = 0;
};
// The owner records this native event after an ordered native transaction.
// It must remain valid and must NOT be rerecorded until this import completes.
struct CudaQueuedNativeFence {
    CUevent event{};
    std::shared_ptr<void> resources;
    std::shared_ptr<CudaQueueNativeFenceObserver> completion_observer{};
};
// Never issued by pump. The granted caller replaces this placeholder with an
// actual trailing fence under a physical-only worker pause, keeping its token.
struct CudaQueuedNativeReservation {
    std::shared_ptr<void> resources;
};
struct CudaQueuedResult;
class CudaOperationToken {
   public:
    CudaOperationToken() = default;
    [[nodiscard]] explicit operator bool() const noexcept {
        return result_ != nullptr;
    }
    bool operator==(const CudaOperationToken&) const = default;

   private:
    friend class CudaOperationQueue;
    std::shared_ptr<CudaQueuedResult> result_;
};

struct CudaQueueEventState;
class CudaQueueEventToken {
   public:
    CudaQueueEventToken() = default;
    [[nodiscard]] explicit operator bool() const noexcept {
        return state_ != nullptr;
    }
    bool operator==(const CudaQueueEventToken&) const = default;

   private:
    friend class CudaOperationQueue;
    std::shared_ptr<CudaQueueEventState> state_;
};
struct CudaEventRecordPreparation {
    CudaOperationToken native_prefix;
    // Native takeover suppresses the old write without completing its immutable
    // historical token. Ordinary logical rerecords retain predecessor ordering.
    bool should_order_previous{true};
};
// CPU-only ownership check on the queue actor, outside queue metadata locks.
// A guard may briefly lock its CPU registry, but must not wait for GPU/control
// or queue progress, call CUDA, or reenter the queue.
// Native ownership changes must exclude the entire check-to-write interval by
// physically pausing the actor. Suppression retains historical logical waits.
class CudaQueueEventRecordGuard {
   public:
    virtual ~CudaQueueEventRecordGuard() = default;
    virtual CUresult prepare_record(CudaEventRecordPreparation& output) noexcept {
        output = {};
        return CUDA_SUCCESS;
    }
    // Once after successful frontend acceptance, before its command exclusion
    // ends. Same CPU-only/nonblocking contract as should_write; no GPU proof.
    virtual CUresult record_accepted() noexcept {
        return CUDA_SUCCESS;
    }
    virtual CUresult should_write(bool& output) noexcept = 0;
};
// Only queue-bound events are accepted. Submission normally depends on the
// previous record and transactionally publishes a new immutable token. A trusted
// native takeover can suppress that dependency, not the historical token.
struct CudaQueuedEventRecord {
    CudaQueueEventToken event;
    std::shared_ptr<CudaQueueEventRecordGuard> guard{};
    bool should_order_previous{true};
};
using CudaQueuedOperation =
    std::variant<CudaQueuedKernel, CudaQueuedCopy, CudaQueuedFree, CudaQueuedMarker,
                 CudaQueuedEventRecord, CudaQueuedNativeFence, CudaQueuedNativeReservation>;

struct CudaKernelCandidate {
    CudaOperationToken token;
    OrderedOperationState state{};
    bool can_retreat{};
};

// Internal backend adapter, not yet connected to application CUDA wrappers.
// One thread externally serializes ALL calls and owns PTDS/context activation.
// Pump/query perform no explicit device wait; shutdown may drain physical work.
// Resource owners must defer blocking destruction outside pump. Native stream
// resources must remain owned through shutdown, including after handle close.
// Bound streams are noncapturing backend streams: do not enqueue unmodeled
// native operations on them. Native context sync is NOT a logical queue fence.
class CudaOperationQueue {
   public:
    CudaOperationQueue(CUcontext context, CudaQueueDispatch dispatch,
                       OperationOrderLimits limits = {});
    ~CudaOperationQueue();
    CudaOperationQueue(const CudaOperationQueue&) = delete;
    CudaOperationQueue& operator=(const CudaOperationQueue&) = delete;
    CudaOperationQueue(CudaOperationQueue&&) = delete;
    CudaOperationQueue& operator=(CudaOperationQueue&&) = delete;

    // A logical PTDS may use an owned explicit physical proxy. The frontend
    // supplies caller-lifetime identity; the queue never chooses caller PTDS.
    [[nodiscard]] OrderedStreamToken bind_stream(CUstream stream, OrderedStreamKind kind,
                                                 std::shared_ptr<void> resources) noexcept;
    [[nodiscard]] bool close_stream(const OrderedStreamToken& stream) noexcept;
    // Newly created, unrecorded, context-local, non-IPC events owned exclusively
    // by this backend. Creation/timing flags remain native. No captured records.
    [[nodiscard]] CudaQueueEventToken bind_event(CUevent event,
                                                 std::shared_ptr<void> resources) noexcept;
    [[nodiscard]] bool close_event(const CudaQueueEventToken& event) noexcept;
    [[nodiscard]] std::optional<CudaOperationToken> event_snapshot(
        const CudaQueueEventToken& event) const noexcept;
    [[nodiscard]] CUresult query_event(const CudaQueueEventToken& event) const noexcept;
    [[nodiscard]] CudaOperationToken wait_event(const OrderedStreamToken& stream,
                                                const CudaQueueEventToken& event) noexcept;
    [[nodiscard]] CudaOperationToken submit(
        const OrderedStreamToken& stream, CudaQueuedOperation operation,
        std::span<const CudaOperationToken> dependencies = {}) noexcept;
    [[nodiscard]] CUresult pump() noexcept;
    // Policy/admission decisions occur outside this queue. No unselected kernel
    // may issue; dependency-ready copies/records/fences still progress normally.
    // Validate the complete selection before any native call. Resume suspended
    // work explicitly before selecting it. Running tokens are not new admission.
    [[nodiscard]] CUresult pump_admitted(std::span<const CudaOperationToken> admitted) noexcept;
    [[nodiscard]] std::vector<CudaKernelCandidate> kernel_candidates() const;
    // Metadata only: native execution adoption must precede background import.
    [[nodiscard]] bool native_fence_is_ready(const CudaOperationToken& operation) const noexcept;
    [[nodiscard]] bool native_reservation_is_ready(
        const CudaOperationToken& operation) const noexcept;
    [[nodiscard]] bool is_native_reservation(const CudaOperationToken& operation) const noexcept;
    // Metadata only. No pending physical execution or unproved native prefix
    // may cross this pause; queued/suspended logical work stays retained.
    [[nodiscard]] CUresult query_native_reservation_boundary(
        const CudaOperationToken& operation) const noexcept;
    [[nodiscard]] bool replace_native_reservation(const CudaOperationToken& operation,
                                                  CudaQueuedNativeFence fence) noexcept;
    [[nodiscard]] CUresult query(const CudaOperationToken& operation) const noexcept;
    [[nodiscard]] std::optional<OrderedOperationState> state(
        const CudaOperationToken& operation) const noexcept;
    [[nodiscard]] CUresult request_retreat(const CudaOperationToken& operation,
                                           bool is_priority = false) noexcept;
    void execution_granted(const CudaOperationToken& operation,
                           CudaQueueExecutionIdentity identity) noexcept;
    [[nodiscard]] bool resume(const CudaOperationToken& operation) noexcept;
    [[nodiscard]] bool cancel_stopped(const CudaOperationToken& operation) noexcept;
    [[nodiscard]] std::optional<CudaOperationToken> stream_snapshot(
        const OrderedStreamToken& stream) const noexcept;
    [[nodiscard]] std::vector<CudaOperationToken> context_snapshot() const;
    // CPU-only implicit completion domains, including retained closed tails.
    [[nodiscard]] std::vector<CudaOperationToken> implicit_completion_snapshot(
        bool is_legacy) const;
    [[nodiscard]] std::optional<std::vector<CudaOperationToken>> completion_snapshot(
        const OrderedStreamToken& stream) const;
    // Nonblocking prerequisite only: no native sync, submission or polling.
    // Owner must prevent interleaving and model trailing native completion
    // before enabling a frontend boundary. Errors take precedence over pending.
    [[nodiscard]] CUresult query_native_boundary(
        std::span<const CudaOperationToken> snapshot) const noexcept;
    [[nodiscard]] std::size_t pending_count() const noexcept;
    [[nodiscard]] CUresult shutdown() noexcept;
    // TRUSTED recovery proof: call only AFTER successful native destruction of
    // this queue's context. Never use a timeout/heartbeat as that proof.
    void retire_after_context_destroy() noexcept;

   private:
    friend class CudaQueueWorker;
    // Only after the previous worker has joined with an empty, healthy queue.
    [[nodiscard]] CUresult adopt_drained_owner() noexcept;
    struct Implementation;
    [[nodiscard]] CUresult pump_selected(
        std::optional<std::span<const CudaOperationToken>> admitted) noexcept;
    std::unique_ptr<Implementation> implementation_;
};

}  // namespace glimmer::interceptor
