#pragma once

#include "cuda_operation_queue.h"
#include "persistent_kernel.h"

#include <functional>

namespace glimmer::interceptor {

enum class CudaNativeReleaseState { kRetained, kReleased, kUncertain };
struct CudaCooperativeReleaseResult {
    CUresult status{};
    CudaNativeReleaseState ownership{CudaNativeReleaseState::kUncertain};
};

struct CudaCooperativePoolDispatch {
    decltype(&cuCtxGetCurrent) context_get_current{};
    // Trusted native hooks may perform quota accounting, never public wrappers.
    std::function<CUresult(CUdeviceptr*, std::size_t)> allocate_device;
    // Native errors may follow successful physical release. Never infer retained
    // ownership just from an error status or retry an uncertain native free.
    std::function<CudaCooperativeReleaseResult(CUdeviceptr)> free_device;
    decltype(&cuMemHostAlloc) allocate_host{};
    std::function<CudaCooperativeReleaseResult(void*)> free_host;
    decltype(&cuMemHostGetDevicePointer) map_host{};
    decltype(&cuMemcpyDtoHAsync) copy_device_to_host{};
    std::shared_ptr<void> accounting_owner;
};

struct CudaCooperativeProgress {
    std::uint64_t observed_blocks{};
    std::uint64_t partial_blocks{};
    std::uint64_t physical_launches{};
};

class CudaCooperativeContinuation : public CudaQueueContinuation {
   public:
    [[nodiscard]] virtual CudaCooperativeProgress progress() const noexcept = 0;
};

struct PreparedCooperativeKernel {
    CudaQueuedCopy initialization;
    CudaQueuedKernel kernel;
    std::shared_ptr<CudaCooperativeContinuation> continuation;
};

// Shared ownership is required. Initialization/release are externally serialized
// context-owner boundaries, never queue callbacks. Prepare is metadata-only and
// externally serialized; lease destruction may occur on the CUDA owner thread.
// Release requires actual queue drain AND destruction of every prepared payload.
class CudaCooperativePool final : public std::enable_shared_from_this<CudaCooperativePool> {
   public:
    CudaCooperativePool(CUcontext context, CudaCooperativePoolDispatch dispatch,
                        std::size_t capacity = 1024, std::size_t audit_block_capacity = 0);
    ~CudaCooperativePool();
    CudaCooperativePool(const CudaCooperativePool&) = delete;
    CudaCooperativePool& operator=(const CudaCooperativePool&) = delete;
    [[nodiscard]] CUresult initialize() noexcept;
    [[nodiscard]] CUresult release() noexcept;
    [[nodiscard]] std::optional<PreparedCooperativeKernel> prepare(
        const CooperativeKernelVariant& variant, std::array<unsigned int, 3> grid,
        std::array<unsigned int, 3> block, void* const* arguments,
        std::shared_ptr<void> kernel_resources) noexcept;
    [[nodiscard]] std::size_t occupied_slots() const noexcept;

   private:
    class Continuation;
    struct Implementation;
    std::unique_ptr<Implementation> implementation_;
};

}  // namespace glimmer::interceptor
