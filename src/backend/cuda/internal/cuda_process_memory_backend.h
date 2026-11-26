#pragma once

#include "cuda_process_memory_owner.h"

#include <cuda_runtime_api.h>

#include <functional>

namespace glimmer::backend::cuda {

class CudaExecutionDomainOwner;

struct CudaProcessMemoryDispatch {
    decltype(&cuInit) initialize{};
    decltype(&cuCtxGetCurrent) context_get_current{};
    decltype(&cuDeviceGetDefaultMemPool) default_pool{};
    decltype(&cuDeviceGetMemPool) current_pool{};
    decltype(&cuDeviceSetMemPool) select_pool{};
    decltype(&cuMemAllocAsync) allocate{};
    decltype(&cuMemAllocFromPoolAsync) allocate_from_pool{};
    decltype(&cudaMemPoolTrimTo) trim{};
    decltype(&cuMemFree) release{};
};
enum class CudaProcessMemoryOperation : std::uint32_t {
    kGetDefaultPool = 1,
    kGetCurrentPool,
    kSetCurrentPool,
    kAllocateAsync,
    kAllocateFromPoolAsync,
    kTrimPool
};
struct CudaProcessMemoryRequest {
    CudaProcessMemoryOperation operation{};
    int device{};
    bool has_output{};
    std::uint64_t pool{}, bytes{}, stream{}, output_seed{};
};
struct CudaProcessMemoryReply {
    CUresult status{CUDA_ERROR_INVALID_VALUE};
    std::optional<std::int32_t> original_status{};
    std::uint64_t id{}, output{};
    bool output_written{}, is_quarantined{};
};

// One serialized service-device SDK gate. The callback must inspect actual
// worker host-call ownership, not infer exclusion from GPU completion.
class CudaProcessMemoryDevice final {
   public:
    CudaProcessMemoryDevice(CudaProcessMemoryDispatch dispatch,
                            std::function<CUresult()> native_entry,
                            std::shared_ptr<CudaExecutionDomainOwner> domain = {});
    ~CudaProcessMemoryDevice();
    [[nodiscard]] CUresult native_health() const noexcept;
    CudaProcessMemoryDevice(const CudaProcessMemoryDevice&) = delete;
    CudaProcessMemoryDevice& operator=(const CudaProcessMemoryDevice&) = delete;
    struct Implementation;

   private:
    friend class CudaProcessMemoryBackend;
    std::unique_ptr<Implementation> implementation_;
};

// Actual SDK factories are separate from the CPU ownership ledger. Composition
// authenticates this attachment; no wire value constructs or transfers it.
class CudaProcessMemoryBackend final {
   public:
    CudaProcessMemoryBackend(CudaProcessMemoryConfig config,
                             std::shared_ptr<core::QuotaLedger> quota,
                             std::shared_ptr<CudaProcessMemoryDevice> device);
    ~CudaProcessMemoryBackend();
    CudaProcessMemoryBackend(const CudaProcessMemoryBackend&) = delete;
    CudaProcessMemoryBackend& operator=(const CudaProcessMemoryBackend&) = delete;
    [[nodiscard]] std::shared_ptr<CudaProcessMemoryOwner> owner() const noexcept;
    [[nodiscard]] CUresult native_health() const noexcept;
    [[nodiscard]] CudaProcessMemoryReply pool_control(const CudaProcessMemoryRequest&) noexcept;
    [[nodiscard]] CudaProcessMemoryResult reserve(const CudaProcessMemoryRequest&) noexcept;
    [[nodiscard]] CUresult cancel(std::uint64_t id) noexcept;
    // Only a trusted queued factory calls this under its existing native scope.
    [[nodiscard]] CudaProcessMemoryReply allocate(
        std::uint64_t id, const CudaProcessMemoryRequest&, CUstream stream,
        const std::shared_ptr<CudaNativeOutputCell>& output,
        std::shared_ptr<CudaProcessMemoryCompletion> tail) noexcept;
    [[nodiscard]] CudaProcessMemoryReply release(std::uint64_t id) noexcept;
    [[nodiscard]] CUresult progress() noexcept;
    [[nodiscard]] CUresult close() noexcept;
    [[nodiscard]] CUresult shutdown() noexcept;
    [[nodiscard]] bool has_allocations() const noexcept;

   private:
    struct Implementation;
    std::unique_ptr<Implementation> implementation_;
};

}  // namespace glimmer::backend::cuda
