#pragma once

#include "cuda_external_execution_bridge.h"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace glimmer::backend::cuda {

struct CudaNcclPrimaryDomainDispatch {
    decltype(&cuDeviceGet) device_get{};
    decltype(&cuDevicePrimaryCtxRetain) primary_retain{};
    decltype(&cuDevicePrimaryCtxRelease) primary_release{};
    decltype(&cuCtxGetCurrent) context_get_current{};
    decltype(&cuCtxSetCurrent) context_set_current{};
    decltype(&cuStreamCreate) stream_create{};
    decltype(&cuStreamQuery) stream_query{};
    decltype(&cuStreamDestroy) stream_destroy{};
};
struct CudaNcclPrimaryDomainConfig {
    std::uint64_t identity{};
    int device_ordinal{-1};
    std::size_t maximum_streams{64};
};
struct CudaNcclPrimaryContextCapture {
    CUresult status{CUDA_ERROR_INVALID_VALUE};
    CudaExecutionContext context{};
};
struct CudaNcclPrimaryStreamCapture {
    CUresult status{CUDA_ERROR_INVALID_VALUE};
    std::uint64_t stream_id{};
    CudaExternalExecutionDomain domain{};
};

// Private owner-thread/PID resource manager, never an application context effect.
// Construction and destruction call no SDK. Only injected original Driver calls
// may retain/release the primary context; this interface cannot reset it.
// Exported immutable CPU capsules may be copied/dropped on arbitrary threads.
// Trusted composition MUST retain a context capsule through every native owner
// (including incomplete init/async/destruction), and a stream capsule through all
// bridges/commands. A stream query is not native communicator terminal authority.
class CudaNcclPrimaryDomain final {
   public:
    CudaNcclPrimaryDomain(CudaNcclPrimaryDomainDispatch dispatch,
                          CudaNcclPrimaryDomainConfig config);
    ~CudaNcclPrimaryDomain();
    CudaNcclPrimaryDomain(const CudaNcclPrimaryDomain&) = delete;
    CudaNcclPrimaryDomain& operator=(const CudaNcclPrimaryDomain&) = delete;

    [[nodiscard]] CUresult initialize() noexcept;
    [[nodiscard]] CudaNcclPrimaryContextCapture capture_context() noexcept;
    [[nodiscard]] CudaNcclPrimaryStreamCapture create_stream() noexcept;
    // Query-success and no external stream capsule are both required. Context
    // capsules alone do not prevent unrelated stream retirement.
    [[nodiscard]] CUresult release_stream(std::uint64_t stream_id) noexcept;
    [[nodiscard]] CUresult close() noexcept;
    // Close first, release all exported capsules, then prove actual stream stop.
    // If the entry current context is this primary context, returns NOT_READY
    // without release: restoring a potentially final-released context is unsafe.
    // Query NOT_READY is retryable; every other native/proof failure is sticky.
    [[nodiscard]] CUresult shutdown() noexcept;

   private:
    struct Implementation;
    std::unique_ptr<Implementation> implementation_;
};

}  // namespace glimmer::backend::cuda
