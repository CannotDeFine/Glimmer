#pragma once

#include "cuda_execution_dispatcher.h"

#include <cuda_runtime_api.h>

#include <functional>

namespace glimmer::backend::cuda {

struct CudaSessionGraphDispatch {
    decltype(&cudaStreamBeginCapture) begin{};
    decltype(&cudaStreamEndCapture) end{};
    decltype(&cudaStreamIsCapturing) state{};
    decltype(&cudaGraphInstantiateWithFlags) instantiate{};
    decltype(&cudaGraphDestroy) destroy{};
    decltype(&cudaGraphExecDestroy) destroy_exec{};
    decltype(&cuGraphLaunch) launch{};
};
struct CudaSessionGraphResult {
    CUresult status{CUDA_ERROR_INVALID_VALUE};
    // Lifecycle calls retain Runtime status; captured nodes and replay retain
    // Driver status. A policy/proof failure never invents original provenance.
    std::optional<std::int32_t> original_status{};
    std::uint64_t id{}, output{};
    std::uint32_t capture_status{};
    bool output_written{}, is_quarantined{};
};
struct CudaSessionGraphSnapshot {
    CUresult status{CUDA_ERROR_INVALID_CONTEXT};
    std::uint64_t stream{}, caller{};
    std::size_t graphs{}, executables{};
    bool has_capture{}, is_unknown{}, is_invalidated{};
};
class CudaSessionGraphLaunch : public CudaExecutionLibraryCall {
   public:
    [[nodiscard]] virtual std::optional<CudaSessionGraphResult> reply() const noexcept = 0;
};
struct CudaSessionGraphLaunchAcceptance {
    ExecutionAcceptance acceptance{.status = ExecutionAcceptStatus::kInvalid, .receipt = {}};
    std::shared_ptr<CudaSessionGraphLaunch> operation{};
};
struct CudaSessionGraphLaunchPreparation {
    CUresult status{CUDA_ERROR_INVALID_HANDLE};
    std::shared_ptr<CudaSessionGraphLaunch> operation{};
};

// Original constructing PID/thread only. Resource leases and SDK object owners
// are independent of source graph lifetime. Destructors never invoke CUDA;
// unknown effects retain native ownership rather than retrying cleanup.
class CudaExecutionGraph final {
   public:
    CudaExecutionGraph(CudaSessionGraphDispatch dispatch, CUcontext context,
                       decltype(&cuCtxGetCurrent) current, std::size_t maximum_objects,
                       std::size_t maximum_nodes);
    ~CudaExecutionGraph();
    CudaExecutionGraph(const CudaExecutionGraph&) = delete;
    CudaExecutionGraph& operator=(const CudaExecutionGraph&) = delete;
    [[nodiscard]] CudaSessionGraphSnapshot snapshot() const noexcept;
    [[nodiscard]] CudaSessionGraphResult begin(std::uint64_t stream, CUstream native, int mode,
                                               std::uint64_t caller,
                                               std::shared_ptr<const void> lifetime) noexcept;
    [[nodiscard]] CudaSessionGraphResult end(std::uint64_t stream, std::uint64_t caller,
                                             bool has_output, std::uint64_t seed) noexcept;
    [[nodiscard]] CudaSessionGraphResult state(std::uint64_t stream, std::uint64_t caller) noexcept;
    [[nodiscard]] CudaSessionGraphResult instantiate(std::uint64_t graph, std::uint64_t flags,
                                                     bool has_output, std::uint64_t seed) noexcept;
    [[nodiscard]] CudaSessionGraphResult destroy(std::uint64_t graph) noexcept;
    [[nodiscard]] CudaSessionGraphResult destroy_exec(std::uint64_t executable) noexcept;
    [[nodiscard]] CudaSessionGraphResult invalidate(std::uint64_t caller) noexcept;
    // Service invokes this only after authenticated actual peer exit. This
    // overrides logical caller matching, never native context/end/cleanup proof.
    [[nodiscard]] CudaSessionGraphResult abort() noexcept;
    // Trusted Session factory only. Resource retention is secured before the
    // original node construction call, never a transport-provided callback.
    [[nodiscard]] CudaSessionGraphResult append(std::uint64_t stream, std::uint64_t caller,
                                                std::shared_ptr<const void> lifetime,
                                                const std::function<CUresult()>& issue) noexcept;
    [[nodiscard]] CudaSessionGraphLaunchPreparation prepare_launch(
        std::uint64_t executable) noexcept;
    [[nodiscard]] CUresult shutdown() noexcept;

   private:
    struct Implementation;
    std::unique_ptr<Implementation> implementation_;
};

}  // namespace glimmer::backend::cuda
