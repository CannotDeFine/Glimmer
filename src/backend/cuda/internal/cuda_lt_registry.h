#pragma once

#include <cuda.h>
#include <cublasLt.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace glimmer::backend::cuda {

// Original SDK entry points are injected. Only the composition executable links
// the vendor component; no CUDA call or native host pointer enters Control.
struct CudaLtDispatch {
    decltype(&cublasLtCreate) create{};
    decltype(&cublasLtDestroy) destroy{};
    decltype(&cublasLtMatrixLayoutCreate) layout_create{};
    decltype(&cublasLtMatrixLayoutDestroy) layout_destroy{};
    decltype(&cublasLtMatrixLayoutSetAttribute) layout_set{};
    decltype(&cublasLtMatrixLayoutGetAttribute) layout_get{};
    decltype(&cublasLtMatmulDescCreate) operation_create{};
    decltype(&cublasLtMatmulDescDestroy) operation_destroy{};
    decltype(&cublasLtMatmulDescSetAttribute) operation_set{};
    decltype(&cublasLtMatmulDescGetAttribute) operation_get{};
    decltype(&cublasLtMatmulPreferenceCreate) preference_create{};
    decltype(&cublasLtMatmulPreferenceDestroy) preference_destroy{};
    decltype(&cublasLtMatmulPreferenceSetAttribute) preference_set{};
    decltype(&cublasLtMatmulPreferenceGetAttribute) preference_get{};
    decltype(&cublasLtGetVersion) version{};
    decltype(&cublasLtGetCudartVersion) runtime_version{};
    decltype(&cublasLtMatmulAlgoGetHeuristic) heuristic{};
    decltype(&cublasLtMatmulAlgoCheck) algorithm_check{};
    decltype(&cublasLtMatmul) matmul{};
    decltype(&cublasLtMatmulAlgoConfigGetAttribute) configuration_get{};
    decltype(&cublasLtMatmulAlgoCapGetAttribute) capability_get{};
};

enum class CudaLtObjectKind : std::uint32_t { kLayout = 1, kOperation, kPreference };
struct CudaLtCreate {
    CudaLtObjectKind kind{CudaLtObjectKind::kLayout};
    cudaDataType type{CUDA_R_32F};
    cublasComputeType_t compute{CUBLAS_COMPUTE_32F};
    cudaDataType scale{CUDA_R_32F};
    std::uint64_t rows{}, columns{};
    std::int64_t leading{};
};
struct CudaLtDeviceReference {
    std::uint64_t session{}, allocation{}, offset{};
    CUdeviceptr address{};
};
enum class CudaLtDisposition { kComplete, kRejected, kDeferred };
struct CudaLtResult {
    CudaLtDisposition disposition{CudaLtDisposition::kRejected};
    CUresult status{CUDA_ERROR_UNKNOWN};
    std::optional<cublasStatus_t> original_status{};
    std::uint64_t id{}, value{};
    std::vector<std::byte> data{};
};
struct CudaLtAttribute {
    std::size_t bytes{};
    bool is_device_pointer{};
};
struct CudaLtAttributeKey {
    std::uint64_t object{};
    std::uint32_t attribute{};
};
[[nodiscard]] std::optional<CudaLtAttribute> cuda_lt_attribute(CudaLtObjectKind kind,
                                                               std::uint32_t attribute) noexcept;

struct CudaLtObjectView {
    CudaLtObjectKind kind{};
    void* native{};
    std::optional<CudaLtDeviceReference> bias;
    // CPU-only lifetime. Its release is not a GPU stop or permission to free
    // buffers; the native executor retains it through the real trailing event.
    std::shared_ptr<const void> lifetime;
};
struct CudaLtRegistrySnapshot {
    std::size_t objects{}, borrowed{};
    bool is_closing{}, is_closed{}, is_quarantined{};
};

// Process/thread-owned CPU metadata, independent of an application CUDA context.
// All SDK calls run on the constructing owner, without transport/registry locks.
// Borrows are immutable and may outlive a close request; they defer both native
// mutation and destruction. Unknown native ownership never retries or recycles.
class CudaLtRegistry final {
   public:
    explicit CudaLtRegistry(CudaLtDispatch dispatch, std::size_t max_objects = 256);
    ~CudaLtRegistry();
    CudaLtRegistry(const CudaLtRegistry&) = delete;
    CudaLtRegistry& operator=(const CudaLtRegistry&) = delete;
    [[nodiscard]] CudaLtResult create(const CudaLtCreate& specification) noexcept;
    [[nodiscard]] CudaLtResult destroy(std::uint64_t id) noexcept;
    [[nodiscard]] CudaLtResult set_attribute(
        CudaLtAttributeKey key, std::span<const std::byte> data,
        std::optional<CudaLtDeviceReference> reference = {}) noexcept;
    [[nodiscard]] CudaLtResult get_attribute(CudaLtAttributeKey key, std::size_t bytes) noexcept;
    [[nodiscard]] CudaLtResult version(bool runtime) noexcept;
    // Original value provenance only; no GPU/resource lifetime or authority.
    [[nodiscard]] CUresult remember_algorithms(
        std::span<const cublasLtMatmulHeuristicResult_t> algorithms,
        std::size_t sdk_version) noexcept;
    [[nodiscard]] CudaLtResult algorithm_capability(const cublasLtMatmulAlgo_t& algorithm,
                                                    std::uint32_t attribute, std::size_t bytes,
                                                    std::uint32_t presence,
                                                    std::span<const std::byte> initial,
                                                    std::size_t initial_written) noexcept;
    [[nodiscard]] std::optional<CudaLtObjectView> borrow(std::uint64_t id,
                                                         CudaLtObjectKind kind) noexcept;
    void close() noexcept;
    [[nodiscard]] CudaLtResult shutdown() noexcept;
    [[nodiscard]] CudaLtRegistrySnapshot snapshot() const noexcept;

   private:
    struct Implementation;
    std::unique_ptr<Implementation> implementation_;
};
}  // namespace glimmer::backend::cuda
