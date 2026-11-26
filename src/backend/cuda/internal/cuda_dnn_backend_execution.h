#pragma once

#include "cuda_dnn_execution.h"

namespace glimmer::backend::cuda {
enum class CudaDnnBackendContextOperation {
    kPrepare,
    kSet,
    kGet,
    kFinalize,
    kInitialize,
    kDestroy
};
struct CudaDnnBackendContextRequest {
    CudaDnnBackendContextOperation operation{};
    CudaDnnBackendAttributeOperation attribute_operation{};
    CudaDnnBackendAttributeHeader header;
    std::uint64_t id{}, generation{};
    CudaDnnBackendAttributeValues values;
};
struct CudaDnnBackendContextResult {
    CudaDnnResult result;
    CudaDnnBackendAttributePreparation preparation;
    CudaDnnBackendAttributeResult attribute;
};
enum class CudaDnnBackendBindingClass : std::uint32_t { kIgnored = 0, kDevice, kByValue };
struct CudaDnnBackendTensorRequirement {
    std::int64_t uid{};
    std::size_t bytes{}, alignment{};
    bool is_by_value{};
};
struct CudaDnnBackendTensorRequirements {
    CUresult status{CUDA_ERROR_INVALID_VALUE};
    std::vector<CudaDnnBackendTensorRequirement> tensors;
};
// Bounded copied graph observations only; still no allocation authority.
[[nodiscard]] CudaDnnBackendTensorRequirements cuda_dnn_backend_tensor_requirements(
    const CudaDnnBackendSnapshot& snapshot) noexcept;
struct CudaDnnBackendExecuteRequirement {
    std::uint32_t index{};
    CudaDnnBackendBindingClass value_class{};
    std::int64_t uid{};
    std::uint64_t slot{};
    std::size_t bytes{}, alignment{};
};
struct CudaDnnBackendExecutePreparation {
    CudaDnnResult result;
    std::uint64_t plan_generation{}, variant_generation{};
    std::vector<CudaDnnBackendExecuteRequirement> requirements;
};
struct CudaDnnBackendExecuteBinding {
    std::uint32_t index{};
    CudaDnnBackendBindingClass value_class{};
    std::uint64_t slot{};
    CudaAllocationRange device;
    std::vector<std::byte> host_bytes;
};
struct CudaSessionDnnBackendExecute {
    std::uint64_t handle{}, plan{}, variant{}, plan_generation{}, variant_generation{};
    std::vector<CudaDnnBackendExecuteBinding> bindings;
};
// Trusted admission products only. The public metadata pack is never executed.
// Shape/range authority and Execute-time caller capture precede this factory.
enum class CudaDnnBackendPointerKind { kNull, kDevice, kHostValue };
struct CudaDnnBackendExecutionPointer {
    CudaDnnBackendPointerKind kind{};
    CudaDeviceMemoryView device;
    std::vector<std::byte> value;
    std::size_t alignment{1};
};
struct CudaDnnBackendExecutionArguments {
    CudaDnnBackendCallView plan;
    std::vector<std::int64_t> uids;
    std::vector<CudaDnnBackendExecutionPointer> pointers;
    CudaDeviceMemoryView workspace;
};
struct CudaDnnBackendExecutionCompletion {
    CudaDnnResult result;
    bool is_ready{}, has_context_loss{}, is_cleanup_uncertain{};
    // A private Set/Finalize failure is diagnostic, not requested Execute status.
    std::optional<cudnnStatus_t> preparation_status;
};
struct CudaDnnBackendExecuteAcceptance {
    CudaDnnComputeAcceptance native;
    std::shared_ptr<const CudaDnnBackendExecutionCompletion> result;
};
[[nodiscard]] std::shared_ptr<CudaExecutionLibraryCall> make_cuda_dnn_backend_call(
    std::shared_ptr<const CudaDnnDispatch> dispatch, cudnnHandle_t handle,
    CudaDnnBackendExecutionArguments arguments, CudaDnnFindContext context,
    std::shared_ptr<CudaDnnBackendExecutionCompletion> result);
}  // namespace glimmer::backend::cuda
