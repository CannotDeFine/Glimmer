#pragma once

#include "cuda_dnn_registry.h"
#include "cuda_execution_session.h"

namespace glimmer::backend::cuda {
struct CudaSessionDnnRnnWeightParams {
    CudaDnnRnnWeightParams parameters;
    CudaAllocationRange weights;
    std::size_t weight_bytes{};
};
enum class CudaDnnComputeKind {
    kActivationForward,
    kAddTensor,
    kPoolingForward,
    kSoftmaxForward,
    kLrnForward,
    kTransformTensor
};
struct CudaSessionDnnCompute {
    CudaDnnComputeKind kind{CudaDnnComputeKind::kActivationForward};
    std::uint64_t handle{};
    CudaDnnObjectView activation;
    std::array<CudaDnnObjectView, 2> tensors;
    std::array<CudaAllocationRange, 2> buffers;
    alignas(double) std::array<std::byte, sizeof(double)> alpha{}, beta{};
    std::size_t scalar_bytes{};
    CudaDnnObjectView pooling;
    std::uint32_t algorithm{}, mode{};
    CudaDnnObjectView lrn;
};
struct CudaDnnComputeAcceptance {
    ExecutionAcceptance acceptance{.status = ExecutionAcceptStatus::kInvalid, .receipt = {}};
    // A failed native metadata query is an original cuDNN error, not an
    // admitted GPU command or a fabricated Driver invalid-value error.
    std::optional<cudnnStatus_t> original_status;
    CUresult rejection_status{CUDA_ERROR_INVALID_VALUE};
};
struct CudaSessionDnnReorder {
    std::uint64_t handle{};
    CudaDnnObjectView filter;
    std::uint32_t reorder{};
    int reorder_bias{};
    // Allocation order: filter input/output, bias input/output.
    std::array<CudaAllocationRange, 4> buffers;
};
[[nodiscard]] bool valid_cuda_dnn_reorder_descriptors(
    const CudaSessionDnnReorder& problem) noexcept;
[[nodiscard]] std::shared_ptr<CudaExecutionLibraryCall> make_cuda_dnn_reorder_call(
    std::shared_ptr<const CudaDnnDispatch> dispatch, cudnnHandle_t handle,
    CudaSessionDnnReorder problem, std::array<CudaDeviceMemoryView, 4> buffers);
inline constexpr std::size_t kCudaDnnMaxConvolutionAlgorithms = 64;
enum class CudaDnnConvolutionQueryKind : std::uint32_t { kWorkspace = 1, kHeuristic, kMaxCount };
enum class CudaDnnBackwardConvolutionQueryKind : std::uint32_t {
    kDataMaxCount = 1,
    kFilterMaxCount,
    kDataWorkspace
};
struct CudaDnnConvolutionPerformance {
    std::uint32_t algorithm{}, status{}, time_bits{};
    std::uint64_t memory{};
    std::uint32_t determinism{}, math{};
    std::array<std::int32_t, 3> reserved{};
};
struct CudaSessionDnnConvolution {
    std::uint64_t handle{};
    // Descriptor order: input tensor, filter, convolution, output tensor.
    // Backward-data workspace queries reuse this order as dy, w, conv, dx;
    // the native call receives w, dy, conv, dx instead.
    std::array<CudaDnnObjectView, 4> descriptors;
    // Allocation order: input, filter, output, optional workspace.
    std::array<CudaAllocationRange, 4> buffers;
    std::uint32_t algorithm{};
    std::size_t workspace_bytes{};
    alignas(double) std::array<std::byte, sizeof(double)> alpha{}, beta{};
    std::size_t scalar_bytes{};
};
struct CudaDnnConvolutionQueryResult {
    CudaDnnResult result;
    std::uint32_t count{};
    std::array<CudaDnnConvolutionPerformance, kCudaDnnMaxConvolutionAlgorithms> performance{};
};
struct CudaDnnConvolutionFindResult {
    CudaDnnConvolutionQueryResult query;
    bool is_ready{};
};
struct CudaDnnFindAcceptance {
    CudaDnnComputeAcceptance native;
    std::shared_ptr<const CudaDnnConvolutionFindResult> result;
};
struct CudaDnnFindContext {
    CUcontext native{};
    decltype(&cuCtxGetCurrent) current{};
    decltype(&cuCtxSynchronize) synchronize{};
};
enum class CudaDnnRnnComputeKind : std::uint32_t { kForward = 1, kBackwardData, kBackwardWeights };
struct CudaSessionDnnRnn {
    CudaDnnRnnComputeKind kind{};
    std::uint64_t handle{};
    CudaDnnRnnView rnn;
    // x-or-dx, y, h, c. Supplied unused state descriptors remain typed pins.
    std::array<CudaDnnObjectView, 4> descriptors;
    // seq, x, y, dy, dx, hx, hy, dhy, dhx, cx, cy, dcy, dcx,
    // weight-or-dweight, workspace, reserve.
    std::array<CudaAllocationRange, 16> buffers;
    std::uint32_t mode{};
    std::size_t weight_bytes{}, workspace_bytes{}, reserve_bytes{};
};
struct CudaDnnRnnCompletion {
    CudaDnnResult result;
    bool is_ready{}, has_context_loss{}, is_cleanup_uncertain{}, has_metadata_quarantine{};
};
struct CudaDnnRnnAcceptance {
    CudaDnnComputeAcceptance native;
    std::shared_ptr<const CudaDnnRnnCompletion> result;
    // Private observations never become requested computation provenance.
    std::optional<cudnnStatus_t> preflight_status;
};
[[nodiscard]] bool valid_cuda_dnn_rnn_descriptors(const CudaSessionDnnRnn& problem) noexcept;
[[nodiscard]] std::shared_ptr<CudaExecutionLibraryCall> make_cuda_dnn_rnn_call(
    std::shared_ptr<const CudaDnnDispatch> dispatch, cudnnHandle_t handle,
    CudaSessionDnnRnn problem, std::array<CudaDeviceMemoryView, 16> buffers,
    CudaDnnFindContext context, std::shared_ptr<CudaDnnRnnCompletion> result,
    std::array<CudaDeviceMemoryView, 4> dropout_states);
enum class CudaDnnAttnComputeKind : std::uint32_t { kForward = 1, kBackwardData, kBackwardWeights };
struct CudaSessionDnnAttn {
    CudaDnnAttnComputeKind kind{};
    std::uint64_t handle{};
    CudaDnnRnnView attn;
    // Q, K, V, O (gradient descriptors use the same semantic positions).
    // Independently absent descriptors retain original early validation.
    std::array<CudaDnnObjectView, 4> descriptors;
    // devQO, devKV, Q, K, V, residual, O, dO, dQ, dK, dV,
    // weights, dweights, workspace, reserve.
    std::array<CudaAllocationRange, 15> buffers;
    std::uint32_t mode{};
    std::int32_t current{};
    std::uint32_t window_presence{};
    std::size_t weight_bytes{}, workspace_bytes{}, reserve_bytes{};
    std::array<std::vector<int>, 2> windows;
};
struct CudaDnnAttnPreparation {
    CudaDnnResult result;
    std::uint32_t first{}, count{};
    bool has_context_loss{};
    std::optional<cudnnStatus_t> preflight_status;
};
struct CudaDnnAttnCompletion {
    CudaDnnResult result;
    bool is_ready{}, has_context_loss{}, is_cleanup_uncertain{}, has_metadata_quarantine{};
};
struct CudaDnnAttnAcceptance {
    CudaDnnComputeAcceptance native;
    std::shared_ptr<const CudaDnnAttnCompletion> result;
    std::optional<cudnnStatus_t> preflight_status;
};
[[nodiscard]] bool valid_cuda_dnn_attn_descriptors(const CudaSessionDnnAttn& problem) noexcept;
[[nodiscard]] std::shared_ptr<CudaExecutionLibraryCall> make_cuda_dnn_attn_call(
    std::shared_ptr<const CudaDnnDispatch> dispatch, cudnnHandle_t handle,
    CudaSessionDnnAttn problem, std::array<CudaDeviceMemoryView, 15> buffers,
    CudaDnnFindContext context, std::shared_ptr<CudaDnnAttnCompletion> result,
    std::array<CudaDeviceMemoryView, 4> dropout_states, std::uint32_t first);
enum class CudaDnnDropoutKind : std::uint32_t { kSet = 1, kRestore };
struct CudaSessionDnnDropout {
    CudaDnnDropoutKind kind{CudaDnnDropoutKind::kSet};
    std::uint64_t handle{};
    std::shared_ptr<CudaDnnDropoutMutation> mutation;
    CudaAllocationRange states;
    std::size_t state_bytes{};
    std::uint32_t probability_bits{};
    std::uint64_t seed{};
};
[[nodiscard]] std::shared_ptr<CudaExecutionLibraryCall> make_cuda_dnn_dropout_call(
    std::shared_ptr<const CudaDnnDispatch> dispatch, cudnnHandle_t handle,
    CudaSessionDnnDropout problem, CudaDeviceMemoryView states, CudaDnnDropoutBinding binding,
    CudaDnnFindContext context);
enum class CudaDnnReductionQueryKind : std::uint32_t { kIndices = 1, kWorkspace };
struct CudaSessionDnnReduction {
    std::uint64_t handle{};
    // Descriptor order: input tensor, reduction, output tensor.
    std::array<CudaDnnObjectView, 3> descriptors;
    // Allocation order: input, output, optional indices, optional workspace.
    std::array<CudaAllocationRange, 4> buffers;
    std::size_t indices_bytes{}, workspace_bytes{};
    alignas(double) std::array<std::byte, sizeof(double)> alpha{}, beta{};
    std::size_t scalar_bytes{};
};
enum class CudaDnnBatchNormKind : std::uint32_t { kForward = 1, kBackward, kReserve };
struct CudaSessionDnnBatchNorm {
    CudaDnnBatchNormKind kind{CudaDnnBatchNormKind::kForward};
    std::uint32_t mode{}, operation{};
    std::uint64_t handle{};
    // x, y, dy, z-or-dz, dx, parameters, optional activation.
    std::array<CudaDnnObjectView, 7> descriptors;
    // x, y, dy, z-or-dz, dx, scale, bias, running-mean-or-dscale,
    // running-variance-or-dbias, saved-mean, saved-inverse-variance,
    // optional workspace, optional reserve.
    std::array<CudaAllocationRange, 13> buffers;
    std::size_t workspace_bytes{}, reserve_bytes{};
    double epsilon{}, average_factor{};
    std::size_t scalar_bytes{};
    // Forward: alpha/beta; backward: alphaData/betaData/alphaParam/betaParam.
    alignas(double) std::array<std::array<std::byte, sizeof(double)>, 4> coefficients{};
};
[[nodiscard]] bool valid_cuda_dnn_batch_norm_descriptors(
    const CudaSessionDnnBatchNorm& problem) noexcept;
[[nodiscard]] std::shared_ptr<CudaExecutionLibraryCall> make_cuda_dnn_batch_norm_call(
    std::shared_ptr<const CudaDnnDispatch> dispatch, cudnnHandle_t handle,
    CudaSessionDnnBatchNorm problem, std::array<CudaDeviceMemoryView, 13> buffers);
[[nodiscard]] bool valid_cuda_dnn_reduction_descriptors(
    const CudaSessionDnnReduction& problem) noexcept;
[[nodiscard]] std::shared_ptr<CudaExecutionLibraryCall> make_cuda_dnn_reduction_call(
    std::shared_ptr<const CudaDnnDispatch> dispatch, cudnnHandle_t handle,
    CudaSessionDnnReduction problem, std::array<CudaDeviceMemoryView, 4> buffers);
[[nodiscard]] bool valid_cuda_dnn_convolution_descriptors(
    const CudaSessionDnnConvolution& problem) noexcept;
[[nodiscard]] std::shared_ptr<CudaExecutionLibraryCall> make_cuda_dnn_convolution_call(
    std::shared_ptr<const CudaDnnDispatch> dispatch, cudnnHandle_t handle,
    CudaSessionDnnConvolution problem, std::array<CudaDeviceMemoryView, 4> buffers);
// The original benchmark runs once on the owning executor. Only this trusted
// call writes the capsule, after native return and context-wide physical proof.
[[nodiscard]] std::shared_ptr<CudaExecutionLibraryCall> make_cuda_dnn_convolution_find_call(
    std::shared_ptr<const CudaDnnDispatch> dispatch, cudnnHandle_t handle,
    CudaSessionDnnConvolution problem, int capacity, CudaDnnFindContext context,
    std::shared_ptr<CudaDnnConvolutionFindResult> result);
[[nodiscard]] std::shared_ptr<CudaExecutionLibraryCall> make_cuda_dnn_call(
    std::shared_ptr<const CudaDnnDispatch> dispatch, cudnnHandle_t handle,
    CudaSessionDnnCompute problem, std::array<CudaDeviceMemoryView, 2> buffers);
}  // namespace glimmer::backend::cuda
