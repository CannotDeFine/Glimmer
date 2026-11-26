#include <cudnn.h>
#include <atomic>

namespace {
std::atomic<unsigned int> calls{};
cudnnStatus_t unexpected() {
    ++calls;
    return CUDNN_STATUS_NOT_SUPPORTED;
}
}  // namespace
extern "C" unsigned int glimmer_dnn_reduction_native_calls() {
    return calls.load();
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnCreate(cudnnHandle_t*) {
    return unexpected();
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnDestroy(cudnnHandle_t) {
    return unexpected();
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetStream(cudnnHandle_t, cudaStream_t) {
    return unexpected();
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetStream(cudnnHandle_t, cudaStream_t*) {
    return unexpected();
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t*) {
    return unexpected();
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t) {
    return unexpected();
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetTensorNdDescriptor(cudnnTensorDescriptor_t,
                                                                cudnnDataType_t, int, const int*,
                                                                const int*) {
    return unexpected();
}
// Exact installed legacy SDK signatures; the fixture must never execute them.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
extern "C" cudnnStatus_t CUDNNWINAPI
cudnnCreateReduceTensorDescriptor(cudnnReduceTensorDescriptor_t*) {
    return unexpected();
}
extern "C" cudnnStatus_t CUDNNWINAPI
cudnnDestroyReduceTensorDescriptor(cudnnReduceTensorDescriptor_t) {
    return unexpected();
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetReduceTensorDescriptor(
    cudnnReduceTensorDescriptor_t, cudnnReduceTensorOp_t, cudnnDataType_t, cudnnNanPropagation_t,
    cudnnReduceTensorIndices_t, cudnnIndicesType_t) {
    return unexpected();
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetReduceTensorDescriptor(
    cudnnReduceTensorDescriptor_t, cudnnReduceTensorOp_t*, cudnnDataType_t*, cudnnNanPropagation_t*,
    cudnnReduceTensorIndices_t*, cudnnIndicesType_t*) {
    return unexpected();
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetReductionIndicesSize(cudnnHandle_t,
                                                                  cudnnReduceTensorDescriptor_t,
                                                                  cudnnTensorDescriptor_t,
                                                                  cudnnTensorDescriptor_t,
                                                                  std::size_t*) {
    return unexpected();
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetReductionWorkspaceSize(cudnnHandle_t,
                                                                    cudnnReduceTensorDescriptor_t,
                                                                    cudnnTensorDescriptor_t,
                                                                    cudnnTensorDescriptor_t,
                                                                    std::size_t*) {
    return unexpected();
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnReduceTensor(cudnnHandle_t, cudnnReduceTensorDescriptor_t,
                                                       void*, std::size_t, void*, std::size_t,
                                                       const void*, cudnnTensorDescriptor_t,
                                                       const void*, const void*,
                                                       cudnnTensorDescriptor_t, void*) {
    return unexpected();
}
#pragma GCC diagnostic pop
