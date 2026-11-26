#include <cudnn.h>
#include <atomic>

namespace {
std::atomic<unsigned int> calls{};
cudnnStatus_t unexpected() {
    ++calls;
    return CUDNN_STATUS_NOT_SUPPORTED;
}
}  // namespace
extern "C" unsigned int glimmer_dnn_batch_norm_native_calls() {
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
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
extern "C" cudnnStatus_t CUDNNWINAPI cudnnCreateActivationDescriptor(cudnnActivationDescriptor_t*) {
    return unexpected();
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnDestroyActivationDescriptor(cudnnActivationDescriptor_t) {
    return unexpected();
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetActivationDescriptor(cudnnActivationDescriptor_t,
                                                                  cudnnActivationMode_t,
                                                                  cudnnNanPropagation_t, double) {
    return unexpected();
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
    cudnnHandle_t, cudnnBatchNormMode_t, cudnnBatchNormOps_t, cudnnTensorDescriptor_t,
    cudnnTensorDescriptor_t, cudnnTensorDescriptor_t, cudnnTensorDescriptor_t,
    cudnnActivationDescriptor_t, std::size_t*) {
    return unexpected();
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetBatchNormalizationBackwardExWorkspaceSize(
    cudnnHandle_t, cudnnBatchNormMode_t, cudnnBatchNormOps_t, cudnnTensorDescriptor_t,
    cudnnTensorDescriptor_t, cudnnTensorDescriptor_t, cudnnTensorDescriptor_t,
    cudnnTensorDescriptor_t, cudnnTensorDescriptor_t, cudnnActivationDescriptor_t, std::size_t*) {
    return unexpected();
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
    cudnnHandle_t, cudnnBatchNormMode_t, cudnnBatchNormOps_t, cudnnActivationDescriptor_t,
    cudnnTensorDescriptor_t, std::size_t*) {
    return unexpected();
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnBatchNormalizationForwardTrainingEx(
    cudnnHandle_t, cudnnBatchNormMode_t, cudnnBatchNormOps_t, const void*, const void*,
    cudnnTensorDescriptor_t, const void*, cudnnTensorDescriptor_t, const void*,
    cudnnTensorDescriptor_t, void*, cudnnTensorDescriptor_t, const void*, const void*, double,
    void*, void*, double, void*, void*, cudnnActivationDescriptor_t, void*, std::size_t, void*,
    std::size_t) {
    return unexpected();
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnBatchNormalizationBackwardEx(
    cudnnHandle_t, cudnnBatchNormMode_t, cudnnBatchNormOps_t, const void*, const void*, const void*,
    const void*, cudnnTensorDescriptor_t, const void*, cudnnTensorDescriptor_t, const void*,
    cudnnTensorDescriptor_t, const void*, cudnnTensorDescriptor_t, void*, cudnnTensorDescriptor_t,
    void*, cudnnTensorDescriptor_t, const void*, const void*, void*, void*, double, const void*,
    const void*, cudnnActivationDescriptor_t, void*, std::size_t, void*, std::size_t) {
    return unexpected();
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnDeriveBNTensorDescriptor(cudnnTensorDescriptor_t,
                                                                   cudnnTensorDescriptor_t,
                                                                   cudnnBatchNormMode_t) {
    return unexpected();
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnBatchNormalizationForwardInference(
    cudnnHandle_t, cudnnBatchNormMode_t, const void*, const void*, cudnnTensorDescriptor_t,
    const void*, cudnnTensorDescriptor_t, void*, cudnnTensorDescriptor_t, const void*, const void*,
    const void*, const void*, double) {
    return unexpected();
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnBatchNormalizationForwardTraining(
    cudnnHandle_t, cudnnBatchNormMode_t, const void*, const void*, cudnnTensorDescriptor_t,
    const void*, cudnnTensorDescriptor_t, void*, cudnnTensorDescriptor_t, const void*, const void*,
    double, void*, void*, double, void*, void*) {
    return unexpected();
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnBatchNormalizationBackward(
    cudnnHandle_t, cudnnBatchNormMode_t, const void*, const void*, const void*, const void*,
    cudnnTensorDescriptor_t, const void*, cudnnTensorDescriptor_t, const void*,
    cudnnTensorDescriptor_t, void*, cudnnTensorDescriptor_t, const void*, void*, void*, double,
    const void*, const void*) {
    return unexpected();
}
#pragma GCC diagnostic pop
