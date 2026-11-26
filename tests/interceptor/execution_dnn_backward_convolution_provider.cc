#include <cudnn.h>

#include <atomic>

namespace {
std::atomic<unsigned int> calls{};
cudnnStatus_t unexpected() {
    ++calls;
    return CUDNN_STATUS_NOT_SUPPORTED;
}
}  // namespace
extern "C" {
unsigned int glimmer_dnn_backward_convolution_native_calls() {
    return calls.load();
}
cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardDataAlgorithmMaxCount(cudnnHandle_t, int*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(cudnnHandle_t, int*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle_t, cudnnFilterDescriptor_t, cudnnTensorDescriptor_t, cudnnConvolutionDescriptor_t,
    cudnnTensorDescriptor_t, cudnnConvolutionBwdDataAlgo_t, std::size_t*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnCreate(cudnnHandle_t*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnDestroy(cudnnHandle_t) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnSetStream(cudnnHandle_t, cudaStream_t) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnSetTensor4dDescriptor(cudnnTensorDescriptor_t, cudnnTensorFormat_t,
                                                     cudnnDataType_t, int, int, int, int) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnCreateFilterDescriptor(cudnnFilterDescriptor_t*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnDestroyFilterDescriptor(cudnnFilterDescriptor_t) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnSetFilter4dDescriptor(cudnnFilterDescriptor_t, cudnnDataType_t,
                                                     cudnnTensorFormat_t, int, int, int, int) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnCreateConvolutionDescriptor(cudnnConvolutionDescriptor_t*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnDestroyConvolutionDescriptor(cudnnConvolutionDescriptor_t) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnSetConvolution2dDescriptor(cudnnConvolutionDescriptor_t, int, int,
                                                          int, int, int, int,
                                                          cudnnConvolutionMode_t, cudnnDataType_t) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnConvolutionForward(cudnnHandle_t, const void*,
                                                  cudnnTensorDescriptor_t, const void*,
                                                  cudnnFilterDescriptor_t, const void*,
                                                  cudnnConvolutionDescriptor_t,
                                                  cudnnConvolutionFwdAlgo_t, void*, std::size_t,
                                                  const void*, cudnnTensorDescriptor_t, void*) {
    return unexpected();
}
}
