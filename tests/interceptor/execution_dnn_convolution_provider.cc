#include <cudnn.h>

#include <atomic>

namespace {
std::atomic<unsigned int> calls{};
cudnnStatus_t unexpected() {
    ++calls;
    return CUDNN_STATUS_NOT_SUPPORTED;
}
}  // namespace
extern "C" unsigned int glimmer_dnn_convolution_native_calls() {
    return calls.load();
}
// Exact legacy ABI exports, never an alternate implementation of remote tokens.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
extern "C" {
cudnnStatus_t CUDNNWINAPI cudnnCreate(cudnnHandle_t*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnDestroy(cudnnHandle_t) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnSetStream(cudnnHandle_t, cudaStream_t) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnGetStream(cudnnHandle_t, cudaStream_t*) {
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
cudnnStatus_t CUDNNWINAPI cudnnGetFilter4dDescriptor(cudnnFilterDescriptor_t, cudnnDataType_t*,
                                                     cudnnTensorFormat_t*, int*, int*, int*, int*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnSetFilterNdDescriptor(cudnnFilterDescriptor_t, cudnnDataType_t,
                                                     cudnnTensorFormat_t, int, const int*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnGetFilterNdDescriptor(cudnnFilterDescriptor_t, int, cudnnDataType_t*,
                                                     cudnnTensorFormat_t*, int*, int*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnGetFilterSizeInBytes(cudnnFilterDescriptor_t, std::size_t*) {
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
cudnnStatus_t CUDNNWINAPI cudnnGetConvolution2dDescriptor(cudnnConvolutionDescriptor_t, int*, int*,
                                                          int*, int*, int*, int*,
                                                          cudnnConvolutionMode_t*,
                                                          cudnnDataType_t*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnSetConvolutionNdDescriptor(cudnnConvolutionDescriptor_t, int,
                                                          const int*, const int*, const int*,
                                                          cudnnConvolutionMode_t, cudnnDataType_t) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionNdDescriptor(cudnnConvolutionDescriptor_t, int, int*,
                                                          int*, int*, int*, cudnnConvolutionMode_t*,
                                                          cudnnDataType_t*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnSetConvolutionGroupCount(cudnnConvolutionDescriptor_t, int) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionGroupCount(cudnnConvolutionDescriptor_t, int*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnSetConvolutionMathType(cudnnConvolutionDescriptor_t,
                                                      cudnnMathType_t) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionMathType(cudnnConvolutionDescriptor_t,
                                                      cudnnMathType_t*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnSetConvolutionReorderType(cudnnConvolutionDescriptor_t,
                                                         cudnnReorderType_t) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionReorderType(cudnnConvolutionDescriptor_t,
                                                         cudnnReorderType_t*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnGetConvolution2dForwardOutputDim(cudnnConvolutionDescriptor_t,
                                                                cudnnTensorDescriptor_t,
                                                                cudnnFilterDescriptor_t, int*, int*,
                                                                int*, int*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionNdForwardOutputDim(cudnnConvolutionDescriptor_t,
                                                                cudnnTensorDescriptor_t,
                                                                cudnnFilterDescriptor_t, int,
                                                                int*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionForwardAlgorithmMaxCount(cudnnHandle_t, int*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionForwardAlgorithm_v7(
    cudnnHandle_t, cudnnTensorDescriptor_t, cudnnFilterDescriptor_t, cudnnConvolutionDescriptor_t,
    cudnnTensorDescriptor_t, int, int*, cudnnConvolutionFwdAlgoPerf_t*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnFindConvolutionForwardAlgorithm(
    cudnnHandle_t, cudnnTensorDescriptor_t, cudnnFilterDescriptor_t, cudnnConvolutionDescriptor_t,
    cudnnTensorDescriptor_t, int, int*, cudnnConvolutionFwdAlgoPerf_t*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionForwardWorkspaceSize(
    cudnnHandle_t, cudnnTensorDescriptor_t, cudnnFilterDescriptor_t, cudnnConvolutionDescriptor_t,
    cudnnTensorDescriptor_t, cudnnConvolutionFwdAlgo_t, std::size_t*) {
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
#pragma GCC diagnostic pop
