#include <cudnn.h>

#include <atomic>

namespace {
std::atomic<unsigned int> calls{};
cudnnStatus_t unexpected() {
    ++calls;
    return CUDNN_STATUS_NOT_SUPPORTED;
}
}  // namespace
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
extern "C" {
unsigned int glimmer_dnn_pooling_native_calls() {
    return calls.load();
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
cudnnStatus_t CUDNNWINAPI cudnnGetStream(cudnnHandle_t, cudaStream_t*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnSetTensorNdDescriptor(cudnnTensorDescriptor_t, cudnnDataType_t, int,
                                                     const int*, const int*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnCreatePoolingDescriptor(cudnnPoolingDescriptor_t*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnDestroyPoolingDescriptor(cudnnPoolingDescriptor_t) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnSetPoolingNdDescriptor(cudnnPoolingDescriptor_t, cudnnPoolingMode_t,
                                                      cudnnNanPropagation_t, int, const int*,
                                                      const int*, const int*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnGetPoolingNdDescriptor(cudnnPoolingDescriptor_t, int,
                                                      cudnnPoolingMode_t*, cudnnNanPropagation_t*,
                                                      int*, int*, int*, int*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnGetPoolingNdForwardOutputDim(cudnnPoolingDescriptor_t,
                                                            cudnnTensorDescriptor_t, int, int*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnPoolingForward(cudnnHandle_t, cudnnPoolingDescriptor_t, const void*,
                                              cudnnTensorDescriptor_t, const void*, const void*,
                                              cudnnTensorDescriptor_t, void*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnSoftmaxForward(cudnnHandle_t, cudnnSoftmaxAlgorithm_t,
                                              cudnnSoftmaxMode_t, const void*,
                                              cudnnTensorDescriptor_t, const void*, const void*,
                                              cudnnTensorDescriptor_t, void*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnSetPooling2dDescriptor(cudnnPoolingDescriptor_t, cudnnPoolingMode_t,
                                                      cudnnNanPropagation_t, int, int, int, int,
                                                      int, int) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnGetPooling2dDescriptor(cudnnPoolingDescriptor_t, cudnnPoolingMode_t*,
                                                      cudnnNanPropagation_t*, int*, int*, int*,
                                                      int*, int*, int*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnGetPooling2dForwardOutputDim(cudnnPoolingDescriptor_t,
                                                            cudnnTensorDescriptor_t, int*, int*,
                                                            int*, int*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnPoolingBackward(cudnnHandle_t, cudnnPoolingDescriptor_t, const void*,
                                               cudnnTensorDescriptor_t, const void*,
                                               cudnnTensorDescriptor_t, const void*,
                                               cudnnTensorDescriptor_t, const void*, const void*,
                                               cudnnTensorDescriptor_t, void*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnSoftmaxBackward(cudnnHandle_t, cudnnSoftmaxAlgorithm_t,
                                               cudnnSoftmaxMode_t, const void*,
                                               cudnnTensorDescriptor_t, const void*,
                                               cudnnTensorDescriptor_t, const void*, const void*,
                                               cudnnTensorDescriptor_t, void*) {
    return unexpected();
}
}
#pragma GCC diagnostic pop
