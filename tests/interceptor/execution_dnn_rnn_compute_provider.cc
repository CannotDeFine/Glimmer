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
unsigned int glimmer_dnn_rnn_compute_native_calls() {
    return calls.load();
}
cudnnStatus_t CUDNNWINAPI cudnnRNNForward(cudnnHandle_t, cudnnRNNDescriptor_t, cudnnForwardMode_t,
                                          const int32_t*, cudnnRNNDataDescriptor_t, const void*,
                                          cudnnRNNDataDescriptor_t, void*, cudnnTensorDescriptor_t,
                                          const void*, void*, cudnnTensorDescriptor_t, const void*,
                                          void*, size_t, const void*, size_t, void*, size_t,
                                          void*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnRNNBackwardData_v8(
    cudnnHandle_t, cudnnRNNDescriptor_t, const int32_t*, cudnnRNNDataDescriptor_t, const void*,
    const void*, cudnnRNNDataDescriptor_t, void*, cudnnTensorDescriptor_t, const void*, const void*,
    void*, cudnnTensorDescriptor_t, const void*, const void*, void*, size_t, const void*, size_t,
    void*, size_t, void*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnRNNBackwardWeights_v8(cudnnHandle_t, cudnnRNNDescriptor_t,
                                                     cudnnWgradMode_t, const int32_t*,
                                                     cudnnRNNDataDescriptor_t, const void*,
                                                     cudnnTensorDescriptor_t, const void*,
                                                     cudnnRNNDataDescriptor_t, const void*, size_t,
                                                     void*, size_t, void*, size_t, void*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnBuildRNNDynamic(cudnnHandle_t, cudnnRNNDescriptor_t, int) {
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
cudnnStatus_t CUDNNWINAPI cudnnCreateRNNDescriptor(cudnnRNNDescriptor_t*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnDestroyRNNDescriptor(cudnnRNNDescriptor_t) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnSetRNNDescriptor_v8(cudnnRNNDescriptor_t, cudnnRNNAlgo_t,
                                                   cudnnRNNMode_t, cudnnRNNBiasMode_t,
                                                   cudnnDirectionMode_t, cudnnRNNInputMode_t,
                                                   cudnnDataType_t, cudnnDataType_t,
                                                   cudnnMathType_t, int32_t, int32_t, int32_t,
                                                   int32_t, cudnnDropoutDescriptor_t, uint32_t) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnCreateRNNDataDescriptor(cudnnRNNDataDescriptor_t*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnDestroyRNNDataDescriptor(cudnnRNNDataDescriptor_t) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnSetRNNDataDescriptor(cudnnRNNDataDescriptor_t, cudnnDataType_t,
                                                    cudnnRNNDataLayout_t, int, int, int, const int*,
                                                    void*) {
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
cudnnStatus_t CUDNNWINAPI cudnnCreateDropoutDescriptor(cudnnDropoutDescriptor_t*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnDestroyDropoutDescriptor(cudnnDropoutDescriptor_t) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnSetDropoutDescriptor(cudnnDropoutDescriptor_t, cudnnHandle_t, float,
                                                    void*, size_t, unsigned long long) {
    return unexpected();
}
}
