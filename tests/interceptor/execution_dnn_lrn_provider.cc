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
unsigned int glimmer_dnn_lrn_native_calls() {
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
cudnnStatus_t CUDNNWINAPI cudnnCreateLRNDescriptor(cudnnLRNDescriptor_t*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnDestroyLRNDescriptor(cudnnLRNDescriptor_t) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnSetLRNDescriptor(cudnnLRNDescriptor_t, unsigned int, double, double,
                                                double) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnLRNCrossChannelForward(cudnnHandle_t, cudnnLRNDescriptor_t,
                                                      cudnnLRNMode_t, const void*,
                                                      cudnnTensorDescriptor_t, const void*,
                                                      const void*, cudnnTensorDescriptor_t, void*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnTransformTensor(cudnnHandle_t, const void*, cudnnTensorDescriptor_t,
                                               const void*, const void*, cudnnTensorDescriptor_t,
                                               void*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnGetLRNDescriptor(cudnnLRNDescriptor_t, unsigned int*, double*,
                                                double*, double*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnLRNCrossChannelBackward(
    cudnnHandle_t, cudnnLRNDescriptor_t, cudnnLRNMode_t, const void*, cudnnTensorDescriptor_t,
    const void*, cudnnTensorDescriptor_t, const void*, cudnnTensorDescriptor_t, const void*,
    const void*, cudnnTensorDescriptor_t, void*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnDivisiveNormalizationForward(cudnnHandle_t, cudnnLRNDescriptor_t,
                                                            cudnnDivNormMode_t, const void*,
                                                            cudnnTensorDescriptor_t, const void*,
                                                            const void*, void*, void*, const void*,
                                                            cudnnTensorDescriptor_t, void*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnDivisiveNormalizationBackward(cudnnHandle_t, cudnnLRNDescriptor_t,
                                                             cudnnDivNormMode_t, const void*,
                                                             cudnnTensorDescriptor_t, const void*,
                                                             const void*, const void*, void*, void*,
                                                             const void*, cudnnTensorDescriptor_t,
                                                             void*, void*) {
    return unexpected();
}
}
