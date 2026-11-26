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
unsigned int glimmer_dnn_rnn_weight_params_native_calls() {
    return calls.load();
}
cudnnStatus_t CUDNNWINAPI cudnnGetRNNWeightParams(cudnnHandle_t, cudnnRNNDescriptor_t, std::int32_t,
                                                  std::size_t, const void*, std::int32_t,
                                                  cudnnTensorDescriptor_t, void**,
                                                  cudnnTensorDescriptor_t, void**) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnCreate(cudnnHandle_t*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnDestroy(cudnnHandle_t) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnCreateRNNDescriptor(cudnnRNNDescriptor_t*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnDestroyRNNDescriptor(cudnnRNNDescriptor_t) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnSetRNNDescriptor_v8(
    cudnnRNNDescriptor_t, cudnnRNNAlgo_t, cudnnRNNMode_t, cudnnRNNBiasMode_t, cudnnDirectionMode_t,
    cudnnRNNInputMode_t, cudnnDataType_t, cudnnDataType_t, cudnnMathType_t, std::int32_t,
    std::int32_t, std::int32_t, std::int32_t, cudnnDropoutDescriptor_t, std::uint32_t) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnCreateDropoutDescriptor(cudnnDropoutDescriptor_t*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnDestroyDropoutDescriptor(cudnnDropoutDescriptor_t) {
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
cudnnStatus_t CUDNNWINAPI cudnnGetTensorNdDescriptor(cudnnTensorDescriptor_t, int, cudnnDataType_t*,
                                                     int*, int*, int*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnAddTensor(cudnnHandle_t, const void*, cudnnTensorDescriptor_t,
                                         const void*, const void*, cudnnTensorDescriptor_t, void*) {
    return unexpected();
}
}
