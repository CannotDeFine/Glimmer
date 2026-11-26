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
unsigned int glimmer_dnn_rnn_query_native_calls() {
    return calls.load();
}
cudnnStatus_t CUDNNWINAPI cudnnGetRNNWeightSpaceSize(cudnnHandle_t, cudnnRNNDescriptor_t,
                                                     std::size_t*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnGetRNNTempSpaceSizes(cudnnHandle_t, cudnnRNNDescriptor_t,
                                                    cudnnForwardMode_t, cudnnRNNDataDescriptor_t,
                                                    std::size_t*, std::size_t*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnGetMultiHeadAttnBuffers(cudnnHandle_t, cudnnAttnDescriptor_t,
                                                       std::size_t*, std::size_t*, std::size_t*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnRNNSetClip_v8(cudnnRNNDescriptor_t, cudnnRNNClipMode_t,
                                             cudnnNanPropagation_t, double, double) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnRNNGetClip_v8(cudnnRNNDescriptor_t, cudnnRNNClipMode_t*,
                                             cudnnNanPropagation_t*, double*, double*) {
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
cudnnStatus_t CUDNNWINAPI cudnnSetRNNDescriptor_v8(
    cudnnRNNDescriptor_t, cudnnRNNAlgo_t, cudnnRNNMode_t, cudnnRNNBiasMode_t, cudnnDirectionMode_t,
    cudnnRNNInputMode_t, cudnnDataType_t, cudnnDataType_t, cudnnMathType_t, std::int32_t,
    std::int32_t, std::int32_t, std::int32_t, cudnnDropoutDescriptor_t, std::uint32_t) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnCreateAttnDescriptor(cudnnAttnDescriptor_t*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnDestroyAttnDescriptor(cudnnAttnDescriptor_t) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnSetAttnDescriptor(cudnnAttnDescriptor_t, unsigned int, int, double,
                                                 cudnnDataType_t, cudnnDataType_t, cudnnMathType_t,
                                                 cudnnDropoutDescriptor_t, cudnnDropoutDescriptor_t,
                                                 int, int, int, int, int, int, int, int, int, int,
                                                 int) {
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
cudnnStatus_t CUDNNWINAPI cudnnSetTensor4dDescriptor(cudnnTensorDescriptor_t, cudnnTensorFormat_t,
                                                     cudnnDataType_t, int, int, int, int) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnAddTensor(cudnnHandle_t, const void*, cudnnTensorDescriptor_t,
                                         const void*, const void*, cudnnTensorDescriptor_t, void*) {
    return unexpected();
}
}
