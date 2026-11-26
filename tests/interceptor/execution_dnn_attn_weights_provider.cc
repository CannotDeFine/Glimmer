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
unsigned int glimmer_dnn_attn_weights_native_calls() {
    return calls.load();
}
cudnnStatus_t CUDNNWINAPI cudnnGetMultiHeadAttnWeights(cudnnHandle_t, cudnnAttnDescriptor_t,
                                                       cudnnMultiHeadAttnWeightKind_t, std::size_t,
                                                       const void*, cudnnTensorDescriptor_t,
                                                       void**) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnCreate(cudnnHandle_t*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnDestroy(cudnnHandle_t) {
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
