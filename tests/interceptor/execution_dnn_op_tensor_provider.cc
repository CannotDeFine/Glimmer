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
unsigned int glimmer_dnn_op_tensor_native_calls() {
    return calls.load();
}
cudnnStatus_t CUDNNWINAPI cudnnCreateOpTensorDescriptor(cudnnOpTensorDescriptor_t*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnDestroyOpTensorDescriptor(cudnnOpTensorDescriptor_t) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnSetOpTensorDescriptor(cudnnOpTensorDescriptor_t, cudnnOpTensorOp_t,
                                                     cudnnDataType_t, cudnnNanPropagation_t) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnGetOpTensorDescriptor(cudnnOpTensorDescriptor_t, cudnnOpTensorOp_t*,
                                                     cudnnDataType_t*, cudnnNanPropagation_t*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnCreateTensorTransformDescriptor(cudnnTensorTransformDescriptor_t*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnDestroyTensorTransformDescriptor(cudnnTensorTransformDescriptor_t) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnSetTensorTransformDescriptor(
    cudnnTensorTransformDescriptor_t, std::uint32_t, cudnnTensorFormat_t, const std::int32_t*,
    const std::int32_t*, const std::uint32_t*, cudnnFoldingDirection_t) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnInitTransformDest(cudnnTensorTransformDescriptor_t,
                                                 cudnnTensorDescriptor_t, cudnnTensorDescriptor_t,
                                                 std::size_t*) {
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
cudnnStatus_t CUDNNWINAPI cudnnGetTensorNdDescriptor(cudnnTensorDescriptor_t, int, cudnnDataType_t*,
                                                     int*, int*, int*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnGetTensorTransformDescriptor(cudnnTensorTransformDescriptor_t,
                                                            std::uint32_t, cudnnTensorFormat_t*,
                                                            std::int32_t*, std::int32_t*,
                                                            std::uint32_t*,
                                                            cudnnFoldingDirection_t*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnOpTensor(cudnnHandle_t, cudnnOpTensorDescriptor_t, const void*,
                                        cudnnTensorDescriptor_t, const void*, const void*,
                                        cudnnTensorDescriptor_t, const void*, const void*,
                                        cudnnTensorDescriptor_t, void*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnTransformTensorEx(cudnnHandle_t, cudnnTensorTransformDescriptor_t,
                                                 const void*, cudnnTensorDescriptor_t, const void*,
                                                 const void*, cudnnTensorDescriptor_t, void*) {
    return unexpected();
}
}
