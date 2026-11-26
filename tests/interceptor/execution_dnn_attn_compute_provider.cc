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
unsigned int glimmer_dnn_attn_compute_native_calls() {
    return calls.load();
}
cudnnStatus_t CUDNNWINAPI cudnnMultiHeadAttnForward(
    cudnnHandle_t, cudnnAttnDescriptor_t, int, const int*, const int*, const int*, const int*,
    cudnnSeqDataDescriptor_t, const void*, const void*, cudnnSeqDataDescriptor_t, const void*,
    cudnnSeqDataDescriptor_t, const void*, cudnnSeqDataDescriptor_t, void*, std::size_t,
    const void*, std::size_t, void*, std::size_t, void*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnMultiHeadAttnBackwardData(
    cudnnHandle_t, cudnnAttnDescriptor_t, const int*, const int*, const int*, const int*,
    cudnnSeqDataDescriptor_t, const void*, cudnnSeqDataDescriptor_t, void*, const void*,
    cudnnSeqDataDescriptor_t, void*, const void*, cudnnSeqDataDescriptor_t, void*, const void*,
    std::size_t, const void*, std::size_t, void*, std::size_t, void*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnMultiHeadAttnBackwardWeights(
    cudnnHandle_t, cudnnAttnDescriptor_t, cudnnWgradMode_t, cudnnSeqDataDescriptor_t, const void*,
    cudnnSeqDataDescriptor_t, const void*, cudnnSeqDataDescriptor_t, const void*,
    cudnnSeqDataDescriptor_t, const void*, std::size_t, const void*, void*, std::size_t, void*,
    std::size_t, void*) {
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
cudnnStatus_t CUDNNWINAPI cudnnCreateSeqDataDescriptor(cudnnSeqDataDescriptor_t*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnDestroySeqDataDescriptor(cudnnSeqDataDescriptor_t) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnSetSeqDataDescriptor(cudnnSeqDataDescriptor_t, cudnnDataType_t, int,
                                                    const int*, const cudnnSeqDataAxis_t*,
                                                    std::size_t, const int*, void*) {
    return unexpected();
}
}
