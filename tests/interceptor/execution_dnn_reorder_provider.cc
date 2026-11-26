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
unsigned int glimmer_dnn_reorder_native_calls() {
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
cudnnStatus_t CUDNNWINAPI cudnnCreateFilterDescriptor(cudnnFilterDescriptor_t*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnDestroyFilterDescriptor(cudnnFilterDescriptor_t) {
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
cudnnStatus_t CUDNNWINAPI cudnnReorderFilterAndBias(cudnnHandle_t, cudnnFilterDescriptor_t,
                                                    decltype(CUDNN_DEFAULT_REORDER), const void*,
                                                    void*, int, const void*, void*) {
    return unexpected();
}
}
