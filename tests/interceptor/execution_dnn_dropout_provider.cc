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
unsigned int glimmer_dnn_dropout_native_calls() {
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
cudnnStatus_t CUDNNWINAPI cudnnCreateDropoutDescriptor(cudnnDropoutDescriptor_t*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnDestroyDropoutDescriptor(cudnnDropoutDescriptor_t) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnDropoutGetStatesSize(cudnnHandle_t, std::size_t*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnSetDropoutDescriptor(cudnnDropoutDescriptor_t, cudnnHandle_t, float,
                                                    void*, std::size_t, unsigned long long) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnRestoreDropoutDescriptor(cudnnDropoutDescriptor_t, cudnnHandle_t,
                                                        float, void*, std::size_t,
                                                        unsigned long long) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnGetDropoutDescriptor(cudnnDropoutDescriptor_t, cudnnHandle_t, float*,
                                                    void**, unsigned long long*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnDropoutGetReserveSpaceSize(cudnnTensorDescriptor_t, std::size_t*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnDropoutForward(cudnnHandle_t, cudnnDropoutDescriptor_t,
                                              cudnnTensorDescriptor_t, const void*,
                                              cudnnTensorDescriptor_t, void*, void*, std::size_t) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnDropoutBackward(cudnnHandle_t, cudnnDropoutDescriptor_t,
                                               cudnnTensorDescriptor_t, const void*,
                                               cudnnTensorDescriptor_t, void*, void*, std::size_t) {
    return unexpected();
}
}
