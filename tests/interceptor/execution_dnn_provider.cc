#include <cudnn.h>

#include <atomic>
#include <cstddef>

namespace {
std::atomic<unsigned int> calls{};
cudnnStatus_t unexpected() {
    ++calls;
    return CUDNN_STATUS_NOT_SUPPORTED;
}
}  // namespace

extern "C" unsigned int glimmer_dnn_native_calls() {
    return calls.load();
}

// These exact legacy SDK ABIs are intentionally exported by the fake provider.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
extern "C" {
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
cudnnStatus_t CUDNNWINAPI cudnnSetTensor4dDescriptor(cudnnTensorDescriptor_t, cudnnTensorFormat_t,
                                                     cudnnDataType_t, int, int, int, int) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnSetTensor4dDescriptorEx(cudnnTensorDescriptor_t, cudnnDataType_t,
                                                       int, int, int, int, int, int, int, int) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnGetTensor4dDescriptor(cudnnTensorDescriptor_t, cudnnDataType_t*,
                                                     int*, int*, int*, int*, int*, int*, int*,
                                                     int*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnSetTensorNdDescriptor(cudnnTensorDescriptor_t, cudnnDataType_t, int,
                                                     const int*, const int*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnSetTensorNdDescriptorEx(cudnnTensorDescriptor_t, cudnnTensorFormat_t,
                                                       cudnnDataType_t, int, const int*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnGetTensorNdDescriptor(cudnnTensorDescriptor_t, int, cudnnDataType_t*,
                                                     int*, int*, int*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnGetTensorSizeInBytes(cudnnTensorDescriptor_t, std::size_t*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnCreateActivationDescriptor(cudnnActivationDescriptor_t*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnDestroyActivationDescriptor(cudnnActivationDescriptor_t) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnSetActivationDescriptor(cudnnActivationDescriptor_t,
                                                       cudnnActivationMode_t, cudnnNanPropagation_t,
                                                       double) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnGetActivationDescriptor(cudnnActivationDescriptor_t,
                                                       cudnnActivationMode_t*,
                                                       cudnnNanPropagation_t*, double*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnActivationForward(cudnnHandle_t, cudnnActivationDescriptor_t,
                                                 const void*, cudnnTensorDescriptor_t, const void*,
                                                 const void*, cudnnTensorDescriptor_t, void*) {
    return unexpected();
}
cudnnStatus_t CUDNNWINAPI cudnnAddTensor(cudnnHandle_t, const void*, cudnnTensorDescriptor_t,
                                         const void*, const void*, cudnnTensorDescriptor_t, void*) {
    return unexpected();
}
}
#pragma GCC diagnostic pop
