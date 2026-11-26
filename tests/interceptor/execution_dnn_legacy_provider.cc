#include <cudnn.h>

#include <atomic>
#include <cstddef>

static_assert(CUDNN_MAJOR == 8);

namespace {
std::atomic<unsigned int> native_calls{};
}
extern "C" unsigned int glimmer_dnn_legacy_native_calls() {
    return native_calls.load();
}

// Every exact ABI is resolvable, but caller-native execution is a test failure.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
// NOLINTBEGIN(bugprone-easily-swappable-parameters): exact installed SDK ABIs.
extern "C" cudnnStatus_t CUDNNWINAPI cudnnOpsInferVersionCheck() {
    ++native_calls;
    return CUDNN_STATUS_INTERNAL_ERROR;
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnOpsTrainVersionCheck() {
    ++native_calls;
    return CUDNN_STATUS_INTERNAL_ERROR;
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnCnnInferVersionCheck() {
    ++native_calls;
    return CUDNN_STATUS_INTERNAL_ERROR;
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnCnnTrainVersionCheck() {
    ++native_calls;
    return CUDNN_STATUS_INTERNAL_ERROR;
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnAdvInferVersionCheck() {
    ++native_calls;
    return CUDNN_STATUS_INTERNAL_ERROR;
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnAdvTrainVersionCheck() {
    ++native_calls;
    return CUDNN_STATUS_INTERNAL_ERROR;
}
extern "C" cudnnStatus_t CUDNNWINAPI
cudnnCreateAlgorithmDescriptor(cudnnAlgorithmDescriptor_t* output) {
    ++native_calls;
    return CUDNN_STATUS_INTERNAL_ERROR;
}
extern "C" cudnnStatus_t CUDNNWINAPI
cudnnDestroyAlgorithmDescriptor(cudnnAlgorithmDescriptor_t descriptor) {
    ++native_calls;
    return CUDNN_STATUS_INTERNAL_ERROR;
}
extern "C" cudnnStatus_t CUDNNWINAPI
cudnnSetAlgorithmDescriptor(cudnnAlgorithmDescriptor_t descriptor, cudnnAlgorithm_t algorithm) {
    ++native_calls;
    return CUDNN_STATUS_INTERNAL_ERROR;
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetRNNMatrixMathType(cudnnRNNDescriptor_t descriptor,
                                                               cudnnMathType_t type) {
    ++native_calls;
    return CUDNN_STATUS_INTERNAL_ERROR;
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetRNNMatrixMathType(cudnnRNNDescriptor_t descriptor,
                                                               cudnnMathType_t* type) {
    ++native_calls;
    return CUDNN_STATUS_INTERNAL_ERROR;
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetRNNBiasMode(cudnnRNNDescriptor_t descriptor,
                                                         cudnnRNNBiasMode_t mode) {
    ++native_calls;
    return CUDNN_STATUS_INTERNAL_ERROR;
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetRNNBiasMode(cudnnRNNDescriptor_t descriptor,
                                                         cudnnRNNBiasMode_t* mode) {
    ++native_calls;
    return CUDNN_STATUS_INTERNAL_ERROR;
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetRNNDescriptor_v6(
    cudnnHandle_t handle, cudnnRNNDescriptor_t descriptor, int hidden_size, int layers,
    cudnnDropoutDescriptor_t dropout, cudnnRNNInputMode_t input_mode,
    cudnnDirectionMode_t direction, cudnnRNNMode_t cell_mode, cudnnRNNAlgo_t algorithm,
    cudnnDataType_t math_precision) {
    ++native_calls;
    return CUDNN_STATUS_INTERNAL_ERROR;
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnRNNSetClip(cudnnHandle_t handle,
                                                     cudnnRNNDescriptor_t descriptor,
                                                     cudnnRNNClipMode_t mode,
                                                     cudnnNanPropagation_t nan, double lower,
                                                     double upper) {
    ++native_calls;
    return CUDNN_STATUS_INTERNAL_ERROR;
}
extern "C" cudnnStatus_t CUDNNWINAPI
cudnnGetRNNWorkspaceSize(cudnnHandle_t handle, cudnnRNNDescriptor_t descriptor, int sequence_length,
                         const cudnnTensorDescriptor_t* sequence, std::size_t* bytes) {
    ++native_calls;
    return CUDNN_STATUS_INTERNAL_ERROR;
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetRNNTrainingReserveSize(
    cudnnHandle_t handle, cudnnRNNDescriptor_t descriptor, int sequence_length,
    const cudnnTensorDescriptor_t* sequence, std::size_t* bytes) {
    ++native_calls;
    return CUDNN_STATUS_INTERNAL_ERROR;
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetRNNParamsSize(cudnnHandle_t handle,
                                                           cudnnRNNDescriptor_t descriptor,
                                                           cudnnTensorDescriptor_t input,
                                                           std::size_t* bytes,
                                                           cudnnDataType_t type) {
    ++native_calls;
    return CUDNN_STATUS_INTERNAL_ERROR;
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetRNNLinLayerMatrixParams(
    cudnnHandle_t handle, cudnnRNNDescriptor_t descriptor, int pseudo_layer,
    cudnnTensorDescriptor_t input, cudnnFilterDescriptor_t filter, const void* weights,
    int linear_layer, cudnnFilterDescriptor_t destination, void** output) {
    ++native_calls;
    return CUDNN_STATUS_INTERNAL_ERROR;
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetRNNLinLayerBiasParams(
    cudnnHandle_t handle, cudnnRNNDescriptor_t descriptor, int pseudo_layer,
    cudnnTensorDescriptor_t input, cudnnFilterDescriptor_t filter, const void* weights,
    int linear_layer, cudnnFilterDescriptor_t destination, void** output) {
    ++native_calls;
    return CUDNN_STATUS_INTERNAL_ERROR;
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetRNNAlgorithmDescriptor(
    cudnnHandle_t handle, cudnnRNNDescriptor_t descriptor, cudnnAlgorithmDescriptor_t algorithm) {
    ++native_calls;
    return CUDNN_STATUS_INTERNAL_ERROR;
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetRNNForwardInferenceAlgorithmMaxCount(
    cudnnHandle_t handle, cudnnRNNDescriptor_t descriptor, int* count) {
    ++native_calls;
    return CUDNN_STATUS_INTERNAL_ERROR;
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnRNNForwardTraining(
    cudnnHandle_t handle, cudnnRNNDescriptor_t descriptor, int sequence_length,
    const cudnnTensorDescriptor_t* x_descriptors, const void* x,
    cudnnTensorDescriptor_t hx_descriptor, const void* hx, cudnnTensorDescriptor_t cx_descriptor,
    const void* cx, cudnnFilterDescriptor_t filter, const void* weights,
    const cudnnTensorDescriptor_t* y_descriptors, void* y, cudnnTensorDescriptor_t hy_descriptor,
    void* hy, cudnnTensorDescriptor_t cy_descriptor, void* cy, void* workspace,
    std::size_t workspace_bytes, void* reserve, std::size_t reserve_bytes) {
    ++native_calls;
    return CUDNN_STATUS_INTERNAL_ERROR;
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnRNNBackwardData(
    cudnnHandle_t handle, cudnnRNNDescriptor_t descriptor, int sequence_length,
    const cudnnTensorDescriptor_t* y_descriptors, const void* y,
    const cudnnTensorDescriptor_t* dy_descriptors, const void* dy,
    cudnnTensorDescriptor_t dhy_descriptor, const void* dhy, cudnnTensorDescriptor_t dcy_descriptor,
    const void* dcy, cudnnFilterDescriptor_t filter, const void* weights,
    cudnnTensorDescriptor_t hx_descriptor, const void* hx, cudnnTensorDescriptor_t cx_descriptor,
    const void* cx, const cudnnTensorDescriptor_t* dx_descriptors, void* dx,
    cudnnTensorDescriptor_t dhx_descriptor, void* dhx, cudnnTensorDescriptor_t dcx_descriptor,
    void* dcx, void* workspace, std::size_t workspace_bytes, void* reserve,
    std::size_t reserve_bytes) {
    ++native_calls;
    return CUDNN_STATUS_INTERNAL_ERROR;
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnRNNBackwardWeights(
    cudnnHandle_t handle, cudnnRNNDescriptor_t descriptor, int sequence_length,
    const cudnnTensorDescriptor_t* x_descriptors, const void* x,
    cudnnTensorDescriptor_t hx_descriptor, const void* hx,
    const cudnnTensorDescriptor_t* y_descriptors, const void* y, const void* workspace,
    std::size_t workspace_bytes, cudnnFilterDescriptor_t filter, void* weights, const void* reserve,
    std::size_t reserve_bytes) {
    ++native_calls;
    return CUDNN_STATUS_INTERNAL_ERROR;
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnCreate(cudnnHandle_t* output) {
    ++native_calls;
    return CUDNN_STATUS_INTERNAL_ERROR;
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnDestroy(cudnnHandle_t handle) {
    ++native_calls;
    return CUDNN_STATUS_INTERNAL_ERROR;
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnCreateRNNDescriptor(cudnnRNNDescriptor_t* output) {
    ++native_calls;
    return CUDNN_STATUS_INTERNAL_ERROR;
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnDestroyRNNDescriptor(cudnnRNNDescriptor_t descriptor) {
    ++native_calls;
    return CUDNN_STATUS_INTERNAL_ERROR;
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t* output) {
    ++native_calls;
    return CUDNN_STATUS_INTERNAL_ERROR;
}
extern "C" cudnnStatus_t CUDNNWINAPI
cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t descriptor) {
    ++native_calls;
    return CUDNN_STATUS_INTERNAL_ERROR;
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnCreateFilterDescriptor(cudnnFilterDescriptor_t* output) {
    ++native_calls;
    return CUDNN_STATUS_INTERNAL_ERROR;
}
extern "C" cudnnStatus_t CUDNNWINAPI
cudnnDestroyFilterDescriptor(cudnnFilterDescriptor_t descriptor) {
    ++native_calls;
    return CUDNN_STATUS_INTERNAL_ERROR;
}
// NOLINTEND(bugprone-easily-swappable-parameters)
#pragma GCC diagnostic pop
