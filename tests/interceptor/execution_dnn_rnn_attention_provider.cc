#include <cudnn.h>

#include <atomic>

namespace {
std::atomic<unsigned int> calls{};
cudnnStatus_t native() {
    ++calls;
    return CUDNN_STATUS_NOT_SUPPORTED;
}
}  // namespace
extern "C" unsigned int glimmer_dnn_rnn_attention_native_calls() {
    return calls.load();
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnCreateDropoutDescriptor(cudnnDropoutDescriptor_t*) {
    return native();
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnDestroyDropoutDescriptor(cudnnDropoutDescriptor_t) {
    return native();
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnCreateRNNDescriptor(cudnnRNNDescriptor_t*) {
    return native();
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnDestroyRNNDescriptor(cudnnRNNDescriptor_t) {
    return native();
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetRNNDescriptor_v8(
    cudnnRNNDescriptor_t, cudnnRNNAlgo_t, cudnnRNNMode_t, cudnnRNNBiasMode_t, cudnnDirectionMode_t,
    cudnnRNNInputMode_t, cudnnDataType_t, cudnnDataType_t, cudnnMathType_t, std::int32_t,
    std::int32_t, std::int32_t, std::int32_t, cudnnDropoutDescriptor_t, std::uint32_t) {
    return native();
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetRNNDescriptor_v8(
    cudnnRNNDescriptor_t, cudnnRNNAlgo_t*, cudnnRNNMode_t*, cudnnRNNBiasMode_t*,
    cudnnDirectionMode_t*, cudnnRNNInputMode_t*, cudnnDataType_t*, cudnnDataType_t*,
    cudnnMathType_t*, std::int32_t*, std::int32_t*, std::int32_t*, std::int32_t*,
    cudnnDropoutDescriptor_t*, std::uint32_t*) {
    return native();
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnCreateAttnDescriptor(cudnnAttnDescriptor_t*) {
    return native();
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnDestroyAttnDescriptor(cudnnAttnDescriptor_t) {
    return native();
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetAttnDescriptor(
    cudnnAttnDescriptor_t, unsigned int, int, double, cudnnDataType_t, cudnnDataType_t,
    cudnnMathType_t, cudnnDropoutDescriptor_t, cudnnDropoutDescriptor_t, int, int, int, int, int,
    int, int, int, int, int, int) {
    return native();
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetAttnDescriptor(
    cudnnAttnDescriptor_t, unsigned int*, int*, double*, cudnnDataType_t*, cudnnDataType_t*,
    cudnnMathType_t*, cudnnDropoutDescriptor_t*, cudnnDropoutDescriptor_t*, int*, int*, int*, int*,
    int*, int*, int*, int*, int*, int*, int*) {
    return native();
}
