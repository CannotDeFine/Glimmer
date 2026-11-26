#include <cudnn.h>

#include <atomic>

namespace {
std::atomic<unsigned int> calls{};
cudnnStatus_t native() {
    ++calls;
    return CUDNN_STATUS_NOT_SUPPORTED;
}
}  // namespace
extern "C" unsigned int glimmer_dnn_sequence_native_calls() {
    return calls.load();
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnCreateRNNDataDescriptor(cudnnRNNDataDescriptor_t*) {
    return native();
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnDestroyRNNDataDescriptor(cudnnRNNDataDescriptor_t) {
    return native();
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetRNNDataDescriptor(cudnnRNNDataDescriptor_t,
                                                               cudnnDataType_t,
                                                               cudnnRNNDataLayout_t, int, int, int,
                                                               const int*, void*) {
    return native();
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetRNNDataDescriptor(cudnnRNNDataDescriptor_t,
                                                               cudnnDataType_t*,
                                                               cudnnRNNDataLayout_t*, int*, int*,
                                                               int*, int, int*, void*) {
    return native();
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnCreateSeqDataDescriptor(cudnnSeqDataDescriptor_t*) {
    return native();
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnDestroySeqDataDescriptor(cudnnSeqDataDescriptor_t) {
    return native();
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetSeqDataDescriptor(cudnnSeqDataDescriptor_t,
                                                               cudnnDataType_t, int, const int*,
                                                               const cudnnSeqDataAxis_t*,
                                                               std::size_t, const int*, void*) {
    return native();
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetSeqDataDescriptor(const cudnnSeqDataDescriptor_t,
                                                               cudnnDataType_t*, int*, int, int*,
                                                               cudnnSeqDataAxis_t*, std::size_t*,
                                                               std::size_t, int*, void*) {
    return native();
}
