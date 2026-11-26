#include <cudnn.h>

#include <atomic>

namespace {
std::atomic<unsigned int> calls{};
cudnnStatus_t unexpected() {
    ++calls;
    return CUDNN_STATUS_NOT_SUPPORTED;
}
}  // namespace
extern "C" unsigned int glimmer_dnn_backend_graph_native_calls() {
    return calls.load();
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnBackendCreateDescriptor(cudnnBackendDescriptorType_t,
                                                                  cudnnBackendDescriptor_t*) {
    return unexpected();
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnBackendDestroyDescriptor(cudnnBackendDescriptor_t) {
    return unexpected();
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnBackendInitialize(cudnnBackendDescriptor_t) {
    return unexpected();
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnBackendFinalize(cudnnBackendDescriptor_t) {
    return unexpected();
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnBackendSetAttribute(cudnnBackendDescriptor_t,
                                                              cudnnBackendAttributeName_t,
                                                              cudnnBackendAttributeType_t,
                                                              std::int64_t, const void*) {
    return unexpected();
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnBackendGetAttribute(cudnnBackendDescriptor_t,
                                                              cudnnBackendAttributeName_t,
                                                              cudnnBackendAttributeType_t,
                                                              std::int64_t, std::int64_t*, void*) {
    return unexpected();
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnBackendExecute(cudnnHandle_t, cudnnBackendDescriptor_t,
                                                         cudnnBackendDescriptor_t) {
    return unexpected();
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnBackendPopulateCudaGraph(cudnnHandle_t,
                                                                   cudnnBackendDescriptor_t,
                                                                   cudnnBackendDescriptor_t,
                                                                   cudaGraph_t) {
    return unexpected();
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnBackendUpdateCudaGraph(cudnnHandle_t,
                                                                 cudnnBackendDescriptor_t,
                                                                 cudnnBackendDescriptor_t,
                                                                 cudaGraph_t) {
    return unexpected();
}
