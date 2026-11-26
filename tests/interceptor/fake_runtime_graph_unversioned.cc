#include <cuda_runtime_api.h>
#include <atomic>
#include <cstddef>

namespace {
std::atomic<unsigned int> g_calls{};
std::atomic<unsigned int> g_update_calls{};
cudaError_t construct(cudaGraphExec_t* output, cudaGraph_t graph) {
    ++g_calls;
    if (!output || !graph) return cudaErrorInvalidValue;
    *output = reinterpret_cast<cudaGraphExec_t>(0xfa010);
    return cudaSuccess;
}
}  // namespace

extern "C" unsigned int glimmer_fake_unversioned_graph_calls() {
    return g_calls.load();
}
extern "C" unsigned int glimmer_fake_unversioned_graph_update_calls() {
    return g_update_calls.load();
}
#if CUDART_VERSION >= 12000
extern "C" cudaError_t CUDARTAPI cudaGraphExecUpdate(cudaGraphExec_t executable, cudaGraph_t graph,
                                                     cudaGraphExecUpdateResultInfo* info) {
    ++g_update_calls;
    if (!executable || !graph || !info) return cudaErrorInvalidValue;
    info->result = cudaGraphExecUpdateSuccess;
    return cudaSuccess;
}
extern "C" cudaError_t CUDARTAPI cudaGraphInstantiate(cudaGraphExec_t* output, cudaGraph_t graph,
                                                      unsigned long long) {
    return construct(output, graph);
}
#else
extern "C" cudaError_t CUDARTAPI cudaGraphExecUpdate(cudaGraphExec_t executable, cudaGraph_t graph,
                                                     cudaGraphNode_t*,
                                                     cudaGraphExecUpdateResult* result) {
    ++g_update_calls;
    if (!executable || !graph || !result) return cudaErrorInvalidValue;
    *result = cudaGraphExecUpdateSuccess;
    return cudaSuccess;
}
extern "C" cudaError_t CUDARTAPI cudaGraphInstantiate(cudaGraphExec_t* output, cudaGraph_t graph,
                                                      cudaGraphNode_t*, char*, std::size_t) {
    return construct(output, graph);
}
#endif
