#include "internal/driver_graph_wrappers.h"

#include <array>
#include <atomic>
#include <string_view>

namespace {
const auto kGraph = reinterpret_cast<CUgraph>(0x6000);
const auto kExecutable = reinterpret_cast<CUgraphExec>(0x5000);
const auto kNode = reinterpret_cast<CUgraphNode>(0x7000);
const auto kEvent = reinterpret_cast<CUevent>(0x3000);
std::array<std::atomic<unsigned int>, 13> g_calls{};
std::atomic<CUresult> g_result{CUDA_SUCCESS};
thread_local CUgraphExec* g_output{};
thread_local CUgraphNode* g_error_node{};
thread_local char* g_log{};
thread_local std::size_t g_log_bytes{};
thread_local CUDA_GRAPH_INSTANTIATE_PARAMS* g_params{};
CUresult create(unsigned int index, CUgraphExec* output, CUgraph graph) {
    ++g_calls[index];
    if (output != g_output || !output || graph != kGraph) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const auto result = g_result.load();
    if (result == CUDA_SUCCESS) {
        *output = kExecutable;
    }
    return result;
}
CUresult mutate(unsigned int index, CUgraphExec executable, bool are_arguments_valid) {
    ++g_calls[index];
    return executable == kExecutable && are_arguments_valid ? g_result.load()
                                                            : CUDA_ERROR_INVALID_VALUE;
}
}  // namespace

extern "C" void glimmer_test_graph_arguments(CUgraphExec* output, CUgraphNode* error_node,
                                             char* log, std::size_t log_bytes,
                                             CUDA_GRAPH_INSTANTIATE_PARAMS* params) {
    g_output = output;
    g_error_node = error_node;
    g_log = log;
    g_log_bytes = log_bytes;
    g_params = params;
}
extern "C" void glimmer_test_graph_result(CUresult result) {
    g_result.store(result);
}
extern "C" unsigned int glimmer_test_graph_calls(unsigned int index) {
    return index < g_calls.size() ? g_calls[index].load() : 0;
}
extern "C" __attribute__((visibility("hidden"))) CUresult CUDAAPI native_cugraphinstantiate(
    CUgraphExec* output, CUgraph graph, CUgraphNode* error_node, char* log, std::size_t log_bytes) {
    return error_node == g_error_node && log == g_log && log_bytes == g_log_bytes
               ? create(0, output, graph)
               : CUDA_ERROR_INVALID_VALUE;
}
extern "C" __attribute__((visibility("hidden"))) CUresult CUDAAPI native_cugraphinstantiate_v2(
    CUgraphExec* output, CUgraph graph, CUgraphNode* error_node, char* log, std::size_t log_bytes) {
    return error_node == g_error_node && log == g_log && log_bytes == g_log_bytes
               ? create(1, output, graph)
               : CUDA_ERROR_INVALID_VALUE;
}
extern "C" __attribute__((visibility("hidden"))) CUresult CUDAAPI
native_cugraphinstantiatewithflags(CUgraphExec* output, CUgraph graph, unsigned long long flags) {
    return flags == 17 ? create(2, output, graph) : CUDA_ERROR_INVALID_VALUE;
}
extern "C" __attribute__((visibility("hidden"))) CUresult CUDAAPI
native_cugraphinstantiatewithparams(CUgraphExec* output, CUgraph graph,
                                    CUDA_GRAPH_INSTANTIATE_PARAMS* params) {
    return params == g_params ? create(3, output, graph) : CUDA_ERROR_INVALID_VALUE;
}
extern "C" __attribute__((visibility("hidden"))) CUresult CUDAAPI
native_cugraphinstantiatewithparams_ptsz(CUgraphExec* output, CUgraph graph,
                                         CUDA_GRAPH_INSTANTIATE_PARAMS* params) {
    return params == g_params ? create(4, output, graph) : CUDA_ERROR_INVALID_VALUE;
}
extern "C" __attribute__((visibility("hidden"))) CUresult CUDAAPI
native_cugraphexecdestroy(CUgraphExec executable) {
    return mutate(5, executable, true);
}
extern "C" __attribute__((visibility("hidden"))) CUresult CUDAAPI
native_cugraphexeceventrecordnodesetevent(CUgraphExec executable, CUgraphNode node, CUevent event) {
    return mutate(6, executable, node == kNode && event == kEvent);
}
extern "C" __attribute__((visibility("hidden"))) CUresult CUDAAPI
native_cugraphexeceventwaitnodesetevent(CUgraphExec executable, CUgraphNode node, CUevent event) {
    return mutate(7, executable, node == kNode && event == kEvent);
}
extern "C" __attribute__((visibility("hidden"))) CUresult CUDAAPI
native_cugraphexecchildgraphnodesetparams(CUgraphExec executable, CUgraphNode node, CUgraph graph) {
    return mutate(8, executable, node == kNode && graph == kGraph);
}
extern "C" __attribute__((visibility("hidden"))) CUresult CUDAAPI
native_cugraphnodesetenabled(CUgraphExec executable, CUgraphNode node, unsigned int is_enabled) {
    return mutate(9, executable, node == kNode && is_enabled == 0);
}
#if CUDA_VERSION >= 12020
extern "C" __attribute__((visibility("hidden"))) CUresult CUDAAPI native_cugraphexecnodesetparams(
    CUgraphExec executable, CUgraphNode node, CUgraphNodeParams* params) {
    return mutate(10, executable, node == kNode && params != nullptr);
}
#endif
extern "C" __attribute__((visibility("hidden"))) CUresult CUDAAPI
native_cugraphexecupdate(CUgraphExec executable, CUgraph graph, CUgraphNode* error_node,
                         CUgraphExecUpdateResult* result) {
    return mutate(11, executable,
                  graph == kGraph && error_node == g_error_node && result != nullptr);
}
extern "C" __attribute__((visibility("hidden"))) CUresult CUDAAPI native_cugraphexecupdate_v2(
    CUgraphExec executable, CUgraph graph, CUgraphExecUpdateResultInfo* result) {
    return mutate(12, executable, graph == kGraph && result != nullptr);
}
// Match the Driver resolver's version/flags order.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
extern "C" void* glimmer_test_graph_symbol(const char* name, int version, cuuint64_t flags) {
    const std::string_view symbol{name};
    if (symbol == "cuGraphInstantiate") {
        return version < 11000   ? reinterpret_cast<void*>(&native_cugraphinstantiate)
               : version < 11040 ? reinterpret_cast<void*>(&native_cugraphinstantiate_v2)
                                 : reinterpret_cast<void*>(&native_cugraphinstantiatewithflags);
    }
    if (symbol == "cuGraphInstantiate_v2")
        return reinterpret_cast<void*>(&native_cugraphinstantiate_v2);
    if (symbol == "cuGraphInstantiateWithFlags")
        return reinterpret_cast<void*>(&native_cugraphinstantiatewithflags);
    if (symbol == "cuGraphInstantiateWithParams") {
        return flags == CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM
                   ? reinterpret_cast<void*>(&native_cugraphinstantiatewithparams_ptsz)
                   : reinterpret_cast<void*>(&native_cugraphinstantiatewithparams);
    }
    if (symbol == "cuGraphInstantiateWithParams_ptsz")
        return reinterpret_cast<void*>(&native_cugraphinstantiatewithparams_ptsz);
    if (symbol == "cuGraphExecDestroy") return reinterpret_cast<void*>(&native_cugraphexecdestroy);
    if (symbol == "cuGraphExecEventRecordNodeSetEvent")
        return reinterpret_cast<void*>(&native_cugraphexeceventrecordnodesetevent);
    if (symbol == "cuGraphExecEventWaitNodeSetEvent")
        return reinterpret_cast<void*>(&native_cugraphexeceventwaitnodesetevent);
    if (symbol == "cuGraphExecChildGraphNodeSetParams")
        return reinterpret_cast<void*>(&native_cugraphexecchildgraphnodesetparams);
    if (symbol == "cuGraphNodeSetEnabled")
        return reinterpret_cast<void*>(&native_cugraphnodesetenabled);
#if CUDA_VERSION >= 12020
    if (symbol == "cuGraphExecNodeSetParams")
        return reinterpret_cast<void*>(&native_cugraphexecnodesetparams);
#endif
    if (symbol == "cuGraphExecUpdate")
        return version < 12000 ? reinterpret_cast<void*>(&native_cugraphexecupdate)
                               : reinterpret_cast<void*>(&native_cugraphexecupdate_v2);
    if (symbol == "cuGraphExecUpdate_v2")
        return reinterpret_cast<void*>(&native_cugraphexecupdate_v2);
    return nullptr;
}

// Public exports delegate to privately bound original implementations.
extern "C" CUresult CUDAAPI cuGraphInstantiate(CUgraphExec* output, CUgraph graph,
                                               CUgraphNode* error_node, char* log,
                                               std::size_t log_bytes) {
    return native_cugraphinstantiate(output, graph, error_node, log, log_bytes);
}
extern "C" CUresult CUDAAPI cuGraphInstantiate_v2(CUgraphExec* output, CUgraph graph,
                                                  CUgraphNode* error_node, char* log,
                                                  std::size_t log_bytes) {
    return native_cugraphinstantiate_v2(output, graph, error_node, log, log_bytes);
}
extern "C" CUresult CUDAAPI cuGraphInstantiateWithFlags(CUgraphExec* output, CUgraph graph,
                                                        unsigned long long flags) {
    return native_cugraphinstantiatewithflags(output, graph, flags);
}
extern "C" CUresult CUDAAPI cuGraphInstantiateWithParams(CUgraphExec* output, CUgraph graph,
                                                         CUDA_GRAPH_INSTANTIATE_PARAMS* params) {
    return native_cugraphinstantiatewithparams(output, graph, params);
}
extern "C" CUresult CUDAAPI cuGraphInstantiateWithParams_ptsz(
    CUgraphExec* output, CUgraph graph, CUDA_GRAPH_INSTANTIATE_PARAMS* params) {
    return native_cugraphinstantiatewithparams_ptsz(output, graph, params);
}
extern "C" CUresult CUDAAPI cuGraphExecDestroy(CUgraphExec executable) {
    return native_cugraphexecdestroy(executable);
}
extern "C" CUresult CUDAAPI cuGraphExecEventRecordNodeSetEvent(CUgraphExec executable,
                                                               CUgraphNode node, CUevent event) {
    return native_cugraphexeceventrecordnodesetevent(executable, node, event);
}
extern "C" CUresult CUDAAPI cuGraphExecEventWaitNodeSetEvent(CUgraphExec executable,
                                                             CUgraphNode node, CUevent event) {
    return native_cugraphexeceventwaitnodesetevent(executable, node, event);
}
extern "C" CUresult CUDAAPI cuGraphExecChildGraphNodeSetParams(CUgraphExec executable,
                                                               CUgraphNode node, CUgraph graph) {
    return native_cugraphexecchildgraphnodesetparams(executable, node, graph);
}
extern "C" CUresult CUDAAPI cuGraphNodeSetEnabled(CUgraphExec executable, CUgraphNode node,
                                                  unsigned int is_enabled) {
    return native_cugraphnodesetenabled(executable, node, is_enabled);
}
#if CUDA_VERSION >= 12020
extern "C" CUresult CUDAAPI cuGraphExecNodeSetParams(CUgraphExec executable, CUgraphNode node,
                                                     CUgraphNodeParams* params) {
    return native_cugraphexecnodesetparams(executable, node, params);
}
#endif
extern "C" CUresult CUDAAPI cuGraphExecUpdate(CUgraphExec executable, CUgraph graph,
                                              CUgraphNode* error_node,
                                              CUgraphExecUpdateResult* result) {
    return native_cugraphexecupdate(executable, graph, error_node, result);
}
extern "C" CUresult CUDAAPI cuGraphExecUpdate_v2(CUgraphExec executable, CUgraph graph,
                                                 CUgraphExecUpdateResultInfo* result) {
    return native_cugraphexecupdate_v2(executable, graph, result);
}
