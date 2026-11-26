#include "internal/driver_graph_wrappers.h"
#include "internal/cuda_native_graph_effects.h"
#include "internal/driver_api_interceptor.h"
#include "internal/driver_dispatch.h"

namespace {
template <typename Operation>
CUresult guard_graph(Operation&& operation) noexcept {
    try {
        const glimmer::interceptor::CooperativeCallScope scope;
        return operation();
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}
template <typename Function, typename... Arguments>
CUresult forward_graph(Function native, Arguments... arguments) noexcept {
    try {
        if (!native) {
            return CUDA_ERROR_NOT_SUPPORTED;
        }
        const glimmer::interceptor::NativeDriverCallScope scope;
        return native(arguments...);
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}
template <typename Function, typename... Arguments>
CUresult construct_graph(Function native, CUgraphExec* output, CUgraph graph, bool should_prepare,
                         Arguments... arguments) {
    if (!native) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return glimmer::interceptor::intercept_graph_instantiation(
        output, graph, should_prepare, [&](CUgraphExec* native_output) {
            return forward_graph(native, native_output, graph, arguments...);
        });
}
template <typename Function, typename... Arguments>
CUresult mutate_graph(Function native, CUgraphExec executable, bool is_destroy,
                      Arguments... arguments) {
    if (!native) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return glimmer::interceptor::intercept_graph_mutation(
        executable, is_destroy, [&] { return forward_graph(native, executable, arguments...); });
}
}  // namespace

extern "C" CUresult CUDAAPI cuGraphInstantiate(CUgraphExec* output, CUgraph graph,
                                               CUgraphNode* error_node, char* log,
                                               std::size_t log_bytes) {
    return guard_graph([&] {
        static const auto native = reinterpret_cast<decltype(&cuGraphInstantiate)>(
            glimmer::interceptor::resolve_native_driver_function("cuGraphInstantiate", false,
                                                                 10000));
        return construct_graph(native, output, graph, true, error_node, log, log_bytes);
    });
}
extern "C" CUresult CUDAAPI cuGraphInstantiate_v2(CUgraphExec* output, CUgraph graph,
                                                  CUgraphNode* error_node, char* log,
                                                  std::size_t log_bytes) {
    return guard_graph([&] {
        static const auto native = reinterpret_cast<decltype(&cuGraphInstantiate_v2)>(
            glimmer::interceptor::resolve_native_driver_function("cuGraphInstantiate", false,
                                                                 11000));
        return construct_graph(native, output, graph, true, error_node, log, log_bytes);
    });
}
extern "C" CUresult CUDAAPI cuGraphInstantiateWithFlags(CUgraphExec* output, CUgraph graph,
                                                        unsigned long long flags) {
    return guard_graph([&] {
        static const auto native = reinterpret_cast<decltype(&cuGraphInstantiateWithFlags)>(
            glimmer::interceptor::resolve_native_driver_function("cuGraphInstantiateWithFlags"));
        return construct_graph(
            native, output, graph,
            glimmer::interceptor::native_graph_flags_allow_effect_preparation(flags), flags);
    });
}
extern "C" CUresult CUDAAPI cuGraphInstantiateWithParams(CUgraphExec* output, CUgraph graph,
                                                         CUDA_GRAPH_INSTANTIATE_PARAMS* params) {
    return guard_graph([&] {
        static const auto native = reinterpret_cast<decltype(&cuGraphInstantiateWithParams)>(
            glimmer::interceptor::resolve_native_driver_function("cuGraphInstantiateWithParams"));
        // Parameters may request upload/device launch; retain original validation
        // and an unprepared row until those progress contracts are modeled.
        return construct_graph(native, output, graph, false, params);
    });
}
extern "C" CUresult CUDAAPI cuGraphInstantiateWithParams_ptsz(
    CUgraphExec* output, CUgraph graph, CUDA_GRAPH_INSTANTIATE_PARAMS* params) {
    return guard_graph([&] {
        static const auto native = reinterpret_cast<decltype(&cuGraphInstantiateWithParams)>(
            glimmer::interceptor::resolve_native_driver_function("cuGraphInstantiateWithParams",
                                                                 true));
        return construct_graph(native, output, graph, false, params);
    });
}
extern "C" CUresult CUDAAPI cuGraphExecDestroy(CUgraphExec executable) {
    return guard_graph([&] {
        static const auto native = reinterpret_cast<decltype(&cuGraphExecDestroy)>(
            glimmer::interceptor::resolve_native_driver_function("cuGraphExecDestroy"));
        return mutate_graph(native, executable, true);
    });
}
extern "C" CUresult CUDAAPI cuGraphExecEventRecordNodeSetEvent(CUgraphExec executable,
                                                               CUgraphNode node, CUevent event) {
    return guard_graph([&] {
        static const auto native = reinterpret_cast<decltype(&cuGraphExecEventRecordNodeSetEvent)>(
            glimmer::interceptor::resolve_native_driver_function(
                "cuGraphExecEventRecordNodeSetEvent"));
        if (!native) return CUDA_ERROR_NOT_SUPPORTED;
        return glimmer::interceptor::intercept_graph_event_update(
            executable, node, event, true,
            [&] { return forward_graph(native, executable, node, event); });
    });
}
extern "C" CUresult CUDAAPI cuGraphExecEventWaitNodeSetEvent(CUgraphExec executable,
                                                             CUgraphNode node, CUevent event) {
    return guard_graph([&] {
        static const auto native = reinterpret_cast<decltype(&cuGraphExecEventWaitNodeSetEvent)>(
            glimmer::interceptor::resolve_native_driver_function(
                "cuGraphExecEventWaitNodeSetEvent"));
        if (!native) return CUDA_ERROR_NOT_SUPPORTED;
        return glimmer::interceptor::intercept_graph_event_update(
            executable, node, event, false,
            [&] { return forward_graph(native, executable, node, event); });
    });
}
extern "C" CUresult CUDAAPI cuGraphExecChildGraphNodeSetParams(CUgraphExec executable,
                                                               CUgraphNode node, CUgraph child) {
    return guard_graph([&] {
        static const auto native = reinterpret_cast<decltype(&cuGraphExecChildGraphNodeSetParams)>(
            glimmer::interceptor::resolve_native_driver_function(
                "cuGraphExecChildGraphNodeSetParams"));
        if (!native) return CUDA_ERROR_NOT_SUPPORTED;
        return glimmer::interceptor::intercept_graph_update(executable, child, node, [&] {
            return forward_graph(native, executable, node, child);
        });
    });
}
extern "C" CUresult CUDAAPI cuGraphNodeSetEnabled(CUgraphExec executable, CUgraphNode node,
                                                  unsigned int is_enabled) {
    return guard_graph([&] {
        static const auto native = reinterpret_cast<decltype(&cuGraphNodeSetEnabled)>(
            glimmer::interceptor::resolve_native_driver_function("cuGraphNodeSetEnabled"));
        if (!native) return CUDA_ERROR_NOT_SUPPORTED;
        return glimmer::interceptor::intercept_graph_node_update(
            executable, node, std::nullopt,
            [&] { return forward_graph(native, executable, node, is_enabled); });
    });
}
#if CUDA_VERSION >= 12020
extern "C" CUresult CUDAAPI cuGraphExecNodeSetParams(CUgraphExec executable, CUgraphNode node,
                                                     CUgraphNodeParams* params) {
    return guard_graph([&] {
        static const auto native = reinterpret_cast<decltype(&cuGraphExecNodeSetParams)>(
            glimmer::interceptor::resolve_native_driver_function("cuGraphExecNodeSetParams"));
        if (!native) return CUDA_ERROR_NOT_SUPPORTED;
        const auto invoke = [&] { return forward_graph(native, executable, node, params); };
        if (params && params->type == CU_GRAPH_NODE_TYPE_GRAPH)
            return glimmer::interceptor::intercept_graph_update(executable, params->graph.graph,
                                                                node, invoke);
        if (params && (params->type == CU_GRAPH_NODE_TYPE_EVENT_RECORD ||
                       params->type == CU_GRAPH_NODE_TYPE_WAIT_EVENT)) {
            const bool is_record = params->type == CU_GRAPH_NODE_TYPE_EVENT_RECORD;
            return glimmer::interceptor::intercept_graph_event_update(
                executable, node, is_record ? params->eventRecord.event : params->eventWait.event,
                is_record, invoke);
        }
        return glimmer::interceptor::intercept_graph_node_update(
            executable, node, params ? std::optional{params->type} : std::nullopt, invoke);
    });
}
#endif
extern "C" CUresult CUDAAPI cuGraphExecUpdate(CUgraphExec executable, CUgraph graph,
                                              CUgraphNode* error_node,
                                              CUgraphExecUpdateResult* update_result) {
    return guard_graph([&] {
        static const auto native = reinterpret_cast<decltype(&cuGraphExecUpdate)>(
            glimmer::interceptor::resolve_native_driver_function("cuGraphExecUpdate", false,
                                                                 10020));
        if (!native) return CUDA_ERROR_NOT_SUPPORTED;
        return glimmer::interceptor::intercept_graph_update(executable, graph, std::nullopt, [&] {
            return forward_graph(native, executable, graph, error_node, update_result);
        });
    });
}
extern "C" CUresult CUDAAPI cuGraphExecUpdate_v2(CUgraphExec executable, CUgraph graph,
                                                 CUgraphExecUpdateResultInfo* result) {
    return guard_graph([&] {
        static const auto native = reinterpret_cast<decltype(&cuGraphExecUpdate_v2)>(
            glimmer::interceptor::resolve_native_driver_function("cuGraphExecUpdate", false,
                                                                 12000));
        if (!native) return CUDA_ERROR_NOT_SUPPORTED;
        return glimmer::interceptor::intercept_graph_update(executable, graph, std::nullopt, [&] {
            return forward_graph(native, executable, graph, result);
        });
    });
}
