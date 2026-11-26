#include "internal/driver_api_interceptor.h"
#include "internal/driver_cooperative_router.h"
#include "internal/driver_graph_wrappers.h"
#include "internal/cuda_native_graph_registry.h"

#include <dlfcn.h>
#include <array>
#include <iostream>
#include <stdexcept>

namespace {
using namespace glimmer::interceptor;
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void check(CUresult result, const char* message) {
    require(result == CUDA_SUCCESS, message);
}
template <typename Function>
Function procedure(const char* name, int version = 12600, cuuint64_t flags = 0) {
    void* address{};
    check(cuGetProcAddress(name, &address, version, flags, nullptr),
          "real versioned graph procedure");
    require(address != nullptr, "real graph procedure output");
    return reinterpret_cast<Function>(address);
}
void registration() {
    check(cuInit(0), "initialize original Driver");
    CUdevice device{};
    check(cuDeviceGet(&device, 0), "actual device");
    CUcontext context{};
    check(cuCtxCreate(&context, 0, device), "actual context");
    CUdeviceptr memory{};
    check(cuMemAlloc(&memory, sizeof(unsigned int)), "actual graph memset destination");
    CUevent event{};
    check(cuEventCreate(&event, CU_EVENT_DISABLE_TIMING), "registered original event");
    CUgraph graph{}, child{};
    check(cuGraphCreate(&graph, 0), "original source graph");
    check(cuGraphCreate(&child, 0), "original child source graph");
    CUgraphNode record{}, wait{}, child_node{}, empty{}, memory_node{};
    check(cuGraphAddEmptyNode(&empty, child, nullptr, 0), "original child empty node");
    check(cuGraphAddEventRecordNode(&record, graph, nullptr, 0, event),
          "original source event node");
    check(cuGraphAddEventWaitNode(&wait, graph, &record, 1, event), "original event wait identity");
    check(cuGraphAddChildGraphNode(&child_node, graph, &wait, 1, child), "original child copy");
    CUDA_MEMSET_NODE_PARAMS memory_params{};
    memory_params.dst = memory;
    memory_params.elementSize = sizeof(unsigned int);
    memory_params.width = memory_params.height = 1;
    check(cuGraphAddMemsetNode(&memory_node, graph, &child_node, 1, &memory_params, context),
          "original known non-event node");
    const auto prepare = [&](CUgraphExec executable) {
        CudaNativeGraphExecutable owned;
        check(cooperative_graph_executable(executable, owned), "public constructor registration");
        require(owned.context() == context && owned.effects().node_count() == 5 &&
                    owned.events().size() == 2 && owned.events()[0].effect.event == event &&
                    owned.events()[1].effect.event == event,
                "actual original identities/effects");
        return owned;
    };
    const auto create = [&] {
        CUgraphExec executable{};
        check(cuGraphInstantiateWithFlags(&executable, graph, 0), "public original constructor");
        return executable;
    };
    // The native constructor may not write an output on validation failure.
    // Preview staging must never read an uninitialized caller output.
    CUgraphExec rejected;
    require(cuGraphInstantiateWithFlags(&rejected, graph, ~0ULL) == CUDA_ERROR_INVALID_VALUE,
            "original invalid flags reject without an uninitialized output read");
    require(cuGraphInstantiateWithFlags(nullptr, graph, 0) == CUDA_ERROR_INVALID_VALUE,
            "original null-output validation remains ordered and exact");
    const auto expect_invalidated = [&](CUgraphExec executable,
                                        const CudaNativeGraphExecutable& old) {
        CudaNativeGraphExecutable current;
        require(cooperative_graph_executable(executable, current) == CUDA_ERROR_NOT_SUPPORTED &&
                    !current && old.events().size() == 2 && old.effects().node_count() == 5,
                "public mutation must retire future plan without erasing retained snapshot");
        check(cuGraphExecDestroy(executable), "original executable destruction");
        require(cooperative_graph_executable(executable, current) == CUDA_ERROR_INVALID_HANDLE,
                "successful original destruction must retire public registration");
    };
    const auto expect_replaced = [&](CUgraphExec executable, const CudaNativeGraphExecutable& old) {
        auto current = prepare(executable);
        {
            CooperativeCallScope scope;
            auto* router = cooperative_source_router();
            require(router && router->graphs().is_current(old) == CUDA_ERROR_INVALID_HANDLE &&
                        old.events().size() == 2 && old.effects().node_count() == 5,
                    "event mutation replaces the generation without changing retained effects");
        }
        check(cuGraphExecDestroy(executable), "original executable destruction after replacement");
        current = {};
        require(cooperative_graph_executable(executable, current) == CUDA_ERROR_INVALID_HANDLE,
                "replacement registration retires with original destruction");
    };
    auto executable = create();
    auto owned = prepare(executable);
    check(cuGraphExecEventRecordNodeSetEvent(executable, record, event), "original record update");
    expect_replaced(executable, owned);
    executable = create();
    owned = prepare(executable);
    check(cuGraphExecEventWaitNodeSetEvent(executable, wait, event), "original wait update");
    expect_replaced(executable, owned);
    executable = create();
    owned = prepare(executable);
    check(cuGraphNodeSetEnabled(executable, memory_node, 0), "original known node disablement");
    expect_replaced(executable, owned);
    executable = create();
    owned = prepare(executable);
    check(cuGraphExecChildGraphNodeSetParams(executable, child_node, child),
          "original child update");
    expect_replaced(executable, owned);
    executable = create();
    owned = prepare(executable);
    require(cuGraphNodeSetEnabled(executable, nullptr, 0) == CUDA_ERROR_INVALID_VALUE,
            "invalid original enablement preserves native status");
    expect_invalidated(executable, owned);
#if CUDA_VERSION >= 12020
    executable = create();
    owned = prepare(executable);
    CUgraphNodeParams generic{};
    generic.type = CU_GRAPH_NODE_TYPE_EVENT_RECORD;
    generic.eventRecord.event = event;
    check(cuGraphExecNodeSetParams(executable, record, &generic), "original generic record update");
    expect_replaced(executable, owned);
    executable = create();
    owned = prepare(executable);
    generic = {};
    generic.type = CU_GRAPH_NODE_TYPE_MEMSET;
    generic.memset.dst = memory;
    generic.memset.value = 7;
    generic.memset.elementSize = sizeof(unsigned int);
    generic.memset.width = generic.memset.height = 1;
    generic.memset.ctx = context;
    check(cuGraphExecNodeSetParams(executable, memory_node, &generic),
          "original generic memset parameters preserve dependencies");
    expect_replaced(executable, owned);
#endif
    executable = create();
    owned = prepare(executable);
    CUgraphExecUpdateResultInfo modern{};
    check(procedure<decltype(&cuGraphExecUpdate_v2)>("cuGraphExecUpdate", 12000)(executable, graph,
                                                                                 &modern),
          "original modern executable update");
    require(modern.result == CU_GRAPH_EXEC_UPDATE_SUCCESS, "original modern update result");
    expect_replaced(executable, owned);
    executable = create();
    owned = prepare(executable);
    CUgraphExecUpdateResult legacy{};
    CUgraphNode error_node{};
    check(procedure<decltype(&cuGraphExecUpdate)>("cuGraphExecUpdate", 10020)(executable, graph,
                                                                              &error_node, &legacy),
          "original legacy executable update");
    require(legacy == CU_GRAPH_EXEC_UPDATE_SUCCESS, "original legacy update result");
    expect_replaced(executable, owned);
    for (int version : {10000, 11000, 11040}) {
        executable = nullptr;
        const auto result =
            version < 11040
                ? procedure<decltype(&cuGraphInstantiate)>("cuGraphInstantiate", version)(
                      &executable, graph, &error_node, nullptr, 0)
                : procedure<decltype(&cuGraphInstantiateWithFlags)>("cuGraphInstantiate", version)(
                      &executable, graph, 0);
        check(result, "real version-specific constructor ABI");
        owned = prepare(executable);
        check(cuGraphExecDestroy(executable), "retire exact original constructor instance");
    }
    for (auto flags :
         {CU_GET_PROC_ADDRESS_LEGACY_STREAM, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM}) {
        executable = nullptr;
        CUDA_GRAPH_INSTANTIATE_PARAMS params{};
        check(procedure<decltype(&cuGraphInstantiateWithParams)>(
                  "cuGraphInstantiateWithParams", 12000, flags)(&executable, graph, &params),
              "real original params/PTDS instantiation");
        CudaNativeGraphExecutable unprepared;
        require(cooperative_graph_executable(executable, unprepared) == CUDA_ERROR_NOT_SUPPORTED &&
                    !unprepared,
                "params/upload contract is not guessed from source graph");
        check(cuGraphExecDestroy(executable), "retire original params instance");
    }
    // Exercise repeated CPU-slot reclamation beyond the registry bound.
    for (unsigned int i = 0; i < 260; ++i) {
        executable = create();
        owned = prepare(executable);
        check(cuGraphExecDestroy(executable), "reclaim public constructor ownership");
    }
    executable = create();
    owned = prepare(executable);
    check(cuGraphDestroy(graph), "destroy original source before executable retirement");
    check(cuGraphDestroy(child), "destroy original child source");
    auto retained = prepare(executable);
    require(retained.events().size() == owned.events().size(),
            "no getter queries destroyed source");
    check(cuGraphExecDestroy(executable), "destroy original executable after source retirement");
    CudaNativeGraphExecutable retired;
    require(cooperative_graph_executable(executable, retired) == CUDA_ERROR_INVALID_HANDLE &&
                retained.events().size() == 2,
            "retained snapshot owns CPU only");
    check(cuEventDestroy(event), "checked actual event destruction");
    check(cuMemFree(memory), "checked actual graph memset allocation release");
    check(cuCtxDestroy(context), "checked context teardown");
    std::cout << "public Driver graph registration lifetime passed; replay remains disabled\n";
}
}  // namespace
int main() {
    try {
        registration();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
