#include "internal/driver_graph_wrappers.h"

#include <dlfcn.h>
#include <array>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template <typename Function>
Function lookup(const char* name) {
    auto result = reinterpret_cast<Function>(dlsym(RTLD_DEFAULT, name));
    require(result != nullptr, "exact graph symbol must resolve");
    return result;
}
using LegacyProc = CUresult(CUDAAPI*)(const char*, void**, int, cuuint64_t);
using ModernProc = CUresult(CUDAAPI*)(const char*, void**, int, cuuint64_t,
                                      CUdriverProcAddressQueryResult*);
template <typename Function>
Function procedure(const char* name, int version, cuuint64_t flags, const char* expected,
                   bool is_modern) {
    void* result{};
    CUdriverProcAddressQueryResult status{};
    const auto code =
        is_modern
            ? lookup<ModernProc>("cuGetProcAddress_v2")(name, &result, version, flags, &status)
            : lookup<LegacyProc>("cuGetProcAddress")(name, &result, version, flags);
    if (code != CUDA_SUCCESS || (is_modern && status != CU_GET_PROC_ADDRESS_SUCCESS) ||
        result != reinterpret_cast<void*>(lookup<Function>(expected))) {
        throw std::runtime_error(
            std::string{"proc-address graph ABI: "} + name + " version=" + std::to_string(version) +
            " modern=" + std::to_string(is_modern) + " result=" + std::to_string(code) +
            " status=" + std::to_string(status) + " selected=" +
            std::to_string(result == reinterpret_cast<void*>(lookup<Function>(expected))));
    }
    return reinterpret_cast<Function>(result);
}

void forwarding() {
    using Arguments =
        void (*)(CUgraphExec*, CUgraphNode*, char*, std::size_t, CUDA_GRAPH_INSTANTIATE_PARAMS*);
    using Result = void (*)(CUresult);
    using Calls = unsigned int (*)(unsigned int);
    const auto args =
        reinterpret_cast<Arguments>(dlsym(RTLD_DEFAULT, "glimmer_test_graph_arguments"));
    const auto set_result =
        reinterpret_cast<Result>(dlsym(RTLD_DEFAULT, "glimmer_test_graph_result"));
    const auto calls = reinterpret_cast<Calls>(dlsym(RTLD_DEFAULT, "glimmer_test_graph_calls"));
    require(args && set_result && calls, "independent original ABI audit");
    const auto graph = reinterpret_cast<CUgraph>(0x6000);
    const auto node = reinterpret_cast<CUgraphNode>(0x7000);
    const auto event = reinterpret_cast<CUevent>(0x3000);
    const auto executable = reinterpret_cast<CUgraphExec>(0x5000);
    CUgraphExec output{};
    CUgraphNode error_node{};
    std::array<char, 19> log{};
    CUDA_GRAPH_INSTANTIATE_PARAMS params{};
    CUgraphExecUpdateResult legacy_update{};
    args(&output, &error_node, log.data(), log.size(), &params);
    // Direct ELF import is checked in addition to dynamic/procedure lookup.
    require(cuGraphInstantiateWithFlags(&output, graph, 17) == CUDA_SUCCESS &&
                output == executable && calls(2) == 1,
            "direct original constructor pointer/flags/output");
    for (auto result : {CUDA_SUCCESS, CUDA_ERROR_LAUNCH_FAILED}) {
        set_result(result);
        for (bool is_modern : {false, true}) {
            require(procedure<decltype(&cuGraphInstantiate)>("cuGraphInstantiate", 10000, 0,
                                                             "cuGraphInstantiate", is_modern)(
                        &output, graph, &error_node, log.data(), log.size()) == result &&
                        procedure<decltype(&cuGraphInstantiate_v2)>(
                            "cuGraphInstantiate", 11000, 0, "cuGraphInstantiate_v2", is_modern)(
                            &output, graph, &error_node, log.data(), log.size()) == result &&
                        procedure<decltype(&cuGraphInstantiateWithFlags)>(
                            "cuGraphInstantiate", 11040, 0, "cuGraphInstantiateWithFlags",
                            is_modern)(&output, graph, 17) == result,
                    "legacy, v2 and flags constructors retain distinct original ABI/errors");
            require(procedure<decltype(&cuGraphExecUpdate)>("cuGraphExecUpdate", 10020, 0,
                                                            "cuGraphExecUpdate", is_modern)(
                        executable, graph, &error_node, &legacy_update) == result,
                    "legacy four-argument update ABI");
            CUgraphExecUpdateResultInfo update{};
            require(procedure<decltype(&cuGraphExecUpdate_v2)>("cuGraphExecUpdate", 12000, 0,
                                                               "cuGraphExecUpdate_v2", is_modern)(
                        executable, graph, &update) == result,
                    "modern three-argument update ABI");
            for (bool is_ptds : {false, true}) {
                require(procedure<decltype(&cuGraphInstantiateWithParams)>(
                            "cuGraphInstantiateWithParams", 12000,
                            is_ptds ? CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM
                                    : CU_GET_PROC_ADDRESS_LEGACY_STREAM,
                            is_ptds ? "cuGraphInstantiateWithParams_ptsz"
                                    : "cuGraphInstantiateWithParams",
                            is_modern)(&output, graph, &params) == result,
                        "params and exact PTDS constructor pointer");
            }
        }
        require(lookup<decltype(&cuGraphExecDestroy)>("cuGraphExecDestroy")(executable) == result &&
                    lookup<decltype(&cuGraphExecEventRecordNodeSetEvent)>(
                        "cuGraphExecEventRecordNodeSetEvent")(executable, node, event) == result &&
                    lookup<decltype(&cuGraphExecEventWaitNodeSetEvent)>(
                        "cuGraphExecEventWaitNodeSetEvent")(executable, node, event) == result &&
                    lookup<decltype(&cuGraphExecChildGraphNodeSetParams)>(
                        "cuGraphExecChildGraphNodeSetParams")(executable, node, graph) == result &&
                    lookup<decltype(&cuGraphNodeSetEnabled)>("cuGraphNodeSetEnabled")(
                        executable, node, 0) == result,
                "native mutation and destruction arguments/errors remain unchanged");
#if CUDA_VERSION >= 12020
        CUgraphNodeParams generic{};
        require(lookup<decltype(&cuGraphExecNodeSetParams)>("cuGraphExecNodeSetParams")(
                    executable, node, &generic) == result,
                "generic executable update forwards its exact params pointer");
#endif
    }
    require(calls(0) == 4 && calls(1) == 4 && calls(2) == 5 && calls(3) == 4 && calls(4) == 4 &&
                calls(11) == 4 && calls(12) == 4,
            "every native constructor/update executes once");
    for (unsigned int index = 5; index <= 10; ++index) {
#if CUDA_VERSION < 12020
        if (index == 10) continue;
#endif
        require(calls(index) == 2, "native mutation is not replayed");
    }
}
}  // namespace
int main() {
    try {
        forwarding();
        std::cout << "original Driver graph lifecycle ABI forwarding passed\n";
        // The Driver dispatch intentionally retains its loader for process lifetime.
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
