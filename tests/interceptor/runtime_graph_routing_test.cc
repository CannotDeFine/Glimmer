#include "internal/runtime_api_bridge.h"
#include "internal/driver_cooperative_router.h"
#include "internal/driver_api_interceptor.h"
#include "internal/symbol_registry.h"

#include <cuda_runtime_api.h>
#include <array>
#include <cstdlib>
#include <dlfcn.h>
#include <iostream>
#include <stdexcept>
#include <string_view>

#ifdef cudaGraphInstantiateWithParams
#undef cudaGraphInstantiateWithParams
#endif

extern "C" unsigned int glimmer_fake_runtime_graph_constructor_calls();
extern "C" unsigned int glimmer_fake_runtime_graph_destroy_calls();
extern "C" unsigned int glimmer_fake_runtime_graph_base_calls(bool is_legacy);
extern "C" unsigned int glimmer_fake_runtime_graph_event_update_calls(bool is_record);
extern "C" unsigned int glimmer_fake_runtime_graph_node_update_calls(bool is_generic);
extern "C" unsigned int glimmer_fake_runtime_graph_update_calls(bool is_legacy);
extern "C" unsigned int glimmer_fake_runtime_graph_child_update_calls();
extern "C" cudaError_t CUDARTAPI legacy_update_import(cudaGraphExec_t, cudaGraph_t,
                                                      cudaGraphNode_t*, cudaGraphExecUpdateResult*);
extern "C" cudaError_t CUDARTAPI modern_update_import(cudaGraphExec_t, cudaGraph_t,
                                                      cudaGraphExecUpdateResultInfo*);
asm(".symver legacy_update_import,cudaGraphExecUpdate@libcudart.so.11.0");
asm(".symver modern_update_import,cudaGraphExecUpdate@libcudart.so.12");
extern "C" cudaError_t CUDARTAPI legacy_graph_import(cudaGraphExec_t*, cudaGraph_t,
                                                     cudaGraphNode_t*, char*, std::size_t);
extern "C" cudaError_t CUDARTAPI modern_graph_import(cudaGraphExec_t*, cudaGraph_t,
                                                     unsigned long long);
asm(".symver legacy_graph_import,cudaGraphInstantiate@libcudart.so.11.0");
asm(".symver modern_graph_import,cudaGraphInstantiate@libcudart.so.12");

namespace {
using namespace glimmer::interceptor;
const auto kGraph = reinterpret_cast<cudaGraph_t>(0xfa020);
const auto kExecutable = reinterpret_cast<cudaGraphExec_t>(0xfa010);
const auto kSentinel = reinterpret_cast<cudaGraphExec_t>(0xfa080);

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Function>
Function wrapper(void* handle, const char* name) {
    const auto symbol = ::dlsym(handle, name);
    require(symbol && symbol == find_interceptor_symbol(name),
            "default/explicit SDK lookup selects exact covered wrapper ABI");
    return reinterpret_cast<Function>(symbol);
}

void check_public_routing() {
    void* sdk = ::dlopen("libcudart.so", RTLD_NOW | RTLD_LOCAL);
    require(sdk, "load independently linked fake SDK for explicit lookup");
    const ProcessCallGate::Call call(cooperative_process_call_gate());
    auto* router = cooperative_source_router();
    require(router, "fake SDK root owns actual CPU graph metadata");
    const auto legacy_address = runtime_graph_constructor_version("libcudart.so.11.0");
    const auto modern_address = runtime_graph_constructor_version("libcudart.so.12");
    require(
        legacy_address && modern_address && legacy_address != modern_address &&
            reinterpret_cast<void*>(&legacy_graph_import) == legacy_address &&
            reinterpret_cast<void*>(&modern_graph_import) == modern_address &&
            dlvsym(RTLD_DEFAULT, "cudaGraphInstantiate", "libcudart.so.11.0") == legacy_address &&
            dlvsym(RTLD_DEFAULT, "cudaGraphInstantiate", "libcudart.so.12") == modern_address &&
            !runtime_graph_constructor_version(nullptr) &&
            !runtime_graph_constructor_version("libcudart.so.99"),
        "versioned imports and exported versions select distinct exact constructor ABIs");
    cudaGraphExec_t base_output{};
    cudaGraphNode_t error_node{};
    std::array<char, 19> log{};
    const auto legacy_before = glimmer_fake_runtime_graph_base_calls(true);
    const auto modern_before = glimmer_fake_runtime_graph_base_calls(false);
    require(legacy_graph_import(&base_output, kGraph, &error_node, log.data(), log.size()) ==
                    cudaSuccess &&
                base_output == kExecutable &&
                error_node == reinterpret_cast<cudaGraphNode_t>(0xfa060) && log.front() == 'L' &&
                log.back() == '!' &&
                glimmer_fake_runtime_graph_base_calls(true) == legacy_before + 1 &&
                glimmer_fake_runtime_graph_base_calls(false) == modern_before,
            "legacy import forwards all five arguments and SDK auxiliary outputs exactly once");
    require(cudaGraphExecDestroy(base_output) == cudaSuccess,
            "retire legacy base constructor slot");
    base_output = kSentinel;
    require(modern_graph_import(&base_output, kGraph, 10000) == cudaErrorApiFailureBase &&
                base_output == kSentinel &&
                glimmer_fake_runtime_graph_base_calls(false) == modern_before + 1,
            "modern import forwards flags and exact Runtime-only error without touching output");
    require(modern_graph_import(&base_output, kGraph, 0) == cudaSuccess &&
                base_output == kExecutable && cudaGraphExecDestroy(base_output) == cudaSuccess,
            "modern base import registers and retires original SDK executable");
    for (void* handle : {RTLD_DEFAULT, sdk}) {
#if CUDART_VERSION >= 12000
        require(dlsym(handle, "cudaGraphInstantiate") == modern_address,
                "default/explicit lookup selects original SDK modern ABI");
#else
        require(dlsym(handle, "cudaGraphInstantiate") == legacy_address,
                "default/explicit lookup selects original SDK legacy ABI");
#endif
        const auto construct = wrapper<decltype(&cudaGraphInstantiateWithFlags)>(
            handle, "cudaGraphInstantiateWithFlags");
        const auto destroy =
            wrapper<decltype(&cudaGraphExecDestroy)>(handle, "cudaGraphExecDestroy");
        const auto before = glimmer_fake_runtime_graph_constructor_calls();
        cudaGraphExec_t output = kSentinel;
        require(construct(&output, kGraph, 10000) == cudaErrorApiFailureBase &&
                    output == kSentinel &&
                    glimmer_fake_runtime_graph_constructor_calls() == before + 1,
                "SDK-only constructor error preserves output and invokes original exactly once");
        require(construct(&output, kGraph, 10001) == cudaErrorApiFailureBase &&
                    output == kExecutable &&
                    glimmer_fake_runtime_graph_constructor_calls() == before + 2,
                "published-on-error SDK handle is staged, returned and retained without replay");
        CudaNativeGraphExecutable owned;
        require(router->graphs().lookup(output, owned) == CUDA_ERROR_NOT_SUPPORTED,
                "failed native constructor publishes no execution permission");
        require(destroy(output) == cudaSuccess,
                "explicit SDK cleanup retires partial registration");
        require(router->graphs().lookup(output, owned) == CUDA_ERROR_INVALID_HANDLE,
                "root destroy retires CPU identity despite cached Driver dispatch");
        const auto destroyed = glimmer_fake_runtime_graph_destroy_calls();
        require(destroy(reinterpret_cast<cudaGraphExec_t>(0xfa099)) == cudaErrorApiFailureBase &&
                    glimmer_fake_runtime_graph_destroy_calls() == destroyed + 1,
                "typed original destructor error is neither converted nor replayed");
        output = kSentinel;
        require(construct(&output, kGraph, 0) == cudaSuccess && output == kExecutable,
                "ordinary flags ABI preserves graph/flags and original output");
        require(destroy(output) == cudaSuccess,
                "ordinary SDK destruction returns constructor slot");
        require(construct(nullptr, kGraph, 0) == cudaErrorInvalidValue,
                "null output receives original SDK validation once");
#if CUDART_VERSION >= 12000
        for (const auto* name :
             {"cudaGraphInstantiateWithParams", "cudaGraphInstantiateWithParams_ptsz"}) {
            const auto params_constructor =
                wrapper<decltype(&cudaGraphInstantiateWithParams)>(handle, name);
            cudaGraphInstantiateParams parameters{};
            parameters.uploadStream = reinterpret_cast<cudaStream_t>(0xfa040);
            const auto legacy_node = reinterpret_cast<cudaGraphNode_t>(0xfa050);
            const auto ptds_node = reinterpret_cast<cudaGraphNode_t>(0xfa051);
            const auto expected_node =
                std::string_view(name).ends_with("_ptsz") ? ptds_node : legacy_node;
            require(params_constructor(&output, kGraph, &parameters) == cudaSuccess &&
                        output == kExecutable &&
                        parameters.result_out == cudaGraphInstantiateSuccess &&
                        parameters.errNode_out == expected_node,
                    "exact params/PTDS original sees same input and publishes SDK output fields");
            require(destroy(output) == cudaSuccess, "params root destruction retires its SDK slot");
        }
#endif
    }
    using Setter = decltype(&cudaGraphExecEventRecordNodeSetEvent);
    for (void* handle : {RTLD_DEFAULT, sdk}) {
        for (const bool is_record : {false, true}) {
            const auto setter =
                wrapper<Setter>(handle, is_record ? "cudaGraphExecEventRecordNodeSetEvent"
                                                  : "cudaGraphExecEventWaitNodeSetEvent");
            cudaGraphExec_t executable{};
            require(cudaGraphInstantiateWithFlags(&executable, kGraph, 0) == cudaSuccess,
                    "prepare independently cached fake SDK executable");
            const auto before = glimmer_fake_runtime_graph_event_update_calls(is_record);
            require(setter(executable, reinterpret_cast<cudaGraphNode_t>(0xfa050),
                           reinterpret_cast<cudaEvent_t>(0xfa099)) == cudaErrorApiFailureBase &&
                        glimmer_fake_runtime_graph_event_update_calls(is_record) == before + 1,
                    "exact root setter preserves SDK-only failure once, without Driver callbacks");
            CudaNativeGraphExecutable unavailable;
            require(router->graphs().lookup(executable, unavailable) == CUDA_ERROR_NOT_SUPPORTED,
                    "SDK failure cannot restore an uncertain executable plan");
            require(
                setter(executable, reinterpret_cast<cudaGraphNode_t>(0xfa050),
                       reinterpret_cast<cudaEvent_t>(0xfa060)) == cudaSuccess &&
                    glimmer_fake_runtime_graph_event_update_calls(is_record) == before + 2 &&
                    router->graphs().lookup(executable, unavailable) == CUDA_ERROR_NOT_SUPPORTED,
                "unprepared SDK success stays native without guessed event dependencies");
            require(cudaGraphExecDestroy(executable) == cudaSuccess,
                    "retire the original SDK executable after setter validation");
            require(setter(nullptr, nullptr, nullptr) == cudaErrorInvalidValue &&
                        glimmer_fake_runtime_graph_event_update_calls(is_record) == before + 3,
                    "null setter input retains original SDK validation once");
        }
    }
    for (void* handle : {RTLD_DEFAULT, sdk}) {
        const auto enable =
            wrapper<decltype(&cudaGraphNodeSetEnabled)>(handle, "cudaGraphNodeSetEnabled");
        cudaGraphExec_t executable{};
        require(cudaGraphInstantiateWithFlags(&executable, kGraph, 0) == cudaSuccess,
                "construct original SDK executable for node mutation validation");
        const auto node = reinterpret_cast<cudaGraphNode_t>(0xfa050);
        const auto before = glimmer_fake_runtime_graph_node_update_calls(false);
        require(enable(executable, node, 10000) == cudaErrorApiFailureBase &&
                    enable(executable, node, 1) == cudaSuccess &&
                    enable(nullptr, nullptr, 0) == cudaErrorInvalidValue &&
                    glimmer_fake_runtime_graph_node_update_calls(false) == before + 3,
                "exact enablement arguments and typed original SDK errors occur once");
        CudaNativeGraphExecutable unavailable;
        require(router->graphs().lookup(executable, unavailable) == CUDA_ERROR_NOT_SUPPORTED,
                "unknown native enablement never guesses an executable plan");
#if CUDART_VERSION >= 12020
        const auto update =
            wrapper<decltype(&cudaGraphExecNodeSetParams)>(handle, "cudaGraphExecNodeSetParams");
        cudaGraphNodeParams params{};
        params.type = cudaGraphNodeTypeKernel;
        params.reserved0[0] = 10000;
        const auto generic_before = glimmer_fake_runtime_graph_node_update_calls(true);
        require(update(executable, node, &params) == cudaErrorApiFailureBase,
                "generic params pointer and Runtime-only SDK error are preserved");
        params.reserved0[0] = 0;
        require(update(executable, node, &params) == cudaSuccess &&
                    update(executable, node, nullptr) == cudaErrorInvalidValue &&
                    glimmer_fake_runtime_graph_node_update_calls(true) == generic_before + 3,
                "generic SDK originals receive valid and null parameters once");
#endif
        require(cudaGraphExecDestroy(executable) == cudaSuccess,
                "retire node mutation's original SDK executable");
    }
    require(!is_inside_runtime_call() && !is_inside_driver_call(),
            "public constructor/destructor restores native caller TLS");
    require(::dlclose(sdk) == 0, "balance explicit fake SDK loader reference");
}

void check_graph_updates() {
    void* sdk = ::dlopen("libcudart.so", RTLD_NOW | RTLD_LOCAL);
    require(sdk, "load independently cached SDK update entry points");
    const ProcessCallGate::Call call(cooperative_process_call_gate());
    auto* router = cooperative_source_router();
    require(router, "update roots have an admitted metadata owner");
    const auto legacy = runtime_graph_update_version("libcudart.so.11.0");
    const auto modern = runtime_graph_update_version("libcudart.so.12");
    require(legacy && modern && legacy != modern &&
                reinterpret_cast<void*>(&legacy_update_import) == legacy &&
                reinterpret_cast<void*>(&modern_update_import) == modern &&
                dlvsym(RTLD_DEFAULT, "cudaGraphExecUpdate", "libcudart.so.11.0") == legacy &&
                dlvsym(RTLD_DEFAULT, "cudaGraphExecUpdate", "libcudart.so.12") == modern &&
                !runtime_graph_update_version(nullptr) &&
                !runtime_graph_update_version("libcudart.so.99"),
            "whole-update ELF imports retain distinct four- and three-argument ABIs");
    cudaGraphExec_t executable{};
    require(cudaGraphInstantiateWithFlags(&executable, kGraph, 0) == cudaSuccess,
            "construct original SDK executable for versioned update validation");
    const auto legacy_before = glimmer_fake_runtime_graph_update_calls(true);
    const auto modern_before = glimmer_fake_runtime_graph_update_calls(false);
    cudaGraphNode_t error_node{};
    cudaGraphExecUpdateResult legacy_result{};
    require(legacy_update_import(executable, kGraph, &error_node, &legacy_result) == cudaSuccess &&
                error_node == reinterpret_cast<cudaGraphNode_t>(0xfa052) &&
                legacy_result == cudaGraphExecUpdateSuccess &&
                glimmer_fake_runtime_graph_update_calls(true) == legacy_before + 1 &&
                glimmer_fake_runtime_graph_update_calls(false) == modern_before,
            "legacy whole update forwards every argument and native output exactly once");
    cudaGraphExecUpdateResultInfo info{};
    require(modern_update_import(executable, reinterpret_cast<cudaGraph_t>(0xfa099), &info) ==
                    cudaErrorApiFailureBase &&
                info.result == cudaGraphExecUpdateSuccess &&
                info.errorNode == reinterpret_cast<cudaGraphNode_t>(0xfa053) &&
                info.errorFromNode == reinterpret_cast<cudaGraphNode_t>(0xfa054) &&
                glimmer_fake_runtime_graph_update_calls(false) == modern_before + 1,
            "SDK-only error preserves all modern output fields without enum conversion");
    require(legacy_update_import(nullptr, nullptr, nullptr, nullptr) == cudaErrorInvalidValue &&
                modern_update_import(nullptr, nullptr, nullptr) == cudaErrorInvalidValue &&
                glimmer_fake_runtime_graph_update_calls(true) == legacy_before + 2 &&
                glimmer_fake_runtime_graph_update_calls(false) == modern_before + 2,
            "both update ABIs preserve original null validation exactly once");
    require(cudaGraphExecDestroy(executable) == cudaSuccess,
            "retire versioned update's original SDK executable");
    for (void* handle : {RTLD_DEFAULT, sdk}) {
        require(dlsym(handle, "cudaGraphExecUpdate") == modern,
                "default and explicit SDK lookup select its default modern update ABI");
        const auto update = reinterpret_cast<decltype(&cudaGraphExecUpdate)>(modern);
        require(cudaGraphInstantiateWithFlags(&executable, kGraph, 0) == cudaSuccess,
                "construct cached SDK graph for whole-update metadata publication");
        CudaNativeGraphExecutable old;
        require(router->graphs().lookup(executable, old) == CUDA_SUCCESS,
                "empty known source starts with prepared metadata");
        const auto before = glimmer_fake_runtime_graph_update_calls(false);
        require(update(executable, kGraph, &info) == cudaSuccess &&
                    glimmer_fake_runtime_graph_update_calls(false) == before + 1 &&
                    router->graphs().is_current(old) == CUDA_ERROR_INVALID_HANDLE,
                "cached SDK success publishes a fresh generation without a Driver callback");
        require(update(executable, reinterpret_cast<cudaGraph_t>(0xfa099), &info) ==
                    cudaErrorApiFailureBase,
                "cached SDK error retains original status");
        CudaNativeGraphExecutable unavailable;
        require(router->graphs().lookup(executable, unavailable) == CUDA_ERROR_NOT_SUPPORTED,
                "native update error does not retain stale replay permission");
        require(update(executable, kGraph, &info) == cudaSuccess &&
                    router->graphs().lookup(executable, unavailable) == CUDA_SUCCESS,
                "successful complete replacement reconstructs known metadata after invalidation");
        const auto child = wrapper<decltype(&cudaGraphExecChildGraphNodeSetParams)>(
            handle, "cudaGraphExecChildGraphNodeSetParams");
        const auto child_before = glimmer_fake_runtime_graph_child_update_calls();
        const auto node = reinterpret_cast<cudaGraphNode_t>(0xfa050);
        require(child(executable, node, kGraph) == cudaSuccess &&
                    child(executable, node, reinterpret_cast<cudaGraph_t>(0xfa099)) ==
                        cudaErrorApiFailureBase &&
                    child(nullptr, nullptr, nullptr) == cudaErrorInvalidValue &&
                    glimmer_fake_runtime_graph_child_update_calls() == child_before + 3,
                "stable child ABI forwards cached originals and preserves errors and nulls once");
        CudaNativeGraphExecutable unknown_child;
        require(router->graphs().lookup(executable, unknown_child) == CUDA_ERROR_NOT_SUPPORTED,
                "unknown child identity cannot reconstruct uncertain parent effects");
        require(cudaGraphExecDestroy(executable) == cudaSuccess,
                "retire cached SDK whole/child mutation slot");
    }
    require(::dlclose(sdk) == 0 && !is_inside_runtime_call() && !is_inside_driver_call(),
            "update roots restore native caller TLS and balance loader ownership");
}

void check_lost_context() {
    const ProcessCallGate::Call call(cooperative_process_call_gate());
    auto* router = cooperative_source_router();
    require(router, "constructor uncertainty has an admitted CPU registry owner");
    unsigned int originals{};
    cudaGraphExec_t output = kSentinel;
    require(
        intercept_runtime_graph_instantiation(
            &output, kGraph, false,
            [&](auto* staged) {
                ++originals;
                *staged = kExecutable;
                require(::setenv("GLIMMER_FAKE_PROGRESS_CONTEXT_ERROR", "1", 1) == 0,
                        "inject failed context observation after original constructor effect");
                return cudaSuccess;
            }) == cudaErrorUnknown &&
            output == kExecutable && originals == 1,
        "post-effect context uncertainty retains and publishes actual handle without issue replay");
    require(::unsetenv("GLIMMER_FAKE_PROGRESS_CONTEXT_ERROR") == 0,
            "restore fixture metadata dispatch without authorizing owner recovery");
    CudaNativeGraphExecutable owned;
    require(router->graphs().lookup(output, owned) == CUDA_ERROR_UNKNOWN,
            "unknown constructor context cannot become replay permission");
    require(intercept_runtime_graph_instantiation(&output, kGraph, false,
                                                  [&](auto*) {
                                                      ++originals;
                                                      return cudaSuccess;
                                                  }) == cudaErrorUnknown &&
                originals == 1,
            "restoring fake native dispatch does not retry or clear retained ownership");
    require(!is_inside_runtime_call() && !is_inside_driver_call(),
            "post-effect ownership error restores original caller TLS");
}

void check_unknown_abi(bool is_preview) {
    void* sdk = dlopen(GLIMMER_UNVERSIONED_RUNTIME_LIBRARY, RTLD_NOW | RTLD_LOCAL);
    require(sdk, "load independent unversioned fake SDK");
    const auto resolve = resolve_real_dlsym();
    const auto native = resolve(sdk, "cudaGraphInstantiate");
    const auto calls =
        reinterpret_cast<unsigned int (*)()>(resolve(sdk, "glimmer_fake_unversioned_graph_calls"));
    require(native && calls && calls() == 0, "unknown ABI has an actual untouched original");
    const auto selected = dlsym(sdk, "cudaGraphInstantiate");
    if (is_preview) {
        require(!selected && calls() == 0, "owned preview rejects unknown ABI without issue");
    } else {
        require(selected == native && dlerror() == nullptr,
                "native passthrough returns exact original pointer without a spurious probe error");
        cudaGraphExec_t output{};
#if CUDART_VERSION >= 12000
        using Constructor =
            cudaError_t(CUDARTAPI*)(cudaGraphExec_t*, cudaGraph_t, unsigned long long);
        const auto result = reinterpret_cast<Constructor>(selected)(&output, kGraph, 0);
#else
        using Constructor = cudaError_t(CUDARTAPI*)(cudaGraphExec_t*, cudaGraph_t, cudaGraphNode_t*,
                                                    char*, std::size_t);
        const auto result =
            reinterpret_cast<Constructor>(selected)(&output, kGraph, nullptr, nullptr, 0);
#endif
        require(result == cudaSuccess && output == kExecutable && calls() == 1,
                "application-known unversioned ABI executes original once outside preview");
    }
    const auto native_update = resolve(sdk, "cudaGraphExecUpdate");
    const auto update_calls = reinterpret_cast<unsigned int (*)()>(
        resolve(sdk, "glimmer_fake_unversioned_graph_update_calls"));
    require(native_update && update_calls && update_calls() == 0,
            "unknown update ABI has an independently untouched SDK original");
    const auto selected_update = dlsym(sdk, "cudaGraphExecUpdate");
    if (is_preview) {
        require(!selected_update && update_calls() == 0,
                "owned preview declines unknown update ABI without original issue");
    } else {
        require(selected_update == native_update && dlerror() == nullptr,
                "native update lookup preserves exact pointer without probe loader errors");
#if CUDART_VERSION >= 12000
        cudaGraphExecUpdateResultInfo info{};
        require(reinterpret_cast<decltype(&cudaGraphExecUpdate)>(selected_update)(
                    kExecutable, kGraph, &info) == cudaSuccess &&
                    info.result == cudaGraphExecUpdateSuccess && update_calls() == 1,
                "application-known modern update ABI executes the unknown original once");
#else
        cudaGraphExecUpdateResult result{};
        require(reinterpret_cast<decltype(&cudaGraphExecUpdate)>(selected_update)(
                    kExecutable, kGraph, nullptr, &result) == cudaSuccess &&
                    result == cudaGraphExecUpdateSuccess && update_calls() == 1,
                "application-known legacy update ABI executes the unknown original once");
#endif
    }
    require(dlclose(sdk) == 0, "balance unknown SDK library reference");
    require(!is_inside_runtime_call() && !is_inside_driver_call(),
            "unknown ABI lookup restores caller TLS without SDK execution permission");
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--unknown-native") {
            check_unknown_abi(false);
        } else if (argc == 2 && std::string_view(argv[1]) == "--unknown-preview") {
            check_unknown_abi(true);
        } else if (argc == 2 && std::string_view(argv[1]) == "--lost-context") {
            check_lost_context();
        } else {
            check_public_routing();
            check_graph_updates();
        }
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
