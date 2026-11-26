#include "internal/driver_api_interceptor.h"
#include "internal/driver_cooperative_router.h"
#include "internal/runtime_api_bridge.h"

#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <iostream>
#include <stdexcept>
#include <string_view>

extern "C" cudaError_t CUDARTAPI legacy_graph_import(cudaGraphExec_t*, cudaGraph_t,
                                                     cudaGraphNode_t*, char*, std::size_t);
asm(".symver legacy_graph_import,cudaGraphInstantiate@libcudart.so.11.0");
extern "C" cudaError_t CUDARTAPI legacy_update_import(cudaGraphExec_t, cudaGraph_t,
                                                      cudaGraphNode_t*, cudaGraphExecUpdateResult*);
asm(".symver legacy_update_import,cudaGraphExecUpdate@libcudart.so.11.0");

namespace {
using namespace glimmer::interceptor;
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void check(cudaError_t value, const char* message) {
    require(value == cudaSuccess, message);
}
void check(CUresult value, const char* message) {
    require(value == CUDA_SUCCESS, message);
}

void run(bool is_native, bool should_initialize_sdk = true, bool uses_nonprimary_context = false,
         bool updates_graph = false) {
    const ProcessCallGate::Call call(cooperative_process_call_gate());
    DriverDispatch driver;
    require(driver.initialize(), "resolve installed legacy SDK's actual Driver");
    void* sdk = dlopen("libcudart.so.11.0", RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
    require(sdk, "legacy SDK is independently linked, not replaced by modern Runtime");
    const auto resolve = resolve_real_dlsym();
    using Launch = cudaError_t(CUDARTAPI*)(cudaGraphExec_t, cudaStream_t);
    const auto launch = reinterpret_cast<Launch>(resolve(sdk, "cudaGraphLaunch"));
    require(launch &&
                runtime_graph_constructor_lookup(sdk) ==
                    runtime_graph_constructor_version("libcudart.so.11.0") &&
                dlerror() == nullptr,
            "explicit actual SDK lookup identifies exact five-argument ABI");
    require(
        dlsym(RTLD_DEFAULT, "cudaGraphInstantiate") ==
                reinterpret_cast<void*>(&legacy_graph_import) &&
            dlsym(sdk, "cudaGraphInstantiate") == reinterpret_cast<void*>(&legacy_graph_import) &&
            dlerror() == nullptr,
        "actual CUDA 11 default/explicit lookup never returns the compiled CUDA 12 alias");
    {
        const NativeDriverCallScope native;
        check(cudaSetDevice(0), "initialize actual legacy Runtime and cached Driver dispatch");
        if (should_initialize_sdk) {
            const auto initialize = reinterpret_cast<RuntimeFreeFunction>(resolve(sdk, "cudaFree"));
            require(initialize, "resolve original legacy SDK initialization");
            check(initialize(nullptr), "initialize legacy SDK before graph ABI qualification");
        }
    }
    if (uses_nonprimary_context) {
        void* create_address{};
        check(driver.get_proc_address("cuCtxCreate", &create_address, 3020,
                                      CU_GET_PROC_ADDRESS_DEFAULT),
              "resolve exact original non-primary context creation ABI");
        const auto create = reinterpret_cast<decltype(&cuCtxCreate)>(create_address);
        require(create, "non-primary creation has original Driver dispatch");
        CUcontext created{};
        const NativeDriverCallScope native;
        check(create(&created, 0, 0), "create application-owned non-primary context");
        require(created, "application receives its original non-primary context");
    }
    CUcontext context{};
    check(driver.context_get_current(&context), "observe actual legacy caller context");
    require(context, "legacy graph caller is bound by actual SDK");
    bool is_primary_active = true;
    if (!should_initialize_sdk) {
        check(driver.primary_context_is_active(0, is_primary_active),
              "observe original cold primary state without initialization");
        require(!is_primary_active, "cold fixture must not warm original SDK before allocation");
    }
    unsigned int* output{};
    check(cudaMalloc(reinterpret_cast<void**>(&output), sizeof(*output)),
          "allocate actual legacy graph output");
    check(driver.primary_context_is_active(0, is_primary_active),
          "observe actual SDK activation after allocation");
    require(is_primary_active != uses_nonprimary_context,
            "original allocation activates primary only for the primary caller");
    std::size_t free_bytes{};
    std::size_t total_bytes{};
    check(cudaMemGetInfo(&free_bytes, &total_bytes), "observe committed legacy allocation charge");
    require(total_bytes == 8388608 && free_bytes == total_bytes - sizeof(*output),
            "cold successful allocation charges exact bytes, without SDK warmup");
    cudaGraph_t graph{};
    cudaGraphExec_t executable{};
    {
        const NativeDriverCallScope native;
        check(cudaGraphCreate(&graph, 0), "create original legacy graph");
        cudaMemsetParams parameters{};
        parameters.dst = output;
        parameters.value = 19;
        parameters.elementSize = sizeof(*output);
        parameters.width = 1;
        parameters.height = 1;
        cudaGraphNode_t node{};
        check(cudaGraphAddMemsetNode(&node, graph, nullptr, 0, &parameters),
              "construct unchanged original legacy memset node");
    }
    cudaError_t constructed{};
    if (is_native) {
        const NativeDriverCallScope native;
        constructed = legacy_graph_import(&executable, graph, nullptr, nullptr, 0);
    } else {
        constructed = legacy_graph_import(&executable, graph, nullptr, nullptr, 0);
    }
    check(constructed, "actual legacy constructor receives its exact five arguments");
    require(executable, "actual legacy executable output is published");
    DriverCooperativeRouter* router{};
    if (!is_native) {
        router = cooperative_source_router();
        require(router, "public legacy constructor has a CPU owner");
        CudaNativeGraphExecutable registered;
        check(router->graphs().lookup(executable, registered),
              "legacy root registers cached original SDK executable");
        require(registered.context() == context && registered.effects().node_count() == 1,
                "legacy executable retains exact original context and node count");
    }
    if (updates_graph) {
        require(
            runtime_graph_update_lookup(sdk) == runtime_graph_update_version("libcudart.so.11.0") &&
                dlsym(RTLD_DEFAULT, "cudaGraphExecUpdate") ==
                    reinterpret_cast<void*>(&legacy_update_import) &&
                dlsym(sdk, "cudaGraphExecUpdate") ==
                    reinterpret_cast<void*>(&legacy_update_import) &&
                dlerror() == nullptr,
            "actual legacy default and explicit lookup retain four-argument update ABI");
        cudaGraph_t replacement{};
        {
            const NativeDriverCallScope native;
            check(cudaGraphCreate(&replacement, 0), "create actual legacy update graph");
            cudaMemsetParams parameters{};
            parameters.dst = output;
            parameters.value = 23;
            parameters.elementSize = sizeof(*output);
            parameters.width = parameters.height = 1;
            cudaGraphNode_t node{};
            check(cudaGraphAddMemsetNode(&node, replacement, nullptr, 0, &parameters),
                  "build changed original legacy memset parameters");
        }
        CudaNativeGraphExecutable before;
        if (router) check(router->graphs().lookup(executable, before), "retain old legacy plan");
        cudaGraphNode_t error_node{};
        cudaGraphExecUpdateResult result{};
        const auto issue = [&] {
            return legacy_update_import(executable, replacement, &error_node, &result);
        };
        if (is_native) {
            const NativeDriverCallScope native;
            check(issue(), "perform actual native four-argument SDK update");
        } else {
            check(issue(), "perform actual owned four-argument SDK update");
        }
        require(result == cudaGraphExecUpdateSuccess,
                "actual legacy SDK confirms performed compatible update");
        if (router) {
            CudaNativeGraphExecutable changed;
            check(router->graphs().lookup(executable, changed),
                  "legacy cached SDK update publishes fresh prepared metadata");
            require(router->graphs().is_current(before) == CUDA_ERROR_INVALID_HANDLE &&
                        changed.effects().node_count() == 1,
                    "legacy update cannot reuse old executable generation");
        }
        const NativeDriverCallScope native;
        check(cudaGraphDestroy(replacement), "destroy supplied legacy update source before replay");
    }
    {
        const NativeDriverCallScope native;
        check(cudaGraphDestroy(graph), "destroy original source before legacy replay");
    }
    for (unsigned int cycle = 0; cycle < 3; ++cycle) {
        if (is_native) {
            const NativeDriverCallScope native;
            check(launch(executable, nullptr), "issue original legacy SDK replay");
        } else {
            check(execute_owned_runtime_graph(executable, nullptr, false,
                                              [&] { return launch(executable, nullptr); }),
                  "issue admitted original legacy SDK replay");
        }
        check(cudaDeviceSynchronize(), "complete actual legacy graph prefix");
        unsigned int observed{};
        {
            const NativeDriverCallScope native;
            check(cudaMemcpy(&observed, output, sizeof(observed), cudaMemcpyDeviceToHost),
                  "read physically completed original legacy graph output");
        }
        require(observed == (updates_graph ? 23U : 19U),
                "legacy graph executes exact original or updated memset parameters");
    }
    check(cudaGraphExecDestroy(executable), "retire original legacy executable");
    if (router) {
        CudaNativeGraphExecutable retired;
        require(router->graphs().lookup(executable, retired) == CUDA_ERROR_INVALID_HANDLE,
                "legacy destruction retires exact CPU executable identity");
        check(router->close_context(context, [] { return CUDA_SUCCESS; }),
              "close private legacy owner before final original cleanup");
    }
    check(cudaFree(output), "release actual legacy output and accounting");
    check(cudaMemGetInfo(&free_bytes, &total_bytes), "observe final restored legacy quota");
    require(total_bytes == 8388608 && free_bytes == total_bytes,
            "legacy release and checked owner cleanup restore complete quota");
    require(cudaPeekAtLastError() == cudaSuccess && cudaGetLastError() == cudaSuccess &&
                !is_inside_runtime_call() && !is_inside_driver_call(),
            "legacy constructor/replay preserve caller last-error and TLS");
    CUcontext after{};
    check(driver.context_get_current(&after), "observe final live legacy caller binding");
    require(after == context, "legacy replay preserves original caller context");
    {
        const NativeDriverCallScope native;
        if (uses_nonprimary_context) {
            check(driver.context_destroy(context), "destroy application-owned non-primary context");
        } else {
            check(cudaDeviceReset(), "checked original legacy SDK final cleanup");
        }
    }
    require(dlclose(sdk) == 0, "balance independently loaded legacy SDK reference");
    std::cout << "legacy_runtime_graph native=" << is_native
              << " initialized_sdk=" << should_initialize_sdk
              << " nonprimary=" << uses_nonprimary_context << " graph_update=" << updates_graph
              << " abi_args=5 update_abi_args=4 nodes=1 replays=3 output="
              << (updates_graph ? 23 : 19) << " last=0 checked_cleanup=1\n";
}
}  // namespace

int main(int argc, char** argv) {
    try {
        require(argc == 2 && (std::string_view(argv[1]) == "--native" ||
                              std::string_view(argv[1]) == "--owned" ||
                              std::string_view(argv[1]) == "--cold-capacity" ||
                              std::string_view(argv[1]) == "--cold-owned" ||
                              std::string_view(argv[1]) == "--update-native" ||
                              std::string_view(argv[1]) == "--update-owned" ||
                              std::string_view(argv[1]) == "--nonprimary-capacity"),
                "select independent native/owned legacy SDK process");
        const std::string_view mode(argv[1]);
        run(mode != "--owned" && mode != "--cold-owned" && mode != "--update-owned",
            mode == "--native" || mode == "--owned" || mode.starts_with("--update-"),
            mode == "--nonprimary-capacity", mode.starts_with("--update-"));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
