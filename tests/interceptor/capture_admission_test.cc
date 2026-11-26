#include <cuda.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>

#include <cstdlib>
#include <iostream>
#include <string_view>

namespace {

struct LaunchCase {
    bool ptds;
    CUstream stream;
    bool rejected;
};

bool check_launches(void* driver, void* runtime, const LaunchCase& test) {
    const auto check_status = [](auto actual, auto expected, std::string_view api) {
        if (actual == expected) {
            return true;
        }
        std::cerr << "capture admission regression: " << api << " result=" << actual
                  << " expected=" << expected << '\n';
        return false;
    };
    using DriverKernel = decltype(&cuLaunchKernel);
    using DriverExtended = decltype(&cuLaunchKernelEx);
    using DriverGraph = decltype(&cuGraphLaunch);
    using RuntimeKernel = decltype(&cudaLaunchKernel);
    using RuntimeExtended = decltype(&cudaLaunchKernelExC);
    using RuntimeGraph = decltype(&cudaGraphLaunch);
    using CompilerKernel =
        cudaError_t (*)(cudaKernel_t, dim3, dim3, void**, std::size_t, cudaStream_t);
    const auto kernel = reinterpret_cast<DriverKernel>(
        dlsym(driver, test.ptds ? "cuLaunchKernel_ptsz" : "cuLaunchKernel"));
    const auto extended = reinterpret_cast<DriverExtended>(
        dlsym(driver, test.ptds ? "cuLaunchKernelEx_ptsz" : "cuLaunchKernelEx"));
    const auto graph = reinterpret_cast<DriverGraph>(
        dlsym(driver, test.ptds ? "cuGraphLaunch_ptsz" : "cuGraphLaunch"));
    if (kernel == nullptr || extended == nullptr || graph == nullptr) {
        return false;
    }

    CUlaunchAttribute attribute{};
    CUlaunchConfig config{};
    config.gridDimX = 3;
    config.gridDimY = config.gridDimZ = 1;
    config.blockDimX = 32;
    config.blockDimY = config.blockDimZ = 1;
    config.sharedMemBytes = 64;
    config.hStream = test.stream;
    config.attrs = &attribute;
    config.numAttrs = 1;
    const auto function = reinterpret_cast<CUfunction>(&config);
    CUresult result = CUDA_SUCCESS;
    void* arguments[] = {&config, &attribute, &result, nullptr};
    const CUresult expected = test.rejected ? CUDA_ERROR_NOT_SUPPORTED : CUDA_SUCCESS;
    bool passed =
        check_status(kernel(function, 3, 1, 1, 32, 1, 1, 64, test.stream, arguments, nullptr),
                     expected, "Driver kernel");
    passed &= check_status(extended(&config, function, arguments, arguments + 3), expected,
                           "Driver extended");
    passed &= check_status(graph(reinterpret_cast<CUgraphExec>(&config), test.stream), expected,
                           "Driver graph");
    result = CUDA_ERROR_LAUNCH_FAILED;
    passed &=
        check_status(extended(&config, function, arguments, arguments + 3),
                     test.rejected ? CUDA_ERROR_NOT_SUPPORTED : result, "Driver native error");

    if (runtime == nullptr) {
        return passed;
    }
    const auto runtime_kernel = reinterpret_cast<RuntimeKernel>(
        dlsym(runtime, test.ptds ? "cudaLaunchKernel_ptsz" : "cudaLaunchKernel"));
    const auto runtime_extended = reinterpret_cast<RuntimeExtended>(
        dlsym(runtime, test.ptds ? "cudaLaunchKernelExC_ptsz" : "cudaLaunchKernelExC"));
    const auto runtime_graph = reinterpret_cast<RuntimeGraph>(
        dlsym(runtime, test.ptds ? "cudaGraphLaunch_ptsz" : "cudaGraphLaunch"));
    const auto compiler_kernel = reinterpret_cast<CompilerKernel>(
        dlsym(runtime, test.ptds ? "__cudaLaunchKernel_ptsz" : "__cudaLaunchKernel"));
    if (runtime_kernel == nullptr || runtime_extended == nullptr || runtime_graph == nullptr ||
        compiler_kernel == nullptr) {
        return false;
    }

    cudaLaunchAttribute runtime_attribute{};
    cudaLaunchConfig_t runtime_config{};
    runtime_config.gridDim = dim3{3};
    runtime_config.blockDim = dim3{32};
    runtime_config.dynamicSmemBytes = 64;
    runtime_config.stream = reinterpret_cast<cudaStream_t>(test.stream);
    runtime_config.attrs = &runtime_attribute;
    runtime_config.numAttrs = 1;
    cudaError_t runtime_result = cudaSuccess;
    void* runtime_arguments[] = {&runtime_config, &runtime_attribute, &runtime_result};
    const cudaError_t runtime_expected = test.rejected ? cudaErrorNotSupported : cudaSuccess;
    passed &= check_status(runtime_kernel(&runtime_config, dim3{3}, dim3{32}, runtime_arguments, 64,
                                          runtime_config.stream),
                           runtime_expected, "Runtime kernel");
    passed &= check_status(runtime_extended(&runtime_config, &runtime_config, runtime_arguments),
                           runtime_expected, "Runtime extended");
    passed &= check_status(compiler_kernel(reinterpret_cast<cudaKernel_t>(&runtime_config), dim3{3},
                                           dim3{32}, runtime_arguments, 64, runtime_config.stream),
                           runtime_expected, "Runtime compiler");
    passed &= check_status(
        runtime_graph(reinterpret_cast<cudaGraphExec_t>(&runtime_config), runtime_config.stream),
        runtime_expected, "Runtime graph");
    runtime_result = cudaErrorLaunchFailure;
    passed &= check_status(runtime_extended(&runtime_config, &runtime_config, runtime_arguments),
                           test.rejected ? cudaErrorNotSupported : runtime_result,
                           "Runtime native error");
    return passed;
}

}  // namespace

int main(int argc, char** argv) {
    const bool load_interceptor = argc >= 2 && std::string_view(argv[1]) == "--load-interceptor";
    const bool missing_capture =
        (argc == 2 && std::string_view(argv[1]) == "--missing-runtime-capture") ||
        (argc == 3 && load_interceptor && std::string_view(argv[2]) == "--missing-runtime-capture");
    if (load_interceptor && missing_capture) {
        std::cout << "Runtime capture sanitizer gate unavailable: late loading cannot resolve "
                     "native Runtime launches\n";
        return 77;
    }
    // ASan must initialize before symbol interception. This mode exercises the
    // Driver wrappers after startup, not Runtime/preload-loader interoperability.
    void* interceptor = load_interceptor
                            ? dlopen(GLIMMER_CAPTURE_INTERCEPTOR_PATH, RTLD_NOW | RTLD_GLOBAL)
                            : nullptr;
    if (load_interceptor && interceptor == nullptr) {
        return EXIT_FAILURE;
    }
    void* driver = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
    void* runtime =
        load_interceptor
            ? nullptr
            : dlopen(missing_capture ? GLIMMER_FAKE_MISSING_CAPTURE_RUNTIME_PATH : "libcudart.so",
                     RTLD_NOW | RTLD_GLOBAL);
    if (driver == nullptr || (!load_interceptor && runtime == nullptr)) {
        return EXIT_FAILURE;
    }
    void* driver_launches = load_interceptor ? interceptor : driver;
    void* runtime_launches = runtime;
    const auto set_status =
        reinterpret_cast<void (*)(int)>(dlsym(driver, "glimmer_fake_set_capture_status"));
    const auto events =
        reinterpret_cast<unsigned int (*)()>(dlsym(driver, "glimmer_fake_internal_event_count"));
    const auto queries = reinterpret_cast<unsigned int (*)(unsigned int)>(
        dlsym(driver, "glimmer_fake_capture_query_count"));
    if (set_status == nullptr || events == nullptr || queries == nullptr) {
        return EXIT_FAILURE;
    }
    int stream_token = 0;
    if (missing_capture) {
        set_status(-1);
        for (const bool ptds : {false, true}) {
            if (!check_launches(driver_launches, runtime_launches, {ptds, nullptr, true}) ||
                events() != 0) {
                std::cerr << "capture admission regression: missing Runtime capture symbols\n";
                return EXIT_FAILURE;
            }
        }
        return EXIT_SUCCESS;
    }
    for (const int status : {static_cast<int>(CU_STREAM_CAPTURE_STATUS_ACTIVE),
                             static_cast<int>(CU_STREAM_CAPTURE_STATUS_INVALIDATED), -1, 3}) {
        set_status(status);
        for (const bool ptds : {false, true}) {
            for (const CUstream stream :
                 {static_cast<CUstream>(nullptr), reinterpret_cast<CUstream>(&stream_token)}) {
                const auto other_queries = queries(ptds ? 0U : 1U);
                const auto matching_queries = queries(ptds ? 1U : 0U);
                if (!check_launches(driver_launches, runtime_launches,
                                    {ptds, stream, status < 0 || status == 3}) ||
                    events() != 0 || queries(ptds ? 0U : 1U) != other_queries ||
                    queries(ptds ? 1U : 0U) <= matching_queries) {
                    std::cerr << "capture admission regression: status=" << status
                              << " ptds=" << ptds << " events=" << events()
                              << " matching_queries=" << matching_queries << "->"
                              << queries(ptds ? 1U : 0U) << " other_queries=" << other_queries
                              << "->" << queries(ptds ? 0U : 1U) << '\n';
                    return EXIT_FAILURE;
                }
            }
        }
    }
    // Graph replay, unlike capture construction, must acquire a lease and event.
    set_status(CU_STREAM_CAPTURE_STATUS_NONE);
    const auto graph =
        reinterpret_cast<decltype(&cuGraphLaunch)>(dlsym(driver_launches, "cuGraphLaunch"));
    if (graph == nullptr ||
        graph(reinterpret_cast<CUgraphExec>(&stream_token), nullptr) != CUDA_SUCCESS ||
        events() != 1) {
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
