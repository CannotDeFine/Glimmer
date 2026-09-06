#include <cuda.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>

#include <cstdlib>
#include <iostream>
#include <string_view>

namespace {

using DriverLaunch = CUresult (*)(const CUlaunchConfig*, CUfunction, void**, void**);
using RuntimeLaunch = cudaError_t (*)(const cudaLaunchConfig_t*, const void*, void**);
using GetProcAddress = CUresult (*)(const char*, void**, int, cuuint64_t,
                                    CUdriverProcAddressQueryResult*);

bool check_driver(void* handle) {
    CUlaunchAttribute attribute{};
    attribute.id = CU_LAUNCH_ATTRIBUTE_IGNORE;
    CUlaunchConfig config{};
    config.gridDimX = 3;
    config.gridDimY = config.gridDimZ = 1;
    config.blockDimX = 32;
    config.blockDimY = config.blockDimZ = 1;
    config.sharedMemBytes = 64;
    config.attrs = &attribute;
    config.numAttrs = 1;
    CUresult result = CUDA_SUCCESS;
    void* arguments[] = {&config, &attribute, &result, nullptr};
    const auto function = reinterpret_cast<CUfunction>(&config);
    const auto get_proc = reinterpret_cast<GetProcAddress>(dlsym(handle, "cuGetProcAddress_v2"));
    if (get_proc == nullptr) {
        return false;
    }
    bool passed = true;
    for (const char* name : {"cuLaunchKernelEx", "cuLaunchKernelEx_ptsz"}) {
        const auto launch = reinterpret_cast<DriverLaunch>(dlsym(handle, name));
        if (launch == nullptr || dlsym(RTLD_DEFAULT, name) != reinterpret_cast<void*>(launch)) {
            return false;
        }
        for (CUstream stream : {static_cast<CUstream>(nullptr), CU_STREAM_PER_THREAD}) {
            config.hStream = stream;
            result = CUDA_SUCCESS;
            passed &= launch(&config, function, arguments, arguments + 3) == CUDA_SUCCESS;
            result = CUDA_ERROR_LAUNCH_FAILED;
            passed &= launch(&config, function, arguments, arguments + 3) == result;
        }
        passed &= launch(nullptr, function, arguments, arguments + 3) == CUDA_ERROR_INVALID_VALUE;
        passed &= launch(&config, nullptr, arguments, arguments + 3) == CUDA_ERROR_INVALID_VALUE;
    }
    struct Alias {
        const char* name;
        const char* ptds;
    };
    const Alias aliases[] = {{"cuLaunchKernelEx", "cuLaunchKernelEx_ptsz"},
                             {"cuGraphLaunch", "cuGraphLaunch_ptsz"},
                             {"cuMemAllocAsync", "cuMemAllocAsync_ptsz"},
                             {"cuStreamSynchronize", "cuStreamSynchronize_ptsz"}};
    for (const auto& alias : aliases) {
        for (const auto flags :
             {CU_GET_PROC_ADDRESS_DEFAULT, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM}) {
            void* launch = nullptr;
            CUdriverProcAddressQueryResult status{};
            const auto lookup_result = get_proc(alias.name, &launch, CUDA_VERSION, flags, &status);
            passed &= lookup_result == CUDA_SUCCESS;
            const char* expected = flags == CU_GET_PROC_ADDRESS_DEFAULT ? alias.name : alias.ptds;
            const bool matched =
                status == CU_GET_PROC_ADDRESS_SUCCESS && launch == dlsym(handle, expected);
            if (!matched || lookup_result != CUDA_SUCCESS) {
                std::cerr << "proc alias mismatch: " << alias.name << " flags=" << flags << '\n';
            }
            passed &= matched;
        }
    }
    return passed;
}

bool check_runtime(void* handle) {
    cudaLaunchAttribute attribute{};
    attribute.id = cudaLaunchAttributeIgnore;
    cudaLaunchConfig_t config{};
    config.gridDim = dim3{3, 1, 1};
    config.blockDim = dim3{32, 1, 1};
    config.dynamicSmemBytes = 64;
    config.attrs = &attribute;
    config.numAttrs = 1;
    cudaError_t result = cudaSuccess;
    void* arguments[] = {&config, &attribute, &result};
    bool passed = true;
    for (const char* name : {"cudaLaunchKernelExC", "cudaLaunchKernelExC_ptsz"}) {
        const auto launch = reinterpret_cast<RuntimeLaunch>(dlsym(handle, name));
        if (launch == nullptr || dlsym(RTLD_DEFAULT, name) != reinterpret_cast<void*>(launch)) {
            return false;
        }
        for (cudaStream_t stream : {static_cast<cudaStream_t>(nullptr), cudaStreamPerThread}) {
            config.stream = stream;
            result = cudaSuccess;
            passed &= launch(&config, &config, arguments) == cudaSuccess;
            result = cudaErrorLaunchFailure;
            passed &= launch(&config, &config, arguments) == result;
        }
        passed &= launch(nullptr, &config, arguments) == cudaErrorInvalidValue;
        passed &= launch(&config, nullptr, arguments) == cudaErrorInvalidValue;
    }
    return passed;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--missing-runtime") {
        cudaLaunchConfig_t config{};
        for (const char* name : {"cudaLaunchKernelExC", "cudaLaunchKernelExC_ptsz"}) {
            const auto launch = reinterpret_cast<RuntimeLaunch>(dlsym(RTLD_DEFAULT, name));
            if (launch == nullptr || launch(&config, &config, nullptr) != cudaErrorNotSupported) {
                return EXIT_FAILURE;
            }
        }
        return EXIT_SUCCESS;
    }
    void* driver = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
    void* runtime = dlopen("libcudart.so", RTLD_NOW | RTLD_GLOBAL);
    if (driver == nullptr || runtime == nullptr) {
        std::cerr << "fake CUDA libraries could not be loaded\n";
        return EXIT_FAILURE;
    }
    const bool driver_passed = check_driver(driver);
    const bool runtime_passed = check_runtime(runtime);
    if (!driver_passed || !runtime_passed) {
        std::cerr << "extended launch forwarding failed: driver=" << driver_passed
                  << " runtime=" << runtime_passed << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
