#include <cuda.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string_view>

namespace {

constexpr std::uint32_t kExpectedValue = 42;

constexpr char kWriteValuePtx[] = R"ptx(
.version 7.0
.target sm_50
.address_size 64

.visible .entry glimmer_write_value(
    .param .u64 output
)
{
    .reg .u64 %rd1;
    .reg .u32 %r1;

    ld.param.u64 %rd1, [output];
    mov.u32 %r1, 42;
    st.global.u32 [%rd1], %r1;
    ret;
}
)ptx";

bool check(CUresult result, std::string_view operation) {
    if (result == CUDA_SUCCESS) {
        return true;
    }

    std::cerr << operation << " failed with CUresult " << static_cast<int>(result) << '\n';
    return false;
}

}  // namespace

int main() {
    if (!check(cuInit(0), "cuInit")) {
        return EXIT_FAILURE;
    }

    CUdevice device{};
    if (!check(cuDeviceGet(&device, 0), "cuDeviceGet")) {
        return EXIT_FAILURE;
    }

    CUcontext context{};
    if (!check(cuCtxCreate(&context, nullptr, 0, device), "cuCtxCreate")) {
        return EXIT_FAILURE;
    }

    CUmodule module{};
    if (!check(cuModuleLoadData(&module, kWriteValuePtx), "cuModuleLoadData")) {
        check(cuCtxDestroy(context), "cuCtxDestroy");
        return EXIT_FAILURE;
    }

    CUfunction function{};
    if (!check(cuModuleGetFunction(&function, module, "glimmer_write_value"),
               "cuModuleGetFunction")) {
        check(cuModuleUnload(module), "cuModuleUnload");
        check(cuCtxDestroy(context), "cuCtxDestroy");
        return EXIT_FAILURE;
    }

    CUdeviceptr output{};
    if (!check(cuMemAlloc_v2(&output, sizeof(kExpectedValue)), "cuMemAlloc_v2")) {
        check(cuModuleUnload(module), "cuModuleUnload");
        check(cuCtxDestroy(context), "cuCtxDestroy");
        return EXIT_FAILURE;
    }

    void* kernel_parameters[] = {&output};
    bool workload_succeeded =
        check(cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, nullptr, kernel_parameters, nullptr),
              "cuLaunchKernel");
    workload_succeeded = check(cuCtxSynchronize(), "cuCtxSynchronize") && workload_succeeded;

    CUlaunchAttribute attribute{};
    attribute.id = CU_LAUNCH_ATTRIBUTE_IGNORE;
    CUlaunchConfig config{};
    config.gridDimX = config.gridDimY = config.gridDimZ = 1;
    config.blockDimX = config.blockDimY = config.blockDimZ = 1;
    config.attrs = &attribute;
    config.numAttrs = 1;
    using ExtendedLaunch = CUresult (*)(const CUlaunchConfig*, CUfunction, void**, void**);
    for (const auto flags :
         {CU_GET_PROC_ADDRESS_DEFAULT, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM}) {
        void* address = nullptr;
        workload_succeeded =
            check(cuGetProcAddress("cuLaunchKernelEx", &address, 11060, flags, nullptr),
                  "get extended launch") &&
            workload_succeeded;
        if (address != nullptr) {
            const auto launch = reinterpret_cast<ExtendedLaunch>(address);
            workload_succeeded =
                check(cuMemsetD32(output, 0, 1), "clear extended output") &&
                check(launch(&config, function, kernel_parameters, nullptr), "extended launch") &&
                check(cuCtxSynchronize(), "extended synchronize") && workload_succeeded;
        } else {
            workload_succeeded = false;
        }
    }

    std::uint32_t result = 0;
    workload_succeeded =
        check(cuMemcpyDtoH_v2(&result, output, sizeof(result)), "cuMemcpyDtoH_v2") &&
        workload_succeeded;
    if (result != kExpectedValue) {
        std::cerr << "kernel returned " << result << ", expected " << kExpectedValue << '\n';
        workload_succeeded = false;
    }

    const bool memory_released = check(cuMemFree_v2(output), "cuMemFree_v2");
    const bool module_unloaded = check(cuModuleUnload(module), "cuModuleUnload");
    const bool context_destroyed = check(cuCtxDestroy(context), "cuCtxDestroy");
    return workload_succeeded && memory_released && module_unloaded && context_destroyed
               ? EXIT_SUCCESS
               : EXIT_FAILURE;
}
