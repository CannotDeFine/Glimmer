#include <cuda.h>
#include <cuda_runtime_api.h>

#include <iostream>
#include <stdexcept>

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
}  // namespace
int main() {
    try {
        require(cuInit(0) == CUDA_SUCCESS, "Driver initialization");
        CUdevice device{};
        CUcontext context{};
        require(cuDeviceGet(&device, 0) == CUDA_SUCCESS &&
                    cuCtxCreate(&context, 0, device) == CUDA_SUCCESS,
                "original context");
        std::size_t initial{}, observed{};
        require(cuCtxGetLimit(&initial, CU_LIMIT_STACK_SIZE) == CUDA_SUCCESS &&
                    cudaDeviceGetLimit(&observed, cudaLimitStackSize) == cudaSuccess &&
                    initial == observed,
                "Driver/Runtime initial stack agreement");
        require(cuCtxSetLimit(CU_LIMIT_STACK_SIZE, 8192) == CUDA_SUCCESS &&
                    cudaDeviceGetLimit(&observed, cudaLimitStackSize) == cudaSuccess &&
                    observed >= 8192,
                "original rounded stack setting");
        require(cudaDeviceSetLimit(cudaLimitStackSize, initial) == cudaSuccess &&
                    cuCtxGetLimit(&observed, CU_LIMIT_STACK_SIZE) == CUDA_SUCCESS &&
                    observed == initial,
                "restore context stack");
        CUfunc_cache original_cache{}, driver_cache{};
        cudaFuncCache runtime_cache{};
        require(cuCtxGetCacheConfig(&original_cache) == CUDA_SUCCESS &&
                    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1) == cudaSuccess &&
                    cuCtxGetCacheConfig(&driver_cache) == CUDA_SUCCESS &&
                    cudaDeviceGetCacheConfig(&runtime_cache) == cudaSuccess &&
                    static_cast<unsigned int>(driver_cache) ==
                        static_cast<unsigned int>(runtime_cache) &&
                    cuCtxSetCacheConfig(original_cache) == CUDA_SUCCESS,
                "shared cache policy");
        unsigned int version{};
        int least{}, greatest{};
        require(cuCtxGetApiVersion(context, &version) == CUDA_SUCCESS && version > 0 &&
                    cuCtxGetStreamPriorityRange(&least, &greatest) == CUDA_SUCCESS &&
                    greatest <= least,
                "actual context capabilities");
        CUexecAffinityParam affinity{};
        const auto affinity_status =
            cuCtxGetExecAffinity(&affinity, CU_EXEC_AFFINITY_TYPE_SM_COUNT);
        const auto driver_reset = cuCtxResetPersistingL2Cache();
        const auto runtime_reset = cudaCtxResetPersistingL2Cache();
        // Architecture-dependent original outcomes are reported for matched
        // comparison, not rewritten as success or full-device participation.
        std::cout << "context_api=" << version << " priority=" << least << ',' << greatest
                  << " affinity_status=" << affinity_status << " affinity_sm="
                  << (affinity_status == CUDA_SUCCESS ? affinity.param.smCount.val : 0)
                  << " driver_reset=" << driver_reset << " runtime_reset=" << runtime_reset << '\n';
        require(cuCtxDestroy(context) == CUDA_SUCCESS, "original context cleanup");
        std::cout << "context_controls_completed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
