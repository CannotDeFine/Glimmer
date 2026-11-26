#include "internal/cuda_cooperative_pool.h"
#include "internal/driver_api_interceptor.h"

#include <cuda.h>

#include <cstdlib>
#include <iostream>

namespace {
using namespace glimmer::interceptor;
bool memory_is(std::size_t expected_free) {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    return cuMemGetInfo(&free_bytes, &total_bytes) == CUDA_SUCCESS && total_bytes == 4096 &&
           free_bytes == expected_free;
}
bool run() {
    CUdevice device = 0;
    CUcontext context = nullptr;
    if (cuInit(0) != CUDA_SUCCESS || cuDeviceGet(&device, 0) != CUDA_SUCCESS ||
        cuCtxCreate(&context, 0, device) != CUDA_SUCCESS) {
        return false;
    }
    CUdeviceptr progress = 0;
    CUdeviceptr application = 0;
    bool passed = memory_is(4096) &&
                  allocate_cooperative_progress(&progress, 1024) == CUDA_SUCCESS && progress != 0 &&
                  memory_is(3072);
    CUdeviceptr rejected = 0;
    passed = passed && cuMemAlloc(&rejected, 4096) == CUDA_ERROR_OUT_OF_MEMORY && rejected == 0 &&
             cuMemAlloc(&application, 3072) == CUDA_SUCCESS && memory_is(0);
    if (progress) {
        const auto released = release_cooperative_progress(progress);
        passed = passed && released.status == CUDA_SUCCESS &&
                 released.ownership == CudaNativeReleaseState::kReleased && memory_is(1024);
    }
    if (application) {
        const auto released = cuMemFree(application);
        passed = passed && released == CUDA_SUCCESS && memory_is(4096);
    }
    const auto destroyed = cuCtxDestroy(context);
    return passed && destroyed == CUDA_SUCCESS;
}
}  // namespace

int main() {
    const bool passed = run();
    if (!passed) {
        std::cerr << "private progress and application quota must share one ledger\n";
    }
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
