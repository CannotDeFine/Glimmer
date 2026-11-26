#include <cuda_runtime_api.h>

namespace {
unsigned int calls{};
}
extern "C" unsigned int glimmer_host_function_native_calls() {
    return calls;
}
extern "C" cudaError_t CUDARTAPI cudaLaunchHostFunc(cudaStream_t, cudaHostFn_t, void*) {
    ++calls;
    return cudaErrorNotSupported;
}
extern "C" cudaError_t CUDARTAPI cudaLaunchHostFunc_ptsz(cudaStream_t, cudaHostFn_t, void*) {
    ++calls;
    return cudaErrorNotSupported;
}
