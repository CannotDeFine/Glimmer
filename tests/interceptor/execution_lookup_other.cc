#include <cuda.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>

extern "C" CUresult CUDAAPI cuCtxGetCurrent(CUcontext*) {
    return CUDA_ERROR_INVALID_DEVICE;
}
extern "C" int cuExecutionLocalMarker() {
    return 29;
}
extern "C" cudaError_t CUDARTAPI cudaMalloc(void**, std::size_t) {
    return cudaErrorInvalidDevice;
}
extern "C" int cudaExecutionLocalMarker() {
    return 31;
}
extern "C" void* glimmer_execution_local_lookup(const char* name) {
    return ::dlsym(RTLD_DEFAULT, name);
}
extern "C" void* glimmer_execution_next_lookup(const char* name) {
    return ::dlsym(RTLD_NEXT, name);
}
asm(".globl cuExecutionAbsoluteZero\n.set cuExecutionAbsoluteZero, 0\n");
