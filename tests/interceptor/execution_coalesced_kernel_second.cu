#include "execution_coalesced_kernel_fixture.cuh"

#if GLIMMER_SYMBOL_RDC
extern __device__ unsigned int symbol_values[12];
__global__ void secondary_symbol_read(unsigned int* output, unsigned int count) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) output[index] += symbol_values[index % 12];
}
#endif
extern "C" void run_secondary_coalesced(unsigned int* output, unsigned int count,
                                        cudaStream_t stream) {
    coalesced_add<3><<<320, 32, 0, stream>>>(output, count);
#if GLIMMER_SYMBOL_RDC
    secondary_symbol_read<<<320, 32, 0, stream>>>(output, count);
#endif
}
extern "C" const void* secondary_coalesced_identity() {
    return reinterpret_cast<const void*>(&coalesced_add<3>);
}
