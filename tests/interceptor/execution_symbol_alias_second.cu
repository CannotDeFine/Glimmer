#include <cuda_runtime.h>

__device__ unsigned int alias_second_value = 73;
__global__ void alias_second_kernel(unsigned int* output) {
    *output = 73;
}
// Keep the second translation unit in the executable without executing GPU work.
extern "C" void glimmer_symbol_alias_second() {}
