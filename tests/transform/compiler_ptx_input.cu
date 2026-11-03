// Compile-only input for actual nvcc line metadata, launch bounds and local data.
// No launch or CUDA initialization occurs in the offline transformation test.
__device__ __noinline__ unsigned int compiler_leaf(unsigned int input) {
    return input + gridDim.x + blockIdx.y + blockIdx.z + gridDim.y + gridDim.z - 2;
}
__device__ __noinline__ unsigned int compiler_middle(unsigned int input) {
    return compiler_leaf(input);
}
extern "C" __global__ __launch_bounds__(32, 2) void compiler_local(unsigned int* output,
                                                                   unsigned int guard) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    volatile unsigned int local[4];
    local[threadIdx.x & 3U] = index;
    const auto value = compiler_middle(local[threadIdx.x & 3U]);
    output[guard + index] += value;
}
