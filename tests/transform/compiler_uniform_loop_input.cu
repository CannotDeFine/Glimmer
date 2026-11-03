// Compile-only input: runtime loop count prevents unrolling, while both the
// zero-trip decision and loop continuation are uniform across the entire CTA.
extern "C" __global__ __launch_bounds__(32, 2) void compiler_uniform_loop(unsigned int* output,
                                                                          unsigned int iterations) {
    unsigned int sum = 0;
#pragma unroll 1
    for (unsigned int iteration = 0; iteration < iterations; ++iteration) {
        __syncthreads();
        sum += __shfl_xor_sync(~0U, threadIdx.x, 1);
    }
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    atomicAdd(output + index, index + gridDim.x + sum);
}
