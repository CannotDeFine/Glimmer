// Compile-only actual nvcc shared-atomic/warp input. The numerical expression is
// index + gridDim.x for a full 32-thread CTA; all participation remains observable.
extern "C" __global__ __launch_bounds__(32, 2) void compiler_warp(unsigned int* output,
                                                                  unsigned int guard) {
    __shared__ unsigned int total;
    if (threadIdx.x == 0) total = 0;
    __syncthreads();
    atomicAdd(&total, threadIdx.x + 1);
    __syncthreads();
    const auto broadcast = __shfl_sync(~0U, threadIdx.x, 0);
    const auto votes = __popc(__ballot_sync(~0U, (threadIdx.x & 1U) == 0));
    const auto matching = __popc(__match_any_sync(~0U, blockIdx.x));
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    atomicAdd(output + guard + index,
              index + gridDim.x + total - 528 + broadcast + votes - 16 + matching - 32);
}
