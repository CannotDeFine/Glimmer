#include "cuda_gemm_completion_election_fixture.h"
#include "internal/cuda_gemm_block_completion.cuh"

namespace {
using namespace glimmer::test::gemm_completion;
template <bool Observes>
__device__ void execute(Shared* shared, unsigned long long* counter, unsigned int hold_zero) {
    if (blockIdx.x == 0 && hold_zero && threadIdx.x == 0) {
        unsigned int released{};
        // Inline PTX writes released; clang-tidy14 does not model asm outputs.
        // NOLINTNEXTLINE(bugprone-infinite-loop)
        do {
            asm volatile("ld.volatile.global.u32 %0, [%1];"
                         : "=r"(released)
                         : "l"(&shared->release)
                         : "memory");
        } while (!released);
    }
    __syncthreads();
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    shared->output[kGuard + index] = 3 * index + 17;
    __syncthreads();
    glimmer::backend::cuda::GemmBlockCompletion<Observes> completion(&shared->completed, counter);
    const bool first = completion.publish(), second = completion.publish();
    if (threadIdx.x == 0) {
        shared->publishers[blockIdx.x] = static_cast<unsigned int>(first) + second;
        asm volatile("st.release.sys.global.u32 [%0], %1;"
                     :
                     : "l"(&shared->done[blockIdx.x]), "r"(1U)
                     : "memory");
    }
}
}  // namespace

extern "C" __global__ void gemm_completion_observed(glimmer::test::gemm_completion::Shared* shared,
                                                    unsigned long long* counter,
                                                    unsigned int hold_zero) {
    execute<true>(shared, counter, hold_zero);
}
extern "C" __global__ void gemm_completion_unobserved(
    glimmer::test::gemm_completion::Shared* shared, unsigned long long* counter,
    unsigned int hold_zero) {
    execute<false>(shared, counter, hold_zero);
}
