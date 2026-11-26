#include "execution_cycle_witness_fixture.h"

#include <cuda_runtime.h>

namespace {
template <unsigned int Kind>
__device__ void record_cycles(cycle_witness::Sample* output, std::uint32_t* values,
                              unsigned int active) {
    using namespace cycle_witness;
    if (!active) return;
    const auto global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const auto warp = threadIdx.x / kWarp;
    const auto identity = blockIdx.x * (kThreads / kWarp) + warp;
    const auto group = Kind == 2 ? warp % 2 : Kind;
    const auto rounds = Kind == 2 ? 64U : Kind == 0 ? 4096U : 2048U;
    auto value = kSeed + group * kGroupSeed + (global_thread + 1) * 17U;
    for (unsigned int sample = 0; sample < kSamples; ++sample) {
#pragma unroll 1
        for (unsigned int round = 0; round < rounds; ++round)
            // This computes the checked recurrence, not a clock-only delay.
            asm volatile("mad.lo.u32 %0, %0, 1664525, 1013904223;" : "+r"(value));
        if (threadIdx.x % kWarp == 0) {
            Sample observed;
            asm volatile(
                "mov.u32 %0, %%smid;\n\t"
                "mov.u64 %1, %%globaltimer;\n\t"
                "mov.u64 %2, %%clock64;\n\t"
                "mov.u64 %3, %%globaltimer;\n\t"
                "mov.u32 %4, %%smid;"
                : "=r"(observed.sm_before), "=l"(observed.before_ns), "=l"(observed.cycle),
                  "=l"(observed.after_ns), "=r"(observed.sm_after));
            observed.identity = identity;
            observed.value = value;
            observed.group = group;
            observed.reserved = 0;
            output[identity * kSamples + sample] = observed;
        }
        __syncthreads();
    }
    values[global_thread] = value;
}
}  // namespace

extern "C" __global__ void cycle_low(cycle_witness::Sample* trace, std::uint32_t* values,
                                     unsigned int active) {
    record_cycles<0>(trace, values, active);
}
extern "C" __global__ void cycle_high(cycle_witness::Sample* trace, std::uint32_t* values,
                                      unsigned int active) {
    record_cycles<1>(trace, values, active);
}
extern "C" __global__ void cycle_calibration(cycle_witness::Sample* trace, std::uint32_t* values,
                                             unsigned int active) {
    record_cycles<2>(trace, values, active);
}
