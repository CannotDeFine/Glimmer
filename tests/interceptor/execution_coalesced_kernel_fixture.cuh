#pragma once

#include <cuda_runtime.h>

// Instantiated in two translation units. The linker coalesces the host stub,
// while ordinary compilation still creates two independent compiler images.
template <unsigned int Increment>
__global__ void coalesced_add(unsigned int* output, unsigned int count) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) output[index] += Increment;
}

extern "C" void run_secondary_coalesced(unsigned int*, unsigned int, cudaStream_t);
extern "C" const void* secondary_coalesced_identity();
