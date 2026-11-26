#include <cstdint>

struct alignas(16) ExecutionBinaryPair {
    std::uint32_t add{}, marker{}, unused[2]{};
};
extern "C" __global__ void binary_add(std::uint32_t* output, std::uint32_t count,
                                      ExecutionBinaryPair values) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] += values.add;
        if (threadIdx.x == 0)
            for (auto remaining = values.marker; remaining != 0; --remaining) {
                const auto observed = *static_cast<volatile std::uint32_t*>(output + index);
                (void)observed;
            }
    }
}
extern "C" __global__ void binary_no_parameters() {}
extern "C" __global__ void binary_high(std::uint32_t* output, unsigned long long duration) {
    const auto start = clock64();
    while (clock64() - start < duration) {
    }
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    output[index] += 1;
}
