#include <cuda_runtime.h>
#include "execution_coalesced_kernel_fixture.cuh"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// Ordinary CUDA application: no Glimmer API, IPC, injected adapter or oracle
// relaxation. Compile the same source with and without relocatable device code.
__device__ unsigned int symbol_values[12] = {1, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
__constant__ unsigned int symbol_coefficients[4] = {2, 4, 6, 8};
__device__ unsigned int symbol_signal;

__global__ void symbol_compute(unsigned int* output, unsigned int count) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) output[index] = symbol_values[index % 12] * symbol_coefficients[index % 4];
}
__global__ void symbol_wait(unsigned int* output) {
    if (threadIdx.x != 0) return;
    while (atomicAdd(&symbol_signal, 0U) == 0) {
    }
    *output = 1;
}
namespace {
void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " +
                                 std::to_string(static_cast<int>(status)));
}
void require(bool valid, const char* message) {
    if (!valid) throw std::runtime_error(message);
}
void exercise() {
    require(secondary_coalesced_identity() == reinterpret_cast<const void*>(&coalesced_add<3>),
            "two compiler translation units share one coalesced host stub");
    constexpr unsigned int count = 4097, guards = 32, marker = 0xfeed1234;
    std::array<unsigned int, 12> values{1, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37}, observed{};
    std::array<unsigned int, 4> coefficients{2, 4, 6, 8}, observed_coefficients{};
    std::size_t bytes{};
    void* address{};
    check(cudaGetSymbolSize(&bytes, symbol_values), "symbol size");
    check(cudaGetSymbolAddress(&address, symbol_values), "symbol address");
    require(bytes == sizeof(values) && address, "original symbol metadata");
    check(cudaMemcpyFromSymbol(observed.data(), symbol_values, sizeof(observed)),
          "initial global values");
    check(cudaMemcpyFromSymbol(observed_coefficients.data(), symbol_coefficients,
                               sizeof(observed_coefficients)),
          "initial constant values");
    require(values == observed && coefficients == observed_coefficients,
            "original module initializers");
    check(cudaMemcpy(observed.data(), address, sizeof(observed), cudaMemcpyDefault),
          "borrowed global address");
    require(values == observed, "symbol address observes the same initialized storage");
    const std::array<unsigned int, 3> patch{41, 43, 47};
    check(cudaMemcpyToSymbol(symbol_values, patch.data(), sizeof(patch), 4 * sizeof(unsigned int)),
          "partial global update");
    for (unsigned int index = 0; index < patch.size(); ++index) values[4 + index] = patch[index];
    unsigned int* output{};
    unsigned int* temporary{};
    check(cudaMalloc(reinterpret_cast<void**>(&output), (count + guards) * sizeof(unsigned int)),
          "output allocation");
    check(cudaMalloc(reinterpret_cast<void**>(&temporary), sizeof(values)), "temporary allocation");
    check(cudaMemcpy(temporary, values.data(), sizeof(values), cudaMemcpyHostToDevice),
          "device symbol source values");
    cudaStream_t stream{};
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream creation");
    const unsigned int coefficient = 9;
    coefficients[1] = coefficient;
    check(cudaMemcpyToSymbolAsync(symbol_coefficients, &coefficient, sizeof(coefficient),
                                  sizeof(unsigned int), cudaMemcpyHostToDevice, stream),
          "async constant update");
    check(cudaMemcpyToSymbolAsync(symbol_values, temporary, sizeof(values), 0,
                                  cudaMemcpyDeviceToDevice, stream),
          "device symbol source");
    check(cudaMemcpyFromSymbolAsync(temporary, symbol_values, sizeof(values), 0,
                                    cudaMemcpyDeviceToDevice, stream),
          "device symbol destination");
    check(cudaMemcpyToSymbolAsync(symbol_values, temporary, sizeof(values), 0, cudaMemcpyDefault,
                                  stream),
          "default-direction device symbol copy");
    std::vector<unsigned int> result(count + guards, marker);
    check(cudaMemcpy(output, result.data(), result.size() * sizeof(result[0]),
                     cudaMemcpyHostToDevice),
          "output guards");
    symbol_compute<<<320, 32, 0, stream>>>(output, count);
    check(cudaGetLastError(), "symbol kernel launch");
    check(cudaStreamSynchronize(stream), "symbol computation completion");
    check(cudaMemcpy(result.data(), output, result.size() * sizeof(result[0]),
                     cudaMemcpyDeviceToHost),
          "complete output");
    for (unsigned int index = 0; index < result.size(); ++index)
        require(result[index] ==
                    (index < count ? values[index % 12] * coefficients[index % 4] : marker),
                "complete symbol dataflow and padding oracle");
    coalesced_add<3><<<320, 32, 0, stream>>>(output, count);
    check(cudaGetLastError(), "first coalesced launch");
    run_secondary_coalesced(output, count, stream);
    check(cudaGetLastError(), "second translation-unit launch");
    check(cudaStreamSynchronize(stream), "coalesced computation completion");
    check(cudaMemcpy(result.data(), output, result.size() * sizeof(result[0]),
                     cudaMemcpyDeviceToHost),
          "coalesced complete output");
    for (unsigned int index = 0; index < result.size(); ++index) {
        const auto extra = GLIMMER_SYMBOL_RDC ? values[index % 12] : 0U;
        require(
            result[index] ==
                (index < count ? values[index % 12] * coefficients[index % 4] + 6 + extra : marker),
            "coalesced kernel and linked shared-storage oracle");
    }
    check(cudaMemcpyFromSymbolAsync(observed.data(), symbol_values, sizeof(observed), 0,
                                    cudaMemcpyDeviceToHost, stream),
          "async global readback");
    check(cudaMemcpyFromSymbolAsync(observed_coefficients.data(), symbol_coefficients,
                                    sizeof(observed_coefficients), 0, cudaMemcpyDefault, stream),
          "async constant readback");
    check(cudaStreamSynchronize(stream), "readback completion");
    require(values == observed && coefficients == observed_coefficients,
            "global and constant storage remain unchanged by compute");
    cudaStream_t signal_stream{};
    unsigned int* signal{};
    check(cudaStreamCreateWithFlags(&signal_stream, cudaStreamNonBlocking),
          "signal stream creation");
    check(cudaMallocHost(reinterpret_cast<void**>(&signal), sizeof(*signal)),
          "pinned signal allocation");
    *signal = 1;
    // This symbol has never been queried. Its first metadata lookup must not
    // drain an already-running kernel that depends on the forthcoming copy.
    symbol_wait<<<1, 32, 0, stream>>>(temporary);
    check(cudaGetLastError(), "symbol waiter launch");
    require(cudaStreamQuery(stream) == cudaErrorNotReady,
            "actual symbol waiter is incomplete before its producer");
    (void)cudaGetLastError();
    check(cudaMemcpyToSymbolAsync(symbol_signal, signal, sizeof(*signal), 0, cudaMemcpyHostToDevice,
                                  signal_stream),
          "first symbol lookup and independent asynchronous producer");
    check(cudaStreamSynchronize(signal_stream), "symbol producer completion");
    check(cudaStreamSynchronize(stream), "symbol waiter completion");
    unsigned int signaled{};
    check(cudaMemcpy(&signaled, temporary, sizeof(signaled), cudaMemcpyDeviceToHost),
          "waiter result");
    require(signaled == 1, "symbol producer releases its exact GPU consumer");
    check(cudaFreeHost(signal), "pinned signal cleanup");
    check(cudaStreamDestroy(signal_stream), "signal stream cleanup");
    check(cudaStreamDestroy(stream), "stream cleanup");
    check(cudaFree(temporary), "temporary cleanup");
    check(cudaFree(output), "output cleanup");
    check(cudaDeviceReset(), "context cleanup");
    std::cout << "Ordinary CUDA symbols verified: outputs=" << result.size()
              << "; global_values=" << values.size() << "; constants=" << coefficients.size()
              << "; workers=320; full_device_qualified=false; signal_dependency=1; "
                 "coalesced_launches=2; shared_rdc_state="
              << GLIMMER_SYMBOL_RDC << '\n';
}
}  // namespace
int main() {
    try {
        exercise();
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
