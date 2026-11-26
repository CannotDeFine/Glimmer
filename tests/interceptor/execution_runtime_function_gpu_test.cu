#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// Ordinary registered Runtime application. Both controls use identical input;
// no Glimmer interface, supplied PTX or application scheduling protocol.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): fixed public kernel ABI.
__global__ void runtime_metadata_kernel(std::uint32_t* output, unsigned int count,
                                        unsigned int seed) {
    if (gridDim.x != 1024 || gridDim.y != 1 || gridDim.z != 1 || blockDim.x != 128 ||
        blockDim.y != 1 || blockDim.z != 1)
        return;
    __shared__ unsigned int block_seed;
    if (threadIdx.x == 0) block_seed = seed ^ blockIdx.x;
    __syncthreads();
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) output[index] = output[index] * 1664525U + 1013904223U + seed + block_seed;
}

namespace {
constexpr unsigned int kBlocks = 1024, kThreads = 128, kGuard = 64;
constexpr std::size_t kValues = std::size_t{kBlocks} * kThreads;
constexpr std::uint32_t kMarker = 0xa9c5e137U;
bool has_failure{};
void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void checked(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        has_failure = true;
        throw std::runtime_error(std::string(operation) + ": " + std::to_string(status));
    }
}
struct Allocation {
    std::uint32_t* value{};
    ~Allocation() {
        if (value && !has_failure && cudaFree(value) != cudaSuccess) has_failure = true;
    }
};
struct Stream {
    cudaStream_t value{};
    ~Stream() {
        if (value && !has_failure && cudaStreamDestroy(value) != cudaSuccess) has_failure = true;
    }
};
using Attributes = std::array<long long, 16>;
Attributes observe(const void* function) {
    struct Guarded {
        std::uint64_t before{0xc9a759e17302bc4dULL};
        cudaFuncAttributes value{};
        std::uint64_t after{0xe583cf6427a190dbULL};
    } output;
    checked(cudaFuncGetAttributes(&output.value, function), "query complete original attributes");
    expect(output.before == 0xc9a759e17302bc4dULL && output.after == 0xe583cf6427a190dbULL,
           "Runtime attribute query preserves outer guards");
    const auto& value = output.value;
    // Reserved bytes and struct padding are intentionally not numerical oracles.
    return {static_cast<long long>(value.sharedSizeBytes),
            static_cast<long long>(value.constSizeBytes),
            static_cast<long long>(value.localSizeBytes),
            value.maxThreadsPerBlock,
            value.numRegs,
            value.ptxVersion,
            value.binaryVersion,
            value.cacheModeCA,
            value.maxDynamicSharedSizeBytes,
            value.preferredShmemCarveout,
            value.clusterDimMustBeSet,
            value.requiredClusterWidth,
            value.requiredClusterHeight,
            value.requiredClusterDepth,
            value.clusterSchedulingPolicyPreference,
            value.nonPortableClusterSizeAllowed};
}
std::uint64_t checksum(const std::vector<std::uint32_t>& values) {
    std::uint64_t hash = 14695981039346656037ULL;
    for (const auto value : values)
        for (unsigned int shift = 0; shift < 32; shift += 8) {
            hash ^= (value >> shift) & 255U;
            hash *= 1099511628211ULL;
        }
    return hash;
}
void run() {
    checked(cudaSetDevice(0), "select original Runtime device");
    const auto* function = reinterpret_cast<const void*>(&runtime_metadata_kernel);
    std::array<Attributes, 8> attributes{};
    attributes[0] = observe(function);  // Genuinely cold before any launch.
    expect(attributes[0][0] >= 4 && attributes[0][3] >= kThreads && attributes[0][6] == 70,
           "original Sm70 kernel metadata, not a private replacement");
    std::array<int, 9> statuses{};
    statuses[0] = cudaFuncSetAttribute(function, cudaFuncAttributeMaxDynamicSharedMemorySize, 0);
    checked(static_cast<cudaError_t>(statuses[0]), "configure original dynamic shared memory");
    statuses[1] =
        cudaFuncSetAttribute(function, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    checked(static_cast<cudaError_t>(statuses[1]), "configure original shared-memory preference");
    attributes[1] = observe(function);
    expect(attributes[1][8] == 0 && attributes[1][9] == 100,
           "complete query observes the two successful public settings");

    Allocation allocation;
    std::vector<std::uint32_t> expected(kValues + std::size_t{2} * kGuard, kMarker);
    for (std::size_t index = 0; index < kValues; ++index)
        expected[index + kGuard] = 0x12345678U + static_cast<std::uint32_t>(index) * 17U;
    checked(cudaMalloc(reinterpret_cast<void**>(&allocation.value),
                       expected.size() * sizeof(expected[0])),
            "allocate complete output and guards");
    checked(cudaMemcpy(allocation.value, expected.data(), expected.size() * sizeof(expected[0]),
                       cudaMemcpyHostToDevice),
            "upload complete non-idempotent initial state");
    checked(cudaDeviceSynchronize(), "order initial upload before nonblocking execution");
    Stream stream;
    checked(cudaStreamCreateWithFlags(&stream.value, cudaStreamNonBlocking),
            "create Runtime stream");
    std::array<std::uint64_t, 2> checksums{};
    for (unsigned int call = 0; call < 2; ++call) {
        auto* output = allocation.value + kGuard;
        auto count = static_cast<unsigned int>(kValues);
        auto seed = 17U + call * 23U;
        std::array<void*, 3> arguments{&output, &count, &seed};
        checked(cudaLaunchKernel(function, dim3{kBlocks}, dim3{kThreads}, arguments.data(), 0,
                                 stream.value),
                "launch exactly one original logical Runtime computation");
        checked(cudaStreamSynchronize(stream.value), "prove exact Runtime logical completion");
        expect(arguments == std::array<void*, 3>{&output, &count, &seed} &&
                   output == allocation.value + kGuard && count == kValues &&
                   seed == 17U + call * 23U,
               "public Runtime launch argument slots and values remain unchanged");
        for (std::size_t index = 0; index < kValues; ++index)
            expected[index + kGuard] = expected[index + kGuard] * 1664525U + 1013904223U + seed +
                                       (seed ^ static_cast<unsigned int>(index / kThreads));
        std::vector<std::uint32_t> actual(expected.size());
        checked(cudaMemcpy(actual.data(), allocation.value, actual.size() * sizeof(actual[0]),
                           cudaMemcpyDeviceToHost),
                "read every output and guard");
        expect(actual == expected, "every original/private result and guard is exactly once");
        checksums[call] = checksum(actual);
        if (call == 0) {
            constexpr std::array<cudaFuncAttribute, 6> selectors{
                cudaFuncAttributeClusterDimMustBeSet,
                cudaFuncAttributeRequiredClusterWidth,
                cudaFuncAttributeRequiredClusterHeight,
                cudaFuncAttributeRequiredClusterDepth,
                cudaFuncAttributeNonPortableClusterSizeAllowed,
                cudaFuncAttributeClusterSchedulingPolicyPreference};
            for (std::size_t index = 0; index < selectors.size(); ++index) {
                statuses[index + 2] = cudaFuncSetAttribute(function, selectors[index], 0);
                attributes[index + 2] = observe(function);
            }
            statuses[8] = cudaFuncSetAttribute(function, static_cast<cudaFuncAttribute>(16), -1);
            expect(statuses[8] == cudaErrorInvalidValue,
                   "unsupported representable Runtime selector is explicit invalid value");
        }
    }
    const auto null_output = cudaFuncGetAttributes(nullptr, function);
    expect(null_output == cudaErrorInvalidValue, "NULL output never reaches native metadata");
    std::cout << "{\"status\":\"passed\",\"attributes\":[";
    for (std::size_t observation = 0; observation < attributes.size(); ++observation) {
        if (observation) std::cout << ',';
        std::cout << '[';
        for (std::size_t field = 0; field < attributes[observation].size(); ++field) {
            if (field) std::cout << ',';
            std::cout << attributes[observation][field];
        }
        std::cout << ']';
    }
    std::cout << "],\"setter_statuses\":[";
    for (std::size_t index = 0; index < statuses.size(); ++index) {
        if (index) std::cout << ',';
        std::cout << statuses[index];
    }
    std::cout << "],\"null_output_status\":" << static_cast<int>(null_output) << ",\"checksums\":["
              << checksums[0] << ',' << checksums[1]
              << "],\"checked_values\":262400,\"logical_calls\":2,\"replays\":0,"
                 "\"full_device_qualified\":false,\"performance_qualified\":false}\n";
}
}  // namespace
int main() {
    try {
        run();
        expect(!has_failure, "original Runtime resource cleanup succeeded");
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        has_failure = true;
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
