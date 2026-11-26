#include <cuda.h>
#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// Ordinary SDK application: no framework headers, supplied PTX or scheduling API.
__global__ void lifecycle_affine(std::uint32_t* output, unsigned int seed) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    output[index] = output[index] * 1664525U + 1013904223U + seed + index;
}

namespace {
constexpr unsigned int kBlocks = 1024, kThreads = 128, kGuard = 64;
constexpr std::size_t kValues = std::size_t{kBlocks} * kThreads;
constexpr std::uint32_t kMarker = 0x91ae73c5;
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void checked(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": Runtime " + std::to_string(status));
}
void checked(CUresult status, const char* operation) {
    if (status != CUDA_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": Driver " + std::to_string(status));
}
void next_phase() {
    std::string line;
    expect(bool(std::getline(std::cin, line)) && line == "run",
           "bounded original application phase input");
}
std::uint64_t checksum(const std::vector<std::uint32_t>& values) {
    std::uint64_t result = 14695981039346656037ULL;
    for (const auto value : values)
        for (unsigned int shift = 0; shift < 32; shift += 8) {
            result ^= (value >> shift) & 255U;
            result *= 1099511628211ULL;
        }
    return result;
}
struct OriginalInput {
    std::vector<std::uint32_t> expected;
    std::uint32_t* output{};
    cudaStream_t stream{};
    explicit OriginalInput(unsigned int generation)
        : expected(kValues + std::size_t{2} * kGuard, kMarker) {
        for (std::size_t index = 0; index < kValues; ++index)
            expected[index + kGuard] =
                0x12345678U + static_cast<std::uint32_t>(index) * 17U + generation;
        checked(
            cudaMalloc(reinterpret_cast<void**>(&output), expected.size() * sizeof(expected[0])),
            "allocate complete state");
        checked(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                "create original stream");
        checked(cudaMemcpy(output, expected.data(), expected.size() * sizeof(expected[0]),
                           cudaMemcpyHostToDevice),
                "upload all state and guards");
        checked(cudaDeviceSynchronize(), "order upload before nonblocking computation");
    }
    OriginalInput(const OriginalInput&) = delete;
    OriginalInput& operator=(const OriginalInput&) = delete;
    // Reset intentionally owns native allocation/stream teardown. No use/free
    // of stale handles or speculative cleanup follows a failed operation.
    std::uint64_t update(unsigned int seed) {
        lifecycle_affine<<<kBlocks, kThreads, 0, stream>>>(output + kGuard, seed);
        checked(cudaGetLastError(), "ordinary registered kernel launch");
        checked(cudaStreamSynchronize(stream), "logical original stream completion");
        for (std::size_t index = 0; index < kValues; ++index)
            expected[index + kGuard] = expected[index + kGuard] * 1664525U + 1013904223U + seed +
                                       static_cast<std::uint32_t>(index);
        std::vector<std::uint32_t> actual(expected.size());
        checked(cudaMemcpy(actual.data(), output, actual.size() * sizeof(actual[0]),
                           cudaMemcpyDeviceToHost),
                "read every output and guard");
        expect(actual == expected, "complete non-idempotent original output mismatch");
        return checksum(actual);
    }
};
void run(bool cycler) {
    next_phase();
    checked(cudaSetDevice(0), "original Runtime-first primary selection");
    std::array<std::uint64_t, 6> sums{};
    std::array<int, 3> reset_current{};
    std::array<unsigned int, 4> primary_flags{};
    if (cycler) {
        for (unsigned int generation = 0; generation < 3; ++generation) {
            if (generation) next_phase();
            checked(cudaSetDevice(0), "reactivate Runtime primary generation");
            OriginalInput input(generation);
            if (!generation) std::this_thread::sleep_for(std::chrono::milliseconds{250});
            CUcontext first{}, second{};
            checked(cuDevicePrimaryCtxRetain(&first, 0), "first explicit primary retain");
            checked(cuDevicePrimaryCtxRetain(&second, 0), "second explicit primary retain");
            expect(first && first == second, "ordinary retains share primary identity");
            sums[std::size_t{2} * generation] = input.update(17 + generation);
            checked(cuDevicePrimaryCtxRelease(0), "release one Driver reference");
            checked(cuDevicePrimaryCtxRelease(0), "release second Driver reference");
            unsigned int flags{};
            int active{};
            checked(cuDevicePrimaryCtxGetState(0, &flags, &active),
                    "Runtime reference remains active");
            primary_flags[generation] = flags;
            expect(active == 1, "Driver releases cannot consume the Runtime-owned reference");
            sums[2 * generation + 1] = input.update(41 + generation);
            checked(cudaDeviceReset(), "reset with original live allocation and stream");
            CUcontext current{};
            checked(cuCtxGetCurrent(&current), "observe actual post-reset current state");
            reset_current[generation] = current != nullptr;
            std::cout << "GEN " << generation << '\n' << std::flush;
        }
        next_phase();
        CUcontext first{}, second{};
        checked(cuDevicePrimaryCtxRetain(&first, 0), "retain fresh Driver-only primary");
        checked(cuDevicePrimaryCtxRetain(&second, 0), "retain second Driver-only reference");
        expect(first && first == second, "fresh original Driver primary identity");
        checked(cuCtxSetCurrent(first), "select original Driver-only context");
        CUdeviceptr output{};
        std::array<std::uint32_t, 32> values{};
        for (std::size_t index = 0; index < values.size(); ++index)
            values[index] = 0xabcd1234U + static_cast<std::uint32_t>(index);
        checked(cuMemAlloc(&output, sizeof(values)), "allocate Driver-last-release state");
        checked(cuMemcpyHtoD(output, values.data(), sizeof(values)), "upload exact Driver state");
        checked(cuDevicePrimaryCtxRelease(0), "nonfinal Driver-only release");
        std::array<std::uint32_t, 32> actual{};
        checked(cuMemcpyDtoH(actual.data(), output, sizeof(actual)),
                "nonfinal reference keeps original state live");
        expect(actual == values, "Driver state survives nonfinal release");
        checked(cuDevicePrimaryCtxRelease(0), "last Driver-only release with live allocation");
        unsigned int flags{};
        int active{};
        checked(cuDevicePrimaryCtxGetState(0, &flags, &active),
                "observe actual final reference retirement");
        primary_flags[3] = flags;
        expect(active == 0, "last reference retires only its original primary context");
        checked(cuCtxSetCurrent(nullptr), "detach after final original release");
    } else {
        OriginalInput input(99);
        std::this_thread::sleep_for(std::chrono::milliseconds{250});
        std::cout << "READY\n" << std::flush;
        for (unsigned int step = 0; step < sums.size(); ++step) {
            next_phase();
            sums[step] = input.update(100 + step);
            std::cout << "STEP " << step << '\n' << std::flush;
        }
        next_phase();
        checked(cudaDeviceReset(), "retire retained sibling primary");
    }
    std::cout << "{\"status\":\"passed\",\"role\":\"" << (cycler ? "cycler" : "sibling")
              << "\",\"kernel_calls\":6,\"checked_values\":"
              << 6 * (kValues + std::size_t{2} * kGuard) + (cycler ? 32 : 0)
              << ",\"resets\":" << (cycler ? 3 : 1)
              << ",\"driver_last_release\":" << (cycler ? "true" : "false") << ",\"checksums\":[";
    for (std::size_t index = 0; index < sums.size(); ++index)
        std::cout << (index ? "," : "") << sums[index];
    std::cout
        << "],\"reset_current\":[" << reset_current[0] << ',' << reset_current[1] << ','
        << reset_current[2] << "],\"primary_flags\":[" << primary_flags[0] << ','
        << primary_flags[1] << ',' << primary_flags[2] << ',' << primary_flags[3]
        << "],\"cleanup\":true,\"full_device_qualified\":false,\"performance_qualified\":false}\n"
        << std::flush;
}
}  // namespace
int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--help") {
        std::cout << "Usage: lifecycle --cycler|--sibling; stdin phase lines: run\n";
        return EXIT_SUCCESS;
    }
    try {
        expect(argc == 2 &&
                   (std::string(argv[1]) == "--cycler" || std::string(argv[1]) == "--sibling"),
               "select one ordinary SDK role");
        run(std::string(argv[1]) == "--cycler");
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
