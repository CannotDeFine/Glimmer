#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <dlfcn.h>

extern "C" cudaError_t CUDARTAPI cudaLaunchHostFunc_ptsz(cudaStream_t, cudaHostFn_t, void*);

// Unchanged SDK-only application in both native and owned execution. Host
// callbacks never call CUDA or receive framework handles or protocol data.
__global__ void host_function_affine(std::uint32_t* values, unsigned int seed) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    values[index] = values[index] * 1664525U + 1013904223U + seed + index;
}

namespace {
constexpr unsigned int kBlocks = 1024, kThreads = 128, kGuard = 64;
constexpr std::size_t kValues = std::size_t{kBlocks} * kThreads;
constexpr std::size_t kExtent = kValues + 2 * kGuard;
constexpr std::uint32_t kMarker = 0x91ae73c5U;
using Launch = decltype(&cudaLaunchHostFunc);
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void checked(cudaError_t result, const char* operation) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + std::to_string(result));
}
struct Callback {
    std::uint32_t* host{};
    const std::vector<std::uint32_t>* expected{};
    bool should_mutate{};
    std::atomic<bool> can_return{}, has_failed{};
    std::atomic<unsigned int> calls{};
};
struct CallbackStorage {
    std::vector<std::uint32_t> expected, final_expected;
    Callback first, second;
};
struct CallbackLifetime {
    std::unique_ptr<CallbackStorage> state{std::make_unique<CallbackStorage>()};
    bool is_verified{};
    ~CallbackLifetime() {
        // A failing SDK call or an erroneously early synchronization may leave
        // application callbacks active. Preserve their data through process
        // exit instead of unwinding objects still referenced by another thread.
        if (!is_verified) (void)state.release();
    }
};
void CUDART_CB host_transform(void* opaque) {
    auto& input = *static_cast<Callback*>(opaque);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!input.can_return.load(std::memory_order_acquire)) {
        if (std::chrono::steady_clock::now() >= deadline) {
            input.has_failed = true;
            ++input.calls;
            return;
        }
        std::this_thread::yield();
    }
    for (std::size_t index = 0; index < kExtent; ++index)
        if (input.host[index] != (*input.expected)[index]) input.has_failed = true;
    if (input.should_mutate)
        for (std::size_t index = 0; index < kValues; ++index)
            input.host[index + kGuard] = input.host[index + kGuard] * 33U + 7U;
    ++input.calls;
}
void update(std::vector<std::uint32_t>& values, unsigned int seed) {
    for (std::size_t index = 0; index < kValues; ++index)
        values[index + kGuard] = values[index + kGuard] * 1664525U + 1013904223U + seed +
                                 static_cast<std::uint32_t>(index);
}
std::uint64_t checksum(const std::uint32_t* values) {
    std::uint64_t result = 14695981039346656037ULL;
    for (std::size_t index = 0; index < kExtent; ++index)
        for (unsigned int shift = 0; shift < 32; shift += 8) {
            result ^= (values[index] >> shift) & 255U;
            result *= 1099511628211ULL;
        }
    return result;
}
std::uint64_t exercise(Launch launch, unsigned int kind, bool ptsz) {
    std::uint32_t *device{}, *host{};
    cudaStream_t stream{};
    checked(cudaMalloc(reinterpret_cast<void**>(&device), kExtent * sizeof(*device)),
            "allocate complete device state");
    checked(cudaMallocHost(reinterpret_cast<void**>(&host), kExtent * sizeof(*host)),
            "allocate original pinned callback storage");
    if (kind == 1 || kind == 2)
        checked(cudaStreamCreateWithFlags(&stream,
                                          kind == 1 ? cudaStreamDefault : cudaStreamNonBlocking),
                "create explicit original stream");
    else if (kind == 3)
        stream = cudaStreamLegacy;
    else if (kind == 4 || ptsz)
        stream = cudaStreamPerThread;
    // Other APIs use an explicit special handle, isolating this callback entry
    // from unrelated PTDS API coverage. Only the suffix's implicit stream is NULL.
    const auto selected = ptsz && kind == 0 ? nullptr : stream;
    CallbackLifetime lifetime;
    auto& expected = lifetime.state->expected;
    expected.assign(kExtent, kMarker);
    for (std::size_t index = 0; index < kValues; ++index)
        expected[index + kGuard] = 0x12345678U + 17U * static_cast<std::uint32_t>(index) + kind;
    std::copy(expected.begin(), expected.end(), host);
    const auto seed = 17U + kind;
    update(expected, seed);
    auto& final_expected = lifetime.state->final_expected;
    final_expected = expected;
    for (std::size_t index = 0; index < kValues; ++index)
        final_expected[index + kGuard] = final_expected[index + kGuard] * 33U + 7U;
    update(final_expected, seed + 23U);
    auto& first = lifetime.state->first;
    auto& second = lifetime.state->second;
    first.host = second.host = host;
    first.expected = &expected;
    first.should_mutate = true;
    second.expected = &final_expected;
    second.can_return = true;

    checked(cudaMemcpyAsync(device, host, kExtent * sizeof(*host), cudaMemcpyHostToDevice, stream),
            "upload complete original input");
    host_function_affine<<<kBlocks, kThreads, 0, stream>>>(device + kGuard, seed);
    checked(cudaGetLastError(), "launch original producer");
    checked(cudaMemcpyAsync(host, device, kExtent * sizeof(*host), cudaMemcpyDeviceToHost, stream),
            "publish original producer before host function");
    checked(launch(selected, host_transform, &first), "enqueue asynchronous CPU transformation");
    // An inline implementation cannot wait for this CPU-only publication and
    // therefore fails the callback's bounded latch, even if numerics later match.
    first.can_return.store(true, std::memory_order_release);
    checked(cudaMemcpyAsync(device, host, kExtent * sizeof(*host), cudaMemcpyHostToDevice, stream),
            "consume host transformation only after callback return");
    host_function_affine<<<kBlocks, kThreads, 0, stream>>>(device + kGuard, seed + 23U);
    checked(cudaGetLastError(), "launch original consumer");
    checked(cudaMemcpyAsync(host, device, kExtent * sizeof(*host), cudaMemcpyDeviceToHost, stream),
            "publish all consumer results");
    checked(launch(selected, host_transform, &second), "enqueue exact final validation callback");
    checked(cudaStreamSynchronize(stream), "join stream including CPU function return");
    expect(first.calls == 1 && second.calls == 1 && !first.has_failed && !second.has_failed,
           "both functions execute exactly once with correct complete predecessors");
    expect(std::equal(final_expected.begin(), final_expected.end(), host),
           "complete non-idempotent dataflow and guards match after synchronization");
    // Both callbacks independently publish their final access before incrementing
    // calls. SDK synchronization alone is not the fixture's lifetime oracle.
    lifetime.is_verified = true;
    const auto result = checksum(host);
    if (kind == 1 || kind == 2)
        checked(cudaStreamDestroy(stream), "destroy completed original stream");
    checked(cudaFreeHost(host), "release original pinned callback storage");
    checked(cudaFree(device), "release original device storage");
    return result;
}
void run(bool cached, bool ptsz) {
    Launch launch = ptsz ? &cudaLaunchHostFunc_ptsz : &cudaLaunchHostFunc;
    if (cached) {
        launch = reinterpret_cast<Launch>(
            ::dlsym(RTLD_DEFAULT, ptsz ? "cudaLaunchHostFunc_ptsz" : "cudaLaunchHostFunc"));
        expect(launch != nullptr, "resolve cached public callback entry");
    }
    checked(cudaSetDevice(0), "select original Runtime primary context");
    std::vector<std::uint64_t> sums(ptsz ? 5 : 3);
    for (unsigned int kind = 0; kind < sums.size(); ++kind)
        sums[kind] = exercise(launch, kind, ptsz);
    checked(cudaDeviceReset(), "retire original context after callback cleanup");
    std::cout << "{\"status\":\"passed\",\"cached\":" << (cached ? "true" : "false")
              << ",\"callbacks\":" << 2 * sums.size() << ",\"kernel_calls\":" << 2 * sums.size()
              << ",\"checked_values\":" << 3 * sums.size() * kExtent << ",\"checksums\":[";
    for (std::size_t index = 0; index < sums.size(); ++index)
        std::cout << (index ? "," : "") << sums[index];
    if (ptsz)
        std::cout << "],\"ptsz\":true";
    else
        std::cout << ']';
    std::cout << ",\"asynchronous\":true,\"cleanup\":true,\"full_device_qualified\":false,"
                 "\"priority_qualified\":false,\"performance_qualified\":false}\n";
}
}  // namespace
int main(int argc, char** argv) {
    try {
        bool cached{}, ptsz{};
        for (int index = 1; index < argc; ++index) {
            const std::string option{argv[index]};
            if (option == "--cached" && !cached)
                cached = true;
            else if (option == "--ptsz" && !ptsz)
                ptsz = true;
            else
                throw std::runtime_error("unknown or duplicate callback selector");
        }
        run(cached, ptsz);
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
