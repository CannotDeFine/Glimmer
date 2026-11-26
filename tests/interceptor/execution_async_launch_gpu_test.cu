#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <dlfcn.h>

// Identical ordinary SDK bytes run natively and under the execution preload.
__global__ void async_launch_affine(std::uint32_t* output, unsigned int seed) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    output[index] = output[index] * 1664525U + 1013904223U + seed + index;
}

namespace {
constexpr unsigned int kBlocks = 1024, kThreads = 128, kGuard = 64;
constexpr std::size_t kValues = std::size_t{kBlocks} * kThreads;
constexpr std::size_t kExtent = kValues + std::size_t{2} * kGuard;
constexpr std::uint32_t kMarker = 0x91ae73c5U;
using Launch = cudaError_t(CUDARTAPI*)(const void*, dim3, dim3, void**, std::size_t, cudaStream_t);
using ThreadCall = cudaError_t(CUDARTAPI*)();
using Clock = std::chrono::steady_clock;
void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void checked(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + std::to_string(status));
}
struct Barrier {
    std::atomic<bool> entered{}, can_return{}, timed_out{}, returned{};
};
struct BarrierOwner {
    std::unique_ptr<Barrier> value{std::make_unique<Barrier>()};
    bool has_stopped{};
    ~BarrierOwner() {
        // An SDK error is not proof that the caller-side callback stopped.
        if (!has_stopped) (void)value.release();
    }
};
void CUDART_CB hold(void* opaque) {
    auto& barrier = *static_cast<Barrier*>(opaque);
    barrier.entered.store(true, std::memory_order_release);
    const auto deadline = Clock::now() + std::chrono::seconds{2};
    while (!barrier.can_return.load(std::memory_order_acquire)) {
        if (Clock::now() >= deadline) {
            barrier.timed_out.store(true, std::memory_order_release);
            break;
        }
        std::this_thread::yield();
    }
    barrier.returned.store(true, std::memory_order_release);
}
void update(std::vector<std::uint32_t>& values, unsigned int seed) {
    for (std::size_t index = 0; index != kValues; ++index)
        values[kGuard + index] = values[kGuard + index] * 1664525U + 1013904223U + seed +
                                 static_cast<std::uint32_t>(index);
}
std::uint64_t checksum(const std::vector<std::uint32_t>& values) {
    std::uint64_t result = 14695981039346656037ULL;
    for (const auto value : values)
        for (unsigned int shift = 0; shift != 32; shift += 8) {
            result ^= (value >> shift) & 255U;
            result *= 1099511628211ULL;
        }
    return result;
}
struct Mismatch {
    std::size_t index{};
    std::uint32_t expected{}, actual{};
};
struct Result {
    std::uint64_t checksum{};
    bool admitted_while_held{};
    std::optional<Mismatch> first_mismatch;
    std::size_t mismatch_count{};
};
Result exercise(Launch launch, unsigned int kind, ThreadCall synchronize) {
    cudaStream_t stream{};
    if (kind)
        checked(cudaStreamCreateWithFlags(&stream,
                                          kind == 1 ? cudaStreamDefault : cudaStreamNonBlocking),
                "create selected ordinary stream");
    std::vector<std::uint32_t> expected(kExtent, kMarker);
    for (std::size_t index = 0; index != kValues; ++index)
        expected[kGuard + index] = 0x12345678U + 17U * static_cast<std::uint32_t>(index) + kind;
    std::uint32_t* device{};
    checked(cudaMalloc(reinterpret_cast<void**>(&device), kExtent * sizeof(*device)),
            "allocate guarded original output");
    checked(cudaMemcpy(device, expected.data(), kExtent * sizeof(*device), cudaMemcpyHostToDevice),
            "initialize every original value and guard");
    // Pageable H2D may return after staging, before its default-stream DMA ends.
    // A nonblocking stream does not inherit that dependency; finish setup before
    // submitting any warmup or starting the independent admission latch.
    checked(cudaStreamSynchronize(nullptr), "finish default-stream input initialization");
    const auto submit = [&](unsigned int requested) {
        auto* output = device + kGuard;
        auto seed = requested;
        void* arguments[]{&output, &seed};
        checked(launch(reinterpret_cast<const void*>(async_launch_affine), dim3{kBlocks},
                       dim3{kThreads}, arguments, 0, stream),
                "admit the original kernel on its application thread");
        output = nullptr;
        seed = UINT32_MAX;
        arguments[0] = nullptr;
    };
    // Remove cold JIT/module preparation from the bounded callback watchdog.
    // This is a correctness latch, not a latency or performance measurement.
    submit(kind + 2U);
    update(expected, kind + 2U);
    checked(cudaStreamSynchronize(stream), "finish exact ordinary warmup");
    BarrierOwner owner;
    auto& barrier = *owner.value;
    checked(cudaLaunchHostFunc(stream, hold, &barrier), "enqueue real CPU dependency");
    const auto deadline = Clock::now() + std::chrono::seconds{5};
    while (!barrier.entered.load(std::memory_order_acquire)) {
        expect(Clock::now() < deadline, "original callback reaches its bounded CPU latch");
        std::this_thread::yield();
    }
    submit(kind + 17U);
    submit(kind + 40U);
    const bool admitted_while_held = !barrier.timed_out.load(std::memory_order_acquire);
    barrier.can_return.store(true, std::memory_order_release);
    update(expected, kind + 17U);
    update(expected, kind + 40U);
    if (synchronize) checked(synchronize(), "legacy device-wide callback/kernel completion");
    checked(cudaStreamSynchronize(stream), "finish original callback and both dependent kernels");
    expect(barrier.returned.load(std::memory_order_acquire),
           "callback independently publishes its last access before cleanup");
    owner.has_stopped = true;
    std::vector<std::uint32_t> observed(kExtent);
    checked(cudaMemcpy(observed.data(), device, kExtent * sizeof(*device), cudaMemcpyDeviceToHost),
            "download every original output and guard");
    std::optional<Mismatch> first_mismatch;
    std::size_t mismatch_count{};
    for (std::size_t index = 0; index != kExtent; ++index)
        if (observed[index] != expected[index]) {
            if (!first_mismatch) first_mismatch = Mismatch{index, expected[index], observed[index]};
            ++mismatch_count;
        }
    const auto result = checksum(observed);
    checked(cudaFree(device), "release completed original output");
    if (stream) checked(cudaStreamDestroy(stream), "destroy completed original stream");
    return {result, admitted_while_held, first_mismatch, mismatch_count};
}
void run(bool cached, bool legacy_thread) {
    Launch launch = &cudaLaunchKernel;
    if (cached) {
        launch = reinterpret_cast<Launch>(::dlsym(RTLD_DEFAULT, "cudaLaunchKernel"));
        expect(launch != nullptr, "cache the exact public Runtime launch ABI");
    }
    ThreadCall synchronize{}, reset{&cudaDeviceReset};
    if (legacy_thread) {
        // These deprecated entry ABIs are the explicit subject of this SDK input.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
        synchronize = &cudaThreadSynchronize;
        reset = &cudaThreadExit;
#pragma GCC diagnostic pop
        if (cached) {
            void* runtime = ::dlopen("libcudart.so.12", RTLD_NOW | RTLD_LOCAL);
            expect(runtime != nullptr,
                   "open original Runtime provider for qualified cached lookup");
            synchronize = reinterpret_cast<ThreadCall>(::dlsym(runtime, "cudaThreadSynchronize"));
            reset = reinterpret_cast<ThreadCall>(::dlsym(runtime, "cudaThreadExit"));
            expect(::dlclose(runtime) == 0, "close only the additional provider lookup reference");
        }
        expect(synchronize && reset, "resolve both exact public legacy thread entries");
    }
    checked(cudaSetDevice(0), "select ordinary Runtime primary context");
    std::array<Result, 3> results{};
    for (unsigned int kind = 0; kind != results.size(); ++kind)
        results[kind] = exercise(launch, kind, synchronize);
    checked(reset(), "normally retire the original context and all commands");
    // Complete all exact checks and ordinary resource retirement before reporting
    // a bounded first mismatch from each stream kind. Never change the oracle.
    for (unsigned int kind = 0; kind != results.size(); ++kind)
        if (const auto& mismatch = results[kind].first_mismatch) {
            const auto* region = mismatch->index < kGuard             ? "prefix-guard"
                                 : mismatch->index < kGuard + kValues ? "value"
                                                                      : "suffix-guard";
            std::cerr << "Numerical mismatch: kind=" << kind << "; index=" << mismatch->index
                      << "; region=" << region << "; expected=" << mismatch->expected
                      << "; actual=" << mismatch->actual
                      << "; mismatch_count=" << results[kind].mismatch_count << "; cleanup=true\n";
        }
    expect(std::ranges::none_of(
               results, [](const auto& result) { return result.first_mismatch.has_value(); }),
           "complete non-idempotent values and guards match exactly after normal cleanup");
    // Old synchronous launch behavior still completes full numerics and normal
    // cleanup through the bounded callback watchdog before this regression fails.
    expect(
        std::ranges::all_of(results, [](const auto& result) { return result.admitted_while_held; }),
        "same application thread returns both launches before releasing their CPU dependency");
    std::cout << "{\"status\":\"passed\",\"cached\":" << (cached ? "true" : "false")
              << ",\"callbacks\":3,\"kernel_calls\":9,\"checked_values\":" << 3 * kExtent
              << ",\"checksums\":[" << results[0].checksum << ',' << results[1].checksum << ','
              << results[2].checksum << ']';
    if (legacy_thread) std::cout << ",\"legacy_thread\":true";
    std::cout << ",\"same_thread_admission\":true,\"cleanup\":true,"
                 "\"full_device_qualified\":false,\"priority_qualified\":false,"
                 "\"performance_qualified\":false}\n";
}
}  // namespace
int main(int argc, char** argv) {
    try {
        bool cached{}, legacy_thread{};
        for (int index = 1; index < argc; ++index) {
            const std::string option{argv[index]};
            if (option == "--cached" && !cached)
                cached = true;
            else if (option == "--legacy-thread" && !legacy_thread)
                legacy_thread = true;
            else
                throw std::runtime_error("unknown or duplicate async SDK mode");
        }
        run(cached, legacy_thread);
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
