#include "cuda_gemm_completion_election_fixture.h"

#include <cuda.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>

namespace {
using namespace glimmer::test::gemm_completion;
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void check(CUresult status) {
    if (status != CUDA_SUCCESS) throw std::runtime_error("CUDA status " + std::to_string(status));
}
struct Resources {
    CUcontext context{};
    CUmodule module{};
    CUdeviceptr counter{}, mapped{};
    Shared* shared{};
    bool is_launched{};
    bool close() noexcept {
        bool result = true;
        if (shared && is_launched) {
            std::atomic_ref(shared->release).store(1, std::memory_order_release);
            // An uncertain stop cannot authorize freeing mapped memory. The
            // isolated failing process retains its owners until process exit.
            if (cuCtxSynchronize() != CUDA_SUCCESS) return false;
            is_launched = false;
        }
        if (shared) {
            result = cuMemFreeHost(shared) == CUDA_SUCCESS && result;
            shared = nullptr;
        }
        if (context) {
            const auto destroyed = cuCtxDestroy(context);
            result = destroyed == CUDA_SUCCESS && result;
            context = nullptr;
            module = nullptr;
            counter = 0;
        }
        return result;
    }
};
struct Scenario {
    bool observes{}, holds_zero{};
};
void run_case(Resources& resources, Scenario scenario) {
    const auto [observes, holds_zero] = scenario;
    auto& shared = *resources.shared;
    shared = {};
    std::ranges::fill(shared.output, kSentinel);
    constexpr unsigned long long initial_claim = 0x4ad921;
    const std::array<unsigned long long, 2> initial{initial_claim, observes ? 0ULL : 0x5d93ULL};
    check(cuMemcpyHtoD(resources.counter, initial.data(), sizeof(initial)));
    CUfunction kernel{};
    check(
        cuModuleGetFunction(&kernel, resources.module,
                            observes ? "gemm_completion_observed" : "gemm_completion_unobserved"));
    unsigned int hold = holds_zero ? 1U : 0U;
    std::array<void*, 3> arguments{&resources.mapped, &resources.counter, &hold};
    resources.is_launched = true;
    check(cuLaunchKernel(kernel, kWorkers, 1, 1, kThreads, 1, 1, 0, nullptr, arguments.data(),
                         nullptr));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{2};
    const auto wait = [&](std::uint32_t& word) {
        while (std::atomic_ref(word).load(std::memory_order_acquire) != 1) {
            if (std::chrono::steady_clock::now() >= deadline) return false;
            std::this_thread::yield();
        }
        return true;
    };
    const bool other_finished = wait(shared.done[1]);
    const bool published =
        observes ? wait(shared.completed)
                 : std::atomic_ref(shared.completed).load(std::memory_order_acquire) == 0;
    // Release/join regardless of the assertion result. A policy failure cannot
    // strand CTA0 or infer resource retirement from a witness alone.
    std::atomic_ref(shared.release).store(1, std::memory_order_release);
    check(cuCtxSynchronize());
    resources.is_launched = false;
    std::array<unsigned long long, 2> actual{};
    check(cuMemcpyDtoH(actual.data(), resources.counter, sizeof(actual)));
    for (unsigned int index = 0; index < std::size(shared.output); ++index) {
        const auto expected = index >= kGuard && index < kGuard + kWorkers * kThreads
                                  ? 3 * (index - kGuard) + 17
                                  : kSentinel;
        expect(shared.output[index] == expected, "every tile output and guard must match");
    }
    expect(std::ranges::all_of(shared.done, [](auto value) { return value == 1; }),
           "all physical CTAs completed actual output after release");
    expect(actual[0] == initial_claim && (!observes ? actual[1] == initial[1] : actual[1] <= 1),
           "completion cannot overwrite the claim count or unobserved private storage");
    const auto publications =
        std::accumulate(std::begin(shared.publishers), std::end(shared.publishers), 0U);
    expect(publications == (observes ? 1U : 0U),
           "exactly one observed publisher, none for default");
    expect(other_finished, "independent nonzero CTA must complete while CTA0 is held");
    expect(published, "a completed nonzero CTA must publish before held CTA0 is released");
    if (observes && holds_zero)
        expect(shared.publishers[0] == 0, "held CTA0 cannot borrow another CTA's publication");
    if (observes) expect(actual[1] == 1, "exact device-only election has one retained winner");
}
}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        std::cout << "Usage: gemm completion election control <fixture.ptx>\n";
        return EXIT_SUCCESS;
    }
    Resources resources;
    std::string error;
    try {
        expect(argc == 2, "exact compiled private-helper PTX path required");
        check(cuInit(0));
        check(cuCtxCreate(&resources.context, CU_CTX_MAP_HOST, 0));
        check(cuModuleLoad(&resources.module, argv[1]));
        check(cuMemAlloc(&resources.counter, 16));
        void* mapping{};
        check(cuMemHostAlloc(&mapping, sizeof(Shared), CU_MEMHOSTALLOC_DEVICEMAP));
        resources.shared = std::construct_at(static_cast<Shared*>(mapping));
        check(cuMemHostGetDevicePointer(&resources.mapped, resources.shared, 0));
        run_case(resources, {.observes = true, .holds_zero = true});
        run_case(resources, {.observes = true, .holds_zero = false});
        run_case(resources, {.observes = false, .holds_zero = true});
    } catch (const std::exception& exception) {
        error = exception.what();
    }
    const bool closed = resources.close();
    if (!closed || !error.empty()) {
        if (!error.empty()) std::cerr << error << '\n';
        if (!closed) std::cerr << "native cleanup failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "{\"status\":\"passed\",\"cases\":3,\"nonzero_cta_before_release\":true,"
                 "\"unique_publisher\":true,\"checked_values\":576,\"cleanup\":true,"
                 "\"full_device_qualified\":false,\"priority_qualified\":false,"
                 "\"performance_qualified\":false}\n";
    return EXIT_SUCCESS;
}
