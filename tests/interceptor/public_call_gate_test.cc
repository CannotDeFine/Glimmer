#include "internal/driver_api_interceptor.h"
#include "internal/driver_dispatch.h"
#include "internal/runtime_api_bridge.h"

#include <cuda_runtime_api.h>
#include <dlfcn.h>

#include <array>
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <thread>

namespace {
using glimmer::interceptor::cooperative_process_call_gate;
using glimmer::interceptor::NativeDriverCallScope;
using glimmer::interceptor::ProcessCallGate;

bool is_preview_enabled{};
thread_local bool expects_private_bypass{};
thread_local bool should_upgrade{};
thread_local bool is_nested{};
thread_local bool should_throw{};
std::atomic<unsigned int> observed_calls{};

void require(bool value, const char* message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}

void original_metadata_callback() {
    ++observed_calls;
    auto& gate = cooperative_process_call_gate();
    require(gate.is_entered_by_current_thread() == (is_preview_enabled && !expects_private_bypass),
            "original invocation must observe the covered caller, not a private worker");
    if (should_throw) {
        throw std::runtime_error("injected original callback exception");
    }
    if (!should_upgrade || is_nested || !is_preview_enabled || expects_private_bypass) {
        return;
    }
    // This CPU-only upgrade proves the real entry guards reused one caller;
    // counting nested Runtime/Driver scopes twice would deadlock here.
    const ProcessCallGate::Transition transition(gate);
    is_nested = true;
    unsigned int flags{};
    const auto driver_result = cuStreamGetFlags(nullptr, &flags);
    const auto runtime_result = cudaStreamGetFlags(nullptr, &flags);
    is_nested = false;
    require(driver_result == CUDA_SUCCESS && runtime_result == cudaSuccess,
            "trusted callbacks must remain reentrant while the outer call owns transition");
}

void check_calls() {
    unsigned int flags{};
    if (is_preview_enabled) {
        ProcessCallGate foreign;
        const ProcessCallGate::Call foreign_call(foreign);
        const auto before = observed_calls.load();
        require(cuStreamGetFlags(nullptr, &flags) == CUDA_ERROR_UNKNOWN &&
                    cudaStreamGetFlags(nullptr, &flags) == cudaErrorUnknown &&
                    foreign.is_entered_by_current_thread() &&
                    !cooperative_process_call_gate().is_entered_by_current_thread() &&
                    observed_calls.load() == before,
                "scope constructor failure must stay inside both C ABIs without native issue");
    }
    should_upgrade = true;
    require(cuStreamGetFlags(nullptr, &flags) == CUDA_SUCCESS &&
                cudaStreamGetFlags(nullptr, &flags) == cudaSuccess,
            "covered Driver and nested Runtime calls must forward original metadata");
    should_upgrade = false;
    require(!cooperative_process_call_gate().is_entered_by_current_thread(),
            "public scopes must end with the original call");
    {
        const NativeDriverCallScope scope;
        expects_private_bypass = true;
        require(cuStreamGetFlags(nullptr, &flags) == CUDA_SUCCESS &&
                    cudaStreamGetFlags(nullptr, &flags) == cudaSuccess,
                "private worker/native calls must bypass application admission");
        expects_private_bypass = false;
    }
    should_throw = true;
    require(cuStreamGetFlags(nullptr, &flags) == CUDA_ERROR_UNKNOWN &&
                cudaStreamGetFlags(nullptr, &flags) == cudaErrorUnknown,
            "original exceptions must be mapped inside both public C ABI guards");
    should_throw = false;
    require(!cooperative_process_call_gate().is_entered_by_current_thread(),
            "exception cleanup must restore the admitted caller count");
}
void concurrent_upgrades() {
    std::atomic<bool> has_failed{};
    std::array<std::thread, 4> callers;
    for (auto& caller : callers) {
        caller = std::thread([&] {
            should_upgrade = true;
            for (unsigned int index = 0; index < 50; ++index) {
                unsigned int flags{};
                if (cudaStreamGetFlags(nullptr, &flags) != cudaSuccess ||
                    cooperative_process_call_gate().is_entered_by_current_thread()) {
                    has_failed.store(true);
                }
            }
        });
    }
    for (auto& caller : callers) {
        caller.join();
    }
    require(!has_failed.load(), "competing public upgrades must drain without leaked admission");
}
}  // namespace

int main(int argc, char** argv) {
    try {
        is_preview_enabled = argc == 1;
        require(argc == 1 || (argc == 2 && std::string_view{argv[1]} == "--disabled"),
                "invalid fixture mode");
        using Setter = void (*)(void (*)());
        const auto setter = reinterpret_cast<Setter>(
            dlsym(RTLD_DEFAULT, "glimmer_test_set_stream_metadata_callback"));
        require(setter != nullptr, "fake original callback dispatch must be available");
        setter(original_metadata_callback);
        check_calls();
        concurrent_upgrades();
        setter(nullptr);
        require(observed_calls.load() >= 204, "public checks must actually reach originals");
        std::cout << "public_call_gate status=ok\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
