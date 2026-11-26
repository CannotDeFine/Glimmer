#include "internal/cuda_deferred_event_binding.h"

#include <array>
#include <atomic>
#include <iostream>
#include <latch>
#include <stdexcept>
#include <thread>

namespace {
using glimmer::interceptor::CudaDeferredEventBinding;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}
void sticky_results() {
    CudaDeferredEventBinding invalid;
    require(!invalid.has_finished() && invalid.bind({}) == CUDA_ERROR_INVALID_VALUE &&
                !invalid.has_finished(),
            "invalid callback does not accept a binding");
    for (const auto native : {CUDA_SUCCESS, CUDA_ERROR_OUT_OF_MEMORY, CUDA_ERROR_INVALID_CONTEXT,
                              CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_UNKNOWN}) {
        CudaDeferredEventBinding binding;
        unsigned int calls{};
        require(binding.bind([&] {
            ++calls;
            return native;
        }) == native &&
                    binding.has_finished(),
                "checked original result completes exactly one attempt");
        for (unsigned int iteration = 0; iteration < 1000; ++iteration) {
            require(binding.bind([&] {
                ++calls;
                return CUDA_SUCCESS;
            }) == native,
                    "completed failure cannot replay binding");
        }
        require(calls == 1 && binding.bind({}) == native,
                "cached result is independent of future callback availability");
    }
    for (const bool is_allocation : {false, true}) {
        CudaDeferredEventBinding binding;
        unsigned int calls{};
        const auto expected = is_allocation ? CUDA_ERROR_OUT_OF_MEMORY : CUDA_ERROR_UNKNOWN;
        const auto throws = [&]() -> CUresult {
            ++calls;
            if (is_allocation) {
                throw std::bad_alloc();
            }
            throw std::runtime_error("uncertain source binding");
        };
        require(binding.bind(throws) == expected && binding.has_finished() &&
                    binding.bind(throws) == expected && calls == 1,
                "callback exception retains sticky failure without escaping or replay");
    }
    CudaDeferredEventBinding existing(true);
    unsigned int unexpected{};
    require(existing.has_finished() && existing.bind([&] {
        ++unexpected;
        return CUDA_ERROR_UNKNOWN;
    }) == CUDA_SUCCESS &&
                unexpected == 0,
            "already bound normal registration executes no import");
}
void competing_callers() {
    for (const auto result : {CUDA_SUCCESS, CUDA_ERROR_NOT_INITIALIZED}) {
        CudaDeferredEventBinding binding;
        std::latch ready{4};
        std::latch start{1};
        std::latch entered{1};
        std::latch release{1};
        std::atomic<unsigned int> callbacks{};
        std::array<CUresult, 4> results{};
        std::array<std::thread, 4> callers;
        std::array<bool, 4> source_threads{};
        for (std::size_t index = 0; index < callers.size(); ++index) {
            callers[index] = std::thread([&, index] {
                const auto original_caller = std::this_thread::get_id();
                ready.count_down();
                start.wait();
                results[index] = binding.bind([&] {
                    source_threads[index] = original_caller == std::this_thread::get_id();
                    callbacks.fetch_add(1);
                    entered.count_down();
                    release.wait();
                    return result;
                });
            });
        }
        ready.wait();
        start.count_down();
        entered.wait();
        const bool was_unfinished = !binding.has_finished();
        release.count_down();
        for (auto& caller : callers) {
            caller.join();
        }
        unsigned int sources{};
        for (std::size_t index = 0; index < results.size(); ++index) {
            require(results[index] == result, "every competitor observes the exact sticky result");
            sources += source_threads[index] ? 1 : 0;
        }
        require(was_unfinished && binding.has_finished() && callbacks.load() == 1 && sources == 1,
                "one callback runs on its original caller, not another worker or importer");
    }
}
}  // namespace

int main() {
    try {
        sticky_results();
        competing_callers();
        std::cout << "deferred event binding ownership passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
