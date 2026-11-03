#include "internal/control_request_sequence.h"

#include <algorithm>
#include <array>
#include <barrier>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <vector>

int main() {
    using glimmer::control::internal::next_epoch_request_sequence;
    constexpr std::size_t k_threads = 4;
    constexpr std::size_t k_per_thread = 2000;
    const auto first = next_epoch_request_sequence();
    std::array<std::vector<std::uint64_t>, k_threads> values;
    std::barrier start(static_cast<std::ptrdiff_t>(k_threads));
    std::array<std::thread, k_threads> threads;
    for (std::size_t i = 0; i < k_threads; ++i) {
        values[i].reserve(k_per_thread);
        threads[i] = std::thread([&, i] {
            start.arrive_and_wait();
            for (std::size_t j = 0; j < k_per_thread; ++j)
                values[i].push_back(next_epoch_request_sequence());
        });
    }
    for (auto& thread : threads) thread.join();
    std::vector<std::uint64_t> combined;
    combined.reserve(k_threads * k_per_thread);
    for (const auto& thread_values : values)
        combined.insert(combined.end(), thread_values.begin(), thread_values.end());
    std::ranges::sort(combined);
    const auto last = next_epoch_request_sequence();
    if (!first || combined.front() != first + 1 || last != first + combined.size() + 1 ||
        std::ranges::adjacent_find(combined) != combined.end()) {
        std::cerr << "Concurrent control request sequence is not unique and monotonic\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
