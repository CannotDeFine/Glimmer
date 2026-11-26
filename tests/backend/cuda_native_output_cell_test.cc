#include "internal/cuda_native_output_cell.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {
using namespace glimmer::backend::cuda;

void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

std::size_t page_bytes() {
    const auto bytes = ::sysconf(_SC_PAGESIZE);
    expect(bytes > 0, "actual Linux base page size is available");
    return static_cast<std::size_t>(bytes);
}

std::shared_ptr<CudaNativeOutputCell> create(std::span<const std::byte> initial) {
    auto result = CudaNativeOutputCell::create(initial);
    expect(result.status == CUDA_SUCCESS && result.cell && result.cell->data(),
           "real Linux output cell creation, without SDK or permissive fallback");
    return std::move(result.cell);
}

void observed(const CudaNativeOutputCell& cell, std::span<const std::byte> expected,
              bool was_written) {
    constexpr std::size_t guard_bytes = 16;
    std::vector<std::byte> bytes(expected.size() + 2 * guard_bytes, std::byte{0xd7});
    auto output = std::span{bytes}.subspan(guard_bytes, expected.size());
    const auto result = cell.snapshot(output);
    expect(result.status == CUDA_SUCCESS && result.was_written == was_written,
           "page provenance reports exact any-write state after proven CPU quiescence");
    expect(std::equal(output.begin(), output.end(), expected.begin(), expected.end()),
           "complete output bytes match the independent expected contents");
    for (std::size_t index = 0; index != guard_bytes; ++index)
        expect(
            bytes[index] == std::byte{0xd7} && bytes[bytes.size() - index - 1] == std::byte{0xd7},
            "snapshot preserves both complete destination guards");
}

template <typename Function>
void on_worker(Function&& function) {
    std::exception_ptr failure;
    std::thread worker([&] {
        try {
            function();
        } catch (...) {
            failure = std::current_exception();
        }
    });
    worker.join();
    if (failure) std::rethrow_exception(failure);
}

void read_only_seed_and_snapshots() {
    std::array<std::byte, 128> initial{};
    for (std::size_t index = 0; index != initial.size(); ++index)
        initial[index] = static_cast<std::byte>((index * 13 + 7) & 255U);
    const auto expected = initial;
    const auto cell = create(initial);
    initial.fill(std::byte{0x31});
    observed(*cell, expected, false);
    std::array<std::byte, 128> worker_copy{};
    on_worker([&] {
        expect(cell->data() != nullptr, "same-process distinct worker may read the native cell");
        std::memcpy(worker_copy.data(), cell->data(), worker_copy.size());
        observed(*cell, expected, false);
    });
    expect(worker_copy == expected, "seed is owned independently of caller storage");
    observed(*cell, expected, false);
    observed(*cell, expected, false);
}

void equal_null_and_poison_writes() {
    void* null_pointer{};
    std::array<std::byte, sizeof(null_pointer)> null_seed{};
    std::memcpy(null_seed.data(), &null_pointer, sizeof(null_pointer));
    const auto null_cell = create(null_seed);
    observed(*null_cell, null_seed, false);
    on_worker([&] {
        // The volatile native-shaped store must occur even though the output
        // bytes remain exactly equal to the original NULL seed.
        *static_cast<void* volatile*>(null_cell->data()) = nullptr;
    });
    observed(*null_cell, null_seed, true);

    std::array<std::byte, 128> poison{};
    poison.fill(std::byte{0xa5});
    const auto poison_cell = create(poison);
    observed(*poison_cell, poison, false);
    *static_cast<volatile unsigned char*>(poison_cell->data()) = 0xa5;
    observed(*poison_cell, poison, true);
    // This is intentionally one equal byte, NOT a claim of a full UID write.
    observed(*null_cell, null_seed, true);
    observed(*poison_cell, poison, true);
}

void changed_restored_and_independent_cells() {
    const std::array<std::byte, 4> seed{std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}};
    const auto first = create(seed);
    const auto second = create(seed);
    auto expected = seed;
    expected[2] = std::byte{0x7f};
    on_worker([&] { static_cast<volatile unsigned char*>(first->data())[2] = 0x7f; });
    observed(*first, expected, true);
    observed(*second, seed, false);
    static_cast<volatile unsigned char*>(first->data())[2] = 3;
    observed(*first, seed, true);
    observed(*second, seed, false);
    static_cast<volatile unsigned char*>(second->data())[3] = 4;
    observed(*first, seed, true);
    observed(*second, seed, true);
    const auto third = create(seed);
    observed(*third, seed, false);
    observed(*first, seed, true);
}

void full_page_and_bounds() {
    const auto page = page_bytes();
    std::vector<std::byte> expected(page, std::byte{0x81});
    const auto cell = create(expected);
    observed(*cell, expected, false);
    static_cast<volatile unsigned char*>(cell->data())[page - 1] = 0x26;
    expected.back() = std::byte{0x26};
    observed(*cell, expected, true);

    const auto empty = CudaNativeOutputCell::create({});
    expect(empty.status == CUDA_ERROR_INVALID_VALUE && !empty.cell, "empty seed is rejected");
    std::vector<std::byte> oversized(page + 1, std::byte{0x63});
    const auto too_large = CudaNativeOutputCell::create(oversized);
    expect(too_large.status == CUDA_ERROR_INVALID_VALUE && !too_large.cell,
           "a seed larger than one actual page is rejected");

    const std::array<std::byte, 8> small_seed{};
    const auto small = create(small_seed);
    std::array<std::byte, 9> output{};
    output.fill(std::byte{0x97});
    for (const auto length : {std::size_t{0}, std::size_t{7}, std::size_t{9}}) {
        const auto result = small->snapshot(std::span{output}.first(length));
        expect(result.status == CUDA_ERROR_INVALID_VALUE && !result.was_written,
               "snapshot requires exactly the original extent");
        for (const auto value : output)
            expect(value == std::byte{0x97}, "bad extent does not mutate output");
    }
    auto* mapping = static_cast<std::byte*>(small->data());
    for (const auto offset : {std::size_t{0}, std::size_t{1}, std::size_t{16}}) {
        const auto result = small->snapshot(std::span{mapping + offset, small_seed.size()});
        expect(result.status == CUDA_ERROR_INVALID_VALUE && !result.was_written,
               "snapshot cannot itself write any part of the isolated source page");
    }
    observed(*small, small_seed, false);
}

void fork_exclusion_and_address_reuse() {
    const std::array<std::byte, 8> seed{};
    auto cell = create(seed);
    auto* address = cell->data();
    const auto page = page_bytes();
    const auto child = ::fork();
    expect(child >= 0, "fork a process-local output-cell control");
    if (!child) {
        std::array<std::byte, 8> output{};
        output.fill(std::byte{0x5c});
        const auto observation = cell->snapshot(output);
        if (cell->data() || observation.status != CUDA_ERROR_INVALID_CONTEXT ||
            observation.was_written)
            ::_exit(1);
        for (const auto value : output)
            if (value != std::byte{0x5c}) ::_exit(2);
        auto* reused = ::mmap(address, page, PROT_READ | PROT_WRITE,
                              MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
        if (reused == MAP_FAILED || reused != address) ::_exit(3);
        *static_cast<volatile unsigned char*>(reused) = 0x37;
        cell.reset();
        // Destruction of the inherited object must not unmap another mapping
        // which now happens to occupy the omitted DONTFORK address.
        if (*static_cast<volatile unsigned char*>(reused) != 0x37) ::_exit(4);
        if (::munmap(reused, page) != 0) ::_exit(5);
        ::_exit(0);
    }
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "child rejects inherited mapping access and preserves reused child address");
    observed(*cell, seed, false);
    *static_cast<volatile unsigned char*>(cell->data()) = 0;
    observed(*cell, seed, true);
}

std::size_t descriptors() {
    std::size_t count{};
    for (const auto& entry : std::filesystem::directory_iterator("/proc/self/fd")) {
        (void)entry;
        ++count;
    }
    return count;
}
std::size_t mappings() {
    std::ifstream input("/proc/self/maps");
    expect(input.is_open(), "read own mapping inventory");
    std::size_t count{};
    std::string line;
    while (std::getline(input, line))
        if (line.find("glimmer-native-output") != std::string::npos) ++count;
    expect(input.eof() && !input.bad(), "complete mapping inventory read");
    return count;
}

void repeated_checked_cleanup() {
    const auto baseline_descriptors = descriptors();
    const auto baseline_mappings = mappings();
    const std::array<std::byte, 16> seed{};
    for (std::size_t iteration = 0; iteration != 64; ++iteration) {
        auto first = create(seed);
        auto second = create(seed);
        auto held = first;
        expect(descriptors() == baseline_descriptors + 4 && mappings() == baseline_mappings + 2,
               "each live isolated cell owns exactly two descriptors and one mapping");
        first.reset();
        observed(*held, seed, false);
        on_worker([&] { static_cast<volatile unsigned char*>(second->data())[7] = 0; });
        observed(*second, seed, true);
        held.reset();
        second.reset();
        expect(descriptors() == baseline_descriptors && mappings() == baseline_mappings,
               "last healthy owner releases every descriptor and mapping without SDK cleanup");
        const auto invalid = CudaNativeOutputCell::create({});
        expect(!invalid.cell && invalid.status == CUDA_ERROR_INVALID_VALUE &&
                   descriptors() == baseline_descriptors && mappings() == baseline_mappings,
               "repeated invalid creation does not retain partial resources");
    }
}
}  // namespace

int main() {
    try {
        read_only_seed_and_snapshots();
        equal_null_and_poison_writes();
        changed_restored_and_independent_cells();
        full_page_and_bounds();
        fork_exclusion_and_address_reuse();
        repeated_checked_cleanup();
        std::cout << "PASS: six native output-cell CPU groups (any-write only; no SDK/GPU claim)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
