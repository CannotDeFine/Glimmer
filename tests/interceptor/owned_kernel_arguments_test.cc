#include "internal/owned_kernel_arguments.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <type_traits>

namespace {
using glimmer::interceptor::OwnedKernelArguments;
using glimmer::interceptor::PtxParameterLayout;
bool expect(bool value, const char* message) {
    if (!value) {
        std::cerr << message << '\n';
    }
    return value;
}
bool run() {
    static_assert(!std::is_copy_constructible_v<OwnedKernelArguments>);
    static_assert(std::is_nothrow_move_constructible_v<OwnedKernelArguments>);
    std::uint64_t pointer = 0x12345678;
    std::uint32_t scalar = 17;
    std::array<std::uint8_t, 16> aggregate{1, 2, 3, 4};
    const std::array<PtxParameterLayout, 3> layout{{{8, 8}, {4, 4}, {16, 16}}};
    std::array<void*, 3> addresses{&pointer, &scalar, aggregate.data()};
    auto original = OwnedKernelArguments::copy(layout, addresses.data());
    if (!expect(original.has_value(), "validated scalar and aggregate arguments must copy")) {
        return false;
    }
    const auto* stored = original->addresses()[0];
    auto moved = std::move(*original);
    original.reset();
    pointer = 0;
    scalar = 0;
    aggregate.fill(0);
    addresses.fill(nullptr);
    std::uint64_t copied_pointer = 0;
    std::uint32_t copied_scalar = 0;
    std::array<std::uint8_t, 16> copied_aggregate{};
    std::memcpy(&copied_pointer, moved.addresses()[0], sizeof(copied_pointer));
    std::memcpy(&copied_scalar, moved.addresses()[1], sizeof(copied_scalar));
    std::memcpy(copied_aggregate.data(), moved.addresses()[2], copied_aggregate.size());
    bool passed =
        expect(moved.addresses()[0] == stored && copied_pointer == 0x12345678 &&
                   copied_scalar == 17 && copied_aggregate[3] == 4,
               "continuation arguments must survive caller mutation, moves, and destruction");
    const std::uint64_t replacement = 91;
    const auto replacement_bytes = std::as_bytes(std::span{&replacement, 1});
    passed &= expect(moved.overwrite(0, 0, replacement_bytes) && moved.addresses()[0] == stored,
                     "bounded update must preserve owned argument addresses");
    std::memcpy(&copied_pointer, moved.addresses()[0], sizeof(copied_pointer));
    passed &= expect(copied_pointer == replacement && !moved.overwrite(3, 0, replacement_bytes) &&
                         !moved.overwrite(0, 1, replacement_bytes) &&
                         !moved.overwrite(0, std::numeric_limits<std::size_t>::max(), {}) &&
                         moved.overwrite(0, 8, {}),
                     "invalid updates must reject without changing valid bytes");
    const auto* overlapping = static_cast<const std::byte*>(moved.addresses()[2]);
    passed &= expect(moved.overwrite(2, 1, std::span{overlapping, 3}),
                     "overlapping owned updates use bounded move semantics");
    std::memcpy(copied_aggregate.data(), moved.addresses()[2], copied_aggregate.size());
    passed &= expect(copied_aggregate[0] == 1 && copied_aggregate[1] == 1 &&
                         copied_aggregate[2] == 2 && copied_aggregate[3] == 3,
                     "overlapping update must retain original source bytes");
    passed &= expect(OwnedKernelArguments::copy({}, nullptr).has_value() &&
                         !OwnedKernelArguments::copy(layout, nullptr).has_value() &&
                         !OwnedKernelArguments::copy(layout, addresses.data()).has_value(),
                     "empty arguments are valid but missing nonempty argument addresses are not");
    addresses[0] = &pointer;
    for (const auto invalid : {PtxParameterLayout{0, 1}, PtxParameterLayout{8, 3},
                               PtxParameterLayout{8, 0}, PtxParameterLayout{8, 256},
                               PtxParameterLayout{std::numeric_limits<std::size_t>::max(), 8}}) {
        const std::array invalid_layout{invalid};
        passed &= expect(!OwnedKernelArguments::copy(invalid_layout, addresses.data()).has_value(),
                         "unbounded or malformed layout must reject before reading caller bytes");
    }
    return passed;
}
}  // namespace
int main() {
    return run() ? EXIT_SUCCESS : EXIT_FAILURE;
}
