#include "internal/cuda_module_jit.h"
#include "glimmer/control/execution_jit_protocol.h"

#include <algorithm>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <new>
#include <utility>

namespace {
thread_local bool should_fail_allocation{};
thread_local unsigned int injected_failures{};
}  // namespace
// Linux64 SDK frame storage uses ordinary C++ new (no over-aligned element).
// Linker wrapping keeps its real matching delete and sanitizer allocation ABI.
extern "C" void* real_jit_test_new(std::size_t bytes) asm("__real__Znwm");
extern "C" void* wrapped_jit_test_new(std::size_t bytes) asm("__wrap__Znwm");
extern "C" void* wrapped_jit_test_new(std::size_t bytes) {
    if (std::exchange(should_fail_allocation, false)) {
        ++injected_failures;
        throw std::bad_alloc();
    }
    return real_jit_test_new(bytes);
}

namespace {
using namespace glimmer::backend::cuda;
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
std::vector<CudaModuleJitOption> example() {
    std::vector<std::byte> time(sizeof(float));
    const float sentinel = -99.5F;
    std::memcpy(time.data(), &sentinel, sizeof(sentinel));
    return {{CU_JIT_THREADS_PER_BLOCK, 128, {}},
            {CU_JIT_MAX_REGISTERS, 64, {}},
            {CU_JIT_WALL_TIME, 0, time},
            {CU_JIT_INFO_LOG_BUFFER, 0, std::vector<std::byte>(64, std::byte{0xa5})},
            {CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES, 64, {}},
            {CU_JIT_ERROR_LOG_BUFFER, 0, std::vector<std::byte>(32, std::byte{0x6b})},
            {CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, 32, {}},
            {CU_JIT_OPTIMIZATION_LEVEL, 4, {}}};
}
void native_slots_and_full_outputs() {
    auto input = example();
    const auto original = input;
    auto frame = CudaModuleJitArguments::create(input);
    expect(frame && frame->inputs().size() == input.size(),
           "create owned SDK frame before any native call");
    input.assign(1, {99, UINT64_MAX, {std::byte{0xff}}});
    expect(std::ranges::equal(frame->inputs(), original),
           "frame retains complete immutable caller inputs");
    auto slots = frame->native_values();
    expect(reinterpret_cast<std::uintptr_t>(slots[0]) == 128 &&
               reinterpret_cast<std::uintptr_t>(slots[1]) == 64 &&
               reinterpret_cast<std::uintptr_t>(slots[2]) == std::bit_cast<std::uint32_t>(-99.5F) &&
               reinterpret_cast<std::uintptr_t>(slots[4]) == 64,
           "integer and float bits occupy slots without pointer dereferences");
    const auto bytes = frame->storage_bytes();
    auto* time = slots[2];
    auto* info = slots[3];
    auto* error = slots[5];
    const float elapsed = 3.25F;
    std::memcpy(&slots[2], &elapsed, sizeof(elapsed));
    std::memcpy(info, "compile", 8);
    *static_cast<char*>(error) = '\0';
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    slots[0] = reinterpret_cast<void*>(256);
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    slots[4] = reinterpret_cast<void*>(8);
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    slots[6] = reinterpret_cast<void*>(1);
    const auto captured = frame->capture();
    expect(captured && captured->size() == original.size() && (*captured)[0].value == 256 &&
               (*captured)[4].value == 8 && (*captured)[6].value == 1,
           "capture actual output slots instead of dereferencing scalar addresses");
    float observed{};
    std::memcpy(&observed, (*captured)[2].data.data(), sizeof(observed));
    expect(
        observed == 3.25F && (*captured)[3].data.size() == 64 && (*captured)[5].data.size() == 32,
        "capture float and full caller-visible log extents");
    const auto* info_text = reinterpret_cast<const char*>((*captured)[3].data.data());
    expect(std::memcmp(info_text, "compile", sizeof("compile")) == 0 &&
               std::ranges::all_of(std::span((*captured)[3].data).subspan(8),
                                   [](auto byte) { return byte == std::byte{0xa5}; }) &&
               (*captured)[5].data.front() == std::byte{} &&
               std::ranges::all_of(std::span((*captured)[5].data).subspan(1),
                                   [](auto byte) { return byte == std::byte{0x6b}; }),
           "every untouched log byte survives exactly");
    frame->reset();
    expect(frame->native_values()[2] == time && frame->native_values()[3] == info &&
               frame->native_values()[5] == error && frame->storage_bytes() == bytes,
           "reset preserves preallocated output storage and charge");
    const auto reset = frame->capture();
    expect(reset && std::ranges::equal(*reset, original),
           "new private compile starts from exact original inputs");
    frame->native_values()[2] = nullptr;
    const auto zero_time = frame->capture();
    expect(zero_time && std::ranges::all_of((*zero_time)[2].data,
                                            [](auto byte) { return byte == std::byte{}; }),
           "a zero float output is valid slot data, never a missing pointee");
}
void independent_native_grammar_and_hostile_sdk() {
    for (std::uint32_t id = 0; id <= 34; ++id) {
        CudaModuleJitOption option{id, 0, {}};
        const auto kind = glimmer::control::execution_jit_storage(id);
        if (kind == glimmer::control::ExecutionJitStorage::kFloat) option.data.resize(4);
        if (kind == glimmer::control::ExecutionJitStorage::kNames) option.data.resize(4);
        std::vector<CudaModuleJitOption> input{option};
        if (id == CU_JIT_INFO_LOG_BUFFER || id == CU_JIT_ERROR_LOG_BUFFER) {
            input[0].data.resize(1);
            input.push_back({id + 1, 1, {}});
        }
        if (id == CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES || id == CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES) {
            input[0].value = 1;
            input.push_back({id - 1, 0, {std::byte{}}});
        }
        expect(static_cast<bool>(CudaModuleJitArguments::create(input)) ==
                   (kind != glimmer::control::ExecutionJitStorage::kUnsupported),
               "CUDA-native enum grammar agrees with independent CUDA-free protocol classifier");
    }
    for (unsigned int mode = 0; mode < 5; ++mode) {
        auto frame = CudaModuleJitArguments::create(example());
        expect(frame != nullptr, "independent hostile SDK frame");
        auto values = frame->native_values();
        if (mode == 0) {
            // NOLINTNEXTLINE(performance-no-int-to-ptr)
            values[2] = reinterpret_cast<void*>(std::uintptr_t{1} << 32);
        }
        if (mode == 1) values[3] = nullptr;
        if (mode == 2) values[1] = nullptr;
        if (mode == 3) frame->native_options()[7] = CU_JIT_MAX_REGISTERS;
        if (mode == 4) {
            // NOLINTNEXTLINE(performance-no-int-to-ptr)
            values[0] = reinterpret_cast<void*>(std::uintptr_t{1} << 32);
        }
        expect(!frame->capture(),
               "malformed pointer/key/input-only/scalar-output SDK mutation rejects");
        frame->reset();
        expect(frame->capture().has_value(),
               "frame reset creates no native retry or cleanup permission");
    }
    auto duplicate = example();
    duplicate.push_back(duplicate.front());
    expect(!CudaModuleJitArguments::create(duplicate), "duplicate input keys reject before SDK");
    auto mismatch = example();
    mismatch[4].value = 63;
    expect(!CudaModuleJitArguments::create(mismatch),
           "input log capacity cannot exceed owned bytes");
    expect(CudaModuleJitArguments::create({}) != nullptr,
           "empty native frame has no invented work");
}
void reset_and_capture_do_not_allocate() {
    const auto input = example();
    auto frame = CudaModuleJitArguments::create(input);
    expect(frame != nullptr, "preallocate immutable and mutable frame storage");
    should_fail_allocation = true;
    frame->reset();
    const auto actual = frame->capture();
    const bool remained_armed = std::exchange(should_fail_allocation, false);
    expect(remained_armed && actual && std::ranges::equal(*actual, input),
           "reset and complete capture consume no ordinary C++ allocation");
    frame->native_values()[3] = nullptr;
    should_fail_allocation = true;
    const auto malformed = frame->capture();
    const bool error_remained_armed = std::exchange(should_fail_allocation, false);
    expect(error_remained_armed && !malformed, "malformed SDK output rejects without allocation");
    should_fail_allocation = true;
    const auto rejected = CudaModuleJitArguments::create(input);
    expect(!rejected && !should_fail_allocation && injected_failures == 1,
           "positive fault control proves frame construction uses the still-armed allocation hook");
}
void referenced_names_use_owned_pointer_arrays() {
    const std::vector<std::byte> names{
        std::byte{1},   std::byte{},    std::byte{},    std::byte{},    std::byte{7},
        std::byte{},    std::byte{},    std::byte{},    std::byte{'k'}, std::byte{'e'},
        std::byte{'r'}, std::byte{'n'}, std::byte{'e'}, std::byte{'l'}, std::byte{}};
    const std::vector<CudaModuleJitOption> options{{CU_JIT_REFERENCED_KERNEL_NAMES, 1, names},
                                                   {CU_JIT_REFERENCED_KERNEL_COUNT, 1, {}},
                                                   {CU_JIT_REFERENCED_VARIABLE_NAMES, 1, names},
                                                   {CU_JIT_REFERENCED_VARIABLE_COUNT, 1, {}}};
    auto frame = CudaModuleJitArguments::create(options);
    expect(frame && frame->capture(),
           "own both exact referenced name families before native entry");
    auto values = frame->native_values();
    auto** kernels = static_cast<const char**>(values[0]);
    auto** variables = static_cast<const char**>(values[2]);
    expect(kernels && variables && kernels != variables && std::strcmp(kernels[0], "kernel") == 0 &&
               std::strcmp(variables[0], "kernel") == 0 && !kernels[1] && !variables[1] &&
               reinterpret_cast<std::uintptr_t>(values[1]) == 1,
           "SDK sees distinct private stable char-pointer arrays and native integer counts");
    for (unsigned int mode = 0; mode < 5; ++mode) {
        if (mode == 0) values[0] = nullptr;
        if (mode == 1) kernels[0] = reinterpret_cast<const char*>(1);
        if (mode == 2) const_cast<char*>(kernels[0])[0] = '!';
        if (mode == 3) kernels[1] = kernels[0];
        if (mode == 4) values[1] = reinterpret_cast<void*>(2);
        should_fail_allocation = true;
        const auto malformed = frame->capture();
        frame->reset();
        const auto restored = frame->capture();
        const bool armed = std::exchange(should_fail_allocation, false);
        expect(!malformed && restored && std::ranges::equal(*restored, options) && armed &&
                   frame->native_values()[0] == kernels && frame->native_values()[2] == variables,
               "native pointer/key/string/tail corruption rejects and resets without allocation or "
               "dereference");
    }
    const std::vector<std::byte> zero(4);
    for (const auto presence : {0U, 1U}) {
        const std::array input{CudaModuleJitOption{CU_JIT_REFERENCED_KERNEL_NAMES, presence, zero}};
        auto empty = CudaModuleJitArguments::create(input);
        expect(empty &&
                   static_cast<bool>(empty->native_values()[0]) == static_cast<bool>(presence) &&
                   empty->capture(),
               "native zero-count pointer presence is not normalized away");
        empty->reset_for_private();
        expect(!empty->has_private_references() && empty->native_options().size() == input.size() &&
                   static_cast<bool>(empty->native_values()[0]) == static_cast<bool>(presence),
               "zero-count reference metadata must not become a restrictive private filter");
    }
    should_fail_allocation = true;
    frame->reset_for_private();
    const auto private_outputs = frame->capture();
    const bool private_valid =
        frame->has_private_references() && private_outputs &&
        frame->native_options().size() == options.size() && (*private_outputs)[1].value == 2 &&
        (*private_outputs)[3].value == 2 && std::strcmp(kernels[1], "*__glimmer*") == 0 &&
        std::strcmp(variables[1], "*__glimmer*") == 0 && !kernels[2] && !variables[2] &&
        std::ranges::equal(frame->inputs(), options);
    frame->reset();
    const auto original_outputs = frame->capture();
    const bool armed = std::exchange(should_fail_allocation, false);
    expect(private_valid && original_outputs && std::ranges::equal(*original_outputs, options) &&
               armed && !kernels[1] && !variables[1],
           "cold private recipe adds only reserved retention and restores exact caller data "
           "without allocation");
}
void private_retention_keeps_public_bounds_and_zero_semantics() {
    std::vector<std::byte> names(4);
    names[0] = std::byte{32};
    for (unsigned int index = 0; index < 32; ++index)
        names.insert(names.end(), {std::byte{2}, std::byte{}, std::byte{}, std::byte{},
                                   std::byte{'x'}, std::byte{}});
    const std::array input{CudaModuleJitOption{CU_JIT_REFERENCED_KERNEL_NAMES, 1, names},
                           CudaModuleJitOption{CU_JIT_REFERENCED_KERNEL_COUNT, 32, {}}};
    auto frame = CudaModuleJitArguments::create(input);
    expect(frame && frame->has_private_references(),
           "preallocate maximum public names and private policy");
    const auto charge = frame->storage_bytes();
    should_fail_allocation = true;
    frame->reset_for_private();
    auto** pointers = static_cast<const char**>(frame->native_values()[0]);
    const auto private_output = frame->capture();
    const bool complete = private_output && (*private_output)[1].value == 33 &&
                          std::strcmp(pointers[31], "x") == 0 &&
                          std::strcmp(pointers[32], "*__glimmer*") == 0;
    frame->reset();
    const auto original = frame->capture();
    const bool armed = std::exchange(should_fail_allocation, false);
    expect(complete && original && std::ranges::equal(*original, input) && armed &&
               frame->storage_bytes() == charge && !pointers[32],
           "one private-only extra reference cannot enlarge public admission or allocate during "
           "reset/capture");
    auto hostile = std::vector<CudaModuleJitOption>(input.begin(), input.end());
    hostile[0].data[0] = std::byte{33};
    hostile[1].value = 33;
    expect(!CudaModuleJitArguments::accepts(hostile),
           "private33 is never an untrusted public bound");
    const std::array zero_count{CudaModuleJitOption{CU_JIT_REFERENCED_KERNEL_COUNT, 0, {}}};
    auto empty = CudaModuleJitArguments::create(zero_count);
    expect(empty && !empty->has_private_references(),
           "count-only zero cannot create a new private filter");
    empty->reset_for_private();
    expect(empty->capture()->size() == 1 && empty->native_values()[0] == nullptr,
           "private compilation preserves an absent names array and its zero scalar");
}
}  // namespace
int main() {
    try {
        native_slots_and_full_outputs();
        independent_native_grammar_and_hostile_sdk();
        reset_and_capture_do_not_allocate();
        referenced_names_use_owned_pointer_arrays();
        private_retention_keeps_public_bounds_and_zero_semantics();
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
