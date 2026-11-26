#include <cuda.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <string_view>

#ifdef cuGetProcAddress
#undef cuGetProcAddress
#endif
extern "C" CUresult CUDAAPI cuGetProcAddress(const char*, void**, int, cuuint64_t);

#ifdef GLIMMER_MODULE_FACTORY_PROVIDER
namespace {
std::atomic<unsigned int> native_calls{};
}
extern "C" unsigned int glimmer_module_factory_native_calls() {
    return native_calls.load();
}
extern "C" CUresult CUDAAPI cuInit(unsigned int) {
    return CUDA_SUCCESS;
}
extern "C" cudaError_t CUDARTAPI cudaRuntimeGetVersion(int* output) {
    if (!output) return cudaErrorInvalidValue;
    *output = CUDART_VERSION;
    return cudaSuccess;
}
extern "C" CUresult CUDAAPI cuModuleLoadData(CUmodule*, const void*) {
    ++native_calls;
    return CUDA_ERROR_UNKNOWN;
}
extern "C" CUresult CUDAAPI cuModuleLoadFatBinary(CUmodule*, const void*) {
    ++native_calls;
    return CUDA_ERROR_UNKNOWN;
}
extern "C" CUresult CUDAAPI cuModuleLoadDataEx(CUmodule*, const void*, unsigned int, CUjit_option*,
                                               void**) {
    ++native_calls;
    return CUDA_ERROR_UNKNOWN;
}
extern "C" CUresult CUDAAPI cuGetProcAddress_v2(const char* name, void** output, int,
                                                cuuint64_t flags,
                                                CUdriverProcAddressQueryResult* status) {
    if (!name || !output || flags) return CUDA_ERROR_INVALID_VALUE;
    *output = nullptr;
    if (std::string_view(name) == "cuModuleLoadData")
        *output = reinterpret_cast<void*>(&cuModuleLoadData);
    if (std::string_view(name) == "cuModuleLoadFatBinary")
        *output = reinterpret_cast<void*>(&cuModuleLoadFatBinary);
    if (std::string_view(name) == "cuModuleLoadDataEx")
        *output = reinterpret_cast<void*>(&cuModuleLoadDataEx);
    if (status)
        *status = *output ? CU_GET_PROC_ADDRESS_SUCCESS : CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
    return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuGetProcAddress(const char* name, void** output, int version,
                                             cuuint64_t flags) {
    return cuGetProcAddress_v2(name, output, version, flags, nullptr);
}
#else
#include "glimmer/control/execution_socket.h"
#include "glimmer/control/execution_jit_protocol.h"
#include "internal/process_identity.h"

#include <chrono>
#include <bit>
#include <cstdlib>
#include <future>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

extern "C" unsigned int glimmer_module_factory_native_calls();
namespace {
using namespace glimmer::control;
using namespace std::chrono_literals;
constexpr std::string_view kPtx = "independently owned module factory PTX fixture";
constexpr unsigned int kLastCase = 39;
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void check(CUresult value) {
    expect(value == CUDA_SUCCESS, "owned public module operation succeeds");
}
struct Functions {
    decltype(&cuModuleLoadData) data{};
    decltype(&cuModuleLoadFatBinary) fat{};
    decltype(&cuModuleLoadDataEx) extended{};
};
bool owned(void* pointer) {
    Dl_info information{};
    return pointer && ::dladdr(pointer, &information) && information.dli_fname &&
           std::string_view(information.dli_fname).ends_with("libglimmer_execution_interceptor.so");
}
std::vector<Functions> routes(const char* path) {
    void* provider = ::dlopen(path, RTLD_NOW | RTLD_LOCAL);
    expect(provider, "load independent original module provider");
    std::vector<Functions> result{{&cuModuleLoadData, &cuModuleLoadFatBinary, &cuModuleLoadDataEx}};
    for (unsigned int route = 0; route < 5; ++route) {
        constexpr std::array names{"cuModuleLoadData", "cuModuleLoadFatBinary",
                                   "cuModuleLoadDataEx"};
        std::array<void*, 3> pointers{};
        for (std::size_t index = 0; index < names.size(); ++index) {
            if (route < 2) {
                pointers[index] = ::dlsym(route ? provider : RTLD_DEFAULT, names[index]);
            } else if (route == 2) {
                check(cuGetProcAddress(names[index], &pointers[index], 12060, 0));
            } else if (route == 3) {
                CUdriverProcAddressQueryResult status = CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
                check(cuGetProcAddress_v2(names[index], &pointers[index], 12060, 0, &status));
                expect(status == CU_GET_PROC_ADDRESS_SUCCESS, "Driver proc lookup status");
            } else {
                cudaDriverEntryPointQueryResult status = cudaDriverEntryPointSymbolNotFound;
                expect(cudaGetDriverEntryPointByVersion(names[index], &pointers[index], 12060, 0,
                                                        &status) == cudaSuccess &&
                           status == cudaDriverEntryPointSuccess,
                       "Runtime proc lookup status");
            }
            expect(owned(pointers[index]), "qualified lookup cannot expose a native factory");
        }
        result.push_back({reinterpret_cast<decltype(&cuModuleLoadData)>(pointers[0]),
                          reinterpret_cast<decltype(&cuModuleLoadFatBinary)>(pointers[1]),
                          reinterpret_cast<decltype(&cuModuleLoadDataEx)>(pointers[2])});
    }
    expect(::dlclose(provider) == 0, "cached functions survive original provider close");
    return result;
}
std::array<std::uint64_t, 4> binary_image() {
    // Raw version-one fatbinary transport, not an actual executable SDK image.
    return {0x00100001ba55ed50, 16, 0x0123456789abcdef, 0xfedcba9876543210};
}
CUresult load(const Functions& functions, unsigned int api, CUmodule* output) {
    auto image = binary_image();
    return api == 0 ? functions.data(output, kPtx.data())
           : api == 1
               ? functions.fat(output, image.data())
               : functions.extended(output, kPtx.data(), 0, reinterpret_cast<CUjit_option*>(1),
                                    reinterpret_cast<void**>(1));
}
struct Control {
    std::atomic<unsigned int> loads{}, unloads{}, closes{};
    std::atomic<void*> protected_log{};
};
struct Case {
    unsigned int api{}, scenario{};
};
struct MappedJitLog {
    std::size_t bytes{};
    void* pointer{MAP_FAILED};
    MappedJitLog(const MappedJitLog&) = delete;
    MappedJitLog& operator=(const MappedJitLog&) = delete;
    MappedJitLog() {
        const auto page = ::sysconf(_SC_PAGESIZE);
        expect(page >= 80, "bounded log page geometry");
        bytes = static_cast<std::size_t>(page);
        pointer =
            ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        expect(pointer != MAP_FAILED, "dedicated late-publication page");
    }
    bool close() noexcept {
        if (pointer == MAP_FAILED) return true;
        if (::munmap(pointer, bytes) != 0) return false;
        pointer = MAP_FAILED;
        return true;
    }
    ~MappedJitLog() {
        if (!close()) std::cerr << "late-publication mapping cleanup failed\n";
    }
};
struct JitFixture {
    std::array<CUjit_option, 5> keys{CU_JIT_MAX_REGISTERS, CU_JIT_THREADS_PER_BLOCK,
                                     CU_JIT_WALL_TIME, CU_JIT_INFO_LOG_BUFFER,
                                     CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES};
    std::array<void*, 5> values{};
    std::array<std::byte, 80> log{};
    float time{-99.5F};
    bool has_zero_time{};
    explicit JitFixture(bool zero_time = false) : has_zero_time(zero_time) {
        log.fill(std::byte{0x77});
        std::fill(log.begin() + 8, log.end() - 8, std::byte{0xa5});
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        values[0] = reinterpret_cast<void*>(32);
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        values[1] = reinterpret_cast<void*>(128);
        // Deliberately not a usable address: float output is in the slot.
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        values[2] = reinterpret_cast<void*>(std::uintptr_t{0x51ac000000000000} |
                                            std::bit_cast<std::uint32_t>(time));
        if (has_zero_time) values[2] = nullptr;
        values[3] = log.data() + 8;
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        values[4] = reinterpret_cast<void*>(64);
    }
    void verify(bool published) const {
        expect(reinterpret_cast<std::uintptr_t>(values[0]) == 32 &&
                   reinterpret_cast<std::uintptr_t>(values[1]) == (published ? 256U : 128U) &&
                   reinterpret_cast<std::uintptr_t>(values[4]) == (published ? 4U : 64U) &&
                   reinterpret_cast<std::uintptr_t>(values[2]) ==
                       ((has_zero_time ? 0 : std::uintptr_t{0x51ac000000000000}) |
                        std::bit_cast<std::uint32_t>(published       ? 3.25F
                                                     : has_zero_time ? 0.F
                                                                     : -99.5F)) &&
                   values[3] == log.data() + 8 && time == -99.5F,
               "all scalar and pointer/float host slots are exact");
        for (std::size_t index = 0; index < log.size(); ++index) {
            const std::array<std::byte, 4> text{std::byte{'j'}, std::byte{'i'}, std::byte{'t'},
                                                std::byte{}};
            const auto expected = index < 8 || index >= 72  ? std::byte{0x77}
                                  : published && index < 12 ? text[index - 8]
                                                            : std::byte{0xa5};
            expect(log[index] == expected,
                   "complete log bytes, untouched tail and both guards survive");
        }
    }
};
std::vector<ExecutionJitOption> jit_input(bool zero_time = false) {
    const auto bits = std::bit_cast<std::array<std::byte, 4>>(zero_time ? 0.F : -99.5F);
    return {{0, 32, {}},
            {1, 128, {}},
            {2, 0, {bits.begin(), bits.end()}},
            {3, 0, std::vector<std::byte>(64, std::byte{0xa5})},
            {4, 64, {}}};
}
std::vector<ExecutionJitOption> jit_names_input(unsigned int variant) {
    const std::vector<std::byte> kernel_names{
        std::byte{2},   std::byte{},    std::byte{},    std::byte{},    std::byte{7},
        std::byte{},    std::byte{},    std::byte{},    std::byte{'k'}, std::byte{'e'},
        std::byte{'r'}, std::byte{'n'}, std::byte{'e'}, std::byte{'l'}, std::byte{},
        std::byte{1},   std::byte{},    std::byte{},    std::byte{},    std::byte{}};
    const std::vector<std::byte> variables{
        std::byte{1},   std::byte{},    std::byte{},    std::byte{},    std::byte{9},
        std::byte{},    std::byte{},    std::byte{},    std::byte{'v'}, std::byte{'a'},
        std::byte{'r'}, std::byte{'i'}, std::byte{'a'}, std::byte{'b'}, std::byte{'l'},
        std::byte{'e'}, std::byte{}};
    auto result = jit_input();
    result.push_back(
        {25, variant == 2 ? 0U : 1U, variant ? std::vector<std::byte>(4) : kernel_names});
    result.push_back({26, variant ? 0U : 2U, {}});
    result.push_back({27, 1, variables});
    result.push_back({28, 1, {}});
    return result;
}
ExecutionWireReply jit_response(unsigned int scenario) {
    auto options = jit_input();
    options[1].value = 256;
    const auto time = std::bit_cast<std::array<std::byte, 4>>(3.25F);
    std::ranges::copy(time, options[2].data.begin());
    std::memcpy(options[3].data.data(), "jit", 4);
    options[4].value = 4;
    ExecutionWireReply reply;
    reply.disposition = ExecutionWireDisposition::kComplete;
    reply.original_status = CUDA_SUCCESS;
    reply.id = 73;
    if (scenario == 20) options[0].value = 33;
    if (scenario == 21) options.pop_back();
    if (scenario == 22) std::swap(options[0], options[1]);
    if (scenario == 23) options[3].data.pop_back();
    if (scenario == 24) options[3].value = 0x10000;
    if (scenario == 26) {
        reply.status = CUDA_ERROR_INVALID_PTX;
        reply.original_status = reply.status;
        reply.id = 0;
    }
    if (scenario == 27 || scenario == 28 || scenario == 29) {
        reply.disposition = ExecutionWireDisposition::kRejected;
        reply.status = scenario == 27 ? CUDA_ERROR_OUT_OF_MEMORY : CUDA_ERROR_NOT_SUPPORTED;
        reply.id = 0;
        if (scenario != 27) reply.original_status.reset();
    }
    if (scenario == 30) reply.id = 0;
    if (scenario == 31) reply.original_status = CUDA_ERROR_INVALID_PTX;
    // A noncanonical pointer shape must still reach the frontend as legal JIT1
    // bytes, so write its hostile slot directly after a valid encoding.
    const auto pointer = options.size() > 3 ? options[3].value : 0;
    if (scenario == 24) options[3].value = 0;
    auto data = encode_execution_jit_options(options);
    expect(data.has_value(), "structurally valid hostile JIT option receipt");
    if (scenario == 24) {
        // Header8 + scalar16 + scalar16 + float20 + key4 = slot at64.
        for (std::size_t byte = 0; byte < 8; ++byte)
            (*data)[64 + byte] = std::byte{static_cast<unsigned char>(pointer >> (8 * byte))};
    }
    if (scenario != 29) reply.data = std::move(*data);
    return reply;
}
struct NamesReplyCase {
    unsigned int scenario{}, variant{};
};
ExecutionWireReply jit_names_response(NamesReplyCase selection) {
    const auto [scenario, variant] = selection;
    auto reply = jit_response(19);
    auto options = decode_execution_jit_options(reply.data).value();
    const auto requested = jit_names_input(variant);
    options.insert(options.end(), requested.begin() + 5, requested.end());
    if (scenario == 34) {
        reply.status = CUDA_ERROR_INVALID_PTX;
        reply.original_status = reply.status;
        reply.id = 0;
    }
    if (scenario == 35) options[5].data[8] = std::byte{'K'};
    if (scenario == 36) options[5].value = 0;
    if (scenario == 38) options[6].value = 3;
    if (scenario == 39) options.pop_back();
    const auto encoded = encode_execution_jit_options(options);
    expect(encoded.has_value(), "hostile names receipts remain structurally valid wire frames");
    reply.data = *encoded;
    return reply;
}
struct MappedJitNames {
    void* pointer{MAP_FAILED};
    std::size_t bytes{};
    const char** kernels{};
    const char** variables{};
    MappedJitNames() {
        const auto page = ::sysconf(_SC_PAGESIZE);
        expect(page >= 128, "name page geometry");
        bytes = static_cast<std::size_t>(page);
        pointer =
            ::mmap(nullptr, 2 * bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        expect(pointer != MAP_FAILED, "allocate owned readonly/guard name pages");
        auto* base = static_cast<char*>(pointer);
        kernels = static_cast<const char**>(pointer);
        variables = reinterpret_cast<const char**>(base + 32);
        std::memcpy(base + bytes - 7, "kernel", 7);
        std::memcpy(base + 64, "variable", 9);
        kernels[0] = base + bytes - 7;
        kernels[1] = base + bytes - 1;
        variables[0] = base + 64;
        expect(
            ::mprotect(base + bytes, bytes, PROT_NONE) == 0 &&
                ::mprotect(pointer, bytes, PROT_READ) == 0,
            "names and pointer arrays are readonly, final terminator borders an unreadable page");
    }
    ~MappedJitNames() {
        if (pointer != MAP_FAILED && ::munmap(pointer, 2 * bytes) != 0) std::terminate();
    }
};
void exercise_jit_names(const std::vector<Functions>& functions, CUcontext context,
                        unsigned int scenario, Control& control) {
    MappedJitNames names;
    unsigned int route{};
    for (const auto& entry : functions) {
        const auto variant = scenario == 36 ? 1U : scenario == 33 ? route++ % 3 : 0U;
        JitFixture original;
        std::vector<CUjit_option> keys(original.keys.begin(), original.keys.end());
        keys.insert(keys.end(),
                    {CU_JIT_REFERENCED_KERNEL_NAMES, CU_JIT_REFERENCED_KERNEL_COUNT,
                     CU_JIT_REFERENCED_VARIABLE_NAMES, CU_JIT_REFERENCED_VARIABLE_COUNT});
        std::vector<void*> values(original.values.begin(), original.values.end());
        // Non-NULL zero-count pointer is deliberately unreadable and must never be read or written.
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        auto* kernel_pointer = variant == 2 ? nullptr
                               : variant    ? reinterpret_cast<void*>(1)
                                            : names.kernels;
        // Native count bits occupy a void* slot, not a dereferenceable address.
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        auto* count_slot = reinterpret_cast<void*>(static_cast<std::uintptr_t>(variant ? 0U : 2U));
        values.insert(values.end(),
                      {kernel_pointer, count_slot, names.variables, reinterpret_cast<void*>(1)});
        const std::array name_slots{values[5], values[6], values[7], values[8]};
        CUmodule module = reinterpret_cast<CUmodule>(0x41);
        if (scenario == 33 && !variant) {
            const auto before = control.loads.load();
            auto invalid = values;
            invalid[6] = reinterpret_cast<void*>(33);
            expect(entry.extended(&module, kPtx.data(), keys.size(), keys.data(), invalid.data()) ==
                       CUDA_ERROR_INVALID_VALUE,
                   "bounded names count rejects before pointer snapshot or SDK intake");
            invalid = values;
            invalid[5] = reinterpret_cast<void*>(1);
            expect(entry.extended(&module, kPtx.data(), keys.size(), keys.data(), invalid.data()) ==
                       CUDA_ERROR_INVALID_VALUE,
                   "unreadable pointer array rejects without an original fallback");
            std::array<void*, 8> aliased{};
            aliased[0] = const_cast<char*>(names.kernels[0]);
            aliased[1] = const_cast<char*>(names.kernels[1]);
            invalid = values;
            invalid[3] = invalid[5] = aliased.data();
            expect(entry.extended(&module, kPtx.data(), keys.size(), keys.data(), invalid.data()) ==
                           CUDA_ERROR_INVALID_VALUE &&
                       control.loads == before && module == reinterpret_cast<CUmodule>(0x41),
                   "input/output aliases and malformed shapes do not submit or publish anything");
        }
        const auto invoke = [&] {
            return entry.extended(&module, kPtx.data(), keys.size(), keys.data(), values.data());
        };
        auto result = invoke();
        const bool published = scenario == 33 || scenario == 34;
        std::ranges::copy(std::span(values).first(5), original.values.begin());
        original.verify(published);
        expect(std::ranges::equal(std::span(values).subspan(5), name_slots) &&
                   std::strcmp(names.kernels[0], "kernel") == 0 && !*names.kernels[1] &&
                   std::strcmp(names.variables[0], "variable") == 0,
               "native error/success never copy back server pointers or readonly names");
        if (scenario == 34) {
            expect(result == CUDA_ERROR_INVALID_PTX && module == reinterpret_cast<CUmodule>(0x41),
                   "original error returns exact output effects without a module");
            JitFixture retry;
            std::ranges::copy(retry.values, values.begin());
            result = invoke();
            std::ranges::copy(std::span(values).first(5), retry.values.begin());
            retry.verify(true);
        }
        if (scenario == 33 || scenario == 34) {
            check(result);
            check(cuModuleUnload(module));
        } else {
            expect(result == CUDA_ERROR_UNKNOWN && module == reinterpret_cast<CUmodule>(0x41),
                   "lost/modified names provenance cannot publish a module or input bytes");
            const auto before = control.loads.load();
            expect(invoke() != CUDA_SUCCESS && cuCtxDestroy(context) != CUDA_SUCCESS &&
                       control.loads == before && !control.unloads && !control.closes,
                   "uncertain name receipts retain ownership without replay");
            return;
        }
    }
    check(cuCtxDestroy(context));
    expect(control.loads == (scenario == 33 ? 6U : 2U) &&
               control.unloads == (scenario == 33 ? 6U : 1U) && control.closes == 1,
           "complete direct/cached names factories retire exact logical modules and context");
}
void exercise_jit(const std::vector<Functions>& functions, CUcontext context, unsigned int scenario,
                  Control& control) {
    if (scenario == 32) {
        MappedJitLog backing;
        void* mapping = backing.pointer;
        JitFixture input;
        std::memcpy(mapping, input.log.data(), input.log.size());
        input.values[3] = static_cast<std::byte*>(mapping) + 8;
        control.protected_log = mapping;
        CUmodule module = reinterpret_cast<CUmodule>(0x41);
        const auto original_time_slot = input.values[2];
        const auto result =
            functions[0].extended(&module, kPtx.data(), 5, input.keys.data(), input.values.data());
        expect(result == CUDA_ERROR_INVALID_VALUE && module == reinterpret_cast<CUmodule>(0x41) &&
                   input.time == -99.5F && input.values[2] == original_time_slot &&
                   reinterpret_cast<std::uintptr_t>(input.values[1]) == 128 &&
                   std::memcmp(mapping, input.log.data(), input.log.size()) == 0,
               "late unwritable JIT output does not partially publish any local result");
        check(cuCtxDestroy(context));
        expect(control.closes == 1 && control.unloads == 0 && backing.close(),
               "known unpublished module remains context-owned until normal close");
        return;
    }
    unsigned int route{};
    for (const auto& entry : functions) {
        JitFixture input(scenario == 19 && route++ % 2 == 1);
        CUmodule module = reinterpret_cast<CUmodule>(0x41);
        const auto call = [&] {
            return entry.extended(&module, kPtx.data(),
                                  static_cast<unsigned int>(input.keys.size()), input.keys.data(),
                                  input.values.data());
        };
        if (scenario == 19) {
            const auto before = control.loads.load();
            CUjit_option known = CU_JIT_MAX_REGISTERS;
            expect(entry.extended(&module, kPtx.data(), 1, reinterpret_cast<CUjit_option*>(1),
                                  input.values.data()) == CUDA_ERROR_INVALID_VALUE &&
                       entry.extended(&module, kPtx.data(), 1, &known,
                                      reinterpret_cast<void**>(1)) == CUDA_ERROR_INVALID_VALUE,
                   "invalid key/value spans reject before JIT submission");
            auto duplicated = input.keys;
            duplicated[1] = duplicated[0];
            expect(entry.extended(&module, kPtx.data(), 5, duplicated.data(),
                                  input.values.data()) == CUDA_ERROR_INVALID_VALUE,
                   "duplicate option keys cannot reach compiler owner");
            auto aliased = input.values;
            aliased[3] = aliased.data();
            expect(entry.extended(&module, kPtx.data(), 5, input.keys.data(), aliased.data()) ==
                       CUDA_ERROR_INVALID_VALUE,
                   "overlapping output spans are an explicit pre-entry rejection");
            std::array<void*, 1> shared_arrays{};
            const std::uint32_t float_key = CU_JIT_WALL_TIME;
            std::memcpy(shared_arrays.data(), &float_key, sizeof(float_key));
            expect(entry.extended(&module, kPtx.data(), 1,
                                  reinterpret_cast<CUjit_option*>(shared_arrays.data()),
                                  shared_arrays.data()) == CUDA_ERROR_INVALID_VALUE,
                   "a float output slot cannot overwrite the aliased option-key array");
            expect(control.loads == before && module == reinterpret_cast<CUmodule>(0x41),
                   "invalid input changes neither remote nor local owner");
            input.verify(false);
        }
        const auto result = call();
        const bool published = scenario == 19 || scenario == 26 || scenario == 27;
        input.verify(published);
        if (scenario == 19) {
            check(result);
            check(cuModuleUnload(module));
            continue;
        }
        const auto expected = scenario == 26   ? CUDA_ERROR_INVALID_PTX
                              : scenario == 27 ? CUDA_ERROR_OUT_OF_MEMORY
                              : scenario == 29 ? CUDA_ERROR_NOT_SUPPORTED
                                               : CUDA_ERROR_UNKNOWN;
        expect(result == expected && module == reinterpret_cast<CUmodule>(0x41),
               "exact JIT failure never publishes a module");
        if (scenario == 26 || scenario == 29) {
            JitFixture retry;
            check(entry.extended(&module, kPtx.data(), static_cast<unsigned int>(retry.keys.size()),
                                 retry.keys.data(), retry.values.data()));
            retry.verify(true);
            check(cuModuleUnload(module));
            check(cuCtxDestroy(context));
        } else {
            const auto before = control.loads.load();
            expect(call() != CUDA_SUCCESS && cuCtxDestroy(context) != CUDA_SUCCESS &&
                       control.loads == before && !control.unloads && !control.closes,
                   "malformed/lost/partial JIT carrier retains origin without replay");
        }
        return;
    }
    check(cuCtxDestroy(context));
    expect(control.loads == 6 && control.unloads == 6 && control.closes == 1,
           "all qualified direct/cached JIT routes finish exact normal owner cleanup");
}
void exercise(const char* provider, const Case& selection, Control& control) {
    const auto api = selection.api;
    const auto scenario = selection.scenario;
    const auto functions = scenario == 0 || scenario == 19 || scenario == 33
                               ? routes(provider)
                               : std::vector<Functions>{{&cuModuleLoadData, &cuModuleLoadFatBinary,
                                                         &cuModuleLoadDataEx}};
    CUcontext context{};
    check(cuCtxCreate(&context, 0, 0));
    if (scenario >= 33) return exercise_jit_names(functions, context, scenario, control);
    if (scenario >= 19) return exercise_jit(functions, context, scenario, control);
    CUmodule module = reinterpret_cast<CUmodule>(0x41);
    if (scenario) {
        CUmodule first{};
        if (scenario == 12) check(load(functions[0], api, &first));
        const auto result = load(functions[0], api, &module);
        const bool clean_error = scenario == 13 || scenario == 14;
        expect(result == (scenario == 13 || scenario == 18 ? CUDA_ERROR_INVALID_PTX
                          : scenario == 14                 ? CUDA_ERROR_NOT_SUPPORTED
                          : scenario == 17                 ? CUDA_ERROR_OUT_OF_MEMORY
                                                           : CUDA_ERROR_UNKNOWN),
               "native/policy status or fail-closed malformed module receipt");
        expect(module == reinterpret_cast<CUmodule>(0x41), "failure never publishes a module");
        const auto before = control.loads.load();
        if (clean_error) {
            check(load(functions[0], api, &module));
            check(cuModuleUnload(module));
            check(cuCtxDestroy(context));
            expect(control.loads == before + 1 && control.unloads == 1 && control.closes == 1,
                   "clean failure allows one successful retry and normal owner cleanup");
        } else {
            expect(load(functions[0], api, &module) != CUDA_SUCCESS &&
                       cuCtxDestroy(context) != CUDA_SUCCESS && control.loads == before &&
                       !control.unloads && !control.closes,
                   "uncertainty retains owner without replay or untrusted release");
        }
        return;
    }
    const auto invalid = [&](const Functions& entry) {
        const auto before = control.loads.load();
        CUjit_option unsupported = CU_JIT_GLOBAL_SYMBOL_NAMES;
        expect(entry.extended(&module, kPtx.data(), 1, &unsupported, reinterpret_cast<void**>(1)) ==
                       CUDA_ERROR_NOT_SUPPORTED &&
                   entry.data(nullptr, kPtx.data()) == CUDA_ERROR_INVALID_VALUE &&
                   entry.fat(&module, reinterpret_cast<void*>(1)) == CUDA_ERROR_INVALID_IMAGE &&
                   module == reinterpret_cast<CUmodule>(0x41) && control.loads == before,
               "unsupported options and invalid spans never submit or mutate outputs");
    };
    for (const auto& entry : functions) {
        invalid(entry);
        for (unsigned int kind = 0; kind < 3; ++kind) {
            check(load(entry, kind, &module));
            expect(module && module != reinterpret_cast<CUmodule>(0x41), "publish opaque owner");
            check(cuModuleUnload(module));
            expect(cuModuleUnload(module) == CUDA_ERROR_INVALID_HANDLE, "no duplicate release");
            module = reinterpret_cast<CUmodule>(0x41);
        }
    }
    // An equal numeric remote identity in a different context is not an alias.
    CUmodule original{};
    check(load(functions[0], 0, &original));
    CUcontext other{};
    check(cuCtxCreate(&other, 0, 0));
    check(load(functions[0], 0, &module));
    expect(module != original && cuModuleUnload(original) == CUDA_ERROR_INVALID_HANDLE,
           "same remote ID is independently scoped to its context");
    check(cuModuleUnload(module));
    check(cuCtxDestroy(other));
    check(cuModuleUnload(original));
    check(cuCtxDestroy(context));
    expect(control.loads == 20 && control.unloads == 20 && control.closes == 2,
           "all six lookup routes plus context control have exact normal cleanup");
}
ExecutionWireReply response(unsigned int scenario) {
    ExecutionWireReply reply;
    reply.disposition = ExecutionWireDisposition::kComplete;
    reply.original_status = CUDA_SUCCESS;
    reply.id = 73;
    switch (scenario) {
        case 1:
            reply.id = 0;
            break;
        case 2:
            reply.address = 1;
            break;
        case 3:
            reply.parameter_bytes = 1;
            break;
        case 4:
            reply.parameter_bytes = 8;
            reply.arguments = {{0, 8}};
            break;
        case 5:
            reply.data = {std::byte{1}};
            break;
        case 6:
            reply.original_status.reset();
            break;
        case 7:
            reply.original_status = CUDA_ERROR_INVALID_VALUE;
            break;
        case 8:
            reply.disposition = ExecutionWireDisposition::kRejected;
            break;
        case 9:
            reply.disposition = ExecutionWireDisposition::kComputeReplacement;
            reply.original_status.reset();
            break;
        case 10:
            reply.status = -1;
            reply.original_status = -1;
            reply.id = 0;
            break;
        case 11:
            reply.status = 1000;
            reply.original_status = 1000;
            reply.id = 0;
            break;
        case 13:
            reply.status = CUDA_ERROR_INVALID_PTX;
            reply.original_status = reply.status;
            reply.id = 0;
            break;
        case 14:
            reply.disposition = ExecutionWireDisposition::kRejected;
            reply.status = CUDA_ERROR_NOT_SUPPORTED;
            reply.original_status.reset();
            reply.id = 0;
            break;
        case 16:
            reply.disposition = ExecutionWireDisposition::kAdmissionAccepted;
            reply.original_status.reset();
            reply.address = 1;
            break;
        case 17:
            // Actual original success, then failed private preparation: retained.
            reply.disposition = ExecutionWireDisposition::kRejected;
            reply.status = CUDA_ERROR_OUT_OF_MEMORY;
            reply.id = 0;
            break;
        case 18:
            reply.disposition = ExecutionWireDisposition::kRejected;
            reply.status = CUDA_ERROR_INVALID_PTX;
            reply.original_status = reply.status;
            reply.id = 0;
            break;
        default:
            break;
    }
    return reply;
}
void run(const char* provider, const Case& selection) {
    const auto scenario = selection.scenario;
    char pattern[] = "/tmp/glimmer-module-factory-XXXXXX";
    const auto* directory = ::mkdtemp(pattern);
    expect(directory, "private socket directory");
    ExecutionSocketConfig config;
    config.socket_path = std::string(directory) + "/service.sock";
    ExecutionSocketServer server(config);
    expect(server.start(), "start independently authenticated module owner");
    const auto identity = current_process_identity();
    expect(identity.has_value() &&
               ::setenv("GLIMMER_EXECUTION_SOCKET", config.socket_path.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_PID", std::to_string(identity->pid).c_str(), 1) ==
                   0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_START",
                        std::to_string(identity->start_time_ticks).c_str(), 1) == 0,
           "configure exact endpoint before first frontend use");
    Control control;
    auto client = std::async(std::launch::async, [&] { exercise(provider, selection, control); });
    const auto deadline = std::chrono::steady_clock::now() + 10s;
    try {
        while (client.wait_for(0ms) != std::future_status::ready) {
            expect(server.poll_for(1), "progress actual frontend IPC");
            if (auto request = server.take_request()) {
                ExecutionPayloadReader input(request->message.payload);
                auto reply = response(0);
                reply.id = 0;
                switch (request->message.header.operation) {
                    case ExecutionOpcode::kHello:
                        break;
                    case ExecutionOpcode::kLoadModuleJit: {
                        std::uint32_t kind{};
                        std::uint64_t bytes{};
                        std::span<const std::byte> encoded;
                        const auto variant = scenario == 36   ? 1U
                                             : scenario == 33 ? control.loads.load() % 3
                                                              : 0U;
                        const auto expected =
                            scenario >= 33
                                ? jit_names_input(variant)
                                : jit_input(scenario == 19 && control.loads.load() % 2 == 1);
                        expect(input.read_u32(kind) && kind == 2 && input.read_u64(bytes) &&
                                   bytes == kPtx.size() && input.read_blob(encoded) &&
                                   input.empty() &&
                                   decode_execution_jit_options(encoded) == expected,
                               "full independent JIT records and image identity cross actual IPC");
                        const auto seals = ::fcntl(request->descriptor.get(), F_GET_SEALS);
                        constexpr int required =
                            F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL;
                        std::vector<char> image(kPtx.size());
                        expect(seals >= 0 && (seals & required) == required &&
                                   ::pread(request->descriptor.get(), image.data(), image.size(),
                                           0) == static_cast<ssize_t>(image.size()) &&
                                   std::string_view(image.data(), image.size()) == kPtx,
                               "only a complete immutable owned image descriptor is submitted");
                        const auto number = ++control.loads;
                        if (scenario == 32)
                            expect(
                                ::mprotect(control.protected_log.load(),
                                           static_cast<std::size_t>(::sysconf(_SC_PAGESIZE)),
                                           PROT_READ) == 0,
                                "log becomes unwritable only after native-result request receipt");
                        if (scenario == 25 || scenario == 37) {
                            server.close_peer(request->connection);
                            continue;
                        }
                        reply = scenario >= 33
                                    ? jit_names_response({.scenario = number == 1 ? scenario : 33,
                                                          .variant = variant})
                                    : jit_response(scenario == 19 || number == 1 ? scenario : 19);
                        break;
                    }
                    case ExecutionOpcode::kLoadPtx:
                    case ExecutionOpcode::kLoadBinary: {
                        std::span<const std::byte> image;
                        expect(input.read_blob(image) && input.empty(),
                               "one complete image snapshot");
                        const auto binary = binary_image();
                        const auto expected =
                            request->message.header.operation == ExecutionOpcode::kLoadPtx
                                ? std::as_bytes(std::span{kPtx.data(), kPtx.size()})
                                : std::as_bytes(std::span{binary});
                        expect(std::ranges::equal(image, expected),
                               "all module bytes independently match");
                        const auto count = ++control.loads;
                        if (scenario == 15) {
                            server.close_peer(request->connection);
                            continue;
                        }
                        reply = response((scenario == 12 ? count == 2 : count == 1) ? scenario : 0);
                        break;
                    }
                    case ExecutionOpcode::kUnloadModule: {
                        std::uint64_t id{};
                        expect(input.read_u64(id) && id == 73 && input.empty(),
                               "exact owned module release");
                        ++control.unloads;
                        break;
                    }
                    case ExecutionOpcode::kClose:
                        ++control.closes;
                        break;
                    default:
                        throw std::runtime_error("unexpected module operation or replay");
                }
                auto header = request->message.header;
                header.kind = ExecutionMessageKind::kReply;
                header.session = request->connection;
                header.incarnation = {31, 37};
                const auto bytes = encode_execution_reply(reply);
                expect(bytes.has_value() && decode_execution_reply(*bytes).has_value() &&
                           server.reply(request->connection, {header, *bytes}),
                       "valid wire envelope carries independent factory receipt");
            }
            expect(std::chrono::steady_clock::now() < deadline, "bounded module factory fixture");
        }
        client.get();
    } catch (...) {
        server.stop();
        (void)::rmdir(directory);
        throw;
    }
    server.stop();
    expect(::rmdir(directory) == 0 && glimmer_module_factory_native_calls() == 0,
           "no original caller-side factory calls; remove closed private socket directory");
}
}  // namespace
int main(int argc, char** argv) {
    try {
        expect(argc == 4, "provider, API index and independently executed receipt scenario");
        const auto api = static_cast<unsigned int>(std::stoul(argv[2]));
        const auto scenario = static_cast<unsigned int>(std::stoul(argv[3]));
        expect(api < 3 && scenario <= kLastCase, "bounded factory controls");
        run(argv[1], {api, scenario});
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
#endif
