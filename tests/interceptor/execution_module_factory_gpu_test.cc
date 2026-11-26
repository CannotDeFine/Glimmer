#include <cuda.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <span>
#include <dlfcn.h>

namespace {
constexpr std::uint32_t kCount = 32768;
constexpr std::size_t kGuard = 32;
constexpr std::uint32_t kSentinel = 0x91ae73c5;
constexpr std::string_view kPtx = R"ptx(
.version 7.0
.target sm_70
.address_size 64
.visible .entry binary_add(.param .u64 output, .param .u32 count,
                          .param .align 16 .b8 values[16]) {
 .reg .pred %p;
 .reg .b32 %r<7>;
 .reg .b64 %rd<4>;
 ld.param.u64 %rd1, [output];
 ld.param.u32 %r1, [count];
 ld.param.u32 %r2, [values];
 mov.u32 %r3, %ctaid.x; mov.u32 %r4, %ntid.x; mov.u32 %r5, %tid.x;
 mad.lo.u32 %r3, %r3, %r4, %r5;
 setp.ge.u32 %p, %r3, %r1; @%p bra DONE;
 mul.wide.u32 %rd2, %r3, 4; add.u64 %rd3, %rd1, %rd2;
 ld.global.u32 %r6, [%rd3]; add.u32 %r6, %r6, %r2;
 st.global.u32 [%rd3], %r6;
DONE: ret;
}
)ptx";
constexpr std::string_view kGlobalPtx = R"ptx(
.version 7.0
.target sm_70
.address_size 64
.visible .global .align 4 .u32 jit_global = 17;
.visible .entry jit_global_add(.param .u64 output, .param .u32 count) {
 .reg .pred %p;
 .reg .b32 %r<7>;
 .reg .b64 %rd<4>;
 ld.param.u64 %rd1, [output]; ld.param.u32 %r1, [count];
 mov.u32 %r2, %ctaid.x; mov.u32 %r3, %ntid.x; mov.u32 %r4, %tid.x;
 mad.lo.u32 %r2, %r2, %r3, %r4;
 setp.ge.u32 %p, %r2, %r1; @%p bra DONE;
 mul.wide.u32 %rd2, %r2, 4; add.u64 %rd3, %rd1, %rd2;
 ld.global.u32 %r5, [jit_global]; ld.global.u32 %r6, [%rd3];
 add.u32 %r6, %r6, %r5; st.global.u32 [%rd3], %r6;
DONE: ret;
}
.visible .entry jit_global_advance() {
 .reg .pred %p;
 .reg .b32 %r<4>;
 mov.u32 %r1, %ctaid.x; mov.u32 %r2, %tid.x;
 or.b32 %r1, %r1, %r2; setp.ne.u32 %p, %r1, 0; @%p bra DONE;
 atom.global.add.u32 %r3, [jit_global], 1;
DONE: ret;
}
)ptx";
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void check(CUresult value, const char* operation) {
    if (value != CUDA_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": CUDA " + std::to_string(value));
}
struct Functions {
    decltype(&cuModuleLoadData) data{&cuModuleLoadData};
    decltype(&cuModuleLoadFatBinary) fat{&cuModuleLoadFatBinary};
    decltype(&cuModuleLoadDataEx) extended{&cuModuleLoadDataEx};
    void* library{};
    explicit Functions(bool cached) {
        if (!cached) return;
        library = ::dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
        expect(library, "open original Driver provider");
        data = reinterpret_cast<decltype(data)>(::dlsym(library, "cuModuleLoadData"));
        fat = reinterpret_cast<decltype(fat)>(::dlsym(library, "cuModuleLoadFatBinary"));
        extended = reinterpret_cast<decltype(extended)>(::dlsym(library, "cuModuleLoadDataEx"));
        if (!data || !fat || !extended) {
            const auto closed = close();
            throw std::runtime_error(closed ? "resolve original module factories"
                                            : "resolve and close Driver provider");
        }
    }
    Functions(const Functions&) = delete;
    Functions& operator=(const Functions&) = delete;
    bool close() noexcept {
        if (!library) return true;
        auto* previous = library;
        library = nullptr;
        return ::dlclose(previous) == 0;
    }
    ~Functions() {
        if (!close()) std::cerr << "module factory provider cleanup failed\n";
    }
};
struct Resources {
    CUcontext context{};
    CUmodule module{};
    CUdeviceptr allocation{};
    bool cleanup_attempted{}, cleanup_ok{true};
    bool close() noexcept {
        if (cleanup_attempted) return cleanup_ok;
        cleanup_attempted = true;
        const auto observed = [&](CUresult value) {
            if (value != CUDA_SUCCESS) {
                cleanup_ok = false;
                std::cerr << "module factory cleanup: CUDA " << value << '\n';
            }
            return value == CUDA_SUCCESS;
        };
        if (context && !observed(cuCtxSynchronize())) return false;
        if (allocation && !observed(cuMemFree(allocation))) return false;
        allocation = 0;
        if (module && !observed(cuModuleUnload(module))) return false;
        module = nullptr;
        if (context && !observed(cuCtxDestroy(context))) return false;
        context = nullptr;
        return cleanup_ok;
    }
    ~Resources() {
        close();
    }
};
std::vector<char> read_image(const std::string& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    expect(input.good(), "open compiler image");
    const auto size = input.tellg();
    expect(size > 0 && size <= std::streamoff{1024} * 1024, "bounded compiler image");
    std::vector<char> result(static_cast<std::size_t>(size));
    input.seekg(0);
    input.read(result.data(), static_cast<std::streamsize>(result.size()));
    expect(input.good(), "read complete compiler image");
    input.close();
    expect(!input.fail(), "close compiler image");
    return result;
}
struct alignas(16) Values {
    std::uint32_t add{}, marker{}, unused[2]{};
};
std::uint64_t checksum(const std::vector<std::uint32_t>& values) {
    std::uint64_t result = 14695981039346656037ULL;
    for (const auto value : values)
        for (unsigned int shift = 0; shift < 32; shift += 8)
            result = (result ^ ((value >> shift) & 255U)) * 1099511628211ULL;
    return result;
}
std::string hex_bytes(const char* pointer, std::size_t count) {
    constexpr std::string_view digits = "0123456789abcdef";
    std::string result;
    result.reserve(count * 2);
    for (std::size_t index = 0; index < count; ++index) {
        const auto byte = static_cast<unsigned char>(pointer[index]);
        result.push_back(digits[byte >> 4]);
        result.push_back(digits[byte & 15]);
    }
    return result;
}
enum class JitReferences { kNone, kNames, kNull };
struct JitStorage {
    static constexpr std::array<CUjit_option, 9> k_keys{CU_JIT_MAX_REGISTERS,
                                                        CU_JIT_THREADS_PER_BLOCK,
                                                        CU_JIT_OPTIMIZATION_LEVEL,
                                                        CU_JIT_INFO_LOG_BUFFER,
                                                        CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
                                                        CU_JIT_ERROR_LOG_BUFFER,
                                                        CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
                                                        CU_JIT_WALL_TIME,
                                                        CU_JIT_LOG_VERBOSE};
    std::vector<CUjit_option> keys{k_keys.begin(), k_keys.end()};
    std::vector<void*> values;
    static constexpr std::array<const char*, 2> k_kernel_names{"binary_*", "jit_global_*"};
    static constexpr std::array<const char*, 1> k_variable_names{"jit_global"};
    std::array<const char*, 2> kernel_names{k_kernel_names};
    std::array<const char*, 1> variable_names{k_variable_names};
    JitReferences references{};
    std::array<char, 1088> info{};
    std::array<char, 576> error{};
    std::uint32_t before{0x5c197a83};
    float wall_time{-99.5F};
    std::uint32_t after{0x1ad7629b};
    explicit JitStorage(JitReferences requested = JitReferences::kNone) : references(requested) {
        info.fill('w');
        error.fill('w');
        std::fill(info.begin() + 32, info.end() - 32, static_cast<char>(0xa5));
        std::fill(error.begin() + 32, error.end() - 32, static_cast<char>(0x6b));
        // Scalar-in-slot inputs follow the original CUDA Driver ABI.
        // NOLINTBEGIN(performance-no-int-to-ptr)
        values = {reinterpret_cast<void*>(64),
                  reinterpret_cast<void*>(32),
                  reinterpret_cast<void*>(3),
                  info.data() + 32,
                  reinterpret_cast<void*>(1024),
                  error.data() + 32,
                  reinterpret_cast<void*>(512),
                  reinterpret_cast<void*>(std::uintptr_t{0x51ac000000000000} |
                                          std::bit_cast<std::uint32_t>(wall_time)),
                  reinterpret_cast<void*>(1)};
        if (references != JitReferences::kNone) {
            keys.insert(keys.end(),
                        {CU_JIT_REFERENCED_KERNEL_NAMES, CU_JIT_REFERENCED_KERNEL_COUNT,
                         CU_JIT_REFERENCED_VARIABLE_NAMES, CU_JIT_REFERENCED_VARIABLE_COUNT});
            values.insert(
                values.end(),
                {references == JitReferences::kNull ? nullptr : kernel_names.data(),
                 references == JitReferences::kNull ? nullptr : reinterpret_cast<void*>(2),
                 references == JitReferences::kNull ? nullptr : variable_names.data(),
                 references == JitReferences::kNull ? nullptr : reinterpret_cast<void*>(1)});
        }
        // NOLINTEND(performance-no-int-to-ptr)
    }
    std::string receipt(unsigned int format, CUresult status) const {
        const auto wall_slot = reinterpret_cast<std::uintptr_t>(values[7]);
        const auto elapsed = std::bit_cast<float>(static_cast<std::uint32_t>(wall_slot));
        expect(keys.size() == (references == JitReferences::kNone ? 9U : 13U) &&
                   std::ranges::equal(std::span(keys).first(9), k_keys) &&
                   reinterpret_cast<std::uintptr_t>(values[0]) == 64 &&
                   reinterpret_cast<std::uintptr_t>(values[2]) == 3 &&
                   reinterpret_cast<std::uintptr_t>(values[8]) == 1 &&
                   values[3] == info.data() + 32 && values[5] == error.data() + 32 &&
                   (wall_slot >> 32) == 0x51ac0000 && wall_time == -99.5F && before == 0x5c197a83 &&
                   after == 0x1ad7629b,
               "JIT input slots, unused float slot bytes and unrelated storage remain unchanged");
        if (references != JitReferences::kNone) {
            const bool empty = references == JitReferences::kNull;
            expect(keys[9] == CU_JIT_REFERENCED_KERNEL_NAMES &&
                       keys[10] == CU_JIT_REFERENCED_KERNEL_COUNT &&
                       keys[11] == CU_JIT_REFERENCED_VARIABLE_NAMES &&
                       keys[12] == CU_JIT_REFERENCED_VARIABLE_COUNT &&
                       values[9] == (empty ? nullptr : kernel_names.data()) &&
                       values[11] == (empty ? nullptr : variable_names.data()) &&
                       reinterpret_cast<std::uintptr_t>(values[10]) == (empty ? 0U : 2U) &&
                       reinterpret_cast<std::uintptr_t>(values[12]) == (empty ? 0U : 1U) &&
                       kernel_names == k_kernel_names && variable_names == k_variable_names,
                   "input-only reference names, counts and pointer arrays remain caller-owned");
        }
        for (const auto* buffer : {&info})
            expect(std::all_of(buffer->begin(), buffer->begin() + 32,
                               [](char byte) { return byte == 'w'; }) &&
                       std::all_of(buffer->end() - 32, buffer->end(),
                                   [](char byte) { return byte == 'w'; }),
                   "both complete info guards survive");
        expect(
            std::all_of(error.begin(), error.begin() + 32, [](char byte) { return byte == 'w'; }) &&
                std::all_of(error.end() - 32, error.end(), [](char byte) { return byte == 'w'; }) &&
                std::isfinite(elapsed) && (elapsed == -99.5F || elapsed >= 0),
            "both error guards and actual wall-time output are valid");
        const std::string_view information(info.data() + 32, 1024), errors(error.data() + 32, 512);
        expect(information.find("__glimmer") == std::string_view::npos &&
                   errors.find("__glimmer") == std::string_view::npos,
               "private compilation cannot overwrite original SDK diagnostics");
        return "{\"format\":" + std::to_string(format) + ",\"status\":" + std::to_string(status) +
               ",\"thread_slot\":" + std::to_string(reinterpret_cast<std::uintptr_t>(values[1])) +
               ",\"info_size\":" + std::to_string(reinterpret_cast<std::uintptr_t>(values[4])) +
               ",\"error_size\":" + std::to_string(reinterpret_cast<std::uintptr_t>(values[6])) +
               ",\"wall_bits\":" + std::to_string(static_cast<std::uint32_t>(wall_slot)) +
               ",\"info\":\"" + hex_bytes(information.data(), information.size()) +
               "\",\"error\":\"" + hex_bytes(errors.data(), errors.size()) +
               "\",\"guards\":true,\"input_slots\":true}";
    }
};
void run(const std::string& directory, bool cached, bool jit, bool referenced_names) {
    Functions functions(cached);
    Resources resources;
    check(cuInit(0), "initialize Driver");
    check(cuCtxCreate(&resources.context, 0, 0), "create context");
    std::vector<std::uint32_t> output(kCount + 2 * kGuard, kSentinel);
    check(cuMemAlloc(&resources.allocation, output.size() * sizeof(output[0])),
          "allocate all guards");
    std::vector<std::uint64_t> checksums;
    std::vector<std::string> jit_receipts;
    std::uint64_t global_checksum{};
    std::string null_names_receipt;
    const auto references = referenced_names ? JitReferences::kNames : JitReferences::kNone;
    if (referenced_names) {
        JitStorage storage(JitReferences::kNull);
        const auto status = functions.extended(&resources.module, kPtx.data(),
                                               static_cast<unsigned int>(storage.keys.size()),
                                               storage.keys.data(), storage.values.data());
        expect(status == CUDA_ERROR_INVALID_VALUE && !resources.module,
               "original SDK rejects NULL name arrays even with zero counts");
        null_names_receipt = storage.receipt(6, status);
    }
    if (jit) {
        auto invalid = std::string(kPtx);
        const auto instruction = invalid.find("ld.global.u32");
        expect(instruction != std::string::npos, "fixed invalid-instruction compiler control");
        invalid.replace(instruction, std::string_view("ld.global.u32").size(),
                        "invalid_opcode.u32");
        JitStorage storage(references);
        const auto result = functions.extended(&resources.module, invalid.c_str(),
                                               static_cast<unsigned int>(storage.keys.size()),
                                               storage.keys.data(), storage.values.data());
        if (result != CUDA_SUCCESS && resources.module) {
            // Do not pass an unproven partial output to module unload. The exact
            // isolated context remains the authoritative cleanup owner.
            resources.module = nullptr;
            throw std::runtime_error("partial compiler failure retained for context shutdown");
        }
        expect(result == CUDA_ERROR_INVALID_PTX && !resources.module,
               "original compiler error is reported with no invented module");
        jit_receipts.push_back(storage.receipt(4, result));
    }
    std::size_t case_number{};
    for (unsigned int api = 0; api < 3; ++api) {
        for (unsigned int format = 0; format < 4; ++format) {
            // Match only formats genuinely documented/accepted by the original
            // FatBinary entry. The common framework parser is not SDK parity.
            if (api == 1 && format < 2) continue;
            const std::array names{"", "kernel.cubin", "kernel.fatbin",
                                   "kernel-uncompressed.fatbin"};
            auto image = format ? read_image(directory + "/" + names[format])
                                : std::vector<char>(kPtx.begin(), kPtx.end());
            if (!format) image.push_back('\0');
            JitStorage storage(references);
            const auto result =
                api == 0 ? functions.data(&resources.module, image.data())
                : api == 1
                    ? functions.fat(&resources.module, image.data())
                    : functions.extended(&resources.module, image.data(),
                                         jit ? static_cast<unsigned int>(storage.keys.size()) : 0,
                                         jit ? storage.keys.data() : nullptr,
                                         jit ? storage.values.data() : nullptr);
            check(result, "load original module image");
            if (jit && api == 2) jit_receipts.push_back(storage.receipt(format, result));
            std::fill(image.begin(), image.end(), 'X');
            CUfunction function{};
            check(cuModuleGetFunction(&function, resources.module, "binary_add"),
                  "find copied-image kernel");
            std::fill(output.begin(), output.end(), kSentinel);
            for (std::size_t index = 0; index < kCount; ++index)
                output[kGuard + index] = static_cast<std::uint32_t>(3 * index + 7);
            check(cuMemcpyHtoD(resources.allocation, output.data(),
                               output.size() * sizeof(output[0])),
                  "upload all input and guard bytes");
            const auto first = static_cast<std::uint32_t>(11 + 2 * case_number);
            const auto second = static_cast<std::uint32_t>(17 + case_number);
            for (const auto add : {first, second}) {
                auto pointer = resources.allocation + kGuard * sizeof(output[0]);
                auto count = kCount;
                Values values{add, 0, {0, 0}};
                std::array<void*, 3> arguments{&pointer, &count, &values};
                check(cuLaunchKernel(function, 1024, 1, 1, 32, 1, 1, 0, nullptr, arguments.data(),
                                     nullptr),
                      "launch copied-image kernel");
                values.add = 0xdeadbeef;
                count = 0;
                pointer = 0;
            }
            check(cuCtxSynchronize(), "physically complete both launches");
            check(cuMemcpyDtoH(output.data(), resources.allocation,
                               output.size() * sizeof(output[0])),
                  "download all output and guard bytes");
            for (std::size_t index = 0; index < output.size(); ++index) {
                const auto expected =
                    index >= kGuard && index < kGuard + kCount
                        ? static_cast<std::uint32_t>(3 * (index - kGuard) + 7 + first + second)
                        : kSentinel;
                expect(output[index] == expected,
                       "every non-idempotent value and both guards agree");
            }
            checksums.push_back(checksum(output));
            check(cuModuleUnload(resources.module), "unload exact image owner");
            resources.module = nullptr;
            ++case_number;
        }
    }
    if (jit) {
        auto source = std::string(kGlobalPtx);
        JitStorage storage(references);
        check(functions.extended(&resources.module, source.c_str(),
                                 static_cast<unsigned int>(storage.keys.size()),
                                 storage.keys.data(), storage.values.data()),
              "compile shared-state JIT module");
        jit_receipts.push_back(storage.receipt(5, CUDA_SUCCESS));
        std::fill(source.begin(), source.end(), 'X');
        CUdeviceptr global{};
        std::size_t global_bytes{};
        check(cuModuleGetGlobal(&global, &global_bytes, resources.module, "jit_global"),
              "find public JIT module state");
        expect(global_bytes == sizeof(std::uint32_t), "exact global extent");
        std::uint32_t word{};
        check(cuMemcpyDtoH(&word, global, sizeof(word)), "read exact original module initializer");
        expect(word == 17, "JIT preserves initialized module data");
        word = 29;
        check(cuMemcpyHtoD(global, &word, sizeof(word)), "initialize public module state");
        std::fill(output.begin(), output.end(), kSentinel);
        for (std::size_t index = 0; index < kCount; ++index)
            output[kGuard + index] = static_cast<std::uint32_t>(3 * index + 7);
        check(cuMemcpyHtoD(resources.allocation, output.data(), output.size() * sizeof(output[0])),
              "upload stateful output guards");
        CUfunction add{}, advance{};
        check(cuModuleGetFunction(&add, resources.module, "jit_global_add"),
              "find stateful read kernel");
        check(cuModuleGetFunction(&advance, resources.module, "jit_global_advance"),
              "find stateful write kernel");
        auto pointer = resources.allocation + kGuard * sizeof(output[0]);
        auto count = kCount;
        std::array<void*, 2> arguments{&pointer, &count};
        check(cuLaunchKernel(add, 1024, 1, 1, 32, 1, 1, 0, nullptr, arguments.data(), nullptr),
              "read public state through selected kernel");
        check(cuLaunchKernel(advance, 1024, 1, 1, 32, 1, 1, 0, nullptr, nullptr, nullptr),
              "write public state exactly once");
        check(cuCtxSynchronize(), "join stateful JIT kernels");
        check(cuMemcpyDtoH(output.data(), resources.allocation, output.size() * sizeof(output[0])),
              "download complete stateful output");
        check(cuMemcpyDtoH(&word, global, sizeof(word)), "read public state after private kernel");
        expect(word == 30, "public and selected kernels share the same initialized module state");
        for (std::size_t index = 0; index < output.size(); ++index) {
            const auto expected = index >= kGuard && index < kGuard + kCount
                                      ? static_cast<std::uint32_t>(3 * (index - kGuard) + 36)
                                      : kSentinel;
            expect(output[index] == expected,
                   "every stateful output and guard has the public-state value");
        }
        global_checksum = checksum(output);
        check(cuModuleUnload(resources.module), "unload stateful module");
        resources.module = nullptr;
    }
    expect(case_number == 10 && resources.close() && functions.close(),
           "complete normal owner cleanup");
    std::cout << "{\"status\":\"passed\",\"cached\":" << (cached ? "true" : "false")
              << ",\"cases\":10,\"kernel_calls\":20,\"checked_values\":328320,\"checksums\":[";
    for (std::size_t index = 0; index < checksums.size(); ++index) {
        if (index) std::cout << ',';
        std::cout << checksums[index];
    }
    std::cout << ']';
    if (referenced_names)
        std::cout << ",\"referenced_names\":true,\"jit_names_null\":" << null_names_receipt;
    if (jit) {
        expect(jit_receipts.size() == 6, "all six complete JIT output receipts retained");
        std::cout << ",\"jit_global\":{\"checked_values\":32832,\"kernel_calls\":2,\"transfer_"
                     "calls\":5,\"global_initial\":17,\"global_after\":30,\"checksum\":"
                  << global_checksum << ",\"guards\":true}";
        std::cout << ",\"jit\":true,\"jit_outputs\":[";
        for (std::size_t index = 0; index < jit_receipts.size(); ++index) {
            if (index) std::cout << ',';
            std::cout << jit_receipts[index];
        }
        std::cout << ']';
    }
    std::cout << ",\"image_overwritten\":true,\"cleanup\":true,\"full_device_qualified\":false,"
                 "\"priority_qualified\":false,\"performance_qualified\":false}\n";
}
}  // namespace
int main(int argc, char** argv) {
    try {
        expect(argc >= 2 && argc <= 5,
               "compiler image directory and optional cached/JIT modes required");
        bool cached{}, jit{}, referenced_names{};
        for (int index = 2; index < argc; ++index) {
            const std::string_view option(argv[index]);
            if (option == "--cached" && !cached)
                cached = true;
            else if (option == "--jit" && !jit)
                jit = true;
            else if (option == "--jit-names" && !referenced_names)
                referenced_names = true;
            else
                throw std::runtime_error("unknown/duplicate module SDK option");
        }
        run(argv[1], cached, jit || referenced_names, referenced_names);
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
