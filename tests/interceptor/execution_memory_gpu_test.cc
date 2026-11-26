#include <cuda.h>
#include <cuda_runtime_api.h>
#include "execution_cached_driver_fixture.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <dlfcn.h>

namespace {
constexpr unsigned int kWorkers = 320, kThreads = 32, kElements = kWorkers * kThreads;
constexpr std::size_t kRegion = 128, kPayload = 64, kPatternBytes = 16 * kRegion;
constexpr const char* kSource = R"ptx(
.version 7.0
.target sm_70
.address_size 64
.visible .entry add(.param .u64 output) {
 .reg .b32 %r<6>; .reg .b64 %rd<4>;
 ld.param.u64 %rd1, [output];
 mov.u32 %r1, %ctaid.x; mov.u32 %r2, %ntid.x; mov.u32 %r3, %tid.x;
 mad.lo.u32 %r4, %r1, %r2, %r3; mul.wide.u32 %rd2, %r4, 4;
 add.u64 %rd3, %rd1, %rd2; ld.global.u32 %r5, [%rd3];
 add.u32 %r5, %r5, 1; st.global.u32 [%rd3], %r5; ret;
}
.visible .entry hold(.param .u64 duration) {
 .reg .pred %p; .reg .b64 %rd<5>;
 ld.param.u64 %rd1, [duration]; mov.u64 %rd2, %globaltimer;
WAIT:
 mov.u64 %rd3, %globaltimer; sub.u64 %rd4, %rd3, %rd2;
 setp.lt.u64 %p, %rd4, %rd1; @%p bra WAIT; ret;
}
)ptx";
void check(CUresult value, const char* operation) {
    if (value != CUDA_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": CUDA " + std::to_string(value));
}
void runtime_check(cudaError_t value, const char* operation) {
    if (value != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": Runtime " + std::to_string(value));
}
void* runtime_pointer(CUdeviceptr address) {
    // Convert numeric GPU UVA for the public Runtime ABI, never a host object.
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    return reinterpret_cast<void*>(address);
}
struct Library {
    void* value{};
    explicit Library(const char* name) : value(::dlopen(name, RTLD_NOW | RTLD_LOCAL)) {
        if (!value) throw std::runtime_error(std::string("load SDK scope: ") + name);
    }
    ~Library() {
        if (::dlclose(value) != 0) std::terminate();
    }
    Library(const Library&) = delete;
    Library& operator=(const Library&) = delete;
};
struct MemoryApi {
    decltype(&cuMemcpyDtoD) copy{&cuMemcpyDtoD};
    decltype(&cuMemcpyDtoDAsync) copy_async{&cuMemcpyDtoDAsync};
    decltype(&cuMemsetD8) fill8{&cuMemsetD8};
    decltype(&cuMemsetD8Async) fill8_async{&cuMemsetD8Async};
    decltype(&cuMemsetD16) fill16{&cuMemsetD16};
    decltype(&cuMemsetD16Async) fill16_async{&cuMemsetD16Async};
    decltype(&cuMemsetD32) fill32{&cuMemsetD32};
    decltype(&cuMemsetD32Async) fill32_async{&cuMemsetD32Async};
    decltype(&cudaMemcpy) runtime_copy{&cudaMemcpy};
    decltype(&cudaMemcpyAsync) runtime_copy_async{&cudaMemcpyAsync};
    decltype(&cudaMemset) runtime_fill{&cudaMemset};
    decltype(&cudaMemsetAsync) runtime_fill_async{&cudaMemsetAsync};
    void bind(bool is_cached) {
        if (!is_cached) return;
        Library driver("libcuda.so.1"), runtime("libcudart.so.12");
        const auto query = reinterpret_cast<decltype(&cuGetProcAddress_v2)>(
            ::dlsym(driver.value, "cuGetProcAddress_v2"));
        if (!query) throw std::runtime_error("resolve public Driver memory metadata");
        const auto resolve = [&](const char* name) {
            void* value{};
            CUdriverProcAddressQueryResult status{};
            check(query(name, &value, 12000, 0, &status), name);
            if (!value || status != CU_GET_PROC_ADDRESS_SUCCESS)
                throw std::runtime_error(std::string("missing controlled Driver memory entry: ") +
                                         name);
            return value;
        };
#define DRIVER_MEMORY(FIELD, NAME) FIELD = reinterpret_cast<decltype(FIELD)>(resolve(NAME))
        DRIVER_MEMORY(copy, "cuMemcpyDtoD");
        DRIVER_MEMORY(copy_async, "cuMemcpyDtoDAsync");
        DRIVER_MEMORY(fill8, "cuMemsetD8");
        DRIVER_MEMORY(fill8_async, "cuMemsetD8Async");
        DRIVER_MEMORY(fill16, "cuMemsetD16");
        DRIVER_MEMORY(fill16_async, "cuMemsetD16Async");
        DRIVER_MEMORY(fill32, "cuMemsetD32");
        DRIVER_MEMORY(fill32_async, "cuMemsetD32Async");
#undef DRIVER_MEMORY
#define RUNTIME_MEMORY(FIELD, NAME)                                          \
    FIELD = reinterpret_cast<decltype(FIELD)>(::dlsym(runtime.value, NAME)); \
    if (!(FIELD))                                                            \
    throw std::runtime_error(std::string("missing controlled Runtime memory entry: ") + (NAME))
        RUNTIME_MEMORY(runtime_copy, "cudaMemcpy");
        RUNTIME_MEMORY(runtime_copy_async, "cudaMemcpyAsync");
        RUNTIME_MEMORY(runtime_fill, "cudaMemset");
        RUNTIME_MEMORY(runtime_fill_async, "cudaMemsetAsync");
#undef RUNTIME_MEMORY
    }
};
void run(bool is_cached, bool is_instrumented) {
    ExecutionCachedDriver driver;
    driver.bind(is_cached);
    MemoryApi memory;
    memory.bind(is_cached);
    check(driver.initialize(0), "initialize");
    CUdevice device{};
    check(driver.get_device(&device, 0), "device");
    int sms{};
    check(driver.device_attribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device),
          "SM availability");
    if (sms != 80) throw std::runtime_error("explicit V100 fixture requires 80 available SMs");
    CUcontext context{};
    CUmodule module{};
    CUfunction add{}, hold{};
    CUdeviceptr source{}, destination{}, patterns{};
    CUstream stream{}, independent{};
    check(driver.create_context(&context, 0, device), "context");
    check(driver.load_module(&module, kSource), "unchanged original module");
    check(driver.get_function(&add, module, "add"), "original producer/consumer");
    check(driver.get_function(&hold, module, "hold"), "original independent tail");
    check(driver.allocate(&source, kElements * sizeof(std::uint32_t)), "source allocation");
    check(driver.allocate(&destination, kElements * sizeof(std::uint32_t)),
          "destination allocation");
    check(driver.allocate(&patterns, kPatternBytes), "guarded pattern allocation");
    check(driver.create_stream(&stream, CU_STREAM_DEFAULT), "blocking application stream");
    check(driver.create_stream(&independent, CU_STREAM_NON_BLOCKING),
          "independent application stream");
    auto* runtime_patterns = runtime_pointer(patterns);
    for (CUdeviceptr pointer : {CUdeviceptr{}, patterns}) {
        check(memory.copy(pointer, pointer, 0), "zero ordinary copy");
        check(memory.copy_async(pointer, pointer, 0, stream), "zero async copy");
        check(memory.fill8(pointer, 7, 0), "zero byte fill");
        check(memory.fill16(pointer, 7, 0), "zero 16-bit fill");
        check(memory.fill32(pointer, 7, 0), "zero 32-bit fill");
        check(memory.fill8_async(pointer, 7, 0, stream), "zero async byte fill");
        check(memory.fill16_async(pointer, 7, 0, stream), "zero async 16-bit fill");
        check(memory.fill32_async(pointer, 7, 0, stream), "zero async 32-bit fill");
        runtime_check(memory.runtime_fill(runtime_pointer(pointer), 7, 0), "zero Runtime fill");
        runtime_check(memory.runtime_fill_async(runtime_pointer(pointer), 7, 0,
                                                reinterpret_cast<cudaStream_t>(stream)),
                      "zero Runtime async fill");
    }
    check(memory.fill32(source, 0, kElements), "initialize source");
    check(memory.fill32(destination, 0, kElements), "initialize destination");
    check(memory.fill8(patterns, 0, kPatternBytes), "initialize guards");
    check(memory.fill16(patterns, 0, 8), "warm typed ordinary fill");
    check(memory.fill8_async(patterns, 0, 16, stream), "warm byte async fill");
    check(memory.fill16_async(patterns, 0, 8, stream), "warm typed async fill");
    check(memory.fill32_async(patterns, 0, 4, stream), "warm word async fill");
    check(memory.copy(patterns, source, 16), "warm ordinary copy");
    check(memory.copy_async(patterns, source, 16, stream), "warm async copy");
    runtime_check(memory.runtime_fill(runtime_patterns, 0, 16), "warm Runtime fill");
    runtime_check(
        memory.runtime_fill_async(runtime_patterns, 0, 16, reinterpret_cast<cudaStream_t>(stream)),
        "warm Runtime async fill");
    runtime_check(memory.runtime_copy(runtime_patterns, runtime_pointer(source), 16,
                                      cudaMemcpyDeviceToDevice),
                  "warm Runtime copy");
    runtime_check(
        memory.runtime_copy_async(runtime_patterns, runtime_pointer(source), 16,
                                  cudaMemcpyDeviceToDevice, reinterpret_cast<cudaStream_t>(stream)),
        "warm Runtime async copy");
    check(driver.synchronize_context(), "complete ordinary warmup");
    std::uint64_t duration = 1000000000;
    void* hold_arguments[]{&duration};
    check(driver.launch(hold, kWorkers, 1, 1, kThreads, 1, 1, 0, independent, hold_arguments,
                        nullptr),
          "independent original tail");
    void* producer_arguments[]{&source};
    check(
        driver.launch(add, kWorkers, 1, 1, kThreads, 1, 1, 0, stream, producer_arguments, nullptr),
        "original producer");
    check(memory.copy_async(destination, source, kElements * sizeof(std::uint32_t), stream),
          "copy behind logical producer");
    void* consumer_arguments[]{&destination};
    check(
        driver.launch(add, kWorkers, 1, 1, kThreads, 1, 1, 0, stream, consumer_arguments, nullptr),
        "original consumer after copy");
    check(memory.copy(patterns, source, kPayload), "ordinary D2D result");
    check(memory.copy_async(patterns + kRegion, source + kPayload, kPayload, stream),
          "async D2D result");
    check(memory.fill8(patterns + 2 * kRegion, 0xAB, kPayload), "ordinary byte pattern");
    check(memory.fill8_async(patterns + 3 * kRegion, 0xCD, kPayload, stream), "async byte pattern");
    check(memory.fill16(patterns + 4 * kRegion, 0x1234, kPayload / 2), "ordinary halfword pattern");
    check(memory.fill16_async(patterns + 5 * kRegion, 0x5678, kPayload / 2, stream),
          "async halfword pattern");
    check(memory.fill32(patterns + 6 * kRegion, 0x12345678, kPayload / 4), "ordinary word pattern");
    check(memory.fill32_async(patterns + 7 * kRegion, 0x89ABCDEF, kPayload / 4, stream),
          "async word pattern");
    runtime_check(memory.runtime_copy(runtime_pointer(patterns + 8 * kRegion),
                                      runtime_pointer(source + 2 * kPayload), kPayload,
                                      cudaMemcpyDeviceToDevice),
                  "Runtime D2D result");
    runtime_check(
        memory.runtime_copy_async(runtime_pointer(patterns + 9 * kRegion),
                                  runtime_pointer(source + 3 * kPayload), kPayload,
                                  cudaMemcpyDefault, reinterpret_cast<cudaStream_t>(stream)),
        "Runtime default async D2D result");
    runtime_check(memory.runtime_fill(runtime_pointer(patterns + 10 * kRegion), -1, kPayload),
                  "Runtime signed byte truncation");
    runtime_check(memory.runtime_fill_async(runtime_pointer(patterns + 11 * kRegion), 0x1EF,
                                            kPayload, reinterpret_cast<cudaStream_t>(stream)),
                  "Runtime async byte truncation");
    check(driver.synchronize_stream(stream), "logical memory/consumer prefix");
    const auto tail = cuStreamQuery(independent);
    if (tail != CUDA_SUCCESS && tail != CUDA_ERROR_NOT_READY) check(tail, "independent tail query");
    if (!is_instrumented && tail != CUDA_ERROR_NOT_READY)
        throw std::runtime_error("memory path added a context-wide completion dependency");
    check(driver.synchronize_context(), "complete actual work before readback");
    std::vector<std::uint32_t> original(kElements), copied(kElements);
    std::array<std::byte, kPatternBytes> observed{}, expected{};
    check(driver.download(original.data(), source, original.size() * sizeof(original.front())),
          "all source outputs");
    check(driver.download(copied.data(), destination, copied.size() * sizeof(copied.front())),
          "all dependent outputs");
    check(driver.download(observed.data(), patterns, observed.size()), "all patterns and guards");
    for (std::size_t index = 0; index < original.size(); ++index)
        if (original[index] != 1 || copied[index] != 2)
            throw std::runtime_error("original producer/copy/consumer output mismatch");
    for (std::size_t region = 0; region < 12; ++region)
        for (std::size_t index = 0; index < kPayload; ++index) {
            std::uint32_t bits{};
            std::size_t width{};
            switch (region) {
                case 0:
                case 1:
                case 8:
                case 9:
                    bits = 1;
                    width = 4;
                    break;
                case 2:
                    bits = 0xAB;
                    width = 1;
                    break;
                case 3:
                    bits = 0xCD;
                    width = 1;
                    break;
                case 4:
                    bits = 0x1234;
                    width = 2;
                    break;
                case 5:
                    bits = 0x5678;
                    width = 2;
                    break;
                case 6:
                    bits = 0x12345678;
                    width = 4;
                    break;
                case 7:
                    bits = 0x89ABCDEF;
                    width = 4;
                    break;
                case 10:
                    bits = 0xFF;
                    width = 1;
                    break;
                default:
                    bits = 0xEF;
                    width = 1;
                    break;
            }
            expected[region * kRegion + index] =
                static_cast<std::byte>((bits >> (8 * (index % width))) & 0xFF);
        }
    if (observed != expected) throw std::runtime_error("typed count/scalar/copy/guard mismatch");
    check(driver.free_memory(source), "free source");
    check(driver.free_memory(destination), "free destination");
    check(driver.free_memory(patterns), "free patterns");
    check(driver.destroy_stream(stream), "destroy blocking stream");
    check(driver.destroy_stream(independent), "destroy independent stream");
    check(driver.unload_module(module), "unload original module");
    check(driver.destroy_context(context), "destroy owned context");
    std::cout << "Ordinary memory application verified: elements=20480; pattern_bytes=2048; "
                 "driver_entries=8; runtime_entries=4; cached="
              << (is_cached ? "true" : "false")
              << "; independent_tail=" << (is_instrumented ? "instrumented_not_qualified" : "true")
              << "; workers=320; full_device_qualified=false; timing_qualified=false\n";
}
}  // namespace
int main(int argc, char** argv) {
    try {
        bool is_cached{}, is_instrumented{};
        for (int index = 1; index < argc; ++index) {
            const std::string_view option{argv[index]};
            if (option == "--cached-lookups" && !is_cached)
                is_cached = true;
            else if (option == "--device-sanitizer-check" && !is_instrumented)
                is_instrumented = true;
            else
                throw std::runtime_error("invalid ordinary memory application option");
        }
        run(is_cached, is_instrumented);
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
