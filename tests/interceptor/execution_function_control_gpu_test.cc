#include <cuda.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <dlfcn.h>

namespace {
constexpr unsigned int kBlocks = 2048;
constexpr unsigned int kThreads = 128;
constexpr std::uint32_t kCount = kBlocks * kThreads;
constexpr std::size_t kGuard = 32;
constexpr std::uint32_t kGuardValue = 0x6b3d9a71;
constexpr int kFailureSentinel = 0x194c7b35;
constexpr const char* kSource = R"ptx(
.version 7.0
.target sm_70
.address_size 64
.visible .entry function_control_output(.param .u64 output, .param .u32 count) {
 .reg .pred %p;
 .reg .b32 %r<7>;
 .reg .b64 %rd<4>;
 ld.param.u64 %rd1, [output];
 ld.param.u32 %r1, [count];
 mov.u32 %r2, %ctaid.x;
 mov.u32 %r3, %ntid.x;
 mov.u32 %r4, %tid.x;
 mad.lo.u32 %r5, %r2, %r3, %r4;
 setp.ge.u32 %p, %r5, %r1;
 @%p bra DONE;
 mad.lo.u32 %r6, %r5, 3, 7;
 mul.wide.u32 %rd2, %r5, 4;
 add.u64 %rd3, %rd1, %rd2;
 st.global.u32 [%rd3], %r6;
DONE:
 ret;
}
)ptx";

void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void check(CUresult status, const char* operation) {
    if (status != CUDA_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": CUDA " + std::to_string(status));
}

struct Functions {
    decltype(&cuFuncGetAttribute) get{&cuFuncGetAttribute};
    decltype(&cuFuncSetAttribute) set{&cuFuncSetAttribute};
    decltype(&cuFuncSetCacheConfig) cache{&cuFuncSetCacheConfig};
    void* library{};

    explicit Functions(bool cached) {
        if (!cached) return;
        library = ::dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
        expect(library, "load the original explicit Driver library");
        get = reinterpret_cast<decltype(get)>(::dlsym(library, "cuFuncGetAttribute"));
        set = reinterpret_cast<decltype(set)>(::dlsym(library, "cuFuncSetAttribute"));
        cache = reinterpret_cast<decltype(cache)>(::dlsym(library, "cuFuncSetCacheConfig"));
        if (!get || !set || !cache) {
            const bool closed = close();
            throw std::runtime_error(closed ? "resolve all three original Driver functions"
                                            : "resolve and close original Driver library");
        }
    }
    Functions(const Functions&) = delete;
    Functions& operator=(const Functions&) = delete;
    bool close() noexcept {
        if (!library) return true;
        auto* owned = library;
        library = nullptr;
        return ::dlclose(owned) == 0;
    }
    ~Functions() {
        if (!close()) std::cerr << "function-control Driver library cleanup failed\n";
    }
};

struct Resources {
    CUcontext context{};
    CUmodule module{};
    CUdeviceptr allocation{};
    bool cleanup_attempted{};
    bool cleanup_ok{true};

    bool close() noexcept {
        if (cleanup_attempted) return cleanup_ok;
        cleanup_attempted = true;
        const auto observed = [&](CUresult status) {
            if (status != CUDA_SUCCESS) {
                cleanup_ok = false;
                std::cerr << "function-control cleanup: CUDA " << status << '\n';
            }
            return status == CUDA_SUCCESS;
        };
        // Do not guess at resource retirement after a failed physical stop.
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

struct Attributes {
    int carveout{};
    int static_shared_bytes{};
    int registers{};
    int maximum_dynamic_shared_bytes{};
};
Attributes observe(const Functions& functions, CUfunction function) {
    Attributes result;
    check(functions.get(&result.carveout, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
                        function),
          "observe original carveout");
    check(functions.get(&result.static_shared_bytes, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, function),
          "observe original static shared memory");
    check(functions.get(&result.registers, CU_FUNC_ATTRIBUTE_NUM_REGS, function),
          "observe original register count");
    check(functions.get(&result.maximum_dynamic_shared_bytes,
                        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, function),
          "observe original maximum dynamic shared memory");
    expect(result.static_shared_bytes == 0, "original PTX has no static shared allocation");
    return result;
}
void print_attributes(const Attributes& attributes) {
    std::cout << "{\"carveout\":" << attributes.carveout
              << ",\"static_shared_bytes\":" << attributes.static_shared_bytes
              << ",\"registers\":" << attributes.registers
              << ",\"maximum_dynamic_shared_bytes\":" << attributes.maximum_dynamic_shared_bytes
              << '}';
}

void run(bool cached) {
    Functions functions(cached);
    Resources resources;
    check(cuInit(0), "initialize Driver");
    CUdevice device{};
    check(cuDeviceGet(&device, 0), "select device");
    check(cuCtxCreate(&resources.context, 0, device), "create context");
    check(cuModuleLoadData(&resources.module, kSource), "load ordinary PTX");
    CUfunction function{};
    check(cuModuleGetFunction(&function, resources.module, "function_control_output"),
          "resolve ordinary function");
    // These are genuine observations, not an assumed native default. In
    // particular, an interceptor must not rewrite the original carveout at lookup.
    const auto initial = observe(functions, function);
    check(functions.set(function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 16384),
          "set original dynamic shared capacity");
    int dynamic_shared{};
    check(functions.get(&dynamic_shared, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, function),
          "read configured dynamic shared capacity");
    expect(dynamic_shared == 16384, "exact configured dynamic shared capacity");
    constexpr std::array requested_carveouts{-1, 0, 100};
    std::array<int, requested_carveouts.size()> observed_carveouts{};
    for (std::size_t index = 0; index < requested_carveouts.size(); ++index) {
        check(functions.set(function, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
                            requested_carveouts[index]),
              "set original carveout");
        check(functions.get(&observed_carveouts[index],
                            CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, function),
              "read configured carveout");
        expect(observed_carveouts[index] == requested_carveouts[index],
               "exact configured carveout request");
    }
    check(functions.cache(function, CU_FUNC_CACHE_PREFER_SHARED), "set original cache preference");
    int invalid_output = kFailureSentinel;
    const auto invalid_status =
        functions.get(&invalid_output, static_cast<CUfunction_attribute>(31), function);
    expect(invalid_status != CUDA_SUCCESS && invalid_output == kFailureSentinel,
           "original invalid-attribute error preserves output");
    const auto configured = observe(functions, function);
    expect(configured.maximum_dynamic_shared_bytes == 16384,
           "cache and failed queries preserve configured shared capacity");

    std::vector<std::uint32_t> output(kCount + 2 * kGuard, kGuardValue);
    for (std::size_t index = kGuard; index < kGuard + kCount; ++index) output[index] = UINT32_MAX;
    const auto allocation_bytes = output.size() * sizeof(output[0]);
    check(cuMemAlloc(&resources.allocation, allocation_bytes), "allocate guarded outputs");
    check(cuMemcpyHtoD(resources.allocation, output.data(), allocation_bytes),
          "upload all output sentinels and guards");
    check(cuCtxSynchronize(), "complete pageable initialization before kernel issue");
    CUdeviceptr destination = resources.allocation + kGuard * sizeof(output[0]);
    std::uint32_t count = kCount;
    const auto original_destination = destination;
    void* arguments[]{&destination, &count};
    const std::array<void*, 2> original_arguments{arguments[0], arguments[1]};
    check(cuLaunchKernel(function, kBlocks, 1, 1, kThreads, 1, 1, 0, nullptr, arguments, nullptr),
          "launch unchanged ordinary function after configuration");
    check(cuCtxSynchronize(), "complete original kernel");
    check(cuMemcpyDtoH(output.data(), resources.allocation, allocation_bytes),
          "read every output and guard");
    expect(destination == original_destination && count == kCount &&
               arguments[0] == original_arguments[0] && arguments[1] == original_arguments[1],
           "launch preserves both argument values and host argument pointers");
    std::uint64_t checksum{};
    for (std::size_t index = 0; index < output.size(); ++index) {
        const bool guard = index < kGuard || index >= kGuard + kCount;
        const auto expected =
            guard ? kGuardValue : static_cast<std::uint32_t>((index - kGuard) * 3 + 7);
        expect(output[index] == expected, "complete output or allocation guard mismatch");
        if (!guard) checksum += output[index];
    }
    expect(resources.close(), "complete original context/module/allocation cleanup");
    expect(functions.close(), "close explicit original Driver library");
    std::cout << "{\"status\":\"passed\",\"cached\":" << (cached ? "true" : "false")
              << ",\"initial\":";
    print_attributes(initial);
    std::cout << ",\"configured\":";
    print_attributes(configured);
    std::cout << ",\"requested_carveouts\":[-1,0,100],\"observed_carveouts\":["
              << observed_carveouts[0] << ',' << observed_carveouts[1] << ','
              << observed_carveouts[2] << "],\"invalid_attribute\":31,\"invalid_status\":"
              << static_cast<int>(invalid_status) << ",\"invalid_output\":" << invalid_output
              << ",\"dynamic_shared_bytes\":" << dynamic_shared << ",\"output_count\":" << kCount
              << ",\"guard_count\":" << 2 * kGuard << ",\"checksum\":" << checksum
              << ",\"guards\":true,\"arguments_unchanged\":true,\"cleanup\":true,"
                 "\"full_device_qualified\":false,\"priority_qualified\":false,"
                 "\"performance_qualified\":false,\"full_goal_qualified\":false}\n";
}
}  // namespace

int main(int argc, char** argv) {
    try {
        expect(argc == 1 || (argc == 2 && std::string_view(argv[1]) == "--cached"),
               "usage: execution_function_control_gpu_test [--cached]");
        run(argc == 2);
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Ordinary function-control input failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
