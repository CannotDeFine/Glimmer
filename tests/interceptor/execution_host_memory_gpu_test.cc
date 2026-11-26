#include <cuda.h>
#include <cuda_runtime_api.h>
#include "execution_cached_driver_fixture.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <dlfcn.h>
#include <sys/mman.h>

namespace {
constexpr unsigned int kWorkers = 320, kThreads = 32, kCount = kWorkers * kThreads;
constexpr std::size_t kBytes = std::size_t{kCount} * sizeof(std::uint32_t);
constexpr const char* kSource = R"ptx(
.version 7.0
.target sm_70
.address_size 64
.visible .entry increment(.param .u64 pointer) {
 .reg .b32 %r<6>; .reg .b64 %d<4>;
 ld.param.u64 %d1,[pointer]; mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x;
 mov.u32 %r3,%tid.x; mad.lo.u32 %r4,%r1,%r2,%r3;
 mul.wide.u32 %d2,%r4,4; add.u64 %d3,%d1,%d2;
 ld.global.u32 %r5,[%d3]; add.u32 %r5,%r5,1; st.global.u32 [%d3],%r5; ret;
}
)ptx";
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void check(CUresult result, const char* operation) {
    if (result != CUDA_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": Driver " + std::to_string(result));
}
void runtime_check(cudaError_t result, const char* operation) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": Runtime " + std::to_string(result));
}
void* gpu_pointer(CUdeviceptr pointer) {
    // Checked native numeric device UVA for the Runtime ABI, not host storage.
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    return reinterpret_cast<void*>(pointer);
}
struct Library {
    void* value{};
    explicit Library(const char* name) : value(::dlopen(name, RTLD_NOW | RTLD_LOCAL)) {
        expect(value, "open installed SDK scope");
    }
    ~Library() {
        if (::dlclose(value) != 0) std::terminate();
    }
    Library(const Library&) = delete;
    Library& operator=(const Library&) = delete;
};
struct HostApi {
    decltype(&cuMemHostAlloc) allocate{&cuMemHostAlloc};
    decltype(&cuMemAllocHost) allocate_default{&cuMemAllocHost};
    decltype(&cuMemFreeHost) release{&cuMemFreeHost};
    decltype(&cuMemHostGetFlags) flags{&cuMemHostGetFlags};
    decltype(&cuMemHostGetDevicePointer) pointer{&cuMemHostGetDevicePointer};
    decltype(&cuMemcpyHtoDAsync) upload{&cuMemcpyHtoDAsync};
    decltype(&cuMemcpyDtoHAsync) download{&cuMemcpyDtoHAsync};
    decltype(&cudaHostAlloc) runtime_allocate{&cudaHostAlloc};
    decltype(&cudaMallocHost) runtime_allocate_default{&cudaMallocHost};
    decltype(&cudaFreeHost) runtime_release{&cudaFreeHost};
    decltype(&cudaHostGetFlags) runtime_flags{&cudaHostGetFlags};
    decltype(&cudaHostGetDevicePointer) runtime_pointer{&cudaHostGetDevicePointer};
    decltype(&cudaMemcpyAsync) runtime_copy{&cudaMemcpyAsync};
    decltype(&cudaMemcpy) runtime_copy_sync{&cudaMemcpy};
    decltype(&cudaDeviceReset) reset{&cudaDeviceReset};
    void bind(bool cached) {
        if (!cached) return;
        Library driver("libcuda.so.1"), runtime("libcudart.so.12");
        const auto query = reinterpret_cast<decltype(&cuGetProcAddress_v2)>(
            ::dlsym(driver.value, "cuGetProcAddress_v2"));
        expect(query, "resolve public SDK Driver entry query");
        const auto resolve = [&](const char* name) {
            void* result{};
            CUdriverProcAddressQueryResult status{};
            check(query(name, &result, 12000, 0, &status), name);
            expect(result && status == CU_GET_PROC_ADDRESS_SUCCESS,
                   "public host entry query succeeds");
            return result;
        };
#define DRIVER(FIELD, NAME) (FIELD) = reinterpret_cast<decltype(FIELD)>(resolve(NAME))
        DRIVER(allocate, "cuMemHostAlloc");
        DRIVER(allocate_default, "cuMemAllocHost");
        DRIVER(release, "cuMemFreeHost");
        DRIVER(flags, "cuMemHostGetFlags");
        DRIVER(pointer, "cuMemHostGetDevicePointer");
        DRIVER(upload, "cuMemcpyHtoDAsync");
        DRIVER(download, "cuMemcpyDtoHAsync");
#undef DRIVER
#define RUNTIME(FIELD, NAME)                                 \
    do {                                                     \
        auto* result = ::dlsym(runtime.value, NAME);         \
        expect(result, "cached Runtime host entry exists");  \
        (FIELD) = reinterpret_cast<decltype(FIELD)>(result); \
    } while (false)
        RUNTIME(runtime_allocate, "cudaHostAlloc");
        RUNTIME(runtime_allocate_default, "cudaMallocHost");
        RUNTIME(runtime_release, "cudaFreeHost");
        RUNTIME(runtime_flags, "cudaHostGetFlags");
        RUNTIME(runtime_pointer, "cudaHostGetDevicePointer");
        RUNTIME(runtime_copy, "cudaMemcpyAsync");
        RUNTIME(runtime_copy_sync, "cudaMemcpy");
        RUNTIME(reset, "cudaDeviceReset");
#undef RUNTIME
    }
};
void run_pageable(bool cached, bool is_runtime) {
    ExecutionCachedDriver driver;
    HostApi api;
    driver.bind(cached, true);
    api.bind(cached);
    check(driver.initialize(0), "metadata initialize");
    int sms{};
    check(driver.device_attribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, 0),
          "available SMs");
    expect(sms == 80, "pageable correctness fixture requires the explicit 80-SM V100");
    CUcontext context{};
    if (is_runtime) {
        runtime_check(cudaFree(nullptr), "ordinary lazy Runtime initialization");
        check(driver.get_current(&context), "Runtime current context");
    } else {
        CUctxCreateParams parameters{};
        check(driver.create_context_v4(&context, &parameters, 0, 0),
              "original application context");
    }
    CUmodule module{};
    CUfunction kernel{};
    CUstream stream{};
    check(driver.load_module(&module, kSource), "original module");
    check(driver.get_function(&kernel, module, "increment"), "original kernel");
    check(driver.create_stream(&stream, CU_STREAM_NON_BLOCKING), "independent stream");
    std::size_t verified{};
    for (const std::size_t count :
         {std::size_t{kCount}, std::size_t{1024 * 1024 + 3}, std::size_t{16 * 1024 * 1024 + 3}}) {
        const auto bytes = count * sizeof(std::uint32_t);
        std::vector<std::uint32_t> input(count, 37), output(count, 0);
        CUdeviceptr device{};
        check(driver.allocate(&device, bytes), "device range");
        const auto transfer = [&](void* host, bool upload, bool asynchronous) {
            if (is_runtime) {
                auto* target = upload ? gpu_pointer(device) : host;
                const void* origin = upload ? host : gpu_pointer(device);
                if (asynchronous)
                    runtime_check(api.runtime_copy(target, origin, bytes, cudaMemcpyDefault,
                                                   reinterpret_cast<cudaStream_t>(stream)),
                                  "Runtime pageable async transfer");
                else
                    runtime_check(api.runtime_copy_sync(
                                      target, origin, bytes,
                                      upload ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost),
                                  "Runtime pageable ordinary transfer");
            } else if (upload) {
                if (asynchronous)
                    check(api.upload(device, host, bytes, stream), "Driver pageable async upload");
                else
                    check(driver.upload(device, host, bytes), "Driver pageable ordinary upload");
            } else {
                if (asynchronous)
                    check(api.download(host, device, bytes, stream),
                          "Driver pageable async download");
                else
                    check(driver.download(host, device, bytes),
                          "Driver pageable ordinary download");
            }
        };
        const auto launch = [&](CUstream selected) {
            void* arguments[]{&device};
            check(driver.launch(kernel, kWorkers, 1, 1, kThreads, 1, 1, 0, selected, arguments,
                                nullptr),
                  "unchanged original pageable consumer");
        };
        transfer(input.data(), true, true);
        std::fill(input.begin(), input.end(),
                  99);  // Source reusable after pageable staging returns.
        launch(stream);
        transfer(output.data(), false, true);
        for (std::size_t index = 0; index < count; ++index)
            expect(output[index] == (index < kCount ? 38U : 37U),
                   "complete first pageable download on API return");
        transfer(output.data(), true, true);
        launch(stream);
        transfer(output.data(), false, true);
        for (std::size_t index = 0; index < count; ++index)
            expect(output[index] == (index < kCount ? 39U : 37U),
                   "pageable D2H-to-H2D alias dataflow and guard bytes");
        transfer(output.data(), true, false);
        launch(nullptr);
        transfer(output.data(), false, false);
        for (std::size_t index = 0; index < count; ++index)
            expect(output[index] == (index < kCount ? 40U : 37U),
                   "ordinary pageable stream ordering and complete readback");
        if (count == kCount) {
            auto* write_only = static_cast<std::uint32_t*>(
                ::mmap(nullptr, bytes, PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
            expect(write_only != MAP_FAILED, "create write-only ordinary host destination");
            for (const bool asynchronous : {false, true}) {
                transfer(write_only, false, asynchronous);
                expect(::mprotect(write_only, bytes, PROT_READ | PROT_WRITE) == 0,
                       "make completed output readable after API return");
                for (std::size_t index = 0; index < count; ++index)
                    expect(write_only[index] == 40,
                           "all write-only destination values are complete");
                expect(::mprotect(write_only, bytes, PROT_WRITE) == 0,
                       "restore write-only destination");
            }
            expect(::munmap(write_only, bytes) == 0, "retire write-only host mapping");
        }
        check(driver.free_memory(device), "device free after staged users");
        verified += count;
    }
    check(driver.destroy_stream(stream), "stream cleanup");
    check(driver.unload_module(module), "original module cleanup");
    if (is_runtime)
        runtime_check(api.reset(), "origin primary reset with staged cleanup");
    else
        check(driver.destroy_context(context), "origin context cleanup with staged users");
    std::cout << "Ordinary pageable application verified: values=" << verified
              << "; snapshot=true; alias_dataflow=true; guards=true; write_only=true; cached="
              << (cached ? "true" : "false") << "; runtime=" << (is_runtime ? "true" : "false")
              << "; workers=320; full_device_qualified=false; timing_qualified=false\n";
}
void run(bool cached, bool is_runtime) {
    ExecutionCachedDriver driver;
    HostApi api;
    driver.bind(cached, true);
    api.bind(cached);
    check(driver.initialize(0), "metadata initialize");
    int sms{};
    check(driver.device_attribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, 0),
          "available SMs");
    expect(sms == 80, "this diagnostic requires the explicit 80-SM V100");
    CUcontext context{};
    CUctxCreateParams context_parameters{};
    CUexecAffinityParam ignored_affinity{};
    context_parameters.execAffinityParams = &ignored_affinity;
    void* host{};
    if (is_runtime) {
        runtime_check(api.runtime_allocate(&host, kBytes, 0), "first Runtime host allocation");
        check(driver.get_current(&context), "Runtime current context");
        expect(context, "first Runtime host allocation initializes a current context");
    } else {
        expect(api.allocate(&host, 4096, 0) == CUDA_ERROR_INVALID_CONTEXT && !host,
               "Driver host allocation does not silently create a current context");
        check(api.release(nullptr), "null Driver free without a current context");
        check(driver.create_context_v4(&context, &context_parameters, 0, 0), "application context");
        check(api.allocate(&host, kBytes, 0), "host allocation");
    }
    CUcontext after{};
    check(driver.get_current(&after), "current context after host allocation");
    expect(after == context,
           "host registration must not push a private context onto the caller stack");
    unsigned int user_flags{};
    check(driver.get_context_flags(&user_flags), "observable application context flags");
    expect(is_runtime ? (user_flags & CU_CTX_MAP_HOST) != 0 : user_flags == 0,
           "private persistent MAP_HOST capability does not change user-context metadata");
    unsigned int flags{};
    check(api.flags(&flags, host), "Driver flags of shared Runtime/Driver allocation");
    expect(flags == CU_MEMHOSTALLOC_DEVICEMAP,
           "native V100 allocation flags are not merely requested flags");
    CUdeviceptr mapped{};
    check(api.pointer(&mapped, host, 0), "original mapped address");
    expect(mapped == reinterpret_cast<CUdeviceptr>(host),
           "original UVA host pointer remains a valid device argument");
    auto* values = static_cast<std::uint32_t*>(host);
    for (std::size_t index = 0; index < kCount; ++index) values[index] = 37;
    auto* interior = static_cast<char*>(host) + 1;
    check(api.flags(&flags, interior), "interior host flags");
    check(api.pointer(&mapped, interior, 0), "interior mapped pointer");
    expect(flags == CU_MEMHOSTALLOC_DEVICEMAP && mapped == reinterpret_cast<CUdeviceptr>(interior),
           "all bytes in a host range preserve mapped-pointer identity");
    expect(api.flags(&flags, static_cast<char*>(host) + kBytes) == CUDA_ERROR_INVALID_VALUE,
           "one-past host range is not a live host allocation");
    CUmodule module{};
    CUfunction kernel{};
    CUstream stream{};
    CUdeviceptr first{}, second{};
    check(driver.load_module(&module, kSource), "original module");
    check(driver.get_function(&kernel, module, "increment"), "original kernel");
    check(driver.allocate(&first, kBytes), "first device range");
    check(driver.allocate(&second, kBytes), "second device range");
    check(driver.create_stream(&stream, CU_STREAM_NON_BLOCKING), "independent stream");
    const auto transfer = [&](CUdeviceptr device, bool is_upload) {
        if (is_runtime)
            runtime_check(
                api.runtime_copy(is_upload ? gpu_pointer(device) : host,
                                 is_upload ? host : gpu_pointer(device), kBytes, cudaMemcpyDefault,
                                 reinterpret_cast<cudaStream_t>(stream)),
                "ordinary Runtime async shared copy");
        else if (is_upload)
            check(api.upload(device, host, kBytes, stream), "ordinary Driver async upload");
        else
            check(api.download(host, device, kBytes, stream), "ordinary Driver async download");
    };
    const auto launch = [&](CUdeviceptr device) {
        void* arguments[]{&device};
        check(driver.launch(kernel, kWorkers, 1, 1, kThreads, 1, 1, 0, stream, arguments, nullptr),
              "original unmodified kernel ABI");
    };
    transfer(first, true);
    launch(first);
    transfer(first, false);
    transfer(first,
             true);  // Must read the preceding GPU-produced D2H bytes, not an early RPC snapshot.
    launch(first);
    transfer(first, false);
    launch(reinterpret_cast<CUdeviceptr>(host));
    transfer(second, true);
    if (is_runtime)
        runtime_check(api.runtime_release(host), "Runtime free after accepted GPU users");
    else
        check(api.release(host), "Driver free after accepted GPU users");
    // Host memory is invalid after free. Validate its last GPU consumer instead.
    std::vector<std::uint32_t> output(kCount);
    check(driver.download(output.data(), second, kBytes), "last GPU consumer readback");
    for (const auto value : output)
        expect(value == 40, "all shared-buffer and mapped-kernel numerical results are exact");
    for (unsigned int requested = 0; requested <= 3; ++requested) {
        void* buffer{};
        if (is_runtime)
            runtime_check(api.runtime_allocate(&buffer, 4096, requested), "Runtime flags variant");
        else
            check(api.allocate(&buffer, 4096, requested), "Driver flags variant");
        check(api.flags(&flags, buffer), "variant flags");
        expect(flags == (requested | CU_MEMHOSTALLOC_DEVICEMAP),
               "all supported public flag bits preserve native normalization");
        if (is_runtime) {
            void* pointer{};
            runtime_check(api.runtime_pointer(&pointer, buffer, 0), "Runtime mapped query");
            runtime_check(api.runtime_flags(&flags, buffer), "Runtime flags query");
            expect(pointer == buffer && flags == (requested | CU_MEMHOSTALLOC_DEVICEMAP),
                   "Runtime queries match original metadata");
            runtime_check(api.runtime_release(buffer), "Runtime variant free");
        } else
            check(api.release(buffer), "Driver variant free");
    }
    void* empty = host;
    if (is_runtime)
        runtime_check(api.runtime_allocate_default(&empty, 0), "zero Runtime host allocation");
    else
        check(api.allocate_default(&empty, 0), "zero Driver host allocation");
    expect(!empty, "zero host allocation returns null without backing");
    check(api.release(nullptr), "null Driver free with current context");
    check(api.upload(0, nullptr, 0, nullptr), "zero-byte Driver async host upload");
    check(api.download(nullptr, 0, 0, nullptr), "zero-byte Driver async host download");
    if (is_runtime) runtime_check(api.runtime_release(nullptr), "null Runtime host free");
    void* default_buffer{};
    if (is_runtime)
        runtime_check(api.runtime_allocate_default(&default_buffer, 4096),
                      "default Runtime host allocation");
    else
        check(api.allocate_default(&default_buffer, 4096), "default Driver host allocation");
    check(driver.set_current(nullptr), "temporarily clear current context");
    expect(api.flags(&flags, default_buffer) == CUDA_ERROR_INVALID_CONTEXT,
           "Driver host flags require a current context");
    check(api.release(default_buffer), "host free does not require a current context");
    check(driver.set_current(context), "restore original application context");
    check(driver.free_memory(first), "free first device range");
    check(driver.free_memory(second), "free second device range");
    check(driver.destroy_stream(stream), "destroy stream");
    check(driver.unload_module(module), "unload original module");
    void* retired{};
    if (is_runtime) {
        runtime_check(api.runtime_allocate(&retired, 4096, 0),
                      "generation-owned Runtime host allocation");
        runtime_check(api.reset(), "reset allocating primary generation");
        expect(api.runtime_flags(&flags, retired) == cudaErrorInvalidValue &&
                   api.runtime_release(retired) == cudaErrorInvalidValue,
               "old Runtime host allocation cannot survive origin reset");
        runtime_check(api.reset(), "retire query's new primary generation");
    } else {
        check(api.allocate(&retired, 4096, CU_MEMHOSTALLOC_PORTABLE),
              "generation-owned portable host allocation");
        check(driver.destroy_context(context), "destroy allocating user context");
        context_parameters = {};  // A valid empty structure differs from NULL itself.
        check(driver.create_context_v4(&context, &context_parameters, 0, 0),
              "fresh survivor context");
        expect(api.flags(&flags, retired) == CUDA_ERROR_INVALID_VALUE &&
                   api.release(retired) == CUDA_ERROR_INVALID_VALUE,
               "portable host allocation cannot outlive its allocating user context");
        check(driver.destroy_context(context), "retire survivor context");
    }
    std::cout
        << "Ordinary public host application verified: values=" << kCount
        << "; mapped_original_argument=true; alias_dataflow=true; origin_lifetime=true; cached="
        << (cached ? "true" : "false") << "; runtime=" << (is_runtime ? "true" : "false")
        << "; workers=320; full_device_qualified=false; timing_qualified=false\n";
}
}  // namespace
int main(int argc, char** argv) {
    try {
        bool cached{}, runtime{}, pageable{};
        for (int index = 1; index < argc; ++index) {
            const std::string argument{argv[index]};
            if (argument == "--cached-lookups")
                cached = true;
            else if (argument == "--runtime")
                runtime = true;
            else if (argument == "--pageable")
                pageable = true;
            else
                throw std::runtime_error("unknown public host application option");
        }
        if (pageable)
            run_pageable(cached, runtime);
        else
            run(cached, runtime);
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
