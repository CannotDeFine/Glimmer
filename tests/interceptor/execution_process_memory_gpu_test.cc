#include <cuda.h>
#include <cuda_runtime_api.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

#include <dlfcn.h>

// Ordinary SDK-only input, identical under native and owned execution. No
// Glimmer headers, scheduler API, IPC fixture or implementation observation.
namespace {
constexpr std::size_t kElements = 4096, kGuard = 32, kExtent = kElements + 2 * kGuard;
constexpr std::uint32_t kCanary = 0xa5c3e187U;
using Image = std::array<std::uint32_t, kExtent>;
void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void checked(CUresult status, const char* operation) {
    if (status != CUDA_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": Driver " + std::to_string(status));
}
void checked_runtime(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": Runtime " + std::to_string(status));
}
struct Library {
    void* handle;
    explicit Library(const char* name) : handle(::dlopen(name, RTLD_NOW | RTLD_LOCAL)) {
        expect(handle != nullptr, "load original SDK scope");
    }
    ~Library() {
        if (::dlclose(handle) != 0) std::terminate();
    }
    Library(const Library&) = delete;
    Library& operator=(const Library&) = delete;
};
struct Api {
    decltype(&cuDeviceGetDefaultMemPool) default_pool{&cuDeviceGetDefaultMemPool};
    decltype(&cuDeviceGetMemPool) current_pool{&cuDeviceGetMemPool};
    decltype(&cuDeviceSetMemPool) select_pool{&cuDeviceSetMemPool};
    decltype(&cuMemAllocAsync) allocate{&cuMemAllocAsync};
    decltype(&cuMemAllocFromPoolAsync) allocate_from_pool{&cuMemAllocFromPoolAsync};
    decltype(&cudaMemPoolTrimTo) trim{&cudaMemPoolTrimTo};
    decltype(&cudaLaunchHostFunc) host_function{&cudaLaunchHostFunc};
    void bind(bool cached) {
        if (!cached) return;
        Library driver("libcuda.so.1"), runtime("libcudart.so.12");
        const auto query = reinterpret_cast<decltype(&cuGetProcAddress_v2)>(
            ::dlsym(driver.handle, "cuGetProcAddress_v2"));
        expect(query != nullptr, "resolve original public Driver entry point");
        const auto resolve = [&](const char* name) {
            void* function{};
            CUdriverProcAddressQueryResult result{};
            checked(query(name, &function, 12000, 0, &result), name);
            expect(function && result == CU_GET_PROC_ADDRESS_SUCCESS, "cache public pool API");
            return function;
        };
        default_pool =
            reinterpret_cast<decltype(default_pool)>(resolve("cuDeviceGetDefaultMemPool"));
        current_pool = reinterpret_cast<decltype(current_pool)>(resolve("cuDeviceGetMemPool"));
        select_pool = reinterpret_cast<decltype(select_pool)>(resolve("cuDeviceSetMemPool"));
        allocate = reinterpret_cast<decltype(allocate)>(resolve("cuMemAllocAsync"));
        allocate_from_pool =
            reinterpret_cast<decltype(allocate_from_pool)>(resolve("cuMemAllocFromPoolAsync"));
        trim = reinterpret_cast<decltype(trim)>(::dlsym(runtime.handle, "cudaMemPoolTrimTo"));
        host_function = reinterpret_cast<decltype(host_function)>(
            ::dlsym(runtime.handle, "cudaLaunchHostFunc"));
        expect(trim && host_function, "cache public Runtime pool/callback APIs");
    }
};
struct Gate {
    std::atomic<bool> can_return{}, expired{};
    std::atomic<unsigned int> calls{};
};
void CUDART_CB wait_for_enqueue(void* opaque) {
    auto& gate = *static_cast<Gate*>(opaque);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!gate.can_return.load(std::memory_order_acquire)) {
        if (std::chrono::steady_clock::now() >= deadline) {
            gate.expired.store(true, std::memory_order_release);
            break;
        }
        std::this_thread::yield();
    }
    gate.calls.fetch_add(1, std::memory_order_release);
}
struct Resources {
    CUcontext first{}, second{};
    CUstream stream{};
    CUdeviceptr pointer{};
    std::unique_ptr<Gate> gate{std::make_unique<Gate>()};
    bool callback_submitted{}, cleanup_attempted{}, cleanup_succeeded{}, is_uncertain{};
    bool cleanup() noexcept {
        if (cleanup_attempted) return cleanup_succeeded;
        cleanup_attempted = true;
        gate->can_return.store(true, std::memory_order_release);
        if (is_uncertain) return false;
        const auto selected_context = second ? second : first;
        if ((pointer || stream || selected_context) &&
            (!selected_context || cuCtxSetCurrent(selected_context) != CUDA_SUCCESS)) {
            is_uncertain = true;
            return false;
        }
        if (stream) {
            const auto retiring = std::exchange(stream, nullptr);
            if (cuStreamSynchronize(retiring) != CUDA_SUCCESS ||
                cuStreamDestroy(retiring) != CUDA_SUCCESS) {
                is_uncertain = true;
                return false;
            }
        }
        if (pointer && cuMemFree(std::exchange(pointer, 0)) != CUDA_SUCCESS) {
            is_uncertain = true;
            return false;
        }
        for (auto* context : {&second, &first}) {
            if (*context && cuCtxDestroy(std::exchange(*context, nullptr)) != CUDA_SUCCESS) {
                is_uncertain = true;
                return false;
            }
        }
        cleanup_succeeded = true;
        return true;
    }
    ~Resources() {
        if (!cleanup()) std::cerr << "Process-memory fixture cleanup failed; no retry\n";
        // A callback may retain this pointer after failed cleanup. Retain it
        // through process exit rather than turn an error into use-after-free.
        if (callback_submitted && gate->calls.load(std::memory_order_acquire) != 1)
            (void)gate.release();
    }
};
void checked_effect(Resources& resources, CUresult status, const char* operation) {
    if (status != CUDA_SUCCESS) resources.is_uncertain = true;
    checked(status, operation);
}
void checked_runtime_effect(Resources& resources, cudaError_t status, const char* operation) {
    if (status != cudaSuccess) resources.is_uncertain = true;
    checked_runtime(status, operation);
}
std::uint64_t run(const Api& api, bool explicit_pool) {
    Resources resources;
    CUmemoryPool initial_default{}, initial_current{};
    checked(api.default_pool(&initial_default, 0), "context-free default pool");
    checked(api.current_pool(&initial_current, 0), "context-free current pool");
    expect(initial_default && initial_current, "real pools have nonnull handles");
    checked(api.select_pool(0, initial_current), "context-free existing pool selection");
    CUcontext created{};
    checked_effect(resources, cuCtxCreate(&created, 0, 0), "create context A");
    if (!created) resources.is_uncertain = true;
    expect(created != nullptr, "context A has actual identity");
    resources.first = created;
    CUstream stream{};
    checked_effect(resources, cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING),
                   "create source stream");
    if (!stream) resources.is_uncertain = true;
    expect(stream != nullptr, "source stream has actual identity");
    resources.stream = stream;
    const auto allocate = [&](CUdeviceptr* output, std::size_t size) {
        return explicit_pool
                   ? api.allocate_from_pool(output, size, initial_default, resources.stream)
                   : api.allocate(output, size, resources.stream);
    };
    CUdeviceptr zero{0x194c7b35};
    checked_effect(resources, allocate(&zero, 0), "zero allocation SDK result");
    if (zero) resources.is_uncertain = true;
    expect(zero == 0, "zero allocation overwrites sentinel with NULL");
    checked_effect(resources, cuStreamSynchronize(resources.stream), "complete zero allocation");

    resources.callback_submitted = true;  // Unknown submit failure still retains callback data.
    checked_runtime_effect(resources,
                           api.host_function(reinterpret_cast<cudaStream_t>(resources.stream),
                                             wait_for_enqueue, resources.gate.get()),
                           "enqueue bounded CPU gate before allocation");
    CUdeviceptr allocated{};
    checked_effect(resources, allocate(&allocated, sizeof(Image)),
                   "allocate behind pending callback");
    if (!allocated) resources.is_uncertain = true;
    expect(allocated != 0, "nonempty allocation returns owned range");
    resources.pointer = allocated;
    checked_effect(resources,
                   cuMemsetD32Async(resources.pointer, kCanary, kExtent, resources.stream),
                   "enqueue allocation consumer without waiting for physical allocator tail");
    resources.gate->can_return.store(true, std::memory_order_release);
    checked_effect(resources, cuStreamSynchronize(resources.stream),
                   "join gate, allocation and consumer");
    expect(resources.gate->calls.load(std::memory_order_acquire) == 1 && !resources.gate->expired,
           "allocation and consumer enqueue must return before the host gate opens");
    Image observed{}, expected{};
    expected.fill(kCanary);
    checked_effect(resources, cuMemcpyDtoH(observed.data(), resources.pointer, sizeof(observed)),
                   "read queued fill");
    expect(observed == expected, "all queued fill values and guards match");
    for (std::size_t index = 0; index < kElements; ++index)
        expected[kGuard + index] =
            static_cast<std::uint32_t>(index * 13 + (explicit_pool ? 31 : 7));
    checked_effect(resources, cuMemcpyHtoD(resources.pointer, expected.data(), sizeof(expected)),
                   "write guarded image");
    // This output is not an association with the context which issued allocation.
    // NOLINTNEXTLINE(performance-no-int-to-ptr): deliberate output sentinel, never dereferenced.
    CUcontext pointer_context = reinterpret_cast<CUcontext>(std::uintptr_t{0x194c7b35});
    checked(
        cuPointerGetAttribute(&pointer_context, CU_POINTER_ATTRIBUTE_CONTEXT, resources.pointer),
        "query real process-owned pointer context");
    expect(pointer_context == nullptr, "pool allocation is not context-associated");
    checked_effect(resources, cuStreamDestroy(std::exchange(resources.stream, nullptr)),
                   "destroy completed A stream");
    checked_effect(resources, cuCtxDestroy(std::exchange(resources.first, nullptr)),
                   "destroy source context A");
    created = nullptr;
    checked_effect(resources, cuCtxCreate(&created, 0, 0), "create context B");
    if (!created) resources.is_uncertain = true;
    expect(created != nullptr, "context B has actual identity");
    resources.second = created;
    CUmemoryPool surviving_default{}, surviving_current{};
    checked(api.default_pool(&surviving_default, 0), "default pool survives source context");
    checked(api.current_pool(&surviving_current, 0), "current pool survives source context");
    expect(surviving_default == initial_default && surviving_current == initial_current,
           "pool identities remain stable across context retirement");
    checked_runtime_effect(resources,
                           api.trim(reinterpret_cast<cudaMemPool_t>(surviving_default), 0),
                           "actual Runtime trim with live allocation");
    checked_effect(resources, cuMemcpyDtoH(observed.data(), resources.pointer, sizeof(observed)),
                   "use surviving pointer in B");
    expect(observed == expected, "entire cross-context image and guards match");
    std::uint64_t checksum{};
    for (const auto value : observed) checksum = checksum * 131U + value;
    const auto retiring = std::exchange(resources.pointer, 0);
    if (explicit_pool) {
        // NOLINTNEXTLINE(performance-no-int-to-ptr): numeric device UVA, never host storage.
        checked_runtime_effect(resources, cudaFree(reinterpret_cast<void*>(retiring)),
                               "Runtime synchronous free from B");
    } else {
        checked_effect(resources, cuMemFree(retiring), "Driver synchronous free from B");
    }
    expect(resources.cleanup(), "normal source-independent exact cleanup");
    return checksum;
}
}  // namespace

int main(int argc, char** argv) {
    try {
        const bool cached = argc == 2 && std::string_view(argv[1]) == "--cached";
        expect(argc == 1 || cached, "usage: execution_process_memory_gpu_test [--cached]");
        Api api;
        api.bind(cached);
        checked(cuInit(0), "initialize original Driver");
        int supported{};
        checked(cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED, 0),
                "query actual device allocator support");
        expect(supported == 1, "memory pools must be supported by the selected device");
        const auto first = run(api, false);
        const auto second = run(api, true);
        std::cout << "{\"status\":\"passed\",\"cached\":" << (cached ? "true" : "false")
                  << ",\"complete_values_and_guards\":" << 4 * kExtent << ",\"checksums\":["
                  << first << ',' << second
                  << "],\"same_stream_enqueue\":true,\"cross_context_free\":true,"
                     "\"normal_cleanup\":true,\"full_device_qualified\":false,"
                     "\"performance_qualified\":false}\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
