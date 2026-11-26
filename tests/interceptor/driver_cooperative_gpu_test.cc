#include "cooperative_ptx_fixture.h"

#include <cuda.h>
#include <dlfcn.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string_view>
#include <thread>

namespace {
constexpr unsigned int kBlocks = 105;
constexpr unsigned int kThreads = 64;
constexpr std::size_t kCount = static_cast<std::size_t>(kBlocks) * kThreads;

bool check(CUresult result, const char* operation) {
    if (result == CUDA_SUCCESS) {
        return true;
    }
    std::cerr << operation << " failed with CUresult " << static_cast<int>(result) << '\n';
    return false;
}

struct Resources {
    CUcontext context{};
    CUcontext high_context{};
    std::array<CUmodule, 2> modules{};
    std::array<CUstream, 3> streams{};
    CUdeviceptr output{};
    CUevent done{};
    std::uint32_t* input{};
    std::uint32_t* readback{};
    std::uint32_t* waited{};
    std::uint32_t* markers{};

    bool cleanup() {
        bool passed = true;
        if (high_context) {
            passed &= check(cuCtxSetCurrent(high_context), "activate high cleanup");
            passed &= check(cuCtxSynchronize(), "drain high context");
            if (streams[1]) {
                passed &= check(cuStreamDestroy(streams[1]), "destroy high stream");
                streams[1] = nullptr;
            }
            if (modules[1]) {
                passed &= check(cuModuleUnload(modules[1]), "unload high module");
                modules[1] = nullptr;
            }
            passed &= check(cuCtxDestroy(high_context), "destroy high context");
            high_context = nullptr;
        }
        if (context) {
            passed &= check(cuCtxSetCurrent(context), "activate low cleanup");
            passed &= check(cuCtxSynchronize(), "cleanup synchronize");
        }
        if (done) {
            passed &= check(cuEventDestroy(done), "destroy event");
            done = nullptr;
        }
        for (auto& stream : streams) {
            if (stream) {
                passed &= check(cuStreamDestroy(stream), "destroy stream");
                stream = nullptr;
            }
        }
        if (output) {
            passed &= check(cuMemFree(output), "free output");
            output = 0;
        }
        for (auto& pointer : {&input, &readback, &waited, &markers}) {
            if (*pointer) {
                passed &= check(cuMemFreeHost(*pointer), "free pinned");
                *pointer = nullptr;
            }
        }
        for (auto& module : modules) {
            if (module) {
                passed &= check(cuModuleUnload(module), "unload module");
                module = nullptr;
            }
        }
        if (context) {
            passed &= check(cuCtxDestroy(context), "destroy context");
            context = nullptr;
        }
        return passed;
    }
    ~Resources() {
        (void)cleanup();
    }
};

struct ApplicationApi {
    decltype(&cuLaunchKernel) launch{cuLaunchKernel};
    decltype(&cuMemcpyHtoDAsync) upload{cuMemcpyHtoDAsync};
    decltype(&cuMemcpyDtoHAsync) download{cuMemcpyDtoHAsync};
    decltype(&cuEventRecord) record{cuEventRecord};
    decltype(&cuStreamWaitEvent) wait{cuStreamWaitEvent};
    decltype(&cuStreamQuery) query{cuStreamQuery};
    decltype(&cuStreamSynchronize) synchronize{cuStreamSynchronize};
};

template <typename Function>
bool resolve(Function& function, const char* base, std::string_view direct, bool uses_proc,
             bool per_thread_default_stream) {
    void* address = nullptr;
    if (uses_proc) {
        CUdriverProcAddressQueryResult status{};
        if (!check(cuGetProcAddress(base, &address, 12600,
                                    per_thread_default_stream
                                        ? CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM
                                        : CU_GET_PROC_ADDRESS_LEGACY_STREAM,
                                    &status),
                   "application proc address") ||
            status != CU_GET_PROC_ADDRESS_SUCCESS) {
            return false;
        }
    } else {
        address = dlsym(RTLD_DEFAULT, direct.data());
    }
    function = reinterpret_cast<Function>(address);
    return function != nullptr;
}

bool run(bool uses_proc, bool uses_dynamic, bool uses_ptds, bool uses_cooperative_high,
         bool releases_pending_output, bool uses_cross_context, bool uses_async_release) {
    Resources resources;
    ApplicationApi api;
    CUdevice device{};
    if (!check(cuInit(0), "initialize") || !check(cuDeviceGet(&device, 0), "device") ||
        !check(cuCtxCreate(&resources.context, 0, device), "context") ||
        !check(cuModuleLoadData(&resources.modules[0], kCooperativePtx), "load low")) {
        return false;
    }
    CUfunction low{};
    CUfunction high{};
    if (uses_cross_context &&
        !check(cuCtxCreate(&resources.high_context, 0, device), "high context")) {
        return false;
    }
    if (!check(cuModuleLoadData(&resources.modules[1], kNativeOncePtx), "load high") ||
        !check(cuModuleGetFunction(&high, resources.modules[1], "native_once"), "high function") ||
        !check(cuStreamCreateWithPriority(&resources.streams[1], CU_STREAM_NON_BLOCKING, -1),
               "high stream") ||
        !check(cuCtxSetCurrent(resources.context), "restore low context")) {
        return false;
    }
    if (!check(cuModuleGetFunction(&low, resources.modules[0], "cooperative"), "low function") ||
        !check(cuStreamCreateWithPriority(&resources.streams[0], CU_STREAM_NON_BLOCKING, 0),
               "low stream") ||
        !check(cuStreamCreate(&resources.streams[2], CU_STREAM_NON_BLOCKING), "wait stream") ||
        !check(cuMemAlloc(&resources.output, kCount * sizeof(std::uint32_t)), "output") ||
        !check(cuMemHostAlloc(reinterpret_cast<void**>(&resources.input),
                              kCount * sizeof(std::uint32_t), 0),
               "input") ||
        !check(cuMemHostAlloc(reinterpret_cast<void**>(&resources.readback),
                              kCount * sizeof(std::uint32_t), 0),
               "readback") ||
        !check(cuMemHostAlloc(reinterpret_cast<void**>(&resources.waited),
                              kCount * sizeof(std::uint32_t), 0),
               "waited") ||
        !check(
            cuMemHostAlloc(reinterpret_cast<void**>(&resources.markers), 2 * sizeof(std::uint32_t),
                           CU_MEMHOSTALLOC_DEVICEMAP | CU_MEMHOSTALLOC_PORTABLE),
            "markers") ||
        !check(cuEventCreate(&resources.done, CU_EVENT_DISABLE_TIMING), "event")) {
        return false;
    }
    CUdeviceptr device_markers{};
    if (!check(cuMemHostGetDevicePointer(&device_markers, resources.markers, 0),
               "mapped markers")) {
        return false;
    }
    CUdeviceptr high_markers = device_markers;
    if (uses_cross_context) {
        if (!check(cuCtxSetCurrent(resources.high_context), "activate high mapping") ||
            !check(cuMemHostGetDevicePointer(&high_markers, resources.markers, 0),
                   "map portable high markers") ||
            !check(cuCtxSetCurrent(resources.context), "restore low mapping")) {
            return false;
        }
    }
    if (uses_proc || uses_dynamic || uses_ptds) {
        if (!resolve(api.launch, "cuLaunchKernel",
                     uses_ptds ? "cuLaunchKernel_ptsz" : "cuLaunchKernel", uses_proc, uses_ptds) ||
            !resolve(api.upload, "cuMemcpyHtoDAsync",
                     uses_ptds ? "cuMemcpyHtoDAsync_v2_ptsz" : "cuMemcpyHtoDAsync_v2", uses_proc,
                     uses_ptds) ||
            !resolve(api.download, "cuMemcpyDtoHAsync",
                     uses_ptds ? "cuMemcpyDtoHAsync_v2_ptsz" : "cuMemcpyDtoHAsync_v2", uses_proc,
                     uses_ptds) ||
            !resolve(api.record, "cuEventRecord",
                     uses_ptds ? "cuEventRecord_ptsz" : "cuEventRecord", uses_proc, uses_ptds) ||
            !resolve(api.wait, "cuStreamWaitEvent",
                     uses_ptds ? "cuStreamWaitEvent_ptsz" : "cuStreamWaitEvent", uses_proc,
                     uses_ptds) ||
            !resolve(api.query, "cuStreamQuery", uses_ptds ? "cuStreamQuery_ptsz" : "cuStreamQuery",
                     uses_proc, uses_ptds) ||
            !resolve(api.synchronize, "cuStreamSynchronize",
                     uses_ptds ? "cuStreamSynchronize_ptsz" : "cuStreamSynchronize", uses_proc,
                     uses_ptds)) {
            return false;
        }
    }
    CUstream low_stream = uses_ptds ? nullptr : resources.streams[0];
    auto release_async = &cuMemFreeAsync;
    if (uses_async_release) {
        if ((uses_ptds || uses_proc || uses_dynamic) &&
            !resolve(release_async, "cuMemFreeAsync",
                     uses_ptds ? "cuMemFreeAsync_ptsz" : "cuMemFreeAsync", uses_proc, uses_ptds)) {
            return false;
        }
        if (release_async(resources.output + sizeof(std::uint32_t), resources.streams[2]) !=
            CUDA_ERROR_INVALID_VALUE) {
            std::cerr << "invalid first Driver free must preserve native immediate error\n";
            return false;
        }
    }
    const std::size_t copy_bytes = kCount * sizeof(std::uint32_t);
    if (api.upload(resources.output, resources.input, copy_bytes + 1, low_stream) !=
            CUDA_ERROR_INVALID_VALUE ||
        api.download(resources.readback, resources.output, copy_bytes + 1, low_stream) !=
            CUDA_ERROR_INVALID_VALUE ||
        cuMemcpyDtoDAsync(resources.output, resources.output, copy_bytes + 1, low_stream) !=
            CUDA_ERROR_INVALID_VALUE) {
        std::cerr << "oversized Driver copy must preserve native immediate errors\n";
        return false;
    }
    for (std::size_t index = 0; index < kCount; ++index) {
        resources.input[index] = 0;
        resources.readback[index] = 0xdeadbeef;
        resources.waited[index] = 0xdeadbeef;
    }
    resources.markers[0] = 0;
    resources.markers[1] = 0;
    if (!check(api.upload(resources.output, resources.input, kCount * sizeof(std::uint32_t),
                          low_stream),
               "initialize output")) {
        return false;
    }
    CUdeviceptr argument_output = resources.output;
    CUdeviceptr argument_marker = device_markers;
    unsigned int work = 50000;
    std::array<void*, 3> parameters{&argument_output, &argument_marker, &work};
    if (!check(api.launch(low, kBlocks, 1, 1, kThreads, 1, 1, 0, low_stream, parameters.data(),
                          nullptr),
               "low launch")) {
        return false;
    }
    // The application remains ordinary CUDA. Its transient argument storage is
    // immediately reused, just as CUDA permits after an accepted launch.
    argument_output = 0;
    argument_marker = 0;
    work = 0;
    parameters.fill(nullptr);
    if (!check(api.download(resources.readback, resources.output, kCount * sizeof(std::uint32_t),
                            low_stream),
               "queued readback") ||
        !check(api.record(resources.done, low_stream), "queued event") ||
        !check(api.wait(resources.streams[2], resources.done, 0), "queued wait") ||
        !check(api.download(resources.waited, resources.output, kCount * sizeof(std::uint32_t),
                            resources.streams[2]),
               "waited readback")) {
        return false;
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{10};
    while (std::atomic_ref<std::uint32_t>(resources.markers[0]).load(std::memory_order_acquire) ==
           0) {
        if (std::chrono::steady_clock::now() >= deadline) {
            std::cerr << "low kernel did not publish application progress\n";
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds{1});
    }
    if (cuEventQuery(resources.done) != CUDA_ERROR_NOT_READY ||
        api.query(low_stream) != CUDA_ERROR_NOT_READY) {
        std::cerr << "pending logical work became ready prematurely\n";
        return false;
    }
    if (uses_async_release) {
        if (!check(release_async(resources.output, resources.streams[2]),
                   "queued output release before high arrival")) {
            return false;
        }
        resources.output = 0;
    }
    CUdeviceptr high_marker = high_markers + sizeof(std::uint32_t);
    std::array<void*, 1> high_parameters{&high_marker};
    if (uses_cross_context &&
        !check(cuCtxSetCurrent(resources.high_context), "activate high caller")) {
        return false;
    }
    // Optional dynamic shared memory conservatively declines transformation;
    // both the cooperative and original caller paths must execute once.
    if (!check(api.launch(high, 1, 1, 1, 1, 1, 1, uses_cooperative_high ? 0 : 4,
                          resources.streams[1], high_parameters.data(), nullptr),
               "native high launch") ||
        !check(api.synchronize(resources.streams[1]), "high synchronize") ||
        resources.markers[1] != 1) {
        return false;
    }
    if (!check(cuCtxSetCurrent(resources.context), "restore low caller")) {
        return false;
    }
    const bool checks_quota =
        uses_async_release && std::getenv("GLIMMER_MEMORY_LIMIT_BYTES") != nullptr;
    std::size_t retained_free{};
    std::size_t quota_total{};
    if (checks_quota && !check(cuMemGetInfo(&retained_free, &quota_total), "retained free quota")) {
        return false;
    }
    if (releases_pending_output) {
        if (!check(cuMemFree(resources.output), "free output with deferred users")) {
            return false;
        }
        resources.output = 0;
        if (!check(cuMemFreeHost(resources.input), "free input with deferred users")) {
            return false;
        }
        resources.input = nullptr;
        if (!check(cuEventQuery(resources.done), "release proves logical completion")) {
            return false;
        }
    }
    if (!check(cuEventSynchronize(resources.done), "low event synchronize") ||
        !check(api.synchronize(resources.streams[2]), "wait synchronize")) {
        return false;
    }
    if (checks_quota) {
        std::size_t recovered_free{};
        std::size_t recovered_total{};
        if (!check(cuMemGetInfo(&recovered_free, &recovered_total), "completed free quota") ||
            recovered_total != quota_total || recovered_free != retained_free + copy_bytes) {
            std::cerr << "Driver queued release must retain then recover its quota exactly once\n";
            return false;
        }
    }
    for (std::size_t index = 0; index < kCount; ++index) {
        if (resources.readback[index] != 1 || resources.waited[index] != 1) {
            std::cerr << "logical thread " << index << " was lost or repeated\n";
            return false;
        }
    }
    if (!resources.cleanup()) {
        return false;
    }
    std::cout << "driver_cooperative_transparency status=ok threads=" << kCount << '\n';
    return true;
}
}  // namespace

int main(int argc, char** argv) {
    try {
        bool uses_proc = false;
        bool uses_dynamic = false;
        bool uses_ptds = false;
        bool uses_cooperative_high = false;
        bool releases_pending_output = false;
        bool uses_cross_context = false;
        bool uses_async_release = false;
        for (int index = 1; index < argc; ++index) {
            const std::string_view argument{argv[index]};
            if (argument == "--proc") {
                uses_proc = true;
            } else if (argument == "--dynamic") {
                uses_dynamic = true;
            } else if (argument == "--ptds") {
                uses_ptds = true;
            } else if (argument == "--cooperative-high") {
                uses_cooperative_high = true;
            } else if (argument == "--release") {
                releases_pending_output = true;
            } else if (argument == "--cross-context") {
                uses_cross_context = true;
            } else if (argument == "--async-release" || argument == "--async-release-ptds") {
                uses_async_release = true;
                uses_ptds = uses_ptds || argument == "--async-release-ptds";
            } else {
                return 1;
            }
        }
        return run(uses_proc, uses_dynamic, uses_ptds, uses_cooperative_high,
                   releases_pending_output, uses_cross_context, uses_async_release)
                   ? 0
                   : 1;
    } catch (...) {
        return 1;
    }
}
