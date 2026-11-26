#include "cooperative_ptx_fixture.h"
#include "glimmer/core/scheduler.h"
#include "internal/owned_kernel_arguments.h"
#include "internal/ptx_transform.h"

#include <cuda.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string_view>
#include <thread>
#include <vector>

namespace {

using glimmer::interceptor::PtxExecutionKind;
constexpr std::uint32_t kGridX = 7;
constexpr std::uint32_t kGridY = 5;
constexpr std::uint32_t kGridZ = 3;
constexpr std::uint32_t kBlocks = kGridX * kGridY * kGridZ;
constexpr std::uint32_t kThreads = 8 * 4 * 2;
constexpr std::uint32_t kElements = kBlocks * kThreads;

bool check(CUresult result, std::string_view operation) {
    if (result != CUDA_SUCCESS) {
        std::cerr << operation << " failed with CUresult " << static_cast<int>(result) << '\n';
        return false;
    }
    return true;
}

struct Resources {
    CUcontext context = nullptr;
    CUmodule module = nullptr;
    CUstream stream = nullptr;
    CUdeviceptr output = 0;
    CUdeviceptr counter = 0;
    void* host_state = nullptr;
    CUdeviceptr mapped_state = 0;
    ~Resources() {
        (void)cleanup();
    }
    bool cleanup() {
        bool passed = true;
        if (context != nullptr) {
            passed &= check(cuCtxSynchronize(), "drain cooperative work");
        }
        if (module != nullptr) {
            passed &= check(cuModuleUnload(module), "unload cooperative module");
            module = nullptr;
        }
        if (stream != nullptr) {
            passed &= check(cuStreamDestroy(stream), "destroy cooperative stream");
            stream = nullptr;
        }
        for (auto* pointer : {&output, &counter}) {
            if (*pointer != 0) {
                passed &= check(cuMemFree(*pointer), "free cooperative storage");
                *pointer = 0;
            }
        }
        if (host_state != nullptr) {
            passed &= check(cuMemFreeHost(host_state), "free retreat state");
            host_state = nullptr;
        }
        if (context != nullptr) {
            passed &= check(cuCtxDestroy(context), "destroy cooperative context");
            context = nullptr;
        }
        return passed;
    }
};

bool validate(Resources& resources) {
    std::vector<std::uint32_t> values(kElements);
    if (!check(cuMemcpyDtoH(values.data(), resources.output, values.size() * sizeof(values[0])),
               "read exactly-once coverage")) {
        return false;
    }
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (values[index] != 1) {
            std::cerr << "logical thread " << index << " executed " << values[index] << " times\n";
            return false;
        }
    }
    return true;
}

bool run(PtxExecutionKind kind) {
    Resources resources;
    CUdevice device = 0;
    const auto transformed = glimmer::interceptor::transform_persistent_ptx(kCooperativePtx, kind);
    if (!transformed.has_value() || transformed->kernels.front().original_name != "cooperative" ||
        transformed->kernels.front().parameter_layout.size() != 3 ||
        !check(cuInit(0), "initialize Driver") || !check(cuDeviceGet(&device, 0), "select GPU") ||
        !check(cuCtxCreate(&resources.context, CU_CTX_MAP_HOST, device), "create context") ||
        !check(cuStreamCreate(&resources.stream, CU_STREAM_NON_BLOCKING), "create stream") ||
        !check(cuMemAlloc(&resources.output, kElements * sizeof(std::uint32_t)),
               "allocate output") ||
        !check(cuMemAlloc(&resources.counter, sizeof(std::uint64_t)), "allocate claim counter") ||
        !check(cuMemHostAlloc(&resources.host_state, 2 * sizeof(std::uint32_t),
                              CU_MEMHOSTALLOC_DEVICEMAP),
               "allocate mapped retreat state") ||
        !check(cuMemHostGetDevicePointer(&resources.mapped_state, resources.host_state, 0),
               "map retreat state") ||
        !check(cuModuleLoadData(&resources.module, transformed->image.c_str()),
               "JIT cooperative variant")) {
        return false;
    }
    auto* state = static_cast<std::uint32_t*>(resources.host_state);
    state[0] = 0;
    state[1] = 0;
    auto marker = resources.mapped_state + sizeof(std::uint32_t);
    CUfunction function = nullptr;
    if (!check(cuModuleGetFunction(&function, resources.module,
                                   transformed->kernels[0].persistent_name.c_str()),
               "find cooperative variant") ||
        !check(cuMemsetD32(resources.output, 0, kElements), "clear output") ||
        !check(cuMemsetD32(resources.counter, 0, 2), "clear claim counter")) {
        return false;
    }
    auto grid_x = kGridX;
    auto grid_y = kGridY;
    auto grid_z = kGridZ;
    std::uint32_t work = kind == PtxExecutionKind::kPreemptive ? 50000 : 0;
    std::array<void*, 8> arguments{
        &resources.output,      &marker, &work, &grid_x, &grid_y, &grid_z, &resources.counter,
        &resources.mapped_state};
    std::array<void*, 3> application_arguments{&resources.output, &marker, &work};
    auto owned = glimmer::interceptor::OwnedKernelArguments::copy(
        transformed->kernels[0].parameter_layout, application_arguments.data());
    if (!owned.has_value()) {
        return false;
    }
    for (std::size_t index = 0; index < application_arguments.size(); ++index) {
        arguments[index] = owned->addresses()[index];
    }
    work = 0;
    application_arguments.fill(nullptr);
    const auto launch = [&](std::uint32_t workers) {
        return check(cuLaunchKernel(function, workers, 1, 1, 8, 4, 2, 0, resources.stream,
                                    arguments.data(), nullptr),
                     "launch cooperative variant");
    };
    if (kind == PtxExecutionKind::kSliced) {
        std::uint32_t start = 0;
        std::uint32_t count = 11;
        arguments[6] = &start;
        arguments[7] = &count;
        for (; start < kBlocks; start += count) {
            if (!launch(count)) {
                return false;
            }
        }
    } else if (kind == PtxExecutionKind::kPreemptive) {
        glimmer::core::Scheduler scheduler(
            4, {.scheduling_policy = glimmer::core::SchedulingPolicy::kPriority});
        const auto low = scheduler.submit({.tenant_id = "low", .memory_bytes = 1, .priority = 1});
        if (!low.accepted() || !scheduler.dispatch_task(low.task_id).has_value()) {
            return false;
        }
        // A pre-existing request must claim no work.
        std::atomic_ref(state[0]).store(1, std::memory_order_release);
        if (!launch(3) || !check(cuStreamSynchronize(resources.stream), "pre-requested retreat")) {
            return false;
        }
        std::uint64_t next = 0;
        if (!check(cuMemcpyDtoH(&next, resources.counter, sizeof(next)),
                   "read untouched counter") ||
            next != 0) {
            return false;
        }
        std::atomic_ref(state[0]).store(0, std::memory_order_release);
        if (!launch(3)) {
            return false;
        }
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{10};
        while (std::atomic_ref(state[1]).load(std::memory_order_acquire) == 0 &&
               std::chrono::steady_clock::now() < deadline) {
            std::this_thread::yield();
        }
        if (std::atomic_ref(state[1]).load(std::memory_order_acquire) == 0) {
            std::cerr << "GPU did not publish progress before retreat deadline\n";
            return false;
        }
        const auto high = scheduler.submit({.tenant_id = "high", .memory_bytes = 1, .priority = 9});
        if (!high.accepted() || !scheduler.retreat_requested(low.task_id)) {
            return false;
        }
        std::atomic_ref(state[0]).store(1, std::memory_order_release);
        if (!check(cuStreamSynchronize(resources.stream), "wait for physical retreat") ||
            !check(cuMemcpyDtoH(&next, resources.counter, sizeof(next)),
                   "read retained progress") ||
            next == 0 || next >= kBlocks) {
            std::cerr << "retreat did not retain partial progress: " << next << '\n';
            return false;
        }
        std::cout << "cooperative_retreat completed_blocks=" << next << " total_blocks=" << kBlocks
                  << '\n';
        CUfunction probe = nullptr;
        void* probe_arguments[] = {&marker};
        if (!scheduler.suspend(low.task_id) || !scheduler.dispatch_task(high.task_id).has_value() ||
            !check(cuModuleGetFunction(&probe, resources.module, "priority_probe"),
                   "find priority probe") ||
            !check(cuLaunchKernel(probe, 1, 1, 1, 1, 1, 1, 0, resources.stream, probe_arguments,
                                  nullptr),
                   "execute high-priority work before resumption") ||
            !check(cuStreamSynchronize(resources.stream), "complete high-priority work") ||
            std::atomic_ref(state[1]).load(std::memory_order_acquire) != 2 ||
            !scheduler.complete(high.task_id) ||
            !scheduler.dispatch_task(low.task_id).has_value()) {
            return false;
        }
        // Resume with different occupancy, without resetting claimed work.
        std::atomic_ref(state[0]).store(0, std::memory_order_release);
        work = 0;
        if (!launch(7)) {
            return false;
        }
        if (!check(cuStreamSynchronize(resources.stream), "complete resumed low work") ||
            !scheduler.complete(low.task_id) || scheduler.usage().used_bytes() != 0) {
            return false;
        }
    } else if (!launch(3)) {
        return false;
    }
    if (!check(cuStreamSynchronize(resources.stream), "finish logical work") ||
        !validate(resources)) {
        return false;
    }
    if (kind == PtxExecutionKind::kBlockStatic) {
        if (!check(cuMemsetD32(resources.output, 0, kElements), "clear overlaunch output") ||
            !launch(kBlocks + 3) ||
            !check(cuStreamSynchronize(resources.stream), "finish idle block-static workers") ||
            !validate(resources)) {
            return false;
        }
    }
    return resources.cleanup();
}

}  // namespace

int main() {
    return run(PtxExecutionKind::kBlockStatic) && run(PtxExecutionKind::kDynamic) &&
                   run(PtxExecutionKind::kSliced) && run(PtxExecutionKind::kPreemptive)
               ? EXIT_SUCCESS
               : EXIT_FAILURE;
}
