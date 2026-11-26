#pragma once

#include "cooperative_ptx_fixture.h"
#include "internal/owned_kernel_arguments.h"
#include "internal/ptx_transform.h"

#include <cuda.h>

#include <array>
#include <atomic>
#include <chrono>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace glimmer::test {

// Internal backend fixture, not an application-transparent launch client.
// CUDA work is explicitly stopped before reporting scheduler suspension.
class CooperativeGpuExecution final {
   public:
    CooperativeGpuExecution() {
        try {
            CUdevice device = 0;
            check(cuInit(0), "initialize remote GPU execution");
            check(cuDeviceGet(&device, 0), "select remote GPU");
            check(cuCtxCreate(&context_, CU_CTX_MAP_HOST, device), "create remote GPU context");
            check(cuStreamCreate(&stream_, CU_STREAM_NON_BLOCKING),
                  "create remote execution stream");
            check(cuMemAlloc(&output_, element_count * sizeof(std::uint32_t)),
                  "allocate remote output");
            check(cuMemAlloc(&counter_, sizeof(std::uint64_t)), "allocate remote counter");
            check(
                cuMemHostAlloc(&host_state_, 2 * sizeof(std::uint32_t), CU_MEMHOSTALLOC_DEVICEMAP),
                "allocate remote retreat state");
            check(cuMemHostGetDevicePointer(&mapped_state_, host_state_, 0),
                  "map remote retreat state");
            auto* state = static_cast<std::uint32_t*>(host_state_);
            state[0] = 0;
            state[1] = 0;
            marker_ = mapped_state_ + sizeof(std::uint32_t);
            const auto transformed = interceptor::transform_persistent_ptx(
                kCooperativePtx, interceptor::PtxExecutionKind::kPreemptive);
            if (!transformed.has_value() ||
                transformed->kernels.front().original_name != "cooperative" ||
                transformed->kernels.front().parameter_layout.size() != 3) {
                throw std::runtime_error("remote GPU fixture PTX should transform");
            }
            check(cuModuleLoadData(&module_, transformed->image.c_str()), "JIT remote variant");
            check(cuModuleGetFunction(&function_, module_,
                                      transformed->kernels[0].persistent_name.c_str()),
                  "resolve remote variant");
            check(cuMemsetD32(output_, 0, element_count), "initialize remote coverage");
            check(cuMemsetD32(counter_, 0, 2), "initialize remote claim state");
            std::uint32_t original_work = 50'000;
            std::array<void*, 3> original_arguments{&output_, &marker_, &original_work};
            owned_ = interceptor::OwnedKernelArguments::copy(
                transformed->kernels[0].parameter_layout, original_arguments.data());
            if (!owned_.has_value()) {
                throw std::runtime_error("remote launch argument ownership failed");
            }
            // original_work and original_arguments die here. Every continuation
            // below can use only backend-owned copies, never caller storage.
        } catch (...) {
            cleanup();
            throw;
        }
    }
    ~CooperativeGpuExecution() {
        cleanup();
    }
    CooperativeGpuExecution(const CooperativeGpuExecution&) = delete;
    CooperativeGpuExecution& operator=(const CooperativeGpuExecution&) = delete;
    CooperativeGpuExecution(CooperativeGpuExecution&&) = delete;
    CooperativeGpuExecution& operator=(CooperativeGpuExecution&&) = delete;

    void launch(std::uint32_t workers) {
        std::array<void*, 8> arguments{nullptr,  nullptr,  nullptr,   &grid_x_,
                                       &grid_y_, &grid_z_, &counter_, &mapped_state_};
        for (std::size_t index = 0; index < owned_->addresses().size(); ++index) {
            arguments[index] = owned_->addresses()[index];
        }
        set_retreat(0);
        check(cuLaunchKernel(function_, workers, 1, 1, 8, 4, 2, 0, stream_, arguments.data(),
                             nullptr),
              "dispatch remote cooperative execution");
    }

    void await_progress() {
        auto* state = static_cast<std::uint32_t*>(host_state_);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
        while (std::atomic_ref(state[1]).load(std::memory_order_acquire) == 0 &&
               std::chrono::steady_clock::now() < deadline) {
            std::this_thread::yield();
        }
        if (std::atomic_ref(state[1]).load(std::memory_order_acquire) == 0) {
            throw std::runtime_error("remote GPU progress deadline expired");
        }
    }

    std::uint64_t stop() {
        set_retreat(1);
        check(cuStreamSynchronize(stream_), "fence remote physical retreat");
        std::uint64_t next = 0;
        check(cuMemcpyDtoH(&next, counter_, sizeof(next)), "read remote retained progress");
        if (next == 0 || next >= logical_block_count) {
            throw std::runtime_error("remote retreat must retain partial work, next=" +
                                     std::to_string(next));
        }
        return next;
    }

    void finish() {
        check(cuStreamSynchronize(stream_), "fence resumed remote execution");
        std::vector<std::uint32_t> values(element_count);
        check(cuMemcpyDtoH(values.data(), output_, values.size() * sizeof(values[0])),
              "read remote exactly-once coverage");
        for (const auto value : values) {
            if (value != 1) {
                throw std::runtime_error("every resumed remote logical thread must execute once");
            }
        }
    }

    void priority_probe() {
        CUfunction probe = nullptr;
        check(cuModuleGetFunction(&probe, module_, "priority_probe"),
              "resolve high-priority probe");
        std::array<void*, 1> arguments{&marker_};
        check(cuLaunchKernel(probe, 1, 1, 1, 1, 1, 1, 0, stream_, arguments.data(), nullptr),
              "launch high-priority process GPU work");
        check(cuStreamSynchronize(stream_), "fence high-priority GPU work");
        auto* state = static_cast<std::uint32_t*>(host_state_);
        if (std::atomic_ref(state[1]).load(std::memory_order_acquire) != 2) {
            throw std::runtime_error(
                "high-priority process must execute real GPU work before completion");
        }
    }

   private:
    static void check(CUresult result, std::string_view operation) {
        if (result != CUDA_SUCCESS) {
            throw std::runtime_error(std::string(operation) + ": CUresult " +
                                     std::to_string(static_cast<int>(result)));
        }
    }
    void set_retreat(std::uint32_t value) {
        auto* state = static_cast<std::uint32_t*>(host_state_);
        std::atomic_ref(state[0]).store(value, std::memory_order_release);
    }
    void cleanup() noexcept {
        if (context_ == nullptr) {
            return;
        }
        // Release the device work before removing any pointer or module used
        // by the launch, including partial constructor/error paths.
        if (host_state_ != nullptr) {
            set_retreat(1);
        }
        const auto drain = cuCtxSynchronize();
        if (drain != CUDA_SUCCESS) {
            // Context destruction below is the recovery boundary after an
            // asynchronous device error; no launch is retried.
            cleanup_failed_ = true;
        }
        if (module_ != nullptr && cuModuleUnload(module_) != CUDA_SUCCESS) {
            cleanup_failed_ = true;
        }
        if (stream_ != nullptr && cuStreamDestroy(stream_) != CUDA_SUCCESS) {
            cleanup_failed_ = true;
        }
        for (const auto pointer : {output_, counter_}) {
            if (pointer != 0 && cuMemFree(pointer) != CUDA_SUCCESS) {
                cleanup_failed_ = true;
            }
        }
        if (host_state_ != nullptr && cuMemFreeHost(host_state_) != CUDA_SUCCESS) {
            cleanup_failed_ = true;
        }
        if (cuCtxDestroy(context_) != CUDA_SUCCESS) {
            cleanup_failed_ = true;
        }
        context_ = nullptr;
        if (cleanup_failed_) {
            std::terminate();
        }
    }

    static constexpr std::uint32_t logical_block_count = 7 * 5 * 3;
    static constexpr std::uint32_t element_count = logical_block_count * 8 * 4 * 2;
    CUcontext context_ = nullptr;
    CUmodule module_ = nullptr;
    CUfunction function_ = nullptr;
    CUstream stream_ = nullptr;
    CUdeviceptr output_ = 0;
    CUdeviceptr counter_ = 0;
    void* host_state_ = nullptr;
    CUdeviceptr mapped_state_ = 0;
    CUdeviceptr marker_ = 0;
    std::uint32_t grid_x_ = 7;
    std::uint32_t grid_y_ = 5;
    std::uint32_t grid_z_ = 3;
    std::optional<interceptor::OwnedKernelArguments> owned_;
    bool cleanup_failed_ = false;
};

}  // namespace glimmer::test
