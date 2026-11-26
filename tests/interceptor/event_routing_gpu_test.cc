#include <cuda.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include "cooperative_ptx_fixture.h"

#include <array>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

extern "C" CUresult CUDAAPI direct_old_capture(CUstream) asm("cuStreamBeginCapture");
extern "C" CUresult CUDAAPI direct_old_capture_ptsz(CUstream) asm("cuStreamBeginCapture_ptsz");

namespace {
void require(bool value, const char* message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}
void check(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
    }
}
void check(CUresult result, const char* operation) {
    if (result != CUDA_SUCCESS) {
        throw std::runtime_error(std::string(operation) + ": CUresult " + std::to_string(result));
    }
}
struct RuntimeResources {
    cudaStream_t producer{};
    cudaStream_t consumer{};
    cudaEvent_t begin{};
    cudaEvent_t end{};
    cudaEvent_t join{};
    void* device{};
    void* host{};
    cudaGraph_t graph{};
    cudaGraphExec_t executable{};
    ~RuntimeResources() {
        bool passed = cudaDeviceSynchronize() == cudaSuccess;
        if (executable) {
            passed &= cudaGraphExecDestroy(executable) == cudaSuccess;
        }
        if (graph) {
            passed &= cudaGraphDestroy(graph) == cudaSuccess;
        }
        for (const auto event : {begin, end, join}) {
            if (event) {
                passed &= cudaEventDestroy(event) == cudaSuccess;
            }
        }
        for (const auto stream : {producer, consumer}) {
            if (stream) {
                passed &= cudaStreamDestroy(stream) == cudaSuccess;
            }
        }
        if (device) {
            passed &= cudaFree(device) == cudaSuccess;
        }
        if (host) {
            passed &= cudaFreeHost(host) == cudaSuccess;
        }
        if (!passed) {
            std::terminate();
        }
    }
};

void runtime_events() {
    RuntimeResources resources;
    check(cudaStreamCreateWithFlags(&resources.producer, cudaStreamNonBlocking),
          "Runtime producer");
    check(cudaStreamCreateWithFlags(&resources.consumer, cudaStreamNonBlocking),
          "Runtime consumer");
    check(cudaEventCreate(&resources.begin), "Runtime timing event");
    check(cudaEventCreateWithFlags(&resources.end, cudaEventBlockingSync),
          "Runtime blocking timing event");
    check(cudaEventCreateWithFlags(&resources.join, cudaEventDisableTiming),
          "Runtime dependency event");
    check(cudaMalloc(&resources.device, 64), "Runtime output");
    check(cudaMallocHost(&resources.host, 64), "Runtime pinned output");
    check(cudaEventQuery(resources.end), "Runtime unrecorded event");
    check(cudaStreamWaitEvent(resources.consumer, resources.end, 0), "Runtime unrecorded wait");
    check(cudaEventRecord(resources.begin, resources.producer), "Runtime start record");
    check(cudaMemsetAsync(resources.device, 7, 64, resources.producer), "Runtime producer data");
    check(cudaEventRecordWithFlags(resources.end, resources.producer, 0),
          "Runtime end record with flags");
    check(cudaStreamWaitEvent(resources.consumer, resources.end, 0), "Runtime cross-stream wait");
    check(cudaMemcpyAsync(resources.host, resources.device, 64, cudaMemcpyDeviceToHost,
                          resources.consumer),
          "Runtime dependent readback");
    check(cudaStreamSynchronize(resources.consumer), "Runtime dependent completion");
    require(static_cast<unsigned char*>(resources.host)[0] == 7, "Runtime wait must order memory");
    check(cudaEventSynchronize(resources.end), "Runtime event synchronization");
    float milliseconds = 0;
    check(cudaEventElapsedTime(&milliseconds, resources.begin, resources.end),
          "Runtime physical timing");
    require(milliseconds >= 0, "Runtime native timing must remain valid");

    using Record = cudaError_t (*)(cudaEvent_t, cudaStream_t);
    using RecordFlags = cudaError_t (*)(cudaEvent_t, cudaStream_t, unsigned int);
    using Wait = cudaError_t (*)(cudaStream_t, cudaEvent_t, unsigned int);
    auto record = reinterpret_cast<Record>(dlsym(RTLD_DEFAULT, "cudaEventRecord_ptsz"));
    auto record_flags =
        reinterpret_cast<RecordFlags>(dlsym(RTLD_DEFAULT, "cudaEventRecordWithFlags_ptsz"));
    auto wait = reinterpret_cast<Wait>(dlsym(RTLD_DEFAULT, "cudaStreamWaitEvent_ptsz"));
    require(record && record_flags && wait, "Runtime PTDS event entry points");
    check(record(resources.end, nullptr), "Runtime PTDS rerecord");
    check(wait(nullptr, resources.end, 0), "Runtime PTDS wait");
    check(record_flags(resources.end, nullptr, 0), "Runtime PTDS record flags");
    check(cudaEventSynchronize(resources.end), "Runtime latest record generation");

    check(cudaStreamBeginCapture(resources.producer, cudaStreamCaptureModeGlobal),
          "Runtime event capture begin");
    check(cudaMemsetAsync(resources.device, 8, 64, resources.producer), "captured producer data");
    check(cudaEventRecordWithFlags(resources.end, resources.producer, 0), "captured record flags");
    check(cudaStreamWaitEvent(resources.consumer, resources.end, 0), "captured cross-stream wait");
    check(cudaMemsetAsync(resources.device, 9, 64, resources.consumer), "captured consumer data");
    check(cudaEventRecord(resources.join, resources.consumer), "captured join record");
    check(cudaStreamWaitEvent(resources.producer, resources.join, 0), "captured stream rejoin");
    check(cudaStreamEndCapture(resources.producer, &resources.graph), "Runtime event capture end");
    check(cudaGraphInstantiate(&resources.executable, resources.graph, 0),
          "Runtime event graph instantiate");
    check(cudaGraphLaunch(resources.executable, resources.producer), "Runtime event graph replay");
    check(cudaStreamSynchronize(resources.producer), "Runtime event graph completion");
    check(cudaMemcpy(resources.host, resources.device, 64, cudaMemcpyDeviceToHost),
          "Runtime graph readback");
    require(static_cast<unsigned char*>(resources.host)[0] == 9,
            "captured event dependencies must remain native");

    cudaError_t thread_result = cudaErrorUnknown;
    std::thread fresh([&] {
        cudaEvent_t event = nullptr;
        thread_result = cudaEventCreate(&event);
        if (thread_result == cudaSuccess) {
            thread_result = cudaEventRecord(event, nullptr);
        }
        if (thread_result == cudaSuccess) {
            thread_result = cudaEventSynchronize(event);
        }
        if (event) {
            const auto destroyed = cudaEventDestroy(event);
            if (thread_result == cudaSuccess) {
                thread_result = destroyed;
            }
        }
    });
    fresh.join();
    check(thread_result, "Runtime event first use on unbound host thread");
}

struct DriverResources {
    CUstream stream{};
    CUevent begin{};
    CUevent end{};
    CUmodule module{};
    CUdeviceptr counter{};
    CUgraph graph{};
    CUgraphExec executable{};
    ~DriverResources() {
        bool passed = cuCtxSynchronize() == CUDA_SUCCESS;
        for (const auto event : {begin, end}) {
            if (event) {
                passed &= cuEventDestroy(event) == CUDA_SUCCESS;
            }
        }
        if (executable) {
            passed &= cuGraphExecDestroy(executable) == CUDA_SUCCESS;
        }
        if (graph) {
            passed &= cuGraphDestroy(graph) == CUDA_SUCCESS;
        }
        if (counter) {
            passed &= cuMemFree(counter) == CUDA_SUCCESS;
        }
        if (module) {
            passed &= cuModuleUnload(module) == CUDA_SUCCESS;
        }
        if (stream) {
            passed &= cuStreamDestroy(stream) == CUDA_SUCCESS;
        }
        if (!passed) {
            std::terminate();
        }
    }
};

void old_driver_capture() {
    using Begin = CUresult(CUDAAPI*)(CUstream);
    using End = CUresult(CUDAAPI*)(CUstream, CUgraph*);
    using Query = CUresult(CUDAAPI*)(const char*, void**, int, cuuint64_t);
    const auto query = reinterpret_cast<Query>(dlsym(RTLD_DEFAULT, "cuGetProcAddress"));
    require(query, "old Driver proc-address resolver available");
    DriverResources resources;
    check(cuStreamCreate(&resources.stream, CU_STREAM_NON_BLOCKING), "old ABI original stream");
    check(cuModuleLoadData(&resources.module, kNativeOncePtx), "old ABI original module");
    CUfunction function{};
    check(cuModuleGetFunction(&function, resources.module, "native_once"),
          "old ABI original entry");
    check(cuMemAlloc(&resources.counter, sizeof(unsigned int)), "old ABI counter");
    const std::array<Begin, 2> direct{&direct_old_capture, &direct_old_capture_ptsz};
    for (const bool is_ptds : {false, true}) {
        const auto flags = is_ptds ? CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM
                                   : CU_GET_PROC_ADDRESS_LEGACY_STREAM;
        const auto dynamic = reinterpret_cast<Begin>(
            dlsym(RTLD_DEFAULT, is_ptds ? "cuStreamBeginCapture_ptsz" : "cuStreamBeginCapture"));
        const auto end = reinterpret_cast<End>(
            dlsym(RTLD_DEFAULT, is_ptds ? "cuStreamEndCapture_ptsz" : "cuStreamEndCapture"));
        void* old_proc{};
        void* modern_proc{};
        check(query("cuStreamBeginCapture", &old_proc, 10000, flags), "old ABI resolver");
        check(cuGetProcAddress("cuStreamBeginCapture", &modern_proc, 10000, flags, nullptr),
              "old ABI extended resolver");
        require(dynamic && end && old_proc && modern_proc,
                "exact old ABI entry available through every original lookup");
        for (const auto begin : {direct[is_ptds], dynamic, reinterpret_cast<Begin>(old_proc),
                                 reinterpret_cast<Begin>(modern_proc)}) {
            check(cuMemsetD32(resources.counter, 0, 1), "initialize original capture counter");
            check(begin(resources.stream), "one-argument original capture begin");
            std::array<void*, 1> arguments{&resources.counter};
            check(cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, resources.stream, arguments.data(),
                                 nullptr),
                  "old ABI capture original kernel");
            check(end(resources.stream, &resources.graph), "old ABI capture end");
            std::array<CUgraphNode, 2> nodes{};
            auto count = nodes.size();
            check(cuGraphGetNodes(resources.graph, nodes.data(), &count), "old ABI original nodes");
            require(count == 1, "old ABI capture contains only the original kernel node");
            CUDA_KERNEL_NODE_PARAMS params{};
            check(cuGraphKernelNodeGetParams(nodes[0], &params), "old ABI original parameters");
            require(params.func == function && params.gridDimX == 1 && params.blockDimX == 1,
                    "old ABI capture preserves original function and dimensions");
            check(cuGraphInstantiateWithFlags(&resources.executable, resources.graph, 0),
                  "old ABI native instantiate");
            check(cuGraphLaunch(resources.executable, resources.stream), "old ABI native replay");
            check(cuStreamSynchronize(resources.stream), "old ABI replay completion");
            unsigned int value{};
            check(cuMemcpyDtoH(&value, resources.counter, sizeof(value)),
                  "old ABI replay readback");
            require(value == 1, "old ABI graph executes the original increment exactly once");
            check(cuGraphExecDestroy(resources.executable), "old ABI executable cleanup");
            resources.executable = nullptr;
            check(cuGraphDestroy(resources.graph), "old ABI graph cleanup");
            resources.graph = nullptr;
        }
    }
}
void driver_events() {
    DriverResources resources;
    check(cuStreamCreate(&resources.stream, CU_STREAM_NON_BLOCKING), "Driver stream");
    check(cuEventCreate(&resources.begin, CU_EVENT_DEFAULT), "Driver timing event");
    check(cuEventCreate(&resources.end, CU_EVENT_BLOCKING_SYNC), "Driver blocking event");
    check(cuEventQuery(resources.end), "Driver unrecorded event");
    check(cuStreamWaitEvent(resources.stream, resources.end, 0), "Driver unrecorded wait");
    check(cuEventRecord(resources.begin, resources.stream), "Driver start record");
    check(cuEventRecordWithFlags(resources.end, resources.stream, 0), "Driver end record flags");
    check(cuEventSynchronize(resources.end), "Driver event sync");
    float milliseconds = 0;
    check(cuEventElapsedTime(&milliseconds, resources.begin, resources.end),
          "Driver physical timing");
    require(milliseconds >= 0, "Driver native timing must remain valid");
    void* pointer = nullptr;
    check(cuGetProcAddress("cuEventRecord", &pointer, 12000,
                           CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM, nullptr),
          "Driver PTDS proc record");
    auto record = reinterpret_cast<CUresult (*)(CUevent, CUstream)>(pointer);
    require(record != nullptr, "Driver record proc result");
    check(record(resources.end, nullptr), "Driver PTDS record");
    check(cuGetProcAddress("cuStreamWaitEvent", &pointer, 12000,
                           CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM, nullptr),
          "Driver PTDS proc wait");
    auto wait = reinterpret_cast<CUresult (*)(CUstream, CUevent, unsigned int)>(pointer);
    require(wait != nullptr, "Driver wait proc result");
    check(wait(nullptr, resources.end, 0), "Driver PTDS wait");
    auto record_flags = reinterpret_cast<CUresult (*)(CUevent, CUstream, unsigned int)>(
        dlsym(RTLD_DEFAULT, "cuEventRecordWithFlags_ptsz"));
    require(record_flags != nullptr, "Driver PTDS flags symbol");
    check(record_flags(resources.end, nullptr, 0), "Driver PTDS flags");
    check(cuEventSynchronize(resources.end), "Driver latest event generation");
}
}  // namespace
int main() {
    try {
        runtime_events();
        driver_events();
        old_driver_capture();
        std::cout
            << "Native Driver/Runtime event, timing, PTDS, capture and lazy-thread checks passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
