#include <cuda.h>

#include <algorithm>
#include <chrono>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <future>
#include <stdexcept>
#include <string>
#include <vector>
#include <thread>

namespace {
void check(CUresult result, const char* operation) {
    if (result != CUDA_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": " + std::to_string(result));
}
constexpr const char* kSource = R"ptx(
.version 7.0
.target sm_70
.address_size 64
.visible .entry add(.param .u64 output, .param .u64 duration,
                    .param .u32 offset, .param .u32 value) {
 .reg .pred %p;
 .reg .b32 %r<8>;
 .reg .b64 %rd<8>;
 ld.param.u64 %rd1, [output]; ld.param.u64 %rd2, [duration];
 ld.param.u32 %r1, [offset]; ld.param.u32 %r2, [value];
 mov.u64 %rd3, %globaltimer;
WAIT:
 mov.u64 %rd4, %globaltimer; sub.u64 %rd5, %rd4, %rd3;
 setp.lt.u64 %p, %rd5, %rd2; @%p bra WAIT;
 mov.u32 %r3, %ctaid.x; mov.u32 %r4, %ntid.x; mov.u32 %r5, %tid.x;
 mad.lo.u32 %r6, %r3, %r4, %r5; add.u32 %r6, %r6, %r1;
 mul.wide.u32 %rd6, %r6, 4; add.u64 %rd7, %rd1, %rd6;
 atom.global.add.u32 %r7, [%rd7], %r2;
 ret;
}
.visible .entry wait_add(.param .u64 output, .param .u64 gate) {
 .reg .pred %p<3>;
 .reg .b32 %r<9>;
 .reg .b64 %rd<8>;
 ld.param.u64 %rd1, [output]; ld.param.u64 %rd2, [gate];
 mov.u64 %rd3, %globaltimer;
WAIT_GATE:
 ld.volatile.global.u32 %r1, [%rd2]; setp.ne.u32 %p1, %r1, 0; @%p1 bra READY;
 mov.u64 %rd4, %globaltimer; sub.u64 %rd5, %rd4, %rd3;
 setp.ge.u64 %p2, %rd5, 2000000000; @%p2 bra TIMEOUT;
 bra WAIT_GATE;
READY: mov.u32 %r2, 1; bra WRITE;
TIMEOUT: mov.u32 %r2, 100;
WRITE:
 mov.u32 %r3, %ctaid.x; mov.u32 %r4, %ntid.x; mov.u32 %r5, %tid.x;
 mad.lo.u32 %r6, %r3, %r4, %r5;
 mul.wide.u32 %rd6, %r6, 4; add.u64 %rd7, %rd1, %rd6;
 atom.global.add.u32 %r7, [%rd7], %r2;
 ret;
}
.visible .entry release_gate(.param .u64 gate) {
 .reg .pred %p;
 .reg .b32 %r;
 .reg .b64 %rd<6>;
 ld.param.u64 %rd1, [gate]; atom.global.exch.b32 %r, [%rd1], 1;
 mov.u64 %rd2, %globaltimer;
RELEASE_WAIT:
 mov.u64 %rd3, %globaltimer; sub.u64 %rd4, %rd3, %rd2;
 setp.lt.u64 %p, %rd4, 1000000000; @%p bra RELEASE_WAIT;
 ret;
}
)ptx";
void run(bool is_device_sanitizer_check) {
    CUdevice device{};
    check(cuDeviceGet(&device, 0), "device");
    int sms{};
    check(cuDeviceGetAttribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device), "SM count");
    if (sms != 80) throw std::runtime_error("explicit 320-CTA fixture requires the 80-SM V100");
    constexpr unsigned int workers = 320, elements = workers * 32;
    CUcontext context{}, observed{};
    check(cuCtxGetCurrent(&observed), "initial current context");
    if (observed) throw std::runtime_error("application must start without a native context");
    check(cuCtxCreate(&context, 0, device), "create context");
    check(cuCtxGetCurrent(&observed), "created current context");
    if (observed != context) throw std::runtime_error("public context identity mismatch");
    CUmodule module{};
    check(cuModuleLoadData(&module, kSource), "load module");
    CUfunction function{};
    check(cuModuleGetFunction(&function, module, "add"), "get function");
    std::vector<std::uint32_t> output(std::size_t{2} * elements, 0);
    CUdeviceptr memory{};
    check(cuMemAlloc(&memory, output.size() * sizeof(output[0])), "allocate");
    check(cuMemcpyHtoD(memory, output.data(), output.size() * sizeof(output[0])), "upload");
    CUstream stream{};
    check(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), "create stream");
    std::uint64_t duration = 500000000;
    std::uint32_t offset = 0, value = 1;
    void* arguments[]{&memory, &duration, &offset, &value};
    check(cuLaunchKernel(function, workers, 1, 1, 32, 1, 1, 0, stream, arguments, nullptr),
          "pointer-array launch");
    duration = 0;
    offset = 0;
    value = 99;
    struct Parameters {
        CUdeviceptr output{};
        std::uint64_t duration{};
        std::uint32_t offset{}, value{};
    } packed{memory, 500000000, elements, 3};
    static_assert(sizeof(Parameters) == 24);
    std::size_t bytes = sizeof(packed);
    void* extra[]{CU_LAUNCH_PARAM_BUFFER_POINTER, &packed, CU_LAUNCH_PARAM_BUFFER_SIZE, &bytes,
                  CU_LAUNCH_PARAM_END};
    check(cuLaunchKernel(function, workers, 1, 1, 32, 1, 1, 0, stream, nullptr, extra),
          "packed launch");
    packed = {};
    check(cuCtxSynchronize(), "complete original GPU work");
    check(cuMemcpyDtoH(output.data(), memory, output.size() * sizeof(output[0])), "download");
    for (std::size_t index = 0; index < output.size(); ++index)
        if (output[index] != (index < elements ? 1U : 3U))
            throw std::runtime_error("exact output mismatch");
    // This is still an ordinary Driver program. The waiting thread must not
    // prevent another thread from issuing the kernel that supplies its gate.
    CUfunction waiting{}, releasing{};
    check(cuModuleGetFunction(&waiting, module, "wait_add"), "get waiting function");
    check(cuModuleGetFunction(&releasing, module, "release_gate"), "get releasing function");
    std::fill(output.begin(), output.end(), 0);
    check(cuMemcpyHtoD(memory, output.data(), output.size() * sizeof(output[0])), "reset output");
    CUdeviceptr gate{};
    check(cuMemAlloc(&gate, 4), "allocate gate");
    std::uint32_t zero{};
    check(cuMemcpyHtoD(gate, &zero, sizeof(zero)), "initialize gate");
    CUstream releasing_stream{};
    check(cuStreamCreate(&releasing_stream, CU_STREAM_NON_BLOCKING),
          "create independent releasing stream");
    void* wait_arguments[]{&memory, &gate};
    check(cuLaunchKernel(waiting, workers, 1, 1, 32, 1, 1, 0, stream, wait_arguments, nullptr),
          "launch bounded gate wait");
    std::atomic<bool> has_entered{};
    auto waiter = std::async(std::launch::async, [&] {
        check(cuCtxSetCurrent(context), "bind waiting thread context");
        has_entered.store(true, std::memory_order_release);
        return cuStreamSynchronize(stream);
    });
    while (!has_entered.load(std::memory_order_acquire)) std::this_thread::yield();
    std::this_thread::sleep_for(std::chrono::milliseconds{100});
    void* release_arguments[]{&gate};
    check(cuLaunchKernel(releasing, workers, 1, 1, 32, 1, 1, 0, releasing_stream, release_arguments,
                         nullptr),
          "submit required gate release while another thread waits");
    check(waiter.get(), "complete only the waiting stream frontier");
    const auto independent = cuStreamQuery(releasing_stream);
    if (independent != CUDA_ERROR_NOT_READY && independent != CUDA_SUCCESS)
        check(independent, "query independent releasing stream");
    // Instrumentation may delay the host's launch return until the releasing
    // kernel has also finished. Device sanitizers retain every numerical and
    // lifetime check, but are not evidence for ordinary concurrent overlap.
    if (!is_device_sanitizer_check && independent != CUDA_ERROR_NOT_READY)
        throw std::runtime_error(
            "later independent stream was incorrectly included in the first wait");
    check(cuCtxSynchronize(), "complete both stream frontiers");
    check(cuMemcpyDtoH(output.data(), memory, output.size() * sizeof(output[0])),
          "download gated output");
    for (std::size_t index = 0; index < output.size(); ++index)
        if (output[index] != (index < elements ? 1U : 0U))
            throw std::runtime_error("gate expired or concurrent frontier outputs mismatch");
    check(cuMemFree(gate), "free gate");
    check(cuStreamDestroy(releasing_stream), "destroy releasing stream");
    check(cuMemFree(memory), "free");
    check(cuStreamDestroy(stream), "destroy stream");
    check(cuModuleUnload(module), "unload");
    check(cuCtxDestroy(context), "destroy context");
    check(cuCtxGetCurrent(&observed), "final current context");
    if (observed) throw std::runtime_error("context remains current after destruction");
    std::cout << "Ordinary Driver application verified: workers=320; elements=" << output.size()
              << "; pointer_array=true; packed_extra=true; concurrent_frontier="
              << (is_device_sanitizer_check ? "instrumented_not_qualified" : "true")
              << "; "
                 "full_device_qualified=false; "
                 "priority_or_timing_qualified=false\n";
}
}  // namespace
int main(int argc, char** argv) {
    try {
        const bool instrumented =
            argc == 2 && std::string_view(argv[1]) == "--device-sanitizer-check";
        if (argc != 1 && !instrumented)
            throw std::runtime_error("unknown ordinary Driver fixture option");
        check(cuInit(0), "initialize");
        run(instrumented);
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
