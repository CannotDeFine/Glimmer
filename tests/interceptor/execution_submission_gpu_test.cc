#include <cuda.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
void check(CUresult result, const char* operation) {
    if (result != CUDA_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": " + std::to_string(result));
}
constexpr unsigned int kWorkers = 320, kThreads = 32, kElements = kWorkers * kThreads;
constexpr const char* kSource = R"ptx(
.version 7.0
.target sm_70
.address_size 64
.visible .entry producer(.param .u64 output, .param .u64 gate, .param .u32 limit) {
 .reg .pred %p<3>;
 .reg .b32 %r<9>;
 .reg .b64 %rd<6>;
 ld.param.u64 %rd1, [output]; ld.param.u64 %rd2, [gate]; ld.param.u32 %r1, [limit];
 mov.u32 %r2, %ctaid.x; mov.u32 %r3, %ntid.x; mov.u32 %r4, %tid.x;
 mad.lo.u32 %r5, %r2, %r3, %r4;
 mul.wide.u32 %rd3, %r5, 4; add.u64 %rd4, %rd1, %rd3;
WAIT:
 ld.volatile.global.u32 %r6, [%rd2]; setp.ne.u32 %p1, %r6, 0; @%p1 bra READY;
 sub.u32 %r1, %r1, 1; setp.ne.u32 %p2, %r1, 0; @%p2 bra WAIT;
 mov.u32 %r7, 100; bra WRITE;
READY: mov.u32 %r7, 1;
WRITE: st.global.u32 [%rd4], %r7; ret;
}
.visible .entry dependent(.param .u64 output) {
 .reg .b32 %r<6>;
 .reg .b64 %rd<4>;
 // A legal but untransformed barrier keeps this helper on its original ABI.
 bar.sync 15;
 ld.param.u64 %rd1, [output];
 mov.u32 %r1, %ctaid.x; mov.u32 %r2, %ntid.x; mov.u32 %r3, %tid.x;
 mad.lo.u32 %r4, %r1, %r2, %r3; add.u32 %r4, %r4, 10240;
 mul.wide.u32 %rd2, %r4, 4; add.u64 %rd3, %rd1, %rd2;
 st.global.u32 [%rd3], 2; ret;
}
.visible .entry release_gate(.param .u64 gate) {
 .reg .pred %p;
 .reg .b32 %r;
 .reg .b64 %rd<6>;
 ld.param.u64 %rd1, [gate]; atom.global.exch.b32 %r, [%rd1], 1;
 mov.u64 %rd2, %globaltimer;
HOLD:
 mov.u64 %rd3, %globaltimer; sub.u64 %rd4, %rd3, %rd2;
 setp.lt.u64 %p, %rd4, 1000000000; @%p bra HOLD;
 ret;
}
)ptx";
void run(bool is_instrumented) {
    CUdevice device{};
    check(cuDeviceGet(&device, 0), "select device");
    int sms{};
    check(cuDeviceGetAttribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device), "SM count");
    if (sms != 80) throw std::runtime_error("fixture requires the identified 80-SM V100");
    CUcontext context{};
    check(cuCtxCreate(&context, 0, device), "create context");
    CUmodule module{};
    check(cuModuleLoadData(&module, kSource), "load unchanged original module");
    CUfunction producer{}, dependent{}, releasing{};
    check(cuModuleGetFunction(&producer, module, "producer"), "get original producer");
    check(cuModuleGetFunction(&dependent, module, "dependent"), "get original dependent");
    check(cuModuleGetFunction(&releasing, module, "release_gate"), "get original release");
    std::vector<std::uint32_t> output(std::size_t{2} * kElements, 0);
    CUdeviceptr memory{}, gate{};
    check(cuMemAlloc(&memory, output.size() * sizeof(output.front())), "allocate output");
    check(cuMemAlloc(&gate, sizeof(std::uint32_t)), "allocate gate");
    std::uint32_t zero{};
    check(cuMemcpyHtoD(memory, output.data(), output.size() * sizeof(output.front())),
          "initialize output");
    check(cuMemcpyHtoD(gate, &zero, sizeof(zero)), "initialize gate");
    CUstream blocking{}, independent{};
    check(cuStreamCreate(&blocking, CU_STREAM_DEFAULT), "create blocking producer stream");
    check(cuStreamCreate(&independent, CU_STREAM_NON_BLOCKING),
          "create nonblocking release stream");
    std::uint32_t limit = 16777216;
    void* producer_arguments[]{&memory, &gate, &limit};
    check(cuLaunchKernel(producer, kWorkers, 1, 1, kThreads, 1, 1, 0, blocking, producer_arguments,
                         nullptr),
          "launch bounded producer");
    std::atomic<bool> has_entered{};
    auto pending = std::async(std::launch::async, [&] {
        check(cuCtxSetCurrent(context), "bind dependent caller context");
        void* arguments[]{&memory};
        has_entered.store(true, std::memory_order_release);
        return cuLaunchKernel(dependent, kWorkers, 1, 1, kThreads, 1, 1, 0, nullptr, arguments,
                              nullptr);
    });
    while (!has_entered.load(std::memory_order_acquire)) std::this_thread::yield();
    // Ordinary benchmark input lets its launcher establish a reproducible
    // arrival boundary; no scheduling or Glimmer API is used by this program.
    std::cout << "Submission request ready\n" << std::flush;
    char command{};
    if (!std::cin.get(command) || command != 'r')
        throw std::runtime_error("missing benchmark release input");
    const auto before_release = cuStreamQuery(nullptr);
    if (before_release != CUDA_ERROR_NOT_READY)
        throw std::runtime_error(
            "producer escaped or logical dependency disappeared before release");
    void* release_arguments[]{&gate};
    check(cuLaunchKernel(releasing, kWorkers, 1, 1, kThreads, 1, 1, 0, independent,
                         release_arguments, nullptr),
          "submit independent releasing original");
    check(pending.get(), "actual dependent SDK result");
    check(cuStreamSynchronize(nullptr), "complete dependent prefix only");
    const auto later = cuStreamQuery(independent);
    if (later != CUDA_SUCCESS && later != CUDA_ERROR_NOT_READY)
        check(later, "observe later stream");
    if (!is_instrumented && later != CUDA_ERROR_NOT_READY)
        throw std::runtime_error("later independent work was included in the dependent frontier");
    check(cuCtxSynchronize(), "complete all actual original work");
    check(cuMemcpyDtoH(output.data(), memory, output.size() * sizeof(output.front())),
          "download all outputs");
    std::uint32_t observed_gate{};
    check(cuMemcpyDtoH(&observed_gate, gate, sizeof(observed_gate)), "observe completed gate");
    for (std::size_t index = 0; index < output.size(); ++index)
        if (output[index] != (index < kElements ? 1U : 2U))
            throw std::runtime_error(
                "producer gate escaped or original output was missing/replayed: index=" +
                std::to_string(index) + "; value=" + std::to_string(output[index]) +
                "; final_gate=" + std::to_string(observed_gate));
    check(cuMemFree(gate), "release gate");
    check(cuMemFree(memory), "release output");
    check(cuStreamDestroy(independent), "destroy release stream");
    check(cuStreamDestroy(blocking), "destroy producer stream");
    check(cuModuleUnload(module), "unload module");
    check(cuCtxDestroy(context), "destroy context");
    std::cout << "Ordinary multiplexed Driver application verified: workers=320; elements="
              << output.size() << "; dependent_submission="
              << (is_instrumented ? "instrumented_not_qualified" : "true")
              << "; full_device_qualified=false; priority_or_timing_qualified=false\n";
}
}  // namespace
int main(int argc, char** argv) {
    try {
        const bool instrumented =
            argc == 2 && std::string_view(argv[1]) == "--device-sanitizer-check";
        if (argc != 1 && !instrumented)
            throw std::runtime_error("unknown ordinary SDK fixture option");
        check(cuInit(0), "initialize Driver");
        run(instrumented);
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
