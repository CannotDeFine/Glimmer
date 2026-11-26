#include "internal/cuda_execution_dispatcher.h"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>

namespace {
using namespace glimmer::backend::cuda;

void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void check(CUresult status, const char* message) {
    if (status != CUDA_SUCCESS)
        throw std::runtime_error(std::string(message) + ": CUDA status " + std::to_string(status));
}

constexpr const char* kSource = R"ptx(
.version 7.0
.target sm_70
.address_size 64
.visible .entry gated_add(.param .u64 output, .param .u64 gate,
                          .param .u32 offset, .param .u32 value) {
 .reg .pred %p<3>;
 .reg .b32 %r<9>;
 .reg .b64 %rd<8>;
 ld.param.u64 %rd1, [output]; ld.param.u64 %rd2, [gate];
 ld.param.u32 %r1, [offset]; ld.param.u32 %r2, [value];
 mov.u64 %rd3, %globaltimer;
WAIT:
 ld.volatile.global.u32 %r3, [%rd2];
 setp.ne.u32 %p1, %r3, 0; @%p1 bra READY;
 mov.u64 %rd4, %globaltimer; sub.u64 %rd5, %rd4, %rd3;
 setp.ge.u64 %p2, %rd5, 2000000000; @%p2 bra EXIT;
 bra WAIT;
READY:
 mov.u32 %r4, %ctaid.x; mov.u32 %r5, %ntid.x; mov.u32 %r6, %tid.x;
 mad.lo.u32 %r7, %r4, %r5, %r6; add.u32 %r7, %r7, %r1;
 mul.wide.u32 %rd6, %r7, 4; add.u64 %rd7, %rd1, %rd6;
 atom.global.add.u32 %r8, [%rd7], %r2;
EXIT: ret;
}
)ptx";

struct Resources {
    CUcontext context{};
    CUmodule module{};
    CUstream stream{};
    CUdeviceptr output{}, device_gate{};
    std::uint32_t* gate{};
    const pid_t process{getpid()};
    const std::thread::id owner{std::this_thread::get_id()};
    std::shared_ptr<bool> cleanup_ok{std::make_shared<bool>(true)};
    ~Resources() {
        if (process != getpid() || owner != std::this_thread::get_id()) {
            *cleanup_ok = false;
            return;
        }
        const auto cleaned = [&](CUresult result) {
            if (result != CUDA_SUCCESS) {
                *cleanup_ok = false;
                std::cerr << "GPU fixture cleanup status " << result << '\n';
            }
        };
        if (gate) std::atomic_ref<std::uint32_t>(*gate).store(1, std::memory_order_release);
        if (context) cleaned(cuCtxSynchronize());
        if (output) cleaned(cuMemFree(output));
        if (gate) cleaned(cuMemFreeHost(gate));
        if (stream) cleaned(cuStreamDestroy(stream));
        if (module) cleaned(cuModuleUnload(module));
        if (context) cleaned(cuCtxDestroy(context));
    }
};
struct Parameters {
    CUdeviceptr output{}, gate{};
    std::uint32_t offset{}, value{};
};
static_assert(sizeof(Parameters) == 24 && offsetof(Parameters, offset) == 16);

void run(const std::shared_ptr<Resources>& resources) {
    CUdevice device{};
    check(cuDeviceGet(&device, 0), "get test device");
    int sms{};
    check(cuDeviceGetAttribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device),
          "read SM count");
    expect(sms > 0, "a real CUDA device is required");
    check(cuCtxCreate(&resources->context, CU_CTX_MAP_HOST, device), "create owned context");
    check(cuModuleLoadData(&resources->module, kSource), "load owned kernel module");
    CUfunction function{};
    check(cuModuleGetFunction(&function, resources->module, "gated_add"), "resolve owned kernel");
    for (std::size_t index = 0; index < 4; ++index) {
        std::size_t offset{}, bytes{};
        check(cuFuncGetParamInfo(function, index, &offset, &bytes),
              "validate native parameter ABI");
        const std::size_t expected_offset = index < 2 ? index * 8 : 16 + (index - 2) * 4;
        expect(offset == expected_offset && bytes == (index < 2 ? 8U : 4U), "exact parameter ABI");
    }
    check(cuStreamCreate(&resources->stream, CU_STREAM_NON_BLOCKING), "create exclusive stream");
    const auto workers = static_cast<unsigned int>(sms) * 4U;
    constexpr unsigned int threads = 32;
    const auto elements = workers * threads;
    const auto total_elements = std::size_t{2} * elements;
    check(cuMemAlloc(&resources->output, total_elements * sizeof(std::uint32_t)),
          "allocate output");
    check(cuMemsetD32(resources->output, 0, total_elements), "initialize output");
    check(cuMemHostAlloc(reinterpret_cast<void**>(&resources->gate), sizeof(std::uint32_t),
                         CU_MEMHOSTALLOC_DEVICEMAP),
          "allocate owned mapped gate");
    check(cuMemHostGetDevicePointer(&resources->device_gate, resources->gate, 0), "map gate");
    std::atomic_ref<std::uint32_t>(*resources->gate).store(0, std::memory_order_release);
    {
        const CudaExecutionDispatch dispatch{cuCtxGetCurrent,    cuLaunchKernel, cuEventCreate,
                                             cuEventRecord,      cuEventQuery,   cuEventDestroy,
                                             cuStreamSynchronize};
        CudaExecutionDispatcher executor({1, 1}, {1, resources->context, resources}, dispatch);
        check(executor.bind_executor(), "bind actual CUDA owner");
        expect(executor.bind_function(1, function, sizeof(Parameters), resources) &&
                   executor.bind_stream(1, resources->stream, resources),
               "bind real owned resources");
        Parameters parameters{resources->output, resources->device_gate, 0, 1};
        CudaExecutionKernel kernel{.function = 1,
                                   .stream = 1,
                                   .grid = {workers, 1, 1},
                                   .block = {threads, 1, 1},
                                   .parameters = std::as_bytes(std::span{&parameters, 1})};
        const auto first = executor.accept_kernel(1, kernel);
        parameters.offset = elements;
        parameters.value = 3;
        const auto second = executor.accept_kernel(2, kernel);
        parameters = {};  // Caller storage is invalidated before either issue.
        expect(first.accepted() && second.accepted(), "own both launch payloads");
        const auto one = executor.issue_next();
        expect(one.state == CudaExecutionIssueState::kIssued, "issue first original kernel");
        check(one.native_status, "first original launch status");
        check(one.completion_status, "first real trailing record");
        expect(executor.poll().retained == 1, "first real event must report NOT_READY");
        const auto two = executor.issue_next();
        expect(two.state == CudaExecutionIssueState::kIssued, "issue second before first stop");
        check(two.native_status, "second original launch status");
        check(two.completion_status, "second real trailing record");
        const auto pending = executor.poll();
        check(pending.status, "query both still-active native submissions");
        expect(pending.retired == 0 && pending.retained == 2 &&
                   executor.acknowledge_result(first.receipt) &&
                   executor.acknowledge_result(second.receipt) &&
                   executor.snapshot().awaiting_physical_stop == 2,
               "both SDK replies must precede physical completion without early retirement");
        std::atomic_ref<std::uint32_t>(*resources->gate).store(1, std::memory_order_release);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{10};
        while (executor.snapshot().outstanding != 0) {
            check(executor.poll().status, "poll actual final physical events");
            expect(std::chrono::steady_clock::now() < deadline, "bounded real retirement");
            std::this_thread::yield();
        }
        expect(executor.shutdown(), "drain exact host and physical ownership");
    }
    std::vector<std::uint32_t> actual(total_elements);
    check(cuMemcpyDtoH(actual.data(), resources->output, actual.size() * sizeof(actual[0])),
          "read complete actual output");
    for (std::size_t index = 0; index < actual.size(); ++index)
        expect(actual[index] == (index < elements ? 1U : 3U),
               "every original executes exactly once");
    std::cout << "Real CUDA pipeline verified: two native issues before either retirement; workers="
              << workers << ", elements=" << total_elements
              << "; full_device_qualified=false; public_integration=false\n";
}
}  // namespace

int main() {
    const auto initialized = cuInit(0);
    if (initialized == CUDA_ERROR_NO_DEVICE) return 77;
    std::shared_ptr<Resources> resources;
    try {
        check(initialized, "initialize CUDA");
        resources = std::make_shared<Resources>();
        const auto cleanup_ok = resources->cleanup_ok;
        run(resources);
        resources.reset();
        expect(*cleanup_ok, "all owned CUDA resources cleanly released");
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        if (resources && resources->gate)
            std::atomic_ref<std::uint32_t>(*resources->gate).store(1, std::memory_order_release);
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
