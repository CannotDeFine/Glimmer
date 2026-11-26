#include "internal/cuda_execution_session.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
using namespace glimmer::backend::cuda;
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void check(CUresult status, const char* message) {
    if (status != CUDA_SUCCESS)
        throw std::runtime_error(std::string(message) + ": CUDA status " + std::to_string(status));
}
void success(const CudaSessionResult& result) {
    expect(result.disposition == CudaSessionDisposition::kComplete,
           "native session operation must complete");
    check(result.status, "native session operation");
}
constexpr const char* kSource = R"ptx(
.version 7.0
.target sm_70
.address_size 64
.visible .entry bounded_add(.param .u64 output, .param .u64 duration,
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
)ptx";

void run() {
    CUdevice device{};
    int sms{};
    check(cuDeviceGet(&device, 0), "select test GPU");
    check(cuDeviceGetAttribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device),
          "read SM count");
    expect(sms > 0, "real GPU SM count");
    const auto workers = static_cast<unsigned int>(sms) * 4U;
    const auto elements = workers * 32U;
    const CudaSessionDispatch dispatch{
        {cuCtxGetCurrent, cuLaunchKernel, cuEventCreate, cuEventRecord, cuEventQuery,
         cuEventDestroy, cuStreamSynchronize},
        cuInit,
        cuDeviceGet,
        cuCtxCreate,
        cuCtxSetCurrent,
        cuCtxDestroy,
        cuModuleLoadData,
        cuModuleUnload,
        cuModuleGetFunction,
        cuFuncGetParamInfo,
        cuMemAlloc,
        cuMemFree,
        cuMemcpyHtoD,
        cuMemcpyDtoH,
        cuStreamCreate,
        cuStreamDestroy};
    CudaExecutionSession session({1, 1}, 1, dispatch);
    success(session.initialize(0));
    std::string source{kSource};
    const auto module = session.load_ptx(source);
    success(module);
    source.clear();
    const auto function = session.get_function(module.id, "bounded_add");
    success(function);
    expect(function.arguments.size() == 4 && function.parameter_bytes >= 24,
           "native ABI must be available");
    const std::vector<CudaArgumentLayout> layout(function.arguments.begin(),
                                                 function.arguments.end());
    std::vector<std::uint32_t> output(std::size_t{2} * elements, 0);
    const auto memory = session.allocate(output.size() * sizeof(output[0]));
    success(memory);
    success(session.upload({memory.id, 0}, std::as_bytes(std::span{output})));
    const auto stream = session.create_stream(CU_STREAM_NON_BLOCKING);
    success(stream);
    std::vector<std::byte> parameters(function.parameter_bytes);
    const auto set = [&](std::size_t index, const auto& value) {
        expect(layout[index].bytes == sizeof(value), "exact native argument size");
        std::memcpy(parameters.data() + layout[index].offset, &value, sizeof(value));
    };
    constexpr std::uint64_t duration_ns = 200000000;
    set(0, memory.address);
    set(1, duration_ns);
    struct OutputOperation {
        std::uint32_t offset{}, value{};
    };
    const auto submit = [&](std::uint64_t sequence, OutputOperation operation) {
        set(2, operation.offset);
        set(3, operation.value);
        return session.accept_kernel(sequence, {.function = function.id,
                                                .stream = stream.id,
                                                .grid = {workers, 1, 1},
                                                .block = {32, 1, 1},
                                                .parameters = parameters});
    };
    const auto first = submit(1, {0, 1}), second = submit(2, {elements, 3});
    std::fill(parameters.begin(), parameters.end(), std::byte{0});
    expect(first.accepted() && second.accepted(), "accept copied server-owned launches");
    const auto one = session.issue_next();
    check(one.native_status, "first original native status");
    check(one.completion_status, "first real trailing event");
    expect(one.state == CudaExecutionIssueState::kIssued && session.poll().retained == 1,
           "first native still pending");
    const auto two = session.issue_next();
    check(two.native_status, "second original native status");
    check(two.completion_status, "second real trailing event");
    const auto pending = session.poll();
    check(pending.status, "pending actual native events");
    expect(two.state == CudaExecutionIssueState::kIssued && pending.retired == 0 &&
               pending.retained == 2,
           "second native launch must precede first retirement");
    const auto early_free = session.release(memory.id);
    expect(
        early_free.disposition == CudaSessionDisposition::kDeferred && !early_free.original_status,
        "in-flight storage release must defer without pretending to call native free");
    expect(session.acknowledge_result(first.receipt) && session.acknowledge_result(second.receipt),
           "ACK original SDK results");
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{10};
    while (session.snapshot().submissions.outstanding) {
        check(session.poll().status, "actual session retirement");
        expect(std::chrono::steady_clock::now() < deadline, "bounded session completion");
        std::this_thread::yield();
    }
    success(session.download({memory.id, 0}, std::as_writable_bytes(std::span{output})));
    for (std::size_t index = 0; index < output.size(); ++index)
        expect(output[index] == (index < elements ? 1U : 3U),
               "all original outputs execute exactly once");
    success(session.release(memory.id));
    success(session.destroy_stream(stream.id));
    success(session.unload_module(module.id));
    const auto replacement = session.allocate(16);
    success(replacement);
    expect(replacement.id != memory.id && !session.release(memory.id).original_status,
           "stale memory ID cannot release a new allocation");
    success(session.shutdown());
    expect(!session.snapshot().has_context && session.snapshot().device_bytes == 0,
           "owned context shutdown retires remaining allocation");
    CUcontext current{};
    check(cuCtxGetCurrent(&current), "observe actual context destruction");
    expect(!current, "factory must not leave its native context current after destruction");
    std::cout << "Server-owned CUDA session verified: workers=" << workers
              << ", elements=" << output.size()
              << "; copied_pipeline=true; resource_lifecycle=true; public_integration=false; "
                 "full_device_qualified=false\n";
}
}  // namespace

int main() {
    const auto initialized = cuInit(0);
    if (initialized == CUDA_ERROR_NO_DEVICE) return 77;
    try {
        check(initialized, "initialize test GPU");
        run();
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
