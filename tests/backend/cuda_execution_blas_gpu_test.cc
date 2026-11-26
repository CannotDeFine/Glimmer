#include "internal/cuda_execution_session.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {
using namespace glimmer::backend::cuda;
void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void check(CUresult status) {
    if (status != CUDA_SUCCESS) throw std::runtime_error("Driver status " + std::to_string(status));
}
void success(const CudaSessionResult& result) {
    check(result.status);
    expect(result.disposition == CudaSessionDisposition::kComplete, "completed native resource");
}
void success(const CudaSessionBlasResult& result) {
    check(result.status);
    expect(result.disposition == CudaSessionDisposition::kComplete &&
               result.original_status == CUBLAS_STATUS_SUCCESS,
           "completed original cuBLAS operation");
}
constexpr std::string_view kDelay = R"ptx(
.version 7.0
.target sm_70
.address_size 64
.visible .entry delay(.param .u64 duration) {
 .reg .b64 %r<5>; .reg .pred %p;
 ld.param.u64 %r1, [duration]; mov.u64 %r2, %globaltimer;
wait:
 mov.u64 %r3, %globaltimer; sub.u64 %r4, %r3, %r2;
 setp.lt.u64 %p, %r4, %r1; @%p bra wait;
 ret;
}
)ptx";
void wait_for_stop(CudaExecutionSession& session) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{20};
    while (session.snapshot().submissions.outstanding) {
        check(session.poll().status);
        expect(std::chrono::steady_clock::now() < deadline, "bounded actual event retirement");
        std::this_thread::yield();
    }
}
void run(bool is_device_tool) {
    auto dispatch =
        CudaSessionDispatch{{cuCtxGetCurrent, cuLaunchKernel, cuEventCreate, cuEventRecord,
                             cuEventQuery, cuEventDestroy, cuStreamSynchronize},
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
    dispatch.blas = {cublasCreate_v2, cublasDestroy_v2, cublasSetStream_v2, cublasSgemm_v2};
    CudaExecutionSession session({1, 1}, 1, dispatch);
    success(session.initialize(0));
    CUdevice device{};
    int sms{};
    check(cuDeviceGet(&device, 0));
    check(cuDeviceGetAttribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));
    expect(sms == 80, "current V100 has 80 available SMs; availability is not participation");
    const auto module = session.load_ptx(kDelay);
    success(module);
    const auto delay = session.get_function(module.id, "delay");
    success(delay);
    expect(delay.arguments.size() == 1 && delay.arguments[0].bytes == 8 &&
               delay.parameter_bytes == 8 && delay.arguments[0].offset == 0,
           "independently validated original delay ABI");
    const auto handle = session.create_blas();
    success(handle);
    const auto stream = session.create_stream(CU_STREAM_NON_BLOCKING);
    success(stream);
    success(session.set_blas_stream(handle.id, stream.id));
    constexpr int rows = 97, columns = 113, depth = 71, leading = 128;
    constexpr std::size_t elements = leading * leading + 32;
    const auto a = session.allocate(elements * sizeof(float));
    const auto b = session.allocate(elements * sizeof(float));
    const auto c = session.allocate(elements * sizeof(float));
    success(a);
    success(b);
    success(c);
    std::vector<float> input_a(elements), input_b(elements), output(elements);
    for (std::size_t index = 0; index < elements; ++index) {
        input_a[index] = static_cast<float>(static_cast<int>(index % 5) - 2);
        input_b[index] = static_cast<float>(static_cast<int>(index % 7) - 3);
    }
    success(session.upload({a.id, 0}, std::as_bytes(std::span{input_a})));
    success(session.upload({b.id, 0}, std::as_bytes(std::span{input_b})));
    std::uint64_t sequence{1};
    unsigned int pipelines{};
    std::size_t checked{};
    for (const auto ta : {CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C}) {
        for (const auto tb : {CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C}) {
            CudaSessionSgemm problem{.handle = handle.id,
                                     .transpose_a = ta,
                                     .transpose_b = tb,
                                     .rows = rows,
                                     .columns = columns,
                                     .depth = depth,
                                     .leading_a = leading,
                                     .leading_b = leading,
                                     .leading_c = leading,
                                     .a = {a.id, 0},
                                     .b = {b.id, 0},
                                     .c = {c.id, 0},
                                     .alpha = 2.0F,
                                     .beta = 0.5F};
            std::fill(output.begin(), output.end(), 6.0F);
            success(session.upload({c.id, 0}, std::as_bytes(std::span{output})));
            // Warm this actual vendor path before observing its ordinary host
            // issue pipeline. Warmup is explicit fixture work, not app replay.
            auto warm = session.accept_sgemm(sequence++, problem);
            expect(warm.accepted() &&
                       session.issue_next().library_status == CUBLAS_STATUS_SUCCESS &&
                       session.acknowledge_result(warm.receipt),
                   "warm original native SGEMM");
            wait_for_stop(session);
            success(session.upload({c.id, 0}, std::as_bytes(std::span{output})));
            const std::uint64_t duration_ns = 150'000'000;
            const auto gate = session.accept_kernel(
                sequence++, {.function = delay.id,
                             .stream = stream.id,
                             .grid = {1, 1, 1},
                             .block = {32, 1, 1},
                             .parameters = std::as_bytes(std::span{&duration_ns, 1})});
            const auto first = session.accept_sgemm(sequence++, problem);
            const auto second = session.accept_sgemm(sequence++, problem);
            expect(gate.accepted() && first.accepted() && second.accepted(),
                   "three owned commands");
            const auto frontier = session.capture_frontier(stream.id);
            check(frontier.status);
            problem.alpha = 99.0F;
            problem.beta = 99.0F;
            expect(session.issue_next().state == CudaExecutionIssueState::kIssued &&
                       session.issue_next().library_status == CUBLAS_STATUS_SUCCESS &&
                       session.issue_next().library_status == CUBLAS_STATUS_SUCCESS,
                   "two original library calls issue before any owner poll");
            const auto pending = session.poll();
            check(pending.status);
            if (pending.retained == 3) ++pipelines;
            expect(!session.native_kernel_pair(),
                   "library calls are not a two-original-kernel witness");
            for (const auto& accepted : {gate, first, second}) {
                const auto reply = session.host_result(accepted.receipt);
                expect(reply && reply->status == 0 && session.acknowledge_result(accepted.receipt),
                       "actual SDK results are acknowledged separately from stop");
            }
            wait_for_stop(session);
            check(session.query_frontier(frontier.frontier));
            success(session.download({c.id, 0}, std::as_writable_bytes(std::span{output})));
            for (std::size_t index = 0; index < output.size(); ++index) {
                const int row = static_cast<int>(index % leading);
                const int column = static_cast<int>(index / leading);
                float expected{6.0F};
                if (row < rows && column < columns) {
                    float sum{};
                    for (int reduction = 0; reduction < depth; ++reduction) {
                        const auto ai = ta == CUBLAS_OP_N ? row + reduction * leading
                                                          : reduction + row * leading;
                        const auto bi = tb == CUBLAS_OP_N ? reduction + column * leading
                                                          : column + reduction * leading;
                        sum += input_a[static_cast<std::size_t>(ai)] *
                               input_b[static_cast<std::size_t>(bi)];
                    }
                    expected = 3.0F * sum + 1.5F;
                }
                expect(output[index] == expected,
                       "every SGEMM output and padding guard matches CPU oracle");
                ++checked;
            }
        }
    }
    expect(is_device_tool || pipelines == 9,
           "ordinary warmed library calls pipeline ahead of actual stop");
    success(session.destroy_blas(handle.id));
    success(session.destroy_stream(stream.id));
    success(session.shutdown());
    std::cout << "Owned cuBLAS verified: values=" << checked
              << "; transpose_pairs=9; pipelines=" << pipelines
              << "; available_sms=80; public_integration=false; full_device_qualified=false; "
              << "performance_qualified=false\n";
}
}  // namespace
int main(int argc, char** argv) {
    try {
        expect(argc == 1 || (argc == 2 && std::string_view(argv[1]) == "--device-tool"),
               "known fixture options");
        check(cuInit(0));
        run(argc == 2);
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
