#include "internal/cuda_execution_session.h"
#include "internal/cuda_execution_domain.h"
#include "internal/cuda_gemm_signal_layout.h"
#include "internal/cuda_lt_registry.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

namespace {
using glimmer::backend::ExecutionAcceptance;
using namespace glimmer::backend::cuda;
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void check(CUresult status) {
    expect(status == CUDA_SUCCESS, ("Driver status " + std::to_string(status)).c_str());
}
void success(const CudaSessionResult& result) {
    check(result.status);
    expect(result.disposition == CudaSessionDisposition::kComplete, "completed owned resource");
}
struct LaunchTrace {
    CUfunction function{};
    CUstream stream{};
    std::array<std::byte, 176> arguments{};
    std::size_t bytes{};
    unsigned int launches{};
    CUdeviceptr counter{}, retreat{}, completed{};
    CUdeviceptr initialized_counter{};
    unsigned int initializations{};
};
LaunchTrace trace;
bool observes_launches{}, replaces_commands{}, uses_sharded_host_retreat{};
struct Mapping {
    std::byte* host{};
    std::size_t bytes{};
    CUdeviceptr device{};
} mapping;
struct SharedShardTrace {
    std::array<Mapping, 2> mappings{};
    std::array<CUdeviceptr, 2> signals{};
    std::size_t count{}, active{};
};
SharedShardTrace* shared_shard_trace{};
CUresult CUDAAPI shared_shard_host_allocate(void** output, std::size_t bytes, unsigned int flags) {
    const auto result = cuMemHostAlloc(output, bytes, flags);
    if (result == CUDA_SUCCESS) {
        expect(shared_shard_trace && shared_shard_trace->count < 2,
               "exactly two shared shard mappings");
        shared_shard_trace->mappings[shared_shard_trace->count++] = {
            static_cast<std::byte*>(*output), bytes, 0};
    }
    return result;
}
CUresult CUDAAPI shared_shard_host_map(CUdeviceptr* output, void* host, unsigned int flags) {
    const auto result = cuMemHostGetDevicePointer(output, host, flags);
    if (result == CUDA_SUCCESS) {
        const auto found = std::ranges::find(shared_shard_trace->mappings, host, &Mapping::host);
        expect(found != shared_shard_trace->mappings.end(), "map the exact retained shard arena");
        found->device = *output;
    }
    return result;
}
// NOLINTBEGIN(bugprone-easily-swappable-parameters): exact original Driver ABI.
CUresult CUDAAPI shared_shard_launch(CUfunction function, unsigned int x, unsigned int y,
                                     unsigned int z, unsigned int bx, unsigned int by,
                                     unsigned int bz, unsigned int shared, CUstream stream,
                                     void** parameters, void** extra) {
    expect(shared_shard_trace && shared_shard_trace->active < 2 && extra,
           "capture the active shared member's actual launch arguments");
    auto& trace = *shared_shard_trace;
    CUdeviceptr signal{};
    std::memcpy(&signal, static_cast<const std::byte*>(extra[1]) + 88, sizeof(signal));
    const auto& arena = trace.mappings[trace.active];
    expect(signal >= arena.device + 128 &&
               signal - arena.device <= arena.bytes - kGemmSignalSlotBytes &&
               (signal - arena.device - 128) % kGemmSignalSlotBytes == 0,
           "actual member kernel receives its own exact sharded signal array");
    trace.signals[trace.active] = signal;
    return cuLaunchKernel(function, x, y, z, bx, by, bz, shared, stream, parameters, extra);
}
// NOLINTEND(bugprone-easily-swappable-parameters)
CUresult CUDAAPI host_allocate(void** output, std::size_t bytes, unsigned int flags) {
    const auto status = cuMemHostAlloc(output, bytes, flags);
    if (status == CUDA_SUCCESS) mapping = {static_cast<std::byte*>(*output), bytes, 0};
    return status;
}
CUresult CUDAAPI host_map(CUdeviceptr* output, void* host, unsigned int flags) {
    const auto status = cuMemHostGetDevicePointer(output, host, flags);
    if (status == CUDA_SUCCESS) {
        expect(host == mapping.host, "exact owned mapped completion arena");
        mapping.device = *output;
    }
    return status;
}
CUresult CUDAAPI upload_async(CUdeviceptr destination, const void* source, std::size_t bytes,
                              CUstream stream) {
    const auto status = cuMemcpyHtoDAsync(destination, source, bytes, stream);
    if (status == CUDA_SUCCESS && bytes == (observes_launches ? 16U : 8U)) {
        std::uint64_t initial{};
        std::memcpy(&initial, source, sizeof(initial));
        expect(!initial, "actual progress initialization uploads zero");
        if (observes_launches) {
            std::memcpy(&initial, static_cast<const std::byte*>(source) + 8, sizeof(initial));
            expect(!initial, "actual private device election also starts at zero");
        }
        trace.initialized_counter = destination;
        ++trace.initializations;
    }
    return status;
}
// NOLINTBEGIN(bugprone-easily-swappable-parameters): exact original Driver ABI.
CUresult CUDAAPI launch(CUfunction function, unsigned int x, unsigned int y, unsigned int z,
                        unsigned int bx, unsigned int by, unsigned int bz, unsigned int shared,
                        CUstream stream, void** parameters, void** extra) {
    expect(x == 320 && y == 1 && z == 1 && bx == 128 && by == 1 && bz == 1 && !shared &&
               !parameters && extra,
           "actual GEMM launch has explicit full-device worker geometry");
    const auto bytes = *static_cast<const std::size_t*>(extra[3]);
    const auto ordinary = bytes - (observes_launches ? 8 : 0);
    expect((ordinary == 96 || ordinary == 104 || ordinary == 128 || ordinary == 168) &&
               bytes <= trace.arguments.size(),
           "actual private provider packing and pointer width");
    if (!trace.launches) {
        trace.function = function;
        trace.stream = stream;
        trace.bytes = bytes;
        std::memcpy(trace.arguments.data(), extra[1], bytes);
        std::memcpy(&trace.counter, trace.arguments.data() + (ordinary == 168 ? 104 : 80), 8);
        std::memcpy(&trace.retreat, trace.arguments.data() + (ordinary == 168 ? 112 : 88), 8);
        if (uses_sharded_host_retreat)
            expect(trace.retreat >= mapping.device &&
                       trace.retreat - mapping.device <= mapping.bytes - kGemmSignalSlotBytes,
                   "actual GEMM shard array belongs to its mapped progress allocation");
        if (observes_launches) {
            std::memcpy(&trace.completed, trace.arguments.data() + ordinary, 8);
            if (!uses_sharded_host_retreat)
                expect(trace.completed == trace.retreat + 4,
                       "actual observed provider uses its distinct owned mapped word");
        }
    } else
        expect(
            trace.function == function && trace.stream == stream && trace.bytes == bytes &&
                std::memcmp(trace.arguments.data(), extra[1], bytes) == 0,
            "every actual continuation retains function, stream, counter, witness and arguments");
    expect(trace.initializations == 1 && trace.initialized_counter == trace.counter,
           "actual counter is initialized once, never replayed on a physical continuation");
    if (uses_sharded_host_retreat) {
        const auto offset = trace.retreat - mapping.device;
        for (std::size_t index = 0; index < kGemmSignalShards; ++index) {
            auto& word = *reinterpret_cast<std::uint32_t*>(mapping.host + offset +
                                                           index * kGemmSignalShardBytes);
            expect(std::atomic_ref(word).load(std::memory_order_acquire) == 0,
                   "actual initial and resumed launches observe all shard resets");
        }
    }
    if (observes_launches) {
        expect(trace.completed >= mapping.device &&
                   trace.completed - mapping.device <= mapping.bytes - sizeof(std::uint32_t),
               "native witness belongs to the exact retained host/device mapping");
        const auto offset = trace.completed - mapping.device;
        auto& word = *reinterpret_cast<std::uint32_t*>(mapping.host + offset);
        expect(std::atomic_ref(word).load(std::memory_order_acquire) == (trace.launches ? 1U : 0U),
               "fresh native command clears stale publication; retained continuation preserves it");
    }
    const auto status =
        cuLaunchKernel(function, x, y, z, bx, by, bz, shared, stream, parameters, extra);
    if (status == CUDA_SUCCESS) ++trace.launches;
    return status;
}
// NOLINTEND(bugprone-easily-swappable-parameters)
CudaSessionDispatch dispatch() {
    CudaSessionDispatch sdk{{cuCtxGetCurrent, cuLaunchKernel, cuEventCreate, cuEventRecord,
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
                            cuStreamDestroy,
                            cuMemHostAlloc,
                            cuMemFreeHost,
                            cuMemHostGetDevicePointer,
                            cuMemcpyHtoDAsync,
                            cuMemcpyDtoHAsync,
                            cuOccupancyMaxActiveBlocksPerMultiprocessor};
    sdk.function_set_attribute = cuFuncSetAttribute;
    sdk.execution.launch_kernel = launch;
    sdk.host_allocate = host_allocate;
    sdk.host_map = host_map;
    sdk.upload_async = upload_async;
    sdk.device_attribute = cuDeviceGetAttribute;
    sdk.stream_wait_event = cuStreamWaitEvent;
    sdk.blas.create = cublasCreate_v2;
    sdk.blas.destroy = cublasDestroy_v2;
    sdk.blas.set_stream = cublasSetStream_v2;
    sdk.blas.sgemm = cublasSgemm_v2;
    sdk.blas.gemm_ex = cublasGemmEx;
    sdk.blas.strided_gemm_ex = cublasGemmStridedBatchedEx;
    sdk.blas.get_version = cublasGetVersion_v2;
    sdk.lt.create = cublasLtCreate;
    sdk.lt.destroy = cublasLtDestroy;
    sdk.lt.layout_create = cublasLtMatrixLayoutCreate;
    sdk.lt.layout_destroy = cublasLtMatrixLayoutDestroy;
    sdk.lt.layout_set = cublasLtMatrixLayoutSetAttribute;
    sdk.lt.layout_get = cublasLtMatrixLayoutGetAttribute;
    sdk.lt.operation_create = cublasLtMatmulDescCreate;
    sdk.lt.operation_destroy = cublasLtMatmulDescDestroy;
    sdk.lt.operation_set = cublasLtMatmulDescSetAttribute;
    sdk.lt.operation_get = cublasLtMatmulDescGetAttribute;
    sdk.lt.preference_create = cublasLtMatmulPreferenceCreate;
    sdk.lt.preference_destroy = cublasLtMatmulPreferenceDestroy;
    sdk.lt.preference_set = cublasLtMatmulPreferenceSetAttribute;
    sdk.lt.preference_get = cublasLtMatmulPreferenceGetAttribute;
    sdk.lt.version = cublasLtGetVersion;
    sdk.lt.matmul = cublasLtMatmul;
    return sdk;
}
template <typename T>
T value(float input) {
    if constexpr (std::is_same_v<T, float>)
        return input;
    else
        return __float2half_rn(input);
}
template <typename T>
float as_float(T input) {
    if constexpr (std::is_same_v<T, float>)
        return input;
    else
        return __half2float(input);
}
template <typename T>
struct Buffer {
    CudaExecutionSession& session;
    CudaSessionResult allocation;
    std::vector<T> input;
    static constexpr std::size_t k_guard = 32;
    Buffer(std::size_t elements, CudaExecutionSession& owner, float fill)
        : session(owner),
          allocation(owner.allocate((elements + 2 * k_guard) * sizeof(T))),
          input(elements + 2 * k_guard, value<T>(-8192.0F)) {
        success(allocation);
        std::fill(input.begin() + k_guard, input.end() - k_guard, value<T>(fill));
        success(session.upload({allocation.id, 0}, std::as_bytes(std::span{input})));
    }
    CudaAllocationRange range() const {
        return {allocation.id, k_guard * sizeof(T)};
    }
    template <class Pattern>
    void pattern(Pattern element) {
        for (std::size_t index = 0; index < input.size() - 2 * k_guard; ++index)
            input[k_guard + index] = value<T>(element(index));
        success(session.upload({allocation.id, 0}, std::as_bytes(std::span{input})));
    }
    void unchanged() const {
        const auto actual = download();
        expect(std::memcmp(actual.data(), input.data(), input.size() * sizeof(T)) == 0,
               "complete original input and head/tail guards remain byte-identical");
    }
    std::vector<T> download() const {
        std::vector<T> actual(input.size());
        success(session.download({allocation.id, 0}, std::as_writable_bytes(std::span{actual})));
        check(session.poll().status);
        return actual;
    }
    void release() {
        success(session.release(allocation.id));
    }
};
std::uint64_t checked_values{}, partials{}, observed_commands{}, commands{};
void execute(CudaExecutionSession& session, const ExecutionAcceptance& accepted,
             std::uint64_t sequence, bool observes, bool requires_partial) {
    trace = {};
    expect(accepted.accepted(), "accept actual owned library operation");
    const auto issued = session.issue_next();
    expect(issued.state == CudaExecutionIssueState::kIssued &&
               issued.native_status == CUDA_SUCCESS &&
               (!issued.library_status || *issued.library_status == CUBLAS_STATUS_SUCCESS) &&
               session.acknowledge_result(accepted.receipt),
           "issue one original logical command");
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{40};
    if (observes && requires_partial) {
        std::optional<CudaSessionPersistentProgress> progress;
        do {
            progress = session.persistent_progress(sequence);
            expect(progress && progress->observes_block_completion &&
                       progress->physical_launches == 1 && !progress->physical_stops &&
                       !progress->completed_blocks,
                   "mapped completion is independent of stop/readback");
            expect(std::chrono::steady_clock::now() < deadline, "bounded real tile completion");
        } while (!progress->has_completed_block);
        ++observed_commands;
        check(session.request_retreat(sequence));
        if (uses_sharded_host_retreat) {
            const auto offset = trace.retreat - mapping.device;
            for (std::size_t index = 0; index < kGemmSignalShards; ++index) {
                auto& word = *reinterpret_cast<std::uint32_t*>(mapping.host + offset +
                                                               index * kGemmSignalShardBytes);
                expect(std::atomic_ref(word).load(std::memory_order_acquire) == 1,
                       "actual request publishes all host signal shards");
            }
        }
        do {
            check(session.poll().status);
            progress = session.persistent_progress(sequence);
            expect(std::chrono::steady_clock::now() < deadline, "bounded real incomplete retreat");
        } while (progress && !progress->physical_stops);
        expect(progress && progress->physical_stops == 1 && progress->completed_blocks > 0 &&
                   progress->completed_blocks < progress->total_blocks &&
                   progress->has_completed_block,
               "actual library work stops incomplete after independently witnessed useful compute");
        std::cout
            << "GEMM_BLOCK_COMPLETION {\"sequence\":" << sequence
            << ",\"completed_blocks\":" << progress->completed_blocks
            << ",\"total_blocks\":" << progress->total_blocks
            << ",\"physical_launches\":1,\"physical_stops\":1,\"pre_stop_publication\":true}\n";
        ++partials;
        const auto resumed = session.resume(sequence);
        expect(resumed.state == CudaExecutionIssueState::kIssued &&
                   resumed.native_status == CUDA_SUCCESS &&
                   session.persistent_progress(sequence)->has_completed_block,
               "same retained logical command preserves witness through physical continuation");
    }
    expect(trace.launches == (replaces_commands ? observes && requires_partial ? 2U : 1U : 0U),
           "actual provider launch count matches exact original logical work and continuation");
    while (session.snapshot().submissions.outstanding) {
        check(session.poll().status);
        expect(std::chrono::steady_clock::now() < deadline, "bounded actual trailing event");
        std::this_thread::yield();
    }
    ++commands;
}
template <typename Element, typename Result>
void gemm(CudaExecutionSession& session, std::uint64_t handle, cublasOperation_t ta,
          cublasOperation_t tb, bool batched, std::uint64_t sequence, bool observes,
          bool is_large = false, bool is_split = false) {
    const int rows = is_split ? 768 : is_large ? 2048 : 65;
    const int columns = is_split ? 768 : is_large ? 2048 : 97;
    const int depth = is_split ? 3072 : is_large ? 8192 : 64;
    const int batches = batched ? 3 : 1;
    const int a_rows = ta == CUBLAS_OP_N ? rows : depth;
    const int a_cols = ta == CUBLAS_OP_N ? depth : rows;
    const int b_rows = tb == CUBLAS_OP_N ? depth : columns;
    const int b_cols = tb == CUBLAS_OP_N ? columns : depth;
    const int lda = a_rows, ldb = b_rows, ldc = rows;
    const auto size_a = static_cast<std::size_t>(lda) * a_cols;
    const auto size_b = static_cast<std::size_t>(ldb) * b_cols;
    const auto size_c = static_cast<std::size_t>(ldc) * columns;
    Buffer<Element> a(size_a * batches, session, 2.0F), b(size_b * batches, session, 4.0F);
    Buffer<Result> c(size_c * batches, session, 6.0F);
    a.pattern([&](std::size_t index) {
        const auto batch = index / size_a;
        const auto local = index % size_a;
        const auto row = ta == CUBLAS_OP_N ? local % lda : local / lda;
        return static_cast<float>((row % 2 + 1) * (batch + 1));
    });
    b.pattern([&](std::size_t index) {
        const auto local = index % size_b;
        const auto column = tb == CUBLAS_OP_N ? local / ldb : local % ldb;
        return static_cast<float>(column % 3 + 1);
    });
    CudaSessionStridedGemmEx problem;
    auto& typed = problem.operation;
    typed.operation = {.handle = handle,
                       .transpose_a = ta,
                       .transpose_b = tb,
                       .rows = rows,
                       .columns = columns,
                       .depth = depth,
                       .leading_a = lda,
                       .leading_b = ldb,
                       .leading_c = ldc,
                       .a = a.range(),
                       .b = b.range(),
                       .c = c.range(),
                       .alpha = is_split ? 1.0F : 0.125F,
                       .beta = is_split ? 0.0F : 0.5F};
    typed.a_type = typed.b_type = std::is_same_v<Element, float> ? CUDA_R_32F : CUDA_R_16F;
    typed.c_type = std::is_same_v<Result, float> ? CUDA_R_32F : CUDA_R_16F;
    std::memcpy(typed.alpha.data(), &typed.operation.alpha, 4);
    std::memcpy(typed.beta.data(), &typed.operation.beta, 4);
    problem.strides = {static_cast<long long>(size_a), static_cast<long long>(size_b),
                       static_cast<long long>(size_c)};
    problem.batches = batches;
    const auto accepted = batched ? session.accept_strided_gemm_ex(sequence, problem)
                                  : session.accept_gemm_ex(sequence, typed);
    execute(session, accepted, sequence, observes, is_large);
    const auto actual = c.download();
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const bool guard =
            index < Buffer<Result>::k_guard || index >= actual.size() - Buffer<Result>::k_guard;
        const auto local = guard ? 0 : index - Buffer<Result>::k_guard;
        const auto row = local % size_c % static_cast<std::size_t>(ldc);
        const auto column = local % size_c / static_cast<std::size_t>(ldc);
        const auto batch = local / size_c;
        const auto sum = static_cast<float>((row % 2 + 1) * (batch + 1) * (column % 3 + 1)) *
                         static_cast<float>(depth);
        const float expected = is_split ? sum : 0.125F * sum + 3.0F;
        expect(as_float(actual[index]) == (guard ? -8192.0F : expected),
               "every native/replacement GEMM output, beta and guard agrees with the exact oracle");
        ++checked_values;
    }
    a.unchanged();
    b.unchanged();
    c.release();
    b.release();
    a.release();
}
struct LibraryCommand {
    std::uint64_t handle, sequence;
};
template <typename Element, typename Result>
void lt(CudaExecutionSession& session, CudaLtRegistry& registry, LibraryCommand command,
        bool observes, bool is_large = false) {
    const int rows = is_large ? 2048 : 65, columns = is_large ? 2048 : 97;
    const int depth = is_large ? 8192 : 64;
    Buffer<Element> a(static_cast<std::size_t>(rows) * depth, session, 2.0F);
    Buffer<Element> b(static_cast<std::size_t>(depth) * columns, session, 4.0F);
    Buffer<Result> output(static_cast<std::size_t>(rows) * columns, session, 6.0F);
    a.pattern([&](std::size_t index) {
        return static_cast<float>(index % static_cast<std::size_t>(rows) % 2 + 1);
    });
    b.pattern([&](std::size_t index) {
        return static_cast<float>(index / static_cast<std::size_t>(depth) % 3 + 1);
    });
    CudaSessionLtMatmul problem;
    const auto operation = registry.create({.kind = CudaLtObjectKind::kOperation});
    expect(operation.status == CUDA_SUCCESS, "create actual Lt operation");
    problem.descriptors.handle = command.handle;
    problem.descriptors.operation = *registry.borrow(operation.id, CudaLtObjectKind::kOperation);
    std::array<std::uint64_t, 4> layouts{};
    const auto input_type = std::is_same_v<Element, float> ? CUDA_R_32F : CUDA_R_16F;
    const auto output_type = std::is_same_v<Result, float> ? CUDA_R_32F : CUDA_R_16F;
    for (std::size_t index = 0; index < layouts.size(); ++index) {
        const auto count_rows = index == 1 ? depth : rows;
        const auto count_cols = index == 0 ? depth : columns;
        const auto created = registry.create({.kind = CudaLtObjectKind::kLayout,
                                              .type = index < 2 ? input_type : output_type,
                                              .rows = static_cast<std::uint64_t>(count_rows),
                                              .columns = static_cast<std::uint64_t>(count_cols),
                                              .leading = count_rows});
        expect(created.status == CUDA_SUCCESS, "create original typed Lt layout");
        layouts[index] = created.id;
        problem.descriptors.matrices[index] =
            *registry.borrow(created.id, CudaLtObjectKind::kLayout);
    }
    problem.matrices = {a.range(), b.range(), output.range(), output.range()};
    problem.scalar_bytes = 4;
    const float alpha = 0.125F, beta = 0.5F;
    std::memcpy(problem.alpha.data(), &alpha, 4);
    std::memcpy(problem.beta.data(), &beta, 4);
    execute(session, session.accept_lt_matmul(command.sequence, problem), command.sequence,
            observes, is_large);
    const auto actual = output.download();
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const bool guard =
            index < Buffer<Result>::k_guard || index >= actual.size() - Buffer<Result>::k_guard;
        const auto local = guard ? 0 : index - Buffer<Result>::k_guard;
        const auto row = local % static_cast<std::size_t>(rows);
        const auto column = local / static_cast<std::size_t>(rows);
        const auto sum =
            static_cast<float>((row % 2 + 1) * (column % 3 + 1)) * static_cast<float>(depth);
        expect(as_float(actual[index]) == (guard ? -8192.0F : 0.125F * sum + 3.0F),
               "every original/replacement Lt output preserves beta exactly once and all guards");
        ++checked_values;
    }
    a.unchanged();
    b.unchanged();
    problem = {};
    for (const auto id : layouts)
        expect(registry.destroy(id).status == CUDA_SUCCESS, "retire Lt layout");
    expect(registry.destroy(operation.id).status == CUDA_SUCCESS, "retire Lt operation");
    output.release();
    b.release();
    a.release();
}
void run(bool replaces, bool observes, unsigned int tile_side = 64,
         bool sharded_host_retreat = false) {
    replaces_commands = replaces;
    observes_launches = observes;
    uses_sharded_host_retreat = sharded_host_retreat;
    const auto sdk = dispatch();
    CudaSessionPersistentConfig settings;
    settings.workers = replaces ? 320 : 0;
    settings.replace_sgemm = replaces;
    settings.gemm_tile_side = replaces ? tile_side : 64;
    settings.observes_block_completion = observes;
    settings.uses_sharded_host_retreat = sharded_host_retreat;
    CudaSessionLimits limits;
    limits.max_device_bytes = std::size_t{512} * 1024 * 1024;
    CudaExecutionSession session({1, 99}, 101, sdk, limits, settings);
    success(session.initialize(0));
    const auto blas = session.create_blas(), lt_handle = session.create_lt();
    expect(blas.status == CUDA_SUCCESS && lt_handle.status == CUDA_SUCCESS,
           "create actual installed SDK handles and validate all provider parameter layouts");
    CudaLtRegistry registry(sdk.lt);
    std::uint64_t sequence{1};
    for (const bool batch : {false, true})
        for (const auto ta : {CUBLAS_OP_N, CUBLAS_OP_T})
            for (const auto tb : {CUBLAS_OP_N, CUBLAS_OP_T}) {
                gemm<float, float>(session, blas.id, ta, tb, batch, sequence++, observes);
                gemm<__half, __half>(session, blas.id, ta, tb, batch, sequence++, observes);
                gemm<__half, float>(session, blas.id, ta, tb, batch, sequence++, observes);
            }
    gemm<float, float>(session, blas.id, CUBLAS_OP_N, CUBLAS_OP_N, false, sequence++, observes,
                       false, true);
    lt<float, float>(session, registry, {lt_handle.id, sequence++}, observes);
    lt<__half, __half>(session, registry, {lt_handle.id, sequence++}, observes);
    lt<__half, float>(session, registry, {lt_handle.id, sequence++}, observes);
    gemm<float, float>(session, blas.id, CUBLAS_OP_N, CUBLAS_OP_N, false, sequence++, observes,
                       true);
    lt<float, float>(session, registry, {lt_handle.id, sequence++}, observes, true);
    expect(commands == 30 && partials == (observes ? 2U : 0U) && observed_commands == partials,
           "complete provider matrix and both real independent large-work witnesses");
    expect(
        session.snapshot().persistent_accepted == (replaces ? 30U : 0U),
        "all thirty operations actually select replacement or original, never a silent fallback");
    registry.close();
    expect(registry.shutdown().status == CUDA_SUCCESS, "normal Lt metadata cleanup");
    expect(session.destroy_blas(blas.id).status == CUDA_SUCCESS &&
               session.destroy_lt(lt_handle.id).status == CUDA_SUCCESS,
           "normal SDK handle cleanup");
    success(session.shutdown());
    std::cout << "{\"status\":\"passed\",\"replaces\":" << (replaces ? "true" : "false")
              << ",\"observes\":" << (observes ? "true" : "false") << ",\"commands\":" << commands
              << ",\"checked_values\":" << checked_values << ",\"partial_retreats\":" << partials
              << ",\"workers\":" << settings.workers;
    if (replaces && tile_side != 64) std::cout << ",\"gemm_tile_side\":" << tile_side;
    if (sharded_host_retreat) std::cout << ",\"sharded_host_retreat\":true";
    std::cout << ",\"full_device_qualified\":false,\"application_priority_qualified\":false,"
                 "\"performance_qualified\":false,\"full_goal_complete\":false}\n";
}

void run_shared_sharded() {
    constexpr int side = 3584, depth = 2048;
    constexpr std::size_t elements = std::size_t{side} * side;
    constexpr std::size_t input_elements = std::size_t{side} * depth;
    constexpr std::size_t guards = 32;
    SharedShardTrace shared_trace;
    shared_shard_trace = &shared_trace;
    struct TraceScope {
        ~TraceScope() {
            shared_shard_trace = nullptr;
        }
    } trace_scope;
    auto sdk = dispatch();
    sdk.execution.launch_kernel = shared_shard_launch;
    sdk.host_allocate = shared_shard_host_allocate;
    sdk.host_map = shared_shard_host_map;
    sdk.upload_async = cuMemcpyHtoDAsync;
    sdk.blas.get_stream = cublasGetStream_v2;
    const CudaExecutionDomainDispatch domain_sdk{cuInit,          cuDeviceGet,     cuCtxCreate,
                                                 cuCtxSetCurrent, cuCtxGetCurrent, cuCtxDestroy};
    CudaExecutionDomainOwner domain(domain_sdk,
                                    {99, 0, CU_CTX_MAP_HOST, 2, std::size_t{512} * 1024 * 1024});
    check(domain.initialize());
    auto first = domain.create_member({{1, 99}, 101});
    auto second = domain.create_member({{2, 99}, 202});
    check(first.status);
    check(second.status);
    CudaSessionPersistentConfig config{320};
    config.replace_sgemm = config.observes_block_completion = config.uses_sharded_host_retreat =
        true;
    config.gemm_tile_side = 32;
    CudaSessionLimits limits;
    limits.max_device_bytes = std::size_t{256} * 1024 * 1024;
    limits.submissions = {2, 4096};
    auto second_limits = limits;
    second_limits.first_resource_id = 1000;
    CudaExecutionSession left({1, 99}, 101, sdk, limits, config, {}, {}, false, {},
                              std::move(first.member));
    CudaExecutionSession right({2, 99}, 202, sdk, second_limits, config, {}, {}, false, {},
                               std::move(second.member));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{100};
    const auto bounded_poll = [&](CudaExecutionSession& session) {
        expect(std::chrono::steady_clock::now() < deadline, "bounded shared-shard GPU work");
        check(session.poll().status);
        std::this_thread::yield();
    };
    const auto settle = [&](CudaExecutionSession& session) {
        do {
            bounded_poll(session);
        } while (session.snapshot().external_work_in_flight ||
                 session.snapshot().submissions.awaiting_physical_stop);
    };
    struct Resources {
        std::uint64_t handle{}, a{}, b{}, c{};
        std::size_t member{};
    };
    std::vector<float> a(input_elements, 1.0F / 16), b(input_elements, 1.0F / 32);
    std::vector<float> initial(elements + guards, 0.25F);
    std::fill(initial.end() - guards, initial.end(), 13.0F);
    const auto prepare = [&](CudaExecutionSession& session) {
        success(session.initialize(0));
        expect(shared_trace.count && shared_trace.count <= 2, "record this member's host mapping");
        const auto member = shared_trace.count - 1;
        const auto handle = session.create_blas();
        expect(handle.status == CUDA_SUCCESS, "create shared-shard BLAS owner");
        const auto a_device = session.allocate(a.size() * sizeof(float));
        const auto b_device = session.allocate(b.size() * sizeof(float));
        const auto c_device = session.allocate(initial.size() * sizeof(float));
        success(a_device);
        success(b_device);
        success(c_device);
        for (const auto& [id, bytes] : {std::pair{a_device.id, std::as_bytes(std::span{a})},
                                        std::pair{b_device.id, std::as_bytes(std::span{b})}}) {
            success(session.upload({id, 0}, bytes));
            settle(session);
        }
        success(session.deactivate());
        return Resources{handle.id, a_device.id, b_device.id, c_device.id, member};
    };
    const auto left_resources = prepare(left);
    const auto right_resources = prepare(right);
    const auto initialize_output = [&](CudaExecutionSession& session, const Resources& resources) {
        success(session.activate());
        success(session.upload({resources.c, 0}, std::as_bytes(std::span{initial})));
        settle(session);
        success(session.deactivate());
    };
    const auto issue = [&](CudaExecutionSession& session, const Resources& resources,
                           std::uint64_t sequence) {
        success(session.activate());
        const CudaSessionSgemm problem{.handle = resources.handle,
                                       .transpose_a = CUBLAS_OP_N,
                                       .transpose_b = CUBLAS_OP_N,
                                       .rows = side,
                                       .columns = side,
                                       .depth = depth,
                                       .leading_a = side,
                                       .leading_b = depth,
                                       .leading_c = side,
                                       .a = {resources.a, 0},
                                       .b = {resources.b, 0},
                                       .c = {resources.c, 0},
                                       .alpha = 1.0F,
                                       .beta = 0.5F};
        const auto accepted = session.accept_sgemm(sequence, problem);
        shared_trace.active = resources.member;
        expect(accepted.accepted() &&
                   session.issue_next().state == CudaExecutionIssueState::kIssued &&
                   session.acknowledge_result(accepted.receipt),
               "shared member issues its exact owned sharded GEMM");
        success(session.deactivate());
    };
    const auto check_shards = [&](std::size_t member, std::uint32_t expected) {
        const auto& arena = shared_trace.mappings[member];
        const auto offset = shared_trace.signals[member] - arena.device;
        for (std::size_t index = 0; index < kGemmSignalShards; ++index) {
            auto& word = *reinterpret_cast<std::uint32_t*>(arena.host + offset +
                                                           index * kGemmSignalShardBytes);
            expect(std::atomic_ref(word).load(std::memory_order_acquire) == expected,
                   "actual shared members have independent signal values");
        }
    };
    const auto retreat = [&](CudaExecutionSession& session, const Resources& resources,
                             std::uint64_t sequence, bool peer_must_be_clear) {
        success(session.activate());
        auto progress = session.persistent_progress(sequence);
        while (progress && !progress->has_completed_block) {
            expect(std::chrono::steady_clock::now() < deadline, "bounded first completed block");
            std::this_thread::yield();
            progress = session.persistent_progress(sequence);
        }
        expect(progress && progress->physical_launches == 1, "exact member has useful GPU work");
        check(session.request_retreat(sequence));
        check_shards(resources.member, 1);
        if (peer_must_be_clear) check_shards(1 - resources.member, 0);
        success(session.deactivate());
    };
    const auto resume = [&](CudaExecutionSession& session, const Resources& resources,
                            std::uint64_t sequence) {
        success(session.activate());
        auto progress = session.persistent_progress(sequence);
        while (progress && !progress->physical_stops) {
            bounded_poll(session);
            progress = session.persistent_progress(sequence);
        }
        expect(progress && progress->completed_blocks &&
                   progress->completed_blocks < progress->total_blocks,
               "actual shared member stopped with work remaining");
        shared_trace.active = resources.member;
        expect(session.resume(sequence).state == CudaExecutionIssueState::kIssued,
               "actual shared member resumes its retained command");
        check_shards(resources.member, 0);
        success(session.deactivate());
    };
    const auto finish = [&](CudaExecutionSession& session) {
        success(session.activate());
        while (session.snapshot().submissions.outstanding) bounded_poll(session);
        success(session.deactivate());
    };
    for (std::uint64_t sequence = 1; sequence <= 2; ++sequence) {
        initialize_output(left, left_resources);
        initialize_output(right, right_resources);
        issue(left, left_resources, sequence);
        issue(right, right_resources, sequence);
        if (sequence == 1) {
            retreat(left, left_resources, sequence, true);
            retreat(right, right_resources, sequence, false);
            resume(left, left_resources, sequence);
            resume(right, right_resources, sequence);
        } else {
            retreat(right, right_resources, sequence, true);
            retreat(left, left_resources, sequence, false);
            resume(right, right_resources, sequence);
            resume(left, left_resources, sequence);
        }
        finish(left);
        finish(right);
        std::vector<float> output(initial.size());
        for (const auto& [session, resources] :
             {std::pair{&left, &left_resources}, std::pair{&right, &right_resources}}) {
            success(session->activate());
            success(
                session->download({resources->c, 0}, std::as_writable_bytes(std::span{output})));
            settle(*session);
            expect(std::ranges::all_of(std::span{output}.first(elements),
                                       [](float value) { return value == 4.125F; }) &&
                       std::ranges::all_of(std::span{output}.last(guards),
                                           [](float value) { return value == 13.0F; }),
                   "every shared result and guard matches the exact non-idempotent oracle");
            success(session->deactivate());
        }
    }
    for (const auto& [session, resources] :
         {std::pair{&left, &left_resources}, std::pair{&right, &right_resources}}) {
        success(session->activate());
        expect(session->snapshot().persistent_accepted == 2 &&
                   session->destroy_blas(resources->handle).status == CUDA_SUCCESS,
               "each member actually selected the provider twice");
        success(session->shutdown());
    }
    check(domain.close());
    check(domain.shutdown());
    std::cout << "{\"status\":\"passed\",\"shared_members\":2,\"partial_retreats\":4,"
                 "\"full_device_qualified\":false,\"performance_qualified\":false}\n";
}
}  // namespace
int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--shared-sharded") {
            run_shared_sharded();
            return EXIT_SUCCESS;
        }
        expect(argc == 2 || argc == 4 || argc == 5,
               "one mode and optional explicit GEMM tile geometry/signal required");
        const std::string_view mode{argv[1]};
        expect(mode == "--native" || mode == "--ordinary" || mode == "--observed",
               "unknown GPU input");
        unsigned int tile_side = 64;
        if (argc >= 4) {
            const std::string_view side{argv[3]};
            expect(std::string_view(argv[2]) == "--gemm-tile-side" &&
                       (side == "32" || side == "64") && mode != "--native",
                   "only replacement inputs select a supported private tile family");
            tile_side = side == "32" ? 32 : 64;
        }
        const bool sharded_retreat = argc == 5 && std::string_view(argv[4]) == "--sharded-retreat";
        expect(argc != 5 || sharded_retreat, "unknown signal argument");
        run(mode != "--native", mode == "--observed", tile_side, sharded_retreat);
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "GEMM completion GPU: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
