#include "internal/cuda_execution_dispatcher.h"
#include "internal/cuda_execution_domain.h"
#include "internal/cuda_execution_session.h"
#include "internal/cuda_profile_cache.h"
#include "glimmer/transform/cuda_module_image.h"
#include "glimmer/transform/ptx_transform.h"
#include "../transform/compiler_ptx_fixture.h"
#include "../transform/computation_ptx_fixture.h"
#include "../transform/device_call_ptx_fixture.h"
#include "../transform/uniform_loop_ptx_fixture.h"
#include "../transform/aggregate_ptx_fixture.h"

#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
using namespace glimmer::backend::cuda;
constexpr unsigned int kWorkers = 320, kThreads = 32, kLogicalBlocks = 65536;
constexpr std::size_t kLowElements = std::size_t{kLogicalBlocks} * kThreads;
constexpr std::size_t kHighElements = std::size_t{kWorkers} * kThreads;
constexpr unsigned char kStatefulPtx[]{
#include "module_state_ptx.inc"
};
alignas(8) constexpr unsigned char kStatefulFatbin[]{
#include "module_state_fatbin.inc"
};
alignas(8) constexpr unsigned char kStatefulCompanion[]{
#include "module_state_companion_fatbin.inc"
};
enum class VariantModule {
    kStatelessPtx,
    kStatefulPtx,
    kStatefulBinary,
    kStatefulCompanion,
    kCompilerPtx,
    kDeviceCallPtx,
    kAtomicPtx,
    kWarpPtx,
    kUniformLoopPtx,
    kReconvergentLoopPtx
};

void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void check(CUresult status, const char* message) {
    if (status != CUDA_SUCCESS)
        throw std::runtime_error(std::string(message) + ": CUDA " + std::to_string(status));
}
void inspect_reconvergent_retreat_oracle();
void inspect_compiler_input() {
    using Kind = glimmer::transform::PtxExecutionKind;
    for (const auto source :
         {kCompilerPtx, kDeviceCallPtx, kAtomicPtx, kWarpPtx, std::string_view{kUniformLoopPtx},
          std::string_view{kReconvergentLoopPtx}}) {
        const auto storage = glimmer::transform::inspect_ptx_static_storage(source);
        expect(storage && *storage == 0,
               "local/debug data must not become module-global reservations");
        for (const auto kind : {Kind::kStatic, Kind::kBlockStatic, Kind::kDynamic,
                                Kind::kPreemptive, Kind::kSliced}) {
            const auto transformed = glimmer::transform::transform_persistent_ptx(source, kind);
            if ((source == kWarpPtx || source == kUniformLoopPtx ||
                 source == kReconvergentLoopPtx) &&
                kind == Kind::kStatic) {
                expect(!transformed, "warp input cannot use independent-thread execution");
                continue;
            }
            expect(transformed && transformed->image.starts_with(source) &&
                       transformed->kernels.size() == 1 &&
                       transformed->kernels.front().parameter_count == 2,
                   "owned compiler input prepares every family and preserves its public ABI");
            const auto prepared_storage =
                glimmer::transform::inspect_ptx_static_storage(transformed->image);
            expect(prepared_storage && *prepared_storage == 0,
                   "generated local/debug data must retain the same static-storage contract");
        }
    }
    inspect_reconvergent_retreat_oracle();
    std::cout << "Compiler input inspected: GPU_initialized=false; native_execution=false\n";
}
void inspect_module_state_inputs() {
    using Kind = glimmer::transform::PtxExecutionKind;
    const auto binary =
        std::as_bytes(std::span{kStatefulFatbin}).first(sizeof(kStatefulFatbin) - 1);
    const auto metadata = glimmer::transform::inspect_cuda_module(binary, 70);
    expect(metadata && metadata->is_ptx_selected && metadata->ptx,
           "nvcc-built fatbinary must select real PTX before stateful preparation");
    expect(metadata->ptx->starts_with(".version 7.4\n"),
           "real nvcc header major/minor restore the stripped version directive");
    const std::string_view source{reinterpret_cast<const char*>(kStatefulPtx),
                                  sizeof(kStatefulPtx) - 1};
    const auto companion_bytes =
        std::as_bytes(std::span{kStatefulCompanion}).first(sizeof(kStatefulCompanion) - 1);
    const auto companion = glimmer::transform::inspect_cuda_module(companion_bytes, 70);
    expect(companion && !companion->is_ptx_selected && companion->ptx &&
               companion->has_complete_kernel_metadata &&
               companion->has_complete_variable_metadata && companion->kernels.size() == 1 &&
               companion->kernels.front().name == "add_once" &&
               companion->kernels.front().arguments ==
                   std::vector<glimmer::transform::CudaImageArgument>{{0, 8}, {8, 4}} &&
               companion->variables.size() == 2 &&
               std::ranges::find(companion->variables,
                                 glimmer::transform::CudaImageVariable{"bias", 4}) !=
                   companion->variables.end() &&
               std::ranges::find(companion->variables,
                                 glimmer::transform::CudaImageVariable{"invocations", 4}) !=
                   companion->variables.end(),
           "real CUBIN companion preserves complete native entry and variable metadata");
    for (const auto input :
         {source, std::string_view{*metadata->ptx}, std::string_view{*companion->ptx}}) {
        expect(input.find(".u32 invocations = 13;") != std::string_view::npos &&
                   input.find(".u32 bias = 7;") != std::string_view::npos,
               "both intake forms retain the original initialized definitions");
        for (const auto kind : {Kind::kStatic, Kind::kBlockStatic, Kind::kDynamic,
                                Kind::kPreemptive, Kind::kSliced}) {
            const auto transformed =
                glimmer::transform::transform_persistent_ptx(input, kind, true);
            expect(transformed && transformed->has_module_variables &&
                       transformed->kernels.size() == 1 &&
                       transformed->kernels.front().original_name == "add_once" &&
                       transformed->image.starts_with(input),
                   "both real inputs prepare each family without duplicating state");
            expect(!glimmer::transform::transform_persistent_ptx(input, kind),
                   "legacy separate-module adapters retain the same stateful rejection");
        }
    }
    std::cout << "Module-state inputs inspected: GPU_initialized=false; native_execution=false\n";
}
constexpr const char* kSource = R"ptx(
.version 7.0
.target sm_70
.address_size 64
.visible .entry tile_add(.param .u64 output, .param .u64 marker,
                        .param .u32 offset, .param .u32 iterations) {
 .reg .pred %p<4>;
 .reg .b32 %r<9>;
 .reg .b64 %rd<9>;
 ld.param.u64 %rd1, [output]; ld.param.u64 %rd2, [marker];
 ld.param.u32 %r1, [offset]; ld.param.u32 %r2, [iterations];
 mov.u32 %r3, %ctaid.x; mov.u32 %r4, %ntid.x; mov.u32 %r5, %tid.x;
 mad.lo.u32 %r6, %r3, %r4, %r5; add.u32 %r6, %r6, %r1;
 mul.wide.u32 %rd3, %r6, 4; add.u64 %rd4, %rd1, %rd3;
 ld.global.u32 %r7, [%rd4]; add.u32 %r7, %r7, 1; st.global.u32 [%rd4], %r7;
 setp.ne.u32 %p1, %r5, 0; @%p1 bra EXIT;
 setp.eq.u32 %p2, %r2, 0; @%p2 bra MARK;
WAIT:
 ld.volatile.global.u32 %r7, [%rd4]; sub.u32 %r2, %r2, 1;
 setp.ne.u32 %p2, %r2, 0; @%p2 bra WAIT;
MARK:
 setp.ne.u32 %p3, %r3, 0; @%p3 bra EXIT;
 st.global.u32 [%rd2], 1; membar.sys;
EXIT: ret;
}
)ptx";

struct alignas(64) HostState {
    std::uint32_t retreat{}, marker{};
    std::uint64_t claimed{};
};
struct Resources {
    CUcontext context{};
    CUmodule module{};
    CUstream low_stream{}, high_stream{};
    CUdeviceptr output{}, counter{}, mapped{};
    HostState* host{};
    std::shared_ptr<bool> cleanup_ok{std::make_shared<bool>(true)};
    ~Resources() {
        const auto cleaned = [&](CUresult status) {
            if (status != CUDA_SUCCESS) *cleanup_ok = false;
        };
        if (host) std::atomic_ref(host->retreat).store(1, std::memory_order_release);
        if (context) cleaned(cuCtxSynchronize());
        if (output) cleaned(cuMemFree(output));
        if (counter) cleaned(cuMemFree(counter));
        if (low_stream) cleaned(cuStreamDestroy(low_stream));
        if (high_stream) cleaned(cuStreamDestroy(high_stream));
        if (module) cleaned(cuModuleUnload(module));
        if (host) cleaned(cuMemFreeHost(host));
        if (context) cleaned(cuCtxDestroy(context));
    }
};
class Progress final : public CudaExecutionPersistentProgress {
   public:
    explicit Progress(std::shared_ptr<Resources> resources) : resources_(std::move(resources)) {}
    CUresult prepare_launch() noexcept override {
        std::atomic_ref(resources_->host->retreat).store(0, std::memory_order_release);
        return CUDA_SUCCESS;
    }
    CUresult request_retreat() noexcept override {
        std::atomic_ref(resources_->host->retreat).store(1, std::memory_order_release);
        return CUDA_SUCCESS;
    }
    CUresult record_readback(CUstream stream) noexcept override {
        return cuMemcpyDtoHAsync(&resources_->host->claimed, resources_->counter,
                                 sizeof(resources_->host->claimed), stream);
    }
    CudaExecutionPersistentStop observe_physical_stop() noexcept override {
        // No CPU observation of a live GPU claim counter. The private trailing
        // event proves all claimed blocks and the stream-ordered readback ended.
        const auto claimed = resources_->host->claimed;
        if (claimed < observed_blocks) return {CUDA_ERROR_UNKNOWN, false};
        observed_blocks = claimed;
        ++physical_stops;
        return {CUDA_SUCCESS, claimed >= kLogicalBlocks};
    }
    std::uint64_t observed_blocks{};
    unsigned int physical_stops{};

   private:
    std::shared_ptr<Resources> resources_;
};
struct Parameters {
    CUdeviceptr output{}, marker{};
    std::uint32_t offset{}, iterations{};
};
struct PrivateParameters {
    Parameters original;
    std::uint32_t x{}, y{1}, z{1}, padding{};
    CUdeviceptr counter{}, retreat{};
};
static_assert(sizeof(Parameters) == 24 && sizeof(PrivateParameters) == 56 &&
              offsetof(PrivateParameters, counter) == 40);

void run(const std::shared_ptr<Resources>& resources) {
    CUdevice device{};
    check(cuDeviceGet(&device, 0), "get device");
    int sms{};
    check(cuDeviceGetAttribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device), "SM count");
    expect(sms == 80, "this correctness fixture explicitly targets the 80-SM V100");
    check(cuCtxCreate(&resources->context, CU_CTX_MAP_HOST, device), "owned context");
    const auto transformed = glimmer::transform::transform_persistent_ptx(
        kSource, glimmer::transform::PtxExecutionKind::kPreemptive);
    expect(transformed && transformed->kernels.size() == 1, "framework generates the variant");
    check(cuModuleLoadData(&resources->module, transformed->image.c_str()), "load generated PTX");
    CUfunction original{}, persistent{};
    check(cuModuleGetFunction(&original, resources->module, "tile_add"), "original entry");
    check(cuModuleGetFunction(&persistent, resources->module,
                              transformed->kernels.front().persistent_name.c_str()),
          "generated entry");
    constexpr std::array<std::size_t, 9> offsets{0, 8, 16, 20, 24, 28, 32, 40, 48};
    constexpr std::array<std::size_t, 9> sizes{8, 8, 4, 4, 4, 4, 4, 8, 8};
    for (std::size_t index = 0; index < offsets.size(); ++index) {
        std::size_t offset{}, bytes{};
        check(cuFuncGetParamInfo(persistent, index, &offset, &bytes), "private native ABI");
        expect(offset == offsets[index] && bytes == sizes[index], "exact generated parameter ABI");
        if (index < 4) {
            check(cuFuncGetParamInfo(original, index, &offset, &bytes), "original native ABI");
            expect(offset == offsets[index] && bytes == sizes[index], "exact original ABI");
        }
    }
    check(cuStreamCreate(&resources->low_stream, CU_STREAM_NON_BLOCKING), "low stream");
    check(cuStreamCreate(&resources->high_stream, CU_STREAM_NON_BLOCKING), "other stream");
    check(cuMemAlloc(&resources->output, (kLowElements + kHighElements) * sizeof(std::uint32_t)),
          "owned output");
    check(cuMemsetD32(resources->output, 0, kLowElements + kHighElements), "zero output");
    check(cuMemAlloc(&resources->counter, sizeof(std::uint64_t)), "owned counter");
    check(cuMemsetD8(resources->counter, 0, sizeof(std::uint64_t)), "zero counter once");
    check(cuMemHostAlloc(reinterpret_cast<void**>(&resources->host), sizeof(HostState),
                         CU_MEMHOSTALLOC_DEVICEMAP),
          "bounded private host state");
    std::construct_at(resources->host);
    check(cuMemHostGetDevicePointer(&resources->mapped, resources->host, 0), "map retreat state");
    const CudaExecutionDispatch dispatch{cuCtxGetCurrent,    cuLaunchKernel, cuEventCreate,
                                         cuEventRecord,      cuEventQuery,   cuEventDestroy,
                                         cuStreamSynchronize};
    CudaExecutionDispatcher executor({1, 1}, {1, resources->context, resources}, dispatch);
    check(executor.bind_executor(), "bind owner");
    expect(executor.bind_function(1, original, sizeof(Parameters), resources) &&
               executor.bind_stream(1, resources->low_stream, resources, true) &&
               executor.bind_stream(2, resources->high_stream, resources, true),
           "bind owned original ABI and independent streams");
    auto progress = std::make_shared<Progress>(resources);
    Parameters parameters{resources->output, resources->mapped + offsetof(HostState, marker), 0,
                          1024};
    PrivateParameters private_parameters{parameters,
                                         kLogicalBlocks,
                                         1,
                                         1,
                                         0,
                                         resources->counter,
                                         resources->mapped + offsetof(HostState, retreat)};
    CudaExecutionKernel logical{.function = 1,
                                .stream = 1,
                                .grid = {kLogicalBlocks, 1, 1},
                                .block = {kThreads, 1, 1},
                                .parameters = std::as_bytes(std::span{&parameters, 1})};
    const auto accepted = executor.accept_persistent_kernel(
        1, logical,
        {persistent, kWorkers, std::as_bytes(std::span{&private_parameters, 1}), progress});
    const auto low_frontier = executor.capture_frontier(1);
    parameters = {};
    private_parameters = {};
    expect(accepted.accepted() && low_frontier.status == CUDA_SUCCESS, "own copied private launch");
    const auto first = executor.issue_next();
    check(first.native_status, "first actual launch");
    check(first.completion_status, "first readback and event");
    expect(first.state == CudaExecutionIssueState::kIssued &&
               executor.acknowledge_result(accepted.receipt),
           "one application SDK reply");
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{30};
    const auto wait = [&] {
        expect(std::chrono::steady_clock::now() < deadline, "bounded physical correctness wait");
        std::this_thread::yield();
    };
    while (std::atomic_ref(resources->host->marker).load(std::memory_order_acquire) == 0) wait();
    check(executor.request_retreat(1), "publish owner retreat");
    while (executor.snapshot().paused == 0) {
        check(executor.poll().status, "actual partial stop query");
        wait();
    }
    expect(progress->observed_blocks > 0 && progress->observed_blocks < kLogicalBlocks &&
               progress->physical_stops == 1 && executor.snapshot().awaiting_physical_stop == 0 &&
               executor.query_frontier(low_frontier.frontier) == CUDA_ERROR_NOT_READY,
           "real incomplete compute stops without settling its logical frontier");
    const auto partial = progress->observed_blocks;
    Parameters high_parameters{resources->output, resources->mapped + offsetof(HostState, marker),
                               static_cast<std::uint32_t>(kLowElements), 0};
    logical.stream = 2;
    logical.grid = {kWorkers, 1, 1};
    logical.parameters = std::as_bytes(std::span{&high_parameters, 1});
    const auto high = executor.accept_kernel(2, logical);
    const auto high_frontier = executor.capture_frontier(2);
    expect(high.accepted() && high_frontier.status == CUDA_SUCCESS, "accept intervening work");
    const auto issued = executor.issue_next();
    check(issued.native_status, "intervening actual launch");
    check(issued.completion_status, "intervening real event");
    expect(executor.acknowledge_result(high.receipt), "acknowledge intervening SDK return");
    while (executor.query_frontier(high_frontier.frontier) == CUDA_ERROR_NOT_READY) {
        check(executor.poll().status, "actual intervening completion");
        wait();
    }
    check(executor.query_frontier(high_frontier.frontier), "intervening logical completion");
    expect(executor.snapshot().paused == 1 && progress->observed_blocks == partial &&
               executor.query_frontier(low_frontier.frontier) == CUDA_ERROR_NOT_READY,
           "completed other compute cannot manufacture completion or progress of paused low");
    const auto resumed = executor.resume(1);
    expect(resumed.state == CudaExecutionIssueState::kIssued, "exact continuation is issued");
    check(resumed.native_status, "actual continuation launch");
    check(resumed.completion_status, "continuation readback and event");
    while (executor.snapshot().outstanding != 0) {
        check(executor.poll().status, "actual final physical stop");
        wait();
    }
    expect(progress->physical_stops == 2 && progress->observed_blocks >= kLogicalBlocks &&
               executor.query_frontier(low_frontier.frontier) == CUDA_SUCCESS &&
               executor.shutdown(),
           "only final exact continuation settles the logical command");
    std::vector<std::uint32_t> actual(kLowElements + kHighElements);
    check(cuMemcpyDtoH(actual.data(), resources->output, actual.size() * sizeof(actual[0])),
          "read all low and intervening outputs");
    for (const auto value : actual) expect(value == 1, "every element executes exactly once");
    std::cout
        << "Persistent physical epochs verified: partial_blocks=" << partial
        << "; total_blocks=" << kLogicalBlocks << "; physical_low_launches=2; workers=" << kWorkers
        << "; exact_elements=" << actual.size()
        << "; full_device_qualified=false; public_integration=false; priority_qualified=false\n";
}

CudaSessionDispatch session_dispatch() {
    CudaSessionDispatch result{{cuCtxGetCurrent, cuLaunchKernel, cuEventCreate, cuEventRecord,
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
                               cuOccupancyMaxActiveBlocksPerMultiprocessor,
                               nullptr,
                               nullptr,
                               nullptr,
                               cuFuncSetAttribute};
    result.device_uuid = cuDeviceGetUuid;
    result.driver_version = cuDriverGetVersion;
    result.device_attribute = cuDeviceGetAttribute;
    result.module_global = cuModuleGetGlobal;
    return result;
}

void run_session_owned(bool retargets_workers = false) {
    const auto dispatch = session_dispatch();
    glimmer::backend::ExecutionSubmissionLimits submissions{2, 4096};
    CudaSessionLimits limits;
    limits.submissions = submissions;
    CudaExecutionSession session({1, 1}, 1, dispatch, limits, {kWorkers});
    const auto success = [](const CudaSessionResult& result) {
        expect(result.disposition == CudaSessionDisposition::kComplete,
               "owned resource creation complete");
        check(result.status, "owned resource operation");
    };
    success(session.initialize(0));
    expect(session.snapshot().device_bytes == 16 && session.snapshot().private_host_bytes == 128,
           "only the session creates and charges private progress resources");
    std::string source{kSource};
    const auto module = session.load_ptx(source);
    success(module);
    source.clear();
    const auto function = session.get_function(module.id, "tile_add");
    success(function);
    expect(
        function.arguments.size() == 4 && function.parameter_bytes == sizeof(Parameters) &&
            session.snapshot().persistent_functions == 1,
        "framework generates and validates private ABI without changing the public original ABI");
    const auto output = session.allocate((kLowElements + kHighElements) * sizeof(std::uint32_t));
    const auto marker = session.allocate(sizeof(std::uint32_t));
    const auto low_stream = session.create_stream(CU_STREAM_NON_BLOCKING);
    const auto high_stream = session.create_stream(CU_STREAM_NON_BLOCKING);
    for (const auto& result : {output, marker, low_stream, high_stream}) success(result);
    std::vector<std::uint32_t> actual(kLowElements + kHighElements, 0);
    success(session.upload({output.id, 0}, std::as_bytes(std::span{actual})));
    Parameters parameters{output.address, marker.address, 0, 1024};
    CudaExecutionKernel kernel{.function = function.id,
                               .stream = low_stream.id,
                               .grid = {kLogicalBlocks, 1, 1},
                               .block = {kThreads, 1, 1},
                               .parameters = std::as_bytes(std::span{&parameters, 1})};
    const auto low = session.accept_kernel(1, kernel);
    const auto low_frontier = session.capture_frontier(low_stream.id);
    expect(low.accepted() && low_frontier.status == CUDA_SUCCESS &&
               session.snapshot().persistent_accepted == 1,
           "actual session persistent selection");
    parameters = {};
    const auto initial = session.issue_next();
    expect(initial.state == CudaExecutionIssueState::kIssued,
           "session issues its own initial variant");
    check(initial.native_status, "initial actual variant SDK return");
    check(initial.completion_status, "initial owned readback/event");
    expect(session.acknowledge_result(low.receipt), "one original application reply");
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{30};
    const auto wait = [&] {
        expect(std::chrono::steady_clock::now() < deadline, "bounded session physical wait");
        std::this_thread::yield();
    };
    std::uint64_t partial{}, physical_launches{};
    for (unsigned int attempt = 0; attempt < 10; ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds{10});
        check(session.request_retreat(1), "session publishes owned retreat flag");
        while (!session.snapshot().submissions.paused) {
            check(session.poll().status, "real session partial event");
            wait();
        }
        const auto progress = session.persistent_progress(1);
        expect(progress && progress->completed_blocks < kLogicalBlocks &&
                   session.query_frontier(low_frontier.frontier) == CUDA_ERROR_NOT_READY,
               "partial stop preserves the same logical command");
        partial = progress->completed_blocks;
        physical_launches = progress->physical_launches;
        if (partial) break;
        const auto retry = session.resume(1);
        check(retry.native_status, "resume a genuinely zero-progress stopped epoch");
        check(retry.completion_status, "zero-progress continuation event");
    }
    expect(partial > 0 && partial < kLogicalBlocks,
           "actual positive incomplete compute is mandatory");
    Parameters high_parameters{output.address, marker.address,
                               static_cast<std::uint32_t>(kLowElements), 0};
    kernel.stream = high_stream.id;
    kernel.grid = {kWorkers, 1, 1};
    kernel.parameters = std::as_bytes(std::span{&high_parameters, 1});
    const auto high = session.accept_kernel(2, kernel);
    const auto high_frontier = session.capture_frontier(high_stream.id);
    expect(high.accepted() && high_frontier.status == CUDA_SUCCESS,
           "intervening session-owned compute");
    const auto issued = session.issue_next();
    check(issued.native_status, "intervening actual SDK result");
    check(issued.completion_status, "intervening private event");
    expect(session.acknowledge_result(high.receipt), "intervening original reply");
    while (session.query_frontier(high_frontier.frontier) == CUDA_ERROR_NOT_READY) {
        check(session.poll().status, "actual intervening stop");
        wait();
    }
    check(session.query_frontier(high_frontier.frontier), "actual intervening completion");
    expect(session.snapshot().persistent_slots == 1 &&
               session.persistent_progress(1)->completed_blocks == partial &&
               session.query_frontier(low_frontier.frontier) == CUDA_ERROR_NOT_READY,
           "other compute cannot complete or advance paused logical low work");
    std::uint64_t retargeted_partial{};
    if (retargets_workers) {
        check(session.retarget_workers(1, {80}), "retarget stopped session PTX to80 workers");
        expect(session.persistent_progress(1)->completed_blocks == partial &&
                   session.persistent_progress(1)->workers == 80 &&
                   session.query_frontier(low_frontier.frontier) == CUDA_ERROR_NOT_READY,
               "worker retarget is not logical progress or completion");
        const auto changed = session.resume(1);
        check(changed.native_status, "actual80-worker continuation");
        check(changed.completion_status, "actual80-worker continuation event");
        for (unsigned int attempt = 0; attempt < 10; ++attempt) {
            std::this_thread::sleep_for(std::chrono::milliseconds{2});
            check(session.request_retreat(1), "retreat actual retargeted interval");
            while (!session.snapshot().submissions.paused) {
                check(session.poll().status, "real retargeted stop");
                expect(session.persistent_progress(1).has_value(),
                       "retarget control requires a second genuinely incomplete stop");
                wait();
            }
            const auto progress = session.persistent_progress(1);
            expect(progress && progress->completed_blocks >= partial &&
                       progress->completed_blocks < kLogicalBlocks && progress->workers == 80,
                   "retargeted interval preserves monotone incomplete logical progress");
            retargeted_partial = progress->completed_blocks;
            physical_launches = progress->physical_launches;
            if (retargeted_partial > partial) break;
            const auto retry = session.resume(1);
            check(retry.native_status,
                  "retry a genuinely stopped zero-additional-progress interval");
            check(retry.completion_status, "retargeted retry event");
        }
        expect(retargeted_partial > partial,
               "retargeted80-worker interval must perform additional useful computation");
        check(session.retarget_workers(1, {160}), "retarget second physical stop to160 workers");
        expect(session.persistent_progress(1)->completed_blocks == retargeted_partial &&
                   session.persistent_progress(1)->workers == 160,
               "second retarget still retains the original exact progress");
    }
    const auto resumed = session.resume(1);
    check(resumed.native_status, "retained session continuation");
    check(resumed.completion_status, "continuation readback/event");
    while (session.snapshot().submissions.outstanding) {
        check(session.poll().status, "real session final stop");
        wait();
    }
    check(session.query_frontier(low_frontier.frontier), "final exact logical completion");
    expect(session.snapshot().persistent_slots == 0 && !session.persistent_progress(1),
           "final retirement returns only known-stopped private progress slots");
    const auto reused = session.accept_kernel(3, kernel);
    expect(reused.accepted() && session.issue_next().state == CudaExecutionIssueState::kIssued &&
               session.acknowledge_result(reused.receipt),
           "reuse a stopped slot for new logical work");
    while (session.snapshot().submissions.outstanding) {
        check(session.poll().status, "real reused-slot stop");
        wait();
    }
    success(session.download({output.id, 0}, std::as_writable_bytes(std::span{actual})));
    for (std::size_t index = 0; index < actual.size(); ++index)
        expect(
            actual[index] == (index < kLowElements ? 1U : 2U),
            "all original and reused-slot outputs are checked, without replay or stale counters");
    expect(session.snapshot().persistent_accepted == 3 && session.snapshot().persistent_slots == 0,
           "all three logical kernels used framework-owned persistent preparation");
    success(session.shutdown());
    expect(!session.snapshot().has_context && session.snapshot().private_host_bytes == 0 &&
               session.snapshot().device_bytes == 0,
           "all framework progress and context resources released");
    std::cout
        << "Session-owned persistent epochs verified: partial_blocks=" << partial
        << "; total_blocks=" << kLogicalBlocks
        << "; physical_low_launches=" << physical_launches + 1 << "; workers=" << kWorkers
        << "; exact_elements=" << actual.size()
        << "; retargeted_workers=" << (retargets_workers ? "320,80,160" : "none")
        << "; retargeted_partial_blocks=" << retargeted_partial
        << "; slots_reused=true; application_private_state=false; full_device_qualified=false; "
           "public_integration=false; priority_qualified=false\n";
}

std::string shared_domain_source() {
    std::string source{kSource};
    const auto entry = source.find(".visible .entry");
    expect(entry != std::string::npos, "shared input retains its original entry");
    source.insert(entry, ".visible .global .align 4 .u32 domain_bias = 0;\n");
    constexpr std::string_view addition = "add.u32 %r7, %r7, 1;";
    const auto location = source.find(addition);
    expect(location != std::string::npos, "shared input has exactly one non-idempotent increment");
    source.replace(location, addition.size(),
                   "ld.global.u32 %r8, [domain_bias]; add.u32 %r8, %r8, 1; "
                   "add.u32 %r7, %r7, %r8;");
    return source;
}
void inspect_shared_domain_input() {
    const auto source = shared_domain_source();
    const auto bytes = glimmer::transform::inspect_ptx_static_storage(source);
    const auto prepared = glimmer::transform::transform_persistent_ptx(
        source, glimmer::transform::PtxExecutionKind::kPreemptive, true);
    expect(bytes && *bytes == 7,
           "shared input charges four-byte global plus the existing three-byte alignment slack");
    expect(prepared && prepared->image.starts_with(source) && prepared->has_module_variables,
           "shared stateful input must prepare in its original module");
    expect(prepared->kernels.size() == 1 && prepared->kernels.front().parameter_count == 4,
           "shared input must retain one four-argument public ABI");
    std::cout << "Shared-domain input inspected: GPU_initialized=false; native_execution=false\n";
}

struct SharedDomainTrace {
    CUresult(CUDAAPI* original_get_shared)(CUsharedconfig*) {};
    CUresult(CUDAAPI* original_set_shared)(CUsharedconfig) {};
    unsigned int configuration_setters{};
    unsigned int context_queries{};
    struct Launch {
        std::size_t member{};
        unsigned int workers{};
        PrivateParameters parameters;
    };
    CUcontext context{};
    std::array<CUmodule, 2> modules{};
    std::array<CUfunction, 2> prepared{};
    std::vector<CUstream> streams;
    std::vector<Launch> launches;
    std::vector<CUdeviceptr> frees;
    unsigned int creates{}, destroys{}, loads{}, unloads{};
    bool has_mismatch{};
};
SharedDomainTrace* shared_domain_trace{};
CUresult CUDAAPI shared_context_get(CUcontext* context) {
    ++shared_domain_trace->context_queries;
    return cuCtxGetCurrent(context);
}
struct SharedDomainTraceScope {
    explicit SharedDomainTraceScope(SharedDomainTrace& trace) {
        expect(!shared_domain_trace, "one owner installs the actual Driver trace");
        shared_domain_trace = &trace;
    }
    ~SharedDomainTraceScope() {
        shared_domain_trace = nullptr;
    }
    SharedDomainTraceScope(const SharedDomainTraceScope&) = delete;
    SharedDomainTraceScope& operator=(const SharedDomainTraceScope&) = delete;
};
CUresult CUDAAPI shared_context_create(CUcontext* context, unsigned int flags, CUdevice device) {
    const auto status = cuCtxCreate(context, flags, device);
    auto& trace = *shared_domain_trace;
    ++trace.creates;
    if (status == CUDA_SUCCESS) trace.context = *context;
    return status;
}
CUresult CUDAAPI shared_context_destroy(CUcontext context) {
    auto& trace = *shared_domain_trace;
    trace.has_mismatch |= context != trace.context;
    ++trace.destroys;
    return cuCtxDestroy(context);
}
CUresult CUDAAPI shared_module_load(CUmodule* module, const void* image) {
    const auto status = cuModuleLoadData(module, image);
    auto& trace = *shared_domain_trace;
    if (status == CUDA_SUCCESS && trace.loads < trace.modules.size())
        trace.modules[trace.loads] = *module;
    else
        trace.has_mismatch = true;
    ++trace.loads;
    return status;
}
CUresult CUDAAPI shared_module_unload(CUmodule module) {
    auto& trace = *shared_domain_trace;
    trace.has_mismatch |= std::ranges::find(trace.modules, module) == trace.modules.end();
    ++trace.unloads;
    return cuModuleUnload(module);
}
CUresult CUDAAPI shared_function_get(CUfunction* function, CUmodule module, const char* name) {
    const auto status = cuModuleGetFunction(function, module, name);
    auto& trace = *shared_domain_trace;
    const auto member = std::ranges::find(trace.modules, module);
    if (status == CUDA_SUCCESS && member != trace.modules.end()) {
        if (std::string_view{name} == "tile_add__glimmer_ptb_preemptive")
            trace.prepared[static_cast<std::size_t>(member - trace.modules.begin())] = *function;
        else if (std::string_view{name} != "tile_add")
            trace.has_mismatch = true;
    } else
        trace.has_mismatch = true;
    return status;
}
CUresult CUDAAPI shared_stream_create(CUstream* stream, unsigned int flags) {
    const auto status = cuStreamCreate(stream, flags);
    auto& trace = *shared_domain_trace;
    trace.has_mismatch |= status != CUDA_SUCCESS || flags != CU_STREAM_NON_BLOCKING;
    if (status == CUDA_SUCCESS) trace.streams.push_back(*stream);
    return status;
}
CUresult CUDAAPI shared_free(CUdeviceptr address) {
    shared_domain_trace->frees.push_back(address);
    return cuMemFree(address);
}
// NOLINTBEGIN(bugprone-easily-swappable-parameters): exact CUDA Driver callback ABI.
CUresult CUDAAPI shared_launch(CUfunction function, unsigned int grid_x, unsigned int grid_y,
                               unsigned int grid_z, unsigned int block_x, unsigned int block_y,
                               unsigned int block_z, unsigned int shared_bytes, CUstream stream,
                               void** parameters, void** extra) {
    auto& trace = *shared_domain_trace;
    const auto member = std::ranges::find(trace.prepared, function);
    trace.has_mismatch |= member == trace.prepared.end() || grid_y != 1 || grid_z != 1 ||
                          block_x != kThreads || block_y != 1 || block_z != 1 || shared_bytes ||
                          parameters || !extra || !stream;
    if (member != trace.prepared.end() && trace.streams.size() == 4)
        trace.has_mismatch |=
            stream !=
            trace.streams[static_cast<std::size_t>(member - trace.prepared.begin()) * 2 + 1];
    else
        trace.has_mismatch = true;
    if (extra && extra[0] == CU_LAUNCH_PARAM_BUFFER_POINTER && extra[1] &&
        extra[2] == CU_LAUNCH_PARAM_BUFFER_SIZE && extra[3] && extra[4] == CU_LAUNCH_PARAM_END &&
        *static_cast<const std::size_t*>(extra[3]) == sizeof(PrivateParameters) &&
        member != trace.prepared.end()) {
        PrivateParameters input;
        std::memcpy(&input, extra[1], sizeof(input));
        trace.has_mismatch |= !input.counter || !input.retreat || input.y != 1 || input.z != 1;
        trace.launches.push_back(
            {static_cast<std::size_t>(member - trace.prepared.begin()), grid_x, input});
    } else
        trace.has_mismatch = true;
    return cuLaunchKernel(function, grid_x, grid_y, grid_z, block_x, block_y, block_z, shared_bytes,
                          stream, parameters, extra);
}
// NOLINTEND(bugprone-easily-swappable-parameters)

CUresult CUDAAPI configuration_get_shared(CUsharedconfig* value) {
    return shared_domain_trace->original_get_shared(value);
}
CUresult CUDAAPI configuration_set_shared(CUsharedconfig value) {
    ++shared_domain_trace->configuration_setters;
    return shared_domain_trace->original_set_shared(value);
}
CUresult CUDAAPI configuration_set_flags(unsigned int flags) {
    ++shared_domain_trace->configuration_setters;
    return cuCtxSetFlags(flags);
}
CUresult CUDAAPI configuration_set_cache(CUfunc_cache cache) {
    ++shared_domain_trace->configuration_setters;
    return cuCtxSetCacheConfig(cache);
}
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): exact native CUDA ABI.
CUresult CUDAAPI configuration_set_limit(CUlimit limit, std::size_t bytes) {
    ++shared_domain_trace->configuration_setters;
    return cuCtxSetLimit(limit, bytes);
}
template <class Function>
Function configuration_symbol(const char* name) {
    void* address{};
    CUdriverProcAddressQueryResult status{};
    check(cuGetProcAddress(name, &address, CUDA_VERSION, CU_GET_PROC_ADDRESS_DEFAULT, &status),
          "resolve exact original legacy configuration ABI");
    expect(address && status == CU_GET_PROC_ADDRESS_SUCCESS,
           "original configuration SDK entry exists");
    return reinterpret_cast<Function>(address);
}

void run_shared_domain_sessions(bool changes_configuration = false,
                                bool uses_exact_commands = false) {
    expect(!uses_exact_commands || changes_configuration,
           "exact pending input includes real logical/physical configuration switching");
    constexpr std::size_t guards = 32;
    SharedDomainTrace trace;
    trace.streams.reserve(4);
    trace.launches.reserve(16);
    trace.frees.reserve(8);
    const SharedDomainTraceScope trace_scope(trace);
    auto dispatch = session_dispatch();
    dispatch.context_create = shared_context_create;
    dispatch.context_destroy = shared_context_destroy;
    dispatch.module_load = shared_module_load;
    dispatch.module_unload = shared_module_unload;
    dispatch.function_get = shared_function_get;
    dispatch.stream_create = shared_stream_create;
    dispatch.release = shared_free;
    dispatch.execution.launch_kernel = shared_launch;
    dispatch.stream_wait_event = cuStreamWaitEvent;
    CudaExecutionDomainDispatch domain_dispatch{
        cuInit,          cuDeviceGet,     shared_context_create,
        cuCtxSetCurrent, cuCtxGetCurrent, shared_context_destroy};
    if (uses_exact_commands) {
        dispatch.execution.context_get_current = shared_context_get;
        domain_dispatch.context_get_current = shared_context_get;
    }
    if (changes_configuration) {
        trace.original_get_shared =
            configuration_symbol<decltype(trace.original_get_shared)>("cuCtxGetSharedMemConfig");
        trace.original_set_shared =
            configuration_symbol<decltype(trace.original_set_shared)>("cuCtxSetSharedMemConfig");
        domain_dispatch.configuration = {cuCtxGetFlags,
                                         configuration_set_flags,
                                         cuCtxGetLimit,
                                         configuration_set_limit,
                                         cuCtxGetCacheConfig,
                                         configuration_set_cache,
                                         configuration_get_shared,
                                         configuration_set_shared};
        dispatch.context_get_flags = cuCtxGetFlags;
        dispatch.context_set_flags = configuration_set_flags;
        dispatch.context.get_cache = cuCtxGetCacheConfig;
        dispatch.context.set_cache = configuration_set_cache;
        dispatch.context.get_limit = cuCtxGetLimit;
        dispatch.context.set_limit = configuration_set_limit;
        dispatch.context.get_shared = configuration_get_shared;
        dispatch.context.set_shared = configuration_set_shared;
    }
    CudaExecutionDomainOwner owner(domain_dispatch,
                                   {99, 0, CU_CTX_MAP_HOST, 2, std::size_t{64} * 1024 * 1024});
    check(owner.initialize(), "create one actual shared physical context");
    auto first = owner.create_member({{1, 99}, 101});
    auto second = owner.create_member({{2, 99}, 202});
    check(first.status, "create first exact member");
    check(second.status, "create second exact member");
    auto high_member = changes_configuration ? first.member : CudaExecutionDomainMember{};
    auto low_member = changes_configuration ? second.member : CudaExecutionDomainMember{};
    auto retained = first.member.view();
    {
        const auto other = second.member.view();
        check(retained.status, "first member native view");
        check(other.status, "second member native view");
        expect(retained.context.native == other.context.native &&
                   retained.logical.context != other.logical.context &&
                   retained.generation != other.generation,
               "actual common native context does not merge logical identity");
    }
    CudaSessionLimits first_limits;
    first_limits.submissions = {2, 4096};
    first_limits.first_resource_id = 1000;
    auto second_limits = first_limits;
    second_limits.first_resource_id = 2000;
    CudaExecutionSession high({1, 99}, 101, dispatch, first_limits, {kWorkers}, {}, {}, false, {},
                              std::move(first.member));
    CudaExecutionSession low({2, 99}, 202, dispatch, second_limits, {kWorkers}, {}, {}, false, {},
                             std::move(second.member));
    const auto success = [](const CudaSessionResult& result) {
        expect(result.disposition == CudaSessionDisposition::kComplete,
               "actual common-domain Session operation must complete");
        check(result.status, "actual common-domain Session operation");
    };
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{60};
    const auto poll = [&](CudaExecutionSession& session) {
        expect(std::chrono::steady_clock::now() < deadline, "bounded common-domain physical wait");
        check(session.poll().status, "actual member event/readback proof");
        std::this_thread::yield();
    };
    const auto settle = [&](CudaExecutionSession& session) {
        do {
            poll(session);
        } while (session.snapshot().external_work_in_flight ||
                 session.snapshot().submissions.awaiting_physical_stop);
    };
    unsigned int exact_commands{}, rejected_duplicates{};
    bool rejects_old_configuration{}, rejects_old_geometry{};
    const auto execute_pending = [&](CudaExecutionSession& session) {
        const auto queries = trace.context_queries;
        const auto pending = session.pending_command();
        check(pending.status, "capture actual exact pending command");
        expect(pending.command && trace.context_queries == queries,
               "pending capture has no SDK or physical side effect");
        const auto result = session.issue(pending.command);
        if (result.state == CudaExecutionIssueState::kIssued) {
            ++exact_commands;
            const auto after = trace.context_queries;
            const auto launches = trace.launches.size();
            const auto duplicate = session.issue(pending.command);
            expect(duplicate.state == CudaExecutionIssueState::kError &&
                       (duplicate.native_status == CUDA_ERROR_NOT_READY ||
                        duplicate.native_status == CUDA_ERROR_INVALID_HANDLE) &&
                       trace.context_queries == after && trace.launches.size() == launches,
                   "an already consumed exact command cannot query SDK, resume or select a "
                   "substitute");
            ++rejected_duplicates;
        }
        return result;
    };
    struct MemberResources {
        CudaSessionResult module, function, bias, output, marker, stream;
    };
    struct MemberInput {
        std::size_t elements{};
        std::uint32_t bias{};
    };
    const auto prepare = [&](CudaExecutionSession& session, MemberInput input) {
        const auto elements = input.elements;
        const auto bias = input.bias;
        success(session.initialize(0));
        MemberResources result;
        result.module = session.load_ptx(shared_domain_source());
        success(result.module);
        result.function = session.get_function(result.module.id, "tile_add");
        result.bias = session.get_global(result.module.id, "domain_bias");
        result.output = session.allocate((elements + guards) * sizeof(std::uint32_t));
        result.marker = session.allocate(sizeof(std::uint32_t));
        result.stream = session.create_stream(CU_STREAM_DEFAULT);
        for (const auto& item :
             {result.function, result.bias, result.output, result.marker, result.stream})
            success(item);
        expect(result.bias.parameter_bytes == sizeof(bias), "exact native mutable-global extent");
        const std::vector<std::uint32_t> zeros(elements + guards, 0);
        success(session.upload({result.output.id, 0}, std::as_bytes(std::span{zeros})));
        settle(session);
        success(session.upload({result.bias.id, 0}, std::as_bytes(std::span{&bias, 1})));
        settle(session);
        success(session.deactivate());
        return result;
    };
    const auto high_resources = prepare(high, {kHighElements, 2});
    const auto low_resources = prepare(low, {kLowElements, 4});
    expect(trace.creates == 1 && trace.loads == 2 && trace.modules[0] != trace.modules[1] &&
               high_resources.bias.address != low_resources.bias.address &&
               trace.streams.size() == 4 && trace.streams[0] != trace.streams[2],
           "actual module globals and default proxies are independently owned on one context");
    struct MemberCommand {
        std::uint64_t sequence{};
        unsigned int blocks{};
        std::uint32_t iterations{};
    };
    const auto issue = [&](CudaExecutionSession& session, const MemberResources& resources,
                           MemberCommand command) {
        Parameters parameters{resources.output.address, resources.marker.address, 0,
                              command.iterations};
        const CudaExecutionKernel kernel{.function = resources.function.id,
                                         .stream = resources.stream.id,
                                         .grid = {command.blocks, 1, 1},
                                         .block = {kThreads, 1, 1},
                                         .parameters = std::as_bytes(std::span{&parameters, 1})};
        const auto accepted = session.accept_kernel(command.sequence, kernel);
        expect(accepted.accepted(), "actual member owns one exact accepted logical command");
        const auto result = uses_exact_commands ? execute_pending(session) : session.issue_next();
        expect(result.state == CudaExecutionIssueState::kIssued, "actual prepared member issue");
        check(result.native_status, "actual member original SDK launch status");
        check(result.completion_status, "actual member private event/readback status");
        expect(session.acknowledge_result(accepted.receipt), "one exact logical host reply");
    };
    success(high.activate());
    std::uint64_t baseline_flags{}, baseline_cache{}, configured_flags{}, configured_cache{},
        physical_epoch{};
    if (changes_configuration) {
        const auto original_flags = high.get_context_flags();
        success(original_flags);
        baseline_flags = original_flags.id;
        const auto before = high.context_control(CudaContextControl::kGetCache, 0);
        expect(before.status == CUDA_SUCCESS && before.original_status == CUDA_SUCCESS,
               "exact original fresh cache preference");
        baseline_cache = before.value;
        success(high.set_context_flags(CU_CTX_SCHED_YIELD));
        const auto changed =
            high.context_control(CudaContextControl::kSetCache, CU_FUNC_CACHE_PREFER_L1);
        expect(changed.status == CUDA_SUCCESS && changed.original_status == CUDA_SUCCESS,
               "actual original cache setter completes once");
        const auto flags = high.get_context_flags();
        const auto cache = high.context_control(CudaContextControl::kGetCache, 0);
        success(flags);
        expect(cache.status == CUDA_SUCCESS && cache.original_status == CUDA_SUCCESS &&
                   high_member.view().configuration_epoch == 3,
               "normalized native flags/cache are retained with an exact logical epoch");
        configured_flags = flags.id;
        configured_cache = cache.value;
        expect((configured_flags & CU_CTX_SCHED_MASK) == CU_CTX_SCHED_YIELD &&
                   (baseline_flags & CU_CTX_SCHED_MASK) != CU_CTX_SCHED_YIELD,
               "high member truly has a different physical configuration, not just different "
               "metadata");
    }
    issue(high, high_resources, {1, kWorkers, 0});
    settle(high);  // Materialize the short member's code before the held low epoch.
    success(high.deactivate());
    success(low.activate());
    if (changes_configuration) {
        const auto flags = low.get_context_flags();
        const auto cache = low.context_control(CudaContextControl::kGetCache, 0);
        success(flags);
        expect(flags.id == baseline_flags && cache.status == CUDA_SUCCESS &&
                   cache.original_status == CUDA_SUCCESS && cache.value == baseline_cache &&
                   low_member.view().configuration_epoch == 1,
               "low restores its fresh original baseline before issuing actual compute");
    }
    issue(low, low_resources, {1, kLogicalBlocks, 1024});
    auto low_frontier = low.capture_command_frontier(1);
    check(low_frontier.status, "capture exact incomplete low command");
    success(low.deactivate());
    if (changes_configuration) {
        const auto setters = trace.configuration_setters;
        const auto deferred = high.activate();
        CUcontext selected{};
        check(cuCtxGetCurrent(&selected), "observe failed configuration activation TLS");
        expect(deferred.disposition == CudaSessionDisposition::kDeferred &&
                   deferred.status == CUDA_ERROR_NOT_READY && !selected &&
                   trace.configuration_setters == setters && trace.destroys == 0 &&
                   trace.unloads == 0,
               "different high configuration defers before setters/cleanup and restores NULL TLS");
    } else {
        success(high.activate());
        issue(high, high_resources, {2, kWorkers, 0});
        settle(high);
        expect(low.snapshot().submissions.awaiting_physical_stop == 1,
               "short member completion does not discharge another member's physical ownership");
        const auto deferred = high.shutdown();
        expect(deferred.disposition == CudaSessionDisposition::kDeferred &&
                   deferred.status == CUDA_ERROR_NOT_READY && trace.destroys == 0 &&
                   trace.unloads == 0 && trace.frees.empty(),
               "actual sibling shutdown defers before native cleanup or context destruction");
        success(high.deactivate());
    }
    success(low.activate());
    std::uint64_t partial{}, low_epochs{};
    for (unsigned int attempt = 0; attempt != 10; ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds{10});
        check(low.request_retreat(1), "request actual shared-context incomplete retreat");
        while (!low.snapshot().submissions.paused) poll(low);
        const auto progress = low.persistent_progress(1);
        expect(progress && progress->completed_blocks < kLogicalBlocks &&
                   low.query_frontier(low_frontier.frontier) == CUDA_ERROR_NOT_READY,
               "actual shared physical stop is not logical completion");
        partial = progress->completed_blocks;
        low_epochs = progress->physical_launches;
        if (partial) break;
        const auto retry = uses_exact_commands ? execute_pending(low) : low.resume(1);
        expect(retry.state == CudaExecutionIssueState::kIssued,
               "only a proven zero-progress stopped epoch may retry");
        check(retry.native_status, "actual zero-progress continuation");
        check(retry.completion_status, "actual zero-progress continuation event");
    }
    expect(partial > 0 && partial < kLogicalBlocks, "real positive incomplete low compute");
    const auto before_switch =
        uses_exact_commands ? low.pending_command() : CudaSessionPendingResult{};
    if (uses_exact_commands)
        expect(before_switch.command && before_switch.command.description().command.is_resume,
               "actual incomplete stop supplies exact continuation identity before high work");
    if (changes_configuration) {
        success(low.deactivate());
        success(high.activate());
        const auto flags = high.get_context_flags();
        const auto cache = high.context_control(CudaContextControl::kGetCache, 0);
        success(flags);
        expect(flags.id == configured_flags && cache.value == configured_cache &&
                   cache.status == CUDA_SUCCESS && cache.original_status == CUDA_SUCCESS,
               "high restores its own actual configuration only after true incomplete stop");
        issue(high, high_resources, {2, kWorkers, 0});
        settle(high);
        success(high.deactivate());
        success(low.activate());
        if (uses_exact_commands) {
            const auto queries = trace.context_queries;
            const auto launches = trace.launches.size();
            const auto stale = low.issue(before_switch.command);
            rejects_old_configuration = stale.state == CudaExecutionIssueState::kError &&
                                        stale.native_status == CUDA_ERROR_INVALID_HANDLE &&
                                        trace.context_queries == queries &&
                                        trace.launches.size() == launches;
            expect(rejects_old_configuration,
                   "actual configuration switch invalidates an old grant before any SDK callback");
        }
    }
    // Read-only independent oracle after all physical work stopped. high's
    // legacy-mode shutdown closed its dispatcher, but has not cleaned resources.
    CUcontext current{};
    check(cuCtxGetCurrent(&current), "prove oracle's actual current domain");
    expect(current == retained.context.native, "closed member oracle remains in the same domain");
    std::vector<std::uint32_t> high_values(kHighElements + guards);
    check(cuMemcpyDtoH(high_values.data(), high_resources.output.address,
                       high_values.size() * sizeof(std::uint32_t)),
          "complete closed-member oracle");
    for (std::size_t index = 0; index != high_values.size(); ++index)
        expect(high_values[index] == (index < kHighElements ? 6U : 0U),
               "both independent high increments and every guard match its private global");
    std::array<std::uint32_t, 2> high_metadata{};
    check(cuMemcpyDtoH(&high_metadata[0], high_resources.bias.address, sizeof(std::uint32_t)),
          "read exact closed-member bias");
    check(cuMemcpyDtoH(&high_metadata[1], high_resources.marker.address, sizeof(std::uint32_t)),
          "read exact closed-member completion marker");
    expect(high_metadata == std::array<std::uint32_t, 2>{2, 1},
           "closed member metadata keeps its own initialized global and GPU-written marker");
    success(low.deactivate());
    success(high.activate());
    success(high.shutdown());
    expect(trace.destroys == 0 && trace.unloads == 1 && !high.snapshot().has_context &&
               high.snapshot().device_bytes == 0 && high.snapshot().private_host_bytes == 0 &&
               owner.snapshot().reserved_device_bytes == low.snapshot().device_bytes,
           "actual member cleanup refunds only its resources without destroying shared context");
    success(low.activate());
    if (changes_configuration) {
        const auto flags = low.get_context_flags();
        const auto cache = low.context_control(CudaContextControl::kGetCache, 0);
        success(flags);
        expect(flags.id == baseline_flags && cache.value == baseline_cache &&
                   cache.status == CUDA_SUCCESS && cache.original_status == CUDA_SUCCESS &&
                   low_member.view().configuration_epoch == 1 &&
                   !low_member.view().has_unobserved_configuration_effects,
               "sibling cleanup preserves baseline and observed epoch of the retained low command");
    }
    const auto before_retarget =
        uses_exact_commands ? low.pending_command() : CudaSessionPendingResult{};
    check(low.retarget_workers(1, {160}), "retarget retained shared command after actual stop");
    if (uses_exact_commands) {
        const auto queries = trace.context_queries;
        const auto launches = trace.launches.size();
        const auto stale = low.issue(before_retarget.command);
        rejects_old_geometry = stale.state == CudaExecutionIssueState::kError &&
                               stale.native_status == CUDA_ERROR_INVALID_HANDLE &&
                               trace.context_queries == queries &&
                               trace.launches.size() == launches;
        expect(rejects_old_geometry,
               "actual worker retarget invalidates an old grant before any SDK callback");
    }
    const auto resumed = uses_exact_commands ? execute_pending(low) : low.resume(1);
    expect(resumed.state == CudaExecutionIssueState::kIssued,
           "resume the same logical low command");
    check(resumed.native_status, "actual160-worker shared continuation");
    check(resumed.completion_status, "actual shared continuation event");
    settle(low);
    check(low.query_frontier(low_frontier.frontier), "actual retained logical low completion");
    expect(!low.persistent_progress(1), "only actual final retirement returns its private slot");
    issue(low, low_resources, {2, kWorkers, 0});
    settle(low);
    std::vector<std::uint32_t> low_values(kLowElements + guards);
    success(
        low.download({low_resources.output.id, 0}, std::as_writable_bytes(std::span{low_values})));
    settle(low);
    for (std::size_t index = 0; index != low_values.size(); ++index)
        expect(low_values[index] == (index < kHighElements  ? 10U
                                     : index < kLowElements ? 5U
                                                            : 0U),
               "every low/reused-slot output and guard detects lost work or replay");
    std::array<std::uint32_t, 2> low_metadata{};
    check(cuMemcpyDtoH(&low_metadata[0], low_resources.bias.address, sizeof(std::uint32_t)),
          "read exact continued-member bias");
    check(cuMemcpyDtoH(&low_metadata[1], low_resources.marker.address, sizeof(std::uint32_t)),
          "read exact continued-member completion marker");
    expect(low_metadata == std::array<std::uint32_t, 2>{4, 1},
           "continued member retains independent global and completion metadata");
    CUdeviceptr high_counter{}, low_counter{};
    std::size_t high_launches{}, low_launches{};
    for (const auto& launch : trace.launches) {
        const auto& input = launch.parameters;
        expect(input.original.offset == 0, "actual member launch retains its public offset ABI");
        if (launch.member == 0) {
            expect(launch.workers == kWorkers && input.x == kWorkers &&
                       input.original.output == high_resources.output.address &&
                       input.original.marker == high_resources.marker.address &&
                       input.original.iterations == 0,
                   "every high physical launch uses its own function/geometry/arguments");
            if (!high_counter) high_counter = input.counter;
            expect(input.counter == high_counter, "high fresh slots remain in their own arena");
            ++high_launches;
        } else {
            if (!low_counter) low_counter = input.counter;
            expect(input.counter == low_counter &&
                       input.original.output == low_resources.output.address &&
                       input.original.marker == low_resources.marker.address &&
                       input.original.iterations == (low_launches <= low_epochs ? 1024U : 0U),
                   "every retained low launch uses the same actual private progress counter");
            const auto expected_workers = low_launches == low_epochs ? 160U : kWorkers;
            expect(launch.workers == expected_workers &&
                       input.x == (low_launches <= low_epochs ? kLogicalBlocks : kWorkers),
                   "actual low epochs preserve geometry across retarget and fresh-slot reuse");
            ++low_launches;
        }
    }
    expect(!trace.has_mismatch && high_launches == 2 && low_launches == low_epochs + 2 &&
               high_counter && low_counter && high_counter != low_counter &&
               std::ranges::find(trace.frees, high_counter) != trace.frees.end() &&
               std::ranges::find(trace.frees, low_counter) == trace.frees.end(),
           "actual cleanup cannot release or replace another member's retained progress counter");
    low_frontier = {};
    if (changes_configuration) {
        const auto view = low_member.view();
        check(view.status, "final actual observed configuration epoch");
        expect(!view.has_unobserved_configuration_effects && view.is_configuration_materialized &&
                   view.configuration_epoch == 1 && view.physical_configuration_epoch == 8 &&
                   trace.configuration_setters >= 7 && trace.configuration_setters <= 12,
               "all actual tails are observed and all SDK mutations match bounded configuration "
               "switches");
        physical_epoch = view.physical_configuration_epoch;
    }
    success(low.shutdown());
    high_member = {};
    low_member = {};
    expect(owner.snapshot().reserved_device_bytes == 0 && trace.unloads == 2 && trace.destroys == 0,
           "both members explicitly retire before native context destruction");
    check(owner.close(), "seal the native domain after member retirement");
    expect(owner.shutdown() == CUDA_ERROR_NOT_READY && trace.destroys == 0,
           "settled exported view is still a real physical-lifetime owner");
    retained = {};
    check(owner.shutdown(), "destroy only the last fully settled physical owner");
    check(cuCtxGetCurrent(&current), "observe final NULL native TLS");
    expect(!current && trace.creates == 1 && trace.destroys == 1 && !trace.has_mismatch,
           "actual shared context creates/destroys once and leaves NULL TLS");
    std::cout << (uses_exact_commands     ? "Exact-command GPU verified: partial_blocks="
                  : changes_configuration ? "Shared-configuration GPU verified: partial_blocks="
                                          : "Shared-domain GPU verified: partial_blocks=")
              << partial << "; total_blocks=" << kLogicalBlocks
              << "; low_physical_launches=" << low_epochs + 1
              << "; workers=320,160; exact_elements=" << high_values.size() + low_values.size() + 4
              << "; native_contexts=1; native_destroys=1; member_modules=2; ";
    if (changes_configuration)
        std::cout << "logical_epochs=3,1; physical_configuration_epoch=" << physical_epoch
                  << "; configuration_setters=" << trace.configuration_setters
                  << "; configurations_restored=true; implicit_effects_verified=true; ";
    if (uses_exact_commands) {
        expect(exact_commands == low_epochs + 4 && rejected_duplicates == exact_commands &&
                   rejects_old_configuration && rejects_old_geometry,
               "every actual exact initial/resume and stale rejection has independent counters");
        std::cout << "exact_pending_commands=" << exact_commands
                  << "; duplicate_tokens_rejected=" << rejected_duplicates
                  << "; stale_configuration_rejected=true; stale_geometry_rejected=true; "
                     "SDK_free_rejections=true; ";
    }
    std::cout << "independent_globals=true; independent_counters=true; "
              << (changes_configuration ? "deferred_configuration=true; "
                                        : "deferred_cleanup=true; ")
              << "retained_counter=true; exported_view_pins=true; slots_reused=true; "
                 "full_device_qualified=false; hardware_co_residency_qualified=false; "
                 "public_integration=false; priority_qualified=false\n";
}

struct ReconvergentSdkTrace {
    CUfunction original{};
    std::array<CUfunction, 4> prepared{};
    std::array<std::array<unsigned int, 3>, 4> launches{};
    std::array<std::byte, 12> parameters{};
    unsigned int logical_blocks{977};
    unsigned int original_launches{};
    std::array<CUstream, 2> streams{};
    std::size_t stream_count{};
    std::optional<CUstream> expected_stream;
    bool has_mismatch{};
};
ReconvergentSdkTrace* reconvergent_trace{};
struct ReconvergentTraceScope {
    explicit ReconvergentTraceScope(ReconvergentSdkTrace* trace) {
        expect(!reconvergent_trace, "one reconvergent SDK trace owner");
        reconvergent_trace = trace;
    }
    ~ReconvergentTraceScope() {
        reconvergent_trace = nullptr;
    }
    ReconvergentTraceScope(const ReconvergentTraceScope&) = delete;
    ReconvergentTraceScope& operator=(const ReconvergentTraceScope&) = delete;
};
CUresult CUDAAPI reconvergent_stream_create(CUstream* stream, unsigned int flags) {
    const auto status = cuStreamCreate(stream, flags);
    if (status == CUDA_SUCCESS) {
        auto& trace = *reconvergent_trace;
        if (trace.stream_count < trace.streams.size())
            trace.streams[trace.stream_count++] = *stream;
        else
            trace.has_mismatch = true;
    }
    return status;
}
CUresult CUDAAPI reconvergent_function_get(CUfunction* function, CUmodule module,
                                           const char* name) {
    const auto status = cuModuleGetFunction(function, module, name);
    if (status == CUDA_SUCCESS) {
        auto& trace = *reconvergent_trace;
        constexpr std::array<std::string_view, 4> names{
            "reconvergent_loop__glimmer_ptb_block_static", "reconvergent_loop__glimmer_ptb_dynamic",
            "reconvergent_loop__glimmer_ptb_preemptive", "reconvergent_loop__glimmer_ptb_sliced"};
        if (std::string_view(name) == "reconvergent_loop")
            trace.original = *function;
        else if (const auto found = std::ranges::find(names, name); found != names.end())
            trace.prepared[static_cast<std::size_t>(found - names.begin())] = *function;
        else
            trace.has_mismatch = true;
    }
    return status;
}
// NOLINTBEGIN(bugprone-easily-swappable-parameters): exact CUDA Driver callback ABI.
CUresult CUDAAPI reconvergent_launch(CUfunction function, unsigned int grid_x, unsigned int grid_y,
                                     unsigned int grid_z, unsigned int block_x,
                                     unsigned int block_y, unsigned int block_z,
                                     unsigned int shared_bytes, CUstream stream, void** parameters,
                                     void** extra) {
    auto& trace = *reconvergent_trace;
    trace.has_mismatch |= grid_y != 1 || grid_z != 1 || block_x != 64 || block_y != 1 ||
                          block_z != 1 || shared_bytes != 0 || parameters || !extra;
    trace.has_mismatch |= trace.expected_stream && stream != *trace.expected_stream;
    if (extra && extra[0] == CU_LAUNCH_PARAM_BUFFER_POINTER &&
        extra[2] == CU_LAUNCH_PARAM_BUFFER_SIZE && extra[4] == CU_LAUNCH_PARAM_END && extra[1] &&
        extra[3]) {
        trace.has_mismatch |=
            *static_cast<const std::size_t*>(extra[3]) < trace.parameters.size() ||
            std::memcmp(extra[1], trace.parameters.data(), trace.parameters.size()) != 0;
    } else
        trace.has_mismatch = true;
    if (function == trace.original) {
        trace.has_mismatch |= grid_x != trace.logical_blocks;
        ++trace.original_launches;
    } else {
        constexpr std::array workers{80U, 160U, 320U};
        const auto selected = std::ranges::find(trace.prepared, function);
        const auto geometry = std::ranges::find(workers, grid_x);
        if (selected == trace.prepared.end() || geometry == workers.end())
            trace.has_mismatch = true;
        else
            ++trace.launches[static_cast<std::size_t>(selected - trace.prepared.begin())]
                            [static_cast<std::size_t>(geometry - workers.begin())];
    }
    return cuLaunchKernel(function, grid_x, grid_y, grid_z, block_x, block_y, block_z, shared_bytes,
                          stream, parameters, extra);
}
// NOLINTEND(bugprone-easily-swappable-parameters)

void validate_reconvergent_trace(const ReconvergentSdkTrace& trace) {
    expect(!trace.has_mismatch && trace.original &&
               std::ranges::all_of(trace.prepared,
                                   [](auto function) { return function != nullptr; }) &&
               (trace.original_launches == 1 || trace.original_launches == 2),
           "SDK observes genuine original and four correctly mapped private functions");
    auto selected = trace.original_launches - 1;
    constexpr std::array workers{80U, 160U, 320U};
    for (std::size_t kind = 0; kind < trace.prepared.size(); ++kind)
        for (std::size_t geometry = 0; geometry < workers.size(); ++geometry) {
            const auto count = trace.launches[kind][geometry];
            const auto expected =
                kind == 3 ? (977U + workers[geometry] - 1) / workers[geometry] : 1U;
            expect(count == expected || count == 2 * expected,
                   "every family/worker candidate actually executes its exact physical intervals");
            selected += count / expected - 1;
            std::cout << "Reconvergent SDK candidate: kind=" << kind + 1
                      << "; workers=" << workers[geometry] << "; physical_launches=" << count
                      << '\n';
        }
    expect(selected == 1,
           "only one final learned choice executes, without replaying application work");
    std::cout << "Reconvergent SDK original: physical_launches=" << trace.original_launches << '\n';
}

struct ReconvergentArguments {
    CUdeviceptr output{};
    std::uint32_t iterations{};
};
std::array<std::byte, 12> reconvergent_parameters(ReconvergentArguments arguments) {
    std::array<std::byte, 12> result{};
    std::memcpy(result.data(), &arguments.output, 8);
    std::memcpy(result.data() + 8, &arguments.iterations, 4);
    return result;
}

struct ReconvergentCheckpoint {
    unsigned int logical_blocks{};
    std::uint32_t iterations{};
    std::uint64_t completed_blocks{};
    unsigned int passes{1};
};
void check_reconvergent_checkpoint(const std::vector<std::uint32_t>& actual,
                                   ReconvergentCheckpoint checkpoint) {
    const auto [logical_blocks, iterations, completed_blocks, passes] = checkpoint;
    constexpr std::size_t guard = 17, threads = 64;
    constexpr std::uint32_t sentinel = 0x12345678;
    expect(completed_blocks <= logical_blocks &&
               actual.size() == std::size_t{logical_blocks} * threads + 2 * guard,
           "complete reconvergent checkpoint has exact geometry and guards");
    for (std::size_t index = 0; index < actual.size(); ++index) {
        auto expected = sentinel;
        if (index >= guard && index - guard < completed_blocks * threads) {
            const auto element = static_cast<std::uint32_t>(index - guard);
            const auto value =
                element + logical_blocks + iterations * (element % 2 == 0 ? 13U : 17U);
            expected += passes * value;
        }
        expect(actual[index] == expected,
               "every completed CTA executes once; unfinished CTAs and both guards stay untouched");
    }
}

void inspect_reconvergent_retreat_oracle() {
    constexpr std::size_t guard = 17, threads = 64, blocks = 2;
    constexpr std::uint32_t sentinel = 0x12345678;
    ReconvergentCheckpoint checkpoint{.logical_blocks = blocks, .iterations = 1024};
    std::vector<std::uint32_t> actual(blocks * threads + 2 * guard, sentinel);
    for (const auto completed : {0U, 1U, 2U})
        for (const auto passes : {1U, 2U}) {
            checkpoint.completed_blocks = completed;
            checkpoint.passes = passes;
            std::ranges::fill(actual, sentinel);
            for (std::size_t index = 0; index < std::size_t{completed} * threads; ++index) {
                const auto lane = index % threads;
                const auto per_iteration = lane % 2 == 0 ? 2 * 3 + 7 : 2 * 7 + 3;
                const auto value = static_cast<std::uint32_t>(index + blocks) +
                                   1024U * static_cast<std::uint32_t>(per_iteration);
                actual[guard + index] += passes * value;
            }
            check_reconvergent_checkpoint(actual, checkpoint);
        }
    checkpoint.completed_blocks = checkpoint.passes = 1;
    std::ranges::fill(actual.begin() + static_cast<std::ptrdiff_t>(guard + threads),
                      actual.end() - static_cast<std::ptrdiff_t>(guard), sentinel);
    // Restore a valid once-executed prefix independently of the checker.
    for (std::size_t index = 0; index < threads; ++index)
        actual[guard + index] = sentinel + static_cast<std::uint32_t>(index + blocks) +
                                1024U * (index % 2 == 0 ? 13U : 17U);
    check_reconvergent_checkpoint(actual, checkpoint);
    const auto rejects = [&](const auto& values, ReconvergentCheckpoint observed) {
        bool was_rejected = false;
        try {
            check_reconvergent_checkpoint(values, observed);
        } catch (const std::runtime_error&) {
            was_rejected = true;
        }
        expect(was_rejected, "oracle rejects fabricated progress, writes, geometry and guards");
    };
    for (const auto index :
         {std::size_t{0}, guard, guard + threads - 1, guard + threads, actual.size() - 1}) {
        auto changed = actual;
        ++changed[index];
        rejects(changed, checkpoint);
    }
    auto replay = actual;
    replay[guard] += replay[guard] - sentinel;
    rejects(replay, checkpoint);
    auto missing = actual;
    missing[guard] = sentinel;
    rejects(missing, checkpoint);
    auto short_output = actual;
    short_output.pop_back();
    rejects(short_output, checkpoint);
    auto forged = checkpoint;
    forged.completed_blocks = 3;
    rejects(actual, forged);
    forged = checkpoint;
    --forged.iterations;
    rejects(actual, forged);
    forged = checkpoint;
    ++forged.logical_blocks;
    rejects(actual, forged);
}

void run_session_reconvergent_retreat() {
    constexpr unsigned int low_blocks = 65536, threads = 64, high_blocks = 320;
    constexpr std::uint32_t low_iterations = 1024, high_iterations = 3;
    constexpr std::size_t guard = 17;
    constexpr std::uint32_t sentinel = 0x12345678;
    const auto success = [](const CudaSessionResult& result) {
        expect(result.disposition == CudaSessionDisposition::kComplete,
               "reconvergent resource operation completes");
        check(result.status, "reconvergent owned resource operation");
    };
    std::vector<std::uint32_t> native_low, native_high;
    for (const bool is_private : {false, true}) {
        ReconvergentSdkTrace trace;
        ReconvergentTraceScope trace_scope(&trace);
        auto dispatch = session_dispatch();
        dispatch.function_get = reconvergent_function_get;
        dispatch.execution.launch_kernel = reconvergent_launch;
        dispatch.stream_create = reconvergent_stream_create;
        CudaSessionLimits limits;
        limits.submissions = {2, 4096};
        CudaSessionPersistentConfig config;
        if (is_private) config.workers = kWorkers;
        CudaExecutionSession session({1, 1}, 1, dispatch, limits, config);
        success(session.initialize(0));
        CUcontext context{};
        check(cuCtxGetCurrent(&context), "capture actual fixture context");
        expect(context != nullptr, "session owns a live original context");
        const auto module = session.load_ptx(kReconvergentLoopPtx);
        success(module);
        const auto function = session.get_function(module.id, "reconvergent_loop");
        success(function);
        expect(function.parameter_bytes == 12 &&
                   session.snapshot().persistent_functions == (is_private ? 1U : 0U),
               "native/private preparation retains the unchanged two-argument original");
        std::vector<std::uint32_t> low_values(std::size_t{low_blocks} * threads + 2 * guard,
                                              sentinel);
        std::vector<std::uint32_t> high_values(std::size_t{high_blocks} * threads + 2 * guard,
                                               sentinel);
        const auto low_output = session.allocate(low_values.size() * sizeof(std::uint32_t));
        const auto high_output = session.allocate(high_values.size() * sizeof(std::uint32_t));
        const auto low_stream = session.create_stream(CU_STREAM_NON_BLOCKING);
        const auto high_stream = session.create_stream(CU_STREAM_NON_BLOCKING);
        for (const auto& result : {low_output, high_output, low_stream, high_stream})
            success(result);
        expect(trace.stream_count == 2 && trace.streams[0] && trace.streams[1] &&
                   trace.streams[0] != trace.streams[1],
               "logical low/high streams bind two distinct actual native streams");
        success(session.upload({low_output.id, 0}, std::as_bytes(std::span{low_values})));
        success(session.upload({high_output.id, 0}, std::as_bytes(std::span{high_values})));
        check(cuCtxSynchronize(), "finish initialization before nonblocking-stream execution");
        const auto read_stopped = [&](const CudaSessionResult& allocation, auto& values) {
            CUcontext current{};
            check(cuCtxGetCurrent(&current), "readback fixture context proof");
            expect(current == context && trace.stream_count == trace.streams.size(),
                   "read only the live session-owned allocation in its original context");
            for (const auto stream : trace.streams)
                check(cuStreamQuery(stream), "both actual compute streams are physically stopped");
            // Fixture-local observation of a physically stopped allocation.
            // Public cold_gate must still defer while logical low work is retained.
            check(cuMemcpyDtoH(values.data(), allocation.address,
                               values.size() * sizeof(std::uint32_t)),
                  "read every stopped output and guard without changing logical ownership");
        };
        const auto low_parameters =
            reconvergent_parameters({.output = low_output.address + guard * sizeof(std::uint32_t),
                                     .iterations = low_iterations});
        const auto high_parameters =
            reconvergent_parameters({.output = high_output.address + guard * sizeof(std::uint32_t),
                                     .iterations = high_iterations});
        const ReconvergentCheckpoint low_complete{.logical_blocks = low_blocks,
                                                  .iterations = low_iterations,
                                                  .completed_blocks = low_blocks};
        const ReconvergentCheckpoint high_complete{.logical_blocks = high_blocks,
                                                   .iterations = high_iterations,
                                                   .completed_blocks = high_blocks};
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{60};
        const auto wait = [&] {
            expect(std::chrono::steady_clock::now() < deadline,
                   "bounded reconvergent physical wait");
            std::this_thread::yield();
        };
        const auto issue = [&](std::uint64_t sequence, bool is_high) {
            const auto& parameters = is_high ? high_parameters : low_parameters;
            const auto blocks = is_high ? high_blocks : low_blocks;
            trace.parameters = parameters;
            trace.logical_blocks = blocks;
            trace.expected_stream = trace.streams[is_high ? 1 : 0];
            const auto accepted =
                session.accept_kernel(sequence, {.function = function.id,
                                                 .stream = is_high ? high_stream.id : low_stream.id,
                                                 .grid = {blocks, 1, 1},
                                                 .block = {threads, 1, 1},
                                                 .parameters = parameters});
            const auto frontier =
                session.capture_frontier(is_high ? high_stream.id : low_stream.id);
            expect(accepted.accepted() && frontier.status == CUDA_SUCCESS,
                   "one retained logical command and its frontier");
            const auto result = session.issue_next();
            expect(result.state == CudaExecutionIssueState::kIssued,
                   "actual reconvergent SDK command issues once");
            check(result.native_status, "reconvergent SDK launch");
            check(result.completion_status, "reconvergent trailing event");
            expect(session.acknowledge_result(accepted.receipt), "one original application reply");
            return frontier.frontier;
        };
        const auto drain = [&](const auto& frontier) {
            while (session.query_frontier(frontier) == CUDA_ERROR_NOT_READY) {
                check(session.poll().status, "reconvergent actual physical completion");
                wait();
            }
            check(session.query_frontier(frontier), "complete reconvergent logical frontier");
        };
        const auto low_frontier = issue(1, false);
        std::uint64_t partial{}, partial_launches{};
        if (is_private) {
            for (unsigned int attempt = 0; attempt < 10; ++attempt) {
                std::this_thread::sleep_for(std::chrono::milliseconds{1});
                check(session.request_retreat(1), "request real reconvergent retreat");
                while (!session.snapshot().submissions.paused) {
                    check(session.poll().status, "reconvergent incomplete physical stop");
                    wait();
                }
                const auto progress = session.persistent_progress(1);
                expect(progress && progress->completed_blocks < low_blocks &&
                           progress->total_blocks == low_blocks &&
                           progress->physical_stops == progress->physical_launches &&
                           session.query_frontier(low_frontier) == CUDA_ERROR_NOT_READY &&
                           session.snapshot().persistent_slots == 1,
                       "ACK cannot complete the logical frontier or release a physically stopped "
                       "slot");
                partial = progress->completed_blocks;
                partial_launches = progress->physical_launches;
                if (partial) break;
                const auto retry = session.resume(1);
                expect(retry.state == CudaExecutionIssueState::kIssued,
                       "a genuine zero-progress epoch resumes the same command");
                check(retry.native_status, "zero-progress continuation launch");
                check(retry.completion_status, "zero-progress continuation event");
            }
            expect(
                partial > 0 && partial < low_blocks,
                "mandatory positive incomplete compute, not a pending launch or finished kernel");
            read_stopped(low_output, low_values);
            auto checkpoint = low_complete;
            checkpoint.completed_blocks = partial;
            check_reconvergent_checkpoint(low_values, checkpoint);
        } else {
            expect(session.request_retreat(1) == CUDA_ERROR_NOT_SUPPORTED &&
                       !session.persistent_progress(1),
                   "original native control cannot fabricate a persistent retreat");
            drain(low_frontier);
            read_stopped(low_output, low_values);
            check_reconvergent_checkpoint(low_values, low_complete);
        }
        drain(issue(2, true));
        read_stopped(high_output, high_values);
        check_reconvergent_checkpoint(high_values, high_complete);
        if (is_private) {
            const auto progress = session.persistent_progress(1);
            expect(progress && progress->completed_blocks == partial &&
                       progress->physical_launches == partial_launches &&
                       session.snapshot().persistent_slots == 1 &&
                       session.snapshot().submissions.paused == 1 &&
                       session.query_frontier(low_frontier) == CUDA_ERROR_NOT_READY,
                   "useful high completion cannot advance or retire the paused low command");
            read_stopped(low_output, low_values);
            auto checkpoint = low_complete;
            checkpoint.completed_blocks = partial;
            check_reconvergent_checkpoint(low_values, checkpoint);
            trace.parameters = low_parameters;
            trace.logical_blocks = low_blocks;
            trace.expected_stream = trace.streams[0];
            const auto resumed = session.resume(1);
            expect(resumed.state == CudaExecutionIssueState::kIssued,
                   "retained reconvergent low command actually resumes");
            check(resumed.native_status, "reconvergent final continuation launch");
            check(resumed.completion_status, "reconvergent final continuation event");
            drain(low_frontier);
            read_stopped(low_output, low_values);
            check_reconvergent_checkpoint(low_values, low_complete);
        }
        expect(!session.persistent_progress(1) && session.snapshot().persistent_slots == 0,
               "only full physical completion retires the original low progress");
        drain(issue(3, true));
        read_stopped(high_output, high_values);
        read_stopped(low_output, low_values);
        check_reconvergent_checkpoint(low_values, low_complete);
        auto checkpoint = high_complete;
        checkpoint.passes = 2;
        check_reconvergent_checkpoint(high_values, checkpoint);
        expect(!trace.has_mismatch && trace.original &&
                   trace.original_launches == (is_private ? 0U : 3U) &&
                   session.snapshot().persistent_accepted == (is_private ? 3U : 0U) &&
                   session.snapshot().persistent_slots == 0,
               "actual SDK execution cannot fall back or replay a logical command");
        if (is_private) {
            expect(trace.prepared[2] && trace.launches[2][2] == partial_launches + 3 &&
                       low_values == native_low && high_values == native_high,
                   "all low epochs and both high commands match full native vectors exactly");
        } else {
            native_low = low_values;
            native_high = high_values;
        }
        for (std::size_t family = 0; family < trace.launches.size(); ++family)
            for (std::size_t geometry = 0; geometry < trace.launches[family].size(); ++geometry)
                if (!is_private || family != 2 || geometry != 2)
                    expect(trace.launches[family][geometry] == 0,
                           "no native fallback or alternate private family/geometry executes");
        success(session.shutdown());
        expect(!session.snapshot().has_context && session.snapshot().device_bytes == 0 &&
                   session.snapshot().private_host_bytes == 0,
               "all original/private contexts, allocations and progress resources are released");
        std::cout << "Reconvergent retreat verified: private=" << is_private
                  << "; workers=" << (is_private ? kWorkers : 0) << "; threads=" << threads
                  << "; low_blocks=" << low_blocks << "; low_iterations=" << low_iterations
                  << "; partial_blocks=" << partial
                  << "; physical_low_launches=" << (is_private ? partial_launches + 1 : 1)
                  << "; high_blocks=" << high_blocks << "; high_iterations=" << high_iterations
                  << "; high_commands=2; low_values=" << low_values.size()
                  << "; high_values=" << high_values.size()
                  << "; full_device_qualified=false; public_integration=false; "
                     "priority_qualified=false\n";
    }
}

void run_session_variants(bool should_profile = false, bool should_compare_families = false,
                          VariantModule module_kind = VariantModule::kStatelessPtx,
                          std::uint32_t loop_iterations = 3) {
    constexpr unsigned int logical_blocks = 977, guard = 17;
    const bool has_reconvergent_loop = module_kind == VariantModule::kReconvergentLoopPtx;
    const bool has_collective_loop =
        has_reconvergent_loop || module_kind == VariantModule::kUniformLoopPtx;
    const auto threads = has_reconvergent_loop ? 64U : kThreads;
    const auto elements = std::size_t{logical_blocks} * threads;
    constexpr std::uint32_t sentinel = 0x12345678;
    constexpr std::string_view stateless_source = R"ptx(
.version 7.0
.target sm_70
.address_size 64
.visible .entry add_once(.param .u64 output, .param .u32 guard) {
 .reg .b32 %r<9>;
 .reg .b64 %rd<4>;
 ld.param.u64 %rd1, [output]; ld.param.u32 %r1, [guard];
 mov.u32 %r2, %ctaid.x; mov.u32 %r3, %ntid.x; mov.u32 %r4, %tid.x;
 mad.lo.u32 %r5, %r2, %r3, %r4; add.u32 %r6, %r5, %r1;
 mul.wide.u32 %rd2, %r6, 4; add.u64 %rd3, %rd1, %rd2;
 ld.global.u32 %r7, [%rd3]; mov.u32 %r8, %nctaid.x;
 add.u32 %r7, %r7, %r5; add.u32 %r7, %r7, %r8;
 st.global.u32 [%rd3], %r7;
 ret;
}
)ptx";
    const bool has_module_state = module_kind == VariantModule::kStatefulPtx ||
                                  module_kind == VariantModule::kStatefulBinary ||
                                  module_kind == VariantModule::kStatefulCompanion;
    const auto source = has_module_state
                            ? std::string_view{reinterpret_cast<const char*>(kStatefulPtx),
                                               sizeof(kStatefulPtx) - 1}
                        : module_kind == VariantModule::kCompilerPtx   ? kCompilerPtx
                        : module_kind == VariantModule::kDeviceCallPtx ? kDeviceCallPtx
                        : module_kind == VariantModule::kAtomicPtx     ? kAtomicPtx
                        : module_kind == VariantModule::kWarpPtx       ? kWarpPtx
                        : has_reconvergent_loop                        ? kReconvergentLoopPtx
                        : has_collective_loop                          ? kUniformLoopPtx
                                                                       : stateless_source;
    const auto binary =
        module_kind == VariantModule::kStatefulCompanion
            ? std::as_bytes(std::span{kStatefulCompanion}).first(sizeof(kStatefulCompanion) - 1)
            : std::as_bytes(std::span{kStatefulFatbin}).first(sizeof(kStatefulFatbin) - 1);
    using Kind = glimmer::transform::PtxExecutionKind;
    const auto success = [](const CudaSessionResult& result) {
        expect(result.disposition == CudaSessionDisposition::kComplete,
               "variant resource operation must complete");
        check(result.status, "variant resource operation");
    };
    std::vector kinds{Kind::kStatic, Kind::kBlockStatic, Kind::kDynamic, Kind::kPreemptive,
                      Kind::kSliced};
    if (module_kind == VariantModule::kWarpPtx || has_collective_loop)
        std::erase(kinds, Kind::kStatic);
    const auto private_candidates = static_cast<unsigned int>(kinds.size()) * 3U;
    const auto configured = should_compare_families ? std::vector{Kind::kPreemptive} : kinds;
    for (const auto kind : configured) {
        CudaSessionLimits limits;
        limits.submissions = {2, 4096};
        std::shared_ptr<CudaProfileCache> profile;
        if (should_profile) {
            CudaProfileConfig config;
            config.warmup_calls = 0;
            config.samples_per_candidate = 1;
            if (should_compare_families) config.kinds = kinds;
            profile = std::make_shared<CudaProfileCache>(std::move(config));
        }
        ReconvergentSdkTrace trace;
        ReconvergentTraceScope trace_scope(has_reconvergent_loop ? &trace : nullptr);
        auto dispatch = session_dispatch();
        if (has_reconvergent_loop) {
            dispatch.function_get = reconvergent_function_get;
            dispatch.execution.launch_kernel = reconvergent_launch;
        }
        CudaExecutionSession session({1, 1}, 1, dispatch, limits, {kWorkers, false, kind}, {},
                                     profile);
        success(session.initialize(0));
        const auto private_bytes = session.snapshot().device_bytes;
        const auto static_bytes = glimmer::transform::inspect_ptx_static_storage(source);
        expect(static_bytes.has_value(), "compiled source has a bounded static-data reservation");
        const auto module = module_kind == VariantModule::kStatefulBinary ||
                                    module_kind == VariantModule::kStatefulCompanion
                                ? session.load_binary(binary)
                                : session.load_ptx(source);
        success(module);
        expect(session.snapshot().device_bytes == private_bytes + *static_bytes,
               "unqueried module variables are already reserved before symbol publication");
        CudaSessionResult bias, invocations;
        if (has_module_state) {
            // Publish symbols before function preparation to catch lazy module replacement.
            bias = session.get_global(module.id, "bias");
            invocations = session.get_global(module.id, "invocations");
            success(bias);
            success(invocations);
            expect(bias.parameter_bytes == 4 && invocations.parameter_bytes == 4,
                   "original symbol sizes survive module preparation");
            std::uint32_t initial{};
            success(session.download({bias.id, 0}, std::as_writable_bytes(std::span{&initial, 1})));
            expect(initial == 7, "constant native initializer is preserved");
            success(session.download({invocations.id, 0},
                                     std::as_writable_bytes(std::span{&initial, 1})));
            expect(initial == 13, "global native initializer is preserved");
        }
        const auto function =
            session.get_function(module.id, has_reconvergent_loop ? "reconvergent_loop"
                                            : has_collective_loop ? "uniform_loop"
                                                                  : "add_once");
        success(function);
        expect(function.parameter_bytes == 12 && session.snapshot().persistent_functions == 1,
               "actual JIT and SDK metadata select the requested private family");
        std::vector<std::uint32_t> actual(elements + std::size_t{2} * guard, sentinel);
        const auto output = session.allocate(actual.size() * sizeof(std::uint32_t));
        success(output);
        success(session.upload({output.id, 0}, std::as_bytes(std::span{actual})));
        std::array<std::byte, 12> parameters{};
        const auto output_address =
            output.address + (has_collective_loop ? guard * sizeof(std::uint32_t) : 0);
        const auto argument = has_collective_loop ? loop_iterations : guard;
        std::memcpy(parameters.data(), &output_address, 8);
        std::memcpy(parameters.data() + 8, &argument, 4);
        trace.parameters = parameters;
        const auto original = parameters;
        const auto passes = should_compare_families ? private_candidates + 2U
                            : should_profile        ? 5U
                                                    : 2U;
        std::uint32_t accumulated_bias{};
        for (std::uint64_t pass = 1; pass <= passes; ++pass) {
            if (has_module_state) {
                const auto value = static_cast<std::uint32_t>(3 * pass + 1);
                success(session.upload({bias.id, 0}, std::as_bytes(std::span{&value, 1})));
                accumulated_bias += value;
            }
            const auto samples_before = profile ? profile->snapshot().samples : 0;
            const auto accepted = session.accept_kernel(pass, {.function = function.id,
                                                               .grid = {logical_blocks, 1, 1},
                                                               .block = {threads, 1, 1},
                                                               .parameters = parameters});
            const auto frontier = session.capture_frontier();
            expect(accepted.accepted() && frontier.status == CUDA_SUCCESS,
                   "accept one logical application launch");
            const auto issued = session.issue_next();
            check(issued.native_status, "actual selected-family launch");
            check(issued.completion_status, "actual selected-family fence");
            expect(issued.state == CudaExecutionIssueState::kIssued &&
                       session.acknowledge_result(accepted.receipt),
                   "one actual SDK result per logical launch");
            unsigned int slices = 1;
            std::uint64_t slice_width{};
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{20};
            while (session.snapshot().submissions.outstanding) {
                check(session.poll().status, "selected-family physical stop");
                if (session.snapshot().submissions.paused) {
                    const auto progress = session.persistent_progress(pass);
                    if (progress && !slice_width) slice_width = progress->completed_blocks;
                    expect((should_compare_families || kind == Kind::kSliced) && progress &&
                               slice_width &&
                               progress->completed_blocks == std::uint64_t{slices} * slice_width &&
                               (!profile || profile->snapshot().samples == samples_before) &&
                               session.query_frontier(frontier.frontier) == CUDA_ERROR_NOT_READY,
                           "slice stop cannot satisfy the full logical frontier");
                    const auto resumed = session.resume(pass);
                    check(resumed.native_status, "selected-family slice continuation");
                    check(resumed.completion_status, "selected-family continuation fence");
                    expect(resumed.state == CudaExecutionIssueState::kIssued,
                           "physical continuation is actually issued");
                    ++slices;
                }
                expect(std::chrono::steady_clock::now() < deadline,
                       "bounded actual variant completion");
                std::this_thread::yield();
            }
            const auto expected_slices =
                slice_width ? (logical_blocks + slice_width - 1) / slice_width : 1;
            expect(slices == expected_slices &&
                       (should_profile || slices == (kind == Kind::kSliced ? 4U : 1U)) &&
                       parameters == original &&
                       session.query_frontier(frontier.frontier) == CUDA_SUCCESS,
                   "all physical intervals complete exactly one unchanged logical launch");
            success(session.download({output.id, 0}, std::as_writable_bytes(std::span{actual})));
            for (std::size_t index = 0; index < actual.size(); ++index) {
                const auto expected =
                    index < guard || index >= guard + elements
                        ? sentinel
                        : sentinel +
                              static_cast<std::uint32_t>(pass * (index - guard + logical_blocks)) +
                              accumulated_bias +
                              (has_collective_loop
                                   ? static_cast<std::uint32_t>(
                                         pass * loop_iterations *
                                         (has_reconvergent_loop
                                              ? ((index - guard) % 2 == 0 ? 13U : 17U)
                                              : (((index - guard) % threads) ^ 1U)))
                                   : 0);
                expect(actual[index] == expected,
                       "every output and both guards match the non-idempotent logical-grid oracle");
            }
            if (has_module_state) {
                std::uint32_t observed{};
                success(session.download({invocations.id, 0},
                                         std::as_writable_bytes(std::span{&observed, 1})));
                expect(observed == 13 + pass,
                       "original and every private family increment the same module state once");
                const auto current_bias = session.get_global(module.id, "bias");
                const auto current_invocations = session.get_global(module.id, "invocations");
                success(current_bias);
                success(current_invocations);
                expect(
                    current_bias.id == bias.id && current_bias.address == bias.address &&
                        current_invocations.id == invocations.id &&
                        current_invocations.address == invocations.address,
                    "published symbol identities remain stable across original/private selection");
            }
        }
        expect((should_profile ? session.snapshot().persistent_accepted >=
                                     (should_compare_families ? private_candidates : 3U)
                               : session.snapshot().persistent_accepted == 2) &&
                   session.snapshot().persistent_slots == 0,
               "all required private candidates execute and retire their leases");
        if (profile)
            expect(profile->snapshot().samples == passes - 1 &&
                       profile->snapshot().trained_cases == 1 && profile->snapshot().hits == 1 &&
                       profile->snapshot().rejected_samples == 0,
                   "one complete sample per candidate, then one measured selection without replay");
        if (has_reconvergent_loop) validate_reconvergent_trace(trace);
        success(session.unload_module(module.id));
        expect(
            session.snapshot().device_bytes ==
                private_bytes + actual.size() * sizeof(std::uint32_t),
            "one proven unload refunds the entire module reservation, not the output allocation");
        if (has_module_state)
            expect(!session.device_memory({bias.id, 0}, 1) &&
                       !session.device_memory({invocations.id, 0}, 1),
                   "one module unload retires both discovered symbol ranges");
        success(session.shutdown());
        std::cout << "Session variant verified: kind=" << static_cast<unsigned int>(kind)
                  << "; workers=" << kWorkers << "; exact_values=" << actual.size() * passes
                  << "; profile=" << should_profile << "; cross_family=" << should_compare_families
                  << "; module_kind=" << static_cast<unsigned int>(module_kind)
                  << "; threads=" << threads << "; loop_iterations=" << loop_iterations
                  << "; full_device_qualified=false; public_integration=false; "
                     "priority_qualified=false\n";
    }
}

struct AggregateSdkTrace {
    using Kind = glimmer::transform::PtxExecutionKind;
    Kind kind{};
    bool is_private{}, has_mismatch{};
    CUfunction original{}, prepared{};
    std::array<std::size_t, 9> offsets{0, 8, 16, 48, 52, 56, 60, 64, 72};
    std::array<std::size_t, 9> sizes{4, 8, 32, 4, 4, 4, 4, 8, 8};
    std::array<unsigned int, 9> original_queries{}, prepared_queries{};
    std::array<std::byte, 52> expected_parameters{};
    unsigned int original_launches{}, prepared_launches{}, expected_blocks{};
    std::size_t arity{}, parameter_bytes{};

    explicit AggregateSdkTrace(std::optional<Kind> selected)
        : kind(selected.value_or(Kind::kStatic)), is_private(selected.has_value()) {
        arity = !is_private                  ? 4
                : kind == Kind::kStatic      ? 5
                : kind == Kind::kBlockStatic ? 7
                : kind == Kind::kDynamic     ? 8
                                             : 9;
        if (kind == Kind::kSliced) {
            offsets[8] = 68;
            sizes[7] = sizes[8] = 4;
        }
        parameter_bytes = offsets[arity - 1] + sizes[arity - 1];
    }
};
AggregateSdkTrace* aggregate_trace{};
struct AggregateTraceScope {
    explicit AggregateTraceScope(AggregateSdkTrace& trace) {
        expect(!aggregate_trace, "one aggregate SDK trace owner");
        aggregate_trace = &trace;
    }
    ~AggregateTraceScope() {
        aggregate_trace = nullptr;
    }
    AggregateTraceScope(const AggregateTraceScope&) = delete;
    AggregateTraceScope& operator=(const AggregateTraceScope&) = delete;
};

CUresult CUDAAPI aggregate_function_get(CUfunction* function, CUmodule module, const char* name) {
    const auto status = cuModuleGetFunction(function, module, name);
    if (status == CUDA_SUCCESS) {
        auto& trace = *aggregate_trace;
        constexpr std::array names{
            "aggregate_once__glimmer_ptb", "aggregate_once__glimmer_ptb_block_static",
            "aggregate_once__glimmer_ptb_dynamic", "aggregate_once__glimmer_ptb_preemptive",
            "aggregate_once__glimmer_ptb_sliced"};
        if (std::string_view(name) == "aggregate_once")
            trace.original = *function;
        else {
            trace.has_mismatch |=
                !trace.is_private ||
                std::string_view(name) != names[static_cast<std::size_t>(trace.kind)];
            trace.prepared = *function;
        }
    }
    return status;
}
// NOLINTBEGIN(bugprone-easily-swappable-parameters): exact CUDA Driver callback ABIs.
CUresult CUDAAPI aggregate_parameter_info(CUfunction function, std::size_t index,
                                          std::size_t* offset, std::size_t* bytes) {
    const auto status = cuFuncGetParamInfo(function, index, offset, bytes);
    if (status == CUDA_SUCCESS) {
        auto& trace = *aggregate_trace;
        const bool original = function == trace.original;
        const auto count = original ? 4 : trace.arity;
        if (index >= count || (!original && function != trace.prepared))
            trace.has_mismatch = true;
        else {
            trace.has_mismatch |= *offset != trace.offsets[index] || *bytes != trace.sizes[index];
            ++(original ? trace.original_queries[index] : trace.prepared_queries[index]);
        }
    }
    return status;
}
CUresult CUDAAPI aggregate_launch(CUfunction function, unsigned int grid_x, unsigned int grid_y,
                                  unsigned int grid_z, unsigned int block_x, unsigned int block_y,
                                  unsigned int block_z, unsigned int shared_bytes, CUstream stream,
                                  void** parameters, void** extra) {
    auto& trace = *aggregate_trace;
    const bool original = function == trace.original;
    trace.has_mismatch |=
        original == trace.is_private || (!original && function != trace.prepared) ||
        grid_x != trace.expected_blocks || grid_y != 1 || grid_z != 1 || block_x != kThreads ||
        block_y != 1 || block_z != 1 || shared_bytes != 0 || parameters || !extra;
    if (extra && extra[0] == CU_LAUNCH_PARAM_BUFFER_POINTER &&
        extra[2] == CU_LAUNCH_PARAM_BUFFER_SIZE && extra[4] == CU_LAUNCH_PARAM_END && extra[1] &&
        extra[3]) {
        trace.has_mismatch |= *static_cast<const std::size_t*>(extra[3]) != trace.parameter_bytes ||
                              std::memcmp(extra[1], trace.expected_parameters.data(),
                                          trace.expected_parameters.size()) != 0;
    } else
        trace.has_mismatch = true;
    ++(original ? trace.original_launches : trace.prepared_launches);
    return cuLaunchKernel(function, grid_x, grid_y, grid_z, block_x, block_y, block_z, shared_bytes,
                          stream, parameters, extra);
}
// NOLINTEND(bugprone-easily-swappable-parameters)

constexpr std::array<std::optional<AggregateSdkTrace::Kind>, 6> kAggregateModes{
    std::nullopt,
    AggregateSdkTrace::Kind::kStatic,
    AggregateSdkTrace::Kind::kBlockStatic,
    AggregateSdkTrace::Kind::kDynamic,
    AggregateSdkTrace::Kind::kPreemptive,
    AggregateSdkTrace::Kind::kSliced};

CudaSessionPersistentConfig aggregate_persistent_config(
    std::optional<AggregateSdkTrace::Kind> mode) {
    CudaSessionPersistentConfig config{};
    if (mode) config = {kWorkers, false, *mode};
    return config;
}

void inspect_aggregate_input() {
    for (const auto mode : kAggregateModes) {
        const auto config = aggregate_persistent_config(mode);
        expect(config.workers == (mode ? 320U : 0U) && !config.replace_sgemm &&
                   config.kind == mode.value_or(AggregateSdkTrace::Kind::kPreemptive),
               "native uses default family; each private family has explicit 320 workers");
        CudaSessionLimits limits;
        limits.submissions = {2, 4096};
        // Only construct and inspect CPU ownership. Never initialize the SDK;
        // the uninitialized session destructor also takes the no-context path.
        CudaExecutionSession session({1, 1}, 1, session_dispatch(), limits, config);
        const auto state = session.snapshot();
        expect(!state.has_context && !state.is_quarantined && !state.device_bytes &&
                   !state.image_bytes && !state.allocations && !state.modules && !state.streams &&
                   !state.functions && !state.private_host_bytes && !state.host_memory_bytes &&
                   !state.host_allocations && !state.persistent_slots &&
                   !state.persistent_functions && !state.persistent_accepted &&
                   !state.submissions.outstanding && !state.has_ready_work && !state.blas_handles &&
                   !state.lt_handles && !state.random_handles && !state.dnn_handles,
               "aggregate configuration inspection creates no native resource or submission");
    }
    std::cout << "Aggregate input configuration verified: modes=6; GPU_initialized=false\n";
}

void run_session_aggregate() {
    using Kind = glimmer::transform::PtxExecutionKind;
    constexpr unsigned int logical_blocks = 1280, guard = 32;
    constexpr std::uint32_t elements = logical_blocks * kThreads - 7;
    constexpr std::uint32_t sentinel = 0x12345678, prefix = 19, suffix = 23;
    constexpr std::array<std::uint32_t, 8> values{3, 5, 7, 11, 13, 17, logical_blocks, elements};
    constexpr auto sum = [values] {
        std::uint32_t value{};
        for (const auto word : values) value += word;
        return value;
    }();
    const auto success = [](const CudaSessionResult& result) {
        expect(result.disposition == CudaSessionDisposition::kComplete,
               "aggregate owned operation completes");
        check(result.status, "aggregate owned result");
    };
    for (const auto mode : kAggregateModes) {
        AggregateSdkTrace trace(mode);
        const AggregateTraceScope tracing(trace);
        trace.expected_blocks = mode ? kWorkers : logical_blocks;
        auto dispatch = session_dispatch();
        dispatch.function_get = aggregate_function_get;
        dispatch.parameter_info = aggregate_parameter_info;
        dispatch.execution.launch_kernel = aggregate_launch;
        CudaSessionLimits limits;
        limits.submissions = {2, 4096};
        CudaExecutionSession session({1, 1}, 1, dispatch, limits,
                                     aggregate_persistent_config(mode));
        success(session.initialize(0));
        const auto private_bytes = session.snapshot().device_bytes;
        const auto module = session.load_ptx(kAggregatePtx);
        success(module);
        const auto function = session.get_function(module.id, "aggregate_once");
        success(function);
        expect(
            function.parameter_bytes == 52 && function.arguments.size() == 4 &&
                session.snapshot().persistent_functions == (mode ? 1U : 0U) && trace.original &&
                (!mode || (trace.prepared && trace.prepared != trace.original)),
            "actual native metadata and exactly one selected private family preserve opaque ABI");
        for (std::size_t index = 0; index < trace.original_queries.size(); ++index) {
            expect(
                trace.original_queries[index] == (index < 4 ? 1U : 0U) &&
                    trace.prepared_queries[index] == (mode && index < trace.arity ? 1U : 0U),
                "every native and private parameter is checked once against independent offsets");
            if (index < 4)
                expect(function.arguments[index].offset == trace.offsets[index] &&
                           function.arguments[index].bytes == trace.sizes[index],
                       "public SDK layout is 0/8/16/48 with an opaque 32-byte aggregate");
        }
        std::vector<std::uint32_t> actual(elements + 2 * guard, sentinel);
        const auto output = session.allocate(actual.size() * sizeof(std::uint32_t));
        success(output);
        success(session.upload({output.id, 0}, std::as_bytes(std::span{actual})));
        check(cuCtxSynchronize(), "complete all aggregate fixture upload before compute");
        const auto address = output.address + guard * sizeof(std::uint32_t);
        auto& expected = trace.expected_parameters;
        expected.fill(std::byte{0xa5});
        std::memcpy(expected.data(), &prefix, 4);
        std::memcpy(expected.data() + 8, &address, 8);
        std::memcpy(expected.data() + 16, values.data(), 32);
        std::memcpy(expected.data() + 48, &suffix, 4);
        for (std::uint64_t pass = 1; pass <= 2; ++pass) {
            auto parameters = expected;
            const auto accepted = session.accept_kernel(pass, {.function = function.id,
                                                               .grid = {logical_blocks, 1, 1},
                                                               .block = {kThreads, 1, 1},
                                                               .parameters = parameters});
            const auto frontier = session.capture_frontier();
            expect(accepted.accepted() && frontier.status == CUDA_SUCCESS &&
                       parameters == expected &&
                       session.snapshot().persistent_accepted == (mode ? pass : 0U),
                   "immutable aggregate admission selects actual native or requested private path");
            parameters.fill(std::byte{0xcc});
            const auto issued = session.issue_next();
            expect(issued.state == CudaExecutionIssueState::kIssued, "one initial physical launch");
            check(issued.native_status, "aggregate original launch status");
            check(issued.completion_status, "aggregate initial physical completion setup");
            const auto receipt = session.host_result(accepted.receipt);
            expect(receipt && receipt->status == CUDA_SUCCESS &&
                       session.acknowledge_result(accepted.receipt) &&
                       !session.acknowledge_result(accepted.receipt) &&
                       session.snapshot().submissions.outstanding == 1 &&
                       session.snapshot().persistent_slots == (mode ? 1U : 0U),
                   "one original host reply never substitutes for complete physical retirement");
            unsigned int launches = 1;
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{20};
            while (session.snapshot().submissions.outstanding) {
                check(session.poll().status, "aggregate physical stop");
                if (session.snapshot().submissions.paused) {
                    const auto progress = session.persistent_progress(pass);
                    expect(mode == Kind::kSliced && progress && launches < 4 &&
                               progress->completed_blocks == launches * std::uint64_t{kWorkers} &&
                               progress->total_blocks == logical_blocks &&
                               progress->physical_launches == launches &&
                               progress->physical_stops == launches &&
                               session.query_frontier(frontier.frontier) == CUDA_ERROR_NOT_READY,
                           "incomplete slice keeps the same aggregate command and bounded counter");
                    const auto resumed = session.resume(pass);
                    expect(resumed.state == CudaExecutionIssueState::kIssued,
                           "aggregate slice continuation is a real physical launch");
                    check(resumed.native_status, "aggregate continuation status");
                    check(resumed.completion_status, "aggregate continuation fence");
                    ++launches;
                }
                expect(std::chrono::steady_clock::now() < deadline, "bounded aggregate retirement");
                std::this_thread::yield();
            }
            expect(
                launches == (mode == Kind::kSliced ? 4U : 1U) &&
                    session.query_frontier(frontier.frontier) == CUDA_SUCCESS &&
                    !session.snapshot().persistent_slots && !trace.has_mismatch &&
                    std::ranges::all_of(parameters,
                                        [](auto byte) { return byte == std::byte{0xcc}; }),
                "all physical launches preserve captured aggregate bytes and retire exactly once");
            success(session.download({output.id, 0}, std::as_writable_bytes(std::span{actual})));
            for (std::size_t index = 0; index < actual.size(); ++index) {
                const auto wanted =
                    index < guard || index >= guard + elements
                        ? sentinel
                        : sentinel + static_cast<std::uint32_t>(
                                         pass * (index - guard + sum + prefix + suffix));
                expect(actual[index] == wanted,
                       "every value, final partial CTA and both guards match independent "
                       "two-launch oracle");
            }
        }
        expect(
            !trace.has_mismatch && trace.original_launches == (mode ? 0U : 2U) &&
                trace.prepared_launches == (!mode                   ? 0U
                                            : mode == Kind::kSliced ? 8U
                                                                    : 2U),
            "exact function identity and launch count exclude silent original fallback or replay");
        success(session.release(output.id));
        success(session.unload_module(module.id));
        expect(session.snapshot().device_bytes == private_bytes &&
                   !session.snapshot().submissions.outstanding &&
                   !session.snapshot().persistent_slots,
               "aggregate allocation/module owners release only after proven complete frontiers");
        success(session.shutdown());
        expect(!session.snapshot().has_context && !session.snapshot().device_bytes,
               "aggregate context shutdown settles all remaining private ownership");
        std::cout << "Aggregate session verified: private=" << mode.has_value()
                  << "; kind=" << static_cast<unsigned int>(trace.kind)
                  << "; workers=" << (mode ? kWorkers : 0U)
                  << "; public_parameter_bytes=52; exact_values=" << actual.size() * 2
                  << "; physical_launches=" << trace.original_launches + trace.prepared_launches
                  << "; full_device_qualified=false; public_integration=false; "
                     "priority_qualified=false; performance_qualified=false\n";
    }
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--inspect-state") {
            inspect_module_state_inputs();
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--inspect-compiler") {
            inspect_compiler_input();
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--inspect-aggregate") {
            inspect_aggregate_input();
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--inspect-shared-domain") {
            inspect_shared_domain_input();
            return EXIT_SUCCESS;
        }
        const auto initialized = cuInit(0);
        if (initialized == CUDA_ERROR_NO_DEVICE) return 77;
        check(initialized, "initialize CUDA");
        if (argc == 2 && std::string_view(argv[1]) == "--session-aggregate") {
            run_session_aggregate();
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--session-owned") {
            run_session_owned();
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--session-retarget") {
            run_session_owned(true);
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--session-shared-domain") {
            run_shared_domain_sessions();
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--session-shared-configuration") {
            run_shared_domain_sessions(true);
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--session-exact-command") {
            run_shared_domain_sessions(true, true);
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--session-variants") {
            run_session_variants();
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--session-profile") {
            run_session_variants(true);
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--session-profile-families") {
            run_session_variants(true, true);
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--session-state") {
            run_session_variants(true, true, VariantModule::kStatefulPtx);
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--session-state-binary") {
            run_session_variants(true, true, VariantModule::kStatefulBinary);
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--session-state-companion") {
            run_session_variants(true, true, VariantModule::kStatefulCompanion);
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--session-compiler") {
            run_session_variants(true, true, VariantModule::kCompilerPtx);
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--session-calls") {
            run_session_variants(true, true, VariantModule::kDeviceCallPtx);
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--session-atomic") {
            run_session_variants(true, true, VariantModule::kAtomicPtx);
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--session-warp") {
            run_session_variants(true, true, VariantModule::kWarpPtx);
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--session-uniform-loop") {
            run_session_variants(true, true, VariantModule::kUniformLoopPtx);
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--session-uniform-zero") {
            run_session_variants(true, true, VariantModule::kUniformLoopPtx, 0);
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--session-reconvergent-loop") {
            run_session_variants(true, true, VariantModule::kReconvergentLoopPtx);
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--session-reconvergent-zero") {
            run_session_variants(true, true, VariantModule::kReconvergentLoopPtx, 0);
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--session-reconvergent-retreat") {
            run_session_reconvergent_retreat();
            return EXIT_SUCCESS;
        }
        expect(argc == 1, "unknown fixture mode");
        auto resources = std::make_shared<Resources>();
        const auto cleanup_ok = resources->cleanup_ok;
        run(resources);
        resources.reset();
        expect(*cleanup_ok, "all owned GPU resources released without error");
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
