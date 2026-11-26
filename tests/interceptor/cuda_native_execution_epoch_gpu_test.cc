#include "internal/cuda_native_execution_epoch.h"

#include <array>
#include <atomic>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {
using namespace glimmer::interceptor;
using namespace glimmer::control;
constexpr NativeEpochExtent kExtent{8, 2, 2, 1'000'000};
constexpr NativeEpochScope kScope{11, 12, 13};
constexpr char kProgram[] = R"ptx(
.version 7.0
.target sm_70
.address_size 64
.visible .entry increment(.param .u64 output, .param .u64 wait_ns) {
 .reg .b64 %pointer, %delay, %start, %now, %elapsed;
 .reg .b32 %value;
 .reg .pred %waiting;
 ld.param.u64 %pointer, [output];
 ld.param.u64 %delay, [wait_ns];
 mov.u64 %start, %globaltimer;
WAIT:
 mov.u64 %now, %globaltimer;
 sub.u64 %elapsed, %now, %start;
 setp.lt.u64 %waiting, %elapsed, %delay;
 @%waiting bra WAIT;
 ld.global.u32 %value, [%pointer];
 add.u32 %value, %value, 1;
 st.global.u32 [%pointer], %value;
 ret;
}
)ptx";
bool has_cleanup_failure{};
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void check(CUresult status, const char* message) {
    if (status != CUDA_SUCCESS)
        throw std::runtime_error(std::string(message) + ": " + std::to_string(status));
}
void cleanup(CUresult status) noexcept {
    has_cleanup_failure |= status != CUDA_SUCCESS;
}
template <typename Predicate>
void until(Predicate predicate) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
    while (!predicate()) {
        require(std::chrono::steady_clock::now() < deadline, "bounded actual GPU progress");
        std::this_thread::yield();
    }
}
template <typename Value>
Value result(std::future<Value> future) {
    require(future.valid(), "accepted actual control request");
    require(future.wait_for(std::chrono::seconds{2}) == std::future_status::ready,
            "bounded actual control response");
    return future.get();
}
struct Resources {
    CUcontext context{};
    CUstream low{};
    CUstream high{};
    CUmodule module{};
    CUfunction increment{};
    CUdeviceptr allocation{};
    CUevent observer{};
    ~Resources() {
        if (observer) cleanup(cuEventDestroy(observer));
        if (allocation) cleanup(cuMemFree(allocation));
        if (module) cleanup(cuModuleUnload(module));
        if (high) cleanup(cuStreamDestroy(high));
        if (low) cleanup(cuStreamDestroy(low));
        if (context) cleanup(cuCtxDestroy(context));
    }
};
CudaNativeEpochDispatch dispatch() {
    return {cuCtxGetCurrent,     cuCtxPushCurrent, cuCtxPopCurrent, cuStreamGetCtx,
            cuStreamIsCapturing, cuEventCreate,    cuEventRecord,   cuEventQuery,
            cuEventDestroy,      nullptr,          nullptr};
}
void launch(Resources& resources, CUstream stream, unsigned long long duration) {
    auto pointer = resources.allocation + sizeof(unsigned int);
    void* arguments[]{&pointer, &duration};
    check(cuLaunchKernel(resources.increment, 1, 1, 1, 1, 1, 1, 0, stream, arguments, nullptr),
          "original launch");
}
void wait_event(CUevent event) {
    until([&] {
        const auto status = cuEventQuery(event);
        if (status == CUDA_ERROR_NOT_READY) return false;
        check(status, "independent physical event query");
        return true;
    });
}
void wait_ready(CudaNativeExecutionEpoch& owner) {
    until([&] {
        const auto state = owner.snapshot();
        require(state.phase != CudaNativeEpochPhase::kRetained &&
                    state.phase != CudaNativeEpochPhase::kRejected,
                "actual owner preparation/authorization succeeds");
        return state.phase == CudaNativeEpochPhase::kReady;
    });
}
void finish(CudaNativeExecutionEpoch& owner, unsigned int calls) {
    until([&] {
        const auto state = owner.snapshot();
        require(state.phase != CudaNativeEpochPhase::kRetained, "actual proof/report succeeds");
        return state.phase == CudaNativeEpochPhase::kComplete;
    });
    owner.shutdown();
    const auto state = owner.snapshot();
    require(state.status == CUDA_SUCCESS && state.execution && !state.execution->has_ownership &&
                state.execution->issued_calls == calls && state.execution->completed_calls == calls,
            "exact entered originals physically completed and acknowledged");
    require(state.created_events == state.destroyed_events, "actual private event cleanup");
}
void execute(Resources& resources, CudaNativeExecutionEpoch& owner,
             unsigned long long duration = 0) {
    std::optional<NativeEpochIssue> call;
    until([&] {
        call = owner.begin_call();
        return call.has_value();
    });
    CUcontext before{};
    check(cuCtxGetCurrent(&before), "original caller context before");
    launch(resources, resources.low, duration);
    CUcontext after{};
    check(cuCtxGetCurrent(&after), "original caller context after");
    require(before == resources.context && after == before,
            "native call remains on unchanged caller context");
    require(owner.original_returned(*call, NativeEpochCallOutcome::kSuccess),
            "original call returns once");
}
void run_lifetime(bool valid_only) {
    Resources resources;
    CUdevice device{};
    check(cuDeviceGet(&device, 0), "device");
    check(cuCtxCreate(&resources.context, 0, device), "context");
    check(cuStreamCreate(&resources.low, CU_STREAM_NON_BLOCKING), "low stream");
    check(cuStreamCreate(&resources.high, CU_STREAM_NON_BLOCKING), "high stream");
    check(cuModuleLoadData(&resources.module, kProgram), "original module");
    check(cuModuleGetFunction(&resources.increment, resources.module, "increment"),
          "original function");
    check(cuMemAlloc(&resources.allocation, 3 * sizeof(unsigned int)), "guarded allocation");
    std::array<unsigned int, 3> data{0x12345678U, 0, 0x76543210U};
    check(cuMemcpyHtoD(resources.allocation, data.data(), sizeof(data)), "initial data");
    check(cuEventCreate(&resources.observer, CU_EVENT_DISABLE_TIMING), "independent observer");
    LaunchGateOptions options;
    options.max_concurrent_launches = 1;
    options.scheduling_policy = glimmer::core::SchedulingPolicy::kPriority;
    auto gate = std::make_shared<AsyncCooperativeLaunchGate>(options, 64, kExtent);
    // Pre-existing native work owns its distinct ordinary unit grant. Its
    // actual trailing proof, not the epoch's preparation, permits that release.
    const auto predecessor = result(gate->submit_and_claim());
    require(predecessor.claim.status == CooperativeClaimStatus::kGranted, "predecessor unit grant");
    launch(resources, resources.low, 100'000'000);
    check(cuEventRecord(resources.observer, resources.low), "predecessor observer record");
    {
        CudaNativeExecutionEpoch owner(resources.context, resources.low, kScope, {kExtent, 0}, gate,
                                       dispatch());
        require(cuEventQuery(resources.observer) == CUDA_ERROR_NOT_READY,
                "actual predecessor still running");
        require(!owner.begin_call() && owner.snapshot().grant.execution.task_id == 0,
                "no original/epoch grant before actual predecessor proof");
        wait_event(resources.observer);
        require(result(gate->complete(predecessor.claim.execution)),
                "predecessor release after actual proof");
        wait_ready(owner);
        for (unsigned int index = 0; index < 4; ++index) execute(resources, owner, 100'000'000);
        check(cuEventRecord(resources.observer, resources.low), "independent low prefix observer");
        require(cuEventQuery(resources.observer) == CUDA_ERROR_NOT_READY,
                "actual low prefix still running");
        auto high = result(gate->submit_and_claim({10, 1}));
        require(high.claim.status == CooperativeClaimStatus::kPending,
                "unproven native epoch holds real shared slot");
        until([&] { return owner.snapshot().execution->phase == NativeEpochPhase::kStopping; });
        require(!owner.begin_call() && owner.snapshot().execution->has_ownership,
                "priority retreat stops new originals");
        require(result(gate->claim(high.task_id)).status == CooperativeClaimStatus::kPending,
                "retreat alone cannot hand off capacity");
        finish(owner, 4);
        wait_event(resources.observer);
        high.claim = result(gate->claim(high.task_id));
        require(high.claim.status == CooperativeClaimStatus::kGranted,
                "high grant after actual native stop");
        launch(resources, resources.high, 0);
        check(cuEventRecord(resources.observer, resources.high), "actual high proof record");
        wait_event(resources.observer);
        require(result(gate->complete(high.claim.execution)),
                "high release after actual GPU proof");
        require(owner.snapshot().recorded_prefixes == 2 && owner.snapshot().proven_prefixes == 2,
                "four originals use two proven prefixes under one grant");
    }
    auto idle_extent = kExtent;
    idle_extent.acceptance_microseconds = 100'000;
    {
        CudaNativeExecutionEpoch idle(resources.context, resources.low, kScope, {idle_extent, 0},
                                      gate, dispatch());
        wait_ready(idle);
        execute(resources, idle);
        finish(idle, 1);  // no further caller and no explicit close
        require(idle.snapshot().recorded_prefixes == 1, "idle monitor records partial real prefix");
    }
    if (!valid_only) {
        CudaNativeExecutionEpoch rejected(resources.context, resources.low, kScope, {kExtent, 0},
                                          gate, dispatch());
        wait_ready(rejected);
        const auto call = rejected.begin_call();
        require(call.has_value(), "invalid original still enters once");
        const auto original_status =
            cuLaunchKernel(nullptr, 1, 1, 1, 1, 1, 1, 0, resources.low, nullptr, nullptr);
        require(original_status != CUDA_SUCCESS, "original CUDA rejection remains authoritative");
        require(rejected.original_returned(*call, NativeEpochCallOutcome::kRejected),
                "original rejection counted without replay");
        finish(rejected, 1);
        require(rejected.snapshot().execution->has_failed,
                "failed terminal after actual trailing proof");
    }
    check(cuMemcpyDtoH(data.data(), resources.allocation, sizeof(data)),
          "complete guarded readback");
    require(data == std::array<unsigned int, 3>{0x12345678U, 7, 0x76543210U},
            "exact original outputs and guards");
    gate->shutdown();
}
void run_default_lifetime(CUstream stream) {
    Resources resources;
    CUdevice device{};
    check(cuDeviceGet(&device, 0), "canonical device");
    check(cuCtxCreate(&resources.context, 0, device), "canonical context");
    check(cuStreamCreate(&resources.high, CU_STREAM_NON_BLOCKING), "canonical high stream");
    check(cuModuleLoadData(&resources.module, kProgram), "canonical unchanged module");
    check(cuModuleGetFunction(&resources.increment, resources.module, "increment"),
          "canonical original function");
    check(cuMemAlloc(&resources.allocation, 3 * sizeof(unsigned int)),
          "canonical guarded allocation");
    std::array<unsigned int, 3> data{0x12345678U, 0, 0x76543210U};
    check(cuMemcpyHtoD(resources.allocation, data.data(), sizeof(data)),
          "canonical initial guards");
    check(cuEventCreate(&resources.observer, CU_EVENT_DISABLE_TIMING),
          "canonical independent observer");
    LaunchGateOptions options;
    options.max_concurrent_launches = 1;
    options.scheduling_policy = glimmer::core::SchedulingPolicy::kPriority;
    auto gate = std::make_shared<AsyncCooperativeLaunchGate>(options, 64, kExtent);
    auto extent = kExtent;
    if (stream == CU_STREAM_PER_THREAD) extent.calls_per_prefix = 1;
    const auto predecessor = result(gate->submit_and_claim());
    require(predecessor.claim.status == CooperativeClaimStatus::kGranted,
            "canonical predecessor admission");
    launch(resources, stream, 100'000'000);
    check(cuEventRecord(resources.observer, stream), "original caller predecessor observer");
    {
        CudaNativeExecutionEpoch owner(resources.context, stream, kScope, {extent, 0}, gate,
                                       dispatch());
        require(
            cuEventQuery(resources.observer) == CUDA_ERROR_NOT_READY &&
                owner.snapshot().phase == CudaNativeEpochPhase::kPreparing && !owner.begin_call(),
            "actual caller default predecessor pending while monitor PTDS is independently idle");
        wait_event(resources.observer);
        require(result(gate->complete(predecessor.claim.execution)),
                "canonical predecessor handoff after actual stop");
        wait_ready(owner);
        NativeEpochGrant retained{};
        for (unsigned int index = 0; index < 4; ++index) {
            std::optional<NativeEpochIssue> call;
            until([&] {
                call = owner.begin_call();
                return call.has_value();
            });
            launch(resources, stream, 100'000'000);
            require(owner.original_returned(*call, NativeEpochCallOutcome::kSuccess),
                    "canonical original return");
            const auto snapshot = owner.snapshot();
            if (index == 0) retained = snapshot.grant;
            require(snapshot.execution && snapshot.execution->issued_calls == index + 1 &&
                        snapshot.grant.execution.task_id == retained.execution.task_id &&
                        snapshot.grant.acceptance_deadline_nanoseconds ==
                            retained.acceptance_deadline_nanoseconds,
                    "canonical originals retain one unchanged grant and deadline");
            CUcontext actual{};
            check(cuCtxGetCurrent(&actual), "canonical original caller context");
            require(actual == resources.context,
                    "caller predecessor/trailing proof cannot alter context");
        }
        check(cuEventRecord(resources.observer, stream),
              "original caller actual trailing observer");
        require(cuEventQuery(resources.observer) == CUDA_ERROR_NOT_READY,
                "actual caller default originals remain pending");
        auto high = result(gate->submit_and_claim({10, 1}));
        require(high.claim.status == CooperativeClaimStatus::kPending,
                "monitor default completion cannot release caller work");
        owner.close();
        require(result(gate->claim(high.task_id)).status == CooperativeClaimStatus::kPending,
                "default epoch stop request does not release actual running capacity");
        finish(owner, 4);
        wait_event(resources.observer);
        require(owner.snapshot().recorded_prefixes == (stream == CU_STREAM_PER_THREAD ? 4U : 2U),
                "PTDS records every actual return; legacy retains bounded monitor prefixes");
        high.claim = result(gate->claim(high.task_id));
        require(high.claim.status == CooperativeClaimStatus::kGranted,
                "canonical actual stop permits high grant");
        launch(resources, resources.high, 0);
        check(cuEventRecord(resources.observer, resources.high),
              "canonical high physical observer");
        wait_event(resources.observer);
        require(result(gate->complete(high.claim.execution)),
                "canonical high report after actual proof");
        CUcontext actual{};
        check(cuCtxGetCurrent(&actual), "context after joined PTDS cleanup");
        require(actual == resources.context,
                "joined cleanup preserves original caller context stack");
    }
    extent.acceptance_microseconds = 100'000;
    {
        CudaNativeExecutionEpoch idle(resources.context, stream, kScope, {extent, 0}, gate,
                                      dispatch());
        wait_ready(idle);
        const auto call = idle.begin_call();
        require(call.has_value(), "canonical idle original admission");
        launch(resources, stream, 0);
        require(idle.original_returned(*call, NativeEpochCallOutcome::kSuccess),
                "canonical idle original return");
        finish(idle, 1);
    }
    check(cuMemcpyDtoH(data.data(), resources.allocation, sizeof(data)),
          "canonical guarded actual readback");
    require(data == std::array<unsigned int, 3>{0x12345678U, 7, 0x76543210U},
            "canonical exact output and guards");
    gate->shutdown();
    std::cout << "canonical_epoch ptds=" << (stream == CU_STREAM_PER_THREAD)
              << " output=7 actual_stop=1 cleanup=1\n";
}

std::atomic<bool> has_recovery_query_failure{};
std::atomic<unsigned int> recovery_queries{};
std::atomic<unsigned int> recovery_synchronizations{};
CUresult recovery_query(CUevent event) {
    ++recovery_queries;
    if (has_recovery_query_failure.load()) return CUDA_ERROR_UNKNOWN;
    return cuEventQuery(event);
}
CUresult recovery_synchronize() {
    ++recovery_synchronizations;
    return cuCtxSynchronize();
}
void run_recovery_lifetime(unsigned int mode) {
    Resources resources;
    CUdevice device{};
    check(cuDeviceGet(&device, 0), "recovery device");
    check(cuCtxCreate(&resources.context, 0, device), "fresh recovery context");
    if (mode == 0)
        check(cuStreamCreate(&resources.low, CU_STREAM_NON_BLOCKING), "recovery low stream");
    const auto stream = mode == 0   ? resources.low
                        : mode == 1 ? CU_STREAM_LEGACY
                                    : CU_STREAM_PER_THREAD;
    check(cuStreamCreate(&resources.high, CU_STREAM_NON_BLOCKING), "recovery high stream");
    check(cuModuleLoadData(&resources.module, kProgram), "unchanged recovery module");
    check(cuModuleGetFunction(&resources.increment, resources.module, "increment"),
          "unchanged recovery function");
    check(cuMemAlloc(&resources.allocation, 3 * sizeof(unsigned int)),
          "recovery guarded allocation");
    std::array<unsigned int, 3> data{0x12345678U, 0, 0x76543210U};
    check(cuMemcpyHtoD(resources.allocation, data.data(), sizeof(data)), "recovery initial guards");
    check(cuEventCreate(&resources.observer, CU_EVENT_DISABLE_TIMING),
          "recovery independent observer");
    LaunchGateOptions options;
    options.max_concurrent_launches = 1;
    options.scheduling_policy = glimmer::core::SchedulingPolicy::kPriority;
    auto gate = std::make_shared<AsyncCooperativeLaunchGate>(options, 64, kExtent);
    auto extent = kExtent;
    if (mode == 2) extent.calls_per_prefix = 1;
    auto native_dispatch = dispatch();
    native_dispatch.event_query = recovery_query;
    native_dispatch.context_synchronize = recovery_synchronize;
    has_recovery_query_failure = false;
    recovery_synchronizations = 0;
    {
        CudaNativeExecutionEpoch owner(resources.context, stream, kScope, {extent, 0}, gate,
                                       native_dispatch);
        wait_ready(owner);
        const auto call = owner.begin_call();
        require(call.has_value(), "recovery original admitted exactly once");
        has_recovery_query_failure = true;
        launch(resources, stream, 500'000'000);
        require(owner.original_returned(*call, NativeEpochCallOutcome::kSuccess),
                "unchanged recovery original returns once");
        check(cuEventRecord(resources.observer, stream),
              "actual original recovery prefix observer");
        owner.close();
        until([&] { return owner.snapshot().phase == CudaNativeEpochPhase::kRetained; });
        owner.shutdown();
        const auto before = owner.snapshot();
        const auto queries = recovery_queries.load();
        require(before.execution && before.execution->has_ownership &&
                    cuEventQuery(resources.observer) == CUDA_ERROR_NOT_READY && !owner.begin_call(),
                "injected projection failure retains actual still-running original scope");
        auto high = result(gate->submit_and_claim({10, 1}));
        require(high.claim.status == CooperativeClaimStatus::kPending &&
                    owner.recover_context_completion({11, 12, 14}) == CUDA_ERROR_INVALID_VALUE &&
                    recovery_synchronizations == 0,
                "wrong scope supplies no SDK proof or priority capacity");
        check(owner.recover_context_completion(kScope),
              "actual original whole-context recovery proof");
        const auto recovered = owner.snapshot();
        CUcontext actual{};
        check(cuCtxGetCurrent(&actual), "actual recovery caller context");
        require(recovered.status == before.status &&
                    recovered.phase == CudaNativeEpochPhase::kComplete &&
                    recovered.execution->has_failed && !recovered.execution->has_ownership &&
                    recovered.execution->issued_calls == 1 &&
                    recovered.execution->completed_calls == 1 &&
                    recovered.created_events == recovered.destroyed_events &&
                    recovery_queries == queries && recovery_synchronizations == 1 &&
                    actual == resources.context &&
                    owner.recover_context_completion(kScope) != CUDA_SUCCESS && !owner.begin_call(),
                "whole-context proof retires exact failed scope without query, original or report "
                "replay");
        check(cuEventQuery(resources.observer),
              "independent original prefix now physically complete");
        high.claim = result(gate->claim(high.task_id));
        require(
            high.claim.status == CooperativeClaimStatus::kGranted,
            "priority capacity transfers only after actual whole-context stop and acknowledgment");
        launch(resources, resources.high, 0);
        check(cuEventRecord(resources.observer, resources.high), "actual recovery high proof");
        wait_event(resources.observer);
        require(result(gate->complete(high.claim.execution)), "recovery high reports actual stop");
    }
    check(cuMemcpyDtoH(data.data(), resources.allocation, sizeof(data)),
          "recovery guarded readback");
    require(data == std::array<unsigned int, 3>{0x12345678U, 2, 0x76543210U},
            "failed original and later priority original execute exactly once");
    gate->shutdown();
    std::cout << "epoch_recovery mode=" << mode
              << " originals=2 sync=1 query_replay=0 failed=1 output=2 guards=1 cleanup=1\n";
}
}  // namespace
int main(int argc, char** argv) try {
    const bool valid_only = argc == 2 && std::string_view(argv[1]) == "--valid-api-only";
    const bool defaults_only = argc == 2 && std::string_view(argv[1]) == "--default-streams";
    const bool recovery_only = argc == 2 && std::string_view(argv[1]) == "--recovery";
    require(argc == 1 || valid_only || defaults_only || recovery_only, "fixture options");
    check(cuInit(0), "initialize CUDA");
    if (recovery_only) {
        for (unsigned int lifetime = 0; lifetime < 2; ++lifetime)
            for (unsigned int mode = 0; mode < 3; ++mode) run_recovery_lifetime(mode);
        require(!has_cleanup_failure, "all recovered actual resource lifetimes cleaned up");
        return 0;
    }
    if (defaults_only) {
        for (unsigned int lifetime = 0; lifetime < 2; ++lifetime)
            for (CUstream stream :
                 {static_cast<CUstream>(nullptr), CU_STREAM_LEGACY, CU_STREAM_PER_THREAD})
                run_default_lifetime(stream);
        require(!has_cleanup_failure, "canonical resource lifetimes clean up");
        std::cout << "Canonical native epoch GPU tests passed (private default/PTDS proof)\n";
        return 0;
    }
    run_lifetime(valid_only);
    run_lifetime(valid_only);
    require(!has_cleanup_failure, "all actual resource lifetimes cleaned up");
    std::cout << "Private native epoch GPU tests passed (not public routing or performance)\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
} catch (...) {
    return 1;
}
