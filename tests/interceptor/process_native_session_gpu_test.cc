#include "internal/driver_cooperative_router.h"
#include "internal/cuda_capture_origin_registry.h"
#include "cooperative_ptx_fixture.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <latch>
#include <stdexcept>
#include <thread>
#include <vector>
#include <string_view>
#include <cerrno>
#include <sys/wait.h>
#include <unistd.h>

namespace {
using namespace glimmer::interceptor;
constexpr auto kNativeHoldPtx = R"ptx(
.version 7.0
.target sm_70
.address_size 64
.visible .entry native_hold(.param .u64 state) {
    .reg .b64 %rd<2>;
    .reg .b32 %r<2>;
    .reg .pred %p;
    ld.param.u64 %rd0, [state];
    add.u64 %rd1, %rd0, 4;
    mov.u32 %r0, 1;
    st.volatile.global.u32 [%rd1], %r0;
    membar.sys;
HOLD:
    ld.volatile.global.u32 %r1, [%rd0];
    setp.eq.u32 %p, %r1, 0;
    @%p bra HOLD;
    ret;
}
)ptx";
void check(CUresult result, const char* operation) {
    if (result != CUDA_SUCCESS) {
        throw std::runtime_error(std::string(operation) + ": " + std::to_string(result));
    }
}
void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}
template <typename Function>
Function native(const DriverDispatch& driver, const char* name, bool is_ptds = false) {
    void* result{};
    check(driver.get_proc_address(name, &result, 12600,
                                  is_ptds ? CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM
                                          : CU_GET_PROC_ADDRESS_LEGACY_STREAM),
          name);
    require(result != nullptr && !is_interceptor_driver_function(result),
            "private fixture requires exact unwrapped native dispatch");
    return reinterpret_cast<Function>(result);
}

void rejected_graph_generation_after_admission(DriverCooperativeRouter& router,
                                               CudaCooperativeFrontend& frontend, CUcontext context,
                                               CUstream high, CUevent record_target,
                                               CUfunction hold, CUdeviceptr mapped,
                                               std::uint32_t* flags) {
    CUstream producer{};
    CUevent dependency{};
    CUgraph graph{};
    CUgraphExec executable{};
    check(cuStreamCreate(&producer, CU_STREAM_NON_BLOCKING), "create held preflight producer");
    check(cuEventCreate(&dependency, CU_EVENT_DISABLE_TIMING), "create logical preflight wait");
    check(router.register_event(context, dependency), "register preflight logical dependency");
    check(frontend.synchronize_stream(high, false), "prime preflight high stream snapshot");
    std::array<void*, 1> arguments{&mapped};
    check(cuLaunchKernel(hold, 1, 1, 1, 1, 1, 1, 0, producer, arguments.data(), nullptr),
          "hold actual native producer prefix before logical acceptance");
    check(frontend.record_event(dependency, producer, false), "accept pending preflight record");
    CUgraphNode wait_node{};
    CUgraphNode record_node{};
    check(cuGraphCreate(&graph, 0), "create preflight generation graph");
    check(cuGraphAddEventWaitNode(&wait_node, graph, nullptr, 0, dependency),
          "add original pending logical wait");
    check(cuGraphAddEventRecordNode(&record_node, graph, &wait_node, 1, record_target),
          "retain original record not yet issued");
    CudaNativeGraphReservation reservation;
    check(router.prepare_graph(context, graph, reservation, true), "prepare preflight graph");
    check(cuGraphInstantiate(&executable, graph, 0), "instantiate preflight graph");
    check(router.graphs().publish(reservation, executable, CUDA_SUCCESS), "publish preflight plan");
    CUresult result = CUDA_ERROR_UNKNOWN;
    std::atomic<unsigned int> originals{};
    std::jthread caller([&] {
        result = cuCtxSetCurrent(context);
        if (result == CUDA_SUCCESS) {
            result = router.native_graph_execution(executable, high, false, [&] {
                ++originals;
                return cuGraphLaunch(executable, high);
            });
        }
    });
    struct Release {
        std::uint32_t* flags;
        ~Release() {
            std::atomic_ref(flags[0]).store(1, std::memory_order_release);
        }
    } release{flags};
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{10};
    CUresult pending = CUDA_SUCCESS;
    while (pending == CUDA_SUCCESS && std::chrono::steady_clock::now() < deadline) {
        pending = frontend.query_stream(high, false);
        std::this_thread::yield();
    }
    require(pending == CUDA_ERROR_NOT_READY && originals == 0,
            "reservation must precede generation retirement while actual dependency is pending");
    router.graphs().invalidate(executable);
    std::atomic_ref(flags[0]).store(1, std::memory_order_release);
    caller.join();
    require(result == CUDA_ERROR_NOT_SUPPORTED && originals == 0,
            "post-admission generation rejection must not call the stale original graph");
    check(frontend.synchronize_stream(high, false),
          "fence rejected grant without false event proof");
    check(frontend.synchronize_event(dependency), "prove actual preflight producer stop");
    CudaEventProjection unchanged;
    check(router.event_projection(record_target, unchanged), "inspect never-issued record target");
    auto stats = frontend.stats();
    const auto acknowledgment_deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds{10};
    while (stats.acknowledged_failures != 1 &&
           std::chrono::steady_clock::now() < acknowledgment_deadline) {
        stats = frontend.stats();
        std::this_thread::yield();
    }
    require(unchanged.is_logical() && !unchanged.issued_native_prefix() &&
                stats.granted_executions == 1 && stats.acknowledged_failures == 1 &&
                stats.acknowledged_completions == 0,
            "pre-issue rejection must preserve logical record ownership and acknowledge once");
    check(frontend.close_event(dependency), "close proven preflight dependency");
    check(cuEventDestroy(dependency), "destroy original preflight event");
    router.forget_event(dependency);
    check(frontend.close_stream(producer), "close proven held producer stream");
    check(cuStreamDestroy(producer), "destroy original held producer stream");
    check(cuGraphExecDestroy(executable), "destroy never-issued retired graph");
    router.graphs().forget(executable);
    check(cuGraphDestroy(graph), "destroy preflight source graph");
}

void issued_graph_record_prefix(bool should_fail_issue = false) {
    check(cuInit(0), "initialize original issued graph Driver");
    DriverDispatch driver;
    require(driver.initialize(), "initialize issued graph dispatch");
    DriverCooperativeRouter router(driver, {}, false);
    CUdevice device{};
    check(cuDeviceGet(&device, 0), "issued graph device");
    CUcontext context{};
    check(cuCtxCreate(&context, 0, device), "issued graph context");
    CUstream high{};
    CUstream consumer{};
    check(cuStreamCreate(&high, CU_STREAM_NON_BLOCKING), "issued graph stream");
    check(cuStreamCreate(&consumer, CU_STREAM_NON_BLOCKING), "fresh record stream");
    CUevent event{};
    CUevent start{};
    check(cuEventCreate(&event, CU_EVENT_DEFAULT), "issued original timing event");
    check(cuEventCreate(&start, CU_EVENT_DEFAULT), "original timing origin");
    check(router.register_event(context, event), "register issued graph event");
    std::shared_ptr<CudaCooperativeFrontend> frontend;
    check(router.frontend(context, frontend), "resolve issued graph frontend");
    check(frontend->synchronize_stream(consumer, false), "prime independent record stream");
    if (!should_fail_issue) {
        CUgraph ambiguous{};
        CUgraphExec ambiguous_exec{};
        CUgraphNode recorded{};
        CUgraphNode waited{};
        check(cuGraphCreate(&ambiguous, 0), "create ambiguous same-event graph");
        check(cuGraphAddEventRecordNode(&recorded, ambiguous, nullptr, 0, event),
              "add ambiguous original record");
        check(cuGraphAddEventWaitNode(&waited, ambiguous, &recorded, 1, event),
              "add ambiguous same-event wait");
        CudaNativeGraphReservation ambiguous_reservation;
        check(router.prepare_graph(context, ambiguous, ambiguous_reservation, true),
              "retain separate same-event record/wait effects");
        check(cuGraphInstantiate(&ambiguous_exec, ambiguous, 0), "instantiate ambiguous graph");
        check(router.graphs().publish(ambiguous_reservation, ambiguous_exec, CUDA_SUCCESS),
              "publish ambiguous original registration");
        unsigned int rejected_originals{};
        require(router.native_graph_execution(ambiguous_exec, high, false,
                                              [&] {
                                                  ++rejected_originals;
                                                  return cuGraphLaunch(ambiguous_exec, high);
                                              }) == CUDA_ERROR_NOT_SUPPORTED &&
                    rejected_originals == 0 && frontend->stats().granted_executions == 0,
                "unknown topology must reject before grant/original issue, not guess node order");
        check(cuGraphExecDestroy(ambiguous_exec), "destroy never-launched ambiguous executable");
        router.graphs().forget(ambiguous_exec);
        check(cuGraphDestroy(ambiguous), "destroy ambiguous source");
        CUcontext foreign_context{};
        CUevent foreign_event{};
        check(cuCtxCreate(&foreign_context, 0, device), "create foreign logical producer context");
        check(cuEventCreate(&foreign_event, CU_EVENT_DISABLE_TIMING), "create foreign event");
        check(router.register_event(foreign_context, foreign_event),
              "register foreign logical event");
        check(cuCtxSetCurrent(context), "restore dependent graph context");
        CUgraph foreign_graph{};
        CUgraphExec foreign_exec{};
        CUgraphNode foreign_wait{};
        check(cuGraphCreate(&foreign_graph, 0), "create original foreign event-wait graph");
        check(cuGraphAddEventWaitNode(&foreign_wait, foreign_graph, nullptr, 0, foreign_event),
              "retain actual original cross-context wait handle");
        CudaNativeGraphReservation foreign_reservation;
        check(router.prepare_graph(context, foreign_graph, foreign_reservation, true),
              "prepare original cross-context wait identity");
        check(cuGraphInstantiate(&foreign_exec, foreign_graph, 0),
              "instantiate foreign wait graph");
        check(router.graphs().publish(foreign_reservation, foreign_exec, CUDA_SUCCESS),
              "publish foreign event-wait registration");
        require(router.native_graph_execution(foreign_exec, high, false,
                                              [&] {
                                                  ++rejected_originals;
                                                  return cuGraphLaunch(foreign_exec, high);
                                              }) == CUDA_ERROR_NOT_SUPPORTED &&
                    rejected_originals == 0 && frontend->stats().granted_executions == 0,
                "foreign logical producers must reject before grant/original graph issue");
        check(cuGraphExecDestroy(foreign_exec), "destroy never-issued foreign wait executable");
        router.graphs().forget(foreign_exec);
        check(cuGraphDestroy(foreign_graph), "destroy foreign wait source graph");
        check(cuCtxSetCurrent(foreign_context), "activate foreign producer cleanup");
        std::shared_ptr<CudaCooperativeFrontend> foreign_frontend;
        check(router.frontend(foreign_context, foreign_frontend), "resolve foreign producer owner");
        check(foreign_frontend->close_event(foreign_event), "close foreign logical event");
        check(cuEventDestroy(foreign_event), "destroy foreign original event");
        router.forget_event(foreign_event);
        check(router.close_context(foreign_context, [] { return CUDA_SUCCESS; }),
              "close foreign producer private owner");
        foreign_frontend.reset();
        check(cuCtxDestroy(foreign_context), "destroy foreign producer original context");
        check(cuCtxSetCurrent(context), "restore issued graph after foreign rejection");
    }
    CUmodule module{};
    CUmodule hold_module{};
    CUfunction kernel{};
    CUfunction hold{};
    check(cuModuleLoadData(&module, kNativeOncePtx), "load unchanged graph counter");
    check(cuModuleGetFunction(&kernel, module, "native_once"), "resolve original graph counter");
    check(cuModuleLoadData(&hold_module, kNativeHoldPtx), "load unchanged business hold kernel");
    check(cuModuleGetFunction(&hold, hold_module, "native_hold"), "resolve original graph hold");
    CUdeviceptr counter{};
    check(cuMemAlloc(&counter, sizeof(std::uint32_t)), "allocate issued graph counter");
    check(cuMemsetD32(counter, 0, 1), "initialize issued graph counter");
    std::uint32_t* flags{};
    check(cuMemHostAlloc(reinterpret_cast<void**>(&flags), 2 * sizeof(*flags),
                         CU_MEMHOSTALLOC_DEVICEMAP),
          "allocate original graph business flags");
    flags[0] = should_fail_issue ? 1U : 0U;
    flags[1] = 0;
    struct Release {
        std::uint32_t*& flags;
        ~Release() {
            if (flags) {
                std::atomic_ref(flags[0]).store(1, std::memory_order_release);
            }
        }
    } release{flags};
    CUdeviceptr mapped{};
    check(cuMemHostGetDevicePointer(&mapped, flags, 0), "map original business flags");
    if (!should_fail_issue) {
        rejected_graph_generation_after_admission(router, *frontend, context, high, event, hold,
                                                  mapped, flags);
        flags[0] = flags[1] = 0;
    }
    const auto before_stats = frontend->stats();
    CUgraph graph{};
    CUgraphExec executable{};
    check(cuGraphCreate(&graph, 0), "create unchanged issued-record graph");
    CUgraphNode kernel_node{};
    CUgraphNode record_node{};
    CUgraphNode hold_node{};
    std::array<void*, 1> counter_arguments{&counter};
    std::array<void*, 1> hold_arguments{&mapped};
    CUDA_KERNEL_NODE_PARAMS parameters{};
    parameters.func = kernel;
    parameters.gridDimX = parameters.gridDimY = parameters.gridDimZ = 1;
    parameters.blockDimX = parameters.blockDimY = parameters.blockDimZ = 1;
    parameters.kernelParams = counter_arguments.data();
    check(cuGraphAddKernelNode(&kernel_node, graph, nullptr, 0, &parameters),
          "add unchanged native graph counter node");
    check(cuGraphAddEventRecordNode(&record_node, graph, &kernel_node, 1, event),
          "record original event before later business work");
    parameters.func = hold;
    parameters.kernelParams = hold_arguments.data();
    check(cuGraphAddKernelNode(&hold_node, graph, &record_node, 1, &parameters),
          "add unchanged original business hold after event");
    CudaNativeGraphReservation reservation;
    check(router.prepare_graph(context, graph, reservation, true),
          "prepare actual graph event effects before original instantiate");
    check(cuGraphInstantiate(&executable, graph, 0), "instantiate unchanged issued graph");
    check(router.graphs().publish(reservation, executable, CUDA_SUCCESS),
          "publish prepared executable without native replay");
    CudaNativeGraphExecutable owned;
    check(router.graphs().lookup(executable, owned), "retain original graph ownership");
    require(owned.effects().node_count() == 3 && owned.events().size() == 1 &&
                owned.events()[0].effect.event == event,
            "graph preparation must preserve its three original nodes");
    check(cuGraphDestroy(graph), "destroy original source before issued graph replay");
    graph = nullptr;
    check(cuEventRecord(start, high), "record native timing origin");
    check(cuEventSynchronize(start), "prove timing origin before unchanged graph");
    unsigned int originals{};
    const auto original = [&] {
        ++originals;
        const auto launched = cuGraphLaunch(executable, high);
        return launched == CUDA_SUCCESS && should_fail_issue ? CUDA_ERROR_INVALID_VALUE : launched;
    };
    const auto result = router.native_graph_execution(executable, high, false, original);
    if (should_fail_issue) {
        require(result == CUDA_ERROR_INVALID_VALUE && originals == 1 &&
                    router.native_graph_execution(executable, high, false, original) ==
                        CUDA_ERROR_INVALID_VALUE &&
                    originals == 1 && frontend->shutdown() == CUDA_ERROR_INVALID_VALUE,
                "uncertain original graph error must retain exact status/pause without replay");
        check(cuCtxSynchronize(), "prove actual issued error-side-effect graph only for test");
        std::uint32_t count{};
        check(cuMemcpyDtoH(&count, counter, sizeof(count)), "inspect original error-side effect");
        require(count == 1, "quarantine must not repeat the already-issued original graph");
        // Retained uncertain worker/control/resources intentionally survive to
        // process exit. This child cannot use native proof to auto-thaw/cleanup.
        ::_exit(EXIT_SUCCESS);
    }
    check(result, "issue original graph through exact native reservation once");
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{10};
    while (std::atomic_ref(flags[1]).load(std::memory_order_acquire) != 1 &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    require(std::atomic_ref(flags[1]).load(std::memory_order_acquire) == 1,
            "native graph must reach later business work without completing its prefix");
    CudaEventProjection issued;
    check(router.event_projection(event, issued), "inspect issued native record epoch");
    require(!issued.is_logical() && issued.issued_native_prefix() &&
                !issued.has_native_completion() &&
                frontend->query_stream(high, false) == CUDA_ERROR_NOT_READY,
            "issued record cannot masquerade as whole graph completion");
    bool needs_boundary = true;
    check(router.logical_event_completion(event, false, needs_boundary),
          "select original native query authority before graph trailing completion");
    require(!needs_boundary && cuEventQuery(event) == CUDA_SUCCESS,
            "mid-graph original event query must not wait for whole graph or context");
    float before{};
    check(cuEventElapsedTime(&before, start, event), "read original mid-graph record timestamp");
    std::shared_ptr<CudaQueueEventRecordGuard> guard;
    check(router.prepare_event_record_guard(issued, guard), "prepare exact fresh record guard");
    check(frontend->record_event(event, consumer, false, guard),
          "asynchronously accept fresh event record while original graph remains pending");
    require(frontend->record_event(event, consumer, false, guard) == CUDA_ERROR_INVALID_VALUE,
            "reused prepared guard must not alter an already-owned deferred write snapshot");
    CudaEventProjection fresh;
    check(router.event_projection(event, fresh), "inspect fresh logical ownership");
    require(
        fresh.is_logical() && !fresh.issued_native_prefix() &&
            router.logical_event_completion(event, false, needs_boundary) == CUDA_ERROR_NOT_READY &&
            !needs_boundary,
        "fresh acceptance cannot reuse the older ready native event as logical completion");
    float held{};
    check(cuEventElapsedTime(&held, start, event), "check old native timestamp during prefix hold");
    require(held == before, "fresh physical write must remain behind the exact native prefix");
    std::atomic_ref(flags[0]).store(1, std::memory_order_release);
    check(frontend->synchronize_event(event), "prove fresh write after actual graph completion");
    float after{};
    check(cuEventElapsedTime(&after, start, event), "read fresh record after native prefix proof");
    std::uint32_t count{};
    check(cuMemcpyDtoH(&count, counter, sizeof(count)), "read exact original issued graph count");
    auto final_stats = frontend->stats();
    const auto acknowledgment_deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds{10};
    while (final_stats.acknowledged_completions != before_stats.acknowledged_completions + 1 &&
           std::chrono::steady_clock::now() < acknowledgment_deadline) {
        final_stats = frontend->stats();
        std::this_thread::yield();
    }
    require(after > before && originals == 1 && count == 1 &&
                final_stats.granted_executions == before_stats.granted_executions + 1 &&
                final_stats.acknowledged_completions == before_stats.acknowledged_completions + 1 &&
                final_stats.acknowledged_failures == before_stats.acknowledged_failures,
            "exact native prefix must order one fresh physical write without graph replay");
    router.graphs().invalidate(executable);
    require(router.native_graph_execution(executable, high, false, original) ==
                    CUDA_ERROR_NOT_SUPPORTED &&
                originals == 1,
            "retired executable plan cannot issue a stale original graph");
    check(frontend->close_event(event), "close issued event logical owner");
    check(cuEventDestroy(event), "destroy original issued event");
    router.forget_event(event);
    check(cuEventDestroy(start), "destroy native timing origin");
    check(frontend->close_stream(high), "close issued original stream owner");
    check(frontend->close_stream(consumer), "close fresh record stream owner");
    check(cuStreamDestroy(high), "destroy original issued stream");
    check(cuStreamDestroy(consumer), "destroy original fresh stream");
    check(cuGraphExecDestroy(executable), "destroy original issued executable");
    router.graphs().forget(executable);
    check(cuModuleUnload(module), "destroy original counter module");
    check(cuModuleUnload(hold_module), "destroy original business module");
    check(cuMemFree(counter), "free original issued graph counter");
    check(cuMemFreeHost(flags), "free proven original graph business flags");
    flags = nullptr;
    check(router.close_context(context, [] { return CUDA_SUCCESS; }),
          "close proven issued graph private context ownership");
    frontend.reset();
    check(cuCtxDestroy(context), "destroy issued graph original context");
    std::cout << "issued_graph_prefix original_once=1 native_mid_query=1 fresh_async_record=1 "
                 "exact_native_prefix=1\n";
}

void isolated_fault_child(const char* mode) {
    const auto child = ::fork();
    require(child >= 0, "start uncertain graph issue child");
    if (child == 0) {
        // Reinitialize CUDA after exec, never use inherited Driver state.
        ::execl("/proc/self/exe", "glimmer_cuda_process_native_session_gpu_test", mode,
                static_cast<char*>(nullptr));
        ::_exit(EXIT_FAILURE);
    }
    int status{};
    pid_t result{};
    do {
        result = ::waitpid(child, &status, 0);
    } while (result < 0 && errno == EINTR);
    require(result == child && WIFEXITED(status) && WEXITSTATUS(status) == EXIT_SUCCESS,
            "isolated native fault child must preserve error without replay");
}

void root_capture_fault(std::string_view mode) {
    check(cuInit(0), "initialize isolated root capture fault");
    DriverDispatch driver;
    require(driver.initialize(), "initialize original root capture dispatch");
    DriverCooperativeRouter router(driver, {}, false);
    ProcessCallGate gate;
    CUdevice device{};
    CUcontext context{};
    CUstream stream{};
    check(cuDeviceGet(&device, 0), "root fault device");
    check(cuCtxCreate(&context, 0, device), "root fault context");
    check(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), "root fault stream");
    std::shared_ptr<CudaCooperativeFrontend> frontend;
    check(router.frontend(context, frontend), "root fault existing private worker");
    check(frontend->synchronize_stream(stream, false), "prime root fault native prefix");
    const auto begin = native<decltype(&cuStreamBeginCapture)>(driver, "cuStreamBeginCapture");
    unsigned int root_calls{};
    unsigned int driver_calls{};
    CudaProcessNativeToken cached;
    const ProcessCallGate::Call call(gate);
    const auto result = router.source_capture_root(
        gate, CudaSourceCaptureKind::kBegin, [&](auto& dispatch) -> CUresult {
            ++root_calls;
            check(router.source_native_token(gate, cached), "cache admitted source token");
            const auto original = [&] {
                ++driver_calls;
                return begin(stream, mode == "--root-capture-swallowed-error"
                                         ? static_cast<CUstreamCaptureMode>(3)
                                         : CU_STREAM_CAPTURE_MODE_GLOBAL);
            };
            if (mode == "--root-capture-missing") {
                return CUDA_SUCCESS;
            }
            if (mode == "--root-capture-unmatched-error") {
                check(original(), "unmatched actual original begin");
                return CUDA_ERROR_INVALID_VALUE;
            }
            if (mode == "--root-capture-mismatch") {
                return dispatch.invoke(context, stream, false, CudaSourceCaptureKind::kEnd,
                                       original);
            }
            const auto dispatched =
                dispatch.invoke(context, stream, false, CudaSourceCaptureKind::kBegin, original);
            if (mode == "--root-capture-swallowed-error") {
                require(dispatched == CUDA_ERROR_INVALID_VALUE,
                        "matched invalid native begin retains original error");
                return CUDA_SUCCESS;
            }
            check(dispatched, "first matched original root begin");
            return dispatch.invoke(context, stream, false, CudaSourceCaptureKind::kBegin, original);
        });
    const bool should_preserve_error = mode == "--root-capture-mismatch" ||
                                       mode == "--root-capture-duplicate" ||
                                       mode == "--root-capture-unmatched-error";
    require(
        result == (should_preserve_error ? CUDA_ERROR_INVALID_VALUE : CUDA_ERROR_UNKNOWN) &&
            root_calls == 1 &&
            driver_calls ==
                (mode == "--root-capture-mismatch" || mode == "--root-capture-missing" ? 0U : 1U) &&
            gate.is_native_only_on_current_thread(),
        "root dispatch uncertainty preserves native error and never replays or thaws");
    unsigned int forbidden{};
    require(router.source_capture_root(gate, CudaSourceCaptureKind::kEnd,
                                       [&](auto&) {
                                           ++forbidden;
                                           return CUDA_SUCCESS;
                                       }) == CUDA_ERROR_UNKNOWN &&
                router.try_finish_source_capture(gate) == CUDA_ERROR_UNKNOWN && forbidden == 0,
            "unknown actual root effects cannot be reconciled as an empty origin set");
    std::shared_ptr<CudaCooperativeFrontend> excluded;
    require(router.frontend(context, excluded) == CUDA_ERROR_UNKNOWN && !excluded,
            "uncertain root keeps private worker ownership excluded");
    const auto forbidden_execution = [&] {
        ++forbidden;
        return CUDA_SUCCESS;
    };
    const auto source_scope = [](const auto& original) { return original(); };
    require(router.native_session_execution(cached, context, stream, false, forbidden_execution,
                                            source_scope, &gate) == CUDA_ERROR_UNKNOWN &&
                router.native_session_execution(cached, context, stream, false, forbidden_execution,
                                                source_scope) == CUDA_ERROR_INVALID_VALUE &&
                forbidden == 0,
            "cached source token cannot bypass quarantine or its bound process gate");
    // Intentional physical/metadata quarantine. Do not manufacture cleanup or
    // inherit initialized CUDA across fork; the parent invokes this after exec.
    ::_exit(0);
}

void source_capture_owner(bool should_recapture_queued_source = false,
                          bool should_use_root = false) {
    ProcessCallGate gate;
    DriverDispatch driver;
    require(driver.initialize(), "initialize original source capture dispatch");
    DriverCooperativeRouter router(driver, {}, false);
    const auto begin = native<decltype(&cuStreamBeginCapture)>(driver, "cuStreamBeginCapture");
    const auto end = native<decltype(&cuStreamEndCapture)>(driver, "cuStreamEndCapture");
    CUdevice device{};
    check(cuDeviceGet(&device, 0), "source owner device");
    std::array<CUcontext, 2> contexts{};
    std::array<CUstream, 2> captures{};
    std::array<std::shared_ptr<CudaCooperativeFrontend>, 2> frontends;
    for (std::size_t index = 0; index < contexts.size(); ++index) {
        check(cuCtxCreate(&contexts[index], 0, device), "create source owner context");
        check(cuStreamCreate(&captures[index], CU_STREAM_NON_BLOCKING), "create source capture");
        check(router.frontend(contexts[index], frontends[index]), "prepare source owner worker");
    }
    check(cuCtxSetCurrent(contexts[0]), "activate native source owner");
    CUstream execution{};
    CUstream queued_execution{};
    check(cuStreamCreate(&execution, CU_STREAM_NON_BLOCKING), "create held native source stream");
    check(cuStreamCreate(&queued_execution, CU_STREAM_NON_BLOCKING), "create queued native source");
    CUmodule module{};
    CUmodule hold_module{};
    CUfunction kernel{};
    CUfunction hold{};
    check(cuModuleLoadData(&module, kNativeOncePtx), "prepare unchanged source graph kernel");
    check(cuModuleGetFunction(&kernel, module, "native_once"), "resolve native source kernel");
    check(cuModuleLoadData(&hold_module, kNativeHoldPtx), "prepare held source kernel");
    check(cuModuleGetFunction(&hold, hold_module, "native_hold"), "resolve held source kernel");
    CUdeviceptr counter{};
    check(cuMemAlloc(&counter, sizeof(unsigned int)), "allocate native source counter");
    check(cuMemsetD32(counter, 0, 1), "initialize native source counter");
    std::uint32_t* flags{};
    check(cuMemHostAlloc(reinterpret_cast<void**>(&flags), 2 * sizeof(*flags),
                         CU_MEMHOSTALLOC_DEVICEMAP),
          "allocate held source business marker");
    flags[0] = flags[1] = 0;
    CUdeviceptr mapped{};
    check(cuMemHostGetDevicePointer(&mapped, flags, 0), "map held source business marker");
    std::jthread queued;
    struct Release {
        std::uint32_t*& flags;
        ~Release() {
            if (flags) {
                std::atomic_ref(flags[0]).store(1, std::memory_order_release);
            }
        }
    } release{flags};
    const auto phase = [&] {
        const ProcessCallGate::Call call(gate);
        return call.is_native_only();
    };
    unsigned int begins{};
    unsigned int ends{};
    unsigned int root_calls{};
    const auto dispatch_source = [&](CUcontext context, CUstream stream, bool is_ptds,
                                     CudaSourceCaptureKind kind,
                                     const CudaCooperativeFrontend::NativeOperation& original) {
        if (!should_use_root) {
            return router.source_capture_transition(gate, context, stream, is_ptds, kind, original);
        }
        return router.source_capture_root(gate, kind, [&](auto& dispatch) {
            ++root_calls;
            require(!is_inside_driver_call() && gate.is_native_only_on_current_thread(),
                    "root callback remains visible under one borrowed transition");
            std::shared_ptr<CudaCooperativeFrontend> excluded;
            require(router.frontend(context, excluded) == CUDA_ERROR_NOT_READY && !excluded,
                    "all private ownership is paused before actual origin discovery");
            if (root_calls == 1) {
                std::atomic<CUresult> foreign{CUDA_SUCCESS};
                std::jthread other([&] {
                    foreign.store(dispatch.invoke(context, stream, is_ptds, kind, original));
                });
                other.join();
                require(foreign.load() == CUDA_ERROR_INVALID_VALUE,
                        "foreign caller cannot consume or damage root capability");
            }
            const ProcessCallGate::Call nested_driver(gate);
            return dispatch.invoke(context, stream, is_ptds, kind, [&] {
                require(is_inside_driver_call(),
                        "only actual Driver original is privately guarded");
                return original();
            });
        });
    };
    const auto transition = [&](std::size_t index, CudaSourceCaptureKind kind,
                                const CudaCooperativeFrontend::NativeOperation& original) {
        check(cuCtxSetCurrent(contexts[index]), "activate actual source capture origin");
        const ProcessCallGate::Call call(gate);
        return dispatch_source(contexts[index], captures[index], false, kind, original);
    };
    unsigned int invalid_roots{};
    const auto forbidden_root = [&](auto&) {
        ++invalid_roots;
        return CUDA_SUCCESS;
    };
    require(router.source_capture_root(gate, CudaSourceCaptureKind::kBegin, forbidden_root) ==
                CUDA_ERROR_INVALID_VALUE,
            "unadmitted root must reject before original invocation");
    {
        const ProcessCallGate::Call call(gate);
        require(router.source_capture_root(gate, static_cast<CudaSourceCaptureKind>(3),
                                           forbidden_root) == CUDA_ERROR_INVALID_VALUE &&
                    router.source_capture_root(gate, CudaSourceCaptureKind::kBegin, {}) ==
                        CUDA_ERROR_INVALID_VALUE &&
                    invalid_roots == 0 && !phase(),
                "invalid root intent/callback cannot issue or change normal phase");
    }
    // Native failure must not replay begin or retain a normal phase without proof.
    const auto invalid = transition(0, CudaSourceCaptureKind::kBegin, [&] {
        ++begins;
        return begin(captures[0], static_cast<CUstreamCaptureMode>(3));
    });
    require(invalid == CUDA_ERROR_INVALID_VALUE && begins == 1 && !phase(),
            "failed native begin returns its exact status after checked NONE/thaw");
    check(transition(0, CudaSourceCaptureKind::kBegin,
                     [&] {
                         ++begins;
                         return begin(captures[0], CU_STREAM_CAPTURE_MODE_GLOBAL);
                     }),
          "begin original GLOBAL source capture");
    require(phase(), "native-only source phase persists after begin returns");
    CudaProcessNativeToken token;
    {
        const ProcessCallGate::Call call(gate);
        check(router.source_native_token(gate, token), "read exact source native capability");
        std::array<void*, 1> arguments{&counter};
        check(router.native_session_execution(
                  token, contexts[0], captures[0], false,
                  [&] {
                      return cuLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, captures[0],
                                            arguments.data(), nullptr);
                  },
                  [](const auto& original) { return original(); }, &gate),
              "construct unchanged original node with no control grant/private event");
    }
    require(router.native_execution_stats().grants == 0,
            "captured construction never enters execution admission");
    check(transition(1, CudaSourceCaptureKind::kBegin,
                     [&] {
                         ++begins;
                         return begin(captures[1], CU_STREAM_CAPTURE_MODE_RELAXED);
                     }),
          "begin overlapping actual source origin");
    check(cuCtxSetCurrent(contexts[0]), "activate held native source execution");
    {
        const ProcessCallGate::Call call(gate);
        std::array<void*, 1> arguments{&mapped};
        check(router.native_session_execution(
                  token, contexts[0], execution, false,
                  [&] {
                      return cuLaunchKernel(hold, 1, 1, 1, 1, 1, 1, 0, execution, arguments.data(),
                                            nullptr);
                  },
                  [](const auto& original) { return original(); }, &gate),
              "admit asynchronous original work during source capture");
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{10};
    while (std::atomic_ref(flags[1]).load(std::memory_order_acquire) != 1 &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    require(std::atomic_ref(flags[1]).load(std::memory_order_acquire) == 1,
            "held native source work must actually execute before queued admission");
    std::latch classified{1};
    std::atomic<CUresult> queued_status{CUDA_ERROR_UNKNOWN};
    std::atomic<unsigned int> queued_issues{};
    queued = std::jthread([&] {
        bool is_first = true;
        try {
            check(cuCtxSetCurrent(contexts[0]), "activate queued source caller");
            const ProcessCallGate::Call call(gate);
            std::array<void*, 1> arguments{&counter};
            queued_status.store(router.native_session_execution(
                token, contexts[0], queued_execution, false,
                [&] {
                    queued_issues.fetch_add(1);
                    return cuLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, queued_execution,
                                          arguments.data(), nullptr);
                },
                [&](const auto& original) {
                    const auto result = original();
                    if (is_first) {
                        is_first = false;
                        classified.count_down();
                    }
                    return result;
                },
                &gate));
        } catch (...) {
            queued_status.store(CUDA_ERROR_UNKNOWN);
        }
        if (is_first) {
            classified.count_down();
        }
    });
    classified.wait();
    std::array<CUgraph, 2> graphs{};
    for (std::size_t index = 0; index < graphs.size(); ++index) {
        check(transition(index, CudaSourceCaptureKind::kEnd,
                         [&] {
                             ++ends;
                             return end(captures[index], &graphs[index]);
                         }),
              "return original end without waiting on detached source admission");
        require(phase(),
                "ended source capture retains native-only ownership while work is pending");
    }
    require(ends == 2 && queued_issues.load() == 0 && router.native_execution_stats().grants == 1,
            "original end returns while the second source waits, with no early native issue");
    {
        const ProcessCallGate::Call call(gate);
        require(router.try_finish_source_capture(gate) == CUDA_ERROR_NOT_READY,
                "NONE alone cannot thaw pending source/native execution");
    }
    CUgraph queued_graph{};
    if (should_recapture_queued_source) {
        check(cuCtxSetCurrent(contexts[0]), "activate waiting source's actual new capture");
        const ProcessCallGate::Call call(gate);
        check(router.source_capture_transition(
                  gate, contexts[0], queued_execution, false, CudaSourceCaptureKind::kBegin,
                  [&] { return begin(queued_execution, CU_STREAM_CAPTURE_MODE_GLOBAL); }),
              "capture the waiting source stream before its execution grant");
    }
    std::atomic_ref(flags[0]).store(1, std::memory_order_release);
    queued.join();
    check(queued_status.load(), "resume and reclassify the original queued source exactly once");
    require(queued_issues.load() == 1,
            "queued source invokes original launch once after reclassification");
    if (should_recapture_queued_source) {
        const ProcessCallGate::Call call(gate);
        check(router.source_capture_transition(
                  gate, contexts[0], queued_execution, false, CudaSourceCaptureKind::kEnd,
                  [&] { return end(queued_execution, &queued_graph); }),
              "end reclassified original captured source without private execution nodes");
    }
    CUresult finished{CUDA_ERROR_NOT_READY};
    while (finished == CUDA_ERROR_NOT_READY && std::chrono::steady_clock::now() < deadline) {
        const ProcessCallGate::Call call(gate);
        finished = router.try_finish_source_capture(gate);
        std::this_thread::yield();
    }
    check(finished, "prove all original source contexts before thaw");
    require(!phase(), "checked source completion restores normal routing");
    const auto stats = router.native_execution_stats();
    require(stats.grants == 2 && stats.completed == (should_recapture_queued_source ? 1U : 2U) &&
                stats.failed == static_cast<unsigned int>(should_recapture_queued_source) &&
                stats.retained == 0,
            "exact source grants and native physical terminal acknowledgments");
    check(cuCtxSetCurrent(contexts[0]), "inspect unchanged source graph");
    std::array<CUgraphNode, 1> nodes{};
    std::size_t count = nodes.size();
    check(cuGraphGetNodes(graphs[0], nodes.data(), &count), "inspect original source node count");
    CUDA_KERNEL_NODE_PARAMS actual{};
    check(cuGraphKernelNodeGetParams(nodes[0], &actual),
          "inspect original source function identity");
    require(count == 1 && actual.func == kernel && actual.gridDimX == 1 && actual.blockDimX == 1,
            "source capture contains only unchanged original function/dimensions");
    CUgraphExec executable{};
    check(cuGraphInstantiateWithFlags(&executable, graphs[0], 0),
          "instantiate original source graph");
    check(frontends[0]->native(
              execution, false, [&] { return cuGraphLaunch(executable, execution); }, true),
          "execute original source graph after complete source ownership reconciliation");
    check(frontends[0]->synchronize_stream(execution, false),
          "prove original source graph completion");
    if (queued_graph) {
        CUgraphExec queued_executable{};
        count = nodes.size();
        check(cuGraphGetNodes(queued_graph, nodes.data(), &count),
              "inspect reclassified original source graph");
        check(cuGraphKernelNodeGetParams(nodes[0], &actual),
              "inspect reclassified source identity");
        require(count == 1 && actual.func == kernel,
                "waiting source becomes one original captured kernel, with no private nodes");
        check(cuGraphInstantiateWithFlags(&queued_executable, queued_graph, 0),
              "instantiate reclassified source graph");
        check(frontends[0]->native(
                  execution, false, [&] { return cuGraphLaunch(queued_executable, execution); },
                  true),
              "replay reclassified original source once after safe thaw");
        check(frontends[0]->synchronize_stream(execution, false),
              "prove reclassified source replay");
        check(cuGraphExecDestroy(queued_executable), "release reclassified executable");
        check(cuGraphDestroy(queued_graph), "release reclassified source graph");
    }
    unsigned int value{};
    check(cuMemcpyDtoH(&value, counter, sizeof(value)), "read original source execution count");
    require(value == 2, "queued original and graph original each execute exactly once");
    check(cuGraphExecDestroy(executable), "destroy original source executable");
    {
        ProcessCallGate foreign;
        const ProcessCallGate::Call call(foreign);
        {
            ProcessCallGate::Transition change(foreign);
            change.set_native_only(true);
        }
        unsigned int forbidden{};
        CudaProcessNativeToken empty;
        require(router.source_native_token(foreign, empty) == CUDA_ERROR_INVALID_VALUE && !empty &&
                    router.source_capture_transition(foreign, contexts[0], captures[0], false,
                                                     CudaSourceCaptureKind::kBegin,
                                                     [&] {
                                                         ++forbidden;
                                                         return CUDA_SUCCESS;
                                                     }) == CUDA_ERROR_INVALID_VALUE &&
                    forbidden == 0,
                "foreign gate cannot consume source ownership or invoke original capture");
    }
    check(transition(0, CudaSourceCaptureKind::kBegin,
                     [&] { return begin(captures[0], CU_STREAM_CAPTURE_MODE_GLOBAL); }),
          "begin source invalidation regression");
    const auto native_sync = native<decltype(&cuStreamSynchronize)>(driver, "cuStreamSynchronize");
    require(native_sync(captures[0]) == CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED,
            "original prohibited sync invalidates actual source capture");
    {
        const ProcessCallGate::Call call(gate);
        require(router.try_finish_source_capture(gate) == CUDA_ERROR_NOT_READY && phase(),
                "INVALIDATED is not NONE and cannot thaw source ownership");
    }
    CUgraph invalidated{};
    require(transition(0, CudaSourceCaptureKind::kEnd,
                       [&] { return end(captures[0], &invalidated); }) ==
                    CUDA_ERROR_STREAM_CAPTURE_INVALIDATED &&
                !invalidated && !phase(),
            "native invalidated end keeps its error while verified NONE allows checked thaw");
    unsigned int partial_calls{};
    require(transition(0, CudaSourceCaptureKind::kBegin,
                       [&]() -> CUresult {
                           ++partial_calls;
                           check(begin(captures[0], CU_STREAM_CAPTURE_MODE_GLOBAL),
                                 "actual begin before injected source exception");
                           throw std::runtime_error("failure after actual source begin");
                       }) == CUDA_ERROR_UNKNOWN &&
                partial_calls == 1 && phase(),
            "exception after native capture begins retains ownership and never replays");
    CUgraph recovered{};
    check(transition(0, CudaSourceCaptureKind::kEnd, [&] { return end(captures[0], &recovered); }),
          "resolve exceptional source ownership using original end/NONE proof");
    require(recovered && !phase(), "explicit native end safely resolves exceptional begin");
    check(cuGraphDestroy(recovered), "release exceptional source graph");
    if (should_use_root) {
        unsigned int root_throws{};
        {
            const ProcessCallGate::Call call(gate);
            require(
                router.source_capture_root(
                    gate, CudaSourceCaptureKind::kBegin,
                    [&](auto& dispatch) -> CUresult {
                        ++root_throws;
                        check(
                            dispatch.invoke(
                                contexts[0], captures[0], false, CudaSourceCaptureKind::kBegin,
                                [&] { return begin(captures[0], CU_STREAM_CAPTURE_MODE_GLOBAL); }),
                            "actual typed begin before root exception");
                        throw std::runtime_error("failure after matched root begin");
                    }) == CUDA_ERROR_UNKNOWN &&
                    root_throws == 1 && phase(),
                "matched root exception retains known active origin without replay");
        }
        CUgraph root_recovered{};
        check(transition(0, CudaSourceCaptureKind::kEnd,
                         [&] { return end(captures[0], &root_recovered); }),
              "reconcile actual matched origin after root exception");
        require(root_recovered && !phase(), "known root origin can prove NONE and thaw safely");
        check(cuGraphDestroy(root_recovered), "release matched-root exceptional graph");
    }
    const auto begin_ptds =
        native<decltype(&cuStreamBeginCapture)>(driver, "cuStreamBeginCapture", true);
    const auto end_ptds = native<decltype(&cuStreamEndCapture)>(driver, "cuStreamEndCapture", true);
    CUgraph ptds_graph{};
    {
        const ProcessCallGate::Call call(gate);
        check(dispatch_source(contexts[0], nullptr, true, CudaSourceCaptureKind::kBegin,
                              [&] { return begin_ptds(nullptr, CU_STREAM_CAPTURE_MODE_GLOBAL); }),
              "begin actual caller PTDS capture through exact original ABI");
        CudaProcessNativeToken ptds_token;
        check(router.source_native_token(gate, ptds_token), "retain caller PTDS source capability");
        std::array<void*, 1> arguments{&counter};
        check(router.native_session_execution(
                  ptds_token, contexts[0], nullptr, true,
                  [&] {
                      return driver.launch_kernel_ptsz(kernel, 1, 1, 1, 1, 1, 1, 0, nullptr,
                                                       arguments.data(), nullptr);
                  },
                  [](const auto& original) { return original(); }, &gate),
              "construct original PTDS kernel with no execution grant");
        check(dispatch_source(contexts[0], nullptr, true, CudaSourceCaptureKind::kEnd,
                              [&] { return end_ptds(nullptr, &ptds_graph); }),
              "end actual caller PTDS origin and prove all contexts");
    }
    require(!phase(), "PTDS origin uses initiating caller and restores checked normal routing");
    count = nodes.size();
    check(cuGraphGetNodes(ptds_graph, nodes.data(), &count),
          "inspect original PTDS graph identity");
    check(cuGraphKernelNodeGetParams(nodes[0], &actual), "inspect unchanged PTDS function");
    require(count == 1 && actual.func == kernel,
            "PTDS capture contains only the original kernel node");
    check(cuGraphInstantiateWithFlags(&executable, ptds_graph, 0),
          "instantiate original PTDS graph");
    check(frontends[0]->native(
              execution, false, [&] { return cuGraphLaunch(executable, execution); }, true),
          "replay original PTDS graph once");
    check(frontends[0]->synchronize_stream(execution, false), "prove original PTDS replay");
    check(cuMemcpyDtoH(&value, counter, sizeof(value)), "read PTDS original source result");
    require(value == 3, "PTDS original replay executes exactly once");
    check(cuGraphExecDestroy(executable), "release PTDS executable");
    check(cuGraphDestroy(ptds_graph), "release original PTDS graph");
    for (unsigned int cycle = 0; cycle < 32; ++cycle) {
        check(transition(0, CudaSourceCaptureKind::kBegin,
                         [&] { return begin(captures[0], CU_STREAM_CAPTURE_MODE_GLOBAL); }),
              "repeat source capture ownership");
        CUgraph repeated{};
        check(
            transition(0, CudaSourceCaptureKind::kEnd, [&] { return end(captures[0], &repeated); }),
            "repeat source native NONE/completion/thaw");
        require(!phase(), "repeated source ownership does not retain an earlier native phase");
        check(cuGraphDestroy(repeated), "release repeated source graph");
    }
    check(cuModuleUnload(hold_module), "release held source module");
    check(cuModuleUnload(module), "release source module");
    check(cuMemFree(counter), "release source counter");
    check(cuMemFreeHost(flags), "release stopped native source marker");
    flags = nullptr;
    check(cuStreamDestroy(execution), "release source execution stream");
    check(cuStreamDestroy(queued_execution), "release queued source execution stream");
    for (std::size_t index = 0; index < contexts.size(); ++index) {
        check(cuCtxSetCurrent(contexts[index]), "activate stopped source cleanup");
        check(cuGraphDestroy(graphs[index]), "release original source graph");
        check(cuStreamDestroy(captures[index]), "release original source stream");
        check(router.close_context(contexts[index], [] { return CUDA_SUCCESS; }),
              "close reconciled source frontend");
        frontends[index].reset();
        check(cuCtxDestroy(contexts[index]), "destroy reconciled source context");
    }
}

void run() {
    check(cuInit(0), "initialize");
    DriverDispatch driver;
    require(driver.initialize(), "initialize original dispatch");
    DriverCooperativeRouter router(driver, {}, false);
    const auto begin_capture =
        native<decltype(&cuStreamBeginCapture)>(driver, "cuStreamBeginCapture");
    const auto end_capture = native<decltype(&cuStreamEndCapture)>(driver, "cuStreamEndCapture");
    const auto is_capturing = native<decltype(&cuStreamIsCapturing)>(driver, "cuStreamIsCapturing");
    CUdevice device{};
    check(cuDeviceGet(&device, 0), "device");
    std::array<CUcontext, 3> contexts{};
    std::array<CUstream, 2> streams{};
    CUstream execution_stream{};
    std::array<std::shared_ptr<CudaCooperativeFrontend>, 2> frontends;
    for (std::size_t index = 0; index < 2; ++index) {
        check(cuCtxCreate(&contexts[index], 0, device), "create owned context");
        check(cuStreamCreate(&streams[index], CU_STREAM_NON_BLOCKING), "create capture stream");
        check(router.frontend(contexts[index], frontends[index]), "initialize owned frontend");
    }
    check(cuCtxSetCurrent(contexts[0]), "activate original graph owner");
    CUmodule module{};
    CUfunction kernel{};
    CUdeviceptr counter{};
    check(cuModuleLoadData(&module, kNativeOncePtx), "prepare original module before capture");
    check(cuModuleGetFunction(&kernel, module, "native_once"), "prepare original function");
    check(cuMemAlloc(&counter, sizeof(unsigned int)), "allocate original counter");
    check(cuMemsetD32(counter, 0, 1), "initialize original counter");
    CUdeviceptr uncaptured_counter{};
    check(cuMemAlloc(&uncaptured_counter, sizeof(unsigned int)),
          "allocate uncaptured native counter");
    check(cuMemsetD32(uncaptured_counter, 0, 1), "initialize uncaptured native counter");
    check(cuStreamCreate(&execution_stream, CU_STREAM_NON_BLOCKING),
          "create uncaptured execution stream");
    CUevent epoch_event{};
    CUevent waited_event{};
    check(cuEventCreate(&epoch_event, CU_EVENT_DISABLE_TIMING), "create logical epoch event");
    check(cuEventCreate(&waited_event, CU_EVENT_DISABLE_TIMING), "create native wait proof");
    check(router.register_event(contexts[0], epoch_event), "register real logical event");
    check(frontends[0]->record_event(epoch_event, execution_stream, false),
          "accept original logical record");
    check(frontends[0]->synchronize_event(epoch_event), "complete original logical record");
    check(frontends[0]->wait_event(streams[0], false, epoch_event),
          "accept immutable historical logical wait");
    check(frontends[0]->synchronize_stream(streams[0], false), "complete historical wait");
    check(frontends[0]->query_event(epoch_event), "old logical record reports success");
    CudaEventProjection old_projection;
    check(router.event_projection(epoch_event, old_projection), "retain old projection");
    require(old_projection.is_logical(), "completed queue event starts logically owned");
    std::shared_ptr<CudaQueueEventRecordGuard> old_guard;
    require(
        router.prepare_event_record_guard({}, old_guard) == CUDA_ERROR_INVALID_VALUE && !old_guard,
        "invalid guard preparation cannot publish an owner");
    check(router.prepare_event_record_guard(old_projection, old_guard),
          "prepare owned CPU record guard before source acceptance");
    bool should_write = false;
    check(old_guard->should_write(should_write), "check actual current registration epoch");
    require(should_write && router.prepare_event_record_guard(old_projection, old_guard) ==
                                CUDA_ERROR_INVALID_VALUE,
            "guard grants current ownership without replacing an occupied output");
    CUmodule hold_module{};
    CUfunction hold_kernel{};
    check(cuModuleLoadData(&hold_module, kNativeHoldPtx), "prepare native epoch hold kernel");
    check(cuModuleGetFunction(&hold_kernel, hold_module, "native_hold"), "prepare hold function");
    std::uint32_t* flags{};
    check(cuMemHostAlloc(reinterpret_cast<void**>(&flags), 2 * sizeof(*flags),
                         CU_MEMHOSTALLOC_DEVICEMAP),
          "allocate actual mapped business marker");
    flags[0] = 0;
    flags[1] = 0;
    struct ReleaseHold {
        std::uint32_t*& flags;
        ~ReleaseHold() {
            if (flags) {
                std::atomic_ref(flags[0]).store(1, std::memory_order_release);
            }
        }
    } release_hold{flags};
    CUdeviceptr mapped_flags{};
    check(cuMemHostGetDevicePointer(&mapped_flags, flags, 0), "get actual marker device alias");
    const DriverCooperativeRouter::NativeIssueScope serialized = [](const auto& operation) {
        return operation();
    };

    CudaProcessNativeToken token;
    CudaCaptureOriginRegistry origins;
    std::array<CudaCaptureOriginToken, 2> origin_tokens;
    check(router.begin_native_session(token), "retain actual multi-context physical pause");
    CudaEventProjection native_projection;
    check(router.event_projection(epoch_event, native_projection), "inspect paused CPU epoch");
    require(!native_projection.is_logical() && !native_projection.has_native_completion() &&
                old_projection.is_logical() &&
                router.publish_logical_event(old_projection) == CUDA_ERROR_NOT_READY,
            "native pause invalidates routing without changing a historical snapshot");
    check(old_guard->should_write(should_write), "observe supersession without GPU calls");
    std::shared_ptr<CudaQueueEventRecordGuard> stale_guard;
    require(!should_write &&
                router.prepare_event_record_guard(old_projection, stale_guard) ==
                    CUDA_ERROR_NOT_READY &&
                !stale_guard,
            "paused native ownership suppresses old writes and rejects stale preparation");
    std::shared_ptr<CudaCooperativeFrontend> rejected;
    require(
        router.frontend(contexts[0], rejected) == CUDA_ERROR_NOT_READY && !rejected &&
            router.frontend(reinterpret_cast<CUcontext>(0xdead), rejected) == CUDA_ERROR_NOT_READY,
        "live session forbids existing private dispatch and new frontend creation");
    unsigned int excluded_calls{};
    const auto excluded = [&] {
        ++excluded_calls;
        return CUDA_SUCCESS;
    };
    require(router.native_boundary(excluded, excluded) == CUDA_ERROR_NOT_READY &&
                router.close_context(contexts[0], excluded) == CUDA_ERROR_NOT_READY &&
                excluded_calls == 0,
            "competing native boundaries must reject without waiting or issuing callbacks");
    CudaProcessNativeToken wrong;
    require(
        router.begin_native_session(wrong) == CUDA_ERROR_NOT_READY && !wrong &&
            router.finish_native_session(wrong, excluded, excluded) == CUDA_ERROR_INVALID_VALUE &&
            excluded_calls == 0,
        "unowned process tokens cannot begin twice or invoke completion");

    check(origins.observe_origin(contexts[0], streams[0], false, origin_tokens[0]),
          "retain first original capture origin before native issue");
    check(begin_capture(streams[0], CU_STREAM_CAPTURE_MODE_GLOBAL),
          "begin original GLOBAL capture");
    CUevent captured_event{};
    CudaEventRegistrationReservation event_registration;
    require(router.prepare_native_event(wrong, contexts[0], event_registration) ==
                    CUDA_ERROR_INVALID_VALUE &&
                !event_registration,
            "foreign session cannot prepare native creation");
    check(router.prepare_native_event(token, contexts[0], event_registration),
          "prepare native registration before original capture-created event");
    check(driver.event_create(&captured_event, CU_EVENT_DISABLE_TIMING),
          "create original event during GLOBAL capture");
    require(router.register_native_event(wrong, contexts[0], captured_event) ==
                CUDA_ERROR_INVALID_VALUE,
            "foreign session cannot register a native handle");
    check(router.publish_native_event(event_registration, captured_event, CUDA_SUCCESS),
          "publish actual capture-created event without allocation or a frontend command");
    require(router.register_native_event(token, contexts[0], captured_event) ==
                CUDA_ERROR_INVALID_HANDLE,
            "duplicate native handle cannot replace its registration");
    CUcontext observed_context{};
    require(
        router.event_frontend(captured_event, observed_context, rejected) == CUDA_ERROR_NOT_READY &&
            !rejected,
        "native event registration must not bind while workers remain paused");
    void* arguments[]{&counter};
    check(router.native_session_execution(
              token, contexts[0], streams[0], false,
              [&] {
                  return driver.launch_kernel(kernel, 1, 1, 1, 1, 1, 1, 0, streams[0], arguments,
                                              nullptr);
              },
              serialized),
          "capture original kernel without private nodes");
    check(router.native_session_execution(
              token, contexts[0], streams[0], false,
              [&] {
                  return driver.event_record_with_flags(captured_event, streams[0],
                                                        CU_EVENT_RECORD_EXTERNAL);
              },
              serialized),
          "capture actual application event record without private nodes");
    require(router.native_execution_stats().grants == 0,
            "captured construction requests no native execution lease");
    unsigned int issue_scopes{};
    const DriverCooperativeRouter::NativeIssueScope fail_before_issue = [&](const auto& body) {
        if (++issue_scopes == 2) {
            throw std::runtime_error("source transition construction failure");
        }
        return body();
    };
    require(router.native_session_execution(token, contexts[0], execution_stream, false, excluded,
                                            fail_before_issue) == CUDA_ERROR_UNKNOWN &&
                excluded_calls == 0 && router.native_execution_stats().failed == 1,
            "failed source scope retires its unissued grant without original CUDA or replay");
    void* uncaptured_arguments[]{&uncaptured_counter};
    check(router.native_session_execution(
              token, contexts[0], execution_stream, false,
              [&] {
                  return driver.launch_kernel(kernel, 1, 1, 1, 1, 1, 1, 0, execution_stream,
                                              uncaptured_arguments, nullptr);
              },
              serialized),
          "admit uncaptured native execution while GLOBAL capture remains active");
    check(cuCtxSetCurrent(contexts[1]), "activate overlapping native capture");
    check(origins.observe_origin(contexts[1], streams[1], false, origin_tokens[1]),
          "retain overlapping origin before native issue");
    check(begin_capture(streams[1], CU_STREAM_CAPTURE_MODE_RELAXED), "begin overlapping capture");
    check(cuCtxSetCurrent(contexts[0]), "restore first capture owner");
    std::array<CUgraph, 2> graphs{};
    check(end_capture(streams[0], &graphs[0]), "end first native capture");
    std::size_t nodes{};
    check(cuGraphGetNodes(graphs[0], nullptr, &nodes), "inspect captured node count");
    require(nodes == 2, "only original kernel and application event nodes belong to capture");
    std::array<CUgraphNode, 2> graph_nodes{};
    check(cuGraphGetNodes(graphs[0], graph_nodes.data(), &nodes),
          "inspect captured original nodes");
    unsigned int kernel_nodes{};
    unsigned int event_nodes{};
    for (const auto node : graph_nodes) {
        CUgraphNodeType type{};
        check(cuGraphNodeGetType(node, &type), "inspect original node type");
        if (type == CU_GRAPH_NODE_TYPE_KERNEL) {
            CUDA_KERNEL_NODE_PARAMS parameters{};
            check(cuGraphKernelNodeGetParams(node, &parameters),
                  "inspect original function identity");
            require(
                parameters.func == kernel && parameters.gridDimX == 1 && parameters.blockDimX == 1,
                "capture preserves original function and shape");
            ++kernel_nodes;
        } else if (type == CU_GRAPH_NODE_TYPE_EVENT_RECORD) {
            CUevent original_event{};
            check(cuGraphEventRecordNodeGetEvent(node, &original_event),
                  "inspect actual captured event");
            require(original_event == captured_event, "capture preserves original event handle");
            ++event_nodes;
        }
    }
    require(kernel_nodes == 1 && event_nodes == 1, "capture must add no private nodes");
    const CudaCaptureOriginRegistry::Query query = [&](CUcontext context, CUstream actual_stream,
                                                       bool is_ptds,
                                                       CUstreamCaptureStatus* status) {
        require(!is_ptds, "hardware fixture uses exact legacy capture query ABI");
        return router.with_context(context, [&] { return is_capturing(actual_stream, status); });
    };
    const auto verify = [&] { return origins.verify_none(query); };
    std::vector<CUcontext> completed;
    const auto complete = [&] {
        CUcontext current{};
        const auto result = driver.context_get_current(&current);
        if (result != CUDA_SUCCESS) {
            return result;
        }
        completed.push_back(current);
        return driver.context_synchronize();
    };
    require(router.finish_native_session(token, verify, complete) == CUDA_ERROR_NOT_READY &&
                completed.empty() && origins.retained_origins() == 1,
            "ending one capture cannot thaw while another native capture remains active");
    check(cuCtxSetCurrent(contexts[1]), "activate final capture owner");
    check(end_capture(streams[1], &graphs[1]), "end final native capture");
    check(origins.reconcile(origin_tokens[1], query),
          "prove ended origin NONE before stream reuse");

    // Reuse the same stream with a new capture generation. An invalidating API
    // error is not NONE; only original end plus checked NONE may retire it.
    CudaCaptureOriginToken invalidated_origin;
    check(origins.observe_origin(contexts[1], streams[1], false, invalidated_origin),
          "observe reused stream before new native begin");
    check(begin_capture(streams[1], CU_STREAM_CAPTURE_MODE_RELAXED),
          "begin original invalidation regression");
    require(origins.reconcile(origin_tokens[1], query) == CUDA_ERROR_INVALID_VALUE,
            "retired origin generation cannot verify a reused capture stream");
    require(driver.stream_synchronize(streams[1]) == CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED,
            "original prohibited stream synchronization invalidates capture");
    CUstreamCaptureStatus invalidated_status = CU_STREAM_CAPTURE_STATUS_NONE;
    check(is_capturing(streams[1], &invalidated_status), "inspect original invalidated state");
    require(invalidated_status == CU_STREAM_CAPTURE_STATUS_INVALIDATED &&
                router.finish_native_session(token, verify, complete) == CUDA_ERROR_NOT_READY &&
                completed.empty(),
            "original invalidation error cannot authorize context completion or thaw");
    CUgraph invalidated_graph{};
    require(end_capture(streams[1], &invalidated_graph) == CUDA_ERROR_STREAM_CAPTURE_INVALIDATED &&
                !invalidated_graph,
            "original invalidated end returns its native error and null graph");

    // This context was not in the private snapshot. Its original work must be
    // observed and proved too, without constructing a new private frontend.
    check(cuCtxSetCurrent(contexts[0]), "activate actual native event epoch");
    void* hold_arguments[]{&mapped_flags};
    check(router.native_session_execution(
              token, contexts[0], execution_stream, false,
              [&] {
                  const auto launched = driver.launch_kernel(
                      hold_kernel, 1, 1, 1, 1, 1, 1, 0, execution_stream, hold_arguments, nullptr);
                  return launched == CUDA_SUCCESS
                             ? driver.event_record(epoch_event, execution_stream)
                             : launched;
              },
              serialized),
          "record actual native epoch behind an unfinished original kernel");
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
    while (std::atomic_ref(flags[1]).load(std::memory_order_acquire) == 0 &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    require(std::atomic_ref(flags[1]).load(std::memory_order_acquire) == 1 &&
                driver.event_query(epoch_event) == CUDA_ERROR_NOT_READY,
            "actual native epoch remains pending despite old logical success");
    native_projection = {};
    check(router.event_projection(epoch_event, native_projection), "select authoritative epoch");
    require(!native_projection.is_logical() && !native_projection.has_native_completion() &&
                router.publish_logical_event(old_projection) == CUDA_ERROR_NOT_READY,
            "new native record cannot regain an old logical completion");
    check(driver.stream_wait_event(streams[0], epoch_event, 0), "wait on actual native record");
    check(driver.event_record(waited_event, streams[0]), "record native dependent wait proof");
    require(driver.event_query(waited_event) == CUDA_ERROR_NOT_READY,
            "native wait must not use the old completed queue record");
    std::atomic_ref(flags[0]).store(1, std::memory_order_release);
    check(driver.event_synchronize(waited_event), "prove actual native producer and wait complete");

    check(cuCtxCreate(&contexts[2], 0, device), "create native-only context while paused");
    check(router.observe_native_session_context(contexts[2]), "observe native-only context");
    CUstream native_stream{};
    CUevent native_event{};
    check(cuStreamCreate(&native_stream, CU_STREAM_NON_BLOCKING),
          "create actual native-only stream while paused");
    check(driver.event_create(&native_event, CU_EVENT_DISABLE_TIMING),
          "create actual native-only event while paused");
    check(router.register_native_event(token, contexts[2], native_event),
          "register native-only event without private context initialization");
    require(
        router.event_frontend(native_event, observed_context, rejected) == CUDA_ERROR_NOT_READY &&
            !rejected,
        "new native-only context cannot initialize through event lookup during pause");
    CUdeviceptr native_counter{};
    check(cuMemAlloc(&native_counter, sizeof(unsigned int)), "native-only allocation");
    check(router.native_session_execution(
              token, contexts[2], native_stream, false,
              [&] {
                  const auto set = cuMemsetD32Async(native_counter, 42, 1, native_stream);
                  return set == CUDA_SUCCESS ? driver.event_record(native_event, native_stream)
                                             : set;
              },
              serialized),
          "admit native-only asynchronous work without a frontend");
    check(cuCtxSetCurrent(contexts[0]), "activate original replay");
    CUgraphExec executable{};
    check(cuGraphInstantiate(&executable, graphs[0], 0), "instantiate original graph");
    check(router.native_session_execution(
              token, contexts[0], nullptr, true,
              [&] { return driver.graph_launch_ptsz(executable, nullptr); }, serialized),
          "admit and execute original graph once");
    check(cuCtxSetCurrent(contexts[2]), "restore original session caller context");
    check(router.finish_native_session(token, verify, complete), "finish verified native session");
    require(origins.retained_origins() == 0, "all original origins retire only after native NONE");
    const auto native_stats = router.native_execution_stats();
    require(native_stats.grants == 5 && native_stats.completed == 4 && native_stats.failed == 1 &&
                native_stats.retained == 0,
            "physical session thaw follows every exact native execution acknowledgment");
    require(completed.size() == 5 && completed[2] == contexts[2] && excluded_calls == 0,
            "all three contexts prove completion before either frontend fence import");
    CUcontext restored{};
    check(driver.context_get_current(&restored), "inspect caller context restoration");
    require(restored == contexts[2], "session completion restores the actual caller context");
    unsigned int actual{};
    check(cuMemcpyDtoH(&actual, native_counter, sizeof(actual)), "read native-only result");
    require(actual == 42, "new native context work must be complete before session releases");
    check(cuCtxSetCurrent(contexts[0]), "activate original output");
    require(
        router.register_native_event(token, contexts[2], native_event) == CUDA_ERROR_INVALID_VALUE,
        "retired session cannot register another native origin");
    std::array<std::shared_ptr<CudaCooperativeFrontend>, 2> imported_owners;
    std::array<CUresult, 2> import_status{};
    std::array<bool, 2> callers_restored{};
    std::latch import_ready{2};
    std::latch import_start{1};
    std::array<std::thread, 2> importers;
    for (std::size_t index = 0; index < importers.size(); ++index) {
        importers[index] = std::thread([&, index] {
            import_status[index] = cuCtxSetCurrent(contexts[0]);
            import_ready.count_down();
            import_start.wait();
            if (import_status[index] != CUDA_SUCCESS) {
                return;
            }
            CUcontext event_context{};
            import_status[index] =
                router.event_frontend(native_event, event_context, imported_owners[index]);
            CUcontext caller{};
            callers_restored[index] = driver.context_get_current(&caller) == CUDA_SUCCESS &&
                                      caller == contexts[0] && event_context == contexts[2];
        });
    }
    import_ready.wait();
    import_start.count_down();
    for (auto& importer : importers) {
        importer.join();
    }
    require(import_status[0] == CUDA_SUCCESS && import_status[1] == CUDA_SUCCESS &&
                callers_restored[0] && callers_restored[1] && imported_owners[0] &&
                imported_owners[0] == imported_owners[1],
            "competing actual cross-context first use initializes and binds exactly once");
    std::shared_ptr<CudaCooperativeFrontend> imported_frontend;
    check(router.event_frontend(native_event, observed_context, imported_frontend),
          "reuse existing native-created event binding");
    require(imported_frontend == imported_owners[0] && observed_context == contexts[2],
            "repeated lazy import retains the same actual context owner");
    CudaEventProjection imported_projection;
    check(router.event_projection(native_event, imported_projection),
          "inspect lazy-bound native epoch");
    require(!imported_projection.is_logical() && imported_projection.has_native_completion(),
            "binding is not a logical record or new completion projection");
    check(driver.context_get_current(&restored), "inspect actual first-use caller context");
    require(restored == contexts[0], "lazy import must preserve the source caller context");
    check(cuCtxSetCurrent(contexts[2]), "activate actual imported stream owner");
    unsigned int* native_readback{};
    check(cuMemAllocHost(reinterpret_cast<void**>(&native_readback), sizeof(*native_readback)),
          "allocate actual pinned native-prefix readback");
    *native_readback = 0;
    check(imported_frontend->copy(native_stream, false,
                                  {CudaQueuedCopyKind::kDeviceToHost, 0, native_counter,
                                   native_readback, sizeof(*native_readback), router.resources()}),
          "import native-created stream prefix and queue pinned readback");
    check(imported_frontend->record_event(native_event, native_stream, false),
          "accept fresh record against native-created event");
    check(router.publish_logical_event(imported_projection),
          "publish fresh native-created event record");
    check(imported_frontend->synchronize_event(native_event),
          "prove imported native prefix and logical copy");
    require(*native_readback == 42, "native-created stream preserves actual earlier native work");
    check(imported_frontend->close_event(native_event), "close actual imported event owner");
    check(driver.event_destroy(native_event), "release original native-created event");
    router.forget_event(native_event);
    CudaEventProjection forgotten;
    require(router.event_projection(native_event, forgotten) == CUDA_ERROR_INVALID_HANDLE,
            "destroy retires native-created registration");
    check(imported_frontend->close_stream(native_stream), "prove imported stream teardown");
    check(cuStreamDestroy(native_stream), "release original native-created stream");
    check(cuMemFreeHost(native_readback), "release physically proven pinned readback");
    check(cuMemFree(native_counter), "release physically proven native-only allocation");
    check(router.close_context(contexts[2], [] { return CUDA_SUCCESS; }),
          "close late private context owner");
    imported_frontend.reset();
    imported_owners.fill(nullptr);
    check(cuCtxSetCurrent(contexts[0]), "restore original output context");
    std::shared_ptr<CudaCooperativeFrontend> captured_owner;
    check(router.event_frontend(captured_event, observed_context, captured_owner),
          "lazily bind actual capture-created event after original replay");
    require(captured_owner == frontends[0] && observed_context == contexts[0],
            "capture-created handle maps to the actual preexisting context owner");
    CudaEventProjection captured_projection;
    check(router.event_projection(captured_event, captured_projection),
          "snapshot original graph event epoch");
    require(!captured_projection.is_logical() && captured_projection.has_native_completion(),
            "original graph record remains native after lazy binding");
    check(captured_owner->record_event(captured_event, execution_stream, false),
          "accept fresh logical record after native graph event");
    check(router.publish_logical_event(captured_projection),
          "restore capture-created event projection");
    check(captured_owner->synchronize_event(captured_event), "prove fresh capture-created record");
    check(captured_owner->close_event(captured_event), "close capture-created queue binding");
    captured_owner.reset();
    native_projection = {};
    check(router.event_projection(epoch_event, native_projection),
          "snapshot fresh acceptance epoch");
    require(!native_projection.is_logical() && native_projection.has_native_completion(),
            "physical completion alone does not restore projection");
    check(frontends[0]->record_event(epoch_event, execution_stream, false),
          "accept fresh logical record without rebinding historical waits");
    check(router.publish_logical_event(native_projection), "publish matching fresh acceptance");
    check(frontends[0]->synchronize_event(epoch_event), "complete fresh logical event");
    CudaEventProjection restored_projection;
    check(router.event_projection(epoch_event, restored_projection), "inspect restored projection");
    require(restored_projection.is_logical() &&
                router.publish_logical_event(old_projection) == CUDA_ERROR_NOT_READY,
            "fresh acceptance restores ownership but cannot revive an earlier epoch");
    check(frontends[0]->close_event(epoch_event), "close physically proven event binding");
    check(driver.event_destroy(epoch_event), "release actual epoch event");
    router.forget_event(epoch_event);
    check(driver.event_destroy(waited_event), "release native wait proof");
    check(cuModuleUnload(hold_module), "release original hold module");
    check(cuMemFreeHost(flags), "release physically proven host signal");
    flags = nullptr;
    check(cuMemcpyDtoH(&actual, counter, sizeof(actual)), "read exact original graph result");
    require(actual == 1, "original graph must execute exactly once");
    check(cuMemcpyDtoH(&actual, uncaptured_counter, sizeof(actual)),
          "read admitted uncaptured output");
    require(actual == 1, "uncaptured native execution preserves exact-once output during capture");
    check(cuMemFree(uncaptured_counter), "release uncaptured counter");
    check(cuStreamDestroy(execution_stream), "release uncaptured stream");
    require(router.finish_native_session(token, excluded, excluded) == CUDA_ERROR_INVALID_VALUE &&
                excluded_calls == 0,
            "completed process token cannot replay native work or completion");
    check(cuGraphExecDestroy(executable), "release original executable");
    check(cuMemFree(counter), "release original counter");
    check(cuModuleUnload(module), "release original module");
    for (std::size_t index = 0; index < 2; ++index) {
        check(cuCtxSetCurrent(contexts[index]), "activate checked frontend cleanup");
        check(frontends[index]->synchronize_context(), "prove imported real trailing fence");
        check(cuGraphDestroy(graphs[index]), "release original captured graph");
        if (index == 0) {
            check(driver.event_destroy(captured_event),
                  "release original captured event after graph teardown");
            router.forget_event(captured_event);
        }
        check(cuStreamDestroy(streams[index]), "release original capture stream");
        check(router.close_context(contexts[index], [] { return CUDA_SUCCESS; }),
              "release private owner");
        frontends[index].reset();
        check(cuCtxDestroy(contexts[index]), "destroy checked original context");
    }
    check(cuCtxDestroy(contexts[2]), "destroy native-only context");
}
constexpr const char* kEpochRouterPtx = R"ptx(
.version 7.0
.target sm_70
.address_size 64
.visible .entry increment(.param .u64 address, .param .u64 duration) {
 .reg .pred p;
 .reg .b64 pointer, delay, start, now, elapsed;
 .reg .b32 value;
 ld.param.u64 pointer, [address];
 ld.param.u64 delay, [duration];
 mov.u64 start, %globaltimer;
wait:
 mov.u64 now, %globaltimer;
 sub.u64 elapsed, now, start;
 setp.lt.u64 p, elapsed, delay;
 @p bra wait;
 ld.global.u32 value, [pointer];
 add.u32 value, value, 1;
 st.global.u32 [pointer], value;
 ret;
}
)ptx";
void process_native_epochs(unsigned int mode) {
    check(cuInit(0), "initialize actual process epochs");
    DriverDispatch driver;
    require(driver.initialize(), "actual process epoch native dispatch");
    ProcessCallGate gate;
    constexpr glimmer::control::NativeEpochExtent extent{8, 2, 2, 1'000'000};
    DriverCooperativeRouter router(driver, {}, false, false, extent);
    std::array<CUcontext, 2> contexts{};
    std::array<CUstream, 2> streams{};
    std::array<CUstream, 2> independent{};
    std::array<CUevent, 2> historical{};
    std::array<CUmodule, 2> modules{};
    std::array<CUfunction, 2> functions{};
    std::array<CUdeviceptr, 2> allocations{};
    std::array<std::shared_ptr<CudaCooperativeFrontend>, 2> owners{};
    CUdevice device{};
    check(cuDeviceGet(&device, 0), "process epoch device");
    for (unsigned int index = 0; index < 2; ++index) {
        check(cuCtxCreate(&contexts[index], 0, device), "fresh process epoch context");
        check(cuStreamCreate(&streams[index], CU_STREAM_DEFAULT), "actual blocking epoch stream");
        check(cuStreamCreate(&independent[index], CU_STREAM_NON_BLOCKING),
              "unrelated actual epoch stream");
        check(cuEventCreate(&historical[index], CU_EVENT_DISABLE_TIMING),
              "actual historical epoch event");
        check(cuModuleLoadData(&modules[index], kEpochRouterPtx),
              "unchanged original epoch module");
        check(cuModuleGetFunction(&functions[index], modules[index], "increment"),
              "original epoch function");
        check(cuMemAlloc(&allocations[index], 3 * sizeof(unsigned int)), "guarded process output");
        const std::array<unsigned int, 3> initial{0x12345678U, 0, 0x76543210U};
        check(cuMemcpyHtoD(allocations[index], initial.data(), sizeof(initial)),
              "initialize actual process guards");
        check(router.frontend(contexts[index], owners[index]), "bind actual process frontend");
        check(router.register_event(contexts[index], historical[index]),
              "register actual historical event");
        check(owners[index]->record_event(historical[index], streams[index], false),
              "record original historical prefix");
        check(owners[index]->synchronize_event(historical[index]),
              "complete original historical prefix");
    }
    const bool is_ptds = mode == 1;
    const auto original_launch =
        native<decltype(&cuLaunchKernel)>(driver, "cuLaunchKernel", is_ptds);
    const auto original_graph = native<decltype(&cuGraphLaunch)>(driver, "cuGraphLaunch", is_ptds);
    const auto original_begin =
        native<decltype(&cuStreamBeginCapture)>(driver, "cuStreamBeginCapture");
    const auto original_end = native<decltype(&cuStreamEndCapture)>(driver, "cuStreamEndCapture");
    const auto stream = [&](unsigned int index) {
        return mode == 0 ? static_cast<CUstream>(nullptr)
               : is_ptds ? CU_STREAM_PER_THREAD
                         : streams[index];
    };
    const auto caller = std::this_thread::get_id();
    std::array<unsigned int, 2> originals{};
    unsigned int forbidden{};
    const auto rejected = [&] {
        ++forbidden;
        return CUDA_SUCCESS;
    };
    const auto launch = [&](unsigned int index, unsigned long long duration) {
        require(std::this_thread::get_id() == caller, "router original remains on actual caller");
        CUcontext actual{};
        check(cuCtxGetCurrent(&actual), "router original actual context");
        require(actual == contexts[index], "router scope switch cannot change original context");
        auto pointer = allocations[index] + sizeof(unsigned int);
        void* arguments[]{&pointer, &duration};
        ++originals[index];
        return original_launch(functions[index], 1, 1, 1, 1, 1, 1, 0, stream(index), arguments,
                               nullptr);
    };
    check(cuCtxSetCurrent(contexts[0]), "activate first epoch caller context");
    require(router.native_execution(gate, owners[0], stream(0), is_ptds, rejected) ==
                CUDA_ERROR_NOT_PERMITTED,
            "unentered gate rejects before original issue");
    glimmer::control::NativeEpochGrant retained{};
    {
        const ProcessCallGate::Call call(gate);
        CUresult empty_context_status = CUDA_ERROR_UNKNOWN;
        CUcontext empty_context = contexts[0];
        std::thread empty_caller([&] {
            const ProcessCallGate::Call empty_call(gate);
            if (cuCtxGetCurrent(&empty_context) == CUDA_SUCCESS && !empty_context)
                empty_context_status =
                    router.native_execution(gate, owners[0], stream(0), is_ptds, rejected);
        });
        empty_caller.join();
        require(!empty_context && empty_context_status == CUDA_ERROR_INVALID_CONTEXT,
                "true-empty actual context rejects before original issue");
        require(router.native_execution(gate, owners[1], stream(0), is_ptds, rejected) ==
                    CUDA_ERROR_INVALID_CONTEXT,
                "foreign frontend rejects before original issue");
        check(router.native_execution(
                  gate, owners[0], stream(0), is_ptds,
                  [&] {
                      require(router.native_execution(gate, owners[0], stream(0), is_ptds,
                                                      rejected) == CUDA_ERROR_NOT_PERMITTED,
                              "reentrant routing rejects without self-wait or duplicate effect");
                      return launch(0, 80'000'000);
                  }),
              "first actual process epoch original");
        retained = owners[0]->native_epoch_snapshot()->grant;
        const auto alias = mode == 0 ? CU_STREAM_LEGACY : is_ptds ? nullptr : stream(0);
        check(router.native_execution(gate, owners[0], alias, is_ptds,
                                      [&] { return launch(0, 80'000'000); }),
              "canonical alias original shares the same process epoch");
        require(owners[0]->native_epoch_snapshot()->grant.execution.task_id ==
                    retained.execution.task_id,
                "same process scope retains immutable authorization");
        check(owners[0]->query_event(historical[0]),
              "historical event excludes later router epoch");
        check(owners[0]->query_stream(independent[0], false),
              "unrelated stream excludes router epoch");
        check(original_begin(independent[0], CU_STREAM_CAPTURE_MODE_THREAD_LOCAL),
              "actual external capture before epoch preflight");
        require(router.native_execution(gate, owners[0], independent[0], false, rejected) ==
                    CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED,
                "captured stream rejects before original issue or epoch closure");
        require(owners[0]->native_epoch_snapshot()->grant.execution.task_id ==
                    retained.execution.task_id,
                "capture rejection retains prior epoch identity");
        CUgraph rejected_capture{};
        check(original_end(independent[0], &rejected_capture), "end external empty capture");
        check(cuGraphDestroy(rejected_capture), "destroy external empty capture");
    }
    {
        ProcessCallGate foreign_gate;
        const ProcessCallGate::Call call(foreign_gate);
        require(router.native_execution(foreign_gate, owners[0], stream(0), is_ptds, rejected) ==
                    CUDA_ERROR_INVALID_VALUE,
                "foreign process gate cannot borrow epoch membership");
    }
    check(cuCtxSetCurrent(contexts[1]), "switch actual caller to second context");
    {
        const ProcessCallGate::Call call(gate);
        check(
            router.native_execution(
                gate, owners[1], stream(1), is_ptds,
                [&] {
                    const auto stopped = owners[0]->native_epoch_snapshot();
                    require(
                        stopped && stopped->phase == CudaNativeEpochPhase::kComplete &&
                            stopped->created_events == stopped->destroyed_events,
                        "other-context epoch physically stopped and joined before new original");
                    const auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                         std::chrono::steady_clock::now().time_since_epoch())
                                         .count();
                    require(
                        static_cast<std::uint64_t>(now) < retained.acceptance_deadline_nanoseconds,
                        "context switch closes promptly rather than awaiting immutable expiry");
                    return launch(1, 0);
                }),
            "actual cross-context original after old epoch stop");
    }
    check(cuCtxSetCurrent(contexts[0]), "restore first context for original graph");
    CUgraph graph{};
    CUgraphExec executable{};
    check(cuGraphCreate(&graph, 0), "original epoch successor graph");
    auto pointer = allocations[0] + sizeof(unsigned int);
    unsigned long long duration{};
    void* arguments[]{&pointer, &duration};
    CUDA_KERNEL_NODE_PARAMS parameters{};
    parameters.func = functions[0];
    parameters.gridDimX = parameters.gridDimY = parameters.gridDimZ = 1;
    parameters.blockDimX = parameters.blockDimY = parameters.blockDimZ = 1;
    parameters.kernelParams = arguments;
    CUgraphNode node{};
    check(cuGraphAddKernelNode(&node, graph, nullptr, 0, &parameters),
          "unchanged original graph node");
    CudaNativeGraphReservation reservation;
    check(router.prepare_graph(contexts[0], graph, reservation, true),
          "prepare original graph ownership");
    check(cuGraphInstantiate(&executable, graph, 0), "instantiate original graph");
    check(router.graphs().publish(reservation, executable, CUDA_SUCCESS),
          "publish original graph identity");
    {
        const ProcessCallGate::Call call(gate);
        check(router.native_execution(gate, owners[0], stream(0), is_ptds,
                                      [&] { return launch(0, 0); }),
              "new epoch before whole original graph");
        check(router.native_graph_execution(
                  executable, stream(0), is_ptds,
                  [&] {
                      require(owners[0]->native_epoch_snapshot()->phase ==
                                  CudaNativeEpochPhase::kComplete,
                              "whole graph closes preceding epoch before original replay");
                      return original_graph(executable, stream(0));
                  }),
              "whole original graph follows actual epoch stop");
        check(owners[0]->synchronize_context(), "actual whole graph completion");
    }
    check(cuGraphExecDestroy(executable), "destroy original executable");
    router.graphs().forget(executable);
    check(cuGraphDestroy(graph), "destroy original source graph");
    check(cuCtxSetCurrent(contexts[1]), "activate second context for process pause");
    CudaProcessNativeToken token;
    {
        const ProcessCallGate::Call call(gate);
        check(router.native_execution(gate, owners[1], stream(1), is_ptds,
                                      [&] { return launch(1, 0); }),
              "actual epoch before process native session");
        ProcessCallGate::Transition transition(gate);
        check(router.begin_native_session(token), "process pause closes all context epochs");
        require(owners[1]->native_epoch_snapshot()->phase == CudaNativeEpochPhase::kComplete,
                "process session starts only after actual epoch stop and cleanup");
        transition.set_native_only(true);
    }
    {
        const ProcessCallGate::Call call(gate);
        require(router.native_execution(gate, owners[1], stream(1), is_ptds, rejected) ==
                    CUDA_ERROR_NOT_PERMITTED,
                "native-only session rejects new epoch before effects");
        ProcessCallGate::Transition transition(gate);
        check(router.finish_native_session(
                  token, [] { return CUDA_SUCCESS; }, [] { return cuCtxSynchronize(); }),
              "finish actual process session before normal thaw");
        transition.set_native_only(false);
    }
    check(cuCtxSetCurrent(contexts[0]), "activate final original caller");
    {
        const ProcessCallGate::Call call(gate);
        check(router.native_execution(gate, owners[0], stream(0), is_ptds,
                                      [&] { return launch(0, 0); }),
              "normal original resumes after process session");
        check(router.source_capture_transition(
                  gate, contexts[0], independent[0], false, CudaSourceCaptureKind::kBegin,
                  [&] {
                      require(owners[0]->native_epoch_snapshot()->phase ==
                                  CudaNativeEpochPhase::kComplete,
                              "source capture closes epoch before original begin");
                      return original_begin(independent[0], CU_STREAM_CAPTURE_MODE_GLOBAL);
                  }),
              "source capture owns drained process epoch");
        CUgraph captured{};
        check(router.source_capture_transition(
                  gate, contexts[0], independent[0], false, CudaSourceCaptureKind::kEnd,
                  [&] { return original_end(independent[0], &captured); }),
              "original source end after epoch closure");
        check(router.try_finish_source_capture(gate), "source ownership completes before thaw");
        check(cuGraphDestroy(captured), "destroy empty original source graph");
    }
    require(forbidden == 0 && originals == std::array<unsigned int, 2>{4, 2},
            "exact original and rejection counts");
    for (unsigned int index = 0; index < 2; ++index) {
        check(cuCtxSetCurrent(contexts[index]), "activate original guarded cleanup context");
        std::array<unsigned int, 3> data{};
        check(cuMemcpyDtoH(data.data(), allocations[index], sizeof(data)),
              "actual guarded original readback");
        require(data == std::array<unsigned int, 3>{0x12345678U, index == 0 ? 5U : 2U, 0x76543210U},
                "actual original/graph counts and guards");
        check(router.close_context(
                  contexts[index],
                  [&] {
                      const auto stopped = owners[index]->native_epoch_snapshot();
                      require(stopped && stopped->phase == CudaNativeEpochPhase::kComplete &&
                                  stopped->created_events == stopped->destroyed_events,
                              "context retirement joins epoch before original destruction");
                      check(cuStreamDestroy(streams[index]), "original stream cleanup");
                      check(cuStreamDestroy(independent[index]),
                            "unrelated original stream cleanup");
                      check(cuEventDestroy(historical[index]), "historical original event cleanup");
                      check(cuModuleUnload(modules[index]), "original module cleanup");
                      check(cuMemFree(allocations[index]), "original guarded allocation cleanup");
                      return cuCtxDestroy(contexts[index]);
                  }),
              "retire exact process context after epoch shutdown");
        owners[index].reset();
    }
    std::cout << "process_epoch mode=" << mode
              << " originals=6 graph=1 cross_context=1 session=1 capture=1 guards=1 cleanup=1\n";
}
std::atomic<bool> recovery_projection_failure{};
std::atomic<bool> recovery_sync_started{};
std::atomic<bool> recovery_waiter_attempted{};
std::atomic<bool> recovery_waiter_entered{};
std::atomic<unsigned int> recovery_sync_calls{};
std::atomic<CUevent> recovery_failed_event{};
std::atomic<unsigned int> recovery_failed_queries{};
GetProcAddressV2Function recovery_original_lookup{};
CUresult recovery_query(CUevent event) {
    if (recovery_projection_failure.exchange(false)) {
        recovery_failed_event = event;
        ++recovery_failed_queries;
        return CUDA_ERROR_LAUNCH_FAILED;
    }
    if (event == recovery_failed_event.load()) ++recovery_failed_queries;
    return cuEventQuery(event);
}
CUresult recovery_destroy(CUevent event) {
    const auto status = cuEventDestroy(event);
    if (status == CUDA_SUCCESS && event == recovery_failed_event.load())
        recovery_failed_event = nullptr;  // only successful destruction ends this live identity
    return status;
}
CUresult recovery_sync() {
    ++recovery_sync_calls;
    recovery_sync_started = true;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
    while (!recovery_waiter_attempted.load()) {
        require(std::chrono::steady_clock::now() < deadline,
                "bounded excluded process recovery waiter");
        std::this_thread::yield();
    }
    require(!recovery_waiter_entered.load(),
            "process transition excludes another Call before actual SDK proof");
    return cuCtxSynchronize();
}
CUresult recovery_lookup(const char* name, void** output, int version, cuuint64_t flags) {
    CUdriverProcAddressQueryResult found{};
    const auto status = recovery_original_lookup(name, output, version, flags, &found);
    if (status != CUDA_SUCCESS) return status;
    if (found != CU_GET_PROC_ADDRESS_SUCCESS) return CUDA_ERROR_NOT_FOUND;
    if (std::strcmp(name, "cuEventQuery") == 0) *output = reinterpret_cast<void*>(recovery_query);
    if (std::strcmp(name, "cuCtxSynchronize") == 0)
        *output = reinterpret_cast<void*>(recovery_sync);
    if (std::strcmp(name, "cuEventDestroy") == 0)
        *output = reinterpret_cast<void*>(recovery_destroy);
    return CUDA_SUCCESS;
}
void process_epoch_recovery(unsigned int mode) {
    check(cuInit(0), "initialize actual process recovery");
    DriverDispatch original_driver;
    require(original_driver.initialize(), "resolve actual SDK recovery lookup");
    recovery_original_lookup =
        native<GetProcAddressV2Function>(original_driver, "cuGetProcAddress");
    DriverFunctionTable table;
    table.get_proc_address = recovery_lookup;
    table.context_get_current =
        native<ContextGetCurrentFunction>(original_driver, "cuCtxGetCurrent");
    table.context_get_device = native<ContextGetDeviceFunction>(original_driver, "cuCtxGetDevice");
    table.context_synchronize =
        native<ContextSynchronizeFunction>(original_driver, "cuCtxSynchronize");
    table.stream_capture_status =
        native<StreamCaptureFunction>(original_driver, "cuStreamIsCapturing");
    DriverDispatch driver(table);
    ProcessCallGate gate;
    constexpr glimmer::control::NativeEpochExtent extent{8, 2, 2, 1'000'000};
    DriverCooperativeRouter router(driver, {}, false, false, extent);
    std::array<CUcontext, 2> contexts{};
    std::array<CUstream, 2> streams{};
    std::array<CUevent, 2> historical{};
    std::array<CUmodule, 2> modules{};
    std::array<CUfunction, 2> functions{};
    std::array<CUdeviceptr, 2> allocations{};
    std::array<std::shared_ptr<CudaCooperativeFrontend>, 2> owners{};
    CUevent pending{};
    CUdevice device{};
    check(cuDeviceGet(&device, 0), "actual recovery device");
    recovery_projection_failure = false;
    recovery_sync_started = false;
    recovery_waiter_attempted = false;
    recovery_waiter_entered = false;
    recovery_sync_calls = 0;
    recovery_failed_event = nullptr;
    recovery_failed_queries = 0;
    for (unsigned int index = 0; index < 2; ++index) {
        check(cuCtxCreate(&contexts[index], 0, device), "fresh actual recovery context");
        check(cuStreamCreate(&streams[index], CU_STREAM_DEFAULT),
              "recovery original blocking stream");
        check(cuEventCreate(&historical[index], CU_EVENT_DISABLE_TIMING),
              "original recovery historical event");
        check(cuModuleLoadData(&modules[index], kEpochRouterPtx),
              "unchanged recovery original module");
        check(cuModuleGetFunction(&functions[index], modules[index], "increment"),
              "unchanged recovery original kernel");
        check(cuMemAlloc(&allocations[index], 3 * sizeof(unsigned int)),
              "actual recovery guarded allocation");
        const std::array<unsigned int, 3> initial{0x12345678U, 0, 0x76543210U};
        check(cuMemcpyHtoD(allocations[index], initial.data(), sizeof(initial)),
              "initialize recovery guards");
        check(router.frontend(contexts[index], owners[index]),
              "initialize actual recovery frontend");
        check(router.register_event(contexts[index], historical[index]),
              "register original recovery event identity");
        check(owners[index]->record_event(historical[index], streams[index], false),
              "record completed historical prefix");
        check(owners[index]->synchronize_event(historical[index]),
              "complete historical recovery prefix");
    }
    check(cuCtxSetCurrent(contexts[0]), "activate failed original context");
    check(cuEventCreate(&pending, CU_EVENT_DISABLE_TIMING),
          "independent pending original observer");
    CUgraph graph{};
    CUgraphExec executable{};
    check(cuGraphCreate(&graph, 0), "original graph survives recovery");
    auto graph_pointer = allocations[0] + sizeof(unsigned int);
    unsigned long long graph_duration{};
    void* graph_arguments[]{&graph_pointer, &graph_duration};
    CUDA_KERNEL_NODE_PARAMS parameters{};
    parameters.func = functions[0];
    parameters.gridDimX = parameters.gridDimY = parameters.gridDimZ = 1;
    parameters.blockDimX = parameters.blockDimY = parameters.blockDimZ = 1;
    parameters.kernelParams = graph_arguments;
    CUgraphNode node{};
    check(cuGraphAddKernelNode(&node, graph, nullptr, 0, &parameters),
          "unchanged pre-recovery graph node");
    CudaNativeGraphReservation reservation;
    check(router.prepare_graph(contexts[0], graph, reservation, true),
          "prepare surviving original graph identity");
    check(cuGraphInstantiate(&executable, graph, 0), "instantiate surviving original graph");
    check(router.graphs().publish(reservation, executable, CUDA_SUCCESS),
          "publish original graph before recovery");
    const bool is_ptds = mode == 2;
    const auto stream = mode == 0   ? streams[0]
                        : mode == 1 ? CU_STREAM_LEGACY
                                    : CU_STREAM_PER_THREAD;
    const auto original_launch =
        native<decltype(&cuLaunchKernel)>(driver, "cuLaunchKernel", is_ptds);
    std::array<unsigned int, 2> originals{};
    const auto launch = [&](unsigned int index, CUstream actual_stream,
                            unsigned long long duration) {
        auto pointer = allocations[index] + sizeof(unsigned int);
        void* arguments[]{&pointer, &duration};
        ++originals[index];
        return original_launch(functions[index], 1, 1, 1, 1, 1, 1, 0, actual_stream, arguments,
                               nullptr);
    };
    glimmer::control::NativeEpochGrant retained{};
    {
        const ProcessCallGate::Call call(gate);
        check(router.native_execution(
                  gate, owners[0], stream, is_ptds,
                  [&] {
                      const auto expected = owners[0]->native_epoch_snapshot()->grant;
                      require(router.recover_native_epoch(gate, contexts[0], expected) ==
                                  CUDA_ERROR_NOT_PERMITTED,
                              "nested routing recovery rejects before transition or self-wait");
                      recovery_projection_failure = true;
                      return launch(0, stream, 500'000'000);
                  }),
              "original kernel returns without replaced private projection error");
        check(cuEventRecord(pending, stream), "observe actual pending original independently");
        static_cast<void>(owners[0]->close_native_epoch(false));
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
        while (owners[0]->query_event(nullptr) == CUDA_ERROR_INVALID_HANDLE) {
            require(std::chrono::steady_clock::now() < deadline,
                    "bounded failed frontend publication");
            std::this_thread::yield();
        }
        const auto before = owners[0]->native_epoch_snapshot();
        require(before && before->execution->has_ownership &&
                    cuEventQuery(pending) == CUDA_ERROR_NOT_READY,
                "failed process owner retains real pending work");
        retained = before->grant;
        check(cuCtxSetCurrent(contexts[1]), "actual recovery caller keeps another context");
        require(router.native_execution(gate, owners[1], streams[1], false,
                                        [&] { return launch(1, streams[1], 0); }) != CUDA_SUCCESS &&
                    originals[1] == 0,
                "failed target prevents another original before independent stop");
        auto stale = retained;
        ++stale.execution.generation;
        require(router.recover_native_epoch(gate, contexts[0], stale) == CUDA_ERROR_INVALID_VALUE &&
                    recovery_sync_calls == 0,
                "stale process grant performs no SDK recovery");
        const auto unknown = reinterpret_cast<CUcontext>(&device);
        require(
            router.recover_native_epoch(gate, unknown, retained) == CUDA_ERROR_INVALID_CONTEXT &&
                recovery_sync_calls == 0,
            "missing actor cannot be imported or activated by recovery");
        {
            const ProcessCallGate::Transition native_only(gate);
            require(router.recover_native_epoch(gate, contexts[0], retained) ==
                            CUDA_ERROR_NOT_PERMITTED &&
                        recovery_sync_calls == 0,
                    "native-only phase rejects before recursive transition or SDK recovery");
        }
    }
    {
        ProcessCallGate foreign;
        const ProcessCallGate::Call foreign_call(foreign);
        require(
            router.recover_native_epoch(foreign, contexts[0], retained) == CUDA_ERROR_NOT_PERMITTED,
            "foreign process gate cannot recover registered target");
    }
    require(router.recover_native_epoch(gate, contexts[0], retained) == CUDA_ERROR_NOT_PERMITTED &&
                recovery_sync_calls == 0,
            "unentered process gate rejects recovery before SDK effects");
    std::thread waiter([&] {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
        while (!recovery_sync_started.load()) {
            if (std::chrono::steady_clock::now() >= deadline) return;
            std::this_thread::yield();
        }
        recovery_waiter_attempted = true;
        const ProcessCallGate::Call call(gate);
        recovery_waiter_entered = true;
    });
    {
        const ProcessCallGate::Call call(gate);
        const auto failed_queries = recovery_failed_queries.load();
        const auto result = router.recover_native_epoch(gate, contexts[0], retained);
        waiter.join();
        check(result, "actual process recovery proof, failed acknowledgment and actor thaw");
        const auto recovered = owners[0]->native_epoch_snapshot();
        CUcontext actual{};
        check(cuCtxGetCurrent(&actual), "actual recovery caller restoration");
        require(
            actual == contexts[1] && recovered->status == CUDA_ERROR_LAUNCH_FAILED &&
                recovered->execution->has_failed && !recovered->execution->has_ownership &&
                recovered->created_events == recovered->destroyed_events && originals[0] == 1 &&
                recovery_sync_calls == 1 && failed_queries == 1 &&
                recovery_failed_queries == failed_queries && !recovery_failed_event.load() &&
                recovery_waiter_entered,
            "one SDK stop restores caller and original identity without query or original replay");
        recovery_projection_failure = false;
        require(router.recover_native_epoch(gate, contexts[0], retained) != CUDA_SUCCESS &&
                    recovery_sync_calls == 1,
                "repeated process recovery cannot replay physical proof or acknowledgment");
        check(cuCtxSetCurrent(contexts[0]), "activate recovered original context");
        check(cuEventQuery(pending), "original physical prefix actually completed");
        check(owners[0]->query_event(historical[0]),
              "target historical identity survives recovery");
        check(router.native_execution(gate, owners[0], stream, is_ptds,
                                      [&] { return launch(0, stream, 0); }),
              "new original executes once after healthy recovery");
        const auto next = owners[0]->native_epoch_snapshot();
        require(next->grant.execution.task_id != retained.execution.task_id &&
                    next->execution->issued_calls == 1,
                "new work receives a fresh independently authorized grant");
        check(router.native_graph_execution(executable, stream, is_ptds,
                                            [&] { return cuGraphLaunch(executable, stream); }),
              "same original whole graph replays after recovered epoch order");
        check(owners[0]->synchronize_context(),
              "complete original graph and recovered logical order");
        check(cuCtxSetCurrent(contexts[1]), "activate preserved other frontend");
        std::shared_ptr<CudaCooperativeFrontend> same;
        check(router.existing_frontend(contexts[1], same), "look up unchanged other frontend");
        require(same == owners[1], "target recovery never replaces another frontend");
        check(owners[1]->query_event(historical[1]),
              "other historical event identity remains complete");
        check(router.native_execution(gate, owners[1], streams[1], false,
                                      [&] { return launch(1, streams[1], 0); }),
              "shared capacity accepts original work in preserved other context");
        check(owners[1]->synchronize_context(), "complete preserved other original scope");
    }
    for (unsigned int index = 0; index < 2; ++index) {
        check(cuCtxSetCurrent(contexts[index]), "activate exact recovery cleanup target");
        std::array<unsigned int, 3> result{};
        check(cuMemcpyDtoH(result.data(), allocations[index], sizeof(result)),
              "actual recovered guarded readback");
        require(
            result == std::array<unsigned int, 3>{0x12345678U, index == 0 ? 3U : 1U, 0x76543210U},
            "failed, new and whole-graph originals execute once without crossing guards");
        check(router.close_context(
                  contexts[index],
                  [&] {
                      if (index == 0) {
                          check(cuEventDestroy(pending), "destroy original recovery observer");
                          check(cuGraphExecDestroy(executable),
                                "destroy surviving original executable");
                          check(cuGraphDestroy(graph), "destroy surviving original graph");
                      }
                      check(cuEventDestroy(historical[index]),
                            "destroy surviving original historical event");
                      check(cuStreamDestroy(streams[index]), "destroy unchanged original stream");
                      check(cuModuleUnload(modules[index]), "destroy unchanged original module");
                      check(cuMemFree(allocations[index]),
                            "destroy guarded original recovery allocation");
                      return cuCtxDestroy(contexts[index]);
                  }),
              "retire recovered context with checked private and original resource cleanup");
        owners[index].reset();
    }
    std::cout << "process_epoch_recovery mode=" << mode
              << " originals=3 graph=1 sync=1 query_replay=0 other_owner=1 guards=1 cleanup=1\n";
}
}  // namespace
int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--epoch-recovery") {
            for (unsigned int lifetime = 0; lifetime < 2; ++lifetime)
                for (unsigned int mode = 0; mode < 3; ++mode) process_epoch_recovery(mode);
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--native-epochs") {
            for (unsigned int lifetime = 0; lifetime < 2; ++lifetime)
                for (unsigned int mode = 0; mode < 3; ++mode) process_native_epochs(mode);
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--uncertain-graph-issue") {
            issued_graph_record_prefix(true);
            return EXIT_FAILURE;
        }
        if (argc == 2 && std::string_view(argv[1]).starts_with("--root-capture-")) {
            root_capture_fault(argv[1]);
            return EXIT_FAILURE;
        }
        isolated_fault_child("--uncertain-graph-issue");
        for (const auto* mode :
             {"--root-capture-missing", "--root-capture-unmatched-error", "--root-capture-mismatch",
              "--root-capture-duplicate", "--root-capture-swallowed-error"}) {
            isolated_fault_child(mode);
        }
        issued_graph_record_prefix();
        run();
        source_capture_owner();
        source_capture_owner(true);
        source_capture_owner(false, true);
        source_capture_owner(true, true);
        std::cout << "process_native_session physical_pause=1 overlapping_capture=1 "
                     "native_context_proof=1\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
