#include "internal/driver_cooperative_router.h"
#include "internal/driver_api_interceptor.h"
#include "internal/runtime_api_bridge.h"
#include "internal/persistent_kernel.h"
#include "cooperative_ptx_fixture.h"

#include <cuda_runtime_api.h>
#include <dlfcn.h>

#include <array>
#include <algorithm>
#include <atomic>
#include <bit>
#include <cstdint>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <new>
#include <stdexcept>
#include <exception>
#include <thread>
#include <sys/wait.h>
#include <unistd.h>
#include <string_view>
#include <string>
#include <filesystem>
#include <latch>

namespace {
thread_local bool should_reject_metadata_allocation{};
}
// Caller-local injection after an actual SDK effect; workers keep allocating.
void* operator new(std::size_t bytes) {
    if (should_reject_metadata_allocation) {
        throw std::bad_alloc();
    }
    if (void* pointer = std::malloc(bytes ? bytes : 1)) {
        return pointer;
    }
    throw std::bad_alloc();
}
void operator delete(void* pointer) noexcept {
    std::free(pointer);
}
void operator delete(void* pointer, std::size_t) noexcept {
    std::free(pointer);
}

#ifdef GLIMMER_HAS_RUNTIME_CAPTURE_KERNEL
extern "C" const void* glimmer_runtime_capture_kernel();
extern "C" const void* glimmer_runtime_ready_kernel(bool is_second);
extern "C" const void* glimmer_runtime_pending_kernel();
extern "C" const void* glimmer_runtime_graph_dependency_kernel();
#endif

namespace {
using namespace glimmer::interceptor;

void require(bool value, const char* message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}
void check(CUresult status, const char* message) {
    if (status != CUDA_SUCCESS) {
        throw std::runtime_error(std::string(message) + ": " + std::to_string(status));
    }
}
void check(cudaError_t status, const char* message) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(status));
    }
}
template <typename Function>
Function runtime_symbol(const char* name) {
    const auto resolver = resolve_real_dlsym();
    const auto symbol = resolver ? resolver(RTLD_DEFAULT, name) : nullptr;
    require(symbol != nullptr, "required actual Runtime symbol missing");
    return reinterpret_cast<Function>(symbol);
}

void check_source_memory(DriverCooperativeRouter& router) {
    auto& gate = cooperative_process_call_gate();
    const auto before = router.native_execution_stats();
    std::size_t initial_free{}, total{}, remaining{};
    check(cuMemGetInfo(&initial_free, &total), "read capture-time quota baseline");
    require(total == 8388608 && initial_free > 4096, "source memory shares configured quota");
    CUdeviceptr pointer{};
    check(cuMemAlloc(&pointer, 4096), "public capture-time Driver allocation");
    check(cuMemGetInfo(&remaining, &total), "read charged Driver source memory");
    require(initial_free - remaining == 4096, "one exact Driver source allocation charge");
    CUdeviceptr rejected = pointer;
    require(cuMemAlloc(&rejected, total) == CUDA_ERROR_OUT_OF_MEMORY && rejected == pointer,
            "source quota rejection preserves existing Driver output");
    check(cuMemFree(pointer), "public capture-time Driver release");
    check(cuMemGetInfo(&remaining, &total), "read Driver source quota restoration");
    require(remaining == initial_free, "Driver source quota restored once");
    check(cuMemAllocManaged(&pointer, 1024, CU_MEM_ATTACH_GLOBAL),
          "public capture-time managed Driver allocation");
    check(cuMemGetInfo(&remaining, &total), "read managed Driver source charge");
    require(initial_free - remaining == 1024, "exact managed Driver source charge");
    CUdeviceptr native_rejected = pointer;
    CUresult native_error{};
    {
        const NativeDriverCallScope native;
        native_error = cuMemAllocManaged(&native_rejected, 16, 0);
    }
    require(native_error != CUDA_SUCCESS, "original invalid managed flags reject");
    rejected = pointer;
    require(cuMemAllocManaged(&rejected, 16, 0) == native_error && rejected == native_rejected,
            "source managed rejection preserves native status and failed output");
    check(cuMemFree(pointer), "release managed Driver source memory");
    std::size_t pitch{};
    check(cuMemAllocPitch(&pointer, &pitch, 32, 3, 4),
          "public capture-time pitched Driver allocation");
    check(cuMemGetInfo(&remaining, &total), "read pitched Driver source charge");
    require(initial_free - remaining == pitch * 3, "exact original Driver pitch charge");
    check(cuMemFree(pointer), "release pitched Driver source memory");
    void* runtime_pointer{};
    check(cudaMalloc(&runtime_pointer, 2048), "public capture-time Runtime allocation");
    check(cuMemGetInfo(&remaining, &total), "read charged Runtime source memory");
    require(initial_free - remaining == 2048, "Runtime source allocation is not double charged");
    void* runtime_rejected = runtime_pointer;
    require(cudaMalloc(&runtime_rejected, total) == cudaErrorMemoryAllocation &&
                runtime_rejected == runtime_pointer,
            "Runtime source quota rejection preserves its original output");
    check(cudaFree(runtime_pointer), "public capture-time Runtime release");
    check(cudaMallocManaged(&runtime_pointer, 1024, cudaMemAttachGlobal),
          "public capture-time managed Runtime allocation");
    check(cuMemGetInfo(&remaining, &total), "read managed Runtime source charge");
    require(initial_free - remaining == 1024, "one managed Runtime source charge");
    check(cudaFree(runtime_pointer), "release managed Runtime source memory");
    check(cudaMallocPitch(&runtime_pointer, &pitch, 32, 3),
          "public capture-time pitched Runtime allocation");
    check(cuMemGetInfo(&remaining, &total), "read pitched Runtime source charge");
    require(initial_free - remaining == pitch * 3, "one original Runtime pitch charge");
    check(cudaFree(runtime_pointer), "release pitched Runtime source memory");
    cudaPitchedPtr pitched{};
    check(cudaMalloc3D(&pitched, cudaExtent{32, 2, 3}),
          "public capture-time three-dimensional Runtime allocation");
    check(cuMemGetInfo(&remaining, &total), "read three-dimensional Runtime source charge");
    require(initial_free - remaining == pitched.pitch * 2 * 3,
            "one original Runtime three-dimensional pitch charge");
    check(cudaFree(pitched.ptr), "release three-dimensional Runtime source memory");
    CUcontext saved_context{};
    check(cuCtxGetCurrent(&saved_context), "observe context before CPU-only quota rejection");
    {
        const NativeDriverCallScope native;
        check(cuCtxSetCurrent(nullptr), "unbind context before CPU-only quota rejection");
    }
    runtime_rejected = nullptr;
    const auto quota_status = cudaMalloc(&runtime_rejected, total);
    {
        const NativeDriverCallScope native;
        check(cuCtxSetCurrent(saved_context), "restore context after CPU-only quota rejection");
    }
    require(quota_status == cudaErrorMemoryAllocation && !runtime_rejected,
            "CPU-only Runtime quota rejection needs no issued allocation context");
    CudaProcessNativeToken token;
    check(router.source_native_token(gate, token), "CPU-only rejection preserves source owner");
    check(cuMemGetInfo(&remaining, &total), "read Runtime source quota restoration");
    require(remaining == initial_free && gate.is_native_only_on_current_thread() &&
                router.native_execution_stats().grants == before.grants,
            "synchronous source memory neither acquires execution grants nor thaws ownership");
}

enum class CaptureErrorKind { kRepeatedBegin, kContextSync, kDriverAllocation, kRuntimeAllocation };

void check_runtime_error(bool is_ptds, CaptureErrorKind kind = CaptureErrorKind::kRepeatedBegin) {
    const bool is_context_error = kind == CaptureErrorKind::kContextSync;
    const bool is_allocation_error = kind == CaptureErrorKind::kDriverAllocation;
    const bool is_driver_error = is_context_error || is_allocation_error;
    using Begin = cudaError_t (*)(cudaStream_t, cudaStreamCaptureMode);
    using End = cudaError_t (*)(cudaStream_t, cudaGraph_t*);
    using LastError = cudaError_t (*)();
    const auto begin =
        runtime_symbol<Begin>(is_ptds ? "cudaStreamBeginCapture_ptsz" : "cudaStreamBeginCapture");
    const auto end =
        runtime_symbol<End>(is_ptds ? "cudaStreamEndCapture_ptsz" : "cudaStreamEndCapture");
    const auto peek_error = runtime_symbol<LastError>("cudaPeekAtLastError");
    const auto get_error = runtime_symbol<LastError>("cudaGetLastError");
    const auto last_error = [&] {
        return std::array<cudaError_t, 3>{peek_error(), get_error(), peek_error()};
    };
    check(cudaSetDevice(0), "bind Runtime error baseline context");
    cudaError_t baseline_error{};
    CUresult baseline_context_error{};
    cudaError_t baseline_end{};
    std::array<cudaError_t, 3> baseline_last{};
    cudaStream_t stream{};
    {
        // Native baseline precedes creation of any private context owner.
        const NativeDriverCallScope native;
        (void)get_error();
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create baseline stream");
        check(begin(stream, cudaStreamCaptureModeGlobal), "begin original error baseline");
        if (is_driver_error) {
            CUdeviceptr pointer{};
            baseline_context_error =
                is_context_error ? cuCtxSynchronize() : cuMemAlloc(&pointer, 4096);
            require(!pointer, "rejected original capture allocation publishes no pointer");
        } else {
            void* pointer{};
            baseline_error = kind == CaptureErrorKind::kRuntimeAllocation
                                 ? cudaMalloc(&pointer, 4096)
                                 : begin(stream, cudaStreamCaptureModeGlobal);
            require(!pointer, "rejected capture Runtime allocation publishes no pointer");
        }
        baseline_last = last_error();
        cudaGraph_t graph{};
        baseline_end = end(stream, &graph);
        if (graph) {
            check(cudaGraphDestroy(graph), "destroy original error baseline graph");
        }
        check(cudaStreamDestroy(stream), "destroy original error baseline stream");
        (void)get_error();
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
              "create owned error stream");
    }
    require(
        is_driver_error ? baseline_context_error != CUDA_SUCCESS : baseline_error != cudaSuccess,
        "invalid original capture operation must reject");
    DriverDispatch driver;
    require(driver.initialize(), "resolve original error metadata");
    CUcontext context{};
    check(driver.context_get_current(&context), "observe actual error context");
    auto& gate = cooperative_process_call_gate();
    const ProcessCallGate::Call call(gate);
    auto* router = cooperative_source_router();
    require(router, "configured source error owner required");
    std::shared_ptr<CudaCooperativeFrontend> frontend;
    check(router->frontend(context, frontend), "prepare error source owner");
    unsigned int context_lookups{};
    const auto transition = [&](CudaSourceCaptureKind kind, const auto& original,
                                RuntimeCaptureObservation& observation, cudaError_t& status) {
        return router->source_capture_root(gate, kind, [&](auto& dispatch) {
            return invoke_runtime_capture_original(
                [&](CUcontext actual_context, CUstream actual_stream, bool actual_ptds,
                    CudaSourceCaptureKind actual_kind, const auto& native_original) {
                    return dispatch.invoke(actual_context, actual_stream, actual_ptds, actual_kind,
                                           native_original);
                },
                original, observation, status,
                [&](CUstream actual_stream, bool actual_ptds, CUcontext& actual_context) {
                    ++context_lookups;
                    return dispatch.lookup_origin_context(actual_stream, actual_ptds,
                                                          actual_context);
                });
        });
    };
    RuntimeCaptureObservation observation;
    cudaError_t status{cudaErrorUnknown};
    check(transition(
              CudaSourceCaptureKind::kBegin,
              [&] { return begin(stream, cudaStreamCaptureModeGlobal); }, observation, status),
          "owned first begin succeeds");
    check(status, "original first begin succeeds");
    if (is_driver_error) {
        CUdeviceptr pointer{};
        const auto rejected = is_context_error ? cuCtxSynchronize() : cuMemAlloc(&pointer, 4096);
        require(rejected == baseline_context_error && !pointer && last_error() == baseline_last &&
                    gate.is_native_only_on_current_thread(),
                "public source native rejection preserves original error and ownership");
    } else if (kind == CaptureErrorKind::kRuntimeAllocation) {
        void* pointer{};
        require(cudaMalloc(&pointer, 4096) == baseline_error && !pointer &&
                    last_error() == baseline_last && gate.is_native_only_on_current_thread(),
                "public Runtime source allocation preserves native GLOBAL error and last-error");
    } else {
        const auto rejected = transition(
            CudaSourceCaptureKind::kBegin,
            [&] { return begin(stream, cudaStreamCaptureModeGlobal); }, observation, status);
        require(rejected != CUDA_SUCCESS && status == baseline_error &&
                    observation.original_driver_calls == 1 && last_error() == baseline_last &&
                    gate.is_native_only_on_current_thread() && !is_inside_runtime_call() &&
                    !is_inside_driver_call(),
                "matched SDK error preserves native last-error, one issue and retained ownership");
    }
    cudaGraph_t graph{};
    const auto finished = transition(
        CudaSourceCaptureKind::kEnd, [&] { return end(stream, &graph); }, observation, status);
    if (status != baseline_end || (finished == CUDA_SUCCESS) != (status == cudaSuccess) ||
        observation.original_driver_calls != 1 || gate.is_native_only_on_current_thread() ||
        context_lookups != (baseline_end == cudaErrorStreamCaptureInvalidated ? 1U : 0U)) {
        throw std::runtime_error(
            "original end/NONE mismatch: Runtime=" + std::to_string(status) +
            ", baseline=" + std::to_string(baseline_end) + ", owner=" + std::to_string(finished) +
            ", Driver calls=" + std::to_string(observation.original_driver_calls) +
            ", native=" + std::to_string(gate.is_native_only_on_current_thread()));
    }
    {
        const NativeDriverCallScope native;
        if (graph) {
            check(cudaGraphDestroy(graph), "destroy owned error graph");
        }
        check(cudaStreamDestroy(stream), "destroy owned error stream");
        (void)get_error();
    }
    check(router->close_context(context, [] { return CUDA_SUCCESS; }), "close error owner");
}

void run(bool is_ptds, bool uses_null_stream,
         cudaStreamCaptureMode mode = cudaStreamCaptureModeGlobal) {
    check(cudaSetDevice(0), "bind actual Runtime primary context");
    DriverDispatch driver;
    require(driver.initialize(), "initialize original Driver metadata");
    CUcontext context{};
    check(driver.context_get_current(&context), "observe real Runtime context without casting");
    require(context != nullptr, "actual Runtime context required");
    auto& gate = cooperative_process_call_gate();
    require(!cooperative_source_router(), "source owner cannot escape admitted caller lifetime");
    const ProcessCallGate::Call source(gate);
    auto* process_router = cooperative_source_router();
    require(process_router != nullptr, "configured process source owner required");
    auto& router = *process_router;
    std::shared_ptr<CudaCooperativeFrontend> frontend;
    check(router.frontend(context, frontend), "prepare actual private source owner");
    CUstream private_stream{};
    CUmodule module{};
    CUfunction kernel{};
    CUdeviceptr counter{};
    CUdeviceptr mirror{};
    unsigned int* pinned{};
    CUresult invalid_event_flags{};
    CUresult null_event_output{};
    {
        const NativeDriverCallScope native;
        check(cuStreamCreate(&private_stream, CU_STREAM_NON_BLOCKING),
              "create actual private stream");
        check(cuModuleLoadData(&module, kNativeOncePtx), "load original kernel before capture");
        check(cuModuleGetFunction(&kernel, module, "native_once"),
              "resolve unchanged original entry");
        check(cuMemAlloc(&counter, sizeof(unsigned int)), "allocate original counter");
        check(cuMemAlloc(&mirror, sizeof(unsigned int)), "allocate original copy destination");
        check(cuMemAllocHost(reinterpret_cast<void**>(&pinned), 3 * sizeof(unsigned int)),
              "allocate original pinned copy input and outputs");
        pinned[0] = 7;
        pinned[1] = 0;
        pinned[2] = 0;
        check(cuMemsetD32(counter, 0, 1), "initialize original counter");
        CUevent invalid{};
        invalid_event_flags = driver.event_create(&invalid, ~0U);
        require(invalid_event_flags != CUDA_SUCCESS && !invalid,
                "native invalid event flags must reject without a handle");
        null_event_output = driver.event_create(nullptr, 0);
        require(null_event_output != CUDA_SUCCESS, "native null event output must reject");
    }
    check(frontend->synchronize_stream(private_stream, false), "prime owned native prefix");
    cudaStream_t runtime_stream{};
    if (!uses_null_stream) {
        check(cudaStreamCreateWithFlags(&runtime_stream, cudaStreamNonBlocking),
              "create opaque original Runtime stream");
    }
    using Begin = cudaError_t (*)(cudaStream_t, cudaStreamCaptureMode);
    using End = cudaError_t (*)(cudaStream_t, cudaGraph_t*);
    const auto begin =
        runtime_symbol<Begin>(is_ptds ? "cudaStreamBeginCapture_ptsz" : "cudaStreamBeginCapture");
    const auto end =
        runtime_symbol<End>(is_ptds ? "cudaStreamEndCapture_ptsz" : "cudaStreamEndCapture");
    const auto native_before = router.native_execution_stats();
    const auto resolver = resolve_real_dlsym();
    const auto public_launch = reinterpret_cast<decltype(&cuLaunchKernel)>(
        resolver ? resolver(RTLD_DEFAULT, is_ptds ? "cuLaunchKernel_ptsz" : "cuLaunchKernel")
                 : nullptr);
    require(public_launch && is_interceptor_driver_function(reinterpret_cast<void*>(public_launch)),
            "actual public Driver launch wrapper required for source routing");
    const auto public_copy = [&]<typename Function>(const char* legacy, const char* ptds) {
        auto* symbol = resolver ? resolver(RTLD_DEFAULT, is_ptds ? ptds : legacy) : nullptr;
        require(symbol && is_interceptor_driver_function(symbol),
                "actual public Driver copy wrapper required for source routing");
        return reinterpret_cast<Function>(symbol);
    };
    const auto h_to_d = public_copy.operator()<decltype(&cuMemcpyHtoDAsync)>(
        "cuMemcpyHtoDAsync_v2", "cuMemcpyHtoDAsync_v2_ptsz");
    const auto d_to_d = public_copy.operator()<decltype(&cuMemcpyDtoDAsync)>(
        "cuMemcpyDtoDAsync_v2", "cuMemcpyDtoDAsync_v2_ptsz");
    const auto d_to_h = public_copy.operator()<decltype(&cuMemcpyDtoHAsync)>(
        "cuMemcpyDtoHAsync_v2", "cuMemcpyDtoHAsync_v2_ptsz");
    const auto uva =
        public_copy.operator()<decltype(&cuMemcpyAsync)>("cuMemcpyAsync", "cuMemcpyAsync_ptsz");
    const auto record =
        public_copy.operator()<decltype(&cuEventRecord)>("cuEventRecord", "cuEventRecord_ptsz");
    const auto record_flags = public_copy.operator()<decltype(&cuEventRecordWithFlags)>(
        "cuEventRecordWithFlags", "cuEventRecordWithFlags_ptsz");
    const auto wait = public_copy.operator()<decltype(&cuStreamWaitEvent)>(
        "cuStreamWaitEvent", "cuStreamWaitEvent_ptsz");
    const auto copy_and_launch = [&](CUstream stream, unsigned int* output) {
        check(h_to_d(counter, pinned, sizeof(unsigned int), stream), "public source HtoD copy");
        check(d_to_d(mirror, counter, sizeof(unsigned int), stream), "public source DtoD copy");
        check(uva(counter, mirror, sizeof(unsigned int), stream), "public source UVA copy");
        std::array<void*, 1> arguments{&counter};
        check(public_launch(kernel, 1, 1, 1, 1, 1, 1, 0, stream, arguments.data(), nullptr),
              "public source original kernel");
        check(d_to_h(output, counter, sizeof(unsigned int), stream), "public source DtoH copy");
    };
    const auto transition = [&](CudaSourceCaptureKind kind,
                                const std::function<cudaError_t()>& original,
                                RuntimeCaptureObservation& observation) {
        const ProcessCallGate::Call call(gate);
        cudaError_t runtime_status{cudaErrorUnknown};
        unsigned int root_calls{};
        const auto owner_status = router.source_capture_root(gate, kind, [&](auto& dispatch) {
            return invoke_runtime_capture_original(
                [&](CUcontext actual_context, CUstream actual_stream, bool actual_ptds,
                    CudaSourceCaptureKind actual_kind, const auto& native_original) {
                    return dispatch.invoke(actual_context, actual_stream, actual_ptds, actual_kind,
                                           native_original);
                },
                [&] {
                    ++root_calls;
                    require(is_inside_runtime_call() && !is_inside_driver_call(),
                            "actual root Runtime call remains visible to its typed leaf");
                    return original();
                },
                observation, runtime_status,
                [&](CUstream actual_stream, bool actual_ptds, CUcontext& actual_context) {
                    return dispatch.lookup_origin_context(actual_stream, actual_ptds,
                                                          actual_context);
                });
        });
        require(!is_inside_runtime_call() && !is_inside_driver_call() && root_calls == 1 &&
                    observation.was_runtime_invoked,
                "one actual Runtime original and balanced permission scope");
        check(runtime_status, "actual original Runtime capture result");
        check(owner_status, "checked real Runtime capture owner result");
    };
    RuntimeCaptureObservation began;
    transition(CudaSourceCaptureKind::kBegin, [&] { return begin(runtime_stream, mode); }, began);
    require(began.original_driver_calls == 1 && began.context == context &&
                began.is_ptds == is_ptds && !began.graph,
            "one actual typed legacy/PTDS begin, without Runtime handle conversion");
    CUevent captured_event{};
    CUevent native_event{};
    CUstream source_stream{};
    CUstream source_priority_stream{};
    {
        const ProcessCallGate::Call call(gate);
        if (mode == cudaStreamCaptureModeRelaxed) {
            check_source_memory(router);
        }
        check(cuStreamCreate(&source_stream, CU_STREAM_NON_BLOCKING),
              "public native-only stream creation");
        check(cuStreamCreateWithPriority(&source_priority_stream, CU_STREAM_NON_BLOCKING, 0),
              "public native-only priority stream creation");
        check(cuEventCreate(&captured_event, 0), "public native-only captured event creation");
        check(cuEventCreate(&native_event, 0), "public native-only uncaptured event creation");
        CUevent baseline_rejected = native_event;
        require(driver.event_create(&baseline_rejected, ~0U) == invalid_event_flags &&
                    baseline_rejected == native_event,
                "original failed creation preserves the preexisting output handle");
        CUevent rejected_event = native_event;
        require(cuEventCreate(&rejected_event, ~0U) == invalid_event_flags &&
                    rejected_event == native_event &&
                    cuEventCreate(nullptr, 0) == null_event_output,
                "source event creation preserves original errors and unchanged failed output");
        CudaEventProjection projection;
        check(router.event_projection(captured_event, projection), "observe source event metadata");
        require(!projection.is_logical() && !projection.has_native_completion(),
                "native creation publishes no private binding or completion proof");
        CUcontext event_context{};
        std::shared_ptr<CudaCooperativeFrontend> event_owner;
        require(router.event_frontend(captured_event, event_context, event_owner) ==
                        CUDA_ERROR_NOT_READY &&
                    !event_owner,
                "public creation does not bind a private frontend during capture");
        copy_and_launch(began.stream, pinned + 1);
        check(record(captured_event, began.stream), "public captured default event record");
        check(record_flags(captured_event, began.stream, CU_EVENT_RECORD_EXTERNAL),
              "public captured original external event record");
        check(wait(began.stream, captured_event, CU_EVENT_WAIT_EXTERNAL),
              "public captured original external event wait");
        require(router.native_execution_stats().grants == native_before.grants,
                "actual Runtime capture construction acquires no execution grant");
        copy_and_launch(source_stream, pinned + 2);
        check(record(native_event, source_stream), "public uncaptured native event record");
        check(record_flags(native_event, source_stream, 0),
              "public uncaptured native event record with flags");
        check(wait(source_stream, native_event, 0), "public uncaptured native event wait");
        require(router.native_execution_stats().grants == native_before.grants + 8,
                "each uncaptured public copy/launch/record/wait receives one native grant");
        if (mode == cudaStreamCaptureModeRelaxed) {
            // Application-selected RELAXED mode, not a framework mode override.
            check(driver.event_synchronize(native_event), "original native event completion");
            check(cuEventQuery(native_event), "public native-only event query");
            check(cuEventSynchronize(native_event), "public native-only event synchronize");
            check(cuStreamSynchronize(source_stream), "public native-only stream synchronize");
            check(cuStreamQuery(source_stream), "public native-only stream query");
            float milliseconds = -1;
            check(cuEventElapsedTime(&milliseconds, native_event, native_event),
                  "public native-only event timing");
            require(milliseconds == 0 && gate.is_native_only_on_current_thread() &&
                        router.native_execution_stats().grants == native_before.grants + 8,
                    "native metadata does not acquire a grant or thaw active capture");
        }
    }
    RuntimeCaptureObservation ended;
    cudaGraph_t opaque_graph{};
    transition(
        CudaSourceCaptureKind::kEnd, [&] { return end(runtime_stream, &opaque_graph); }, ended);
    require(ended.original_driver_calls == 1 && ended.context == context &&
                ended.stream == began.stream && ended.is_ptds == is_ptds && ended.graph &&
                opaque_graph,
            "actual Driver graph output observed without casting opaque Runtime graph");
    {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (gate.is_native_only_on_current_thread()) {
            const auto status = router.try_finish_source_capture(gate);
            require(status == CUDA_SUCCESS || status == CUDA_ERROR_NOT_READY,
                    "native execution completion permits checked source reconciliation");
            require(std::chrono::steady_clock::now() < deadline,
                    "actual SDK NONE/native completion must eventually restore routing");
            std::this_thread::yield();
        }
        const auto native_after = router.native_execution_stats();
        require(native_after.completed == native_before.completed + 8 &&
                    native_after.failed == native_before.failed,
                "eight actual native completions acknowledged before source thaw");
        require(pinned[1] == 0 && pinned[2] == 8,
                "captured copies have not executed, uncaptured readback completes exactly");
    }
    std::array<CUgraphNode, 7> nodes{};
    std::size_t node_count{};
    check(cuGraphGetNodes(ended.graph, nullptr, &node_count), "count every original graph node");
    require(node_count == nodes.size(), "capture contains no additional private graph node");
    check(cuGraphGetNodes(ended.graph, nodes.data(), &node_count), "inspect actual original graph");
    require(node_count == nodes.size(), "actual SDK capture contains only seven original nodes");
    unsigned int kernel_nodes{};
    unsigned int copy_nodes{};
    unsigned int record_nodes{};
    unsigned int wait_nodes{};
    for (const auto node : nodes) {
        CUgraphNodeType type{};
        check(cuGraphNodeGetType(node, &type), "inspect original graph node type");
        if (type == CU_GRAPH_NODE_TYPE_KERNEL) {
            CUDA_KERNEL_NODE_PARAMS params{};
            check(cuGraphKernelNodeGetParams(node, &params), "inspect original captured kernel");
            require(params.func == kernel && params.gridDimX == 1 && params.blockDimX == 1,
                    "actual SDK capture preserves original function and dimensions");
            ++kernel_nodes;
        } else if (type == CU_GRAPH_NODE_TYPE_MEMCPY) {
            ++copy_nodes;
        } else if (type == CU_GRAPH_NODE_TYPE_EVENT_RECORD) {
            CUevent event{};
            check(cuGraphEventRecordNodeGetEvent(node, &event), "inspect original graph record");
            require(event == captured_event, "capture preserves original recorded event handle");
            ++record_nodes;
        } else if (type == CU_GRAPH_NODE_TYPE_WAIT_EVENT) {
            CUevent event{};
            check(cuGraphEventWaitNodeGetEvent(node, &event), "inspect original graph wait");
            require(event == captured_event, "capture preserves original waited event handle");
            ++wait_nodes;
        }
    }
    require(kernel_nodes == 1 && copy_nodes == 4 && record_nodes == 1 && wait_nodes == 1,
            "capture adds no private node");
    CUgraphExec executable{};
    {
        const NativeDriverCallScope native;
        check(cuGraphInstantiateWithFlags(&executable, ended.graph, 0),
              "instantiate original graph");
    }
    unsigned int original_launches{};
    check(frontend->native(
              private_stream, false,
              [&] {
                  ++original_launches;
                  return driver.graph_launch(executable, private_stream);
              },
              true),
          "admit actual original replay once");
    check(frontend->synchronize_stream(private_stream, false), "prove original replay fence");
    unsigned int value{};
    {
        const NativeDriverCallScope native;
        check(cuMemcpyDtoH(&value, counter, sizeof(value)), "read actual original replay result");
        check(cuGraphExecDestroy(executable), "destroy original executable after physical proof");
        check(cudaGraphDestroy(opaque_graph), "destroy graph through its original Runtime handle");
    }
    require(original_launches == 1 && value == 8 && pinned[1] == 8 && pinned[2] == 8,
            "original replay executes its copies and kernel exactly once");
    check(record(captured_event, source_stream), "fresh public logical source event record");
    CudaEventProjection fresh;
    check(router.event_projection(captured_event, fresh), "inspect fresh public logical event");
    require(fresh.is_logical(),
            "native-created event imports only after thaw and fresh acceptance");
    check(cuEventSynchronize(captured_event), "physically complete fresh public logical record");
    check(cuEventDestroy(captured_event), "destroy checked imported source event");
    check(cuEventDestroy(native_event), "destroy checked native source event");
    check(cuStreamSynchronize(source_priority_stream), "import native-created priority stream");
    check(cuStreamDestroy(source_priority_stream), "destroy imported priority stream after thaw");
    check(cuStreamDestroy(source_stream), "destroy imported execution stream after thaw");
    check(router.close_context(context, [] { return CUDA_SUCCESS; }),
          "close checked private owner");
    frontend.reset();
    {
        const NativeDriverCallScope native;
        check(cuStreamDestroy(private_stream), "destroy private original stream");
        check(cuMemFree(counter), "release original counter");
        check(cuMemFree(mirror), "release original copy destination");
        check(cuMemFreeHost(pinned), "release original pinned copy buffers");
        check(cuModuleUnload(module), "release original module");
        if (runtime_stream) {
            check(cudaStreamDestroy(runtime_stream), "destroy original Runtime stream");
        }
    }
}
#ifdef GLIMMER_HAS_RUNTIME_CAPTURE_KERNEL
std::string source_module_image() {
    return std::string(kCooperativePtx) + R"ptx(
.visible .entry late_priority_probe(.param .u64 marker)
{
    .reg .u64 %pointer;
    ld.param.u64 %pointer, [marker];
    st.global.u32 [%pointer], 2;
    membar.sys;
    ret;
}
)ptx";
}

bool check_source_module_native() {
    check(cuInit(0), "resolve native module baseline dispatch before trusted bypass");
    const NativeDriverCallScope native;
    check(cudaSetDevice(0), "bind native source module baseline context");
    CUstream stream{};
    CUmodule module{};
    CUfunction function{};
    CUdeviceptr marker{};
    std::uint32_t* host_marker{};
    CUgraph graph{};
    check(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING),
          "create original module baseline stream");
    check(cuMemHostAlloc(reinterpret_cast<void**>(&host_marker), 2 * sizeof(std::uint32_t),
                         CU_MEMHOSTALLOC_DEVICEMAP),
          "allocate original module baseline marker");
    check(cuMemHostGetDevicePointer(&marker, host_marker, 0),
          "map original module baseline marker");
    marker += sizeof(std::uint32_t);
    check(cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_RELAXED),
          "begin original module baseline capture");
    auto image = source_module_image();
    const auto loaded = cuModuleLoadData(&module, image.c_str());
    image.assign(image.size(), 'x');
    const auto looked_up =
        loaded == CUDA_SUCCESS ? cuModuleGetFunction(&function, module, "priority_probe") : loaded;
    std::array<void*, 1> parameters{&marker};
    const auto launched =
        looked_up == CUDA_SUCCESS
            ? cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, stream, parameters.data(), nullptr)
            : looked_up;
    const auto ended = cuStreamEndCapture(stream, &graph);
    std::cout << "source_module_native load=" << loaded << " lookup=" << looked_up
              << " kernel=" << launched << " end=" << ended << '\n';
    if (graph) {
        check(cuGraphDestroy(graph), "destroy original module baseline graph");
    }
    if (module) {
        check(cuModuleUnload(module), "destroy original module baseline module");
    }
    check(cuMemFreeHost(host_marker), "release original module baseline marker");
    check(cuStreamDestroy(stream), "release original module baseline stream");
    return loaded == CUDA_SUCCESS && looked_up == CUDA_SUCCESS && launched == CUDA_SUCCESS &&
           ended == CUDA_SUCCESS;
}

struct RuntimeEventLifetimeResult {
    cudaError_t created{};
    cudaError_t destroyed{};
    cudaError_t rejected_creation{};
    std::array<cudaError_t, 3> rejection_last_error{};
    bool was_failed_output_unchanged{};
    std::array<cudaError_t, 3> last_error{};
    CUresult ended{};
};

RuntimeEventLifetimeResult check_runtime_source_event_lifetime(bool is_native,
                                                               bool is_capture_created,
                                                               bool is_driver_created = false,
                                                               bool is_context_unbound = false,
                                                               bool is_creation_unbound = false,
                                                               bool uses_default_create = false) {
    check(cuInit(0), "initialize Runtime event lifetime dispatch");
    DriverDispatch driver;
    require(driver.initialize(), "resolve typed Runtime event lifetime context");
    CUcontext context{};
    CUstream stream{};
    {
        const NativeDriverCallScope native;
        check(cudaSetDevice(0), "bind Runtime event lifetime context");
        check(driver.context_get_current(&context), "observe actual Runtime event context");
        check(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), "create typed event capture stream");
        (void)cudaGetLastError();
    }
    cudaEvent_t event{};
    const auto create = [&] {
        if (is_driver_created) {
            // SDK-documented identical event types; no opaque-handle cast.
            check(cuEventCreate(&event, CU_EVENT_DISABLE_TIMING), "create actual Driver event");
            return cudaSuccess;
        }
        return uses_default_create ? cudaEventCreate(&event)
                                   : cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
    };
    RuntimeEventLifetimeResult result;
    if (!is_capture_created) {
        if (is_native) {
            const NativeDriverCallScope native;
            result.created = create();
        } else {
            result.created = create();
        }
        check(result.created, "create ordinary Runtime event before capture");
    }
    auto& gate = cooperative_process_call_gate();
    const ProcessCallGate::Call call(gate);
    auto* router = is_native ? nullptr : cooperative_source_router();
    std::shared_ptr<CudaCooperativeFrontend> frontend;
    CudaPausedNativeStats before{};
    const auto begin = [&] { return cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_RELAXED); };
    if (is_native) {
        const NativeDriverCallScope native;
        check(begin(), "begin native Runtime event lifetime capture");
    } else {
        require(router, "configured Runtime event source owner");
        check(router->frontend(context, frontend), "prepare Runtime event source owner");
        before = router->native_execution_stats();
        check(router->source_capture_transition(gate, context, stream, false,
                                                CudaSourceCaptureKind::kBegin, begin),
              "retain Runtime event source capture");
    }
    const auto lifetime = [&] {
        if (is_creation_unbound) {
            const NativeDriverCallScope native;
            check(cuCtxSetCurrent(nullptr), "unbind caller before original Runtime creation");
        }
        if (is_capture_created) {
            if (!is_driver_created) {
                cudaEvent_t rejected{};
                result.rejected_creation = cudaEventCreateWithFlags(&rejected, 0x80000000U);
                result.was_failed_output_unchanged = rejected == nullptr;
                result.rejection_last_error = {cudaPeekAtLastError(), cudaGetLastError(),
                                               cudaPeekAtLastError()};
                require(result.rejected_creation != cudaSuccess,
                        "original SDK invalid event flags must reject");
            }
            if (!is_native && !is_driver_created) {
                unsigned int originals{};
                cudaEvent_t failed_output;
                require(
                    intercept_cooperative_runtime_event_create(&failed_output, cudaEventDefault,
                                                               [&] {
                                                                   ++originals;
                                                                   return cudaErrorApiFailureBase;
                                                               }) == cudaErrorApiFailureBase &&
                        originals == 1,
                    "failed Runtime creation preserves error without reading unset output");
                require(intercept_cooperative_runtime_event_create(&failed_output,
                                                                   cudaEventInterprocess,
                                                                   [&] {
                                                                       ++originals;
                                                                       return cudaSuccess;
                                                                   }) == cudaErrorNotSupported &&
                            originals == 1,
                        "unsupported IPC cannot borrow a Runtime source creation permission");
            }
            result.created = create();
            if (!is_native && !is_driver_created && result.created == cudaSuccess) {
                CudaEventProjection created;
                check(router->event_projection(event, created),
                      "actual Runtime root creation publishes exact source registration");
                require(created.context() == context && !created.is_logical() &&
                            !created.has_native_completion(),
                        "SDK lazy context supplies native owner, not logical completion");
            }
        }
        if (is_context_unbound) {
            const NativeDriverCallScope native;
            check(cuCtxSetCurrent(nullptr), "unbind caller before original Runtime destroy");
        }
        if (!is_native && is_driver_created) {
            CudaEventProjection before_destroy;
            check(router->event_projection(event, before_destroy),
                  "snapshot Runtime destroy owner");
            unsigned int originals{};
            require(intercept_cooperative_runtime_event_destroy(event,
                                                                [&] {
                                                                    ++originals;
                                                                    return cudaErrorApiFailureBase;
                                                                }) == cudaErrorApiFailureBase &&
                        originals == 1,
                    "Runtime failure outside CUresult range remains exact and is not replayed");
            CudaEventProjection retained;
            check(router->refresh_event_projection(before_destroy, retained),
                  "failed original Runtime destruction preserves exact owner");
            require(router->source_event_destroy(
                        gate, event,
                        [&] {
                            ++originals;
                            return CUDA_SUCCESS;
                        },
                        CudaSourceMemoryOrigin::kRuntime) == CUDA_ERROR_INVALID_VALUE &&
                        originals == 1,
                    "missing borrowed Runtime issue proof rejects before original call");
        }
        if (result.created == cudaSuccess) {
            result.destroyed = cudaEventDestroy(event);
        }
        result.last_error = {cudaPeekAtLastError(), cudaGetLastError(), cudaPeekAtLastError()};
        if (is_context_unbound) {
            CUcontext after{};
            check(driver.context_get_current(&after), "observe native destroy context requirement");
            std::cout << "runtime_event_destroy_context native=" << is_native
                      << " had_current_after=" << static_cast<bool>(after)
                      << " destroy=" << result.destroyed << std::endl;
            const NativeDriverCallScope native;
            check(cuCtxSetCurrent(context), "restore caller for actual capture end");
        }
    };
    if (is_native) {
        const NativeDriverCallScope native;
        lifetime();
    } else {
        lifetime();
        if ((is_driver_created || is_capture_created) && result.destroyed == cudaSuccess) {
            CudaEventProjection retired;
            if (router->event_projection(event, retired) != CUDA_ERROR_INVALID_HANDLE) {
                std::cerr << "Runtime destroy succeeded but exact source event owner survived\n";
                std::_Exit(EXIT_FAILURE);
            }
        }
        require(gate.is_native_only_on_current_thread() &&
                    router->native_execution_stats().grants == before.grants,
                "Runtime event lifetime cannot thaw or acquire a construction grant");
    }
    CUgraph graph{};
    if (is_native) {
        const NativeDriverCallScope native;
        result.ended = cuStreamEndCapture(stream, &graph);
    } else {
        result.ended = router->source_capture_transition(
            gate, context, stream, false, CudaSourceCaptureKind::kEnd,
            [&] { return cuStreamEndCapture(stream, &graph); });
        if (gate.is_native_only_on_current_thread()) {
            std::cerr << "Runtime event lifetime retained failed source ownership\n";
            std::_Exit(EXIT_FAILURE);
        }
        check(router->close_context(context, [] { return CUDA_SUCCESS; }),
              "close proven Runtime event private owner");
        frontend.reset();
    }
    {
        const NativeDriverCallScope native;
        if (graph) {
            std::size_t count{};
            check(cuGraphGetNodes(graph, nullptr, &count), "count original Runtime event graph");
            require(count == 0, "event lifetime metadata cannot add private graph nodes");
            check(cuGraphDestroy(graph), "destroy original Runtime event graph");
        }
        check(cuStreamDestroy(stream), "release Runtime event capture stream");
    }
    std::cout << "runtime_event_lifetime native=" << is_native
              << " capture_created=" << is_capture_created << " create=" << result.created
              << " driver_created=" << is_driver_created << " destroy=" << result.destroyed
              << " creation_unbound=" << is_creation_unbound
              << " default_create=" << uses_default_create
              << " rejected_creation=" << result.rejected_creation << " end=" << result.ended
              << " last=" << result.last_error[0] << ',' << result.last_error[1] << ','
              << result.last_error[2] << '\n';
    return result;
}

struct EventRetirementResult {
    CUresult destroy{};
    CUresult end{};
    CUresult instantiate{CUDA_ERROR_NOT_READY};
    CUresult launch{CUDA_ERROR_NOT_READY};
    bool has_graph{};
};

EventRetirementResult check_source_event_retirement(bool is_native, bool has_record,
                                                    bool is_native_created,
                                                    bool should_corrupt_retirement = false) {
    check(cuInit(0), "initialize source event retirement dispatch");
    DriverDispatch driver;
    require(driver.initialize(), "resolve event retirement metadata");
    CUcontext context{};
    CUstream stream{};
    CUmodule module{};
    CUfunction function{};
    CUevent event{};
    std::uint32_t* host_marker{};
    CUdeviceptr marker{};
    {
        const NativeDriverCallScope native;
        check(cudaSetDevice(0), "bind event retirement context");
        check(driver.context_get_current(&context), "observe event retirement context");
        check(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), "create event retirement stream");
        check(cuModuleLoadData(&module, kCooperativePtx), "load original event retirement module");
        check(cuModuleGetFunction(&function, module, "priority_probe"),
              "get original event retirement function");
        check(cuMemHostAlloc(reinterpret_cast<void**>(&host_marker), sizeof(std::uint32_t),
                             CU_MEMHOSTALLOC_DEVICEMAP),
              "allocate event retirement marker");
        check(cuMemHostGetDevicePointer(&marker, host_marker, 0), "map event retirement marker");
    }
    *host_marker = 0;
    if (!is_native_created) {
        if (is_native) {
            const NativeDriverCallScope native;
            check(cuEventCreate(&event, CU_EVENT_DISABLE_TIMING), "create native baseline event");
        } else {
            check(cuEventCreate(&event, CU_EVENT_DISABLE_TIMING), "create bound source event");
        }
    }
    auto& gate = cooperative_process_call_gate();
    const ProcessCallGate::Call call(gate);
    auto* router = is_native ? nullptr : cooperative_source_router();
    std::shared_ptr<CudaCooperativeFrontend> frontend;
    CudaPausedNativeStats before{};
    if (!is_native) {
        require(router, "configured source event retirement owner required");
        check(router->frontend(context, frontend), "prepare source event retirement owner");
        before = router->native_execution_stats();
    }
    const auto begin = [&] { return cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_RELAXED); };
    if (is_native) {
        const NativeDriverCallScope native;
        check(begin(), "begin native event retirement capture");
    } else {
        check(router->source_capture_transition(gate, context, stream, false,
                                                CudaSourceCaptureKind::kBegin, begin),
              "retain event retirement source pause");
    }
    const auto retired_event = [&] {
        if (is_native_created) {
            check(cuEventCreate(&event, CU_EVENT_DISABLE_TIMING),
                  "create capture-time source event");
        }
        std::array<void*, 1> parameters{&marker};
        check(cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, stream, parameters.data(), nullptr),
              "capture original kernel before event retirement");
        if (has_record) {
            check(cuEventRecordWithFlags(event, stream, CU_EVENT_RECORD_EXTERNAL),
                  "capture original external record before event retirement");
        }
        if (!is_native) {
            CudaEventProjection owned;
            check(router->event_projection(event, owned), "snapshot exact source event owner");
            unsigned int rejected_calls{};
            require(router->source_event_destroy(gate, event,
                                                 [&] {
                                                     ++rejected_calls;
                                                     return CUDA_ERROR_LAUNCH_FAILED;
                                                 }) == CUDA_ERROR_LAUNCH_FAILED &&
                        rejected_calls == 1,
                    "failed original destruction preserves exact status without replay");
            CudaEventProjection retained;
            check(router->refresh_event_projection(owned, retained),
                  "failed original destruction retains the same event registration");
            if (should_corrupt_retirement) {
                CudaProcessNativeToken cached;
                check(router->source_native_token(gate, cached), "cache source event capability");
                unsigned int original_calls{};
                const auto status = router->source_event_destroy(gate, event, [&] {
                    ++original_calls;
                    check(driver.event_destroy(event), "destroy actual source event once");
                    router->forget_event(event);  // Inject a lost CPU commit owner.
                    return CUDA_SUCCESS;
                });
                require(status == CUDA_ERROR_INVALID_HANDLE && original_calls == 1 &&
                            gate.is_native_only_on_current_thread() &&
                            router->try_finish_source_capture(gate) == status,
                        "post-destroy CPU failure retains the failed physical owner");
                require(router->source_event_destroy(gate, event,
                                                     [&] {
                                                         ++original_calls;
                                                         return CUDA_SUCCESS;
                                                     }) == status &&
                            original_calls == 1,
                        "failed owner cannot retry successful original destruction");
                require(router->native_session_execution(
                            cached, context, stream, false,
                            [&] {
                                ++original_calls;
                                return CUDA_SUCCESS;
                            },
                            [](const auto& issue) { return issue(); }, &gate) == status &&
                            original_calls == 1,
                        "cached execution capability cannot issue after CPU retirement failure");
                // Intentional physical quarantine; no clean teardown is asserted.
                std::_Exit(EXIT_SUCCESS);
            }
        }
        return cuEventDestroy(event);
    };
    CUgraph graph{};
    CUgraphExec executable{};
    EventRetirementResult result;
    if (is_native) {
        const NativeDriverCallScope native;
        result.destroy = retired_event();
        result.end = cuStreamEndCapture(stream, &graph);
    } else {
        result.destroy = retired_event();
        if (result.destroy == CUDA_ERROR_NOT_READY) {
            std::cerr << "public source event destroy rejected: " << result.destroy << '\n';
            std::cerr.flush();
            std::_Exit(EXIT_FAILURE);
        }
        CudaEventProjection retired;
        require(router->event_projection(event, retired) == CUDA_ERROR_INVALID_HANDLE && !retired,
                "successful destruction removes the exact event registration during pause");
        require(gate.is_native_only_on_current_thread() &&
                    router->native_execution_stats().grants == before.grants && *host_marker == 0,
                "event retirement cannot thaw, issue captured work or obtain a grant");
        result.end = router->source_capture_transition(
            gate, context, stream, false, CudaSourceCaptureKind::kEnd,
            [&] { return cuStreamEndCapture(stream, &graph); });
        require(!gate.is_native_only_on_current_thread(),
                "event retirement end requires actual NONE/context proof");
    }
    result.has_graph = graph != nullptr;
    {
        const NativeDriverCallScope native;
        if (graph) {
            std::size_t count{};
            check(cuGraphGetNodes(graph, nullptr, &count), "count original event retirement nodes");
            require(count == (has_record ? 2U : 1U), "event retirement cannot add private nodes");
            // Capture success does not prove retained event ownership. The
            // native 560.35.05 driver crashes when this destroyed event is
            // instantiated; this fixture cannot claim that lifetime support.
            if (!has_record) {
                result.instantiate = cuGraphInstantiateWithFlags(&executable, graph, 0);
            }
            if (result.instantiate == CUDA_SUCCESS) {
                if (is_native) {
                    result.launch = driver.graph_launch(executable, stream);
                } else {
                    result.launch = frontend->native(
                        stream, false, [&] { return driver.graph_launch(executable, stream); },
                        true);
                    check(frontend->synchronize_stream(stream, false),
                          "prove admitted event retirement replay fence");
                }
                if (result.launch == CUDA_SUCCESS) {
                    check(cuStreamSynchronize(stream), "prove original event retirement replay");
                    require(*host_marker == 2, "original retirement graph kernel executes once");
                }
            }
        }
    }
    if (router) {
        CUevent fresh{};
        check(cuEventCreate(&fresh, CU_EVENT_DISABLE_TIMING), "create fresh post-retirement event");
        check(cuEventRecord(fresh, stream), "accept fresh logical event after source proof");
        check(cuEventSynchronize(fresh), "prove fresh post-retirement event record");
        check(cuEventDestroy(fresh), "retire fresh independent event registration");
        check(router->close_context(context, [] { return CUDA_SUCCESS; }),
              "close proven event retirement owner");
        frontend.reset();
    }
    {
        const NativeDriverCallScope native;
        if (executable) {
            check(cuGraphExecDestroy(executable), "destroy original event retirement executable");
        }
        if (graph) {
            check(cuGraphDestroy(graph), "destroy original event retirement graph");
        }
        if (result.destroy != CUDA_SUCCESS) {
            check(cuEventDestroy(event), "release failed-retirement event after proof");
        }
        check(cuModuleUnload(module), "release original event retirement module");
        check(cuMemFreeHost(host_marker), "release event retirement marker");
        check(cuStreamDestroy(stream), "release event retirement stream");
    }
    std::cout << "source_event_retirement native=" << is_native << " record=" << has_record
              << " native_created=" << is_native_created << " destroy=" << result.destroy
              << " end=" << result.end << " graph=" << result.has_graph
              << " instantiate=" << result.instantiate << " launch=" << result.launch << '\n';
    return result;
}

constexpr auto kHeldPrefixPtx = R"ptx(
.version 7.0
.target sm_70
.address_size 64
.visible .entry hold_event_prefix(.param .u64 state) {
    .reg .u64 %rd<2>;
    .reg .u32 %r<2>;
    .reg .pred %p;
    ld.param.u64 %rd0, [state];
    add.u64 %rd1, %rd0, 4;
    st.volatile.global.u32 [%rd1], 1;
    membar.sys;
HELD:
    ld.volatile.global.u32 %r0, [%rd0];
    setp.eq.u32 %p, %r0, 0;
    @%p bra HELD;
    ret;
}
)ptx";

void check_pending_source_event_retirement(bool is_native, bool uses_runtime_destroy = false,
                                           bool is_stream_destroy = false,
                                           bool has_instrumentation_release = false) {
    DriverDispatch driver;
    require(driver.initialize(), "resolve pending source event dispatch");
    CUcontext context{};
    CUstream captured{};
    CUstream execution{};
    CUmodule module{};
    CUfunction hold{};
    std::uint32_t* flags{};
    CUdeviceptr mapped{};
    {
        const NativeDriverCallScope native;
        check(cudaSetDevice(0), "bind pending event retirement context");
        check(driver.context_get_current(&context), "observe pending event retirement context");
        check(cuStreamCreate(&captured, CU_STREAM_NON_BLOCKING), "create pending capture stream");
        check(cuStreamCreate(&execution, CU_STREAM_NON_BLOCKING),
              "create pending execution stream");
        check(cuModuleLoadData(&module, kHeldPrefixPtx), "load original event-prefix hold kernel");
        check(cuModuleGetFunction(&hold, module, "hold_event_prefix"), "resolve event-prefix hold");
        check(cuMemHostAlloc(reinterpret_cast<void**>(&flags), 2 * sizeof(std::uint32_t),
                             CU_MEMHOSTALLOC_DEVICEMAP),
              "allocate original pending event business state");
        check(cuMemHostGetDevicePointer(&mapped, flags, 0), "map pending event business state");
    }
    flags[0] = flags[1] = 0;
    struct ReleaseHeldPrefix {
        std::uint32_t* flags;
        ~ReleaseHeldPrefix() {
            if (flags) {
                std::atomic_ref(flags[0]).store(1, std::memory_order_release);
            }
        }
    } release{flags};
    CUevent event{};
    if (is_native) {
        const NativeDriverCallScope native;
        check(cuEventCreate(&event, CU_EVENT_DISABLE_TIMING), "create native pending event");
    } else {
        check(cuEventCreate(&event, CU_EVENT_DISABLE_TIMING), "create bound pending event");
    }
    auto& gate = cooperative_process_call_gate();
    const ProcessCallGate::Call call(gate);
    auto* router = is_native ? nullptr : cooperative_source_router();
    std::shared_ptr<CudaCooperativeFrontend> frontend;
    CudaPausedNativeStats before{};
    const auto begin = [&] {
        return cuStreamBeginCapture(captured, CU_STREAM_CAPTURE_MODE_RELAXED);
    };
    if (is_native) {
        const NativeDriverCallScope native;
        check(begin(), "begin original pending event capture");
    } else {
        require(router, "configured pending event source owner");
        check(router->frontend(context, frontend), "prepare pending event private owner");
        if (is_stream_destroy) {
            check(frontend->synchronize_stream(execution, false),
                  "bind and drain original pending stream before source pause");
        }
        before = router->native_execution_stats();
        check(router->source_capture_transition(gate, context, captured, false,
                                                CudaSourceCaptureKind::kBegin, begin),
              "retain pending event source capture");
    }
    const auto issue = [&] {
        std::array<void*, 1> parameters{&mapped};
        check(
            driver.launch_kernel(hold, 1, 1, 1, 1, 1, 1, 0, execution, parameters.data(), nullptr),
            "issue original pending event producer");
        return driver.event_record(event, execution);
    };
    if (is_native) {
        check(issue(), "record original event behind held business prefix");
    } else {
        CudaProcessNativeToken token;
        check(router->source_native_token(gate, token), "observe pending source capability");
        // Internal multi-call original issue fixture, not public graph activation.
        check(router->native_session_execution(
                  token, context, execution, false, issue,
                  [](const auto& operation) { return operation(); }, &gate),
              "admit exact pending native event prefix once");
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (std::atomic_ref(flags[1]).load(std::memory_order_acquire) == 0) {
        require(std::chrono::steady_clock::now() < deadline, "actual event producer must start");
        std::this_thread::yield();
    }
    require(driver.event_query(event) == CUDA_ERROR_NOT_READY,
            "destroy regression requires an actually pending original event");
    // Device instrumentation can wait inside the original pending-stream destroy.
    // Keep the strict return-before-release hardware test separate: this optional
    // original/Source comparison checks memory and fence safety, not latency.
    require(!has_instrumentation_release || is_stream_destroy,
            "independent instrumentation release is stream-destruction specific");
    std::jthread instrumentation_release;
    if (has_instrumentation_release) {
        instrumentation_release = std::jthread([flags] {
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
            std::atomic_ref(flags[0]).store(1, std::memory_order_release);
        });
    }
    const auto destroy_begin = std::chrono::steady_clock::now();
    if (is_native) {
        const NativeDriverCallScope native;
        if (is_stream_destroy) {
            if (uses_runtime_destroy) {
                check(cudaStreamDestroy(execution), "destroy actual pending native Runtime stream");
            } else {
                check(driver.stream_destroy(execution), "destroy actual pending native stream");
            }
        } else if (uses_runtime_destroy) {
            check(cudaEventDestroy(event), "destroy actual pending native Runtime event");
        } else {
            check(driver.event_destroy(event), "destroy actual pending native event");
        }
    } else {
        if (is_stream_destroy) {
            if (uses_runtime_destroy) {
                check(cudaStreamDestroy(execution),
                      "public Runtime destroys pending source stream");
            } else {
                check(cuStreamDestroy(execution), "public Driver destroys pending source stream");
            }
        } else if (uses_runtime_destroy) {
            check(cudaEventDestroy(event), "public Runtime source destroy actual pending event");
        } else {
            check(cuEventDestroy(event), "public source destroy actual pending event");
        }
        const auto after = router->native_execution_stats();
        require(after.grants == before.grants + 1 && after.completed == before.completed &&
                    after.retained == before.retained + 1 &&
                    gate.is_native_only_on_current_thread(),
                "pending event destruction neither completes its fence nor releases capacity");
    }
    const bool returns_before_release =
        std::atomic_ref(flags[0]).load(std::memory_order_acquire) == 0;
    const auto destroy_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - destroy_begin)
                                .count();
    require(has_instrumentation_release || returns_before_release,
            "successful destruction must return before business prefix release");
    if (instrumentation_release.joinable()) {
        instrumentation_release.join();
    }
    CUgraph graph{};
    if (is_native) {
        const NativeDriverCallScope native;
        check(cuStreamEndCapture(captured, &graph), "end original pending event capture");
    } else {
        check(router->source_capture_transition(
                  gate, context, captured, false, CudaSourceCaptureKind::kEnd,
                  [&] { return cuStreamEndCapture(captured, &graph); }),
              "end source capture while event's private prefix is still pending");
        require(has_instrumentation_release || gate.is_native_only_on_current_thread(),
                "capture end alone cannot thaw pending event execution");
    }
    std::atomic_ref(flags[0]).store(1, std::memory_order_release);
    if (router) {
        while (gate.is_native_only_on_current_thread()) {
            const auto status = router->try_finish_source_capture(gate);
            require(status == CUDA_SUCCESS || status == CUDA_ERROR_NOT_READY,
                    "destroyed user event cannot break private trailing fence proof");
            require(std::chrono::steady_clock::now() < deadline, "pending event proof must finish");
            std::this_thread::yield();
        }
        const auto after = router->native_execution_stats();
        require(after.completed == before.completed + 1 && after.failed == before.failed &&
                    after.retained == before.retained,
                "private fence acknowledges exactly its original pending execution");
        if (is_stream_destroy) {
            check(cuEventDestroy(event), "retire still-live user event after source stream proof");
        }
        check(router->close_context(context, [] { return CUDA_SUCCESS; }),
              "close pending event owner after actual proof");
        frontend.reset();
    }
    {
        const NativeDriverCallScope native;
        if (is_stream_destroy) {
            check(cuCtxSynchronize(), "prove destroyed stream prefix without querying its handle");
            if (is_native) {
                check(cuEventDestroy(event), "destroy still-live native event after stream proof");
            }
        } else {
            check(cuStreamSynchronize(execution),
                  "prove original pending event producer completion");
        }
        check(cuGraphDestroy(graph), "destroy original empty pending event capture");
        check(cuModuleUnload(module), "release original event-prefix hold module");
        release.flags = nullptr;
        check(cuMemFreeHost(flags), "release physically proven pending event business state");
        if (!is_stream_destroy) {
            check(cuStreamDestroy(execution), "destroy original pending execution stream");
        }
        check(cuStreamDestroy(captured), "destroy original pending capture stream");
    }
    std::cout << "source_event_pending native=" << is_native
              << " runtime_destroy=" << uses_runtime_destroy
              << " stream_destroy=" << is_stream_destroy
              << " instrumentation_release=" << has_instrumentation_release
              << " destroy_before_release=" << returns_before_release
              << " destroy_ms=" << destroy_ms << " exact_private_completion=1" << std::endl;
}

struct ModuleRetirementResult {
    CUresult unload{};
    CUresult end{};
    bool has_graph{};
};

ModuleRetirementResult check_source_module_retirement(bool is_native) {
    check(cuInit(0), "initialize source module retirement dispatch");
    DriverDispatch driver;
    require(driver.initialize(), "resolve actual module retirement metadata");
    CUcontext context{};
    CUstream stream{};
    CUmodule original_module{};
    CUfunction original_function{};
    CUdeviceptr marker{};
    std::uint32_t* host_marker{};
    {
        const NativeDriverCallScope native;
        check(cudaSetDevice(0), "bind original module retirement context");
        check(driver.context_get_current(&context), "observe module retirement context");
        check(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), "create module retirement stream");
        check(cuModuleLoadData(&original_module, kCooperativePtx),
              "load retained original capture module");
        check(cuModuleGetFunction(&original_function, original_module, "priority_probe"),
              "get retained original capture function");
        check(cuMemHostAlloc(reinterpret_cast<void**>(&host_marker), 2 * sizeof(std::uint32_t),
                             CU_MEMHOSTALLOC_DEVICEMAP),
              "allocate module retirement marker");
        check(cuMemHostGetDevicePointer(&marker, host_marker, 0), "map module retirement marker");
    }
    host_marker[0] = host_marker[1] = 0;
    marker += sizeof(std::uint32_t);
    auto& gate = cooperative_process_call_gate();
    const ProcessCallGate::Call call(gate);
    auto* router = is_native ? nullptr : cooperative_source_router();
    std::shared_ptr<CudaCooperativeFrontend> frontend;
    CudaPausedNativeStats before{};
    if (!is_native) {
        require(router, "configured source module retirement owner required");
        check(router->frontend(context, frontend), "prepare source module retirement owner");
        before = router->native_execution_stats();
    }
    const auto begin = [&] { return cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_RELAXED); };
    if (is_native) {
        const NativeDriverCallScope native;
        check(begin(), "begin native module retirement capture");
    } else {
        check(router->source_capture_transition(gate, context, stream, false,
                                                CudaSourceCaptureKind::kBegin, begin),
              "retain source module retirement pause");
    }
    CUmodule retired{};
    CUfunction retired_function{};
    CUgraph graph{};
    ModuleRetirementResult result;
    const auto construction = [&] {
        check(cuModuleLoadData(&retired, kCooperativePtx), "load unused source retirement module");
        check(cuModuleGetFunction(&retired_function, retired, "cooperative"),
              "get original function before source retirement");
        std::array<void*, 1> parameters{&marker};
        check(cuLaunchKernel(original_function, 1, 1, 1, 1, 1, 1, 0, stream, parameters.data(),
                             nullptr),
              "capture retained original kernel before module retirement");
        result.unload = cuModuleUnload(retired);
    };
    if (is_native) {
        const NativeDriverCallScope native;
        construction();
        result.end = cuStreamEndCapture(stream, &graph);
    } else {
        construction();
        if (result.unload == CUDA_ERROR_NOT_READY) {
            std::cerr << "public source module unload rejected: " << result.unload << '\n';
            std::cerr.flush();
            std::_Exit(EXIT_FAILURE);
        }
        require(gate.is_native_only_on_current_thread() &&
                    router->native_execution_stats().grants == before.grants && host_marker[1] == 0,
                "module retirement cannot thaw, issue captured work or acquire a grant");
        result.end = router->source_capture_transition(
            gate, context, stream, false, CudaSourceCaptureKind::kEnd,
            [&] { return cuStreamEndCapture(stream, &graph); });
        require(!gate.is_native_only_on_current_thread(),
                "module retirement end needs actual NONE/context proof before normal routing");
    }
    result.has_graph = graph != nullptr;
    if (graph) {
        const NativeDriverCallScope native;
        std::size_t count{};
        check(cuGraphGetNodes(graph, nullptr, &count), "count original retirement graph nodes");
        require(count == 1, "module retirement adds no private graph nodes");
        CUgraphNode node{};
        check(cuGraphGetNodes(graph, &node, &count), "get original retirement graph node");
        CUDA_KERNEL_NODE_PARAMS parameters{};
        check(cuGraphKernelNodeGetParams(node, &parameters), "get retirement graph function");
        require(parameters.func == original_function && parameters.gridDimX == 1 &&
                    parameters.blockDimX == 1,
                "retirement retains the actual original graph function and dimensions");
        check(cuGraphDestroy(graph), "destroy original retirement graph");
    }
    if (router) {
        check(router->close_context(context, [] { return CUDA_SUCCESS; }),
              "close proven module retirement private owner");
        frontend.reset();
    }
    {
        const NativeDriverCallScope native;
        if (result.unload != CUDA_SUCCESS) {
            check(cuModuleUnload(retired), "release failed-retirement original module after proof");
        }
        check(cuModuleUnload(original_module), "release retained original retirement module");
        check(cuMemFreeHost(host_marker), "release module retirement marker");
        check(cuStreamDestroy(stream), "release module retirement stream");
    }
    std::cout << "source_module_retirement native=" << is_native << " unload=" << result.unload
              << " end=" << result.end << " graph=" << result.has_graph << " native_grants=0\n";
    return result;
}

template <typename Function>
Function original_runtime_symbol(const char* name) {
    const auto resolver = resolve_real_dlsym();
    require(resolver, "resolve original loader before borrowing Runtime library");
    Dl_info runtime_info{};
    require(::dladdr(reinterpret_cast<void*>(&cudaPeekAtLastError), &runtime_info) != 0 &&
                runtime_info.dli_fname,
            "identify actual linked SDK without assuming its library major version");
    void* runtime_library = ::dlopen(runtime_info.dli_fname, RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
    require(runtime_library, "retain actual already-loaded Runtime library for native baseline");
    const auto native = reinterpret_cast<Function>(resolver(runtime_library, name));
    // The executable links this SDK; closing our lookup reference cannot unload
    // it. Balance the temporary handle before any later test assertion can throw.
    require(::dlclose(runtime_library) == 0, "release checked native-baseline library reference");
    require(native, "required actual original SDK entry is present");
    return native;
}

void check_native_runtime_error_tls(bool is_warm) {
    DriverDispatch driver;
    require(driver.initialize(), "resolve original caller-error qualification dispatch");
    check(cuInit(0), "publish original caller-error Driver dispatch");
    const NativeDriverCallScope native;
    const auto count = original_runtime_symbol<decltype(&cudaGetDeviceCount)>("cudaGetDeviceCount");
    const auto device = original_runtime_symbol<decltype(&cudaGetDevice)>("cudaGetDevice");
    const auto peek =
        original_runtime_symbol<decltype(&cudaPeekAtLastError)>("cudaPeekAtLastError");
    const auto get = original_runtime_symbol<decltype(&cudaGetLastError)>("cudaGetLastError");
    const auto set_device = original_runtime_symbol<decltype(&cudaSetDevice)>("cudaSetDevice");
    const auto reset = original_runtime_symbol<RuntimeDeviceResetFunction>("cudaDeviceReset");
    if (is_warm) {
        check(set_device(0), "initialize original warm SDK caller");
    }
    CUcontext before{}, after{};
    check(driver.context_get_current(&before), "observe original pre-error caller");
    unsigned int flags{};
    int active_before{}, active_after{};
    check(cuDevicePrimaryCtxGetState(0, &flags, &active_before),
          "observe original pre-error primary state");
    check(get(), "clear original initial SDK error");
    const auto produce = [&] {
        const auto result = count(nullptr);
        require(result == cudaErrorInvalidValue && peek() == result && peek() == result,
                "actual SDK null count produces stable nonzero caller error");
        return result;
    };
    const auto seeded = produce();
    check(driver.context_get_current(&after), "observe original post-error caller");
    check(cuDevicePrimaryCtxGetState(0, &flags, &active_after),
          "observe error producer activation effect");
    require(after == before && active_after == active_before && get() == seeded &&
                peek() == cudaSuccess,
            "qualified error producer preserves context/primary and get clears exact error");
    const auto observe = [&](const char* name, const std::function<cudaError_t()>& operation) {
        check(get(), "clear prior observation error");
        const auto error = produce();
        const auto status = operation();
        const auto observed = peek();
        check(driver.context_get_current(&after),
              "observe caller after original successful SDK metadata");
        std::cout << "native_error_tls warm=" << is_warm << " operation=" << name
                  << " seeded=" << error << " status=" << status << " last=" << observed
                  << " caller_preserved=" << (after == before) << std::endl;
        require(status == cudaSuccess && after == before && observed == error &&
                    get() == observed && peek() == cudaSuccess,
                "original successful SDK metadata has measured stable caller/TLS behavior");
    };
    observe("get_device_count", [&] {
        int value{};
        return count(&value);
    });
    observe("get_device", [&] {
        int value{};
        return device(&value);
    });
#ifdef GLIMMER_HAS_RUNTIME_CAPTURE_KERNEL
    if (is_warm) {
        const auto name = original_runtime_symbol<decltype(&cudaFuncGetName)>("cudaFuncGetName");
        const auto function =
            original_runtime_symbol<decltype(&cudaGetFuncBySymbol)>("cudaGetFuncBySymbol");
        observe("func_get_name", [&] {
            const char* value{};
            return name(&value, glimmer_runtime_capture_kernel());
        });
        observe("get_func_by_symbol", [&] {
            cudaFunction_t value{};
            return function(&value, glimmer_runtime_capture_kernel());
        });
    }
#endif
    const auto parent_error = produce();
    bool child_has_independent_error{};
    std::thread child([&] {
        const NativeDriverCallScope child_native;
        const auto initial = peek();
        const auto error = count(nullptr);
        child_has_independent_error = initial == cudaSuccess && error == parent_error &&
                                      peek() == error && get() == error && peek() == cudaSuccess;
    });
    child.join();
    require(child_has_independent_error && peek() == parent_error && get() == parent_error &&
                peek() == cudaSuccess,
            "actual SDK error TLS is independent across host threads");
    check(cuDevicePrimaryCtxGetState(0, &flags, &active_after),
          "observe actual final error-probe primary state");
    if (active_after != 0) {
        check(set_device(0), "bind actual SDK-owned primary before checked probe cleanup");
        check(reset(), "destroy actual SDK-owned error-probe resources once");
    }
    check(cuCtxSetCurrent(nullptr), "clear actual native post-cleanup caller binding");
    check(cuDevicePrimaryCtxGetState(0, &flags, &active_after),
          "prove no hidden error-probe primary ownership");
    require(active_after == 0 && peek() == cudaSuccess,
            "native error probe ends with inactive primary and cleared error");
    std::cout << "native_error_tls warm=" << is_warm
              << " producer_no_activation=1 thread_isolation=1 checked_cleanup=1\n";
}

void check_native_context_destroy_version(bool is_modern) {
    DriverDispatch driver;
    require(driver.initialize(), "resolve original context-destroy qualifier");
    check(cuInit(0), "publish original context-destroy process dispatch");
    const NativeDriverCallScope native;
    using Destroy = CUresult(CUDAAPI*)(CUcontext);
    void* address{};
    const int version = is_modern ? 4000 : 2000;
    check(driver.get_proc_address("cuCtxDestroy", &address, version, 0),
          "resolve exact original destroy version");
    require(address, "original historical context-destroy entry exists");
    const auto destroy = reinterpret_cast<Destroy>(address);
    for (const bool is_primary : {false, true}) {
        for (unsigned int caller = 0; caller < 4; ++caller) {
            CUcontext foreign{}, target{};
            check(cuCtxCreate(&foreign, 0, 0), "create independently owned lower-stack context");
            if (is_primary) {
                check(cuDevicePrimaryCtxRetain(&target, 0),
                      "retain independently owned original primary target");
                check(cuCtxPushCurrent(target),
                      "push retained original primary above independent context");
            } else {
                check(cuCtxCreate(&target, 0, 0),
                      "create original user target above independent context");
            }
            CUdeviceptr pointer{};
            CUstream stream{};
            CUevent event{};
            check(cuMemAlloc(&pointer, 4096), "allocate original native context resource");
            check(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING),
                  "create original native context stream");
            check(cuEventCreate(&event, CU_EVENT_DISABLE_TIMING),
                  "create original native context event");
            CUcontext expected = caller == 0 ? target : caller == 2 ? foreign : nullptr;
            check(cuCtxSetCurrent(expected),
                  "select exact bound/null/foreign native destruction caller");
            CUcontext before{};
            check(driver.context_get_current(&before), "observe actual native pre-destroy caller");
            if (caller == 1) {
                while (before) {
                    CUcontext popped{};
                    check(cuCtxPopCurrent(&popped),
                          "empty original caller stack for true null case");
                    check(driver.context_get_current(&before), "prove original null caller stack");
                }
            }
            const auto result = destroy(target);
            CUcontext after{};
            check(driver.context_get_current(&after),
                  "observe original destruction caller before any rebind");
            std::cout << "native_context_destroy_result version=" << version
                      << " primary=" << is_primary << " caller=" << caller << " status=" << result
                      << " before_null=" << (before == nullptr)
                      << " before_foreign=" << (before == foreign)
                      << " before_target=" << (before == target)
                      << " after_null=" << (after == nullptr)
                      << " after_foreign=" << (after == foreign)
                      << " after_target=" << (after == target) << std::endl;
            if (is_primary) {
                unsigned int flags{};
                int active{};
                check(cuDevicePrimaryCtxGetState(0, &flags, &active),
                      "observe original primary state after rejected destruction");
                require(result == CUDA_ERROR_INVALID_CONTEXT && active == 1 && after == before,
                        "original primary destroy rejects while preserving primary ownership");
                // The measured rejection leaves the primary alive. Rebind only
                // after observing the SDK caller, then explicitly clean up its
                // still-owned resources; never retry the rejected destroy.
                check(cuCtxSetCurrent(target), "bind surviving original primary for cleanup");
                check(cuMemFree(pointer), "release surviving primary allocation");
                check(cuStreamDestroy(stream), "release surviving primary stream");
                check(cuEventDestroy(event), "release surviving primary event");
                check(cuCtxSetCurrent(nullptr), "unbind surviving original primary");
                check(cuDevicePrimaryCtxRelease(0), "release sole original primary reference");
                check(cuDevicePrimaryCtxGetState(0, &flags, &active),
                      "prove rejected-destroy probe leaves no primary ownership");
                require(active == 0, "original primary cleanup deactivates the sole reference");
            } else {
                require(result == CUDA_SUCCESS && after == (caller == 1 ? nullptr : foreign),
                        "original user destroy reveals the independently owned lower stack");
            }
            // Never inspect user-context resources after successful destruction.
            std::cout << "native_context_destroy version=" << version << " primary=" << is_primary
                      << " caller=" << caller << " after_null=" << (after == nullptr)
                      << " after_foreign=" << (after == foreign)
                      << " after_target=" << (after == target) << " old_resource_queries=0"
                      << std::endl;
            check(cuCtxSetCurrent(nullptr), "clear observed original SDK post-destruction binding");
            check(cuCtxDestroy(foreign), "destroy still-owned independent context exactly once");
            CUcontext final{};
            check(driver.context_get_current(&final),
                  "observe final original context stack cleanup");
            require(final == nullptr, "native context-stack qualifier leaves no binding");
        }
    }
}

struct PrimaryDestroyProbeOptions {
    bool is_cached{};
    bool is_modern{true};
    unsigned int caller{};
    int fault{-1};
};

void check_owned_primary_destroy(PrimaryDestroyProbeOptions options) {
    const auto is_cached = options.is_cached;
    const auto is_modern = options.is_modern;
    const auto caller = options.caller;
    const auto fault = options.fault;
    check(cuInit(0), "publish owned primary-destroy routing");
    DriverDispatch driver;
    require(driver.initialize(), "resolve owned primary-destroy original dispatch");
    using Destroy = CUresult(CUDAAPI*)(CUcontext);
    const auto controlled = ::dlsym(RTLD_DEFAULT, is_modern ? "cuCtxDestroy_v2" : "cuCtxDestroy");
    require(controlled, "exact public destruction ownership entry exists");
    auto destroy = reinterpret_cast<Destroy>(controlled);
    if (is_cached) {
        void* address{};
        cudaDriverEntryPointQueryResult query{};
        const NativeDriverCallScope native;
        check(cudaGetDriverEntryPointByVersion("cuCtxDestroy", &address, is_modern ? 4000 : 2000, 0,
                                               &query),
              "warm exact original SDK destruction cache");
        require(query == cudaDriverEntryPointSuccess && address == controlled,
                "actual SDK cache retains the exact controlled destruction ownership entry");
        destroy = reinterpret_cast<Destroy>(address);
    }
    CUcontext foreign{}, target{};
    CUstream stream{};
    CUdeviceptr output{}, mapped{};
    std::uint32_t* marker{};
    constexpr unsigned int k_blocks = 84, k_threads = 80;
    constexpr std::size_t k_elements = static_cast<std::size_t>(k_blocks) * k_threads;
    {
        const NativeDriverCallScope native;
        check(cuCtxCreate(&foreign, 0, 0), "create independently owned lower context stack");
        check(cuDevicePrimaryCtxRetain(&target, 0), "retain sole application primary reference");
        check(cuCtxPushCurrent(target), "push primary target above original lower stack");
        check(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), "create original surviving stream");
        check(cuMemHostAlloc(reinterpret_cast<void**>(&marker), 4096, CU_MEMHOSTALLOC_DEVICEMAP),
              "allocate original page-aligned pending-work marker");
        check(cuMemHostGetDevicePointer(&mapped, marker, 0), "map original progress marker");
        *marker = 0;
    }
    check(cuMemAlloc(&output, k_elements * sizeof(std::uint32_t)),
          "allocate charged primary output through unchanged API");
    {
        const NativeDriverCallScope native;
        check(cuMemsetD32(output, 0, k_elements), "initialize exact surviving output");
    }
    CUmodule module{};
    CUfunction function{};
    CUevent event{};
    check(cuModuleLoadData(&module, kCooperativePtx), "load original primary PTX");
    check(cuModuleGetFunction(&function, module, "cooperative"), "resolve surviving function");
    check(cuEventCreate(&event, CU_EVENT_DISABLE_TIMING), "register surviving original event");
    DriverCooperativeRouter* router{};
    {
        const ProcessCallGate::Call observation(cooperative_process_call_gate());
        router = cooperative_source_router();
    }
    require(router, "ordinary setup configures primary owner");
    std::shared_ptr<CudaCooperativeFrontend> owner;
    check(router->existing_frontend(target, owner), "observe already imported primary actor");
    require(owner != nullptr, "public setup imported the primary actor");
    CudaEventProjection registered;
    check(router->event_projection(event, registered), "snapshot exact event registration");
    std::size_t available{}, total{};
    check(cuMemGetInfo(&available, &total), "snapshot charged primary quota");
    const auto produce =
        original_runtime_symbol<decltype(&cudaGetDeviceCount)>("cudaGetDeviceCount");
    const auto peek =
        original_runtime_symbol<decltype(&cudaPeekAtLastError)>("cudaPeekAtLastError");
    const auto get = original_runtime_symbol<decltype(&cudaGetLastError)>("cudaGetLastError");
    std::array<std::uint32_t, k_elements> readback{};
    if (fault >= 0) {
        const std::array results{CUDA_ERROR_OUT_OF_MEMORY, CUDA_ERROR_NOT_READY, CUDA_SUCCESS};
        unsigned int originals{};
        const auto original = [&]() -> CUresult {
            ++originals;
            require(cooperative_process_call_gate().is_native_only_on_current_thread(),
                    "controlled destroy owns its exclusive original issue");
            if (fault == 3) {
                throw std::runtime_error("controlled unknown destruction effect");
            }
            return results.at(static_cast<std::size_t>(fault));
        };
        const ProcessCallGate::Call call(cooperative_process_call_gate());
        CudaPrimaryDestroyDisposition disposition;
        const auto& handled = disposition.was_handled;
        const auto& uncertain = disposition.has_uncertain_lifetime;
        const auto status = router->try_rejected_primary_destroy(cooperative_process_call_gate(),
                                                                 target, original, disposition);
        const auto expected =
            fault < 2 ? results[static_cast<std::size_t>(fault)] : CUDA_ERROR_UNKNOWN;
        require(status == expected && handled && uncertain && originals == 1 &&
                    router->normal_owner_status() != CUDA_SUCCESS,
                "unqualified destroy result preserves native error and retains failed ownership");
        require(router->try_rejected_primary_destroy(cooperative_process_call_gate(), target,
                                                     original, disposition) != CUDA_SUCCESS &&
                    handled && !uncertain && originals == 1 &&
                    cuCtxDestroy(target) == CUDA_ERROR_UNKNOWN,
                "sticky destruction failure forbids original replay and public retry");
        std::cout << "owned_primary_destroy_quarantine fault=" << fault << " status=" << status
                  << " originals=" << originals << " retained=1" << std::endl;
        // Controlled callbacks establish no actual SDK lifetime contract. This
        // exec-isolated uncertain owner is intentionally retained, not cleaned.
        std::_Exit(EXIT_SUCCESS);
    }
    for (unsigned int cycle = 0; cycle < 3; ++cycle) {
        unsigned int work = cycle == 0 ? 150000 : 1000;
        std::array<void*, 3> arguments{&output, &mapped, &work};
        check(cuLaunchKernel(function, k_blocks, 1, 1, k_threads, 1, 1, 0, stream, arguments.data(),
                             nullptr),
              "accept original kernel before primary destroy rejection");
        check(cuEventRecord(event, stream), "accept original surviving event record");
        if (cycle == 0) {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{30};
            while (std::atomic_ref<std::uint32_t>(*marker).load(std::memory_order_acquire) == 0) {
                require(std::chrono::steady_clock::now() < deadline,
                        "actual accepted primary work begins before destruction");
                std::this_thread::yield();
            }
            require(owner->stats().acknowledged_completions == 0,
                    "primary destroy starts with actually pending accepted execution");
        }
        CUcontext before{};
        {
            const NativeDriverCallScope native;
            check(cuCtxSetCurrent(caller == 0   ? target
                                  : caller == 1 ? nullptr
                                                : foreign),
                  "select exact original primary-destroy caller");
            check(driver.context_get_current(&before),
                  "observe exact original pre-rejection caller");
            if (caller == 1) {
                while (before) {
                    CUcontext popped{};
                    check(cuCtxPopCurrent(&popped),
                          "empty original caller stack for null rejection");
                    check(driver.context_get_current(&before),
                          "prove actual null rejection caller");
                }
            }
            check(get(), "clear prior actual Runtime TLS");
            require(produce(nullptr) == cudaErrorInvalidValue && peek() == cudaErrorInvalidValue,
                    "seed exact nonzero actual SDK caller error without activation");
        }
        require(destroy(target) == CUDA_ERROR_INVALID_CONTEXT,
                "ordinary primary destroy preserves exact original SDK rejection");
        CUcontext after{};
        check(driver.context_get_current(&after),
              "observe SDK post-rejection caller before rebind");
        std::cout << "owned_primary_destroy_observed cached=" << is_cached << " caller=" << caller
                  << " cycle=" << cycle << " after_null=" << (after == nullptr)
                  << " after_foreign=" << (after == foreign)
                  << " after_target=" << (after == target) << " last=" << peek()
                  << " driver_guard=" << is_inside_driver_call()
                  << " runtime_guard=" << is_inside_runtime_call() << std::endl;
        require(after == before && peek() == cudaErrorInvalidValue &&
                    get() == cudaErrorInvalidValue && peek() == cudaSuccess &&
                    !is_inside_driver_call() && !is_inside_runtime_call(),
                "primary rejection preserves measured SDK stack, nonzero TLS and lexical guards");
        std::shared_ptr<CudaCooperativeFrontend> retained;
        check(router->existing_frontend(target, retained), "observe surviving primary actor");
        CudaEventProjection projection;
        check(router->refresh_event_projection(registered, projection),
              "observe surviving exact event registration");
        require(retained == owner && projection.context() == target,
                "rejected destruction cannot close actor or retire exact event identity");
        {
            const NativeDriverCallScope native;
            check(cuCtxSetCurrent(target), "explicitly bind surviving primary after observation");
        }
        std::size_t remaining{};
        check(cuMemGetInfo(&remaining, &total), "observe unchanged live primary charge");
        require(remaining == available, "rejected destruction cannot release live primary quota");
        check(cuEventSynchronize(event), "complete surviving accepted event dependency");
        check(cuStreamSynchronize(stream), "complete surviving original application stream");
        check(cuMemcpyDtoH(readback.data(), output, sizeof(readback)),
              "read surviving exact output");
        require(std::all_of(readback.begin(), readback.end(),
                            [&](auto value) { return value == cycle + 1; }),
                "surviving primary work executes every logical output exactly once");
    }
    check(cuEventDestroy(event), "destroy still-owned original event once");
    check(cuModuleUnload(module), "unload still-owned original source module once");
    check(cuMemFree(output), "release surviving application charge once");
    check(cuStreamDestroy(stream), "destroy still-owned original stream once");
    {
        const NativeDriverCallScope native;
        check(cuMemFreeHost(marker), "release original primary progress marker once");
    }
    check(cuDevicePrimaryCtxRelease(0),
          "close primary owner through qualified public last release");
    owner.reset();
    {
        const NativeDriverCallScope native;
        check(cuCtxSetCurrent(nullptr), "clear actual last-release caller before lower cleanup");
        check(cuCtxDestroy(foreign), "destroy independently owned lower context once");
    }
    std::cout << "owned_primary_destroy cached=" << is_cached << " caller=" << caller
              << " cycles=3 pending=1 exact_outputs=" << k_elements * 3 << " tls_error=1"
              << " version=" << (is_modern ? 4000 : 2000) << '\n';
}

struct UserDestroyProbeOptions {
    bool is_modern{};
    bool is_cached{};
    bool has_extra_reference{};
    unsigned int caller{};
    int fault{-1};
};

void check_owned_user_destroy(UserDestroyProbeOptions options) {
    check(cuInit(0), "publish public user-destroy dispatch");
    DriverDispatch driver;
    require(driver.initialize(), "resolve independent user-destroy originals");
    using Destroy = CUresult(CUDAAPI*)(CUcontext);
    using Attach = CUresult(CUDAAPI*)(CUcontext*, unsigned int);
    using Detach = CUresult(CUDAAPI*)(CUcontext);
    void *attach_address{}, *detach_address{};
    check(driver.get_proc_address("cuCtxAttach", &attach_address, 2000, 0),
          "resolve exact original extra-reference acquisition");
    check(driver.get_proc_address("cuCtxDetach", &detach_address, 2000, 0),
          "resolve exact original extra-reference release");
    require(attach_address && detach_address, "original reference entries exist");
    const auto attach = reinterpret_cast<Attach>(attach_address);
    const auto detach = reinterpret_cast<Detach>(detach_address);
    const auto controlled =
        ::dlsym(RTLD_DEFAULT, options.is_modern ? "cuCtxDestroy_v2" : "cuCtxDestroy");
    const auto modern = reinterpret_cast<Destroy>(::dlsym(RTLD_DEFAULT, "cuCtxDestroy_v2"));
    require(controlled && modern, "exact public user destruction exports exist");
    auto destroy = reinterpret_cast<Destroy>(controlled);
    if (options.is_cached) {
        const NativeDriverCallScope native;
        void* address{};
        cudaDriverEntryPointQueryResult query{};
        check(cudaGetDriverEntryPointByVersion("cuCtxDestroy", &address,
                                               options.is_modern ? 4000 : 2000, 0, &query),
              "warm actual SDK user-destruction cache");
        require(query == cudaDriverEntryPointSuccess && address == controlled,
                "actual SDK cache retains the exact controlled user-destroy ABI");
        destroy = reinterpret_cast<Destroy>(address);
    }
    constexpr std::size_t k_page_bytes = 65536;
    constexpr unsigned int k_blocks = 84, k_threads = 80;
    constexpr std::size_t k_elements = static_cast<std::size_t>(k_blocks) * k_threads;
    std::unique_ptr<void, decltype(&std::free)> pages(std::aligned_alloc(4096, k_page_bytes),
                                                      &std::free);
    require(pages != nullptr, "allocate independently owned host business storage");
    auto* markers = static_cast<std::uint32_t*>(pages.get());
    auto* readback = reinterpret_cast<std::uint32_t*>(static_cast<char*>(pages.get()) + 4096);
    CUcontext foreign{};
    {
        const NativeDriverCallScope native;
        check(cuCtxCreate(&foreign, 0, 0), "create independent registration owner");
        check(cuMemHostRegister(pages.get(), k_page_bytes,
                                CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP),
              "register foreign-owned portable readback and markers");
    }
    std::size_t baseline{}, total{};
    check(cuMemGetInfo(&baseline, &total), "snapshot user-lifecycle device quota baseline");
    const bool is_rejected = !options.is_modern && options.has_extra_reference;
    const auto produce =
        original_runtime_symbol<decltype(&cudaGetDeviceCount)>("cudaGetDeviceCount");
    const auto peek =
        original_runtime_symbol<decltype(&cudaPeekAtLastError)>("cudaPeekAtLastError");
    const auto get = original_runtime_symbol<decltype(&cudaGetLastError)>("cudaGetLastError");
    std::size_t exact_values{};
    for (unsigned int generation = 0; generation < 3; ++generation) {
        CUcontext target{}, attached{};
        CUstream stream{}, high_stream{};
        CUdeviceptr output{}, mapped{};
        CUmodule module{};
        CUfunction low{}, high{};
        CUevent event{};
        {
            const NativeDriverCallScope native;
            check(cuCtxSetCurrent(foreign), "select live lower context before new generation");
            check(cuCtxCreate(&target, 0, 0), "create modern application user context");
            if (options.has_extra_reference) {
                check(attach(&attached, 0), "acquire original extra user reference once");
                require(attached == target, "original extra reference identifies target");
            }
            check(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), "create original user stream");
            int least{}, greatest{};
            check(cuCtxGetStreamPriorityRange(&least, &greatest), "observe actual priority range");
            require(least > greatest, "user recovery requires distinct actual stream priorities");
            check(cuStreamCreateWithPriority(&high_stream, CU_STREAM_NON_BLOCKING, greatest),
                  "create original priority insertion stream");
            check(cuMemHostGetDevicePointer(&mapped, pages.get(), 0),
                  "map independently registered foreign business storage");
        }
        check(cuMemAlloc(&output, k_elements * sizeof(std::uint32_t)), "charge user output once");
        {
            const NativeDriverCallScope native;
            check(cuMemsetD32(output, 0, k_elements), "initialize exact output generation");
        }
        check(cuModuleLoadData(&module, kCooperativePtx),
              "prepare original user cooperative source");
        check(cuModuleGetFunction(&low, module, "cooperative"), "resolve user low source");
        check(cuModuleGetFunction(&high, module, "priority_probe"),
              "resolve original priority probe");
        check(cuEventCreate(&event, CU_EVENT_DISABLE_TIMING),
              "register exact user event generation");
        DriverCooperativeRouter* router{};
        {
            const ProcessCallGate::Call observation(cooperative_process_call_gate());
            router = cooperative_source_router();
        }
        require(router, "ordinary user setup configures process router");
        std::shared_ptr<CudaCooperativeFrontend> owner;
        check(router->existing_frontend(target, owner), "observe existing application user actor");
        require(owner != nullptr, "ordinary resource setup imports user actor");
        CudaEventProjection registration;
        check(router->event_projection(event, registration), "snapshot user event identity");
        std::size_t available{};
        check(cuMemGetInfo(&available, &total), "snapshot live user resource charge");
        bool has_physical_pause{}, has_priority_output{};
        const unsigned int cycles = is_rejected ? 3 : 1;
        for (unsigned int cycle = 0; cycle < cycles; ++cycle) {
            markers[0] = markers[1] = 0;
            std::fill_n(readback, k_elements, 0);
            unsigned int work = cycle < 2 ? 150000 : 1000;
            std::array<void*, 3> arguments{&output, &mapped, &work};
            check(cuLaunchKernel(low, k_blocks, 1, 1, k_threads, 1, 1, 0, stream, arguments.data(),
                                 nullptr),
                  "accept unchanged user kernel");
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{30};
            while (std::atomic_ref(markers[0]).load(std::memory_order_acquire) == 0) {
                require(std::chrono::steady_clock::now() < deadline,
                        "accepted user kernel begins before lifecycle issue");
                std::this_thread::yield();
            }
            if (cycle == 0) {
                require(owner->stats().acknowledged_completions == 0,
                        "first user destruction owns pending accepted logical work");
            }
            if (cycle == 1) {
                CUdeviceptr high_marker = mapped + sizeof(std::uint32_t);
                std::array<void*, 1> high_arguments{&high_marker};
                std::array<std::uint32_t, k_elements> sampled{};
                check(owner->native(
                          high_stream, false,
                          [&] {
                              const NativeDriverCallScope native;
                              check(cuMemcpyDtoH(sampled.data(), output, sizeof(sampled)),
                                    "sample actual acknowledged pause after original rejection");
                              has_physical_pause =
                                  std::any_of(sampled.begin(), sampled.end(),
                                              [&](auto value) { return value == cycle; });
                              return driver.launch_kernel(high, 1, 1, 1, 1, 1, 1, 0, high_stream,
                                                          high_arguments.data(), nullptr);
                          },
                          true),
                      "insert actual priority work into the recovered same actor");
            }
            check(cuMemcpyDtoHAsync(readback, output, k_elements * sizeof(std::uint32_t), stream),
                  "accept readback into independently owned host storage before destroy");
            check(cuEventRecord(event, stream), "accept exact event before user destruction");
            CUcontext before{};
            {
                const NativeDriverCallScope native;
                if (options.caller == 2) {
                    check(cuCtxSetCurrent(foreign), "select independent foreign SDK caller");
                } else if (options.caller == 1) {
                    CUcontext current{};
                    check(driver.context_get_current(&current), "observe pre-empty original stack");
                    while (current) {
                        CUcontext popped{};
                        check(cuCtxPopCurrent(&popped), "empty actual original caller stack");
                        check(driver.context_get_current(&current),
                              "observe remaining original stack");
                    }
                }
                check(driver.context_get_current(&before), "observe exact SDK lifecycle caller");
                check(get(), "clear original SDK error before user destruction");
                require(
                    produce(nullptr) == cudaErrorInvalidValue && peek() == cudaErrorInvalidValue,
                    "seed actual nonzero caller TLS without private Runtime replacement");
            }
            if (options.fault >= 0) {
                const std::array results{CUDA_ERROR_OUT_OF_MEMORY, CUDA_ERROR_NOT_READY,
                                         CUDA_ERROR_INVALID_CONTEXT};
                unsigned int originals{}, retirements{};
                const auto original = [&]() -> CUresult {
                    ++originals;
                    CUcontext caller{};
                    check(driver.context_get_current(&caller),
                          "observe controlled user original caller");
                    require(
                        caller == before &&
                            cooperative_process_call_gate().is_native_only_on_current_thread() &&
                            std::all_of(readback, readback + k_elements,
                                        [](auto value) { return value == 1; }),
                        "user fault issue owns exclusion after exact accepted readback drain");
                    if (options.fault == 3) {
                        throw std::runtime_error("controlled unknown user destruction effect");
                    }
                    if (options.fault >= 4 && options.fault <= 5) {
                        return driver.context_destroy(target);
                    }
                    return results.at(static_cast<std::size_t>(options.fault));
                };
                const auto retire = [&](CUcontext context) -> CUresult {
                    ++retirements;
                    require(context == target &&
                                cooperative_process_call_gate().is_native_only_on_current_thread(),
                            "post-destruction cleanup retains exclusion and exact CPU target");
                    if (options.fault == 5) {
                        throw std::bad_alloc();
                    }
                    return CUDA_ERROR_OUT_OF_MEMORY;
                };
                const ProcessCallGate::Call call(cooperative_process_call_gate());
                CudaContextDestroyDisposition disposition;
                const auto status =
                    router->destroy_context(cooperative_process_call_gate(), target, original,
                                            retire, options.is_modern, disposition);
                const auto expected = options.fault == 0 || options.fault == 4
                                          ? CUDA_ERROR_OUT_OF_MEMORY
                                      : options.fault == 1 ? CUDA_ERROR_NOT_READY
                                      : options.fault == 2 ? CUDA_ERROR_INVALID_CONTEXT
                                                           : CUDA_ERROR_UNKNOWN;
                CUcontext after{};
                check(driver.context_get_current(&after),
                      "observe post-fault original caller without querying old resources");
                require(
                    status == expected && disposition.was_handled &&
                        disposition.has_uncertain_lifetime && originals == 1 &&
                        retirements == (options.fault == 4 || options.fault == 5 ? 1U : 0U) &&
                        router->normal_owner_status() != CUDA_SUCCESS &&
                        after == (options.fault == 4 || options.fault == 5 ? foreign : before) &&
                        peek() == cudaErrorInvalidValue && get() == cudaErrorInvalidValue &&
                        peek() == cudaSuccess && !is_inside_driver_call() &&
                        !is_inside_runtime_call(),
                    "user uncertainty preserves original result/caller/TLS and sticky ownership");
                require(
                    router->destroy_context(cooperative_process_call_gate(), target, original,
                                            retire, options.is_modern,
                                            disposition) != CUDA_SUCCESS &&
                        !disposition.has_uncertain_lifetime && originals == 1 &&
                        destroy(target) != CUDA_SUCCESS,
                    "uncertain user destruction cannot retry original issue or import an actor");
                CudaEventProjection retained;
                check(router->event_projection(event, retained),
                      "observe only retained CPU event registration after user uncertainty");
                require(retained.context() == registration.context(),
                        "uncertain user effects cannot discard event ownership");
                std::cout << "owned_user_destroy_quarantine fault=" << options.fault
                          << " status=" << status << " originals=" << originals
                          << " retirements=" << retirements << " drained_values=" << k_elements
                          << " retained=1" << std::endl;
                // These controlled callbacks do not qualify SDK error effects.
                // Retain uncertain ownership until exec exit, even when actual
                // successful destruction precedes a failed CPU cleanup callback.
                std::_Exit(EXIT_SUCCESS);
            }
            const auto status = destroy(target);
            require(status == (is_rejected ? CUDA_ERROR_INVALID_CONTEXT : CUDA_SUCCESS),
                    "ordinary user destruction preserves the exact original ABI result");
            CUcontext after{};
            check(driver.context_get_current(&after),
                  "observe original post-SDK caller without rebind");
            require((is_rejected ? after == before
                                 : after == (options.caller == 1 ? nullptr : foreign)) &&
                        peek() == cudaErrorInvalidValue && get() == cudaErrorInvalidValue &&
                        peek() == cudaSuccess && !is_inside_driver_call() &&
                        !is_inside_runtime_call(),
                    "user lifecycle preserves original caller, nonzero TLS and lexical guards");
            require(std::all_of(readback, readback + k_elements,
                                [&](auto value) { return value == cycle + 1; }),
                    "lifecycle returns only after exact queued host readback completion");
            exact_values += k_elements;
            if (cycle == 1) {
                has_priority_output = markers[1] == 2;
            }
            if (is_rejected) {
                std::shared_ptr<CudaCooperativeFrontend> retained;
                check(router->existing_frontend(target, retained),
                      "observe surviving same user actor");
                CudaEventProjection current;
                check(router->refresh_event_projection(registration, current),
                      "reuse exact user event identity");
                require(retained == owner && current.context() == target,
                        "rejected user destroy cannot retire queue or registration");
                const NativeDriverCallScope native;
                check(cuCtxSetCurrent(target),
                      "bind independently qualified surviving user context");
            }
        }
        if (is_rejected) {
            std::size_t remaining{};
            check(cuMemGetInfo(&remaining, &total), "observe complete surviving user charge");
            const auto stats = owner->stats();
            require(
                remaining == available && has_physical_pause && has_priority_output &&
                    stats.granted_executions == 5 && stats.retreat_requests == 1 &&
                    stats.acknowledged_suspensions == 1 && stats.logical_resumes == 1 &&
                    stats.acknowledged_completions == 4 && stats.acknowledged_failures == 0,
                "recovery preserves full quota, original preparation and actual priority handoff");
            check(cuEventSynchronize(event), "reuse original user event after recovery");
            check(cuStreamSynchronize(stream), "reuse original user stream after recovery");
            check(cuEventDestroy(event), "explicitly destroy surviving application event");
            check(cuModuleUnload(module), "explicitly unload surviving application source");
            check(cuMemFree(output), "explicitly release surviving application allocation");
            check(cuStreamDestroy(stream), "explicitly destroy surviving application stream");
            check(cuStreamDestroy(high_stream), "explicitly destroy surviving high stream");
            {
                const NativeDriverCallScope native;
                check(detach(attached),
                      "release original extra reference while user context lives");
            }
            check(modern(target), "public modern final destruction of the surviving user owner");
        }
        // Success is destruction: never pass any old application CUDA handle again.
        CudaEventProjection forgotten;
        require(router->event_projection(event, forgotten) != CUDA_SUCCESS,
                "successful user destruction retires CPU event registration");
        owner.reset();
        {
            const NativeDriverCallScope native;
            check(cuCtxSetCurrent(foreign), "bind independent context for quota observation");
        }
        std::size_t restored{};
        check(cuMemGetInfo(&restored, &total), "observe fully restored context device quota");
        require(restored == baseline, "successful user destruction retires exact context charges");
    }
    {
        const NativeDriverCallScope native;
        check(cuMemHostUnregister(pages.get()), "unregister independent foreign host storage once");
        check(cuCtxDestroy(foreign), "destroy independent foreign registration owner once");
    }
    std::cout << "owned_user_destroy modern=" << options.is_modern
              << " cached=" << options.is_cached << " extra=" << options.has_extra_reference
              << " caller=" << options.caller << " generations=3 exact_values=" << exact_values
              << " tls_error=1"
              << " priority_handoff=" << is_rejected << " checked_cleanup=1\n";
}

RuntimeDeviceResetFunction original_runtime_reset() {
    const auto native_reset =
        original_runtime_symbol<RuntimeDeviceResetFunction>("cudaDeviceReset");
    require(native_reset != &cudaDeviceReset,
            "native reset baseline must use the original SDK, not the public wrapper");
    return native_reset;
}

struct FailedResetProbe {
    cudaError_t result;
    DriverDispatch& driver;
    DriverCooperativeRouter& router;
    CUcontext context;
    const std::uint32_t* readback;
    unsigned int originals{};
    bool has_drained_before_issue{};
    bool has_original_caller{};
};
thread_local FailedResetProbe* g_failed_reset_probe{};

cudaError_t reset_without_native_effect() {
    auto& probe = *g_failed_reset_probe;
    ++probe.originals;
    CUcontext caller{};
    check(probe.driver.context_get_current(&caller), "observe controlled original reset caller");
    probe.has_original_caller = caller == probe.context;
    std::shared_ptr<CudaCooperativeFrontend> closed;
    const auto status = probe.router.existing_frontend(probe.context, closed);
    probe.has_drained_before_issue =
        status == CUDA_ERROR_DEINITIALIZED && !closed &&
        std::all_of(probe.readback, probe.readback + 1024, [](auto value) { return value == 7; });
    // Controlled validation failure only. Do not infer actual SDK effects from
    // this callback, and never query old resources after a destructive reset.
    return probe.result;
}

void check_failed_runtime_reset(cudaError_t error) {
    const ProcessCallGate::Call call(cooperative_process_call_gate());
    DriverDispatch driver;
    require(driver.initialize(), "resolve failed reset original dispatch");
    check(cuInit(0), "initialize failed reset routing");
    {
        const NativeDriverCallScope native;
        check(cudaSetDevice(0), "initialize failed reset original SDK");
        check(cudaGetLastError(), "start controlled reset with clean SDK error state");
    }
    CUcontext context{};
    check(driver.context_get_current(&context), "observe failed reset caller");
    auto* router = cooperative_source_router();
    require(router && context, "failed reset has a bound original owner");
    void* pointer{};
    cudaEvent_t event{};
    CUstream stream{};
    check(cudaMalloc(&pointer, 4096), "allocate charged failed reset resource");
    check(cudaEventCreate(&event), "register failed reset event");
    check(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), "create failed reset stream");
    std::shared_ptr<CudaCooperativeFrontend> owner;
    check(router->frontend(context, owner), "initialize failed reset actor");
    CudaEventProjection before;
    check(router->event_projection(event, before), "observe exact pre-failure registration");
    std::uint32_t* readback{};
    {
        const NativeDriverCallScope native;
        check(cuMemAllocHost(reinterpret_cast<void**>(&readback), 4096),
              "allocate failed reset readback outside application quota");
        check(cuMemsetD32(reinterpret_cast<CUdeviceptr>(pointer), 7, 1024),
              "initialize original failed reset allocation");
    }
    std::fill_n(readback, 1024, 0);
    check(cuMemcpyDtoHAsync(readback, reinterpret_cast<CUdeviceptr>(pointer), 4096, stream),
          "accept logical copy before controlled reset failure");
    check(cuEventRecord(event, stream), "accept event dependency before failed reset");
    FailedResetProbe probe{error, driver, *router, context, readback};
    g_failed_reset_probe = &probe;
    const auto status = intercept_runtime_device_reset(reset_without_native_effect);
    const auto retry = intercept_runtime_device_reset(reset_without_native_effect);
    g_failed_reset_probe = nullptr;
    CUcontext after{};
    check(driver.context_get_current(&after), "observe failed reset original caller");
    const std::array last{cudaPeekAtLastError(), cudaGetLastError(), cudaPeekAtLastError()};
    const bool has_completed_copy =
        std::all_of(readback, readback + 1024, [](auto value) { return value == 7; });
    CudaEventProjection retained;
    check(router->refresh_event_projection(before, retained),
          "failed reset preserves exact registration identity");
    std::shared_ptr<CudaCooperativeFrontend> unusable;
    const auto owner_status = router->existing_frontend(context, unusable);
    const auto record_status = cuEventRecord(event, stream);
    std::size_t available{}, total{};
    check(cuMemGetInfo(&available, &total), "observe surviving failed reset charge");
    CUdeviceptr base{};
    std::size_t bytes{};
    {
        // Safe only because the controlled callback explicitly had no effect.
        const NativeDriverCallScope native;
        check(cuMemGetAddressRange(&base, &bytes, reinterpret_cast<CUdeviceptr>(pointer)),
              "prove controlled failure preserves original allocation");
        check(cuEventQuery(event), "prove drained original event physically completes");
    }
    // Trusted fixture cleanup is not application recovery or a reset retry.
    // Remove the already-joined CPU owner while its known-live context is bound.
    check(router->close_context(context, [] { return CUDA_SUCCESS; }),
          "retire known-live controlled-failure fixture metadata");
    owner.reset();
    check(cudaFree(pointer), "explicitly release retained failed reset charge");
    {
        const NativeDriverCallScope native;
        check(cuEventDestroy(event), "destroy surviving original failed reset event");
        check(cuStreamDestroy(stream), "destroy surviving original failed reset stream");
        check(cuMemFreeHost(readback), "release completed failed reset readback");
    }
    std::size_t cleaned{}, clean_total{};
    check(cuMemGetInfo(&cleaned, &clean_total), "observe exact failed reset fixture cleanup");
    {
        const NativeDriverCallScope native;
        check(original_runtime_reset()(), "check final original SDK reset after explicit cleanup");
    }
    std::cout << "controlled_failed_runtime_reset error=" << error << " status=" << status
              << " originals=" << probe.originals << " retry=" << retry
              << " owner_status=" << owner_status << " record_status=" << record_status
              << " charge=" << total - available << " completed_copy=" << has_completed_copy
              << " checked_cleanup=1 recovery_qualified=0" << std::endl;
    require(status == error && probe.originals == 1 && retry == cudaErrorUnknown &&
                after == context && last == std::array{cudaSuccess, cudaSuccess, cudaSuccess} &&
                !is_inside_driver_call() && !is_inside_runtime_call(),
            "typed original failure is once-only and preserves caller/TLS without reset retry");
    require(probe.has_drained_before_issue && probe.has_original_caller && has_completed_copy &&
                owner_status == CUDA_ERROR_DEINITIALIZED && !unusable &&
                record_status != CUDA_SUCCESS && retained.context() == context &&
                base == reinterpret_cast<CUdeviceptr>(pointer) && bytes >= 4096 &&
                total == 8388608 && total - available == 4096 && cleaned == clean_total &&
                clean_total == total,
            "failed reset drains but retains ownership and charge, never claiming recovery");
}

void check_unbound_reset_cycle(bool is_native, unsigned int cycle, bool is_cached = false,
                               bool is_unbound = true) {
    const ProcessCallGate::Call call(cooperative_process_call_gate());
    DriverDispatch driver;
    require(driver.initialize(), "resolve unbound reset original dispatch");
    const auto native_reset = original_runtime_reset();
    check(cuInit(0), "initialize unbound reset CPU routing");
    {
        const NativeDriverCallScope native;
        check(cudaSetDevice(0), "initialize selected original Runtime device");
    }
    CUcontext context{};
    check(driver.context_get_current(&context), "observe reset target before unbinding");
    auto* router = cooperative_source_router();
    require(router, "unbound reset has CPU routing owner");
    void* pointer{};
    cudaEvent_t event{};
    if (is_native) {
        const NativeDriverCallScope native;
        check(original_runtime_symbol<RuntimeMallocFunction>("cudaMalloc")(&pointer, 4096),
              "allocate original reset-owned memory without framework accounting");
        check(original_runtime_symbol<decltype(&cudaEventCreate)>("cudaEventCreate")(&event),
              "create original reset-owned event without framework metadata");
    } else {
        check(cudaMalloc(&pointer, 4096), "allocate charged reset-owned memory");
        check(cudaEventCreate(&event), "create fresh metadata-only reset-owned event");
        CudaEventProjection registered;
        check(router->event_projection(event, registered), "observe pre-reset CPU registration");
        require(registered.context() == context, "event belongs to the actual SDK reset target");
    }
    std::shared_ptr<CudaCooperativeFrontend> owner;
    if (!is_native) {
        check(router->frontend(context, owner), "initialize exact reset generation actor");
    }
    std::size_t charged{}, total_before{};
    check(cuMemGetInfo(&charged, &total_before), "observe charged reset generation before SDK");
    if (is_unbound) {
        const NativeDriverCallScope native;
        check(cuCtxSetCurrent(nullptr), "unbind caller immediately before original reset");
    }
    if (!is_native && cycle == 0) {
        const auto missing = intercept_runtime_device_reset(nullptr);
        CudaEventProjection unchanged;
        check(router->event_projection(event, unchanged),
              "missing reset retains event registration");
        std::shared_ptr<CudaCooperativeFrontend> unchanged_owner;
        check(router->existing_frontend(context, unchanged_owner),
              "missing reset retains existing actor without closing or import");
        CUcontext caller{};
        check(driver.context_get_current(&caller), "missing reset preserves original caller");
        require(missing == cudaErrorNotSupported && (is_unbound ? !caller : caller == context) &&
                    unchanged.context() == context && unchanged_owner == owner,
                "missing original cannot issue reset or retire existing ownership");
    }
    cudaError_t status{};
    if (is_native) {
        const NativeDriverCallScope native;
        status = native_reset();
    } else if (is_cached) {
        // Actual SDK entry without a trusted native scope or Runtime root.
        status = native_reset();
    } else {
        status = cudaDeviceReset();
    }
    CUcontext after{};
    check(driver.context_get_current(&after), "observe SDK reset caller before reinitialization");
    const std::array last{cudaPeekAtLastError(), cudaGetLastError(), cudaPeekAtLastError()};
    CudaEventProjection retired;
    const auto registration_status = router->event_projection(event, retired);
    std::shared_ptr<CudaCooperativeFrontend> retained;
    check(router->existing_frontend(context, retained), "inspect retired actor without import");
    {
        const NativeDriverCallScope native;
        check(cudaSetDevice(0), "reinitialize only after observing the original reset result");
    }
    std::size_t available{}, total{};
    check(cuMemGetInfo(&available, &total), "observe reset quota after SDK reinitialization");
    const bool has_expected_ownership =
        is_unbound
            ? available == charged && retained == owner &&
                  (is_native ? registration_status == CUDA_ERROR_INVALID_HANDLE
                             : registration_status == CUDA_SUCCESS)
            : !retained && registration_status == CUDA_ERROR_INVALID_HANDLE && available == total;
    if (is_unbound) {
        // Native qualification proves these resources survive this no-op.
        // Never perform these queries in the bound, destructive reset case.
        CUdeviceptr base{};
        std::size_t bytes{};
        {
            const NativeDriverCallScope native;
            check(cuMemGetAddressRange(&base, &bytes, reinterpret_cast<CUdeviceptr>(pointer)),
                  "prove original allocation survives unbound SDK reset");
            check(cuEventQuery(event), "prove original event survives unbound SDK reset");
        }
        require(base == reinterpret_cast<CUdeviceptr>(pointer) && bytes >= 4096,
                "unbound SDK success cannot destroy application memory");
        if (is_native) {
            const NativeDriverCallScope native;
            check(original_runtime_symbol<RuntimeFreeFunction>("cudaFree")(pointer),
                  "release surviving original unbound allocation");
            check(original_runtime_symbol<decltype(&cudaEventDestroy)>("cudaEventDestroy")(event),
                  "release surviving original unbound event");
        } else {
            check(cudaFree(pointer), "explicitly release surviving charged allocation");
            check(cudaEventDestroy(event), "explicitly retire surviving event registration");
            check(router->close_context(context, [] { return CUDA_SUCCESS; }),
                  "close surviving actor after explicit application cleanup");
        }
    }
    owner.reset();
    retained.reset();
    std::size_t cleaned{}, clean_total{};
    check(cuMemGetInfo(&cleaned, &clean_total), "observe checked reset fixture quota cleanup");
    // Bound reset keys are inspected only in CPU metadata, never by CUDA.
    {
        const NativeDriverCallScope native;
        check(native_reset(), "check final SDK reset without querying destroyed handles");
    }
    std::cout << "normal_runtime_unbound_reset native=" << is_native << " cached=" << is_cached
              << " unbound=" << is_unbound << " cycle=" << cycle << " status=" << status
              << " post_context=" << static_cast<bool>(after)
              << " registration=" << registration_status << " quota=" << available << '/' << total
              << " last=" << last[0] << ',' << last[1] << ',' << last[2] << " checked_cleanup=1"
              << std::endl;
    require(status == cudaSuccess && (is_unbound ? !after : after == context) &&
                last == std::array{cudaSuccess, cudaSuccess, cudaSuccess},
            "reset retains original SDK status, caller binding and TLS");
    require(has_expected_ownership && total == total_before && total == 8388608 &&
                cleaned == clean_total && clean_total == total,
            "reset preserves surviving ownership or retires destroyed ownership, never inferred "
            "from SDK success alone");
}

void check_original_unbound_reset_effect() {
    DriverDispatch driver;
    require(driver.initialize(), "resolve native unbound reset effect dispatch");
    const NativeDriverCallScope native;
    check(driver.init(0), "initialize native unbound reset effect");
    check(cudaSetDevice(0), "initialize actual SDK reset baseline");
    CUcontext before{};
    check(driver.context_get_current(&before), "observe original active reset target");
    void* pointer{};
    cudaEvent_t event{};
    check(original_runtime_symbol<RuntimeMallocFunction>("cudaMalloc")(&pointer, 4096),
          "allocate original unbound reset effect resource");
    check(original_runtime_symbol<decltype(&cudaEventCreate)>("cudaEventCreate")(&event),
          "create original unbound reset effect resource");
    check(cuCtxSetCurrent(nullptr), "unbind native SDK reset effect caller");
    const auto status = original_runtime_reset()();
    CUcontext after{}, primary{};
    check(driver.context_get_current(&after), "observe unbound reset before later SDK calls");
    unsigned int flags{};
    int active{};
    check(cuDevicePrimaryCtxGetState(0, &flags, &active), "observe native reset primary activity");
    check(driver.active_primary_context(0, primary), "borrow only still-active primary identity");
    require(status == cudaSuccess && !after && active == 1 && primary == before,
            "qualified unbound SDK reset leaves its original primary active");
    check(cuCtxSetCurrent(primary), "bind proven surviving primary for checked cleanup");
    CUdeviceptr base{};
    std::size_t bytes{};
    check(cuMemGetAddressRange(&base, &bytes, reinterpret_cast<CUdeviceptr>(pointer)),
          "prove allocation survives SDK unbound reset with active primary");
    check(cuEventQuery(event), "prove original fresh event survives SDK unbound reset");
    check(original_runtime_symbol<RuntimeFreeFunction>("cudaFree")(pointer),
          "release proven surviving original allocation");
    check(original_runtime_symbol<decltype(&cudaEventDestroy)>("cudaEventDestroy")(event),
          "release proven surviving original event");
    check(original_runtime_reset()(), "check original bound cleanup after effect qualification");
    std::cout << "native_unbound_reset_effect status=" << status << " active=" << active
              << " allocation_survives=" << (base == reinterpret_cast<CUdeviceptr>(pointer))
              << " event_survives=1 checked_cleanup=1" << std::endl;
    require(base == reinterpret_cast<CUdeviceptr>(pointer) && bytes >= 4096,
            "native unbound reset does not destroy application resources");
}

void check_original_registered_reset_effect() {
    DriverDispatch driver;
    require(driver.initialize(), "resolve original registered reset effect");
    const NativeDriverCallScope native;
    check(cudaSetDevice(0), "initialize original registered reset baseline");
    alignas(4096) std::array<std::uint32_t, 7168> storage{};
    check(cuMemHostRegister(storage.data(), sizeof(storage), CU_MEMHOSTREGISTER_DEVICEMAP),
          "register application-owned original reset output");
    check(original_runtime_reset()(), "reset actual SDK with caller-owned host registration");
    check(cudaSetDevice(0), "reinitialize original SDK after host reset");
    const auto registered =
        cuMemHostRegister(storage.data(), sizeof(storage), CU_MEMHOSTREGISTER_DEVICEMAP);
    require(registered == CUDA_SUCCESS || registered == CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED,
            "native reset host-registration effect must be known before fixture cleanup");
    check(cuMemHostUnregister(storage.data()), "release proven live application host registration");
    check(original_runtime_reset()(), "check final original registered reset cleanup");
    std::cout << "native_registered_reset_effect re_register=" << registered
              << " registration_survives="
              << (registered == CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED) << " checked_cleanup=1"
              << std::endl;
}

void check_unbound_reset(bool is_native) {
    for (unsigned int cycle = 0; cycle < 3; ++cycle) {
        check_unbound_reset_cycle(is_native, cycle);
    }
}

enum class PrimaryReleaseCaller { kBound, kUnbound, kForeign };

void check_native_primary_release_version(bool is_modern) {
    const ProcessCallGate::Call call(cooperative_process_call_gate());
    DriverDispatch driver;
    require(driver.initialize(), "resolve original primary release dispatch");
    check(cuInit(0), "publish the original Driver table before trusted baseline forwarding");
    const NativeDriverCallScope native;
    check(driver.init(0), "initialize original primary release SDK");
    using Release = CUresult(CUDAAPI*)(CUdevice);
    void* address{};
    const int version = is_modern ? 11000 : 7000;
    check(driver.get_proc_address("cuDevicePrimaryCtxRelease", &address, version, 0),
          "resolve exact original primary release ownership version");
    require(address, "original primary release ownership entry exists");
    const auto release = reinterpret_cast<Release>(address);
    const std::array callers{PrimaryReleaseCaller::kBound, PrimaryReleaseCaller::kUnbound,
                             PrimaryReleaseCaller::kForeign};
    for (const bool has_extra_reference : {false, true}) {
        for (const bool has_private_code : {false, true}) {
            for (const auto caller : callers) {
                CUcontext context{}, extra{}, foreign{};
                check(cuDevicePrimaryCtxRetain(&context, 0), "retain original primary generation");
                check(cuCtxSetCurrent(context), "bind original release resource owner");
                if (has_extra_reference) {
                    check(cuDevicePrimaryCtxRetain(&extra, 0), "retain independent primary usage");
                    require(extra == context, "independent reference names the actual primary");
                }
                CUdeviceptr pointer{};
                CUstream stream{};
                CUevent event{};
                check(cuMemAlloc(&pointer, 4096), "allocate original release-owned storage");
                check(cuMemsetD32(pointer, 7, 1), "initialize original release-owned storage");
                check(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING),
                      "create original release-owned stream");
                check(cuEventCreate(&event, 0), "create original release-owned event");
                check(cuEventRecord(event, stream), "record original release-owned event");
                check(cuEventSynchronize(event), "complete original release-owned prefix");
                CUlibrary library{};
                CUfunction function{};
                if (has_private_code) {
                    check(driver.library_load_data(&library, kCooperativePtx),
                          "load original context-independent code library");
                    CUkernel kernel{};
                    check(driver.library_get_kernel(&kernel, library, "priority_probe"),
                          "resolve original context-independent kernel");
                    check(driver.kernel_get_function(&function, kernel),
                          "bind original library function to actual primary");
                    check(driver.function_load(function), "load original primary library function");
                }
                // Progress host storage is not assumed to share context reset
                // lifetime. Qualify checked pre-release retirement instead.
                void* host{};
                CUdeviceptr mapped{};
                check(cuMemHostAlloc(&host, 32, CU_MEMHOSTALLOC_DEVICEMAP),
                      "allocate original mapped progress storage");
                check(cuMemHostGetDevicePointer(&mapped, host, 0), "map original progress storage");
                check(cuMemFreeHost(host), "retire mapped progress storage before native release");
                CUcontext expected = context;
                if (caller == PrimaryReleaseCaller::kUnbound) {
                    expected = nullptr;
                    check(cuCtxSetCurrent(nullptr), "unbind original release caller");
                } else if (caller == PrimaryReleaseCaller::kForeign) {
                    check(cuCtxCreate(&foreign, 0, 0), "create independent foreign release caller");
                    expected = foreign;
                }
                const auto result = release(0);
                CUcontext after{};
                check(driver.context_get_current(&after), "observe exact original release caller");
                unsigned int flags{};
                int active = -1;
                check(cuDevicePrimaryCtxGetState(0, &flags, &active),
                      "observe actual post-release primary activity");
                const bool has_original_effect = result == CUDA_SUCCESS && after == expected &&
                                                 active == (has_extra_reference ? 1 : 0);
                if (has_extra_reference) {
                    // Activity plus this retained reference establishes survival.
                    // No resource from the last-release branch is queried.
                    check(cuCtxPushCurrent(context), "borrow surviving release target for cleanup");
                    CUdeviceptr base{};
                    std::size_t bytes{};
                    check(cuMemGetAddressRange(&base, &bytes, pointer),
                          "prove original allocation survives nonlast release");
                    check(cuEventQuery(event), "prove original event survives nonlast release");
                    unsigned int value{};
                    check(cuMemcpyDtoH(&value, pointer, sizeof(value)),
                          "prove original allocation contents survive nonlast release");
                    require(base == pointer && bytes >= 4096 && value == 7,
                            "nonlast original release retains exact application storage");
                    if (has_private_code) {
                        check(driver.function_load(function),
                              "prove original code survives nonlast release");
                    }
                    check(cuEventDestroy(event), "destroy surviving original release event");
                    check(cuStreamDestroy(stream), "destroy surviving original release stream");
                    check(cuMemFree(pointer), "free surviving original release allocation");
                    CUcontext popped{};
                    check(cuCtxPopCurrent(&popped), "restore exact surviving release caller");
                    require(popped == context, "checked cleanup popped only its borrowed target");
                }
                if (library) {
                    check(driver.library_unload(library),
                          "retire context-independent code on original post-release caller");
                    int after_unload = -1;
                    check(cuDevicePrimaryCtxGetState(0, &flags, &after_unload),
                          "prove library cleanup cannot activate a destroyed primary");
                    require(after_unload == active,
                            "global code cleanup preserves primary activity");
                }
                check(cuCtxSetCurrent(nullptr), "unbind original post-release caller once");
                if (foreign) {
                    check(cuCtxDestroy(foreign), "destroy independent foreign release caller");
                    check(cuCtxSetCurrent(nullptr), "clear restored native caller stack");
                }
                if (has_extra_reference) {
                    check(release(0), "release final independent primary usage after cleanup");
                }
                check(cuDevicePrimaryCtxGetState(0, &flags, &active),
                      "observe terminal original primary release generation");
                std::cout << "native_primary_release version=" << version
                          << " extra_reference=" << has_extra_reference
                          << " private_code=" << has_private_code
                          << " caller=" << static_cast<unsigned int>(caller) << " status=" << result
                          << " caller_preserved=" << (after == expected)
                          << " terminal_inactive=" << (active == 0) << std::endl;
                require(has_original_effect && active == 0,
                        "original primary release preserves caller and exact reference lifetime");
            }
        }
    }
}

void check_native_primary_reset_version(bool is_modern, bool is_native = true) {
    const ProcessCallGate::Call call(cooperative_process_call_gate());
    DriverDispatch driver;
    require(driver.initialize(), "resolve original primary reset version probe");
    std::optional<NativeDriverCallScope> native;
    if (is_native) {
        native.emplace();
    } else {
        check(cuInit(0), "initialize modeled primary reset ownership");
    }
    check(driver.init(0), "initialize original primary reset ABI probe");
    using Reset = CUresult(CUDAAPI*)(CUdevice);
    void* address{};
    const int version = is_modern ? 11000 : 7000;
    check(is_native ? driver.get_proc_address("cuDevicePrimaryCtxReset", &address, version, 0)
                    : intercept_get_proc_address("cuDevicePrimaryCtxReset", &address, version, 0),
          "resolve exact original or modeled primary reset ownership version");
    require(address, "original versioned primary reset entry is present");
    const auto reset = reinterpret_cast<Reset>(address);
    const auto inactive = reset(0);
    unsigned int flags{};
    int active{};
    check(cuDevicePrimaryCtxGetState(0, &flags, &active),
          "observe original cold primary reset state");
    std::cout << "native_primary_reset native=" << is_native << " version=" << version
              << " cold=1 status=" << inactive << " active=" << active << std::endl;
    for (bool has_extra_reference : {false, true}) {
        {
            const NativeDriverCallScope warmup;
            check(cudaSetDevice(0), "initialize original Runtime primary for ABI qualification");
        }
        CUcontext context{};
        check(driver.context_get_current(&context), "observe exact direct reset caller");
        CUcontext extra{};
        if (has_extra_reference) {
            check(cuDevicePrimaryCtxRetain(&extra, 0),
                  "retain independent application primary usage");
        }
        void* pointer{};
        cudaEvent_t event{};
        std::shared_ptr<CudaCooperativeFrontend> owner;
        if (is_native) {
            check(original_runtime_symbol<RuntimeMallocFunction>("cudaMalloc")(&pointer, 4096),
                  "allocate original primary reset ABI resource");
            check(original_runtime_symbol<decltype(&cudaEventCreate)>("cudaEventCreate")(&event),
                  "create original primary reset ABI event");
        } else {
            check(cudaMalloc(&pointer, 4096), "allocate modeled primary reset ABI resource");
            check(cudaEventCreate(&event), "create tracked primary reset ABI event");
            check(cooperative_source_router()->frontend(context, owner),
                  "create exact direct primary reset actor");
        }
        const auto status = reset(0);
        check(cuDevicePrimaryCtxGetState(0, &flags, &active),
              "observe exact post-reset native state");
        CUcontext after{};
        check(driver.context_get_current(&after), "observe original primary reset caller");
        const int observed_active = active;
        bool has_retired = is_native;
        if (!is_native && status == CUDA_SUCCESS) {
            CudaEventProjection projection;
            std::shared_ptr<CudaCooperativeFrontend> retained;
            check(cooperative_source_router()->existing_frontend(context, retained),
                  "inspect direct reset actor CPU-only");
            has_retired = !retained && cooperative_source_router()->event_projection(
                                           event, projection) == CUDA_ERROR_INVALID_HANDLE;
        }
        if (status != CUDA_SUCCESS) {
            // A rejected original reset has not supplied destruction proof.
            // This diagnostic only frees live original resources after the SDK
            // reports its documented PRIMARY_CONTEXT_ACTIVE validation error.
            require(status == CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE,
                    "unexpected reset error cannot authorize speculative resource cleanup");
            check(original_runtime_symbol<RuntimeFreeFunction>("cudaFree")(pointer),
                  "release surviving original rejected-reset allocation");
            check(original_runtime_symbol<decltype(&cudaEventDestroy)>("cudaEventDestroy")(event),
                  "release surviving original rejected-reset event");
        }
        if (has_extra_reference) {
            check(cuDevicePrimaryCtxRelease(0), "balance independent original primary usage once");
        }
        owner.reset();
        {
            const NativeDriverCallScope reinitialize;
            check(cudaSetDevice(0), "reinitialize only after direct reset caller observation");
        }
        std::size_t available{}, total{};
        if (!is_native) {
            check(cuMemGetInfo(&available, &total), "observe direct reset quota retirement");
            has_retired &= available == total && total == 8388608;
        }
        {
            const NativeDriverCallScope cleanup;
            check(original_runtime_reset()(),
                  "check original Runtime cleanup after ABI qualification");
        }
        std::cout << "native_primary_reset native=" << is_native << " version=" << version
                  << " cold=0 extra_reference=" << has_extra_reference << " status=" << status
                  << " active=" << observed_active << " post_context=" << static_cast<bool>(after)
                  << " checked_cleanup=1" << std::endl;
        require(
            status == CUDA_SUCCESS && observed_active == 0 && after == context && has_retired,
            "both reset ownership versions preserve native effect and retire exact modeled state");
    }
}

void check_cold_reset(bool is_native, bool is_cached = false) {
    const ProcessCallGate::Call call(cooperative_process_call_gate());
    DriverDispatch driver;
    require(driver.initialize(), "resolve cold reset without initializing Driver or Runtime");
    const auto native_reset = original_runtime_reset();
    // No cuInit/current-context probe precedes reset. Instrumentation may
    // initialize Driver state itself; the application still first enters CUDA
    // through the actual SDK reset rather than a fixture initialization call.
    CUcontext after{};
    cudaError_t status{};
    if (is_native) {
        const NativeDriverCallScope native;
        status = native_reset();
    } else if (is_cached) {
        status = native_reset();
    } else {
        status = cudaDeviceReset();
    }
    check(driver.context_get_current(&after), "observe cold SDK reset before later initialization");
    const std::array last{cudaPeekAtLastError(), cudaGetLastError(), cudaPeekAtLastError()};
    CUcontext primary{};
    check(driver.active_primary_context(0, primary),
          "inspect post-reset primary without activation");
    auto* router = cooperative_source_router();
    require(router, "cold reset has CPU process owner");
    std::shared_ptr<CudaCooperativeFrontend> owner;
    if (primary) {
        check(router->existing_frontend(primary, owner), "inspect cold reset without actor import");
    }
    {
        const NativeDriverCallScope native;
        check(native_reset(), "check final cold reset SDK cleanup");
    }
    std::cout << "normal_runtime_cold_reset native=" << is_native << " status=" << status
              << " post_context=" << static_cast<bool>(after)
              << " actor=" << static_cast<bool>(owner) << " last=" << last[0] << ',' << last[1]
              << ',' << last[2] << " checked_cleanup=1" << std::endl;
    require(status == cudaSuccess && !after && !owner &&
                last == std::array{cudaSuccess, cudaSuccess, cudaSuccess},
            "cold reset preserves SDK initialization, unbound caller and TLS without actor import");
}

constexpr unsigned int kResetBlocks = 105;
constexpr unsigned int kResetThreads = 64;
constexpr std::size_t kResetElements = static_cast<std::size_t>(kResetBlocks) * kResetThreads;

thread_local unsigned int g_checkpoint_originals{};
cudaError_t checkpoint_failure_without_native_effect() {
    ++g_checkpoint_originals;
    return static_cast<cudaError_t>(cudaErrorApiFailureBase + 7);
}

void check_reversible_context_checkpoint() {
    const ProcessCallGate::Call call(cooperative_process_call_gate());
    DriverDispatch driver;
    require(driver.initialize(), "resolve reversible checkpoint dispatch");
    check(cuInit(0), "initialize checkpoint routing");
    {
        const NativeDriverCallScope native;
        check(cudaSetDevice(0), "initialize checkpoint original SDK");
        check(cudaGetLastError(), "start checkpoint with clean SDK TLS");
    }
    CUcontext context{};
    check(driver.context_get_current(&context), "observe original checkpoint caller");
    auto* router = cooperative_source_router();
    require(router, "checkpoint has existing context router");
    CUstream stream{}, high_stream{};
    CUdeviceptr output{}, mapped{};
    std::uint32_t* markers{};
    std::uint32_t* readback{};
    check(cuMemAlloc(&output, kResetElements * sizeof(std::uint32_t)),
          "allocate charged checkpoint output");
    {
        const NativeDriverCallScope native;
        check(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), "create checkpoint low stream");
        check(cuStreamCreateWithPriority(&high_stream, CU_STREAM_NON_BLOCKING, -1),
              "create checkpoint high stream");
        check(cuMemsetD32(output, 0, kResetElements), "initialize checkpoint exact-once output");
        check(cuMemHostAlloc(reinterpret_cast<void**>(&markers), 4096, CU_MEMHOSTALLOC_DEVICEMAP),
              "allocate checkpoint mapped markers");
        check(cuMemHostGetDevicePointer(&mapped, markers, 0), "map checkpoint progress markers");
        check(cuMemAllocHost(reinterpret_cast<void**>(&readback),
                             kResetElements * sizeof(std::uint32_t)),
              "allocate checkpoint pinned readback");
    }
    CUmodule module{};
    CUfunction low{}, high{};
    check(cuModuleLoadData(&module, kCooperativePtx), "load checkpoint original PTX");
    check(cuModuleGetFunction(&low, module, "cooperative"), "prepare checkpoint low function");
    check(cuModuleGetFunction(&high, module, "priority_probe"), "prepare checkpoint high function");
    cudaEvent_t event{};
    check(cudaEventCreate(&event), "create checkpoint public Runtime event");
    std::shared_ptr<CudaCooperativeFrontend> owner;
    check(router->existing_frontend(context, owner), "observe checkpoint original actor");
    require(owner != nullptr, "qualified module creates checkpoint actor");
    std::size_t before{}, total{};
    check(cuMemGetInfo(&before, &total), "observe retained checkpoint charge");
    bool has_pending_pause{};
    g_checkpoint_originals = 0;
    for (unsigned int cycle = 0; cycle < 3; ++cycle) {
        markers[0] = 0;
        markers[1] = 0;
        unsigned int work = cycle == 1 ? 150000 : 1000;
        std::array<void*, 3> arguments{&output, &mapped, &work};
        check(cuLaunchKernel(low, kResetBlocks, 1, 1, kResetThreads, 1, 1, 0, stream,
                             arguments.data(), nullptr),
              "reuse original public kernel around reversible checkpoint");
        if (cycle == 1) {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{10};
            while (std::atomic_ref<std::uint32_t>(markers[0]).load(std::memory_order_acquire) ==
                   0) {
                require(std::chrono::steady_clock::now() < deadline,
                        "resumed checkpoint kernel starts physical work");
                std::this_thread::sleep_for(std::chrono::milliseconds{1});
            }
            CUdeviceptr high_marker = mapped + sizeof(std::uint32_t);
            std::array<void*, 1> high_arguments{&high_marker};
            std::array<std::uint32_t, kResetElements> sampled{};
            check(owner->native(
                      high_stream, false,
                      [&] {
                          const NativeDriverCallScope native;
                          check(cuMemcpyDtoH(sampled.data(), output, sizeof(sampled)),
                                "sample physically paused work after checkpoint resume");
                          has_pending_pause =
                              std::any_of(sampled.begin(), sampled.end(),
                                          [&](auto value) { return value == cycle; });
                          return driver.launch_kernel(high, 1, 1, 1, 1, 1, 1, 0, high_stream,
                                                      high_arguments.data(), nullptr);
                      },
                      true),
                  "insert priority work into resumed original actor");
        }
        check(cuMemcpyDtoHAsync(readback, output, kResetElements * sizeof(std::uint32_t), stream),
              "accept public readback before checkpoint join");
        check(cuEventRecord(event, stream), "record surviving public event before checkpoint");
        CudaWorkerCheckpoint checkpoint;
        // Device instrumentation expands every volatile load in the deliberately
        // long preemption kernel. Keep a bounded wait without weakening work or
        // accepting a pending queue as a completed checkpoint.
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{60};
        CUresult parked{};
        do {
            parked = owner->checkpoint(checkpoint);
            require(parked == CUDA_SUCCESS || parked == CUDA_ERROR_NOT_READY,
                    "checkpoint admits only completed logical/control state");
            std::this_thread::sleep_for(std::chrono::milliseconds{1});
        } while (parked == CUDA_ERROR_NOT_READY && std::chrono::steady_clock::now() < deadline);
        if (parked == CUDA_ERROR_NOT_READY) {
            const auto pending = owner->stats();
            std::cout << "checkpoint_admission_timeout cycle=" << cycle
                      << " grants=" << pending.granted_executions
                      << " completed=" << pending.acknowledged_completions
                      << " retreat=" << pending.retreat_requests
                      << " suspended=" << pending.acknowledged_suspensions
                      << " resumed=" << pending.logical_resumes << std::endl;
        }
        check(parked, "join checkpoint without retiring resource identities");
        require(std::all_of(readback, readback + kResetElements,
                            [&](auto value) { return value == cycle + 1; }),
                "checkpoint proves all 6720 exact-once outputs before original issue");
        const auto original = checkpoint_failure_without_native_effect();
        CUcontext after{};
        check(driver.context_get_current(&after), "observe controlled failure checkpoint caller");
        const std::array last{cudaPeekAtLastError(), cudaGetLastError(), cudaPeekAtLastError()};
        std::size_t retained{}, retained_total{};
        check(cuMemGetInfo(&retained, &retained_total),
              "observe unchanged parked ownership charge");
        require(original == static_cast<cudaError_t>(cudaErrorApiFailureBase + 7) &&
                    after == context && last == std::array{cudaSuccess, cudaSuccess, cudaSuccess} &&
                    retained == before && retained_total == total,
                "known non-destructive original failure preserves caller, TLS and entire quota");
        // Only the callback's explicit no-effect contract authorizes resumption.
        // Neither an arbitrary SDK error nor context pointer equality would do so.
        check(owner->resume_checkpoint(checkpoint), "resume independently proven surviving owner");
        check(cuEventQuery(event), "reuse exact public event binding after worker handoff");
        check(cuStreamWaitEvent(nullptr, event, 0),
              "reuse public event dependency after checkpoint");
        check(cuStreamSynchronize(nullptr), "prove post-checkpoint public wait ordering");
    }
    const auto stats = owner->stats();
    require(has_pending_pause && stats.granted_executions == 5 && stats.retreat_requests == 1 &&
                stats.acknowledged_suspensions == 1 && stats.logical_resumes == 1 &&
                stats.acknowledged_completions == 4 && stats.acknowledged_failures == 0 &&
                g_checkpoint_originals == 3,
            "resumed actor preserves original cooperative preparation and actual priority handoff");
    check(cudaEventDestroy(event), "retire public checkpoint event");
    check(cuModuleUnload(module), "retire original checkpoint source after resumed work");
    check(cuMemFree(output), "release charged checkpoint output");
    check(router->close_context(context, [] { return CUDA_SUCCESS; }), "close resumed actor");
    owner.reset();
    {
        const NativeDriverCallScope native;
        check(cuStreamDestroy(stream), "destroy checkpoint low stream");
        check(cuStreamDestroy(high_stream), "destroy checkpoint high stream");
        check(cuMemFreeHost(markers), "free stopped checkpoint markers");
        check(cuMemFreeHost(readback), "free completed checkpoint readback");
    }
    std::size_t cleaned{}, clean_total{};
    check(cuMemGetInfo(&cleaned, &clean_total), "observe complete checkpoint fixture recovery");
    require(cleaned == clean_total && clean_total == 8388608, "checkpoint cleanup restores quota");
    {
        const NativeDriverCallScope native;
        check(original_runtime_reset()(), "check final checkpoint original SDK cleanup");
    }
    std::cout << "reversible_context_checkpoint cycles=3 originals=3 outputs=6720 "
                 "grants=5 retreat=1 suspended=1 resumed=1 completed=4 checked_cleanup=1 "
                 "arbitrary_reset_recovery=0"
              << std::endl;
}

void check_cold_runtime_graph_constructor(bool is_native) {
    DriverDispatch driver;
    require(driver.initialize(), "resolve cold graph original Driver metadata");
    check(driver.init(0), "initialize Driver without warming Runtime context");
    cudaGraph_t graph{};
    CUcontext before{}, after{};
    {
        const NativeDriverCallScope native;
        check(cudaGraphCreate(&graph, 0), "create original cold empty SDK graph");
        check(driver.context_get_current(&before), "observe original pre-constructor binding");
    }
    cudaGraphExec_t executable{};
    cudaError_t status{};
    if (is_native) {
        const NativeDriverCallScope native;
        status = cudaGraphInstantiateWithFlags(&executable, graph, 0);
    } else {
        status = cudaGraphInstantiateWithFlags(&executable, graph, 0);
    }
    check(driver.context_get_current(&after), "observe actual post-constructor binding");
    const auto peek = cudaPeekAtLastError();
    const auto last = cudaGetLastError();
    check(status, "cold native/public constructor preserves original SDK success");
    require(executable && peek == cudaSuccess && last == cudaSuccess && !is_inside_runtime_call() &&
                !is_inside_driver_call(),
            "cold constructor preserves actual handle, last-error and lexical TLS");
    if (!is_native) {
        const ProcessCallGate::Call call(cooperative_process_call_gate());
        auto* router = cooperative_source_router();
        require(router, "cold public constructor owns CPU executable registration");
        CudaNativeGraphExecutable owned;
        const auto found = router->graphs().lookup(executable, owned);
        require(after ? found == CUDA_SUCCESS && owned.context() == after &&
                            owned.effects().node_count() == 0
                      : found == CUDA_ERROR_NOT_SUPPORTED,
                "actual late context determines prepared versus contextless metadata");
        check(cudaGraphExecDestroy(executable), "retire cold public executable");
        CudaNativeGraphExecutable retired;
        require(router->graphs().lookup(executable, retired) == CUDA_ERROR_INVALID_HANDLE,
                "cold executable retirement removes exact CPU identity");
    } else {
        const NativeDriverCallScope native;
        check(cudaGraphExecDestroy(executable), "destroy cold original executable");
    }
    {
        const NativeDriverCallScope native;
        check(cudaGraphDestroy(graph), "destroy cold original source graph");
        check(cudaDeviceReset(), "checked final original cold SDK cleanup");
    }
    std::cout << "cold_runtime_graph_constructor native=" << is_native << " status=" << status
              << " bound_before=" << (before != nullptr) << " bound_after=" << (after != nullptr)
              << " last=" << last << " checked_cleanup=1" << std::endl;
}

void check_owned_runtime_graph(bool is_ptds, bool uses_null_stream, bool fails_after_issue = false,
                               bool uses_runtime_constructor = false,
                               bool uses_base_constructor = false, bool updates_events = false,
                               bool updates_nodes = false, unsigned int graph_update_kind = 0) {
    const bool updates_child = graph_update_kind == 2 || graph_update_kind == 3;
    static_assert(std::is_same_v<cudaGraphExec_t, CUgraphExec>);
    static_assert(std::is_same_v<cudaStream_t, CUstream>);
    DriverDispatch driver;
    require(driver.initialize(), "resolve owned SDK graph native dispatch");
    {
        const NativeDriverCallScope native;
        check(cudaSetDevice(0), "initialize original SDK graph caller and cached dispatch");
    }
    const auto sdk_launch = original_runtime_symbol<RuntimeGraphLaunchFunction>(
        is_ptds ? "cudaGraphLaunch_ptsz" : "cudaGraphLaunch");
    const auto host_kernel = glimmer_runtime_capture_kernel();
    cudaFunction_t original_kernel{};
    CUstream low_stream{}, high_stream{};
    CUdeviceptr output{}, device_markers{};
    std::uint32_t* markers{};
    unsigned int* counter{};
    {
        const NativeDriverCallScope native;
        check(runtime_symbol<RuntimeGetFuncBySymbolFunction>("cudaGetFuncBySymbol")(
                  &original_kernel, host_kernel),
              "load actual SDK graph function without changing node ABI");
        int least{}, greatest{};
        check(cuCtxGetStreamPriorityRange(&least, &greatest), "query graph stream priorities");
        require(least > greatest, "graph handoff requires actual distinct stream priorities");
        check(cuStreamCreateWithPriority(&low_stream, CU_STREAM_NON_BLOCKING, least),
              "create original low graph producer stream");
        if (!uses_null_stream) {
            check(cuStreamCreateWithPriority(&high_stream, CU_STREAM_NON_BLOCKING, greatest),
                  "create original high graph replay stream");
        }
        check(cuMemHostAlloc(reinterpret_cast<void**>(&markers), 8, CU_MEMHOSTALLOC_DEVICEMAP),
              "allocate mapped graph progress markers");
        check(cuMemHostGetDevicePointer(&device_markers, markers, 0),
              "resolve actual graph marker address");
        check(original_runtime_symbol<RuntimeMallocFunction>("cudaMalloc")(
                  reinterpret_cast<void**>(&counter), sizeof(*counter)),
              "allocate original SDK graph counter");
        check(cudaMemset(counter, 0, sizeof(*counter)), "initialize original SDK graph counter");
    }
    markers[0] = markers[1] = 0;
    check(cuMemAlloc(&output, kResetElements * sizeof(std::uint32_t)),
          "allocate charged graph dependency output");
    check(cuMemsetD32(output, 0, kResetElements), "initialize graph dependency output");
    const ProcessCallGate::Call call(cooperative_process_call_gate());
    auto* router = cooperative_source_router();
    require(router, "owned Runtime graph has an admitted process router");
    CUcontext context{};
    check(driver.context_get_current(&context), "observe actual SDK graph caller");
    std::shared_ptr<CudaCooperativeFrontend> owner;
    check(router->frontend(context, owner), "initialize exact graph context owner");
    CUmodule module{};
    CUfunction low{};
    check(cuModuleLoadData(&module, kCooperativePtx), "prepare original graph dependency kernel");
    check(cuModuleGetFunction(&low, module, "cooperative"), "resolve qualified graph producer");
    CUevent waited{}, recorded{}, retired_waited{}, retired_recorded{}, outside_recorded{};
    check(cuEventCreate(&waited, CU_EVENT_DISABLE_TIMING), "create owned graph wait event");
    check(cuEventCreate(&recorded, CU_EVENT_DISABLE_TIMING), "create owned graph record event");
    if (updates_child)
        check(cuEventCreate(&outside_recorded, CU_EVENT_DISABLE_TIMING),
              "create unrelated outer graph record event");
    CUgraph graph{}, source_child{};
    CUgraphNode child_node{};
    std::array<CUgraphNode, 3> nodes{};
    CUgraphExec executable{};
    {
        const NativeDriverCallScope native;
        check(cuGraphCreate(&graph, 0), "construct original SDK-compatible graph");
        check(cuGraphAddEventWaitNode(&nodes[0], graph, nullptr, 0, waited),
              "add original dependency wait node");
        std::array<void*, 1> arguments{&counter};
        CUDA_KERNEL_NODE_PARAMS params{};
        params.func = original_kernel;
        params.gridDimX = params.gridDimY = params.gridDimZ = 1;
        params.blockDimX = params.blockDimY = params.blockDimZ = 1;
        params.kernelParams = arguments.data();
        check(cuGraphAddKernelNode(&nodes[1], graph, &nodes[0], 1, &params),
              "add unchanged actual Runtime kernel node");
        check(cuGraphAddEventRecordNode(&nodes[2], graph, &nodes[1], 1, recorded),
              "add original native record node");
        std::size_t count{};
        check(cuGraphGetNodes(graph, nullptr, &count), "count complete original graph");
        CUDA_KERNEL_NODE_PARAMS actual{};
        check(cuGraphKernelNodeGetParams(nodes[1], &actual), "check original graph function ABI");
        require(count == 3 && actual.func == original_kernel && actual.gridDimX == 1 &&
                    actual.blockDimX == 1,
                "owned replay must preserve all original nodes and function identity");
        if (updates_child) {
            source_child = graph;
            check(cuGraphCreate(&graph, 0), "create original parent graph");
            check(cuGraphAddChildGraphNode(&child_node, graph, nullptr, 0, source_child),
                  "copy original three-node child into parent scope");
            CUgraphNode outer{};
            check(cuGraphAddEventRecordNode(&outer, graph, &child_node, 1, outside_recorded),
                  "retain unrelated original outer record effect");
        }
    }
    if (uses_runtime_constructor) {
        using BaseConstructor =
            cudaError_t(CUDARTAPI*)(cudaGraphExec_t*, cudaGraph_t, unsigned long long);
        const auto base = reinterpret_cast<BaseConstructor>(
            dlvsym(RTLD_DEFAULT, "cudaGraphInstantiate", "libcudart.so.12"));
        require(!uses_base_constructor || base, "exact modern base constructor export required");
        require(!uses_base_constructor || (runtime_graph_constructor_lookup(RTLD_DEFAULT) ==
                                               reinterpret_cast<void*>(base) &&
                                           dlerror() == nullptr),
                "actual CUDA 12 default lookup selects modern ABI without alternate-probe error");
        check(uses_base_constructor ? base(&executable, graph, 0)
                                    : cudaGraphInstantiateWithFlags(&executable, graph, 0),
              "actual SDK root registers executable despite cached Driver dispatch");
        CudaNativeGraphExecutable observed;
        check(router->graphs().lookup(executable, observed),
              "inspect SDK root prepared registration");
        require(observed.context() == context &&
                    observed.effects().node_count() == (updates_child ? 5U : 3U) &&
                    observed.events().size() == (updates_child ? 3U : 2U),
                "Runtime root retains exact immutable original graph effects");
    } else {
        check(cuGraphInstantiateWithFlags(&executable, graph, 0),
              "register actual native executable before owned Runtime replay");
    }
    if (updates_events || updates_nodes) {
        CudaNativeGraphExecutable previous;
        check(router->graphs().lookup(executable, previous),
              "retain graph plan before SDK mutation");
        CUevent new_waited{}, new_recorded{};
        check(cuEventCreate(&new_waited, CU_EVENT_DISABLE_TIMING),
              "create replacement graph wait event");
        check(cuEventCreate(&new_recorded, CU_EVENT_DISABLE_TIMING),
              "create replacement graph record event");
        if (updates_nodes) {
            cudaGraphNodeParams params{};
            params.type = cudaGraphNodeTypeWaitEvent;
            params.eventWait.event = new_waited;
            check(cudaGraphExecNodeSetParams(executable, nodes[0], &params),
                  "generic public SDK root replaces original graph wait event");
            params = {};
            params.type = cudaGraphNodeTypeEventRecord;
            params.eventRecord.event = new_recorded;
            check(runtime_symbol<decltype(&cudaGraphExecNodeSetParams)>(
                      "cudaGraphExecNodeSetParams")(executable, nodes[2], &params),
                  "dynamic generic public SDK root replaces original record event");
        } else {
            check(cudaGraphExecEventWaitNodeSetEvent(executable, nodes[0], new_waited),
                  "public SDK root changes original executable wait event");
            using Setter = decltype(&cudaGraphExecEventRecordNodeSetEvent);
            check(runtime_symbol<Setter>("cudaGraphExecEventRecordNodeSetEvent")(
                      executable, nodes[2], new_recorded),
                  "dynamic public SDK root changes original executable record event");
        }
        CudaNativeGraphExecutable changed;
        check(router->graphs().lookup(executable, changed),
              "SDK mutation retains a replayable replacement");
        require(std::ranges::any_of(changed.events(),
                                    [&](const auto& bound) {
                                        return bound.effect.kind ==
                                                   CudaNativeGraphEventKind::kWait &&
                                               bound.effect.event == new_waited;
                                    }) &&
                    std::ranges::any_of(changed.events(),
                                        [&](const auto& bound) {
                                            return bound.effect.kind ==
                                                       CudaNativeGraphEventKind::kRecord &&
                                                   bound.effect.event == new_recorded;
                                        }) &&
                    std::ranges::none_of(previous.events(),
                                         [&](const auto& bound) {
                                             return bound.effect.event == new_waited ||
                                                    bound.effect.event == new_recorded;
                                         }) &&
                    router->graphs().is_current(previous) == CUDA_ERROR_INVALID_HANDLE,
                "SDK success replaces exact effects without mutating a retained prior snapshot");
        if (updates_nodes) {
            retired_waited = waited;
            retired_recorded = recorded;
        } else {
            check(cuEventDestroy(waited), "retire old graph wait before future SDK issue");
            check(cuEventDestroy(recorded), "retire old graph record before future SDK issue");
        }
        waited = new_waited;
        recorded = new_recorded;
    }
    if (graph_update_kind != 0) {
        CudaNativeGraphExecutable previous;
        check(router->graphs().lookup(executable, previous),
              "retain original plan before complete SDK graph replacement");
        CUevent new_waited{}, new_recorded{};
        check(cuEventCreate(&new_waited, CU_EVENT_DISABLE_TIMING),
              "create complete replacement wait registration");
        check(cuEventCreate(&new_recorded, CU_EVENT_DISABLE_TIMING),
              "create complete replacement record registration");
        CUgraph replacement{};
        {
            const NativeDriverCallScope native;
            check(cuGraphCreate(&replacement, 0), "create topologically identical replacement");
            std::array<CUgraphNode, 3> changed_nodes{};
            check(cuGraphAddEventWaitNode(&changed_nodes[0], replacement, nullptr, 0, new_waited),
                  "build actual replacement dependency wait");
            CUDA_KERNEL_NODE_PARAMS params{};
            params.func = original_kernel;
            params.gridDimX = 2;
            params.gridDimY = params.gridDimZ = 1;
            params.blockDimX = params.blockDimY = params.blockDimZ = 1;
            std::array<void*, 1> arguments{&counter};
            params.kernelParams = arguments.data();
            check(
                cuGraphAddKernelNode(&changed_nodes[1], replacement, &changed_nodes[0], 1, &params),
                "build replacement original kernel with changed grid");
            check(cuGraphAddEventRecordNode(&changed_nodes[2], replacement, &changed_nodes[1], 1,
                                            new_recorded),
                  "build actual replacement native record");
        }
        if (graph_update_kind == 1) {
            cudaGraphExecUpdateResultInfo info{};
            check(runtime_symbol<decltype(&cudaGraphExecUpdate)>("cudaGraphExecUpdate")(
                      executable, replacement, &info),
                  "cached modern SDK root performs complete whole update");
            require(info.result == cudaGraphExecUpdateSuccess,
                    "actual SDK confirms compatible topology and performed update");
            CudaNativeGraphExecutable before_setter;
            check(router->graphs().lookup(executable, before_setter),
                  "whole update retains a fresh plan before original-node setter");
            CUevent final_recorded{};
            check(cuEventCreate(&final_recorded, CU_EVENT_DISABLE_TIMING),
                  "create subsequent original-node record binding");
            check(cudaGraphExecEventRecordNodeSetEvent(executable, nodes[2], final_recorded),
                  "SDK setter still addresses instantiation-time node after whole update");
            require(router->graphs().is_current(before_setter) == CUDA_ERROR_INVALID_HANDLE,
                    "subsequent original-node setter replaces the whole-update generation");
            {
                const NativeDriverCallScope native;
                check(cuGraphDestroy(replacement), "retire replacement before its old event");
            }
            check(cuEventDestroy(new_recorded), "retire superseded whole-update record event");
            new_recorded = final_recorded;
        } else {
            if (graph_update_kind == 2) {
                check(cudaGraphExecChildGraphNodeSetParams(executable, child_node, replacement),
                      "stable SDK root replaces the exact known child scope");
            } else {
                cudaGraphNodeParams params{};
                params.type = cudaGraphNodeTypeGraph;
                params.graph.graph = replacement;
                check(runtime_symbol<decltype(&cudaGraphExecNodeSetParams)>(
                          "cudaGraphExecNodeSetParams")(executable, child_node, &params),
                      "generic cached SDK root replaces child without altering outer effects");
            }
            const NativeDriverCallScope native;
            check(cuGraphDestroy(replacement), "retire supplied child before executable replay");
        }
        CudaNativeGraphExecutable changed;
        check(router->graphs().lookup(executable, changed),
              "complete SDK replacement publishes replayable new dependencies");
        const auto has_event = [](const auto& owned, CUevent event) {
            return std::ranges::any_of(
                owned.events(), [&](const auto& bound) { return bound.effect.event == event; });
        };
        require(has_event(changed, new_waited) && has_event(changed, new_recorded) &&
                    !has_event(changed, waited) && !has_event(changed, recorded) &&
                    !has_event(previous, new_waited) && !has_event(previous, new_recorded) &&
                    (!updates_child || (has_event(previous, outside_recorded) &&
                                        has_event(changed, outside_recorded))) &&
                    router->graphs().is_current(previous) == CUDA_ERROR_INVALID_HANDLE,
                "new scope effects and unrelated parent effects preserve immutable old snapshots");
        retired_waited = waited;
        retired_recorded = recorded;
        waited = new_waited;
        recorded = new_recorded;
    }
    const auto retire_source = [&] {
        {
            const NativeDriverCallScope native;
            check(cuGraphDestroy(graph), "destroy source graph before retained executable replay");
            if (source_child)
                check(cuGraphDestroy(source_child), "destroy separately supplied original child");
        }
        if (retired_waited) {
            check(cuEventDestroy(retired_waited),
                  "retire original wait after node/source lifetime");
            check(cuEventDestroy(retired_recorded),
                  "retire original record after node/source lifetime");
            retired_waited = retired_recorded = nullptr;
        }
    };
    if (updates_nodes) {
        cudaGraphNodeParams params{};
        params.type = cudaGraphNodeTypeKernel;
        params.kernel.func = const_cast<void*>(host_kernel);
        params.kernel.gridDim = dim3{2, 1, 1};
        params.kernel.blockDim = dim3{1, 1, 1};
        std::array<void*, 1> arguments{&counter};
        params.kernel.kernelParams = arguments.data();
        check(cudaGraphExecNodeSetParams(executable, nodes[1], &params),
              "generic SDK kernel update changes original grid to two blocks");
        check(cudaGraphNodeSetEnabled(executable, nodes[1], 0),
              "disable original SDK kernel without disabling dependency plan");
        CudaNativeGraphExecutable prepared;
        check(router->graphs().lookup(executable, prepared),
              "known kernel parameters and enablement retain prepared dependencies");
    } else {
        retire_source();
    }
    unsigned int originals{};
    {
        const NativeDriverCallScope native;
        check(cuCtxSetCurrent(nullptr), "unbind original caller before private graph preflight");
    }
    const auto unbound = execute_owned_runtime_graph(executable, high_stream, is_ptds, [&] {
        ++originals;
        return sdk_launch(executable, high_stream);
    });
    CUcontext unbound_after{};
    check(driver.context_get_current(&unbound_after), "observe rejected unbound graph caller");
    {
        const NativeDriverCallScope native;
        check(cuCtxSetCurrent(context),
              "restore proven live graph caller after no-issue rejection");
    }
    require(unbound == cudaErrorUnknown && !unbound_after && originals == 0,
            "private graph contract cannot guess SDK lazy binding or issue on an unbound caller");
    unsigned int work{1000};
    std::array<void*, 3> parameters{&output, &device_markers, &work};
    check(cuLaunchKernel(low, kResetBlocks, 1, 1, kResetThreads, 1, 1, 0, low_stream,
                         parameters.data(), nullptr),
          "accept logical graph dependency producer");
    check(cuEventRecord(waited, low_stream), "record pending logical graph dependency");
    bool has_incomplete_pause{};
    for (unsigned int cycle = 0; cycle < 3; ++cycle) {
        if (updates_nodes && cycle == 1) {
            check(runtime_symbol<decltype(&cudaGraphNodeSetEnabled)>("cudaGraphNodeSetEnabled")(
                      executable, nodes[1], 1),
                  "dynamic SDK root reenables kernel before original source retirement");
            retire_source();
        }
        if (cycle == 1 && !uses_null_stream) {
            markers[0] = 0;
            work = 150000;
            check(cuLaunchKernel(low, kResetBlocks, 1, 1, kResetThreads, 1, 1, 0, low_stream,
                                 parameters.data(), nullptr),
                  "accept fresh low producer behind completed graph dependency");
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{10};
            while (std::atomic_ref<std::uint32_t>(markers[0]).load(std::memory_order_acquire) ==
                   0) {
                require(std::chrono::steady_clock::now() < deadline,
                        "fresh low graph producer must make actual progress");
                std::this_thread::yield();
            }
        }
        const auto result = execute_owned_runtime_graph(executable, high_stream, is_ptds, [&] {
            ++originals;
            std::array<std::uint32_t, kResetElements> sample{};
            check(cuMemcpyDtoH(sample.data(), output, sizeof(sample)),
                  "sample physically stopped graph dependency before actual SDK issue");
            if (cycle == 0) {
                require(std::ranges::all_of(sample, [](auto value) { return value == 1; }),
                        "logical graph wait must complete its producer before native grant");
            } else if (cycle == 1 && !uses_null_stream) {
                has_incomplete_pause =
                    std::ranges::any_of(sample, [](auto value) { return value == 1; });
                require(has_incomplete_pause,
                        "native graph insertion physically retreats low work");
            }
            const auto native = sdk_launch(executable, high_stream);
            return native == cudaSuccess && fails_after_issue ? cudaErrorApiFailureBase : native;
        });
        if (fails_after_issue) {
            check(driver.stream_synchronize(high_stream), "prove one actual uncertain SDK replay");
            unsigned int value{};
            {
                const NativeDriverCallScope native;
                check(cudaMemcpy(&value, counter, sizeof(value), cudaMemcpyDeviceToHost),
                      "read actual once-only SDK effect after injected root error");
            }
            const auto retry = execute_owned_runtime_graph(executable, high_stream, is_ptds, [&] {
                ++originals;
                return sdk_launch(executable, high_stream);
            });
            require(result == cudaErrorApiFailureBase && retry == cudaErrorUnknown &&
                        originals == 1 && value == 1 && !is_inside_driver_call() &&
                        !is_inside_runtime_call(),
                    "post-issue Runtime error retains exact enum and forbids native retry");
            std::cout << "owned_runtime_graph_uncertain originals=1 value=1 error="
                      << static_cast<int>(result) << " retained_ownership=1 retry_issues=0"
                      << std::endl;
            // The injected status provides no completion/cleanup permission.
            // Keep the failed reservation and native resources until exec exit.
            std::_Exit(EXIT_SUCCESS);
        }
        check(result, "owned actual Runtime whole-graph replay");
        check(owner->synchronize_stream(high_stream, is_ptds), "prove exact native graph prefix");
        check(cuEventQuery(recorded), "native graph record has original query authority");
        if (outside_recorded)
            check(cuEventQuery(outside_recorded),
                  "unrelated outer record survives child replacement");
        unsigned int value{};
        {
            const NativeDriverCallScope native;
            check(cudaMemcpy(&value, counter, sizeof(value), cudaMemcpyDeviceToHost),
                  "read exact repeated original Runtime graph result");
        }
        CUcontext after{};
        check(driver.context_get_current(&after), "verify original SDK graph caller unchanged");
        const std::array last{cudaPeekAtLastError(), cudaGetLastError(), cudaPeekAtLastError()};
        const auto expected_counter = updates_nodes            ? cycle * 2
                                      : graph_update_kind != 0 ? (cycle + 1) * 2
                                                               : cycle + 1;
        require(originals == cycle + 1 && value == expected_counter && after == context &&
                    last == std::array{cudaSuccess, cudaSuccess, cudaSuccess} &&
                    !is_inside_driver_call() && !is_inside_runtime_call(),
                "SDK replay is exact once with unchanged caller and native last-error/TLS");
        check(owner->synchronize_stream(low_stream, false),
              "complete resumed graph producer exactly once");
    }
    std::array<std::uint32_t, kResetElements> output_values{};
    check(cuMemcpyDtoH(output_values.data(), output, sizeof(output_values)),
          "read exact graph producer output after all physical completions");
    const unsigned int expected_output = uses_null_stream ? 1 : 2;
    require(
        std::ranges::all_of(output_values, [&](auto value) { return value == expected_output; }),
        "graph dependency and priority resume lose or repeat no logical thread");
    const auto expected_completed = uses_null_stream ? 4U : 5U;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{10};
    auto stats = owner->stats();
    while (stats.acknowledged_completions != expected_completed) {
        require(stats.acknowledged_failures == 0 && std::chrono::steady_clock::now() < deadline,
                "graph native prefixes require terminal control acknowledgment");
        std::this_thread::yield();
        stats = owner->stats();
    }
    require(stats.granted_executions == (uses_null_stream ? 4U : 6U) &&
                stats.retreat_requests == (uses_null_stream ? 0U : 1U) &&
                stats.acknowledged_suspensions == (uses_null_stream ? 0U : 1U) &&
                stats.logical_resumes == (uses_null_stream ? 0U : 1U) &&
                stats.acknowledged_failures == 0 && (uses_null_stream || has_incomplete_pause),
            "Runtime bridge keeps dependency-before-grant and actual distinct-priority handoff");
    if (uses_runtime_constructor) {
        check(cudaGraphExecDestroy(executable),
              "actual SDK root retires its executable registration");
    } else {
        check(cuGraphExecDestroy(executable), "retire actual native graph executable");
    }
    require(cudaGraphLaunch(executable, high_stream) == cudaErrorUnknown &&
                cuGraphLaunch(executable, high_stream) == CUDA_ERROR_INVALID_HANDLE,
            "public replay rejects retired metadata before querying or issuing the SDK handle");
    require(execute_owned_runtime_graph(executable, high_stream, is_ptds,
                                        [&] {
                                            ++originals;
                                            return sdk_launch(executable, high_stream);
                                        }) == cudaErrorUnknown &&
                originals == 3,
            "retired CPU graph identity cannot issue or query its destroyed native handle");
    if (uses_runtime_constructor) {
        cudaGraph_t empty{};
        check(cudaGraphCreate(&empty, 0), "create original source for SDK slot reuse");
        for (unsigned int cycle = 0; cycle < 260; ++cycle) {
            cudaGraphExec_t reused{};
            check(cudaGraphInstantiateWithFlags(&reused, empty, 0),
                  "reuse prepared SDK graph slot");
            check(cudaGraphExecDestroy(reused), "retire SDK graph slot without capacity leak");
        }
#if CUDART_VERSION >= 12000
        using Instantiate = decltype(&cudaGraphInstantiateWithParams);
        for (const auto* name :
             {"cudaGraphInstantiateWithParams", "cudaGraphInstantiateWithParams_ptsz"}) {
            const auto construct = reinterpret_cast<Instantiate>(::dlsym(RTLD_DEFAULT, name));
            require(construct, "covered actual Runtime params ABI is available");
            cudaGraphExec_t unprepared{};
            cudaGraphInstantiateParams parameters{};
            check(construct(&unprepared, empty, &parameters),
                  "preserve original SDK params constructor");
            CudaNativeGraphExecutable metadata;
            require(router->graphs().lookup(unprepared, metadata) == CUDA_ERROR_NOT_SUPPORTED,
                    "unmodeled params/upload constructor does not authorize replay");
            check(cudaGraphExecDestroy(unprepared), "retire original SDK params executable");
        }
#endif
        check(cudaGraphDestroy(empty), "destroy original SDK reuse graph");
    }
    check(cuEventDestroy(recorded), "retire completed native graph record event");
    check(cuEventDestroy(waited), "retire completed logical graph wait event");
    if (outside_recorded)
        check(cuEventDestroy(outside_recorded), "retire completed unrelated outer record");
    check(cuModuleUnload(module), "retire drained graph producer module");
    check(cuMemFree(output), "release charged graph dependency output");
    check(router->close_context(context, [] { return CUDA_SUCCESS; }), "close checked graph owner");
    owner.reset();
    {
        const NativeDriverCallScope native;
        check(original_runtime_symbol<RuntimeFreeFunction>("cudaFree")(counter),
              "free completed original SDK graph counter");
        check(cuStreamDestroy(low_stream), "destroy completed low graph stream");
        if (high_stream) {
            check(cuStreamDestroy(high_stream), "destroy completed high graph stream");
        }
        check(cuMemFreeHost(markers), "free physically stopped graph markers");
    }
    std::size_t available{}, total{};
    check(cuMemGetInfo(&available, &total), "verify complete owned graph quota recovery");
    require(available == total && total == 8388608, "checked graph cleanup restores exact quota");
    {
        const NativeDriverCallScope native;
        check(original_runtime_reset()(), "check final original SDK graph cleanup");
    }
    std::cout << "owned_runtime_graph ptds=" << is_ptds << " null=" << uses_null_stream
              << " runtime_constructor=" << uses_runtime_constructor
              << " base_constructor=" << uses_base_constructor
              << " event_updates=" << updates_events << " node_updates=" << updates_nodes
              << " graph_update_kind=" << graph_update_kind
              << " originals=3 nodes=" << (updates_child ? 5 : 3) << " output=" << expected_output
              << " grants=" << stats.granted_executions << " retreat=" << stats.retreat_requests
              << " completed=" << stats.acknowledged_completions
              << " checked_cleanup=1 public_activation=0" << std::endl;
}

struct PendingResetProbe {
    DriverDispatch& driver;
    DriverCooperativeRouter& router;
    CUcontext context;
    CUdeviceptr output;
    std::uint32_t* markers;
    RuntimeDeviceResetFunction original;
    std::shared_ptr<CudaCooperativeFrontend> owner;
    std::latch selected{1};
    std::latch may_close{1};
    std::array<std::uint32_t, kResetElements> sampled{};
    unsigned int originals{};
    CUcontext after{};
    std::array<cudaError_t, 3> last{};
    bool has_selected{};
    bool has_pending_pause{};
    bool has_completed_high{};
    bool has_preserved_caller{};
    bool has_exact_handoff{};
};
thread_local PendingResetProbe* g_pending_reset_probe{};

cudaError_t reset_after_logical_proof() {
    auto& probe = *g_pending_reset_probe;
    ++probe.originals;
    CUcontext caller{};
    check(probe.driver.context_get_current(&caller), "observe untouched reset SDK caller");
    probe.has_preserved_caller = caller == probe.context;
    // close_context has joined the private worker before invoking the SDK.
    // Inspect CPU terminal counters and sample live original output here, never
    // submit these handles after actual reset destroys the primary context.
    const auto stats = probe.owner->stats();
    probe.has_exact_handoff = stats.granted_executions == 3 && stats.retreat_requests == 1 &&
                              stats.acknowledged_suspensions == 1 && stats.logical_resumes == 1 &&
                              stats.acknowledged_completions == 2 &&
                              stats.acknowledged_failures == 0;
    check(probe.router.with_context(probe.context,
                                    [&] {
                                        const NativeDriverCallScope native;
                                        check(cuMemcpyDtoH(probe.sampled.data(), probe.output,
                                                           sizeof(probe.sampled)),
                                              "sample logical output before SDK reset destruction");
                                        probe.has_completed_high = probe.markers[1] == 2;
                                        check(cuMemFreeHost(probe.markers),
                                              "release proven stopped mapped marker before reset");
                                        probe.markers = nullptr;
                                        return CUDA_SUCCESS;
                                    }),
          "restore bound SDK reset caller after live output sampling");
    const NativeDriverCallScope native;
    return probe.original();
}

void check_pending_bound_reset(bool is_cached = false) {
    const ProcessCallGate::Call call(cooperative_process_call_gate());
    DriverDispatch driver;
    require(driver.initialize(), "resolve pending unbound reset original dispatch");
    check(cuInit(0), "initialize pending reset CPU routing");
    {
        const NativeDriverCallScope native;
        check(cudaSetDevice(0), "initialize original pending reset primary");
    }
    CUcontext context{};
    check(driver.context_get_current(&context), "observe pending reset primary");
    auto* router = cooperative_source_router();
    require(router, "pending reset has CPU router");
    CUstream stream{}, high_stream{};
    CUdeviceptr output{}, mapped{};
    std::uint32_t* markers{};
    // Separate pages prevent overlapping host registrations. SDK reset owns
    // the non-portable registrations, but never this caller-owned CPU storage.
    alignas(4096) std::array<std::uint32_t, 1024> marker_storage{};
    alignas(4096) std::array<std::uint32_t, 7168> readback{};
    {
        const NativeDriverCallScope native;
        check(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), "create pending reset low stream");
        check(cuStreamCreateWithPriority(&high_stream, CU_STREAM_NON_BLOCKING, -1),
              "create pending reset priority stream");
        check(cuMemAlloc(&output, kResetElements * sizeof(std::uint32_t)),
              "allocate original pending reset output");
        check(cuMemsetD32(output, 0, kResetElements),
              "initialize pending reset exactly-once output");
        if (is_cached) {
            markers = marker_storage.data();
            check(cuMemHostRegister(markers, sizeof(marker_storage), CU_MEMHOSTREGISTER_DEVICEMAP),
                  "register caller-owned cached reset markers");
            check(cuMemHostRegister(readback.data(), sizeof(readback), 0),
                  "register caller-owned cached reset readback");
        } else {
            check(cuMemHostAlloc(reinterpret_cast<void**>(&markers), 2 * sizeof(std::uint32_t),
                                 CU_MEMHOSTALLOC_DEVICEMAP),
                  "allocate mapped pending reset markers");
        }
        check(cuMemHostGetDevicePointer(&mapped, markers, 0), "map pending reset markers");
    }
    markers[0] = 0;
    markers[1] = 0;
    CUmodule module{};
    CUfunction low{}, high{};
    check(cuModuleLoadData(&module, kCooperativePtx), "load public qualified reset PTX");
    check(cuModuleGetFunction(&low, module, "cooperative"), "resolve public reset low function");
    check(cuModuleGetFunction(&high, module, "priority_probe"), "warm public reset high function");
    std::shared_ptr<CudaCooperativeFrontend> owner;
    check(router->existing_frontend(context, owner), "observe prepared pending reset owner");
    require(owner != nullptr, "public reset PTX preparation creates the actual private actor");
    void* charged{};
    cudaEvent_t event{};
    check(cudaMalloc(&charged, 4096), "allocate tracked pending reset resource");
    check(cudaEventCreate(&event), "create tracked pending reset metadata event");
    unsigned int work = 150000;
    std::array<void*, 3> arguments{&output, &mapped, &work};
    check(cuLaunchKernel(low, kResetBlocks, 1, 1, kResetThreads, 1, 1, 0, stream, arguments.data(),
                         nullptr),
          "launch ordinary low work before reset");
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{10};
    while (std::atomic_ref<std::uint32_t>(markers[0]).load(std::memory_order_acquire) == 0) {
        require(std::chrono::steady_clock::now() < deadline,
                "pending reset low work makes progress");
        std::this_thread::sleep_for(std::chrono::milliseconds{1});
    }
    CUdeviceptr high_marker = mapped + sizeof(std::uint32_t);
    std::array<void*, 1> high_arguments{&high_marker};
    PendingResetProbe probe{driver, *router, context, output, markers, original_runtime_reset(),
                            owner};
    std::optional<std::jthread> caller;
    cudaError_t status{cudaErrorUnknown};
    bool has_released{};
    const auto release = [&] {
        if (!has_released) {
            has_released = true;
            probe.may_close.count_down();
        }
    };
    try {
        const auto inserted = owner->native(
            high_stream, false,
            [&] {
                // The acknowledged native pause physically stops low work.
                // Sample actual incomplete output before thaw, not after a
                // timing-sensitive high-stream synchronization on the caller.
                const NativeDriverCallScope native;
                check(cuMemcpyDtoH(probe.sampled.data(), output, sizeof(probe.sampled)),
                      "sample physically stopped incomplete reset output");
                probe.has_pending_pause = static_cast<std::size_t>(std::ranges::count(
                                              probe.sampled, 1U)) < kResetElements;
                check(driver.launch_kernel(high, 1, 1, 1, 1, 1, 1, 0, high_stream,
                                           high_arguments.data(), nullptr),
                      "issue original priority work during proven low pause");
                caller.emplace([&] {
                    g_pending_reset_probe = &probe;
                    try {
                        const ProcessCallGate::Call reset_call(cooperative_process_call_gate());
                        {
                            const NativeDriverCallScope bind;
                            check(cuCtxSetCurrent(context), "bind actual destructive reset caller");
                        }
                        probe.has_selected = true;
                        probe.selected.count_down();
                        // Main must finish its held pause and unbind before
                        // this caller can join workers and destroy the primary.
                        probe.may_close.wait();
                        if (is_cached) {
                            CUcontext before{};
                            check(driver.context_get_current(&before), "observe cached SDK caller");
                            probe.has_preserved_caller = before == context;
                            ++probe.originals;
                            // No Runtime wrapper, typed root or native guard.
                            // The SDK's actual cached Driver reset owns cleanup.
                            status = probe.original();
                        } else {
                            status = intercept_runtime_device_reset(reset_after_logical_proof);
                        }
                        check(driver.context_get_current(&probe.after),
                              "observe original reset caller before any reinitialization");
                        probe.last = {cudaPeekAtLastError(), cudaGetLastError(),
                                      cudaPeekAtLastError()};
                    } catch (...) {
                        status = cudaErrorUnknown;
                    }
                    g_pending_reset_probe = nullptr;
                    if (!probe.has_selected) {
                        probe.selected.count_down();
                    }
                });
                probe.selected.wait();
                return CUDA_SUCCESS;
            },
            true);
        if (is_cached) {
            check(cuMemcpyDtoHAsync(readback.data(), output, sizeof(probe.sampled), stream),
                  "accept ordered caller-owned readback before cached reset destruction");
        }
        {
            const NativeDriverCallScope native;
            check(cuCtxSetCurrent(nullptr), "unbind main caller before permitting owner closure");
        }
        release();
        if (caller && caller->joinable()) {
            caller->join();
        }
        check(inserted, "finish priority pause independently before joining reset caller");
    } catch (...) {
        release();
        if (caller && caller->joinable()) {
            caller->join();
        }
        throw;
    }
    if (is_cached) {
        // The private worker has joined. Neither access submits a destroyed
        // CUDA handle: these arrays belong to the application's CPU caller.
        std::copy_n(readback.begin(), kResetElements, probe.sampled.begin());
        probe.has_completed_high = marker_storage[1] == 2;
        probe.markers = nullptr;
        const auto stats = probe.owner->stats();
        probe.has_exact_handoff =
            stats.granted_executions == 3 && stats.retreat_requests == 1 &&
            stats.acknowledged_suspensions == 1 && stats.logical_resumes == 1 &&
            stats.acknowledged_completions == 2 && stats.acknowledged_failures == 0;
    }
    CudaEventProjection retired;
    const auto registration = router->event_projection(event, retired);
    std::shared_ptr<CudaCooperativeFrontend> retained;
    check(router->existing_frontend(context, retained),
          "inspect closed pending reset actor CPU-only");
    probe.owner.reset();
    owner.reset();
    {
        const NativeDriverCallScope native;
        check(cudaSetDevice(0), "reinitialize after pending reset SDK observation");
        if (is_cached) {
            check(cuMemHostRegister(marker_storage.data(), sizeof(marker_storage),
                                    CU_MEMHOSTREGISTER_DEVICEMAP),
                  "prove old cached marker registration retired before new generation");
            check(cuMemHostRegister(readback.data(), sizeof(readback), 0),
                  "prove old cached readback registration retired before new generation");
            check(cuMemHostUnregister(marker_storage.data()), "release new marker registration");
            check(cuMemHostUnregister(readback.data()), "release new readback registration");
        }
    }
    std::size_t available{}, total{};
    check(cuMemGetInfo(&available, &total), "observe full pending reset quota recovery");
    {
        const NativeDriverCallScope native;
        check(probe.original(), "check final SDK cleanup after pending reset proof");
    }
    const auto completed = static_cast<std::size_t>(std::ranges::count(probe.sampled, 1U));
    std::cout << "normal_runtime_pending_reset cached=" << is_cached << " status=" << status
              << " sdk_calls=" << probe.originals << " completed_before_destroy=" << completed
              << '/' << kResetElements << " pending_at_pause=" << probe.has_pending_pause
              << " exact_handoff=" << probe.has_exact_handoff << " registration=" << registration
              << " quota=" << available << '/' << total << " checked_cleanup=1" << std::endl;
    require(status == cudaSuccess && probe.after == context && probe.has_preserved_caller &&
                probe.originals == 1 &&
                probe.last == std::array{cudaSuccess, cudaSuccess, cudaSuccess},
            "pending reset reaches the actual SDK once on its original bound caller");
    require(completed == kResetElements && probe.has_pending_pause && probe.has_completed_high &&
                probe.has_exact_handoff && !probe.markers && !retained &&
                registration == CUDA_ERROR_INVALID_HANDLE && available == total && total == 8388608,
            "reset drains logical work, acknowledges exact handoff and retires private/quota "
            "ownership");
}

void check_nonprimary_reset(bool is_native, bool has_active_primary = false) {
    const ProcessCallGate::Call call(cooperative_process_call_gate());
    DriverDispatch driver;
    require(driver.initialize(), "resolve non-primary reset original dispatch");
    const auto native_reset = original_runtime_reset();
    check(cuInit(0), "initialize non-primary reset regression");
    CUdevice device{};
    CUcontext context{};
    {
        const NativeDriverCallScope native;
        check(cuDeviceGet(&device, 0), "resolve non-primary reset device");
        if (has_active_primary) {
            check(cudaSetDevice(0), "initialize an independent original primary context");
        }
        check(cuCtxCreate(&context, 0, device), "create original non-primary reset context");
        check(cudaFree(nullptr), "initialize Runtime on application Driver context");
    }
    bool is_charge_retained = true;
    bool is_native_lifetime_preserved = true;
    for (unsigned int cycle = 0; cycle < 3; ++cycle) {
        std::size_t before{}, total{}, charged{}, after{}, released{};
        CUdeviceptr pointer{};
        std::shared_ptr<CudaCooperativeFrontend> owner;
        if (!is_native) {
            auto* router = cooperative_source_router();
            require(router, "non-primary reset has an existing CPU router");
            check(router->frontend(context, owner), "initialize non-primary actor before reset");
        }
        if (is_native) {
            const NativeDriverCallScope native;
            check(cuMemAlloc(&pointer, 4096), "allocate original non-primary resource");
        } else {
            check(cuMemGetInfo(&before, &total), "read non-primary quota baseline");
            check(cuMemAlloc(&pointer, 4096), "allocate charged non-primary resource");
            check(cuMemGetInfo(&charged, &total), "read non-primary resource charge");
            require(before - charged == 4096, "Driver allocation charged exactly once");
        }
        cudaError_t status{};
        if (is_native) {
            const NativeDriverCallScope native;
            status = native_reset();
        } else {
            status = cudaDeviceReset();
        }
        CUcontext current{};
        check(driver.context_get_current(&current), "observe actual post-reset context");
        const std::array last{cudaPeekAtLastError(), cudaGetLastError(), cudaPeekAtLastError()};
        CUdeviceptr base{};
        std::size_t bytes{};
        {
            const NativeDriverCallScope native;
            check(cuMemGetAddressRange(&base, &bytes, pointer),
                  "prove Driver allocation survives non-primary Runtime reset");
        }
        is_native_lifetime_preserved &= status == cudaSuccess && current == context &&
                                        base == pointer && bytes >= 4096 &&
                                        last == std::array{cudaSuccess, cudaSuccess, cudaSuccess};
        if (!is_native) {
            check(cuMemGetInfo(&after, &total), "observe charge after non-primary reset");
            // Reset releases checked private progress, not application Driver memory.
            is_charge_retained &= total - after == 4096;
            check(cuMemFree(pointer), "explicitly release surviving Driver allocation");
            check(cuMemGetInfo(&released, &total), "observe exact explicit-release recovery");
            is_charge_retained &= released == total;
            std::shared_ptr<CudaCooperativeFrontend> retired;
            check(cooperative_source_router()->existing_frontend(context, retired),
                  "inspect retired non-primary actor without importing a new one");
            require(!retired, "successful reset retires its checked old private actor");
        } else {
            const NativeDriverCallScope native;
            check(cuMemFree(pointer), "release original surviving Driver allocation");
        }
        owner.reset();
        std::cout << "normal_runtime_nonprimary_reset native=" << is_native
                  << " active_primary=" << has_active_primary << " cycle=" << cycle
                  << " status=" << status << " context_preserved=" << (current == context)
                  << " allocation_survives=" << (base == pointer)
                  << " charge_retained=" << (is_native || total - after == 4096)
                  << " checked_release=1" << std::endl;
    }
    check(cuCtxDestroy(context), "destroy original non-primary context after checked cleanup");
    if (has_active_primary) {
        const NativeDriverCallScope native;
        check(native_reset(), "release independent original primary after checked cleanup");
    }
    require(is_native_lifetime_preserved, "non-primary reset retains original Driver lifetime/TLS");
    require(is_charge_retained, "Runtime-state reset must not release surviving Driver quota");
}

void check_pending_cached_nonprimary_reset(DriverDispatch& driver, DriverCooperativeRouter& router,
                                           const std::shared_ptr<CudaCooperativeFrontend>& owner,
                                           CUcontext context, CUstream stream, CUevent event,
                                           CUdeviceptr counter, RuntimeDeviceResetFunction reset) {
#ifdef GLIMMER_HAS_RUNTIME_CAPTURE_KERNEL
    CUdeviceptr output{}, mapped{};
    CUstream high_stream{};
    std::uint32_t* markers{};
    std::uint32_t* readback{};
    check(cuMemAlloc(&output, kResetElements * sizeof(std::uint32_t)),
          "allocate charged pending non-primary output");
    {
        const NativeDriverCallScope native;
        check(cuStreamCreateWithPriority(&high_stream, CU_STREAM_NON_BLOCKING, -1),
              "create pending non-primary priority stream");
        check(cuMemsetD32(output, 0, kResetElements), "initialize pending non-primary output");
        check(cuMemHostAlloc(reinterpret_cast<void**>(&markers), 8, CU_MEMHOSTALLOC_DEVICEMAP),
              "allocate pending non-primary mapped markers");
        check(cuMemHostGetDevicePointer(&mapped, markers, 0), "map pending non-primary markers");
        check(cuMemAllocHost(reinterpret_cast<void**>(&readback),
                             kResetElements * sizeof(std::uint32_t)),
              "allocate pending non-primary pinned readback");
    }
    markers[0] = markers[1] = 0;
    CUmodule module{};
    CUfunction low{}, high{};
    check(cuModuleLoadData(&module, kCooperativePtx), "load pending non-primary ordinary PTX");
    check(cuModuleGetFunction(&low, module, "cooperative"), "prepare pending non-primary low work");
    check(cuModuleGetFunction(&high, module, "priority_probe"),
          "prepare non-primary priority work");
    const auto settled_stats = [&] {
        // CUDA completion snapshots do not acknowledge asynchronous control
        // reports. Join an empty checkpoint before comparing terminal counts;
        // the independently live non-primary owner can then resume unchanged.
        CudaWorkerCheckpoint checkpoint;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{60};
        CUresult result{};
        do {
            result = owner->checkpoint(checkpoint);
            require(result == CUDA_SUCCESS || result == CUDA_ERROR_NOT_READY,
                    "terminal counter checkpoint has no uncertain ownership");
            require(std::chrono::steady_clock::now() < deadline,
                    "terminal counter checkpoint completes all control acknowledgments");
            if (result == CUDA_ERROR_NOT_READY) {
                std::this_thread::sleep_for(std::chrono::milliseconds{1});
            }
        } while (result == CUDA_ERROR_NOT_READY);
        check(owner->resume_checkpoint(checkpoint),
              "resume independently live non-primary actor after joined counter observation");
        // stats() is an admitted worker command, not a parked-owner accessor.
        // Resuming admits that read; no new work can change the settled counts.
        return owner->stats();
    };
    const auto before_stats = settled_stats();
    unsigned int work = 150000;
    std::array<void*, 3> arguments{&output, &mapped, &work};
    check(cuLaunchKernel(low, kResetBlocks, 1, 1, kResetThreads, 1, 1, 0, stream, arguments.data(),
                         nullptr),
          "accept long low work before cached non-primary reset");
    const auto progress_deadline = std::chrono::steady_clock::now() + std::chrono::seconds{60};
    while (std::atomic_ref<std::uint32_t>(markers[0]).load(std::memory_order_acquire) == 0) {
        require(std::chrono::steady_clock::now() < progress_deadline,
                "pending non-primary kernel makes actual progress");
        std::this_thread::sleep_for(std::chrono::milliseconds{1});
    }
    // Kernel argument bytes carry a GPU virtual address, not CPU provenance.
    auto* runtime_counter = std::bit_cast<unsigned int*>(counter);
    std::array<void*, 1> runtime_arguments{&runtime_counter};
    check(cuMemcpyDtoHAsync(readback, output, kResetElements * sizeof(std::uint32_t), stream),
          "accept ordered pending non-primary readback");
    check(cuEventRecord(event, stream), "accept exact pending non-primary event epoch");
    CudaEventProjection registered;
    check(router.event_projection(event, registered), "snapshot pending non-primary event owner");
    std::size_t charged{}, available{}, total{};
    check(cuMemGetInfo(&charged, &total), "snapshot all pending non-primary charges");
    std::array<std::uint32_t, kResetElements> sampled{};
    std::atomic<bool> is_done{};
    cudaError_t status{cudaErrorUnknown};
    CUcontext after{};
    std::array<cudaError_t, 3> last{};
    unsigned int originals{};
    std::optional<std::jthread> caller;
    bool has_pending_pause{};
    bool has_returned_during_pause{};
    CUdeviceptr high_marker = mapped + sizeof(std::uint32_t);
    std::array<void*, 1> high_arguments{&high_marker};
    const auto inserted = owner->native(
        high_stream, false,
        [&] {
            const NativeDriverCallScope native;
            check(cuMemcpyDtoH(sampled.data(), output, sizeof(sampled)),
                  "sample physically stopped incomplete non-primary work");
            has_pending_pause =
                static_cast<std::size_t>(std::ranges::count(sampled, 1U)) < kResetElements;
            caller.emplace([&] {
                try {
                    const ProcessCallGate::Call reset_call(cooperative_process_call_gate());
                    {
                        const NativeDriverCallScope bind;
                        check(cuCtxSetCurrent(context), "bind independent non-primary SDK caller");
                    }
                    ++originals;
                    // Neither a Runtime root nor a native invocation guard.
                    status = reset();
                    check(driver.context_get_current(&after), "observe pending reset SDK caller");
                    last = {cudaPeekAtLastError(), cudaGetLastError(), cudaPeekAtLastError()};
                    {
                        const NativeDriverCallScope unbind;
                        check(cuCtxSetCurrent(nullptr), "unbind checked pending reset caller");
                    }
                } catch (...) {
                    status = cudaErrorUnknown;
                }
                is_done.store(true, std::memory_order_release);
            });
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
            while (!is_done.load(std::memory_order_acquire) &&
                   std::chrono::steady_clock::now() < deadline) {
                std::this_thread::sleep_for(std::chrono::milliseconds{1});
            }
            has_returned_during_pause = is_done.load(std::memory_order_acquire);
            // Never join a potentially blocked reset while retaining the pause.
            return driver.launch_kernel(high, 1, 1, 1, 1, 1, 1, 0, high_stream,
                                        high_arguments.data(), nullptr);
        },
        true);
    if (caller) {
        caller->join();
    }
    check(inserted, "finish pending non-primary priority insertion");
    check(cuEventSynchronize(event), "complete retained pending non-primary event prefix");
    CudaEventProjection retained;
    check(router.refresh_event_projection(registered, retained),
          "pending SDK reset preserves exact event registration");
    std::shared_ptr<CudaCooperativeFrontend> retained_owner;
    check(router.existing_frontend(context, retained_owner), "observe retained pending actor");
    // An opaque Runtime kernel may require a drained native boundary. Issue it
    // only after proving the independent cooperative pending-reset handoff.
    check(cudaLaunchKernel(glimmer_runtime_capture_kernel(), dim3{2}, dim3{1},
                           runtime_arguments.data(), 0, stream),
          "launch actual Runtime kernel after pending cached reset");
    check(cuEventRecord(event, stream), "record post-reset Runtime kernel on retained event");
    check(cuEventSynchronize(event), "complete post-reset actual Runtime kernel");
    check(cuStreamSynchronize(high_stream), "complete original priority fence after cached reset");
    check(owner->synchronize_context(), "complete every retained control acknowledgment");
    check(cuMemGetInfo(&available, &total), "observe all preserved pending charges");
    unsigned int result{};
    {
        const NativeDriverCallScope native;
        check(cuMemcpyDtoH(&result, counter, sizeof(result)),
              "observe queued Runtime kernel result");
    }
    const auto stats = settled_stats();
    const bool has_completed =
        std::all_of(readback, readback + kResetElements, [](auto value) { return value == 1; });
    check(cuStreamDestroy(high_stream), "close pending non-primary priority stream");
    check(cuModuleUnload(module), "retire checked pending non-primary source module");
    check(cuMemFree(output), "release completed pending non-primary charge");
    {
        const NativeDriverCallScope native;
        check(cuMemFreeHost(readback), "release checked pending non-primary readback");
        check(cuMemFreeHost(markers), "release checked pending non-primary markers");
    }
    std::cout << "cached_nonprimary_pending originals=" << originals << " status=" << status
              << " incomplete_pause=" << has_pending_pause
              << " returned_in_pause=" << has_returned_during_pause
              << " grants=" << stats.granted_executions - before_stats.granted_executions
              << " completed="
              << stats.acknowledged_completions - before_stats.acknowledged_completions
              << " exact_counter=" << result << std::endl;
    require(status == cudaSuccess && originals == 1 && after == context &&
                last == std::array{cudaSuccess, cudaSuccess, cudaSuccess} && has_pending_pause &&
                has_returned_during_pause && has_completed && result == 18 &&
                retained_owner == owner && charged == available &&
                stats.retreat_requests - before_stats.retreat_requests == 1 &&
                stats.acknowledged_suspensions - before_stats.acknowledged_suspensions == 1 &&
                stats.logical_resumes - before_stats.logical_resumes == 1 &&
                stats.granted_executions - before_stats.granted_executions == 4 &&
                stats.acknowledged_completions - before_stats.acknowledged_completions == 3,
            "cached non-primary reset preserves pending work, native state and exact handoff");
#else
    (void)driver;
    (void)router;
    (void)owner;
    (void)context;
    (void)stream;
    (void)event;
    (void)counter;
    (void)reset;
    throw std::runtime_error("pending cached reset requires the actual SDK kernel");
#endif
}

void check_cached_nonprimary_reset(bool is_native, bool has_active_primary) {
#ifdef GLIMMER_HAS_RUNTIME_CAPTURE_KERNEL
    const ProcessCallGate::Call call(cooperative_process_call_gate());
    DriverDispatch driver;
    require(driver.initialize(), "resolve cached non-primary reset original dispatch");
    const auto reset = original_runtime_reset();
    check(cuInit(0), "initialize cached non-primary reset routing");
    CUdevice device{};
    CUcontext context{};
    CUstream stream{};
    CUevent event{};
    CUdeviceptr pointer{};
    {
        const NativeDriverCallScope native;
        check(cuDeviceGet(&device, 0), "resolve cached non-primary device");
        if (has_active_primary) {
            check(cudaSetDevice(0), "activate independent cached-reset primary");
        }
        check(cuCtxCreate(&context, 0, device), "create cached-reset application context");
        check(cudaFree(nullptr), "warm actual SDK cache inside trusted native scope");
        check(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), "create independent Driver stream");
        if (is_native) {
            check(cuEventCreate(&event, 0), "create independent original Driver event");
        }
    }
    if (!is_native) {
        check(cuEventCreate(&event, 0), "create independently owned logical Driver event");
    }
    if (is_native) {
        const NativeDriverCallScope native;
        check(cuMemAlloc(&pointer, 4096), "allocate original cached-reset Driver storage");
    } else {
        check(cuMemAlloc(&pointer, 4096), "allocate charged cached-reset Driver storage");
    }
    {
        const NativeDriverCallScope native;
        check(cuMemsetD32(pointer, 0, 1), "initialize exact cached-reset counter");
    }
    auto* router = cooperative_source_router();
    std::shared_ptr<CudaCooperativeFrontend> owner;
    if (!is_native) {
        require(router, "cached non-primary reset has a configured router");
        check(router->frontend(context, owner), "initialize exact cached-reset actor");
    }
    auto* counter = std::bit_cast<unsigned int*>(pointer);
    std::array<void*, 1> arguments{&counter};
    const auto native_launch =
        original_runtime_symbol<decltype(&cudaLaunchKernel)>("cudaLaunchKernel");
    const auto native_launch_ptds =
        original_runtime_symbol<decltype(&cudaLaunchKernel)>("cudaLaunchKernel_ptsz");
    const auto public_launch_ptds =
        runtime_symbol<decltype(&cudaLaunchKernel)>("cudaLaunchKernel_ptsz");
    {
        // Private SDK warmup must remain native even though its cache bindings
        // are controlled for subsequent public invocations.
        const NativeDriverCallScope native;
        const std::array entries{is_native ? native_launch : &cudaLaunchKernel,
                                 is_native ? native_launch_ptds : public_launch_ptds};
        for (const auto issue : entries) {
            check(issue(glimmer_runtime_capture_kernel(), dim3{2}, dim3{1}, arguments.data(), 0,
                        stream),
                  "private Runtime kernel warmup must preserve actual SDK status");
        }
        check(cuStreamSynchronize(stream), "complete original private Runtime warmup");
        unsigned int result{};
        check(cuMemcpyDtoH(&result, pointer, sizeof(result)), "observe once-only private warmup");
        require(result == 4, "both private warmup aliases issue original kernel exactly once");
    }
    const auto launch = [&] {
        const auto invoke = [&] {
            const auto issue = is_native ? native_launch : &cudaLaunchKernel;
            check(issue(glimmer_runtime_capture_kernel(), dim3{2}, dim3{1}, arguments.data(), 0,
                        stream),
                  "launch actual registered Runtime kernel across cached reset");
            check(cuEventRecord(event, stream), "record independently owned Driver event");
            check(cuEventSynchronize(event), "complete actual Runtime kernel dependency");
        };
        if (is_native) {
            const NativeDriverCallScope native;
            invoke();
        } else {
            invoke();
        }
    };
    bool has_preserved_lifetime = true;
    bool has_preserved_charge = true;
    for (unsigned int cycle = 0; cycle < 3; ++cycle) {
        launch();
        CudaEventProjection registered;
        std::size_t before{}, after{}, total{};
        if (!is_native) {
            check(router->event_projection(event, registered), "snapshot pre-reset Driver event");
            check(cuMemGetInfo(&before, &total), "snapshot complete cached-reset charge");
        }
        cudaError_t status{};
        if (is_native) {
            const NativeDriverCallScope native;
            status = reset();
        } else {
            // Actual cached SDK entry: no Runtime wrapper or invocation bypass.
            status = reset();
        }
        CUcontext current{};
        check(driver.context_get_current(&current), "observe cached-reset original caller");
        const std::array last{cudaPeekAtLastError(), cudaGetLastError(), cudaPeekAtLastError()};
        CUdeviceptr base{};
        std::size_t bytes{};
        {
            const NativeDriverCallScope native;
            check(cuMemGetAddressRange(&base, &bytes, pointer), "observe surviving Driver storage");
            check(cuEventQuery(event), "observe original completed Driver event after SDK reset");
        }
        has_preserved_lifetime &= status == cudaSuccess && current == context && base == pointer &&
                                  bytes >= 4096 &&
                                  last == std::array{cudaSuccess, cudaSuccess, cudaSuccess};
        if (!is_native) {
            CudaEventProjection retained;
            check(router->refresh_event_projection(registered, retained),
                  "cached reset preserves exact Driver event registration");
            std::shared_ptr<CudaCooperativeFrontend> retained_owner;
            check(router->existing_frontend(context, retained_owner),
                  "cached reset observes exact actor without creating another");
            require(retained_owner == owner, "non-destructive cached reset keeps original actor");
            check(cuMemGetInfo(&after, &total), "observe complete retained cached-reset charge");
            has_preserved_charge &= before == after;
        }
        launch();
        unsigned int result{};
        {
            const NativeDriverCallScope native;
            check(cuMemcpyDtoH(&result, pointer, sizeof(result)),
                  "read exact post-reset kernel result");
        }
        has_preserved_lifetime &= result == 4 * (cycle + 1) + 4;
        std::cout << "cached_nonprimary_reset native=" << is_native
                  << " active_primary=" << has_active_primary << " cycle=" << cycle
                  << " status=" << status << " exact_counter=" << result
                  << " context_preserved=" << (current == context)
                  << " charge_preserved=" << (is_native || before == after) << std::endl;
    }
    if (!is_native) {
        check_pending_cached_nonprimary_reset(driver, *router, owner, context, stream, event,
                                              pointer, reset);
        check(cuEventDestroy(event), "retire surviving logical Driver event");
        check(cuStreamDestroy(stream), "retire surviving logical Driver stream");
        check(cuMemFree(pointer), "explicitly release surviving cached-reset application charge");
        check(router->close_context(context, [] { return CUDA_SUCCESS; }),
              "close surviving cached-reset private actor before native destruction");
        owner.reset();
        std::size_t available{}, total{};
        check(cuMemGetInfo(&available, &total), "observe checked cached-reset quota recovery");
        has_preserved_charge &= available == total;
    } else {
        const NativeDriverCallScope native;
        check(cuEventDestroy(event), "destroy original surviving Driver event");
        check(cuStreamDestroy(stream), "destroy original surviving Driver stream");
        check(cuMemFree(pointer), "free original surviving Driver storage");
    }
    check(cuCtxDestroy(context), "destroy checked cached-reset non-primary context");
    if (has_active_primary) {
        const NativeDriverCallScope native;
        check(reset(), "release independent original primary after cached-reset cleanup");
    }
    require(has_preserved_lifetime, "cached SDK non-primary reset preserves native lifetime/TLS");
    require(has_preserved_charge, "cached SDK non-primary reset preserves exact owned quota");
#else
    (void)is_native;
    (void)has_active_primary;
    throw std::runtime_error("cached non-primary reset requires the actual registered SDK kernel");
#endif
}

void check_cold_device_sync(bool is_native) {
    const ProcessCallGate::Call call(cooperative_process_call_gate());
    DriverDispatch driver;
    require(driver.initialize(), "resolve first-use device-sync dispatch");
    check(driver.init(0), "initialize Driver without initializing Runtime device sync");
    CUcontext before{}, after{};
    check(driver.context_get_current(&before), "observe original cold device-sync caller");
    require(!before, "exec-isolated first-use device sync starts unbound");
    const auto synchronize = [&] { return cudaDeviceSynchronize(); };
    cudaError_t status{};
    if (is_native) {
        const NativeDriverCallScope native;
        status = synchronize();
    } else {
        status = synchronize();
    }
    check(driver.context_get_current(&after), "observe SDK first-use sync context");
    const std::array last{cudaPeekAtLastError(), cudaGetLastError(), cudaPeekAtLastError()};
    auto* router = cooperative_source_router();
    require(router, "first-use device sync has CPU process owner");
    std::shared_ptr<CudaCooperativeFrontend> owner;
    if (after) {
        check(router->existing_frontend(after, owner),
              "inspect first-use sync without actor import");
    }
    require(!owner, "device sync with no accepted logical work creates no private actor");
    {
        const NativeDriverCallScope native;
        check(cudaDeviceReset(), "release first-use SDK context after successful observation");
    }
    std::cout << "normal_runtime_cold_device_sync native=" << is_native << " status=" << status
              << " lazy_binding=" << static_cast<bool>(after) << " actor_import=0 last=" << last[0]
              << ',' << last[1] << ',' << last[2] << " checked_cleanup=1" << std::endl;
    require(
        status == cudaSuccess && after && last == std::array{cudaSuccess, cudaSuccess, cudaSuccess},
        "native and model first-use device sync preserve SDK initialization and TLS");
}

void check_native_device_sync() {
    DriverDispatch driver;
    require(driver.initialize(), "resolve native device synchronization baseline");
    const NativeDriverCallScope native;
    check(cudaSetDevice(0), "bind original device synchronization baseline");
    check(cudaDeviceSynchronize(), "warm original SDK device synchronization");
    CUcontext context{};
    check(driver.context_get_current(&context), "observe original device sync context");
    CUmodule module{};
    CUfunction function{};
    CUstream stream{};
    CUdeviceptr output{}, mapped{};
    std::uint32_t* markers{};
    constexpr unsigned int k_blocks = 105, k_threads = 64;
    constexpr std::size_t k_elements = static_cast<std::size_t>(k_blocks) * k_threads;
    check(cuModuleLoadData(&module, kCooperativePtx), "load original device-sync work");
    check(cuModuleGetFunction(&function, module, "cooperative"), "resolve original sync work");
    check(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), "create original device-sync stream");
    check(cuMemAlloc(&output, k_elements * sizeof(std::uint32_t)), "allocate original sync output");
    check(cuMemsetD32(output, 0, k_elements), "initialize original sync output");
    check(cuMemHostAlloc(reinterpret_cast<void**>(&markers), 2 * sizeof(std::uint32_t),
                         CU_MEMHOSTALLOC_DEVICEMAP),
          "allocate original sync marker");
    check(cuMemHostGetDevicePointer(&mapped, markers, 0), "map original sync marker");
    markers[0] = markers[1] = 0;
    unsigned int work = 50000;
    std::array<void*, 3> parameters{&output, &mapped, &work};
    check(cuLaunchKernel(function, k_blocks, 1, 1, k_threads, 1, 1, 0, stream, parameters.data(),
                         nullptr),
          "launch original nonpreemptive device-sync work");
    const auto synchronized = cudaDeviceSynchronize();
    CUcontext after{};
    check(driver.context_get_current(&after), "observe original sync caller after SDK");
    const std::array last{cudaPeekAtLastError(), cudaGetLastError(), cudaPeekAtLastError()};
    std::array<std::uint32_t, k_elements> values{};
    check(cuMemcpyDtoH(values.data(), output, sizeof(values)), "sample original sync output");
    const auto completed = static_cast<std::size_t>(std::ranges::count(values, 1U));
    check(cuCtxSynchronize(), "physically prove original sync cleanup");
    check(cuModuleUnload(module), "release original sync module");
    check(cuStreamDestroy(stream), "release original sync stream");
    check(cuMemFree(output), "release original sync output");
    check(cuMemFreeHost(markers), "release original sync marker");
    std::cout << "normal_runtime_device_sync native=1 status=" << synchronized
              << " output=" << completed << " expected=" << k_elements
              << " context_preserved=" << (after == context) << " last=" << last[0] << ','
              << last[1] << ',' << last[2] << " checked_cleanup=1" << std::endl;
    require(synchronized == cudaSuccess && completed == k_elements && after == context &&
                last == std::array{cudaSuccess, cudaSuccess, cudaSuccess},
            "original device synchronization proves every output and preserves caller state");
}

void check_source_module(bool probes_device_sync = false) {
    check(cudaSetDevice(0), "bind actual source module context");
    DriverDispatch driver;
    require(driver.initialize(), "resolve source module metadata");
    CUcontext context{};
    check(driver.context_get_current(&context), "observe source module context");
    CUstream stream{};
    CUstream high_stream{};
    CUdeviceptr output{};
    CUdeviceptr device_markers{};
    std::uint32_t* markers{};
    constexpr unsigned int k_blocks = 105;
    constexpr unsigned int k_threads = 64;
    constexpr std::size_t k_elements = static_cast<std::size_t>(k_blocks) * k_threads;
    {
        const NativeDriverCallScope native;
        check(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), "create source module stream");
        check(cuStreamCreateWithPriority(&high_stream, CU_STREAM_NON_BLOCKING, -1),
              "create source module priority stream");
        check(cuMemAlloc(&output, k_elements * sizeof(std::uint32_t)),
              "allocate source module output");
        check(cuMemHostAlloc(reinterpret_cast<void**>(&markers), 2 * sizeof(std::uint32_t),
                             CU_MEMHOSTALLOC_DEVICEMAP),
              "allocate source module progress markers");
        check(cuMemHostGetDevicePointer(&device_markers, markers, 0), "map source module markers");
        check(cuMemsetD32(output, 0, k_elements), "initialize source module output");
    }
    markers[0] = 0;
    markers[1] = 0;
    auto& gate = cooperative_process_call_gate();
    const ProcessCallGate::Call call(gate);
    auto* router = cooperative_source_router();
    require(router, "configured source module owner required");
    std::shared_ptr<CudaCooperativeFrontend> frontend;
    check(router->frontend(context, frontend), "prepare actual source module owner");
    if (probes_device_sync) {
        const NativeDriverCallScope native;
        check(cudaDeviceSynchronize(), "warm actual SDK synchronization before pending work");
    }
    check(router->source_capture_transition(
              gate, context, stream, false, CudaSourceCaptureKind::kBegin,
              [&] { return cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_RELAXED); }),
          "retain source ownership before module load");
    const auto before = router->native_execution_stats();
    CUmodule baseline{};
    CUfunction baseline_function{};
    {
        const NativeDriverCallScope native;
        check(cuModuleLoadData(&baseline, kCooperativePtx), "load original source module baseline");
        check(cuModuleGetFunction(&baseline_function, baseline, "cooperative"),
              "lookup original source function baseline");
    }
    CUmodule module{};
    auto source_image = source_module_image();
    const auto loaded = cuModuleLoadData(&module, source_image.c_str());
    if (loaded != CUDA_SUCCESS) {
        // The isolated pre-fix probe must not destruct an intentionally retained
        // capture owner, or mistake that quarantine for clean resource disposal.
        std::cerr << "public source module load rejected: " << loaded << '\n';
        std::cerr.flush();
        std::_Exit(EXIT_FAILURE);
    }
    source_image.assign(source_image.size(), 'x');
    CUfunction function{};
    check(cuModuleGetFunction(&function, module, "cooperative"),
          "public source lookup retains original function");
    CUfunction high_function{};
    check(cuModuleGetFunction(&high_function, module, "priority_probe"),
          "source lookup retains the original priority function");
    CUdeviceptr high_marker = device_markers + sizeof(std::uint32_t);
    std::array<void*, 1> high_parameters{&high_marker};
    check(
        cuLaunchKernel(high_function, 1, 1, 1, 1, 1, 1, 0, stream, high_parameters.data(), nullptr),
        "capture original source-created priority kernel");
    require(function && gate.is_native_only_on_current_thread() &&
                router->native_execution_stats().grants == before.grants,
            "source module preparation cannot thaw or acquire execution capacity");
    const auto transformed =
        transform_persistent_ptx(kCooperativePtx, PtxExecutionKind::kPreemptive);
    require(transformed && !transformed->kernels.empty(), "source module is qualified PTX");
    CUfunction private_function{};
    {
        const NativeDriverCallScope native;
        require(cuModuleGetFunction(&private_function, module,
                                    transformed->kernels.front().persistent_name.c_str()) ==
                    CUDA_ERROR_NOT_FOUND,
                "original source module contains no framework-private entry");
    }
    CUgraph graph{};
    check(
        router->source_capture_transition(gate, context, stream, false, CudaSourceCaptureKind::kEnd,
                                          [&] { return cuStreamEndCapture(stream, &graph); }),
        "finish original source module capture with actual proof");
    std::size_t nodes{};
    CUgraphExec executable{};
    {
        const NativeDriverCallScope native;
        check(cuGraphGetNodes(graph, nullptr, &nodes), "observe original source graph node count");
        require(nodes == 1, "source module APIs add only the requested original kernel node");
        CUgraphNode node{};
        check(cuGraphGetNodes(graph, &node, &nodes), "observe original source kernel node");
        CUDA_KERNEL_NODE_PARAMS params{};
        check(cuGraphKernelNodeGetParams(node, &params),
              "observe source-created graph function identity");
        require(params.func == high_function && params.gridDimX == 1 && params.blockDimX == 1,
                "source-created graph retains original function and dimensions");
        check(cuGraphInstantiateWithFlags(&executable, graph, 0),
              "instantiate original source-created graph");
    }
    unsigned int original_replays{};
    check(frontend->native(
              stream, false,
              [&] {
                  ++original_replays;
                  return driver.graph_launch(executable, stream);
              },
              true),
          "admit original source-created graph once");
    check(frontend->synchronize_stream(stream, false),
          "prove original source-created replay completion");
    require(original_replays == 1 && markers[1] == 2,
            "original source-created replay physically completes once");
    {
        const NativeDriverCallScope native;
        check(cuGraphExecDestroy(executable), "release source-created graph executable");
        check(cuGraphDestroy(graph), "release source module graph");
        check(cuModuleUnload(baseline),
              "release original source module baseline after capture proof");
    }
    markers[1] = 0;
    const auto acknowledged_stats = [&](std::uint64_t completions) {
        auto stats = frontend->stats();
        const auto acknowledgment_deadline =
            std::chrono::steady_clock::now() + std::chrono::seconds{5};
        while (stats.acknowledged_completions < completions) {
            require(stats.acknowledged_failures == 0 &&
                        std::chrono::steady_clock::now() < acknowledgment_deadline,
                    "physical completion must receive terminal control acknowledgment");
            std::this_thread::sleep_for(std::chrono::milliseconds{1});
            stats = frontend->stats();
        }
        return stats;
    };
    const auto coordinated_before = acknowledged_stats(1);
    require(coordinated_before.granted_executions == 1 &&
                coordinated_before.acknowledged_completions == 1,
            "original replay terminal acknowledgment precedes fresh handoff baseline");
    CUdeviceptr argument_output = output;
    CUdeviceptr argument_marker = device_markers;
    unsigned int work = 50000;
    std::array<void*, 3> parameters{&argument_output, &argument_marker, &work};
    check(cuLaunchKernel(function, k_blocks, 1, 1, k_threads, 1, 1, 0, stream, parameters.data(),
                         nullptr),
          "launch source-created low kernel through public CUDA");
    argument_output = 0;
    argument_marker = 0;
    work = 0;
    parameters.fill(nullptr);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{10};
    while (std::atomic_ref<std::uint32_t>(markers[0]).load(std::memory_order_acquire) == 0) {
        require(std::chrono::steady_clock::now() < deadline,
                "source-created low kernel makes actual progress");
        std::this_thread::sleep_for(std::chrono::milliseconds{1});
    }
    require(cuStreamQuery(stream) == CUDA_ERROR_NOT_READY,
            "source-created low kernel remains pending");
    high_function = nullptr;
    check(cuModuleGetFunction(&high_function, module, "late_priority_probe"),
          "public warm function lookup while source-created low work is active");
    require(cuStreamQuery(stream) == CUDA_ERROR_NOT_READY,
            "warm original function lookup cannot drain accepted low work");
    cudaError_t device_synchronized{cudaErrorUnknown};
    std::size_t device_completed{};
    CUcontext device_sync_context{};
    std::array<cudaError_t, 3> device_sync_last{};
    unsigned int device_originals{};
    bool has_preserved_device_errors{true};
    std::optional<std::jthread> device_caller;
    std::latch device_sdk_returned{1};
    if (probes_device_sync) {
        check(
            frontend->native(
                high_stream, false,
                [&] {
                    check(driver.launch_kernel(high_function, 1, 1, 1, 1, 1, 1, 0, high_stream,
                                               high_parameters.data(), nullptr),
                          "issue original priority kernel during physical low pause");
                    device_caller.emplace([&] {
                        bool has_notified{};
                        try {
                            {
                                const NativeDriverCallScope native;
                                check(cuCtxSetCurrent(context), "bind independent SDK sync caller");
                            }
                            const ProcessCallGate::Call sdk_call(cooperative_process_call_gate());
                            for (const auto expected :
                                 {cudaErrorInvalidValue, cudaErrorApiFailureBase,
                                  cudaErrorLaunchFailure, cudaErrorNotSupported}) {
                                unsigned int error_originals{};
                                const auto failed = intercept_normal_runtime_device_completion([&] {
                                    ++error_originals;
                                    return expected;
                                });
                                has_preserved_device_errors &=
                                    failed == expected && error_originals == 1;
                            }
                            device_synchronized = intercept_normal_runtime_device_completion([&] {
                                                      ++device_originals;
                                                      const NativeDriverCallScope native;
                                                      const auto status = cudaDeviceSynchronize();
                                                      has_notified = true;
                                                      device_sdk_returned.count_down();
                                                      return status;
                                                  }).value_or(cudaErrorUnknown);
                            check(driver.context_get_current(&device_sync_context),
                                  "observe original post-SDK sync context");
                            device_sync_last = {cudaPeekAtLastError(), cudaGetLastError(),
                                                cudaPeekAtLastError()};
                            std::array<std::uint32_t, k_elements> sampled{};
                            const NativeDriverCallScope native;
                            check(cuMemcpyDtoH(sampled.data(), output, sizeof(sampled)),
                                  "sample actually stopped output without queue completion");
                            device_completed =
                                static_cast<std::size_t>(std::ranges::count(sampled, 1U));
                        } catch (...) {
                            device_synchronized = cudaErrorUnknown;
                        }
                        if (!has_notified) {
                            device_sdk_returned.count_down();
                        }
                    });
                    device_sdk_returned.wait();
                    return CUDA_SUCCESS;
                },
                true),
            "admit original priority execution around the device-sync diagnostic");
    } else {
        check(cuLaunchKernel(high_function, 1, 1, 1, 1, 1, 1, 0, high_stream,
                             high_parameters.data(), nullptr),
              "launch warm source-created priority function without cold preparation");
    }
    if (device_caller) {
        device_caller->join();
    }
    check(cuStreamSynchronize(high_stream), "physically finish source-created priority task");
    require(markers[1] == 2, "source-created high task executes once");
    check(cuStreamSynchronize(stream), "resume and physically complete source-created low task");
    std::array<std::uint32_t, k_elements> values{};
    {
        const NativeDriverCallScope native;
        check(cuMemcpyDtoH(values.data(), output, sizeof(values)),
              "read source-created exactly-once output");
    }
    for (const auto value : values) {
        require(value == 1,
                "source-created logical thread executes exactly once across retreat/resume");
    }
    const auto coordinated_after =
        acknowledged_stats(coordinated_before.acknowledged_completions + 2);
    const bool has_exact_handoff =
        coordinated_after.granted_executions - coordinated_before.granted_executions == 3 &&
        coordinated_after.retreat_requests - coordinated_before.retreat_requests == 1 &&
        coordinated_after.acknowledged_suspensions - coordinated_before.acknowledged_suspensions ==
            1 &&
        coordinated_after.logical_resumes - coordinated_before.logical_resumes == 1 &&
        coordinated_after.acknowledged_completions - coordinated_before.acknowledged_completions ==
            2 &&
        coordinated_after.acknowledged_failures == coordinated_before.acknowledged_failures;
    if (!has_exact_handoff) {
        throw std::runtime_error(
            "source module handoff: grants=" +
            std::to_string(coordinated_after.granted_executions -
                           coordinated_before.granted_executions) +
            ", retreats=" +
            std::to_string(coordinated_after.retreat_requests -
                           coordinated_before.retreat_requests) +
            ", suspensions=" +
            std::to_string(coordinated_after.acknowledged_suspensions -
                           coordinated_before.acknowledged_suspensions) +
            ", resumes=" +
            std::to_string(coordinated_after.logical_resumes - coordinated_before.logical_resumes) +
            ", completions=" +
            std::to_string(coordinated_after.acknowledged_completions -
                           coordinated_before.acknowledged_completions));
    }
    check(cuModuleUnload(module), "release public source-loaded module after thaw");
    check(router->close_context(context, [] { return CUDA_SUCCESS; }),
          "close source module private owner");
    frontend.reset();
    {
        const NativeDriverCallScope native;
        check(cuStreamDestroy(stream), "release source module stream");
        check(cuStreamDestroy(high_stream), "release source module priority stream");
        check(cuMemFree(output), "release source module output");
        check(cuMemFreeHost(markers), "release source module progress markers");
    }
    std::cout << "source_module actual_sdk=1 original_entry=1 private_entries=0 native_grants=0 "
                 "original_replays=1 exact_output=6720 grants=3 retreats=1 suspensions=1 resumes=1 "
                 "completions=2\n";
    if (probes_device_sync) {
        std::cout << "normal_runtime_device_sync native=0 status=" << device_synchronized
                  << " output=" << device_completed << " expected=" << k_elements
                  << " sdk_calls=" << device_originals
                  << " context_preserved=" << (device_sync_context == context)
                  << " last=" << device_sync_last[0] << ',' << device_sync_last[1] << ','
                  << device_sync_last[2] << " checked_cleanup=1" << std::endl;
        require(device_synchronized == cudaSuccess && device_completed == k_elements &&
                    device_originals == 1 && has_preserved_device_errors &&
                    device_sync_context == context &&
                    device_sync_last == std::array{cudaSuccess, cudaSuccess, cudaSuccess},
                "device synchronization cannot return success before logical low completion");
    }
}

enum class SourceFaultKind {
    kKernelMissing,
    kMemoryThrow,
    kMemoryContextChange,
    kRuntimeMemoryContextChange
};

void check_source_quarantine(SourceFaultKind kind) {
    check(cudaSetDevice(0), "bind actual quarantine context");
    DriverDispatch driver;
    require(driver.initialize(), "resolve quarantine metadata");
    CUcontext context{};
    check(driver.context_get_current(&context), "observe quarantine context");
    CUstream stream{};
    {
        const NativeDriverCallScope native;
        check(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), "create quarantine stream");
    }
    auto& gate = cooperative_process_call_gate();
    const ProcessCallGate::Call call(gate);
    auto* router = cooperative_source_router();
    require(router, "configured quarantine owner required");
    std::shared_ptr<CudaCooperativeFrontend> frontend;
    check(router->frontend(context, frontend), "prepare real quarantine owner");
    check(router->source_capture_transition(
              gate, context, stream, false, CudaSourceCaptureKind::kBegin,
              [&] {
                  return cuStreamBeginCapture(stream,
                                              kind == SourceFaultKind::kRuntimeMemoryContextChange
                                                  ? CU_STREAM_CAPTURE_MODE_RELAXED
                                                  : CU_STREAM_CAPTURE_MODE_GLOBAL);
              }),
          "retain actual capture before missing kernel callback");
    CudaProcessNativeToken cached;
    check(router->source_native_token(gate, cached), "cache valid source token before uncertainty");
    unsigned int roots{};
    unsigned int originals{};
    ProcessCallGate foreign;
    require(router->source_memory_metadata(foreign,
                                           [&] {
                                               ++originals;
                                               return CUDA_SUCCESS;
                                           }) == CUDA_ERROR_INVALID_VALUE &&
                router->source_memory_metadata(gate, {}) == CUDA_ERROR_INVALID_VALUE &&
                router->source_memory_metadata(
                    gate,
                    [&] {
                        ++originals;
                        return CUDA_SUCCESS;
                    },
                    CudaSourceMemoryOrigin::kRuntime) == CUDA_ERROR_INVALID_VALUE &&
                originals == 0,
            "foreign/empty/unowned Runtime memory invocation rejects before original issue");
    require(router->source_module_metadata(foreign,
                                           [&] {
                                               ++originals;
                                               return CUDA_SUCCESS;
                                           }) == CUDA_ERROR_INVALID_VALUE &&
                router->source_module_metadata(gate, {}) == CUDA_ERROR_INVALID_VALUE &&
                originals == 0,
            "foreign/empty source module permission rejects before native issue");
    if (kind == SourceFaultKind::kKernelMissing) {
        const auto status = invoke_runtime_source_kernel_original(
            {{1, 1, 1}, {1, 1, 1}, 0, false},
            [&] {
                ++roots;
                return cudaSuccess;
            },
            [&] {
                require(
                    router->quarantine_source_call(gate, CUDA_ERROR_UNKNOWN) == CUDA_ERROR_UNKNOWN,
                    "missing callback publishes actual source failure");
            });
        require(status == cudaErrorUnknown, "missing kernel callback reports Runtime uncertainty");
    } else if (kind == SourceFaultKind::kRuntimeMemoryContextChange) {
        std::size_t initial_free{}, remaining{}, total{};
        check(cuMemGetInfo(&initial_free, &total), "read uncertain Runtime allocation baseline");
        void* pointer{};
        // This injection surrounds a real, fully accounted Runtime allocation.
        // The outer borrowed flag states that this callback actually issues it.
        const bool has_native_issue = true;
        const auto status = intercept_cooperative_runtime_memory_boundary(
            [&] {
                ++roots;
                check(cudaMalloc(&pointer, 1024), "issue actual uncertain Runtime allocation");
                const NativeDriverCallScope native;
                check(cuCtxSetCurrent(nullptr), "inject post-accounting Runtime context loss");
                return cudaSuccess;
            },
            has_native_issue);
        require(status == cudaErrorUnknown && pointer,
                "successful SDK allocation reports uncertain source ownership");
        {
            const NativeDriverCallScope native;
            check(cuCtxSetCurrent(context), "restore fixture context without thawing owner");
        }
        check(cuMemGetInfo(&remaining, &total),
              "read retained uncertain Runtime allocation charge");
        require(initial_free - remaining == 1024,
                "post-issue ownership failure cannot cancel successful allocation accounting");
    } else {
        const auto status = router->source_memory_metadata(gate, [&]() -> CUresult {
            ++roots;
            if (kind == SourceFaultKind::kMemoryThrow) {
                throw std::runtime_error("uncertain source memory callback");
            }
            const NativeDriverCallScope native;
            check(cuCtxSetCurrent(nullptr), "inject source memory context change");
            return CUDA_ERROR_INVALID_VALUE;
        });
        require(status == (kind == SourceFaultKind::kMemoryThrow ? CUDA_ERROR_UNKNOWN
                                                                 : CUDA_ERROR_INVALID_VALUE),
                "memory uncertainty quarantines but preserves a known original Driver error");
    }
    require(roots == 1 && gate.is_native_only_on_current_thread() &&
                router->try_finish_source_capture(gate) == CUDA_ERROR_UNKNOWN,
            "unknown source effects cannot release the physical capture owner");
    require(router->quarantine_source_call(gate, CUDA_ERROR_LAUNCH_FAILED) == CUDA_ERROR_UNKNOWN &&
                router->try_finish_source_capture(gate) == CUDA_ERROR_UNKNOWN,
            "repeated quarantine preserves the first source failure");
    require(router->source_memory_metadata(gate,
                                           [&] {
                                               ++originals;
                                               return CUDA_SUCCESS;
                                           }) == CUDA_ERROR_UNKNOWN &&
                originals == 0,
            "failed source cannot invoke another memory original");
    require(router->source_module_metadata(gate,
                                           [&] {
                                               ++originals;
                                               return CUDA_SUCCESS;
                                           }) == CUDA_ERROR_UNKNOWN &&
                originals == 0,
            "failed source cannot invoke a module original");
    require(router->native_session_execution(
                cached, context, stream, false,
                [&] {
                    ++originals;
                    return CUDA_SUCCESS;
                },
                [](const auto& issue) { return issue(); }, &gate) == CUDA_ERROR_UNKNOWN &&
                originals == 0,
            "cached source capability cannot issue after missing dispatch");
    // Intentional quarantine: no cleanup or leak-free recovery is asserted.
    std::_Exit(EXIT_SUCCESS);
}

void inspect_native_source_stream_destroy(bool uses_runtime, bool is_origin,
                                          bool has_joined_member = false,
                                          bool uses_null_target = false) {
    check(cuInit(0), "initialize native source stream lifetime inspection");
    const NativeDriverCallScope native;
    check(cudaSetDevice(0), "bind native stream inspection context");
    CUstream origin{};
    CUstream other{};
    check(cuStreamCreate(&origin, CU_STREAM_NON_BLOCKING), "create original capture origin");
    check(cuStreamCreate(&other, CU_STREAM_NON_BLOCKING), "create original unrelated stream");
    CUevent joined{};
    if (has_joined_member) {
        check(cuEventCreate(&joined, CU_EVENT_DISABLE_TIMING), "create original joining event");
    }
    check(cuStreamBeginCapture(origin, CU_STREAM_CAPTURE_MODE_RELAXED),
          "begin original stream inspection capture");
    if (has_joined_member) {
        check(cuEventRecord(joined, origin), "capture original event joining dependency");
        check(cuStreamWaitEvent(other, joined, 0), "join other stream to original capture");
        CUstreamCaptureStatus status{};
        check(cuStreamIsCapturing(other, &status), "observe actual joined member before destroy");
        std::cout << "native_stream_destroy joined_before=" << status << std::endl;
        require(status == CU_STREAM_CAPTURE_STATUS_ACTIVE,
                "native inspection requires actual active joined member");
    }
    const auto target = uses_null_target ? nullptr : (is_origin ? origin : other);
    std::cout << "native_stream_destroy before_issue runtime=" << uses_runtime
              << " origin=" << is_origin << std::endl;
    if (uses_runtime) {
        const auto result = cudaStreamDestroy(target);
        std::cout << "native_stream_destroy after_issue runtime_status=" << result << std::endl;
    } else {
        const auto result = cuStreamDestroy(target);
        std::cout << "native_stream_destroy after_issue driver_status=" << result << std::endl;
    }
    if (has_joined_member && is_origin) {
        CUstreamCaptureStatus status{};
        const auto queried = cuStreamIsCapturing(other, &status);
        std::cout << "native_stream_destroy joined_query=" << queried << " joined_after=" << status
                  << std::endl;
        const auto waited = cuStreamWaitEvent(other, joined, 0);
        status = static_cast<CUstreamCaptureStatus>(3);
        const auto after_wait = cuStreamIsCapturing(other, &status);
        std::cout << "native_stream_destroy old_record_wait=" << waited
                  << " after_wait_query=" << after_wait << " after_wait_status=" << status
                  << std::endl;
    }
    if (!is_origin) {
        CUgraph graph{};
        const auto ended = cuStreamEndCapture(origin, &graph);
        std::cout << "native_stream_destroy live_origin_end=" << ended
                  << " has_graph=" << static_cast<bool>(graph) << std::endl;
        if (graph) {
            check(cuGraphDestroy(graph), "destroy unrelated stream inspection graph");
        }
    } else {
        // Never query/end a potentially destroyed origin to guess capture NONE.
        CUstream fresh{};
        check(cuStreamCreate(&fresh, CU_STREAM_NON_BLOCKING), "create fresh inspection origin");
        const auto begun = cuStreamBeginCapture(fresh, CU_STREAM_CAPTURE_MODE_RELAXED);
        CUgraph graph{};
        const auto ended = begun == CUDA_SUCCESS ? cuStreamEndCapture(fresh, &graph) : begun;
        std::cout << "native_stream_destroy fresh_begin=" << begun << " fresh_end=" << ended
                  << " has_graph=" << static_cast<bool>(graph) << std::endl;
        if (graph) {
            check(cuGraphDestroy(graph), "destroy fresh stream inspection graph");
        }
    }
    std::cout << "native_stream_destroy context_sync=" << cuCtxSynchronize() << std::endl;
    // Isolated SDK inspection only; do not retry uncertain original destruction.
    std::_Exit(EXIT_SUCCESS);
}

std::array<cudaError_t, 3> check_source_stream_retirement(bool is_native, bool uses_runtime,
                                                          bool is_origin, bool has_joined_member,
                                                          bool has_binding, bool is_caller_unbound,
                                                          bool has_overlapping_capture = false) {
    check(cuInit(0), "initialize actual source stream retirement");
    DriverDispatch driver;
    require(driver.initialize(), "resolve source stream retirement metadata");
    CUcontext context{};
    CUstream origin{}, other{}, overlapping{};
    CUevent event{};
    {
        const NativeDriverCallScope native;
        check(cudaSetDevice(0), "bind stream retirement actual context");
        check(driver.context_get_current(&context), "observe stream retirement actual context");
        check(cuStreamCreate(&origin, CU_STREAM_NON_BLOCKING), "create retirement capture origin");
        if (has_binding || is_origin) {
            check(cuStreamCreate(&other, CU_STREAM_NON_BLOCKING), "create retirement other stream");
        }
        check(cuEventCreate(&event, CU_EVENT_DISABLE_TIMING),
              "create original retirement join event");
        if (has_overlapping_capture) {
            check(cuStreamCreate(&overlapping, CU_STREAM_NON_BLOCKING),
                  "create independent overlapping retirement origin");
        }
    }
    const auto target = is_origin ? origin : other;
    auto& gate = cooperative_process_call_gate();
    const ProcessCallGate::Call call(gate);
    auto* router = is_native ? nullptr : cooperative_source_router();
    std::shared_ptr<CudaCooperativeFrontend> frontend;
    CudaPausedNativeStats before{};
    if (!is_native) {
        require(router, "configured stream retirement source owner");
        check(router->frontend(context, frontend), "prepare stream retirement private owner");
        if (has_binding) {
            check(frontend->synchronize_stream(target, false),
                  "bind and physically drain original target before capture");
        }
        before = router->native_execution_stats();
    }
    const auto begin = [&] { return cuStreamBeginCapture(origin, CU_STREAM_CAPTURE_MODE_RELAXED); };
    if (is_native) {
        const NativeDriverCallScope native;
        check(begin(), "begin original native retirement capture");
    } else {
        check(router->source_capture_transition(gate, context, origin, false,
                                                CudaSourceCaptureKind::kBegin, begin),
              "retain actual source retirement capture");
    }
    const auto construct = [&] {
        if (has_overlapping_capture) {
            const auto begin_overlap = [&] {
                return cuStreamBeginCapture(overlapping, CU_STREAM_CAPTURE_MODE_RELAXED);
            };
            if (is_native) {
                check(begin_overlap(), "begin independent original overlapping capture");
            } else {
                check(
                    router->source_capture_transition(gate, context, overlapping, false,
                                                      CudaSourceCaptureKind::kBegin, begin_overlap),
                    "retain independent source overlapping capture");
            }
        }
        if (!other) {
            check(cuStreamCreate(&other, CU_STREAM_NON_BLOCKING),
                  "create original unbound stream during capture");
        }
        if (has_joined_member) {
            check(cuEventRecord(event, origin), "capture original retirement join dependency");
            check(cuStreamWaitEvent(other, event, 0), "join actual retirement capture member");
        }
    };
    if (is_native) {
        const NativeDriverCallScope native;
        construct();
    } else {
        construct();
        if (uses_runtime) {
            unsigned int originals{};
            require(intercept_cooperative_runtime_stream_destroy(target,
                                                                 [&] {
                                                                     ++originals;
                                                                     return cudaErrorApiFailureBase;
                                                                 }) == cudaErrorApiFailureBase &&
                        originals == 1,
                    "failed stream SDK status outside CUresult remains exact without replay");
        }
    }
    const auto actual_target = is_origin ? origin : other;
    if (is_caller_unbound) {
        const NativeDriverCallScope native;
        check(cuCtxSetCurrent(nullptr), "unbind caller before original Runtime stream destruction");
    }
    std::array<cudaError_t, 3> last{};
    const auto destroy = [&] {
        if (uses_runtime) {
            check(cudaStreamDestroy(actual_target), "destroy original source Runtime stream once");
            cudaError_t rejected{};
            {
                const NativeDriverCallScope native;
                rejected = cudaStreamDestroy(nullptr);
            }
            require(rejected != cudaSuccess && cudaStreamDestroy(nullptr) == rejected,
                    "implicit invalid destruction preserves original Runtime SDK validation");
        } else {
            check(cuStreamDestroy(actual_target), "destroy original source Driver stream once");
            const auto rejected = driver.stream_destroy(nullptr);
            require(rejected != CUDA_SUCCESS && cuStreamDestroy(nullptr) == rejected,
                    "implicit invalid destruction preserves original Driver SDK validation");
        }
        last = {cudaPeekAtLastError(), cudaGetLastError(), cudaPeekAtLastError()};
    };
    if (is_native) {
        const NativeDriverCallScope native;
        destroy();
    } else {
        destroy();
        const auto after = router->native_execution_stats();
        require(after.grants == before.grants && after.retained == before.retained &&
                    gate.is_native_only_on_current_thread(),
                "successful source destruction does not grant, fence, complete or thaw");
        CudaProcessNativeToken token;
        check(router->source_native_token(gate, token), "observe healthy post-destroy session");
    }
    if (is_caller_unbound) {
        const NativeDriverCallScope native;
        check(cuCtxSetCurrent(context), "restore caller only after original SDK destroy returns");
    }
    CUgraph graph{};
    if (is_origin) {
        if (has_joined_member) {
            const auto observe_live = [&] {
                CUstreamCaptureStatus status{};
                check(cuStreamIsCapturing(other, &status), "query only still-live joined member");
                require(status == CU_STREAM_CAPTURE_STATUS_NONE,
                        "SDK origin destruction ends its still-live joined member capture");
                require(cuStreamWaitEvent(other, event, 0) == CUDA_ERROR_INVALID_VALUE,
                        "old captured dependency preserves original SDK rejection after destroy");
            };
            if (is_native) {
                const NativeDriverCallScope native;
                observe_live();
            } else {
                observe_live();
            }
        }
        if (router) {
            require(router->try_finish_source_capture(gate) ==
                        (has_overlapping_capture ? CUDA_ERROR_NOT_READY : CUDA_SUCCESS),
                    "destroyed origin cannot release independent overlapping capture");
        }
    } else if (is_native) {
        const NativeDriverCallScope native;
        check(cuStreamEndCapture(origin, &graph), "end live original stream retirement capture");
    } else {
        check(router->source_capture_transition(gate, context, origin, false,
                                                CudaSourceCaptureKind::kEnd,
                                                [&] { return cuStreamEndCapture(origin, &graph); }),
              "end still-live source origin after member/unrelated destruction");
    }
    CUgraph overlapping_graph{};
    if (has_overlapping_capture) {
        const auto end_overlap = [&] {
            return cuStreamEndCapture(overlapping, &overlapping_graph);
        };
        if (is_native) {
            const NativeDriverCallScope native;
            check(end_overlap(), "end actual independent original overlapping capture");
        } else {
            require(gate.is_native_only_on_current_thread(),
                    "independent overlap retains source pause after target retirement");
            check(router->source_capture_transition(gate, context, overlapping, false,
                                                    CudaSourceCaptureKind::kEnd, end_overlap),
                  "end last actual source overlap with all live origin proof");
        }
    }
    if (router) {
        require(!gate.is_native_only_on_current_thread(),
                "normal routing requires actual surviving-origin/context proof");
        if (has_joined_member) {
            check(cuEventDestroy(event), "retire exact record-only retirement event");
        }
        check(router->close_context(context, [] { return CUDA_SUCCESS; }),
              "close physically proven source stream retirement owner");
        frontend.reset();
    }
    {
        const NativeDriverCallScope native;
        if (graph) {
            std::size_t count{};
            check(cuGraphGetNodes(graph, nullptr, &count),
                  "count original stream retirement graph");
            require(count == 0, "stream destruction adds no private capture graph node");
            check(cuGraphDestroy(graph), "destroy original stream retirement graph");
        }
        if (is_native || !has_joined_member) {
            check(cuEventDestroy(event), "destroy original unregistered retirement event");
        }
        check(cuStreamDestroy(is_origin ? other : origin),
              "destroy only still-live original stream");
        if (overlapping_graph) {
            check(cuGraphDestroy(overlapping_graph),
                  "destroy independent overlapping original graph");
            check(cuStreamDestroy(overlapping), "destroy still-live overlapping original stream");
        }
    }
    std::cout << "source_stream_retirement native=" << is_native << " runtime=" << uses_runtime
              << " origin=" << is_origin << " joined=" << has_joined_member
              << " bound=" << has_binding << " caller_unbound=" << is_caller_unbound
              << " overlapping=" << has_overlapping_capture
              << " construction_grants=0 private_retained=0\n";
    return last;
}

void inspect_source_capture_join(bool is_native, bool uses_runtime_record = false,
                                 bool uses_record_flags = false, bool is_ptds = false,
                                 bool is_caller_unbound = false, bool uses_runtime_wait = false) {
    check(cuInit(0), "initialize original capture-joining wait inspection");
    DriverDispatch driver;
    require(driver.initialize(), "resolve actual capture-joining context");
    CUcontext context{};
    CUstream origin{}, member{};
    CUevent event{};
    CUmodule module{};
    CUfunction function{};
    std::uint32_t* marker{};
    CUdeviceptr mapped{};
    {
        const NativeDriverCallScope native;
        check(cudaSetDevice(0), "bind capture join inspection context");
        check(driver.context_get_current(&context), "observe actual join inspection context");
        check(cuStreamCreate(&origin, CU_STREAM_NON_BLOCKING), "create joining origin");
        check(cuStreamCreate(&member, CU_STREAM_NON_BLOCKING), "create joining destination");
        check(cuEventCreate(&event, CU_EVENT_DISABLE_TIMING), "create original joining event");
        check(cuModuleLoadData(&module, kCooperativePtx), "load original join business module");
        check(cuModuleGetFunction(&function, module, "priority_probe"),
              "resolve original join business function");
        check(cuMemHostAlloc(reinterpret_cast<void**>(&marker), sizeof(std::uint32_t),
                             CU_MEMHOSTALLOC_DEVICEMAP),
              "allocate original join business marker");
        check(cuMemHostGetDevicePointer(&mapped, marker, 0), "map original join business marker");
    }
    *marker = 0;
    auto& gate = cooperative_process_call_gate();
    const ProcessCallGate::Call call(gate);
    auto* router = is_native ? nullptr : cooperative_source_router();
    std::shared_ptr<CudaCooperativeFrontend> frontend;
    CudaPausedNativeStats before{};
    const auto begin = [&] { return cuStreamBeginCapture(origin, CU_STREAM_CAPTURE_MODE_RELAXED); };
    if (is_native) {
        const NativeDriverCallScope native;
        check(begin(), "begin original native join capture");
    } else {
        require(router, "configured capture join source owner");
        check(router->frontend(context, frontend), "prepare capture join private owner");
        before = router->native_execution_stats();
        check(router->source_capture_transition(gate, context, origin, false,
                                                CudaSourceCaptureKind::kBegin, begin),
              "retain capture join source owner");
    }
    const auto record = [&](CUstream stream) {
        if (uses_runtime_record) {
            cudaError_t result{};
            if (uses_record_flags) {
                const auto entry = runtime_symbol<decltype(&cudaEventRecordWithFlags)>(
                    is_ptds ? "cudaEventRecordWithFlags_ptsz" : "cudaEventRecordWithFlags");
                result = entry(event, stream, 0);
            } else {
                const auto entry = runtime_symbol<decltype(&cudaEventRecord)>(
                    is_ptds ? "cudaEventRecord_ptsz" : "cudaEventRecord");
                result = entry(event, stream);
            }
            check(result, "original Runtime captured record result");
            return CUDA_SUCCESS;  // Checked precondition, not SDK error conversion.
        }
        if (uses_record_flags) {
            const auto entry = runtime_symbol<decltype(&cuEventRecordWithFlags)>(
                is_ptds ? "cuEventRecordWithFlags_ptsz" : "cuEventRecordWithFlags");
            return entry(event, stream, 0);
        }
        const auto entry = runtime_symbol<decltype(&cuEventRecord)>(is_ptds ? "cuEventRecord_ptsz"
                                                                            : "cuEventRecord");
        return entry(event, stream);
    };
    const auto wait = [&](CUstream stream) {
        if (uses_runtime_wait) {
            const auto entry = runtime_symbol<decltype(&cudaStreamWaitEvent)>(
                is_ptds ? "cudaStreamWaitEvent_ptsz" : "cudaStreamWaitEvent");
            check(entry(stream, event, 0), "original Runtime capture joining wait result");
            return CUDA_SUCCESS;  // Checked precondition, not SDK error conversion.
        }
        const auto entry = runtime_symbol<decltype(&cuStreamWaitEvent)>(
            is_ptds ? "cuStreamWaitEvent_ptsz" : "cuStreamWaitEvent");
        return entry(stream, event, 0);
    };
    const auto kernel = [&](CUstream stream) {
        std::array<void*, 1> arguments{&mapped};
        return cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, stream, arguments.data(), nullptr);
    };
    const auto construct = [&] {
        check(kernel(origin), "capture original origin kernel");
        if (is_caller_unbound) {
            const NativeDriverCallScope native;
            check(cuCtxSetCurrent(nullptr), "unbind caller before original Runtime record");
        }
        if (!is_native && uses_runtime_record) {
            unsigned int originals{};
            require(intercept_cooperative_runtime_event_record(event, origin, is_ptds, 0,
                                                               [&] {
                                                                   ++originals;
                                                                   return cudaErrorApiFailureBase;
                                                               }) == cudaErrorApiFailureBase &&
                        originals == 1,
                    "failed SDK record preserves non-CUresult Runtime error without replay");
        }
        check(record(origin), "capture original joining event");
        check(wait(member), "join actual original destination");
        check(kernel(member), "capture original member kernel");
        check(record(member), "capture original rejoining event");
        check(wait(origin), "rejoin original capture origin");
        if (uses_runtime_record) {
            const auto entry = runtime_symbol<decltype(&cudaEventRecordWithFlags)>(
                is_ptds ? "cudaEventRecordWithFlags_ptsz" : "cudaEventRecordWithFlags");
            check(entry(event, origin, cudaEventRecordExternal),
                  "capture original Runtime external record without joining provenance");
        } else {
            const auto entry = runtime_symbol<decltype(&cuEventRecordWithFlags)>(
                is_ptds ? "cuEventRecordWithFlags_ptsz" : "cuEventRecordWithFlags");
            check(entry(event, origin, CU_EVENT_RECORD_EXTERNAL),
                  "capture original Driver external record without joining provenance");
        }
        if (uses_runtime_wait) {
            const auto entry = runtime_symbol<decltype(&cudaStreamWaitEvent)>(
                is_ptds ? "cudaStreamWaitEvent_ptsz" : "cudaStreamWaitEvent");
            check(entry(origin, event, cudaEventWaitExternal),
                  "capture original Runtime external wait node");
        } else {
            const auto entry = runtime_symbol<decltype(&cuStreamWaitEvent)>(
                is_ptds ? "cuStreamWaitEvent_ptsz" : "cuStreamWaitEvent");
            check(entry(origin, event, CU_EVENT_WAIT_EXTERNAL),
                  "capture original Driver external wait node");
        }
    };
    if (is_native) {
        const NativeDriverCallScope native;
        construct();
    } else {
        construct();
    }
    CUstreamCaptureStatus status{};
    {
        const NativeDriverCallScope native;
        check(cuStreamIsCapturing(member, &status), "observe live joined capture member");
    }
    const auto after = router ? router->native_execution_stats() : CudaPausedNativeStats{};
    std::cout << "source_capture_join native=" << is_native
              << " runtime_record=" << uses_runtime_record << " runtime_wait=" << uses_runtime_wait
              << " flags=" << uses_record_flags << " ptds=" << is_ptds
              << " caller_unbound=" << is_caller_unbound << " member_status=" << status
              << " construction_grants=" << after.grants - before.grants
              << " construction_retained=" << after.retained - before.retained << std::endl;
    if (status != CU_STREAM_CAPTURE_STATUS_ACTIVE || after.grants != before.grants ||
        after.retained != before.retained || *marker != 0) {
        std::cerr << "capture joining changed SDK construction or admitted private execution\n";
        std::_Exit(EXIT_FAILURE);
    }
    CUgraph graph{};
    if (is_native) {
        const NativeDriverCallScope native;
        check(cuStreamEndCapture(origin, &graph), "end original native joined capture");
    } else {
        check(router->source_capture_transition(gate, context, origin, false,
                                                CudaSourceCaptureKind::kEnd,
                                                [&] { return cuStreamEndCapture(origin, &graph); }),
              "end actual source joined capture with all participants NONE");
        require(!gate.is_native_only_on_current_thread(),
                "successful joined capture end requires actual source proof");
        CudaEventProjection registered;
        check(router->event_projection(event, registered), "inspect exact record-only owner");
        require(!registered.allows_logical_binding(),
                "unobserved creation flags never authorize logical import");
    }
    CUgraphExec executable{};
    {
        const NativeDriverCallScope native;
        std::size_t count{};
        check(cuGraphGetNodes(graph, nullptr, &count), "count unchanged joined original graph");
        require(count == 4, "only two original kernels and requested external record/wait nodes");
        std::array<CUgraphNode, 4> nodes{};
        check(cuGraphGetNodes(graph, nodes.data(), &count), "inspect joined original nodes");
        unsigned int kernel_count{}, record_count{}, wait_count{};
        for (const auto node : nodes) {
            CUgraphNodeType type{};
            check(cuGraphNodeGetType(node, &type), "inspect actual joined graph node type");
            if (type == CU_GRAPH_NODE_TYPE_KERNEL) {
                CUDA_KERNEL_NODE_PARAMS parameters{};
                check(cuGraphKernelNodeGetParams(node, &parameters), "inspect actual join kernel");
                require(parameters.func == function && parameters.gridDimX == 1 &&
                            parameters.blockDimX == 1,
                        "joined graph preserves original function and dimensions");
                ++kernel_count;
            } else if (type == CU_GRAPH_NODE_TYPE_EVENT_RECORD) {
                CUevent recorded{};
                check(cuGraphEventRecordNodeGetEvent(node, &recorded),
                      "inspect original joined graph external record");
                require(recorded == event, "external record preserves original event");
                ++record_count;
            } else if (type == CU_GRAPH_NODE_TYPE_WAIT_EVENT) {
                CUevent waited{};
                check(cuGraphEventWaitNodeGetEvent(node, &waited),
                      "inspect original joined graph external wait");
                require(waited == event, "external wait preserves original event");
                ++wait_count;
            }
        }
        require(kernel_count == 2 && record_count == 1 && wait_count == 1,
                "capture adds no private nodes and changes no external event flags");
        check(cuGraphInstantiateWithFlags(&executable, graph, 0),
              "instantiate original join graph");
    }
    if (is_native) {
        const NativeDriverCallScope native;
        check(cuGraphLaunch(executable, origin), "replay original native joined graph");
        check(cuCtxSynchronize(), "prove actual native joined replay completion");
    } else {
        check(frontend->native(
                  origin, false,
                  [&] {
                      const NativeDriverCallScope native;
                      return cuGraphLaunch(executable, origin);
                  },
                  true),
              "admit actual original joined graph replay with physical proof");
        check(frontend->synchronize_stream(origin, false),
              "prove actual joined replay private trailing fence");
    }
    require(*marker == 2, "exact two original joined kernels execute only on replay");
    cudaError_t completed_metadata_wait{cudaSuccess};
    if (!is_native) {
        check(router->native_boundary([] { return CUDA_SUCCESS; },
                                      [&] { return driver.context_synchronize(); }),
              "prove completed record-only epoch without importing logical flags");
        CudaEventProjection completed;
        check(router->event_projection(event, completed), "observe proven record-only epoch");
        require(completed.has_native_completion() && !completed.allows_logical_binding(),
                "physical completion never grants unknown creation flags logical permission");
        const auto entry = runtime_symbol<decltype(&cudaStreamWaitEvent)>(
            is_ptds ? "cudaStreamWaitEvent_ptsz" : "cudaStreamWaitEvent");
        completed_metadata_wait = entry(origin, event, 0);
    }
    if (is_native) {
        const NativeDriverCallScope native;
        check(cuEventDestroy(event), "destroy native original join event");
    } else {
        if (uses_runtime_record) {
            check(cudaEventDestroy(event), "cached normal Runtime retires record-only owner");
        } else {
            check(cuEventDestroy(event), "normal Driver retires record-only owner");
        }
        CudaEventProjection retired;
        require(router->event_projection(event, retired) == CUDA_ERROR_INVALID_HANDLE,
                "normal SDK destruction leaves no stale record-only registration");
        CUevent fresh{};
        check(cuEventCreate(&fresh, CU_EVENT_DISABLE_TIMING),
              "create fresh ordinary logical event");
        CudaEventProjection rebound;
        check(router->event_projection(fresh, rebound), "observe fresh actual creation owner");
        require(rebound.allows_logical_binding(),
                "fresh actual non-IPC creation retains ordinary logical binding contract");
        check(cuEventDestroy(fresh), "destroy fresh ordinary join event");
        check(router->close_context(context, [] { return CUDA_SUCCESS; }),
              "close proven joined capture private owner");
        frontend.reset();
    }
    {
        const NativeDriverCallScope native;
        check(cuGraphExecDestroy(executable), "destroy original joined executable");
        check(cuGraphDestroy(graph), "destroy original joined graph");
        check(cuStreamDestroy(origin), "destroy original joined origin stream");
        check(cuStreamDestroy(member), "destroy original joined member stream");
        check(cuModuleUnload(module), "unload original join business module");
        check(cuMemFreeHost(marker), "release original join business marker");
    }
    std::cout << "completed_record_only_wait native=" << is_native
              << " sdk_status=" << completed_metadata_wait << " checked_cleanup=1" << std::endl;
    require(completed_metadata_wait == cudaSuccess,
            "completed metadata-only event waits stay native rather than importing logical flags");
}

void check_runtime_event_quarantine(bool is_creation, bool loses_context, bool is_record = false) {
    check(cuInit(0), "initialize Runtime event quarantine dispatch");
    DriverDispatch driver;
    require(driver.initialize(), "resolve Runtime event quarantine context");
    CUcontext context{};
    CUstream stream{};
    cudaEvent_t event{};
    {
        const NativeDriverCallScope native;
        check(cudaSetDevice(0), "bind Runtime event quarantine context");
        check(driver.context_get_current(&context), "observe Runtime quarantine context");
        check(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), "create Runtime quarantine stream");
    }
    if (!is_creation) {
        check(cuEventCreate(&event, CU_EVENT_DISABLE_TIMING), "create bound quarantine event");
    }
    auto& gate = cooperative_process_call_gate();
    const ProcessCallGate::Call call(gate);
    auto* router = cooperative_source_router();
    require(router, "configured Runtime event quarantine owner");
    std::shared_ptr<CudaCooperativeFrontend> frontend;
    check(router->frontend(context, frontend), "prepare Runtime quarantine private owner");
    check(router->source_capture_transition(
              gate, context, stream, false, CudaSourceCaptureKind::kBegin,
              [&] { return cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_RELAXED); }),
          "retain actual Runtime quarantine capture");
    CudaProcessNativeToken cached;
    check(router->source_native_token(gate, cached), "cache Runtime event source capability");
    unsigned int originals{};
    const auto original = [&] {
        ++originals;
        {
            const NativeDriverCallScope native;
            if (is_creation) {
                check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming),
                      "issue actual original Runtime creation exactly once");
            } else if (is_record) {
                check(cudaEventRecord(event, stream),
                      "issue actual original Runtime captured record exactly once");
            } else {
                check(cudaEventDestroy(event), "issue actual original Runtime destroy once");
            }
            if (loses_context) {
                check(cuCtxSetCurrent(nullptr), "inject lost post-SDK actual context proof");
            }
        }
        if (!loses_context) {
            if (is_creation) {
                check(router->register_native_event(cached, context, event),
                      "inject competing CPU registration after successful original creation");
            } else {
                router->forget_event(event);  // Inject a lost exact retirement owner.
            }
        }
        return cudaSuccess;
    };
    const auto invoke = [&] {
        return is_record
                   ? intercept_cooperative_runtime_event_record(event, stream, false, 0, original)
               : is_creation ? intercept_cooperative_runtime_event_create(
                                   &event, cudaEventDisableTiming, original)
                             : intercept_cooperative_runtime_event_destroy(event, original);
    };
    require(
        invoke() == cudaErrorUnknown && originals == 1 && gate.is_native_only_on_current_thread(),
        "post-SDK failure retains source pause without successful public return");
    const auto failed = router->try_finish_source_capture(gate);
    require(
        failed == (is_creation || loses_context ? CUDA_ERROR_UNKNOWN : CUDA_ERROR_INVALID_HANDLE) &&
            invoke() == cudaErrorUnknown && originals == 1,
        "failed Runtime event owner is sticky and cannot repeat SDK issue");
    require(router->native_session_execution(
                cached, context, stream, false,
                [&] {
                    ++originals;
                    return CUDA_SUCCESS;
                },
                [](const auto& issue) { return issue(); }, &gate) == failed &&
                originals == 1,
            "cached native capability cannot execute after Runtime ownership failure");
    std::cout << "runtime_event_quarantine creation=" << is_creation << " record=" << is_record
              << " context_lost=" << loses_context << " sdk_calls=" << originals
              << " owner_status=" << failed << std::endl;
    // Intentional physical quarantine in an exec-isolated process, not cleanup.
    std::_Exit(EXIT_SUCCESS);
}

void check_source_stream_quarantine(bool loses_context) {
    check(cuInit(0), "initialize actual stream retirement fault dispatch");
    DriverDispatch driver;
    require(driver.initialize(), "resolve stream retirement fault metadata");
    CUcontext context{};
    CUstream captured{}, target{};
    {
        const NativeDriverCallScope native;
        check(cudaSetDevice(0), "bind actual stream retirement fault context");
        check(driver.context_get_current(&context), "observe actual stream fault context");
        check(cuStreamCreate(&captured, CU_STREAM_NON_BLOCKING), "create stream fault origin");
        check(cuStreamCreate(&target, CU_STREAM_NON_BLOCKING), "create stream fault target");
    }
    auto& gate = cooperative_process_call_gate();
    const ProcessCallGate::Call call(gate);
    auto* router = cooperative_source_router();
    require(router, "configured source stream fault owner");
    std::shared_ptr<CudaCooperativeFrontend> frontend;
    check(router->frontend(context, frontend), "prepare actual stream fault private owner");
    check(frontend->synchronize_stream(target, false), "bind and drain stream fault target");
    check(router->source_capture_transition(
              gate, context, captured, false, CudaSourceCaptureKind::kBegin,
              [&] { return cuStreamBeginCapture(captured, CU_STREAM_CAPTURE_MODE_RELAXED); }),
          "retain actual stream fault capture owner");
    CudaProcessNativeToken cached;
    check(router->source_native_token(gate, cached), "cache actual stream fault capability");
    unsigned int originals{}, retirements{};
    const auto original = [&] {
        ++originals;
        const NativeDriverCallScope native;
        if (loses_context) {
            check(cudaStreamDestroy(target), "issue successful original Runtime destruction once");
            check(cuCtxSetCurrent(nullptr), "lose actual post-SDK stream destruction context");
        } else {
            check(cuStreamDestroy(target), "issue successful original Driver destruction once");
        }
        return CUDA_SUCCESS;
    };
    const auto retire = [&](CUcontext actual) {
        ++retirements;
        require(actual == context, "CPU retirement receives saved actual stream context");
        return CUDA_ERROR_OUT_OF_MEMORY;
    };
    unsigned int unissued{};
    require(router->source_stream_destroy(
                gate, target,
                [&] {
                    ++unissued;
                    return CUDA_SUCCESS;
                },
                retire, CudaSourceMemoryOrigin::kRuntime) == CUDA_ERROR_INVALID_VALUE &&
                unissued == 0 && retirements == 0,
            "missing borrowed Runtime effect proof rejects before original SDK issue");
    const auto invoke = [&] {
        if (loses_context) {
            return intercept_cooperative_runtime_stream_destroy(target, [&] {
                       check(original(), "successful checked Runtime stream destruction");
                       return cudaSuccess;
                   }) == cudaErrorUnknown;
        }
        return router->source_stream_destroy(gate, target, original, retire) ==
               CUDA_ERROR_OUT_OF_MEMORY;
    };
    require(invoke() && originals == 1 && gate.is_native_only_on_current_thread(),
            "post-SDK stream failure retains source pause without retry or successful return");
    const auto failed = router->try_finish_source_capture(gate);
    require(failed == (loses_context ? CUDA_ERROR_UNKNOWN : CUDA_ERROR_OUT_OF_MEMORY) && invoke() &&
                originals == 1 && retirements == (loses_context ? 0U : 1U),
            "stream ownership failure is sticky and cannot repeat original or retirement");
    require(router->native_session_execution(
                cached, context, captured, false,
                [&] {
                    ++originals;
                    return CUDA_SUCCESS;
                },
                [](const auto& issue) { return issue(); }, &gate) == failed &&
                originals == 1,
            "cached stream capability cannot issue after post-SDK ownership failure");
    std::cout << "source_stream_quarantine context_lost=" << loses_context
              << " sdk_calls=" << originals << " retirements=" << retirements
              << " owner_status=" << failed << std::endl;
    // Exec-isolated intentional physical quarantine, not cleanup or recovery.
    std::_Exit(EXIT_SUCCESS);
}

void run_source_quarantine_child(const char* flag) {
    const auto child = ::fork();
    require(child >= 0, "fork source kernel fault child");
    if (child == 0) {
        ::execl("/proc/self/exe", "glimmer_cuda_runtime_capture_owner_gpu_test", flag,
                static_cast<char*>(nullptr));
        std::_Exit(EXIT_FAILURE);
    }
    int status{};
    pid_t waited{};
    do {
        waited = ::waitpid(child, &status, 0);
    } while (waited == -1 && errno == EINTR);
    require(waited == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
            "actual source uncertainty retains ownership in isolated child");
}

std::array<cudaError_t, 4> check_normal_runtime_handle_retirement(bool is_stream,
                                                                  bool is_native = false,
                                                                  bool has_binding = true,
                                                                  bool is_caller_unbound = false) {
    const ProcessCallGate::Call call(cooperative_process_call_gate());
    check(cuInit(0), "initialize normal Runtime lifetime regression");
    DriverDispatch driver;
    require(driver.initialize(), "resolve normal Runtime lifetime original metadata");
    CUcontext context{};
    {
        const NativeDriverCallScope native;
        check(cudaSetDevice(0), "bind actual normal Runtime lifetime context");
        check(driver.context_get_current(&context), "observe actual normal lifetime context");
    }
    auto* router = cooperative_source_router();
    require(router, "normal lifetime preview owner must exist");
    std::shared_ptr<CudaCooperativeFrontend> frontend;
    if (!is_native) {
        check(router->frontend(context, frontend), "observe normal lifetime actor");
    }
    const auto unbind = [&] {
        if (is_caller_unbound) {
            const NativeDriverCallScope native;
            check(cuCtxSetCurrent(nullptr), "unbind normal original Runtime caller");
        }
    };
    const auto restore = [&] {
        const NativeDriverCallScope native;
        check(cuCtxSetCurrent(context), "restore only after original Runtime destroy returns");
    };
    std::array<cudaError_t, 4> last{};
    const auto rejected_null = [&] {
        const auto status = is_stream ? cudaStreamDestroy(nullptr) : cudaEventDestroy(nullptr);
        require(status != cudaSuccess, "original implicit lifetime validation must reject");
        last = {status, cudaPeekAtLastError(), cudaGetLastError(), cudaPeekAtLastError()};
    };
    if (is_stream) {
        CUstream stream{};
        {
            const NativeDriverCallScope native;
            check(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), "create original normal stream");
        }
        if (frontend && has_binding) {
            const NativeDriverCallScope native;
            check(frontend->synchronize_stream(stream, false), "bind and drain logical stream");
        }
        const auto inspect = [&](CUresult expected) {
            const NativeDriverCallScope native;
            CudaNativeBoundaryToken boundary;
            CUresult retained{};
            check(router->with_context(context,
                                       [&] {
                                           check(frontend->pause(boundary),
                                                 "pause exact normal actor CPU inspection");
                                           retained = frontend->retire_paused_stream(boundary,
                                                                                     stream, false);
                                           return frontend->finish_pause(boundary, [&] {
                                               return driver.context_synchronize();
                                           });
                                       }),
                  "finish normal actor inspection with actual context proof");
            require(retained == expected,
                    "normal Runtime stream destruction must retain/retire exact actor identity");
        };
        unbind();
        if (!is_native) {
            bool effect{};
            const auto rejected = router->native_stream_destroy(
                stream, [] { return CUDA_ERROR_INVALID_VALUE; },
                [](CUcontext) { return CUDA_SUCCESS; },
                [&] { return driver.context_synchronize(); }, &effect);
            check(rejected == CUDA_ERROR_INVALID_VALUE ? CUDA_SUCCESS : rejected,
                  "normal boundary must preserve separately signaled SDK failure");
            inspect(has_binding ? CUDA_SUCCESS : CUDA_ERROR_INVALID_HANDLE);
            unsigned int originals{};
            require(intercept_cooperative_runtime_stream_destroy(stream,
                                                                 [&] {
                                                                     ++originals;
                                                                     return cudaErrorApiFailureBase;
                                                                 }) == cudaErrorApiFailureBase &&
                        originals == 1,
                    "failed normal stream original preserves non-CUresult status exactly once");
            inspect(has_binding ? CUDA_SUCCESS : CUDA_ERROR_INVALID_HANDLE);
        }
        if (is_native) {
            const NativeDriverCallScope native;
            check(cudaStreamDestroy(stream), "destroy original normal Runtime stream once");
        } else {
            check(cudaStreamDestroy(stream), "cached Runtime destroys actual logical stream once");
        }
        if (is_native) {
            const NativeDriverCallScope native;
            rejected_null();
        } else {
            rejected_null();
        }
        if (!is_native) {
            inspect(CUDA_ERROR_INVALID_HANDLE);
        }
        restore();
    } else {
        CUevent event{};
        if (is_native || !has_binding) {
            const NativeDriverCallScope native;
            check(cuEventCreate(&event, CU_EVENT_DISABLE_TIMING), "create original normal event");
        } else {
            check(cuEventCreate(&event, CU_EVENT_DISABLE_TIMING), "create public logical event");
            check(cuEventRecord(event, nullptr), "accept logical record before Runtime destroy");
        }
        CudaEventProjection before;
        if (!is_native && has_binding) {
            check(router->event_projection(event, before), "snapshot public logical event owner");
        }
        unbind();
        if (!is_native) {
            unsigned int originals{};
            require(intercept_normal_runtime_event_destroy(event,
                                                           [&] {
                                                               ++originals;
                                                               return cudaErrorApiFailureBase;
                                                           }) == cudaErrorApiFailureBase &&
                        originals == 1,
                    "failed normal event original preserves non-CUresult status exactly once");
            if (has_binding) {
                CudaEventProjection retained;
                check(router->refresh_event_projection(before, retained),
                      "failed Runtime destruction keeps exact original registration");
            }
        }
        if (is_native) {
            const NativeDriverCallScope native;
            check(cudaEventDestroy(event), "destroy original normal Runtime event once");
        } else {
            check(cudaEventDestroy(event), "cached Runtime destroys actual logical event once");
        }
        if (is_native) {
            const NativeDriverCallScope native;
            rejected_null();
        } else {
            rejected_null();
        }
        CudaEventProjection after;
        if (!is_native && has_binding) {
            require(router->refresh_event_projection(before, after) == CUDA_ERROR_INVALID_HANDLE,
                    "successful normal Runtime event destroy must retire exact CPU registration");
        }
        restore();
    }
    // Keep this same actor across the next real handle-creation/reuse cycle.
    frontend.reset();
    std::cout << "normal_runtime_handle_retirement stream=" << is_stream << " native=" << is_native
              << " bound=" << has_binding << " caller_unbound=" << is_caller_unbound
              << " exact_cpu_retirement=1" << std::endl;
    return last;
}

void check_normal_runtime_retirement_quarantine(bool is_stream) {
    const ProcessCallGate::Call call(cooperative_process_call_gate());
    DriverDispatch driver;
    require(driver.initialize(), "resolve normal lifetime fault original dispatch");
    CUcontext context{};
    {
        const NativeDriverCallScope native;
        check(cudaSetDevice(0), "bind normal lifetime fault actual context");
        check(driver.context_get_current(&context), "observe normal lifetime fault context");
    }
    auto* router = cooperative_source_router();
    require(router, "normal lifetime fault owner");
    std::shared_ptr<CudaCooperativeFrontend> frontend;
    check(router->frontend(context, frontend), "prepare normal fault actor");
    CUevent event{};
    CUstream stream{};
    if (is_stream) {
        check(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), "create exact normal fault stream");
        check(frontend->synchronize_stream(stream, false), "bind exact normal fault stream");
    } else {
        check(cuEventCreate(&event, CU_EVENT_DISABLE_TIMING), "create exact normal fault event");
    }
    unsigned int originals{}, detachments{};
    bool has_native_effect{};
    const auto original = [&] {
        ++originals;
        const NativeDriverCallScope native;
        const auto sdk = is_stream ? cudaStreamDestroy(stream) : cudaEventDestroy(event);
        require(sdk == cudaSuccess, "normal quarantine follows actual SDK success");
        has_native_effect = true;
        if (!is_stream) {
            router->forget_event(event);  // Inject lost post-SDK CPU ownership, not native cleanup.
        }
        return CUDA_SUCCESS;
    };
    const auto complete = [&] { return driver.context_synchronize(); };
    const auto detach = [&](CUcontext actual) {
        require(actual == context, "normal release retirement retains pre-destroy context");
        ++detachments;
        return CUDA_ERROR_OUT_OF_MEMORY;
    };
    unsigned int forbidden{};
    const auto excluded = [&] {
        ++forbidden;
        return CUDA_SUCCESS;
    };
    require((is_stream ? router->native_stream_destroy(stream, excluded, detach, complete, nullptr)
                       : router->native_event_destroy(event, excluded, complete, nullptr)) ==
                    CUDA_ERROR_INVALID_VALUE &&
                forbidden == 0,
            "normal destruction needs borrowed effect before original issue");
    const auto failed =
        is_stream
            ? router->native_stream_destroy(stream, original, detach, complete, &has_native_effect)
            : router->native_event_destroy(event, original, complete, &has_native_effect);
    require(failed == (is_stream ? CUDA_ERROR_OUT_OF_MEMORY : CUDA_ERROR_INVALID_HANDLE) &&
                originals == 1 && detachments == (is_stream ? 1U : 0U),
            "normal post-effect CPU failure quarantines without another SDK call");
    require(router->frontend(context, frontend) == CUDA_ERROR_UNKNOWN,
            "failed normal actor cannot resume or reinitialize");
    has_native_effect = false;
    const auto repeated =
        is_stream
            ? router->native_stream_destroy(stream, original, detach, complete, &has_native_effect)
            : router->native_event_destroy(event, original, complete, &has_native_effect);
    require(repeated == CUDA_ERROR_UNKNOWN && originals == 1,
            "normal failed owner forbids destruction retry and cached issue");
    std::cout << "normal_runtime_quarantine stream=" << is_stream << " sdk_calls=" << originals
              << " cpu_status=" << failed << std::endl;
    // Intentional exec-isolated ownership retention, not recovery or clean teardown.
    std::_Exit(EXIT_SUCCESS);
}

void check_normal_runtime_record_quarantine(bool should_throw = false, bool is_wait = false,
                                            bool should_fail_allocation = false) {
    const ProcessCallGate::Call call(cooperative_process_call_gate());
    DriverDispatch driver;
    require(driver.initialize(), "resolve actual record fault dispatch");
    CUcontext context{};
    {
        const NativeDriverCallScope native;
        check(cudaSetDevice(0), "bind original record fault context");
        check(driver.context_get_current(&context), "observe original record fault context");
    }
    auto* router = cooperative_source_router();
    require(router, "actual record fault owner");
    std::shared_ptr<CudaCooperativeFrontend> frontend;
    check(router->frontend(context, frontend), "prepare actual record fault actor");
    CUevent event{};
    check(cuEventCreate(&event, CU_EVENT_DISABLE_TIMING), "create actual record fault event");
    unsigned int originals{};
    const auto invoke = [&](const std::function<cudaError_t()>& original) {
        return is_wait ? intercept_normal_runtime_event_wait(nullptr, event, false, 0, original)
                       : intercept_normal_runtime_event_record(event, nullptr, false, original);
    };
    const auto failed = invoke([&] {
        ++originals;
        const NativeDriverCallScope native;
        const auto sdk =
            is_wait ? cudaStreamWaitEvent(nullptr, event, 0) : cudaEventRecord(event, nullptr);
        require(sdk == cudaSuccess, "quarantine follows one real SDK record effect");
        if (should_fail_allocation) {
            should_reject_metadata_allocation = true;
        } else {
            router->forget_event(event);  // Lost CPU ownership after the actual SDK effect.
        }
        if (should_throw) {
            throw std::runtime_error("injected uncertainty after original SDK record");
        }
        return sdk;
    });
    should_reject_metadata_allocation = false;
    require(failed == cudaErrorUnknown && originals == 1 &&
                router->normal_owner_status() == CUDA_ERROR_UNKNOWN,
            "post-effect record failure retains failed ownership");
    require(router->frontend(context, frontend) == CUDA_ERROR_UNKNOWN,
            "failed record actor cannot reinitialize");
    if (should_fail_allocation) {
        router->forget_event(event);  // Also forbid retry through the unknown-handle fallback.
    }
    const auto repeated = invoke([&] {
        ++originals;
        return cudaSuccess;
    });
    require(repeated == cudaErrorUnknown && originals == 1,
            "failed record owner forbids unknown-handle SDK retry");
    std::cout << "normal_runtime_record_quarantine sdk_calls=" << originals
              << " thrown=" << should_throw << " wait=" << is_wait
              << " allocation=" << should_fail_allocation << " retained_owner=1" << std::endl;
    // Exec-isolated uncertain ownership, not a claim of successful cleanup.
    std::_Exit(EXIT_SUCCESS);
}

void check_normal_runtime_creation_quarantine(bool should_throw, bool has_duplicate) {
    const ProcessCallGate::Call call(cooperative_process_call_gate());
    DriverDispatch driver;
    require(driver.initialize(), "resolve actual creation fault dispatch");
    CUcontext context{};
    {
        const NativeDriverCallScope native;
        check(cudaSetDevice(0), "bind original creation fault context");
        check(driver.context_get_current(&context), "observe original creation fault context");
    }
    auto* router = cooperative_source_router();
    require(router, "actual creation fault owner");
    std::shared_ptr<CudaCooperativeFrontend> frontend;
    check(router->frontend(context, frontend), "prepare creation fault owner");
    cudaEvent_t event{};
    unsigned int originals{};
    const auto failed = intercept_normal_runtime_event_create(&event, cudaEventDefault, [&] {
        ++originals;
        const NativeDriverCallScope native;
        const auto sdk = cudaEventCreate(&event);
        require(sdk == cudaSuccess, "creation quarantine follows one actual SDK effect");
        if (should_throw) {
            throw std::runtime_error("injected uncertainty after actual SDK creation");
        }
        if (has_duplicate) {
            check(router->register_event(context, event), "inject duplicate CPU event owner");
        } else {
            check(cuCtxSetCurrent(nullptr), "inject missing post-SDK creation context");
        }
        return sdk;
    });
    require(failed == cudaErrorUnknown && originals == 1 &&
                router->normal_owner_status() == CUDA_ERROR_UNKNOWN,
            "unproven post-SDK creation retains unhealthy ownership");
    const auto repeated = intercept_normal_runtime_event_create(&event, cudaEventDefault, [&] {
        ++originals;
        return cudaSuccess;
    });
    require(repeated == cudaErrorUnknown && originals == 1,
            "failed creation registry forbids unknown-handle SDK retry");
    std::cout << "normal_runtime_create_quarantine sdk_calls=" << originals
              << " thrown=" << should_throw << " duplicate=" << has_duplicate << " retained_owner=1"
              << std::endl;
    // Intentional exec-isolated uncertain ownership, not successful cleanup.
    std::_Exit(EXIT_SUCCESS);
}

void check_normal_runtime_cold_creation(bool is_native) {
    const ProcessCallGate::Call call(cooperative_process_call_gate());
    DriverDispatch driver;
    require(driver.initialize(), "resolve cold actual creation dispatch");
    CUcontext before{}, after{};
    {
        const NativeDriverCallScope native;
        check(driver.init(0), "initialize Driver without warming Runtime creation");
        check(driver.context_get_current(&before), "observe cold SDK caller");
    }
    require(!before, "isolated cold creation starts without a bound context");
    cudaEvent_t event{};
    std::array<cudaError_t, 3> last{};
    const auto create = [&] {
        check(cudaEventCreate(&event), "perform first actual cold SDK event creation");
        check(driver.context_get_current(&after), "observe actual cold SDK context effect");
        last = {cudaPeekAtLastError(), cudaGetLastError(), cudaPeekAtLastError()};
    };
    if (is_native) {
        const NativeDriverCallScope native;
        create();
    } else {
        create();
    }
    require(event && after && last == std::array<cudaError_t, 3>{},
            "native and model cold creation preserve successful lazy binding and SDK TLS");
    auto* router = cooperative_source_router();
    require(router, "cold creation has CPU router");
    std::shared_ptr<CudaCooperativeFrontend> frontend;
    check(router->existing_frontend(after, frontend), "inspect cold creation without actor import");
    require(!frontend, "creation alone must not initialize a private actor");
    CudaEventProjection projection;
    const auto found = router->event_projection(event, projection);
    require(is_native ? found == CUDA_ERROR_INVALID_HANDLE
                      : found == CUDA_SUCCESS && projection.context() == after &&
                            projection.has_native_completion() && !projection.is_logical(),
            "only model root publishes exact fresh-unrecorded CPU provenance");
    const auto finish = [&] {
        check(cudaEventQuery(event), "actual unrecorded event remains ready");
        check(cudaEventDestroy(event), "destroy cold-created SDK event once");
    };
    if (is_native) {
        const NativeDriverCallScope native;
        finish();
    } else {
        finish();
    }
    CudaEventProjection retired;
    require(router->event_projection(event, retired) == CUDA_ERROR_INVALID_HANDLE,
            "cold creation/destruction leaves no stale CPU registration");
    check(router->existing_frontend(after, frontend), "inspect cold cleanup without actor import");
    require(!frontend, "cold metadata lifecycle submits no private execution");
    std::cout << "normal_runtime_cold_create native=" << is_native
              << " actual_sdk=1 lazy_binding=1 sdk_last=0,0,0 actor_import=0 checked_cleanup=1\n";
}

struct RuntimeCompletionObservation {
    std::array<cudaError_t, 3> last{};
    CUcontext caller_context{};
    std::array<cudaError_t, 4> record_errors{};
    std::array<cudaError_t, 4> flag_errors{};
    std::array<cudaError_t, 3> record_last{};
    CUcontext record_context{};
    cudaError_t legacy_query{};
    std::array<cudaError_t, 4> wait_errors{};
    std::array<cudaError_t, 4> wait_flag_errors{};
    std::array<cudaError_t, 3> wait_last{};
    CUcontext wait_context{};
    std::array<cudaError_t, 3> create_last{};
    CUcontext create_context{};
    std::array<cudaError_t, 4> create_errors{};
    std::array<cudaError_t, 4> create_flag_errors{};
    std::array<std::array<cudaError_t, 4>, 5> timing_errors{};
    float timing_output{-1.0F};
    std::array<cudaError_t, 3> device_sync_last{};
    CUcontext device_sync_context{};
    bool operator==(const RuntimeCompletionObservation&) const = default;
};

struct RuntimeRecordProbeOptions {
    bool is_ptds{};
    bool uses_flags{};
    bool uses_implicit_stream{};
    bool is_caller_unbound{};
    bool has_prior_error{};
    bool uses_runtime_wait{};
    bool uses_stream_completion{};
    bool uses_queued_release{};
    bool uses_runtime_creation{};
    bool uses_default_create{};
    bool uses_elapsed_time{};
    unsigned int timing_case{};
    bool uses_timing_interleaving{};
    bool uses_device_completion{};
};

RuntimeCompletionObservation check_normal_runtime_pending_completion(
    bool is_native = false, bool uses_runtime_record = false, bool has_registration = true,
    bool is_caller_unbound = false, bool has_prior_error = false,
    bool uses_logical_consumer = false, RuntimeRecordProbeOptions record_options = {}) {
    const ProcessCallGate::Call call(cooperative_process_call_gate());
    DriverDispatch driver;
    require(driver.initialize(), "resolve actual cached Runtime completion dispatch");
    CUcontext context{};
    CUstream stream{};
    CUstream consumer{};
    CUstream unbound_completion_stream{};
    CUmodule module{};
    CUfunction hold{};
    std::uint32_t* flags{};
    CUdeviceptr mapped{};
    {
        const NativeDriverCallScope native;
        check(cudaSetDevice(0), "bind actual cached completion context");
        check(driver.context_get_current(&context), "observe actual cached completion context");
        check(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), "create native completion stream");
        if (uses_logical_consumer && !record_options.uses_implicit_stream) {
            check(cuStreamCreate(&consumer, CU_STREAM_NON_BLOCKING),
                  "create empty original consumer stream");
        }
        check(cuModuleLoadData(&module, kHeldPrefixPtx), "load original held completion kernel");
        if (record_options.uses_stream_completion) {
            check(cuStreamCreate(&unbound_completion_stream, CU_STREAM_NON_BLOCKING),
                  "create exact unbound completion stream");
        }
        check(cuModuleGetFunction(&hold, module, "hold_event_prefix"), "resolve held original");
        check(cuMemHostAlloc(reinterpret_cast<void**>(&flags), 2 * sizeof(std::uint32_t),
                             CU_MEMHOSTALLOC_DEVICEMAP),
              "allocate actual held completion state");
        check(cuMemHostGetDevicePointer(&mapped, flags, 0), "map actual held completion state");
    }
    flags[0] = flags[1] = 0;
    struct Release {
        std::uint32_t* flags;
        ~Release() {
            if (flags) {
                std::atomic_ref(flags[0]).store(1, std::memory_order_release);
            }
        }
    } release{flags};
    auto* router = cooperative_source_router();
    require(router, "cached completion has actual process owner");
    std::shared_ptr<CudaCooperativeFrontend> frontend;
    check(router->frontend(context, frontend), "prepare cached completion actor");
    CUevent event{};
    CUevent dependency{};
    CUevent timing_peer{};
    CUevent timing_race_event{};
    RuntimeCompletionObservation observation;
    CUdeviceptr queued_release{};
    constexpr std::size_t k_release_bytes = 65536;
    std::size_t release_baseline{};
    bool has_retained_release{true}, has_recovered_release{true};
    bool has_early_record{true}, has_early_empty_wait{true};
    bool has_native_timing_observation{true}, has_unknown_timing_peer{true};
    std::optional<std::jthread> record_release;
    const auto record = [&](cudaEvent_t target, cudaStream_t target_stream) {
        return record_options.uses_flags
                   ? runtime_symbol<decltype(&cudaEventRecordWithFlags)>(
                         record_options.is_ptds
                             ? "cudaEventRecordWithFlags_ptsz"
                             : "cudaEventRecordWithFlags")(target, target_stream, 0)
                   : runtime_symbol<decltype(&cudaEventRecord)>(
                         record_options.is_ptds ? "cudaEventRecord_ptsz" : "cudaEventRecord")(
                         target, target_stream);
    };
    if (record_options.uses_runtime_creation) {
        const auto create = [&](cudaEvent_t* output) {
            return record_options.uses_default_create
                       ? cudaEventCreate(output)
                       : cudaEventCreateWithFlags(output, cudaEventDisableTiming);
        };
        {
            const NativeDriverCallScope native;
            cudaEvent_t warmed{};
            check(create(&warmed), "warm actual cached Runtime creation");
            check(record(warmed, consumer), "warm cached record without touching target event");
            check(cuCtxSynchronize(), "prove independent warm record complete");
            check(cudaEventDestroy(warmed), "destroy original warm creation once");
        }
        const auto reject_create = [&] {
            const auto failed = create(nullptr);
            observation.create_errors = {failed, cudaPeekAtLastError(), cudaGetLastError(),
                                         cudaPeekAtLastError()};
            cudaEvent_t unchanged{};
            const auto bad_flags = cudaEventCreateWithFlags(&unchanged, ~0U);
            observation.create_flag_errors = {bad_flags, cudaPeekAtLastError(), cudaGetLastError(),
                                              cudaPeekAtLastError()};
            require(failed != cudaSuccess && bad_flags != cudaSuccess && !unchanged,
                    "SDK validates creation output and flags without publishing a handle");
            if (!is_native) {
                unsigned int originals{};
                require(intercept_normal_runtime_event_create(&unchanged, cudaEventDefault,
                                                              [&] {
                                                                  ++originals;
                                                                  return cudaErrorApiFailureBase;
                                                              }) == cudaErrorApiFailureBase &&
                            originals == 1 && !unchanged,
                        "failed create outside CUresult preserves one SDK call and output");
            }
        };
        if (is_native) {
            const NativeDriverCallScope native;
            reject_create();
        } else {
            reject_create();
        }
        {
            const NativeDriverCallScope native;
            if (record_options.has_prior_error) {
                require(cudaEventQuery(nullptr) != cudaSuccess, "seed original creation error");
            }
            if (record_options.is_caller_unbound) {
                check(cuCtxSetCurrent(nullptr), "unbind SDK creation caller");
            }
        }
        const auto create_and_observe = [&] {
            check(create(&event), "create actual application Runtime completion event");
            check(driver.context_get_current(&observation.create_context),
                  "observe original post-SDK creation context");
            observation.create_last = {cudaPeekAtLastError(), cudaGetLastError(),
                                       cudaPeekAtLastError()};
        };
        if (is_native) {
            const NativeDriverCallScope native;
            create_and_observe();
        } else {
            create_and_observe();
        }
        {
            const NativeDriverCallScope native;
            check(cuCtxSetCurrent(context), "restore creation cleanup context after observation");
        }
    } else if (has_registration) {
        check(cuEventCreate(&event, record_options.uses_elapsed_time ? CU_EVENT_DEFAULT
                                                                     : CU_EVENT_DISABLE_TIMING),
              "create exact logical completion event");
    } else {
        const NativeDriverCallScope native;
        check(cuEventCreate(&event, CU_EVENT_DISABLE_TIMING), "create original unknown event");
    }
    if (uses_logical_consumer) {
        check(cuEventCreate(&dependency, CU_EVENT_DISABLE_TIMING),
              "create actual logical-only consumer dependency");
        if (record_options.uses_elapsed_time) {
            if (record_options.uses_timing_interleaving) {
                check(cuEventCreate(&timing_race_event, CU_EVENT_DEFAULT),
                      "create known ready sampling-window event");
            }
            if (record_options.uses_timing_interleaving) {
                check(cuEventCreate(&timing_peer, CU_EVENT_DEFAULT),
                      "create known peer for ordered two-registration sampling");
            } else {
                const NativeDriverCallScope native;
                check(cuEventCreate(&timing_peer, CU_EVENT_DEFAULT),
                      "create unregistered original timing-enabled peer event");
            }
            const auto observe_unrecorded = [&] {
                float unset{-1.0F};
                const auto unrecorded = cudaEventElapsedTime(&unset, timing_peer, timing_peer);
                observation.timing_errors[4] = {unrecorded, cudaPeekAtLastError(),
                                                cudaGetLastError(), cudaPeekAtLastError()};
            };
            if (is_native) {
                const NativeDriverCallScope native;
                observe_unrecorded();
            } else {
                observe_unrecorded();
            }
        }
        const NativeDriverCallScope native;
        if (record_options.uses_runtime_wait) {
            const auto wait = runtime_symbol<decltype(&cudaStreamWaitEvent)>(
                record_options.is_ptds ? "cudaStreamWaitEvent_ptsz" : "cudaStreamWaitEvent");
            check(wait(consumer, dependency, 0), "warm original cached Runtime wait");
        }
        if (!record_options.uses_runtime_creation) {
            check(record(event, consumer), "warm cached original Runtime consumer record");
        }
        if (timing_peer) {
            check(record(timing_peer, consumer), "record independent completed timing peer");
        }
        if (timing_race_event) {
            check(record(timing_race_event, consumer), "warm completed sampling-window target");
        }
        check(cuCtxSynchronize(), "prove warm original marker complete before held producer");
        if (record_options.uses_device_completion) {
            check(cudaDeviceSynchronize(), "warm original device sync before pending prefixes");
        }
        if (record_options.uses_elapsed_time) {
            float milliseconds{};
            check(cudaEventElapsedTime(&milliseconds, event, event),
                  "warm actual SDK timing on a completed independent prefix");
        }
        if (record_options.uses_stream_completion) {
            const auto query = runtime_symbol<decltype(&cudaStreamQuery)>(
                record_options.is_ptds ? "cudaStreamQuery_ptsz" : "cudaStreamQuery");
            const auto synchronize = runtime_symbol<decltype(&cudaStreamSynchronize)>(
                record_options.is_ptds ? "cudaStreamSynchronize_ptsz" : "cudaStreamSynchronize");
            check(query(consumer), "warm actual original stream query");
            check(synchronize(consumer), "warm actual original stream synchronization");
        }
    }
    if (record_options.uses_queued_release) {
        const auto allocate = [&] {
            if (!is_native) {
                std::size_t total{};
                check(cuMemGetInfo(&release_baseline, &total),
                      "observe exact release quota baseline");
            }
            check(cuMemAlloc(&queued_release, k_release_bytes),
                  "allocate queued release proof buffer");
        };
        if (is_native) {
            const NativeDriverCallScope native;
            allocate();
        } else {
            allocate();
        }
    }
    {
        const NativeDriverCallScope native;
        check(frontend->native(
                  stream, false,
                  [&] {
                      std::array<void*, 1> parameters{&mapped};
                      return driver.launch_kernel(hold, 1, 1, 1, 1, 1, 1, 0, stream,
                                                  parameters.data(), nullptr);
                  },
                  true),
              "admit held original prefix with independent actual execution fence");
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (std::atomic_ref(flags[1]).load(std::memory_order_acquire) == 0) {
        require(std::chrono::steady_clock::now() < deadline, "held original prefix must start");
        std::this_thread::yield();
    }
    if (uses_logical_consumer) {
        const auto prepare_consumer = [&] {
            if (record_options.uses_runtime_creation) {
                record_release.emplace([flags] {
                    std::this_thread::sleep_for(std::chrono::milliseconds(250));
                    std::atomic_ref(flags[0]).store(1, std::memory_order_release);
                });
                const auto wait_empty = runtime_symbol<decltype(&cudaStreamWaitEvent)>(
                    record_options.is_ptds ? "cudaStreamWaitEvent_ptsz" : "cudaStreamWaitEvent");
                check(wait_empty(consumer, event, 0), "accept actual fresh-unrecorded event wait");
                has_early_empty_wait =
                    std::atomic_ref(flags[0]).load(std::memory_order_acquire) == 0;
            }
            check(cuEventRecord(dependency, stream), "record held producer dependency");
            if (record_options.uses_runtime_wait) {
                const auto wait = runtime_symbol<decltype(&cudaStreamWaitEvent)>(
                    record_options.is_ptds ? "cudaStreamWaitEvent_ptsz" : "cudaStreamWaitEvent");
                const auto bad_event = wait(consumer, nullptr, 0);
                observation.wait_errors = {bad_event, cudaPeekAtLastError(), cudaGetLastError(),
                                           cudaPeekAtLastError()};
                require(bad_event != cudaSuccess, "original invalid wait rejects held work");
                const auto bad_flags = wait(consumer, dependency, ~0U);
                observation.wait_flag_errors = {bad_flags, cudaPeekAtLastError(),
                                                cudaGetLastError(), cudaPeekAtLastError()};
                require(bad_flags != cudaSuccess, "original invalid wait flags reject");
                if (record_options.has_prior_error) {
                    const NativeDriverCallScope native;
                    require(cudaEventQuery(nullptr) != cudaSuccess, "seed original wait error");
                }
                if (record_options.is_caller_unbound) {
                    const NativeDriverCallScope native;
                    check(cuCtxSetCurrent(nullptr), "unbind original caller before wait");
                }
                if (!is_native) {
                    unsigned int originals{};
                    require(
                        intercept_normal_runtime_event_wait(consumer, dependency,
                                                            record_options.is_ptds, 0,
                                                            [&] {
                                                                ++originals;
                                                                return cudaErrorApiFailureBase;
                                                            }) == cudaErrorApiFailureBase &&
                            originals == 1,
                        "failed SDK wait outside CUresult preserves one call without acceptance");
                }
                check(wait(consumer, dependency, 0), "accept cached Runtime consumer dependency");
                check(driver.context_get_current(&observation.wait_context),
                      "observe actual post-SDK wait caller context");
                observation.wait_last = {cudaPeekAtLastError(), cudaGetLastError(),
                                         cudaPeekAtLastError()};
                if (record_options.is_caller_unbound) {
                    const NativeDriverCallScope native;
                    check(cuCtxSetCurrent(context),
                          "restore wait cleanup context after SDK effect");
                }
            } else {
                const auto wait = runtime_symbol<decltype(&cuStreamWaitEvent)>(
                    record_options.is_ptds ? "cuStreamWaitEvent_ptsz" : "cuStreamWaitEvent");
                check(wait(consumer, dependency, 0), "accept exact consumer dependency");
            }
            const auto reject = [&] {
                const auto failed = record(nullptr, consumer);
                observation.record_errors = {failed, cudaPeekAtLastError(), cudaGetLastError(),
                                             cudaPeekAtLastError()};
                require(failed != cudaSuccess, "original invalid record rejects while work held");
                if (record_options.uses_flags) {
                    const auto entry = runtime_symbol<decltype(&cudaEventRecordWithFlags)>(
                        record_options.is_ptds ? "cudaEventRecordWithFlags_ptsz"
                                               : "cudaEventRecordWithFlags");
                    const auto bad_flags = entry(event, consumer, ~0U);
                    observation.flag_errors = {bad_flags, cudaPeekAtLastError(), cudaGetLastError(),
                                               cudaPeekAtLastError()};
                    require(bad_flags != cudaSuccess, "original invalid record flags reject");
                }
                if (record_options.has_prior_error) {
                    const NativeDriverCallScope native;
                    require(cudaEventQuery(nullptr) != cudaSuccess, "seed original record error");
                }
                if (record_options.is_caller_unbound) {
                    const NativeDriverCallScope native;
                    check(cuCtxSetCurrent(nullptr), "unbind original caller before record");
                }
            };
            reject();
            if (!is_native) {
                unsigned int originals{};
                require(
                    intercept_normal_runtime_event_record(event, consumer, record_options.is_ptds,
                                                          [&] {
                                                              ++originals;
                                                              return cudaErrorApiFailureBase;
                                                          }) == cudaErrorApiFailureBase &&
                        originals == 1,
                    "failed SDK record outside CUresult preserves one call without acceptance");
            }
            check(record(event, consumer), "record cached Runtime consumer marker");
            has_early_record = std::atomic_ref(flags[0]).load(std::memory_order_acquire) == 0;
            check(driver.context_get_current(&observation.record_context),
                  "observe actual post-SDK record caller context");
            observation.record_last = {cudaPeekAtLastError(), cudaGetLastError(),
                                       cudaPeekAtLastError()};
            if (record_options.is_caller_unbound) {
                const NativeDriverCallScope native;
                check(cuCtxSetCurrent(context), "restore record cleanup context after SDK effect");
            }
            if (record_options.uses_queued_release) {
                const auto release_async = runtime_symbol<decltype(&cuMemFreeAsync)>(
                    record_options.is_ptds ? "cuMemFreeAsync_ptsz" : "cuMemFreeAsync");
                check(release_async(queued_release, consumer),
                      "accept stream-ordered free behind held logical consumer");
            }
        };
        if (is_native) {
            const NativeDriverCallScope native;
            prepare_consumer();
        } else {
            prepare_consumer();
        }
    } else if (is_native || !has_registration) {
        const NativeDriverCallScope native;
        check(cuEventRecord(event, stream), "record original event after actual native prefix");
    } else if (uses_runtime_record) {
        check(cudaEventRecord(event, stream), "accept Runtime record after held native prefix");
    } else {
        check(cuEventRecord(event, stream),
              "accept logical record after actual held native prefix");
    }
    if (record_options.uses_stream_completion && record_options.is_ptds &&
        record_options.uses_implicit_stream) {
        if (is_native) {
            const NativeDriverCallScope native;
            observation.legacy_query = cudaStreamQuery(nullptr);
        } else {
            observation.legacy_query = cudaStreamQuery(nullptr);
        }
    }
    if (record_options.uses_stream_completion) {
        const auto query_unknown = [&] {
            check(cudaStreamQuery(unbound_completion_stream),
                  "unbound native query remains ready while independent work is held");
        };
        if (is_native) {
            const NativeDriverCallScope native;
            query_unknown();
        } else {
            query_unknown();
        }
        bool has_binding{true};
        check(frontend->complete_stream_snapshot(unbound_completion_stream, false, false,
                                                 has_binding),
              "inspect exact native query's binding without importing work");
        require(!has_binding, "normal unknown query must not import a private fence");
    }
    cudaError_t original_query{};
    float milliseconds{-1.0F};
    if (!is_native && !record_options.uses_runtime_creation) {
        for (bool should_wait : {false, true}) {
            unsigned int originals{};
            require(intercept_normal_runtime_event_completion(event, should_wait,
                                                              [&] {
                                                                  ++originals;
                                                                  return cudaErrorApiFailureBase;
                                                              }) == cudaErrorApiFailureBase &&
                        originals == 1 &&
                        std::atomic_ref(flags[0]).load(std::memory_order_acquire) == 0,
                    "original completion error outside CUresult preserves one call without wait");
        }
    }
    {
        const NativeDriverCallScope native;
        original_query =
            record_options.uses_elapsed_time ? cudaEventElapsedTime(&milliseconds, event, event)
            : record_options.uses_stream_completion
                ? runtime_symbol<decltype(&cudaStreamQuery)>(
                      record_options.is_ptds ? "cudaStreamQuery_ptsz" : "cudaStreamQuery")(consumer)
                : cudaEventQuery(event);
        (void)cudaGetLastError();
        if (has_prior_error) {
            require(cudaEventQuery(nullptr) != cudaSuccess, "seed actual original SDK error");
        }
        if (is_caller_unbound) {
            check(cuCtxSetCurrent(nullptr), "unbind caller before actual Runtime query");
        }
    }
    if (record_options.uses_stream_completion && !is_native) {
        for (bool should_wait : {false, true}) {
            unsigned int originals{};
            require(intercept_normal_runtime_stream_completion(consumer, record_options.is_ptds,
                                                               should_wait,
                                                               [&] {
                                                                   ++originals;
                                                                   return cudaErrorApiFailureBase;
                                                               }) == cudaErrorApiFailureBase &&
                        originals == 1 &&
                        std::atomic_ref(flags[0]).load(std::memory_order_acquire) == 0,
                    "SDK stream error outside CUresult preserves one call without private drain");
        }
    }
    CUresult logical_query{};
    cudaError_t runtime_query{};
    const CUevent timing_start = record_options.timing_case == 2 ? timing_peer : event;
    const CUevent timing_end = record_options.timing_case == 1 ? timing_peer : event;
    cudaError_t interleaved_record{cudaSuccess};
    CUresult interleaved_logical{CUDA_SUCCESS};
    unsigned int interleaved_originals{};
    std::optional<std::jthread> timing_rerecord;
    std::latch rerecord_started{1};
    std::latch sample_finished{1};
    float prior_timing{-1.0F};
    if (record_options.uses_timing_interleaving) {
        const NativeDriverCallScope native;
        check(cudaEventElapsedTime(&prior_timing, timing_peer, timing_race_event),
              "observe exact old timestamps before sampling-window rerecord");
    }
    const auto query = [&] {
        if (record_options.uses_elapsed_time) {
            float unrelated{-1.0F};
            const auto native_only = cudaEventElapsedTime(&unrelated, timing_peer, timing_peer);
            has_native_timing_observation =
                native_only == cudaSuccess && unrelated == 0.0F &&
                std::atomic_ref(flags[0]).load(std::memory_order_acquire) == 0;
            CudaEventProjection unknown;
            has_unknown_timing_peer =
                router->event_projection(timing_peer, unknown) ==
                (record_options.uses_timing_interleaving ? CUDA_SUCCESS
                                                         : CUDA_ERROR_INVALID_HANDLE);
            const auto observe_error = [&](unsigned int index, float* output, CUevent first,
                                           CUevent last) {
                const auto status = cudaEventElapsedTime(output, first, last);
                observation.timing_errors[index] = {status, cudaPeekAtLastError(),
                                                    cudaGetLastError(), cudaPeekAtLastError()};
            };
            observe_error(0, &milliseconds, timing_start, nullptr);
            observe_error(1, &milliseconds, nullptr, timing_end);
            observe_error(2, nullptr, timing_start, timing_end);
            observe_error(3, &milliseconds, event, dependency);
            if (has_prior_error) {
                const NativeDriverCallScope native;
                require(cudaEventQuery(nullptr) != cudaSuccess, "seed native timing caller error");
            }
            if (is_caller_unbound) {
                const NativeDriverCallScope native;
                check(cuCtxSetCurrent(nullptr), "unbind exact caller before timing observation");
            }
            milliseconds = -1.0F;
        }
        if (record_options.uses_stream_completion) {
            const auto driver_query = runtime_symbol<decltype(&cuStreamQuery)>(
                record_options.is_ptds ? "cuStreamQuery_ptsz" : "cuStreamQuery");
            logical_query = router->with_context(context, [&] { return driver_query(consumer); });
        } else if (has_registration && !is_native) {
            logical_query = cuEventQuery(event);
        } else {
            const NativeDriverCallScope native;
            logical_query = router->with_context(context, [&] { return cuEventQuery(event); });
        }
        if (record_options.uses_timing_interleaving) {
            const auto sample_after_record = [&](float* output) {
                ++interleaved_originals;
                timing_rerecord.emplace([&] {
                    rerecord_started.count_down();
                    try {
                        if (is_native) {
                            sample_finished.wait();
                        }
                        {
                            const NativeDriverCallScope native;
                            check(cuCtxSetCurrent(context), "bind actual rerecord caller context");
                        }
                        if (is_native) {
                            const NativeDriverCallScope native;
                            interleaved_record = record(timing_race_event, consumer);
                        } else {
                            interleaved_record = record(timing_race_event, consumer);
                        }
                    } catch (...) {
                        interleaved_record = cudaErrorUnknown;
                    }
                });
                rerecord_started.wait();
                const NativeDriverCallScope native;
                return cudaEventElapsedTime(output, timing_peer, timing_race_event);
            };
            milliseconds = -1.0F;
            const auto measured =
                is_native ? std::optional<cudaError_t>{sample_after_record(&milliseconds)}
                          : intercept_normal_runtime_event_elapsed_time(
                                &milliseconds, timing_peer, timing_race_event, sample_after_record);
            runtime_query = measured.value_or(cudaErrorUnknown);
            sample_finished.count_down();
            if (timing_rerecord) {
                timing_rerecord->join();
            }
            interleaved_logical = cuEventQuery(timing_race_event);
        } else {
            runtime_query = record_options.uses_elapsed_time
                                ? cudaEventElapsedTime(&milliseconds, timing_start, timing_end)
                            : record_options.uses_stream_completion
                                ? runtime_symbol<decltype(&cudaStreamQuery)>(
                                      record_options.is_ptds ? "cudaStreamQuery_ptsz"
                                                             : "cudaStreamQuery")(consumer)
                                : cudaEventQuery(event);
        }
        check(driver.context_get_current(&observation.caller_context),
              "observe actual post-SDK caller context before restoration");
        observation.last = {cudaPeekAtLastError(), cudaGetLastError(), cudaPeekAtLastError()};
        if (record_options.uses_elapsed_time) {
            observation.timing_output = milliseconds;
        }
    };
    if (is_native) {
        const NativeDriverCallScope native;
        query();
    } else {
        query();
    }
    const auto release_quota = [&] {
        std::size_t available{}, total{};
        check(router->with_context(context, [&] { return cuMemGetInfo(&available, &total); }),
              "observe release quota without changing SDK caller context");
        return available;
    };
    if (record_options.uses_queued_release && !is_native) {
        has_retained_release = release_baseline >= k_release_bytes &&
                               release_quota() == release_baseline - k_release_bytes;
    }
    if (!record_options.uses_runtime_creation) {
        require(std::atomic_ref(flags[0]).load(std::memory_order_acquire) == 0,
                "pending Runtime query must return before original business release");
    }
    std::jthread delayed_release([flags] {
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        std::atomic_ref(flags[0]).store(1, std::memory_order_release);
    });
    cudaError_t synchronized{};
    if (record_options.uses_device_completion) {
        const NativeDriverCallScope native;
        if (has_prior_error) {
            require(cudaEventQuery(nullptr) != cudaSuccess, "seed original device-sync error");
        }
        if (is_caller_unbound) {
            check(cuCtxSetCurrent(nullptr), "unbind exact original device-sync caller");
        }
    }
    const auto synchronize = [&] {
        return record_options.uses_device_completion ? cudaDeviceSynchronize()
               : record_options.uses_stream_completion
                   ? runtime_symbol<decltype(&cudaStreamSynchronize)>(
                         record_options.is_ptds ? "cudaStreamSynchronize_ptsz"
                                                : "cudaStreamSynchronize")(consumer)
                   : cudaEventSynchronize(event);
    };
    if (is_native) {
        const NativeDriverCallScope native;
        synchronized = synchronize();
    } else {
        synchronized = synchronize();
    }
    if (record_options.uses_device_completion) {
        check(driver.context_get_current(&observation.device_sync_context),
              "observe actual SDK device-sync context before private cleanup");
        observation.device_sync_last = {cudaPeekAtLastError(), cudaGetLastError(),
                                        cudaPeekAtLastError()};
    }
    const bool has_sync_completion = std::atomic_ref(flags[0]).load(std::memory_order_acquire) == 1;
    if (record_options.uses_queued_release && !is_native) {
        has_recovered_release = release_quota() == release_baseline;
    }
    delayed_release.join();
    if (record_release) {
        record_release->join();
    }
    std::atomic_ref(flags[0]).store(1, std::memory_order_release);
    {
        const NativeDriverCallScope native;
        check(cuCtxSetCurrent(context), "restore cleanup context only after original Runtime sync");
        check(frontend->synchronize_stream(stream, false), "prove exact held prefix and record");
        if (uses_logical_consumer) {
            check(frontend->synchronize_stream(consumer, record_options.is_ptds),
                  "prove consumer dependency complete before freeing either event");
        }
    }
    cudaError_t completed_timing{cudaSuccess};
    if (record_options.uses_elapsed_time) {
        completed_timing = cudaEventElapsedTime(&milliseconds, timing_start, timing_end);
        check(cudaEventDestroy(timing_peer), "retire completed distinct timing peer once");
        if (timing_race_event) {
            check(cudaEventDestroy(timing_race_event), "retire completed rerecord target once");
        }
    }
    check(cudaEventDestroy(event), "retire actual completed normal Runtime event once");
    if (dependency) {
        check(cudaEventDestroy(dependency), "retire completed producer dependency once");
    }
    check(router->close_context(context, [] { return CUDA_SUCCESS; }),
          "close actual cached completion actor after physical proof");
    frontend.reset();
    {
        const NativeDriverCallScope native;
        check(cuCtxSynchronize(), "prove every original held completion resource unused");
        check(cuModuleUnload(module), "unload proven original held completion module");
        release.flags = nullptr;
        check(cuMemFreeHost(flags), "release actual completed business state");
        check(cuStreamDestroy(stream), "destroy live original completion stream");
        if (consumer) {
            check(cuStreamDestroy(consumer), "destroy proven original consumer stream");
        }
        if (unbound_completion_stream) {
            check(cuStreamDestroy(unbound_completion_stream), "destroy original unbound stream");
        }
    }
    std::cout << "normal_runtime_pending native=" << is_native
              << " runtime_record=" << uses_runtime_record << " registered=" << has_registration
              << " logical_consumer=" << uses_logical_consumer
              << " record_ptds=" << record_options.is_ptds
              << " record_flags=" << record_options.uses_flags
              << " record_implicit=" << record_options.uses_implicit_stream
              << " record_unbound=" << record_options.is_caller_unbound
              << " runtime_wait=" << record_options.uses_runtime_wait
              << " stream_completion=" << record_options.uses_stream_completion
              << " queued_release=" << record_options.uses_queued_release
              << " runtime_creation=" << record_options.uses_runtime_creation
              << " elapsed_time=" << record_options.uses_elapsed_time
              << " timing_case=" << record_options.timing_case << " milliseconds=" << milliseconds
              << " record_before_release=" << has_early_record
              << " empty_wait_before_release=" << has_early_empty_wait
              << " caller_unbound=" << is_caller_unbound << " prior_error=" << has_prior_error
              << " original_query=" << original_query << " logical_query=" << logical_query
              << " runtime_query=" << runtime_query << " last=" << observation.last[0] << ','
              << observation.last[1] << ',' << observation.last[2] << " checked_cleanup=1"
              << " sync_status=" << synchronized << " sync_after_release=" << has_sync_completion
              << " quota_retained=" << has_retained_release
              << " quota_recovered=" << has_recovered_release << std::endl;
    require(has_early_record, "application-created Runtime event record must not drain work");
    if (record_options.uses_timing_interleaving) {
        std::cout << "timing_interleaving native=" << is_native
                  << " sdk_calls=" << interleaved_originals << " rerecord=" << interleaved_record
                  << " logical=" << interleaved_logical << " timing=" << runtime_query
                  << " checked_cleanup=1" << std::endl;
        require(interleaved_record == cudaSuccess && interleaved_originals == 1 &&
                    interleaved_logical == CUDA_ERROR_NOT_READY,
                "SDK sample contains one accepted pending concurrent rerecord");
    }
    require(has_early_empty_wait, "fresh-unrecorded Runtime wait must not drain unrelated work");
    require(logical_query == CUDA_ERROR_NOT_READY &&
                runtime_query ==
                    (record_options.uses_timing_interleaving ? cudaSuccess : cudaErrorNotReady),
            "cached Runtime query must not report success for an actually pending logical record");
    require(synchronized == cudaSuccess && has_sync_completion,
            "cached Runtime sync must prove logical completion before reporting success");
    require(has_retained_release && has_recovered_release,
            "pending private free retains quota until exact stream completion, then restores it");
    if (record_options.uses_elapsed_time) {
        require(observation.timing_output ==
                    (record_options.uses_timing_interleaving ? prior_timing : -1.0F),
                "pending logical timing must not publish an early SDK sample");
        require(has_native_timing_observation && has_unknown_timing_peer,
                "unknown native timing observes no binding and drains no unrelated work");
        require(completed_timing == cudaSuccess &&
                    (record_options.timing_case != 0 || milliseconds == 0.0F),
                "completed logical events retain exact original SDK timing");
        for (const auto& errors : observation.timing_errors) {
            require(errors[0] != cudaSuccess, "original SDK rejects invalid timing arguments");
        }
    }
    return observation;
}

void check_runtime_code_ready() {
    // This isolated cache proof owns no source transition or execution grant.
    check(cuInit(0), "initialize actual private-code proof process");
    const NativeDriverCallScope native;
    check(cudaSetDevice(0), "bind actual private Runtime code context");
    DriverDispatch driver;
    require(driver.initialize(), "resolve actual private-code Driver metadata");
    CUcontext context{};
    check(driver.context_get_current(&context), "observe private Runtime code context");
    const auto get_name = runtime_symbol<RuntimeFuncGetNameFunction>("cudaFuncGetName");
    const auto get_function = runtime_symbol<RuntimeGetFuncBySymbolFunction>("cudaGetFuncBySymbol");
    const auto is_loaded = runtime_symbol<decltype(&cuFuncIsLoaded)>("cuFuncIsLoaded");
    std::array<const void*, 2> hosts{glimmer_runtime_ready_kernel(false),
                                     glimmer_runtime_ready_kernel(true)};
    std::array<const char*, 2> names{};
    for (std::size_t index = 0; index < hosts.size(); ++index) {
        check(get_name(&names[index], hosts[index]), "obtain actual original Runtime entry name");
        require(names[index], "actual original Runtime entry name must be present");
    }
    Dl_info location{};
    require(::dladdr(hosts[0], &location) && location.dli_fname,
            "actual Runtime host entry must identify its ELF");
    const auto binary = std::filesystem::absolute(location.dli_fname).string();
    PersistentKernelRuntime runtime(driver, 3, false, true);
    std::array<CUfunction, 2> functions{};
    for (std::size_t index = 0; index < hosts.size(); ++index) {
        cudaFunction_t original{};
        check(get_function(&original, hosts[index]), "obtain actual original Runtime function");
        const auto variant = runtime.runtime_kernel(context, hosts[index], names[index], binary,
                                                    GLIMMER_CUOBJDUMP_PATH, original);
        require(variant && variant->kernel && variant->function,
                "qualified actual Runtime private function must publish");
        functions[index] = variant->function;
        CUfunctionLoadingState state = CU_FUNCTION_LOADING_STATE_UNLOADED;
        check(is_loaded(&state, variant->function), "query actual private GPU code loading state");
        require(state == CU_FUNCTION_LOADING_STATE_LOADED,
                "published actual Runtime private GPU code must already be loaded");
        state = CU_FUNCTION_LOADING_STATE_UNLOADED;
        check(is_loaded(&state, original), "query actual original GPU code loading state");
        require(variant->original_function == original && variant->original_module &&
                    state == CU_FUNCTION_LOADING_STATE_LOADED,
                "published Runtime variant must match a loaded actual original function");
    }
    require(functions[0] != functions[1] &&
                runtime.runtime_cooperative_source_is_prepared(context, binary),
            "two distinct actual functions share one terminal prepared source");
    runtime.forget_context(context);
    require(!runtime.runtime_cooperative_source_is_prepared(context, binary),
            "checked actual private library retirement removes source readiness");
    std::cout
        << "runtime_code_ready actual_sdk=1 loaded_private_functions=2 loaded_original_functions=2 "
           "original_kernel_issues=0 application_loading_mode_unchanged=1\n";
}

void check_runtime_kernel(bool is_ptds, bool uses_null_stream) {
    check(cudaSetDevice(0), "bind actual Runtime source kernel context");
    DriverDispatch driver;
    require(driver.initialize(), "resolve actual source kernel Driver metadata");
    cudaStream_t captured{};
    cudaStream_t native_stream{};
    unsigned int* counter{};
    const auto host_kernel = glimmer_runtime_capture_kernel();
    cudaFunction_t actual_kernel{};
    {
        const NativeDriverCallScope native;
        if (!uses_null_stream) {
            check(cudaStreamCreateWithFlags(&captured, cudaStreamNonBlocking),
                  "create original opaque capture stream");
        }
        check(cudaStreamCreateWithFlags(&native_stream, cudaStreamNonBlocking),
              "create original opaque execution stream");
        check(cudaMalloc(reinterpret_cast<void**>(&counter), sizeof(unsigned int)),
              "allocate original Runtime counter");
        check(cudaMemset(counter, 0, sizeof(unsigned int)), "initialize original Runtime counter");
        const auto get_function =
            runtime_symbol<RuntimeGetFuncBySymbolFunction>("cudaGetFuncBySymbol");
        check(get_function(&actual_kernel, host_kernel), "load original Runtime kernel only");
    }
    CUcontext context{};
    check(driver.context_get_current(&context), "observe actual source kernel context");
    auto& gate = cooperative_process_call_gate();
    const ProcessCallGate::Call call(gate);
    auto* router = cooperative_source_router();
    require(router, "configured source kernel owner required");
    std::shared_ptr<CudaCooperativeFrontend> frontend;
    check(router->frontend(context, frontend), "prepare kernel source owner");
    const auto begin = runtime_symbol<cudaError_t (*)(cudaStream_t, cudaStreamCaptureMode)>(
        is_ptds ? "cudaStreamBeginCapture_ptsz" : "cudaStreamBeginCapture");
    const auto end = runtime_symbol<cudaError_t (*)(cudaStream_t, cudaGraph_t*)>(
        is_ptds ? "cudaStreamEndCapture_ptsz" : "cudaStreamEndCapture");
    const auto transition = [&](cudaStream_t stream, CudaSourceCaptureKind kind,
                                RuntimeCaptureObservation& observed, cudaGraph_t& graph) {
        cudaError_t status{};
        const auto owned = router->source_capture_root(gate, kind, [&](auto& dispatch) {
            return invoke_runtime_capture_original(
                [&](CUcontext actual_context, CUstream actual_stream, bool actual_ptds,
                    CudaSourceCaptureKind actual_kind, const auto& original) {
                    return dispatch.invoke(actual_context, actual_stream, actual_ptds, actual_kind,
                                           original);
                },
                [&] {
                    return kind == CudaSourceCaptureKind::kBegin
                               ? begin(stream, cudaStreamCaptureModeGlobal)
                               : end(stream, &graph);
                },
                observed, status,
                [&](CUstream actual_stream, bool actual_ptds, CUcontext& actual_context) {
                    return dispatch.lookup_origin_context(actual_stream, actual_ptds,
                                                          actual_context);
                });
        });
        check(status, "actual SDK kernel capture transition");
        check(owned, "source kernel capture transition ownership");
        require(observed.original_driver_calls == 1 && observed.context == context &&
                    observed.is_ptds == is_ptds,
                "one typed kernel capture origin");
    };
    RuntimeCaptureObservation captured_origin, native_origin;
    cudaGraph_t graph{}, empty_graph{};
    transition(captured, CudaSourceCaptureKind::kBegin, captured_origin, graph);
    // A second original empty capture reveals its typed Driver stream without
    // converting the opaque Runtime handle. The first origin stays active.
    transition(native_stream, CudaSourceCaptureKind::kBegin, native_origin, empty_graph);
    transition(native_stream, CudaSourceCaptureKind::kEnd, native_origin, empty_graph);
    using Launch =
        cudaError_t(CUDARTAPI*)(const void*, dim3, dim3, void**, std::size_t, cudaStream_t);
    const auto launch = reinterpret_cast<Launch>(
        ::dlsym(RTLD_DEFAULT, is_ptds ? "cudaLaunchKernel_ptsz" : "cudaLaunchKernel"));
    const auto extended = reinterpret_cast<decltype(&cuLaunchKernelEx)>(
        ::dlsym(RTLD_DEFAULT, is_ptds ? "cuLaunchKernelEx_ptsz" : "cuLaunchKernelEx"));
    using RuntimeExtended = cudaError_t(CUDARTAPI*)(const cudaLaunchConfig_t*, const void*, void**);
    const auto runtime_extended = reinterpret_cast<RuntimeExtended>(
        ::dlsym(RTLD_DEFAULT, is_ptds ? "cudaLaunchKernelExC_ptsz" : "cudaLaunchKernelExC"));
    require(launch && runtime_extended && extended &&
                is_interceptor_driver_function(reinterpret_cast<void*>(extended)),
            "public SDK Runtime and extended Driver source wrappers required");
    std::array<void*, 1> arguments{&counter};
    const auto launch_extended = [&](CUstream stream) {
        CUlaunchConfig config{};
        config.gridDimX = config.gridDimY = config.gridDimZ = 1;
        config.blockDimX = config.blockDimY = config.blockDimZ = 1;
        config.hStream = stream;
        check(extended(&config, actual_kernel, arguments.data(), nullptr),
              "public original extended Driver source launch");
    };
    const auto launch_runtime_extended = [&](cudaStream_t stream) {
        cudaLaunchAttribute attribute{};
        attribute.id = cudaLaunchAttributeIgnore;
        cudaLaunchConfig_t config{};
        config.gridDim = config.blockDim = dim3(1);
        config.stream = stream;
        if (uses_null_stream) {
            config.attrs = &attribute;
            config.numAttrs = 1;
        }
        check(runtime_extended(&config, host_kernel, arguments.data()),
              "public original extended Runtime source launch");
    };
    const auto before = router->native_execution_stats();
    check(launch(host_kernel, dim3(1), dim3(1), arguments.data(), 0, captured),
          "public original captured Runtime kernel");
    launch_extended(captured_origin.stream);
    launch_runtime_extended(captured);
    require(router->native_execution_stats().grants == before.grants,
            "Runtime/extended graph construction has no execution grant");
    check(launch(host_kernel, dim3(1), dim3(1), arguments.data(), 0, native_stream),
          "public original uncaptured Runtime kernel");
    launch_extended(native_origin.stream);
    launch_runtime_extended(native_stream);
    require(router->native_execution_stats().grants == before.grants + 3,
            "Runtime/extended native execution receives three actual grants");
    transition(captured, CudaSourceCaptureKind::kEnd, captured_origin, graph);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (gate.is_native_only_on_current_thread()) {
        const auto status = router->try_finish_source_capture(gate);
        require(status == CUDA_SUCCESS || status == CUDA_ERROR_NOT_READY,
                "source kernel completion permits reconciliation");
        require(std::chrono::steady_clock::now() < deadline, "source kernel completion deadline");
        std::this_thread::yield();
    }
    const auto after = router->native_execution_stats();
    require(after.completed == before.completed + 3 && after.failed == before.failed,
            "all three original source executions physically complete before thaw");
    cudaLaunchConfig_t normal_config{};
    normal_config.gridDim = normal_config.blockDim = dim3(1);
    normal_config.stream = native_stream;
    require(
        runtime_extended(&normal_config, host_kernel, arguments.data()) == cudaErrorNotSupported &&
            router->native_execution_stats().grants == after.grants,
        "normal cooperative Runtime extended launch still rejects without issue or grant");
    std::size_t count{};
    check(cuGraphGetNodes(captured_origin.graph, nullptr, &count),
          "count original SDK kernel nodes");
    require(count == 3, "SDK kernel capture adds only its three original nodes");
    std::array<CUgraphNode, 3> nodes{};
    check(cuGraphGetNodes(captured_origin.graph, nodes.data(), &count),
          "inspect original kernel graph");
    for (const auto node : nodes) {
        CUDA_KERNEL_NODE_PARAMS params{};
        check(cuGraphKernelNodeGetParams(node, &params), "inspect original SDK kernel identity");
        require(params.func == actual_kernel && params.gridDimX == 1 && params.gridDimY == 1 &&
                    params.gridDimZ == 1 && params.blockDimX == 1 && params.blockDimY == 1 &&
                    params.blockDimZ == 1 && params.sharedMemBytes == 0,
                "captured SDK function/dimensions remain original");
    }
    CUgraphExec executable{};
    unsigned int value{};
    {
        const NativeDriverCallScope native;
        check(cudaMemcpy(&value, counter, sizeof(value), cudaMemcpyDeviceToHost),
              "read completed source kernel result before replay");
        require(value == 3, "construction does not execute and native issue is exact once");
        check(cuGraphInstantiateWithFlags(&executable, captured_origin.graph, 0),
              "instantiate original SDK kernel graph");
    }
    check(frontend->native(
              native_origin.stream, is_ptds,
              [&] { return driver.graph_launch(executable, native_origin.stream); }, true),
          "admit original SDK kernel graph replay once");
    check(frontend->synchronize_stream(native_origin.stream, is_ptds),
          "prove original SDK replay fence");
    {
        const NativeDriverCallScope native;
        check(cudaMemcpy(&value, counter, sizeof(value), cudaMemcpyDeviceToHost),
              "read exact SDK replay");
        check(cuGraphExecDestroy(executable), "destroy completed SDK executable");
        check(cudaGraphDestroy(graph), "destroy actual opaque SDK graph");
        check(cudaGraphDestroy(empty_graph), "destroy original empty graph");
    }
    require(value == 6, "Runtime and extended original kernels execute exactly once in replay");
    check(router->close_context(context, [] { return CUDA_SUCCESS; }),
          "close checked SDK kernel owner");
    frontend.reset();
    {
        const NativeDriverCallScope native;
        check(cudaFree(counter), "release original source kernel counter");
        check(cudaStreamDestroy(native_stream), "destroy original native Runtime stream");
        if (captured) {
            check(cudaStreamDestroy(captured), "destroy original captured Runtime stream");
        }
    }
}
void check_normal_null_free() {
    // The first public call deliberately precedes device setup and native baselines.
    check(cudaFree(nullptr), "cold public null free initializes through the original SDK");
    const auto native_free = original_runtime_symbol<RuntimeFreeFunction>("cudaFree");
    const auto native_reset =
        original_runtime_symbol<RuntimeDeviceResetFunction>("cudaDeviceReset");
    const auto produce =
        original_runtime_symbol<decltype(&cudaGetDeviceCount)>("cudaGetDeviceCount");
    const auto peek =
        original_runtime_symbol<decltype(&cudaPeekAtLastError)>("cudaPeekAtLastError");
    const auto get = original_runtime_symbol<decltype(&cudaGetLastError)>("cudaGetLastError");
    DriverDispatch driver;
    require(driver.initialize(), "resolve actual null-release context observer");
    CUcontext context{};
    check(driver.context_get_current(&context), "observe original cold null-release context");
    require(context && peek() == cudaSuccess, "cold SDK binds context without changing error TLS");
    check(cudaDeviceReset(), "retire first public cold null-release lifetime");
    for (unsigned int lifetime = 0; lifetime < 2; ++lifetime) {
        std::array<cudaError_t, 3> baseline{};
        bool native_keeps_context{}, native_active{};
        {
            const NativeDriverCallScope native;
            check(native_free(nullptr), "reset-separated original cold null free");
            CUcontext native_context{}, native_after{};
            check(driver.context_get_current(&native_context), "observe original baseline context");
            (void)get();
            require(produce(nullptr) != cudaSuccess, "seed native warmed error TLS");
            check(native_free(nullptr), "original warmed null free");
            baseline = {peek(), get(), peek()};
            check(native_reset(), "retire original null-release baseline context");
            check(driver.context_get_current(&native_after), "observe original reset context");
            require(!native_after || native_after == native_context,
                    "original reset does not substitute a foreign context");
            native_keeps_context = native_after != nullptr;
            check(driver.primary_context_is_active(0, native_active),
                  "observe original reset primary activity without activation");
        }
        check(cudaFree(nullptr), "reset-separated public cold null free");
        check(driver.context_get_current(&context), "read new original public context");
        require(context, "public cold null free preserves original lazy context initialization");
        (void)get();
        require(produce(nullptr) != cudaSuccess, "seed public warmed error TLS");
        check(cudaFree(nullptr), "public warmed null free preserves SDK invocation");
        require(std::array<cudaError_t, 3>{peek(), get(), peek()} == baseline,
                "public null-release nonzero error TLS exactly matches the original SDK");
        CUcontext after{};
        check(driver.context_get_current(&after), "read caller context after warmed null free");
        require(after == context, "warmed null free does not replace caller context");
        void* allocation{};
        check(cudaMalloc(&allocation, 3 * sizeof(unsigned int)),
              "cold null free leaves allocation accounting healthy");
        const std::array<unsigned int, 3> initial{0x12345678U, 0, 0x76543210U};
        check(cudaMemcpy(allocation, initial.data(), sizeof(initial), cudaMemcpyHostToDevice),
              "seed guarded public null-release allocation");
        auto* counter = static_cast<unsigned int*>(allocation) + 1;
        void* arguments[]{&counter};
        check(cudaLaunchKernel(glimmer_runtime_capture_kernel(), dim3{1}, dim3{1}, arguments, 0,
                               nullptr),
              "warm unchanged application kernel and private resource accounting");
        check(cudaDeviceSynchronize(), "complete first public kernel before quota baseline");
        std::size_t free_before{}, free_after{}, total{};
        check(cudaMemGetInfo(&free_before, &total), "observe live allocation before null release");
        require(total == 8388608, "null free preserves configured quota visibility");
        check(cudaLaunchKernel(glimmer_runtime_capture_kernel(), dim3{1}, dim3{1}, arguments, 0,
                               nullptr),
              "enqueue second public kernel before null-release boundary");
        check(cudaFree(nullptr), "null release retains normal pending-work memory boundary");
        check(cudaMemGetInfo(&free_after, &total), "observe live allocation after null release");
        require(free_after == free_before, "null release retires no live application charge");
        std::array<unsigned int, 3> output{};
        {
            const NativeDriverCallScope native;
            check(cudaMemcpy(output.data(), allocation, sizeof(output), cudaMemcpyDeviceToHost),
                  "independent SDK readback after public null-release boundary");
        }
        require(output[0] == initial[0] && output[1] == 2 && output[2] == initial[2],
                "pending original kernel output is exact once with intact guards");
        check(cudaFree(allocation), "nonnull tracked release still retires its allocation");
        check(cudaDeviceReset(), "retire checked public null-release lifetime");
        check(driver.context_get_current(&after), "observe original reset context cleanup");
        bool active{};
        check(driver.primary_context_is_active(0, active), "observe public reset primary activity");
        require((native_keeps_context ? after == context : !after) && active == native_active &&
                    !active && !is_inside_driver_call() && !is_inside_runtime_call(),
                "reset preserves measured SDK context state and retires primary resources");
    }
    std::cout << "runtime_null_free cold=1 lifetimes=2 tls=1 quota=1 output=2 guards=1 cleanup=1\n";
}

std::atomic<bool> runtime_recovery_projection_failure{};
std::atomic<CUevent> runtime_recovery_failed_event{};
std::atomic<unsigned int> runtime_recovery_failed_queries{};
std::atomic<unsigned int> runtime_recovery_driver_syncs{};
GetProcAddressV2Function runtime_recovery_original_lookup{};
CUresult runtime_recovery_query(CUevent event) {
    if (runtime_recovery_projection_failure.exchange(false)) {
        runtime_recovery_failed_event = event;
        ++runtime_recovery_failed_queries;
        return CUDA_ERROR_LAUNCH_FAILED;
    }
    if (event == runtime_recovery_failed_event.load()) ++runtime_recovery_failed_queries;
    return cuEventQuery(event);
}
CUresult runtime_recovery_destroy(CUevent event) {
    const auto status = cuEventDestroy(event);
    if (status == CUDA_SUCCESS && event == runtime_recovery_failed_event.load())
        runtime_recovery_failed_event = nullptr;
    return status;
}
CUresult runtime_recovery_driver_sync() {
    ++runtime_recovery_driver_syncs;
    return CUDA_ERROR_UNKNOWN;  // a second internal SDK synchronization cannot supply proof
}
CUresult runtime_recovery_lookup(const char* name, void** output, int version, cuuint64_t flags) {
    CUdriverProcAddressQueryResult found{};
    const auto status = runtime_recovery_original_lookup(name, output, version, flags, &found);
    if (status != CUDA_SUCCESS) return status;
    if (found != CU_GET_PROC_ADDRESS_SUCCESS) return CUDA_ERROR_NOT_FOUND;
    if (std::strcmp(name, "cuEventQuery") == 0)
        *output = reinterpret_cast<void*>(runtime_recovery_query);
    if (std::strcmp(name, "cuEventDestroy") == 0)
        *output = reinterpret_cast<void*>(runtime_recovery_destroy);
    if (std::strcmp(name, "cuCtxSynchronize") == 0)
        *output = reinterpret_cast<void*>(runtime_recovery_driver_sync);
    return CUDA_SUCCESS;
}
void check_runtime_epoch_recovery(unsigned int mode, bool is_cached, bool should_seed_error = true,
                                  bool uses_process_owner = false) {
    DriverDispatch sdk_driver;
    require(sdk_driver.initialize(), "actual Runtime recovery Driver dispatch");
    void* lookup{};
    require(sdk_driver.get_proc_address("cuGetProcAddress", &lookup, 12600,
                                        CU_GET_PROC_ADDRESS_LEGACY_STREAM) == CUDA_SUCCESS &&
                lookup,
            "actual unwrapped SDK recovery proc-address function");
    runtime_recovery_original_lookup = reinterpret_cast<GetProcAddressV2Function>(lookup);
    DriverFunctionTable table;
    table.get_proc_address = runtime_recovery_lookup;
    const auto resolve = [&](const char* name, bool uses_ptds = false) {
        void* output{};
        require(sdk_driver.get_proc_address(
                    name, &output, 12600,
                    uses_ptds ? CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM
                              : CU_GET_PROC_ADDRESS_LEGACY_STREAM) == CUDA_SUCCESS &&
                    output && !is_interceptor_driver_function(output),
                "actual original SDK recovery dispatch member");
        return output;
    };
    table.context_get_current =
        reinterpret_cast<ContextGetCurrentFunction>(resolve("cuCtxGetCurrent"));
    table.context_get_device =
        reinterpret_cast<ContextGetDeviceFunction>(resolve("cuCtxGetDevice"));
    table.context_synchronize =
        reinterpret_cast<ContextSynchronizeFunction>(resolve("cuCtxSynchronize"));
    table.stream_capture_status =
        reinterpret_cast<StreamCaptureFunction>(resolve("cuStreamIsCapturing"));
    DriverDispatch driver(table);
    using Launch = cudaError_t (*)(const void*, dim3, dim3, void**, std::size_t, cudaStream_t);
    const bool is_ptds = mode == 1;
    const auto sdk_launch =
        original_runtime_symbol<Launch>(is_ptds ? "cudaLaunchKernel_ptsz" : "cudaLaunchKernel");
    const auto launch =
        is_cached ? sdk_launch
                  : runtime_symbol<Launch>(is_ptds ? "cudaLaunchKernel_ptsz" : "cudaLaunchKernel");
    const auto sdk_sync =
        original_runtime_symbol<RuntimeDeviceSynchronizeFunction>("cudaDeviceSynchronize");
    const auto synchronize =
        is_cached ? sdk_sync
                  : runtime_symbol<RuntimeDeviceSynchronizeFunction>("cudaDeviceSynchronize");
    const auto get_function =
        original_runtime_symbol<decltype(&cudaGetFuncBySymbol)>("cudaGetFuncBySymbol");
    const auto produce =
        original_runtime_symbol<decltype(&cudaGetDeviceCount)>("cudaGetDeviceCount");
    const auto peek =
        original_runtime_symbol<decltype(&cudaPeekAtLastError)>("cudaPeekAtLastError");
    const auto get = original_runtime_symbol<decltype(&cudaGetLastError)>("cudaGetLastError");
    const std::array<unsigned int, 3> initial{0x12345678U, 0, 0x76543210U};
    unsigned int* allocation{};
    cudaStream_t stream{};
    CUcontext context{};
    cudaFunction_t function{};
    CUevent pending{}, historical{};
    std::array<cudaError_t, 3> baseline{};
    {
        const NativeDriverCallScope native;
        check(cudaSetDevice(0), "actual Runtime recovery device");
        check(cudaFree(nullptr), "actual fresh Runtime recovery context");
        void* memory{};
        check(cudaMalloc(&memory, sizeof(initial)), "guarded actual Runtime recovery allocation");
        allocation = static_cast<unsigned int*>(memory);
        check(cudaMemcpy(allocation, initial.data(), sizeof(initial), cudaMemcpyHostToDevice),
              "initialize actual recovery guards");
        if (mode == 2) check(cudaStreamCreate(&stream), "actual recovery blocking stream");
        check(get_function(&function, glimmer_runtime_pending_kernel()),
              "actual pending original Runtime kernel identity");
        check(sdk_driver.context_get_current(&context), "actual SDK recovery context");
        check(cuEventCreate(&pending, CU_EVENT_DISABLE_TIMING), "actual pending observer");
        check(cuEventCreate(&historical, CU_EVENT_DISABLE_TIMING), "actual historical event");
        (void)get();
        if (should_seed_error)
            require(produce(nullptr) != cudaSuccess, "seed actual baseline SDK error TLS");
        check(sdk_sync(), "actual original Runtime completion baseline");
        baseline = {peek(), get(), peek()};
    }
    require(baseline[0] == baseline[1] && baseline[2] == cudaSuccess &&
                (should_seed_error ? baseline[0] != cudaSuccess : baseline[0] == cudaSuccess),
            "SDK baseline independently qualifies declared completion TLS mode");
    runtime_recovery_projection_failure = false;
    runtime_recovery_failed_event = nullptr;
    runtime_recovery_failed_queries = 0;
    runtime_recovery_driver_syncs = 0;
    auto& gate = cooperative_process_call_gate();
    const ProcessCallGate::Call call(gate);
    std::optional<DriverCooperativeRouter> isolated;
    auto* selected = uses_process_owner ? cooperative_source_router() : nullptr;
    if (!uses_process_owner) {
        isolated.emplace(driver, glimmer::control::LaunchGateOptions{}, false, false,
                         glimmer::control::NativeEpochExtent{8, 2, 2, 1'000'000});
        selected = &*isolated;
    }
    require(selected && selected->has_native_epoch_policy(),
            "exact enabled recovery process owner");
    auto& router = *selected;
    std::shared_ptr<CudaCooperativeFrontend> owner;
    check(router.frontend(context, owner), "actual Runtime recovery frontend");
    check(router.register_event(context, historical), "actual historical recovery identity");
    const auto actual_stream = is_ptds ? CU_STREAM_PER_THREAD : reinterpret_cast<CUstream>(stream);
    check(owner->record_event(historical, actual_stream, is_ptds), "original historical prefix");
    check(owner->synchronize_event(historical), "complete historical recovery prefix");
    CUgraph graph{};
    CUgraphExec executable{};
    unsigned int* counter = allocation + 1;
    unsigned long long duration = 500'000'000;
    void* arguments[]{&counter, &duration};
    {
        const NativeDriverCallScope native;
        check(cuGraphCreate(&graph, 0), "original graph before Runtime recovery");
        cudaFunction_t graph_function{};
        check(get_function(&graph_function, glimmer_runtime_capture_kernel()),
              "actual original recovery graph function");
        void* graph_arguments[]{&counter};
        CUDA_KERNEL_NODE_PARAMS parameters{};
        parameters.func = graph_function;
        parameters.gridDimX = parameters.gridDimY = parameters.gridDimZ = 1;
        parameters.blockDimX = parameters.blockDimY = parameters.blockDimZ = 1;
        parameters.kernelParams = graph_arguments;
        CUgraphNode node{};
        check(cuGraphAddKernelNode(&node, graph, nullptr, 0, &parameters),
              "original unchanged recovery graph node");
        CudaNativeGraphReservation reservation;
        check(router.prepare_graph(context, graph, reservation, true), "prepare original graph");
        check(cuGraphInstantiate(&executable, graph, 0), "actual original graph executable");
        check(router.graphs().publish(reservation, executable, CUDA_SUCCESS),
              "publish actual original graph identity");
    }
    unsigned int originals{}, completions{}, quarantines{};
    RuntimeCooperativeLaunch plan{function, std::nullopt, {1, 1, 1}, {1, 1, 1}};
    plan.context = context;
    const RuntimeNativeKernelExecution execution = [&](CUfunction actual, CUstream origin,
                                                       bool actual_ptds, const auto& original) {
        require(actual == function && actual_ptds == is_ptds,
                "actual SDK leaf binds original Runtime recovery function and PTDS");
        ++originals;
        return router.native_execution(gate, owner, origin, actual_ptds, [&] {
            runtime_recovery_projection_failure = duration != 0;
            return original();
        });
    };
    const auto quarantine = [&] {
        ++quarantines;
        router.quarantine_normal_launch(context, CUDA_ERROR_UNKNOWN);
    };
    require(
        router.has_native_epoch_policy() &&
            !invoke_runtime_context_completion_original(
                [&](const auto& original) {
                    return router.recover_current_native_epoch(gate, original);
                },
                [&] {
                    ++completions;
                    return synchronize();
                },
                quarantine) &&
            completions == 0,
        "enabled but unbound healthy ownership declines before SDK effects or cold actor import");
    const auto public_launch = runtime_symbol<decltype(&cuLaunchKernel)>(
        is_ptds ? "cuLaunchKernel_ptsz" : "cuLaunchKernel");
    const auto issue_original = [&] {
        if (uses_process_owner) {
            check(public_launch(function, 1, 1, 1, 1, 1, 1, 0, stream, arguments, nullptr),
                  "actual public original Driver fallback uses the configured process owner");
            ++originals;
            runtime_recovery_projection_failure = duration != 0;
        } else {
            check(invoke_runtime_cooperative_launch_original(
                      plan, is_ptds,
                      [&] {
                          return launch(glimmer_runtime_pending_kernel(), dim3{1}, dim3{1},
                                        arguments, 0, stream);
                      },
                      quarantine, execution),
                  "original Runtime kernel enters one physically owned pending epoch");
        }
    };
    constexpr std::size_t late_bytes = std::size_t{1024} * 1024;
    CUdeviceptr late_allocation{};
    CUstream late_stream{};
    CUevent late_pending{};
    std::size_t baseline_free{}, baseline_total{};
    if (uses_process_owner) {
        check(cuMemGetInfo(&baseline_free, &baseline_total),
              "public quota before tracked late allocation");
        check(cuMemAlloc(&late_allocation, late_bytes), "public tracked late-release allocation");
        check(cuStreamCreate(&late_stream, CU_STREAM_NON_BLOCKING),
              "public independent late-release stream");
        const NativeDriverCallScope native;
        check(cuEventCreate(&late_pending, CU_EVENT_DISABLE_TIMING),
              "independent late-prefix observer");
    }
    issue_original();
    {
        const NativeDriverCallScope native;
        const auto record =
            reinterpret_cast<decltype(&cuEventRecord)>(resolve("cuEventRecord", is_ptds));
        check(record(pending, actual_stream), "independent actual pending original observer");
    }
    static_cast<void>(owner->close_native_epoch(false));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
    while (is_cached ? owner->native_epoch_snapshot()->phase != CudaNativeEpochPhase::kRetained
                     : owner->query_event(nullptr) == CUDA_ERROR_INVALID_HANDLE) {
        require(std::chrono::steady_clock::now() < deadline, "bounded retained Runtime recovery");
        std::this_thread::yield();
    }
    const auto before = owner->native_epoch_snapshot();
    {
        const NativeDriverCallScope native;
        require(before && before->execution->has_ownership &&
                    cuEventQuery(pending) == CUDA_ERROR_NOT_READY,
                "retained Runtime failure still owns actual pending original work");
        if (should_seed_error)
            require(produce(nullptr) != cudaSuccess, "seed actual owned completion error TLS");
    }
    std::atomic<bool> sync_started{}, waiter_attempted{}, waiter_entered{};
    std::exception_ptr waiter_error;
    std::thread waiter([&] {
        const auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds{5};
        while (!sync_started.load()) {
            if (std::chrono::steady_clock::now() >= timeout) return;
            std::this_thread::yield();
        }
        waiter_attempted = true;
        const ProcessCallGate::Call competing(gate);
        waiter_entered = true;
        if (uses_process_owner) {
            bool has_pushed{};
            try {
                {
                    const NativeDriverCallScope native;
                    check(cuCtxPushCurrent(context), "late caller owns actual recovery context");
                    has_pushed = true;
                    unsigned long long late_duration = 500'000'000;
                    void* late_arguments[]{&counter, &late_duration};
                    check(cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, late_stream, late_arguments,
                                         nullptr),
                          "independent native prefix issues after the earlier SDK completion");
                    check(cuEventRecord(late_pending, late_stream),
                          "observe actual later pending prefix");
                }
                check(cuMemFreeAsync(late_allocation, late_stream),
                      "actual public asynchronous free uses the same recovered process queue");
            } catch (...) {
                waiter_error = std::current_exception();
            }
            if (has_pushed) {
                const NativeDriverCallScope native;
                CUcontext popped{};
                const auto restored = cuCtxPopCurrent(&popped);
                if ((restored != CUDA_SUCCESS || popped != context) && !waiter_error)
                    waiter_error = std::make_exception_ptr(
                        std::runtime_error("late caller context restoration"));
            }
        }
    });
    const auto original_completion = [&] {
        ++completions;
        require(is_inside_runtime_call() && is_inside_driver_call() &&
                    gate.is_native_only_on_current_thread(),
                "actual SDK recovery completion owns caller and process exclusion");
        sync_started = true;
        const auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds{5};
        while (!waiter_attempted.load()) {
            require(std::chrono::steady_clock::now() < timeout, "bounded excluded SDK waiter");
            std::this_thread::yield();
        }
        require(!waiter_entered.load(), "process exclusion precedes real SDK completion");
        const auto result = synchronize();
        if (uses_process_owner && result == cudaSuccess)
            require(cuEventQuery(pending) == CUDA_SUCCESS,
                    "original SDK completion independently proves the actual pending epoch before "
                    "acknowledgment");
        return result;
    };
    const auto completed = uses_process_owner
                               ? intercept_normal_runtime_device_completion(original_completion)
                               : intercept_normal_runtime_device_completion(
                                     original_completion, [&](const auto& original) {
                                         return router.recover_current_native_epoch(gate, original);
                                     });
    waiter.join();
    if (waiter_error) std::rethrow_exception(waiter_error);
    require(
        completed == cudaSuccess && completions == 1 && originals == 1 && quarantines == 0 &&
            waiter_entered && runtime_recovery_driver_syncs == 0 &&
            runtime_recovery_failed_queries == 1 && !runtime_recovery_failed_event.load(),
        "one actual Runtime SDK proof heals exact frontend without another sync or query replay");
    const std::array observed{peek(), get(), peek()};
    require(observed == baseline, "borrowed completion preserves exact measured SDK TLS mode");
    CUcontext after{};
    check(sdk_driver.context_get_current(&after), "actual borrowed SDK return context");
    const auto recovered = owner->native_epoch_snapshot();
    require(after == context && recovered->status == before->status &&
                recovered->execution->has_failed && !recovered->execution->has_ownership &&
                recovered->created_events == recovered->destroyed_events &&
                recovered->execution->issued_calls == 1 && !is_inside_driver_call() &&
                !is_inside_runtime_call(),
            "old error/count and actual context survive borrowed recovery and logical thaw");
    if (uses_process_owner) {
        std::size_t charged_free{}, charged_total{};
        check(cuMemGetInfo(&charged_free, &charged_total),
              "observe retained later public release charge");
        {
            const NativeDriverCallScope native;
            require(
                cuEventQuery(late_pending) == CUDA_ERROR_NOT_READY,
                "later asynchronous free remains behind actual work not covered by the old proof");
        }
        require(charged_total == baseline_total && baseline_total == 8388608 &&
                    charged_free + late_bytes == baseline_free &&
                    cooperative_submitted_release_count(context) == 1,
                "earlier Runtime recovery cannot retire a later public release generation");
        check(cudaDeviceSynchronize(),
              "subsequent ordinary public Runtime completion drains the same queue");
        check(cuMemGetInfo(&charged_free, &charged_total), "observe exact later quota restoration");
        require(charged_free == baseline_free && charged_total == baseline_total &&
                    cooperative_submitted_release_count(context) == 0,
                "only subsequent completion retires the exact later submitted public release");
    }
    require(!invoke_runtime_context_completion_original(
                [&](const auto& original) {
                    return router.recover_current_native_epoch(gate, original);
                },
                [&] {
                    ++completions;
                    return synchronize();
                },
                quarantine) &&
                completions == 1,
            "healthy frontend declines before another Runtime SDK original");
    check(owner->query_event(historical), "historical event survives Runtime recovery");
    duration = 0;
    issue_original();
    require(
        owner->native_epoch_snapshot()->grant.execution.task_id != before->grant.execution.task_id,
        "new Runtime original cannot reuse the failed immutable grant");
    check(router.native_graph_execution(executable, actual_stream, is_ptds,
                                        [&] { return cuGraphLaunch(executable, actual_stream); }),
          "same original whole graph replays after borrowed Runtime recovery");
    check(owner->synchronize_context(), "complete subsequent original and graph order");
    require(originals == 2 && completions == 1 && quarantines == 0,
            "recovery never replays originals or publishes false global quarantine");
    std::array<unsigned int, 3> output{};
    {
        const NativeDriverCallScope native;
        check(cudaMemcpy(output.data(), allocation, sizeof(output), cudaMemcpyDeviceToHost),
              "actual Runtime recovery guarded output");
    }
    require(output ==
                std::array<unsigned int, 3>{0x12345678U, uses_process_owner ? 4U : 3U, 0x76543210U},
            "old/new/whole-graph Runtime originals execute once with intact guards");
    check(router.close_context(context, [] { return CUDA_SUCCESS; }),
          "retire healthy Runtime recovery frontend and identities");
    owner.reset();
    {
        const NativeDriverCallScope native;
        check(cuGraphExecDestroy(executable), "destroy original recovery graph executable");
        check(cuGraphDestroy(graph), "destroy original recovery graph");
        check(cuEventDestroy(pending), "destroy independent actual pending observer");
        check(cuEventDestroy(historical), "destroy original historical event");
        if (stream) check(cudaStreamDestroy(stream), "destroy actual recovery blocking stream");
        if (uses_process_owner) {
            check(cuEventDestroy(late_pending), "destroy actual later pending observer");
            check(cuStreamDestroy(late_stream), "destroy actual public late-release stream");
        }
        check(cudaFree(allocation), "retire actual recovery application memory");
        if (!uses_process_owner)
            check(cudaDeviceReset(), "retire fresh actual Runtime recovery lifetime");
    }
    if (uses_process_owner)
        check(cudaDeviceReset(), "actual public reset retires the unified process lifetime");
    std::cout << (uses_process_owner ? "process_epoch_recovery" : "runtime_epoch_recovery")
              << " mode=" << mode << " cached=" << is_cached
              << " originals=2 completion=1 driver_sync=0 tls=1 seed=" << should_seed_error
              << " graph=1 query_replay=0 guards=1 cleanup=1";
    if (uses_process_owner) std::cout << " same_owner=1 late_prefix=1 later_completion=1 quota=1";
    std::cout << '\n';
}

void check_process_epoch_capture(unsigned int mode, bool is_native, bool is_cold = false) {
    std::optional<NativeDriverCallScope> native_baseline;
    if (is_native) native_baseline.emplace();
    DriverDispatch driver;
    require(driver.initialize(), "resolve epoch capture SDK dispatch");
    const bool is_ptds = mode == 1;
    const auto select = [is_native]<typename Function>(const char* name) {
        return is_native ? original_runtime_symbol<Function>(name) : runtime_symbol<Function>(name);
    };
    const auto set_device = select.operator()<decltype(&cudaSetDevice)>("cudaSetDevice");
    const auto allocate = select.operator()<decltype(&cudaMalloc)>("cudaMalloc");
    const auto release = select.operator()<decltype(&cudaFree)>("cudaFree");
    const auto copy = select.operator()<decltype(&cudaMemcpy)>("cudaMemcpy");
    const auto create =
        select.operator()<decltype(&cudaStreamCreateWithFlags)>("cudaStreamCreateWithFlags");
    const auto destroy = select.operator()<decltype(&cudaStreamDestroy)>("cudaStreamDestroy");
    const auto launch = select.operator()<decltype(&cudaLaunchKernel)>(
        is_ptds ? "cudaLaunchKernel_ptsz" : "cudaLaunchKernel");
    const auto begin = select.operator()<decltype(&cudaStreamBeginCapture)>(
        is_ptds ? "cudaStreamBeginCapture_ptsz" : "cudaStreamBeginCapture");
    const auto end = select.operator()<decltype(&cudaStreamEndCapture)>(
        is_ptds ? "cudaStreamEndCapture_ptsz" : "cudaStreamEndCapture");
    const auto instantiate = select.operator()<decltype(&cudaGraphInstantiateWithFlags)>(
        "cudaGraphInstantiateWithFlags");
    const auto graph_destroy = select.operator()<decltype(&cudaGraphDestroy)>("cudaGraphDestroy");
    const auto replay = select.operator()<decltype(&cudaGraphLaunch)>(
        is_ptds ? "cudaGraphLaunch_ptsz" : "cudaGraphLaunch");
    const auto executable_destroy =
        select.operator()<decltype(&cudaGraphExecDestroy)>("cudaGraphExecDestroy");
    const auto synchronize =
        select.operator()<decltype(&cudaDeviceSynchronize)>("cudaDeviceSynchronize");
    const auto reset = select.operator()<decltype(&cudaDeviceReset)>("cudaDeviceReset");
    const auto sdk_record = original_runtime_symbol<decltype(&cudaEventRecord)>(
        is_ptds ? "cudaEventRecord_ptsz" : "cudaEventRecord");
    const auto sdk_event_create =
        original_runtime_symbol<decltype(&cudaEventCreateWithFlags)>("cudaEventCreateWithFlags");
    const auto sdk_event_query =
        original_runtime_symbol<decltype(&cudaEventQuery)>("cudaEventQuery");
    const auto sdk_event_destroy =
        original_runtime_symbol<decltype(&cudaEventDestroy)>("cudaEventDestroy");
    check(set_device(0), "select epoch capture device");
    check(release(nullptr), "initialize epoch capture primary context");
    CUcontext context{};
    check(driver.context_get_current(&context), "observe epoch capture original context");
    auto& gate = cooperative_process_call_gate();
    DriverCooperativeRouter* router{};
    std::shared_ptr<CudaCooperativeFrontend> owner;
    if (!is_native) {
        const ProcessCallGate::Call call(gate);
        router = cooperative_source_router();
        require(router && router->has_native_epoch_policy(), "enabled actual epoch capture owner");
        if (is_cold) {
            check(router->existing_frontend(context, owner), "CPU-only cold owner observation");
            require(!owner, "ordinary public operations start with no privately prepared actor");
        } else {
            check(router->frontend(context, owner), "prepare actual epoch capture frontend");
        }
    }
    const std::array<unsigned int, 3> initial{0x12345678U, 0, 0x76543210U};
    void* memory{};
    check(allocate(&memory, sizeof(initial)), "allocate epoch capture guarded output");
    check(copy(memory, initial.data(), sizeof(initial), cudaMemcpyHostToDevice),
          "initialize epoch capture guards");
    cudaStream_t producer{}, captured{}, consumer{};
    if (mode == 2) check(create(&producer, cudaStreamDefault), "create blocking epoch producer");
    check(create(&captured, cudaStreamNonBlocking), "create explicit original capture stream");
    check(create(&consumer, cudaStreamNonBlocking), "create original whole-graph consumer");
    auto* counter = static_cast<unsigned int*>(memory) + 1;
    unsigned long long duration_nanoseconds = 300'000'000;
    void* arguments[]{&counter, &duration_nanoseconds};
    require(!gate.is_entered_by_current_thread(), "fixture must not admit public launch calls");
    check(launch(glimmer_runtime_pending_kernel(), dim3{1}, dim3{1}, arguments, 0, producer),
          "public first original epoch capture predecessor");
    if (!is_native && is_cold) {
        const ProcessCallGate::Call call(gate);
        check(router->existing_frontend(context, owner),
              "CPU-only public-created owner observation");
        require(owner != nullptr,
                "ordinary public calls create their actual frontend without a seam");
    }
    const auto first = owner ? owner->native_epoch_snapshot() : std::nullopt;
    check(launch(glimmer_runtime_pending_kernel(), dim3{1}, dim3{1}, arguments, 0, producer),
          "public second original epoch capture predecessor");
    if (owner) {
        const auto second = owner->native_epoch_snapshot();
        require(first && first->execution && first->execution->issued_calls == 1 && second &&
                    second->execution && second->execution->issued_calls == 2 &&
                    second->execution->has_ownership &&
                    first->grant.execution.task_id == second->grant.execution.task_id,
                "public calls independently admit two originals under one immutable grant");
    }
    cudaEvent_t observer{};
    {
        const NativeDriverCallScope native;
        check(sdk_event_create(&observer, cudaEventDisableTiming), "independent pending observer");
        check(sdk_record(observer, producer), "independent SDK trailing original record");
        require(sdk_event_query(observer) == cudaErrorNotReady,
                "actual predecessors remain pending before public capture admission");
    }
    require(!gate.is_entered_by_current_thread(), "fixture must not admit public capture root");
    check(begin(captured, cudaStreamCaptureModeGlobal), "public capture admits pending epoch");
    if (owner) {
        const ProcessCallGate::Call call(gate);
        const auto closed = owner->native_epoch_snapshot();
        require(closed && closed->phase == CudaNativeEpochPhase::kComplete &&
                    closed->status == CUDA_SUCCESS && closed->execution &&
                    !closed->execution->has_ownership && closed->execution->issued_calls == 2 &&
                    closed->execution->completed_calls == 2,
                "capture admission physically retires both originals without fixture completion");
        require(gate.is_native_only_on_current_thread(), "public capture classifies later calls");
    }
    duration_nanoseconds = 0;
    check(launch(glimmer_runtime_pending_kernel(), dim3{1}, dim3{1}, arguments, 0, captured),
          "public source kernel constructs original graph without executing");
    cudaGraph_t graph{};
    check(end(captured, &graph), "public end capture reconciles original graph");
    CUcontext after{};
    check(driver.context_get_current(&after), "observe original caller after capture transitions");
    require(after == context && !is_inside_runtime_call() &&
                !runtime_cooperative_launch_is_pending() &&
                (is_native || !is_inside_driver_call()) && !gate.is_entered_by_current_thread(),
            "capture returns preserve actual context and balance caller admission/Runtime scopes");
    if (owner) {
        const ProcessCallGate::Call call(gate);
        require(!gate.is_native_only_on_current_thread(),
                "public capture end restores normal calls");
        const auto closed = owner->native_epoch_snapshot();
        require(closed && closed->execution && closed->execution->issued_calls == 2 &&
                    !closed->execution->has_ownership,
                "captured construction neither reissues nor acquires original execution");
    }
    const auto nodes = original_runtime_symbol<decltype(&cudaGraphGetNodes)>("cudaGraphGetNodes");
    const auto parameters = original_runtime_symbol<decltype(&cudaGraphKernelNodeGetParams)>(
        "cudaGraphKernelNodeGetParams");
    std::size_t count{};
    cudaGraphNode_t node{};
    cudaKernelNodeParams params{};
    {
        const NativeDriverCallScope native;
        check(nodes(graph, nullptr, &count), "inspect original captured node count");
        require(count == 1, "public capture keeps exactly one original kernel node");
        check(nodes(graph, &node, &count), "inspect actual captured node");
        check(parameters(node, &params), "inspect actual SDK captured parameters");
    }
    require(params.func == glimmer_runtime_pending_kernel() && params.gridDim.x == 1 &&
                params.gridDim.y == 1 && params.gridDim.z == 1 && params.blockDim.x == 1 &&
                params.blockDim.y == 1 && params.blockDim.z == 1 && params.sharedMemBytes == 0,
            "captured original SDK function and dimensions remain unchanged");
    // Native capture need not drain unrelated producer work. Complete only after
    // both capture transitions; public admission itself was checked above.
    check(synchronize(), "public post-capture completion before guarded readback");
    std::array<unsigned int, 3> output{};
    check(copy(output.data(), memory, sizeof(output), cudaMemcpyDeviceToHost),
          "read epoch predecessors without graph execution");
    require(output == std::array<unsigned int, 3>{0x12345678U, 2, 0x76543210U},
            "capture construction does not execute original graph kernel");
    cudaGraphExec_t executable{};
    check(instantiate(&executable, graph, 0), "public instantiate original captured graph");
    check(graph_destroy(graph), "public destroy source before whole original replay");
    check(replay(executable, consumer), "public replay original whole captured graph");
    check(synchronize(), "public complete whole original replay");
    check(copy(output.data(), memory, sizeof(output), cudaMemcpyDeviceToHost),
          "read exact whole original graph output");
    require(output == std::array<unsigned int, 3>{0x12345678U, 3, 0x76543210U},
            "two normal originals and one graph replay preserve guards and exact-once output");
    check(executable_destroy(executable), "public destroy original captured executable");
    {
        const NativeDriverCallScope native;
        check(sdk_event_destroy(observer), "destroy checked independent observer");
    }
    check(destroy(consumer), "public destroy graph consumer");
    check(destroy(captured), "public destroy original capture stream");
    if (producer) check(destroy(producer), "public destroy blocking producer");
    check(release(memory), "public release guarded epoch capture allocation");
    check(reset(), "public reset final epoch capture lifetime");
    if (owner) {
        const auto closed = owner->native_epoch_snapshot();
        require(closed && closed->execution && !closed->execution->has_ownership &&
                    closed->created_events == closed->destroyed_events,
                "epoch capture lifetime retains only balanced completed CPU history");
        check(cudaFree(nullptr), "public initialize quota inspection lifetime");
        std::size_t free_bytes{}, total{};
        check(cudaMemGetInfo(&free_bytes, &total), "public inspect checked reset quota");
        require(total == 8388608 && free_bytes == total, "all application/private quota restored");
        check(cudaDeviceReset(), "public reset empty inspection lifetime");
    }
    std::cout << (is_cold ? "process_cold_epoch_capture" : "process_epoch_capture")
              << " mode=" << mode << " native=" << is_native
              << " pending=1 independent_admission=" << !is_native
              << " capture=1 original_node=1 construction_output=2 replay_output=3 guards=1"
                 " cleanup=1 quota_checked="
              << !is_native << '\n';
}

class CaptureOriginalHold final {
   public:
    void wait() {
        permission_.wait();
    }
    void release() noexcept {
        if (!has_released_.exchange(true)) permission_.count_down();
    }

   private:
    std::latch permission_{1};
    std::atomic<bool> has_released_{};
};
struct CaptureOriginalRelease final {
    CaptureOriginalHold& hold;
    ~CaptureOriginalRelease() {
        hold.release();
    }
};

void check_cached_capture_overlap(unsigned int mode) {
    DriverDispatch driver;
    require(driver.initialize(), "resolve actual overlapping capture dispatch");
    const bool is_ptds = mode == 1;
    const auto launch_name = is_ptds ? "cudaLaunchKernel_ptsz" : "cudaLaunchKernel";
    const auto launch = runtime_symbol<decltype(&cudaLaunchKernel)>(launch_name);
    const auto sdk_launch = original_runtime_symbol<decltype(&cudaLaunchKernel)>(launch_name);
    const auto begin = runtime_symbol<decltype(&cudaStreamBeginCapture)>(
        is_ptds ? "cudaStreamBeginCapture_ptsz" : "cudaStreamBeginCapture");
    const auto end = runtime_symbol<decltype(&cudaStreamEndCapture)>(
        is_ptds ? "cudaStreamEndCapture_ptsz" : "cudaStreamEndCapture");
    const auto sdk_device = original_runtime_symbol<decltype(&cudaSetDevice)>("cudaSetDevice");
    const auto sdk_free = original_runtime_symbol<decltype(&cudaFree)>("cudaFree");
    const auto sdk_record = original_runtime_symbol<decltype(&cudaEventRecord)>(
        is_ptds ? "cudaEventRecord_ptsz" : "cudaEventRecord");
    const auto sdk_query = original_runtime_symbol<decltype(&cudaEventQuery)>("cudaEventQuery");
    const auto sdk_peek =
        original_runtime_symbol<decltype(&cudaPeekAtLastError)>("cudaPeekAtLastError");
    const auto sdk_get = original_runtime_symbol<decltype(&cudaGetLastError)>("cudaGetLastError");
    check(cudaSetDevice(0), "select overlapping capture device");
    check(cudaFree(nullptr), "initialize actual overlapping primary context");
    CUcontext context{};
    check(driver.context_get_current(&context), "observe overlapping original caller context");
    auto& gate = cooperative_process_call_gate();
    DriverCooperativeRouter* router{};
    std::shared_ptr<CudaCooperativeFrontend> owner;
    {
        const ProcessCallGate::Call call(gate);
        router = cooperative_source_router();
        require(router && router->has_native_epoch_policy(), "actual overlapping epoch policy");
        check(router->existing_frontend(context, owner), "observe cold overlapping actor");
        require(!owner, "overlap fixture must not privately construct a frontend");
    }
    const std::array<unsigned int, 3> initial{0x12345678U, 0, 0x76543210U};
    void* memory{};
    check(cudaMalloc(&memory, sizeof(initial)), "public overlapping guarded allocation");
    check(cudaMemcpy(memory, initial.data(), sizeof(initial), cudaMemcpyHostToDevice),
          "public overlapping guard initialization");
    cudaStream_t producer{}, captured{}, consumer{};
    if (mode == 2) check(cudaStreamCreate(&producer), "public overlapping blocking producer");
    check(cudaStreamCreateWithFlags(&captured, cudaStreamNonBlocking),
          "public overlapping explicit capture stream");
    check(cudaStreamCreateWithFlags(&consumer, cudaStreamNonBlocking),
          "public overlapping whole-graph consumer");
    auto* counter = static_cast<unsigned int*>(memory) + 1;
    unsigned long long duration_nanoseconds = 300'000'000;
    void* arguments[]{&counter, &duration_nanoseconds};
    check(launch(glimmer_runtime_pending_kernel(), dim3{1}, dim3{1}, arguments, 0, producer),
          "public overlapping original creates actual epoch");
    {
        const ProcessCallGate::Call call(gate);
        check(router->existing_frontend(context, owner), "observe ordinary public-created actor");
        require(owner != nullptr, "actual public-created actor is required for overlap proof");
    }
    cudaFunction_t function{};
    cudaEvent_t observer{};
    {
        const NativeDriverCallScope native;
        check(original_runtime_symbol<decltype(&cudaGetFuncBySymbol)>("cudaGetFuncBySymbol")(
                  &function, glimmer_runtime_pending_kernel()),
              "observe actual overlapping cached SDK function");
        check(original_runtime_symbol<decltype(&cudaEventCreateWithFlags)>(
                  "cudaEventCreateWithFlags")(&observer, cudaEventDisableTiming),
              "create independent actual cached original observer");
    }
    CaptureOriginalHold hold;
    std::atomic<bool> has_entered{}, has_original_returned{}, has_capture_returned{};
    unsigned int original_calls{}, quarantines{};
    bool has_pending_original{}, has_balanced_original{}, has_closed_capture{};
    std::exception_ptr original_error, capture_error;
    cudaGraph_t graph{};
    const auto bind = [&] {
        const NativeDriverCallScope native;
        check(sdk_device(0), "bind actual SDK thread device");
        check(sdk_free(nullptr), "bind actual SDK thread primary context");
        CUcontext actual{};
        check(driver.context_get_current(&actual), "observe actual SDK thread binding");
        require(actual == context, "overlapping caller uses the actual shared application context");
    };
    std::jthread original_thread([&] {
        try {
            bind();
            {
                const NativeDriverCallScope native;
                (void)sdk_get();
            }
            {
                const ProcessCallGate::Call call(gate);
                RuntimeCooperativeLaunch plan{function, std::nullopt, {1, 1, 1}, {1, 1, 1}};
                plan.context = context;
                check(invoke_runtime_cooperative_launch_original(
                          std::move(plan), is_ptds,
                          [&] {
                              has_entered = true;
                              hold.wait();
                              ++original_calls;
                              return sdk_launch(glimmer_runtime_pending_kernel(), dim3{1}, dim3{1},
                                                arguments, 0, producer);
                          },
                          [&] {
                              ++quarantines;
                              router->quarantine_normal_launch(context, CUDA_ERROR_UNKNOWN);
                          }),
                      "actual cached normal original returns without an execution provider");
                require(std::array{sdk_peek(), sdk_get(), sdk_peek()} ==
                            std::array{cudaSuccess, cudaSuccess, cudaSuccess},
                        "cached normal original preserves clean actual caller TLS");
                {
                    const NativeDriverCallScope native;
                    check(sdk_record(observer, producer), "actual cached trailing original record");
                    has_pending_original = sdk_query(observer) == cudaErrorNotReady;
                }
                require(has_pending_original,
                        "cached original is genuinely pending before Call exit");
            }
            CUcontext after{};
            check(driver.context_get_current(&after),
                  "observe cached original caller after return");
            has_balanced_original =
                after == context && !is_inside_runtime_call() && !is_inside_driver_call() &&
                !runtime_cooperative_launch_is_pending() && !gate.is_entered_by_current_thread();
        } catch (...) {
            original_error = std::current_exception();
        }
        has_original_returned = true;
    });
    std::jthread capture_thread;
    // Constructed after both thread owners: release before either destructor joins.
    const CaptureOriginalRelease release_on_exit{hold};
    const auto wait_until = [](const auto& ready) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{10};
        while (!ready()) {
            if (std::chrono::steady_clock::now() >= deadline) return false;
            std::this_thread::yield();
        }
        return true;
    };
    require(wait_until([&] { return has_entered.load() || has_original_returned.load(); }),
            "cached normal caller must reach its original boundary");
    if (has_entered) {
        capture_thread = std::jthread([&] {
            try {
                bind();
                check(begin(captured, cudaStreamCaptureModeGlobal),
                      "actual public capture waits for the existing normal caller");
                {
                    const ProcessCallGate::Call call(gate);
                    const auto closed = owner->native_epoch_snapshot();
                    has_closed_capture =
                        closed && closed->execution &&
                        closed->phase == CudaNativeEpochPhase::kComplete &&
                        closed->status == CUDA_SUCCESS && !closed->execution->has_ownership &&
                        closed->execution->issued_calls == closed->execution->completed_calls &&
                        gate.is_native_only_on_current_thread();
                }
                require(has_closed_capture,
                        "public overlap capture physically retires actual originals");
                unsigned long long captured_duration{};
                void* captured_arguments[]{&counter, &captured_duration};
                check(launch(glimmer_runtime_pending_kernel(), dim3{1}, dim3{1}, captured_arguments,
                             0, captured),
                      "actual public overlapping original graph construction");
                check(end(captured, &graph), "actual public capture ends on its original caller");
                CUcontext after{};
                check(driver.context_get_current(&after),
                      "observe public capture caller after end");
                require(after == context && !is_inside_driver_call() && !is_inside_runtime_call() &&
                            !gate.is_entered_by_current_thread(),
                        "public overlap transitions balance actual caller context and scopes");
            } catch (...) {
                capture_error = std::current_exception();
            }
            has_capture_returned = true;
        });
    }
    bool has_waiting_transition = false;
    (void)wait_until([&] {
        const auto observed = gate.snapshot();
        has_waiting_transition = observed.owns_transition && observed.waiting_transitions == 1 &&
                                 observed.active_calls == 1 && observed.admitted_callers == 2;
        return has_waiting_transition || has_capture_returned.load() ||
               has_original_returned.load();
    });
    cudaStreamCaptureStatus actual_status = cudaStreamCaptureStatusInvalidated;
    if (has_waiting_transition) {
        const NativeDriverCallScope native;
        check(original_runtime_symbol<decltype(&cudaStreamIsCapturing)>(
                  is_ptds ? "cudaStreamIsCapturing_ptsz" : "cudaStreamIsCapturing")(captured,
                                                                                    &actual_status),
              "independent SDK observation before releasing the waiting original");
    }
    hold.release();
    original_thread.join();
    if (capture_thread.joinable()) capture_thread.join();
    if (original_error) std::rethrow_exception(original_error);
    if (capture_error) std::rethrow_exception(capture_error);
    require(has_waiting_transition && actual_status == cudaStreamCaptureStatusNone &&
                original_calls == 1 && quarantines == 0 && has_pending_original &&
                has_balanced_original && has_closed_capture,
            "observed waiting transition cannot issue capture or duplicate the cached original");
    const auto balanced = gate.snapshot();
    require(balanced.active_calls == 0 && balanced.admitted_callers == 0 &&
                balanced.waiting_transitions == 0 && !balanced.owns_transition &&
                !balanced.is_native_only,
            "overlapping original and public capture finish with balanced process admission");
    std::size_t count{};
    cudaGraphNode_t node{};
    cudaKernelNodeParams parameters{};
    {
        const NativeDriverCallScope native;
        const auto nodes =
            original_runtime_symbol<decltype(&cudaGraphGetNodes)>("cudaGraphGetNodes");
        check(nodes(graph, nullptr, &count), "observe original overlapping graph node count");
        require(count == 1, "overlapping capture contains only its original SDK kernel");
        check(nodes(graph, &node, &count), "observe actual overlapping original graph node");
        check(original_runtime_symbol<decltype(&cudaGraphKernelNodeGetParams)>(
                  "cudaGraphKernelNodeGetParams")(node, &parameters),
              "observe actual overlapping captured SDK parameters");
    }
    require(parameters.func == glimmer_runtime_pending_kernel() && parameters.gridDim.x == 1 &&
                parameters.gridDim.y == 1 && parameters.gridDim.z == 1 &&
                parameters.blockDim.x == 1 && parameters.blockDim.y == 1 &&
                parameters.blockDim.z == 1 && parameters.sharedMemBytes == 0,
            "overlapping capture preserves original SDK kernel identity and dimensions");
    check(cudaDeviceSynchronize(), "public complete overlapping construction without replay");
    std::array<unsigned int, 3> output{};
    check(cudaMemcpy(output.data(), memory, sizeof(output), cudaMemcpyDeviceToHost),
          "public overlapping pre-replay guarded readback");
    require(output == std::array<unsigned int, 3>{0x12345678U, 2, 0x76543210U},
            "two actual originals execute exactly once and graph construction does not execute");
    cudaGraphExec_t executable{};
    check(cudaGraphInstantiateWithFlags(&executable, graph, 0),
          "public overlapping graph instantiate");
    check(cudaGraphDestroy(graph),
          "public destroy overlapping source before original whole replay");
    check(runtime_symbol<decltype(&cudaGraphLaunch)>(
              is_ptds ? "cudaGraphLaunch_ptsz" : "cudaGraphLaunch")(executable, consumer),
          "public overlapping original whole graph replay");
    check(cudaDeviceSynchronize(), "public complete overlapping original whole replay");
    check(cudaMemcpy(output.data(), memory, sizeof(output), cudaMemcpyDeviceToHost),
          "public overlapping final guarded readback");
    require(output == std::array<unsigned int, 3>{0x12345678U, 3, 0x76543210U},
            "cached original overlap preserves actual whole-graph output and guards");
    check(cudaGraphExecDestroy(executable), "public overlapping executable destruction");
    {
        const NativeDriverCallScope native;
        check(original_runtime_symbol<decltype(&cudaEventDestroy)>("cudaEventDestroy")(observer),
              "checked independent overlapping observer destruction");
    }
    check(cudaStreamDestroy(consumer), "public overlapping consumer destruction");
    check(cudaStreamDestroy(captured), "public overlapping source stream destruction");
    if (producer)
        check(cudaStreamDestroy(producer), "public overlapping blocking stream destruction");
    check(cudaFree(memory), "public overlapping guarded allocation release");
    check(cudaDeviceReset(), "public reset overlapping original lifetime");
    const auto retired = owner->native_epoch_snapshot();
    require(retired && retired->execution && !retired->execution->has_ownership &&
                retired->created_events == retired->destroyed_events,
            "overlapping retirement retains only balanced historical CPU ownership");
    check(cudaFree(nullptr), "public initialize overlapping quota observation");
    std::size_t available{}, total{};
    check(cudaMemGetInfo(&available, &total), "public overlapping reset quota observation");
    require(total == 8388608 && available == total,
            "overlapping public reset restores exact quota");
    check(cudaDeviceReset(), "public reset empty overlapping observation lifetime");
    std::cout << "cached_capture_overlap mode=" << mode
              << " cold=1 transition_wait=1 sdk_uncaptured=1 cached_originals=1 provider=0"
                 " pending=1 tls=1 context=1 physical_stop=1 nodes=1 construction_output=2"
                 " replay_output=3 guards=1 quota=1 cleanup=1\n";
}

void check_source_event_graph(bool is_ptds, bool is_native, bool should_check_rejection) {
    DriverDispatch driver;
    require(driver.initialize(), "resolve source graph original dispatch");
    std::optional<NativeDriverCallScope> native_reference;
    if (is_native) native_reference.emplace();
    const auto symbol = [is_native]<typename Function>(const char* name) {
        return is_native ? original_runtime_symbol<Function>(name) : runtime_symbol<Function>(name);
    };
    const auto allocate = symbol.operator()<decltype(&cudaMalloc)>("cudaMalloc");
    const auto release = symbol.operator()<decltype(&cudaFree)>("cudaFree");
    const auto copy = symbol.operator()<decltype(&cudaMemcpy)>("cudaMemcpy");
    const auto synchronize =
        symbol.operator()<decltype(&cudaDeviceSynchronize)>("cudaDeviceSynchronize");
    const auto begin = symbol.operator()<decltype(&cudaStreamBeginCapture)>(
        is_ptds ? "cudaStreamBeginCapture_ptsz" : "cudaStreamBeginCapture");
    const auto end = symbol.operator()<decltype(&cudaStreamEndCapture)>(
        is_ptds ? "cudaStreamEndCapture_ptsz" : "cudaStreamEndCapture");
    const auto launch = symbol.operator()<decltype(&cudaGraphLaunch)>(
        is_ptds ? "cudaGraphLaunch_ptsz" : "cudaGraphLaunch");
    const auto record = symbol.operator()<decltype(&cudaEventRecord)>(
        is_ptds ? "cudaEventRecord_ptsz" : "cudaEventRecord");
    const auto wait = symbol.operator()<decltype(&cudaStreamWaitEvent)>(
        is_ptds ? "cudaStreamWaitEvent_ptsz" : "cudaStreamWaitEvent");
    check(symbol.operator()<decltype(&cudaSetDevice)>("cudaSetDevice")(0), "source graph device");
    check(release(nullptr), "source graph primary binding");
    CUcontext context{};
    check(driver.context_get_current(&context), "source graph actual context");
    auto& gate = cooperative_process_call_gate();
    DriverCooperativeRouter* router{};
    if (!is_native) {
        const ProcessCallGate::Call call(gate);
        router = cooperative_source_router();
        require(router, "source graph actual process router");
    }
    std::array<unsigned int, 3> guards{0x1234abcdU, 0, 0xdcba4321U};
    void* memory{};
    check(allocate(&memory, sizeof(guards)), "source graph guarded allocation");
    check(copy(memory, guards.data(), sizeof(guards), cudaMemcpyHostToDevice),
          "source graph initialize");
    auto* counter = static_cast<unsigned int*>(memory) + 1;
    unsigned int* marker{};
    CUdeviceptr marker_device{};
    check(cuMemHostAlloc(reinterpret_cast<void**>(&marker), sizeof(*marker),
                         CU_MEMHOSTALLOC_DEVICEMAP),
          "source graph mapped start marker");
    check(cuMemHostGetDevicePointer(&marker_device, marker, 0), "source graph marker address");
    auto* marker_argument = std::bit_cast<unsigned int*>(marker_device);
    cudaStream_t origin{}, consumer{}, joined{};
    const auto create =
        symbol.operator()<decltype(&cudaStreamCreateWithFlags)>("cudaStreamCreateWithFlags");
    check(create(&origin, cudaStreamNonBlocking), "source graph capture stream");
    check(create(&consumer, cudaStreamNonBlocking), "source graph independent stream");
    check(create(&joined, cudaStreamNonBlocking), "source graph wait destination");
    cudaEvent_t waited{}, recorded{};
    const auto event_create =
        symbol.operator()<decltype(&cudaEventCreateWithFlags)>("cudaEventCreateWithFlags");
    check(event_create(&waited, cudaEventDisableTiming), "source graph input event");
    check(event_create(&recorded, cudaEventDisableTiming), "source graph output event");
    check(record(waited, consumer), "source graph completed input record");
    check(synchronize(), "source graph initial input completion");
    cudaGraph_t source{};
    cudaGraphExec_t executable{};
    std::array<cudaGraphNode_t, 4> nodes{};
    check(symbol.operator()<decltype(&cudaGraphCreate)>("cudaGraphCreate")(&source, 0),
          "source graph create");
    check(symbol.operator()<decltype(&cudaGraphAddEventWaitNode)>("cudaGraphAddEventWaitNode")(
              &nodes[0], source, nullptr, 0, waited),
          "source graph wait node");
    cudaKernelNodeParams parameters{};
    parameters.func = const_cast<void*>(glimmer_runtime_capture_kernel());
    parameters.gridDim = parameters.blockDim = dim3{1};
    void* marker_arguments[]{&marker_argument};
    parameters.kernelParams = marker_arguments;
    const auto add_kernel =
        symbol.operator()<decltype(&cudaGraphAddKernelNode)>("cudaGraphAddKernelNode");
    check(add_kernel(&nodes[1], source, &nodes[0], 1, &parameters),
          "source graph original start kernel");
    unsigned long long duration_nanoseconds = 600'000'000;
    void* arguments[]{&counter, &duration_nanoseconds};
    parameters.func = const_cast<void*>(glimmer_runtime_pending_kernel());
    parameters.kernelParams = arguments;
    check(add_kernel(&nodes[2], source, &nodes[1], 1, &parameters),
          "source graph original pending kernel");
    check(symbol.operator()<decltype(&cudaGraphAddEventRecordNode)>("cudaGraphAddEventRecordNode")(
              &nodes[3], source, &nodes[2], 1, recorded),
          "source graph record node");
    check(symbol.operator()<decltype(&cudaGraphInstantiateWithFlags)>(
              "cudaGraphInstantiateWithFlags")(&executable, source, 0),
          "source graph original instantiate");
    {
        const NativeDriverCallScope native;
        std::size_t count{};
        cudaKernelNodeParams actual{};
        check(original_runtime_symbol<decltype(&cudaGraphGetNodes)>("cudaGraphGetNodes")(
                  source, nullptr, &count),
              "source graph independently inspect node count");
        check(original_runtime_symbol<decltype(&cudaGraphKernelNodeGetParams)>(
                  "cudaGraphKernelNodeGetParams")(nodes[2], &actual),
              "source graph independently inspect original business kernel");
        require(count == 4 && actual.func == parameters.func && actual.gridDim.x == 1 &&
                    actual.blockDim.x == 1 && actual.sharedMemBytes == 0,
                "source graph retains original nodes and business kernel ABI");
    }
    const auto destroy_graph = symbol.operator()<decltype(&cudaGraphDestroy)>("cudaGraphDestroy");
    check(destroy_graph(source), "source graph retire original source before replay");
    const auto sdk_get = original_runtime_symbol<decltype(&cudaGetLastError)>("cudaGetLastError");
    const auto sdk_peek =
        original_runtime_symbol<decltype(&cudaPeekAtLastError)>("cudaPeekAtLastError");
    const auto is_pending = [&] {
        const NativeDriverCallScope native;
        const auto exchange =
            original_runtime_symbol<decltype(&cudaThreadExchangeStreamCaptureMode)>(
                "cudaThreadExchangeStreamCaptureMode");
        auto saved = cudaStreamCaptureModeRelaxed;
        check(exchange(&saved), "source graph independent observer mode");
        const auto queried =
            original_runtime_symbol<decltype(&cudaEventQuery)>("cudaEventQuery")(recorded);
        check(exchange(&saved), "source graph restore observer mode");
        require(saved == cudaStreamCaptureModeRelaxed &&
                    (queried == cudaSuccess || queried == cudaErrorNotReady),
                "source graph checked independent event status");
        return queried == cudaErrorNotReady;
    };
    for (unsigned int cycle = 0; cycle < 2; ++cycle) {
        std::atomic_ref<unsigned int>(*marker).store(0);
        check(begin(origin, cudaStreamCaptureModeGlobal),
              "source graph begin actual GLOBAL capture");
        check(record(recorded, origin), "source graph supersede earlier captured record");
        const auto before = is_native ? CudaPausedNativeStats{} : router->native_execution_stats();
        (void)sdk_get();
        check(launch(executable, consumer), "source graph original independent replay");
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{10};
        while (std::atomic_ref<unsigned int>(*marker).load() != 1) {
            require(std::chrono::steady_clock::now() < deadline, "source graph actual GPU start");
            std::this_thread::yield();
        }
        require(is_pending(), "source graph original record is independently pending");
        require(std::array{sdk_peek(), sdk_get(), sdk_peek()} ==
                    std::array{cudaSuccess, cudaSuccess, cudaSuccess},
                "source graph SDK error TLS");
        CUcontext actual{};
        check(driver.context_get_current(&actual), "source graph original return context");
        require(actual == context && !gate.is_entered_by_current_thread() &&
                    !is_inside_runtime_call() && (is_native || !is_inside_driver_call()),
                "source graph balanced original caller scopes");
        if (cycle == 1) {
            check(wait(joined, recorded, 0),
                  "source graph overwritten record wait cannot join capture");
            const NativeDriverCallScope native;
            cudaStreamCaptureStatus status{};
            check(original_runtime_symbol<decltype(&cudaStreamIsCapturing)>(
                      "cudaStreamIsCapturing")(joined, &status),
                  "source graph independently observe wait classification");
            require(status == cudaStreamCaptureStatusNone,
                    "source graph record clears capture provenance");
        }
        cudaGraph_t captured{};
        check(end(origin, &captured), "source graph end original capture");
        if (cycle == 0 && !is_native) {
            require(is_pending(), "source graph end cannot drain independent original work");
            const auto execution = router->native_execution_stats();
            require(gate.snapshot().is_native_only && execution.retained == 1 &&
                        execution.grants == before.grants + 1 &&
                        execution.completed == before.completed,
                    "source graph retains exact native physical ownership after capture end");
        }
        check(synchronize(), "source graph completion proves physical stop before normal routing");
        if (!is_native) {
            const auto execution = router->native_execution_stats();
            require(!gate.snapshot().is_native_only && execution.retained == 0 &&
                        execution.grants == before.grants + 1 + cycle &&
                        execution.completed == before.completed + 1 + cycle &&
                        execution.failed == before.failed,
                    "source graph exact graph and uncaptured wait grants complete without failure");
        }
        {
            const NativeDriverCallScope native;
            std::size_t count{};
            check(original_runtime_symbol<decltype(&cudaGraphGetNodes)>("cudaGraphGetNodes")(
                      captured, nullptr, &count),
                  "source graph no private capture nodes");
            require(count == 0,
                    "source graph record-only capture remains the original empty graph");
        }
        check(destroy_graph(captured), "source graph checked captured graph destruction");
    }
    if (should_check_rejection) {
        check(begin(origin, cudaStreamCaptureModeGlobal), "source graph rejected capture begins");
        const auto before = is_native ? CudaPausedNativeStats{} : router->native_execution_stats();
        (void)sdk_get();
        require(launch(executable, origin) == cudaErrorStreamCaptureUnsupported,
                "source graph preserves actual SDK captured-record graph rejection 900");
        require(sdk_peek() == cudaErrorStreamCaptureUnsupported &&
                    sdk_get() == cudaErrorStreamCaptureUnsupported && sdk_peek() == cudaSuccess,
                "source graph preserves original rejected SDK error TLS");
        cudaGraph_t rejected{};
        require(end(origin, &rejected) == cudaErrorStreamCaptureInvalidated && !rejected,
                "source graph preserves original invalidated end 901 and null output");
        require(sdk_get() == cudaErrorStreamCaptureInvalidated && sdk_peek() == cudaSuccess,
                "source graph preserves original end error TLS");
        check(synchronize(), "source graph rejected capture checked physical recovery");
        if (!is_native) {
            const auto after = router->native_execution_stats();
            require(!gate.snapshot().is_native_only && after.retained == 0 &&
                        after.grants == before.grants && after.failed == before.failed,
                    "source graph captured construction rejection has no execution grant");
        }
    }
    check(copy(guards.data(), memory, sizeof(guards), cudaMemcpyDeviceToHost),
          "source graph guarded readback");
    require(guards == std::array<unsigned int, 3>{0x1234abcdU, 2, 0xdcba4321U},
            "source graph each original business kernel executes exactly once");
    check(symbol.operator()<decltype(&cudaGraphExecDestroy)>("cudaGraphExecDestroy")(executable),
          "source graph executable cleanup");
    const auto destroy_event = symbol.operator()<decltype(&cudaEventDestroy)>("cudaEventDestroy");
    check(destroy_event(recorded), "source graph output cleanup");
    check(destroy_event(waited), "source graph input cleanup");
    const auto destroy_stream =
        symbol.operator()<decltype(&cudaStreamDestroy)>("cudaStreamDestroy");
    check(destroy_stream(joined), "source graph destination cleanup");
    check(destroy_stream(consumer), "source graph consumer cleanup");
    check(destroy_stream(origin), "source graph origin cleanup");
    check(cuMemFreeHost(marker), "source graph mapped marker cleanup");
    check(release(memory), "source graph allocation cleanup");
    const auto reset = symbol.operator()<decltype(&cudaDeviceReset)>("cudaDeviceReset");
    check(reset(), "source graph lifetime reset");
    if (!is_native) {
        check(release(nullptr), "source graph fresh quota observation");
        std::size_t available{}, total{};
        check(cudaMemGetInfo(&available, &total), "source graph reset quota");
        require(total == 8388608 && available == total, "source graph full quota restored");
        check(reset(), "source graph empty observation reset");
    }
    std::cout << "source_event_graph ptds=" << is_ptds << " native=" << is_native
              << " nodes=4 source_retired=1 pending=1 provenance=1 effects=2 tls=1 context=1 "
                 "cleanup=1 quota="
              << !is_native << " rejection=" << should_check_rejection << '\n';
}

void check_public_graph_dependency(unsigned int mode, bool is_native) {
    DriverDispatch driver;
    require(driver.initialize(), "resolve actual public graph dependency dispatch");
    std::optional<NativeDriverCallScope> native_reference;
    if (is_native) native_reference.emplace();
    const bool is_ptds = mode == 1;
    const auto symbol = [is_native]<typename Function>(const char* name) {
        return is_native ? original_runtime_symbol<Function>(name) : runtime_symbol<Function>(name);
    };
    const auto device = symbol.operator()<decltype(&cudaSetDevice)>("cudaSetDevice");
    const auto release = symbol.operator()<decltype(&cudaFree)>("cudaFree");
    const auto allocate = symbol.operator()<decltype(&cudaMalloc)>("cudaMalloc");
    const auto copy = symbol.operator()<decltype(&cudaMemcpy)>("cudaMemcpy");
    const auto synchronize =
        symbol.operator()<decltype(&cudaDeviceSynchronize)>("cudaDeviceSynchronize");
    const auto reset = symbol.operator()<decltype(&cudaDeviceReset)>("cudaDeviceReset");
    check(device(0), "select public graph dependency device");
    check(release(nullptr), "initialize actual graph dependency primary context");
    CUcontext context{};
    check(driver.context_get_current(&context), "observe actual graph dependency caller");
    auto& gate = cooperative_process_call_gate();
    DriverCooperativeRouter* router{};
    std::shared_ptr<CudaCooperativeFrontend> owner;
    if (!is_native) {
        const ProcessCallGate::Call call(gate);
        router = cooperative_source_router();
        require(router && router->has_native_epoch_policy(), "actual public graph epoch policy");
        check(router->existing_frontend(context, owner), "observe absent public graph actor");
        require(!owner, "graph dependency fixture cannot privately construct its actor");
    }
    void* producer_memory{};
    void* guarded_memory{};
    std::array<unsigned int, kResetElements> zeros{};
    const std::array<unsigned int, 4> initial{0x12345678U, 0, 0, 0x76543210U};
    check(allocate(&producer_memory, sizeof(zeros)), "public graph producer allocation");
    check(allocate(&guarded_memory, sizeof(initial)), "public graph guarded allocation");
    check(copy(producer_memory, zeros.data(), sizeof(zeros), cudaMemcpyHostToDevice),
          "public graph producer initialization");
    check(copy(guarded_memory, initial.data(), sizeof(initial), cudaMemcpyHostToDevice),
          "public graph output guard initialization");
    std::uint32_t* markers{};
    CUdeviceptr device_markers{};
    check(cuMemHostAlloc(reinterpret_cast<void**>(&markers), 8, CU_MEMHOSTALLOC_DEVICEMAP),
          "public graph mapped producer progress allocation");
    check(cuMemHostGetDevicePointer(&device_markers, markers, 0),
          "public graph mapped producer progress address");
    markers[0] = markers[1] = 0;
    int least{}, greatest{};
    check(cuCtxGetStreamPriorityRange(&least, &greatest), "actual graph stream priority range");
    require(least > greatest, "public graph test requires distinct actual priorities");
    cudaStream_t producer{}, consumer{};
    const auto create =
        symbol.operator()<decltype(&cudaStreamCreateWithPriority)>("cudaStreamCreateWithPriority");
    if (mode == 2) check(create(&producer, cudaStreamDefault, least), "public blocking producer");
    check(create(&consumer, cudaStreamNonBlocking, greatest), "public high graph consumer");
    const auto create_event =
        symbol.operator()<decltype(&cudaEventCreateWithFlags)>("cudaEventCreateWithFlags");
    cudaEvent_t waited{}, recorded{};
    check(create_event(&waited, cudaEventDisableTiming), "public graph input event creation");
    check(create_event(&recorded, cudaEventDisableTiming), "public graph output event creation");
    CUmodule module{};
    CUfunction low{};
    check(cuModuleLoadData(&module, kCooperativePtx), "public qualified producer module load");
    check(cuModuleGetFunction(&low, module, "cooperative"), "public qualified producer lookup");
    auto* counter = static_cast<unsigned int*>(guarded_memory) + 1;
    auto* premature_reads = counter + 1;
    auto* input = static_cast<unsigned int*>(producer_memory);
    unsigned int elements = kResetElements;
    void* graph_arguments[]{&counter, &premature_reads, &input, &elements};
    cudaKernelNodeParams parameters{};
    parameters.func = const_cast<void*>(glimmer_runtime_graph_dependency_kernel());
    parameters.gridDim = dim3{1};
    parameters.blockDim = dim3{1};
    parameters.kernelParams = graph_arguments;
    cudaGraph_t graph{};
    std::array<cudaGraphNode_t, 3> nodes{};
    check(symbol.operator()<decltype(&cudaGraphCreate)>("cudaGraphCreate")(&graph, 0),
          "public original dependency graph creation");
    check(symbol.operator()<decltype(&cudaGraphAddEventWaitNode)>("cudaGraphAddEventWaitNode")(
              &nodes[0], graph, nullptr, 0, waited),
          "public original graph wait node");
    check(symbol.operator()<decltype(&cudaGraphAddKernelNode)>("cudaGraphAddKernelNode")(
              &nodes[1], graph, &nodes[0], 1, &parameters),
          "public original graph dependency-read kernel");
    check(symbol.operator()<decltype(&cudaGraphAddEventRecordNode)>("cudaGraphAddEventRecordNode")(
              &nodes[2], graph, &nodes[1], 1, recorded),
          "public original graph trailing event record");
    {
        const NativeDriverCallScope native;
        std::size_t count{};
        cudaKernelNodeParams actual{};
        cudaEvent_t actual_wait{}, actual_record{};
        check(original_runtime_symbol<decltype(&cudaGraphGetNodes)>("cudaGraphGetNodes")(
                  graph, nullptr, &count),
              "inspect original public graph node count");
        check(original_runtime_symbol<decltype(&cudaGraphKernelNodeGetParams)>(
                  "cudaGraphKernelNodeGetParams")(nodes[1], &actual),
              "inspect original public graph kernel identity");
        check(original_runtime_symbol<decltype(&cudaGraphEventWaitNodeGetEvent)>(
                  "cudaGraphEventWaitNodeGetEvent")(nodes[0], &actual_wait),
              "inspect original public graph wait identity");
        check(original_runtime_symbol<decltype(&cudaGraphEventRecordNodeGetEvent)>(
                  "cudaGraphEventRecordNodeGetEvent")(nodes[2], &actual_record),
              "inspect original public graph record identity");
        require(count == 3 && actual.func == parameters.func && actual.gridDim.x == 1 &&
                    actual.gridDim.y == 1 && actual.gridDim.z == 1 && actual.blockDim.x == 1 &&
                    actual.blockDim.y == 1 && actual.blockDim.z == 1 &&
                    actual.sharedMemBytes == 0 && actual_wait == waited &&
                    actual_record == recorded,
                "dependency routing cannot replace original graph functions, nodes or events");
        std::size_t edges{};
        const auto get_edges =
            original_runtime_symbol<decltype(&cudaGraphGetEdges)>("cudaGraphGetEdges");
        check(get_edges(graph, nullptr, nullptr, &edges), "inspect original graph edge count");
        require(edges == 2, "original dependency graph has two edges only");
        std::array<cudaGraphNode_t, 2> from{}, to{};
        check(get_edges(graph, from.data(), to.data(), &edges), "inspect original graph edges");
        const bool has_expected_edges =
            (from[0] == nodes[0] && to[0] == nodes[1] && from[1] == nodes[1] &&
             to[1] == nodes[2]) ||
            (from[1] == nodes[0] && to[1] == nodes[1] && from[0] == nodes[1] && to[0] == nodes[2]);
        require(has_expected_edges, "original wait/kernel/record topology is unchanged");
    }
    cudaGraphExec_t executable{};
    check(symbol.operator()<decltype(&cudaGraphInstantiateWithFlags)>(
              "cudaGraphInstantiateWithFlags")(&executable, graph, 0),
          "public dependency graph instantiation before producer acceptance");
    check(symbol.operator()<decltype(&cudaGraphDestroy)>("cudaGraphDestroy")(graph),
          "public source destruction before producer acceptance and whole replay");
    if (!is_native) {
        const ProcessCallGate::Call call(gate);
        check(router->existing_frontend(context, owner),
              "observe actual public-created graph actor");
        require(owner != nullptr, "public graph operations must create their actual owner");
    }
    const auto before = is_native ? CudaCoordinationStats{} : owner->stats();
    CUdeviceptr device_output = reinterpret_cast<CUdeviceptr>(producer_memory);
    unsigned int work = 100'000;
    void* producer_arguments[]{&device_output, &device_markers, &work};
    const auto producer_launch = runtime_symbol<decltype(&cuLaunchKernel)>(
        is_ptds ? "cuLaunchKernel_ptsz" : "cuLaunchKernel");
    check(producer_launch(low, kResetBlocks, 1, 1, kResetThreads, 1, 1, 0, producer,
                          producer_arguments, nullptr),
          "ordinary public cooperative producer acceptance");
    check(symbol.operator()<decltype(&cudaEventRecord)>(
              is_ptds ? "cudaEventRecord_ptsz" : "cudaEventRecord")(waited, producer),
          "public pending logical graph input record");
    bool has_logical_pending{};
    if (!is_native) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{10};
        while (std::atomic_ref<std::uint32_t>(markers[0]).load(std::memory_order_acquire) == 0) {
            require(std::chrono::steady_clock::now() < deadline,
                    "actual producer must begin before graph dependency proof");
            std::this_thread::yield();
        }
        {
            const ProcessCallGate::Call call(gate);
            CudaEventProjection projection;
            check(router->event_projection(waited, projection),
                  "inspect current logical graph wait");
            has_logical_pending =
                projection.is_logical() && owner->query_event(waited) == CUDA_ERROR_NOT_READY;
        }
        require(has_logical_pending, "whole graph must encounter a truly pending logical producer");
        require(symbol.operator()<decltype(&cudaEventQuery)>("cudaEventQuery")(waited) ==
                    cudaErrorNotReady,
                "public query cannot report completion of the pending logical record");
    }
    const auto sdk_get = original_runtime_symbol<decltype(&cudaGetLastError)>("cudaGetLastError");
    const auto sdk_peek =
        original_runtime_symbol<decltype(&cudaPeekAtLastError)>("cudaPeekAtLastError");
    {
        const NativeDriverCallScope native;
        (void)sdk_get();
    }
    check(symbol.operator()<decltype(&cudaGraphLaunch)>(
              is_ptds ? "cudaGraphLaunch_ptsz" : "cudaGraphLaunch")(executable, consumer),
          "ordinary public whole graph honors its pending logical input");
    CUcontext after_launch{};
    check(driver.context_get_current(&after_launch), "observe graph caller after public replay");
    require(after_launch == context && !is_inside_runtime_call() &&
                (is_native || !is_inside_driver_call()) && !gate.is_entered_by_current_thread() &&
                std::array{sdk_peek(), sdk_get(), sdk_peek()} ==
                    std::array{cudaSuccess, cudaSuccess, cudaSuccess},
            "public whole replay preserves its actual caller context and clean SDK TLS");
    check(synchronize(), "public completion of producer and original whole graph");
    check(symbol.operator()<decltype(&cudaEventQuery)>("cudaEventQuery")(recorded),
          "public original graph record query authority");
    {
        const NativeDriverCallScope native;
        check(original_runtime_symbol<decltype(&cudaEventQuery)>("cudaEventQuery")(recorded),
              "independent actual SDK graph record completion");
    }
    std::array<unsigned int, 4> output{};
    check(copy(output.data(), guarded_memory, sizeof(output), cudaMemcpyDeviceToHost),
          "public exact-once graph dependency readback");
    require(output == std::array<unsigned int, 4>{0x12345678U, 1, 0, 0x76543210U},
            "original graph executes once and never observes premature producer output");
    check(copy(zeros.data(), producer_memory, sizeof(zeros), cudaMemcpyDeviceToHost),
          "public exact producer output readback");
    require(std::ranges::all_of(zeros, [](auto value) { return value == 1; }),
            "every logical producer thread executes exactly once");
    if (!is_native) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{10};
        auto after = owner->stats();
        while (after.acknowledged_completions < before.acknowledged_completions + 2) {
            require(after.acknowledged_failures == before.acknowledged_failures &&
                        std::chrono::steady_clock::now() < deadline,
                    "physical graph completion must receive both terminal control acknowledgments");
            std::this_thread::yield();
            after = owner->stats();
        }
        require(after.granted_executions == before.granted_executions + 2 &&
                    after.acknowledged_completions == before.acknowledged_completions + 2 &&
                    after.retreat_requests == before.retreat_requests &&
                    after.acknowledged_suspensions == before.acknowledged_suspensions &&
                    after.logical_resumes == before.logical_resumes &&
                    after.acknowledged_failures == before.acknowledged_failures,
                "dependency-blocked graph cannot retreat its producer before input completion");
    }
    check(symbol.operator()<decltype(&cudaGraphExecDestroy)>("cudaGraphExecDestroy")(executable),
          "public completed graph executable destruction");
    const auto destroy_event = symbol.operator()<decltype(&cudaEventDestroy)>("cudaEventDestroy");
    check(destroy_event(recorded), "public original graph record destruction");
    check(destroy_event(waited), "public completed dependency event destruction");
    check(cuModuleUnload(module), "public completed producer module destruction");
    check(symbol.operator()<decltype(&cudaStreamDestroy)>("cudaStreamDestroy")(consumer),
          "public graph consumer destruction");
    if (producer)
        check(symbol.operator()<decltype(&cudaStreamDestroy)>("cudaStreamDestroy")(producer),
              "public graph blocking producer destruction");
    check(cuMemFreeHost(markers), "public completed mapped progress destruction");
    check(release(guarded_memory), "public guarded graph allocation release");
    check(release(producer_memory), "public completed producer allocation release");
    check(reset(), "public graph dependency lifetime reset");
    if (!is_native) {
        check(release(nullptr), "initialize public graph quota observation");
        std::size_t available{}, total{};
        check(cudaMemGetInfo(&available, &total), "observe public graph reset quota");
        require(total == 8388608 && available == total, "public graph reset restores full quota");
        check(reset(), "public empty graph observation reset");
    }
    std::cout << "public_graph_dependency mode=" << mode << " native=" << is_native
              << " cold=1 logical_pending=" << has_logical_pending
              << " nodes=3 edges=2 source_retired=1 graph_effects=1 premature_reads=0"
                 " producer_threads=6720 grants="
              << (is_native ? 0 : 2)
              << " producer_retreats=0 tls=1 context=1 guards=1 quota=" << !is_native
              << " cleanup=1\n";
}

void check_detached_capture_end(unsigned int mode, bool is_native) {
    DriverDispatch driver;
    require(driver.initialize(), "resolve actual detached capture dispatch");
    std::optional<NativeDriverCallScope> native_reference;
    if (is_native) native_reference.emplace();
    const bool is_ptds = mode == 1;
    const auto symbol = [is_native]<typename Function>(const char* name) {
        return is_native ? original_runtime_symbol<Function>(name) : runtime_symbol<Function>(name);
    };
    const auto launch = symbol.operator()<decltype(&cudaLaunchKernel)>(
        is_ptds ? "cudaLaunchKernel_ptsz" : "cudaLaunchKernel");
    const auto begin = symbol.operator()<decltype(&cudaStreamBeginCapture)>(
        is_ptds ? "cudaStreamBeginCapture_ptsz" : "cudaStreamBeginCapture");
    const auto end = symbol.operator()<decltype(&cudaStreamEndCapture)>(
        is_ptds ? "cudaStreamEndCapture_ptsz" : "cudaStreamEndCapture");
    const auto device = symbol.operator()<decltype(&cudaSetDevice)>("cudaSetDevice");
    const auto release = symbol.operator()<decltype(&cudaFree)>("cudaFree");
    const auto allocate = symbol.operator()<decltype(&cudaMalloc)>("cudaMalloc");
    const auto copy = symbol.operator()<decltype(&cudaMemcpy)>("cudaMemcpy");
    const auto synchronize =
        symbol.operator()<decltype(&cudaDeviceSynchronize)>("cudaDeviceSynchronize");
    const auto reset = symbol.operator()<decltype(&cudaDeviceReset)>("cudaDeviceReset");
    const auto sdk_device = original_runtime_symbol<decltype(&cudaSetDevice)>("cudaSetDevice");
    const auto sdk_free = original_runtime_symbol<decltype(&cudaFree)>("cudaFree");
    const auto sdk_get = original_runtime_symbol<decltype(&cudaGetLastError)>("cudaGetLastError");
    const auto sdk_peek =
        original_runtime_symbol<decltype(&cudaPeekAtLastError)>("cudaPeekAtLastError");
    check(device(0), "select detached capture device");
    check(release(nullptr), "initialize actual detached capture primary context");
    CUcontext context{};
    check(driver.context_get_current(&context), "observe detached capture caller context");
    auto& gate = cooperative_process_call_gate();
    DriverCooperativeRouter* router{};
    std::shared_ptr<CudaCooperativeFrontend> owner;
    if (!is_native) {
        const ProcessCallGate::Call call(gate);
        router = cooperative_source_router();
        require(router && router->has_native_epoch_policy(), "actual detached epoch policy");
        check(router->existing_frontend(context, owner), "observe absent detached actor");
        require(!owner, "detached fixture cannot privately prepare an actor");
    }
    const std::array<unsigned int, 3> initial{0x12345678U, 0, 0x76543210U};
    void* memory{};
    check(allocate(&memory, sizeof(initial)), "detached guarded allocation");
    check(copy(memory, initial.data(), sizeof(initial), cudaMemcpyHostToDevice),
          "detached guard initialization");
    cudaStream_t producer{}, captured{}, consumer{};
    if (mode == 2)
        check(symbol.operator()<decltype(&cudaStreamCreate)>("cudaStreamCreate")(&producer),
              "detached blocking producer creation");
    const auto create =
        symbol.operator()<decltype(&cudaStreamCreateWithFlags)>("cudaStreamCreateWithFlags");
    check(create(&captured, cudaStreamNonBlocking), "detached explicit capture stream");
    check(create(&consumer, cudaStreamNonBlocking), "detached whole-graph consumer");
    cudaEvent_t observer{};
    {
        const NativeDriverCallScope native;
        check(original_runtime_symbol<decltype(&cudaEventCreateWithFlags)>(
                  "cudaEventCreateWithFlags")(&observer, cudaEventDisableTiming),
              "create independent detached execution observer");
        (void)sdk_get();
    }
    auto* counter = static_cast<unsigned int*>(memory) + 1;
    unsigned long long warm_duration_nanoseconds = 100'000'000;
    void* warm_arguments[]{&counter, &warm_duration_nanoseconds};
    check(launch(glimmer_runtime_pending_kernel(), dim3{1}, dim3{1}, warm_arguments, 0, producer),
          "ordinary detached launch creates actual public epoch actor");
    if (!is_native) {
        const ProcessCallGate::Call call(gate);
        check(router->existing_frontend(context, owner), "observe public-created detached actor");
        require(owner != nullptr, "ordinary operations must create the actual detached actor");
    }
    CaptureOriginalHold issue_hold;
    std::atomic<bool> has_bound{}, has_returned{};
    std::exception_ptr waiting_error;
    bool has_balanced_caller{};
    std::jthread waiting_thread([&] {
        try {
            std::optional<NativeDriverCallScope> native;
            if (is_native) native.emplace();
            {
                const NativeDriverCallScope binding;
                check(sdk_device(0), "bind waiting caller actual SDK device");
                check(sdk_free(nullptr), "bind waiting caller actual shared primary context");
                (void)sdk_get();
            }
            CUcontext actual{};
            check(driver.context_get_current(&actual), "observe waiting caller context");
            require(actual == context, "waiting caller cannot substitute its application context");
            has_bound = true;
            issue_hold.wait();
            unsigned long long second_duration_nanoseconds = 200'000'000;
            void* second_arguments[]{&counter, &second_duration_nanoseconds};
            check(launch(glimmer_runtime_pending_kernel(), dim3{1}, dim3{1}, second_arguments, 0,
                         producer),
                  "second original resumes actual public source admission");
            require(std::array{sdk_peek(), sdk_get(), sdk_peek()} ==
                        std::array{cudaSuccess, cudaSuccess, cudaSuccess},
                    "waiting original preserves clean actual caller error TLS");
            check(driver.context_get_current(&actual), "observe resumed caller original context");
            has_balanced_caller = actual == context && !gate.is_entered_by_current_thread() &&
                                  !is_inside_runtime_call() &&
                                  !runtime_cooperative_launch_is_pending() &&
                                  (is_native || !is_inside_driver_call());
        } catch (...) {
            waiting_error = std::current_exception();
        }
        has_returned = true;
    });
    // Bind before GLOBAL capture; release before the thread owner joins on unwind.
    const CaptureOriginalRelease release_on_exit{issue_hold};
    const auto binding_deadline = std::chrono::steady_clock::now() + std::chrono::seconds{10};
    while (!has_bound.load() && !has_returned.load()) {
        require(std::chrono::steady_clock::now() < binding_deadline,
                "waiting caller must bind before GLOBAL capture begins");
        std::this_thread::yield();
    }
    if (!has_bound) {
        waiting_thread.join();
        if (waiting_error) std::rethrow_exception(waiting_error);
        require(false, "waiting caller unexpectedly returned before context binding");
    }
    check(begin(captured, cudaStreamCaptureModeGlobal), "begin detached GLOBAL capture");
    if (!is_native) {
        const auto epoch = owner->native_epoch_snapshot();
        require(epoch && epoch->execution && epoch->phase == CudaNativeEpochPhase::kComplete &&
                    epoch->status == CUDA_SUCCESS && !epoch->execution->has_ownership &&
                    epoch->execution->issued_calls == 1 && epoch->execution->completed_calls == 1,
                "capture physically completes the actual public-created normal epoch");
    }
    const auto before = is_native ? CudaPausedNativeStats{} : router->native_execution_stats();
    unsigned long long captured_duration_nanoseconds{};
    void* captured_arguments[]{&counter, &captured_duration_nanoseconds};
    check(
        launch(glimmer_runtime_pending_kernel(), dim3{1}, dim3{1}, captured_arguments, 0, captured),
        "detached original graph construction");
    if (!is_native)
        require(router->native_execution_stats().grants == before.grants,
                "captured construction acquires no native execution grant");
    unsigned long long first_duration_nanoseconds = 1'500'000'000;
    void* first_arguments[]{&counter, &first_duration_nanoseconds};
    check(launch(glimmer_runtime_pending_kernel(), dim3{1}, dim3{1}, first_arguments, 0, producer),
          "uncaptured first original occupies the native execution slot");
    {
        const NativeDriverCallScope native;
        check(original_runtime_symbol<decltype(&cudaEventRecord)>(
                  is_ptds ? "cudaEventRecord_ptsz" : "cudaEventRecord")(observer, producer),
              "independent trailing record after first uncaptured original");
    }
    const auto is_pending = [&] {
        const NativeDriverCallScope native;
        const auto exchange =
            original_runtime_symbol<decltype(&cudaThreadExchangeStreamCaptureMode)>(
                "cudaThreadExchangeStreamCaptureMode");
        const auto query = original_runtime_symbol<decltype(&cudaEventQuery)>("cudaEventQuery");
        auto saved = cudaStreamCaptureModeRelaxed;
        check(exchange(&saved), "temporarily relax only independent observer capture mode");
        const auto queried = query(observer);
        check(exchange(&saved), "restore original observer capture mode before application calls");
        require(saved == cudaStreamCaptureModeRelaxed,
                "independent observer mode restored exactly");
        require(queried == cudaSuccess || queried == cudaErrorNotReady,
                "independent SDK observer cannot hide a capture/API failure");
        return queried == cudaErrorNotReady;
    };
    require(is_pending(), "first uncaptured execution is genuinely pending");
    issue_hold.release();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{10};
    bool has_detached_wait{};
    while (true) {
        if (!is_native) {
            const auto observed = gate.snapshot();
            const auto execution = router->native_execution_stats();
            has_detached_wait = observed.active_calls == 0 && observed.admitted_callers == 1 &&
                                !observed.owns_transition && observed.waiting_transitions == 0 &&
                                observed.is_native_only && execution.retained == 2 &&
                                execution.grants == before.grants + 1 &&
                                execution.completed == before.completed;
        }
        if (has_detached_wait || has_returned.load()) break;
        require(std::chrono::steady_clock::now() < deadline,
                "actual detached admission must become observable before deadline");
        std::this_thread::yield();
    }
    require(is_native || has_detached_wait, "second public caller must really detach for control");
    if (is_native) {
        waiting_thread.join();
        if (waiting_error) std::rethrow_exception(waiting_error);
    }
    require(is_native || is_pending(),
            "observed admission cannot rely on already completed first execution");
    cudaGraph_t graph{};
    check(end(captured, &graph), "public capture end returns while source admission waits");
    const bool has_pending_at_end = is_pending();
    require(graph && (is_native || has_pending_at_end),
            "capture end cannot drain or lose the pending source original");
    CUcontext after_end{};
    check(driver.context_get_current(&after_end), "observe original capture caller after end");
    require(after_end == context && !is_inside_runtime_call() &&
                (is_native || !is_inside_driver_call()) && !gate.is_entered_by_current_thread(),
            "capture end balances its own actual caller context and admission scopes");
    {
        const NativeDriverCallScope native;
        cudaStreamCaptureStatus status = cudaStreamCaptureStatusInvalidated;
        check(original_runtime_symbol<decltype(&cudaStreamIsCapturing)>(
                  is_ptds ? "cudaStreamIsCapturing_ptsz" : "cudaStreamIsCapturing")(captured,
                                                                                    &status),
              "independently observe actual original capture end");
        require(status == cudaStreamCaptureStatusNone,
                "SDK capture ends before waiting source issue");
    }
    if (!is_native) {
        const auto observed = gate.snapshot();
        const auto execution = router->native_execution_stats();
        require(observed.active_calls == 0 && observed.admitted_callers == 1 &&
                    observed.is_native_only && !observed.owns_transition &&
                    observed.waiting_transitions == 0 && execution.retained == 2 &&
                    execution.grants == before.grants + 1 && !has_returned.load(),
                "end retains native-only routing and detached caller without early thaw or issue");
    }
    if (waiting_thread.joinable()) waiting_thread.join();
    if (waiting_error) std::rethrow_exception(waiting_error);
    require(has_balanced_caller, "resumed caller balances actual TLS/context and admission scopes");
    check(synchronize(),
          "public completion retires original source executions before normal routing");
    if (!is_native) {
        const auto observed = gate.snapshot();
        const auto execution = router->native_execution_stats();
        require(!observed.is_native_only && observed.active_calls == 0 &&
                    observed.admitted_callers == 0 && !observed.owns_transition &&
                    execution.retained == 0 && execution.grants == before.grants + 2 &&
                    execution.completed == before.completed + 2 &&
                    execution.failed == before.failed,
                "two physical original completions precede balanced normal routing");
    }
    std::size_t count{};
    cudaGraphNode_t node{};
    cudaKernelNodeParams parameters{};
    {
        const NativeDriverCallScope native;
        const auto nodes =
            original_runtime_symbol<decltype(&cudaGraphGetNodes)>("cudaGraphGetNodes");
        check(nodes(graph, nullptr, &count), "observe detached original graph node count");
        require(count == 1, "detached admission cannot inject graph nodes");
        check(nodes(graph, &node, &count), "observe detached original graph node");
        check(original_runtime_symbol<decltype(&cudaGraphKernelNodeGetParams)>(
                  "cudaGraphKernelNodeGetParams")(node, &parameters),
              "observe detached original graph parameters");
    }
    require(parameters.func == glimmer_runtime_pending_kernel() && parameters.gridDim.x == 1 &&
                parameters.gridDim.y == 1 && parameters.gridDim.z == 1 &&
                parameters.blockDim.x == 1 && parameters.blockDim.y == 1 &&
                parameters.blockDim.z == 1 && parameters.sharedMemBytes == 0,
            "detached capture preserves original SDK kernel identity and dimensions");
    std::array<unsigned int, 3> output{};
    check(copy(output.data(), memory, sizeof(output), cudaMemcpyDeviceToHost),
          "detached original construction readback");
    require(output == std::array<unsigned int, 3>{0x12345678U, 3, 0x76543210U},
            "three originals execute once and graph construction does not execute");
    cudaGraphExec_t executable{};
    check(symbol.operator()<decltype(&cudaGraphInstantiateWithFlags)>(
              "cudaGraphInstantiateWithFlags")(&executable, graph, 0),
          "detached original graph instantiation");
    check(symbol.operator()<decltype(&cudaGraphDestroy)>("cudaGraphDestroy")(graph),
          "destroy detached graph source before retained whole replay");
    check(symbol.operator()<decltype(&cudaGraphLaunch)>(
              is_ptds ? "cudaGraphLaunch_ptsz" : "cudaGraphLaunch")(executable, consumer),
          "detached original whole graph replay");
    check(synchronize(), "complete detached original whole replay");
    check(copy(output.data(), memory, sizeof(output), cudaMemcpyDeviceToHost),
          "detached original guarded replay readback");
    require(output == std::array<unsigned int, 3>{0x12345678U, 4, 0x76543210U},
            "detached admission preserves original replay and both guards");
    check(symbol.operator()<decltype(&cudaGraphExecDestroy)>("cudaGraphExecDestroy")(executable),
          "checked detached executable destruction");
    {
        const NativeDriverCallScope native;
        check(original_runtime_symbol<decltype(&cudaEventDestroy)>("cudaEventDestroy")(observer),
              "checked detached independent observer destruction");
    }
    const auto destroy = symbol.operator()<decltype(&cudaStreamDestroy)>("cudaStreamDestroy");
    check(destroy(consumer), "checked detached consumer destruction");
    check(destroy(captured), "checked detached capture stream destruction");
    if (producer) check(destroy(producer), "checked detached blocking producer destruction");
    check(release(memory), "checked detached guarded allocation release");
    check(reset(), "checked detached original lifetime reset");
    if (!is_native) {
        const auto retired = owner->native_epoch_snapshot();
        require(retired && retired->execution && !retired->execution->has_ownership &&
                    retired->created_events == retired->destroyed_events,
                "detached reset retains only balanced historical CPU epoch metadata");
        check(release(nullptr), "initialize detached quota observation lifetime");
        std::size_t available{}, total{};
        check(cudaMemGetInfo(&available, &total), "observe detached public reset quota");
        require(total == 8388608 && available == total, "detached reset restores full exact quota");
        check(reset(), "reset empty detached quota observation lifetime");
    }
    std::cout << "detached_capture_end mode=" << mode << " native=" << is_native
              << " cold=1 detached=" << !is_native << " pending_at_end=" << has_pending_at_end
              << " retained_phase=" << !is_native
              << " originals=3 source_grants=" << (is_native ? 0 : 2)
              << " tls=1 context=1 nodes=1 construction_output=3 replay_output=4 guards=1 quota="
              << !is_native << " cleanup=1\n";
}

void check_process_runtime_lifecycle(unsigned int mode, bool is_cached, bool should_seed_error) {
    DriverDispatch driver;
    require(driver.initialize(), "resolve actual lifecycle SDK dispatch");
    using Launch = cudaError_t (*)(const void*, dim3, dim3, void**, std::size_t, cudaStream_t);
    const bool is_ptds = mode == 1;
    const auto sdk_launch =
        original_runtime_symbol<Launch>(is_ptds ? "cudaLaunchKernel_ptsz" : "cudaLaunchKernel");
    const auto public_launch =
        runtime_symbol<Launch>(is_ptds ? "cudaLaunchKernel_ptsz" : "cudaLaunchKernel");
    const auto sdk_record = original_runtime_symbol<decltype(&cudaEventRecord)>(
        is_ptds ? "cudaEventRecord_ptsz" : "cudaEventRecord");
    const auto public_record = runtime_symbol<decltype(&cudaEventRecord)>(
        is_ptds ? "cudaEventRecord_ptsz" : "cudaEventRecord");
    const auto sdk_reset = original_runtime_symbol<RuntimeDeviceResetFunction>("cudaDeviceReset");
    const auto sdk_set_device = original_runtime_symbol<decltype(&cudaSetDevice)>("cudaSetDevice");
    const auto sdk_free = original_runtime_symbol<decltype(&cudaFree)>("cudaFree");
    const auto sdk_malloc = original_runtime_symbol<decltype(&cudaMalloc)>("cudaMalloc");
    const auto sdk_copy = original_runtime_symbol<decltype(&cudaMemcpy)>("cudaMemcpy");
    const auto sdk_sync =
        original_runtime_symbol<RuntimeDeviceSynchronizeFunction>("cudaDeviceSynchronize");
    const auto sdk_stream_create =
        original_runtime_symbol<decltype(&cudaStreamCreate)>("cudaStreamCreate");
    const auto sdk_elapsed =
        original_runtime_symbol<decltype(&cudaEventElapsedTime)>("cudaEventElapsedTime");
    const auto get_function =
        original_runtime_symbol<decltype(&cudaGetFuncBySymbol)>("cudaGetFuncBySymbol");
    const auto produce =
        original_runtime_symbol<decltype(&cudaGetDeviceCount)>("cudaGetDeviceCount");
    const auto peek =
        original_runtime_symbol<decltype(&cudaPeekAtLastError)>("cudaPeekAtLastError");
    const auto get = original_runtime_symbol<decltype(&cudaGetLastError)>("cudaGetLastError");
    const auto seed = [&] {
        const NativeDriverCallScope native;
        (void)get();
        if (should_seed_error)
            require(produce(nullptr) != cudaSuccess, "seed actual lifecycle SDK error TLS");
    };
    const std::array<unsigned int, 3> initial{0x12345678U, 0, 0x76543210U};
    std::array<cudaError_t, 3> launch_tls{}, reset_tls{};
    CUcontext native_after{};
    CUresult native_context_status{};
    {
        const NativeDriverCallScope native;
        check(sdk_set_device(0), "native lifecycle baseline device");
        check(sdk_free(nullptr), "native lifecycle baseline primary context");
        void* memory{};
        check(sdk_malloc(&memory, sizeof(initial)), "native lifecycle baseline allocation");
        auto* counter = static_cast<unsigned int*>(memory) + 1;
        check(sdk_copy(memory, initial.data(), sizeof(initial), cudaMemcpyHostToDevice),
              "native lifecycle baseline guards");
        cudaStream_t stream{};
        if (mode == 2) check(sdk_stream_create(&stream), "native lifecycle blocking stream");
        unsigned long long duration_nanoseconds{};
        void* arguments[]{&counter, &duration_nanoseconds};
        seed();
        for (unsigned int issue = 0; issue < 2; ++issue)
            check(sdk_launch(glimmer_runtime_pending_kernel(), dim3{1}, dim3{1}, arguments, 0,
                             stream),
                  "native lifecycle original launch baseline");
        launch_tls = {peek(), get(), peek()};
        check(sdk_sync(), "native baseline output completion");
        std::array<unsigned int, 3> output{};
        check(sdk_copy(output.data(), memory, sizeof(output), cudaMemcpyDeviceToHost),
              "native baseline exact guarded readback");
        require(output == std::array<unsigned int, 3>{0x12345678U, 2, 0x76543210U},
                "native lifecycle baseline exact-once output");
        duration_nanoseconds = 500'000'000;
        check(sdk_launch(glimmer_runtime_pending_kernel(), dim3{1}, dim3{1}, arguments, 0, stream),
              "native SDK pending original before matched reset");
        CUevent observer{};
        check(cuEventCreate(&observer, CU_EVENT_DISABLE_TIMING), "native reset pending observer");
        check(sdk_record(observer, stream), "native reset original trailing record");
        require(cuEventQuery(observer) == CUDA_ERROR_NOT_READY,
                "matched native reset baseline starts with actual pending work");
        seed();
        check(sdk_reset(), "native SDK reset baseline owns all remaining resources");
        reset_tls = {peek(), get(), peek()};
        native_context_status = driver.context_get_current(&native_after);
        check(native_context_status, "observe native SDK reset caller binding");
    }
    require(launch_tls[0] == launch_tls[1] && launch_tls[2] == cudaSuccess &&
                (should_seed_error ? launch_tls[0] != cudaSuccess : launch_tls[0] == cudaSuccess),
            "independent SDK launch baseline qualifies declared TLS mode");
    {
        const NativeDriverCallScope native;
        check(sdk_set_device(0), "fresh actual enabled lifecycle device");
        check(sdk_free(nullptr), "fresh actual enabled lifecycle primary");
    }
    const ProcessCallGate::Call call(cooperative_process_call_gate());
    auto* router = cooperative_source_router();
    require(router && router->has_native_epoch_policy(), "actual enabled process lifecycle owner");
    CUcontext context{};
    check(driver.context_get_current(&context), "actual lifecycle caller context");
    std::size_t free_before{}, free_with_owner{}, total{}, remaining{};
    check(cuMemGetInfo(&free_before, &total), "actual lifecycle quota baseline");
    require(total == 8388608 && free_before == total,
            "original SDK baseline leaves the fresh public quota fully available");
    std::shared_ptr<CudaCooperativeFrontend> owner;
    check(router->frontend(context, owner), "actual singleton lifecycle frontend");
    check(cuMemGetInfo(&free_with_owner, &total), "actual frontend private progress charge");
    require(free_with_owner < free_before,
            "actual frontend progress resources are separately charged");
    void* memory{};
    check(cudaMalloc(&memory, sizeof(initial)), "public lifecycle guarded allocation");
    check(cudaMemcpy(memory, initial.data(), sizeof(initial), cudaMemcpyHostToDevice),
          "public lifecycle initialize guards");
    check(cuMemGetInfo(&remaining, &total), "actual lifecycle allocation charge");
    require(free_with_owner - remaining == sizeof(initial),
            "exact public lifecycle allocation charge");
    cudaStream_t stream{}, consumer{};
    if (mode == 2) check(cudaStreamCreate(&stream), "public lifecycle blocking stream");
    check(cudaStreamCreateWithFlags(&consumer, cudaStreamNonBlocking), "public lifecycle consumer");
    cudaEvent_t start{}, end{}, graph_done{}, reset_event{};
    check(cudaEventCreate(&start), "public lifecycle native timing start");
    check(cudaEventCreate(&end), "public lifecycle native timing end");
    check(cudaEventCreateWithFlags(&graph_done, cudaEventDisableTiming),
          "public graph record event");
    check(public_record(start, stream), "public lifecycle original start record");
    cudaFunction_t function{};
    {
        const NativeDriverCallScope native;
        check(get_function(&function, glimmer_runtime_pending_kernel()),
              "fresh actual SDK kernel identity");
    }
    auto* counter = static_cast<unsigned int*>(memory) + 1;
    unsigned long long duration_nanoseconds = 200'000'000;
    void* arguments[]{&counter, &duration_nanoseconds};
    RuntimeCooperativeLaunch plan{function, std::nullopt, {1, 1, 1}, {1, 1, 1}};
    plan.context = context;
    unsigned int quarantines{};
    const auto issue = [&] {
        const auto original = [&] {
            return sdk_launch(glimmer_runtime_pending_kernel(), dim3{1}, dim3{1}, arguments, 0,
                              stream);
        };
        return is_cached ? invoke_runtime_cooperative_launch_original(
                               plan, is_ptds, original,
                               [&] {
                                   ++quarantines;
                                   router->quarantine_normal_launch(context, CUDA_ERROR_UNKNOWN);
                               })
                         : public_launch(glimmer_runtime_pending_kernel(), dim3{1}, dim3{1},
                                         arguments, 0, stream);
    };
    seed();
    check(issue(), "first normal Runtime original enters actual singleton epoch");
    const auto first = owner->native_epoch_snapshot();
    require(first && first->execution && first->execution->issued_calls == 1 &&
                first->execution->has_ownership,
            "first normal Runtime original has actual physical epoch ownership");
    check(issue(), "second normal Runtime original shares singleton epoch");
    const auto second = owner->native_epoch_snapshot();
    require(second && second->execution && second->execution->issued_calls == 2 &&
                second->grant.execution.task_id == first->grant.execution.task_id &&
                quarantines == 0,
            "two exact original Runtime issues retain one immutable grant without provider seam");
    require(std::array{peek(), get(), peek()} == launch_tls,
            "normal Runtime epoch preserves native SDK TLS");
    CUevent observer{};
    {
        const NativeDriverCallScope native;
        check(cuEventCreate(&observer, CU_EVENT_DISABLE_TIMING),
              "independent original pending observer");
        check(sdk_record(observer, stream), "independent original trailing observer record");
        require(cuEventQuery(observer) == CUDA_ERROR_NOT_READY,
                "actual Runtime originals genuinely pending");
    }
    check(public_record(end, stream), "public lifecycle original end record");
    check(runtime_symbol<decltype(&cudaStreamWaitEvent)>(
              is_ptds ? "cudaStreamWaitEvent_ptsz" : "cudaStreamWaitEvent")(consumer, end, 0),
          "public original consumer dependency");
    check(cudaEventSynchronize(end), "public event completion after normal Runtime epoch");
    float milliseconds{-1.0F}, native_milliseconds{-2.0F};
    if (is_cached) {
        const auto measured = intercept_normal_runtime_event_elapsed_time(
            &milliseconds, start, end, [&](float* output) {
                const NativeDriverCallScope native;
                return sdk_elapsed(output, start, end);
            });
        require(measured.has_value(), "normal cached timing root accepts registered events");
        check(*measured, "normal cached SDK event timing");
    } else {
        check(cudaEventElapsedTime(&milliseconds, start, end), "public Runtime event timing");
    }
    {
        const NativeDriverCallScope native;
        check(sdk_elapsed(&native_milliseconds, start, end),
              "independent SDK timing on identical completed events");
        check(cuEventDestroy(observer), "checked independent observer destruction");
    }
    require(milliseconds == native_milliseconds && milliseconds >= 350.0F,
            "normal timing publishes exact SDK sample including original pending work");
    cudaGraph_t graph{};
    cudaGraphExec_t executable{};
    std::array<cudaGraphNode_t, 3> nodes{};
    check(cudaGraphCreate(&graph, 0), "public Runtime graph construction");
    check(cudaGraphAddEventWaitNode(&nodes[0], graph, nullptr, 0, end),
          "public original graph wait node");
    duration_nanoseconds = 0;
    cudaKernelNodeParams parameters{};
    parameters.func = const_cast<void*>(glimmer_runtime_pending_kernel());
    parameters.gridDim = parameters.blockDim = dim3{1};
    parameters.kernelParams = arguments;
    check(cudaGraphAddKernelNode(&nodes[1], graph, &nodes[0], 1, &parameters),
          "public original graph kernel node");
    check(cudaGraphAddEventRecordNode(&nodes[2], graph, &nodes[1], 1, graph_done),
          "public original graph record node");
    std::size_t node_count{};
    cudaKernelNodeParams actual_parameters{};
    cudaEvent_t actual_wait{}, actual_record{};
    check(cudaGraphGetNodes(graph, nullptr, &node_count), "public original graph node count");
    check(cudaGraphKernelNodeGetParams(nodes[1], &actual_parameters),
          "public original graph kernel identity");
    check(cudaGraphEventWaitNodeGetEvent(nodes[0], &actual_wait),
          "public original graph wait identity");
    check(cudaGraphEventRecordNodeGetEvent(nodes[2], &actual_record),
          "public original graph record identity");
    require(node_count == 3 && actual_parameters.func == parameters.func &&
                actual_parameters.gridDim.x == 1 && actual_parameters.gridDim.y == 1 &&
                actual_parameters.gridDim.z == 1 && actual_parameters.blockDim.x == 1 &&
                actual_parameters.blockDim.y == 1 && actual_parameters.blockDim.z == 1 &&
                actual_wait == end && actual_record == graph_done,
            "public graph construction retains all original kernel and event identities");
    check(cudaGraphInstantiateWithFlags(&executable, graph, 0),
          "public original graph executable constructor");
    CudaNativeGraphExecutable registered;
    check(router->graphs().lookup(executable, registered),
          "actual public constructor CPU graph registration");
    require(registered.context() == context && registered.effects().node_count() == 3 &&
                registered.events().size() == 2,
            "actual public graph owns exact original wait/kernel/record effects");
    check(cudaGraphDestroy(graph), "public source graph destruction before replay");
    check(runtime_symbol<RuntimeGraphLaunchFunction>(
              is_ptds ? "cudaGraphLaunch_ptsz" : "cudaGraphLaunch")(executable, consumer),
          "public unchanged original whole-graph replay");
    check(cudaDeviceSynchronize(), "public completion proves whole graph and dependencies");
    check(cudaEventSynchronize(graph_done), "public original graph record completion");
    std::array<unsigned int, 3> output{};
    check(cudaMemcpy(output.data(), memory, sizeof(output), cudaMemcpyDeviceToHost),
          "public lifecycle guarded readback");
    require(output == std::array<unsigned int, 3>{0x12345678U, 3, 0x76543210U},
            "two normal originals plus one unchanged graph kernel execute exactly once");
    check(cudaGraphExecDestroy(executable), "public graph executable destruction");
    CudaNativeGraphExecutable retired_graph;
    require(router->graphs().lookup(executable, retired_graph) == CUDA_ERROR_INVALID_HANDLE &&
                !retired_graph,
            "public graph destruction retires CPU executable identity");
    check(cudaEventDestroy(start), "public timing start destruction");
    check(cudaEventDestroy(end), "public timing end destruction");
    check(cudaEventDestroy(graph_done), "public graph record event destruction");
    check(cudaStreamDestroy(consumer), "public consumer stream destruction");
    check(cudaFree(memory), "public lifecycle checked output release");
    check(cuMemGetInfo(&remaining, &total), "public lifecycle quota restoration before reset");
    require(remaining == free_with_owner,
            "application release retains the exact live frontend charge");

    check(cudaMalloc(&memory, sizeof(initial)), "public reset owns fresh charged output");
    check(cudaMemcpy(memory, initial.data(), sizeof(initial), cudaMemcpyHostToDevice),
          "public pending reset initialization");
    counter = static_cast<unsigned int*>(memory) + 1;
    check(cudaEventCreateWithFlags(&reset_event, cudaEventDisableTiming),
          "public reset event identity");
    check(cuMemGetInfo(&remaining, &total), "exact application and frontend charges before reset");
    require(free_with_owner - remaining == sizeof(initial),
            "reset owns one exact application charge");
    CudaEventProjection reset_registration;
    check(router->event_projection(reset_event, reset_registration),
          "actual registered reset event");
    {
        const NativeDriverCallScope native;
        check(cuEventCreate(&observer, CU_EVENT_DISABLE_TIMING),
              "independent pending reset observer");
    }
    duration_nanoseconds = 500'000'000;
    check(issue(), "normal Runtime pending original before public reset");
    const auto pending = owner->native_epoch_snapshot();
    require(pending && pending->execution && pending->execution->issued_calls == 1 &&
                pending->execution->has_ownership &&
                pending->grant.execution.task_id != first->grant.execution.task_id,
            "public reset receives a live newly granted original epoch");
    {
        const NativeDriverCallScope native;
        check(sdk_record(observer, stream), "independent original reset trailing record");
        require(cuEventQuery(observer) == CUDA_ERROR_NOT_READY,
                "public reset starts with genuine pending work");
    }
    seed();
    check(cudaDeviceReset(), "actual public reset retires pending epoch without a private close");
    require(std::array{peek(), get(), peek()} == reset_tls,
            "public reset preserves measured native SDK TLS");
    CUcontext after{};
    const auto context_status = driver.context_get_current(&after);
    require(
        context_status == native_context_status && (after == nullptr) == (native_after == nullptr),
        "public reset preserves SDK post-return caller binding state");
    std::shared_ptr<CudaCooperativeFrontend> retired;
    check(router->existing_frontend(context, retired), "CPU-only old reset frontend lookup");
    CudaEventProjection retired_event;
    const auto completed = owner->native_epoch_snapshot();
    require(!retired && completed && completed->execution &&
                completed->phase == CudaNativeEpochPhase::kComplete &&
                completed->status == CUDA_SUCCESS && !completed->execution->has_ownership &&
                completed->execution->issued_calls == 1 &&
                completed->execution->completed_calls == 1 &&
                completed->created_events == completed->destroyed_events && quarantines == 0 &&
                router->event_projection(reset_event, retired_event) == CUDA_ERROR_INVALID_HANDLE &&
                !retired_event && !is_inside_runtime_call() && !is_inside_driver_call() &&
                !runtime_cooperative_launch_is_pending(),
            "public reset retires old actor/event/epoch identities and balances caller scopes");
    // Reset owns memory, stream, reset_event and observer. Never query/free their
    // invalidated SDK handles; only fresh-context quota observation follows.
    {
        const NativeDriverCallScope native;
        check(sdk_free(nullptr), "fresh SDK primary after public reset");
    }
    check(cuMemGetInfo(&remaining, &total), "fresh-context quota after public reset");
    require(remaining == free_before,
            "public reset restores application and private frontend charges");
    check(cudaDeviceReset(), "checked empty fresh lifetime cleanup");
    std::cout << "process_runtime_lifecycle mode=" << mode << " cached=" << is_cached
              << " originals=2 grant=1 tls=1 seed=" << should_seed_error
              << " event=1 timing=1 graph=1 reset_pending=1 quota=1 guards=1 cleanup=1\n";
}

void check_normal_runtime_epochs(unsigned int mode, bool is_cached) {
    DriverDispatch driver;
    require(driver.initialize(), "actual Runtime epoch Driver dispatch");
    using Launch = cudaError_t (*)(const void*, dim3, dim3, void**, std::size_t, cudaStream_t);
    const bool is_ptds = mode == 1;
    const auto name = is_ptds ? "cudaLaunchKernel_ptsz" : "cudaLaunchKernel";
    const auto sdk_launch = original_runtime_symbol<Launch>(name);
    const auto launch = is_cached ? sdk_launch : runtime_symbol<Launch>(name);
    const auto get_function =
        original_runtime_symbol<decltype(&cudaGetFuncBySymbol)>("cudaGetFuncBySymbol");
    const auto produce =
        original_runtime_symbol<decltype(&cudaGetDeviceCount)>("cudaGetDeviceCount");
    const auto peek =
        original_runtime_symbol<decltype(&cudaPeekAtLastError)>("cudaPeekAtLastError");
    const auto get = original_runtime_symbol<decltype(&cudaGetLastError)>("cudaGetLastError");
    unsigned int* allocation{};
    cudaStream_t stream{};
    cudaFunction_t function{};
    CUcontext context{};
    const std::array<unsigned int, 3> initial{0x12345678U, 0, 0x76543210U};
    {
        const NativeDriverCallScope native;
        check(cudaSetDevice(0), "bind actual Runtime epoch device before allocation");
        check(cudaFree(nullptr), "fresh actual Runtime primary context");
        void* memory{};
        check(cudaMalloc(&memory, sizeof(initial)), "guarded actual Runtime epoch allocation");
        allocation = static_cast<unsigned int*>(memory);
        check(cudaMemcpy(allocation, initial.data(), sizeof(initial), cudaMemcpyHostToDevice),
              "initialize Runtime epoch guards");
        if (mode == 2) check(cudaStreamCreate(&stream), "actual blocking Runtime epoch stream");
        check(get_function(&function, glimmer_runtime_capture_kernel()),
              "actual SDK original function identity");
        check(driver.context_get_current(&context), "actual SDK epoch context identity");
    }
    auto pointer = allocation + 1;
    void* arguments[]{&pointer};
    const auto invoke = [&](Launch selected, dim3 grid) {
        return selected(glimmer_runtime_capture_kernel(), grid, dim3{1}, arguments, 0, stream);
    };
    std::array<cudaError_t, 3> baseline{};
    cudaError_t validation{};
    std::array<cudaError_t, 3> validation_tls{};
    {
        const NativeDriverCallScope native;
        (void)get();
        require(produce(nullptr) != cudaSuccess, "seed actual native Runtime error TLS");
        check(invoke(sdk_launch, dim3{1}), "first original SDK baseline kernel");
        check(invoke(sdk_launch, dim3{1}), "second original SDK baseline kernel");
        baseline = {peek(), get(), peek()};
        validation = invoke(sdk_launch, dim3{0});
        validation_tls = {peek(), get(), peek()};
    }
    require(validation == cudaErrorInvalidConfiguration && validation_tls[0] == validation &&
                validation_tls[1] == validation && validation_tls[2] == cudaSuccess,
            "actual SDK zero-sized launch rejection is qualified before epoch entry");
    {
        DriverCooperativeRouter failed_router(driver, {}, false);
        require(failed_router.normal_owner_status() == CUDA_SUCCESS,
                "actor-free normal owner initially healthy");
        failed_router.quarantine_normal_launch(nullptr, CUDA_ERROR_INVALID_VALUE);
        failed_router.quarantine_normal_launch(context, CUDA_ERROR_UNKNOWN);
        unsigned int effects{};
        require(failed_router.normal_owner_status() == CUDA_ERROR_UNKNOWN &&
                    failed_router.native_boundary(
                        [&] {
                            ++effects;
                            return CUDA_SUCCESS;
                        },
                        [&] {
                            ++effects;
                            return CUDA_SUCCESS;
                        }) == CUDA_ERROR_UNKNOWN &&
                    effects == 0,
                "actor-free normal quarantine excludes original and completion effects");
    }
    auto& gate = cooperative_process_call_gate();
    DriverCooperativeRouter router(driver, {}, false, false,
                                   glimmer::control::NativeEpochExtent{8, 2, 2, 1'000'000});
    std::shared_ptr<CudaCooperativeFrontend> owner;
    check(router.frontend(context, owner), "actual Runtime epoch frontend");
    RuntimeCooperativeLaunch plan{function, std::nullopt, {1, 1, 1}, {1, 1, 1}};
    plan.context = context;
    unsigned int originals{}, quarantines{};
    const auto caller = std::this_thread::get_id();
    const RuntimeNativeKernelExecution execution = [&](CUfunction actual, CUstream actual_stream,
                                                       bool actual_ptds, const auto& original) {
        require(
            actual == function && actual_ptds == is_ptds && std::this_thread::get_id() == caller,
            "actual SDK normal leaf supplies original function, PTDS and caller");
        ++originals;
        return router.native_execution(gate, owner, actual_stream, actual_ptds, original);
    };
    const auto quarantine = [&] {
        ++quarantines;
        router.quarantine_normal_launch(context, CUDA_ERROR_UNKNOWN);
    };
    const ProcessCallGate::Call call(gate);
    {
        const NativeDriverCallScope native;
        require(produce(nullptr) != cudaSuccess, "seed owned Runtime caller error TLS");
    }
    check(invoke_runtime_cooperative_launch_original(
              plan, is_ptds, [&] { return invoke(launch, dim3{1}); }, quarantine, execution),
          "first actual matched Runtime epoch original");
    const auto grant = owner->native_epoch_snapshot()->grant;
    check(invoke_runtime_cooperative_launch_original(
              plan, is_ptds, [&] { return invoke(launch, dim3{1}); }, quarantine, execution),
          "second actual matched Runtime epoch original");
    require(owner->native_epoch_snapshot()->grant.execution.task_id == grant.execution.task_id,
            "actual Runtime originals retain one immutable epoch grant");
    const std::array observed{peek(), get(), peek()};
    require(observed == baseline, "actual normal Runtime epoch preserves measured nonzero SDK TLS");
    plan.grid[0] = 0;
    const auto rejected = invoke_runtime_cooperative_launch_original(
        plan, is_ptds, [&] { return invoke(launch, dim3{0}); }, quarantine, execution);
    const std::array rejected_tls{peek(), get(), peek()};
    require(rejected == validation && rejected_tls == validation_tls && originals == 2 &&
                quarantines == 0,
            "actual Runtime validation rejects without epoch entry, replay or TLS replacement");
    CUcontext after{};
    check(driver.context_get_current(&after), "actual Runtime epoch caller after rejection");
    require(after == context && !is_inside_runtime_call() && !is_inside_driver_call() &&
                !runtime_cooperative_launch_is_pending(),
            "actual Runtime context/TLS restored");
    check(owner->close_native_epoch(), "close physically proven actual Runtime epoch");
    std::array<unsigned int, 3> output{};
    {
        const NativeDriverCallScope native;
        check(cudaMemcpy(output.data(), allocation, sizeof(output), cudaMemcpyDeviceToHost),
              "actual Runtime epoch guarded readback");
    }
    require(output == std::array<unsigned int, 3>{0x12345678U, 4, 0x76543210U},
            "two native and two epoch originals preserve actual guards and exact output");
    check(router.close_context(context, [] { return CUDA_SUCCESS; }),
          "retire actual Runtime epoch owner");
    owner.reset();
    {
        const NativeDriverCallScope native;
        if (stream) check(cudaStreamDestroy(stream), "destroy actual Runtime epoch stream");
        check(cudaFree(allocation), "release actual Runtime epoch output");
        check(cudaDeviceReset(), "destroy actual Runtime epoch primary lifetime");
    }
    std::cout << "runtime_epoch mode=" << mode << " cached=" << is_cached
              << " originals=2 grant=1 tls=1 validation=1 output=4 guards=1 cleanup=1\n";
}
#endif
}  // namespace

int main(int argc, char** argv) {
    try {
#ifdef GLIMMER_HAS_RUNTIME_CAPTURE_KERNEL
        if (argc == 2 && std::string_view(argv[1]).starts_with("--environment-")) {
            const auto mode_name = std::string_view(argv[1]).substr(14);
            require(mode_name == "runtime-lifecycle" || mode_name == "runtime-lifecycle-clean" ||
                        mode_name == "graph-dependency" || mode_name == "detached-capture-end" ||
                        mode_name == "cold-capture",
                    "known public environment GPU mode");
            // No trusted startup call: ordinary roots must select their launch environment.
            for (unsigned int lifetime = 0; lifetime < 2; ++lifetime)
                for (unsigned int mode = 0; mode < 3; ++mode) {
                    if (mode_name.starts_with("runtime-lifecycle")) {
                        for (bool is_cached : {false, true})
                            check_process_runtime_lifecycle(mode, is_cached,
                                                            mode_name == "runtime-lifecycle");
                    } else if (mode_name == "graph-dependency") {
                        check_public_graph_dependency(mode, true);
                        check_public_graph_dependency(mode, false);
                    } else if (mode_name == "detached-capture-end") {
                        check_detached_capture_end(mode, true);
                        check_detached_capture_end(mode, false);
                    } else {
                        check_process_epoch_capture(mode, true, true);
                        check_process_epoch_capture(mode, false, true);
                    }
                }
            return EXIT_SUCCESS;
        }
        if (argc == 2 && (std::string_view(argv[1]) == "--source-event-graph" ||
                          std::string_view(argv[1]) == "--source-event-graph-clean")) {
            const bool should_check_rejection = std::string_view(argv[1]) == "--source-event-graph";
            for (unsigned int lifetime = 0; lifetime < 2; ++lifetime)
                for (bool is_ptds : {false, true}) {
                    check_source_event_graph(is_ptds, true, should_check_rejection);
                    check_source_event_graph(is_ptds, false, should_check_rejection);
                }
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--public-graph-dependency") {
            {
                const ProcessCallGate::Call call(cooperative_process_call_gate());
                check(configure_cooperative_native_epochs({{8, 2, 2, 1'000'000}, {}}),
                      "configure actual public graph process before initialization");
            }
            for (unsigned int lifetime = 0; lifetime < 2; ++lifetime)
                for (unsigned int mode = 0; mode < 3; ++mode) {
                    check_public_graph_dependency(mode, true);
                    check_public_graph_dependency(mode, false);
                }
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--detached-capture-end") {
            {
                const ProcessCallGate::Call call(cooperative_process_call_gate());
                check(configure_cooperative_native_epochs({{8, 2, 2, 1'000'000}, {}}),
                      "configure actual detached capture process before initialization");
            }
            for (unsigned int lifetime = 0; lifetime < 2; ++lifetime)
                for (unsigned int mode = 0; mode < 3; ++mode) {
                    check_detached_capture_end(mode, true);
                    check_detached_capture_end(mode, false);
                }
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--cached-capture-overlap") {
            {
                const ProcessCallGate::Call call(cooperative_process_call_gate());
                check(configure_cooperative_native_epochs({{8, 2, 2, 1'000'000}, {}}),
                      "configure actual overlapping capture process owner before initialization");
            }
            for (unsigned int lifetime = 0; lifetime < 2; ++lifetime)
                for (unsigned int mode = 0; mode < 3; ++mode) {
                    check_process_epoch_capture(mode, true, true);
                    check_cached_capture_overlap(mode);
                }
            return EXIT_SUCCESS;
        }
        if (argc == 2 && (std::string_view(argv[1]) == "--process-epoch-capture" ||
                          std::string_view(argv[1]) == "--process-cold-epoch-capture")) {
            {
                const ProcessCallGate::Call call(cooperative_process_call_gate());
                check(configure_cooperative_native_epochs({{8, 2, 2, 1'000'000}, {}}),
                      "configure actual public capture process owner before initialization");
            }
            const bool is_cold = std::string_view(argv[1]) == "--process-cold-epoch-capture";
            for (unsigned int lifetime = 0; lifetime < 2; ++lifetime)
                for (unsigned int mode = 0; mode < 3; ++mode) {
                    check_process_epoch_capture(mode, true, is_cold);
                    check_process_epoch_capture(mode, false, is_cold);
                }
            return EXIT_SUCCESS;
        }
        if (argc == 2 && (std::string_view(argv[1]) == "--process-runtime-lifecycle" ||
                          std::string_view(argv[1]) == "--process-runtime-lifecycle-clean")) {
            {
                const ProcessCallGate::Call call(cooperative_process_call_gate());
                check(configure_cooperative_native_epochs({{8, 2, 2, 1'000'000}, {}}),
                      "configure actual normal lifecycle process owner before initialization");
            }
            const bool should_seed_error =
                std::string_view(argv[1]) == "--process-runtime-lifecycle";
            for (unsigned int lifetime = 0; lifetime < 2; ++lifetime)
                for (unsigned int mode = 0; mode < 3; ++mode)
                    for (bool is_cached : {false, true})
                        check_process_runtime_lifecycle(mode, is_cached, should_seed_error);
            return EXIT_SUCCESS;
        }
        if (argc == 2 && (std::string_view(argv[1]) == "--process-epoch-recovery" ||
                          std::string_view(argv[1]) == "--process-epoch-recovery-clean")) {
            {
                auto& gate = cooperative_process_call_gate();
                const ProcessCallGate::Call call(gate);
                check(configure_cooperative_native_epochs(
                          {{8, 2, 2, 1'000'000},
                           {runtime_recovery_query, runtime_recovery_destroy,
                            runtime_recovery_driver_sync}}),
                      "configure actual process epoch owner before its public initialization");
            }
            const bool should_seed_error = std::string_view(argv[1]) == "--process-epoch-recovery";
            for (unsigned int lifetime = 0; lifetime < 2; ++lifetime)
                for (unsigned int mode = 0; mode < 3; ++mode)
                    for (bool is_cached : {false, true})
                        check_runtime_epoch_recovery(mode, is_cached, should_seed_error, true);
            return EXIT_SUCCESS;
        }
        if (argc == 2 && (std::string_view(argv[1]) == "--runtime-epoch-recovery" ||
                          std::string_view(argv[1]) == "--runtime-epoch-recovery-clean")) {
            const bool should_seed_error = std::string_view(argv[1]) == "--runtime-epoch-recovery";
            for (unsigned int lifetime = 0; lifetime < 2; ++lifetime)
                for (unsigned int mode = 0; mode < 3; ++mode)
                    for (bool is_cached : {false, true})
                        check_runtime_epoch_recovery(mode, is_cached, should_seed_error);
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--normal-null-free") {
            check_normal_null_free();
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--normal-runtime-epochs") {
            for (unsigned int lifetime = 0; lifetime < 2; ++lifetime)
                for (unsigned int mode = 0; mode < 3; ++mode)
                    for (bool is_cached : {false, true})
                        check_normal_runtime_epochs(mode, is_cached);
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--owned-user-lifecycle-probe") {
            for (const auto* abi : {"legacy", "modern"}) {
                for (const auto* entry : {"direct", "cached"}) {
                    for (const auto* references : {"single", "extra"}) {
                        for (unsigned int caller = 0; caller < 3; ++caller) {
                            const auto flag = std::string("--owned-user-destroy-") + abi + "-" +
                                              entry + "-" + references + "-" +
                                              std::to_string(caller);
                            run_source_quarantine_child(flag.c_str());
                        }
                    }
                }
            }
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--owned-user-quarantine-probe") {
            for (unsigned int fault = 0; fault < 6; ++fault) {
                const auto flag = std::string("--owned-user-fault-") + std::to_string(fault);
                run_source_quarantine_child(flag.c_str());
            }
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]).starts_with("--owned-user-fault-")) {
            const std::string_view flag(argv[1]);
            require(flag.back() >= '0' && flag.back() <= '5', "qualified user destruction fault");
            const auto fault = flag.back() - '0';
            check_owned_user_destroy({.is_modern = true, .fault = fault});
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]).starts_with("--owned-user-destroy-")) {
            const std::string_view flag(argv[1]);
            require(flag.back() >= '0' && flag.back() <= '2', "qualified user-destroy caller");
            check_owned_user_destroy(
                {.is_modern = flag.find("-modern-") != std::string_view::npos,
                 .is_cached = flag.find("-cached-") != std::string_view::npos,
                 .has_extra_reference = flag.find("-extra-") != std::string_view::npos,
                 .caller = static_cast<unsigned int>(flag.back() - '0')});
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]).starts_with("--owned-primary-fault-")) {
            const auto flag = std::string_view(argv[1]);
            require(flag.back() >= '0' && flag.back() <= '3', "qualified destruction fault");
            check_owned_primary_destroy({.fault = flag.back() - '0'});
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]).starts_with("--owned-primary-destroy-")) {
            const auto flag = std::string_view(argv[1]);
            const bool cached = flag.ends_with("cached-0") || flag.ends_with("cached-1") ||
                                flag.ends_with("cached-2");
            require(flag.back() >= '0' && flag.back() <= '2', "qualified destruction caller");
            check_owned_primary_destroy(
                {.is_cached = cached,
                 .is_modern = !flag.starts_with("--owned-primary-destroy-legacy-"),
                 .caller = static_cast<unsigned int>(flag.back() - '0')});
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--owned-primary-legacy-rejection-probe") {
            for (const auto* flag : {"--owned-primary-destroy-legacy-direct-0",
                                     "--owned-primary-destroy-legacy-direct-1",
                                     "--owned-primary-destroy-legacy-direct-2",
                                     "--owned-primary-destroy-legacy-cached-0",
                                     "--owned-primary-destroy-legacy-cached-1",
                                     "--owned-primary-destroy-legacy-cached-2"}) {
                run_source_quarantine_child(flag);
            }
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--owned-primary-rejection-probe") {
            for (const auto* flag :
                 {"--owned-primary-destroy-direct-0", "--owned-primary-destroy-direct-1",
                  "--owned-primary-destroy-direct-2", "--owned-primary-destroy-cached-0",
                  "--owned-primary-destroy-cached-1", "--owned-primary-destroy-cached-2",
                  "--owned-primary-fault-0", "--owned-primary-fault-1", "--owned-primary-fault-2",
                  "--owned-primary-fault-3"}) {
                run_source_quarantine_child(flag);
            }
            return EXIT_SUCCESS;
        }
        if (argc == 2 && (std::string_view(argv[1]) == "--native-error-tls-cold" ||
                          std::string_view(argv[1]) == "--native-error-tls-warm")) {
            check_native_runtime_error_tls(std::string_view(argv[1]) == "--native-error-tls-warm");
            return EXIT_SUCCESS;
        }
        if (argc == 2 && (std::string_view(argv[1]) == "--native-context-destroy-legacy" ||
                          std::string_view(argv[1]) == "--native-context-destroy-modern")) {
            check_native_context_destroy_version(std::string_view(argv[1]) ==
                                                 "--native-context-destroy-modern");
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--native-caller-semantics-probe") {
            for (const auto* flag :
                 {"--native-error-tls-cold", "--native-error-tls-warm",
                  "--native-context-destroy-legacy", "--native-context-destroy-modern"}) {
                run_source_quarantine_child(flag);
            }
            return EXIT_SUCCESS;
        }
        if (argc == 2 && (std::string_view(argv[1]) == "--native-primary-release-legacy" ||
                          std::string_view(argv[1]) == "--native-primary-release-modern")) {
            check_native_primary_release_version(std::string_view(argv[1]).ends_with("-modern"));
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--native-primary-release-probe") {
            run_source_quarantine_child("--native-primary-release-legacy");
            run_source_quarantine_child("--native-primary-release-modern");
            return EXIT_SUCCESS;
        }
        if (argc == 2 && (std::string_view(argv[1]) == "--cold-runtime-graph-native" ||
                          std::string_view(argv[1]) == "--cold-runtime-graph-public")) {
            check_cold_runtime_graph_constructor(std::string_view(argv[1]) ==
                                                 "--cold-runtime-graph-native");
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--runtime-graph-constructor-probe") {
            for (const auto* child :
                 {"--cold-runtime-graph-native", "--cold-runtime-graph-public",
                  "--runtime-graph-constructor-legacy", "--runtime-graph-constructor-ptds",
                  "--runtime-graph-constructor-null", "--runtime-graph-base-legacy",
                  "--runtime-graph-base-ptds", "--runtime-graph-base-null"}) {
                run_source_quarantine_child(child);
            }
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--runtime-graph-event-update-probe") {
            for (const auto* child :
                 {"--runtime-graph-event-update-legacy", "--runtime-graph-event-update-ptds",
                  "--runtime-graph-event-update-null"})
                run_source_quarantine_child(child);
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--runtime-graph-update-probe") {
            for (const auto* mode :
                 {"--runtime-graph-update-whole-legacy", "--runtime-graph-update-whole-ptds",
                  "--runtime-graph-update-whole-null", "--runtime-graph-update-child-legacy",
                  "--runtime-graph-update-child-ptds", "--runtime-graph-update-child-null",
                  "--runtime-graph-update-generic-legacy", "--runtime-graph-update-generic-ptds",
                  "--runtime-graph-update-generic-null"})
                run_source_quarantine_child(mode);
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]).starts_with("--runtime-graph-update-")) {
            const std::string_view mode{argv[1]};
            const auto kind = mode.starts_with("--runtime-graph-update-whole-")     ? 1U
                              : mode.starts_with("--runtime-graph-update-child-")   ? 2U
                              : mode.starts_with("--runtime-graph-update-generic-") ? 3U
                                                                                    : 0U;
            require(kind != 0 && (mode.ends_with("-legacy") || mode.ends_with("-ptds") ||
                                  mode.ends_with("-null")),
                    "select exact whole/child/generic SDK update and stream mode");
            check_owned_runtime_graph(!mode.ends_with("-legacy"), mode.ends_with("-null"), false,
                                      true, false, false, false, kind);
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--runtime-graph-node-update-probe") {
            for (const auto* child :
                 {"--runtime-graph-node-update-legacy", "--runtime-graph-node-update-ptds",
                  "--runtime-graph-node-update-null"})
                run_source_quarantine_child(child);
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]).starts_with("--runtime-graph-node-update-")) {
            const auto mode = std::string_view(argv[1]);
            require(mode == "--runtime-graph-node-update-legacy" ||
                        mode == "--runtime-graph-node-update-ptds" ||
                        mode == "--runtime-graph-node-update-null",
                    "unknown generic node mutation fixture mode");
            check_owned_runtime_graph(mode != "--runtime-graph-node-update-legacy",
                                      mode == "--runtime-graph-node-update-null", false, true,
                                      false, false, true);
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]).starts_with("--runtime-graph-event-update-")) {
            const auto mode = std::string_view(argv[1]);
            require(mode == "--runtime-graph-event-update-legacy" ||
                        mode == "--runtime-graph-event-update-ptds" ||
                        mode == "--runtime-graph-event-update-null",
                    "unknown graph mutation fixture mode");
            check_owned_runtime_graph(mode != "--runtime-graph-event-update-legacy",
                                      mode == "--runtime-graph-event-update-null", false, true,
                                      false, true);
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]).starts_with("--runtime-graph-base-")) {
            const auto mode = std::string_view(argv[1]);
            require(mode == "--runtime-graph-base-legacy" || mode == "--runtime-graph-base-ptds" ||
                        mode == "--runtime-graph-base-null",
                    "unknown modern base graph constructor fixture mode");
            check_owned_runtime_graph(mode != "--runtime-graph-base-legacy",
                                      mode == "--runtime-graph-base-null", false, true, true);
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]).starts_with("--runtime-graph-constructor-")) {
            const auto mode = std::string_view(argv[1]);
            require(mode == "--runtime-graph-constructor-legacy" ||
                        mode == "--runtime-graph-constructor-ptds" ||
                        mode == "--runtime-graph-constructor-null",
                    "unknown Runtime constructor fixture mode");
            check_owned_runtime_graph(mode != "--runtime-graph-constructor-legacy",
                                      mode == "--runtime-graph-constructor-null", false, true);
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--owned-runtime-graph-probe") {
            for (const auto* child :
                 {"--owned-runtime-graph-legacy", "--owned-runtime-graph-ptds",
                  "--owned-runtime-graph-null", "--owned-runtime-graph-fault"}) {
                run_source_quarantine_child(child);
            }
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]).starts_with("--owned-runtime-graph-")) {
            const auto mode = std::string_view(argv[1]);
            require(
                mode == "--owned-runtime-graph-legacy" || mode == "--owned-runtime-graph-ptds" ||
                    mode == "--owned-runtime-graph-null" || mode == "--owned-runtime-graph-fault",
                "unknown owned Runtime graph fixture mode");
            check_owned_runtime_graph(
                mode == "--owned-runtime-graph-ptds" || mode == "--owned-runtime-graph-null",
                mode == "--owned-runtime-graph-null", mode == "--owned-runtime-graph-fault");
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--reversible-context-checkpoint") {
            check_reversible_context_checkpoint();
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--normal-runtime-failed-reset-probe") {
            for (const auto* child : {"--failed-reset-invalid", "--failed-reset-unknown",
                                      "--failed-reset-runtime-only"}) {
                run_source_quarantine_child(child);
            }
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]).starts_with("--failed-reset-")) {
            const auto flag = std::string_view(argv[1]);
            require(flag == "--failed-reset-invalid" || flag == "--failed-reset-unknown" ||
                        flag == "--failed-reset-runtime-only",
                    "recognized controlled failed reset case");
            check_failed_runtime_reset(flag == "--failed-reset-invalid" ? cudaErrorInvalidValue
                                       : flag == "--failed-reset-unknown"
                                           ? cudaErrorUnknown
                                           : static_cast<cudaError_t>(cudaErrorApiFailureBase + 7));
            return EXIT_SUCCESS;
        }
        if (argc == 2 && (std::string_view(argv[1]) == "--native-primary-reset-legacy" ||
                          std::string_view(argv[1]) == "--native-primary-reset-modern")) {
            check_native_primary_reset_version(std::string_view(argv[1]).ends_with("-modern"));
            return EXIT_SUCCESS;
        }
        if (argc == 2 && (std::string_view(argv[1]) == "--model-primary-reset-legacy" ||
                          std::string_view(argv[1]) == "--model-primary-reset-modern")) {
            check_native_primary_reset_version(std::string_view(argv[1]).ends_with("-modern"),
                                               false);
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--normal-runtime-cached-reset-model") {
            for (unsigned int cycle = 0; cycle < 3; ++cycle) {
                check_unbound_reset_cycle(false, cycle, true);
            }
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--normal-runtime-cached-bound-reset-model") {
            for (unsigned int cycle = 0; cycle < 3; ++cycle) {
                check_unbound_reset_cycle(false, cycle, true, false);
            }
            return EXIT_SUCCESS;
        }
        if (argc == 2 && (std::string_view(argv[1]) == "--normal-runtime-bound-reset-native" ||
                          std::string_view(argv[1]) == "--normal-runtime-bound-reset-model")) {
            for (unsigned int cycle = 0; cycle < 3; ++cycle) {
                check_unbound_reset_cycle(std::string_view(argv[1]).ends_with("-native"), cycle,
                                          false, false);
            }
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--normal-runtime-cached-cold-reset-model") {
            check_cold_reset(false, true);
            return EXIT_SUCCESS;
        }
        if (argc == 2 &&
            std::string_view(argv[1]) == "--normal-runtime-cached-pending-reset-model") {
            check_pending_bound_reset(true);
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--normal-runtime-cached-reset-probe") {
            for (const auto* child :
                 {"--native-unbound-reset-effect", "--native-registered-reset-effect",
                  "--native-primary-reset-legacy", "--native-primary-reset-modern",
                  "--model-primary-reset-legacy", "--model-primary-reset-modern",
                  "--normal-runtime-cold-reset-native", "--normal-runtime-cached-cold-reset-model",
                  "--normal-runtime-bound-reset-native", "--normal-runtime-bound-reset-model",
                  "--normal-runtime-cached-bound-reset-model",
                  "--normal-runtime-cached-reset-model",
                  "--normal-runtime-cached-pending-reset-model"}) {
                run_source_quarantine_child(child);
            }
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--native-unbound-reset-effect") {
            check_original_unbound_reset_effect();
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--native-registered-reset-effect") {
            check_original_registered_reset_effect();
            return EXIT_SUCCESS;
        }
        if (argc == 2 && (std::string_view(argv[1]) == "--normal-runtime-cold-reset-native" ||
                          std::string_view(argv[1]) == "--normal-runtime-cold-reset-model")) {
            check_cold_reset(std::string_view(argv[1]).ends_with("-native"));
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--normal-runtime-pending-reset-model") {
            check_pending_bound_reset();
            return EXIT_SUCCESS;
        }
        if (argc == 2 && (std::string_view(argv[1]) == "--normal-runtime-unbound-reset-native" ||
                          std::string_view(argv[1]) == "--normal-runtime-unbound-reset-model")) {
            check_unbound_reset(std::string_view(argv[1]).ends_with("-native"));
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--normal-runtime-unbound-reset-probe") {
            run_source_quarantine_child("--normal-runtime-cold-reset-native");
            run_source_quarantine_child("--normal-runtime-cold-reset-model");
            run_source_quarantine_child("--normal-runtime-unbound-reset-native");
            run_source_quarantine_child("--normal-runtime-unbound-reset-model");
            run_source_quarantine_child("--normal-runtime-pending-reset-model");
            return EXIT_SUCCESS;
        }
        if (argc == 2 &&
            (std::string_view(argv[1]) == "--normal-runtime-nonprimary-reset-native" ||
             std::string_view(argv[1]) == "--normal-runtime-nonprimary-reset-model" ||
             std::string_view(argv[1]) == "--normal-runtime-nonprimary-reset-native-active" ||
             std::string_view(argv[1]) == "--normal-runtime-nonprimary-reset-model-active")) {
            const std::string_view flag(argv[1]);
            check_nonprimary_reset(flag.find("native") != std::string_view::npos,
                                   flag.ends_with("-active"));
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--normal-runtime-nonprimary-reset-probe") {
            run_source_quarantine_child("--normal-runtime-nonprimary-reset-native");
            run_source_quarantine_child("--normal-runtime-nonprimary-reset-model");
            run_source_quarantine_child("--normal-runtime-nonprimary-reset-native-active");
            run_source_quarantine_child("--normal-runtime-nonprimary-reset-model-active");
            run_source_quarantine_child("--cached-nonprimary-reset-native");
            run_source_quarantine_child("--cached-nonprimary-reset-model");
            run_source_quarantine_child("--cached-nonprimary-reset-native-active");
            run_source_quarantine_child("--cached-nonprimary-reset-model-active");
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]).starts_with("--cached-nonprimary-reset-")) {
            const std::string_view flag(argv[1]);
            check_cached_nonprimary_reset(flag.find("native") != std::string_view::npos,
                                          flag.ends_with("-active"));
            return EXIT_SUCCESS;
        }
        if (argc == 2 && (std::string_view(argv[1]) == "--normal-runtime-cold-device-sync-native" ||
                          std::string_view(argv[1]) == "--normal-runtime-cold-device-sync-model")) {
            check_cold_device_sync(std::string_view(argv[1]) ==
                                   "--normal-runtime-cold-device-sync-native");
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--normal-runtime-device-sync-probe") {
            run_source_quarantine_child("--normal-runtime-cold-device-sync-native");
            run_source_quarantine_child("--normal-runtime-cold-device-sync-model");
            check_native_device_sync();
            for (bool is_ptds : {false, true}) {
                for (bool implicit : {false, true}) {
                    for (bool unbound : {false, true}) {
                        RuntimeRecordProbeOptions options;
                        options.is_ptds = is_ptds;
                        options.uses_implicit_stream = implicit;
                        options.is_caller_unbound = unbound;
                        options.has_prior_error = unbound;
                        options.uses_runtime_wait = true;
                        options.uses_queued_release = true;
                        options.uses_device_completion = true;
                        const auto native = check_normal_runtime_pending_completion(
                            true, true, true, unbound, unbound, true, options);
                        const auto logical = check_normal_runtime_pending_completion(
                            false, true, true, unbound, unbound, true, options);
                        require(logical == native,
                                "device completion retains actual SDK error and caller state");
                    }
                }
            }
            check_source_module(true);
            return EXIT_SUCCESS;
        }
        if (argc == 2 &&
            std::string_view(argv[1]) == "--normal-runtime-timing-serialization-probe") {
            RuntimeRecordProbeOptions options;
            options.uses_elapsed_time = true;
            options.uses_timing_interleaving = true;
            const auto native = check_normal_runtime_pending_completion(true, true, true, false,
                                                                        false, true, options);
            const auto model = check_normal_runtime_pending_completion(false, true, true, false,
                                                                       false, true, options);
            require(model.last == native.last && model.caller_context == native.caller_context,
                    "concurrent timing preserves actual native SDK caller observations");
            check_source_module();
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--normal-runtime-event-timing-probe") {
            for (unsigned int timing_case : {0U, 1U, 2U}) {
                for (bool is_ptds : {false, true}) {
                    for (bool uses_implicit : {false, true}) {
                        for (bool unbound : {false, true}) {
                            RuntimeRecordProbeOptions options;
                            options.is_ptds = is_ptds;
                            options.uses_implicit_stream = uses_implicit;
                            options.is_caller_unbound = unbound;
                            options.has_prior_error = unbound;
                            options.uses_elapsed_time = true;
                            options.timing_case = timing_case;
                            const auto native = check_normal_runtime_pending_completion(
                                true, true, true, unbound, unbound, true, options);
                            const auto logical = check_normal_runtime_pending_completion(
                                false, true, true, unbound, unbound, true, options);
                            require(
                                logical == native,
                                "cached Runtime timing preserves native caller, output and TLS");
                        }
                    }
                }
            }
            check_source_module();
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--normal-runtime-event-creation-probe") {
            run_source_quarantine_child("--normal-runtime-cold-create-native");
            run_source_quarantine_child("--normal-runtime-cold-create-model");
            for (bool uses_default_create : {false, true}) {
                for (bool is_ptds : {false, true}) {
                    for (bool uses_implicit : {false, true}) {
                        for (bool unbound : {false, true}) {
                            const RuntimeRecordProbeOptions options{
                                is_ptds, false, uses_implicit, unbound, unbound,
                                true,    false, false,         true,    uses_default_create};
                            const auto native = check_normal_runtime_pending_completion(
                                true, true, true, false, false, true, options);
                            const auto logical = check_normal_runtime_pending_completion(
                                false, true, true, false, false, true, options);
                            require(logical == native,
                                    "cached Runtime creation preserves SDK caller and error state");
                        }
                    }
                }
            }
            run_source_quarantine_child("--normal-runtime-create-context-quarantine");
            run_source_quarantine_child("--normal-runtime-create-throw-quarantine");
            run_source_quarantine_child("--normal-runtime-create-duplicate-quarantine");
            check_source_module();
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--normal-runtime-cold-create-native") {
            check_normal_runtime_cold_creation(true);
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--normal-runtime-cold-create-model") {
            check_normal_runtime_cold_creation(false);
            return EXIT_SUCCESS;
        }
        if (argc == 2 &&
            std::string_view(argv[1]) == "--normal-runtime-create-context-quarantine") {
            check_normal_runtime_creation_quarantine(false, false);
        }
        if (argc == 2 && std::string_view(argv[1]) == "--normal-runtime-create-throw-quarantine") {
            check_normal_runtime_creation_quarantine(true, false);
        }
        if (argc == 2 &&
            std::string_view(argv[1]) == "--normal-runtime-create-duplicate-quarantine") {
            check_normal_runtime_creation_quarantine(false, true);
        }
        if (argc == 2 && std::string_view(argv[1]) == "--normal-runtime-stream-completion-probe") {
            for (bool is_ptds : {false, true}) {
                for (bool uses_flags : {false, true}) {
                    for (bool uses_implicit : {false, true}) {
                        for (bool unbound : {false, true}) {
                            const RuntimeRecordProbeOptions options{
                                is_ptds, uses_flags, uses_implicit, unbound,
                                unbound, true,       true,          true};
                            const auto native = check_normal_runtime_pending_completion(
                                true, true, true, unbound, unbound, true, options);
                            const auto logical = check_normal_runtime_pending_completion(
                                false, true, true, unbound, unbound, true, options);
                            require(
                                logical == native,
                                "cached stream completion preserves SDK caller and error state");
                        }
                    }
                }
            }
            check_source_module();
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--normal-runtime-consumer-wait-probe") {
            for (bool is_ptds : {false, true}) {
                for (bool uses_flags : {false, true}) {
                    for (bool uses_implicit : {false, true}) {
                        for (bool unbound : {false, true}) {
                            const RuntimeRecordProbeOptions options{
                                is_ptds, uses_flags, uses_implicit, unbound, unbound, true};
                            const auto native = check_normal_runtime_pending_completion(
                                true, true, true, false, false, true, options);
                            const auto logical = check_normal_runtime_pending_completion(
                                false, true, true, false, false, true, options);
                            require(logical == native,
                                    "cached Runtime wait preserves original SDK caller state");
                        }
                    }
                }
            }
            run_source_quarantine_child("--normal-runtime-wait-quarantine");
            run_source_quarantine_child("--normal-runtime-wait-throw-quarantine");
            run_source_quarantine_child("--normal-runtime-wait-allocation-quarantine");
            check_source_module();
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--normal-runtime-wait-quarantine") {
            check_normal_runtime_record_quarantine(false, true);
        }
        if (argc == 2 && std::string_view(argv[1]) == "--normal-runtime-wait-throw-quarantine") {
            check_normal_runtime_record_quarantine(true, true);
        }
        if (argc == 2 &&
            std::string_view(argv[1]) == "--normal-runtime-wait-allocation-quarantine") {
            check_normal_runtime_record_quarantine(false, true, true);
        }
        if (argc == 2 && std::string_view(argv[1]) == "--normal-runtime-record-quarantine") {
            check_normal_runtime_record_quarantine();
        }
        if (argc == 2 && std::string_view(argv[1]) == "--normal-runtime-record-throw-quarantine") {
            check_normal_runtime_record_quarantine(true);
        }
        if (argc == 2 &&
            std::string_view(argv[1]) == "--normal-runtime-record-allocation-quarantine") {
            check_normal_runtime_record_quarantine(false, false, true);
        }
        if (argc == 2 && std::string_view(argv[1]) == "--normal-runtime-consumer-record-probe") {
            for (bool is_ptds : {false, true}) {
                for (bool uses_flags : {false, true}) {
                    for (bool uses_implicit : {false, true}) {
                        for (bool unbound : {false, true}) {
                            const RuntimeRecordProbeOptions options{
                                is_ptds, uses_flags, uses_implicit, unbound, unbound};
                            const auto native = check_normal_runtime_pending_completion(
                                true, true, true, false, false, true, options);
                            const auto logical = check_normal_runtime_pending_completion(
                                false, true, true, false, false, true, options);
                            require(logical == native,
                                    "cached consumer record preserves native SDK caller state");
                        }
                    }
                }
            }
            run_source_quarantine_child("--normal-runtime-record-quarantine");
            run_source_quarantine_child("--normal-runtime-record-throw-quarantine");
            run_source_quarantine_child("--normal-runtime-record-allocation-quarantine");
            check_source_module();
            return EXIT_SUCCESS;
        }
        if (argc == 2 &&
            (std::string_view(argv[1]) == "--normal-runtime-pending-completion-probe" ||
             std::string_view(argv[1]) == "--native-runtime-pending-completion-probe" ||
             std::string_view(argv[1]) == "--normal-runtime-record-completion-probe")) {
            const auto result = check_normal_runtime_pending_completion(
                std::string_view(argv[1]) == "--native-runtime-pending-completion-probe",
                std::string_view(argv[1]) == "--normal-runtime-record-completion-probe");
            if (std::string_view(argv[1]) != "--native-runtime-pending-completion-probe") {
                const auto unknown = check_normal_runtime_pending_completion(false, false, false);
                require(unknown == result, "unknown pending event preserves original SDK status");
                const auto native = check_normal_runtime_pending_completion(true);
                require(native == result, "logical Runtime pending query preserves SDK last error");
                for (bool has_prior_error : {false, true}) {
                    const auto unbound = check_normal_runtime_pending_completion(
                        false, false, true, true, has_prior_error);
                    const auto original = check_normal_runtime_pending_completion(
                        true, false, true, true, has_prior_error);
                    require(unbound == original,
                            "logical completion preserves original lazy context and prior error");
                }
                check_source_module();
            }
            return EXIT_SUCCESS;
        }
        if (argc == 2 && (std::string_view(argv[1]) == "--normal-runtime-event-quarantine" ||
                          std::string_view(argv[1]) == "--normal-runtime-stream-quarantine")) {
            check_normal_runtime_retirement_quarantine(std::string_view(argv[1]) ==
                                                       "--normal-runtime-stream-quarantine");
        }
        if (argc == 2 && (std::string_view(argv[1]) == "--normal-runtime-event-destroy-probe" ||
                          std::string_view(argv[1]) == "--normal-runtime-stream-destroy-probe")) {
            const ProcessCallGate::Call call(cooperative_process_call_gate());
            const bool is_stream =
                std::string_view(argv[1]) == "--normal-runtime-stream-destroy-probe";
            for (unsigned int iteration = 0; iteration < 3; ++iteration) {
                for (bool has_binding : {false, true}) {
                    for (bool is_caller_unbound : {false, true}) {
                        const auto native = check_normal_runtime_handle_retirement(
                            is_stream, true, has_binding, is_caller_unbound);
                        const auto source = check_normal_runtime_handle_retirement(
                            is_stream, false, has_binding, is_caller_unbound);
                        require(native == source,
                                "normal Runtime lifetime preserves SDK last-error state");
                    }
                }
            }
            CUcontext context{};
            check(cuCtxGetCurrent(&context), "observe actual normal matrix teardown context");
            auto* router = cooperative_source_router();
            require(router, "normal matrix owner remains admitted");
            check(router->close_context(context, [] { return CUDA_SUCCESS; }),
                  "close same normal actor only after complete real handle-reuse matrix");
            run_source_quarantine_child(is_stream ? "--normal-runtime-stream-quarantine"
                                                  : "--normal-runtime-event-quarantine");
            check_source_module();
            return EXIT_SUCCESS;
        }
        if (argc == 2 &&
            (std::string_view(argv[1]) == "--native-pending-driver-stream-device-probe" ||
             std::string_view(argv[1]) == "--native-pending-runtime-stream-device-probe")) {
            check_pending_source_event_retirement(
                true, std::string_view(argv[1]) == "--native-pending-runtime-stream-device-probe",
                true, true);
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--native-source-capture-join-probe") {
            inspect_source_capture_join(true);
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--source-stream-overlap-origin-probe") {
            const auto native =
                check_source_stream_retirement(true, false, true, true, true, false, true);
            const auto source =
                check_source_stream_retirement(false, false, true, true, true, false, true);
            require(native == source, "isolated overlap origin preserves original SDK results");
            return EXIT_SUCCESS;
        }
        if (argc == 2 &&
            (std::string_view(argv[1]) == "--source-driver-stream-destroy-probe" ||
             std::string_view(argv[1]) == "--source-runtime-stream-destroy-probe" ||
             std::string_view(argv[1]) == "--source-driver-stream-destroy-device-probe" ||
             std::string_view(argv[1]) == "--source-runtime-stream-destroy-device-probe")) {
            const bool uses_runtime =
                std::string_view(argv[1]) == "--source-runtime-stream-destroy-probe" ||
                std::string_view(argv[1]) == "--source-runtime-stream-destroy-device-probe";
            const bool has_instrumentation_release =
                std::string_view(argv[1]).find("-device-probe") != std::string_view::npos;
            for (bool is_origin : {false, true}) {
                for (bool has_joined_member : {false, true}) {
                    for (bool has_binding : {false, true}) {
                        for (bool is_caller_unbound : {false, true}) {
                            if (is_caller_unbound && !uses_runtime) {
                                continue;
                            }
                            const auto native = check_source_stream_retirement(
                                true, uses_runtime, is_origin, has_joined_member, has_binding,
                                is_caller_unbound);
                            const auto source = check_source_stream_retirement(
                                false, uses_runtime, is_origin, has_joined_member, has_binding,
                                is_caller_unbound);
                            require(native == source,
                                    "source destruction preserves actual SDK Runtime last error");
                        }
                    }
                }
            }
            for (bool is_origin : {false, true}) {
                const auto native = check_source_stream_retirement(true, uses_runtime, is_origin,
                                                                   true, true, uses_runtime, true);
                const auto source = check_source_stream_retirement(false, uses_runtime, is_origin,
                                                                   true, true, uses_runtime, true);
                require(native == source,
                        "overlapping capture destruction preserves SDK result and last error");
            }
            check_pending_source_event_retirement(true, uses_runtime, true,
                                                  has_instrumentation_release);
            check_pending_source_event_retirement(false, uses_runtime, true,
                                                  has_instrumentation_release);
            run_source_quarantine_child("--source-stream-release-quarantine");
            if (uses_runtime) {
                run_source_quarantine_child("--source-runtime-stream-context-quarantine");
            }
            check_source_module();
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--source-stream-release-quarantine") {
            check_source_stream_quarantine(false);
        }
        if (argc == 2 &&
            std::string_view(argv[1]) == "--source-runtime-stream-context-quarantine") {
            check_source_stream_quarantine(true);
        }
        if (argc == 2 && std::string_view(argv[1]) == "--source-capture-join-probe") {
            for (bool uses_runtime_record : {false, true}) {
                for (bool uses_record_flags : {false, true}) {
                    for (bool is_ptds : {false, true}) {
                        for (bool uses_runtime_wait : {false, true}) {
                            inspect_source_capture_join(true, uses_runtime_record,
                                                        uses_record_flags, is_ptds,
                                                        uses_runtime_record, uses_runtime_wait);
                            inspect_source_capture_join(false, uses_runtime_record,
                                                        uses_record_flags, is_ptds,
                                                        uses_runtime_record, uses_runtime_wait);
                        }
                    }
                }
            }
            run_source_quarantine_child("--runtime-event-record-context-quarantine");
            run_source_quarantine_child("--runtime-event-record-owner-quarantine");
            check_source_module();
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--runtime-event-record-context-quarantine") {
            check_runtime_event_quarantine(false, true, true);
        }
        if (argc == 2 && std::string_view(argv[1]) == "--runtime-event-record-owner-quarantine") {
            check_runtime_event_quarantine(false, false, true);
        }
        if (argc == 2 && std::string_view(argv[1]) == "--native-source-driver-origin-destroy") {
            inspect_native_source_stream_destroy(false, true);
            return EXIT_FAILURE;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--native-source-driver-null-destroy") {
            inspect_native_source_stream_destroy(false, false, false, true);
            return EXIT_FAILURE;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--native-source-runtime-origin-destroy") {
            inspect_native_source_stream_destroy(true, true);
            return EXIT_FAILURE;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--native-source-driver-unrelated-destroy") {
            inspect_native_source_stream_destroy(false, false);
            return EXIT_FAILURE;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--native-source-runtime-unrelated-destroy") {
            inspect_native_source_stream_destroy(true, false);
            return EXIT_FAILURE;
        }
        if (argc == 2 &&
            std::string_view(argv[1]) == "--native-source-driver-joined-origin-destroy") {
            inspect_native_source_stream_destroy(false, true, true);
            return EXIT_FAILURE;
        }
        if (argc == 2 &&
            std::string_view(argv[1]) == "--native-source-runtime-joined-origin-destroy") {
            inspect_native_source_stream_destroy(true, true, true);
            return EXIT_FAILURE;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--native-source-driver-member-destroy") {
            inspect_native_source_stream_destroy(false, false, true);
            return EXIT_FAILURE;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--native-source-runtime-member-destroy") {
            inspect_native_source_stream_destroy(true, false, true);
            return EXIT_FAILURE;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--runtime-source-event-lifetime-probe") {
            for (const bool is_capture_created : {false, true}) {
                const auto native = check_runtime_source_event_lifetime(true, is_capture_created);
                const auto source = check_runtime_source_event_lifetime(false, is_capture_created);
                require(native.created == source.created && native.destroyed == source.destroyed &&
                            native.ended == source.ended && native.last_error == source.last_error,
                        "Runtime source event lifetime must preserve native SDK results");
            }
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--runtime-source-event-destroy-probe") {
            for (const bool is_capture_created : {false, true}) {
                for (const bool is_context_unbound : {false, true}) {
                    const auto native = check_runtime_source_event_lifetime(
                        true, is_capture_created, true, is_context_unbound);
                    const auto source = check_runtime_source_event_lifetime(
                        false, is_capture_created, true, is_context_unbound);
                    require(native.destroyed == source.destroyed && native.ended == source.ended &&
                                native.last_error == source.last_error,
                            "cached Runtime destruction must retire the exact source owner");
                }
            }
            run_source_quarantine_child("--runtime-event-destroy-context-quarantine");
            run_source_quarantine_child("--runtime-event-destroy-owner-quarantine");
            check_pending_source_event_retirement(true, true);
            check_pending_source_event_retirement(false, true);
            check_source_module();
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--runtime-source-event-create-probe") {
            for (bool is_creation_unbound : {false, true}) {
                for (bool uses_default_create : {false, true}) {
                    const auto native = check_runtime_source_event_lifetime(
                        true, true, false, false, is_creation_unbound, uses_default_create);
                    const auto source = check_runtime_source_event_lifetime(
                        false, true, false, false, is_creation_unbound, uses_default_create);
                    require(
                        native.created == source.created && native.destroyed == source.destroyed &&
                            native.ended == source.ended && native.last_error == source.last_error,
                        "source Runtime creation preserves SDK result and last-error behavior");
                    require(native.rejected_creation == source.rejected_creation &&
                                native.was_failed_output_unchanged ==
                                    source.was_failed_output_unchanged &&
                                native.rejection_last_error == source.rejection_last_error,
                            "SDK failure preserves output and native last-error state");
                }
            }
            run_source_quarantine_child("--runtime-event-create-context-quarantine");
            run_source_quarantine_child("--runtime-event-create-owner-quarantine");
            check_source_module();
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--runtime-event-create-context-quarantine") {
            check_runtime_event_quarantine(true, true);
            return EXIT_FAILURE;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--runtime-event-create-owner-quarantine") {
            check_runtime_event_quarantine(true, false);
            return EXIT_FAILURE;
        }
        if (argc == 2 &&
            std::string_view(argv[1]) == "--runtime-event-destroy-context-quarantine") {
            check_runtime_event_quarantine(false, true);
            return EXIT_FAILURE;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--runtime-event-destroy-owner-quarantine") {
            check_runtime_event_quarantine(false, false);
            return EXIT_FAILURE;
        }
        if (argc == 2 &&
            std::string_view(argv[1]) == "--runtime-source-event-no-context-native-probe") {
            (void)check_runtime_source_event_lifetime(true, false, true, true);
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--source-event-retirement-native-probe") {
            for (const bool has_record : {false, true}) {
                for (const bool is_native_created : {false, true}) {
                    (void)check_source_event_retirement(true, has_record, is_native_created);
                }
            }
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--source-event-retirement-probe") {
            for (const bool has_record : {false, true}) {
                for (const bool is_native_created : {false, true}) {
                    const auto native =
                        check_source_event_retirement(true, has_record, is_native_created);
                    const auto source =
                        check_source_event_retirement(false, has_record, is_native_created);
                    require(source.destroy == native.destroy && source.end == native.end &&
                                source.has_graph == native.has_graph &&
                                source.instantiate == native.instantiate &&
                                source.launch == native.launch,
                            "event retirement must preserve original native lifetime/results");
                }
            }
            run_source_quarantine_child("--source-event-retirement-quarantine");
            check_pending_source_event_retirement(true);
            check_pending_source_event_retirement(false);
            check_source_module();
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--source-event-retirement-quarantine") {
            (void)check_source_event_retirement(false, false, false, true);
            return EXIT_FAILURE;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--source-module-retirement-native-probe") {
            (void)check_source_module_retirement(true);
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--source-module-retirement-probe") {
            const auto native = check_source_module_retirement(true);
            const auto source = check_source_module_retirement(false);
            require(source.unload == native.unload && source.end == native.end &&
                        source.has_graph == native.has_graph,
                    "source retirement must preserve actual native unload/end results");
            check_source_module();
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--runtime-code-ready-probe") {
            check_runtime_code_ready();
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--source-module-native-probe") {
            return check_source_module_native() ? EXIT_SUCCESS : EXIT_FAILURE;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--source-module-probe") {
            check_source_module();
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--source-kernel-missing") {
            check_source_quarantine(SourceFaultKind::kKernelMissing);
        }
        if (argc == 2 && std::string_view(argv[1]) == "--source-memory-throw") {
            check_source_quarantine(SourceFaultKind::kMemoryThrow);
        }
        if (argc == 2 && std::string_view(argv[1]) == "--source-memory-context") {
            check_source_quarantine(SourceFaultKind::kMemoryContextChange);
        }
        if (argc == 2 && std::string_view(argv[1]) == "--source-runtime-memory-context") {
            check_source_quarantine(SourceFaultKind::kRuntimeMemoryContextChange);
        }
        for (const auto* flag : {"--source-kernel-missing", "--source-memory-throw",
                                 "--source-memory-context", "--source-runtime-memory-context"}) {
            run_source_quarantine_child(flag);
        }
#else
        (void)argc;
        (void)argv;
#endif
        run(false, false);
        run(true, false);
        run(true, true);
        run(false, false, cudaStreamCaptureModeRelaxed);
        run(true, false, cudaStreamCaptureModeRelaxed);
        run(true, true, cudaStreamCaptureModeRelaxed);
        check_runtime_error(false);
        check_runtime_error(true);
        check_runtime_error(false, CaptureErrorKind::kContextSync);
        check_runtime_error(true, CaptureErrorKind::kContextSync);
        check_runtime_error(false, CaptureErrorKind::kDriverAllocation);
        check_runtime_error(true, CaptureErrorKind::kDriverAllocation);
        check_runtime_error(false, CaptureErrorKind::kRuntimeAllocation);
        check_runtime_error(true, CaptureErrorKind::kRuntimeAllocation);
#ifdef GLIMMER_HAS_RUNTIME_CAPTURE_KERNEL
        check_runtime_kernel(false, false);
        check_runtime_kernel(true, false);
        check_runtime_kernel(true, true);
        std::cout << "runtime_source_kernel actual_sdk=1 opaque_runtime_stream=1 "
                     "legacy_ptds_null=3 original_kernel_nodes=3 native_grants=3 "
                     "exact_output=6 runtime_extended=1 missing_dispatch_quarantine=1 "
                     "source_memory_quarantine=3\n";
#endif
        std::cout << "runtime_capture_owner actual_sdk=1 typed_origin=1 original_nodes=1 "
                     "original_replay_once=1 source_copies=4 source_events=1 native_grants=8 "
                     "source_memory_quota=1\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
