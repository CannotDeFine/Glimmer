#include "cooperative_ptx_fixture.h"
#include "glimmer/core/scheduler.h"
#include "internal/cuda_operation_queue.h"
#include "internal/cuda_queue_worker.h"
#include "internal/cuda_admission_coordinator.h"
#include "internal/cuda_cooperative_pool.h"
#include "internal/cuda_cooperative_frontend.h"
#include "internal/cuda_event_projection_registry.h"
#include "internal/cuda_native_graph_effects.h"
#include "internal/cuda_native_graph_registry.h"
#include "internal/persistent_kernel.h"
#include "internal/ptx_transform.h"

#include <cuda.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {
using namespace glimmer::interceptor;
constexpr std::array<unsigned int, 3> kGrid{7, 5, 3};
constexpr std::array<unsigned int, 3> kBlock{8, 4, 2};
constexpr std::uint32_t kBlocks = 7 * 5 * 3;
constexpr std::size_t kElements = static_cast<std::size_t>(kBlocks) * 8 * 4 * 2;
constexpr std::size_t kMemoryBytes = kElements * sizeof(std::uint32_t);

void require(bool value, const char* message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}
void check(CUresult result, const char* operation) {
    if (result != CUDA_SUCCESS) {
        throw std::runtime_error(std::string(operation) + ": CUresult " +
                                 std::to_string(static_cast<int>(result)));
    }
}
struct Resources final : CudaQueueReleaseObserver {
    CUcontext context{};
    CUmodule module{};
    CUmodule native_module{};
    CUgraph graph{};
    CUgraph child_graph{};
    CUgraphExec graph_exec{};
    CUstream producer{};
    CUstream consumer{};
    CUstream priority{};
    CUevent begin{};
    CUevent end{};
    CUevent native_fence{};
    CUdeviceptr output{};
    CUdeviceptr mirror{};
    CUdeviceptr counter{};
    CUdeviceptr mapped_state{};
    void* host_state{};
    void* host_values{};

    explicit Resources(bool should_test_defaults) {
        try {
            CUdevice device = 0;
            check(cuInit(0), "initialize queue GPU");
            check(cuDeviceGet(&device, 0), "select queue GPU");
            check(cuCtxCreate(&context, 0, device), "create ordinary queue context");
            check(cuStreamCreate(&producer,
                                 should_test_defaults ? CU_STREAM_DEFAULT : CU_STREAM_NON_BLOCKING),
                  "create producer stream");
            check(cuStreamCreate(&consumer, CU_STREAM_NON_BLOCKING), "create consumer stream");
            check(cuStreamCreate(&priority, CU_STREAM_NON_BLOCKING), "create priority stream");
            check(cuEventCreate(&begin, CU_EVENT_DEFAULT), "create start timing event");
            check(cuEventCreate(&end, CU_EVENT_DEFAULT), "create end timing event");
            check(cuEventCreate(&native_fence, CU_EVENT_DISABLE_TIMING), "create native fence");
            check(cuMemAlloc(&output, kMemoryBytes), "allocate output");
            check(cuMemAlloc(&mirror, kMemoryBytes), "allocate mirror");
            check(cuMemAlloc(&counter, sizeof(std::uint64_t)), "allocate retained progress");
            check(cuMemHostAlloc(&host_state, 16, CU_MEMHOSTALLOC_DEVICEMAP),
                  "allocate retreat and progress state");
            check(cuMemHostGetDevicePointer(&mapped_state, host_state, 0), "map retreat state");
            check(cuMemHostAlloc(&host_values, kMemoryBytes, 0), "allocate pinned readback");
            auto* state = static_cast<std::uint32_t*>(host_state);
            state[0] = 0;
            state[1] = 0;
            std::fill_n(static_cast<std::uint32_t*>(host_values), kElements, 0);
            check(cuMemsetD32Async(mirror, 0, kElements, producer), "initialize native mirror");
            check(cuEventRecord(native_fence, producer), "record preceding native work");
            check(cuMemsetD32(counter, 0, 2), "initialize claim counter");
        } catch (...) {
            cleanup();
            throw;
        }
    }
    ~Resources() override {
        cleanup();
    }
    Resources(const Resources&) = delete;
    Resources& operator=(const Resources&) = delete;
    Resources(Resources&&) = delete;
    Resources& operator=(Resources&&) = delete;
    void confirmed_release(CUdeviceptr pointer) noexcept override {
        if (pointer == output) {
            output = 0;
        }
    }
    void context_destroyed(CUdeviceptr pointer) noexcept override {
        confirmed_release(pointer);
    }
    void cleanup() noexcept {
        if (!context) {
            return;
        }
        if (host_state) {
            std::atomic_ref(static_cast<std::uint32_t*>(host_state)[0])
                .store(1, std::memory_order_release);
        }
        bool passed = cuCtxSynchronize() == CUDA_SUCCESS;
        if (graph_exec) {
            passed &= cuGraphExecDestroy(graph_exec) == CUDA_SUCCESS;
        }
        for (const auto source : {graph, child_graph}) {
            if (source) {
                passed &= cuGraphDestroy(source) == CUDA_SUCCESS;
            }
        }
        for (const auto event : {begin, end, native_fence}) {
            if (event) {
                passed &= cuEventDestroy(event) == CUDA_SUCCESS;
            }
        }
        if (module) {
            passed &= cuModuleUnload(module) == CUDA_SUCCESS;
        }
        if (native_module) {
            passed &= cuModuleUnload(native_module) == CUDA_SUCCESS;
        }
        for (const auto stream : {producer, consumer, priority}) {
            if (stream) {
                passed &= cuStreamDestroy(stream) == CUDA_SUCCESS;
            }
        }
        for (const auto pointer : {output, mirror, counter}) {
            if (pointer) {
                passed &= cuMemFree(pointer) == CUDA_SUCCESS;
            }
        }
        for (const auto host_pointer : {host_state, host_values}) {
            if (host_pointer) {
                passed &= cuMemFreeHost(host_pointer) == CUDA_SUCCESS;
            }
        }
        passed &= cuCtxDestroy(context) == CUDA_SUCCESS;
        context = nullptr;
        if (!passed) {
            std::terminate();
        }
    }
};

class Progress final : public CudaQueueContinuation {
   public:
    explicit Progress(std::shared_ptr<Resources> resources) : resources_(std::move(resources)) {}
    CUresult prepare_launch(std::array<unsigned int, 3>& grid,
                            OwnedKernelArguments&) noexcept override {
        grid = {worker_blocks, 1, 1};
        std::atomic_ref(static_cast<std::uint32_t*>(resources_->host_state)[0])
            .store(0, std::memory_order_release);
        return CUDA_SUCCESS;
    }
    CUresult request_retreat() noexcept override {
        std::atomic_ref(static_cast<std::uint32_t*>(resources_->host_state)[0])
            .store(1, std::memory_order_release);
        return CUDA_SUCCESS;
    }
    CUresult stage_completion(CUstream stream) noexcept override {
        return cuMemcpyDtoHAsync(static_cast<std::uint64_t*>(resources_->host_state) + 1,
                                 resources_->counter, sizeof(std::uint64_t), stream);
    }
    CUresult inspect_stopped(bool& complete) noexcept override {
        claimed_blocks = static_cast<std::uint64_t*>(resources_->host_state)[1];
        complete = claimed_blocks >= kBlocks;
        if (!complete) {
            stopped_partial_blocks = claimed_blocks;
        }
        return CUDA_SUCCESS;
    }
    unsigned int worker_blocks{3};
    std::uint64_t claimed_blocks{};
    std::uint64_t stopped_partial_blocks{};

   private:
    std::shared_ptr<Resources> resources_;
};

struct EventRecordAudit final {
    static inline std::atomic<EventRecordAudit*> active{};
    const CUevent event;
    std::atomic<unsigned int> records{};
    std::atomic<CUstream> last_stream{};
    explicit EventRecordAudit(CUevent handle) : event(handle) {
        EventRecordAudit* expected = nullptr;
        require(active.compare_exchange_strong(expected, this), "one event audit owner at a time");
    }
    ~EventRecordAudit() {
        active.store(nullptr);
    }
    EventRecordAudit(const EventRecordAudit&) = delete;
    EventRecordAudit& operator=(const EventRecordAudit&) = delete;
    EventRecordAudit(EventRecordAudit&&) = delete;
    EventRecordAudit& operator=(EventRecordAudit&&) = delete;
    static CUresult record(CUevent handle, CUstream stream) {
        if (auto* audit = active.load(); audit && audit->event == handle) {
            audit->records.fetch_add(1);
            audit->last_stream.store(stream);
        }
        return cuEventRecord(handle, stream);
    }
};
class ProjectionRecordGuard final : public CudaQueueEventRecordGuard {
   public:
    ProjectionRecordGuard(std::shared_ptr<CudaEventProjectionRegistry> registry_value,
                          CudaEventProjection snapshot_value)
        : registry(std::move(registry_value)), snapshot(std::move(snapshot_value)) {}
    CUresult should_write(bool& output) noexcept override {
        return registry->matches_native_epoch(snapshot, output);
    }
    CUresult prepare_record(CudaEventRecordPreparation& output) noexcept override {
        CudaEventProjection current;
        const auto status = registry->lookup_owned(snapshot, current);
        if (status != CUDA_SUCCESS) {
            return status;
        }
        output = {};
        if (!current.is_logical()) {
            const auto& issued = current.issued_native_prefix();
            if (!issued && !current.has_native_completion()) {
                return CUDA_ERROR_NOT_READY;
            }
            output.should_order_previous = false;
            output.native_prefix = issued ? *issued : CudaOperationToken{};
        }
        snapshot = std::move(current);
        prefix = output.native_prefix;
        return CUDA_SUCCESS;
    }
    CUresult record_accepted() noexcept override {
        return registry->publish_logical(snapshot, prefix ? &prefix : nullptr);
    }

   private:
    const std::shared_ptr<CudaEventProjectionRegistry> registry;
    CudaEventProjection snapshot;
    CudaOperationToken prefix;
};
struct ProjectionCompletion final : CudaQueueNativeFenceObserver {
    const std::shared_ptr<CudaEventProjectionRegistry> registry;
    std::vector<CudaEventProjection> epochs;
    std::atomic<unsigned int> calls{};
    const std::shared_ptr<CudaOperationToken> prefix{std::make_shared<CudaOperationToken>()};
    explicit ProjectionCompletion(std::shared_ptr<CudaEventProjectionRegistry> value)
        : registry(std::move(value)) {}
    CUresult native_reservation_accepted(const CudaOperationToken& token) noexcept override {
        if (*prefix || !token) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        *prefix = token;
        return CUDA_SUCCESS;
    }
    CUresult confirmed_native_completion() noexcept override {
        const auto status = registry->complete_native_events(epochs);
        calls.fetch_add(1);
        return status;
    }
};
CudaQueueDispatch native_dispatch() {
    return {cuCtxGetCurrent,   cuLaunchKernel, cuMemcpyHtoDAsync,   cuMemcpyDtoHAsync,
            cuMemcpyDtoDAsync, cuMemFreeAsync, cuEventCreate,       cuEventRecord,
            cuEventQuery,      cuEventDestroy, cuStreamSynchronize, cuStreamWaitEvent,
            cuEventSynchronize};
}
CudaCooperativePoolDispatch native_pool_dispatch(const std::shared_ptr<Resources>& resources) {
    return {cuCtxGetCurrent,
            cuMemAlloc,
            [](CUdeviceptr pointer) {
                const auto result = cuMemFree(pointer);
                return CudaCooperativeReleaseResult{
                    result, result == CUDA_SUCCESS ? CudaNativeReleaseState::kReleased
                                                   : CudaNativeReleaseState::kUncertain};
            },
            cuMemHostAlloc,
            [](void* pointer) {
                const auto result = cuMemFreeHost(pointer);
                return CudaCooperativeReleaseResult{
                    result, result == CUDA_SUCCESS ? CudaNativeReleaseState::kReleased
                                                   : CudaNativeReleaseState::kUncertain};
            },
            cuMemHostGetDevicePointer,
            cuMemcpyDtoHAsync,
            resources};
}
DriverFunctionTable metadata_dispatch() {
    DriverFunctionTable table;
    table.module_load_data = cuModuleLoadData;
    table.module_load_data_ex = cuModuleLoadDataEx;
    table.module_get_function = cuModuleGetFunction;
    table.module_unload = cuModuleUnload;
    table.context_get_current = cuCtxGetCurrent;
    table.context_get_device = cuCtxGetDevice;
    table.device_get_attribute = cuDeviceGetAttribute;
    table.stream_capture_status = cuStreamIsCapturing;
    table.occupancy = cuOccupancyMaxActiveBlocksPerMultiprocessor;
    table.launch_kernel = cuLaunchKernel;
    return table;
}
template <typename Predicate>
void pump_until(CudaOperationQueue& queue, Predicate predicate, const char* failure,
                const CudaOperationToken& admitted = {}) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{10};
    while (!predicate() && std::chrono::steady_clock::now() < deadline) {
        std::vector<CudaOperationToken> selection;
        for (const auto& candidate : queue.kernel_candidates()) {
            if (candidate.token == admitted && candidate.state == OrderedOperationState::kQueued) {
                selection.push_back(candidate.token);
            }
        }
        check(queue.pump_admitted(selection), "pump admitted queue");
        std::this_thread::yield();
    }
    require(predicate(), failure);
}

void run(bool should_test_defaults) {
    auto resources = std::make_shared<Resources>(should_test_defaults);
    DriverDispatch driver(metadata_dispatch());
    PersistentKernelRuntime kernels(driver, 3, false, true);
    check(kernels.load_module(&resources->module, kCooperativePtx),
          "load owned cooperative module");
    CUfunction original = nullptr;
    check(kernels.get_function(&original, resources->module, "cooperative"),
          "resolve original kernel");
    const auto prepared =
        kernels.cooperative_kernel(original, kGrid, kBlock, 0, resources->producer, false);
    require(prepared.has_value(), "ordinary original kernel must prepare cooperative metadata");
    auto marker = resources->mapped_state + sizeof(std::uint32_t);
    auto grid = kGrid;
    std::uint32_t work = 50000;
    std::array<void*, 8> argument_addresses{
        &resources->output,      &marker, &work, &grid[0], &grid[1], &grid[2], &resources->counter,
        &resources->mapped_state};
    auto layout = prepared->parameter_layout;
    layout.insert(layout.end(), {{4, 4}, {4, 4}, {4, 4}, {8, 8}, {8, 8}});
    auto arguments = OwnedKernelArguments::copy(layout, argument_addresses.data());
    require(arguments.has_value(), "queue must own complete original/private argument ABI");
    argument_addresses.fill(nullptr);
    work = 0;
    grid.fill(0);

    auto progress = std::make_shared<Progress>(resources);
    CudaOperationQueue queue(resources->context, native_dispatch());
    const auto producer = queue.bind_stream(
        resources->producer,
        should_test_defaults ? OrderedStreamKind::kBlocking : OrderedStreamKind::kNonBlocking,
        resources);
    const auto consumer =
        queue.bind_stream(should_test_defaults ? CU_STREAM_PER_THREAD : resources->consumer,
                          should_test_defaults ? OrderedStreamKind::kPerThreadDefault
                                               : OrderedStreamKind::kNonBlocking,
                          resources);
    const auto priority =
        queue.bind_stream(resources->priority, OrderedStreamKind::kNonBlocking, resources);
    const auto begin = queue.bind_event(resources->begin, resources);
    const auto end = queue.bind_event(resources->end, resources);
    require(begin && end && queue.query_event(end) == CUDA_SUCCESS, "bind native timing events");
    const auto imported =
        queue.submit(producer, CudaQueuedNativeFence{resources->native_fence, resources});
    const auto begin_record = queue.submit(producer, CudaQueuedEventRecord{begin});
    const auto initialize =
        queue.submit(producer, CudaQueuedCopy{CudaQueuedCopyKind::kHostToDevice, resources->output,
                                              0, resources->host_values, kMemoryBytes, resources});
    const auto low =
        queue.submit(producer, CudaQueuedKernel{prepared->function, kGrid, kBlock, 0,
                                                std::move(*arguments), resources, progress});
    const auto record = queue.submit(producer, CudaQueuedEventRecord{end});
    const auto wait = queue.wait_event(consumer, end);
    const auto device_copy = queue.submit(
        consumer, CudaQueuedCopy{CudaQueuedCopyKind::kDeviceToDevice, resources->mirror,
                                 resources->output, nullptr, kMemoryBytes, resources});
    const auto host_copy = queue.submit(
        consumer, CudaQueuedCopy{CudaQueuedCopyKind::kDeviceToHost, 0, resources->mirror,
                                 resources->host_values, kMemoryBytes, resources});
    const auto release =
        queue.submit(consumer, CudaQueuedFree{resources->output, resources, resources});
    require(imported && begin_record && initialize && low && record && wait && device_copy &&
                host_copy && release,
            "all owned GPU operations must enqueue");
    CudaOperationToken legacy_boundary;
    if (should_test_defaults) {
        const auto legacy =
            queue.bind_stream(nullptr, OrderedStreamKind::kLegacyDefault, resources);
        legacy_boundary = queue.submit(legacy, CudaQueuedMarker{});
        require(static_cast<bool>(legacy_boundary), "legacy boundary must enqueue");
    }
    const auto context_snapshot = queue.context_snapshot();

    glimmer::core::Scheduler scheduler(
        4, {.scheduling_policy = glimmer::core::SchedulingPolicy::kPriority});
    const auto low_task = scheduler.submit({.tenant_id = "low", .memory_bytes = 1, .priority = 1});
    require(low_task.accepted() && scheduler.dispatch_task(low_task.task_id).has_value(),
            "low scheduler execution must dispatch");
    auto* state = static_cast<std::uint32_t*>(resources->host_state);
    pump_until(
        queue, [&] { return std::atomic_ref(state[1]).load(std::memory_order_acquire) == 1; },
        "GPU must publish partial progress", low);
    check(queue.query(imported), "prove preceding native work completed");
    const auto high_task =
        scheduler.submit({.tenant_id = "high", .memory_bytes = 1, .priority = 9});
    require(high_task.accepted() && scheduler.retreat_requested(low_task.task_id),
            "higher priority work must request retreat");
    check(queue.request_retreat(low), "request physical retreat");
    pump_until(
        queue, [&] { return queue.state(low) == OrderedOperationState::kSuspended; },
        "event-backed stop must suspend logical execution");
    require(progress->claimed_blocks > 0 && progress->claimed_blocks < kBlocks &&
                queue.query(record) == CUDA_ERROR_NOT_READY &&
                queue.query(wait) == CUDA_ERROR_NOT_READY &&
                queue.query(device_copy) == CUDA_ERROR_NOT_READY &&
                queue.query(host_copy) == CUDA_ERROR_NOT_READY &&
                queue.query(release) == CUDA_ERROR_NOT_READY &&
                (!legacy_boundary || queue.query(legacy_boundary) == CUDA_ERROR_NOT_READY),
            "retreat must retain partial progress and every dependent operation");
    require(queue.query_event(end) == CUDA_ERROR_NOT_READY &&
                queue.query_native_boundary(context_snapshot) == CUDA_ERROR_NOT_READY &&
                cuEventQuery(resources->end) == CUDA_SUCCESS,
            "unissued native event completion cannot substitute for logical completion");
    std::vector<std::uint32_t> untouched(kElements);
    check(cuMemcpyDtoH(untouched.data(), resources->mirror, kMemoryBytes),
          "inspect deferred mirror");
    require(std::ranges::all_of(untouched, [](auto value) { return value == 0; }) &&
                std::ranges::all_of(
                    std::span(static_cast<std::uint32_t*>(resources->host_values), kElements),
                    [](auto value) { return value == 0; }),
            "device and host copies must not physically issue while producer is paused");
    CUdeviceptr allocation_base = 0;
    std::size_t allocation_bytes = 0;
    check(cuMemGetAddressRange(&allocation_base, &allocation_bytes, resources->output),
          "prove deferred source allocation remains live");
    require(allocation_base == resources->output && allocation_bytes >= kMemoryBytes &&
                scheduler.suspend(low_task.task_id) &&
                scheduler.dispatch_task(high_task.task_id).has_value(),
            "only proven physical stop may release scheduler execution capacity");

    CUfunction probe = nullptr;
    check(cuModuleGetFunction(&probe, resources->module, "priority_probe"),
          "resolve priority probe");
    std::array<void*, 1> probe_addresses{&marker};
    const std::array<PtxParameterLayout, 1> probe_layout{{{8, 8}}};
    auto probe_arguments = OwnedKernelArguments::copy(probe_layout, probe_addresses.data());
    require(probe_arguments.has_value(), "priority arguments must be owned");
    const auto high = queue.submit(
        priority, CudaQueuedKernel{
                      probe, {1, 1, 1}, {1, 1, 1}, 0, std::move(*probe_arguments), resources, {}});
    require(static_cast<bool>(high), "high priority CUDA operation must enqueue");
    pump_until(
        queue, [&] { return queue.query(high) == CUDA_SUCCESS; },
        "high priority GPU operation must finish before resumption", high);
    require(std::atomic_ref(state[1]).load(std::memory_order_acquire) == 2 &&
                queue.state(low) == OrderedOperationState::kSuspended &&
                scheduler.complete(high_task.task_id) &&
                scheduler.dispatch_task(low_task.task_id).has_value(),
            "high GPU work must overtake, not complete, the paused producer");

    const auto stopped_blocks = progress->claimed_blocks;
    progress->worker_blocks = 7;
    require(queue.resume(low), "logical producer must resume with retained progress");
    pump_until(
        queue,
        [&] {
            return queue.query(release) == CUDA_SUCCESS &&
                   (!legacy_boundary || queue.query(legacy_boundary) == CUDA_SUCCESS);
        },
        "resumed logical operation and dependent operations must drain", low);
    require(resources->output == 0,
            "confirmed queued free must retire the RAII handle before resource cleanup");
    require(std::ranges::all_of(
                std::span(static_cast<std::uint32_t*>(resources->host_values), kElements),
                [](auto value) { return value == 1; }) &&
                scheduler.complete(low_task.task_id) && scheduler.usage().used_bytes() == 0 &&
                queue.pending_count() == 0,
            "every logical thread must execute once before copy/free and scheduler completion");
    for (const auto& snapshot : context_snapshot) {
        check(queue.query(snapshot), "logical context synchronization snapshot");
    }
    check(queue.query_native_boundary(context_snapshot), "logical native-boundary readiness");
    check(queue.query_event(end), "logical public-handle record completion");
    check(cuEventQuery(resources->end), "physical public-handle record completion");
    float milliseconds = 0;
    check(cuEventElapsedTime(&milliseconds, resources->begin, resources->end),
          "native timing events");
    require(milliseconds >= 0, "event timing must remain available without a fixed duration claim");
    check(queue.shutdown(), "drain and destroy queue events");
    std::cout << "cuda_queue_retreat stopped_blocks=" << stopped_blocks
              << " total_blocks=" << kBlocks
              << " resumed_workers=7 defaults=" << should_test_defaults << " exactly_once=1\n";
}

class WorkerCoordinator final : public CudaQueueCoordinator {
   public:
    CUresult progress(CudaOperationQueue& queue) noexcept override {
        try {
            std::vector<CudaOperationToken> selection;
            for (const auto& candidate : queue.kernel_candidates()) {
                if (candidate.state == OrderedOperationState::kQueued) {
                    selection.push_back(candidate.token);
                }
            }
            return queue.pump_admitted(selection);
        } catch (...) {
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
    }
    void stopped(CudaOperationQueue&, CUresult result) noexcept override {
        stop_result = result;
    }
    CUresult stop_result{CUDA_ERROR_NOT_READY};
};

void autonomous_worker() {
    auto resources = std::make_shared<Resources>(false);
    check(cuModuleLoadData(&resources->module, kCooperativePtx), "load worker original module");
    CUfunction probe = nullptr;
    check(cuModuleGetFunction(&probe, resources->module, "priority_probe"), "resolve worker probe");
    auto marker = resources->mapped_state + sizeof(std::uint32_t);
    std::array<void*, 1> addresses{&marker};
    const std::array<PtxParameterLayout, 1> layout{{{8, 8}}};
    auto arguments = OwnedKernelArguments::copy(layout, addresses.data());
    require(arguments.has_value(), "worker must own application argument bytes");
    auto coordinator = std::make_shared<WorkerCoordinator>();
    CudaQueueWorker worker(resources->context,
                           {native_dispatch(), cuCtxPushCurrent, cuCtxPopCurrent}, coordinator);
    check(worker.status(), "activate autonomous queue context");
    CudaNativeBoundaryToken native_boundary;
    check(worker.try_pause_native(native_boundary), "pause empty native boundary");
    check(cuMemsetD32Async(resources->mirror, 0, kElements, resources->priority),
          "perform original caller native operation");
    check(cuStreamBeginCapture(resources->priority, CU_STREAM_CAPTURE_MODE_GLOBAL),
          "begin original caller global capture while owner is paused");
    check(
        cuLaunchKernel(probe, 1, 1, 1, 1, 1, 1, 0, resources->priority, addresses.data(), nullptr),
        "capture original kernel without internal queue nodes");
    CUgraph graph = nullptr;
    check(cuStreamEndCapture(resources->priority, &graph), "end original caller capture");
    std::size_t graph_nodes = 0;
    check(cuGraphGetNodes(graph, nullptr, &graph_nodes), "inspect native captured graph");
    require(graph_nodes == 1, "capture must contain only the original application kernel");
    check(cuGraphDestroy(graph), "release captured native graph");
    check(cuEventRecord(resources->native_fence, resources->priority),
          "record actual trailing native completion");
    OrderedStreamToken stream;
    check(worker.finish_native(
              native_boundary,
              [&](CudaOperationQueue& queue) {
                  stream = queue.bind_stream(resources->priority, OrderedStreamKind::kNonBlocking,
                                             resources);
                  return stream && queue.submit(
                                       stream,
                                       CudaQueuedNativeFence{resources->native_fence, resources})
                             ? CUDA_SUCCESS
                             : CUDA_ERROR_UNKNOWN;
              }),
          "import caller native fence before autonomous issue");
    CudaOperationToken token;
    check(worker.invoke([&](CudaOperationQueue& queue) {
        token = queue.submit(
            stream,
            CudaQueuedKernel{probe, {1, 1, 1}, {1, 1, 1}, 0, std::move(*arguments), resources, {}});
        return token ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
    }),
          "accept autonomous launch");
    marker = 0;
    addresses.fill(nullptr);
    // No caller pump or query drives the GPU. Only the owning background thread
    // can issue the queued launch after command acceptance.
    auto* state = static_cast<std::uint32_t*>(resources->host_state);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{10};
    while (std::atomic_ref(state[1]).load(std::memory_order_acquire) != 2 &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    require(std::atomic_ref(state[1]).load(std::memory_order_acquire) == 2,
            "GPU launch must progress after API return without caller pumping");
    CUcontext caller_context = nullptr;
    check(cuCtxGetCurrent(&caller_context), "inspect caller context");
    require(caller_context == resources->context, "worker must not alter caller context stack");
    check(worker.shutdown(), "join autonomous owner after physical drain");
    require(coordinator->stop_result == CUDA_SUCCESS, "coordinator must receive real stop proof");
    std::cout << "cuda_queue_worker autonomous=1 owned_arguments=1 caller_context_preserved=1 "
                 "native_boundary=1\n";
}
void autonomous_preemption(bool is_native_priority) {
    auto resources = std::make_shared<Resources>(false);
    DriverDispatch driver(metadata_dispatch());
    PersistentKernelRuntime kernels(driver, 3, false, true);
    check(kernels.load_module(&resources->module, kCooperativePtx),
          "load autonomous cooperative module");
    CUfunction original = nullptr;
    check(kernels.get_function(&original, resources->module, "cooperative"),
          "resolve autonomous original");
    const auto prepared =
        kernels.cooperative_kernel(original, kGrid, kBlock, 0, resources->producer, false);
    require(prepared.has_value(), "autonomous original must prepare");
    auto marker = resources->mapped_state + sizeof(std::uint32_t);
    std::uint32_t work = 50000;
    std::array<void*, 3> addresses{&resources->output, &marker, &work};
    auto pool =
        std::make_shared<CudaCooperativePool>(resources->context, native_pool_dispatch(resources));
    check(pool->initialize(), "prime framework-owned bounded progress pool");
    auto launch = pool->prepare(*prepared, kGrid, kBlock, addresses.data(), resources);
    require(launch.has_value(), "framework must own private ABI without application progress");
    auto progress = launch->continuation;
    // Resolve and own all priority metadata before the low kernel starts. Tool
    // instrumentation may otherwise delay function preparation until low ends.
    CUfunction probe = nullptr;
    check(cuModuleGetFunction(&probe, resources->module, "priority_probe"),
          "resolve autonomous high probe");
    std::array<void*, 1> probe_addresses{&marker};
    const std::array<PtxParameterLayout, 1> probe_layout{{{8, 8}}};
    auto probe_arguments = OwnedKernelArguments::copy(probe_layout, probe_addresses.data());
    require(probe_arguments.has_value(), "autonomous high arguments must be owned");
    glimmer::control::LaunchGateOptions options;
    options.scheduling_policy = glimmer::core::SchedulingPolicy::kPriority;
    auto control = std::make_shared<glimmer::control::AsyncCooperativeLaunchGate>(options);
    auto coordinator = std::make_shared<CudaAdmissionCoordinator>(control);
    CudaQueueWorker worker(resources->context,
                           {native_dispatch(), cuCtxPushCurrent, cuCtxPopCurrent}, coordinator);
    check(worker.status(), "activate autonomous preemption owner");
    OrderedStreamToken priority;
    CudaOperationToken readback;
    check(worker.invoke([&](CudaOperationQueue& queue) {
        const auto producer =
            queue.bind_stream(resources->producer, OrderedStreamKind::kNonBlocking, resources);
        priority =
            queue.bind_stream(resources->priority, OrderedStreamKind::kNonBlocking, resources);
        const auto fence =
            queue.submit(producer, CudaQueuedNativeFence{resources->native_fence, resources});
        const auto initialize = queue.submit(
            producer, CudaQueuedCopy{CudaQueuedCopyKind::kHostToDevice, resources->output, 0,
                                     resources->host_values, kMemoryBytes, resources});
        const auto initialize_progress = queue.submit(producer, std::move(launch->initialization));
        const auto low = queue.submit(producer, std::move(launch->kernel));
        readback = queue.submit(
            producer, CudaQueuedCopy{CudaQueuedCopyKind::kDeviceToHost, 0, resources->output,
                                     resources->host_values, kMemoryBytes, resources});
        if (!low || !coordinator->track(queue, low, {.relative_priority = 1})) {
            if (low) {
                static_cast<void>(queue.cancel_stopped(low));
            }
            return CUDA_ERROR_UNKNOWN;
        }
        return fence && initialize && initialize_progress && priority && readback
                   ? CUDA_SUCCESS
                   : CUDA_ERROR_UNKNOWN;
    }),
          "accept autonomous low launch and ordered readback");
    addresses.fill(nullptr);
    launch.reset();
    work = 0;
    auto* state = static_cast<std::uint32_t*>(resources->host_state);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{10};
    while (std::atomic_ref(state[1]).load(std::memory_order_acquire) != 1 &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    require(std::atomic_ref(state[1]).load(std::memory_order_acquire) == 1,
            "autonomous low must progress without caller pumping");
    CudaOperationToken high;
    check(worker.invoke([&](CudaOperationQueue& queue) {
        high = is_native_priority
                   ? queue.submit(priority, CudaQueuedNativeReservation{resources})
                   : queue.submit(priority, CudaQueuedKernel{probe,
                                                             {1, 1, 1},
                                                             {1, 1, 1},
                                                             0,
                                                             std::move(*probe_arguments),
                                                             resources,
                                                             {}});
        if (!high || !coordinator->track(queue, high, {.relative_priority = 9})) {
            if (high) {
                static_cast<void>(queue.cancel_stopped(high));
            }
            return CUDA_ERROR_UNKNOWN;
        }
        return CUDA_SUCCESS;
    }),
          "accept autonomous priority arrival");
    if (is_native_priority) {
        std::optional<glimmer::control::TaskExecutionIdentity> grant;
        while (!grant && std::chrono::steady_clock::now() < deadline) {
            check(worker.invoke([&](CudaOperationQueue& queue) {
                grant = coordinator->native_reservation_grant(queue, high);
                return CUDA_SUCCESS;
            }),
                  "poll native priority generation without GPU wait");
            std::this_thread::yield();
        }
        require(grant.has_value(), "native priority must obtain the shared generation grant");
        CudaNativeBoundaryToken boundary;
        CUresult paused = CUDA_ERROR_NOT_READY;
        while (paused == CUDA_ERROR_NOT_READY && std::chrono::steady_clock::now() < deadline) {
            paused = worker.try_pause_native_reservation(high, boundary);
            std::this_thread::yield();
        }
        check(paused, "pause physical owner while preserving suspended logical low");
        check(cuLaunchKernel(probe, 1, 1, 1, 1, 1, 1, 0, resources->priority,
                             probe_addresses.data(), nullptr),
              "caller issue original priority once");
        check(cuEventRecord(resources->native_fence, resources->priority),
              "record original priority physical completion");
        check(worker.finish_native(boundary,
                                   [&](CudaOperationQueue& queue) {
                                       return coordinator->commit_native_reservation(
                                                  queue, high, *grant,
                                                  {resources->native_fence, resources})
                                                  ? CUDA_SUCCESS
                                                  : CUDA_ERROR_UNKNOWN;
                                   }),
              "replace original priority reservation without changing dependent token");
    }
    marker = 0;
    probe_addresses.fill(nullptr);
    while (std::atomic_ref(state[1]).load(std::memory_order_acquire) != 2 &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    require(std::atomic_ref(state[1]).load(std::memory_order_acquire) == 2,
            "control arrival must automatically pause low and issue priority work");
    CUresult completion = CUDA_ERROR_NOT_READY;
    CudaCoordinationStats stats;
    while (completion == CUDA_ERROR_NOT_READY && std::chrono::steady_clock::now() < deadline) {
        check(worker.invoke([&](CudaOperationQueue& queue) {
            stats = coordinator->stats();
            completion = queue.query(readback);
            if (completion == CUDA_SUCCESS && stats.acknowledged_completions != 2) {
                completion = CUDA_ERROR_NOT_READY;
            }
            return CUDA_SUCCESS;
        }),
              "query autonomous logical completion");
        std::this_thread::yield();
    }
    check(completion, "complete autonomous restored low and deferred readback");
    check(worker.shutdown(), "drain autonomous physical ownership");
    check(coordinator->control_status(), "acknowledge autonomous terminal control ownership");
    control->shutdown();
    const auto physical_progress = progress->progress();
    progress.reset();
    require(pool->occupied_slots() == 0, "actual drain must release framework progress ownership");
    check(pool->release(), "release framework progress after proven physical drain");
    const auto has_exact_output = std::ranges::all_of(
        std::span(static_cast<std::uint32_t*>(resources->host_values), kElements),
        [](auto value) { return value == 1; });
    std::cout << "cuda_queue_preemption autonomous=1 framework_progress=1 stopped_blocks="
              << physical_progress.partial_blocks << " total_blocks=" << kBlocks
              << " grants=" << stats.granted_executions << " retreat=" << stats.retreat_requests
              << " suspend=" << stats.acknowledged_suspensions
              << " resume=" << stats.logical_resumes
              << " launches=" << physical_progress.physical_launches
              << " native_priority=" << is_native_priority << " exactly_once=" << has_exact_output
              << '\n';
    require(
        stats.granted_executions == 3 && stats.retreat_requests == 1 &&
            stats.acknowledged_suspensions == 1 && stats.logical_resumes == 1 &&
            physical_progress.partial_blocks > 0 && physical_progress.partial_blocks < kBlocks &&
            physical_progress.physical_launches == 2 && has_exact_output,
        "automatic generation handoff must retain partial progress and execute every thread once");
}

void autonomous_native_execution() {
    auto resources = std::make_shared<Resources>(false);
    check(cuModuleLoadData(&resources->module, kNativeOncePtx),
          "load original native execution module");
    CUfunction original = nullptr;
    check(cuModuleGetFunction(&original, resources->module, "native_once"),
          "resolve original native once kernel");
    auto marker = resources->mapped_state + sizeof(std::uint32_t);
    std::array<void*, 1> arguments{&marker};
    auto control = std::make_shared<glimmer::control::AsyncCooperativeLaunchGate>();
    auto coordinator = std::make_shared<CudaAdmissionCoordinator>(control);
    CudaQueueWorker worker(resources->context,
                           {native_dispatch(), cuCtxPushCurrent, cuCtxPopCurrent}, coordinator);
    check(worker.status(), "activate native execution coordinator");
    OrderedStreamToken stream;
    check(worker.invoke([&](CudaOperationQueue& queue) {
        stream = queue.bind_stream(resources->priority, OrderedStreamKind::kNonBlocking, resources);
        return stream ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
    }),
          "bind native execution stream");
    for (const bool should_fail : {false, true}) {
        CudaNativeBoundaryToken boundary;
        check(worker.try_pause_native(boundary), "pause for caller-issued native execution");
        const auto task = control->submit({}).get();
        require(task.has_value(), "native execution must use the shared cooperative gate");
        const auto lease = control->claim(*task).get();
        require(lease.status == glimmer::control::CooperativeClaimStatus::kGranted,
                "caller must own an exact generation before native issue");
        const auto native_status = cuLaunchKernel(original, should_fail ? 0 : 1, 1, 1, 1, 1, 1, 0,
                                                  resources->priority, arguments.data(), nullptr);
        require(native_status == (should_fail ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS),
                "original caller native status must remain unchanged");
        check(cuEventRecord(resources->native_fence, resources->priority),
              "record actual native execution stop fence");
        CudaOperationToken token;
        check(worker.finish_native(
                  boundary,
                  [&](CudaOperationQueue& queue) {
                      token = queue.submit(
                          stream, CudaQueuedNativeFence{resources->native_fence, resources});
                      return token && coordinator->adopt_native_execution(
                                          queue, token, lease.execution, native_status)
                                 ? CUDA_SUCCESS
                                 : CUDA_ERROR_UNKNOWN;
                  }),
              "adopt caller-issued native generation before resuming progress");
        CUresult result = CUDA_ERROR_NOT_READY;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{10};
        while (result == CUDA_ERROR_NOT_READY && std::chrono::steady_clock::now() < deadline) {
            check(worker.invoke([&](CudaOperationQueue& queue) {
                result = queue.query(token);
                const auto stats = coordinator->stats();
                if (result == CUDA_SUCCESS && (should_fail ? stats.acknowledged_failures != 1
                                                           : stats.acknowledged_completions != 1)) {
                    result = CUDA_ERROR_NOT_READY;
                }
                return CUDA_SUCCESS;
            }),
                  "query adopted native execution completion");
            std::this_thread::yield();
        }
        check(result, "prove actual caller-issued native execution stop");
    }
    check(worker.shutdown(), "drain adopted native ownership");
    check(coordinator->control_status(), "acknowledge original native status reports");
    control->shutdown();
    const auto stats = coordinator->stats();
    require(static_cast<std::uint32_t*>(resources->host_state)[1] == 1 &&
                stats.granted_executions == 2 && stats.acknowledged_completions == 1 &&
                stats.acknowledged_failures == 1 && stats.retreat_requests == 0,
            "native execution must issue once and preserve failed native launch reporting");
    std::cout << "cuda_queue_native_execution caller_issued=1 generations=2 completed=1 failed=1 "
                 "exactly_once=1\n";
}
void cooperative_frontend(bool is_native_priority, bool should_supersede_event = false,
                          bool is_graph_priority = false, bool should_takeover_early = false) {
    auto resources = std::make_shared<Resources>(false);
    check(cuStreamDestroy(resources->priority), "replace original priority stream");
    resources->priority = nullptr;
    int least = 0;
    int greatest = 0;
    check(cuCtxGetStreamPriorityRange(&least, &greatest), "read native stream priority range");
    require(greatest < 0, "frontend fixture requires a distinct native high priority");
    check(cuStreamCreateWithPriority(&resources->priority, CU_STREAM_NON_BLOCKING, greatest),
          "create original high-priority application stream");
    DriverDispatch driver(metadata_dispatch());
    PersistentKernelRuntime kernels(driver, 3, false, true);
    check(kernels.load_module(&resources->module, kCooperativePtx), "prepare frontend PTX module");
    CUfunction original = nullptr;
    CUfunction probe = nullptr;
    check(kernels.get_function(&original, resources->module, "cooperative"),
          "resolve frontend low");
    check(kernels.get_function(&probe, resources->module, "priority_probe"),
          "resolve frontend high");
    const auto low =
        kernels.cooperative_kernel(original, kGrid, kBlock, 0, resources->producer, false);
    const auto high =
        kernels.cooperative_kernel(probe, {1, 1, 1}, {1, 1, 1}, 0, resources->priority, false);
    require(low && high, "frontend must obtain both variants before physical low execution");
    glimmer::control::LaunchGateOptions options;
    options.scheduling_policy = glimmer::core::SchedulingPolicy::kPriority;
    auto gate = std::make_shared<glimmer::control::AsyncCooperativeLaunchGate>(options);
    EventRecordAudit audit(resources->end);
    auto queue_dispatch = native_dispatch();
    queue_dispatch.event_record = EventRecordAudit::record;
    auto registry = std::make_shared<CudaEventProjectionRegistry>();
    check(registry->register_event(resources->context, resources->end, true),
          "register actual event epoch");
    check(registry->register_event(resources->context, resources->begin, true),
          "register unrelated logical event projection");
    check(registry->register_event(resources->context, resources->native_fence, false),
          "register unrelated native epoch without completion publication");
    CudaEventProjection projection;
    check(registry->lookup(resources->end, projection), "snapshot logical event write ownership");
    CudaCooperativeFrontend frontend(
        resources->context,
        {{queue_dispatch, cuCtxPushCurrent, cuCtxPopCurrent},
         native_pool_dispatch(resources),
         cuStreamCreate,
         cuStreamDestroy,
         cuStreamGetFlags,
         cuStreamGetPriority},
        gate, {.progress_slots = 8, .import_events = 8, .per_thread_streams = 2});
    check(frontend.initialize(), "initialize context-local frontend resources");
    check(frontend.bind_event(resources->end, resources), "bind original completion event");
    // The frontend owns every private argument, counter and retreat flag. The
    // fixture supplies only the original kernel ABI, stream and borrowed buffers.
    auto marker = resources->mapped_state + sizeof(std::uint32_t);
    CudaNativeGraphEffects graph_effects;
    CudaNativeGraphRegistry graph_registry;
    CudaNativeGraphExecutable graph_owner;
    unsigned int graph_launches{};
    if (is_graph_priority) {
        check(cuModuleLoadData(&resources->native_module, kNativeOncePtx),
              "load original native graph counter kernel");
        CUfunction graph_kernel{};
        check(cuModuleGetFunction(&graph_kernel, resources->native_module, "native_once"),
              "resolve original untransformed graph counter function");
        check(cuGraphCreate(&resources->graph, 0), "create original parent priority graph");
        check(cuGraphCreate(&resources->child_graph, 0), "create original child priority graph");
        std::array<void*, 1> graph_arguments{&resources->counter};
        CUDA_KERNEL_NODE_PARAMS parameters{};
        parameters.func = graph_kernel;
        parameters.gridDimX = parameters.gridDimY = parameters.gridDimZ = 1;
        parameters.blockDimX = parameters.blockDimY = parameters.blockDimZ = 1;
        parameters.kernelParams = graph_arguments.data();
        CUgraphNode kernel_node{};
        CUgraphNode record_node{};
        CUgraphNode child_node{};
        check(cuGraphAddKernelNode(&kernel_node, resources->child_graph, nullptr, 0, &parameters),
              "add unchanged original priority kernel and parameter bytes");
        check(cuGraphAddEventRecordNode(&record_node, resources->child_graph, &kernel_node, 1,
                                        resources->end),
              "add original event record after original high kernel");
        check(cuGraphAddChildGraphNode(&child_node, resources->graph, nullptr, 0,
                                       resources->child_graph),
              "retain actual CUDA-owned child graph copy");
        graph_arguments.fill(nullptr);
        parameters.kernelParams = nullptr;
        check(prepare_native_graph_effects(
                  resources->graph,
                  {cuGraphGetNodes, cuGraphNodeGetType, cuGraphChildGraphNodeGetGraph,
                   cuGraphEventRecordNodeGetEvent, cuGraphEventWaitNodeGetEvent},
                  graph_effects),
              "prepare bounded actual child-graph event effects before instantiation");
        require(graph_effects && graph_effects.node_count() == 3 &&
                    graph_effects.events().size() == 1 &&
                    graph_effects.events()[0].event == resources->end &&
                    graph_effects.events()[0].kind == CudaNativeGraphEventKind::kRecord &&
                    graph_effects.events()[0].node != record_node,
                "snapshot must identify the original CUDA child copy, not a guessed source node");
        CUgraph actual_child{};
        check(cuGraphChildGraphNodeGetGraph(child_node, &actual_child),
              "inspect actual child copy");
        std::array<CUgraphNode, 2> original_nodes{};
        std::size_t original_count = original_nodes.size();
        check(cuGraphGetNodes(actual_child, original_nodes.data(), &original_count),
              "inspect unchanged actual native child nodes");
        require(original_count == 2, "child must contain exactly the original kernel/event nodes");
        unsigned int original_kernels{};
        unsigned int original_records{};
        for (const auto node : original_nodes) {
            CUgraphNodeType type{};
            check(cuGraphNodeGetType(node, &type), "inspect original child node type");
            if (type == CU_GRAPH_NODE_TYPE_KERNEL) {
                CUDA_KERNEL_NODE_PARAMS actual{};
                check(cuGraphKernelNodeGetParams(node, &actual),
                      "inspect original kernel identity");
                require(actual.func == graph_kernel && actual.gridDimX == 1 &&
                            actual.gridDimY == 1 && actual.gridDimZ == 1 && actual.blockDimX == 1 &&
                            actual.blockDimY == 1 && actual.blockDimZ == 1,
                        "no transformed function, dimensions or private nodes enter the graph");
                ++original_kernels;
            } else if (type == CU_GRAPH_NODE_TYPE_EVENT_RECORD) {
                require(node == graph_effects.events()[0].node,
                        "actual original event node identity");
                ++original_records;
            }
        }
        require(original_kernels == 1 && original_records == 1, "only the original graph effects");
        CudaNativeGraphReservation graph_reservation;
        check(
            graph_registry.prepare(resources->context, graph_effects, *registry, graph_reservation),
            "prepare original executable CPU owner");
        check(cuGraphInstantiateWithFlags(&resources->graph_exec, resources->graph, 0),
              "instantiate original graph without modifying nodes");
        check(graph_registry.publish(graph_reservation, resources->graph_exec, CUDA_SUCCESS),
              "publish actual original executable without new allocation");
        check(graph_registry.lookup(resources->graph_exec, graph_owner),
              "retain executable effects independently of original graph lifetime");
        check(cuGraphDestroy(resources->graph), "destroy source graph before asynchronous replay");
        resources->graph = nullptr;
        check(cuGraphDestroy(resources->child_graph),
              "destroy original child source before replay");
        resources->child_graph = nullptr;
    }
    auto native_completion = std::make_shared<ProjectionCompletion>(registry);
    if (should_supersede_event) {
        if (is_graph_priority) {
            native_completion->epochs.reserve(graph_owner.events().size());
            for (const auto& bound : graph_owner.events()) {
                require(bound.effect.kind == CudaNativeGraphEventKind::kRecord,
                        "this fixture must not bypass a pending graph wait dependency");
                native_completion->epochs.push_back(bound.registration);
            }
        } else {
            native_completion->epochs.push_back(projection);
        }
    }
    std::uint32_t work = 50000;
    std::array<void*, 3> arguments{&resources->output, &marker, &work};
    check(frontend.copy(resources->producer, false,
                        {CudaQueuedCopyKind::kHostToDevice, resources->output, 0,
                         resources->host_values, kMemoryBytes, resources}),
          "queue original initialization");
    check(frontend.launch(*low, resources->producer, false, kGrid, kBlock, arguments.data(),
                          resources),
          "accept ordinary low invocation into frontend");
    work = 0;
    arguments.fill(nullptr);
    check(frontend.copy(resources->producer, false,
                        {CudaQueuedCopyKind::kDeviceToHost, 0, resources->output,
                         resources->host_values, kMemoryBytes, resources}),
          "queue original deferred readback");
    check(frontend.record_event(resources->end, resources->producer, false,
                                std::make_shared<ProjectionRecordGuard>(registry, projection)),
          "queue original logical record");
    check(frontend.wait_event(resources->consumer, false, resources->end),
          "retain immutable historical wait before native supersession");
    auto* state = static_cast<std::uint32_t*>(resources->host_state);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{10};
    while (std::atomic_ref(state[1]).load(std::memory_order_acquire) != 1 &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    require(std::atomic_ref(state[1]).load(std::memory_order_acquire) == 1,
            "frontend low must start without caller pumping");
    require(frontend.query_event(resources->end) == CUDA_ERROR_NOT_READY,
            "logical record must not report completion while original invocation is running");
    std::array<void*, 1> high_arguments{&marker};
    if (is_native_priority) {
        check(frontend.native(
                  resources->priority, false,
                  [&] {
                      if (should_supersede_event && !should_takeover_early) {
                          for (auto& owned : native_completion->epochs) {
                              CudaEventProjection affected;
                              check(registry->lookup_owned(owned, affected),
                                    "resolve the exact original event registration, not a reused "
                                    "handle");
                              owned = std::move(affected);
                          }
                          check(registry->invalidate_native_events(native_completion->epochs),
                                "atomically retire only actual native execution record epochs");
                          require(audit.records.load() == 0 &&
                                      !static_cast<const std::uint32_t*>(resources->host_values)[0],
                                  "deferred record and readback must not precede physical retreat");
                      }
                      if (is_graph_priority) {
                          check(graph_registry.is_current(graph_owner),
                                "verify unchanged executable registration before original issue");
                          ++graph_launches;
                          const auto launched =
                              cuGraphLaunch(resources->graph_exec, resources->priority);
                          return launched == CUDA_SUCCESS && should_takeover_early
                                     ? registry->invalidate_native_events(native_completion->epochs,
                                                                          native_completion->prefix)
                                     : launched;
                      }
                      const auto launched =
                          cuLaunchKernel(probe, 1, 1, 1, 1, 1, 1, 0, resources->priority,
                                         high_arguments.data(), nullptr);
                      return launched == CUDA_SUCCESS && should_supersede_event
                                 ? EventRecordAudit::record(resources->end, resources->priority)
                                 : launched;
                  },
                  true, should_supersede_event ? native_completion : nullptr),
              "insert once-issued original caller native high execution");
    } else {
        check(frontend.launch(*high, resources->priority, false, {1, 1, 1}, {1, 1, 1},
                              high_arguments.data(), resources),
              "accept original high invocation");
    }
    high_arguments.fill(nullptr);
    if (should_takeover_early) {
        CudaEventProjection issued;
        check(registry->lookup(resources->end, issued), "inspect issued native takeover prefix");
        check(frontend.record_event(resources->end, resources->priority, false,
                                    std::make_shared<ProjectionRecordGuard>(registry, issued)),
              "accept independent fresh record before low logical completion");
        check(frontend.synchronize_event(resources->end), "complete prefix-only native takeover");
        require(frontend.query_stream(resources->producer, false) == CUDA_ERROR_NOT_READY &&
                    frontend.query_stream(resources->consumer, false) == CUDA_ERROR_NOT_READY &&
                    audit.records == 1 && audit.last_stream == resources->priority,
                "fresh takeover must not depend on or complete the historical low record/wait");
    }
    check(frontend.synchronize_stream(resources->priority, false), "await logical high completion");
    if (should_supersede_event) {
        CudaEventProjection completed;
        CudaEventProjection unrelated_logical;
        CudaEventProjection unrelated_native;
        check(registry->lookup(resources->end, completed), "inspect exact native fence proof");
        check(registry->lookup(resources->begin, unrelated_logical),
              "inspect unrelated logical event projection");
        check(registry->lookup(resources->native_fence, unrelated_native),
              "inspect unrelated unproven native epoch");
        require(native_completion->calls.load() == 1 && completed.has_native_completion() &&
                    completed.is_logical() == should_takeover_early &&
                    unrelated_logical.is_logical() && !unrelated_native.has_native_completion(),
                "actual high fence proves its epoch without a context-wide drain or projection");
    }
    check(frontend.synchronize_event(resources->end),
          "await exactly-once low continuation and readback");
    check(frontend.synchronize_stream(resources->consumer, false),
          "complete historical wait only after resumed logical producer");
    require(audit.records.load() ==
                    static_cast<unsigned int>(!is_graph_priority || should_takeover_early) &&
                (is_graph_priority
                     ? graph_launches == 1 && graph_effects.node_count() == 3
                     : audit.last_stream.load() ==
                           (should_supersede_event ? resources->priority : resources->producer)),
            "deferred queue write must not overwrite the newer native record after resume");
    check(cuEventQuery(resources->end), "original latest native event remains authoritative");
    if (should_supersede_event && !should_takeover_early) {
        CudaEventProjection fresh;
        check(registry->lookup(resources->end, fresh), "snapshot new native epoch");
        check(frontend.record_event(resources->end, resources->producer, false,
                                    std::make_shared<ProjectionRecordGuard>(registry, fresh)),
              "accept a fresh logical record without discarding historical binding");
        check(registry->publish_logical(fresh), "restore logical projection on fresh acceptance");
        check(frontend.synchronize_event(resources->end), "complete current epoch's actual write");
        require(audit.records.load() == (is_graph_priority ? 1U : 2U) &&
                    audit.last_stream.load() == resources->producer,
                "fresh current record writes exactly once after native completion proof");
    }
    const auto* values = static_cast<const std::uint32_t*>(resources->host_values);
    require(std::all_of(values, values + kElements, [](auto value) { return value == 1; }),
            "frontend resume must not skip or duplicate original blocks");
    check(frontend.close_event(resources->end), "close original event binding");
    check(frontend.shutdown(),
          "physically drain frontend ownership before native resource cleanup");
    require(native_completion->calls.load() == static_cast<unsigned int>(should_supersede_event),
            "terminal drain must not repeat the exact native completion notification");
    const auto stats = frontend.stats();
    if (is_graph_priority) {
        std::uint32_t native_count{};
        check(cuMemcpyDtoH(&native_count, resources->counter, sizeof(native_count)),
              "read independent original native graph execution count after actual stop");
        require(
            native_count == 1,
            "original graph high kernel must execute once, independently of resumed low markers");
    }
    require(stats.granted_executions == 3 && stats.retreat_requests == 1 &&
                stats.acknowledged_suspensions == 1 && stats.logical_resumes == 1 &&
                stats.acknowledged_completions == 2 && stats.acknowledged_failures == 0,
            "frontend must prove retreat, native-priority handoff and new-generation resume");
    gate->shutdown();
    std::cout << "cuda_cooperative_frontend native_priority=" << is_native_priority
              << " superseded_event=" << should_supersede_event
              << " early_native_takeover=" << should_takeover_early
              << " native_graph=" << graph_launches
              << " grants=3 suspended=1 resumed=1 owned_arguments=1 exactly_once=1\n";
}
void native_graph_waits_for_logical_producer() {
    auto resources = std::make_shared<Resources>(false);
    check(cuStreamDestroy(resources->priority), "replace dependent graph stream");
    resources->priority = nullptr;
    int least{};
    int greatest{};
    check(cuCtxGetStreamPriorityRange(&least, &greatest), "read graph wait priority range");
    require(greatest < 0, "graph wait test requires a distinct high priority");
    check(cuStreamCreateWithPriority(&resources->priority, CU_STREAM_NON_BLOCKING, greatest),
          "create dependent high-priority graph stream");
    DriverDispatch driver(metadata_dispatch());
    PersistentKernelRuntime kernels(driver, 3, false, true);
    check(kernels.load_module(&resources->module, kCooperativePtx), "load graph wait producer");
    CUfunction original{};
    check(kernels.get_function(&original, resources->module, "cooperative"),
          "resolve graph wait producer");
    const auto low =
        kernels.cooperative_kernel(original, kGrid, kBlock, 0, resources->producer, false);
    require(low.has_value(), "graph wait producer must be qualified before execution");
    check(cuModuleLoadData(&resources->native_module, kNativeOncePtx),
          "load unchanged graph wait consumer");
    CUfunction graph_kernel{};
    check(cuModuleGetFunction(&graph_kernel, resources->native_module, "native_once"),
          "resolve unchanged original graph function");
    check(cuGraphCreate(&resources->graph, 0), "create original dependent graph");
    CUgraphNode wait_node{};
    CUgraphNode kernel_node{};
    check(cuGraphAddEventWaitNode(&wait_node, resources->graph, nullptr, 0, resources->end),
          "add original wait for application event");
    std::array<void*, 1> graph_arguments{&resources->counter};
    CUDA_KERNEL_NODE_PARAMS parameters{};
    parameters.func = graph_kernel;
    parameters.gridDimX = parameters.gridDimY = parameters.gridDimZ = 1;
    parameters.blockDimX = parameters.blockDimY = parameters.blockDimZ = 1;
    parameters.kernelParams = graph_arguments.data();
    check(cuGraphAddKernelNode(&kernel_node, resources->graph, &wait_node, 1, &parameters),
          "add unchanged original graph kernel after event wait");
    CudaNativeGraphEffects effects;
    check(prepare_native_graph_effects(
              resources->graph,
              {cuGraphGetNodes, cuGraphNodeGetType, cuGraphChildGraphNodeGetGraph,
               cuGraphEventRecordNodeGetEvent, cuGraphEventWaitNodeGetEvent},
              effects),
          "snapshot original graph wait before executable creation");
    require(effects.node_count() == 2 && effects.events().size() == 1 &&
                effects.events()[0].kind == CudaNativeGraphEventKind::kWait &&
                effects.events()[0].event == resources->end,
            "graph effects must retain only the unchanged original wait handle");
    CUDA_KERNEL_NODE_PARAMS inspected{};
    check(cuGraphKernelNodeGetParams(kernel_node, &inspected), "inspect original graph ABI");
    require(inspected.func == graph_kernel && inspected.gridDimX == 1 && inspected.blockDimX == 1 &&
                *static_cast<const CUdeviceptr*>(inspected.kernelParams[0]) == resources->counter,
            "event dependency preparation must not rewrite the original graph kernel");
    check(cuGraphInstantiate(&resources->graph_exec, resources->graph, 0),
          "instantiate original dependent graph");
    check(cuGraphDestroy(resources->graph), "destroy source graph before native replay");
    resources->graph = nullptr;
    graph_arguments.fill(nullptr);
    glimmer::control::LaunchGateOptions options;
    options.scheduling_policy = glimmer::core::SchedulingPolicy::kPriority;
    auto gate = std::make_shared<glimmer::control::AsyncCooperativeLaunchGate>(options);
    CudaCooperativeFrontend frontend(resources->context,
                                     {{native_dispatch(), cuCtxPushCurrent, cuCtxPopCurrent},
                                      native_pool_dispatch(resources),
                                      cuStreamCreate,
                                      cuStreamDestroy,
                                      cuStreamGetFlags,
                                      cuStreamGetPriority},
                                     gate, {.progress_slots = 4, .import_events = 8});
    check(frontend.initialize(), "initialize original graph wait frontend");
    check(frontend.bind_event(resources->end, resources), "bind actual waited producer event");
    auto marker = resources->mapped_state + sizeof(std::uint32_t);
    std::uint32_t work = 50000;
    std::array<void*, 3> arguments{&resources->output, &marker, &work};
    check(frontend.copy(resources->producer, false,
                        {CudaQueuedCopyKind::kHostToDevice, resources->output, 0,
                         resources->host_values, kMemoryBytes, resources}),
          "initialize graph wait producer output");
    check(frontend.launch(*low, resources->producer, false, kGrid, kBlock, arguments.data(),
                          resources),
          "accept qualified low producer before graph wait");
    check(frontend.copy(resources->producer, false,
                        {CudaQueuedCopyKind::kDeviceToHost, 0, resources->output,
                         resources->host_values, kMemoryBytes, resources}),
          "queue exact producer readback before event");
    check(frontend.record_event(resources->end, resources->producer, false),
          "accept pending producer logical record");
    work = 0;
    arguments.fill(nullptr);
    auto* state = static_cast<std::uint32_t*>(resources->host_state);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{10};
    while (std::atomic_ref(state[1]).load(std::memory_order_acquire) != 1 &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    require(std::atomic_ref(state[1]).load(std::memory_order_acquire) == 1 &&
                frontend.query_event(resources->end) == CUDA_ERROR_NOT_READY,
            "actual low producer must still be pending before dependent high admission");
    const std::array waits{effects.events()[0].event};
    unsigned int original_launches{};
    check(frontend.native(
              resources->priority, false,
              [&] {
                  const auto* values = static_cast<const std::uint32_t*>(resources->host_values);
                  require(std::all_of(values, values + kElements,
                                      [](auto value) { return value == 1; }) &&
                              cuEventQuery(resources->end) == CUDA_SUCCESS,
                          "original graph issue must follow exact low event completion");
                  ++original_launches;
                  return cuGraphLaunch(resources->graph_exec, resources->priority);
              },
              true, {}, waits),
          "admit original graph only after its logical wait dependency");
    check(frontend.synchronize_stream(resources->priority, false), "prove native graph stop");
    check(frontend.close_event(resources->end), "retire graph wait logical event");
    check(frontend.shutdown(), "drain original graph wait ownership");
    std::uint32_t count{};
    check(cuMemcpyDtoH(&count, resources->counter, sizeof(count)), "read original graph count");
    const auto stats = frontend.stats();
    require(original_launches == 1 && count == 1 && stats.granted_executions == 2 &&
                stats.acknowledged_completions == 2 && stats.retreat_requests == 0 &&
                stats.logical_resumes == 0 && stats.acknowledged_failures == 0,
            "dependent native graph must finish once without pausing its required producer");
    gate->shutdown();
    std::cout << "cuda_native_graph_wait logical_dependency=1 original_nodes=2 grants=2 "
                 "retreat=0 exactly_once=1\n";
}
constexpr const char* kFrontendEpochPtx = R"ptx(
.version 7.0
.target sm_70
.address_size 64
.visible .entry epoch_increment(.param .u64 address, .param .u64 duration) {
 .reg .pred p;
 .reg .b64 pointer, delay, start, now, elapsed;
 .reg .b32 value;
 ld.param.u64 pointer, [address];
 ld.param.u64 delay, [duration];
 mov.u64 start, %clock64;
wait:
 mov.u64 now, %clock64;
 sub.u64 elapsed, now, start;
 setp.lt.u64 p, elapsed, delay;
 @p bra wait;
 ld.global.u32 value, [pointer];
 add.u32 value, value, 1;
 st.global.u32 [pointer], value;
 ret;
}
)ptx";
template <typename Predicate>
void frontend_epoch_until(Predicate predicate) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{10};
    while (!predicate()) {
        require(std::chrono::steady_clock::now() < deadline,
                "bounded actual frontend epoch progress");
        std::this_thread::yield();
    }
}
template <typename Value>
Value frontend_epoch_result(std::future<Value> future) {
    require(future.valid() && future.wait_for(std::chrono::seconds{3}) == std::future_status::ready,
            "bounded actual shared control result");
    return future.get();
}
template <typename Function>
Function frontend_epoch_original(const char* name, Function legacy, bool per_thread_scope) {
    if (!per_thread_scope) return legacy;
    void* address{};
    CUdriverProcAddressQueryResult status = CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
    check(cuGetProcAddress(name, &address, 12000, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM,
                           &status),
          "resolve actual original PTDS ABI");
    require(address && status == CU_GET_PROC_ADDRESS_SUCCESS, "actual original PTDS symbol");
    return reinterpret_cast<Function>(address);
}
void native_epoch_frontend(int default_mode = -1) {
    using namespace glimmer::control;
    auto resources = std::make_shared<Resources>(default_mode == 4);
    const bool per_thread_scope = default_mode == 2 || default_mode == 3;
    const auto original_launch =
        frontend_epoch_original("cuLaunchKernel", cuLaunchKernel, per_thread_scope);
    const auto original_record =
        frontend_epoch_original("cuEventRecord", cuEventRecord, per_thread_scope);
    const auto original_graph =
        frontend_epoch_original("cuGraphLaunch", cuGraphLaunch, per_thread_scope);
    const CUstream stream = default_mode == 0 || default_mode == 2 ? nullptr
                            : default_mode == 1                    ? CU_STREAM_LEGACY
                            : default_mode == 3                    ? CU_STREAM_PER_THREAD
                                                                   : resources->producer;
    const CUstream alias = default_mode == 0   ? CU_STREAM_LEGACY
                           : default_mode == 1 ? nullptr
                           : default_mode == 2 ? CU_STREAM_PER_THREAD
                           : default_mode == 3 ? nullptr
                                               : stream;
    check(cuModuleLoadData(&resources->native_module, kFrontendEpochPtx),
          "load unchanged epoch kernel");
    CUfunction increment{};
    check(cuModuleGetFunction(&increment, resources->native_module, "epoch_increment"),
          "original epoch function");
    auto metadata = metadata_dispatch();
    metadata.stream_capture_status_ptsz =
        frontend_epoch_original("cuStreamIsCapturing", cuStreamIsCapturing, per_thread_scope);
    DriverDispatch driver(metadata);
    PersistentKernelRuntime kernels(driver, 3, false, true);
    check(kernels.load_module(&resources->module, kCooperativePtx),
          "prepare mixed cooperative module");
    CUfunction original{};
    check(kernels.get_function(&original, resources->module, "cooperative"),
          "mixed cooperative original");
    const auto low =
        kernels.cooperative_kernel(original, kGrid, kBlock, 0, stream, per_thread_scope);
    require(low.has_value(), "mixed cooperative source is qualified");
    std::array<std::uint32_t, 3> guarded{0x12345678U, 0, 0x76543210U};
    check(cuMemcpyHtoD(resources->mirror, guarded.data(), sizeof(guarded)),
          "initialize guarded original output");
    CUdeviceptr pointer = resources->mirror + sizeof(std::uint32_t);
    constexpr NativeEpochExtent extent{8, 2, 2, 1'000'000};
    LaunchGateOptions control_options;
    control_options.scheduling_policy = glimmer::core::SchedulingPolicy::kPriority;
    auto gate = std::make_shared<AsyncCooperativeLaunchGate>(control_options, 64, extent);
    CudaFrontendOptions options;
    options.progress_slots = 4;
    options.import_events = 8;
    options.per_thread_streams = 2;
    options.native_epoch_policy = extent;
    if (default_mode < 0) options.native_epoch_policy->acceptance_microseconds = 100'000;
    options.native_epoch_dispatch = {cuCtxGetCurrent, cuCtxPushCurrent,    cuCtxPopCurrent,
                                     cuStreamGetCtx,  cuStreamIsCapturing, cuEventCreate,
                                     cuEventRecord,   cuEventQuery,        cuEventDestroy};
    CudaCooperativeFrontend frontend(resources->context,
                                     {{native_dispatch(), cuCtxPushCurrent, cuCtxPopCurrent},
                                      native_pool_dispatch(resources),
                                      cuStreamCreate,
                                      cuStreamDestroy,
                                      cuStreamGetFlags,
                                      cuStreamGetPriority},
                                     gate, options);
    check(frontend.initialize(), "initialize actual epoch membership");
    check(frontend.bind_event(resources->end, resources), "bind historical actual event");
    check(frontend.record_event(resources->end, stream, per_thread_scope),
          "record historical actual prefix");
    check(frontend.synchronize_event(resources->end), "complete historical actual prefix");
    const auto caller = std::this_thread::get_id();
    unsigned int originals{};
    const auto launch = [&](CUstream stream, unsigned long long duration) {
        require(std::this_thread::get_id() == caller, "actual original remains on its caller");
        CUcontext actual{};
        check(cuCtxGetCurrent(&actual), "original caller actual context");
        require(actual == resources->context, "actual context is unchanged");
        void* arguments[]{&pointer, &duration};
        ++originals;
        return original_launch(increment, 1, 1, 1, 1, 1, 1, 0, stream, arguments, nullptr);
    };
    NativeEpochGrant retained{};
    for (unsigned int index = 0; index < 4; ++index) {
        const auto original_stream = index % 2 == 0 ? stream : alias;
        check(frontend.native_epoch(
                  original_stream, [&] { return launch(original_stream, 100'000'000); },
                  per_thread_scope),
              "original launch under shared epoch");
        const auto snapshot = frontend.native_epoch_snapshot();
        require(snapshot && snapshot->execution && snapshot->execution->issued_calls == index + 1,
                "actual originals counted exactly once");
        if (index == 0) retained = snapshot->grant;
        require(snapshot->grant.execution.task_id == retained.execution.task_id &&
                    snapshot->grant.execution.generation == retained.execution.generation &&
                    snapshot->grant.acceptance_deadline_nanoseconds ==
                        retained.acceptance_deadline_nanoseconds,
                "one actual independent grant for four originals");
    }
    check(original_record(resources->native_fence, stream),
          "independent actual epoch stop observer");
    require(cuEventQuery(resources->native_fence) == CUDA_ERROR_NOT_READY,
            "native originals physically pending");
    check(frontend.query_event(resources->end), "historical event excludes actual later epoch");
    check(frontend.query_stream(resources->consumer, false),
          "unrelated actual nonblocking stream query");
    auto high = frontend_epoch_result(gate->submit_and_claim({10, 1}));
    require(high.claim.status == CooperativeClaimStatus::kPending,
            "native epoch retains shared priority capacity");
    if (default_mode >= 0) {
        CUresult foreign_query = CUDA_ERROR_UNKNOWN;
        std::thread other([&] {
            check(cuCtxPushCurrent(resources->context), "foreign PTDS caller context");
            foreign_query = frontend.query_stream(nullptr, true);
            CUcontext popped{};
            check(cuCtxPopCurrent(&popped), "foreign PTDS caller context restoration");
            require(popped == resources->context, "exact foreign context restoration");
        });
        other.join();
        require(foreign_query == (default_mode <= 1 ? CUDA_ERROR_NOT_READY : CUDA_SUCCESS),
                "another actual caller PTDS is independent except for legacy dependencies");
        require(frontend.query_stream(nullptr, false) == CUDA_ERROR_NOT_READY,
                "legacy query includes actual blocking and caller PTDS epochs");
    }
    require(frontend.query_stream(alias, per_thread_scope) == CUDA_ERROR_NOT_READY,
            "affected actual query returns without GPU drain");
    check(frontend.synchronize_stream(alias, per_thread_scope), "actual stop before logical thaw");
    check(cuEventQuery(resources->native_fence), "independent actual stop proof");
    const auto stopped = frontend.native_epoch_snapshot();
    require(stopped && stopped->phase == CudaNativeEpochPhase::kComplete &&
                stopped->proven_prefixes == (per_thread_scope ? 4U : 2U) &&
                stopped->created_events == stopped->destroyed_events,
            "actual two-prefix proof and private cleanup precede thaw");
    high.claim = frontend_epoch_result(gate->claim(high.task_id));
    require(high.claim.status == CooperativeClaimStatus::kGranted,
            "high admission only after native stop");
    check(launch(resources->priority, 0), "actual high original launch");
    check(cuEventRecord(resources->native_fence, resources->priority), "actual high stop observer");
    check(cuEventSynchronize(resources->native_fence), "actual high stop");
    require(frontend_epoch_result(gate->complete(high.claim.execution)),
            "high acknowledgment follows actual proof");
    auto marker = resources->mapped_state + sizeof(std::uint32_t);
    std::uint32_t work{};
    std::array<void*, 3> arguments{&resources->output, &marker, &work};
    check(frontend.copy(stream, per_thread_scope,
                        {CudaQueuedCopyKind::kHostToDevice, resources->output, 0,
                         resources->host_values, kMemoryBytes, resources}),
          "initialize mixed cooperative output after actual native stop");
    check(
        frontend.launch(*low, stream, per_thread_scope, kGrid, kBlock, arguments.data(), resources),
        "cooperative queue accepts after native thaw");
    check(frontend.copy(stream, per_thread_scope,
                        {CudaQueuedCopyKind::kDeviceToHost, 0, resources->output,
                         resources->host_values, kMemoryBytes, resources}),
          "read exact mixed cooperative output");
    check(frontend.native_epoch(
              stream,
              [&] {
                  require(
                      std::all_of(static_cast<std::uint32_t*>(resources->host_values),
                                  static_cast<std::uint32_t*>(resources->host_values) + kElements,
                                  [](auto value) { return value == 1; }),
                      "native entry follows actual cooperative/readback stop");
                  return launch(stream, 0);
              },
              per_thread_scope),
          "native epoch after drained cooperative queue");
    // A whole original graph replay remains a separate unit execution. Its
    // source nodes and arguments are untouched by epoch membership.
    check(cuGraphCreate(&resources->graph, 0), "create original whole graph");
    unsigned long long duration{};
    void* graph_arguments[]{&pointer, &duration};
    CUDA_KERNEL_NODE_PARAMS parameters{};
    parameters.func = increment;
    parameters.gridDimX = parameters.gridDimY = parameters.gridDimZ = 1;
    parameters.blockDimX = parameters.blockDimY = parameters.blockDimZ = 1;
    parameters.kernelParams = graph_arguments;
    CUgraphNode node{};
    check(cuGraphAddKernelNode(&node, resources->graph, nullptr, 0, &parameters),
          "unchanged original graph node");
    check(cuGraphInstantiate(&resources->graph_exec, resources->graph, 0),
          "instantiate original whole graph");
    check(frontend.native(
              stream, per_thread_scope,
              [&] { return original_graph(resources->graph_exec, stream); }, true),
          "whole original graph closes preceding epoch");
    check(frontend.synchronize_context(), "prove mixed epoch/cooperative/graph completion");
    check(frontend.native_epoch(
              stream, [&] { return launch(stream, 0); }, per_thread_scope),
          "enter actual idle partial epoch");
    frontend_epoch_until([&] {
        const auto snapshot = frontend.native_epoch_snapshot();
        require(snapshot && snapshot->phase != CudaNativeEpochPhase::kRetained,
                "actual idle epoch cleanup succeeds");
        return snapshot->phase == CudaNativeEpochPhase::kComplete;
    });
    CudaWorkerCheckpoint checkpoint;
    frontend_epoch_until([&] { return frontend.checkpoint(checkpoint) == CUDA_SUCCESS; });
    check(frontend.resume_checkpoint(checkpoint), "restore actual completed epoch membership");
    check(frontend.close_event(resources->end), "close actual historical event");
    check(frontend.shutdown(), "drain and join actual epoch frontend");
    check(cuMemcpyDtoH(guarded.data(), resources->mirror, sizeof(guarded)),
          "guarded actual original readback");
    require(originals == 7 && guarded == std::array<std::uint32_t, 3>{0x12345678U, 8, 0x76543210U},
            "actual original counts, graph replay and guards remain exact");
    gate->shutdown();
    std::cout << "cuda_frontend_epoch mode=" << default_mode
              << " originals=7 graph=1 prefix_proof=" << (per_thread_scope ? 4 : 2)
              << " mixed_cooperative=1 guarded_output=8\n";
}
}  // namespace

int main(int argc, char** argv) {
    try {
        const bool epochs_only = argc == 2 && std::string_view(argv[1]) == "--native-epochs";
        const bool defaults_only = argc == 2 && std::string_view(argv[1]) == "--default-epochs";
        require(argc == 1 || epochs_only || defaults_only, "queue GPU fixture options");
        if (defaults_only) {
            for (unsigned int lifetime = 0; lifetime < 2; ++lifetime)
                for (int mode = 0; mode < 5; ++mode) native_epoch_frontend(mode);
            return EXIT_SUCCESS;
        }
        if (epochs_only) {
            native_epoch_frontend();
            native_epoch_frontend();
            return EXIT_SUCCESS;
        }
        run(false);
        run(true);
        autonomous_worker();
        autonomous_preemption(false);
        autonomous_preemption(true);
        autonomous_native_execution();
        cooperative_frontend(false);
        cooperative_frontend(true);
        cooperative_frontend(true, true);
        cooperative_frontend(true, true, true);
        cooperative_frontend(true, true, true, true);
        native_graph_waits_for_logical_producer();
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
