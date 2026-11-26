#include "internal/cuda_cooperative_frontend.h"
#include "cooperative_ptx_fixture.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>

namespace {
using namespace glimmer::interceptor;
namespace control = glimmer::control;
constexpr unsigned int kBlocks = 84;
constexpr unsigned int kThreads = 80;
constexpr std::size_t kElements = static_cast<std::size_t>(kBlocks) * kThreads;

void require(bool value, const char* message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}
void check(CUresult result, const char* message) {
    if (result != CUDA_SUCCESS) {
        throw std::runtime_error(std::string(message) + ": " + std::to_string(result));
    }
}
template <typename Function>
Function original(const DriverDispatch& driver, const char* name) {
    void* address{};
    check(driver.get_proc_address(name, &address, 12600, CU_GET_PROC_ADDRESS_LEGACY_STREAM),
          "resolve exact original native dispatch");
    require(address != nullptr && !is_interceptor_driver_function(address),
            "required dispatch is original, never an interposer");
    return reinterpret_cast<Function>(address);
}
CudaFrontendDispatch frontend_dispatch(const DriverDispatch& driver) {
    CudaFrontendDispatch dispatch;
    auto& queue = dispatch.worker.queue;
    queue.context_get_current =
        original<decltype(queue.context_get_current)>(driver, "cuCtxGetCurrent");
    queue.launch_kernel = original<decltype(queue.launch_kernel)>(driver, "cuLaunchKernel");
    queue.copy_host_to_device =
        original<decltype(queue.copy_host_to_device)>(driver, "cuMemcpyHtoDAsync");
    queue.copy_device_to_host =
        original<decltype(queue.copy_device_to_host)>(driver, "cuMemcpyDtoHAsync");
    queue.copy_device_to_device =
        original<decltype(queue.copy_device_to_device)>(driver, "cuMemcpyDtoDAsync");
    queue.free_async = original<decltype(queue.free_async)>(driver, "cuMemFreeAsync");
    queue.event_create = original<decltype(queue.event_create)>(driver, "cuEventCreate");
    queue.event_record = original<decltype(queue.event_record)>(driver, "cuEventRecord");
    queue.event_query = original<decltype(queue.event_query)>(driver, "cuEventQuery");
    queue.event_destroy = original<decltype(queue.event_destroy)>(driver, "cuEventDestroy");
    queue.stream_synchronize =
        original<decltype(queue.stream_synchronize)>(driver, "cuStreamSynchronize");
    queue.stream_wait_event =
        original<decltype(queue.stream_wait_event)>(driver, "cuStreamWaitEvent");
    queue.event_synchronize =
        original<decltype(queue.event_synchronize)>(driver, "cuEventSynchronize");
    dispatch.worker.context_push =
        original<decltype(dispatch.worker.context_push)>(driver, "cuCtxPushCurrent");
    dispatch.worker.context_pop =
        original<decltype(dispatch.worker.context_pop)>(driver, "cuCtxPopCurrent");
    dispatch.worker.enter_native_scope = enter_native_driver_scope;
    dispatch.worker.leave_native_scope = leave_native_driver_scope;
    dispatch.stream_create = original<decltype(dispatch.stream_create)>(driver, "cuStreamCreate");
    dispatch.stream_destroy =
        original<decltype(dispatch.stream_destroy)>(driver, "cuStreamDestroy");
    dispatch.stream_get_flags =
        original<decltype(dispatch.stream_get_flags)>(driver, "cuStreamGetFlags");
    dispatch.stream_get_priority =
        original<decltype(dispatch.stream_get_priority)>(driver, "cuStreamGetPriority");
    auto& pool = dispatch.pool;
    pool.context_get_current = queue.context_get_current;
    pool.allocate_device = [&driver](CUdeviceptr* pointer, std::size_t bytes) {
        return driver.mem_alloc(pointer, bytes);
    };
    pool.free_device = [&driver](CUdeviceptr pointer) {
        const auto result = driver.mem_free(pointer);
        return CudaCooperativeReleaseResult{result, result == CUDA_SUCCESS
                                                        ? CudaNativeReleaseState::kReleased
                                                        : CudaNativeReleaseState::kUncertain};
    };
    pool.allocate_host = original<decltype(pool.allocate_host)>(driver, "cuMemHostAlloc");
    const auto free_host = original<decltype(&cuMemFreeHost)>(driver, "cuMemFreeHost");
    pool.free_host = [free_host](void* pointer) {
        const auto result = free_host(pointer);
        return CudaCooperativeReleaseResult{result, result == CUDA_SUCCESS
                                                        ? CudaNativeReleaseState::kReleased
                                                        : CudaNativeReleaseState::kUncertain};
    };
    pool.map_host = original<decltype(pool.map_host)>(driver, "cuMemHostGetDevicePointer");
    pool.copy_device_to_host = queue.copy_device_to_host;
    pool.accounting_owner = std::make_shared<int>(0);
    return dispatch;
}
CudaWorkerCheckpoint join(CudaCooperativeFrontend& frontend) {
    CudaWorkerCheckpoint checkpoint;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{60};
    CUresult result{};
    do {
        result = frontend.checkpoint(checkpoint);
        require(result == CUDA_SUCCESS || result == CUDA_ERROR_NOT_READY,
                "checkpoint accepts only completed logical/control work");
        std::this_thread::sleep_for(std::chrono::milliseconds{1});
    } while (result == CUDA_ERROR_NOT_READY && std::chrono::steady_clock::now() < deadline);
    check(result, "join actual completed native owner");
    return checkpoint;
}

void qualify(bool is_modern) {
    DriverDispatch driver;
    require(driver.initialize(), "initialize original dispatch");
    check(cuInit(0), "publish original process Driver table without Runtime activation");
    const NativeDriverCallScope native;
    check(driver.init(0), "initialize native Driver");
    void* address{};
    const int version = is_modern ? 11000 : 7000;
    check(driver.get_proc_address("cuDevicePrimaryCtxRelease", &address, version,
                                  CU_GET_PROC_ADDRESS_DEFAULT),
          "select exact release ownership version");
    require(address != nullptr, "versioned original release exists");
    const auto release = reinterpret_cast<decltype(&cuDevicePrimaryCtxRelease)>(address);
    CUcontext context{};
    check(cuDevicePrimaryCtxRetain(&context, 0), "retain sole original primary reference");
    check(cuCtxSetCurrent(context), "bind original lifecycle caller");
    control::LaunchGateOptions options;
    options.scheduling_policy = glimmer::core::SchedulingPolicy::kPriority;
    auto gate = std::make_shared<control::AsyncCooperativeLaunchGate>(options);
    auto owner = std::make_unique<CudaCooperativeFrontend>(
        context, frontend_dispatch(driver), gate,
        CudaFrontendOptions{.progress_slots = 32, .import_events = 16, .per_thread_streams = 2});
    check(owner->initialize(), "initialize standalone private owner");
    PersistentKernelRuntime kernels(driver, 4, false, true);
    CUmodule module{};
    CUfunction low{}, high{};
    check(kernels.load_module(&module, kCooperativePtx), "load original source module");
    check(kernels.get_function(&low, module, "cooperative"), "lookup original cooperative source");
    check(kernels.get_function(&high, module, "priority_probe"), "lookup original priority source");
    kernels.prepare_module_source(low);
    const auto variant =
        kernels.cooperative_kernel(low, {kBlocks, 1, 1}, {kThreads, 1, 1}, 0, nullptr, false);
    require(variant.has_value(), "actual source produced a cooperative private variant");
    CUstream stream{}, high_stream{};
    CUevent event{};
    CUdeviceptr output{}, mapped{};
    std::uint32_t* markers{};
    std::uint32_t* readback{};
    check(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), "create surviving original low stream");
    check(cuStreamCreateWithPriority(&high_stream, CU_STREAM_NON_BLOCKING, -1),
          "create priority stream");
    check(cuEventCreate(&event, CU_EVENT_DEFAULT), "create surviving original event");
    check(owner->bind_event(event, std::make_shared<int>(1)),
          "bind immutable original event identity");
    check(driver.mem_alloc(&output, kElements * sizeof(std::uint32_t)), "allocate original output");
    check(cuMemsetD32(output, 0, kElements), "initialize exact-once output");
    check(cuMemHostAlloc(reinterpret_cast<void**>(&markers), 4096, CU_MEMHOSTALLOC_DEVICEMAP),
          "allocate live mapped markers");
    check(cuMemHostGetDevicePointer(&mapped, markers, 0), "map live markers");
    check(cuMemAllocHost(reinterpret_cast<void**>(&readback), kElements * sizeof(std::uint32_t)),
          "allocate live pinned readback");
    bool has_pending_pause{};
    unsigned int original_calls{};
    for (unsigned int cycle = 0; cycle < 3; ++cycle) {
        markers[0] = 0;
        markers[1] = 0;
        unsigned int work = cycle == 1 ? 150000 : 1000;
        std::array<void*, 3> arguments{&output, &mapped, &work};
        check(owner->launch(*variant, stream, false, {kBlocks, 1, 1}, {kThreads, 1, 1},
                            arguments.data(), std::make_shared<int>(1)),
              "launch retained cooperative variant");
        if (cycle == 1) {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{10};
            while (std::atomic_ref<std::uint32_t>(markers[0]).load(std::memory_order_acquire) ==
                   0) {
                require(std::chrono::steady_clock::now() < deadline,
                        "physical low work starts after pool turnover");
                std::this_thread::sleep_for(std::chrono::milliseconds{1});
            }
            CUdeviceptr high_marker = mapped + sizeof(std::uint32_t);
            std::array<void*, 1> high_arguments{&high_marker};
            check(owner->native(
                      high_stream, false,
                      [&] {
                          std::array<std::uint32_t, kElements> sampled{};
                          check(cuMemcpyDtoH(sampled.data(), output, sizeof(sampled)),
                                "sample actually suspended work");
                          has_pending_pause =
                              std::any_of(sampled.begin(), sampled.end(),
                                          [&](auto value) { return value == cycle; });
                          return driver.launch_kernel(high, 1, 1, 1, 1, 1, 1, 0, high_stream,
                                                      high_arguments.data(), nullptr);
                      },
                      true),
                  "insert physical priority work into original actor");
        }
        check(owner->copy(stream, false,
                          {CudaQueuedCopyKind::kDeviceToHost, 0, output, readback,
                           kElements * sizeof(std::uint32_t), std::make_shared<int>(1)}),
              "accept ordered readback");
        check(owner->record_event(event, stream, false), "record original event generation");
        const auto checkpoint = join(*owner);
        require(std::all_of(readback, readback + kElements,
                            [&](auto value) { return value == cycle + 1; }),
                "all 6720 outputs completed exactly once");
        if (cycle == 1) {
            require(std::atomic_ref<std::uint32_t>(markers[1]).load(std::memory_order_acquire) == 2,
                    "the inserted original priority kernel actually completed its output");
        }
        CUcontext extra{};
        check(cuDevicePrimaryCtxRetain(&extra, 0),
              "retain independently owned extra primary reference");
        require(extra == context, "extra reference belongs to exact original primary");
        check(owner->release_checkpoint_progress(checkpoint),
              "free only private progress before original release");
        check(release(0), "issue exact original nonlast release once");
        ++original_calls;
        unsigned int flags{};
        int active{};
        CUcontext after{};
        check(cuDevicePrimaryCtxGetState(0, &flags, &active),
              "prove primary survived original release");
        check(driver.context_get_current(&after), "observe unchanged original caller binding");
        require(active == 1 && after == context,
                "native nonlast release preserves live primary and caller");
        check(owner->restore_checkpoint_progress(checkpoint),
              "recreate only private progress in surviving target");
        check(owner->resume_checkpoint(checkpoint),
              "resume the same worker queue and immutable bindings");
        check(owner->query_event(event), "reuse exact original completed event");
        check(owner->wait_event(nullptr, true, event),
              "reuse original event through preserved PTDS proxy");
        check(owner->synchronize_stream(nullptr, true), "prove resumed PTDS ordering");
    }
    const auto stats = owner->stats();
    require(has_pending_pause && stats.granted_executions == 5 && stats.retreat_requests == 1 &&
                stats.acknowledged_suspensions == 1 && stats.logical_resumes == 1 &&
                stats.acknowledged_completions == 4 && stats.acknowledged_failures == 0,
            "pool turnover preserves exact cooperative physical handoff and terminal control");
    const auto checkpoint = join(*owner);
    check(owner->release_checkpoint_progress(checkpoint),
          "free private progress before actual last release");
    check(cuMemFreeHost(markers), "free stopped mapped caller storage while target lives");
    check(cuMemFreeHost(readback), "free completed caller readback while target lives");
    check(release(0), "issue exact original last release once");
    ++original_calls;
    unsigned int flags{};
    int active{};
    CUcontext after{};
    check(cuDevicePrimaryCtxGetState(0, &flags, &active),
          "observe actual destruction without old handle query");
    check(driver.context_get_current(&after), "observe native post-destruction caller binding");
    require(active == 0 && after == context,
            "actual last release destroys primary without popping caller");
    check(owner->retire_checkpoint_after_context_destroy(checkpoint),
          "consume only destroyed CPU ownership");
    kernels.forget_context(context);  // Context-independent private library cleanup only.
    require(owner->stats().acknowledged_completions == stats.acknowledged_completions,
            "destroyed CPU retirement preserves terminal evidence");
    check(owner->shutdown(), "idempotent destroyed CPU teardown");
    owner.reset();
    check(cuDevicePrimaryCtxGetState(0, &flags, &active),
          "private library cleanup does not reactivate primary");
    require(active == 0 && original_calls == 4, "no repeated release or hidden primary activation");
    check(cuCtxSetCurrent(nullptr), "clear native unchanged destroyed caller binding");
    // New generation ownership never reuses old stream/event/function/allocation handles.
    CUcontext fresh{};
    check(cuDevicePrimaryCtxRetain(&fresh, 0), "retain independently new primary generation");
    check(cuCtxSetCurrent(fresh), "bind independently new primary generation");
    CudaCooperativeFrontend fresh_owner(
        fresh, frontend_dispatch(driver), gate,
        {.progress_slots = 2, .import_events = 2, .per_thread_streams = 2});
    check(fresh_owner.initialize(), "initialize new private generation after last release");
    check(fresh_owner.shutdown(), "check new generation live resource teardown");
    check(cuCtxSetCurrent(nullptr), "unbind live new generation before last release");
    check(release(0), "release independently new generation once");
    check(cuDevicePrimaryCtxGetState(0, &flags, &active), "prove new generation reference cleanup");
    check(driver.context_get_current(&after), "observe final unbound native caller");
    require(active == 0 && after == nullptr,
            "new generation final release leaves no active primary or invented caller binding");
    gate->shutdown();
    std::cout << "primary_release_owner version=" << version
              << " nonlast=3 last=1 outputs=6720 grants=5 retreat=1 suspended=1 resumed=1 "
                 "completed=4 pending_pause=1 new_generation=1 public_release_routing=0\n";
}
}  // namespace

int main(int argc, char** argv) {
    try {
        require(argc == 2 && (std::string_view(argv[1]) == "--legacy" ||
                              std::string_view(argv[1]) == "--modern"),
                "select release ownership version");
        qualify(std::string_view(argv[1]) == "--modern");
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "primary release owner qualification failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
