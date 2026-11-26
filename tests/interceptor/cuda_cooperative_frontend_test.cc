#include "internal/cuda_cooperative_frontend.h"
#include "internal/cuda_binary_slice.h"
#include "internal/cuda_process_native_session.h"
#include "internal/cuda_event_projection_registry.h"
#include "internal/diagnostics.h"

#include <atomic>
#include <array>
#include <chrono>
#include <bit>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <latch>
#include <barrier>
#include <cerrno>
#include <cstdlib>
#include <new>
#include <vector>
#include <charconv>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <sys/wait.h>
#include <unistd.h>

namespace {
thread_local bool should_reject_retirement_allocation{};
thread_local bool is_epoch_queue_actor{};
std::atomic<bool> should_reject_epoch_marker_allocation{};
std::atomic<unsigned int> rejected_epoch_marker_allocations{};
}  // namespace
// Caller-local failure; background actor/control threads retain normal allocation.
void* operator new(std::size_t bytes) {
    if (is_epoch_queue_actor && should_reject_epoch_marker_allocation.load()) {
        ++rejected_epoch_marker_allocations;
        throw std::bad_alloc();
    }
    if (should_reject_retirement_allocation) {
        throw std::bad_alloc();
    }
    if (void* pointer = std::malloc(bytes ? bytes : 1)) {
        return pointer;
    }
    throw std::bad_alloc();
}
// Keep replacement deallocation opaque to GCC's caller-side allocation-kind
// inference: this fixture's matching replacement new also uses malloc/free.
[[gnu::noinline]] void operator delete(void* pointer) noexcept {
    std::free(pointer);
}
[[gnu::noinline]] void operator delete(void* pointer, std::size_t) noexcept {
    std::free(pointer);
}

namespace {
using namespace glimmer::interceptor;
const auto kContext = reinterpret_cast<CUcontext>(0x1000);
thread_local CUcontext current_context{};
thread_local unsigned int caller_context_queries{};
thread_local std::latch* native_context_probe{};
std::atomic<bool> is_ready{true};
std::atomic<bool> should_fail_epoch_query{};
std::atomic<unsigned int> native_calls{};
std::atomic<unsigned int> launched_value{};
std::atomic<unsigned int> kernel_calls{};
std::atomic<unsigned int> event_count{};
std::atomic<unsigned int> stream_count{};
std::atomic<unsigned int> allocation_count{};
std::atomic<unsigned int> async_free_calls{};
std::atomic<CUstream> observed_record_stream{};
std::atomic<unsigned int> observed_stream_records{};
bool should_fail_destroy{};
bool should_fail_create{};
bool should_create_null{};
bool should_fail_record{};
std::atomic<unsigned int> destroy_calls{};
std::atomic<CUevent> failed_wait_record{};
struct Event {
    bool is_recorded{};
};
struct Stream {
    unsigned int flags{};
    int priority{};
};
void require(bool value, const char* message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}
void check(CUresult result, const char* operation) {
    if (result != CUDA_SUCCESS) {
        throw std::runtime_error(std::string(operation) + ": " + std::to_string(result));
    }
}
CUresult context(CUcontext* output) {
    ++native_calls;
    ++caller_context_queries;
    if (native_context_probe) {
        native_context_probe->count_down();
        native_context_probe = nullptr;
    }
    *output = current_context ? current_context : kContext;
    return CUDA_SUCCESS;
}
CUresult push(CUcontext value) {
    ++native_calls;
    current_context = value;
    return CUDA_SUCCESS;
}
CUresult pop(CUcontext* output) {
    ++native_calls;
    *output = current_context;
    current_context = nullptr;
    return CUDA_SUCCESS;
}
CUresult allocate(CUdeviceptr* output, std::size_t bytes) {
    ++native_calls;
    *output = reinterpret_cast<CUdeviceptr>(new std::byte[bytes]);
    ++allocation_count;
    return CUDA_SUCCESS;
}
CudaCooperativeReleaseResult free_device(CUdeviceptr value) {
    ++native_calls;
    delete[] std::bit_cast<std::byte*>(value);
    --allocation_count;
    return {CUDA_SUCCESS, CudaNativeReleaseState::kReleased};
}
CUresult free_async(CUdeviceptr value, CUstream) {
    ++async_free_calls;
    return free_device(value).status;
}
CUresult host_allocate(void** output, std::size_t bytes, unsigned int) {
    ++native_calls;
    *output = ::operator new(bytes, std::align_val_t{16});
    ++allocation_count;
    return CUDA_SUCCESS;
}
CudaCooperativeReleaseResult host_free(void* value) {
    ++native_calls;
    ::operator delete(value, std::align_val_t{16});
    --allocation_count;
    return {CUDA_SUCCESS, CudaNativeReleaseState::kReleased};
}
CUresult map(CUdeviceptr* output, void* value, unsigned int) {
    ++native_calls;
    *output = reinterpret_cast<CUdeviceptr>(value);
    return CUDA_SUCCESS;
}
CUresult device_to_host(void* destination, CUdeviceptr source, std::size_t bytes, CUstream) {
    ++native_calls;
    std::memcpy(destination, std::bit_cast<void*>(source), bytes);
    return CUDA_SUCCESS;
}
CUresult host_to_device(CUdeviceptr destination, const void* source, std::size_t bytes, CUstream) {
    ++native_calls;
    std::memcpy(std::bit_cast<void*>(destination), source, bytes);
    return CUDA_SUCCESS;
}
CUresult event_create(CUevent* output, unsigned int) {
    ++native_calls;
    if (should_create_null) {
        *output = nullptr;
        return CUDA_SUCCESS;
    }
    *output = reinterpret_cast<CUevent>(new Event{});
    ++event_count;
    return should_fail_create ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
}
CUresult event_record(CUevent value, CUstream stream) {
    ++native_calls;
    if (stream && stream == observed_record_stream.load()) {
        ++observed_stream_records;
    }
    reinterpret_cast<Event*>(value)->is_recorded = true;
    if (value == failed_wait_record.load()) {
        return CUDA_ERROR_LAUNCH_FAILED;
    }
    return should_fail_record ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
}
CUresult event_query(CUevent value) {
    ++native_calls;
    if (should_fail_epoch_query.load()) return CUDA_ERROR_LAUNCH_FAILED;
    return !reinterpret_cast<Event*>(value)->is_recorded || is_ready.load() ? CUDA_SUCCESS
                                                                            : CUDA_ERROR_NOT_READY;
}
CUresult event_destroy(CUevent value) {
    ++native_calls;
    ++destroy_calls;
    delete reinterpret_cast<Event*>(value);
    --event_count;
    return should_fail_destroy ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
}
CUresult stream_create(CUstream* output, unsigned int flags) {
    ++native_calls;
    *output = reinterpret_cast<CUstream>(new Stream{flags, 0});
    ++stream_count;
    return CUDA_SUCCESS;
}
CUresult stream_destroy(CUstream value) {
    ++native_calls;
    delete reinterpret_cast<Stream*>(value);
    --stream_count;
    return CUDA_SUCCESS;
}
CUresult stream_flags(CUstream value, unsigned int* output) {
    ++native_calls;
    *output = reinterpret_cast<Stream*>(value)->flags;
    return CUDA_SUCCESS;
}
CUresult stream_priority(CUstream value, int* output) {
    ++native_calls;
    *output = reinterpret_cast<Stream*>(value)->priority;
    return CUDA_SUCCESS;
}
CUresult stream_sync(CUstream) {
    ++native_calls;
    return is_ready.load() ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
}
CUresult wait(CUstream, CUevent, unsigned int) {
    ++native_calls;
    return CUDA_SUCCESS;
}
CUresult event_sync(CUevent value) {
    return event_query(value);
}
// NOLINTBEGIN(bugprone-easily-swappable-parameters): exact CUDA native ABI.
CUresult kernel(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int,
                unsigned int, unsigned int, CUstream, void** arguments, void**) {
    ++native_calls;
    ++kernel_calls;
    launched_value.store(*static_cast<unsigned int*>(arguments[0]));
    const auto counter = *static_cast<CUdeviceptr*>(arguments[4]);
    *std::bit_cast<std::uint64_t*>(counter) = 3;
    return CUDA_SUCCESS;
}
// NOLINTEND(bugprone-easily-swappable-parameters)
CudaFrontendDispatch dispatch() {
    CudaQueueDispatch queue;
    queue.context_get_current = context;
    queue.launch_kernel = kernel;
    queue.copy_host_to_device = host_to_device;
    queue.copy_device_to_host = device_to_host;
    queue.free_async = free_async;
    queue.event_create = event_create;
    queue.event_record = event_record;
    queue.event_query = event_query;
    queue.event_destroy = event_destroy;
    queue.stream_synchronize = stream_sync;
    queue.stream_wait_event = wait;
    queue.event_synchronize = event_sync;
    return {{queue, push, pop},
            {context, allocate, free_device, host_allocate, host_free, map, device_to_host,
             std::make_shared<int>(0)},
            stream_create,
            stream_destroy,
            stream_flags,
            stream_priority};
}
std::shared_ptr<glimmer::control::AsyncCooperativeLaunchGate> control() {
    glimmer::control::LaunchGateOptions options;
    options.scheduling_policy = glimmer::core::SchedulingPolicy::kPriority;
    return std::make_shared<glimmer::control::AsyncCooperativeLaunchGate>(options);
}

void native_handoff_needs_no_caller_subscription() {
    const auto previous_context = current_context;
    current_context = kContext;
    glimmer::control::LaunchGateOptions options;
    options.scheduling_policy = glimmer::core::SchedulingPolicy::kPriority;
    auto gate = std::make_shared<glimmer::control::AsyncCooperativeLaunchGate>(options, 2);
    CudaCooperativeFrontend frontend(kContext, dispatch(), gate);
    check(frontend.initialize(), "initialize worker before exhausting subscriptions");
    auto remaining = gate->subscribe_progress();
    require(remaining && !gate->subscribe_progress(),
            "fixture must exhaust progress subscriptions after the worker subscribes");
    unsigned int issued{};
    const auto caller = std::this_thread::get_id();
    for (unsigned int index = 0; index < 32; ++index) {
        check(frontend.native(
                  nullptr, false,
                  [&] {
                      require(std::this_thread::get_id() == caller && current_context == kContext,
                              "owner-driven handoff keeps original work on its caller/context");
                      ++issued;
                      return CUDA_SUCCESS;
                  },
                  true),
              "native handoff must not allocate a per-caller progress subscription");
        check(frontend.synchronize_stream(nullptr, false), "prove handed-off native completion");
    }
    check(frontend.shutdown(), "drain owner-driven native handoffs");
    const auto stats = frontend.stats();
    require(issued == 32 && stats.granted_executions == 32 &&
                stats.acknowledged_completions == 32 && stats.acknowledged_failures == 0,
            "each handoff requires its exact grant, original call and physical completion");
    remaining.reset();
    gate->shutdown();
    require(event_count == 0 && stream_count == 0 && allocation_count == 0,
            "owner-driven handoff must release every proven native resource");
    current_context = previous_context;
}
std::atomic<unsigned int> slice_count{};
std::array<std::atomic<unsigned int>, 4> slice_order{};
unsigned int gemm_slice_width{1};  // Set before issue; read only by the serialized owner.
// NOLINTBEGIN(bugprone-easily-swappable-parameters): exact CUDA native ABI.
CUresult mapped_kernel(CUfunction, unsigned int gx, unsigned int gy, unsigned int gz,
                       unsigned int bx, unsigned int by, unsigned int bz, unsigned int shared,
                       CUstream, void** arguments, void**) {
    int count{};
    std::uint64_t block{};
    if (bx == 256) {
        std::uint32_t rows{};
        std::memcpy(&rows, static_cast<const std::byte*>(arguments[0]) + 104, sizeof(rows));
        count = rows == 129 ? 2049 : 1024;
        std::memcpy(&block, static_cast<const std::byte*>(arguments[0]) + 176, sizeof(block));
        const auto planes = rows == 1 ? 1U
                            : gemm_slice_width == 1
                                ? 1U
                                : std::min(8U, 19U - static_cast<unsigned int>(block));
        require((rows == 129 || rows == 1) && gx == (rows == 129 ? 2U : 1U) && gy == gx &&
                    gz == planes && by == 1 && bz == 1 && shared == 0 &&
                    block % gemm_slice_width == 0,
                "frontend preserves GEMM XY and bounded physical range geometry");
        block /= gemm_slice_width;
    } else {
        require(gx == 1 && gy == 1 && gz == 1 && bx == 128 && by == 1 && bz == 1 && shared == 0,
                "frontend issues exactly one private CTA");
        count = *static_cast<int*>(arguments[0]);
        std::memcpy(&block, static_cast<const std::byte*>(arguments[2]) + 24, sizeof(block));
    }
    require((count == 2049 && block < 3) || (count == 1024 && block == 0),
            "frontend preserves copied arguments and the retained logical index");
    const auto slot = slice_count.load();
    require(slot < slice_order.size(), "frontend cannot replay a slice");
    slice_order[slot] = count == 1024 ? 10 : static_cast<unsigned int>(block);
    if (count == 2049 && block == 0) is_ready = false;
    slice_count = slot + 1;
    return CUDA_SUCCESS;
}
// NOLINTEND(bugprone-easily-swappable-parameters)
void sliced_priority_uses_physical_proof(bool is_gemm = false, bool uses_gemm_ranges = false) {
    gemm_slice_width = uses_gemm_ranges ? 8U : 1U;
    auto gate = control();
    auto native = dispatch();
    native.worker.queue.launch_kernel = mapped_kernel;
    CudaCooperativeFrontend frontend(kContext, native, gate);
    check(frontend.initialize(), "initialize sliced frontend");
    // Unwind must release fake physical work before frontend teardown.
    struct ReleaseProof {
        ~ReleaseProof() {
            is_ready = true;
        }
    } release_proof;
    current_context = kContext;
    CUstream low{}, high{};
    CUevent event{};
    check(stream_create(&low, CU_STREAM_NON_BLOCKING), "create low slice stream");
    check(stream_create(&high, CU_STREAM_NON_BLOCKING), "create high slice stream");
    reinterpret_cast<Stream*>(high)->priority = -1;
    check(event_create(&event, 0), "create slice event");
    check(frontend.bind_event(event, std::make_shared<int>(0)), "bind slice event");
    auto resources = std::make_shared<int>(1);
    std::weak_ptr<void> lifetime = resources;
    auto prepare = [&](int count) {
        if (is_gemm) {
            alignas(8) std::array<std::byte, 176> bytes{};
            const auto put = [&]<typename T>(std::size_t offset, T value) {
                std::memcpy(bytes.data() + offset, &value, sizeof(value));
            };
            const unsigned int m = count == 2049 ? 129 : 1;
            const unsigned int n = count == 2049 ? 33 : 1;
            for (auto offset : {0U, 8U, 16U, 168U})
                put(offset, std::uint64_t{0x10000} * (offset + 1));
            put(24, std::uint64_t{m} * 4);
            put(32, std::uint64_t{n} * 4);
            put(40, std::uint64_t{m} * 32);
            put(48, std::uint64_t{n} * 32);
            put(56, std::uint64_t{m});
            put(64, std::uint64_t{n});
            put(72, std::uint64_t{m} * n);
            put(80, std::uint64_t{m} * n);
            for (auto offset : {88U, 96U, 100U, 104U}) put(offset, m);
            for (auto offset : {92U, 108U}) put(offset, n);
            for (auto offset : {112U, 128U, 132U}) put(offset, std::uint32_t{1});
            std::array<void*, 1> arguments{bytes.data()};
            auto kernel = prepare_mapped_gemm_slices(
                reinterpret_cast<CUfunction>(0x2000),
                {(m + 127) / 128, (n + 31) / 32,
                 count == 2049 ? (uses_gemm_ranges ? 19U : 3U) : 1U},
                {256, 1, 1}, arguments.data(), resources, false,
                uses_gemm_ranges ? QualifiedGemmMapping::kRange : QualifiedGemmMapping::kPlane);
            require(kernel.has_value(), "prepare trusted owned GEMM planes");
            bytes.fill(std::byte{0xff});
            return std::move(*kernel);
        }
        float alpha = 1;
        std::array<std::uint64_t, 3> pointers{0x10000, 0x20000, 0x30000};
        std::array<void*, 3> arguments{&count, &alpha, pointers.data()};
        auto kernel =
            prepare_mapped_add_slices(reinterpret_cast<CUfunction>(0x2000),
                                      {(static_cast<unsigned int>(count) + 1023) / 1024, 1, 1},
                                      {128, 1, 1}, arguments.data(), resources);
        require(kernel.has_value(), "prepare trusted owned slice");
        count = 0;
        pointers.fill(0);
        return std::move(*kernel);
    };
    require(
        frontend.launch_sliced(nullptr, low, false, prepare(1024)) == CUDA_ERROR_INVALID_VALUE &&
            slice_count == 0,
        "foreign context rejects before issue");
    for (const auto invalid :
         {std::array{0U, 1U, 1U}, std::array{1U, 0U, 1U}, std::array{1U, 1U, 0U},
          std::array{65536U, 65536U, 1U}, std::array{65536U, 1U, 65536U}}) {
        auto kernel = prepare(1024);
        kernel.grid = invalid;
        require(frontend.launch_sliced(kContext, low, false, std::move(kernel)) ==
                        CUDA_ERROR_INVALID_VALUE &&
                    slice_count == 0,
                "empty or overflowed logical XYZ must reject before admission");
    }
    // Establish high binding before holding low's physical event pending.
    check(frontend.launch_sliced(kContext, high, false, prepare(1024)), "warm high binding");
    check(frontend.synchronize_stream(high, false), "complete warm high slice");
    slice_count = 0;
    check(frontend.launch_sliced(kContext, low, false, prepare(2049)), "accept low slices");
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{3};
    while (slice_count == 0 && std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
    require(slice_count == 1, "first low slice must physically issue");
    check(frontend.record_event(event, low, false), "record full logical slice completion");
    check(frontend.launch_sliced(kContext, high, false, prepare(1024)), "accept priority slice");
    resources.reset();
    while (frontend.stats().retreat_requests == 0 && std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
    require(frontend.stats().retreat_requests > 0 && slice_count == 1 && !lifetime.expired() &&
                frontend.stats().acknowledged_suspensions == 0 &&
                frontend.query_event(event) == CUDA_ERROR_NOT_READY,
            "priority request cannot replace physical stop or complete a logical event");
    is_ready = true;
    check(frontend.synchronize_stream(high, false), "complete priority insertion");
    check(frontend.synchronize_event(event), "complete retained low continuation");
    require(slice_count == 4 && slice_order[0] == 0 && slice_order[1] == 10 &&
                slice_order[2] == 1 && slice_order[3] == 2,
            "higher priority runs between low slices with no loss or replay");
    check(frontend.close_event(event), "close sliced event");
    check(event_destroy(event), "destroy sliced event");
    check(frontend.close_stream(low), "close low slice stream");
    check(frontend.close_stream(high), "close high slice stream");
    check(stream_destroy(low), "destroy low slice stream");
    check(stream_destroy(high), "destroy high slice stream");
    check(frontend.shutdown(), "drain sliced frontend");
    const auto stats = frontend.stats();
    require(stats.granted_executions == 5 && stats.acknowledged_suspensions == 2 &&
                stats.logical_resumes == 2 && stats.acknowledged_completions == 3 &&
                stats.acknowledged_failures == 0 && lifetime.expired(),
            "fresh grants and physical acknowledgments retain then release owned resources");
    gate->shutdown();
    require(event_count == 0 && stream_count == 0 && allocation_count == 0,
            "sliced frontend releases every owned resource");
}
void bindings_and_owned_arguments() {
    auto gate = control();
    CudaCooperativeFrontend frontend(
        kContext, dispatch(), gate,
        {.progress_slots = 4, .import_events = 12, .per_thread_streams = 2});
    check(frontend.initialize(), "initialize bounded frontend");
    current_context = reinterpret_cast<CUcontext>(0x3000);
    require(frontend.synchronize_context() == CUDA_ERROR_INVALID_CONTEXT,
            "wrong caller context must reject without owner mutation");
    current_context = kContext;
    auto resources = std::make_shared<int>(0);
    CUstream stream = nullptr;
    CUevent event = nullptr;
    check(stream_create(&stream, CU_STREAM_NON_BLOCKING), "create original stream");
    check(event_create(&event, 0), "create original event");
    check(frontend.bind_event(event, resources), "bind original event");
    require(frontend.bind_event(event, resources) == CUDA_ERROR_INVALID_HANDLE,
            "duplicate event binding must reject without native issue");
    check(frontend.query_event(event), "unrecorded event has no logical predecessor");
    is_ready.store(false);
    unsigned int value = 123;
    std::array<void*, 1> arguments{&value};
    CooperativeKernelVariant variant{
        kContext, reinterpret_cast<CUfunction>(0x2000), {{4, 4}}, {3, 1}, 1};
    check(
        frontend.launch(variant, stream, false, {3, 1, 1}, {1, 1, 1}, arguments.data(), resources),
        "accept original argument storage");
    value = 999;
    arguments.fill(nullptr);
    check(frontend.record_event(event, stream, false), "record logical completion");
    require(kernel_calls.load() == 0, "unproved native prefix must block physical issue");
    require(frontend.query_event(event) == CUDA_ERROR_NOT_READY,
            "native unrecorded event success cannot complete the logical record");
    require(frontend.query_stream(stream, false) == CUDA_ERROR_NOT_READY,
            "stream query must retain queued invocation");
    is_ready.store(true);
    check(frontend.synchronize_event(event), "await immutable event record");
    require(kernel_calls.load() == 1 && launched_value.load() == 123,
            "deferred issue must use exactly one owned original argument copy");
    check(frontend.wait_event(nullptr, true, event), "queue caller PTDS event dependency");
    check(frontend.synchronize_stream(nullptr, true), "synchronize caller PTDS proxy");
    for (unsigned int iteration = 0; iteration < 40; ++iteration) {
        CUresult result = CUDA_ERROR_UNKNOWN;
        std::thread caller([&] { result = frontend.synchronize_stream(nullptr, true); });
        caller.join();
        check(result, "recycle a physically stopped PTDS proxy after caller lifetime ends");
    }
    check(frontend.close_event(event), "close event after logical users drain");
    check(event_destroy(event), "destroy original event after frontend close");
    check(frontend.close_stream(stream), "close original stream after logical drain");
    check(stream_destroy(stream), "destroy original stream");
    check(frontend.shutdown(), "drain and release frontend");
    require(frontend.stats().acknowledged_completions == 1,
            "one logical invocation must report once");
    gate->shutdown();
    require(event_count.load() == 0 && stream_count.load() == 0 && allocation_count.load() == 0,
            "checked teardown must release every owned native resource");
}
void reversible_checkpoint_preserves_dependencies() {
    auto gate = control();
    CudaCooperativeFrontend frontend(
        kContext, dispatch(), gate,
        {.progress_slots = 2, .import_events = 2, .per_thread_streams = 2});
    check(frontend.initialize(), "initialize reversible checkpoint owner");
    CUstream stream{};
    CUevent event{};
    check(stream_create(&stream, CU_STREAM_NON_BLOCKING), "create checkpoint original stream");
    check(event_create(&event, CU_EVENT_DEFAULT), "create checkpoint original event");
    check(frontend.bind_event(event, std::make_shared<int>(1)), "bind checkpoint event");
    is_ready = false;
    check(frontend.record_event(event, stream, false), "accept pending checkpoint record");
    CudaWorkerCheckpoint checkpoint;
    require(frontend.checkpoint(checkpoint) == CUDA_ERROR_NOT_READY && !checkpoint,
            "pending work cannot be checkpointed or canceled as completed");
    is_ready = true;
    check(frontend.synchronize_event(event), "prove pending checkpoint record completion");
    CudaFrontendEventWait historic;
    check(frontend.prepare_event_wait(event, historic), "retain immutable checkpoint dependency");
    const auto events_before = event_count.load();
    const auto streams_before = stream_count.load();
    const auto allocations_before = allocation_count.load();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{3};
    CUresult result{};
    do {
        result = frontend.checkpoint(checkpoint);
        require(result == CUDA_SUCCESS || result == CUDA_ERROR_NOT_READY,
                "checkpoint admission has no unexpected failure");
        std::this_thread::yield();
    } while (result == CUDA_ERROR_NOT_READY && std::chrono::steady_clock::now() < deadline);
    check(result, "join completed checkpoint worker");
    const auto parked_calls = native_calls.load();
    require(frontend.shutdown() == CUDA_ERROR_NOT_READY &&
                frontend.resume_checkpoint({}) == CUDA_ERROR_INVALID_VALUE &&
                native_calls == parked_calls && event_count == events_before &&
                stream_count == streams_before && allocation_count == allocations_before,
            "parked owner keeps resources and rejects cleanup or empty resume without CUDA");
    current_context = reinterpret_cast<CUcontext>(0x2000);
    require(frontend.resume_checkpoint(checkpoint) == CUDA_ERROR_INVALID_CONTEXT,
            "wrong caller cannot activate a checkpoint worker");
    const auto wrong_binding_calls = native_calls.load();
    require(frontend.release_checkpoint_progress(checkpoint) == CUDA_ERROR_INVALID_CONTEXT &&
                native_calls == wrong_binding_calls + 1 && allocation_count == allocations_before,
            "wrong release binding cannot free resources or poison surviving ownership");
    current_context = kContext;
    require(
        frontend.retire_checkpoint_after_context_destroy(checkpoint) == CUDA_ERROR_INVALID_VALUE,
        "unreleased native pool forbids CPU-only destruction retirement");
    check(frontend.release_checkpoint_progress(checkpoint), "release live empty checkpoint pool");
    const auto released_calls = native_calls.load();
    require(allocation_count == allocations_before - 2 && event_count == events_before &&
                stream_count == streams_before &&
                frontend.release_checkpoint_progress(checkpoint) == CUDA_ERROR_INVALID_VALUE &&
                frontend.resume_checkpoint(checkpoint) == CUDA_ERROR_INVALID_VALUE &&
                frontend.restore_checkpoint_progress({}) == CUDA_ERROR_INVALID_VALUE &&
                native_calls == released_calls,
            "pool release preserves original identities and rejects unsafe/repeated transitions");
    current_context = reinterpret_cast<CUcontext>(0x2000);
    require(frontend.restore_checkpoint_progress(checkpoint) == CUDA_ERROR_INVALID_CONTEXT &&
                allocation_count == allocations_before - 2 && native_calls == released_calls + 1,
            "wrong target binding rejects before new pool ownership or worker activation");
    current_context = kContext;
    check(frontend.restore_checkpoint_progress(checkpoint),
          "restore proven surviving private pool");
    const auto restored_calls = native_calls.load();
    require(frontend.restore_checkpoint_progress(checkpoint) == CUDA_ERROR_INVALID_VALUE &&
                native_calls == restored_calls,
            "restored pool cannot be duplicated under the same checkpoint");
    check(frontend.resume_checkpoint(checkpoint), "resume proven surviving original owner");
    require(frontend.resume_checkpoint(checkpoint) == CUDA_ERROR_INVALID_VALUE,
            "consumed checkpoint cannot replay a handoff");
    check(frontend.query_event(event), "same event binding retains completed record");
    check(frontend.wait_event_snapshot(nullptr, true, historic),
          "same queue accepts historical prefix after joined handoff");
    check(frontend.synchronize_stream(nullptr, true), "prove preserved PTDS wait dependency");
    check(frontend.record_event(event, stream, false), "rerecord surviving event after checkpoint");
    check(frontend.synchronize_event(event), "prove new event generation after checkpoint");
    require(event_count == events_before && stream_count == streams_before &&
                allocation_count == allocations_before,
            "progress turnover preserves original handles and restores private resource counts");
    const auto stale = checkpoint;
    checkpoint = {};
    check(frontend.checkpoint(checkpoint), "join second completed progress generation");
    const auto second_parked_calls = native_calls.load();
    require(frontend.release_checkpoint_progress(stale) == CUDA_ERROR_INVALID_VALUE &&
                native_calls == second_parked_calls,
            "stale checkpoint cannot free a newer progress generation");
    check(frontend.release_checkpoint_progress(checkpoint),
          "release second exact progress generation");
    const auto second_released_calls = native_calls.load();
    require(
        frontend.restore_checkpoint_progress(stale) == CUDA_ERROR_INVALID_VALUE &&
            frontend.retire_checkpoint_after_context_destroy(stale) == CUDA_ERROR_INVALID_VALUE &&
            native_calls == second_released_calls,
        "stale checkpoint cannot restore or retire a newer joined pool-free owner");
    check(frontend.restore_checkpoint_progress(checkpoint),
          "restore second exact progress generation");
    check(frontend.resume_checkpoint(checkpoint), "resume second exact queue generation");
    check(frontend.close_event(event), "retire checkpoint event after resumed completion");
    check(event_destroy(event), "destroy original checkpoint event");
    check(frontend.close_stream(stream), "retire resumed checkpoint stream");
    check(stream_destroy(stream), "destroy original checkpoint stream");
    check(frontend.shutdown(), "check ordinary cleanup after explicit resume");
    gate->shutdown();
    require(event_count == 0 && stream_count == 0 && allocation_count == 0,
            "resumed checkpoint releases all resources exactly once");
}

// The empty checkpoint fixture creates only private handles on the caller.
// Keep their original fake-context lifetime separate from actor CPU retirement.
std::vector<CUevent> checkpoint_context_events;
std::vector<CUstream> checkpoint_context_streams;
CUresult checkpoint_event_create(CUevent* output, unsigned int flags) {
    const auto result = event_create(output, flags);
    if (result == CUDA_SUCCESS) {
        checkpoint_context_events.push_back(*output);
    }
    return result;
}
CUresult checkpoint_stream_create(CUstream* output, unsigned int flags) {
    const auto result = stream_create(output, flags);
    if (result == CUDA_SUCCESS) {
        checkpoint_context_streams.push_back(*output);
    }
    return result;
}
void checkpoint_retirement_is_cuda_free() {
    auto gate = control();
    auto native_dispatch = dispatch();
    native_dispatch.worker.queue.event_create = checkpoint_event_create;
    native_dispatch.stream_create = checkpoint_stream_create;
    auto frontend = std::make_unique<CudaCooperativeFrontend>(
        kContext, native_dispatch, gate,
        CudaFrontendOptions{.progress_slots = 2, .import_events = 2, .per_thread_streams = 2});
    check(frontend->initialize(), "initialize CPU-only retirement owner");
    CudaWorkerCheckpoint checkpoint;
    check(frontend->checkpoint(checkpoint), "join empty retirement owner");
    const auto parked = native_calls.load();
    require(
        frontend->retire_checkpoint_after_context_destroy(checkpoint) == CUDA_ERROR_INVALID_VALUE &&
            native_calls == parked && allocation_count == 2,
        "retirement cannot abandon live native progress allocations");
    check(frontend->release_checkpoint_progress(checkpoint), "release pool before original effect");
    require(allocation_count == 0, "live pool teardown frees device and mapped host storage");
    // Model the independently proven original context effect, not frontend
    // cleanup: these native resources cease to exist before CPU retirement.
    for (const auto event : checkpoint_context_events) {
        delete reinterpret_cast<Event*>(event);
        --event_count;
    }
    for (const auto stream : checkpoint_context_streams) {
        delete reinterpret_cast<Stream*>(stream);
        --stream_count;
    }
    checkpoint_context_events.clear();
    checkpoint_context_streams.clear();
    current_context = reinterpret_cast<CUcontext>(0xdead);
    const auto destroyed = native_calls.load();
    check(frontend->retire_checkpoint_after_context_destroy(checkpoint),
          "retire independently destroyed context CPU ownership");
    require(
        frontend->retire_checkpoint_after_context_destroy(checkpoint) == CUDA_ERROR_INVALID_VALUE &&
            frontend->restore_checkpoint_progress(checkpoint) == CUDA_ERROR_INVALID_VALUE &&
            frontend->resume_checkpoint(checkpoint) == CUDA_ERROR_INVALID_VALUE &&
            frontend->query_event(reinterpret_cast<CUevent>(0x3000)) == CUDA_ERROR_DEINITIALIZED &&
            frontend->stats().acknowledged_completions == 0 && native_calls == destroyed,
        "retired actor rejects reuse without querying or reactivating destroyed handles");
    check(frontend->shutdown(), "idempotent closed CPU teardown");
    frontend.reset();
    gate->shutdown();
    require(
        native_calls == destroyed && event_count == 0 && stream_count == 0 && allocation_count == 0,
        "retired frontend destructors do not touch any destroyed CUDA handle");
    current_context = kContext;
}

void checkpoint_progress_failure_retains_ownership() {
    for (const bool fails_restore : {false, true}) {
        const auto child = fork();
        require(child >= 0, "fork isolated uncertain checkpoint pool");
        if (child == 0) {
            try {
                auto gate = control();
                auto native_dispatch = dispatch();
                unsigned int release_calls{}, allocation_calls{};
                bool should_fail{};
                native_dispatch.pool.free_device = [&](CUdeviceptr pointer) {
                    ++release_calls;
                    auto result = free_device(pointer);
                    if (should_fail && !fails_restore) {
                        result.status = CUDA_ERROR_LAUNCH_FAILED;
                    }
                    return result;
                };
                native_dispatch.pool.allocate_device = [&](CUdeviceptr* output, std::size_t bytes) {
                    ++allocation_calls;
                    const auto result = allocate(output, bytes);
                    return should_fail && fails_restore ? CUDA_ERROR_LAUNCH_FAILED : result;
                };
                auto frontend = std::make_unique<CudaCooperativeFrontend>(
                    kContext, native_dispatch, gate,
                    CudaFrontendOptions{.progress_slots = 2, .import_events = 2});
                check(frontend->initialize(), "initialize uncertain checkpoint pool owner");
                CudaWorkerCheckpoint checkpoint;
                check(frontend->checkpoint(checkpoint), "join uncertain pool owner");
                if (fails_restore) {
                    check(frontend->release_checkpoint_progress(checkpoint),
                          "release old pool before partial restore");
                }
                should_fail = true;
                const auto status = fails_restore
                                        ? frontend->restore_checkpoint_progress(checkpoint)
                                        : frontend->release_checkpoint_progress(checkpoint);
                const auto failed_calls = native_calls.load();
                require(status == CUDA_ERROR_LAUNCH_FAILED &&
                            frontend->restore_checkpoint_progress(checkpoint) ==
                                CUDA_ERROR_INVALID_VALUE &&
                            frontend->release_checkpoint_progress(checkpoint) ==
                                CUDA_ERROR_INVALID_VALUE &&
                            frontend->resume_checkpoint(checkpoint) == CUDA_ERROR_INVALID_VALUE &&
                            frontend->retire_checkpoint_after_context_destroy(checkpoint) ==
                                CUDA_ERROR_INVALID_VALUE &&
                            frontend->shutdown() == CUDA_ERROR_NOT_READY && release_calls == 1 &&
                            allocation_calls == (fails_restore ? 2U : 1U) &&
                            native_calls == failed_calls,
                        "uncertain pool effect preserves exact error without retry or restart");
                frontend.reset();
                require(native_calls == failed_calls,
                        "unfinished owner destruction never retries uncertain CUDA effects");
                // This is intentional uncertainty quarantine, not leak-free
                // ordinary cleanup. Verify it before terminating the child.
                _exit(EXIT_SUCCESS);
            } catch (const std::exception& error) {
                std::cerr << "checkpoint pool quarantine failure: " << error.what() << '\n';
                _exit(EXIT_FAILURE);
            }
        }
        int status{};
        pid_t waited{};
        do {
            waited = waitpid(child, &status, 0);
        } while (waited < 0 && errno == EINTR);
        require(waited == child && WIFEXITED(status) && WEXITSTATUS(status) == EXIT_SUCCESS,
                "uncertain pool child must prove ownership before termination");
    }
}

void queued_release_retains_predecessors_and_resources() {
    struct Observer final : CudaQueueReleaseObserver {
        std::atomic<unsigned int> confirmations{};
        std::atomic<unsigned int> context_destructions{};
        void confirmed_release(CUdeviceptr) noexcept override {
            ++confirmations;
        }
        void context_destroyed(CUdeviceptr) noexcept override {
            ++context_destructions;
        }
    };
    auto gate = control();
    CudaCooperativeFrontend frontend(
        kContext, dispatch(), gate,
        {.progress_slots = 2, .import_events = 4, .per_thread_streams = 1});
    check(frontend.initialize(), "initialize release frontend");
    current_context = kContext;
    check(frontend.synchronize_stream(nullptr, true), "prime release PTDS binding");
    CUdeviceptr pointer{};
    check(allocate(&pointer, 16), "allocate original release storage");
    auto resources = std::make_shared<int>(0);
    auto observer = std::make_shared<Observer>();
    const auto before = async_free_calls.load();
    const auto rejected = frontend.release(nullptr, true, {pointer, {}, observer});
    require(rejected.status == CUDA_ERROR_INVALID_VALUE && !rejected.has_owned_release &&
                async_free_calls.load() == before,
            "invalid admission retains caller ownership and makes no native free");
    current_context = reinterpret_cast<CUcontext>(0xdead);
    const auto wrong_context = frontend.release(nullptr, true, {pointer, resources, observer});
    require(wrong_context.status == CUDA_ERROR_INVALID_CONTEXT &&
                !wrong_context.has_owned_release && async_free_calls.load() == before &&
                observer->confirmations.load() == 0,
            "wrong caller context must reject before native free and retain release ownership");
    current_context = kContext;
    is_ready.store(false);
    unsigned int value = 321;
    std::array<void*, 1> arguments{&value};
    CooperativeKernelVariant variant{
        kContext, reinterpret_cast<CUfunction>(0x2000), {{4, 4}}, {3, 1}, 1};
    check(
        frontend.launch(variant, nullptr, true, {3, 1, 1}, {1, 1, 1}, arguments.data(), resources),
        "queue release predecessor");
    auto release_resources = std::make_shared<int>(0);
    std::weak_ptr<int> retained = release_resources;
    const auto accepted = frontend.release(nullptr, true, {pointer, release_resources, observer});
    check(accepted.status, "accept release behind unfinished predecessor");
    release_resources.reset();
    require(accepted.has_owned_release && !retained.expired() &&
                async_free_calls.load() == before && observer->confirmations.load() == 0,
            "acceptance must neither block on predecessor nor issue/confirm free early");
    is_ready.store(true);
    check(frontend.synchronize_stream(nullptr, true), "prove queued release completion");
    require(async_free_calls.load() == before + 1 && observer->confirmations.load() == 1 &&
                observer->context_destructions.load() == 0 && retained.expired(),
            "native free executes once and physical proof releases payload ownership");
    check(frontend.shutdown(), "release frontend teardown");
    gate->shutdown();
    require(event_count.load() == 0 && stream_count.load() == 0 && allocation_count.load() == 0,
            "release frontend must not leak native resources");
}
CUresult session_context(CUcontext target,
                         const CudaCooperativeFrontend::NativeOperation& operation) {
    const auto previous = current_context;
    current_context = target;
    CUresult result = CUDA_ERROR_UNKNOWN;
    try {
        result = operation();
    } catch (...) {
        current_context = previous;
        throw;
    }
    current_context = previous;
    return result;
}

void retained_process_native_session() {
    const auto second_context = reinterpret_cast<CUcontext>(0x2000);
    const auto native_context = reinterpret_cast<CUcontext>(0x3000);
    auto gate = control();
    const CudaFrontendOptions options{
        .progress_slots = 2, .import_events = 2, .per_thread_streams = 0};
    auto first = std::make_shared<CudaCooperativeFrontend>(kContext, dispatch(), gate, options);
    auto second =
        std::make_shared<CudaCooperativeFrontend>(second_context, dispatch(), gate, options);
    current_context = kContext;
    check(first->initialize(), "initialize first session context");
    current_context = second_context;
    check(second->initialize(), "initialize second session context");
    current_context = kContext;
    // Repeat without recreating workers: tokens and trailing imports must not
    // accumulate or authorize a later session after their lifetime ends.
    CudaProcessNativeToken previous;
    for (unsigned int iteration = 0; iteration < 40; ++iteration) {
        CUevent event{};
        check(event_create(&event, CU_EVENT_DISABLE_TIMING), "create session retirement event");
        check(first->bind_event(event, std::make_shared<int>(0)), "bind session retirement event");
        CUstream stream{};
        check(stream_create(&stream, CU_STREAM_NON_BLOCKING), "create session retirement stream");
        check(first->synchronize_stream(stream, false), "bind actual session retirement stream");
        auto reservation = std::make_shared<int>(0);
        std::weak_ptr<int> held = reservation;
        CudaProcessNativeSession session(
            {{kContext, first, first}, {second_context, second, second}}, session_context,
            reservation, 3);
        reservation.reset();
        CudaProcessNativeToken token;
        unsigned int verifier_calls{};
        std::vector<CUcontext> completed;
        const auto complete = [&] {
            completed.push_back(current_context);
            return stream_sync(CU_STREAM_LEGACY);
        };
        require(session.finish(
                    token, [] { return CUDA_SUCCESS; }, complete) == CUDA_ERROR_INVALID_VALUE &&
                    completed.empty(),
                "unbegun session cannot invoke proof or thaw");
        check(session.begin(token), "retain process pause after begin returns");
        const auto paused_calls = native_calls.load();
        require(session.retire_event({}, kContext, event, true) == CUDA_ERROR_INVALID_VALUE &&
                    session.retire_event(token, native_context, event, true) ==
                        CUDA_ERROR_INVALID_CONTEXT,
                "missing capability/context cannot retire session metadata");
        check(session.retire_event(token, kContext, event, false),
              "prepare exact retained session event retirement");
        if (previous) {
            require(
                session.retire_event(previous, kContext, event, true) == CUDA_ERROR_INVALID_VALUE,
                "old session token cannot retire current actor event");
        }
        check(session.retire_event(token, kContext, event, true),
              "retire logical event under exact session pause");
        require(session.retire_stream({}, kContext, stream, true) == CUDA_ERROR_INVALID_VALUE &&
                    session.retire_stream(token, native_context, stream, true) ==
                        CUDA_ERROR_INVALID_CONTEXT &&
                    (!previous || session.retire_stream(previous, kContext, stream, true) ==
                                      CUDA_ERROR_INVALID_VALUE),
                "stream retirement requires exact live session/context capability");
        check(session.retire_stream(token, kContext, stream, false),
              "prepare exact retained session stream retirement");
        check(session.retire_stream(token, kContext, stream, true),
              "retire session logical stream without activating any native context");
        require(session.retire_stream(token, kContext, stream, true) == CUDA_ERROR_INVALID_HANDLE,
                "session stream metadata is retired once");
        require(native_calls.load() == paused_calls && session.is_paused(),
                "session retirement does not activate context, call CUDA or thaw");
        require(session.is_paused() && token && !held.expired() &&
                    native_calls.load() == paused_calls && session.status() == CUDA_ERROR_NOT_READY,
                "live tokens retain the reservation and reject private worker polling");
        require(session.begin(token) == CUDA_ERROR_INVALID_VALUE,
                "a live pause cannot be begun or overwritten twice");
        if (previous) {
            require(session.finish(
                        previous,
                        [&] {
                            ++verifier_calls;
                            return CUDA_SUCCESS;
                        },
                        complete) == CUDA_ERROR_INVALID_VALUE &&
                        verifier_calls == 0 && completed.empty(),
                    "a previous session capability cannot thaw current workers");
        }
        require(session.finish(
                    token,
                    [&] {
                        ++verifier_calls;
                        return CUDA_ERROR_NOT_READY;
                    },
                    complete) == CUDA_ERROR_NOT_READY &&
                    verifier_calls == 1 && completed.empty() && !held.expired(),
                "active or invalidated capture classification retains the exact pause");
        check(session.observe_native_context(native_context),
              "retain newly observed native context");
        check(session.observe_native_context(native_context),
              "native context observation is idempotent");
        require(session.observe_native_context(nullptr) == CUDA_ERROR_INVALID_VALUE,
                "null context observation cannot authorize completion");
        check(session.finish(
                  token,
                  [&] {
                      ++verifier_calls;
                      require(session.finish(
                                  token, [] { return CUDA_SUCCESS; }, complete) ==
                                  CUDA_ERROR_INVALID_VALUE,
                              "proof callback cannot reenter finish and replay native completion");
                      return CUDA_SUCCESS;
                  },
                  [&] {
                      if (completed.size() < 3) {
                          const auto before = native_calls.load();
                          require(
                              session.is_paused() && native_calls.load() == before,
                              "all contexts must prove native completion before any worker thaw");
                      }
                      return complete();
                  }),
              "finish all-context native session with exact physical proof");
        require(
            completed == std::vector<CUcontext>{kContext, second_context, native_context, kContext,
                                                second_context} &&
                verifier_calls == 2 && held.expired() && session.status() == CUDA_SUCCESS,
            "native-only contexts participate in proof and reservation releases only after thaw");
        const auto proofs = completed.size();
        require(session.retire_event(token, kContext, event, true) == CUDA_ERROR_INVALID_VALUE,
                "finished session cannot retire any event");
        check(event_destroy(event), "destroy simulated retired session original event");
        require(session.finish(
                    token, [] { return CUDA_SUCCESS; }, complete) == CUDA_ERROR_INVALID_VALUE &&
                    completed.size() == proofs &&
                    session.observe_native_context(native_context) == CUDA_ERROR_INVALID_VALUE,
                "finished capabilities cannot replay completion or accept new native work");
        check(stream_destroy(stream), "destroy simulated retired session original stream");
        previous = token;
        check(first->synchronize_context(), "drain first imported native fence");
        current_context = second_context;
        check(second->synchronize_context(), "drain second imported native fence");
        current_context = kContext;
    }
    check(first->shutdown(), "release first process-session frontend");
    current_context = second_context;
    check(second->shutdown(), "release second process-session frontend");
    current_context = kContext;
    gate->shutdown();
    require(event_count == 0 && stream_count == 0 && allocation_count == 0,
            "finished process sessions must not leak native owners");
}

void failed_process_session_retains_ownership(unsigned int mode) {
    const auto child = ::fork();
    require(child >= 0, "process-session failure child must start");
    if (child == 0) {
        try {
            // Intentional failed/forgotten pause quarantine is process-lifetime
            // ownership, not a leak-free teardown or capture recovery claim.
            const auto second_context = reinterpret_cast<CUcontext>(0x2000);
            auto gate = control();
            const CudaFrontendOptions options{
                .progress_slots = 2, .import_events = 2, .per_thread_streams = 0};
            auto first =
                std::make_shared<CudaCooperativeFrontend>(kContext, dispatch(), gate, options);
            auto second = std::make_shared<CudaCooperativeFrontend>(second_context, dispatch(),
                                                                    gate, options);
            current_context = kContext;
            check(first->initialize(), "initialize failed-session first owner");
            current_context = second_context;
            check(second->initialize(), "initialize failed-session second owner");
            current_context = kContext;
            auto reservation = std::make_shared<int>(0);
            std::weak_ptr<int> retained = reservation;
            unsigned int activation_calls{};
            auto session = std::make_unique<CudaProcessNativeSession>(
                std::vector<CudaProcessNativeContext>{{kContext, first, first},
                                                      {second_context, second, second}},
                [&](CUcontext context, const auto& operation) {
                    ++activation_calls;
                    if (mode == 0 && activation_calls == 2) {
                        return CUDA_ERROR_INVALID_CONTEXT;
                    }
                    return session_context(context, operation);
                },
                reservation, 2);
            reservation.reset();
            CudaProcessNativeToken token;
            unsigned int verifier_calls{};
            unsigned int completion_calls{};
            if (mode == 0) {
                require(session->begin(token) == CUDA_ERROR_INVALID_CONTEXT && !token,
                        "partial pause must reject without publishing a native capability");
            } else {
                check(session->begin(token), "begin failed or forgotten session");
                if (mode != 4) {
                    CUresult expected = CUDA_ERROR_LAUNCH_FAILED;
                    CUresult result = CUDA_ERROR_UNKNOWN;
                    if (mode == 3) {
                        expected = CUDA_ERROR_OUT_OF_MEMORY;
                        result =
                            session->observe_native_context(reinterpret_cast<CUcontext>(0x3000));
                    } else {
                        if (mode == 5) {
                            expected = CUDA_ERROR_UNKNOWN;
                        }
                        result = session->finish(
                            token,
                            [&] {
                                ++verifier_calls;
                                if (mode == 5) {
                                    throw std::runtime_error(
                                        "injected inactive verifier exception");
                                }
                                return mode == 1 ? CUDA_ERROR_LAUNCH_FAILED : CUDA_SUCCESS;
                            },
                            [&] {
                                ++completion_calls;
                                require(session->is_paused(),
                                        "no context may thaw before all native completion checks "
                                        "succeed");
                                return completion_calls == 2 ? CUDA_ERROR_LAUNCH_FAILED
                                                             : CUDA_SUCCESS;
                            });
                    }
                    require(
                        result == expected && session->status() == expected &&
                            first->query_stream(CU_STREAM_LEGACY, false) == expected &&
                            second->query_stream(CU_STREAM_LEGACY, false) == expected,
                        "failed proof must retain the pause and wake frontend callers to error");
                }
            }
            const auto verifications = verifier_calls;
            const auto completions = completion_calls;
            require(!retained.expired(), "failed or forgotten session retains process reservation");
            if (mode != 4) {
                require(session->finish(
                            token,
                            [&] {
                                ++verifier_calls;
                                return CUDA_SUCCESS;
                            },
                            [&] {
                                ++completion_calls;
                                return CUDA_SUCCESS;
                            }) == CUDA_ERROR_INVALID_VALUE &&
                            verifier_calls == verifications && completion_calls == completions,
                        "failure cannot replay verification, completion or token import");
            }
            session.reset();
            require(!retained.expired(),
                    "destruction cannot auto-thaw or release uncertain ownership");
            ::_exit(EXIT_SUCCESS);
        } catch (...) {
            ::_exit(EXIT_FAILURE);
        }
    }
    int status{};
    pid_t result{};
    do {
        result = ::waitpid(child, &status, 0);
    } while (result < 0 && errno == EINTR);
    require(result == child && WIFEXITED(status) && WEXITSTATUS(status) == EXIT_SUCCESS,
            "process-native failure retains ownership without replay or implicit thaw");
}

void native_transaction_keeps_rejected_payload() {
    auto gate = control();
    CudaCooperativeFrontend frontend(
        kContext, dispatch(), gate,
        {.progress_slots = 4, .import_events = 12, .per_thread_streams = 2});
    check(frontend.initialize(), "initialize transaction frontend");
    CUstream stream = nullptr;
    check(stream_create(&stream, CU_STREAM_NON_BLOCKING), "create transaction stream");
    check(frontend.synchronize_stream(stream, false), "prime caller stream binding");
    std::atomic<bool> entered{};
    std::atomic<bool> released{};
    std::atomic<bool> copied{};
    CUresult native_result = CUDA_ERROR_UNKNOWN;
    CUresult copy_result = CUDA_ERROR_UNKNOWN;
    unsigned int original = 789;
    unsigned int destination = 0;
    auto resources = std::make_shared<int>(0);
    std::thread transaction([&] {
        native_result = frontend.native(
            stream, false,
            [&] {
                entered.store(true);
                while (!released.load()) {
                    std::this_thread::yield();
                }
                return CUDA_SUCCESS;
            },
            true);
    });
    while (!entered.load()) {
        std::this_thread::yield();
    }
    const auto calls = native_calls.load();
    std::thread submitter([&] {
        copy_result = frontend.copy(
            stream, false,
            {CudaQueuedCopyKind::kHostToDevice, reinterpret_cast<CUdeviceptr>(&destination), 0,
             &original, sizeof(original), resources});
        copied.store(true);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds{10});
    require(!copied.load() && native_calls.load() == calls,
            "paused native transaction must exclude metadata issue and private polling");
    released.store(true);
    transaction.join();
    submitter.join();
    check(native_result, "import once-issued caller native execution");
    check(copy_result, "retry unconsumed copy payload after transaction");
    check(frontend.synchronize_stream(stream, false), "await deferred copy");
    require(destination == original, "rejected command must retain its payload");
    unsigned int issued = 0;
    const auto original_caller = std::this_thread::get_id();
    for (unsigned int iteration = 0; iteration < 40; ++iteration) {
        check(frontend.native(
                  stream, false,
                  [&] {
                      require(std::this_thread::get_id() == original_caller &&
                                  current_context == kContext,
                              "combined admission must preserve caller-thread original execution");
                      ++issued;
                      return CUDA_SUCCESS;
                  },
                  true),
              "reuse physically stopped native import slot");
        check(frontend.synchronize_stream(stream, false), "prove reusable trailing native fence");
    }
    require(issued == 40, "bounded import arena must not impose a lifetime launch count limit");
    check(frontend.close_stream(stream), "close transaction stream");
    check(stream_destroy(stream), "destroy transaction stream");
    check(frontend.shutdown(), "release transaction frontend");
    gate->shutdown();
    require(event_count.load() == 0 && stream_count.load() == 0 && allocation_count.load() == 0,
            "transaction resources must release after physical completion");
}

struct NativeWaitClassifier final : CudaFrontendNativeWaitGuard {
    CUevent native_event{};
    CUresult status{CUDA_SUCCESS};
    std::latch* entered{};
    std::atomic<unsigned int> calls{};
    std::thread::id actor{};
    CUresult should_wait(CUevent event, bool& output) noexcept override {
        actor = std::this_thread::get_id();
        ++calls;
        output = event != native_event;
        if (entered) {
            entered->count_down();
            entered = nullptr;
        }
        return status;
    }
};

void native_execution_waits_before_admission() {
    auto gate = control();
    CudaCooperativeFrontend frontend(kContext, dispatch(), gate,
                                     {.progress_slots = 2, .import_events = 12});
    check(frontend.initialize(), "initialize native dependency frontend");
    CUstream low{};
    CUstream high{};
    CUevent event{};
    CUevent unrecorded{};
    CUevent native_event{};
    check(stream_create(&low, CU_STREAM_NON_BLOCKING), "create dependency producer");
    check(stream_create(&high, CU_STREAM_NON_BLOCKING), "create dependent native stream");
    reinterpret_cast<Stream*>(high)->priority = -1;
    check(event_create(&event, 0), "create waited logical event");
    check(event_create(&unrecorded, 0), "create unrecorded logical event");
    check(event_create(&native_event, 0), "create native-only event");
    auto resources = std::make_shared<int>(0);
    check(frontend.bind_event(event, resources), "bind waited event");
    check(frontend.bind_event(unrecorded, resources), "bind empty event snapshot");
    check(frontend.synchronize_stream(low, false), "prime producer native prefix");
    check(frontend.synchronize_stream(high, false), "prime dependent native prefix");
    unsigned int source = 321;
    unsigned int destination{};
    is_ready = false;
    check(frontend.copy(
              low, false,
              {CudaQueuedCopyKind::kHostToDevice, reinterpret_cast<CUdeviceptr>(&destination), 0,
               &source, sizeof(source), resources}),
          "accept pending producer data");
    check(frontend.record_event(event, low, false), "accept pending logical event record");
    std::latch classified{1};
    auto classifier = std::make_shared<NativeWaitClassifier>();
    classifier->native_event = native_event;
    classifier->entered = &classified;
    const std::array waits{event, event, unrecorded, native_event};
    std::atomic<unsigned int> originals{};
    std::thread::id native_caller;
    CUresult result = CUDA_ERROR_UNKNOWN;
    std::jthread caller([&] {
        native_caller = std::this_thread::get_id();
        result = frontend.native(
            high, false,
            [&] {
                ++originals;
                return destination == source ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
            },
            true, {}, waits, classifier);
    });
    struct ReleaseProducer {
        ~ReleaseProducer() {
            is_ready = true;
        }
    } release_producer;
    classified.wait();
    // This later actor command proves snapshot resolution/submission finished.
    // Rerecording on the reserved stream would form a cycle if the native wait
    // were retargeted to the new rather than the immutable previous record.
    check(frontend.record_event(event, high, false), "rerecord after native wait snapshot");
    require(originals == 0 && frontend.stats().granted_executions == 0,
            "a pending logical producer must exclude native grant and original issue");
    is_ready = true;
    caller.join();
    check(result, "issue native operation after exact waited predecessor completes");
    check(frontend.synchronize_event(event), "complete later independent rerecord");
    require(originals == 1 && classifier->calls == 3 && classifier->actor != native_caller,
            "distinct waits classify once on actor; original issues once on caller");

    const auto operation = [&] {
        ++originals;
        return CUDA_SUCCESS;
    };
    const std::array foreign_wait{native_event};
    require(frontend.native(high, false, operation, true, {}, foreign_wait) ==
                CUDA_ERROR_INVALID_HANDLE,
            "a foreign logical event cannot become a cross-queue dependency");
    auto rejected = std::make_shared<NativeWaitClassifier>();
    rejected->status = CUDA_ERROR_INVALID_VALUE;
    const std::array logical_wait{event};
    require(frontend.native(high, false, operation, true, {}, logical_wait, rejected) ==
                    CUDA_ERROR_INVALID_VALUE &&
                rejected->calls == 1,
            "classification failure returns unchanged before reservation");
    const std::array<CUevent, 1> null_wait{nullptr};
    require(
        frontend.native(high, false, operation, true, {}, null_wait) == CUDA_ERROR_INVALID_HANDLE &&
            frontend.native(high, false, operation, false, {}, logical_wait) ==
                CUDA_ERROR_INVALID_VALUE,
        "invalid handles and non-execution boundaries reject waits");
    const std::vector oversized(4097, event);
    require(
        frontend.native(high, false, operation, true, {}, oversized) == CUDA_ERROR_INVALID_VALUE &&
            originals == 1 && frontend.stats().granted_executions == 1,
        "bounded validation errors cannot call original or obtain a grant");
    const std::array empty_wait{unrecorded};
    check(frontend.native(high, false, operation, true, {}, empty_wait),
          "unrecorded logical event adds no predecessor");
    check(frontend.synchronize_stream(high, false), "prove empty-dependency execution");
    require(originals == 2 && frontend.stats().granted_executions == 2 &&
                frontend.stats().retreat_requests == 0,
            "dependencies do not create extra grants or unnecessary retreat");
    check(frontend.close_event(event), "close waited event");
    check(frontend.close_event(unrecorded), "close unrecorded event");
    check(frontend.close_stream(low), "close producer stream");
    check(frontend.close_stream(high), "close dependent native stream");
    check(frontend.shutdown(), "release native dependency resources");
    check(event_destroy(event), "destroy waited original event");
    check(event_destroy(unrecorded), "destroy unrecorded original event");
    check(event_destroy(native_event), "destroy native original event");
    check(stream_destroy(low), "destroy producer original stream");
    check(stream_destroy(high), "destroy dependent original stream");
    gate->shutdown();
    require(event_count == 0 && stream_count == 0 && allocation_count == 0,
            "native dependency validation and issue must not leak resources");
}

void failed_native_wait_never_issues() {
    const auto child = ::fork();
    require(child >= 0, "start failed native dependency child");
    if (child == 0) {
        try {
            // The failed worker deliberately retains uncertain native ownership.
            // Healthy teardown is leak-checked independently in the parent.
            auto gate = control();
            CudaCooperativeFrontend frontend(kContext, dispatch(), gate,
                                             {.progress_slots = 2, .import_events = 8});
            check(frontend.initialize(), "initialize failed native dependency");
            CUstream low{};
            CUstream high{};
            CUevent event{};
            check(stream_create(&low, CU_STREAM_NON_BLOCKING), "create failed producer");
            check(stream_create(&high, CU_STREAM_NON_BLOCKING), "create failed dependent");
            reinterpret_cast<Stream*>(high)->priority = -1;
            check(event_create(&event, 0), "create failed waited event");
            check(frontend.bind_event(event, std::make_shared<int>(0)), "bind failed record");
            check(frontend.synchronize_stream(low, false), "prime failed producer prefix");
            check(frontend.synchronize_stream(high, false), "prime failed dependent prefix");
            is_ready = false;
            unsigned int source = 123;
            unsigned int destination{};
            check(frontend.copy(low, false,
                                {CudaQueuedCopyKind::kHostToDevice,
                                 reinterpret_cast<CUdeviceptr>(&destination), 0, &source,
                                 sizeof(source), std::make_shared<int>(0)}),
                  "queue predecessor before targeted record failure");
            failed_wait_record = event;
            check(frontend.record_event(event, low, false), "accept doomed record");
            std::latch classified{1};
            auto classifier = std::make_shared<NativeWaitClassifier>();
            classifier->entered = &classified;
            const std::array waits{event};
            std::atomic<unsigned int> originals{};
            CUresult result = CUDA_ERROR_UNKNOWN;
            std::jthread caller([&] {
                result = frontend.native(
                    high, false,
                    [&] {
                        ++originals;
                        return CUDA_SUCCESS;
                    },
                    true, {}, waits, classifier);
            });
            struct ReleaseProducer {
                ~ReleaseProducer() {
                    is_ready = true;
                }
            } release_producer;
            classified.wait();
            require(frontend.query_stream(high, false) == CUDA_ERROR_NOT_READY,
                    "reservation must be pending before targeted producer failure");
            is_ready = true;
            caller.join();
            require(result == CUDA_ERROR_LAUNCH_FAILED && originals == 0 &&
                        frontend.stats().granted_executions == 0,
                    "failed producer must wake native caller without issue or control grant");
            ::_exit(EXIT_SUCCESS);
        } catch (...) {
            ::_exit(EXIT_FAILURE);
        }
    }
    int status{};
    pid_t result{};
    do {
        result = ::waitpid(child, &status, 0);
    } while (result < 0 && errno == EINTR);
    require(result == child && WIFEXITED(status) && WEXITSTATUS(status) == EXIT_SUCCESS,
            "native dependency failure must not hang or replay original issue");
}

struct PrefixCompletion final : CudaQueueNativeFenceObserver {
    std::shared_ptr<CudaEventProjectionRegistry> registry;
    std::vector<CudaEventProjection> epochs;
    std::shared_ptr<CudaOperationToken> prefix{std::make_shared<CudaOperationToken>()};
    std::atomic<unsigned int> accepted{};
    std::atomic<unsigned int> completed{};
    CUresult native_reservation_accepted(const CudaOperationToken& token) noexcept override {
        ++accepted;
        if (!token || *prefix) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        *prefix = token;
        return CUDA_SUCCESS;
    }
    CUresult confirmed_native_completion() noexcept override {
        ++completed;
        return registry->complete_native_events(epochs);
    }
};
struct PrefixRecordGuard final : CudaQueueEventRecordGuard {
    std::shared_ptr<CudaEventProjectionRegistry> registry;
    CudaEventProjection owned;
    CudaOperationToken dependency;
    unsigned int prepared{};
    unsigned int accepted{};
    CUresult prepare_record(CudaEventRecordPreparation& output) noexcept override {
        ++prepared;
        CudaEventProjection current;
        const auto result = registry->lookup_owned(owned, current);
        if (result != CUDA_SUCCESS) {
            return result;
        }
        owned = std::move(current);
        output = {};
        if (!owned.is_logical()) {
            const auto& prefix = owned.issued_native_prefix();
            if (!prefix && !owned.has_native_completion()) {
                return CUDA_ERROR_NOT_READY;
            }
            output.native_prefix = prefix ? *prefix : CudaOperationToken{};
            output.should_order_previous = false;
        }
        dependency = output.native_prefix;
        return CUDA_SUCCESS;
    }
    CUresult record_accepted() noexcept override {
        ++accepted;
        return registry->publish_logical(owned, dependency ? &dependency : nullptr);
    }
    CUresult should_write(bool& output) noexcept override {
        return registry->matches_native_epoch(owned, output);
    }
};
void fresh_record_owns_issued_native_prefix() {
    auto gate = control();
    CudaCooperativeFrontend frontend(kContext, dispatch(), gate,
                                     {.progress_slots = 2, .import_events = 8});
    check(frontend.initialize(), "initialize issued-prefix frontend");
    CUstream high{};
    CUstream consumer{};
    CUevent event{};
    check(stream_create(&high, CU_STREAM_NON_BLOCKING), "create native prefix stream");
    check(stream_create(&consumer, CU_STREAM_NON_BLOCKING), "create fresh record stream");
    check(event_create(&event, 0), "create prefix event");
    check(frontend.bind_event(event, std::make_shared<int>(0)), "bind original prefix event");
    check(frontend.synchronize_stream(high, false), "prime high prefix binding");
    check(frontend.synchronize_stream(consumer, false), "prime fresh record binding");
    auto registry = std::make_shared<CudaEventProjectionRegistry>();
    check(registry->register_event(kContext, event, true), "register prefix event epoch");
    CudaEventProjection initial;
    check(registry->lookup(event, initial), "lookup logical prefix event");
    std::array rejected_epochs{initial};
    require(
        registry->invalidate_native_events(
            rejected_epochs, std::make_shared<CudaOperationToken>()) == CUDA_ERROR_INVALID_VALUE &&
            rejected_epochs[0].is_logical(),
        "an empty token is neither issued record nor native completion");
    auto unrelated = std::make_shared<PrefixCompletion>();
    unrelated->registry = registry;
    check(frontend.native(
              high, false, [] { return CUDA_SUCCESS; }, true, unrelated),
          "prepare unrelated completed reservation token");
    check(frontend.synchronize_stream(high, false), "complete unrelated native prefix");
    auto completion = std::make_shared<PrefixCompletion>();
    completion->registry = registry;
    completion->epochs.push_back(initial);
    is_ready = false;
    unsigned int originals{};
    check(frontend.native(
              high, false,
              [&] {
                  ++originals;
                  require(completion->accepted == 1 && *completion->prefix,
                          "exact prefix is accepted before original issue, not guessed later");
                  check(event_record(event, high), "issue original native event record");
                  return registry->invalidate_native_events(completion->epochs, completion->prefix);
              },
              true, completion),
          "publish original-success native event prefix before actor thaw");
    CudaEventProjection native;
    check(registry->lookup(event, native), "inspect issued native epoch");
    CudaOperationToken empty;
    require(!native.is_logical() && native.issued_native_prefix() == completion->prefix &&
                !native.has_native_completion() && completion->completed == 0 &&
                registry->publish_logical(native) == CUDA_ERROR_NOT_READY &&
                registry->publish_logical(native, &empty) == CUDA_ERROR_NOT_READY &&
                registry->publish_logical(native, unrelated->prefix.get()) == CUDA_ERROR_NOT_READY,
            "issue/import and foreign or missing prefix cannot prove record acceptance");
    auto guard = std::make_shared<PrefixRecordGuard>();
    guard->registry = registry;
    guard->owned = native;
    check(frontend.record_event(event, consumer, false, guard),
          "accept fresh record without waiting for native GPU completion");
    CudaEventProjection logical;
    check(registry->lookup(event, logical), "inspect prefix-ordered logical acceptance");
    require(logical.is_logical() && !logical.issued_native_prefix() &&
                frontend.query_event(event) == CUDA_ERROR_NOT_READY && guard->prepared == 1 &&
                guard->accepted == 1 && completion->completed == 0,
            "logical acceptance retires native routing but cannot complete a pending prefix");
    is_ready = true;
    check(frontend.synchronize_event(event), "write fresh event after exact native prefix proof");
    require(originals == 1 && completion->accepted == 1 && completion->completed == 1,
            "native execution and its exact completion cannot replay");
    check(frontend.close_event(event), "close prefix event");
    check(frontend.close_stream(high), "close native prefix stream");
    check(frontend.close_stream(consumer), "close fresh record stream");
    check(frontend.shutdown(), "release prefix-owned frontend");
    check(event_destroy(event), "destroy original prefix event");
    check(stream_destroy(high), "destroy original native stream");
    check(stream_destroy(consumer), "destroy original fresh record stream");
    gate->shutdown();
    require(event_count == 0 && stream_count == 0 && allocation_count == 0,
            "issued prefix metadata owns no leaking native resources");
}
class AcceptanceGuard final : public CudaQueueEventRecordGuard {
   public:
    std::shared_ptr<CudaEventProjectionRegistry> registry;
    CudaEventProjection projection;
    std::latch* entered{};
    std::latch* release{};
    std::atomic<unsigned int> calls{};
    std::thread::id actor;
    CUresult publication{CUDA_SUCCESS};
    bool should_suppress{};
    CUresult record_accepted() noexcept override {
        ++calls;
        actor = std::this_thread::get_id();
        if (entered) {
            entered->count_down();
            // Test-only finite model of a CPU publication in flight.
            release->wait();
        }
        return publication == CUDA_SUCCESS ? registry->publish_logical(projection) : publication;
    }
    CUresult should_write(bool& output) noexcept override {
        if (should_suppress) {
            output = false;
            return CUDA_SUCCESS;
        }
        return registry->matches_native_epoch(projection, output);
    }
};
void event_publication_is_part_of_acceptance() {
    current_context = kContext;
    is_ready.store(true);
    auto gate = control();
    CudaCooperativeFrontend frontend(
        kContext, dispatch(), gate,
        {.progress_slots = 4, .import_events = 12, .per_thread_streams = 2});
    check(frontend.initialize(), "initialize event acceptance frontend");
    CUstream low{};
    CUstream high{};
    CUevent event{};
    check(stream_create(&low, CU_STREAM_NON_BLOCKING), "create event acceptance low stream");
    check(stream_create(&high, CU_STREAM_NON_BLOCKING), "create native acceptance high stream");
    std::bit_cast<Stream*>(high)->priority = -1;
    check(event_create(&event, 0), "create event acceptance original handle");
    auto resources = std::make_shared<int>(0);
    check(frontend.bind_event(event, resources), "bind event acceptance owner");
    check(frontend.synchronize_stream(low, false), "prime event acceptance low binding");
    check(frontend.synchronize_stream(high, false), "prime event acceptance native binding");
    auto registry = std::make_shared<CudaEventProjectionRegistry>();
    check(registry->register_event(kContext, event, false),
          "register native event acceptance epoch");
    check(registry->complete_native(kContext), "prove epoch before fresh logical acceptance");
    auto guard = std::make_shared<AcceptanceGuard>();
    guard->registry = registry;
    check(registry->lookup(event, guard->projection), "prepare exact acceptance guard");
    std::latch accepted{1};
    std::latch release{1};
    std::latch native_started{1};
    guard->entered = &accepted;
    guard->release = &release;
    std::jthread recording;
    std::jthread native_source;
    struct Release {
        std::latch& latch;
        ~Release() {
            if (!latch.try_wait()) {
                latch.count_down();
            }
        }
    } release_on_error{release};
    CUresult recorded{CUDA_ERROR_UNKNOWN};
    CUresult native_result{CUDA_ERROR_UNKNOWN};
    std::atomic<bool> native_was_invoked{};
    recording = std::jthread([&] {
        current_context = kContext;
        recorded = frontend.record_event(event, low, false, guard);
    });
    accepted.wait();
    native_source = std::jthread([&] {
        current_context = kContext;
        native_context_probe = &native_started;
        native_result = frontend.native(
            high, false,
            [&] {
                native_was_invoked.store(true);
                CudaEventProjection logical;
                const auto found = registry->lookup(event, logical);
                if (found != CUDA_SUCCESS || !logical.is_logical()) {
                    return CUDA_ERROR_UNKNOWN;
                }
                const auto invalidated = registry->invalidate_native_event(logical);
                return invalidated == CUDA_SUCCESS ? event_record(event, high) : invalidated;
            },
            true);
    });
    native_started.wait();
    require(!native_was_invoked.load(),
            "native pause cannot cross an actor's in-flight acceptance publication");
    release.count_down();
    const auto recording_caller = recording.get_id();
    recording.join();
    native_source.join();
    check(recorded, "record publishes logical ownership before returning");
    check(native_result, "native owner supersedes only the fully accepted record");
    check(frontend.synchronize_context(), "prove native acceptance prefix");
    CudaEventProjection native;
    check(registry->lookup(event, native), "inspect native supersession after publication");
    require(guard->calls == 1 && guard->actor != recording_caller && !native.is_logical() &&
                native_was_invoked.load(),
            "publication runs once on actor and cannot undo a later native epoch");

    auto rejected = std::make_shared<AcceptanceGuard>();
    rejected->registry = registry;
    rejected->projection = native;
    require(frontend.record_event(nullptr, low, false, rejected) == CUDA_ERROR_INVALID_HANDLE &&
                rejected->calls == 0,
            "rejected record cannot invoke publication");
    rejected->publication = CUDA_ERROR_INVALID_VALUE;
    rejected->should_suppress = true;
    require(frontend.record_event(event, low, false, rejected) == CUDA_ERROR_INVALID_VALUE &&
                rejected->calls == 1,
            "accepted publication error is preserved without repeating source acceptance");
    check(frontend.synchronize_event(event), "retain failed-publication record through completion");
    require(rejected->calls == 1, "physical progress cannot repeat CPU acceptance publication");
    check(frontend.close_event(event), "close event acceptance handle after users drain");
    check(frontend.close_stream(low), "close event acceptance low stream");
    check(frontend.close_stream(high), "close event acceptance high stream");
    check(frontend.shutdown(), "shutdown event acceptance owners");
    check(event_destroy(event), "destroy original event acceptance handle");
    check(stream_destroy(low), "destroy event acceptance low stream");
    check(stream_destroy(high), "destroy event acceptance high stream");
}

void caller_record_contribution_precedes_worker_pause() {
    current_context = kContext;
    is_ready.store(true);
    auto gate = control();
    CudaCooperativeFrontend frontend(kContext, dispatch(), gate,
                                     {.progress_slots = 2, .import_events = 8});
    check(frontend.initialize(), "initialize caller record frontend");
    CUstream stream{};
    CUevent event{};
    check(stream_create(&stream, CU_STREAM_NON_BLOCKING), "create caller record stream");
    check(event_create(&event, 0), "create caller record event");
    check(frontend.bind_event(event, std::make_shared<int>(0)), "bind caller record event");
    check(frontend.synchronize_stream(stream, false), "prime caller record stream");
    auto registry = std::make_shared<CudaEventProjectionRegistry>();
    check(registry->register_event(kContext, event, true), "register caller record projection");
    auto guard = std::make_shared<AcceptanceGuard>();
    guard->registry = registry;
    check(registry->lookup(event, guard->projection), "prepare caller record guard");
    require(frontend.admit_event_metadata(event, {}) == CUDA_ERROR_INVALID_VALUE,
            "missing caller record callback accepts nothing");
    unsigned int failures{};
    require(frontend.admit_event_metadata(event,
                                          [&] {
                                              ++failures;
                                              return CUDA_ERROR_INVALID_VALUE;
                                          }) == CUDA_ERROR_INVALID_VALUE &&
                failures == 1 && guard->calls == 0,
            "failed caller validation returns once without record acceptance");
    CudaNativeBoundaryToken rejected_pause;
    require(frontend.admit_event_metadata(event, [&] { return frontend.pause(rejected_pause); }) ==
                    CUDA_ERROR_INVALID_VALUE &&
                !rejected_pause,
            "caller contribution cannot wait for its own native pause");
    std::latch entered{1};
    std::latch release{1};
    std::latch native_started{1};
    std::jthread recording;
    std::jthread native_source;
    struct Release {
        std::latch& latch;
        ~Release() {
            if (!latch.try_wait()) {
                latch.count_down();
            }
        }
    } release_on_error{release};
    CUresult recorded{CUDA_ERROR_UNKNOWN};
    CUresult native_result{CUDA_ERROR_UNKNOWN};
    std::atomic<unsigned int> validations{};
    std::atomic<bool> native_was_invoked{};
    std::thread::id validation_caller;
    recording = std::jthread([&] {
        current_context = kContext;
        recorded = frontend.admit_event_metadata(event, [&] {
            ++validations;
            validation_caller = std::this_thread::get_id();
            entered.count_down();
            release.wait();
            return frontend.record_event(event, stream, false, guard);
        });
    });
    entered.wait();
    native_source = std::jthread([&] {
        current_context = kContext;
        native_context_probe = &native_started;
        native_result = frontend.native(
            stream, false,
            [&] {
                native_was_invoked.store(true);
                return guard->calls == 1 ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
            },
            false);
    });
    native_started.wait();
    require(!native_was_invoked.load(), "native issue cannot cross admitted caller validation");
    release.count_down();
    const auto original_caller = recording.get_id();
    recording.join();
    native_source.join();
    check(recorded, "caller publishes record before leaving its contribution");
    check(native_result, "native worker pause follows complete caller acceptance");
    require(validations == 1 && validation_caller == original_caller && guard->calls == 1,
            "SDK callback stays on caller and publication executes exactly once");
    check(frontend.close_event(event), "close admitted caller event");
    check(frontend.close_stream(stream), "close admitted caller stream");
    check(frontend.shutdown(), "release admitted caller resources");
    check(event_destroy(event), "destroy admitted caller original event");
    check(stream_destroy(stream), "destroy admitted caller original stream");
    gate->shutdown();
    require(event_count == 0 && stream_count == 0 && allocation_count == 0,
            "caller contribution owns no leaking native resources");
}

void immutable_wait_snapshots_and_serialized_callers() {
    current_context = kContext;
    is_ready.store(true);
    auto gate = control();
    CudaCooperativeFrontend frontend(kContext, dispatch(), gate,
                                     {.progress_slots = 2, .import_events = 8});
    CudaCooperativeFrontend foreign(kContext, dispatch(), gate,
                                    {.progress_slots = 2, .import_events = 2});
    check(frontend.initialize(), "initialize historical snapshot frontend");
    check(foreign.initialize(), "initialize distinct snapshot owner");
    CUstream producer{}, history_consumer{}, empty_consumer{};
    for (auto* output : {&producer, &history_consumer, &empty_consumer}) {
        check(stream_create(output, CU_STREAM_NON_BLOCKING), "create snapshot stream");
        check(frontend.synchronize_stream(*output, false), "prime snapshot stream prefix");
    }
    CUevent event{}, unrecorded{};
    for (auto* output : {&event, &unrecorded}) {
        check(event_create(output, 0), "create snapshot event");
        check(frontend.bind_event(*output, std::make_shared<int>(0)), "bind exact snapshot event");
    }
    check(frontend.record_event(event, producer, false), "record original historical point");
    check(frontend.synchronize_event(event), "prove original historical point complete");
    CudaFrontendEventWait history, empty;
    check(frontend.prepare_event_wait(event, history), "capture immutable completed point");
    check(frontend.prepare_event_wait(unrecorded, empty), "capture valid empty event point");
    require(
        history && empty && frontend.prepare_event_wait(event, history) == CUDA_ERROR_INVALID_VALUE,
        "a prepared empty prefix is valid; occupied output cannot be replaced");
    // Every private command checks this caller's context first. Worker polling
    // is concurrent and must not be mistaken for CUDA issue by this rejected call.
    const auto before_foreign = caller_context_queries;
    require(
        foreign.wait_event_snapshot(history_consumer, false, history) == CUDA_ERROR_INVALID_VALUE &&
            frontend.wait_event_snapshot(history_consumer, false, {}) == CUDA_ERROR_INVALID_VALUE &&
            caller_context_queries == before_foreign,
        "foreign or missing snapshot rejects before private CUDA");
    struct Release {
        ~Release() {
            observed_record_stream.store(nullptr);
            is_ready.store(true);
        }
    } release;
    is_ready.store(false);
    check(frontend.native(
              producer, false, [] { return CUDA_SUCCESS; }, true),
          "import independently pending native producer");
    check(frontend.record_event(event, producer, false), "rerecord historical event while pending");
    check(frontend.record_event(unrecorded, producer, false), "record formerly empty event");
    const auto expect_marker = [&](CUstream consumer, const CudaFrontendEventWait& snapshot) {
        observed_stream_records.store(0);
        observed_record_stream.store(consumer);
        check(frontend.wait_event_snapshot(consumer, false, snapshot),
              "accept exact historical dependency without rereading latest event");
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (observed_stream_records.load() == 0) {
            require(std::chrono::steady_clock::now() < deadline,
                    "historical marker must issue while newer event prefix remains pending");
            std::this_thread::yield();
        }
        require(frontend.query_event(event) == CUDA_ERROR_NOT_READY &&
                    frontend.query_event(unrecorded) == CUDA_ERROR_NOT_READY,
                "historical marker issue cannot finish either newer record");
        observed_record_stream.store(nullptr);
    };
    expect_marker(history_consumer, history);
    expect_marker(empty_consumer, empty);
    is_ready.store(true);
    check(frontend.synchronize_context(), "prove every historical and new prefix before close");
    check(frontend.close_event(event), "retire exact snapshot registration");
    check(frontend.bind_event(event, std::make_shared<int>(0)),
          "rebind same native handle identity");
    require(
        frontend.wait_event_snapshot(history_consumer, false, history) == CUDA_ERROR_INVALID_HANDLE,
        "retired snapshot cannot name the same handle's new binding");
    unsigned int recursive_calls{};
    require(frontend.admit_event_metadata(event,
                                          [&] {
                                              return frontend.admit_event_metadata(event, [&] {
                                                  ++recursive_calls;
                                                  return CUDA_SUCCESS;
                                              });
                                          }) == CUDA_ERROR_INVALID_VALUE &&
                recursive_calls == 0,
            "recursive same-event validation rejects without invoking another SDK call");
    std::barrier start{3};
    std::atomic<unsigned int> active{}, calls{}, overlaps{};
    std::atomic<CUresult> result{CUDA_SUCCESS};
    const auto validate = [&] {
        if (active.fetch_add(1) != 0) {
            ++overlaps;
        }
        ++calls;
        std::this_thread::yield();
        --active;
        return CUDA_SUCCESS;
    };
    const auto caller = [&] {
        current_context = kContext;
        start.arrive_and_wait();
        for (unsigned int iteration = 0; iteration < 100; ++iteration) {
            const auto status = frontend.admit_event_metadata(event, validate);
            if (status != CUDA_SUCCESS) {
                result.store(status);
                break;
            }
        }
    };
    std::jthread first(caller), second(caller);
    start.arrive_and_wait();
    first.join();
    second.join();
    check(result.load(), "serialize competing event SDK callers");
    require(calls == 200 && active == 0 && overlaps == 0,
            "same registration's SDK validation callbacks never overlap");
    check(frontend.close_event(event), "close reused snapshot event");
    check(frontend.close_event(unrecorded), "close empty snapshot event");
    for (auto stream : {producer, history_consumer, empty_consumer}) {
        check(frontend.close_stream(stream), "close physically proven snapshot stream");
    }
    check(frontend.shutdown(), "release historical snapshot frontend");
    check(foreign.shutdown(), "release distinct CPU snapshot owner");
    for (auto value : {event, unrecorded}) {
        check(event_destroy(value), "destroy original snapshot event");
    }
    for (auto stream : {producer, history_consumer, empty_consumer}) {
        check(stream_destroy(stream), "destroy original snapshot stream");
    }
    gate->shutdown();
    require(event_count == 0 && stream_count == 0 && allocation_count == 0,
            "immutable snapshots retain no native resources after proven cleanup");
}

void completion_looks_up_exact_existing_streams() {
    current_context = kContext;
    is_ready.store(true);
    auto gate = control();
    CudaCooperativeFrontend frontend(kContext, dispatch(), gate,
                                     {.progress_slots = 2, .import_events = 6});
    struct Release {
        ~Release() {
            is_ready.store(true);
        }
    } release;
    check(frontend.initialize(), "initialize exact stream completion owner");
    CUstream producer{}, unknown{};
    CUevent event{};
    check(stream_create(&producer, CU_STREAM_NON_BLOCKING), "create known completion stream");
    check(stream_create(&unknown, CU_STREAM_NON_BLOCKING), "create unbound native stream");
    const auto original_events = event_count.load();
    const auto original_streams = stream_count.load();
    bool found{true};
    check(frontend.complete_stream_snapshot(unknown, false, false, found),
          "unknown stream completion remains CPU-only");
    require(!found && event_count == original_events && stream_count == original_streams,
            "unknown query imports no fence, stream or queue binding");
    check(frontend.synchronize_stream(producer, false), "bind known stream once");
    check(frontend.complete_stream_snapshot(producer, false, false, found),
          "known drained stream is complete");
    require(found, "known completion preserves exact binding");
    is_ready.store(false);
    check(frontend.native(
              producer, false, [] { return CUDA_SUCCESS; }, true),
          "import pending exact producer prefix");
    require(
        frontend.complete_stream_snapshot(producer, false, false, found) == CUDA_ERROR_NOT_READY &&
            found,
        "known query cannot complete a held physical prefix");
    check(event_create(&event, 0), "create PTDS completion dependency");
    check(frontend.bind_event(event, std::make_shared<int>(0)), "bind completion dependency");
    check(frontend.record_event(event, producer, false), "record pending completion dependency");
    check(frontend.wait_event(nullptr, true, event), "accept exact caller PTDS dependency");
    require(frontend.complete_stream_snapshot(CU_STREAM_PER_THREAD, false, false, found) ==
                    CUDA_ERROR_NOT_READY &&
                found,
            "literal PTDS identity matches caller's null PTDS binding");
    require(
        frontend.complete_stream_snapshot(nullptr, false, false, found) == CUDA_ERROR_NOT_READY &&
            found,
        "unbound legacy completion observes blocking/PTDS tails without importing a fence");
    std::atomic<CUresult> other_status{CUDA_ERROR_UNKNOWN};
    bool other_found{true};
    std::jthread other([&] {
        current_context = kContext;
        other_status.store(frontend.complete_stream_snapshot(nullptr, true, false, other_found));
    });
    other.join();
    require(other_status == CUDA_SUCCESS && !other_found,
            "another caller's PTDS query cannot see this caller's binding");
    std::latch entered{1};
    std::atomic<bool> finished{};
    std::atomic<CUresult> synchronized{CUDA_ERROR_UNKNOWN};
    bool synchronized_binding{};
    std::jthread waiter([&] {
        current_context = kContext;
        entered.count_down();
        synchronized.store(
            frontend.complete_stream_snapshot(producer, false, true, synchronized_binding));
        finished.store(true);
    });
    entered.wait();
    require(!finished, "bound synchronization cannot finish while exact prefix is pending");
    is_ready.store(true);
    waiter.join();
    require(synchronized == CUDA_SUCCESS && synchronized_binding,
            "bound synchronization proves its accepted prefix after physical completion");
    check(frontend.synchronize_context(), "prove all completion fixture resources");
    check(frontend.shutdown(), "close exact completion owner after proof");
    check(event_destroy(event), "destroy proven completion dependency");
    check(stream_destroy(producer), "destroy proven completion stream");
    check(stream_destroy(unknown), "destroy unused original stream");
    gate->shutdown();
    require(event_count == 0 && stream_count == 0 && allocation_count == 0,
            "completion lookup owns no extra native resources");
}

void process_pause_requires_actual_native_completion() {
    auto gate = control();
    CudaCooperativeFrontend frontend(
        kContext, dispatch(), gate,
        {.progress_slots = 2, .import_events = 2, .per_thread_streams = 0});
    check(frontend.initialize(), "initialize process pause");
    CUevent retired_event{};
    check(event_create(&retired_event, CU_EVENT_DISABLE_TIMING), "create paused retirement event");
    check(frontend.bind_event(retired_event, std::make_shared<int>(0)),
          "bind paused retirement event");
    CUstream retired_stream{};
    check(stream_create(&retired_stream, CU_STREAM_NON_BLOCKING),
          "create paused original retirement stream");
    check(frontend.synchronize_stream(retired_stream, false),
          "bind and physically prove original retirement stream");
    CudaNativeBoundaryToken boundary;
    check(frontend.pause(boundary), "pause context owner");
    const auto before = native_calls.load();
    unsigned int proof_calls = 0;
    CudaNativeBoundaryToken invalid;
    require(frontend.retire_paused_event(invalid, retired_event, true) == CUDA_ERROR_INVALID_VALUE,
            "foreign pause cannot retire a logical binding");
    check(frontend.retire_paused_event(boundary, retired_event, false),
          "prepare source retirement without closing binding");
    check(frontend.retire_paused_event(boundary, retired_event, false),
          "rejected original destruction leaves the binding available");
    should_reject_retirement_allocation = true;
    const auto allocation_failure = frontend.retire_paused_event(boundary, retired_event, true);
    should_reject_retirement_allocation = false;
    require(allocation_failure == CUDA_ERROR_OUT_OF_MEMORY && native_calls.load() == before &&
                frontend.native_boundary_is_paused(boundary),
            "mailbox allocation failure preserves pause and calls no private CUDA");
    check(frontend.retire_paused_event(boundary, retired_event, false),
          "unissued failed retirement leaves exact logical binding intact");
    check(frontend.retire_paused_event(boundary, retired_event, true),
          "retire binding without thaw or private GPU calls");
    require(
        frontend.retire_paused_stream(invalid, retired_stream, true) == CUDA_ERROR_INVALID_VALUE &&
            frontend.retire_paused_stream(boundary, nullptr, true) == CUDA_ERROR_INVALID_VALUE,
        "wrong pause or implicit stream cannot authorize source retirement");
    check(frontend.retire_paused_stream(boundary, retired_stream, false),
          "prepare exact paused stream retirement");
    should_reject_retirement_allocation = true;
    const auto stream_allocation_failure =
        frontend.retire_paused_stream(boundary, retired_stream, true);
    should_reject_retirement_allocation = false;
    require(stream_allocation_failure == CUDA_ERROR_OUT_OF_MEMORY &&
                native_calls.load() == before && frontend.native_boundary_is_paused(boundary),
            "failed stream retirement command leaves pause and makes no CUDA call");
    check(frontend.retire_paused_stream(boundary, retired_stream, false),
          "failed original stream destruction preserves association");
    check(frontend.retire_paused_stream(boundary, retired_stream, true),
          "commit CPU-only paused stream retirement");
    require(frontend.retire_paused_stream(boundary, retired_stream, true) ==
                    CUDA_ERROR_INVALID_HANDLE &&
                native_calls.load() == before,
            "stream retirement is exact once and performs no native cleanup");
    require(
        frontend.retire_paused_event(boundary, retired_event, true) == CUDA_ERROR_INVALID_HANDLE &&
            native_calls.load() == before && frontend.native_boundary_is_paused(boundary),
        "retirement is exact-once CPU metadata under the same physical pause");
    require(frontend.finish_pause(invalid,
                                  [&] {
                                      ++proof_calls;
                                      return CUDA_SUCCESS;
                                  }) == CUDA_ERROR_INVALID_VALUE &&
                proof_calls == 0 && native_calls.load() == before,
            "a stale or foreign boundary must not invoke native completion");
    std::atomic<bool> started{};
    std::atomic<bool> completed{};
    std::atomic<CUresult> blocked_result{CUDA_ERROR_UNKNOWN};
    std::thread blocked([&] {
        started = true;
        blocked_result = frontend.synchronize_context();
        completed = true;
    });
    while (!started.load()) {
        std::this_thread::yield();
    }
    const bool was_excluded = !completed.load() && native_calls.load() == before;
    check(frontend.finish_pause(boundary,
                                [&] {
                                    ++proof_calls;
                                    return stream_sync(CU_STREAM_LEGACY);
                                }),
          "import actual native context completion");
    blocked.join();
    require(was_excluded && blocked_result == CUDA_SUCCESS && proof_calls == 1 && completed &&
                frontend.finish_pause(boundary,
                                      [&] {
                                          ++proof_calls;
                                          return CUDA_SUCCESS;
                                      }) == CUDA_ERROR_INVALID_VALUE &&
                proof_calls == 1,
            "exact boundary generation can thaw once without replaying native work");
    require(frontend.retire_paused_event(boundary, retired_event, true) == CUDA_ERROR_INVALID_VALUE,
            "finished generation cannot retire a later event");
    require(
        frontend.retire_paused_stream(boundary, retired_stream, true) == CUDA_ERROR_INVALID_VALUE,
        "finished generation cannot retire a later stream");
    check(event_destroy(retired_event), "destroy simulated retired original event");
    check(stream_destroy(retired_stream), "destroy simulated retired original stream");
    check(frontend.shutdown(), "release process pause resources");
    gate->shutdown();
    require(event_count == 0 && stream_count == 0 && allocation_count == 0,
            "preallocated process fence must release on checked teardown");
}

void uncertain_cleanup_never_replays(unsigned int mode) {
    const auto child = ::fork();
    require(child >= 0, "uncertain ownership child must start");
    if (child == 0) {
        try {
            // These tests intentionally retain an uncertain owner until process
            // exit. The parent exercises ordinary teardown under LeakSanitizer.
            auto gate = control();
            CudaCooperativeFrontend frontend(
                kContext, dispatch(), gate,
                {.progress_slots = 2, .import_events = 2, .per_thread_streams = 0});
            if (mode == 0 || mode == 3) {
                should_fail_create = true;
                should_create_null = mode == 3;
                require(frontend.initialize() ==
                            (mode == 3 ? CUDA_ERROR_UNKNOWN : CUDA_ERROR_INVALID_VALUE),
                        "ambiguous native creation must reject initialization");
                const auto calls = destroy_calls.load();
                require(frontend.shutdown() == CUDA_ERROR_UNKNOWN && destroy_calls.load() == calls,
                        "uncertain created handle must not be automatically destroyed");
            } else {
                check(frontend.initialize(), "initialize uncertain ownership child");
                if (mode == 1) {
                    should_fail_destroy = true;
                    require(frontend.shutdown() == CUDA_ERROR_INVALID_VALUE,
                            "actual release followed by native error must be retained");
                    const auto calls = destroy_calls.load();
                    require(
                        frontend.shutdown() == CUDA_ERROR_UNKNOWN && destroy_calls.load() == calls,
                        "failed native destroy must never replay on retry");
                } else {
                    check(frontend.synchronize_stream(nullptr, false),
                          "prime native transaction binding");
                    should_fail_record = true;
                    unsigned int issued = 0;
                    require(frontend.native(
                                nullptr, false,
                                [&] {
                                    ++issued;
                                    return CUDA_SUCCESS;
                                },
                                true) == CUDA_ERROR_INVALID_VALUE &&
                                issued == 1,
                            "post-issue native record error must not replay the invocation");
                    should_fail_record = false;
                    require(frontend.shutdown() == CUDA_ERROR_INVALID_VALUE && issued == 1,
                            "failed native fence must retain its worker pause and execution");
                }
            }
            ::_exit(EXIT_SUCCESS);
        } catch (...) {
            ::_exit(EXIT_FAILURE);
        }
    }
    int status = 0;
    pid_t result = -1;
    do {
        result = ::waitpid(child, &status, 0);
    } while (result < 0 && errno == EINTR);
    require(result == child && WIFEXITED(status) && WEXITSTATUS(status) == EXIT_SUCCESS,
            "uncertain native ownership must fail closed without cleanup replay");
}
void invalid_completion_interval_has_no_native_effect() {
    require(CudaFrontendOptions{}.worker.poll_interval == std::chrono::microseconds{25},
            "normal frontend selects a positive submillisecond physical interval");
    auto gate = control();
    for (const auto interval : {std::chrono::microseconds{0}, std::chrono::microseconds{-1}}) {
        CudaFrontendOptions options;
        options.worker.poll_interval = interval;
        const auto before = native_calls.load();
        CudaCooperativeFrontend frontend(kContext, dispatch(), gate, options);
        require(frontend.initialize() == CUDA_ERROR_INVALID_VALUE && native_calls == before &&
                    frontend.shutdown() == CUDA_SUCCESS && event_count == 0 && stream_count == 0 &&
                    allocation_count == 0,
                "invalid completion interval rejects before context proof or native resources");
    }
    gate->shutdown();
}

std::uint64_t timing_field(std::string_view line, std::string_view key) {
    const auto begin = line.find(key);
    require(begin != std::string_view::npos, "native timing must include every named field");
    const auto value = line.substr(begin + key.size());
    std::uint64_t result{};
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    require(parsed.ec == std::errc{} && parsed.ptr != value.data() &&
                (parsed.ptr == value.data() + value.size() || *parsed.ptr == ' '),
            "native timing fields must be unsigned decimal metadata");
    return result;
}

class DiagnosticDescriptor final {
   public:
    explicit DiagnosticDescriptor(int descriptor) noexcept : descriptor_(descriptor) {}
    ~DiagnosticDescriptor() {
        if (close() != 0) {
            std::_Exit(EXIT_FAILURE);
        }
    }
    DiagnosticDescriptor(const DiagnosticDescriptor&) = delete;
    DiagnosticDescriptor& operator=(const DiagnosticDescriptor&) = delete;
    [[nodiscard]] int get() const noexcept {
        return descriptor_;
    }
    [[nodiscard]] int close() noexcept {
        const auto descriptor = std::exchange(descriptor_, -1);
        return descriptor < 0 ? 0 : ::close(descriptor);
    }

   private:
    int descriptor_;
};

void native_phase_diagnostics_are_opt_in() {
    int descriptors[2]{};
    require(::pipe(descriptors) == 0, "create isolated diagnostic pipe");
    DiagnosticDescriptor input(descriptors[0]);
    DiagnosticDescriptor output_pipe(descriptors[1]);
    const auto child = ::fork();
    if (child == 0) {
        if (input.close() != 0 || ::dup2(output_pipe.get(), STDERR_FILENO) != STDERR_FILENO ||
            output_pipe.close() != 0) {
            std::_Exit(EXIT_FAILURE);
        }
        try {
            NativeTimingObservation maximum;
            maximum.task_id = std::numeric_limits<std::uint64_t>::max();
            maximum.generation = maximum.task_id;
            maximum.started_nanoseconds = maximum.task_id;
            maximum.ended_nanoseconds = maximum.task_id;
            maximum.admission_retries = maximum.task_id;
            maximum.physical_retries = maximum.task_id;
            maximum.original_status = std::numeric_limits<std::uint32_t>::max();
            maximum.phase_nanoseconds.fill(maximum.task_id);
            maximum.admission_nanoseconds.fill(maximum.task_id);
            errno = EDOM;
            report_native_timing_diagnostic(maximum);
            require(errno == EDOM, "bounded diagnostic reporting preserves caller errno");
            const CooperativeControlTimingObservation maximum_control{
                .task_id = maximum.task_id,
                .generation = maximum.task_id,
                .is_submission = true,
                .claim_status = std::numeric_limits<std::uint32_t>::max(),
                .enqueued_nanoseconds = maximum.task_id,
                .started_nanoseconds = maximum.task_id,
                .finished_nanoseconds = maximum.task_id,
                .observed_nanoseconds = maximum.task_id};
            report_cooperative_control_timing_diagnostic(maximum_control);
            require(errno == EDOM, "control timing reporting preserves caller errno");
            for (const bool enabled : {false, true}) {
                current_context = kContext;
                auto gate = control();
                CudaFrontendOptions options;
                options.progress_slots = 2;
                options.import_events = 12;
                options.should_trace_native_timings = enabled;
                CudaCooperativeFrontend frontend(kContext, dispatch(), gate, options);
                check(frontend.initialize(), "initialize diagnostic frontend");
                CUstream stream{};
                check(stream_create(&stream, CU_STREAM_NON_BLOCKING), "create diagnostic stream");
                unsigned int originals{};
                for (const bool execution : {true, false}) {
                    check(frontend.native(
                              stream, false,
                              [&] {
                                  ++originals;
                                  return CUDA_SUCCESS;
                              },
                              execution),
                          "original issue with opt-in diagnostic");
                }
                require(frontend.native(stream, false, {}, true) == CUDA_ERROR_INVALID_VALUE &&
                            originals == 2,
                        "invalid issue produces no completed sample or original replay");
                check(frontend.synchronize_stream(stream, false), "complete diagnostic fences");
                check(frontend.close_stream(stream), "close diagnostic stream binding");
                check(stream_destroy(stream), "destroy original diagnostic stream");
                check(frontend.shutdown(), "release diagnostic private resources");
                gate->shutdown();
                require(event_count == 0 && stream_count == 0 && allocation_count == 0,
                        "enabled and disabled diagnostics preserve proven cleanup");
            }
            std::_Exit(EXIT_SUCCESS);
        } catch (...) {
            std::_Exit(EXIT_FAILURE);
        }
    }
    const auto write_closed = output_pipe.close();
    if (child < 0) {
        const auto read_closed = input.close();
        require(write_closed == 0 && read_closed == 0, "close failed diagnostic fork descriptors");
        require(false, "fork isolated diagnostic fixture");
    }
    std::string output;
    std::array<char, 1024> buffer{};
    ssize_t bytes{};
    do {
        bytes = ::read(input.get(), buffer.data(), buffer.size());
        if (bytes > 0) {
            output.append(buffer.data(), static_cast<std::size_t>(bytes));
        }
    } while (bytes > 0 || (bytes < 0 && errno == EINTR));
    const auto read_closed = input.close();
    int status{};
    pid_t waited{};
    do {
        waited = ::waitpid(child, &status, 0);
    } while (waited < 0 && errno == EINTR);
    require(write_closed == 0 && read_closed == 0 && bytes == 0 && waited == child &&
                WIFEXITED(status) && WEXITSTATUS(status) == EXIT_SUCCESS,
            "diagnostic fixture completes originals and closes every descriptor");
    constexpr std::array phases{" prepare_ns=", " reservation_ns=",     " admission_ns=",
                                " pause_ns=",   " heartbeat_start_ns=", " original_ns=",
                                " fence_ns=",   " heartbeat_stop_ns=",  " commit_ns="};
    std::size_t samples{};
    std::size_t executions{};
    std::size_t control_samples{};
    std::uint64_t control_task{};
    std::uint64_t control_generation{};
    std::size_t begin{};
    while (begin < output.size()) {
        const auto end = output.find('\n', begin);
        require(end != std::string::npos && end - begin < 1024,
                "every diagnostic fits a terminated fixed buffer");
        const std::string_view line(output.data() + begin, end - begin);
        if (line.starts_with("[glimmer] cooperative control timing ")) {
            require(end - begin < 512, "control diagnostic fits its fixed buffer");
            const auto task = timing_field(line, " task=");
            const auto generation = timing_field(line, " generation=");
            const auto enqueued = timing_field(line, " enqueued_ns=");
            const auto started = timing_field(line, " started_ns=");
            const auto finished = timing_field(line, " finished_ns=");
            const auto observed = timing_field(line, " observed_ns=");
            if (control_samples == 0) {
                require(
                    task == std::numeric_limits<std::uint64_t>::max() && generation == task &&
                        enqueued == task && started == task && finished == task &&
                        observed == task &&
                        timing_field(line, " status=") == std::numeric_limits<std::uint32_t>::max(),
                    "maximum-width control metadata is not truncated");
            } else {
                require(task != 0 && generation != 0 && enqueued > 0 && enqueued <= started &&
                            started <= finished && finished <= observed &&
                            timing_field(line, " submit=") == 1 &&
                            timing_field(line, " status=") == 1,
                        "observed real grant preserves identity and timestamp ordering");
                control_task = task;
                control_generation = generation;
            }
            ++control_samples;
            begin = end + 1;
            continue;
        }
        require(line.starts_with("[glimmer] native timing "), "only requested diagnostics appear");
        require(timing_field(line, " thread=") == static_cast<std::uint64_t>(child),
                "native timing must identify the reporting caller, not a background worker");
        const auto task = timing_field(line, " task=");
        const auto generation = timing_field(line, " generation=");
        if (samples == 0) {
            require(task == std::numeric_limits<std::uint64_t>::max() &&
                        timing_field(line, " total_ns=") == task &&
                        timing_field(line, " original_status=") ==
                            std::numeric_limits<std::uint32_t>::max() &&
                        timing_field(line, " admission_owner_probe_ns=") == task &&
                        timing_field(line, " admission_signal_wait_ns=") == task &&
                        timing_field(line, " admission_pause_probe_ns=") == task &&
                        timing_field(line, " admission_other_ns=") == 0,
                    "maximum metadata fits and total duration saturates without wrapping");
        } else {
            const bool execution = timing_field(line, " execution=") == 1;
            executions += execution ? 1 : 0;
            require(execution ? (task != 0 && generation != 0) : (task == 0 && generation == 0),
                    "only an exact admitted execution has a task generation");
            require(!execution || (task == control_task && generation == control_generation),
                    "control observation correlates to the exact native execution");
            std::uint64_t total{};
            for (const auto* phase : phases) {
                const auto duration = timing_field(line, phase);
                require(duration <= std::numeric_limits<std::uint64_t>::max() - total,
                        "fixture durations sum without overflow");
                total += duration;
            }
            require(total == timing_field(line, " total_ns=") &&
                        timing_field(line, " started_ns=") > 0 &&
                        timing_field(line, " ended_ns=") >= timing_field(line, " started_ns=") &&
                        total ==
                            timing_field(line, " ended_ns=") - timing_field(line, " started_ns=") &&
                        timing_field(line, " original_status=") == CUDA_SUCCESS,
                    "disjoint phases reconcile and preserve exact original status");
            const auto owner = timing_field(line, " admission_owner_probe_ns=");
            const auto signal = timing_field(line, " admission_signal_wait_ns=");
            const auto pause = timing_field(line, " admission_pause_probe_ns=");
            const auto other = timing_field(line, " admission_other_ns=");
            const auto admission = timing_field(line, " admission_ns=");
            require(owner <= admission && signal <= admission - owner &&
                        pause <= admission - owner - signal &&
                        other == admission - owner - signal - pause &&
                        (execution ? owner > 0 : owner == 0 && signal == 0 && pause == 0),
                    "admission subsets include the combined probe but not later physical retries");
        }
        ++samples;
        begin = end + 1;
    }
    require(samples == 3 && executions == 1 && control_samples == 2,
            "disabled and rejected transactions emit nothing; enabled exact-once calls emit once");
}
constexpr glimmer::control::NativeEpochExtent kFrontendExtent{8, 2, 2, 1'000'000};
std::atomic<bool> has_epoch_capture{};
std::thread::id epoch_ptds_caller;
std::atomic<unsigned int> epoch_ptds_records{};
CUresult epoch_stream_context(CUstream, CUcontext* output) {
    *output = kContext;
    return CUDA_SUCCESS;
}
CUresult epoch_stream_capture(CUstream, CUstreamCaptureStatus* output) {
    *output = has_epoch_capture ? CU_STREAM_CAPTURE_STATUS_ACTIVE : CU_STREAM_CAPTURE_STATUS_NONE;
    return CUDA_SUCCESS;
}
CUresult epoch_default_record(CUevent event, CUstream stream) {
    if (stream == CU_STREAM_PER_THREAD) {
        require(std::this_thread::get_id() == epoch_ptds_caller,
                "frontend PTDS predecessor and every original trailing record stay on caller");
        ++epoch_ptds_records;
    }
    return event_record(event, stream);
}
CudaFrontendOptions epoch_options() {
    CudaFrontendOptions options;
    options.progress_slots = 2;
    options.import_events = 4;
    options.per_thread_streams = 2;
    options.native_epoch_policy = kFrontendExtent;
    options.native_epoch_dispatch = {
        context,      push,         pop,         epoch_stream_context, epoch_stream_capture,
        event_create, event_record, event_query, event_destroy};
    return options;
}
std::shared_ptr<glimmer::control::AsyncCooperativeLaunchGate> epoch_control(
    bool is_enabled = true) {
    glimmer::control::LaunchGateOptions options;
    options.scheduling_policy = glimmer::core::SchedulingPolicy::kPriority;
    return std::make_shared<glimmer::control::AsyncCooperativeLaunchGate>(
        options, 64, is_enabled ? std::optional{kFrontendExtent} : std::nullopt);
}
template <typename Predicate>
void epoch_until(Predicate predicate) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{3};
    while (!predicate()) {
        require(std::chrono::steady_clock::now() < deadline, "bounded frontend epoch progress");
        std::this_thread::yield();
    }
}
void epoch_query_domains_and_exact_calls() {
    current_context = kContext;
    auto gate = epoch_control();
    CudaCooperativeFrontend frontend(kContext, dispatch(), gate, epoch_options());
    check(frontend.initialize(), "initialize epoch frontend");
    CUstream stream{}, unrelated{};
    CUevent historical{};
    check(stream_create(&stream, CU_STREAM_NON_BLOCKING), "epoch original stream");
    check(stream_create(&unrelated, CU_STREAM_NON_BLOCKING), "unrelated original stream");
    check(event_create(&historical, 0), "historical original event");
    check(frontend.bind_event(historical, std::make_shared<int>(0)), "bind historical event");
    check(frontend.record_event(historical, stream, false), "record before native epoch");
    check(frontend.synchronize_event(historical), "prove historical event");
    const auto caller = std::this_thread::get_id();
    unsigned int originals{};
    glimmer::control::NativeEpochGrant grant{};
    for (unsigned int i = 0; i < 4; ++i) {
        check(frontend.native_epoch(stream,
                                    [&] {
                                        require(std::this_thread::get_id() == caller,
                                                "original remains on actual caller");
                                        require(current_context == kContext,
                                                "original caller context is unchanged");
                                        is_ready = false;
                                        ++originals;
                                        return CUDA_SUCCESS;
                                    }),
              "join exact native extent");
        const auto snapshot = frontend.native_epoch_snapshot();
        require(snapshot && snapshot->execution && snapshot->execution->issued_calls == i + 1,
                "one count for each original entry");
        if (i == 0) grant = snapshot->grant;
        require(snapshot->grant.execution.task_id == grant.execution.task_id &&
                    snapshot->grant.execution.generation == grant.execution.generation &&
                    snapshot->grant.acceptance_deadline_nanoseconds ==
                        grant.acceptance_deadline_nanoseconds,
                "joined calls retain one immutable authorization and deadline");
    }
    current_context = reinterpret_cast<CUcontext>(0x3000);
    require(frontend.native_epoch(stream,
                                  [&] {
                                      ++originals;
                                      return CUDA_SUCCESS;
                                  }) == CUDA_ERROR_INVALID_CONTEXT &&
                frontend.query_stream(stream, false) == CUDA_ERROR_INVALID_CONTEXT,
            "wrong context rejects warm entry and affected query immediately");
    current_context = kContext;
    check(frontend.query_event(historical), "old event excludes later epoch");
    check(frontend.query_stream(unrelated, false), "unrelated nonblocking stream stays complete");
    require(frontend.query_stream(stream, false) == CUDA_ERROR_NOT_READY && originals == 4,
            "affected query closes acceptance without waiting for withheld physical proof");
    is_ready = true;
    check(frontend.synchronize_stream(stream, false), "wait for epoch proof and logical thaw");
    const auto stopped = frontend.native_epoch_snapshot();
    require(stopped && stopped->phase == CudaNativeEpochPhase::kComplete &&
                stopped->created_events == stopped->destroyed_events &&
                stopped->proven_prefixes == 2,
            "two physical prefixes and all private resources settle before synchronization");
    check(frontend.close_event(historical), "close historical event");
    check(event_destroy(historical), "destroy historical event");
    check(frontend.close_stream(stream), "close epoch stream");
    check(stream_destroy(stream), "destroy epoch stream");
    check(stream_destroy(unrelated), "destroy unrelated stream");
    check(frontend.shutdown(), "shutdown completed epoch frontend");
    gate->shutdown();
    require(event_count == 0 && stream_count == 0 && allocation_count == 0,
            "epoch membership resources are released exactly once");
}
void epoch_idle_caller_exit_and_boundaries() {
    auto gate = epoch_control();
    auto options = epoch_options();
    options.native_epoch_policy->acceptance_microseconds = 10'000;
    CudaCooperativeFrontend frontend(kContext, dispatch(), gate, options);
    check(frontend.initialize(), "initialize idle epoch frontend");
    CUstream stream{};
    check(stream_create(&stream, CU_STREAM_NON_BLOCKING), "idle original stream");
    check(frontend.native_epoch(stream, [] { return CUDA_SUCCESS; }), "issue idle partial epoch");
    epoch_until([&] {
        const auto snapshot = frontend.native_epoch_snapshot();
        return snapshot && snapshot->phase == CudaNativeEpochPhase::kComplete;
    });
    const auto first = frontend.native_epoch_snapshot()->grant.execution;
    std::thread caller([&] {
        current_context = kContext;
        check(frontend.native_epoch(stream, [] { return CUDA_SUCCESS; }),
              "issue on exiting caller");
    });
    caller.join();
    epoch_until([&] {
        const auto snapshot = frontend.native_epoch_snapshot();
        return snapshot && snapshot->phase == CudaNativeEpochPhase::kComplete;
    });
    require(frontend.native_epoch_snapshot()->grant.execution.task_id != first.task_id,
            "new caller lifetime has a fresh independent epoch");
    check(frontend.native_epoch(stream, [] { return CUDA_SUCCESS; }),
          "epoch before event metadata");
    CUevent event{};
    check(event_create(&event, 0), "boundary event");
    check(frontend.bind_event(event, std::make_shared<int>(0)), "binding closes preceding epoch");
    check(frontend.record_event(event, stream, false), "record after exact epoch stop");
    check(frontend.wait_event(nullptr, true, event), "PTDS dependency after epoch");
    check(frontend.native_epoch(stream, [] { return CUDA_SUCCESS; }), "epoch before checkpoint");
    CudaWorkerCheckpoint checkpoint;
    epoch_until([&] { return frontend.checkpoint(checkpoint) == CUDA_SUCCESS; });
    check(frontend.resume_checkpoint(checkpoint), "restore completed epoch checkpoint");
    check(frontend.native_epoch(stream, [] { return CUDA_SUCCESS; }), "epoch before process pause");
    CudaNativeBoundaryToken boundary;
    epoch_until([&] { return frontend.pause(boundary) == CUDA_SUCCESS; });
    check(frontend.finish_pause(boundary, [] { return CUDA_SUCCESS; }),
          "finish independent process pause");
    check(frontend.native_epoch(stream, [] { return CUDA_SUCCESS; }), "epoch before stream close");
    check(frontend.close_stream(stream), "stream close includes physical epoch");
    check(stream_destroy(stream), "destroy stopped stream");
    check(frontend.close_event(event), "close dependency event");
    check(event_destroy(event), "destroy dependency event");
    check(frontend.shutdown(), "shutdown idle frontend");
    gate->shutdown();
    require(event_count == 0 && stream_count == 0 && allocation_count == 0,
            "idle/caller/checkpoint/process closure releases resources");
}
void epoch_scope_change_and_original_rejection() {
    auto gate = epoch_control();
    CudaCooperativeFrontend frontend(kContext, dispatch(), gate, epoch_options());
    check(frontend.initialize(), "initialize scope-change owner");
    CUstream first{}, second{};
    check(stream_create(&first, CU_STREAM_NON_BLOCKING), "first scope");
    check(stream_create(&second, CU_STREAM_NON_BLOCKING), "second scope");
    check(frontend.native_epoch(first,
                                [] {
                                    is_ready = false;
                                    return CUDA_SUCCESS;
                                }),
          "first scope issue");
    const auto old = frontend.native_epoch_snapshot();
    std::thread proof([&] {
        epoch_until([&] {
            const auto snapshot = frontend.native_epoch_snapshot();
            return snapshot && snapshot->execution &&
                   snapshot->execution->phase == NativeEpochPhase::kStopping;
        });
        is_ready = true;
    });
    unsigned int rejected{};
    require(frontend.native_epoch(second,
                                  [&] {
                                      ++rejected;
                                      return CUDA_ERROR_INVALID_VALUE;
                                  }) == CUDA_ERROR_INVALID_VALUE,
            "new scope waits for old stop and preserves original rejection status");
    proof.join();
    const auto current = frontend.native_epoch_snapshot();
    require(current && current->execution && old && old->execution &&
                current->grant.execution.task_id != old->grant.execution.task_id && rejected == 1,
            "changed stream gets a distinct scope and one rejected original");
    unsigned int source = 17, destination{};
    check(frontend.copy(
              second, false,
              {CudaQueuedCopyKind::kHostToDevice, reinterpret_cast<CUdeviceptr>(&destination), 0,
               &source, sizeof(source), std::make_shared<int>(0)}),
          "copy closes failed original epoch after proof");
    check(frontend.synchronize_context(), "settle rejected original and dependent copy");
    require(destination == source && frontend.native_epoch_snapshot()->execution->has_failed,
            "known rejected original is reported once without preventing later ordered work");
    check(frontend.close_stream(first), "close first scope");
    check(frontend.close_stream(second), "close second scope");
    check(stream_destroy(first), "destroy first scope");
    check(stream_destroy(second), "destroy second scope");
    check(frontend.shutdown(), "shutdown changed-scope owner");
    gate->shutdown();
    require(event_count == 0 && stream_count == 0 && allocation_count == 0,
            "changed-scope and rejection resources settle exactly once");
}
void canonical_frontend_epoch(unsigned int mode) {
    current_context = kContext;
    epoch_ptds_caller = std::this_thread::get_id();
    epoch_ptds_records = 0;
    auto gate = epoch_control();
    auto options = epoch_options();
    options.native_epoch_dispatch.event_record = epoch_default_record;
    CudaCooperativeFrontend frontend(kContext, dispatch(), gate, options);
    check(frontend.initialize(), "initialize canonical frontend");
    CUstream blocking{}, unrelated{};
    CUevent historical{};
    check(stream_create(&blocking, CU_STREAM_DEFAULT), "original blocking stream");
    check(stream_create(&unrelated, CU_STREAM_NON_BLOCKING), "original independent stream");
    check(event_create(&historical, 0), "canonical historical event");
    check(frontend.bind_event(historical, std::make_shared<int>(0)),
          "bind canonical historical event");
    const bool is_ptds = mode == 2 || mode == 3;
    const CUstream stream = mode == 0 || mode == 2 ? nullptr
                            : mode == 1            ? CU_STREAM_LEGACY
                            : mode == 3            ? CU_STREAM_PER_THREAD
                                                   : blocking;
    check(frontend.record_event(historical, stream, is_ptds), "historical canonical record");
    check(frontend.synchronize_event(historical), "complete canonical historical prefix");
    unsigned int originals{};
    check(frontend.native_epoch(
              stream,
              [&] {
                  ++originals;
                  is_ready = false;
                  return CUDA_SUCCESS;
              },
              is_ptds),
          "first canonical original");
    const auto grant = frontend.native_epoch_snapshot()->grant;
    const CUstream alias = mode < 2  ? (stream ? nullptr : CU_STREAM_LEGACY)
                           : is_ptds ? (stream ? nullptr : CU_STREAM_PER_THREAD)
                                     : stream;
    check(frontend.native_epoch(
              alias,
              [&] {
                  ++originals;
                  return CUDA_SUCCESS;
              },
              is_ptds),
          "join canonical null/explicit alias");
    const auto joined = frontend.native_epoch_snapshot();
    require(joined && joined->execution && joined->execution->issued_calls == 2 && originals == 2 &&
                joined->grant.execution.task_id == grant.execution.task_id &&
                joined->grant.extent.calls_per_prefix == (is_ptds ? 1U : 2U),
            "aliases retain exact authorization; PTDS requests width one before grant");
    if (is_ptds)
        require(epoch_ptds_records == 3, "one caller predecessor and two caller trailing records");
    check(frontend.query_event(historical), "historical canonical event stays complete");
    check(frontend.query_stream(unrelated, false), "nonblocking stream excludes canonical epoch");
    CUresult other_ptds{};
    std::thread other([&] {
        current_context = kContext;
        other_ptds = frontend.query_stream(nullptr, true);
    });
    other.join();
    require(other_ptds == (mode < 2 ? CUDA_ERROR_NOT_READY : CUDA_SUCCESS),
            "another caller's PTDS only includes genuine legacy implicit dependency");
    require(frontend.query_stream(nullptr, false) == CUDA_ERROR_NOT_READY,
            "legacy query includes blocking/PTDS epoch implicit dependencies");
    if (mode < 2)
        require(frontend.query_stream(blocking, false) == CUDA_ERROR_NOT_READY,
                "blocking query includes live legacy epoch");
    require(frontend.query_stream(alias, is_ptds) == CUDA_ERROR_NOT_READY,
            "affected canonical alias query cannot wait for physical proof");
    is_ready = true;
    check(frontend.synchronize_stream(stream, is_ptds), "prove and join canonical epoch");
    const auto complete = frontend.native_epoch_snapshot();
    require(complete && complete->phase == CudaNativeEpochPhase::kComplete &&
                complete->created_events == complete->destroyed_events,
            "canonical synchronization includes proof, report and joined cleanup");
    check(frontend.close_event(historical), "close canonical historical event");
    check(event_destroy(historical), "destroy canonical event");
    if (mode == 4) check(frontend.close_stream(blocking), "close bound blocking scope");
    check(stream_destroy(blocking), "destroy original blocking stream");
    check(stream_destroy(unrelated), "destroy original independent stream");
    check(frontend.shutdown(), "shutdown canonical frontend");
    gate->shutdown();
    require(event_count == 0 && stream_count == 0 && allocation_count == 0,
            "canonical frontend resources settle exactly once");
}
void captured_epoch_rejects_before_import() {
    auto gate = epoch_control();
    CudaCooperativeFrontend frontend(kContext, dispatch(), gate, epoch_options());
    check(frontend.initialize(), "initialize caller-capture preflight");
    has_epoch_capture = true;
    unsigned int originals{};
    const auto events = event_count.load();
    require(frontend.native_epoch(
                nullptr,
                [&] {
                    ++originals;
                    return CUDA_SUCCESS;
                },
                true) == CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED &&
                originals == 0 && event_count == events && !frontend.native_epoch_snapshot(),
            "caller capture rejects before prefix import or owner acquisition");
    has_epoch_capture = false;
    check(frontend.shutdown(), "unsupported capture does not poison unused frontend");
    gate->shutdown();
}
void epoch_missing_configuration_has_no_native_effect() {
    auto gate = epoch_control();
    auto options = epoch_options();
    options.native_epoch_dispatch.event_query = nullptr;
    CudaCooperativeFrontend frontend(kContext, dispatch(), gate, options);
    const auto before = native_calls.load();
    require(frontend.initialize() == CUDA_ERROR_INVALID_VALUE && native_calls == before,
            "missing epoch proof dispatch is rejected before native initialization");
    gate->shutdown();
}
std::unique_ptr<NativeExecutionEpoch> reject_epoch_model(NativeEpochScope, NativeEpochAuthorization,
                                                         NativeExecutionEpoch::TimePoint) {
    throw std::bad_alloc();
}
CUresult epoch_actor_push(CUcontext value) {
    is_epoch_queue_actor = true;
    return push(value);
}
void epoch_known_rejection_does_not_retry(bool should_fail_model) {
    auto gate = epoch_control(should_fail_model);
    auto options = epoch_options();
    if (should_fail_model) options.native_epoch_monitor.create_execution = reject_epoch_model;
    CudaCooperativeFrontend frontend(kContext, dispatch(), gate, options);
    check(frontend.initialize(), "initialize rejected epoch frontend");
    CUstream stream{};
    check(stream_create(&stream, CU_STREAM_NON_BLOCKING), "rejected original stream");
    unsigned int originals{};
    const auto status = frontend.native_epoch(stream, [&] {
        ++originals;
        return CUDA_SUCCESS;
    });
    require(status == (should_fail_model ? CUDA_ERROR_OUT_OF_MEMORY : CUDA_ERROR_NOT_PERMITTED) &&
                originals == 0,
            "known failed acquisition/model returns without replay or original entry");
    check(frontend.synchronize_context(), "known unused failure thaws ordinary frontend");
    check(frontend.close_stream(stream), "close rejected binding");
    check(stream_destroy(stream), "destroy rejected stream");
    check(frontend.shutdown(), "shutdown rejected epoch frontend");
    gate->shutdown();
    require(event_count == 0 && stream_count == 0 && allocation_count == 0,
            "known unused rejection cleans all resources");
}
void epoch_failed_proof_or_report_retains_pause(unsigned int mode) {
    const auto child = ::fork();
    require(child >= 0, "fork isolated retained epoch ownership");
    if (child == 0) {
        try {
            current_context = kContext;
            auto gate = epoch_control();
            auto table = dispatch();
            table.worker.context_push = epoch_actor_push;
            CudaCooperativeFrontend frontend(kContext, table, gate, epoch_options());
            check(frontend.initialize(), "initialize failed epoch owner");
            CUstream stream{};
            check(stream_create(&stream, CU_STREAM_NON_BLOCKING), "failed epoch original stream");
            unsigned int originals{};
            check(frontend.native_epoch(stream,
                                        [&] {
                                            ++originals;
                                            if (mode == 0) should_fail_record = true;
                                            if (mode == 1) should_fail_epoch_query = true;
                                            if (mode == 2) gate->shutdown();
                                            if (mode == 3)
                                                should_reject_epoch_marker_allocation = true;
                                            return CUDA_SUCCESS;
                                        }),
                  "entered original status is preserved before failed proof/report");
            // Partial prefixes close on the fixed deadline without another
            // original call. Known record/query/report failure cannot thaw.
            epoch_until([&] {
                const auto snapshot = frontend.native_epoch_snapshot();
                return snapshot && snapshot->phase == (mode == 3 ? CudaNativeEpochPhase::kComplete
                                                                 : CudaNativeEpochPhase::kRetained);
            });
            epoch_until([&] { return frontend.query_event(nullptr) != CUDA_ERROR_INVALID_HANDLE; });
            require(frontend.synchronize_context() != CUDA_SUCCESS &&
                        frontend.native_epoch(stream,
                                              [&] {
                                                  ++originals;
                                                  return CUDA_SUCCESS;
                                              }) != CUDA_SUCCESS &&
                        originals == 1,
                    "failed proof/report retains pause and never replays original");
            const auto retained = frontend.native_epoch_snapshot();
            require(retained && retained->execution &&
                        retained->execution->has_ownership == (mode != 3) &&
                        retained->execution->issued_calls == 1,
                    "unacknowledged ownership remains exact after failure");
            if (mode < 2)
                require(retained->created_events > retained->destroyed_events,
                        "unproved native events stay retained");
            if (mode == 3)
                require(rejected_epoch_marker_allocations > 0 &&
                            retained->created_events == retained->destroyed_events,
                        "marker allocation fails only after proven and acknowledged physical stop");
            if (mode >= 2) {
                const auto calls = native_calls.load();
                require(
                    frontend.recover_native_epoch(retained->grant) == CUDA_ERROR_UNKNOWN &&
                        native_calls == calls,
                    "lost terminal or marker failure cannot authorize another physical recovery");
            }
            gate->shutdown();
            ::_exit(EXIT_SUCCESS);  // quarantine is intentional, not a leak-free success path
        } catch (...) {
            ::_exit(EXIT_FAILURE);
        }
    }
    int status{};
    require(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
            "retained proof/report failure regression passes");
}
thread_local std::array<CUcontext, 8> recovery_context_stack{};
thread_local std::size_t recovery_context_depth{};
std::atomic<unsigned int> recovery_synchronizations{};
std::atomic<bool> should_fail_recovery_synchronization{};
CUresult recovery_push(CUcontext value) {
    require(recovery_context_depth < recovery_context_stack.size(),
            "bounded recovery context stack");
    recovery_context_stack[recovery_context_depth++] = current_context;
    current_context = value;
    return CUDA_SUCCESS;
}
CUresult recovery_pop(CUcontext* value) {
    require(recovery_context_depth != 0, "balanced recovery context stack");
    *value = current_context;
    current_context = recovery_context_stack[--recovery_context_depth];
    return CUDA_SUCCESS;
}
CUresult recovery_synchronize() {
    require(current_context == kContext, "actual whole-context proof binds exact frontend context");
    ++recovery_synchronizations;
    if (should_fail_recovery_synchronization.load()) return CUDA_ERROR_LAUNCH_FAILED;
    is_ready = true;
    return CUDA_SUCCESS;
}
CudaFrontendOptions recovery_options() {
    auto options = epoch_options();
    options.native_epoch_dispatch.context_push = recovery_push;
    options.native_epoch_dispatch.context_pop = recovery_pop;
    options.native_epoch_dispatch.context_synchronize = recovery_synchronize;
    return options;
}
void epoch_frontend_recovery(unsigned int mode, bool failed_record, bool is_borrowed = false) {
    current_context = kContext;
    recovery_synchronizations = 0;
    auto gate = epoch_control();
    CudaCooperativeFrontend frontend(kContext, dispatch(), gate, recovery_options());
    check(frontend.initialize(), "initialize recoverable frontend");
    CUstream explicit_stream{};
    if (mode == 0)
        check(stream_create(&explicit_stream, CU_STREAM_NON_BLOCKING), "recovery original stream");
    const auto stream = mode == 0   ? explicit_stream
                        : mode == 1 ? CU_STREAM_LEGACY
                                    : CU_STREAM_PER_THREAD;
    CUevent historical{};
    check(event_create(&historical, 0), "recovery historical event");
    check(frontend.bind_event(historical, std::make_shared<int>(0)),
          "bind recovery historical event");
    check(frontend.record_event(historical, stream, mode == 2),
          "record recovery historical prefix");
    check(frontend.synchronize_event(historical), "complete recovery historical prefix");
    unsigned int originals{};
    check(frontend.native_epoch(
              stream,
              [&] {
                  ++originals;
                  const auto active = frontend.native_epoch_snapshot();
                  require(
                      active &&
                          frontend.recover_native_epoch(active->grant) == CUDA_ERROR_NOT_READY &&
                          recovery_synchronizations == 0,
                      "entered original excludes recovery before SDK or actor effects");
                  is_ready = false;
                  should_fail_record = failed_record;
                  should_fail_epoch_query = !failed_record;
                  return CUDA_SUCCESS;
              },
              mode == 2),
          "original status survives private proof failure");
    static_cast<void>(frontend.close_native_epoch(false));
    epoch_until([&] { return frontend.query_event(nullptr) != CUDA_ERROR_INVALID_HANDLE; });
    const auto retained = frontend.native_epoch_snapshot();
    require(retained && retained->phase == CudaNativeEpochPhase::kRetained &&
                retained->execution->has_ownership && originals == 1,
            "exclusive epoch failure retains physical capacity and actor pause");
    const auto calls = native_calls.load();
    auto stale = retained->grant;
    ++stale.acceptance_deadline_nanoseconds;
    require(frontend.recover_native_epoch(stale) == CUDA_ERROR_INVALID_VALUE &&
                native_calls == calls && recovery_synchronizations == 0,
            "stale immutable grant cannot perform any recovery SDK effect");
    auto waiting = gate->submit_and_claim({10, 1}).get();
    require(waiting.claim.status == glimmer::control::CooperativeClaimStatus::kPending,
            "failed frontend still holds shared priority capacity");
    should_fail_record = false;
    should_fail_epoch_query = false;
    unsigned int borrowed_originals{};
    const auto borrowed = [&] {
        ++borrowed_originals;
        require(current_context == kContext && recovery_context_depth == 0,
                "borrowed frontend proof never pushes the application context");
        return recovery_synchronize();
    };
    const auto recovered_status =
        is_borrowed ? frontend.recover_native_epoch(borrowed)
                    : std::optional{frontend.recover_native_epoch(retained->grant)};
    require(recovered_status.has_value(), "retained frontend recovery cannot decline");
    check(*recovered_status, "recover exact failed frontend");
    require(borrowed_originals == (is_borrowed ? 1U : 0U),
            "borrowed SDK completion is observed exactly once");
    const auto recovered = frontend.native_epoch_snapshot();
    require(recovered && recovered->status == retained->status &&
                recovered->execution->has_failed && !recovered->execution->has_ownership &&
                recovered->execution->issued_calls == 1 &&
                recovered->created_events == recovered->destroyed_events &&
                recovery_synchronizations == 1 && current_context == kContext &&
                recovery_context_depth == 0,
            "one physical proof precedes healthy actor thaw without erasing old error or count");
    require(frontend.recover_native_epoch(retained->grant) != CUDA_SUCCESS &&
                recovery_synchronizations == 1,
            "retired epoch cannot be recovered or reported twice");
    if (is_borrowed)
        require(!frontend.recover_native_epoch(borrowed) && borrowed_originals == 1,
                "healthy recovered frontend declines without invoking another SDK completion");
    waiting.claim = gate->claim(waiting.task_id).get();
    require(waiting.claim.status == glimmer::control::CooperativeClaimStatus::kGranted &&
                gate->complete(waiting.claim.execution).get(),
            "actual stop permits priority handoff");
    check(frontend.query_event(historical),
          "recovery preserves completed historical event identity");
    check(frontend.native_epoch(
              stream,
              [&] {
                  ++originals;
                  return CUDA_SUCCESS;
              },
              mode == 2),
          "new original obtains an independent epoch after recovery");
    const auto next = frontend.native_epoch_snapshot();
    require(next && next->grant.execution.task_id != retained->grant.execution.task_id &&
                originals == 2,
            "new work never replays or reuses the retired grant");
    check(frontend.synchronize_context(), "finish new recovery generation");
    check(frontend.close_event(historical), "close recovery historical binding");
    check(event_destroy(historical), "destroy original recovery event");
    if (explicit_stream) {
        check(frontend.close_stream(explicit_stream), "close recovery stream binding");
        check(stream_destroy(explicit_stream), "destroy original recovery stream");
    }
    check(frontend.shutdown(), "shutdown healthy recovered frontend");
    gate->shutdown();
    require(event_count == 0 && stream_count == 0 && allocation_count == 0,
            "recovered frontend retires every normal host/native resource");
}
void epoch_unrecoverable_frontend(unsigned int mode, bool is_borrowed = false) {
    const auto child = ::fork();
    require(child >= 0, "fork isolated unrecoverable frontend");
    if (child == 0) {
        try {
            current_context = kContext;
            recovery_synchronizations = 0;
            auto gate = epoch_control();
            auto table = dispatch();
            table.worker.context_push = epoch_actor_push;
            CudaCooperativeFrontend frontend(kContext, table, gate, recovery_options());
            check(frontend.initialize(), "initialize retained recovery frontend");
            CUstream stream{};
            check(stream_create(&stream, CU_STREAM_NON_BLOCKING), "unrecoverable original stream");
            unsigned int originals{};
            check(frontend.native_epoch(stream,
                                        [&] {
                                            ++originals;
                                            should_fail_epoch_query = true;
                                            return CUDA_SUCCESS;
                                        }),
                  "original return precedes isolated recovery failure");
            static_cast<void>(frontend.close_native_epoch(false));
            epoch_until([&] { return frontend.query_event(nullptr) != CUDA_ERROR_INVALID_HANDLE; });
            const auto retained = frontend.native_epoch_snapshot();
            require(retained && retained->execution->has_ownership,
                    "retained recovery starts with ownership");
            should_fail_epoch_query = false;
            if (mode == 0) should_fail_recovery_synchronization = true;
            if (mode == 1) frontend.fail_native_boundary(retained->status);
            if (mode == 2) should_reject_epoch_marker_allocation = true;
            const auto result =
                is_borrowed ? frontend.recover_native_epoch([] { return recovery_synchronize(); })
                            : std::optional{frontend.recover_native_epoch(retained->grant)};
            should_reject_epoch_marker_allocation = false;
            require(result && *result != CUDA_SUCCESS &&
                        recovery_synchronizations == (mode == 1 ? 0U : 1U) &&
                        frontend.synchronize_context() != CUDA_SUCCESS && originals == 1,
                    "failed proof, additional same-error boundary or marker cannot heal frontend");
            const auto after = frontend.native_epoch_snapshot();
            require(
                after->execution->has_ownership == (mode != 2),
                "marker failure does not fabricate ownership after actual failed acknowledgment");
            const auto calls = recovery_synchronizations.load();
            require(frontend.recover_native_epoch(retained->grant) != CUDA_SUCCESS &&
                        recovery_synchronizations == calls,
                    "failed recovery cannot repeat SDK proof, cleanup, terminal or marker");
            ::_exit(EXIT_SUCCESS);  // intentional failed quarantine, not leak-free recovery
        } catch (...) {
            ::_exit(EXIT_FAILURE);
        }
    }
    int status{};
    require(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
            "isolated unrecoverable frontend retains exact failed ownership");
}
}  // namespace

int main() {
    try {
        native_handoff_needs_no_caller_subscription();
        invalid_completion_interval_has_no_native_effect();
        epoch_query_domains_and_exact_calls();
        epoch_idle_caller_exit_and_boundaries();
        epoch_scope_change_and_original_rejection();
        for (unsigned int mode = 0; mode < 5; ++mode) canonical_frontend_epoch(mode);
        captured_epoch_rejects_before_import();
        epoch_missing_configuration_has_no_native_effect();
        epoch_known_rejection_does_not_retry(false);
        epoch_known_rejection_does_not_retry(true);
        for (unsigned int mode = 0; mode < 4; ++mode)
            epoch_failed_proof_or_report_retains_pause(mode);
        for (unsigned int mode = 0; mode < 3; ++mode) {
            epoch_frontend_recovery(mode, false);
            epoch_frontend_recovery(mode, true);
            epoch_frontend_recovery(mode, false, true);
            epoch_frontend_recovery(mode, true, true);
            epoch_unrecoverable_frontend(mode);
            epoch_unrecoverable_frontend(mode, true);
        }
        native_phase_diagnostics_are_opt_in();
        bindings_and_owned_arguments();
        sliced_priority_uses_physical_proof();
        slice_count = 0;
        sliced_priority_uses_physical_proof(true);
        slice_count = 0;
        sliced_priority_uses_physical_proof(true, true);
        reversible_checkpoint_preserves_dependencies();
        checkpoint_retirement_is_cuda_free();
        checkpoint_progress_failure_retains_ownership();
        queued_release_retains_predecessors_and_resources();
        retained_process_native_session();
        for (unsigned int mode = 0; mode < 6; ++mode) {
            failed_process_session_retains_ownership(mode);
        }
        native_transaction_keeps_rejected_payload();
        native_execution_waits_before_admission();
        failed_native_wait_never_issues();
        fresh_record_owns_issued_native_prefix();
        event_publication_is_part_of_acceptance();
        caller_record_contribution_precedes_worker_pause();
        immutable_wait_snapshots_and_serialized_callers();
        completion_looks_up_exact_existing_streams();
        process_pause_requires_actual_native_completion();
        uncertain_cleanup_never_replays(0);
        uncertain_cleanup_never_replays(1);
        uncertain_cleanup_never_replays(2);
        uncertain_cleanup_never_replays(3);
        std::cout
            << "cooperative_frontend owned_arguments=1 logical_events=1 native_transaction=1\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "cooperative frontend failure: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
